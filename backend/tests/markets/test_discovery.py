import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from prediction_desk.audit.log import AuditLog
from prediction_desk.clock import SimClock
from prediction_desk.db.engine import make_sessionmaker
from prediction_desk.db.models import AuditEvent, EventContract, Release
from prediction_desk.domain.enums import Comparator, MappingStatus, Measure, Series, Venue
from prediction_desk.markets.discovery import DiscoveryResult, discover_contracts
from prediction_desk.markets.mapping import MAPPER_VERSION
from prediction_desk.markets.types import MarketRef, OrderBook, VenueAdapter, VenueMarket
from prediction_desk.net.http import ProviderError

CPI_RULES = (
    "If the Consumer Price Index (CPI) increases by more than 0.3% (single-decimal) "
    "in November 2026, then the market resolves to Yes."
)
CLOSE = datetime(2026, 12, 10, 13, 29, tzinfo=UTC)
NOW = datetime(2026, 11, 20, tzinfo=UTC)


class FakeAdapter:
    venue = Venue.KALSHI

    def __init__(
        self,
        markets: Mapping[str, Sequence[VenueMarket]],
        errors: Mapping[str, Exception] | None = None,
    ) -> None:
        self.markets = markets
        self.errors = errors or {}

    async def list_markets(self, series_code: str) -> list[VenueMarket]:
        if series_code in self.errors:
            raise self.errors[series_code]
        return list(self.markets.get(series_code, []))

    async def order_book(self, ref: MarketRef) -> OrderBook:
        raise NotImplementedError


def _market(floor: object = 0.3, mid: str = "KXCPI-26NOV-T0.3", **kw: object) -> VenueMarket:
    base: dict[str, object] = {
        "venue": Venue.KALSHI,
        "market_id": mid,
        "event_id": "KXCPI-26NOV",
        "series_code": "KXCPI",
        "title": "CPI above 0.3",
        "rules_text": CPI_RULES,
        "close_time": CLOSE,
        "status": "active",
        "strike": {
            "strike_type": "greater",
            "floor_strike": floor,
            "cap_strike": None,
            "custom_strike": None,
        },
        "yes_token_id": None,
    }
    base.update(kw)
    return VenueMarket(**base)  # type: ignore[arg-type]


async def _seed_release(session: AsyncSession) -> None:
    session.add(
        Release(
            series=Series.CPI,
            period="2026-11",
            scheduled_at=datetime(2026, 12, 10, 13, 30, tzinfo=UTC),
        )
    )
    await session.commit()


async def _run(
    sessions: async_sessionmaker[AsyncSession],
    adapter: VenueAdapter,
    series: Sequence[str] = ("KXCPI",),
    clock: SimClock | None = None,
) -> DiscoveryResult:
    clock = clock or SimClock(NOW)
    return await discover_contracts(
        sessions, {Venue.KALSHI: (adapter, series)}, AuditLog(clock), clock
    )


async def _audit_count(session: AsyncSession) -> int:
    return (await session.execute(select(func.count()).select_from(AuditEvent))).scalar_one()


async def test_discovery_maps_and_audits_once(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    await _seed_release(session)
    result = await _run(sessions, FakeAdapter({"KXCPI": [_market()]}))
    assert (result.mapped, result.unmapped, result.changed, result.unchanged) == (1, 0, 1, 0)
    assert result.failures == []
    events = (await session.execute(select(AuditEvent))).scalars().all()
    assert len(events) == 1
    ev = events[0]
    assert ev.actor == "system:discovery"
    assert ev.entity == "event_contract"
    assert ev.entity_id == "kalshi:KXCPI-26NOV-T0.3"
    assert ev.action == "contract_mapped"
    assert ev.payload["status"] == "mapped"
    assert ev.payload["measure"] == "cpi_mom"
    assert ev.payload["comparator"] == "gt"
    assert ev.payload["strike_lo"] == "0.3000"
    assert ev.payload["strike_hi"] is None
    assert ev.payload["reason"] is None
    assert ev.payload["mapper"] == MAPPER_VERSION
    row = (await session.execute(select(EventContract))).scalar_one()
    assert row.mapping_status is MappingStatus.MAPPED
    assert row.mapping_confidence == Decimal("1.00")
    assert row.mapped_by == MAPPER_VERSION
    assert row.measure is Measure.CPI_MOM
    assert row.comparator is Comparator.GT
    assert row.release_id is not None
    assert ev.payload["release_id"] == row.release_id


async def test_discovery_rerun_is_silent(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    await _seed_release(session)
    adapter = FakeAdapter({"KXCPI": [_market()]})
    await _run(sessions, adapter)
    result = await _run(sessions, adapter)
    assert (result.mapped, result.changed, result.unchanged) == (1, 0, 1)
    assert await _audit_count(session) == 1


async def test_rerun_after_db_roundtrip_is_unchanged(
    session: AsyncSession, engine: AsyncEngine
) -> None:
    """Each run reads contracts back through a fresh session, so values round-trip the DB."""
    await _seed_release(session)
    markets = [_market(mid=f"KXCPI-26NOV-T{i}", floor=Decimal(i) / 10) for i in range(3)]
    markets.append(_market(mid="KXCPI-26NOV-BAD", floor=None))
    adapter = FakeAdapter({"KXCPI": markets})
    first = await _run(make_sessionmaker(engine), adapter)
    assert first.changed == 4
    count = await _audit_count(session)
    again = await _run(make_sessionmaker(engine), adapter)
    assert again.unchanged == 4
    assert again.changed == 0
    assert await _audit_count(session) == count


async def test_discovery_terms_change_remaps(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    await _seed_release(session)
    await _run(sessions, FakeAdapter({"KXCPI": [_market(floor=0.3)]}))
    result = await _run(sessions, FakeAdapter({"KXCPI": [_market(floor=0.4)]}))
    assert (result.changed, result.unchanged) == (1, 0)
    actions = (
        (await session.execute(select(AuditEvent.action).order_by(AuditEvent.seq))).scalars().all()
    )
    assert actions == ["contract_mapped", "contract_remapped"]
    row = (await session.execute(select(EventContract))).scalar_one()
    assert row.strike_lo == Decimal("0.4")


async def test_descriptive_change_is_not_audited(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    await _seed_release(session)
    await _run(sessions, FakeAdapter({"KXCPI": [_market()]}))
    result = await _run(sessions, FakeAdapter({"KXCPI": [_market(title="renamed")]}))
    assert (result.changed, result.unchanged) == (0, 1)
    assert await _audit_count(session) == 1
    row = (await session.execute(select(EventContract))).scalar_one()
    assert row.title == "renamed"
    assert row.updated_at == NOW


async def test_discovery_unmapped_reason_persisted(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    result = await _run(sessions, FakeAdapter({"KXCPI": [_market()]}))
    assert (result.mapped, result.unmapped) == (0, 1)
    row = (await session.execute(select(EventContract))).scalar_one()
    assert row.mapping_status is MappingStatus.UNMAPPED
    assert row.unmapped_reason == "no_release"
    assert row.mapping_confidence == Decimal("0.00")
    assert row.mapped_by == MAPPER_VERSION
    assert row.measure is None
    assert row.release_id is None
    ev = (await session.execute(select(AuditEvent))).scalar_one()
    assert ev.action == "contract_unmapped"
    assert ev.payload["reason"] == "no_release"


async def test_discovery_remaps_when_release_appears(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    adapter = FakeAdapter({"KXCPI": [_market()]})
    await _run(sessions, adapter)
    await _seed_release(session)
    result = await _run(sessions, adapter)
    assert (result.mapped, result.changed) == (1, 1)
    row = (await session.execute(select(EventContract))).scalar_one()
    assert row.mapping_status is MappingStatus.MAPPED
    assert row.unmapped_reason is None


async def test_discovery_continues_after_provider_error(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    await _seed_release(session)
    adapter = FakeAdapter(
        {"KXCPICORE": [_market()]}, errors={"KXCPI": ProviderError("kalshi", "boom")}
    )
    result = await _run(sessions, adapter, ("KXCPI", "KXCPICORE"))
    assert result.failures == [("KXCPI", "ProviderError: kalshi: boom")]
    assert result.changed == 1
    assert await _audit_count(session) == 1


async def test_discovery_db_error_rolls_back_and_raises(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    await _seed_release(session)
    bad = _market(title=None)  # NOT NULL violation on flush
    with pytest.raises(Exception, match="title"):
        await _run(sessions, FakeAdapter({"KXCPI": [_market(mid="ok"), bad]}))
    # The service rolled back its own session; the caller's session was never touched.
    assert (
        await session.execute(select(func.count()).select_from(EventContract))
    ).scalar_one() == 0
    assert await _audit_count(session) == 0


async def test_discovery_audit_chain_verifies(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    await _seed_release(session)
    await _run(sessions, FakeAdapter({"KXCPI": [_market(floor=0.3), _market(mid="b", floor=0.5)]}))
    await _run(sessions, FakeAdapter({"KXCPI": [_market(floor=0.4), _market(mid="b", floor=0.5)]}))
    verified = await AuditLog(SimClock(NOW)).verify(session)
    assert verified.ok
    assert verified.checked == 3


async def test_discovery_unexpected_error_keeps_committed_series(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    await _seed_release(session)
    adapter = FakeAdapter({"KXCPI": [_market()]}, errors={"KXCPICORE": RuntimeError("bug")})
    with pytest.raises(RuntimeError, match="bug"):
        await _run(sessions, adapter, ("KXCPI", "KXCPICORE"))
    assert (
        await session.execute(select(func.count()).select_from(EventContract))
    ).scalar_one() == 1
    assert await _audit_count(session) == 1


async def test_updated_at_bumps_only_on_descriptive_change(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    await _seed_release(session)
    clock = SimClock(NOW)
    await _run(sessions, FakeAdapter({"KXCPI": [_market()]}), clock=clock)
    clock.advance(timedelta(hours=1))
    unchanged = await _run(sessions, FakeAdapter({"KXCPI": [_market()]}), clock=clock)
    assert unchanged.unchanged == 1
    row = (
        await session.execute(select(EventContract).execution_options(populate_existing=True))
    ).scalar_one()
    assert (row.created_at, row.updated_at) == (NOW, NOW)
    clock.advance(timedelta(hours=1))
    await _run(sessions, FakeAdapter({"KXCPI": [_market(title="renamed")]}), clock=clock)
    row = (
        await session.execute(select(EventContract).execution_options(populate_existing=True))
    ).scalar_one()
    assert (row.created_at, row.updated_at) == (NOW, NOW + timedelta(hours=2))
    assert await _audit_count(session) == 1


async def test_duplicate_market_ids_keep_first_and_warn(
    session: AsyncSession,
    sessions: async_sessionmaker[AsyncSession],
    caplog: pytest.LogCaptureFixture,
) -> None:
    await _seed_release(session)
    markets = [_market(floor=0.3), _market(floor=0.3), _market(floor=0.4)]
    with caplog.at_level(logging.WARNING, logger="prediction_desk.markets.discovery"):
        result = await _run(sessions, FakeAdapter({"KXCPI": markets}))
    assert (result.mapped, result.changed, result.unchanged) == (1, 1, 0)
    row = (await session.execute(select(EventContract))).scalar_one()
    assert row.strike_lo == Decimal("0.3")
    ev = (await session.execute(select(AuditEvent))).scalar_one()
    assert ev.payload["strike_lo"] == "0.3000"
    assert caplog.text.count("duplicate market") == 1
    assert "KXCPI-26NOV-T0.3" in caplog.text


async def test_audit_payload_names_linked_release(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    await _seed_release(session)
    await _run(sessions, FakeAdapter({"KXCPI": [_market(), _market(mid="BAD", floor=None)]}))
    events = {
        ev.entity_id: ev.payload for ev in (await session.execute(select(AuditEvent))).scalars()
    }
    mapped = events["kalshi:KXCPI-26NOV-T0.3"]
    assert mapped["release_period"] == "2026-11"
    assert mapped["release_scheduled_at"] == "2026-12-10T13:30:00+00:00"
    unmapped = events["kalshi:BAD"]
    assert unmapped["release_period"] is None
    assert unmapped["release_scheduled_at"] is None

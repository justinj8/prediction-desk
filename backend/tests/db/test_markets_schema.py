from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError, StatementError
from sqlalchemy.ext.asyncio import AsyncSession

from prediction_desk.db.models import DataObservation, EventContract, MarketSnapshot, Release
from prediction_desk.domain.enums import (
    Comparator,
    DataSource,
    MappingStatus,
    Measure,
    Series,
    Venue,
)

NOW = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)


def _contract(
    market_id: str = "KXCPI-26NOV-T0.3",
    venue: Venue = Venue.KALSHI,
    **overrides: object,
) -> EventContract:
    fields: dict[str, object] = {
        "venue": venue,
        "venue_market_id": market_id,
        "venue_event_id": "KXCPI-26NOV",
        "title": "CPI above 0.3%?",
        "rules_text": "Resolves yes if CPI m/m is above 0.3%.",
        "close_time": NOW,
        "mapping_status": MappingStatus.UNMAPPED,
        "mapping_confidence": Decimal("0.00"),
        "mapped_by": "test",
        "created_at": NOW,
        "updated_at": NOW,
    }
    fields.update(overrides)
    return EventContract(**fields)


async def _release(session: AsyncSession) -> int:
    release = Release(series=Series.CPI, period="2026-10", scheduled_at=NOW)
    session.add(release)
    await session.flush()
    return release.id


async def test_enum_values_stored_lowercase(session: AsyncSession) -> None:
    session.add(_contract())
    await session.commit()
    assert (
        await session.execute(text("SELECT venue::text FROM event_contracts"))
    ).scalar_one() == "kalshi"


async def test_mapped_contract_requires_terms(session: AsyncSession) -> None:
    release_id = await _release(session)
    session.add(
        _contract(
            mapping_status=MappingStatus.MAPPED,
            release_id=release_id,
            measure=None,
            comparator=Comparator.GT,
            strike_lo=Decimal("0.3"),
        )
    )
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_mapped_contract_with_terms_ok(session: AsyncSession) -> None:
    release_id = await _release(session)
    session.add(
        _contract(
            mapping_status=MappingStatus.MAPPED,
            release_id=release_id,
            measure=Measure.CPI_MOM,
            comparator=Comparator.GT,
            strike_lo=Decimal("0.3"),
        )
    )
    await session.commit()
    row = (await session.execute(select(EventContract))).scalar_one()
    assert row.measure is Measure.CPI_MOM and row.strike_lo == Decimal("0.3000")


async def test_unmapped_contract_allows_null_terms(session: AsyncSession) -> None:
    session.add(_contract(unmapped_reason="no_release"))
    await session.commit()
    row = (await session.execute(select(EventContract))).scalar_one()
    assert row.release_id is None and row.measure is None and row.strike_lo is None


async def test_venue_market_id_unique_per_venue(session: AsyncSession) -> None:
    session.add_all([_contract("M1", Venue.KALSHI), _contract("M1", Venue.POLYMARKET)])
    await session.commit()
    session.add(_contract("M1", Venue.KALSHI))
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_observation_vintages_coexist(session: AsyncSession) -> None:
    for vintage in (datetime(2026, 11, 1, 5, tzinfo=UTC), datetime(2026, 12, 1, 5, tzinfo=UTC)):
        session.add(
            DataObservation(
                source=DataSource.FRED,
                series_id="CPIAUCSL",
                obs_date=date(2026, 10, 1),
                vintage_ts=vintage,
                published_on=vintage.date(),
                value=Decimal("320.123456"),
                ingested_at=NOW,
            )
        )
    await session.commit()
    rows = (await session.execute(select(DataObservation))).scalars().all()
    assert len(rows) == 2


async def test_snapshot_round_trip_with_empty_side(session: AsyncSession) -> None:
    contract = _contract()
    session.add(contract)
    await session.flush()
    session.add(
        MarketSnapshot(
            contract_id=contract.id,
            ts=NOW,
            best_bid=None,
            best_ask=Decimal("0.42000"),
            ask_size=Decimal("10.00"),
            book={"bids": [], "asks": [["0.42", "10"]]},
        )
    )
    await session.commit()
    row = (await session.execute(select(MarketSnapshot))).scalar_one()
    assert row.best_bid is None and row.best_ask == Decimal("0.42000")
    assert row.book == {"bids": [], "asks": [["0.42", "10"]]}


async def test_snapshot_naive_ts_rejected(session: AsyncSession) -> None:
    contract = _contract()
    session.add(contract)
    await session.flush()
    session.add(MarketSnapshot(contract_id=contract.id, ts=datetime(2026, 11, 10, 12, 0), book={}))
    with pytest.raises(StatementError, match="timezone"):
        await session.commit()


@pytest.mark.parametrize("confidence", [Decimal("1.01"), Decimal("-0.01"), Decimal("9.99")])
async def test_mapping_confidence_must_be_in_unit_range(
    session: AsyncSession, confidence: Decimal
) -> None:
    session.add(_contract(mapping_confidence=confidence))
    with pytest.raises(IntegrityError, match="ck_event_contracts_confidence_range"):
        await session.commit()


@pytest.mark.parametrize("confidence", [Decimal("0"), Decimal("0.5"), Decimal("1")])
async def test_mapping_confidence_bounds_inclusive(
    session: AsyncSession, confidence: Decimal
) -> None:
    session.add(_contract(mapping_confidence=confidence))
    await session.commit()


async def test_snapshot_sizes_keep_six_decimals_and_venue_ts(session: AsyncSession) -> None:
    contract = _contract()
    session.add(contract)
    await session.flush()
    size = Decimal("123456789012345678.123456")  # Numeric(24, 6) upper range
    session.add(
        MarketSnapshot(
            contract_id=contract.id,
            ts=NOW,
            venue_ts=NOW,
            bid_size=size,
            ask_size=Decimal("0.000001"),
            book={},
        )
    )
    await session.commit()
    row = (await session.execute(select(MarketSnapshot))).scalar_one()
    assert (row.bid_size, row.ask_size, row.venue_ts) == (size, Decimal("0.000001"), NOW)

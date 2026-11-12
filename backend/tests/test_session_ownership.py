"""Ruling R9: batch services own short sessions, never hold a transaction across HTTP, roll
back on failure, and leave no connection idle in a transaction."""

import json
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from prediction_desk.audit.log import AuditLog
from prediction_desk.calendar.schedule import sync_releases
from prediction_desk.clock import SimClock
from prediction_desk.db.models import (
    AuditEvent,
    DataObservation,
    EventContract,
    MarketSnapshot,
    Release,
)
from prediction_desk.domain.enums import Comparator, MappingStatus, Measure, Series, Venue
from prediction_desk.ingestion.alpha_vantage import AlphaVantageClient
from prediction_desk.ingestion.fred import FredClient
from prediction_desk.ingestion.store import ingest_alpha_vantage, ingest_fred_series
from prediction_desk.markets.discovery import discover_contracts
from prediction_desk.markets.kalshi import KalshiAdapter
from prediction_desk.markets.snapshots import record_snapshots
from prediction_desk.net.http import ProviderError, ProviderHttpError, ResilientHttp

NOW = datetime(2026, 11, 20, tzinfo=UTC)
FIXTURES = Path(__file__).parent / "ingestion" / "fixtures"
KEY = "test-key-not-real"
CPI_RULES = "If CPI increases by more than 0.3% (single-decimal) in November 2026"

Handler = Callable[[httpx.Request], httpx.Response]


def _guarded(engine: AsyncEngine, handler: Handler, calls: list[str]) -> httpx.MockTransport:
    """Every request asserts that no pooled connection (hence no transaction) is held."""

    def wrapped(req: httpx.Request) -> httpx.Response:
        calls.append(req.url.path)
        assert engine.pool.checkedout() == 0, f"connection held during HTTP {req.url.path}"  # type: ignore[attr-defined]
        return handler(req)

    return httpx.MockTransport(wrapped)


def _http(provider: str, transport: httpx.MockTransport) -> ResilientHttp:
    return ResilientHttp(provider, "https://example.test", SimClock(NOW), transport=transport)


async def _assert_clean(engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession]) -> None:
    """The run closed its sessions and left nothing idle in a transaction."""
    assert engine.pool.checkedout() == 0  # type: ignore[attr-defined]
    async with sessions() as s:
        assert (await s.execute(text("SELECT 1"))).scalar_one() == 1
        stuck = await s.scalar(
            text(
                "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() "
                "AND pid <> pg_backend_pid() AND state LIKE 'idle in transaction%'"
            )
        )
        assert stuck == 0


async def _count(sessions: async_sessionmaker[AsyncSession], model: Any) -> int:
    async with sessions() as s:
        return int(await s.scalar(select(func.count()).select_from(model)) or 0)


# ---- sync_releases ---------------------------------------------------------------------------


async def test_sync_releases_fails_mid_run_without_partial_writes(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession]
) -> None:
    calls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.params["release_id"] == "10":
            return httpx.Response(
                200, content=(FIXTURES / "fred_release_dates_10.json").read_bytes()
            )
        return httpx.Response(400, json={"error_message": "boom"})

    fred = FredClient(_http("fred", _guarded(engine, handler, calls)), KEY)
    with pytest.raises(ProviderHttpError):
        await sync_releases(sessions, fred, since=date(2024, 1, 1))
    assert len(calls) == 2  # CPI fetched fine, NFP failed: neither may be half-written
    assert await _count(sessions, Release) == 0
    await _assert_clean(engine, sessions)


async def test_sync_releases_no_connection_during_http(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession]
) -> None:
    calls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        name = "fred_release_dates_10.json"
        if req.url.params["release_id"] == "50":
            name = "fred_release_dates_empty.json"
        return httpx.Response(200, content=(FIXTURES / name).read_bytes())

    fred = FredClient(_http("fred", _guarded(engine, handler, calls)), KEY)
    result = await sync_releases(sessions, fred, since=date(2024, 1, 1))
    assert result.inserted > 0 and len(calls) == 2
    await _assert_clean(engine, sessions)


# ---- ingestion -------------------------------------------------------------------------------


async def test_ingest_fred_fails_without_writes(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession]
) -> None:
    calls: list[str] = []
    handler: Handler = lambda req: httpx.Response(400, json={})  # noqa: E731
    fred = FredClient(_http("fred", _guarded(engine, handler, calls)), KEY)
    with pytest.raises(ProviderHttpError):
        await ingest_fred_series(sessions, fred, SimClock(NOW), "CPIAUCSL", date(2024, 1, 1))
    assert calls
    assert await _count(sessions, DataObservation) == 0
    await _assert_clean(engine, sessions)


async def test_ingest_fred_writes_after_http(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession]
) -> None:
    calls: list[str] = []
    body = (FIXTURES / "fred_observations_cpiaucsl.json").read_bytes()
    handler: Handler = lambda req: httpx.Response(200, content=body)  # noqa: E731
    fred = FredClient(_http("fred", _guarded(engine, handler, calls)), KEY)
    assert await ingest_fred_series(sessions, fred, SimClock(NOW), "X", date(2024, 1, 1)) == 4
    await _assert_clean(engine, sessions)


async def test_ingest_alpha_vantage_reads_then_fetches_outside_txn(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession]
) -> None:
    calls: list[str] = []
    ok = (FIXTURES / "alpha_vantage_cpi.json").read_bytes()
    limited = (FIXTURES / "alpha_vantage_rate_limit.json").read_bytes()
    bodies = [ok, limited]
    handler: Handler = lambda req: httpx.Response(200, content=bodies.pop(0))  # noqa: E731
    clock = SimClock(NOW)
    av = AlphaVantageClient(_http("alpha_vantage", _guarded(engine, handler, calls)), KEY)
    assert await ingest_alpha_vantage(sessions, av, clock, "CPI:monthly") == 3
    clock.advance(timedelta(days=2))
    with pytest.raises(ProviderError):
        await ingest_alpha_vantage(sessions, av, clock, "CPI:monthly")
    assert len(calls) == 2
    assert await _count(sessions, DataObservation) == 3
    await _assert_clean(engine, sessions)


# ---- discovery -------------------------------------------------------------------------------


def _kalshi_market(ticker: str, event: str, series: str) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "event_ticker": event,
        "title": "t",
        "rules_primary": CPI_RULES,
        "close_time": "2026-12-10T13:29:00Z",
        "status": "active",
        "strike_type": "greater",
        "floor_strike": 0.3,
    }


async def test_discovery_provider_failure_mid_run_is_isolated(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession]
) -> None:
    async with sessions() as s, s.begin():
        s.add(
            Release(
                series=Series.CPI,
                period="2026-11",
                scheduled_at=datetime(2026, 12, 10, 13, 30, tzinfo=UTC),
            )
        )
    calls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        series = req.url.params["series_ticker"]
        if series == "KXCPICORE":
            return httpx.Response(503)
        market = _kalshi_market("KXCPI-26NOV-T0.3", "KXCPI-26NOV", series)
        return httpx.Response(200, content=json.dumps({"markets": [market], "cursor": ""}))

    clock = SimClock(NOW)
    http = ResilientHttp(
        "kalshi",
        "https://example.test",
        clock,
        transport=_guarded(engine, handler, calls),
        max_attempts=1,
    )
    adapter = KalshiAdapter(http, clock)
    result = await discover_contracts(
        sessions, {Venue.KALSHI: (adapter, ("KXCPI", "KXCPICORE"))}, AuditLog(clock), clock
    )
    assert [code for code, _ in result.failures] == ["KXCPICORE"]
    assert result.failures[0][1].startswith("ProviderUnavailable: ")
    assert result.mapped == 1
    assert await _count(sessions, EventContract) == 1
    assert await _count(sessions, AuditEvent) == 1
    await _assert_clean(engine, sessions)


# ---- snapshots -------------------------------------------------------------------------------


async def _mapped_contract(sessions: async_sessionmaker[AsyncSession], market_id: str) -> int:
    async with sessions() as s, s.begin():
        release = Release(series=Series.CPI, period=f"p-{market_id}", scheduled_at=NOW)
        s.add(release)
        await s.flush()
        row = EventContract(
            venue=Venue.KALSHI,
            venue_market_id=market_id,
            venue_event_id="E",
            title="t",
            rules_text="r",
            close_time=NOW + timedelta(days=1),
            release_id=release.id,
            measure=Measure.CPI_MOM,
            comparator=Comparator.GT,
            strike_lo=Decimal("0.3"),
            unit="pct",
            resolution_precision=1,
            resolution_source="BLS",
            mapping_status=MappingStatus.MAPPED,
            mapping_confidence=Decimal("1"),
            mapped_by="rules:v1",
            created_at=NOW,
            updated_at=NOW,
        )
        s.add(row)
        await s.flush()
        return row.id


async def test_snapshots_provider_failure_mid_run_keeps_other_books(
    engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession]
) -> None:
    good = await _mapped_contract(sessions, "GOOD")
    bad = await _mapped_contract(sessions, "BAD")
    calls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if "BAD" in req.url.path:
            return httpx.Response(200, content=json.dumps({"orderbook": {}}))
        book = {"orderbook_fp": {"yes_dollars": [["0.40", "1"]], "no_dollars": None}}
        return httpx.Response(200, content=json.dumps(book))

    clock = SimClock(NOW)
    adapter = KalshiAdapter(_http("kalshi", _guarded(engine, handler, calls)), clock)
    result = await record_snapshots(sessions, {Venue.KALSHI: adapter}, clock)
    assert result.recorded == 1
    assert result.failures == [(bad, "ProviderError: kalshi: malformed order book: KeyError")]
    async with sessions() as s:
        snaps = (await s.execute(select(MarketSnapshot.contract_id))).scalars().all()
    assert snaps == [good]
    assert len(calls) == 2
    await _assert_clean(engine, sessions)

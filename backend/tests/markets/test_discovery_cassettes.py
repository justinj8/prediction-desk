"""Real recorded venue data through the real adapters and discover_contracts, against the DB."""

from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from prediction_desk.audit.log import AuditLog
from prediction_desk.calendar.schedule import load_fomc_meetings, scheduled_at
from prediction_desk.clock import SimClock
from prediction_desk.db.models import AuditEvent, EventContract, Release
from prediction_desk.domain.enums import MappingStatus, Series, Venue
from prediction_desk.markets.discovery import discover_contracts
from prediction_desk.markets.kalshi import KalshiAdapter
from prediction_desk.markets.polymarket import POLYMARKET_TAGS, PolymarketAdapter
from prediction_desk.net.http import ResilientHttp

CASSETTES = Path(__file__).parent / "cassettes"
RECORDED = [
    str(CASSETTES / "test_kalshi" / "test_lists_open_kxcpi_markets.yaml"),
    str(CASSETTES / "test_kalshi" / "test_lists_fed_decision_custom_strikes.yaml"),
    str(
        CASSETTES / "test_polymarket" / "test_lists_only_single_meeting_fed_and_us_cpi_events.yaml"
    ),
]
NOW = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)

# CPI releases the recorded markets resolve on, as (period, BLS release day):
CPI_RELEASES = (
    # Polymarket October events close 2026-11-11 04:59Z; their rules name the October report
    # "scheduled to be released on November 10, 2026, at 8:30 AM ET".
    ("2026-10", date(2026, 11, 10)),
    # KXCPI-26NOV closes 2026-12-10 13:29Z, one minute before the 08:30 ET release that day.
    ("2026-11", date(2026, 12, 10)),
    # KXCPI-26DEC closes 2027-01-13 13:29Z, one minute before the 08:30 ET release that day.
    ("2026-12", date(2027, 1, 13)),
)
# The packaged FOMC calendar ends at 2027-12-08 (ruling R12), so the January 2028 meeting is
# the one documented exception: unmapped with reason no_release.
NO_RELEASE_EVENT = "KXFEDDECISION-28JAN"


async def _seed(sessions: async_sessionmaker[AsyncSession]) -> None:
    async with sessions() as s, s.begin():
        for meeting in load_fomc_meetings():
            s.add(
                Release(
                    series=Series.FOMC,
                    period=meeting.decision_date.isoformat(),
                    scheduled_at=meeting.decision_at,
                )
            )
        for period, day in CPI_RELEASES:
            s.add(
                Release(
                    series=Series.CPI, period=period, scheduled_at=scheduled_at(Series.CPI, day)
                )
            )


@pytest.mark.vcr(*RECORDED)
async def test_recorded_markets_discover_and_link(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    await _seed(sessions)
    clock = SimClock(NOW)
    kalshi = KalshiAdapter(
        ResilientHttp("kalshi", "https://api.elections.kalshi.com/trade-api/v2", clock), clock
    )
    polymarket = PolymarketAdapter(
        ResilientHttp("polymarket-gamma", "https://gamma-api.polymarket.com", clock),
        ResilientHttp("polymarket-clob", "https://clob.polymarket.com", clock),
        clock,
    )
    result = await discover_contracts(
        sessions,
        {
            Venue.KALSHI: (kalshi, ("KXCPI", "KXFEDDECISION")),
            Venue.POLYMARKET: (polymarket, POLYMARKET_TAGS),
        },
        AuditLog(clock),
        clock,
    )

    assert result.failures == []
    assert (result.mapped, result.unmapped, result.changed) == (107, 5, 112)
    rows = (await session.execute(select(EventContract))).scalars().all()
    assert len(rows) == 112
    unmapped = {r.venue_market_id: r.unmapped_reason for r in rows if r.release_id is None}
    assert len(unmapped) == 5
    assert all(mid.startswith(f"{NO_RELEASE_EVENT}-") for mid in unmapped)
    assert set(unmapped.values()) == {"no_release"}
    by_venue = {
        venue: sum(r.mapping_status is MappingStatus.MAPPED and r.venue is venue for r in rows)
        for venue in Venue
    }
    assert by_venue == {Venue.KALSHI: 65, Venue.POLYMARKET: 42}

    releases = {r.id: r for r in (await session.execute(select(Release))).scalars()}
    for r in rows:
        if r.release_id is not None:
            assert (
                abs(releases[r.release_id].scheduled_at - r.close_time).total_seconds() <= 48 * 3600
            )
    assert await session.scalar(select(func.count()).select_from(AuditEvent)) == 112
    assert (await AuditLog(clock).verify(session)).ok

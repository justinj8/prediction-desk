from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from prediction_desk.calendar.schedule import (
    FomcMeeting,
    SyncResult,
    load_fomc_meetings,
    parse_fomc_meetings,
    scheduled_at,
    sync_releases,
)
from prediction_desk.clock import SimClock
from prediction_desk.db.models import DataObservation, Release
from prediction_desk.domain.enums import DataSource, Series
from prediction_desk.ingestion.fred import FredClient
from prediction_desk.net.http import ResilientHttp

T0 = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
BASE = "https://api.stlouisfed.org/fred"
KEY = "test-key-not-real"
SINCE = date(2024, 1, 1)
FIXTURES = Path(__file__).parents[1] / "ingestion" / "fixtures"


def _fred(by_release: dict[str, str], requests: list[httpx.Request] | None = None) -> FredClient:
    """FRED stub: release/dates answers from hand-built fixtures keyed by release_id."""

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/fred/release/dates"
        if requests is not None:
            requests.append(req)
        rid = req.url.params["release_id"]
        return httpx.Response(200, content=(FIXTURES / by_release[rid]).read_bytes())

    http = ResilientHttp("fred", BASE, SimClock(T0), transport=httpx.MockTransport(handler))
    return FredClient(http, KEY)


def _obs(series_id: str, obs: date, published_on: date) -> DataObservation:
    return DataObservation(
        source=DataSource.FRED,
        series_id=series_id,
        obs_date=obs,
        vintage_ts=datetime(published_on.year, published_on.month, published_on.day, 5, tzinfo=UTC),
        published_on=published_on,
        value=1,
        ingested_at=T0,
    )


async def _releases(session: AsyncSession, series: Series) -> list[Release]:
    rows = await session.execute(
        select(Release).where(Release.series == series).order_by(Release.scheduled_at)
    )
    return list(rows.scalars())


NOV_20 = {"10": "fred_release_dates_2025_11_20.json", "50": "fred_release_dates_empty.json"}


def test_fomc_yaml_loads_32_meetings_utc() -> None:
    meetings = load_fomc_meetings()
    assert len(meetings) == 32
    by_date = {m.decision_date: m for m in meetings}
    assert by_date[date(2026, 12, 9)].decision_at == datetime(2026, 12, 9, 19, 0, tzinfo=UTC)
    assert by_date[date(2026, 9, 16)].decision_at == datetime(2026, 9, 16, 18, 0, tzinfo=UTC)
    assert isinstance(meetings[0], FomcMeeting)


def test_fomc_yaml_is_packaged_resource() -> None:
    from importlib.resources import files

    assert files("prediction_desk.calendar").joinpath("fomc_meetings.yaml").is_file()


@pytest.mark.parametrize(
    "body",
    [
        "decision_time: '14:00'\ntz: America/New_York\nmeetings: []\n",
        "decision_time: '14:00'\ntz: America/New_York\nmeetings:\n"
        "  - decision_date: '2026-03-18'\n  - decision_date: '2026-03-18'\n",
        "decision_time: '14:00'\ntz: America/New_York\nmeetings:\n"
        "  - decision_date: '2026-04-29'\n  - decision_date: '2026-03-18'\n",
    ],
)
def test_fomc_yaml_validation_rejects_bad_input(body: str) -> None:
    with pytest.raises(ValueError):
        parse_fomc_meetings(body)


def test_scheduled_at_cpi_0830_et() -> None:
    assert scheduled_at(Series.CPI, date(2026, 12, 10)) == datetime(
        2026, 12, 10, 13, 30, tzinfo=UTC
    )
    assert scheduled_at(Series.NFP, date(2026, 7, 2)) == datetime(2026, 7, 2, 12, 30, tzinfo=UTC)


def test_scheduled_at_fomc_agrees_with_meeting_decision_at() -> None:
    for m in load_fomc_meetings():
        assert scheduled_at(Series.FOMC, m.decision_date) == m.decision_at


async def test_sync_uses_data_driven_period(session: AsyncSession) -> None:
    session.add_all(
        [
            _obs("CPIAUCSL", date(2025, 8, 1), date(2025, 11, 20)),
            _obs("CPIAUCSL", date(2025, 9, 1), date(2025, 11, 20)),
            # Different series and different publish date must not influence the period.
            _obs("PAYEMS", date(2025, 10, 1), date(2025, 11, 20)),
            _obs("CPIAUCSL", date(2025, 10, 1), date(2025, 12, 18)),
        ]
    )
    await session.commit()
    requests: list[httpx.Request] = []

    result = await sync_releases(session, _fred(NOV_20, requests), since=SINCE)

    cpi = await _releases(session, Series.CPI)
    assert [(r.period, r.scheduled_at) for r in cpi] == [
        ("2025-09", datetime(2025, 11, 20, 13, 30, tzinfo=UTC))
    ]
    assert result.inserted == 1 + 32 and result.updated == 0
    assert await _releases(session, Series.NFP) == []
    assert {r.url.params["release_id"] for r in requests} == {"10", "50"}
    for r in requests:
        p = r.url.params
        assert p["realtime_start"] == "2024-01-01"
        assert p["realtime_end"] == "9999-12-31"
        assert p["include_release_dates_with_no_data"] == "true"
        assert p["file_type"] == "json"
        assert p["api_key"] == KEY


async def test_sync_future_release_assumes_previous_month(session: AsyncSession) -> None:
    files = {"10": "fred_release_dates_10.json", "50": "fred_release_dates_empty.json"}
    await sync_releases(session, _fred(files), since=SINCE)

    cpi = await _releases(session, Series.CPI)
    assert [r.period for r in cpi] == ["2024-01", "2024-02", "2024-03"]
    assert cpi[0].scheduled_at == datetime(2024, 2, 13, 13, 30, tzinfo=UTC)
    # 2024-03-12 is EDT (DST began 2024-03-10): 08:30 ET == 12:30 UTC.
    assert cpi[1].scheduled_at == datetime(2024, 3, 12, 12, 30, tzinfo=UTC)
    assert cpi[2].scheduled_at == datetime(2024, 4, 10, 12, 30, tzinfo=UTC)


async def test_sync_since_filters_release_dates(session: AsyncSession) -> None:
    files = {"10": "fred_release_dates_10.json", "50": "fred_release_dates_empty.json"}
    await sync_releases(session, _fred(files), since=date(2024, 3, 12))

    assert [r.period for r in await _releases(session, Series.CPI)] == ["2024-02", "2024-03"]
    assert all(
        r.scheduled_at.date() >= date(2024, 3, 12) for r in await _releases(session, Series.FOMC)
    )


async def test_sync_is_idempotent(session: AsyncSession) -> None:
    fred = _fred(NOV_20)
    first = await sync_releases(session, fred, since=SINCE)
    second = await sync_releases(session, fred, since=SINCE)

    assert first == SyncResult(inserted=33, updated=0)
    assert second == SyncResult(inserted=0, updated=0)


async def test_sync_fomc_rows_one_per_meeting(session: AsyncSession) -> None:
    await sync_releases(session, _fred(NOV_20), since=SINCE)
    fomc = await _releases(session, Series.FOMC)
    assert len(fomc) == 32
    assert fomc[0].period == "2024-01-31"
    assert fomc[0].scheduled_at == datetime(2024, 1, 31, 19, 0, tzinfo=UTC)


async def test_sync_corrects_assumed_period(session: AsyncSession) -> None:
    fred = _fred(NOV_20)
    await sync_releases(session, fred, since=SINCE)
    (assumed,) = await _releases(session, Series.CPI)
    assert assumed.period == "2025-10"
    original_id = assumed.id

    session.add(_obs("CPIAUCSL", date(2025, 9, 1), date(2025, 11, 20)))
    await session.commit()
    result = await sync_releases(session, fred, since=SINCE)

    assert result == SyncResult(inserted=0, updated=1)
    (fixed,) = await _releases(session, Series.CPI)
    assert fixed.id == original_id
    assert fixed.period == "2025-09"


async def test_sync_does_not_rewrite_period_once_actual_value_recorded(
    session: AsyncSession,
) -> None:
    fred = _fred(NOV_20)
    await sync_releases(session, fred, since=SINCE)
    (row,) = await _releases(session, Series.CPI)
    row.actual_value = 1  # type: ignore[assignment]
    session.add(_obs("CPIAUCSL", date(2025, 9, 1), date(2025, 11, 20)))
    await session.commit()

    result = await sync_releases(session, fred, since=SINCE)

    assert result == SyncResult(inserted=1, updated=0)
    assert sorted(r.period for r in await _releases(session, Series.CPI)) == ["2025-09", "2025-10"]


async def test_sync_updates_scheduled_at_when_it_differs(session: AsyncSession) -> None:
    session.add(
        Release(
            series=Series.CPI, period="2025-10", scheduled_at=datetime(2025, 11, 20, tzinfo=UTC)
        )
    )
    await session.commit()

    result = await sync_releases(session, _fred(NOV_20), since=SINCE)

    assert result == SyncResult(inserted=32, updated=1)
    (row,) = await _releases(session, Series.CPI)
    assert row.scheduled_at == datetime(2025, 11, 20, 13, 30, tzinfo=UTC)

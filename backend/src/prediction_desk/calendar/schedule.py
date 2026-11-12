"""Release calendar: FOMC decision dates (packaged YAML) plus FRED release dates for CPI/NFP.

Rows in ``releases`` are data, not decisions, so syncing writes no audit events. Every change
to an existing row's ``scheduled_at`` or ``period`` is logged at WARNING, and a row that already
has an ``actual_value`` never has its ``scheduled_at`` moved.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from importlib.resources import files
from itertools import pairwise
from typing import Any
from zoneinfo import ZoneInfo

import yaml
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from prediction_desk.db.models import DataObservation, Release
from prediction_desk.domain.enums import DataSource, Series
from prediction_desk.ingestion.fred import FRED_RELEASE_IDS, FredClient

_NY = ZoneInfo("America/New_York")
_RELEASE_TIME: dict[Series, time] = {
    Series.CPI: time(8, 30),
    Series.NFP: time(8, 30),
    Series.FOMC: time(14, 0),
}
_ANCHOR_SERIES: dict[Series, str] = {Series.CPI: "CPIAUCSL", Series.NFP: "PAYEMS"}
_FOMC_YAML = "fomc_meetings.yaml"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FomcMeeting:
    decision_date: date
    decision_at: datetime  # UTC


@dataclass
class SyncResult:
    inserted: int
    updated: int


def scheduled_at(series: Series, day: date) -> datetime:
    """Local release time on ``day`` (New York), as UTC."""
    return datetime.combine(day, _RELEASE_TIME[series], tzinfo=_NY).astimezone(UTC)


def parse_fomc_meetings(text: str) -> list[FomcMeeting]:
    raw: Any = yaml.safe_load(text)
    try:
        decision_time = time.fromisoformat(str(raw["decision_time"]))
        tz = ZoneInfo(str(raw["tz"]))
        days = [date.fromisoformat(str(m["decision_date"])) for m in raw["meetings"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed FOMC calendar: {exc!r}") from None
    if not days:
        raise ValueError("FOMC calendar has no meetings")
    if any(a >= b for a, b in pairwise(days)):
        raise ValueError("FOMC calendar dates must be unique and strictly ascending")
    return [
        FomcMeeting(d, datetime.combine(d, decision_time, tzinfo=tz).astimezone(UTC)) for d in days
    ]


def load_fomc_meetings() -> list[FomcMeeting]:
    text = files("prediction_desk.calendar").joinpath(_FOMC_YAML).read_text(encoding="utf-8")
    return parse_fomc_meetings(text)


def _assumed_period(release_day: date) -> str:
    prev = release_day.replace(day=1) - timedelta(days=1)
    return f"{prev.year:04d}-{prev.month:02d}"


async def _data_periods(session: AsyncSession, series_id: str, release_day: date) -> set[str]:
    """'YYYY-MM' of every anchor observation FRED says was published on ``release_day``."""
    days = await session.scalars(
        select(DataObservation.obs_date)
        .distinct()
        .where(
            DataObservation.source == DataSource.FRED,
            DataObservation.series_id == series_id,
            DataObservation.published_on == release_day,
        )
    )
    return {f"{d.year:04d}-{d.month:02d}" for d in days}


async def sync_releases(
    sessions: async_sessionmaker[AsyncSession], fred: FredClient, *, since: date
) -> SyncResult:
    """Upsert CPI/NFP releases from FRED release dates and FOMC decisions from the YAML.

    Raises ``ProviderError`` if any FRED call fails, and then writes nothing: every release
    date is fetched before the single write transaction opens, so a run never half-syncs.
    """
    fetched: dict[Series, list[date]] = {}
    for series in (Series.CPI, Series.NFP):
        fetched[series] = await fred.release_dates(FRED_RELEASE_IDS[series], since)
    async with sessions() as session, session.begin():
        return await _write_releases(session, fetched, since)


async def _write_releases(
    session: AsyncSession, fetched: dict[Series, list[date]], since: date
) -> SyncResult:
    result = SyncResult(inserted=0, updated=0)

    for series, days in fetched.items():
        existing = {
            r.period: r
            for r in (await session.scalars(select(Release).where(Release.series == series)))
        }
        for day in days:
            if day < since:
                continue
            supported = await _data_periods(session, _ANCHOR_SERIES[series], day)
            period = max(supported) if supported else _assumed_period(day)
            _upsert(session, existing, series, period, day, supported, result)

    existing = {
        r.period: r
        for r in (await session.scalars(select(Release).where(Release.series == Series.FOMC)))
    }
    for meeting in load_fomc_meetings():
        if meeting.decision_date < since:
            continue
        period = meeting.decision_date.isoformat()
        row = existing.get(period)
        if row is None:
            row = Release(series=Series.FOMC, period=period, scheduled_at=meeting.decision_at)
            session.add(row)
            existing[period] = row
            result.inserted += 1
        elif _move_scheduled_at(row, meeting.decision_at):
            result.updated += 1

    return result


def _move_scheduled_at(row: Release, when: datetime) -> bool:
    """Move a row to the calendar's time; never once its actual value is known. Logs both."""
    if row.scheduled_at == when:
        return False
    if row.actual_value is not None:
        logger.warning(
            "release %s %s has actual_value; keeping scheduled_at %s (calendar says %s)",
            row.series.value,
            row.period,
            row.scheduled_at.isoformat(),
            when.isoformat(),
        )
        return False
    logger.warning(
        "release %s %s scheduled_at changed %s -> %s",
        row.series.value,
        row.period,
        row.scheduled_at.isoformat(),
        when.isoformat(),
    )
    row.scheduled_at = when
    return True


def _upsert(
    session: AsyncSession,
    existing: dict[str, Release],
    series: Series,
    period: str,
    day: date,
    supported: set[str],
    result: SyncResult,
) -> None:
    when = scheduled_at(series, day)
    row = existing.get(period)
    if row is not None:
        if _move_scheduled_at(row, when):
            result.updated += 1
        return
    # A row on the same release day whose period no anchor data supports (an earlier assumed
    # period): correct the lowest-id one in place. Rows that anchor data supports are real
    # releases and are never rewritten.
    stale = [
        old
        for old in existing.values()
        if old.actual_value is None
        and old.period not in supported
        and old.scheduled_at.astimezone(_NY).date() == day
    ]
    if stale:
        old = min(stale, key=lambda r: r.id)
        logger.warning(
            "release %s on %s: period %s -> %s", series.value, day.isoformat(), old.period, period
        )
        del existing[old.period]
        old.period = period
        existing[period] = old
        result.updated += 1
        return
    row = Release(series=series, period=period, scheduled_at=when)
    session.add(row)
    existing[period] = row
    result.inserted += 1

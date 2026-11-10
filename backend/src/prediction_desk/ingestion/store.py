"""Persist macro observations: FRED/ALFRED vintages and Alpha Vantage cross-check series.

Rows are immutable per (source, series, obs_date, vintage_ts), so re-runs insert nothing new.
FRED rows carry their ALFRED vintage time; Alpha Vantage has no vintages, so its rows are
stamped with the live fetch time.

The batch services here own their sessions (ruling R9): they take a session factory, never
hold a transaction open across an HTTP call, and roll back on any exception.
"""

from collections.abc import Iterable
from datetime import date, timedelta

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from prediction_desk.clock import Clock
from prediction_desk.db.models import DataObservation
from prediction_desk.domain.enums import DataSource
from prediction_desk.ingestion.alpha_vantage import AlphaVantageClient
from prediction_desk.ingestion.fred import FredClient, fred_vintage_ts

_BATCH = 2000


async def upsert_observations(session: AsyncSession, rows: Iterable[DataObservation]) -> int:
    """INSERT ... ON CONFLICT DO NOTHING; returns the number of rows actually inserted."""
    values = [
        {
            "source": r.source,
            "series_id": r.series_id,
            "obs_date": r.obs_date,
            "vintage_ts": r.vintage_ts,
            "published_on": r.published_on,
            "value": r.value,
            "ingested_at": r.ingested_at,
        }
        for r in rows
    ]
    inserted = 0
    for i in range(0, len(values), _BATCH):
        stmt = (
            insert(DataObservation)
            .values(values[i : i + _BATCH])
            .on_conflict_do_nothing()
            .returning(DataObservation.series_id)
        )
        inserted += len((await session.execute(stmt)).all())
    return inserted


async def ingest_fred_series(
    sessions: async_sessionmaker[AsyncSession],
    client: FredClient,
    clock: Clock,
    series_id: str,
    observation_start: date,
) -> int:
    """Fetch every ALFRED vintage of one series and insert the new rows; returns that count.

    Raises ``ProviderError`` (nothing is written) if the fetch fails. All HTTP happens before
    the single write transaction opens.
    """
    fetched = await client.observations_all_vintages(series_id, observation_start)
    now = clock.now()
    rows = [
        DataObservation(
            source=DataSource.FRED,
            series_id=o.series_id,
            obs_date=o.obs_date,
            vintage_ts=fred_vintage_ts(o.published_on),
            published_on=o.published_on,
            value=o.value,
            ingested_at=now,
        )
        for o in fetched
    ]
    async with sessions() as session, session.begin():
        return await upsert_observations(session, rows)


async def ingest_alpha_vantage(
    sessions: async_sessionmaker[AsyncSession],
    client: AlphaVantageClient,
    clock: Clock,
    series_id: str,
    *,
    min_interval: timedelta = timedelta(hours=24),
) -> int:
    """Fetch one Alpha Vantage series unless stored data is younger than ``min_interval``.

    The free tier allows 25 calls/day, so a fresh vintage short-circuits before any request.
    Raises ``ProviderError`` (``ProviderRateLimited`` on a quota notice) and writes nothing
    if the fetch fails. The freshness read and the write use separate short sessions; no
    transaction is open during the request.

    LIVE writer only: rows are stamped ``vintage_ts = clock.now()``, the fetch time. Calling
    this under a ``SimClock`` set in the past would write today's data into the past and
    leak it into replay; replay must read stored rows, never re-ingest.
    """
    now = clock.now()
    async with sessions() as session:
        latest = await session.scalar(
            select(func.max(DataObservation.vintage_ts)).where(
                DataObservation.source == DataSource.ALPHA_VANTAGE,
                DataObservation.series_id == series_id,
            )
        )
    if latest is not None and latest > now - min_interval:
        return 0
    fetched = await client.series(series_id)
    rows = [
        DataObservation(
            source=DataSource.ALPHA_VANTAGE,
            series_id=series_id,
            obs_date=obs_date,
            vintage_ts=now,
            published_on=None,
            value=value,
            ingested_at=now,
        )
        for obs_date, value in fetched
    ]
    async with sessions() as session, session.begin():
        return await upsert_observations(session, rows)

"""Persist FRED vintages. Rows are immutable per vintage, so re-runs insert nothing."""

from collections.abc import Iterable
from datetime import date

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from prediction_desk.clock import Clock
from prediction_desk.db.models import DataObservation
from prediction_desk.domain.enums import DataSource
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
    session: AsyncSession,
    client: FredClient,
    clock: Clock,
    series_id: str,
    observation_start: date,
) -> int:
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
    inserted = await upsert_observations(session, rows)
    await session.commit()
    return inserted

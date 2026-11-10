"""Point-in-time macro reads: what value was known as of T, never later revisions."""

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import distinct_on
from sqlalchemy.ext.asyncio import AsyncSession

from prediction_desk.db.models import DataObservation
from prediction_desk.domain.enums import DataSource
from prediction_desk.domain.types import require_utc


@dataclass(frozen=True)
class Point:
    obs_date: date
    value: Decimal
    vintage_ts: datetime


class MacroDataProvider:
    async def series_asof(
        self,
        session: AsyncSession,
        source: DataSource,
        series_id: str,
        asof: datetime,
        start: date | None = None,
    ) -> list[Point]:
        asof = require_utc(asof)
        stmt = (
            select(DataObservation.obs_date, DataObservation.value, DataObservation.vintage_ts)
            .ext(distinct_on(DataObservation.obs_date))
            .where(
                DataObservation.source == source,
                DataObservation.series_id == series_id,
                DataObservation.vintage_ts <= asof,
            )
            .order_by(DataObservation.obs_date, DataObservation.vintage_ts.desc())
        )
        if start is not None:
            stmt = stmt.where(DataObservation.obs_date >= start)
        result = await session.execute(stmt)
        return [Point(r.obs_date, r.value, r.vintage_ts) for r in result]

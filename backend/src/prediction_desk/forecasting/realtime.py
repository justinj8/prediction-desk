"""Point-in-time training sets.

For a reference month m, features use only data known just before m was first
published (t_m = first_vintage_ts(m) - 1us), and the target is the first-print
value of m, never today's revised one. This is where look-ahead bias is
prevented.
"""

from collections.abc import Awaitable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Protocol
from zoneinfo import ZoneInfo

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from prediction_desk.db.models import DataObservation
from prediction_desk.domain.enums import DataSource
from prediction_desk.ingestion.provider import MacroDataProvider

_NY = ZoneInfo("America/New_York")
_Q = Decimal("0.000001")
_HUNDRED = Decimal(100)


class InsufficientData(Exception):
    def __init__(self, measure: str, have: int, need: int) -> None:
        super().__init__(f"{measure}: insufficient data ({have} < {need})")
        self.measure = measure
        self.have = have
        self.need = need


async def first_vintage_ts(
    session: AsyncSession, source: DataSource, series_id: str, obs_date: date
) -> datetime | None:
    stmt = select(func.min(DataObservation.vintage_ts)).where(
        DataObservation.source == source,
        DataObservation.series_id == series_id,
        DataObservation.obs_date == obs_date,
    )
    result: datetime | None = (await session.execute(stmt)).scalar_one()
    return result


async def first_print(
    session: AsyncSession, source: DataSource, series_id: str, obs_date: date
) -> Decimal | None:
    stmt = (
        select(DataObservation.value)
        .where(
            DataObservation.source == source,
            DataObservation.series_id == series_id,
            DataObservation.obs_date == obs_date,
        )
        .order_by(DataObservation.vintage_ts)
        .limit(1)
    )
    return (await session.execute(stmt)).scalar_one_or_none()


async def levels_asof(
    provider: MacroDataProvider,
    session: AsyncSession,
    series_id: str,
    asof: datetime,
    start: date,
) -> dict[date, Decimal]:
    points = await provider.series_asof(session, DataSource.FRED, series_id, asof, start)
    return {p.obs_date: p.value for p in points}


def _prev_month(month: date) -> date:
    return date(month.year - 1, 12, 1) if month.month == 1 else date(month.year, month.month - 1, 1)


def pct_change(levels: Mapping[date, Decimal], month: date) -> Decimal | None:
    """100 * (L[m] / L[m-1 month] - 1), quantized to 1e-6; None if not computable."""
    cur = levels.get(month)
    prev = levels.get(_prev_month(month))
    if cur is None or prev is None or prev == 0:
        return None
    return (_HUNDRED * (cur / prev - 1)).quantize(_Q, rounding=ROUND_HALF_EVEN)


def month_mean(points: Mapping[date, Decimal], month: date) -> Decimal | None:
    """Mean of observations dated within month's calendar month; None if none."""
    vals = [v for d, v in points.items() if d.year == month.year and d.month == month.month]
    if not vals:
        return None
    return (sum(vals, Decimal(0)) / len(vals)).quantize(_Q, rounding=ROUND_HALF_EVEN)


@dataclass(frozen=True)
class Sample:
    month: date
    asof: datetime
    features: tuple[Decimal, ...]
    target: Decimal


class FeatureFn(Protocol):
    def __call__(
        self, session: AsyncSession, provider: MacroDataProvider, month: date, asof: datetime, /
    ) -> Awaitable[tuple[Decimal, ...] | None]: ...


class TargetFn(Protocol):
    def __call__(
        self, session: AsyncSession, provider: MacroDataProvider, month: date, /
    ) -> Awaitable[Decimal | None]: ...


def _month_index(month: date) -> int:
    return month.year * 12 + (month.month - 1)


def _from_index(idx: int) -> date:
    return date(idx // 12, idx % 12 + 1, 1)


async def build_samples(
    session: AsyncSession,
    provider: MacroDataProvider,
    *,
    target_series: str,
    target_fn: TargetFn,
    feature_fn: FeatureFn,
    asof: datetime,
    lookback_months: int = 96,
) -> list[Sample]:
    """Ascending point-in-time samples for months first published at or before asof."""
    end = _month_index(asof.astimezone(_NY).date())
    samples: list[Sample] = []
    for idx in range(end - lookback_months, end + 1):
        month = _from_index(idx)
        fv = await first_vintage_ts(session, DataSource.FRED, target_series, month)
        if fv is None or fv > asof:
            continue
        t_m = fv - timedelta(microseconds=1)
        features = await feature_fn(session, provider, month, t_m)
        if features is None:
            continue
        target = await target_fn(session, provider, month)
        if target is None:
            continue
        samples.append(Sample(month=month, asof=t_m, features=features, target=target))
    return samples

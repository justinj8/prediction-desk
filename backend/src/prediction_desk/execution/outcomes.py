"""Official measure outcomes (U1): the FRED first print decides, an admin may override.

``official_raw_value`` computes a measure from the first-vintage index levels at full precision
(no 1e-6 pre-rounding, which could flip a value like 0.2499996 across a grid boundary).
``reported_value`` then rounds it onto the contract grid ROUND_HALF_UP. A venue's own resolution
is only ever compared against these values (settlement), never used to pay out.
"""

from datetime import date, datetime, timedelta
from decimal import (
    ROUND_HALF_EVEN,
    ROUND_HALF_UP,
    Context,
    Decimal,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    localcontext,
)
from typing import Any, Final

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from prediction_desk.audit.log import AuditLog
from prediction_desk.clock import Clock
from prediction_desk.db.models import EventContract, MeasureOutcome, Release, Settlement
from prediction_desk.db.uow import unit_of_work
from prediction_desk.domain.enums import MEASURE_SERIES, Comparator, DataSource, Measure
from prediction_desk.domain.types import require_utc
from prediction_desk.forecasting.realtime import first_vintage_ts, levels_asof
from prediction_desk.ingestion.provider import MacroDataProvider
from prediction_desk.markets.mapping import MEASURE_SPECS, ContractTerms

SOURCE_FIRST_PRINT: Final = "fred_first_print"
SOURCE_OVERRIDE: Final = "admin_override"
OUTCOMES_ACTOR: Final = "system:outcomes"

_FRED = DataSource.FRED
_HUNDRED: Final = Decimal(100)
_ONE: Final = Decimal(1)
_THOUSAND: Final = Decimal(1000)
_STEP_BP: Final = Decimal(25)
_MAX_BP: Final = Decimal(50)
_RAW_QUANT: Final = Decimal("0.00000001")  # measure_outcomes.raw_value is Numeric(20, 8)
_DFEDTARU: Final = "DFEDTARU"

_CONTEXT = Context(
    prec=28, rounding=ROUND_HALF_EVEN, traps=[InvalidOperation, DivisionByZero, Overflow]
)

_CPI_SERIES: Final[dict[Measure, str]] = {
    Measure.CPI_MOM: "CPIAUCSL",
    Measure.CORE_CPI_MOM: "CPILFESL",
    Measure.CPI_YOY: "CPIAUCSL",
    Measure.CORE_CPI_YOY: "CPILFESL",
}
_CPI_LAG_MONTHS: Final[dict[Measure, int]] = {
    Measure.CPI_MOM: 1,
    Measure.CORE_CPI_MOM: 1,
    Measure.CPI_YOY: 12,
    Measure.CORE_CPI_YOY: 12,
}


def reported_value(measure: Measure, raw: Decimal) -> Decimal:
    """Round ``raw`` onto the measure's grid, ties away from zero (ROUND_HALF_UP).

    FOMC snaps to the nearest 25 bp (a tie at +-12.5 goes to +-25) and clamps to [-50, 50], the
    open-ended tails of the outcome grid (-50 means a cut of 50 or more).
    """
    with localcontext(_CONTEXT):
        if measure is Measure.FOMC_CHANGE_BP:
            steps = (raw / _STEP_BP).quantize(_ONE, rounding=ROUND_HALF_UP)
            return max(-_MAX_BP, min(_MAX_BP, steps * _STEP_BP)).quantize(_ONE)
        precision = MEASURE_SPECS[measure][1]
        rounded = raw.quantize(Decimal(1).scaleb(-precision), rounding=ROUND_HALF_UP)
        return rounded.quantize(_ONE) if precision < 0 else rounded


def contract_yes_won(terms: ContractTerms, reported: Decimal) -> bool:
    lo = terms.strike_lo
    if terms.comparator is Comparator.GT:
        return reported > lo
    if terms.comparator is Comparator.GTE:
        return reported >= lo
    if terms.comparator is Comparator.LT:
        return reported < lo
    if terms.comparator is Comparator.LTE:
        return reported <= lo
    if terms.comparator is Comparator.EQ:
        return reported == lo
    # BETWEEN, the only comparator left
    if terms.strike_hi is None:
        raise ValueError("BETWEEN contract has no strike_hi")
    return lo <= reported <= terms.strike_hi


def _shift(month: date, n: int) -> date:
    idx = month.year * 12 + (month.month - 1) + n
    return date(idx // 12, idx % 12 + 1, 1)


def _month_of(period: str) -> date:
    return date(int(period[:4]), int(period[5:7]), 1)


async def _monthly_levels(
    session: AsyncSession,
    provider: MacroDataProvider,
    series_id: str,
    month: date,
    lag_months: int,
    asof: datetime,
) -> tuple[Decimal, Decimal, datetime] | None:
    """(L[m], L[m - lag], first vintage of m) from the index as first published for m."""
    fv = await first_vintage_ts(session, _FRED, series_id, month)
    if fv is None or fv > asof:
        return None
    prev = _shift(month, -lag_months)
    levels = await levels_asof(provider, session, series_id, fv, prev)
    cur, old = levels.get(month), levels.get(prev)
    if cur is None or old is None:
        return None
    return cur, old, fv


async def official_raw_value(
    session: AsyncSession,
    provider: MacroDataProvider,
    release: Release,
    measure: Measure,
    asof: datetime,
) -> tuple[Decimal, datetime] | None:
    """(raw value at full precision, source vintage ts), or None while the data is not out."""
    asof = require_utc(asof)
    if measure in _CPI_SERIES:
        got = await _monthly_levels(
            session,
            provider,
            _CPI_SERIES[measure],
            _month_of(release.period),
            _CPI_LAG_MONTHS[measure],
            asof,
        )
        if got is None:
            return None
        cur, old, fv = got
        with localcontext(_CONTEXT):
            return _HUNDRED * (cur / old - _ONE), fv
    if measure is Measure.NFP_CHANGE:
        got = await _monthly_levels(session, provider, "PAYEMS", _month_of(release.period), 1, asof)
        if got is None:
            return None
        cur, old, fv = got
        with localcontext(_CONTEXT):
            return (cur - old) * _THOUSAND, fv
    # FOMC_CHANGE_BP: target upper bound the day after the decision less the decision day's.
    day = date.fromisoformat(release.period)
    after = day + timedelta(days=1)
    stamp = await first_vintage_ts(session, _FRED, _DFEDTARU, after)
    if stamp is None or stamp > asof:
        return None
    levels = await levels_asof(provider, session, _DFEDTARU, asof, day)
    before_rate, after_rate = levels.get(day), levels.get(after)
    if before_rate is None or after_rate is None:
        return None
    with localcontext(_CONTEXT):
        return (after_rate - before_rate) * _HUNDRED, stamp


def _payload(row: MeasureOutcome | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "raw_value": str(row.raw_value),
        "reported_value": str(row.reported_value),
        "source": row.source,
        "recorded_by": row.recorded_by,
    }


async def _locked_release(session: AsyncSession, release_id: int) -> Release:
    release = (
        await session.execute(select(Release).where(Release.id == release_id).with_for_update())
    ).scalar_one_or_none()
    if release is None:
        raise ValueError(f"unknown release {release_id}")
    return release


async def record_outcomes(
    sessions: async_sessionmaker[AsyncSession],
    provider: MacroDataProvider,
    audit: AuditLog,
    clock: Clock,
    release_id: int,
) -> dict[Measure, Decimal]:
    """Record the first-print outcome of every measure of the release that has data and no row.

    Measures whose data is not out yet are skipped (they stay open and a later call picks them
    up); recorded or overridden outcomes are never overwritten. Returns the reported values
    written.
    """
    now = require_utc(clock.now())
    written: dict[Measure, Decimal] = {}
    async with unit_of_work(sessions) as session:
        release = await _locked_release(session, release_id)
        existing = set(
            (
                await session.execute(
                    select(MeasureOutcome.measure).where(MeasureOutcome.release_id == release_id)
                )
            ).scalars()
        )
        events: list[tuple[Measure, dict[str, Any]]] = []
        for measure, series in MEASURE_SERIES.items():
            if series is not release.series or measure in existing:
                continue
            got = await official_raw_value(session, provider, release, measure, now)
            if got is None:
                continue
            raw, vintage = got
            reported = reported_value(measure, raw)
            stored_raw = raw.quantize(_RAW_QUANT, rounding=ROUND_HALF_EVEN)
            session.add(
                MeasureOutcome(
                    release_id=release_id,
                    measure=measure,
                    raw_value=stored_raw,
                    reported_value=reported,
                    source=SOURCE_FIRST_PRINT,
                    source_vintage_ts=vintage,
                    recorded_by=OUTCOMES_ACTOR,
                    reason=None,
                    recorded_at=now,
                )
            )
            written[measure] = reported
            events.append(
                (
                    measure,
                    {
                        "raw_value": str(stored_raw),
                        "reported_value": str(reported),
                        "source": SOURCE_FIRST_PRINT,
                        "source_vintage_ts": vintage,
                    },
                )
            )
        await session.flush()
        for measure, payload in events:
            await audit.append(
                session,
                actor=OUTCOMES_ACTOR,
                entity="measure_outcome",
                entity_id=f"{release_id}:{measure.value}",
                action="outcome.recorded",
                payload=payload,
            )
    return written


async def override_outcome(
    session: AsyncSession,
    audit: AuditLog,
    clock: Clock,
    *,
    release_id: int,
    measure: Measure,
    raw_value: Decimal,
    actor: str,
    reason: str,
) -> None:
    """Replace (or create) a measure's outcome, on the caller's transaction, before settlement."""
    if not actor.startswith("admin:") or not actor.removeprefix("admin:").strip():
        raise ValueError("actor must be admin:<login>")
    if not reason.strip():
        raise ValueError("an override needs a reason")
    release = await _locked_release(session, release_id)
    if MEASURE_SERIES[measure] is not release.series:
        raise ValueError(f"measure {measure.value} does not belong to release {release_id}")
    settled = (
        await session.execute(
            select(Settlement.id)
            .join(EventContract, EventContract.id == Settlement.contract_id)
            .where(EventContract.release_id == release_id)
            .limit(1)
        )
    ).first()
    if settled is not None:
        raise ValueError(f"release {release_id} has settled positions; outcome is final")
    now = require_utc(clock.now())
    stored_raw = raw_value.quantize(_RAW_QUANT, rounding=ROUND_HALF_EVEN)
    reported = reported_value(measure, raw_value)
    row = await session.get(MeasureOutcome, (release_id, measure), with_for_update=True)
    old = _payload(row)
    if row is None:
        row = MeasureOutcome(release_id=release_id, measure=measure)
        session.add(row)
    row.raw_value = stored_raw
    row.reported_value = reported
    row.source = SOURCE_OVERRIDE
    row.source_vintage_ts = None
    row.recorded_by = actor
    row.reason = reason
    row.recorded_at = now
    await session.flush()
    await audit.append(
        session,
        actor=actor,
        entity="measure_outcome",
        entity_id=f"{release_id}:{measure.value}",
        action="outcome.overridden",
        payload={"old": old, "new": _payload(row), "reason": reason},
    )

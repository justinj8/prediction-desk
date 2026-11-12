"""FOMC base model `fomc-implied-v1`.

Infers the market-implied policy move from the 4-week T-bill (DTB4WK) against the effective
fed funds rate (DFF), net of their historical spread, then blends the interpolated implied
distribution with a "the Fed rarely surprises" prior (a one-hot on the nearest grid point, or
on "no change" when the meeting is too far away for the bill to see it). All arithmetic is
exact Decimal. Outcomes are bp; the tails mean "cut >= 50" and "hike >= 50".
"""

from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_EVEN, Decimal
from math import floor
from zoneinfo import ZoneInfo

from sqlalchemy.ext.asyncio import AsyncSession

from prediction_desk.db.models import Release
from prediction_desk.domain.enums import Measure, Series
from prediction_desk.domain.types import require_utc
from prediction_desk.forecasting.base import ModelOutput
from prediction_desk.forecasting.distributions import Categorical
from prediction_desk.forecasting.realtime import InsufficientData, levels_asof
from prediction_desk.ingestion.provider import MacroDataProvider

_NY = ZoneInfo("America/New_York")
_DFF = "DFF"
_BILL = "DTB4WK"
_RATES_HISTORY_DAYS = 400
MAX_RATE_AGE_DAYS = 7  # ruling R8: stale rates fail closed
_Q = Decimal("0.000001")
_BILL_TENOR_DAYS = 28
_STEP = Decimal(25)
_CAP = Decimal(50)
_HUNDRED = Decimal(100)
_ZERO = Decimal(0)
_ONE = Decimal(1)


def _fmt(x: Decimal) -> str:
    """Diagnostic string at the same 1e-6 quantum as cpi.py (display only)."""
    return str(x.quantize(_Q, rounding=ROUND_HALF_EVEN))


def _median(values: list[Decimal]) -> Decimal:
    """Exact median; the mean of the two middle values for an even count."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _nearest_grid(implied: Decimal) -> Decimal:
    """Nearest multiple of 25; an exact midpoint goes toward zero."""
    q = implied / _STEP
    lo = Decimal(floor(q))
    rem = q - lo
    if rem < Decimal("0.5"):
        return lo * _STEP
    if rem > Decimal("0.5"):
        return (lo + 1) * _STEP
    return min(lo, lo + 1, key=abs) * _STEP


def _interpolate(implied: Decimal) -> dict[Decimal, Decimal]:
    """Split mass linearly between the two neighbouring grid outcomes."""
    lo = Decimal(floor(implied / _STEP)) * _STEP
    if lo >= _CAP:
        return {_CAP: _ONE}
    w_hi = (implied - lo) / _STEP
    return {lo: _ONE - w_hi, lo + _STEP: w_hi}


class FomcModelV1:
    """Market-implied FOMC model.

    Note (flag for Plan 5): `days_to` is measured from the asof ET calendar date, not from the
    latest bill observation date. A weekend bill print is 1-3 days older than asof, so
    `frac_after` is slightly understated (the bill has covered a bit less of the post-meeting
    window than assumed). Rates older than MAX_RATE_AGE_DAYS fail closed (ruling R8).
    """

    series = Series.FOMC
    model_version = "fomc-implied-v1"
    OUTCOMES = (-50, -25, 0, 25, 50)
    PRIOR_WEIGHT = Decimal("0.5")
    SPREAD_LOOKBACK = 250
    MIN_SPREAD_OBS = 20
    MIN_FRAC_AFTER = Decimal("0.25")

    async def predict(
        self,
        session: AsyncSession,
        provider: MacroDataProvider,
        release: Release,
        asof: datetime,
    ) -> ModelOutput:
        asof = require_utc(asof)
        if release.scheduled_at <= asof:
            raise ValueError("release already published as of asof")

        asof_date = asof.astimezone(_NY).date()
        start = asof_date - timedelta(days=_RATES_HISTORY_DAYS)
        dff_pts = await levels_asof(provider, session, _DFF, asof, start)
        bill_pts = await levels_asof(provider, session, _BILL, asof, start)
        if not dff_pts or not bill_pts:
            raise InsufficientData("fomc_rates", 0, 1)
        dff_date, bill_date = max(dff_pts), max(bill_pts)
        age_days = max((asof_date - dff_date).days, (asof_date - bill_date).days)
        if age_days > MAX_RATE_AGE_DAYS:
            raise InsufficientData("fomc_rates_stale", age_days, MAX_RATE_AGE_DAYS)
        dff = dff_pts[dff_date]
        bill = bill_pts[bill_date]

        paired = sorted(dff_pts.keys() & bill_pts.keys())[-self.SPREAD_LOOKBACK :]
        if len(paired) < self.MIN_SPREAD_OBS:
            raise InsufficientData("fomc_spread", len(paired), self.MIN_SPREAD_OBS)
        spread = _median([bill_pts[d] - dff_pts[d] for d in paired])

        days_to = (date.fromisoformat(release.period) - asof_date).days
        if days_to < 0:
            raise ValueError("release period precedes asof date")
        frac_after = Decimal(max(0, _BILL_TENOR_DAYS - days_to)) / Decimal(_BILL_TENOR_DAYS)

        implied_txt = "n/a"
        if frac_after < self.MIN_FRAC_AFTER:
            probs: dict[Decimal, Decimal] = {_ZERO: _ONE}
        else:
            raw = (bill - dff - spread) * _HUNDRED / frac_after
            implied = min(max(raw, -_CAP), _CAP)
            implied_txt = _fmt(implied)
            interp = _interpolate(implied)
            nearest = _nearest_grid(implied)
            keep = _ONE - self.PRIOR_WEIGHT
            probs = {g: keep * p for g, p in interp.items()}
            probs[nearest] = probs.get(nearest, _ZERO) + self.PRIOR_WEIGHT

        dist = Categorical(
            tuple((o, probs.get(Decimal(o), _ZERO)) for o in self.OUTCOMES),
        )
        return ModelOutput(
            model_version=self.model_version,
            distributions={Measure.FOMC_CHANGE_BP: dist},
            n_train=len(paired),
            diagnostics={
                "fomc.dff": _fmt(dff),
                "fomc.bill": _fmt(bill),
                "fomc.spread": _fmt(spread),
                "fomc.days_to": str(days_to),
                "fomc.frac_after": _fmt(frac_after),
                "fomc.implied_bp": implied_txt,
                "fomc.n_spread": str(len(paired)),
            },
        )

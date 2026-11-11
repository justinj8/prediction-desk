"""CPI base model `cpi-ridge-v1`.

MoM measures are ridge regressions on point-in-time features with a Student-t whose scale is
the out-of-sample RMS (floored). YoY measures transform the NSA MoM (the mean exactly):

    YoY = 100 (r - 1) + r * X_nsa,   r = L_nsa[m-1] / L_nsa[m-12],   X_nsa = X_sa + s_m

so the YoY mean is 100 (r - 1) + r * (mu_mom + s_m), with s_m the recent-year mean of
(NSA MoM - SA MoM) for m's calendar month. Ruling R6: sigma_yoy = r * sqrt(sigma_mom^2 + var_s),
var_s being the sample variance of those per-year offsets.
"""

from datetime import date, datetime
from decimal import ROUND_HALF_EVEN, Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from prediction_desk.db.models import Release
from prediction_desk.domain.enums import DataSource, Measure, Series
from prediction_desk.domain.types import require_utc
from prediction_desk.forecasting.base import ModelOutput
from prediction_desk.forecasting.distributions import StudentT
from prediction_desk.forecasting.realtime import (
    FeatureFn,
    InsufficientData,
    Sample,
    TargetFn,
    build_samples,
    first_vintage_ts,
    levels_asof,
    month_mean,
    pct_change,
)
from prediction_desk.forecasting.ridge import fit_ridge, oos_residuals, oos_sigma, predict
from prediction_desk.ingestion.provider import MacroDataProvider

ALPHA = Decimal("1.0")
MIN_TRAIN = 36
OOS_MIN_TRAIN = 24  # ruling R5: within MIN_TRAIN, so >= 12 out-of-sample residuals
SIGMA_WINDOW = 24
SIGMA_FLOOR = Decimal("0.05")  # pct
NU = Decimal("5")
SEASONAL_YEARS = 5
SEASONAL_MIN_YEARS = 3
SEASONAL_SEARCH_YEARS = 10

_FRED = DataSource.FRED
_GAS = "GASREGW"
_PPI = "PPIACO"
_SHELTER = "CUSR0000SAH1"
_Q = Decimal("0.000001")
_Q_RATIO = Decimal("0.000000001")
_HUNDRED = Decimal(100)

# measure -> (SA series, NSA series)
_YOY_SERIES: dict[Measure, tuple[str, str]] = {
    Measure.CPI_YOY: ("CPIAUCSL", "CPIAUCNS"),
    Measure.CORE_CPI_YOY: ("CPILFESL", "CPILFENS"),
}


def _shift(month: date, n: int) -> date:
    idx = month.year * 12 + (month.month - 1) + n
    return date(idx // 12, idx % 12 + 1, 1)


def _parse_period(period: str) -> date:
    return date(int(period[:4]), int(period[5:7]), 1)


def _make_feature_fn(series_id: str, *, with_gas: bool) -> FeatureFn:
    """Features for month m, every read as of `asof` (= t_m when training)."""

    async def feature_fn(
        session: AsyncSession, provider: MacroDataProvider, month: date, asof: datetime, /
    ) -> tuple[Decimal, ...] | None:
        prev = _shift(month, -1)
        own = await levels_asof(provider, session, series_id, asof, _shift(month, -3))
        feats: list[Decimal | None] = [
            pct_change(own, prev),
            pct_change(own, _shift(month, -2)),
        ]
        if with_gas:
            pts = await provider.series_asof(session, _FRED, _GAS, asof, prev)
            monthly: dict[date, Decimal] = {}
            for m in (prev, month):
                mean = month_mean({p.obs_date: p.value for p in pts}, m)
                if mean is not None:
                    monthly[m] = mean
            feats.append(pct_change(monthly, month))
        for sid in (_PPI, _SHELTER):
            lv = await levels_asof(provider, session, sid, asof, _shift(month, -2))
            feats.append(pct_change(lv, prev))
        if any(f is None for f in feats):
            return None
        return tuple(f for f in feats if f is not None)

    return feature_fn


def _make_target_fn(series_id: str) -> TargetFn:
    """First-print MoM: both L[m] and L[m-1] as published in m's first vintage."""

    async def target_fn(
        session: AsyncSession, provider: MacroDataProvider, month: date, /
    ) -> Decimal | None:
        fv = await first_vintage_ts(session, _FRED, series_id, month)
        if fv is None:
            return None
        levels = await levels_asof(provider, session, series_id, fv, _shift(month, -1))
        return pct_change(levels, month)

    return target_fn


def _mom_series(core: bool) -> str:
    return "CPILFESL" if core else "CPIAUCSL"


async def mom_samples(
    session: AsyncSession, provider: MacroDataProvider, *, core: bool, asof: datetime
) -> list[Sample]:
    """Point-in-time training samples for headline (core=False) or core MoM."""
    sid = _mom_series(core)
    return await build_samples(
        session,
        provider,
        target_series=sid,
        target_fn=_make_target_fn(sid),
        feature_fn=_make_feature_fn(sid, with_gas=not core),
        asof=asof,
    )


class CpiModelV1:
    series = Series.CPI
    model_version = "cpi-ridge-v1"

    async def predict(
        self,
        session: AsyncSession,
        provider: MacroDataProvider,
        release: Release,
        asof: datetime,
    ) -> ModelOutput:
        asof = require_utc(asof)
        month = _parse_period(release.period)
        published = await first_vintage_ts(session, _FRED, "CPIAUCSL", month)
        if published is not None and published <= asof:
            raise ValueError("release already published as of asof")
        dists: dict[Measure, StudentT] = {}
        diagnostics: dict[str, str] = {}
        n_trains: list[int] = []

        for measure, core in ((Measure.CPI_MOM, False), (Measure.CORE_CPI_MOM, True)):
            dist, n_train, rms = await self._fit_mom(
                session, provider, measure, core=core, month=month, asof=asof
            )
            dists[measure] = dist
            n_trains.append(n_train)
            diagnostics[f"{measure.value}.n_train"] = str(n_train)
            diagnostics[f"{measure.value}.oos_rms"] = str(rms)

        for yoy, mom in (
            (Measure.CPI_YOY, Measure.CPI_MOM),
            (Measure.CORE_CPI_YOY, Measure.CORE_CPI_MOM),
        ):
            dist_yoy, r, s_m, var_s = await self._yoy(
                session, provider, yoy, dists[mom], month, asof
            )
            dists[yoy] = dist_yoy
            diagnostics[f"{yoy.value}.r"] = str(r)
            diagnostics[f"{yoy.value}.s_m"] = str(s_m)
            diagnostics[f"{yoy.value}.var_s"] = str(var_s)

        return ModelOutput(
            model_version=self.model_version,
            distributions=dists,
            n_train=min(n_trains),
            diagnostics=diagnostics,
        )

    async def _fit_mom(
        self,
        session: AsyncSession,
        provider: MacroDataProvider,
        measure: Measure,
        *,
        core: bool,
        month: date,
        asof: datetime,
    ) -> tuple[StudentT, int, Decimal]:
        samples = await mom_samples(session, provider, core=core, asof=asof)
        if len(samples) < MIN_TRAIN:
            raise InsufficientData(measure.value, len(samples), MIN_TRAIN)
        sid = _mom_series(core)
        x_now = await _make_feature_fn(sid, with_gas=not core)(session, provider, month, asof)
        if x_now is None:
            raise InsufficientData(f"{measure.value}_features", 0, 1)
        X = [s.features for s in samples]
        y = [s.target for s in samples]
        mu = predict(fit_ridge(X, y, ALPHA), x_now)
        residuals = oos_residuals(X, y, ALPHA, OOS_MIN_TRAIN)
        sigma = oos_sigma(residuals, SIGMA_WINDOW, SIGMA_FLOOR)
        rms = oos_sigma(residuals, SIGMA_WINDOW, Decimal(0))
        return StudentT(mu, sigma, NU), len(samples), rms

    async def _yoy(
        self,
        session: AsyncSession,
        provider: MacroDataProvider,
        measure: Measure,
        mom: StudentT,
        month: date,
        asof: datetime,
    ) -> tuple[StudentT, Decimal, Decimal, Decimal]:
        sa_id, nsa_id = _YOY_SERIES[measure]
        start = date(month.year - SEASONAL_SEARCH_YEARS - 1, 1, 1)
        nsa = await levels_asof(provider, session, nsa_id, asof, start)
        sa = await levels_asof(provider, session, sa_id, asof, start)

        offsets: list[Decimal] = []
        for k in range(1, SEASONAL_SEARCH_YEARS + 1):
            ref = date(month.year - k, month.month, 1)
            nsa_pct = pct_change(nsa, ref)
            sa_pct = pct_change(sa, ref)
            if nsa_pct is None or sa_pct is None:
                continue
            offsets.append(nsa_pct - sa_pct)
            if len(offsets) == SEASONAL_YEARS:
                break
        if len(offsets) < SEASONAL_MIN_YEARS:
            raise InsufficientData(f"{measure.value}_seasonal", len(offsets), SEASONAL_MIN_YEARS)
        s_m = (sum(offsets, Decimal(0)) / len(offsets)).quantize(_Q, rounding=ROUND_HALF_EVEN)

        last, base = nsa.get(_shift(month, -1)), nsa.get(_shift(month, -12))
        if last is None or base is None or base == 0:
            have = sum(1 for v in (last, base) if v is not None)
            raise InsufficientData(f"{measure.value}_levels", have, 2)
        r = (last / base).quantize(_Q_RATIO, rounding=ROUND_HALF_EVEN)

        # Ruling R6: the seasonal offset is itself uncertain, so it widens sigma.
        mean_exact = sum(offsets, Decimal(0)) / len(offsets)
        var_s = (
            sum(((o - mean_exact) ** 2 for o in offsets), Decimal(0)) / (len(offsets) - 1)
        ).quantize(_Q, rounding=ROUND_HALF_EVEN)
        sigma = r * (mom.sigma**2 + var_s).quantize(_Q, rounding=ROUND_HALF_EVEN).sqrt()
        mu = _HUNDRED * (r - 1) + r * (mom.mu + s_m)
        return StudentT(mu, sigma, mom.nu), r, s_m, var_s

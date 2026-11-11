"""NFP base model `nfp-ridge-v1`.

Ridge regression on point-in-time features for the first-print monthly change in nonfarm
payrolls (PAYEMS). Everything is fitted in thousands (the unit FRED / BLS publish), with a
Normal whose scale is the out-of-sample RMS (floored at 50k jobs). The output is converted
to JOBS with an exact affine map, because pricing integrates over the BLS 1,000-job grid.
"""

from datetime import date, datetime
from decimal import ROUND_HALF_EVEN, Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from prediction_desk.db.models import Release
from prediction_desk.domain.enums import DataSource, Measure, Series
from prediction_desk.domain.types import require_utc
from prediction_desk.forecasting.base import ModelOutput
from prediction_desk.forecasting.distributions import Normal, affine
from prediction_desk.forecasting.realtime import (
    InsufficientData,
    Sample,
    build_samples,
    first_vintage_ts,
    levels_asof,
    month_mean,
)
from prediction_desk.forecasting.ridge import fit_ridge, oos_residuals, oos_sigma, predict
from prediction_desk.ingestion.provider import MacroDataProvider

_FRED = DataSource.FRED
_PAYEMS = "PAYEMS"
_ICSA = "ICSA"
_ADP = "ADPWNUSNERSA"
_JOLTS = "JTSJOL"
_PERSONS_PER_THOUSAND = Decimal(1000)
_JOBS_PER_THOUSAND = Decimal(1000)
_Q = Decimal("0.000001")
_MEASURE = Measure.NFP_CHANGE.value


def _shift(month: date, n: int) -> date:
    idx = month.year * 12 + (month.month - 1) + n
    return date(idx // 12, idx % 12 + 1, 1)


def _parse_period(period: str) -> date:
    return date(int(period[:4]), int(period[5:7]), 1)


async def _feature_fn(
    session: AsyncSession, provider: MacroDataProvider, month: date, asof: datetime, /
) -> tuple[Decimal, ...] | None:
    """Features (thousands) for month m, every read as of `asof` (= t_m when training)."""
    prev = _shift(month, -1)

    payems = await levels_asof(provider, session, _PAYEMS, asof, _shift(month, -2))
    last, before = payems.get(prev), payems.get(_shift(month, -2))
    payems_chg = None if last is None or before is None else last - before

    weekly_chg: list[Decimal | None] = []
    for sid in (_ICSA, _ADP):  # persons -> thousands
        pts = await levels_asof(provider, session, sid, asof, prev)
        cur, old = month_mean(pts, month), month_mean(pts, prev)
        weekly_chg.append(
            None
            if cur is None or old is None
            else ((cur - old) / _PERSONS_PER_THOUSAND).quantize(_Q, rounding=ROUND_HALF_EVEN)
        )

    jolts = await levels_asof(provider, session, _JOLTS, asof, _shift(month, -3))
    j_now, j_old = jolts.get(_shift(month, -2)), jolts.get(_shift(month, -3))
    jolts_chg = None if j_now is None or j_old is None else j_now - j_old

    feats = (payems_chg, *weekly_chg, jolts_chg)
    if any(f is None for f in feats):
        return None
    return tuple(f for f in feats if f is not None)


async def _target_fn(
    session: AsyncSession, provider: MacroDataProvider, month: date, /
) -> Decimal | None:
    """First-print change: L[m] - L[m-1] as published in m's first vintage."""
    fv = await first_vintage_ts(session, _FRED, _PAYEMS, month)
    if fv is None:
        return None
    prev = _shift(month, -1)
    levels = await levels_asof(provider, session, _PAYEMS, fv, prev)
    cur, old = levels.get(month), levels.get(prev)
    return None if cur is None or old is None else cur - old


async def nfp_samples(
    session: AsyncSession, provider: MacroDataProvider, asof: datetime
) -> list[Sample]:
    """Point-in-time training samples (thousands) for months first published by `asof`."""
    return await build_samples(
        session,
        provider,
        target_series=_PAYEMS,
        target_fn=_target_fn,
        feature_fn=_feature_fn,
        asof=asof,
    )


class NfpModelV1:
    series = Series.NFP
    model_version = "nfp-ridge-v1"
    ALPHA = Decimal("1.0")
    MIN_TRAIN = 36
    OOS_MIN_TRAIN = 24  # ruling R5: within MIN_TRAIN, so >= 12 out-of-sample residuals
    SIGMA_WINDOW = 24
    SIGMA_FLOOR = Decimal("50")  # thousands == 50k jobs

    async def predict(
        self,
        session: AsyncSession,
        provider: MacroDataProvider,
        release: Release,
        asof: datetime,
    ) -> ModelOutput:
        asof = require_utc(asof)
        month = _parse_period(release.period)
        published = await first_vintage_ts(session, _FRED, _PAYEMS, month)
        if published is not None and published <= asof:
            raise ValueError("release already published as of asof")

        samples = await nfp_samples(session, provider, asof)
        if len(samples) < self.MIN_TRAIN:
            raise InsufficientData(_MEASURE, len(samples), self.MIN_TRAIN)
        x_now = await _feature_fn(session, provider, month, asof)
        if x_now is None:
            raise InsufficientData(f"{_MEASURE}_features", 0, 1)

        X = [s.features for s in samples]
        y = [s.target for s in samples]
        mu = predict(fit_ridge(X, y, self.ALPHA), x_now)
        residuals = oos_residuals(X, y, self.ALPHA, self.OOS_MIN_TRAIN)
        sigma = oos_sigma(residuals, self.SIGMA_WINDOW, self.SIGMA_FLOOR)
        rms = oos_sigma(residuals, self.SIGMA_WINDOW, Decimal(0))

        jobs = affine(Normal(mu, sigma), Decimal(0), _JOBS_PER_THOUSAND)
        return ModelOutput(
            model_version=self.model_version,
            distributions={Measure.NFP_CHANGE: jobs},
            n_train=len(samples),
            diagnostics={
                f"{_MEASURE}.n_train": str(len(samples)),
                f"{_MEASURE}.oos_rms": str(rms),
            },
        )

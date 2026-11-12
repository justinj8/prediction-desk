from collections.abc import Awaitable, Callable
from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from prediction_desk.db.models import DataObservation, Release
from prediction_desk.domain.enums import Comparator, DataSource, Measure, Series
from prediction_desk.forecasting.base import BaseModel, ModelOutput
from prediction_desk.forecasting.distributions import Normal
from prediction_desk.forecasting.nfp import NfpModelV1, nfp_samples
from prediction_desk.forecasting.realtime import InsufficientData
from prediction_desk.ingestion.provider import MacroDataProvider
from prediction_desk.markets.mapping import ContractTerms
from prediction_desk.pricing.fair import fair_prob
from synthetic_macro import (
    SERIES_ADP,
    SERIES_ICSA,
    SERIES_JOLTS,
    SERIES_PAYEMS,
    SyntheticNfpTruth,
    add_months,
    nfp_first_vintage_for,
)

D = Decimal
Seeder = Callable[..., Awaitable[SyntheticNfpTruth]]
PROVIDER = MacroDataProvider()
_CONFORMS: BaseModel = NfpModelV1()  # mypy verifies the Protocol
K = D(1000)
NO_US = timedelta(microseconds=1)


def release_for(month: date) -> Release:
    return Release(
        series=Series.NFP,
        period=f"{month.year:04d}-{month.month:02d}",
        scheduled_at=nfp_first_vintage_for(month),
    )


def normal(out: ModelOutput) -> Normal:
    dist = out.distributions[Measure.NFP_CHANGE]
    assert isinstance(dist, Normal)
    return dist


async def predict(session: AsyncSession, truth: SyntheticNfpTruth) -> ModelOutput:
    return await NfpModelV1().predict(session, PROVIDER, release_for(truth.ref_month), truth.asof)


def obs(sid: str, obs_date: date, vintage_ts: datetime, value: str) -> DataObservation:
    return DataObservation(
        source=DataSource.FRED,
        series_id=sid,
        obs_date=obs_date,
        vintage_ts=vintage_ts,
        published_on=vintage_ts.date(),
        value=D(value),
        ingested_at=vintage_ts,
    )


def test_constants() -> None:
    assert D("1.0") == NfpModelV1.ALPHA
    assert NfpModelV1.MIN_TRAIN == 36
    assert NfpModelV1.OOS_MIN_TRAIN == 24
    assert NfpModelV1.SIGMA_WINDOW == 24
    assert D("50") == NfpModelV1.SIGMA_FLOOR  # thousands: 50k jobs
    assert NfpModelV1.model_version == "nfp-ridge-v1"
    assert NfpModelV1.series == Series.NFP


async def test_predict_returns_normal_nfp_change(
    session: AsyncSession, seed_synthetic_nfp: Seeder
) -> None:
    truth = await seed_synthetic_nfp(session)
    out = await predict(session, truth)
    assert out.model_version == "nfp-ridge-v1"
    assert set(out.distributions) == {Measure.NFP_CHANGE}
    assert normal(out).sigma > 0
    assert set(out.diagnostics) == {"nfp_change.n_train", "nfp_change.oos_rms"}
    assert all(isinstance(v, str) for v in out.diagnostics.values())
    assert out.n_train >= NfpModelV1.MIN_TRAIN
    assert out.n_train == int(out.diagnostics["nfp_change.n_train"])


async def test_output_in_jobs_units(session: AsyncSession, seed_synthetic_nfp: Seeder) -> None:
    truth = await seed_synthetic_nfp(session)
    dist = normal(await predict(session, truth))
    # Truth is in thousands; the output must be ~1000x it, in jobs.
    assert abs(dist.mu - K * truth.true_change_k) < D(25_000)
    assert dist.mu > D(100_000)  # an unscaled (thousands) mu would be ~350
    assert dist.sigma >= K * NfpModelV1.SIGMA_FLOOR


async def test_mean_tracks_known_relationship(
    session: AsyncSession, seed_synthetic_nfp: Seeder
) -> None:
    truth = await seed_synthetic_nfp(session)
    dist = normal(await predict(session, truth))
    # The reference month's conditional mean is far from the unconditional mean, so a
    # feature-blind (intercept-only) model cannot pass.
    gap = abs(truth.unconditional_change_k - truth.true_change_k)
    assert gap > D(50)
    assert abs(dist.mu / K - truth.true_change_k) < gap / 4


async def test_sigma_floor_50k(session: AsyncSession, seed_synthetic_nfp: Seeder) -> None:
    truth = await seed_synthetic_nfp(session, noiseless=True)
    out = await predict(session, truth)
    assert normal(out).sigma == D(50_000)
    assert D(out.diagnostics["nfp_change.oos_rms"]) < NfpModelV1.SIGMA_FLOOR


async def test_sigma_above_floor_equals_unfloored_oos_rms(
    session: AsyncSession, seed_synthetic_nfp: Seeder
) -> None:
    truth = await seed_synthetic_nfp(session, noise_sd=80.0)
    out = await predict(session, truth)
    rms = D(out.diagnostics["nfp_change.oos_rms"])
    assert rms > NfpModelV1.SIGMA_FLOOR
    assert normal(out).sigma == K * rms


async def test_insufficient_history_raises(
    session: AsyncSession, seed_synthetic_nfp: Seeder
) -> None:
    truth = await seed_synthetic_nfp(session, months=30)
    with pytest.raises(InsufficientData) as exc:
        await predict(session, truth)
    assert exc.value.measure == "nfp_change"
    assert exc.value.need == NfpModelV1.MIN_TRAIN
    assert exc.value.have < NfpModelV1.MIN_TRAIN


@pytest.mark.parametrize("series_id", [SERIES_ICSA, SERIES_ADP, SERIES_JOLTS])
async def test_missing_feature_series_skips_samples_then_raises(
    session: AsyncSession, seed_synthetic_nfp: Seeder, series_id: str
) -> None:
    truth = await seed_synthetic_nfp(session)
    await session.execute(delete(DataObservation).where(DataObservation.series_id == series_id))
    await session.commit()
    with pytest.raises(InsufficientData) as exc:
        await predict(session, truth)
    assert (exc.value.measure, exc.value.have) == ("nfp_change", 0)


async def test_prediction_features_unavailable_raises(
    session: AsyncSession, seed_synthetic_nfp: Seeder
) -> None:
    truth = await seed_synthetic_nfp(session)
    # Only the reference month's claims are missing: training is intact, the live vector is not.
    await session.execute(
        delete(DataObservation).where(
            DataObservation.series_id == SERIES_ICSA,
            DataObservation.obs_date >= truth.ref_month,
        )
    )
    await session.commit()
    with pytest.raises(InsufficientData) as exc:
        await predict(session, truth)
    assert exc.value.measure == "nfp_change_features"


@pytest.mark.parametrize("delta", [timedelta(0), timedelta(days=5)])
async def test_already_published_release_raises(
    session: AsyncSession, seed_synthetic_nfp: Seeder, delta: timedelta
) -> None:
    truth = await seed_synthetic_nfp(session)
    last = add_months(truth.ref_month, -1)
    asof = truth.first_vintage[last] + delta
    with pytest.raises(ValueError, match="release already published as of asof"):
        await NfpModelV1().predict(session, PROVIDER, release_for(last), asof)


async def test_published_guard_runs_before_the_sufficiency_check(
    session: AsyncSession, seed_synthetic_nfp: Seeder
) -> None:
    """R7 ordering: with too little history AND a published release, the caller must hear
    "already published" (a caller bug), not InsufficientData (a data gap that gets priced as
    unpriceable)."""
    truth = await seed_synthetic_nfp(session, months=30)  # 29 samples < MIN_TRAIN
    last = add_months(truth.ref_month, -1)
    asof = truth.first_vintage[last]
    with pytest.raises(ValueError, match="release already published as of asof"):
        await NfpModelV1().predict(session, PROVIDER, release_for(last), asof)
    # The same thin history before publication is a data gap, not a caller error.
    with pytest.raises(InsufficientData):
        await predict(session, truth)


async def test_prediction_ignores_data_after_asof(
    session: AsyncSession, seed_synthetic_nfp: Seeder
) -> None:
    truth = await seed_synthetic_nfp(session)
    before = await predict(session, truth)

    later = truth.ref_first_vintage  # > asof by 1us
    assert later > truth.asof
    ref = truth.ref_month
    extra: list[DataObservation] = []
    # PAYEMS: wild first prints of the reference / next month, and late revisions of the two
    # months the features read (m-1, m-2) plus older ones.
    extra.append(obs(SERIES_PAYEMS, ref, later, "999999"))
    extra.append(obs(SERIES_PAYEMS, add_months(ref, 1), later + timedelta(days=30), "5"))
    for k in (1, 2, 5, 40):
        extra.append(obs(SERIES_PAYEMS, add_months(ref, -k), later + timedelta(days=60), "1"))
    # ICSA / ADP: late revisions of existing weeks and brand-new weeks, in months m and m-1.
    for sid in (SERIES_ICSA, SERIES_ADP):
        for month in (ref, add_months(ref, -1)):
            for day in (1, 8):
                extra.append(obs(sid, date(month.year, month.month, day), later, "9999999"))
            extra.append(obs(sid, date(month.year, month.month, 29), later, "1"))
    # JTSJOL: late revisions of m-2 and m-3, plus later months.
    for k in (2, 3):
        extra.append(obs(SERIES_JOLTS, add_months(ref, -k), later + timedelta(days=45), "1"))
    for k in (0, 1):
        extra.append(obs(SERIES_JOLTS, add_months(ref, -k), later, "99999"))
    session.add_all(extra)
    await session.commit()

    after = await predict(session, truth)
    assert after.n_train == before.n_train
    assert dict(after.diagnostics) == dict(before.diagnostics)
    assert normal(after).params() == normal(before).params()


async def test_sample_features_match_definition(
    session: AsyncSession, seed_synthetic_nfp: Seeder
) -> None:
    truth = await seed_synthetic_nfp(session)
    samples = await nfp_samples(session, PROVIDER, truth.asof)
    month = date(2021, 11, 1)
    s = next(x for x in samples if x.month == month)
    p1, p2, p3 = (add_months(month, -k) for k in (1, 2, 3))
    expected = (
        truth.payems_levels[p1] - truth.payems_levels[p2],
        truth.claims_k[month] - truth.claims_k[p1],
        truth.adp_k[month] - truth.adp_k[p1],
        truth.jolts[p2] - truth.jolts[p3],
    )
    assert len(s.features) == 4
    for got, want in zip(s.features, expected, strict=True):
        assert abs(got - want) <= D("0.000001")
    assert s.target == truth.payems_levels[month] - truth.payems_levels[p1]
    assert s.asof == truth.first_vintage[month] - NO_US


async def test_training_uses_pre_revision_feature_and_first_vintage_target(
    session: AsyncSession, seed_synthetic_nfp: Seeder
) -> None:
    truth = await seed_synthetic_nfp(session)
    month = date(2021, 11, 1)
    prev, prev2 = add_months(month, -1), add_months(month, -2)
    # BLS revises m-1 in the same release that first prints m.
    revised = truth.payems_levels[prev] + D(777)
    session.add(obs(SERIES_PAYEMS, prev, truth.first_vintage[month], str(revised)))
    await session.commit()
    by_month = {s.month: s for s in await nfp_samples(session, PROVIDER, truth.asof)}

    levels = truth.payems_levels
    # Feature for m is read 1us before that vintage: the pre-revision m-1 level.
    assert by_month[month].features[0] == levels[prev] - levels[prev2]
    # The target is m's change as published in m's first vintage: against the revised m-1.
    assert by_month[month].target == levels[month] - revised
    assert by_month[month].target != levels[month] - levels[prev]
    # The next month's feature sees the revision (it is known by then).
    nxt = add_months(month, 1)
    assert by_month[nxt].features[0] == levels[month] - revised
    # m-1's own sample is untouched.
    assert by_month[prev].target == levels[prev] - levels[prev2]


async def test_pricing_nfp_strike_uses_thousand_grid(
    session: AsyncSession, seed_synthetic_nfp: Seeder
) -> None:
    truth = await seed_synthetic_nfp(session)
    dist = normal(await predict(session, truth))
    # Reported value R = round(X, -3 digits): GT 90000 needs R >= 91000, i.e. X >= 90500.
    terms = ContractTerms(Measure.NFP_CHANGE, Comparator.GT, D("90000"), None, "jobs", -3, "bls")
    assert fair_prob(terms, dist) == dist.prob_between(D("90500"), None)
    # Near the mean the grid matters: P(X >= strike + 500) differs from P(X > strike).
    strike = (dist.mu - D(20_000)).quantize(D("1"), rounding="ROUND_HALF_EVEN")
    strike = (strike / K).quantize(D("1")) * K
    near = ContractTerms(Measure.NFP_CHANGE, Comparator.GT, strike, None, "jobs", -3, "bls")
    assert fair_prob(near, dist) == dist.prob_between(strike + D(500), None)
    assert fair_prob(near, dist) != dist.prob_between(strike, None)

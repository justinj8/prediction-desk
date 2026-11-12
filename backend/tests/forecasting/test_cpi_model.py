from collections.abc import Awaitable, Callable
from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest
from scipy.stats import t as student_t
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from prediction_desk.db.models import DataObservation, Release
from prediction_desk.domain.enums import Comparator, DataSource, Measure, Series
from prediction_desk.forecasting.base import BaseModel, ModelOutput
from prediction_desk.forecasting.cpi import (
    ALPHA,
    MIN_TRAIN,
    NU,
    SIGMA_FLOOR,
    CpiModelV1,
    mom_samples,
)
from prediction_desk.forecasting.distributions import StudentT
from prediction_desk.forecasting.realtime import InsufficientData
from prediction_desk.ingestion.fred import fred_vintage_ts
from prediction_desk.ingestion.provider import MacroDataProvider
from prediction_desk.markets.mapping import ContractTerms
from prediction_desk.pricing.fair import fair_prob
from synthetic_macro import (
    SERIES_GAS,
    SERIES_PPI,
    SERIES_SHELTER,
    SyntheticCpiTruth,
    add_months,
    first_vintage_for,
)

D = Decimal
Seeder = Callable[..., Awaitable[SyntheticCpiTruth]]
PROVIDER = MacroDataProvider()
_CONFORMS: BaseModel = CpiModelV1()  # mypy verifies the Protocol


def release_for(month: date) -> Release:
    return Release(
        series=Series.CPI,
        period=f"{month.year:04d}-{month.month:02d}",
        scheduled_at=first_vintage_for(month),
    )


def pct(levels: dict[date, Decimal], month: date) -> Decimal:
    return D(100) * (levels[month] / levels[add_months(month, -1)] - 1)


def student(out_dist: object) -> StudentT:
    assert isinstance(out_dist, StudentT)
    return out_dist


async def predict(session: AsyncSession, truth: SyntheticCpiTruth) -> ModelOutput:
    return await CpiModelV1().predict(session, PROVIDER, release_for(truth.ref_month), truth.asof)


def test_constants() -> None:
    assert D("1.0") == ALPHA
    assert MIN_TRAIN == 36
    assert D("0.10") == SIGMA_FLOOR  # R13 (was 0.05)
    assert D("5") == NU
    assert CpiModelV1.model_version == "cpi-ridge-v1"
    assert CpiModelV1.series == Series.CPI


async def test_predict_returns_four_measures(
    session: AsyncSession, seed_synthetic_cpi: Seeder
) -> None:
    truth = await seed_synthetic_cpi(session)
    out = await predict(session, truth)
    assert out.model_version == "cpi-ridge-v1"
    assert set(out.distributions) == {
        Measure.CPI_MOM,
        Measure.CORE_CPI_MOM,
        Measure.CPI_YOY,
        Measure.CORE_CPI_YOY,
    }
    for dist in out.distributions.values():
        assert isinstance(dist, StudentT)
        assert dist.nu == NU
        assert dist.sigma > 0
    assert set(out.diagnostics) == {
        "cpi_mom.n_train",
        "cpi_mom.oos_rms",
        "core_cpi_mom.n_train",
        "core_cpi_mom.oos_rms",
        "cpi_yoy.r",
        "cpi_yoy.s_m",
        "cpi_yoy.var_s",
        "core_cpi_yoy.var_s",
        "core_cpi_yoy.r",
        "core_cpi_yoy.s_m",
    }
    assert out.n_train >= MIN_TRAIN
    assert out.n_train == min(
        int(out.diagnostics["cpi_mom.n_train"]), int(out.diagnostics["core_cpi_mom.n_train"])
    )
    assert all(isinstance(v, str) for v in out.diagnostics.values())


async def test_mom_mean_tracks_known_relationship(
    session: AsyncSession, seed_synthetic_cpi: Seeder
) -> None:
    truth = await seed_synthetic_cpi(session)
    out = await predict(session, truth)
    head = student(out.distributions[Measure.CPI_MOM])
    core = student(out.distributions[Measure.CORE_CPI_MOM])
    assert abs(head.mu - truth.true_headline_mom_mean) < D("0.05")
    assert abs(core.mu - truth.true_core_mom_mean) < D("0.05")
    # The reference month's conditional mean is far from the unconditional mean, so a
    # feature-blind (intercept-only) model cannot pass.
    gap = abs(truth.unconditional_headline_mom_mean - truth.true_headline_mom_mean)
    assert gap > D("0.15")
    assert abs(head.mu - truth.true_headline_mom_mean) < gap / 4
    # Same for core (CORE_REF_SHOCK): an intercept-only core model would sit at the
    # unconditional mean, a full `core_gap` away from the truth.
    core_gap = abs(truth.unconditional_core_mom_mean - truth.true_core_mom_mean)
    assert core_gap > D("0.15")
    assert abs(core.mu - truth.true_core_mom_mean) < core_gap / 4


async def test_sigma_floor_on_noiseless_data(
    session: AsyncSession, seed_synthetic_cpi: Seeder
) -> None:
    truth = await seed_synthetic_cpi(session, noiseless=True)
    out = await predict(session, truth)
    head = student(out.distributions[Measure.CPI_MOM])
    core = student(out.distributions[Measure.CORE_CPI_MOM])
    assert head.sigma == SIGMA_FLOOR
    assert core.sigma == SIGMA_FLOOR
    # A strike ~4 floors below the mean must still price strictly inside (0, 1). GT strike k on
    # the 0.1 grid needs X >= k + 0.05, so the standardized cut is (k + 0.05 - mu) / 0.10; the
    # expected value comes straight from scipy's t(5) survival function (R13 floor = 0.10).
    strike = (head.mu - D("0.4")).quantize(D("0.1"))
    terms = ContractTerms(Measure.CPI_MOM, Comparator.GT, strike, None, "pct", 1, "bls")
    p = fair_prob(terms, head)
    z = float((strike + D("0.05") - head.mu) / D("0.10"))
    assert float(p) == pytest.approx(float(student_t.sf(z, 5)), abs=5e-6)  # Prob has 5 dp
    assert D("0.99") < p < D(1)


async def test_yoy_mean_affine_and_sigma_includes_seasonal_variance(
    session: AsyncSession, seed_synthetic_cpi: Seeder
) -> None:
    truth = await seed_synthetic_cpi(session)
    out = await predict(session, truth)
    m = truth.ref_month
    for mom_measure, yoy_measure, nsa_id, sa_id, prefix in (
        (Measure.CPI_MOM, Measure.CPI_YOY, "CPIAUCNS", "CPIAUCSL", "cpi_yoy"),
        (Measure.CORE_CPI_MOM, Measure.CORE_CPI_YOY, "CPILFENS", "CPILFESL", "core_cpi_yoy"),
    ):
        nsa = truth.first_print_levels[nsa_id]
        sa = truth.first_print_levels[sa_id]
        # Hand-computed r and s_m (5 most recent prior years, same calendar month).
        r = nsa[add_months(m, -1)] / nsa[add_months(m, -12)]
        offsets = [
            pct(nsa, date(m.year - k, m.month, 1)) - pct(sa, date(m.year - k, m.month, 1))
            for k in range(1, 6)
        ]
        s_m = sum(offsets, D(0)) / 5
        # The seeded seasonal pattern implies offset ~ F_k - F_{k-1}.
        factors = truth.seasonal_pct[nsa_id]
        assert abs(s_m - (factors[m.month - 1] - factors[m.month - 2])) < D("0.15")

        mom = student(out.distributions[mom_measure])
        yoy = student(out.distributions[yoy_measure])
        mean_o = s_m
        var_s = sum(((o - mean_o) ** 2 for o in offsets), D(0)) / 4  # ddof=1, 5 years
        assert var_s > D("0.0001")  # seeded offsets vary across years
        # Mean is the affine value; sigma carries the seasonal-offset uncertainty (R6).
        assert abs(yoy.mu - (100 * (r - 1) + r * (mom.mu + s_m))) < D("1e-5")
        assert abs(yoy.sigma - r * (mom.sigma**2 + var_s).sqrt()) < D("1e-5")
        assert yoy.sigma > r * mom.sigma
        assert abs(D(out.diagnostics[f"{prefix}.var_s"]) - var_s) < D("1e-5")
        assert yoy.nu == mom.nu
        assert abs(D(out.diagnostics[f"{prefix}.r"]) - r) < D("1e-6")
        assert abs(D(out.diagnostics[f"{prefix}.s_m"]) - s_m) < D("1e-5")


async def test_insufficient_history_raises(
    session: AsyncSession, seed_synthetic_cpi: Seeder
) -> None:
    truth = await seed_synthetic_cpi(session, months=30)
    with pytest.raises(InsufficientData) as exc:
        await predict(session, truth)
    assert exc.value.measure == "cpi_mom"
    assert exc.value.need == MIN_TRAIN
    assert exc.value.have < MIN_TRAIN


async def test_missing_feature_series_skips_samples_then_raises(
    session: AsyncSession, seed_synthetic_cpi: Seeder
) -> None:
    truth = await seed_synthetic_cpi(session)
    await session.execute(delete(DataObservation).where(DataObservation.series_id == SERIES_PPI))
    await session.commit()
    with pytest.raises(InsufficientData) as exc:
        await predict(session, truth)
    assert (exc.value.measure, exc.value.have, exc.value.need) == ("cpi_mom", 0, MIN_TRAIN)


async def test_prediction_features_unavailable_raises(
    session: AsyncSession, seed_synthetic_cpi: Seeder
) -> None:
    truth = await seed_synthetic_cpi(session)
    # Only the reference month's gasoline is missing: training is intact, the live vector is not.
    await session.execute(
        delete(DataObservation).where(
            DataObservation.series_id == SERIES_GAS,
            DataObservation.obs_date >= truth.ref_month,
        )
    )
    await session.commit()
    with pytest.raises(InsufficientData) as exc:
        await predict(session, truth)
    assert exc.value.measure == "cpi_mom_features"


async def test_insufficient_seasonal_history_raises(
    session: AsyncSession, seed_synthetic_cpi: Seeder
) -> None:
    truth = await seed_synthetic_cpi(session)
    # Keep Dec 2022 + Jan 2023 (one usable January pct) and the r base month; drop the rest.
    cutoff = date(truth.ref_month.year - 2, 12, 1)
    await session.execute(
        delete(DataObservation).where(
            DataObservation.series_id == "CPIAUCNS",
            DataObservation.obs_date < cutoff,
            DataObservation.obs_date != add_months(truth.ref_month, -12),
        )
    )
    await session.commit()
    with pytest.raises(InsufficientData) as exc:
        await predict(session, truth)
    assert (exc.value.measure, exc.value.have, exc.value.need) == ("cpi_yoy_seasonal", 1, 3)


async def test_missing_yoy_base_levels_raises(
    session: AsyncSession, seed_synthetic_cpi: Seeder
) -> None:
    truth = await seed_synthetic_cpi(session)
    base = add_months(truth.ref_month, -12)
    await session.execute(
        delete(DataObservation).where(
            DataObservation.series_id == "CPIAUCNS", DataObservation.obs_date == base
        )
    )
    await session.commit()
    with pytest.raises(InsufficientData) as exc:
        await predict(session, truth)
    assert exc.value.measure == "cpi_yoy_levels"
    assert exc.value.need == 2


async def test_prediction_ignores_data_after_asof(
    session: AsyncSession, seed_synthetic_cpi: Seeder
) -> None:
    truth = await seed_synthetic_cpi(session)
    before = await predict(session, truth)

    later = truth.ref_first_vintage  # > asof by 1us
    assert later > truth.asof
    ref = truth.ref_month
    extra: list[DataObservation] = []

    def add(sid: str, obs_date: date, vintage_ts: datetime, value: str) -> None:
        extra.append(
            DataObservation(
                source=DataSource.FRED,
                series_id=sid,
                obs_date=obs_date,
                vintage_ts=vintage_ts,
                published_on=vintage_ts.date(),
                value=D(value),
                ingested_at=vintage_ts,
            )
        )

    # The reference month itself and the next one get (wild) first prints after asof.
    for sid in ("CPIAUCSL", "CPILFESL", "CPIAUCNS", "CPILFENS", SERIES_PPI, SERIES_SHELTER):
        add(sid, ref, later, "999")
        add(sid, add_months(ref, 1), later + timedelta(days=30), "5")
    # A late revision of old months, a late gasoline print, a late gasoline revision.
    for sid in ("CPIAUCSL", "CPILFESL", "CPIAUCNS", "CPILFENS"):
        for k in (3, 14, 40):
            add(sid, add_months(ref, -k), later + timedelta(days=60), "1")
    # Late revisions of the YoY inputs: m-1, m-12 and a prior-year Dec/Jan pair used by s_m.
    for sid in ("CPIAUCSL", "CPILFESL", "CPIAUCNS", "CPILFENS"):
        for k in (1, 12, 24, 25, 36, 37):
            add(sid, add_months(ref, -k), later + timedelta(days=90), "777")
    add(SERIES_GAS, date(ref.year, ref.month, 1), later, "9.999")
    add(SERIES_GAS, date(ref.year, ref.month, 29), later, "0.001")
    session.add_all(extra)
    await session.commit()

    after = await predict(session, truth)
    assert after.n_train == before.n_train
    assert dict(after.diagnostics) == dict(before.diagnostics)
    for measure, dist in before.distributions.items():
        assert after.distributions[measure].params() == dist.params()


async def test_training_target_uses_first_print_of_revised_month(
    session: AsyncSession, seed_synthetic_cpi: Seeder
) -> None:
    truth = await seed_synthetic_cpi(session)
    rev = truth.revision
    assert rev.revised != rev.first_print
    samples = await mom_samples(session, PROVIDER, core=False, asof=truth.asof)
    by_month = {s.month: s for s in samples}
    sa = truth.first_print_levels["CPIAUCSL"]
    # Both the revised month and the next one must be scored against first-print levels.
    for month in (rev.obs_date, add_months(rev.obs_date, 1)):
        assert by_month[month].target == pct(sa, month).quantize(D("0.000001"))
        # What a (wrong) latest-levels target would have been:
        wrong_levels = dict(sa)
        wrong_levels[rev.obs_date] = rev.revised
        assert abs(pct(wrong_levels, month) - by_month[month].target) > D("1")


async def test_sample_features_match_definition(
    session: AsyncSession, seed_synthetic_cpi: Seeder
) -> None:
    truth = await seed_synthetic_cpi(session)
    samples = await mom_samples(session, PROVIDER, core=False, asof=truth.asof)
    month = date(2021, 11, 1)
    s = next(x for x in samples if x.month == month)
    sa = truth.first_print_levels["CPIAUCSL"]
    ppi = truth.first_print_levels[SERIES_PPI]
    shelter = truth.first_print_levels[SERIES_SHELTER]
    gas = truth.gas_monthly
    q = D("0.000001")
    expected = (
        pct(sa, add_months(month, -1)),
        pct(sa, add_months(month, -2)),
        D(100) * (gas[month] / gas[add_months(month, -1)] - 1),
        pct(ppi, add_months(month, -1)),
        pct(shelter, add_months(month, -1)),
    )
    assert len(s.features) == 5
    for got, want in zip(s.features, expected, strict=True):
        assert abs(got - want) <= q
    # t_m is strictly before the month's first vintage and nothing later is used.
    assert s.asof == truth.first_vintage[month] - timedelta(microseconds=1)
    assert s.asof < fred_vintage_ts(date(2021, 12, 12))
    core_samples = await mom_samples(session, PROVIDER, core=True, asof=truth.asof)
    cs = next(x for x in core_samples if x.month == month)
    assert len(cs.features) == 4  # no gasoline feature for core


async def test_sigma_above_floor_equals_unfloored_oos_rms(
    session: AsyncSession, seed_synthetic_cpi: Seeder
) -> None:
    truth = await seed_synthetic_cpi(session, noise_sd=0.4)
    out = await predict(session, truth)
    for measure, key in (
        (Measure.CPI_MOM, "cpi_mom.oos_rms"),
        (Measure.CORE_CPI_MOM, "core_cpi_mom.oos_rms"),
    ):
        rms = D(out.diagnostics[key])
        assert rms > SIGMA_FLOOR
        assert student(out.distributions[measure]).sigma == rms


@pytest.mark.parametrize("delta", [timedelta(0), timedelta(days=5)])
async def test_already_published_release_raises(
    session: AsyncSession, seed_synthetic_cpi: Seeder, delta: timedelta
) -> None:
    truth = await seed_synthetic_cpi(session)
    last = add_months(truth.ref_month, -1)
    asof = truth.first_vintage[last] + delta
    with pytest.raises(ValueError, match="release already published as of asof"):
        await CpiModelV1().predict(session, PROVIDER, release_for(last), asof)

from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from prediction_desk.db.models import DataObservation, Release
from prediction_desk.domain.enums import Comparator, DataSource, Measure, Series
from prediction_desk.forecasting.base import BaseModel, ModelOutput
from prediction_desk.forecasting.distributions import Categorical
from prediction_desk.forecasting.fomc import FomcModelV1
from prediction_desk.forecasting.realtime import InsufficientData
from prediction_desk.ingestion.fred import fred_vintage_ts
from prediction_desk.ingestion.provider import MacroDataProvider
from prediction_desk.markets.mapping import ContractTerms
from prediction_desk.pricing.fair import fair_prob
from synthetic_macro import SERIES_DFF, SERIES_DTB4WK, SyntheticRatesTruth

D = Decimal
Seeder = Callable[..., Awaitable[SyntheticRatesTruth]]
PROVIDER = MacroDataProvider()
_CONFORMS: BaseModel = FomcModelV1()  # mypy verifies the Protocol
_NY = ZoneInfo("America/New_York")
LAST_OBS = date(2026, 3, 10)  # asof = 2026-03-11 00:00 ET


def release_on(meeting: date) -> Release:
    return Release(
        series=Series.FOMC,
        period=meeting.isoformat(),
        scheduled_at=datetime.combine(meeting, time(14), tzinfo=_NY).astimezone(UTC),
    )


async def predict(
    session: AsyncSession, truth: SyntheticRatesTruth, days_to: int = 0
) -> ModelOutput:
    release = release_on(truth.asof_et_date + timedelta(days=days_to))
    return await FomcModelV1().predict(session, PROVIDER, release, truth.asof)


def probs(out: ModelOutput) -> dict[int, Decimal]:
    dist = out.distributions[Measure.FOMC_CHANGE_BP]
    assert isinstance(dist, Categorical)
    return dict(dist.outcomes)


def expected(**kw: str) -> dict[int, Decimal]:
    full = {-50: D(0), -25: D(0), 0: D(0), 25: D(0), 50: D(0)}
    for k, v in kw.items():
        full[int(k.replace("m", "-").lstrip("p"))] = D(v)
    return full


def test_constants() -> None:
    assert FomcModelV1.model_version == "fomc-implied-v1"
    assert FomcModelV1.series is Series.FOMC
    assert FomcModelV1.OUTCOMES == (-50, -25, 0, 25, 50)
    assert D("0.5") == FomcModelV1.PRIOR_WEIGHT
    assert FomcModelV1.SPREAD_LOOKBACK == 250
    assert D("0.25") == FomcModelV1.MIN_FRAC_AFTER
    assert FomcModelV1.MIN_SPREAD_OBS == 20


async def test_no_change_priced_flat(session: AsyncSession, seed_synthetic_rates: Seeder) -> None:
    truth = await seed_synthetic_rates(session, last_obs=LAST_OBS)
    out = await predict(session, truth)
    assert probs(out) == expected(**{"0": "1"})
    assert out.model_version == "fomc-implied-v1"
    assert out.n_train == 250
    diag = out.diagnostics
    assert set(diag) == {
        "fomc.dff",
        "fomc.bill",
        "fomc.spread",
        "fomc.days_to",
        "fomc.frac_after",
        "fomc.implied_bp",
        "fomc.n_spread",
    }
    # Diagnostics are quantized to 1e-6 (as in cpi.py), independent of the DB NUMERIC scale.
    assert dict(diag) == {
        "fomc.dff": "5.330000",
        "fomc.bill": "5.350000",
        "fomc.spread": "0.020000",
        "fomc.days_to": "0",
        "fomc.frac_after": "1.000000",
        "fomc.implied_bp": "0.000000",
        "fomc.n_spread": "250",
    }


async def test_implied_cut_interpolates(
    session: AsyncSession, seed_synthetic_rates: Seeder
) -> None:
    # implied -10: interp {-25: 0.4, 0: 0.6}; nearest 0 -> {-25: 0.2, 0: 0.3 + 0.5}
    truth = await seed_synthetic_rates(session, last_obs=LAST_OBS, last_gap=D("-0.10"))
    out = await predict(session, truth)
    assert probs(out) == expected(m25="0.2", **{"0": "0.8"})
    assert D(out.diagnostics["fomc.implied_bp"]) == -10


async def test_midpoint_tie_cut_goes_to_zero(
    session: AsyncSession, seed_synthetic_rates: Seeder
) -> None:
    # implied -12.5: interp {-25: .5, 0: .5}; nearest tie -> 0 -> {-25: .25, 0: .25 + .5}
    truth = await seed_synthetic_rates(session, last_obs=LAST_OBS, last_gap=D("-0.125"))
    assert probs(await predict(session, truth)) == expected(m25="0.25", **{"0": "0.75"})


async def test_midpoint_tie_hike_goes_to_25(
    session: AsyncSession, seed_synthetic_rates: Seeder
) -> None:
    # implied +37.5: interp {25: .5, 50: .5}; nearest tie -> 25 -> {25: .25 + .5, 50: .25}
    truth = await seed_synthetic_rates(session, last_obs=LAST_OBS, last_gap=D("0.375"))
    assert probs(await predict(session, truth)) == expected(p25="0.75", p50="0.25")


async def test_implied_exactly_on_grid_and_edge(
    session: AsyncSession, seed_synthetic_rates: Seeder
) -> None:
    truth = await seed_synthetic_rates(session, last_obs=LAST_OBS, last_gap=D("0.25"))
    assert probs(await predict(session, truth)) == expected(p25="1")
    # exactly +50 puts all mass on the hike tail
    s2 = await seed_synthetic_rates(
        session, last_obs=LAST_OBS - timedelta(days=1000), last_gap=D("0.50")
    )
    assert probs(await predict(session, s2)) == expected(p50="1")


async def test_far_meeting_uses_prior_only(
    session: AsyncSession, seed_synthetic_rates: Seeder
) -> None:
    # days_to 25 -> frac_after 3/28 < 0.25: the bill signal is ignored.
    truth = await seed_synthetic_rates(session, last_obs=LAST_OBS, last_gap=D("-0.50"))
    out = await predict(session, truth, days_to=25)
    assert probs(out) == expected(**{"0": "1"})
    assert out.diagnostics["fomc.implied_bp"] == "n/a"
    assert out.diagnostics["fomc.days_to"] == "25"
    assert out.diagnostics["fomc.frac_after"] == "0.107143"
    # beyond 28 days the fraction is 0
    far = await predict(session, truth, days_to=40)
    assert far.diagnostics["fomc.frac_after"] == "0.000000"
    assert probs(far) == expected(**{"0": "1"})


async def test_frac_after_boundary_is_inclusive(
    session: AsyncSession, seed_synthetic_rates: Seeder
) -> None:
    # days_to 21 -> frac 0.25 (not < 0.25): implied = -0.025 * 100 / 0.25 = -10 bp.
    truth = await seed_synthetic_rates(session, last_obs=LAST_OBS, last_gap=D("-0.025"))
    out = await predict(session, truth, days_to=21)
    assert probs(out) == expected(m25="0.2", **{"0": "0.8"})
    assert out.diagnostics["fomc.frac_after"] == "0.250000"


async def test_clamp_extremes(session: AsyncSession, seed_synthetic_rates: Seeder) -> None:
    truth = await seed_synthetic_rates(session, last_obs=LAST_OBS, last_gap=D("-2.0"))
    out = await predict(session, truth)
    assert probs(out) == expected(m50="1")
    assert D(out.diagnostics["fomc.implied_bp"]) == -50
    up = await seed_synthetic_rates(
        session, last_obs=LAST_OBS - timedelta(days=1000), last_gap=D("2.0")
    )
    assert probs(await predict(session, up)) == expected(p50="1")


@pytest.mark.parametrize("gap", ["-0.37", "-0.04", "0.13", "0.29"])
async def test_probabilities_sum_to_one(
    session: AsyncSession, seed_synthetic_rates: Seeder, gap: str
) -> None:
    truth = await seed_synthetic_rates(session, last_obs=LAST_OBS, last_gap=D(gap))
    p = probs(await predict(session, truth))
    assert sum(p.values(), D(0)) == 1
    assert all(v >= 0 for v in p.values())


async def test_kalshi_polymarket_tails_agree(
    session: AsyncSession, seed_synthetic_rates: Seeder
) -> None:
    truth = await seed_synthetic_rates(session, last_obs=LAST_OBS, last_gap=D("-0.30"))
    dist = (await predict(session, truth)).distributions[Measure.FOMC_CHANGE_BP]

    def terms(cmp: Comparator, strike: str) -> ContractTerms:
        return ContractTerms(
            Measure.FOMC_CHANGE_BP, cmp, D(strike), None, "bp", 0, "Federal Reserve"
        )

    kalshi_cut = fair_prob(terms(Comparator.LT, "-25"), dist)
    poly_cut = fair_prob(terms(Comparator.LTE, "-50"), dist)
    assert kalshi_cut == poly_cut
    assert kalshi_cut > 0
    assert fair_prob(terms(Comparator.GT, "25"), dist) == fair_prob(
        terms(Comparator.GTE, "50"), dist
    )


@pytest.mark.parametrize("delta", [timedelta(0), timedelta(hours=1)])
async def test_already_published_release_raises(
    session: AsyncSession, seed_synthetic_rates: Seeder, delta: timedelta
) -> None:
    truth = await seed_synthetic_rates(session, last_obs=LAST_OBS)
    release = release_on(truth.asof_et_date)
    asof = release.scheduled_at + delta
    with pytest.raises(ValueError, match="release already published as of asof"):
        await FomcModelV1().predict(session, PROVIDER, release, asof)


async def test_insufficient_spread_raises(
    session: AsyncSession, seed_synthetic_rates: Seeder
) -> None:
    truth = await seed_synthetic_rates(session, last_obs=LAST_OBS, days=19)
    with pytest.raises(InsufficientData) as exc:
        await predict(session, truth)
    assert (exc.value.measure, exc.value.have, exc.value.need) == ("fomc_spread", 19, 20)
    ok = await seed_synthetic_rates(session, last_obs=LAST_OBS - timedelta(days=2000), days=20)
    assert (await predict(session, ok)).n_train == 20


async def test_missing_rates_raises(session: AsyncSession) -> None:
    truth = SyntheticRatesTruth(
        LAST_OBS, fred_vintage_ts(LAST_OBS), LAST_OBS + timedelta(days=1), D(0), D(0), D(0)
    )
    with pytest.raises(InsufficientData) as exc:
        await predict(session, truth)
    assert (exc.value.measure, exc.value.have, exc.value.need) == ("fomc_rates", 0, 1)


async def test_only_one_series_raises(session: AsyncSession) -> None:
    session.add(_obs(SERIES_DFF, LAST_OBS, "5.33"))
    await session.commit()
    truth = SyntheticRatesTruth(
        LAST_OBS, fred_vintage_ts(LAST_OBS), LAST_OBS + timedelta(days=1), D(0), D(0), D(0)
    )
    with pytest.raises(InsufficientData, match="fomc_rates"):
        await predict(session, truth)


def _obs(sid: str, d: date, value: str, vts: datetime | None = None) -> DataObservation:
    v = vts or fred_vintage_ts(d)
    return DataObservation(
        source=DataSource.FRED,
        series_id=sid,
        obs_date=d,
        vintage_ts=v,
        published_on=(v - timedelta(days=1)).date(),
        value=D(value),
        ingested_at=v,
    )


async def test_spread_is_median_not_mean(
    session: AsyncSession, seed_synthetic_rates: Seeder
) -> None:
    # 10 of the last 250 days carry a +1.00 spread shock: median stays 0.02, mean is 0.06.
    shocks: Mapping[int, Decimal] = {b: D("1.00") for b in range(10, 20)}
    truth = await seed_synthetic_rates(session, last_obs=LAST_OBS, shocks=shocks)
    out = await predict(session, truth)
    assert D(out.diagnostics["fomc.spread"]) == D("0.02")
    assert probs(out) == expected(**{"0": "1"})


async def test_spread_even_count_averages_middle_pair(
    session: AsyncSession, seed_synthetic_rates: Seeder
) -> None:
    # 20 paired days (even): 10 at spread 0.02, 10 at 0.04 -> median 0.03.
    shocks = {b: D("0.02") for b in range(10)}
    truth = await seed_synthetic_rates(session, last_obs=LAST_OBS, days=20, shocks=shocks)
    out = await predict(session, truth)
    assert D(out.diagnostics["fomc.spread"]) == D("0.03")


async def test_spread_uses_only_dates_in_both_series(
    session: AsyncSession, seed_synthetic_rates: Seeder
) -> None:
    truth = await seed_synthetic_rates(session, last_obs=LAST_OBS)
    # Extra DFF-only and DTB4WK-only days far in the past must not enter the spread.
    session.add(_obs(SERIES_DFF, LAST_OBS - timedelta(days=340), "9.99"))
    session.add(_obs(SERIES_DTB4WK, LAST_OBS - timedelta(days=341), "0.01"))
    await session.commit()
    out = await predict(session, truth)
    assert out.diagnostics["fomc.n_spread"] == "250"
    assert D(out.diagnostics["fomc.spread"]) == D("0.02")


async def test_prediction_ignores_data_after_asof(
    session: AsyncSession, seed_synthetic_rates: Seeder
) -> None:
    truth = await seed_synthetic_rates(session, last_obs=LAST_OBS, last_gap=D("-0.10"))
    before = await predict(session, truth)
    later = fred_vintage_ts(LAST_OBS + timedelta(days=30))
    for k in range(1, 6):
        d = LAST_OBS + timedelta(days=k)
        session.add(_obs(SERIES_DFF, d, "9.00"))
        session.add(_obs(SERIES_DTB4WK, d, "1.00"))
    # post-asof revisions of recent dates
    for k in range(0, 40):
        d = LAST_OBS - timedelta(days=k)
        session.add(_obs(SERIES_DFF, d, "8.00", later))
        session.add(_obs(SERIES_DTB4WK, d, "0.50", later))
    await session.commit()
    after = await predict(session, truth)
    assert probs(after) == probs(before)
    assert dict(after.diagnostics) == dict(before.diagnostics)


async def test_spread_odd_count_takes_middle_value(
    session: AsyncSession, seed_synthetic_rates: Seeder
) -> None:
    # 21 paired days (odd): 10 at 0.02, 10 at 0.06, 1 at 0.04 -> median 0.04.
    shocks = {**{b: D("0.04") for b in range(1, 11)}, **{0: D("0.02")}}
    truth = await seed_synthetic_rates(session, last_obs=LAST_OBS, days=21, shocks=shocks)
    out = await predict(session, truth)
    assert D(out.diagnostics["fomc.spread"]) == D("0.04")


async def test_latest_rates_are_used_not_oldest(
    session: AsyncSession, seed_synthetic_rates: Seeder
) -> None:
    # DFF stepped up on the last day (bill = DFF + spread): latest, not oldest, must be read.
    older = {b: D("5.33") for b in range(1, 300)}
    truth = await seed_synthetic_rates(session, last_obs=LAST_OBS, dff=D("5.58"), dff_by_back=older)
    out = await predict(session, truth)
    assert out.diagnostics["fomc.dff"] == "5.580000"
    assert out.diagnostics["fomc.bill"] == "5.600000"
    assert probs(out) == expected(**{"0": "1"})


async def test_spread_uses_last_250_not_oldest_250(
    session: AsyncSession, seed_synthetic_rates: Seeder
) -> None:
    # 400 paired days; the oldest 150 (back 250..399) carry a +1.00 shock. The last 250 are
    # clean, so spread 0.02; the oldest 250 would be 60% shocked (median 1.02).
    shocks = {b: D("1.00") for b in range(250, 400)}
    truth = await seed_synthetic_rates(session, last_obs=LAST_OBS, days=400, shocks=shocks)
    out = await predict(session, truth)
    assert out.n_train == 250
    assert out.diagnostics["fomc.spread"] == "0.020000"
    assert out.diagnostics["fomc.n_spread"] == "250"


async def test_spread_ignores_pairs_older_than_400_days(
    session: AsyncSession, seed_synthetic_rates: Seeder
) -> None:
    # asof ET date = LAST_OBS + 1, so the window starts at LAST_OBS - 399 (inclusive).
    truth = await seed_synthetic_rates(session, last_obs=LAST_OBS, days=100)
    # 120 heavily shocked pairs ending one day before the window: must be ignored.
    old_end = LAST_OBS - timedelta(days=400)
    await seed_synthetic_rates(
        session, last_obs=old_end, days=120, shocks={b: D("1.00") for b in range(120)}
    )
    # A single clean pair exactly on the window start: must be included.
    await seed_synthetic_rates(session, last_obs=LAST_OBS - timedelta(days=399), days=1)
    out = await predict(session, truth)
    assert out.n_train == 101
    assert out.diagnostics["fomc.spread"] == "0.020000"


@pytest.mark.parametrize("which", ["bill", "dff"])
async def test_stale_series_fails_closed(
    session: AsyncSession, seed_synthetic_rates: Seeder, which: str
) -> None:
    lag = {"dff_lag_days": 0, "bill_lag_days": 0}
    lag[f"{which}_lag_days"] = 30
    truth = await seed_synthetic_rates(session, last_obs=LAST_OBS, **lag)
    with pytest.raises(InsufficientData) as exc:
        await predict(session, truth)
    # age = asof ET date - newest obs date = 1 + 30 days
    assert (exc.value.measure, exc.value.have, exc.value.need) == ("fomc_rates_stale", 31, 7)


async def test_staleness_boundary_age_7_passes_8_raises(
    session: AsyncSession, seed_synthetic_rates: Seeder
) -> None:
    # age = 1 + lag
    ok = await seed_synthetic_rates(session, last_obs=LAST_OBS, bill_lag_days=6)
    assert (await predict(session, ok)).n_train > 0
    bad_end = LAST_OBS - timedelta(days=2000)
    bad = await seed_synthetic_rates(session, last_obs=bad_end, dff_lag_days=7)
    with pytest.raises(InsufficientData) as exc:
        await predict(session, bad)
    assert (exc.value.measure, exc.value.have, exc.value.need) == ("fomc_rates_stale", 8, 7)


async def test_staleness_reports_larger_age(
    session: AsyncSession, seed_synthetic_rates: Seeder
) -> None:
    truth = await seed_synthetic_rates(
        session, last_obs=LAST_OBS, dff_lag_days=10, bill_lag_days=20
    )
    with pytest.raises(InsufficientData) as exc:
        await predict(session, truth)
    assert exc.value.have == 21


async def test_asof_late_evening_et_uses_et_date(
    session: AsyncSession, seed_synthetic_rates: Seeder
) -> None:
    # 21:00 ET on 2026-03-10 is already 2026-03-11 in UTC. ET days_to = 21 -> frac 0.25
    # (implied -10 -> {-25: .2, 0: .8}); a UTC date would give days_to 20 -> frac 8/28.
    await seed_synthetic_rates(session, last_obs=date(2026, 3, 9), last_gap=D("-0.025"))
    asof = datetime(2026, 3, 10, 21, 0, tzinfo=_NY).astimezone(UTC)
    assert asof.date() == date(2026, 3, 11)
    out = await FomcModelV1().predict(session, PROVIDER, release_on(date(2026, 3, 31)), asof)
    assert out.diagnostics["fomc.days_to"] == "21"
    assert probs(out) == expected(m25="0.2", **{"0": "0.8"})


async def test_period_before_asof_date_raises(
    session: AsyncSession, seed_synthetic_rates: Seeder
) -> None:
    truth = await seed_synthetic_rates(session, last_obs=LAST_OBS)
    # Inconsistent row: period is yesterday (ET) but scheduled_at is still in the future.
    release = Release(
        series=Series.FOMC,
        period=truth.last_obs.isoformat(),
        scheduled_at=truth.asof + timedelta(days=1),
    )
    with pytest.raises(ValueError, match="release period precedes asof date"):
        await FomcModelV1().predict(session, PROVIDER, release, truth.asof)

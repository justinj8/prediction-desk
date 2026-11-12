import hashlib
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import ROUND_DOWN, Decimal, DivisionByZero, InvalidOperation, Overflow, getcontext
from types import MappingProxyType
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from prediction_desk.audit.hashing import canonical_json
from prediction_desk.db.base import Base
from prediction_desk.db.models import EventContract, MarketSnapshot, Release
from prediction_desk.domain.enums import Comparator, MappingStatus, Measure, Series, Venue
from prediction_desk.domain.types import to_prob
from prediction_desk.forecasting.base import ModelOutput
from prediction_desk.forecasting.distributions import Categorical, StudentT
from prediction_desk.forecasting.realtime import InsufficientData
from prediction_desk.forecasting.registry import MODELS
from prediction_desk.ingestion.provider import MacroDataProvider
from prediction_desk.markets.terms import rules_hash, terms_from_row
from prediction_desk.markets.types import OrderBook
from prediction_desk.pricing import release as release_mod
from prediction_desk.pricing.edge import EdgeQuote, Unavailable, best_side, evaluate_edge
from prediction_desk.pricing.fair import fair_prob
from prediction_desk.pricing.fees import FeeSchedule
from prediction_desk.pricing.release import PRICING_VERSION, ReleasePricing, price_release
from synthetic_macro import SyntheticCpiTruth, add_months, first_vintage_for

D = Decimal
PROVIDER = MacroDataProvider()
NOW = datetime(2026, 11, 11, 12, 0, tzinfo=UTC)
GOOD_BOOK: dict[str, Any] = {"bids": [["0.40", "100"]], "asks": [["0.50", "100"]]}
Seeder = Callable[..., Awaitable[SyntheticCpiTruth]]


class StubModel:
    """Fixed CPI_MOM distribution, so logic tests need no macro data."""

    series = Series.CPI

    def __init__(self, mu: str = "0.30", version: str = "stub-v1") -> None:
        self.model_version = version
        self._mu = D(mu)

    async def predict(
        self, session: AsyncSession, provider: MacroDataProvider, release: Release, asof: datetime
    ) -> ModelOutput:
        dist = StudentT(self._mu, D("0.10"), D("5"))
        return ModelOutput(self.model_version, {Measure.CPI_MOM: dist}, 40, {})


class RaisingModel:
    series = Series.CPI
    model_version = "raising"

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def predict(
        self, session: AsyncSession, provider: MacroDataProvider, release: Release, asof: datetime
    ) -> ModelOutput:
        raise self._exc


def stub(mu: str = "0.30", version: str = "stub-v1") -> dict[Series, Any]:
    return {Series.CPI: StubModel(mu, version)}


async def make_release(session: AsyncSession, period: str = "2026-10") -> Release:
    release = Release(series=Series.CPI, period=period, scheduled_at=NOW + timedelta(days=1))
    session.add(release)
    await session.commit()
    return release


async def add_contract(
    session: AsyncSession,
    release: Release | None,
    market_id: str,
    *,
    strike_hi: str | None = None,
    source: str = "BLS",
    venue: Venue = Venue.KALSHI,
    event: str = "E",
    measure: Measure = Measure.CPI_MOM,
    comparator: Comparator = Comparator.GT,
    strike: str = "0.3",
    precision: int = 1,
    unit: str = "pct",
    rules: str = "rules",
    status: MappingStatus = MappingStatus.MAPPED,
) -> EventContract:
    row = EventContract(
        venue=venue,
        venue_market_id=market_id,
        venue_event_id=event,
        yes_token_id=None,
        title="t",
        rules_text=rules,
        close_time=NOW + timedelta(days=1),
        release_id=None if release is None else release.id,
        measure=measure,
        comparator=comparator,
        strike_lo=D(strike),
        strike_hi=None if strike_hi is None else D(strike_hi),
        unit=unit,
        resolution_precision=precision,
        resolution_source=source,
        mapping_status=status,
        mapping_confidence=D("1.00"),
        mapped_by="rules:v2",
        created_at=NOW,
        updated_at=NOW,
    )
    session.add(row)
    await session.commit()
    return row


async def add_snapshot(
    session: AsyncSession,
    contract: EventContract,
    age: timedelta = timedelta(seconds=10),
    book: Any = None,
    at: datetime = NOW,
) -> MarketSnapshot:
    snap = MarketSnapshot(
        contract_id=contract.id,
        ts=at - age,
        venue_ts=None,
        book=GOOD_BOOK if book is None else book,
    )
    session.add(snap)
    await session.commit()
    return snap


async def price(
    session: AsyncSession, release: Release, models: Any = None, asof: datetime = NOW, **kw: Any
) -> ReleasePricing:
    return await price_release(
        session, PROVIDER, release, asof, models=stub() if models is None else models, **kw
    )


async def test_prices_every_mapped_contract(
    session: AsyncSession, seed_synthetic_cpi: Seeder
) -> None:
    truth = await seed_synthetic_cpi(session)
    release = Release(
        series=Series.CPI, period=f"{truth.ref_month:%Y-%m}", scheduled_at=truth.ref_first_vintage
    )
    session.add(release)
    await session.commit()
    asof = truth.asof
    a = await add_contract(session, release, "A", strike="0.2")
    b = await add_contract(session, release, "B", strike="0.3")
    c = await add_contract(
        session,
        release,
        "C",
        venue=Venue.POLYMARKET,
        event="P",
        measure=Measure.CPI_YOY,
        strike="3.0",
    )
    d = await add_contract(
        session, release, "D", measure=Measure.CORE_CPI_YOY, comparator=Comparator.GT, strike="3.1"
    )
    await add_contract(session, release, "U", status=MappingStatus.UNMAPPED)
    snaps = [await add_snapshot(session, x, at=asof) for x in (a, b, c, d)]
    out = await MODELS[Series.CPI].predict(session, PROVIDER, release, asof)

    result = await price_release(session, PROVIDER, release, asof)

    assert result.release_id == release.id
    assert result.asof == asof
    assert result.model_version == "cpi-ridge-v1"
    assert result.unpriceable == {}
    assert set(result.distributions) == {
        Measure.CPI_MOM,
        Measure.CORE_CPI_MOM,
        Measure.CPI_YOY,
        Measure.CORE_CPI_YOY,
    }
    assert result.distributions[Measure.CPI_MOM]["family"] == "student_t"
    assert set(result.distributions[Measure.CPI_MOM]) == {"family", "mu", "nu", "sigma"}
    assert [p.contract_id for p in result.contracts] == [a.id, b.id, c.id, d.id]
    assert len(result.inputs_hash) == 64
    for p, row, snap in zip(result.contracts, (a, b, c, d), snaps, strict=True):
        terms = terms_from_row(row)
        assert p.venue == row.venue
        assert p.terms == terms
        assert p.snapshot_id == snap.id
        assert p.quote_age_s == 10
        assert p.fair_yes == fair_prob(terms, out.distributions[terms.measure])
        book = OrderBook.from_json(GOOD_BOOK, asof)
        yes, no = evaluate_edge(p.fair_yes, book, row.venue, 10, asof)
        assert (p.yes, p.no) == (yes, no)
        assert p.quote == best_side((yes, no))
        assert isinstance(p.quote, EdgeQuote)
    # The strike sits near its own measure's mean, so any other measure's distribution
    # (or a mixed-up contract) gives a different, non-degenerate fair value.
    core = result.contracts[3]
    assert D("0.2") < core.fair_yes < D("0.8")
    own = out.distributions[Measure.CORE_CPI_YOY]
    assert core.fair_yes == fair_prob(core.terms, own)
    for m, other in out.distributions.items():
        if m is not Measure.CORE_CPI_YOY:
            assert core.fair_yes != fair_prob(core.terms, other)
    assert isinstance(result.distributions[Measure.CPI_MOM], MappingProxyType)


async def test_result_independent_of_ambient_decimal_context(
    session: AsyncSession, seed_synthetic_cpi: Seeder
) -> None:
    """I2: the whole body, model.predict included, runs under a pinned Decimal context."""
    truth = await seed_synthetic_cpi(session)
    release = Release(
        series=Series.CPI, period=f"{truth.ref_month:%Y-%m}", scheduled_at=truth.ref_first_vintage
    )
    session.add(release)
    await session.commit()
    rows = [
        await add_contract(session, release, "A", strike="0.2"),
        await add_contract(
            session, release, "C", venue=Venue.POLYMARKET, event="P",
            measure=Measure.CPI_YOY, strike="3.0",
        ),
        await add_contract(
            session, release, "D", measure=Measure.CORE_CPI_YOY, comparator=Comparator.GT,
            strike="3.1",
        ),
    ]  # fmt: skip
    for row in rows:
        await add_snapshot(session, row, at=truth.asof)
    baseline = await price_release(session, PROVIDER, release, truth.asof)
    assert len(baseline.contracts) == 3

    ctx = getcontext()
    saved = (ctx.prec, ctx.rounding)
    try:
        ctx.prec = 6
        ctx.rounding = ROUND_DOWN
        skewed = await price_release(session, PROVIDER, release, truth.asof)
        assert (getcontext().prec, getcontext().rounding) == (6, ROUND_DOWN)  # caller's untouched
    finally:
        ctx.prec, ctx.rounding = saved
    assert skewed.inputs_hash == baseline.inputs_hash
    assert [p.fair_yes for p in skewed.contracts] == [p.fair_yes for p in baseline.contracts]
    assert skewed == baseline


async def test_unmapped_and_other_release_contracts_excluded(session: AsyncSession) -> None:
    r1 = await make_release(session, "2026-10")
    r2 = await make_release(session, "2026-11")
    a = await add_contract(session, r1, "A")
    await add_contract(session, r2, "B")
    await add_contract(session, r1, "U", status=MappingStatus.UNMAPPED)
    result = await price(session, r1)
    assert [p.contract_id for p in result.contracts] == [a.id]
    assert result.rejected == {}  # the unmapped row is never even loaded


async def test_stale_and_missing_snapshots_unavailable(session: AsyncSession) -> None:
    release = await make_release(session)
    none = await add_contract(session, release, "none")
    stale = await add_contract(session, release, "stale")
    edge60 = await add_contract(session, release, "edge60")
    edge59 = await add_contract(session, release, "edge59")
    edge_over = await add_contract(session, release, "over")
    future_only = await add_contract(session, release, "future")
    frac = await add_contract(session, release, "frac")
    await add_snapshot(session, stale, timedelta(seconds=61))
    await add_snapshot(session, edge60, timedelta(seconds=60))
    await add_snapshot(session, edge59, timedelta(seconds=59))
    await add_snapshot(session, edge_over, timedelta(seconds=60, milliseconds=1))
    await add_snapshot(session, future_only, timedelta(seconds=-5))
    await add_snapshot(session, frac, timedelta(seconds=59, milliseconds=600))

    result = await price(session, release)
    by_id = {p.contract_id: p for p in result.contracts}

    p = by_id[none.id]
    assert (p.snapshot_id, p.quote_age_s) == (None, None)
    assert p.quote == p.yes == p.no == Unavailable("no_snapshot")

    p = by_id[stale.id]
    assert p.quote == p.yes == p.no == Unavailable("stale_snapshot")
    assert p.quote_age_s == 61
    assert p.snapshot_id is not None

    # Strictly greater than max_quote_age is stale; exactly equal is fresh.
    assert isinstance(by_id[edge60.id].quote, EdgeQuote)
    assert by_id[edge60.id].quote_age_s == 60
    assert isinstance(by_id[edge59.id].quote, EdgeQuote)
    assert by_id[edge59.id].quote_age_s == 59
    assert by_id[edge_over.id].quote == Unavailable("stale_snapshot")
    assert by_id[edge_over.id].quote_age_s == 60  # truncated
    assert isinstance(by_id[frac.id].quote, EdgeQuote)
    assert by_id[frac.id].quote_age_s == 59  # truncated, not rounded to 60

    # A snapshot after asof is never used.
    assert by_id[future_only.id].quote == Unavailable("no_snapshot")
    assert by_id[future_only.id].snapshot_id is None


async def test_future_snapshot_ignored_in_favour_of_older(session: AsyncSession) -> None:
    release = await make_release(session)
    c = await add_contract(session, release, "A")
    old = await add_snapshot(session, c, timedelta(seconds=20))
    await add_snapshot(session, c, timedelta(seconds=-5))
    (p,) = (await price(session, release)).contracts
    assert p.snapshot_id == old.id
    assert p.quote_age_s == 20


async def test_max_quote_age_parameter(session: AsyncSession) -> None:
    release = await make_release(session)
    c = await add_contract(session, release, "A")
    await add_snapshot(session, c, timedelta(seconds=30))
    (p,) = (await price(session, release, max_quote_age=timedelta(seconds=29))).contracts
    assert p.quote == Unavailable("stale_snapshot")
    (p,) = (await price(session, release, max_quote_age=timedelta(seconds=30))).contracts
    assert isinstance(p.quote, EdgeQuote)


@pytest.mark.parametrize(
    "bad_book",
    [
        {},  # KeyError: no "bids"
        [],  # TypeError: list indexed by str
        {"bids": "x", "asks": []},  # ValueError: side is not a sequence
        {"bids": [["abc", "1"]], "asks": []},  # ValueError: not numeric
        {"bids": [["0.3", "1"], ["0.4", "1"]], "asks": []},  # ValueError: not descending
        {"bids": [["1.5", "1"]], "asks": []},  # ValueError: price out of range
    ],
)
async def test_corrupt_snapshot_unavailable_not_raise(session: AsyncSession, bad_book: Any) -> None:
    release = await make_release(session)
    bad = await add_contract(session, release, "bad")
    good = await add_contract(session, release, "good")
    snap = await add_snapshot(session, bad, timedelta(seconds=7), book=bad_book)
    await add_snapshot(session, good)

    result = await price(session, release)
    by_id = {p.contract_id: p for p in result.contracts}
    p = by_id[bad.id]
    assert p.quote == p.yes == p.no == Unavailable("bad_snapshot")
    assert p.snapshot_id == snap.id
    assert p.quote_age_s == 7
    assert isinstance(by_id[good.id].quote, EdgeQuote)


async def test_insufficient_data_marks_unpriceable_not_raise(
    session: AsyncSession, seed_synthetic_cpi: Seeder
) -> None:
    truth = await seed_synthetic_cpi(session, months=30)
    release = Release(
        series=Series.CPI, period=f"{truth.ref_month:%Y-%m}", scheduled_at=truth.ref_first_vintage
    )
    session.add(release)
    await session.commit()
    a = await add_contract(session, release, "A")
    await add_contract(session, release, "B", measure=Measure.CPI_YOY, strike="3.0")
    await add_snapshot(session, a, at=truth.asof)

    result = await price_release(session, PROVIDER, release, truth.asof)

    assert result.unpriceable == {
        Measure.CPI_MOM: "insufficient_data:cpi_mom",  # the failing measure, for both
        Measure.CPI_YOY: "insufficient_data:cpi_mom",
    }
    assert result.contracts == ()
    assert result.model_version is None
    assert dict(result.distributions) == {}
    assert result.coherence == ()
    assert len(result.inputs_hash) == 64


async def test_insufficient_data_from_stub_model(session: AsyncSession) -> None:
    release = await make_release(session)
    await add_contract(session, release, "A")
    models = {Series.CPI: RaisingModel(InsufficientData("cpi_mom", 3, 36))}
    result = await price(session, release, models)
    assert result.unpriceable == {Measure.CPI_MOM: "insufficient_data:cpi_mom"}


async def test_no_model_marks_unpriceable(session: AsyncSession) -> None:
    release = await make_release(session)
    await add_contract(session, release, "A")
    await add_contract(session, release, "B", measure=Measure.CPI_YOY, strike="3.0")
    result = await price(session, release, models={})
    assert result.unpriceable == {Measure.CPI_MOM: "no_model", Measure.CPI_YOY: "no_model"}
    assert result.contracts == ()
    assert result.model_version is None


async def test_no_distribution_marks_measure_and_drops_contract(session: AsyncSession) -> None:
    release = await make_release(session)
    a = await add_contract(session, release, "A")
    await add_contract(
        session,
        release,
        "N",
        measure=Measure.NFP_CHANGE,
        strike="150000",
        precision=-3,
        unit="jobs",
    )
    result = await price(session, release)
    assert result.unpriceable == {Measure.NFP_CHANGE: "no_distribution"}
    assert [p.contract_id for p in result.contracts] == [a.id]


async def test_published_release_error_propagates(
    session: AsyncSession, seed_synthetic_cpi: Seeder
) -> None:
    truth = await seed_synthetic_cpi(session)
    published = add_months(truth.ref_month, -1)  # first-printed before truth.asof (R7)
    release = Release(
        series=Series.CPI, period=f"{published:%Y-%m}", scheduled_at=first_vintage_for(published)
    )
    session.add(release)
    await session.commit()
    await add_contract(session, release, "A")
    with pytest.raises(ValueError, match="already published"):
        await price_release(session, PROVIDER, release, truth.asof)


@pytest.mark.parametrize(
    "offset",
    [timedelta(0), timedelta(hours=1), timedelta(days=3)],
    ids=["at_scheduled", "plus_1h", "plus_3d"],
)
@pytest.mark.parametrize("series", [Series.CPI, Series.NFP, Series.FOMC])
async def test_price_at_or_after_scheduled_at_raises_before_any_model(
    session: AsyncSession, series: Series, offset: timedelta
) -> None:
    """R12: scheduled_at <= asof raises for every series, before the model is called."""
    release = Release(series=series, period="2026-10", scheduled_at=NOW - offset)
    session.add(release)
    await session.commit()
    model = RecordingModel()
    with pytest.raises(ValueError, match="release already published as of asof"):
        await price(session, release, {series: model})
    assert model.calls == []


@pytest.mark.parametrize("series", [Series.CPI, Series.NFP, Series.FOMC])
async def test_price_one_microsecond_before_scheduled_at_still_prices(
    session: AsyncSession, series: Series
) -> None:
    release = Release(series=series, period="2026-10", scheduled_at=NOW + timedelta(microseconds=1))
    session.add(release)
    await session.commit()
    model = RecordingModel()
    result = await price(session, release, {series: model})
    assert len(model.calls) == 1
    assert result.model_version == "stub-v1"


async def test_cpi_inside_vintage_window_after_release_time_raises(
    session: AsyncSession, seed_synthetic_cpi: Seeder
) -> None:
    """The CPI vintage is stamped 00:00 ET on D+1, so the model's own guard stays silent from
    the 08:30 ET release until midnight; price_release must still refuse (R12)."""
    truth = await seed_synthetic_cpi(session)
    release_time = truth.ref_first_vintage - timedelta(hours=15, minutes=30)  # 08:30 ET on D
    release = Release(
        series=Series.CPI, period=f"{truth.ref_month:%Y-%m}", scheduled_at=release_time
    )
    session.add(release)
    await session.commit()
    asof = release_time + timedelta(hours=1)
    assert asof < truth.ref_first_vintage  # the model's own R7 guard would not fire here
    out = await MODELS[Series.CPI].predict(session, PROVIDER, release, asof)  # prices fine
    assert out.model_version == "cpi-ridge-v1"
    with pytest.raises(ValueError, match="release already published as of asof"):
        await price_release(session, PROVIDER, release, asof)
    # Just before the release it still prices.
    await price_release(session, PROVIDER, release, release_time - timedelta(microseconds=1))


@pytest.mark.parametrize("qty", [0, -1])
async def test_probe_qty_below_one_raises_even_with_no_books(
    session: AsyncSession, qty: int
) -> None:
    """m2: validated at entry, so it cannot hide behind missing snapshots or R11's catches."""
    release = await make_release(session)
    await add_contract(session, release, "A")  # mapped, but no snapshot at all
    model = RecordingModel()
    with pytest.raises(ValueError, match="probe_qty"):
        await price(session, release, {Series.CPI: model}, probe_qty=qty)
    assert model.calls == []


async def test_negative_max_quote_age_raises_but_zero_is_allowed(session: AsyncSession) -> None:
    release = await make_release(session)
    c = await add_contract(session, release, "A")
    await add_snapshot(session, c, timedelta(0))
    with pytest.raises(ValueError, match="max_quote_age"):
        await price(session, release, max_quote_age=timedelta(microseconds=-1))
    (p,) = (await price(session, release, max_quote_age=timedelta(0))).contracts
    assert isinstance(p.quote, EdgeQuote)  # age 0 is not > 0
    (p,) = (await price(session, release, probe_qty=1)).contracts
    assert isinstance(p.yes, EdgeQuote)
    assert p.yes.qty == 1


async def test_other_model_errors_propagate(session: AsyncSession) -> None:
    release = await make_release(session)
    await add_contract(session, release, "A")
    with pytest.raises(RuntimeError, match="boom"):
        await price(session, release, {Series.CPI: RaisingModel(RuntimeError("boom"))})


async def test_contracts_ordered_by_id_and_keys_by_measure(session: AsyncSession) -> None:
    release = await make_release(session)
    a = await add_contract(session, release, "A", strike="0.1")
    b = await add_contract(session, release, "B", strike="0.2")
    c = await add_contract(session, release, "C", strike="0.3")
    for x in (a, b, c):
        await add_snapshot(session, x)
    # Updating a row moves its heap tuple, so an unordered scan would return b, c, a.
    a.title = "changed"
    await session.commit()
    result = await price(session, release)
    assert [p.contract_id for p in result.contracts] == [a.id, b.id, c.id]


async def test_unpriceable_keys_sorted(session: AsyncSession) -> None:
    release = await make_release(session)
    await add_contract(
        session, release, "Z", measure=Measure.NFP_CHANGE, strike="1000", precision=-3, unit="jobs"
    )
    await add_contract(session, release, "Y", measure=Measure.CORE_CPI_YOY, strike="3.0")
    await add_contract(session, release, "X", measure=Measure.CORE_CPI_MOM, strike="0.3")
    result = await price(session, release, models={})
    assert list(result.unpriceable) == [
        Measure.CORE_CPI_MOM,
        Measure.CORE_CPI_YOY,
        Measure.NFP_CHANGE,
    ]


# ---- parameters reach the collaborators --------------------------------------------------------


class RecordingModel:
    series = Series.CPI
    model_version = "recording"

    def __init__(self) -> None:
        self.calls: list[tuple[AsyncSession, MacroDataProvider, Release, datetime]] = []

    async def predict(
        self, session: AsyncSession, provider: MacroDataProvider, release: Release, asof: datetime
    ) -> ModelOutput:
        self.calls.append((session, provider, release, asof))
        return await StubModel().predict(session, provider, release, asof)


async def test_model_receives_exact_session_provider_release_asof(session: AsyncSession) -> None:
    release = await make_release(session)
    await add_contract(session, release, "A")
    model = RecordingModel()
    await price(session, release, {Series.CPI: model})
    assert len(model.calls) == 1
    got_session, got_provider, got_release, got_asof = model.calls[0]
    assert got_session is session
    assert got_provider is PROVIDER
    assert got_release is release
    assert got_asof == NOW


async def test_probe_qty_reaches_edge(session: AsyncSession) -> None:
    release = await make_release(session)
    c = await add_contract(session, release, "A")
    await add_snapshot(session, c)
    (p,) = (await price(session, release, probe_qty=7)).contracts
    assert isinstance(p.yes, EdgeQuote)
    assert isinstance(p.no, EdgeQuote)
    assert p.yes.qty == p.no.qty == 7
    assert p.yes.fee_cents == 13  # ceil(0.07 * 7 * 0.25 * 100) = ceil(12.25)
    book = OrderBook.from_json(GOOD_BOOK, NOW)
    assert (p.yes, p.no) == evaluate_edge(p.fair_yes, book, Venue.KALSHI, 7, NOW)


async def test_fee_schedule_injection_and_asof_date(session: AsyncSession) -> None:
    # asof is 00:00:05 New York on Nov 11; the snapshot (10 s older) is still Nov 10 there. The
    # schedule effective Nov 11 must apply, because fees are looked up at asof, not snapshot time.
    asof = datetime(2026, 11, 11, 5, 0, 5, tzinfo=UTC)
    release = await make_release(session)
    c = await add_contract(session, release, "A")
    await add_snapshot(session, c, at=asof)
    schedules = [
        FeeSchedule(Venue.KALSHI, date(2024, 1, 1), "kalshi_quadratic", D("0.07")),
        FeeSchedule(Venue.KALSHI, date(2026, 11, 11), "kalshi_quadratic", D("0.14")),
        FeeSchedule(Venue.POLYMARKET, date(2024, 1, 1), "zero", D("0")),
    ]
    (p,) = (await price(session, release, asof=asof, fee_schedules=schedules)).contracts
    assert isinstance(p.yes, EdgeQuote)
    assert p.yes.fee_cents == 35  # ceil(0.14 * 10 * 0.25 * 100); the 0.07 schedule gives 18
    (default,) = (await price(session, release, asof=asof)).contracts
    assert isinstance(default.yes, EdgeQuote)
    assert default.yes.fee_cents == 18


# ---- inputs_hash ----------------------------------------------------------------------------


async def _hashed_world(session: AsyncSession) -> tuple[Release, EventContract, EventContract]:
    release = await make_release(session)
    a = await add_contract(session, release, "A", rules="rules A")
    b = await add_contract(session, release, "B", venue=Venue.POLYMARKET, event="P", strike="0.4")
    await add_snapshot(session, a)
    await add_snapshot(session, b)
    return release, a, b


async def test_inputs_hash_stable_across_sessions(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    release, _, _ = await _hashed_world(session)
    first = await price(session, release)
    again = await price(session, release)
    async with sessions() as other:
        second = await price_release(other, PROVIDER, release, NOW, models=stub())
    assert first.inputs_hash == again.inputs_hash == second.inputs_hash
    assert first == again


def _contract_json(row: EventContract, snapshot_id: int | None) -> dict[str, Any]:
    return {
        "id": row.id,
        "measure": row.measure,
        "comparator": row.comparator,
        "strike_lo": None if row.strike_lo is None else format(row.strike_lo.normalize(), "f"),
        "strike_hi": None if row.strike_hi is None else format(row.strike_hi.normalize(), "f"),
        "unit": row.unit,
        "resolution_precision": row.resolution_precision,
        "resolution_source": row.resolution_source,
        "rules_hash": rules_hash(row),
        "snapshot_id": snapshot_id,
    }


async def test_inputs_hash_matches_documented_payload(session: AsyncSession) -> None:
    release, a, b = await _hashed_world(session)
    nfp = await add_contract(  # mapped but unpriceable: still hashed, snapshot null
        session, release, "N", measure=Measure.NFP_CHANGE, strike="1000", precision=-3, unit="jobs"
    )
    result = await price(session, release)
    snaps = (await session.execute(select(MarketSnapshot).order_by(MarketSnapshot.id))).scalars()
    s_a, s_b = list(snaps)
    dist = StudentT(D("0.30"), D("0.10"), D("5"))
    payload = {
        "pricing_version": "pricing:v1",
        "release_id": release.id,
        "asof": NOW,
        "model_version": "stub-v1",
        "distributions": {"cpi_mom": {"family": "student_t", **dist.params()}},
        "contracts": [
            _contract_json(a, s_a.id),
            _contract_json(b, s_b.id),
            _contract_json(nfp, None),
        ],
        "unpriceable": {"nfp_change": "no_distribution"},
        "rejected": {},
        "polymarket_event_sizes": {"P": 1},
        "probe_qty": 10,
        "max_quote_age_us": 60_000_000,
        "fee_schedules": {
            "kalshi": {
                "effective_from": date(2024, 1, 1),
                "kind": "kalshi_quadratic",
                "rate": D("0.07"),
            },
            "polymarket": {"effective_from": date(2024, 1, 1), "kind": "zero", "rate": "0"},
        },
    }
    assert PRICING_VERSION == "pricing:v1"
    assert result.inputs_hash == hashlib.sha256(canonical_json(payload)).hexdigest()


async def test_inputs_hash_sensitive_to_rules_text(session: AsyncSession) -> None:
    release, a, _ = await _hashed_world(session)
    before = await price(session, release)
    a.rules_text = "rules A, amended"
    await session.commit()
    after = await price(session, release)
    assert after.inputs_hash != before.inputs_hash
    assert after.contracts == before.contracts  # prices are untouched; only the hash moves


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("strike_lo", D("0.31")),
        ("strike_hi", D("0.5")),
        ("resolution_precision", 2),
        ("resolution_source", "BEA"),
        ("unit", "percent"),
        ("comparator", Comparator.GTE),
        ("measure", Measure.CORE_CPI_MOM),
    ],
)
async def test_inputs_hash_sensitive_to_contract_terms(
    session: AsyncSession, field: str, value: Any
) -> None:
    release, a, _ = await _hashed_world(session)
    before = await price(session, release)
    setattr(a, field, value)
    await session.commit()
    assert (await price(session, release)).inputs_hash != before.inputs_hash


async def test_terms_change_with_same_rules_text_moves_fair_and_hash(
    session: AsyncSession,
) -> None:
    release, a, _ = await _hashed_world(session)
    before = await price(session, release)
    a.strike_lo = D("0.2")  # spec-valid; rules_text untouched
    await session.commit()
    after = await price(session, release)
    assert [p.contract_id for p in after.contracts] == [p.contract_id for p in before.contracts]
    assert after.contracts[0].contract_id == a.id
    assert after.contracts[0].fair_yes != before.contracts[0].fair_yes
    assert after.inputs_hash != before.inputs_hash


async def test_spec_mismatch_after_remap_is_rejected_not_priced(session: AsyncSession) -> None:
    release, a, b = await _hashed_world(session)
    a.resolution_precision = 2  # same rules text, now off MEASURE_SPECS
    await session.commit()
    result = await price(session, release)
    assert result.rejected == {a.id: "bad_terms"}
    assert [p.contract_id for p in result.contracts] == [b.id]


async def test_inputs_hash_sensitive_to_unpriced_mapped_contract(session: AsyncSession) -> None:
    release, _, _ = await _hashed_world(session)
    n = await add_contract(
        session, release, "N", measure=Measure.NFP_CHANGE, strike="1000", precision=-3, unit="jobs"
    )
    before = await price(session, release)
    n.strike_lo = D("2000")
    await session.commit()
    assert (await price(session, release)).inputs_hash != before.inputs_hash


async def test_inputs_hash_sensitive_to_unpriceable_reason(session: AsyncSession) -> None:
    release = await make_release(session)
    await add_contract(session, release, "A")
    insufficient = {Series.CPI: RaisingModel(InsufficientData("cpi_mom", 3, 36))}
    h_none = (await price(session, release, models={})).inputs_hash
    h_insufficient = (await price(session, release, insufficient)).inputs_hash
    assert h_none != h_insufficient


async def test_inputs_hash_sensitive_to_which_measure_lacked_data(session: AsyncSession) -> None:
    release = await make_release(session)
    await add_contract(session, release, "A")
    h_mom = (
        await price(
            session, release, {Series.CPI: RaisingModel(InsufficientData("cpi_mom", 3, 36))}
        )
    ).inputs_hash
    h_rates = (
        await price(session, release, {Series.CPI: RaisingModel(InsufficientData("rates", 3, 36))})
    ).inputs_hash
    assert h_mom != h_rates


async def test_inputs_hash_sensitive_to_rejected(
    session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = await make_release(session)
    await add_contract(session, release, "A")
    bad = await add_contract(session, release, "B")  # no snapshot either way, same venue
    before = await price(session, release)
    real = terms_from_row

    def reject_b(row: EventContract) -> Any:
        if row.id == bad.id:
            raise ValueError("bad")
        return real(row)

    monkeypatch.setattr(release_mod, "terms_from_row", reject_b)
    after = await price(session, release)
    assert after.rejected == {bad.id: "bad_terms"}
    assert after.inputs_hash != before.inputs_hash


@pytest.mark.parametrize(
    ("raw", "text"),
    [
        ("0.3000", "0.3"),
        ("0.3", "0.3"),
        ("-0", "0"),
        ("-0.0000", "0"),
        ("0", "0"),
        ("0.00", "0"),
        ("1E+2", "100"),
        ("150000.0000", "150000"),
        ("-0.25", "-0.25"),
    ],
)
def test_dec_is_scale_independent_and_has_no_negative_zero(raw: str, text: str) -> None:
    assert release_mod._dec(D(raw)) == text


@pytest.mark.parametrize("bad", ["NaN", "sNaN", "Infinity", "-Infinity"])
def test_dec_rejects_non_finite(bad: str) -> None:
    with pytest.raises(ValueError, match="finite"):
        release_mod._dec(D(bad))


def test_dec_none_is_none() -> None:
    assert release_mod._dec(None) is None


async def test_nan_strike_is_rejected_and_hashed_without_aborting(session: AsyncSession) -> None:
    release, a, b = await _hashed_world(session)
    finite = await price(session, release)
    a.strike_lo = D("NaN")  # NUMERIC(14, 4) accepts NaN (but not Infinity)
    await session.commit()
    nan = await price(session, release)
    assert nan.rejected == {a.id: "bad_terms"}
    assert [p.contract_id for p in nan.contracts] == [b.id]
    assert nan.inputs_hash != finite.inputs_hash


@pytest.mark.parametrize(
    ("raw", "text"), [("NaN", "NaN"), ("Infinity", "Infinity"), ("0.30", "0.3")]
)
def test_strike_text(raw: str, text: str) -> None:
    assert release_mod._strike_text(D(raw)) == text
    assert release_mod._strike_text(None) is None


async def test_inputs_hash_sensitive_to_unmapped_polymarket_sibling(
    session: AsyncSession,
) -> None:
    """I4: an unmapped (or other-release) sibling changes the bracket's size n, hence the
    coherence verdict, hence must change the hash even though no priced input moved."""
    release, _, _ = await _hashed_world(session)  # polymarket event "P" has one priced contract
    before = await price(session, release)
    await add_contract(
        session, None, "Psib", venue=Venue.POLYMARKET, event="P", status=MappingStatus.UNMAPPED
    )
    after = await price(session, release)
    assert after.contracts == before.contracts
    assert after.coherence == ("polymarket event P: incomplete bracket (1 of 2 mapped)",)
    assert before.coherence == ("polymarket event P: incomplete bracket (1 of 1 mapped)",)
    assert after.inputs_hash != before.inputs_hash


async def test_inputs_hash_sensitive_to_newer_snapshot(session: AsyncSession) -> None:
    release, a, _ = await _hashed_world(session)
    before = await price(session, release)
    await add_snapshot(session, a, timedelta(seconds=2))
    after = await price(session, release)
    assert after.inputs_hash != before.inputs_hash


async def test_inputs_hash_ignores_snapshot_after_asof(session: AsyncSession) -> None:
    release, a, _ = await _hashed_world(session)
    before = await price(session, release)
    await add_snapshot(session, a, timedelta(seconds=-5))
    assert (await price(session, release)).inputs_hash == before.inputs_hash


async def test_inputs_hash_sensitive_to_probe_qty(session: AsyncSession) -> None:
    release, _, _ = await _hashed_world(session)
    h10 = (await price(session, release, probe_qty=10)).inputs_hash
    h11 = (await price(session, release, probe_qty=11)).inputs_hash
    assert h10 != h11


async def test_inputs_hash_sensitive_to_max_quote_age(session: AsyncSession) -> None:
    release, _, _ = await _hashed_world(session)
    h60 = (await price(session, release)).inputs_hash
    h61 = (await price(session, release, max_quote_age=timedelta(seconds=61))).inputs_hash
    h61_us = (
        await price(session, release, max_quote_age=timedelta(seconds=61, microseconds=1))
    ).inputs_hash
    assert len({h60, h61, h61_us}) == 3


async def test_inputs_hash_sensitive_to_asof(session: AsyncSession) -> None:
    release, _, _ = await _hashed_world(session)
    h0 = (await price(session, release)).inputs_hash
    h1 = (await price(session, release, asof=NOW + timedelta(seconds=1))).inputs_hash
    assert h0 != h1


async def test_inputs_hash_sensitive_to_model_version_and_params(session: AsyncSession) -> None:
    release, _, _ = await _hashed_world(session)
    base = (await price(session, release)).inputs_hash
    assert (await price(session, release, stub(version="stub-v2"))).inputs_hash != base
    assert (await price(session, release, stub(mu="0.31"))).inputs_hash != base


def _schedules(
    effective_from: date = date(2024, 1, 1),
    kind: Any = "kalshi_quadratic",
    rate: str = "0.07",
) -> Sequence[FeeSchedule]:
    return [
        FeeSchedule(Venue.KALSHI, effective_from, kind, D(rate)),
        FeeSchedule(Venue.POLYMARKET, date(2024, 1, 1), "zero", D("0")),
    ]


@pytest.mark.parametrize(
    "changed",
    [
        _schedules(effective_from=date(2026, 1, 1)),
        _schedules(rate="0.08"),
        _schedules(kind="zero", rate="0"),
    ],
    ids=["effective_from", "rate", "kind"],
)
async def test_inputs_hash_sensitive_to_fee_schedule(
    session: AsyncSession, changed: Sequence[FeeSchedule]
) -> None:
    release, _, _ = await _hashed_world(session)
    base = await price(session, release, fee_schedules=_schedules())
    assert base.inputs_hash == (await price(session, release)).inputs_hash  # packaged == same
    assert (await price(session, release, fee_schedules=changed)).inputs_hash != base.inputs_hash


async def test_inputs_hash_sensitive_to_pricing_version(
    session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, _, _ = await _hashed_world(session)
    before = await price(session, release)
    monkeypatch.setattr(release_mod, "PRICING_VERSION", "pricing:v2")
    assert (await price(session, release)).inputs_hash != before.inputs_hash


async def test_missing_fee_schedule_propagates(session: AsyncSession) -> None:
    release, _, _ = await _hashed_world(session)
    with pytest.raises(LookupError):
        await price(session, release, fee_schedules=[])


async def test_inputs_hash_defined_when_nothing_priced(session: AsyncSession) -> None:
    release = await make_release(session)
    empty = await price(session, release)
    assert empty.contracts == ()
    assert len(empty.inputs_hash) == 64


# ---- per-contract rejection (R11) -------------------------------------------------------------


async def test_malformed_contracts_rejected_alone(session: AsyncSession) -> None:
    release = await make_release(session)
    good = await add_contract(session, release, "G")
    await add_snapshot(session, good)
    no_hi = await add_contract(  # BETWEEN needs strike_hi
        session, release, "BT", comparator=Comparator.BETWEEN, strike="0.2"
    )
    inverted = await add_contract(
        session, release, "INV", comparator=Comparator.BETWEEN, strike="0.5", strike_hi="0.2"
    )
    bp = await add_contract(session, release, "BP", unit="bp", strike="0")  # not a CPI unit
    stale = await add_contract(  # a rules:v1 NFP row: precision 0 instead of -3
        session,
        release,
        "V1",
        measure=Measure.NFP_CHANGE,
        strike="150000",
        precision=0,
        unit="jobs",
    )
    nan = await add_contract(session, release, "NAN", strike="NaN")  # NUMERIC accepts NaN
    other = await add_contract(session, release, "G2", strike="0.4")
    await add_snapshot(session, other)

    result = await price(session, release)
    assert [p.contract_id for p in result.contracts] == [good.id, other.id]
    assert result.rejected == {
        no_hi.id: "bad_terms",
        inverted.id: "bad_terms",
        bp.id: "bad_terms",
        stale.id: "bad_terms",
        nan.id: "bad_terms",
    }
    assert list(result.rejected) == sorted(result.rejected)
    assert isinstance(result.rejected, MappingProxyType)
    assert result.unpriceable == {}  # a rejected contract's measure is not "unpriceable"


class CategoricalStub:
    """Returns a categorical for a continuous measure: every CPI contract then mismatches."""

    series = Series.CPI
    model_version = "categorical-stub"

    async def predict(
        self, session: AsyncSession, provider: MacroDataProvider, release: Release, asof: datetime
    ) -> ModelOutput:
        dist = Categorical(((0, D("1")),))
        return ModelOutput(self.model_version, {Measure.CPI_MOM: dist}, 1, {})


async def test_distribution_family_mismatch_rejected_alone(session: AsyncSession) -> None:
    release = await make_release(session)
    a = await add_contract(session, release, "A")
    result = await price(session, release, {Series.CPI: CategoricalStub()})
    assert result.contracts == ()
    assert result.rejected == {a.id: "terms_distribution_mismatch"}


@pytest.mark.parametrize("exc", [InvalidOperation("bad"), DivisionByZero("bad"), Overflow("bad")])
async def test_arithmetic_errors_reject_the_contract_alone(
    session: AsyncSession, monkeypatch: pytest.MonkeyPatch, exc: ArithmeticError
) -> None:
    """I3: the per-contract catches cover ArithmeticError (InvalidOperation etc.), not just
    ValueError, for both terms_from_row and fair_prob."""
    release = await make_release(session)
    good = await add_contract(session, release, "G")
    await add_snapshot(session, good)
    bad_terms = await add_contract(session, release, "BT", strike="0.1")
    bad_fair = await add_contract(session, release, "BF", strike="0.2")
    real_terms, real_fair = terms_from_row, fair_prob

    def terms(row: EventContract) -> Any:
        if row.id == bad_terms.id:
            raise exc
        return real_terms(row)

    def fair(t: Any, dist: Any) -> Any:
        if t.strike_lo == D("0.2"):
            raise exc
        return real_fair(t, dist)

    monkeypatch.setattr(release_mod, "terms_from_row", terms)
    monkeypatch.setattr(release_mod, "fair_prob", fair)
    result = await price(session, release)
    assert [p.contract_id for p in result.contracts] == [good.id]
    assert result.rejected == {
        bad_terms.id: "bad_terms",
        bad_fair.id: "terms_distribution_mismatch",
    }


# ---- coherence ------------------------------------------------------------------------------


PARTITION = [
    (Comparator.LTE, "0.1"),
    (Comparator.EQ, "0.2"),
    (Comparator.EQ, "0.3"),
    (Comparator.EQ, "0.4"),
    (Comparator.GT, "0.4"),
]


async def _bracket(
    session: AsyncSession,
    release: Release,
    event: str,
    parts: list[tuple[Comparator, str]],
    tag: str,
) -> list[EventContract]:
    return [
        await add_contract(
            session,
            release,
            f"{tag}{i}",
            venue=Venue.POLYMARKET,
            event=event,
            comparator=cmp,
            strike=k,
        )
        for i, (cmp, k) in enumerate(parts)
    ]


async def test_coherence_clean_on_single_distribution(session: AsyncSession) -> None:
    release = await make_release(session)
    # Kalshi threshold ladder (shared event id; bracket rules are Polymarket-only).
    for i, k in enumerate(("0.1", "0.2", "0.3", "0.4")):
        await add_contract(session, release, f"K{i}", strike=k, event="KE")
    await _bracket(session, release, "PE", PARTITION, "P")
    result = await price(session, release)
    assert len(result.contracts) == 9
    assert result.coherence == ()


async def test_bracket_complete_partition_with_bad_sum_reports_sum(session: AsyncSession) -> None:
    release = await make_release(session)
    # No 0.4 bucket: a structurally complete partition whose probabilities miss ~0.2 of mass.
    await _bracket(
        session, release, "PS", [PARTITION[0], PARTITION[1], PARTITION[2], PARTITION[4]], "S"
    )
    result = await price(session, release)
    (msg,) = result.coherence
    assert msg.startswith("polymarket event PS: bracket probabilities sum to ")


async def test_bracket_three_of_four_mapped_is_incomplete(session: AsyncSession) -> None:
    release = await make_release(session)
    await _bracket(session, release, "PI", PARTITION[:3] + PARTITION[4:], "I")
    await add_contract(
        session, None, "Iun", venue=Venue.POLYMARKET, event="PI", status=MappingStatus.UNMAPPED
    )
    result = await price(session, release)
    assert result.coherence == ("polymarket event PI: incomplete bracket (4 of 5 mapped)",)


async def test_bracket_unmapped_sibling_counts_toward_n(session: AsyncSession) -> None:
    release = await make_release(session)
    other = await make_release(session, "2026-11")
    await _bracket(session, release, "PU", [PARTITION[0], PARTITION[4]], "U")
    await add_contract(  # unmapped, no release
        session, None, "Ux", venue=Venue.POLYMARKET, event="PU", status=MappingStatus.UNMAPPED
    )
    await add_contract(session, other, "Uy", venue=Venue.POLYMARKET, event="PU")  # other release
    result = await price(session, release)
    assert result.coherence == ("polymarket event PU: incomplete bracket (2 of 4 mapped)",)


async def test_bracket_unpriced_sibling_is_incomplete(session: AsyncSession) -> None:
    release = await make_release(session)
    await _bracket(session, release, "PN", PARTITION[:3] + PARTITION[4:], "N")
    await add_contract(  # mapped to this release but the model has no distribution for it
        session,
        release,
        "Nx",
        venue=Venue.POLYMARKET,
        event="PN",
        measure=Measure.NFP_CHANGE,
        strike="1000",
        unit="jobs",
    )
    result = await price(session, release)
    assert result.coherence == ("polymarket event PN: incomplete bracket (4 of 5 mapped)",)


async def test_bracket_two_lower_tails_is_not_a_partition(session: AsyncSession) -> None:
    release = await make_release(session)
    parts = [(Comparator.LT, "0.2"), (Comparator.LTE, "0.1"), (Comparator.GT, "0.4")]
    await _bracket(session, release, "PL", parts, "L")
    result = await price(session, release)
    assert result.coherence == ("polymarket event PL: incomplete bracket (3 of 3 mapped)",)


async def test_bracket_without_upper_tail_is_not_a_partition(session: AsyncSession) -> None:
    release = await make_release(session)
    await _bracket(session, release, "PW", PARTITION[:4], "W")
    result = await price(session, release)
    assert result.coherence == ("polymarket event PW: incomplete bracket (4 of 4 mapped)",)


async def test_bracket_lone_contract_is_incomplete(session: AsyncSession) -> None:
    release = await make_release(session)
    await add_contract(session, release, "P3", venue=Venue.POLYMARKET, event="LONE", strike="0.9")
    result = await price(session, release)
    assert result.coherence == ("polymarket event LONE: incomplete bracket (1 of 1 mapped)",)


async def test_coherence_reports_sorted_violations(
    session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = await make_release(session)
    # A single distribution never yields a non-monotonic ladder, so the two off-grid GT strikes
    # (0.25, 0.35) are given inverted probabilities: more mass at the higher strike.
    forced = {D("0.25"): D("0.2"), D("0.35"): D("0.6")}
    real = fair_prob

    def skewed(terms: Any, dist: Any) -> Any:
        if terms.comparator == Comparator.GT and terms.strike_lo in forced:
            return to_prob(forced[terms.strike_lo])
        return real(terms, dist)

    monkeypatch.setattr(release_mod, "fair_prob", skewed)
    await add_contract(session, release, "K1", strike="0.25")
    await add_contract(session, release, "K2", strike="0.35")
    # On Polymarket the same pair, as one event, is also not a partition; the second event is a
    # complete partition with a bad sum.
    await add_contract(session, release, "P1", venue=Venue.POLYMARKET, event="PM", strike="0.25")
    await add_contract(session, release, "P2", venue=Venue.POLYMARKET, event="PM", strike="0.35")
    await _bracket(session, release, "PS", [PARTITION[0], PARTITION[1], PARTITION[4]], "S")
    result = await price(session, release)
    assert len(result.coherence) == 4
    assert result.coherence == tuple(sorted(result.coherence))
    kalshi, incomplete, bad_sum, poly = result.coherence
    assert kalshi.startswith("kalshi: cpi_mom gt: strike 0.2500 prob 0.20000 vs strike 0.3500")
    assert incomplete == "polymarket event PM: incomplete bracket (2 of 2 mapped)"
    assert bad_sum.startswith("polymarket event PS: bracket probabilities sum to ")
    assert poly.startswith("polymarket: cpi_mom gt: strike 0.2500 prob 0.20000 vs strike 0.3500")


# ---- read-only ------------------------------------------------------------------------------


async def _counts(session: AsyncSession) -> dict[str, int]:
    return {
        t.name: (await session.execute(select(func.count()).select_from(t))).scalar_one()
        for t in Base.metadata.sorted_tables
    }


async def test_no_writes(session: AsyncSession) -> None:
    release, _, _ = await _hashed_world(session)
    before = await _counts(session)
    await price(session, release)
    assert await _counts(session) == before
    assert not session.new
    assert not session.dirty
    assert not session.deleted


async def test_callers_pending_and_dirty_state_is_not_autoflushed(session: AsyncSession) -> None:
    release, a, _ = await _hashed_world(session)
    pending = Release(series=Series.NFP, period="pending", scheduled_at=NOW)
    session.add(pending)
    a.title = "dirtied by the caller"
    assert a in session.dirty
    await price(session, release)
    assert pending in session.new
    assert pending.id is None
    assert a in session.dirty
    await session.rollback()
    left = (
        await session.execute(
            select(func.count()).select_from(Release).where(Release.period == "pending")
        )
    ).scalar_one()
    assert left == 0


class QueryingModel:
    """Runs its own ORM query inside predict, which would autoflush a pending object."""

    series = Series.CPI
    model_version = "querying"

    def __init__(self) -> None:
        self.seen: list[str] = []

    async def predict(
        self, session: AsyncSession, provider: MacroDataProvider, release: Release, asof: datetime
    ) -> ModelOutput:
        rows = (await session.execute(select(Release))).scalars().all()
        self.seen = sorted(r.period for r in rows)
        return await StubModel().predict(session, provider, release, asof)


async def test_model_queries_do_not_autoflush_callers_pending_objects(
    session: AsyncSession,
) -> None:
    release, _, _ = await _hashed_world(session)
    pending = Release(series=Series.NFP, period="pending-from-model", scheduled_at=NOW)
    session.add(pending)
    model = QueryingModel()
    await price(session, release, {Series.CPI: model})
    assert model.seen == ["2026-10"]  # the model's own select ran, but never saw the pending row
    assert pending in session.new
    assert pending.id is None
    await session.rollback()
    left = (
        await session.execute(
            select(func.count()).select_from(Release).where(Release.period == "pending-from-model")
        )
    ).scalar_one()
    assert left == 0


async def test_does_not_commit_or_roll_back_callers_transaction(session: AsyncSession) -> None:
    release, _, _ = await _hashed_world(session)
    marker = Release(series=Series.NFP, period="marker", scheduled_at=NOW)
    session.add(marker)
    await session.flush()  # uncommitted, inside the caller's transaction
    assert session.in_transaction()
    await price(session, release)
    assert session.in_transaction()  # not committed away...
    assert marker in session  # ...and the unit of work was not discarded
    assert marker.id is not None
    await session.rollback()
    left = (
        await session.execute(
            select(func.count()).select_from(Release).where(Release.period == "marker")
        )
    ).scalar_one()
    assert left == 0  # the marker was never committed

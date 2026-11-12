import hashlib
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from prediction_desk.audit.hashing import canonical_json
from prediction_desk.db.base import Base
from prediction_desk.db.models import EventContract, MarketSnapshot, Release
from prediction_desk.domain.enums import Comparator, MappingStatus, Measure, Series, Venue
from prediction_desk.forecasting.base import ModelOutput
from prediction_desk.forecasting.distributions import StudentT
from prediction_desk.forecasting.realtime import InsufficientData
from prediction_desk.forecasting.registry import MODELS
from prediction_desk.ingestion.provider import MacroDataProvider
from prediction_desk.markets.terms import rules_hash, terms_from_row
from prediction_desk.markets.types import OrderBook
from prediction_desk.pricing import fees
from prediction_desk.pricing.edge import EdgeQuote, Unavailable, best_side, evaluate_edge
from prediction_desk.pricing.fair import fair_prob
from prediction_desk.pricing.fees import FeeSchedule
from prediction_desk.pricing.release import ReleasePricing, price_release
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
    release: Release,
    market_id: str,
    *,
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
        release_id=release.id,
        measure=measure,
        comparator=comparator,
        strike_lo=D(strike),
        unit=unit,
        resolution_precision=precision,
        resolution_source="BLS",
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
    await add_contract(session, release, "U", status=MappingStatus.UNMAPPED)
    snaps = [await add_snapshot(session, x, at=asof) for x in (a, b, c)]
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
    assert [p.contract_id for p in result.contracts] == [a.id, b.id, c.id]
    assert len(result.inputs_hash) == 64
    for p, row, snap in zip(result.contracts, (a, b, c), snaps, strict=True):
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


async def test_unmapped_and_other_release_contracts_excluded(session: AsyncSession) -> None:
    r1 = await make_release(session, "2026-10")
    r2 = await make_release(session, "2026-11")
    a = await add_contract(session, r1, "A")
    await add_contract(session, r2, "B")
    await add_contract(session, r1, "U", status=MappingStatus.UNMAPPED)
    result = await price(session, r1)
    assert [p.contract_id for p in result.contracts] == [a.id]


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
        Measure.CPI_MOM: "insufficient_data",
        Measure.CPI_YOY: "insufficient_data",
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
    assert result.unpriceable == {Measure.CPI_MOM: "insufficient_data"}


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
    await add_contract(session, release, "Z", measure=Measure.NFP_CHANGE, strike="1000", unit="j")
    await add_contract(session, release, "Y", measure=Measure.CORE_CPI_YOY, strike="3.0")
    await add_contract(session, release, "X", measure=Measure.CORE_CPI_MOM, strike="0.3")
    result = await price(session, release, models={})
    assert list(result.unpriceable) == [
        Measure.CORE_CPI_MOM,
        Measure.CORE_CPI_YOY,
        Measure.NFP_CHANGE,
    ]


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


async def test_inputs_hash_matches_documented_payload(session: AsyncSession) -> None:
    release, a, b = await _hashed_world(session)
    result = await price(session, release)
    snaps = (await session.execute(select(MarketSnapshot).order_by(MarketSnapshot.id))).scalars()
    s_a, s_b = list(snaps)
    dist = StudentT(D("0.30"), D("0.10"), D("5"))
    payload = {
        "release_id": release.id,
        "asof": NOW,
        "model_version": "stub-v1",
        "distributions": {"cpi_mom": {"family": "student_t", **dist.params()}},
        "contracts": [
            {"id": a.id, "rules_hash": rules_hash(a), "snapshot_id": s_a.id},
            {"id": b.id, "rules_hash": rules_hash(b), "snapshot_id": s_b.id},
        ],
        "probe_qty": 10,
        "fee_effective_from": {"kalshi": date(2024, 1, 1), "polymarket": date(2024, 1, 1)},
    }
    assert result.inputs_hash == hashlib.sha256(canonical_json(payload)).hexdigest()


async def test_inputs_hash_sensitive_to_rules_text(session: AsyncSession) -> None:
    release, a, _ = await _hashed_world(session)
    before = await price(session, release)
    a.rules_text = "rules A, amended"
    await session.commit()
    after = await price(session, release)
    assert after.inputs_hash != before.inputs_hash
    assert after.contracts == before.contracts  # prices are untouched; only the hash moves


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


async def test_inputs_hash_sensitive_to_fee_schedule(
    session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, _, _ = await _hashed_world(session)
    before = await price(session, release)
    newer = (
        FeeSchedule(Venue.KALSHI, date(2024, 1, 1), "kalshi_quadratic", D("0.07")),
        FeeSchedule(Venue.KALSHI, date(2026, 1, 1), "kalshi_quadratic", D("0.07")),
        FeeSchedule(Venue.POLYMARKET, date(2024, 1, 1), "zero", D("0")),
    )
    monkeypatch.setattr(fees, "_packaged_schedules", lambda: newer)
    after = await price(session, release)
    assert after.inputs_hash != before.inputs_hash
    assert after.contracts == before.contracts


async def test_inputs_hash_defined_when_nothing_priced(session: AsyncSession) -> None:
    release = await make_release(session)
    empty = await price(session, release)
    assert empty.contracts == ()
    assert len(empty.inputs_hash) == 64


# ---- coherence ------------------------------------------------------------------------------


async def test_coherence_clean_on_single_distribution(session: AsyncSession) -> None:
    release = await make_release(session)
    # Kalshi threshold ladder (shared event id; bracket sums are Polymarket-only).
    for i, k in enumerate(("0.1", "0.2", "0.3", "0.4")):
        await add_contract(session, release, f"K{i}", strike=k, event="KE")
    # Polymarket exhaustive partition of the reported value: <=0.1, 0.2, 0.3, 0.4, >0.4.
    parts = [
        (Comparator.LTE, "0.1"),
        (Comparator.EQ, "0.2"),
        (Comparator.EQ, "0.3"),
        (Comparator.EQ, "0.4"),
        (Comparator.GT, "0.4"),
    ]
    for i, (cmp, k) in enumerate(parts):
        await add_contract(
            session, release, f"P{i}", venue=Venue.POLYMARKET, event="PE", comparator=cmp, strike=k
        )
    result = await price(session, release)
    assert len(result.contracts) == 9
    assert result.coherence == ()


async def test_coherence_reports_sorted_violations(session: AsyncSession) -> None:
    release = await make_release(session)
    # Same strike on different reporting grids gives different probabilities: a monotonic
    # violation (equal strikes, unequal probabilities). On Polymarket the same pair, sharing an
    # event, also breaks the bracket sum.
    await add_contract(session, release, "K1", strike="0.3", precision=1)
    await add_contract(session, release, "K2", strike="0.3", precision=2)
    await add_contract(
        session, release, "P1", venue=Venue.POLYMARKET, event="PE", strike="0.3", precision=1
    )
    await add_contract(
        session, release, "P2", venue=Venue.POLYMARKET, event="PE", strike="0.3", precision=2
    )
    # A lone Polymarket contract is not a bracket set.
    await add_contract(session, release, "P3", venue=Venue.POLYMARKET, event="LONE", strike="0.9")
    result = await price(session, release)
    assert len(result.coherence) == 3
    assert result.coherence == tuple(sorted(result.coherence))
    kalshi, event, poly = result.coherence
    assert kalshi.startswith("kalshi: cpi_mom gt: strike 0.3000 prob ")
    assert event.startswith("polymarket event PE: bracket probabilities sum to ")
    assert poly.startswith("polymarket: cpi_mom gt: strike 0.3000 prob ")
    assert not any("LONE" in m for m in result.coherence)


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

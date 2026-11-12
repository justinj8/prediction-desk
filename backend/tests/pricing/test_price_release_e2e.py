"""Permanent end-to-end check of ``price_release`` with the REAL model registry.

Synthetic CPI / NFP / rates data are seeded, Kalshi and Polymarket contracts are mapped with the
terms ``MEASURE_SPECS`` fixes (never hand-typed units), fresh order books are stored, and every
release is priced through ``MODELS``. Nothing is stubbed: a regression anywhere between the
seeded data and the final ``ReleasePricing`` (model, terms validation, fair value, books, edges,
coherence, hash) shows up here.
"""

from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from prediction_desk.db.models import EventContract, MarketSnapshot, Release
from prediction_desk.domain.enums import Comparator, MappingStatus, Measure, Series, Venue
from prediction_desk.domain.types import to_prob
from prediction_desk.forecasting.registry import MODELS
from prediction_desk.ingestion.provider import MacroDataProvider
from prediction_desk.markets.mapping import MEASURE_SPECS
from prediction_desk.pricing.edge import EdgeQuote
from prediction_desk.pricing.fair import fair_prob
from prediction_desk.pricing.release import ReleasePricing, price_release
from synthetic_macro import (
    SyntheticCpiTruth,
    SyntheticNfpTruth,
    SyntheticRatesTruth,
)

D = Decimal
C = Comparator
PROVIDER = MacroDataProvider()
BOOK: dict[str, Any] = {"bids": [["0.40", "1000"]], "asks": [["0.50", "1000"]]}
CREATED = datetime(2020, 1, 1, tzinfo=UTC)
_NY = ZoneInfo("America/New_York")

# (comparator, strike) ladders / brackets
Leg = tuple[Comparator, str]


async def _map(
    session: AsyncSession,
    release: Release,
    asof: datetime,
    venue: Venue,
    event: str,
    measure: Measure,
    legs: Sequence[Leg],
) -> list[EventContract]:
    """Mapped contracts for one venue event, with fresh snapshots (5 s old at ``asof``)."""
    unit, precision, source = MEASURE_SPECS[measure]
    rows: list[EventContract] = []
    for i, (comparator, strike) in enumerate(legs):
        row = EventContract(
            venue=venue,
            venue_market_id=f"{event}-{i}",
            venue_event_id=event,
            yes_token_id=None,
            title=f"{event} {comparator.value} {strike}",
            rules_text=f"{measure.value} {comparator.value} {strike}",
            close_time=release.scheduled_at,
            release_id=release.id,
            measure=measure,
            comparator=comparator,
            strike_lo=D(strike),
            strike_hi=None,
            unit=unit,
            resolution_precision=precision,
            resolution_source=source,
            mapping_status=MappingStatus.MAPPED,
            mapping_confidence=D("1.00"),
            mapped_by="rules:v2",
            created_at=CREATED,
            updated_at=CREATED,
        )
        session.add(row)
        rows.append(row)
    await session.flush()
    session.add_all(
        MarketSnapshot(contract_id=r.id, ts=asof - timedelta(seconds=5), venue_ts=None, book=BOOK)
        for r in rows
    )
    await session.commit()
    return rows


async def _price_and_check(
    session: AsyncSession,
    sessions: async_sessionmaker[AsyncSession],
    release: Release,
    asof: datetime,
    mapped: Sequence[EventContract],
    model_version: str,
) -> ReleasePricing:
    result = await price_release(session, PROVIDER, release, asof)  # the real MODELS registry

    assert result.model_version == model_version
    assert result.unpriceable == {}
    assert result.rejected == {}
    assert result.coherence == ()
    assert sorted(p.contract_id for p in result.contracts) == sorted(r.id for r in mapped)
    assert [p.contract_id for p in result.contracts] == sorted(
        p.contract_id for p in result.contracts
    )
    by_id = {r.id: r for r in mapped}
    # Each contract is priced on its OWN measure's distribution (a mix-up would still leave
    # ladders monotone and complete brackets summing to 1).
    expected_out = await MODELS[release.series].predict(session, PROVIDER, release, asof)
    assert result.distributions.keys() == expected_out.distributions.keys()
    for p in result.contracts:
        assert p.fair_yes == fair_prob(p.terms, expected_out.distributions[p.terms.measure])
    for p in result.contracts:
        assert isinstance(p.fair_yes, Decimal)
        assert D(0) <= p.fair_yes <= D(1)
        assert to_prob(p.fair_yes) == p.fair_yes  # a valid Prob, not a raw Decimal
        assert p.terms.measure == by_id[p.contract_id].measure
        assert p.snapshot_id is not None
        assert p.quote_age_s == 5
        assert isinstance(p.yes, EdgeQuote)
        assert isinstance(p.no, EdgeQuote)
        assert isinstance(p.quote, EdgeQuote)

    again = await price_release(session, PROVIDER, release, asof)
    async with sessions() as fresh:
        other = await price_release(fresh, PROVIDER, release, asof)
    assert len(result.inputs_hash) == 64
    assert again.inputs_hash == result.inputs_hash
    assert other.inputs_hash == result.inputs_hash
    assert other == result
    return result


async def test_cpi_release_end_to_end(
    session: AsyncSession,
    sessions: async_sessionmaker[AsyncSession],
    seed_synthetic_cpi: Callable[..., Awaitable[SyntheticCpiTruth]],
) -> None:
    truth = await seed_synthetic_cpi(session)
    release = Release(
        series=Series.CPI, period=f"{truth.ref_month:%Y-%m}", scheduled_at=truth.ref_first_vintage
    )
    session.add(release)
    await session.commit()
    asof = truth.asof

    mapped: list[EventContract] = []
    # Kalshi: threshold ladders on the MoM measures (headline mean ~0.43, core ~0.45).
    mapped += await _map(
        session, release, asof, Venue.KALSHI, "KXCPI-REF", Measure.CPI_MOM,
        [(C.GT, k) for k in ("0.2", "0.3", "0.4", "0.5", "0.6")],
    )  # fmt: skip
    mapped += await _map(
        session, release, asof, Venue.KALSHI, "KXCPICORE-REF", Measure.CORE_CPI_MOM,
        [(C.GT, k) for k in ("0.3", "0.4", "0.5", "0.6")],
    )  # fmt: skip
    # Polymarket: complete YoY brackets (LTE tail, EQ buckets, GTE tail) around each mean
    # (headline ~0.12, core ~3.13).
    mapped += await _map(
        session, release, asof, Venue.POLYMARKET, "inflation-us-annual-ref", Measure.CPI_YOY,
        [(C.LTE, "-0.1"), (C.EQ, "0.0"), (C.EQ, "0.1"), (C.EQ, "0.2"), (C.GTE, "0.3")],
    )  # fmt: skip
    mapped += await _map(
        session, release, asof, Venue.POLYMARKET, "core-cpi-yoy-ref", Measure.CORE_CPI_YOY,
        [(C.LTE, "2.9"), (C.EQ, "3.0"), (C.EQ, "3.1"), (C.EQ, "3.2"), (C.GTE, "3.3")],
    )  # fmt: skip

    result = await _price_and_check(session, sessions, release, asof, mapped, "cpi-ridge-v1")
    assert set(result.distributions) == {
        Measure.CPI_MOM,
        Measure.CORE_CPI_MOM,
        Measure.CPI_YOY,
        Measure.CORE_CPI_YOY,
    }
    # The ladders really are priced (descending), not degenerate placeholders.
    kalshi_head = [p.fair_yes for p in result.contracts if p.terms.measure is Measure.CPI_MOM]
    assert kalshi_head == sorted(kalshi_head, reverse=True)
    assert kalshi_head[0] > kalshi_head[-1]


async def test_nfp_release_end_to_end(
    session: AsyncSession,
    sessions: async_sessionmaker[AsyncSession],
    seed_synthetic_nfp: Callable[..., Awaitable[SyntheticNfpTruth]],
) -> None:
    truth = await seed_synthetic_nfp(session)
    release = Release(
        series=Series.NFP, period=f"{truth.ref_month:%Y-%m}", scheduled_at=truth.ref_first_vintage
    )
    session.add(release)
    await session.commit()
    asof = truth.asof

    centre = int((truth.true_change_k * 1000).quantize(D("10000")))  # jobs, on the 1,000 grid
    strikes = [str(centre + step) for step in (-60_000, -30_000, 0, 30_000, 60_000)]
    mapped = await _map(
        session, release, asof, Venue.KALSHI, "KXPAYROLLS-REF", Measure.NFP_CHANGE,
        [(C.GT, k) for k in strikes],
    )  # fmt: skip

    result = await _price_and_check(session, sessions, release, asof, mapped, "nfp-ridge-v1")
    assert set(result.distributions) == {Measure.NFP_CHANGE}
    fairs = [p.fair_yes for p in result.contracts]
    assert fairs == sorted(fairs, reverse=True)
    assert fairs[0] > fairs[-1]


async def test_fomc_release_end_to_end(
    session: AsyncSession,
    sessions: async_sessionmaker[AsyncSession],
    seed_synthetic_rates: Callable[..., Awaitable[SyntheticRatesTruth]],
) -> None:
    # The bill implies a ~10bp cut with the meeting on the asof ET date: mass on {-25, 0}.
    truth = await seed_synthetic_rates(session, last_obs=date(2026, 3, 10), last_gap=D("-0.10"))
    meeting = truth.asof_et_date
    release = Release(
        series=Series.FOMC,
        period=meeting.isoformat(),
        scheduled_at=datetime.combine(meeting, time(14), tzinfo=_NY).astimezone(UTC),
    )
    session.add(release)
    await session.commit()
    asof = truth.asof

    mapped: list[EventContract] = []
    # Kalshi: LT/EQ/GT around the categorical outcomes ("Cut >25" is LT -25).
    mapped += await _map(
        session, release, asof, Venue.KALSHI, "KXFEDDECISION-REF", Measure.FOMC_CHANGE_BP,
        [(C.LT, "-25"), (C.EQ, "-25"), (C.EQ, "0"), (C.EQ, "25"), (C.GT, "25")],
    )  # fmt: skip
    # Polymarket: the complete bracket, tails at -50 / +50.
    mapped += await _map(
        session, release, asof, Venue.POLYMARKET, "fed-decision-in-march", Measure.FOMC_CHANGE_BP,
        [(C.LTE, "-50"), (C.EQ, "-25"), (C.EQ, "0"), (C.EQ, "25"), (C.GTE, "50")],
    )  # fmt: skip

    result = await _price_and_check(session, sessions, release, asof, mapped, "fomc-implied-v1")
    assert set(result.distributions) == {Measure.FOMC_CHANGE_BP}
    # implied -10bp: interp {-25: 0.4, 0: 0.6}, blended with the nearest-grid prior 0:
    # P(-25) = 0.2, P(0) = 0.8 (see test_fomc_model.test_implied_cut_interpolates).
    poly = {
        (p.terms.comparator, p.terms.strike_lo): p.fair_yes
        for p in result.contracts
        if p.venue is Venue.POLYMARKET
    }
    assert poly[(C.EQ, D("-25"))] == D("0.2")
    assert poly[(C.EQ, D("0"))] == D("0.8")
    assert poly[(C.GTE, D("50"))] == D(0)

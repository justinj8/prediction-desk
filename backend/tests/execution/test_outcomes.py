"""Official outcomes (U1): FRED first print -> raw -> reported, admin override, settlement guard."""

from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from desk_fixtures import BASE_TS, add_contract, make_book, make_release
from prediction_desk.audit.log import AuditLog
from prediction_desk.clock import SimClock
from prediction_desk.db.models import (
    AuditEvent,
    DataObservation,
    MeasureOutcome,
    Settlement,
)
from prediction_desk.domain.enums import (
    Comparator,
    DataSource,
    Measure,
    PositionSide,
    Series,
    Venue,
)
from prediction_desk.execution.outcomes import (
    contract_yes_won,
    official_raw_value,
    override_outcome,
    record_outcomes,
    reported_value,
)
from prediction_desk.ingestion.provider import MacroDataProvider
from prediction_desk.markets.mapping import MEASURE_SPECS, ContractTerms
from synthetic_macro import (
    SyntheticCpiTruth,
    SyntheticNfpTruth,
    add_months,
    nfp_first_vintage_for,
)

D = Decimal
FAR = datetime(2030, 1, 1, tzinfo=UTC)
PROVIDER = MacroDataProvider()
ADMIN = "admin:justin"


@pytest.mark.parametrize(
    ("measure", "raw", "expected"),
    [
        (Measure.CPI_MOM, "0.25", "0.3"),
        (Measure.CPI_MOM, "-0.05", "-0.1"),
        (Measure.CPI_MOM, "0.2499996", "0.2"),
        (Measure.CPI_YOY, "3.05", "3.1"),
        (Measure.NFP_CHANGE, "150000", "150000"),
        (Measure.NFP_CHANGE, "149500", "150000"),
        (Measure.NFP_CHANGE, "-149500", "-150000"),
        (Measure.FOMC_CHANGE_BP, "-75", "-50"),
        (Measure.FOMC_CHANGE_BP, "75", "50"),
        (Measure.FOMC_CHANGE_BP, "30", "25"),
        (Measure.FOMC_CHANGE_BP, "12.5", "25"),
        (Measure.FOMC_CHANGE_BP, "-12.5", "-25"),
        (Measure.FOMC_CHANGE_BP, "12.4", "0"),
        (Measure.FOMC_CHANGE_BP, "-25", "-25"),
    ],
)
def test_reported_value_half_up_on_grid(measure: Measure, raw: str, expected: str) -> None:
    got = reported_value(measure, D(raw))
    assert got == D(expected)
    assert measure in MEASURE_SPECS


def _terms(comparator: Comparator, lo: str, hi: str | None = None) -> ContractTerms:
    return ContractTerms(
        Measure.CPI_MOM, comparator, D(lo), None if hi is None else D(hi), "pct", 1, "BLS"
    )


@pytest.mark.parametrize(
    ("comparator", "lo", "hi", "reported", "won"),
    [
        (Comparator.GT, "0.2", None, "0.3", True),
        (Comparator.GT, "0.2", None, "0.2", False),
        (Comparator.GTE, "0.2", None, "0.2", True),
        (Comparator.GTE, "0.2", None, "0.1", False),
        (Comparator.LT, "0.2", None, "0.1", True),
        (Comparator.LT, "0.2", None, "0.2", False),
        (Comparator.LTE, "0.2", None, "0.2", True),
        (Comparator.LTE, "0.2", None, "0.3", False),
        (Comparator.EQ, "0.2", None, "0.2", True),
        (Comparator.EQ, "0.2", None, "0.3", False),
        (Comparator.BETWEEN, "0.1", "0.3", "0.1", True),
        (Comparator.BETWEEN, "0.1", "0.3", "0.3", True),
        (Comparator.BETWEEN, "0.1", "0.3", "0.4", False),
        (Comparator.BETWEEN, "0.1", "0.3", "0.0", False),
    ],
)
def test_contract_yes_won_each_comparator(
    comparator: Comparator, lo: str, hi: str | None, reported: str, won: bool
) -> None:
    assert contract_yes_won(_terms(comparator, lo, hi), D(reported)) is won


def test_between_without_upper_strike_is_an_error() -> None:
    with pytest.raises(ValueError, match="strike_hi"):
        contract_yes_won(_terms(Comparator.BETWEEN, "0.1"), D("0.2"))


def _mom(levels: dict[date, Decimal], month: date) -> Decimal:
    return 100 * (levels[month] / levels[add_months(month, -1)] - 1)


async def test_cpi_mom_from_first_print_matches_synthetic_truth(
    session: AsyncSession, seed_synthetic_cpi: Callable[..., Awaitable[SyntheticCpiTruth]]
) -> None:
    truth = await seed_synthetic_cpi(session)
    month = add_months(truth.ref_month, -5)
    release = await make_release(session, Series.CPI, f"{month:%Y-%m}")
    levels = truth.first_print_levels
    for measure, sid in (
        (Measure.CPI_MOM, "CPIAUCSL"),
        (Measure.CORE_CPI_MOM, "CPILFESL"),
    ):
        got = await official_raw_value(session, PROVIDER, release, measure, FAR)
        assert got is not None
        assert got[0] == _mom(levels[sid], month)
        assert got[1] == truth.first_vintage[month]
    for measure, sid in (
        (Measure.CPI_YOY, "CPIAUCSL"),
        (Measure.CORE_CPI_YOY, "CPILFESL"),
    ):
        got = await official_raw_value(session, PROVIDER, release, measure, FAR)
        assert got is not None
        yoy = 100 * (levels[sid][month] / levels[sid][add_months(month, -12)] - 1)
        assert got[0] == yoy


async def test_raw_value_is_not_pre_quantized(session: AsyncSession) -> None:
    """0.2499996 must stay below the 0.25 boundary: no 1e-6 rounding before the grid."""
    vts = datetime(2026, 9, 12, 5, tzinfo=UTC)
    for obs, value in ((date(2026, 7, 1), "1000.000000"), (date(2026, 8, 1), "1002.499996")):
        session.add(_obs("CPIAUCSL", obs, vts, D(value)))
    await session.commit()
    release = await make_release(session, Series.CPI, "2026-08")
    got = await official_raw_value(session, PROVIDER, release, Measure.CPI_MOM, FAR)
    assert got is not None
    assert D("0.2499996") <= got[0] < D("0.25")
    assert reported_value(Measure.CPI_MOM, got[0]) == D("0.2")


def _obs(series_id: str, obs_date: date, vintage: datetime, value: Decimal) -> DataObservation:
    return DataObservation(
        source=DataSource.FRED,
        series_id=series_id,
        obs_date=obs_date,
        vintage_ts=vintage,
        published_on=vintage.date(),
        value=value,
        ingested_at=vintage,
    )


async def test_later_revision_does_not_change_outcome(
    session: AsyncSession, seed_synthetic_cpi: Callable[..., Awaitable[SyntheticCpiTruth]]
) -> None:
    truth = await seed_synthetic_cpi(session)
    month = truth.revision.obs_date
    release = await make_release(session, Series.CPI, f"{month:%Y-%m}")
    before = await official_raw_value(session, PROVIDER, release, Measure.CPI_MOM, FAR)
    assert before is not None
    # a second, much later vintage of the same month with a very different index
    session.add(_obs("CPIAUCSL", month, FAR - timedelta(days=1), truth.revision.first_print * 2))
    await session.commit()
    after = await official_raw_value(session, PROVIDER, release, Measure.CPI_MOM, FAR)
    assert after == before
    assert before[0] == _mom(truth.first_print_levels["CPIAUCSL"], month)


async def test_nfp_change_in_jobs(
    session: AsyncSession, seed_synthetic_nfp: Callable[..., Awaitable[SyntheticNfpTruth]]
) -> None:
    truth = await seed_synthetic_nfp(session)
    month = add_months(truth.ref_month, -4)
    release = await make_release(session, Series.NFP, f"{month:%Y-%m}")
    got = await official_raw_value(session, PROVIDER, release, Measure.NFP_CHANGE, FAR)
    assert got is not None
    lv = truth.payems_levels
    assert got == ((lv[month] - lv[add_months(month, -1)]) * 1000, nfp_first_vintage_for(month))


async def _seed_dfedtaru(
    session: AsyncSession, d: date, before: str, after: str, vts: datetime | None = None
) -> None:
    for obs, v in ((d, before), (d + timedelta(days=1), after)):
        stamp = vts or datetime.combine(obs + timedelta(days=1), datetime.min.time(), UTC)
        session.add(_obs("DFEDTARU", obs, stamp, D(v)))
    await session.commit()


async def test_fomc_from_dfedtaru_step(session: AsyncSession) -> None:
    d = date(2026, 10, 28)
    await _seed_dfedtaru(session, d, "4.50", "4.25")
    release = await make_release(session, Series.FOMC, d.isoformat())
    got = await official_raw_value(session, PROVIDER, release, Measure.FOMC_CHANGE_BP, FAR)
    assert got == (D("-25"), datetime(2026, 10, 30, tzinfo=UTC))
    assert reported_value(Measure.FOMC_CHANGE_BP, got[0]) == D("-25")


async def test_missing_data_returns_none_and_record_skips(
    session: AsyncSession,
    sessions: async_sessionmaker[AsyncSession],
    seed_synthetic_cpi: Callable[..., Awaitable[SyntheticCpiTruth]],
) -> None:
    truth = await seed_synthetic_cpi(session)
    month = add_months(truth.ref_month, -3)
    # drop core CPI for the month: headline settles, core stays open
    for row in (
        await session.execute(
            select(DataObservation).where(
                DataObservation.series_id == "CPILFESL", DataObservation.obs_date == month
            )
        )
    ).scalars():
        await session.delete(row)
    await session.commit()
    release = await make_release(session, Series.CPI, f"{month:%Y-%m}")
    assert await official_raw_value(session, PROVIDER, release, Measure.CORE_CPI_MOM, FAR) is None
    # asof before the first vintage of the month: nothing is out yet
    early = truth.first_vintage[month] - timedelta(seconds=1)
    assert await official_raw_value(session, PROVIDER, release, Measure.CPI_MOM, early) is None

    clock = SimClock(FAR)
    written = await record_outcomes(sessions, PROVIDER, AuditLog(clock), clock, release.id)
    assert set(written) == {Measure.CPI_MOM, Measure.CPI_YOY}
    async with sessions() as s:
        rows = (await s.execute(select(MeasureOutcome))).scalars().all()
        assert {r.measure for r in rows} == set(written)
        assert all(r.source == "fred_first_print" for r in rows)
        assert all(r.recorded_by == "system:outcomes" for r in rows)
        assert all(r.recorded_at == FAR for r in rows)
        assert all(r.source_vintage_ts is not None for r in rows)
        events = (
            (await s.execute(select(AuditEvent).where(AuditEvent.action == "outcome.recorded")))
            .scalars()
            .all()
        )
        assert len(events) == len(rows)


async def test_record_does_not_overwrite_and_is_rerunnable(
    session: AsyncSession,
    sessions: async_sessionmaker[AsyncSession],
    seed_synthetic_nfp: Callable[..., Awaitable[SyntheticNfpTruth]],
) -> None:
    truth = await seed_synthetic_nfp(session)
    month = add_months(truth.ref_month, -2)
    release = await make_release(session, Series.NFP, f"{month:%Y-%m}")
    clock = SimClock(FAR)
    audit = AuditLog(clock)
    first = await record_outcomes(sessions, PROVIDER, audit, clock, release.id)
    assert set(first) == {Measure.NFP_CHANGE}
    async with sessions() as s, s.begin():
        await override_outcome(
            s, audit, clock, release_id=release.id, measure=Measure.NFP_CHANGE,
            raw_value=D("123456"), actor=ADMIN, reason="BLS note",
        )  # fmt: skip
    assert await record_outcomes(sessions, PROVIDER, audit, clock, release.id) == {}
    async with sessions() as s:
        row = await s.get(MeasureOutcome, (release.id, Measure.NFP_CHANGE))
        assert row is not None and row.source == "admin_override"


async def test_record_unknown_release_is_an_error(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    clock = SimClock(FAR)
    with pytest.raises(ValueError, match="release"):
        await record_outcomes(sessions, PROVIDER, AuditLog(clock), clock, 999)


async def test_override_requires_admin_and_audits_old_new(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    d = date(2026, 10, 28)
    await _seed_dfedtaru(session, d, "4.50", "4.25")
    release = await make_release(session, Series.FOMC, d.isoformat())
    clock = SimClock(FAR)
    audit = AuditLog(clock)
    await record_outcomes(sessions, PROVIDER, audit, clock, release.id)

    async def go(actor: str, reason: str, measure: Measure = Measure.FOMC_CHANGE_BP) -> None:
        async with sessions() as s, s.begin():
            await override_outcome(
                s, audit, clock, release_id=release.id, measure=measure,
                raw_value=D("0"), actor=actor, reason=reason,
            )  # fmt: skip

    for actor in ("system:outcomes", "justin", "admin:", "admin:   "):
        with pytest.raises(ValueError, match="actor"):
            await go(actor, "why")
    with pytest.raises(ValueError, match="reason"):
        await go(ADMIN, "  ")
    with pytest.raises(ValueError, match="measure"):
        await go(ADMIN, "why", Measure.CPI_MOM)
    async with sessions() as s:
        row = await s.get(MeasureOutcome, (release.id, Measure.FOMC_CHANGE_BP))
        assert row is not None and row.reported_value == D("-25")

    clock.set(FAR + timedelta(hours=1))
    await go(ADMIN, "Fed statement corrected")
    async with sessions() as s:
        row = await s.get(MeasureOutcome, (release.id, Measure.FOMC_CHANGE_BP))
        assert row is not None
        assert (row.raw_value, row.reported_value) == (D("0"), D("0"))
        assert (row.source, row.recorded_by, row.reason) == (
            "admin_override",
            ADMIN,
            "Fed statement corrected",
        )
        assert row.recorded_at == FAR + timedelta(hours=1)
        ev = (
            await s.execute(select(AuditEvent).where(AuditEvent.action == "outcome.overridden"))
        ).scalar_one()
        assert ev.actor == ADMIN
        assert D(ev.payload["old"]["reported_value"]) == D("-25")
        assert ev.payload["old"]["source"] == "fred_first_print"
        assert D(ev.payload["new"]["reported_value"]) == 0
        assert ev.payload["reason"] == "Fed statement corrected"


async def test_override_can_create_a_missing_outcome(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    release = await make_release(session, Series.NFP, "2026-10")
    clock = SimClock(FAR)
    async with sessions() as s, s.begin():
        await override_outcome(
            s, AuditLog(clock), clock, release_id=release.id, measure=Measure.NFP_CHANGE,
            raw_value=D("149500"), actor=ADMIN, reason="manual",
        )  # fmt: skip
    async with sessions() as s:
        row = await s.get(MeasureOutcome, (release.id, Measure.NFP_CHANGE))
        assert row is not None and row.reported_value == D("150000")
        ev = (await s.execute(select(AuditEvent))).scalar_one()
        assert ev.payload["old"] is None


async def test_override_unknown_release_is_an_error(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    clock = SimClock(FAR)
    async with sessions() as s, s.begin():
        with pytest.raises(ValueError, match="release"):
            await override_outcome(
                s, AuditLog(clock), clock, release_id=999, measure=Measure.NFP_CHANGE,
                raw_value=D("1"), actor=ADMIN, reason="x",
            )  # fmt: skip


async def test_override_after_settlement_refused(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    release = await make_release(session, Series.CPI, "2026-10")
    contract = await add_contract(
        session, release, venue=Venue.KALSHI, measure=Measure.CPI_MOM,
        comparator=Comparator.GT, strike_lo=D("0.2"),
    )  # fmt: skip
    book = await make_book(session)
    session.add(
        Settlement(
            book_id=book.id, contract_id=contract.id, side=PositionSide.YES, qty=1,
            yes_won=True, payout_cents=100, realized_pnl_cents=40, reported_value=D("0.3"),
            venue_yes_won=None, mismatch=False, settled_at=BASE_TS,
        )
    )  # fmt: skip
    await session.commit()
    clock = SimClock(FAR)
    audit = AuditLog(clock)
    with pytest.raises(ValueError, match="settled"):
        async with sessions() as s, s.begin():
            await override_outcome(
                s, audit, clock, release_id=release.id, measure=Measure.CPI_MOM,
                raw_value=D("0.1"), actor=ADMIN, reason="too late",
            )  # fmt: skip
    async with sessions() as s:
        assert (await s.execute(select(MeasureOutcome))).first() is None
        assert (await s.execute(select(AuditEvent))).first() is None


async def test_each_measure_family_returns_none_without_its_inputs(session: AsyncSession) -> None:
    vts = datetime(2026, 9, 12, 5, tzinfo=UTC)
    # the month is out but the prior month's level is not
    session.add(_obs("CPIAUCSL", date(2026, 8, 1), vts, D("1000")))
    session.add(_obs("PAYEMS", date(2026, 8, 1), vts, D("150000")))
    await session.commit()
    cpi = await make_release(session, Series.CPI, "2026-08")
    nfp = await make_release(session, Series.NFP, "2026-08")
    assert await official_raw_value(session, PROVIDER, cpi, Measure.CPI_MOM, FAR) is None
    assert await official_raw_value(session, PROVIDER, nfp, Measure.NFP_CHANGE, FAR) is None
    # no PAYEMS at all for another month
    nfp2 = await make_release(session, Series.NFP, "2025-01")
    assert await official_raw_value(session, PROVIDER, nfp2, Measure.NFP_CHANGE, FAR) is None

    # FOMC: nothing published, then published but after asof, then decision-day level missing
    d = date(2026, 10, 28)
    fomc = await make_release(session, Series.FOMC, d.isoformat())
    assert await official_raw_value(session, PROVIDER, fomc, Measure.FOMC_CHANGE_BP, FAR) is None
    session.add(
        _obs("DFEDTARU", d + timedelta(days=1), datetime(2026, 10, 30, tzinfo=UTC), D("4.25"))
    )
    await session.commit()
    early = datetime(2026, 10, 29, tzinfo=UTC)
    assert await official_raw_value(session, PROVIDER, fomc, Measure.FOMC_CHANGE_BP, early) is None
    assert await official_raw_value(session, PROVIDER, fomc, Measure.FOMC_CHANGE_BP, FAR) is None

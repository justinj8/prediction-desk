"""One CPI release through the whole desk: price, risk-decide, fill, mark, outcome, settle.

Synthetic CPI is seeded, two Kalshi GT contracts are mapped with fresh books, and the release is
priced with the real model registry. A pass-through officer reviews the slate on a REPLAY book,
every approval is filled by the paper broker, and after the clock passes the release the
first-print outcomes are recorded and the positions settle. The audit chain must verify and the
audit actions must appear in the order the desk ran.
"""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import timedelta
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from desk_fixtures import add_contract, add_snapshot
from prediction_desk.audit.log import AuditLog
from prediction_desk.clock import SimClock
from prediction_desk.db.models import (
    AuditEvent,
    BookState,
    DataObservation,
    Order,
    Position,
    Release,
    RiskDecision,
)
from prediction_desk.db.uow import unit_of_work
from prediction_desk.domain.enums import (
    BookMode,
    Comparator,
    DataSource,
    Measure,
    OrderStatus,
    Series,
    Slot,
    Venue,
)
from prediction_desk.execution.broker import PaperBroker
from prediction_desk.execution.outcomes import record_outcomes
from prediction_desk.execution.settlement import settle_release
from prediction_desk.ingestion.provider import MacroDataProvider
from prediction_desk.pricing.release import price_release
from prediction_desk.risk.book import create_book, mark_book
from prediction_desk.risk.engine import RiskEngine, SlateItem
from synthetic_macro import SyntheticCpiTruth

D = Decimal
PROVIDER = MacroDataProvider()


class FakeOfficer:
    """Approves every slate item at the engine's own quantity."""

    def __init__(self) -> None:
        self.reviewed = 0

    async def review(self, slate: Sequence[SlateItem]) -> Mapping[int, int]:
        self.reviewed += len(slate)
        return {item.contract_id: item.qty_engine for item in slate}


def _first_print(series_id: str, truth: SyntheticCpiTruth, mom_pct: str) -> DataObservation:
    """The reference month's first vintage: the prior level grown by ``mom_pct`` percent."""
    prior = truth.first_print_levels[series_id][max(m for m in truth.first_print_levels[series_id])]
    level = (prior * (1 + D(mom_pct) / 100)).quantize(D("0.001"))
    vintage = truth.ref_first_vintage
    return DataObservation(
        source=DataSource.FRED,
        series_id=series_id,
        obs_date=truth.ref_month,
        vintage_ts=vintage,
        published_on=vintage.date(),
        value=level,
        ingested_at=vintage,
    )


async def test_cpi_release_price_risk_fill_mark_settle(
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

    # Priced two hours before the release, well inside the 5 minute cutoff; the model sees
    # no data for the reference month yet.
    asof = truth.ref_first_vintage - timedelta(hours=2)
    clock = SimClock(asof)
    audit = AuditLog(clock)

    # Headline MoM is forecast near 0.43: "> 0.2" is very likely YES, "> 0.9" very likely NO.
    # Books: YES asks 0.60 / bids 0.50, so YES costs 0.60 and NO costs 0.50.
    bids, asks = [("0.50", "5000")], [("0.60", "5000")]
    yes_contract = await add_contract(
        session,
        release,
        venue=Venue.KALSHI,
        measure=Measure.CPI_MOM,
        comparator=Comparator.GT,
        strike_lo=D("0.2"),
        venue_event_id="KXCPI-E2E",
    )
    no_contract = await add_contract(
        session,
        release,
        venue=Venue.KALSHI,
        measure=Measure.CPI_MOM,
        comparator=Comparator.GT,
        strike_lo=D("0.9"),
        venue_event_id="KXCPI-E2E",
    )
    for contract in (yes_contract, no_contract):
        await add_snapshot(session, contract, asof - timedelta(seconds=5), bids, asks)

    async with unit_of_work(sessions) as s:
        book_id = (
            await create_book(
                s, audit, clock, mode=BookMode.REPLAY, replay_batch_id=1, cash_cents=1_000_000
            )
        ).id

    # 1. price
    pricing = await price_release(session, PROVIDER, release, asof)
    assert pricing.unpriceable == {} and pricing.rejected == {}
    assert len(pricing.contracts) == 2

    # 2. risk decision with a pass-through officer
    officer = FakeOfficer()
    engine = RiskEngine(clock, audit)
    result = await engine.evaluate(
        sessions,
        book_id=book_id,
        run_key="e2e-run",
        slot=Slot.T24H,
        pricing=pricing,
        healthy_venues=frozenset({Venue.KALSHI}),
        officer=officer,
    )
    assert result.killed is False
    assert len(result.decision_ids) == 2
    assert len(result.approved) == 2  # both sides carry a real edge
    assert officer.reviewed == 2

    # 3. fill every approval
    broker = PaperBroker(clock, audit)
    for approved in result.approved:
        order = await broker.submit(sessions, approved)
        assert order.status in (OrderStatus.FILLED, OrderStatus.PARTIAL)
        assert order.filled_qty > 0

    # 4. mark
    async with unit_of_work(sessions) as s:
        marked = await mark_book(s, audit, clock, book_id, asof)
        assert marked.kill_switch_on is False

    # 5. the release prints: headline +0.4% (> 0.2, not > 0.9), core +0.3%
    session.add_all(
        [_first_print("CPIAUCSL", truth, "0.4"), _first_print("CPILFESL", truth, "0.3")]
    )
    await session.commit()
    clock.set(truth.ref_first_vintage + timedelta(hours=1))
    outcomes = await record_outcomes(sessions, PROVIDER, audit, clock, release.id)
    assert outcomes[Measure.CPI_MOM] == D("0.4")
    assert Measure.CORE_CPI_MOM in outcomes

    # 6. settle
    settled = await settle_release(sessions, audit, clock, book_id=book_id, release_id=release.id)
    assert len(settled.settled) == 2
    assert settled.pending == () and settled.mismatches == ()

    async with sessions() as s:
        book = await s.get_one(BookState, book_id, populate_existing=True)
        positions = (await s.scalars(select(Position).where(Position.book_id == book_id))).all()
        orders = (await s.scalars(select(Order).where(Order.book_id == book_id))).all()
        decisions = {d.id for d in (await s.scalars(select(RiskDecision))).all()}
        events = (await s.scalars(select(AuditEvent).order_by(AuditEvent.seq))).all()
        chain = await AuditLog(clock).verify(s)

    assert len(positions) == 2
    assert all(p.settled for p in positions)
    assert {p.side.value for p in positions} == {"yes", "no"}
    assert book.cash_cents == 1_000_000 + sum(p.realized_pnl_cents for p in positions)
    assert book.cash_cents != 1_000_000
    assert len(orders) == 2
    assert all(o.risk_decision_id in decisions for o in orders)
    assert chain.ok

    actions = [e.action for e in events]
    assert actions[0] == "book.created"
    assert actions.count("risk.decided") == 1
    assert actions.count("order.submitted") == 2
    assert actions.count("outcome.recorded") == 2
    assert actions.count("position.settled") == 2
    last = {a: len(actions) - 1 - actions[::-1].index(a) for a in set(actions)}
    first = {a: actions.index(a) for a in set(actions)}
    assert (
        first["book.created"]
        < first["risk.decided"]
        < first["order.submitted"]
        <= last["order.submitted"]
        < first["outcome.recorded"]
        <= last["outcome.recorded"]
        < first["position.settled"]
    )

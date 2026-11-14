"""Settlement: pay out on the stored reported value, flag venue mismatches, re-mark the book."""

from datetime import timedelta
from decimal import Decimal

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from desk_fixtures import BASE_TS, add_contract, add_snapshot, make_book, make_release
from prediction_desk.audit.log import AuditLog
from prediction_desk.clock import SimClock
from prediction_desk.db.models import (
    AuditEvent,
    BookState,
    EventContract,
    MeasureOutcome,
    Order,
    Position,
    Release,
    RiskDecision,
    Settlement,
)
from prediction_desk.domain.enums import (
    BookMode,
    Comparator,
    MappingStatus,
    Measure,
    OrderStatus,
    PositionSide,
    Slot,
    Venue,
    Verdict,
)
from prediction_desk.execution.broker import PaperBroker
from prediction_desk.execution.settlement import SettleResult, settle_release
from prediction_desk.pricing.fees import FeeSchedule
from prediction_desk.risk.book import exposure
from prediction_desk.risk.engine import _approve

D = Decimal
NOW = BASE_TS + timedelta(days=2)


async def _outcome(
    session: AsyncSession, release: Release, measure: Measure, reported: str
) -> None:
    session.add(
        MeasureOutcome(
            release_id=release.id,
            measure=measure,
            raw_value=D(reported),
            reported_value=D(reported),
            source="fred_first_print",
            source_vintage_ts=BASE_TS,
            recorded_by="system:outcomes",
            reason=None,
            recorded_at=BASE_TS,
        )
    )
    await session.commit()


async def _position(
    session: AsyncSession,
    book: BookState,
    contract: EventContract,
    *,
    side: PositionSide = PositionSide.YES,
    qty: int = 100,
    cost: int = 5_300,
) -> Position:
    row = Position(
        book_id=book.id,
        contract_id=contract.id,
        side=side,
        qty=qty,
        cost_cents=cost,
        mark_price=D("0.5"),
        updated_at=BASE_TS,
    )
    session.add(row)
    await session.commit()
    return row


async def _fresh(session: AsyncSession, model: type, key: int) -> object:
    return (
        await session.execute(
            select(model).where(model.id == key).execution_options(populate_existing=True)  # type: ignore[attr-defined]
        )
    ).scalar_one()


async def _setup(
    session: AsyncSession, *, cash: int = 900_000, comparator: Comparator = Comparator.GT
) -> tuple[Release, EventContract, BookState]:
    release = await make_release(session)
    contract = await add_contract(
        session,
        release,
        venue=Venue.KALSHI,
        measure=Measure.CPI_MOM,
        comparator=comparator,
        strike_lo=D("0.3"),
    )
    book = await make_book(session, mode=BookMode.REPLAY, cash_cents=cash)
    return release, contract, book


async def _settle(
    sessions: async_sessionmaker[AsyncSession],
    book: BookState,
    release: Release,
    venue: dict[int, bool] | None = None,
) -> SettleResult:
    clock = SimClock(NOW)
    return await settle_release(
        sessions,
        AuditLog(clock),
        clock,
        book_id=book.id,
        release_id=release.id,
        venue_yes_won=venue,
    )


async def test_yes_winner_pays_100_per_contract_and_books_pnl(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    release, contract, book = await _setup(session)
    pos = await _position(session, book, contract, qty=100, cost=5_300)
    await _outcome(session, release, Measure.CPI_MOM, "0.4")

    result = await _settle(sessions, book, release)

    assert result == SettleResult(settled=(contract.id,), pending=(), mismatches=(), expired=())
    pos_row = await _fresh(session, Position, pos.id)
    assert isinstance(pos_row, Position)
    assert (pos_row.settled, pos_row.realized_pnl_cents, pos_row.mark_price) == (True, 4_700, D(1))
    book_row = await _fresh(session, BookState, book.id)
    assert isinstance(book_row, BookState)
    assert book_row.cash_cents == 910_000
    settlement = (await session.execute(select(Settlement))).scalar_one()
    assert (settlement.side, settlement.qty, settlement.yes_won) == (PositionSide.YES, 100, True)
    assert (settlement.payout_cents, settlement.realized_pnl_cents) == (10_000, 4_700)
    assert settlement.reported_value == D("0.4")
    assert settlement.venue_yes_won is None and settlement.mismatch is False
    assert settlement.settled_at == NOW
    audit = (
        await session.execute(select(AuditEvent).where(AuditEvent.action == "position.settled"))
    ).scalar_one()
    assert audit.payload["payout_cents"] == 10_000
    assert audit.payload["realized_pnl_cents"] == 4_700


async def test_no_position_wins_when_yes_loses(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    release, contract, book = await _setup(session)
    pos = await _position(session, book, contract, side=PositionSide.NO, qty=50, cost=2_000)
    await _outcome(session, release, Measure.CPI_MOM, "0.2")

    await _settle(sessions, book, release)

    pos_row = await _fresh(session, Position, pos.id)
    assert isinstance(pos_row, Position)
    assert (pos_row.realized_pnl_cents, pos_row.mark_price) == (3_000, D(0))
    book_row = await _fresh(session, BookState, book.id)
    assert isinstance(book_row, BookState)
    assert book_row.cash_cents == 905_000
    settlement = (await session.execute(select(Settlement))).scalar_one()
    assert (settlement.yes_won, settlement.payout_cents) == (False, 5_000)


async def test_losing_side_pays_nothing(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    release, contract, book = await _setup(session)
    pos = await _position(session, book, contract, side=PositionSide.YES, qty=10, cost=500)
    await _outcome(session, release, Measure.CPI_MOM, "0.2")

    await _settle(sessions, book, release)

    pos_row = await _fresh(session, Position, pos.id)
    assert isinstance(pos_row, Position)
    assert pos_row.realized_pnl_cents == -500
    book_row = await _fresh(session, BookState, book.id)
    assert isinstance(book_row, BookState)
    assert book_row.cash_cents == 900_000


async def test_missing_measure_outcome_leaves_position_pending(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    release, mom, book = await _setup(session)
    core = await add_contract(
        session,
        release,
        venue=Venue.KALSHI,
        measure=Measure.CORE_CPI_MOM,
        comparator=Comparator.GT,
        strike_lo=D("0.3"),
    )
    await _position(session, book, mom)
    core_pos = await _position(session, book, core)
    await _outcome(session, release, Measure.CPI_MOM, "0.4")

    result = await _settle(sessions, book, release)

    assert result == SettleResult(settled=(mom.id,), pending=(core.id,), mismatches=(), expired=())
    core_row = await _fresh(session, Position, core_pos.id)
    assert isinstance(core_row, Position)
    assert core_row.settled is False and core_row.realized_pnl_cents == 0
    assert await session.scalar(select(func.count()).select_from(Settlement)) == 1

    # The missing outcome arrives later; the second pass settles only the core position.
    await _outcome(session, release, Measure.CORE_CPI_MOM, "0.1")
    again = await _settle(sessions, book, release)
    assert again == SettleResult(settled=(core.id,), pending=(), mismatches=(), expired=())


async def test_unmapped_contract_stays_pending(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    release, contract, book = await _setup(session)
    pos = await _position(session, book, contract)
    session.add(
        RiskDecision(
            book_id=book.id,
            run_key="run-1",
            slot=Slot.T24H,
            release_id=release.id,
            contract_id=contract.id,
            side=PositionSide.YES,
            inputs_hash="ab" * 32,
            engine_version="risk:v1",
            fair_yes=D("0.6"),
            limit_price=D("0.50"),
            stake_final_cents=100,
            qty=100,
            verdict=Verdict.APPROVED,
            reasons=[],
            limits_version=1,
            limits_snapshot={},
            decided_at=BASE_TS,
        )
    )
    await _outcome(session, release, Measure.CPI_MOM, "0.4")
    # What discovery's _set_mapping does to a contract it can no longer map.
    contract.mapping_status = MappingStatus.UNMAPPED
    contract.unmapped_reason = "remapped"
    contract.release_id = None
    contract.measure = None
    contract.comparator = None
    contract.strike_lo = None
    contract.strike_hi = None
    contract.unit = None
    contract.resolution_precision = None
    contract.resolution_source = None
    await session.commit()

    result = await _settle(sessions, book, release)

    # Its approved decision never got an order, so settlement also expires it (R14).
    assert result == SettleResult(
        settled=(), pending=(contract.id,), mismatches=(), expired=(contract.id,)
    )
    pos_row = await _fresh(session, Position, pos.id)
    assert isinstance(pos_row, Position)
    assert pos_row.settled is False
    assert await session.scalar(select(func.count()).select_from(Settlement)) == 0
    book_row = await _fresh(session, BookState, book.id)
    assert isinstance(book_row, BookState)
    assert book_row.cash_cents == 900_000
    event = (
        await session.execute(
            select(AuditEvent).where(AuditEvent.action == "settlement.pending_unmapped")
        )
    ).scalar_one()
    assert (event.entity, event.entity_id) == ("position", str(pos.id))


async def test_mismatch_flagged_but_payout_follows_our_outcome(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    release, contract, book = await _setup(session)
    await _position(session, book, contract, qty=100, cost=5_300)
    await _outcome(session, release, Measure.CPI_MOM, "0.4")

    result = await _settle(sessions, book, release, {contract.id: False})

    assert result == SettleResult(
        settled=(contract.id,), pending=(), mismatches=(contract.id,), expired=()
    )
    settlement = (await session.execute(select(Settlement))).scalar_one()
    assert (settlement.yes_won, settlement.venue_yes_won, settlement.mismatch) == (
        True,
        False,
        True,
    )
    assert settlement.payout_cents == 10_000
    actions = (
        (
            await session.execute(
                select(AuditEvent.action).where(AuditEvent.action.like("%mismatch"))
            )
        )
        .scalars()
        .all()
    )
    assert actions == ["settlement.mismatch"]


async def test_agreeing_venue_is_not_a_mismatch(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    release, contract, book = await _setup(session)
    await _position(session, book, contract)
    await _outcome(session, release, Measure.CPI_MOM, "0.4")

    result = await _settle(sessions, book, release, {contract.id: True})

    assert result.mismatches == ()
    settlement = (await session.execute(select(Settlement))).scalar_one()
    assert (settlement.venue_yes_won, settlement.mismatch) == (True, False)


async def test_settle_twice_is_idempotent(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    release, contract, book = await _setup(session)
    await _position(session, book, contract)
    await _outcome(session, release, Measure.CPI_MOM, "0.4")

    await _settle(sessions, book, release)
    again = await _settle(sessions, book, release)

    assert again == SettleResult(settled=(), pending=(), mismatches=(), expired=())
    assert await session.scalar(select(func.count()).select_from(Settlement)) == 1
    book_row = await _fresh(session, BookState, book.id)
    assert isinstance(book_row, BookState)
    assert book_row.cash_cents == 910_000


async def test_big_realized_loss_trips_drawdown_via_mark(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    # Equity 1_000_000 = 700_000 cash + 300_000 cost basis. Losing it all is a 30% drawdown from
    # the 1_000_000 peak; the day start is 700_000, so the daily-loss rule cannot fire.
    release, contract, book = await _setup(session, cash=700_000)
    book.equity_cents = 1_000_000
    book.peak_equity_cents = 1_000_000
    book.day_start_equity_cents = 700_000
    book.day_start_date = NOW.date()  # the mark does not roll the day
    await session.commit()
    await _position(session, book, contract, qty=500, cost=300_000)
    await _outcome(session, release, Measure.CPI_MOM, "0.2")

    await _settle(sessions, book, release)

    book_row = await _fresh(session, BookState, book.id)
    assert isinstance(book_row, BookState)
    assert book_row.equity_cents == 700_000
    assert book_row.kill_switch_on is True
    tripped = (
        await session.execute(select(AuditEvent).where(AuditEvent.action == "kill_switch.tripped"))
    ).scalar_one()
    assert str(tripped.payload["reason"]).startswith("drawdown")


async def test_equity_after_settlement_equals_cash_when_flat(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    release, contract, book = await _setup(session)
    await _position(session, book, contract, qty=100, cost=5_300)
    await _outcome(session, release, Measure.CPI_MOM, "0.4")

    await _settle(sessions, book, release)

    book_row = await _fresh(session, BookState, book.id)
    assert isinstance(book_row, BookState)
    assert book_row.equity_cents == book_row.cash_cents == 910_000


async def test_unknown_release_is_an_error(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    _, _, book = await _setup(session)
    clock = SimClock(NOW)
    with pytest.raises(ValueError, match="release"):
        await settle_release(sessions, AuditLog(clock), clock, book_id=book.id, release_id=9999)


async def test_position_opened_through_the_broker_settles(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    release, contract, book = await _setup(session, cash=1_000_000)
    await add_snapshot(session, contract, BASE_TS - timedelta(seconds=10), [], [("0.50", "100")])
    clock = SimClock(BASE_TS)
    decision = RiskDecision(
        book_id=book.id,
        run_key="run-1",
        slot=Slot.T24H,
        release_id=release.id,
        contract_id=contract.id,
        side=PositionSide.YES,
        inputs_hash="ab" * 32,
        engine_version="risk:v1",
        fair_yes=D("0.6"),
        limit_price=D("0.50"),
        stake_final_cents=100,
        qty=100,
        verdict=Verdict.APPROVED,
        reasons=[],
        limits_version=1,
        limits_snapshot={},
        decided_at=BASE_TS,
    )
    session.add(decision)
    await session.commit()
    fees = (FeeSchedule(Venue.KALSHI, BASE_TS.date().replace(year=2020), "zero", D("0")),)
    broker = PaperBroker(clock, AuditLog(clock), fees)
    submitted = await broker.submit(sessions, _approve(decision))
    assert submitted.cost_cents == 5_000
    await _outcome(session, release, Measure.CPI_MOM, "0.4")

    result = await _settle(sessions, book, release)

    assert result.settled == (contract.id,)
    book_row = await _fresh(session, BookState, book.id)
    assert isinstance(book_row, BookState)
    assert book_row.cash_cents == 1_000_000 - 5_000 + 10_000
    assert book_row.equity_cents == book_row.cash_cents


def _decision(
    book: BookState, release: Release, contract: EventContract, *, run_key: str = "run-1"
) -> RiskDecision:
    return RiskDecision(
        book_id=book.id,
        run_key=run_key,
        slot=Slot.T24H,
        release_id=release.id,
        contract_id=contract.id,
        side=PositionSide.YES,
        inputs_hash="ab" * 32,
        engine_version="risk:v2",
        fair_yes=D("0.6"),
        limit_price=D("0.50"),
        stake_final_cents=1_250,
        qty=25,
        verdict=Verdict.APPROVED,
        reasons=[],
        limits_version=1,
        limits_snapshot={},
        decided_at=BASE_TS,
    )


async def test_unsubmitted_approval_expires_at_settlement(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    release, contract, book = await _setup(session)
    later = await make_release(session, period="2026-11")
    other = await add_contract(
        session,
        later,
        venue=Venue.KALSHI,
        measure=Measure.CPI_MOM,
        comparator=Comparator.GT,
        strike_lo=D("0.3"),
        venue_event_id="ev-2",
    )
    orphan = _decision(book, release, contract)
    kept = _decision(book, later, other)
    session.add_all([orphan, kept])
    await session.commit()
    assert (await exposure(session, book.id)).portfolio == 2_500

    result = await _settle(sessions, book, release)
    again = await _settle(sessions, book, release)

    assert result == SettleResult(settled=(), pending=(), mismatches=(), expired=(contract.id,))
    assert again == SettleResult(settled=(), pending=(), mismatches=(), expired=())
    order = (await session.execute(select(Order))).scalar_one()
    assert order.client_order_id == _approve(orphan).client_order_id
    assert (order.risk_decision_id, order.status, order.status_reason) == (
        orphan.id,
        OrderStatus.REFUSED,
        "expired",
    )
    assert (order.qty, order.filled_qty, order.created_at) == (25, 0, NOW)
    exp = await exposure(session, book.id)
    assert (exp.portfolio, exp.event) == (1_250, {later.id: 1_250})  # only the later release
    event = (
        await session.execute(select(AuditEvent).where(AuditEvent.action == "order.expired"))
    ).scalar_one()
    assert (event.entity, event.entity_id, event.actor) == (
        "order",
        str(order.id),
        "system:settlement",
    )
    assert event.payload["risk_decision_id"] == orphan.id
    assert event.payload["status_reason"] == "expired"
    # A late submit of the expired approval finds the refusal (D8), so nothing ever fills.
    clock = SimClock(BASE_TS)
    late = await PaperBroker(clock, AuditLog(clock)).submit(sessions, _approve(orphan))
    assert (late.order_id, late.status) == (order.id, OrderStatus.REFUSED)

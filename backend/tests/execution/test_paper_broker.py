"""PaperBroker: IOC fills against the stored book, idempotency and re-checks under the lock."""

import asyncio
from datetime import date, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import func, insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from desk_fixtures import BASE_TS, add_contract, add_snapshot, make_book, make_release
from prediction_desk.audit.log import AuditLog
from prediction_desk.clock import SimClock
from prediction_desk.db.models import (
    AuditEvent,
    BookState,
    EventContract,
    Fill,
    MarketSnapshot,
    Order,
    Position,
    RiskDecision,
)
from prediction_desk.domain.enums import (
    BookMode,
    Comparator,
    Measure,
    OrderStatus,
    PositionSide,
    Slot,
    Venue,
    Verdict,
)
from prediction_desk.execution.broker import LiveBroker, OrderResult, PaperBroker
from prediction_desk.pricing.fees import FeeSchedule
from prediction_desk.risk.engine import ApprovedOrder, _approve
from prediction_desk.risk.kill_switch import trip_kill_switch

D = Decimal
NOW = BASE_TS
KALSHI_FEES = (FeeSchedule(Venue.KALSHI, date(2020, 1, 1), "kalshi_quadratic", D("0.07")),)
WORKED_ASKS = [("0.50", "100"), ("0.53", "200")]


class Env:
    def __init__(self, book: BookState, contract: EventContract, clock: SimClock) -> None:
        self.book = book
        self.contract = contract
        self.clock = clock
        self.audit = AuditLog(clock)
        self.broker = PaperBroker(clock, self.audit, KALSHI_FEES)


async def _env(
    session: AsyncSession,
    *,
    asks: list[tuple[str, str]] | None = None,
    bids: list[tuple[str, str]] | None = None,
    snap_age_s: int = 10,
    cash: int = 1_000_000,
) -> Env:
    release = await make_release(session)
    contract = await add_contract(
        session,
        release,
        venue=Venue.KALSHI,
        measure=Measure.CPI_MOM,
        comparator=Comparator.GT,
        strike_lo=D("0.3"),
    )
    await add_snapshot(
        session,
        contract,
        NOW - timedelta(seconds=snap_age_s),
        [] if bids is None else bids,
        WORKED_ASKS if asks is None else asks,
    )
    book = await make_book(session, mode=BookMode.REPLAY, cash_cents=cash)
    return Env(book, contract, SimClock(NOW))


async def _approved(
    session: AsyncSession,
    env: Env,
    *,
    qty: int,
    limit: str,
    side: PositionSide = PositionSide.YES,
    run_key: str = "run-1",
) -> ApprovedOrder:
    decision = RiskDecision(
        book_id=env.book.id,
        run_key=run_key,
        slot=Slot.T24H,
        release_id=env.contract.release_id,
        contract_id=env.contract.id,
        side=side,
        inputs_hash="ab" * 32,
        engine_version="risk:v1",
        fair_yes=D("0.6"),
        limit_price=D(limit),
        stake_final_cents=100,
        qty=qty,
        verdict=Verdict.APPROVED,
        reasons=[],
        limits_version=1,
        limits_snapshot={},
        decided_at=NOW,
    )
    session.add(decision)
    await session.commit()
    return _approve(decision)


async def _count(session: AsyncSession, model: type) -> int:
    return (await session.execute(select(func.count()).select_from(model))).scalar_one()


async def _book(session: AsyncSession, book_id: int) -> BookState:
    return (
        await session.execute(
            select(BookState)
            .where(BookState.id == book_id)
            .execution_options(populate_existing=True)
        )
    ).scalar_one()


async def test_fills_full_qty_at_or_below_limit_and_moves_cash(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    env = await _env(session)
    approved = await _approved(session, env, qty=250, limit="0.53")

    result = await env.broker.submit(sessions, approved)

    assert result == OrderResult(
        order_id=result.order_id,
        status=OrderStatus.FILLED,
        filled_qty=250,
        cost_cents=12_950,
        fees_cents=437,
    )
    fill = (await session.execute(select(Fill))).scalar_one()
    assert (fill.qty, fill.cost_cents, fill.fees_cents, fill.slippage_cents) == (
        250,
        12_950,
        437,
        450,
    )
    assert fill.avg_price == D("0.518")
    assert fill.ts == NOW
    order = (await session.execute(select(Order))).scalar_one()
    assert (order.status, order.filled_qty, order.status_reason) == (OrderStatus.FILLED, 250, None)
    assert order.client_order_id == approved.client_order_id
    assert order.risk_decision_id == approved.risk_decision_id
    position = (await session.execute(select(Position))).scalar_one()
    assert (position.qty, position.cost_cents, position.side) == (250, 13_387, PositionSide.YES)
    assert position.settled is False
    assert position.realized_pnl_cents == 0
    assert (await _book(session, env.book.id)).cash_cents == 986_613
    audit = (
        await session.execute(select(AuditEvent).where(AuditEvent.action == "order.submitted"))
    ).scalar_one()
    assert audit.payload["status"] == "filled"
    assert audit.payload["cost_cents"] == 12_950
    assert audit.payload["fees_cents"] == 437
    assert audit.payload["slippage_cents"] == 450
    assert audit.payload["order_id"] == order.id


async def test_partial_when_depth_at_limit_insufficient(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    env = await _env(session, asks=[("0.50", "100"), ("0.53", "50.9"), ("0.60", "500")])
    approved = await _approved(session, env, qty=250, limit="0.53")

    result = await env.broker.submit(sessions, approved)

    # 100 @ 0.50 + 50 whole contracts @ 0.53 (0.9 is not a contract); 0.60 is above the limit.
    assert result.status is OrderStatus.PARTIAL
    assert result.filled_qty == 150
    assert result.cost_cents == 5_000 + 2_650
    order = (await session.execute(select(Order))).scalar_one()
    assert (order.status, order.filled_qty, order.qty) == (OrderStatus.PARTIAL, 150, 250)
    assert order.status_reason == "ioc_remainder_cancelled"
    position = (await session.execute(select(Position))).scalar_one()
    assert position.qty == 150
    assert position.cost_cents == result.cost_cents + result.fees_cents
    assert (await _book(session, env.book.id)).cash_cents == 1_000_000 - position.cost_cents


async def test_cancelled_when_nothing_at_or_below_limit(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    env = await _env(session, asks=[("0.60", "100")])
    approved = await _approved(session, env, qty=10, limit="0.55")

    result = await env.broker.submit(sessions, approved)

    assert result.status is OrderStatus.CANCELLED
    assert (result.filled_qty, result.cost_cents, result.fees_cents) == (0, 0, 0)
    order = (await session.execute(select(Order))).scalar_one()
    assert order.status_reason == "nothing_at_limit"
    assert await _count(session, Fill) == 0
    assert await _count(session, Position) == 0
    assert (await _book(session, env.book.id)).cash_cents == 1_000_000
    assert await _count(session, AuditEvent) == 1


async def test_cancelled_when_levels_have_no_whole_contract(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    env = await _env(session, asks=[("0.50", "0.5")])
    approved = await _approved(session, env, qty=10, limit="0.55")
    result = await env.broker.submit(sessions, approved)
    assert result.status is OrderStatus.CANCELLED


async def test_refused_when_kill_switch_flipped_after_approval(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    env = await _env(session)
    approved = await _approved(session, env, qty=10, limit="0.53")
    async with sessions() as s, s.begin():
        await trip_kill_switch(s, env.audit, env.clock, env.book.id, reason="test", actor="admin:t")

    result = await env.broker.submit(sessions, approved)

    assert result.status is OrderStatus.REFUSED
    assert result.filled_qty == 0
    order = (await session.execute(select(Order))).scalar_one()
    assert order.status_reason == "kill_switch_on"
    assert await _count(session, Fill) == 0
    assert await _count(session, Position) == 0
    assert (await _book(session, env.book.id)).cash_cents == 1_000_000


async def test_refused_when_book_stale(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    env = await _env(session, snap_age_s=61)
    approved = await _approved(session, env, qty=10, limit="0.53")
    result = await env.broker.submit(sessions, approved)
    assert result.status is OrderStatus.REFUSED
    order = (await session.execute(select(Order))).scalar_one()
    assert order.status_reason == "no_fresh_book"
    assert await _count(session, Fill) == 0


async def test_refused_when_no_snapshot_at_or_before_now(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    env = await _env(session, snap_age_s=-3600)  # the only snapshot is in the future
    approved = await _approved(session, env, qty=10, limit="0.53")
    result = await env.broker.submit(sessions, approved)
    assert result.status is OrderStatus.REFUSED
    assert (await session.execute(select(Order))).scalar_one().status_reason == "no_fresh_book"


async def test_refused_when_stored_book_malformed(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    env = await _env(session)
    snap = (await session.execute(select(MarketSnapshot))).scalar_one()
    snap.book = {"bids": [], "asks": [["abc", "1"]]}
    await session.commit()
    approved = await _approved(session, env, qty=10, limit="0.53")
    result = await env.broker.submit(sessions, approved)
    assert result.status is OrderStatus.REFUSED
    assert (await session.execute(select(Order))).scalar_one().status_reason == "no_fresh_book"


async def test_refused_when_cash_insufficient(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    env = await _env(session, cash=13_000)
    approved = await _approved(session, env, qty=250, limit="0.53")

    result = await env.broker.submit(sessions, approved)

    assert result.status is OrderStatus.REFUSED
    assert (result.filled_qty, result.cost_cents, result.fees_cents) == (0, 0, 0)
    order = (await session.execute(select(Order))).scalar_one()
    assert (order.status_reason, order.filled_qty) == ("insufficient_cash", 0)
    assert await _count(session, Fill) == 0
    assert await _count(session, Position) == 0
    assert (await _book(session, env.book.id)).cash_cents == 13_000


async def test_submit_twice_one_order(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    env = await _env(session)
    approved = await _approved(session, env, qty=250, limit="0.53")

    first = await env.broker.submit(sessions, approved)
    second = await env.broker.submit(sessions, approved)

    assert second == first
    assert await _count(session, Order) == 1
    assert await _count(session, Fill) == 1
    assert (await session.execute(select(Position))).scalar_one().qty == 250
    assert (await _book(session, env.book.id)).cash_cents == 986_613
    assert await _count(session, AuditEvent) == 1


async def test_concurrent_submits_one_order(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    env = await _env(session)
    approved = await _approved(session, env, qty=250, limit="0.53")

    results = await asyncio.gather(
        env.broker.submit(sessions, approved), env.broker.submit(sessions, approved)
    )

    assert results[0] == results[1]
    assert await _count(session, Order) == 1
    assert (await _book(session, env.book.id)).cash_cents == 986_613


async def test_resubmit_of_refused_order_returns_refusal(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    env = await _env(session, snap_age_s=120)
    approved = await _approved(session, env, qty=10, limit="0.53")
    first = await env.broker.submit(sessions, approved)
    second = await env.broker.submit(sessions, approved)
    assert first.status is OrderStatus.REFUSED
    assert second == first


async def test_no_side_fills_against_bids(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    # YES bids 0.50 x 100 and 0.40 x 100 -> NO asks 0.50 and 0.60.
    env = await _env(session, asks=[("0.90", "10")], bids=[("0.50", "100"), ("0.40", "100")])
    approved = await _approved(session, env, qty=150, limit="0.60", side=PositionSide.NO)

    result = await env.broker.submit(sessions, approved)

    assert result.status is OrderStatus.FILLED
    assert result.cost_cents == 5_000 + 3_000
    fill = (await session.execute(select(Fill))).scalar_one()
    assert fill.slippage_cents == 0 + 500  # 50 contracts one tick-ten worse than the best
    position = (await session.execute(select(Position))).scalar_one()
    assert position.side is PositionSide.NO


async def test_top_up_accumulates_position_cost(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    env = await _env(session, asks=[("0.50", "1000")])
    first = await env.broker.submit(
        sessions, await _approved(session, env, qty=100, limit="0.50", run_key="a")
    )
    second = await env.broker.submit(
        sessions, await _approved(session, env, qty=50, limit="0.50", run_key="b")
    )

    assert first.cost_cents == 5_000 and second.cost_cents == 2_500
    position = (await session.execute(select(Position))).scalar_one()
    assert position.qty == 150
    assert position.cost_cents == (
        first.cost_cents + first.fees_cents + second.cost_cents + second.fees_cents
    )
    assert (await _book(session, env.book.id)).cash_cents == 1_000_000 - position.cost_cents
    assert await _count(session, Order) == 2


async def test_top_up_refuses_settled_or_opposite_position(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    env = await _env(session, asks=[("0.50", "1000")], bids=[("0.40", "1000")])
    await env.broker.submit(
        sessions, await _approved(session, env, qty=10, limit="0.50", run_key="a")
    )
    opposite = await _approved(
        session, env, qty=10, limit="0.60", side=PositionSide.NO, run_key="b"
    )
    with pytest.raises(RuntimeError, match="opposite"):
        await env.broker.submit(sessions, opposite)

    position = (await session.execute(select(Position))).scalar_one()
    position.settled = True
    await session.commit()
    again = await _approved(session, env, qty=10, limit="0.50", run_key="c")
    with pytest.raises(RuntimeError, match="settled"):
        await env.broker.submit(sessions, again)
    assert await _count(session, Order) == 1  # both failed submits rolled back


def test_live_broker_refuses_to_construct() -> None:
    with pytest.raises(RuntimeError, match=r"live trading is disabled \(TRADING_MODE=paper\)"):
        LiveBroker()


async def test_order_without_decision_impossible(session: AsyncSession) -> None:
    env = await _env(session)
    with pytest.raises(IntegrityError):
        await session.execute(
            insert(Order).values(
                client_order_id="e" * 64,
                risk_decision_id=None,
                book_id=env.book.id,
                contract_id=env.contract.id,
                side=PositionSide.YES,
                limit_price=D("0.5"),
                qty=1,
                filled_qty=0,
                status=OrderStatus.CANCELLED,
                created_at=NOW,
            )
        )

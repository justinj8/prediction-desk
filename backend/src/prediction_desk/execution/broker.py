"""Paper broker: turns a risk-approved order into an order, a fill, a position and a cash move.

Orders are IOC marketable limits (D3): the fill walks the side's levels priced at or below the
limit, whole contracts only, and the remainder is cancelled. Everything happens in one
``unit_of_work`` under the book lock (D2), so a kill-switch trip that committed before the lock
is seen here and a double submit finds the first order (D8).

Order of checks under the lock (ruling R14):

1. the ``RiskDecision`` behind the approval must exist, be APPROVED and match it (book,
   contract, side, limit, qty, client_order_id); a mismatch is an invariant failure that raises
   ``RuntimeError`` and writes nothing;
2. dedupe on ``client_order_id``: a resubmit returns the first order unchanged;
3. refusals, first match wins: ``past_cutoff`` (now at or after the decision's release
   ``scheduled_at - cutoff_minutes``), ``kill_switch_on``, ``no_fresh_book``, then after the
   walk ``position_settled`` and ``insufficient_cash``.

Every refusal is the same single write: one REFUSED order with no fill and one
``order.submitted`` audit event.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import ROUND_CEILING, ROUND_HALF_UP, Decimal
from math import floor
from typing import Final, Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from prediction_desk.audit.log import AuditLog
from prediction_desk.clock import Clock
from prediction_desk.db.models import (
    BookState,
    EventContract,
    Fill,
    Order,
    Position,
    Release,
    RiskDecision,
)
from prediction_desk.db.uow import unit_of_work
from prediction_desk.domain.enums import OrderStatus, Verdict
from prediction_desk.domain.types import require_utc
from prediction_desk.markets.snapshots import latest_snapshot_asof
from prediction_desk.markets.types import PARSE_ERRORS, OrderBook
from prediction_desk.pricing.book import side_levels
from prediction_desk.pricing.fees import FeeSchedule, fill_fee_cents, schedule_for
from prediction_desk.risk.book import lock_book
from prediction_desk.risk.engine import ApprovedOrder, client_order_id, release_cutoff
from prediction_desk.risk.limits import load_active_limits

BROKER_ACTOR: Final = "system:paper_broker"
_CENT: Final = Decimal(100)
_ONE: Final = Decimal(1)
_AVG_QUANT: Final = Decimal("0.00000001")


@dataclass(frozen=True)
class OrderResult:
    order_id: int
    status: OrderStatus
    filled_qty: int
    cost_cents: int
    fees_cents: int


class Broker(Protocol):
    async def submit(
        self, sessions: async_sessionmaker[AsyncSession], approved: ApprovedOrder
    ) -> OrderResult: ...


class LiveBroker:
    """Placeholder that cannot be built: live trading is disabled in this phase."""

    def __init__(self) -> None:
        raise RuntimeError("live trading is disabled (TRADING_MODE=paper)")


class PaperBroker:
    def __init__(
        self,
        clock: Clock,
        audit: AuditLog,
        fee_schedules: Sequence[FeeSchedule] | None = None,
    ) -> None:
        self._clock = clock
        self._audit = audit
        self._fee_schedules = fee_schedules

    async def submit(
        self, sessions: async_sessionmaker[AsyncSession], approved: ApprovedOrder
    ) -> OrderResult:
        async with unit_of_work(sessions) as session:
            book = await lock_book(session, approved.book_id)
            decision = await self._decision(session, approved)
            existing = (
                await session.execute(
                    select(Order).where(Order.client_order_id == approved.client_order_id)
                )
            ).scalar_one_or_none()
            if existing is not None:
                return await self._existing_result(session, existing)
            return await self._place(session, book, approved, decision)

    @staticmethod
    async def _decision(session: AsyncSession, approved: ApprovedOrder) -> RiskDecision:
        """The stored decision behind ``approved``; raises unless it is APPROVED and matches."""
        decision = await session.get(RiskDecision, approved.risk_decision_id)
        if (
            decision is None
            or decision.verdict is not Verdict.APPROVED
            or (
                decision.book_id,
                decision.contract_id,
                decision.side,
                decision.limit_price,
                decision.qty,
                client_order_id(decision),
            )
            != (
                approved.book_id,
                approved.contract_id,
                approved.side,
                approved.limit_price,
                approved.qty,
                approved.client_order_id,
            )
        ):
            raise RuntimeError(
                f"approved order does not match risk decision {approved.risk_decision_id}"
            )
        return decision

    @staticmethod
    async def _existing_result(session: AsyncSession, order: Order) -> OrderResult:
        fill = (
            await session.execute(select(Fill).where(Fill.order_id == order.id))
        ).scalar_one_or_none()
        return OrderResult(
            order_id=order.id,
            status=order.status,
            filled_qty=order.filled_qty,
            cost_cents=0 if fill is None else fill.cost_cents,
            fees_cents=0 if fill is None else fill.fees_cents,
        )

    async def _place(
        self,
        session: AsyncSession,
        book: BookState,
        approved: ApprovedOrder,
        decision: RiskDecision,
    ) -> OrderResult:
        now = require_utc(self._clock.now())
        limits = await load_active_limits(session)
        release = await session.get_one(Release, decision.release_id)
        snap = await latest_snapshot_asof(session, approved.contract_id, now)
        order_book: OrderBook | None = None
        if snap is not None and now - require_utc(snap.ts) <= timedelta(
            seconds=limits.max_quote_age_s
        ):
            try:
                order_book = OrderBook.from_json(snap.book, snap.ts, snap.venue_ts)
            except PARSE_ERRORS:
                order_book = None

        refusal: str | None = None
        if now >= release_cutoff(release, limits):
            refusal = "past_cutoff"
        elif book.kill_switch_on:
            refusal = "kill_switch_on"
        elif snap is None or order_book is None:
            refusal = "no_fresh_book"

        filled = 0
        exact_cost = Decimal(0)
        cost = fees = slippage = 0
        avg = Decimal(0)
        takes: list[tuple[Decimal, Decimal]] = []
        if refusal is None and order_book is not None:
            best: Decimal | None = None
            remaining = approved.qty
            for level in side_levels(order_book, approved.side.value):
                if level.price > approved.limit_price or remaining == 0:
                    break
                take = min(floor(level.size), remaining)
                if take < 1:
                    continue
                best = level.price if best is None else best
                takes.append((level.price, Decimal(take)))
                exact_cost += level.price * take
                filled += take
                remaining -= take
            if filled:
                assert best is not None  # a level was taken
                contract = await session.get(EventContract, approved.contract_id)
                assert contract is not None  # FK on the decision
                avg = exact_cost / filled
                cost = int((exact_cost * _CENT).to_integral_value(ROUND_CEILING))
                fees = fill_fee_cents(
                    schedule_for(contract.venue, now, self._fee_schedules), avg, filled, takes
                )
                slippage = int(
                    ((exact_cost - best * filled) * _CENT).to_integral_value(ROUND_HALF_UP)
                )
                held = await self._position(session, approved)
                if held is not None and held.settled:
                    refusal = "position_settled"
                elif book.cash_cents - cost - fees < 0:
                    refusal = "insufficient_cash"
                if refusal is not None:
                    filled = cost = fees = slippage = 0
                    avg = Decimal(0)

        if refusal is not None:
            status, reason = OrderStatus.REFUSED, refusal
        elif filled == approved.qty:
            status, reason = OrderStatus.FILLED, None
        elif filled:
            status, reason = OrderStatus.PARTIAL, "ioc_remainder_cancelled"
        else:
            status, reason = OrderStatus.CANCELLED, "nothing_at_limit"

        order = Order(
            client_order_id=approved.client_order_id,
            risk_decision_id=approved.risk_decision_id,
            book_id=approved.book_id,
            contract_id=approved.contract_id,
            side=approved.side,
            limit_price=approved.limit_price,
            qty=approved.qty,
            filled_qty=filled,
            status=status,
            status_reason=reason,
            created_at=now,
        )
        session.add(order)
        await session.flush()

        if filled:
            assert snap is not None  # a fill needs a book
            session.add(
                Fill(
                    order_id=order.id,
                    qty=filled,
                    avg_price=avg.quantize(_AVG_QUANT, rounding=ROUND_HALF_UP),
                    cost_cents=cost,
                    fees_cents=fees,
                    slippage_cents=slippage,
                    snapshot_id=snap.id,
                    ts=now,
                )
            )
            await self._apply_to_position(session, approved, filled, cost + fees, now)
            book.cash_cents -= cost + fees
            book.updated_at = now
        await session.flush()

        await self._audit.append(
            session,
            actor=BROKER_ACTOR,
            entity="order",
            entity_id=str(order.id),
            action="order.submitted",
            payload={
                "order_id": order.id,
                "client_order_id": approved.client_order_id,
                "risk_decision_id": approved.risk_decision_id,
                "book_id": approved.book_id,
                "contract_id": approved.contract_id,
                "side": approved.side,
                "limit_price": approved.limit_price,
                "qty": approved.qty,
                "filled_qty": filled,
                "status": status,
                "status_reason": reason,
                "avg_price": avg.quantize(_AVG_QUANT, rounding=ROUND_HALF_UP) if filled else None,
                "cost_cents": cost,
                "fees_cents": fees,
                "slippage_cents": slippage,
                "snapshot_id": None if snap is None else snap.id,
                "cash_cents": book.cash_cents,
            },
        )
        return OrderResult(
            order_id=order.id, status=status, filled_qty=filled, cost_cents=cost, fees_cents=fees
        )

    @staticmethod
    async def _position(session: AsyncSession, approved: ApprovedOrder) -> Position | None:
        return (
            await session.execute(
                select(Position)
                .where(
                    Position.book_id == approved.book_id,
                    Position.contract_id == approved.contract_id,
                )
                .with_for_update()
            )
        ).scalar_one_or_none()

    async def _apply_to_position(
        self,
        session: AsyncSession,
        approved: ApprovedOrder,
        filled: int,
        basis_cents: int,
        now: datetime,
    ) -> None:
        position = await self._position(session, approved)
        if position is None:
            session.add(
                Position(
                    book_id=approved.book_id,
                    contract_id=approved.contract_id,
                    side=approved.side,
                    qty=filled,
                    cost_cents=basis_cents,
                    updated_at=now,
                )
            )
            return
        if position.side is not approved.side:
            raise RuntimeError(f"position {position.id} holds the opposite side")
        position.qty += filled
        position.cost_cents += basis_cents
        position.updated_at = now

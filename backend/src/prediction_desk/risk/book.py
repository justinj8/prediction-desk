"""Book state: creation, the row lock, exposure (D1) and mark-to-market with auto-trips."""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import ROUND_HALF_EVEN, Decimal
from zoneinfo import ZoneInfo

from sqlalchemy import ColumnElement, exists, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from prediction_desk.audit.log import AuditLog
from prediction_desk.clock import Clock
from prediction_desk.db.models import BookState, EventContract, Order, Position, RiskDecision
from prediction_desk.domain.enums import BookMode, PositionSide, Verdict
from prediction_desk.domain.types import require_utc
from prediction_desk.markets.snapshots import latest_snapshot_asof
from prediction_desk.risk.kill_switch import trip_kill_switch
from prediction_desk.risk.limits import load_active_limits

ET = ZoneInfo("America/New_York")
SYSTEM_ACTOR = "system:risk"
_MARK_QUANT = Decimal("0.00001")
_ONE = Decimal(1)
_TWO = Decimal(2)
_CENTS = Decimal(100)


def et_date(ts: datetime) -> date:
    """The America/New_York calendar day of ``ts`` (D7)."""
    return require_utc(ts).astimezone(ET).date()


@dataclass(frozen=True)
class Exposure:
    contract: Mapping[int, int]
    event: Mapping[int, int]
    portfolio: int


async def create_book(
    session: AsyncSession,
    audit: AuditLog,
    clock: Clock,
    *,
    mode: BookMode,
    replay_batch_id: int | None,
    cash_cents: int = 1_000_000,
) -> BookState:
    now = require_utc(clock.now())
    book = BookState(
        mode=mode,
        replay_batch_id=replay_batch_id,
        cash_cents=cash_cents,
        equity_cents=cash_cents,
        peak_equity_cents=cash_cents,
        day_start_equity_cents=cash_cents,
        day_start_date=et_date(now),
        kill_switch_on=False,
        created_at=now,
        updated_at=now,
    )
    session.add(book)
    await session.flush()
    await audit.append(
        session,
        actor=SYSTEM_ACTOR,
        entity="book",
        entity_id=str(book.id),
        action="book.created",
        payload={
            "mode": mode,
            "replay_batch_id": replay_batch_id,
            "cash_cents": cash_cents,
        },
    )
    return book


async def lock_book(session: AsyncSession, book_id: int) -> BookState:
    """``SELECT ... FOR UPDATE`` on the book row, refreshing any stale cached copy (D2)."""
    book = (
        await session.execute(
            select(BookState)
            .where(BookState.id == book_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
    ).scalar_one_or_none()
    if book is None:
        raise LookupError(f"book {book_id} not found")
    return book


def held_release_id(book_id: int) -> ColumnElement[int | None]:
    """The release a held position belongs to, for a query over ``Position`` joined to its
    ``EventContract``: the contract's own release, else (discovery unmapped the contract, so
    ``release_id`` is NULL) the release of this book's first approved decision on it. NULL when
    neither is known. Shared by ``exposure`` and settlement so both attribute alike."""
    decided = (
        select(RiskDecision.release_id)
        .where(
            RiskDecision.book_id == book_id,
            RiskDecision.contract_id == Position.contract_id,
            RiskDecision.verdict == Verdict.APPROVED,
        )
        .order_by(RiskDecision.id)
        .limit(1)
        .scalar_subquery()
    )
    return func.coalesce(EventContract.release_id, decided)


async def exposure(session: AsyncSession, book_id: int) -> Exposure:
    """Capital at risk in cents: open position cost plus approved-but-unordered stakes (D1)."""
    by_contract: dict[int, int] = defaultdict(int)
    by_event: dict[int, int] = defaultdict(int)
    held = await session.execute(
        select(Position.contract_id, held_release_id(book_id), Position.cost_cents)
        .join(EventContract, EventContract.id == Position.contract_id)
        .where(Position.book_id == book_id, Position.settled.is_(False))
    )
    pending = await session.execute(
        select(
            RiskDecision.contract_id, RiskDecision.release_id, RiskDecision.stake_final_cents
        ).where(
            RiskDecision.book_id == book_id,
            RiskDecision.verdict == Verdict.APPROVED,
            RiskDecision.stake_final_cents.is_not(None),
            ~exists().where(Order.risk_decision_id == RiskDecision.id),
        )
    )
    portfolio = 0
    for contract_id, release_id, maybe_cents in [*held.all(), *pending.all()]:
        cents = maybe_cents or 0  # pending rows are filtered non-null in SQL
        if release_id is None:  # unmapped and never decided by this book: fail closed
            raise RuntimeError(f"no release known for held contract {contract_id}")
        by_contract[contract_id] += cents
        by_event[release_id] += cents
        portfolio += cents
    return Exposure(contract=dict(by_contract), event=dict(by_event), portfolio=portfolio)


async def _mark_price(
    session: AsyncSession, position: Position, asof: datetime
) -> tuple[Decimal, bool]:
    """YES-price mark for a position and whether it came from a fresh two-sided book."""
    snap = await latest_snapshot_asof(session, position.contract_id, asof)
    if snap is not None and snap.best_bid is not None and snap.best_ask is not None:
        return ((snap.best_bid + snap.best_ask) / _TWO).quantize(
            _MARK_QUANT, rounding=ROUND_HALF_EVEN
        ), True
    if position.mark_price is not None:
        return position.mark_price, False
    paid = (Decimal(position.cost_cents) / (position.qty * _CENTS)).quantize(
        _MARK_QUANT, rounding=ROUND_HALF_EVEN
    )
    # The cost basis is what was paid for this side; express it as a YES price.
    return (paid if position.side is PositionSide.YES else _ONE - paid), False


async def mark_book(
    session: AsyncSession, audit: AuditLog, clock: Clock, book_id: int, asof: datetime
) -> BookState:
    """Re-mark open positions, roll the ET day, update the peak and auto-trip the kill switch."""
    book = await lock_book(session, book_id)
    limits = await load_active_limits(session)
    now = require_utc(clock.now())
    asof = require_utc(asof)
    positions = (
        (
            await session.execute(
                select(Position)
                .where(Position.book_id == book_id, Position.settled.is_(False))
                .order_by(Position.id)
            )
        )
        .scalars()
        .all()
    )
    value = 0
    for position in positions:
        mark, _ = await _mark_price(session, position, asof)
        position.mark_price = mark
        position.updated_at = now
        side_mark = mark if position.side is PositionSide.YES else _ONE - mark
        value += int((position.qty * side_mark * _CENTS).quantize(_ONE, rounding=ROUND_HALF_EVEN))
    equity = book.cash_cents + value
    previous = book.equity_cents

    if et_date(asof) > book.day_start_date:
        book.day_start_equity_cents = previous
        book.day_start_date = et_date(asof)
    book.equity_cents = equity
    book.peak_equity_cents = max(book.peak_equity_cents, equity)
    book.updated_at = now
    await session.flush()

    if equity != previous:
        await audit.append(
            session,
            actor=SYSTEM_ACTOR,
            entity="book",
            entity_id=str(book_id),
            action="book.marked",
            payload={
                "asof": asof,
                "previous_equity_cents": previous,
                "equity_cents": equity,
                "cash_cents": book.cash_cents,
            },
        )

    day_start = book.day_start_equity_cents
    peak = book.peak_equity_cents
    if equity - day_start <= -limits.daily_loss_kill * day_start:
        await trip_kill_switch(
            session,
            audit,
            clock,
            book_id,
            reason=f"daily_loss: equity {equity} vs day start {day_start}",
            actor=SYSTEM_ACTOR,
        )
    if peak - equity >= limits.drawdown_kill * peak:
        await trip_kill_switch(
            session,
            audit,
            clock,
            book_id,
            reason=f"drawdown: equity {equity} vs peak {peak}",
            actor=SYSTEM_ACTOR,
        )
    return book

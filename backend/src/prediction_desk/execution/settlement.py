"""Settlement: close positions on the officially reported value (U1), pay out, re-mark the book.

The payout follows the stored ``measure_outcomes.reported_value`` (never recomputed from the raw
value). A venue's own resolution, when supplied, is only compared and flagged on mismatch.
Lock order is book, then release, so settlement serialises with ``override_outcome``.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Final

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from prediction_desk.audit.log import AuditLog
from prediction_desk.clock import Clock
from prediction_desk.db.models import EventContract, MeasureOutcome, Position, Release, Settlement
from prediction_desk.db.uow import unit_of_work
from prediction_desk.domain.enums import PositionSide
from prediction_desk.domain.types import require_utc
from prediction_desk.execution.outcomes import contract_yes_won
from prediction_desk.markets.terms import terms_from_row
from prediction_desk.risk.book import lock_book, mark_book

SETTLEMENT_ACTOR: Final = "system:settlement"
_CENTS_PER_CONTRACT: Final = 100


@dataclass(frozen=True)
class SettleResult:
    settled: tuple[int, ...]
    pending: tuple[int, ...]
    mismatches: tuple[int, ...]


async def settle_release(
    sessions: async_sessionmaker[AsyncSession],
    audit: AuditLog,
    clock: Clock,
    *,
    book_id: int,
    release_id: int,
    venue_yes_won: Mapping[int, bool] | None = None,
) -> SettleResult:
    """Settle every open position of the book on the release that has an official outcome.

    A position whose measure has no outcome yet, or whose contract is no longer mapped, stays
    open and is listed in ``pending``; nothing settles on a guess. Re-running skips settled
    positions. The book is re-marked last, which may trip the kill switch on a realized loss.
    """
    venue = venue_yes_won or {}
    now = require_utc(clock.now())
    settled: list[int] = []
    pending: list[int] = []
    mismatches: list[int] = []
    async with unit_of_work(sessions) as session:
        book = await lock_book(session, book_id)
        release = (
            await session.execute(select(Release).where(Release.id == release_id).with_for_update())
        ).scalar_one_or_none()
        if release is None:
            raise ValueError(f"unknown release {release_id}")
        rows = (
            await session.execute(
                select(Position, EventContract)
                .join(EventContract, EventContract.id == Position.contract_id)
                .where(
                    Position.book_id == book_id,
                    Position.settled.is_(False),
                    EventContract.release_id == release_id,
                )
                .order_by(Position.id)
            )
        ).all()
        events: list[tuple[str, str, dict[str, object]]] = []
        for position, contract in rows:
            try:
                terms = terms_from_row(contract)
            except ValueError:
                pending.append(contract.id)
                continue
            outcome = await session.get(MeasureOutcome, (release_id, terms.measure))
            if outcome is None:
                pending.append(contract.id)
                continue
            reported = outcome.reported_value
            yes_won = contract_yes_won(terms, reported)
            wins = (position.side is PositionSide.YES) == yes_won
            payout = position.qty * _CENTS_PER_CONTRACT if wins else 0
            realized = payout - position.cost_cents
            venue_result = venue.get(contract.id)
            mismatch = venue_result is not None and venue_result != yes_won
            book.cash_cents += payout
            book.updated_at = now
            position.settled = True
            position.realized_pnl_cents = realized
            position.mark_price = Decimal(1) if yes_won else Decimal(0)
            position.updated_at = now
            session.add(
                Settlement(
                    book_id=book_id,
                    contract_id=contract.id,
                    side=position.side,
                    qty=position.qty,
                    yes_won=yes_won,
                    payout_cents=payout,
                    realized_pnl_cents=realized,
                    reported_value=reported,
                    venue_yes_won=venue_result,
                    mismatch=mismatch,
                    settled_at=now,
                )
            )
            settled.append(contract.id)
            payload: dict[str, object] = {
                "contract_id": contract.id,
                "side": position.side,
                "qty": position.qty,
                "yes_won": yes_won,
                "reported_value": str(reported),
                "payout_cents": payout,
                "realized_pnl_cents": realized,
            }
            events.append(("position", str(position.id), {"action": "position.settled", **payload}))
            if mismatch:
                mismatches.append(contract.id)
                events.append(
                    (
                        "settlement",
                        f"{book_id}:{contract.id}",
                        {
                            "action": "settlement.mismatch",
                            "our_yes_won": yes_won,
                            "venue_yes_won": venue_result,
                            "reported_value": str(reported),
                        },
                    )
                )
        await session.flush()
        for entity, entity_id, body in events:
            action = str(body.pop("action"))
            await audit.append(
                session,
                actor=SETTLEMENT_ACTOR,
                entity=entity,
                entity_id=entity_id,
                action=action,
                payload=body,
            )
        await mark_book(session, audit, clock, book_id, now)
    return SettleResult(tuple(settled), tuple(pending), tuple(mismatches))

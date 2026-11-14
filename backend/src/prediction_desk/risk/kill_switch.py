"""The kill switch: trips are idempotent, resets are admin-only and never automatic (spec 7.4)."""

from sqlalchemy.ext.asyncio import AsyncSession

from prediction_desk.audit.log import AuditLog
from prediction_desk.clock import Clock
from prediction_desk.db.models import BookState
from prediction_desk.domain.types import require_utc

ADMIN_PREFIX = "admin:"


async def _locked(session: AsyncSession, book_id: int) -> BookState:
    # Same lock as book.lock_book; kept local so book.py can import this module.
    book = await session.get(BookState, book_id, with_for_update=True, populate_existing=True)
    if book is None:
        raise LookupError(f"book {book_id} not found")
    return book


async def trip_kill_switch(
    session: AsyncSession,
    audit: AuditLog,
    clock: Clock,
    book_id: int,
    *,
    reason: str,
    actor: str,
) -> bool:
    """Turn the switch on. Returns False, writing nothing, when it is already on."""
    if not reason.strip():
        raise ValueError("kill switch trip needs a reason")
    book = await _locked(session, book_id)
    if book.kill_switch_on:
        return False
    now = require_utc(clock.now())
    book.kill_switch_on = True
    book.kill_reason = reason
    book.kill_actor = actor
    book.kill_at = now
    book.updated_at = now
    await session.flush()
    await audit.append(
        session,
        actor=actor,
        entity="book",
        entity_id=str(book_id),
        action="kill_switch.tripped",
        payload={"reason": reason},
    )
    return True


async def reset_kill_switch(
    session: AsyncSession,
    audit: AuditLog,
    clock: Clock,
    book_id: int,
    *,
    reason: str,
    actor: str,
) -> None:
    """Turn the switch off. Admin actors only; nothing calls this automatically."""
    if not actor.startswith(ADMIN_PREFIX) or not actor.removeprefix(ADMIN_PREFIX).strip():
        raise PermissionError(f"only an {ADMIN_PREFIX}<login> actor may reset the kill switch")
    if not reason.strip():
        raise ValueError("kill switch reset needs a reason")
    book = await _locked(session, book_id)
    if not book.kill_switch_on:
        raise ValueError("kill switch is not on")
    previous = {"reason": book.kill_reason, "actor": book.kill_actor}
    book.kill_switch_on = False
    book.kill_reason = None
    book.kill_actor = None
    book.kill_at = None
    book.updated_at = require_utc(clock.now())
    await session.flush()
    await audit.append(
        session,
        actor=actor,
        entity="book",
        entity_id=str(book_id),
        action="kill_switch.reset",
        payload={
            "reason": reason,
            "tripped_reason": previous["reason"],
            "tripped_by": previous["actor"],
        },
    )

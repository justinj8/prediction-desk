from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from desk_fixtures import BASE_TS, make_book
from prediction_desk.audit.log import AuditLog
from prediction_desk.clock import SimClock
from prediction_desk.db.models import AuditEvent, BookState
from prediction_desk.db.uow import unit_of_work
from prediction_desk.risk.book import mark_book
from prediction_desk.risk.kill_switch import reset_kill_switch, trip_kill_switch


async def _count(session: AsyncSession, action: str) -> int:
    return (
        await session.execute(
            select(func.count()).select_from(AuditEvent).where(AuditEvent.action == action)
        )
    ).scalar_one()


async def _fresh(session: AsyncSession, book_id: int) -> BookState:
    session.expire_all()
    book = await session.get(BookState, book_id)
    assert book is not None
    return book


async def test_trip_is_idempotent_single_audit_event(
    sessions: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    book = await make_book(session)
    clock = SimClock(BASE_TS)
    audit = AuditLog(clock)
    async with unit_of_work(sessions) as s:
        assert await trip_kill_switch(s, audit, clock, book.id, reason="first", actor="system:x")
    clock.set(datetime(2026, 11, 11, 13, 0, tzinfo=UTC))
    async with unit_of_work(sessions) as s:
        assert not await trip_kill_switch(s, audit, clock, book.id, reason="second", actor="y")
    row = await _fresh(session, book.id)
    assert (row.kill_switch_on, row.kill_reason, row.kill_actor) == (True, "first", "system:x")
    assert row.kill_at == BASE_TS
    assert await _count(session, "kill_switch.tripped") == 1


async def test_trip_requires_reason(
    sessions: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    book = await make_book(session)
    clock = SimClock(BASE_TS)
    async with unit_of_work(sessions) as s:
        with pytest.raises(ValueError, match="reason"):
            await trip_kill_switch(s, AuditLog(clock), clock, book.id, reason=" ", actor="a")
    assert (await _fresh(session, book.id)).kill_switch_on is False


async def test_reset_requires_admin_and_reason(
    sessions: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    book = await make_book(session)
    clock = SimClock(BASE_TS)
    audit = AuditLog(clock)
    async with unit_of_work(sessions) as s:
        await trip_kill_switch(s, audit, clock, book.id, reason="x", actor="system:risk")
    async with unit_of_work(sessions) as s:
        with pytest.raises(PermissionError):
            await reset_kill_switch(s, audit, clock, book.id, reason="ok", actor="system:risk")
        with pytest.raises(ValueError, match="reason"):
            await reset_kill_switch(s, audit, clock, book.id, reason="", actor="admin:justin")
    assert (await _fresh(session, book.id)).kill_switch_on is True
    async with unit_of_work(sessions) as s:
        await reset_kill_switch(s, audit, clock, book.id, reason="reviewed", actor="admin:justin")
    row = await _fresh(session, book.id)
    assert row.kill_switch_on is False
    assert row.kill_reason is None and row.kill_actor is None and row.kill_at is None
    event = (
        await session.execute(select(AuditEvent).where(AuditEvent.action == "kill_switch.reset"))
    ).scalar_one()
    assert event.actor == "admin:justin"
    assert event.payload["reason"] == "reviewed"


async def test_reset_when_off_raises(
    sessions: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    book = await make_book(session)
    clock = SimClock(BASE_TS)
    async with unit_of_work(sessions) as s:
        with pytest.raises(ValueError, match="not on"):
            await reset_kill_switch(
                s, AuditLog(clock), clock, book.id, reason="r", actor="admin:justin"
            )


async def test_reset_never_automatic(
    sessions: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    book = await make_book(session)
    book.cash_cents = 900_000
    await session.commit()
    clock = SimClock(BASE_TS)
    audit = AuditLog(clock)
    async with unit_of_work(sessions) as s:
        await mark_book(s, audit, clock, book.id, BASE_TS)
    assert (await _fresh(session, book.id)).kill_switch_on is True
    book.cash_cents = 1_100_000  # equity fully recovers
    await session.commit()
    async with unit_of_work(sessions) as s:
        await mark_book(s, audit, clock, book.id, BASE_TS)
    row = await _fresh(session, book.id)
    assert row.equity_cents == 1_100_000
    assert row.kill_switch_on is True
    assert await _count(session, "kill_switch.reset") == 0


async def test_manual_trip_by_admin_records_actor(
    sessions: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    book = await make_book(session)
    clock = SimClock(BASE_TS)
    async with unit_of_work(sessions) as s:
        await trip_kill_switch(
            s, AuditLog(clock), clock, book.id, reason="manual halt", actor="admin:justin"
        )
    row = await _fresh(session, book.id)
    assert row.kill_actor == "admin:justin"
    event = (
        await session.execute(select(AuditEvent).where(AuditEvent.action == "kill_switch.tripped"))
    ).scalar_one()
    assert event.actor == "admin:justin"
    assert event.payload["reason"] == "manual halt"


async def test_missing_book_raises(sessions: async_sessionmaker[AsyncSession]) -> None:
    clock = SimClock(BASE_TS)
    audit = AuditLog(clock)
    async with unit_of_work(sessions) as s:
        with pytest.raises(LookupError):
            await trip_kill_switch(s, audit, clock, 999, reason="r", actor="a")
        with pytest.raises(LookupError):
            await reset_kill_switch(s, audit, clock, 999, reason="r", actor="admin:x")

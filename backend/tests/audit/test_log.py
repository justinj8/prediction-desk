import asyncio
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from enum import StrEnum
from uuid import UUID

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from prediction_desk.audit.hashing import GENESIS_HASH
from prediction_desk.audit.log import AuditLog
from prediction_desk.clock import SimClock
from prediction_desk.db.models import AuditEvent

T0 = datetime(2024, 3, 12, 12, 0, tzinfo=UTC)


async def _append(log: AuditLog, s: AsyncSession, i: int) -> AuditEvent:
    ev = await log.append(
        s, actor="system", entity="test", entity_id=str(i), action="noop", payload={"i": i}
    )
    await s.commit()
    return ev


async def _append_uncommitted(log: AuditLog, s: AsyncSession, i: int) -> AuditEvent:
    return await log.append(
        s, actor="system", entity="test", entity_id=str(i), action="noop", payload={"i": i}
    )


async def test_empty_chain_verifies(session: AsyncSession) -> None:
    result = await AuditLog(SimClock(T0)).verify(session)
    assert result.ok and result.checked == 0 and result.first_broken_seq is None


async def test_chain_links_and_verifies(session: AsyncSession) -> None:
    log = AuditLog(SimClock(T0))
    first = await _append(log, session, 1)
    second = await _append(log, session, 2)
    assert first.prev_hash == GENESIS_HASH and second.prev_hash == first.hash
    result = await log.verify(session)
    assert result.ok and result.checked == 2 and result.first_broken_seq is None


async def test_append_uses_injected_clock(session: AsyncSession) -> None:
    clock = SimClock(T0)
    clock.advance(timedelta(minutes=7))
    ev = await _append(AuditLog(clock), session, 1)
    assert ev.ts == T0 + timedelta(minutes=7)


async def test_append_rejects_naive_clock_timestamp(session: AsyncSession) -> None:
    class NaiveClock:
        def now(self) -> datetime:
            return datetime(2024, 3, 12, 12, 0)

    with pytest.raises(ValueError, match="timezone"):
        await AuditLog(NaiveClock()).append(
            session, actor="system", entity="t", entity_id="1", action="noop", payload={}
        )
    await session.rollback()


async def test_rolled_back_append_leaves_no_row_and_gap_still_verifies(
    session: AsyncSession,
) -> None:
    log = AuditLog(SimClock(T0))
    await _append(log, session, 1)
    await log.append(
        session, actor="system", entity="test", entity_id="x", action="noop", payload={}
    )
    await session.rollback()
    await _append(log, session, 3)
    rows = (await session.execute(select(AuditEvent).order_by(AuditEvent.seq))).scalars().all()
    assert len(rows) == 2 and rows[1].seq > rows[0].seq + 1  # gap from rollback
    assert (await log.verify(session)).ok


@pytest.mark.parametrize(
    "stmt",
    ["UPDATE audit_events SET action = 'x'", "DELETE FROM audit_events", "TRUNCATE audit_events"],
)
async def test_mutations_are_rejected(session: AsyncSession, stmt: str) -> None:
    await _append(AuditLog(SimClock(T0)), session, 1)
    with pytest.raises(DBAPIError, match="append-only"):
        await session.execute(text(stmt))


async def _tamper(session: AsyncSession, stmt: str, params: dict[str, object]) -> None:
    await session.execute(text("ALTER TABLE audit_events DISABLE TRIGGER USER"))
    await session.execute(text(stmt), params)
    await session.execute(text("ALTER TABLE audit_events ENABLE TRIGGER USER"))
    await session.commit()


async def _seqs(session: AsyncSession) -> list[int]:
    return list((await session.execute(select(AuditEvent.seq).order_by(AuditEvent.seq))).scalars())


async def test_tampering_is_detected(session: AsyncSession) -> None:
    log = AuditLog(SimClock(T0))
    for i in range(3):
        await _append(log, session, i)
    seq2 = (await _seqs(session))[1]
    await _tamper(
        session, "UPDATE audit_events SET payload = '{\"i\": 99}' WHERE seq = :s", {"s": seq2}
    )
    result = await log.verify(session)
    assert not result.ok and result.first_broken_seq == seq2
    assert result.checked == 2  # stops at the broken row, which is counted


@pytest.mark.parametrize(
    "assignment",
    [
        "actor = 'admin:mallory'",
        "action = 'cancel'",
        "entity_id = '999'",
        "entity = 'other'",
        "ts = ts + interval '1 second'",
    ],
)
async def test_tampering_with_any_hashed_field_is_detected(
    session: AsyncSession, assignment: str
) -> None:
    log = AuditLog(SimClock(T0))
    for i in range(3):
        await _append(log, session, i)
    seq2 = (await _seqs(session))[1]
    await _tamper(session, f"UPDATE audit_events SET {assignment} WHERE seq = :s", {"s": seq2})
    result = await log.verify(session)
    assert not result.ok and result.first_broken_seq == seq2 and result.checked == 2


async def test_broken_prev_hash_link_is_detected(session: AsyncSession) -> None:
    log = AuditLog(SimClock(T0))
    for i in range(3):
        await _append(log, session, i)
    seq1 = (await _seqs(session))[0]
    # Deleting the first row leaves the second pointing at a hash that is no longer the previous.
    await _tamper(session, "DELETE FROM audit_events WHERE seq = :s", {"s": seq1})
    seq2 = (await _seqs(session))[0]
    result = await log.verify(session)
    assert not result.ok and result.first_broken_seq == seq2 and result.checked == 1


async def test_rich_typed_payload_roundtrips_and_verifies(engine: AsyncEngine) -> None:
    class Side(StrEnum):
        YES = "yes"

    maker = async_sessionmaker(engine, expire_on_commit=False)
    log = AuditLog(SimClock(T0))
    payload = {
        "price": Decimal("0.12300"),
        "at": datetime(2024, 3, 12, 8, 30, tzinfo=timezone(timedelta(hours=-5))),
        "id": UUID("12345678-1234-5678-1234-567812345678"),
        "side": Side.YES,
        "nested": {"z": [Decimal("1.50"), {"b": 2, "a": 1}], "a": "héllo"},
    }
    async with maker() as s:
        await log.append(
            s, actor="system", entity="order", entity_id="1", action="place", payload=payload
        )
        await s.commit()
    async with maker() as fresh:  # new session: payload comes back as plain JSONB
        stored = (await fresh.execute(select(AuditEvent.payload))).scalar_one()
        assert stored["price"] == "0.12300" and stored["side"] == "yes"
        result = await log.verify(fresh)
    assert result.ok and result.checked == 1


async def test_verify_is_independent_of_session_time_zone(session: AsyncSession) -> None:
    log = AuditLog(SimClock(T0))
    await _append(log, session, 1)
    await _append(log, session, 2)
    await session.execute(text("SET TIME ZONE 'America/New_York'"))
    # Raw SQL bypasses UTCDateTime: confirms the driver really returns local time here.
    raw_ts = (await session.execute(text("SELECT ts FROM audit_events LIMIT 1"))).scalar_one()
    assert raw_ts.utcoffset() != timedelta(0)
    orm_ts = (await session.execute(select(AuditEvent.ts).limit(1))).scalar_one()
    assert orm_ts.tzinfo is UTC  # the ORM column type normalizes it back
    assert (await log.verify(session)).ok


async def test_concurrent_appends_do_not_fork_chain(engine: AsyncEngine) -> None:
    maker = async_sessionmaker(engine, expire_on_commit=False)
    log = AuditLog(SimClock(T0))

    async def worker(i: int) -> None:
        async with maker() as s:
            await _append(log, s, i)

    await asyncio.gather(*(worker(i) for i in range(20)))
    async with maker() as s:
        result = await log.verify(s)
        rows = (await s.execute(select(AuditEvent.prev_hash))).scalars().all()
    assert result.ok and result.checked == 20
    assert len(set(rows)) == 20  # no two rows share a parent: no fork


@pytest.mark.parametrize(
    "assignment",
    [
        "prev_hash = upper(prev_hash)",  # would make compute_hash raise ValueError
        "payload = '{\"i\": 1.5}'::jsonb",  # would make canonical_json raise TypeError
    ],
)
async def test_tampering_that_breaks_recompute_is_reported_not_raised(
    session: AsyncSession, assignment: str
) -> None:
    log = AuditLog(SimClock(T0))
    for i in range(3):
        await _append(log, session, i)
    seq2 = (await _seqs(session))[1]
    await _tamper(session, f"UPDATE audit_events SET {assignment} WHERE seq = :s", {"s": seq2})
    result = await log.verify(session)
    assert not result.ok and result.first_broken_seq == seq2 and result.checked == 2


async def test_verify_ignores_stale_identity_map_objects(session: AsyncSession) -> None:
    log = AuditLog(SimClock(T0))
    held = [await _append(log, session, i) for i in range(3)]  # keep the ORM objects alive
    await _tamper(
        session,
        "UPDATE audit_events SET payload = '{\"i\": 99}' WHERE seq = :s",
        {"s": held[1].seq},
    )
    assert held[1].payload == {"i": 1}  # identity map still holds the pre-tamper value
    result = await log.verify(session)
    assert not result.ok and result.first_broken_seq == held[1].seq


@pytest.mark.parametrize("level", ["REPEATABLE READ", "SERIALIZABLE"])
async def test_append_refuses_non_read_committed_isolation(
    db_url: str, engine: AsyncEngine, level: str
) -> None:
    strict = create_async_engine(db_url, isolation_level=level)
    try:
        async with async_sessionmaker(strict)() as s:
            with pytest.raises(
                RuntimeError, match=f"READ COMMITTED isolation, got {level.lower()}"
            ):
                await _append_uncommitted(AuditLog(SimClock(T0)), s, 1)
            await s.rollback()
    finally:
        await strict.dispose()
    async with async_sessionmaker(engine)() as s:
        count = (await s.execute(select(func.count()).select_from(AuditEvent))).scalar_one()
    assert count == 0


async def test_first_append_matches_golden_hash(session: AsyncSession) -> None:
    ev = await _append_uncommitted(AuditLog(SimClock(T0)), session, 1)
    await session.commit()
    assert ev.seq == 1 and ev.prev_hash == GENESIS_HASH
    assert ev.hash == "b956a4d07efc6f039a0995f4de03242486b83b77d323715afc816a43acbf122c"


async def test_two_appends_in_one_transaction_chain_and_verify(session: AsyncSession) -> None:
    log = AuditLog(SimClock(T0))
    first = await _append_uncommitted(log, session, 1)
    second = await _append_uncommitted(log, session, 2)
    await session.commit()
    assert first.prev_hash == GENESIS_HASH and second.prev_hash == first.hash
    result = await log.verify(session)
    assert result.ok and result.checked == 2

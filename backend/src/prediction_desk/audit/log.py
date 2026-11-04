"""Append-only, hash-chained audit log written inside the caller's transaction."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from prediction_desk.audit.hashing import GENESIS_HASH, canonical_json, compute_hash
from prediction_desk.clock import Clock
from prediction_desk.db.models import AuditEvent
from prediction_desk.domain.types import require_utc

AUDIT_LOCK_KEY: int = 7_310_001


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    checked: int
    first_broken_seq: int | None


def _envelope(
    actor: str, entity: str, entity_id: str, action: str, stored: Mapping[str, Any]
) -> dict[str, Any]:
    """The hash input: every identity field plus the payload, so none can be rewritten unseen."""
    return {
        "actor": actor,
        "entity": entity,
        "entity_id": entity_id,
        "action": action,
        "payload": stored,
    }


class AuditLog:
    def __init__(self, clock: Clock) -> None:
        self._clock = clock

    async def append(
        self,
        session: AsyncSession,
        *,
        actor: str,
        entity: str,
        entity_id: str,
        action: str,
        payload: Mapping[str, Any],
    ) -> AuditEvent:
        """Append one event on the caller's session. Never commits.

        Usage contract:

        - READ COMMITTED only. Under REPEATABLE READ or SERIALIZABLE the transaction snapshot
          predates the lock, so the tail read could miss a concurrent append and fork the
          chain. Any other level raises ``RuntimeError`` (after the lock is taken, before
          anything is written); the caller must roll back. The isolation level is never
          changed on the caller's behalf.
        - Call it last in the transaction, right before commit, and after taking any row
          locks. The advisory lock is global (one key for the whole log) and is held until
          the caller commits or rolls back, so every other appender waits on this transaction.
        - It never commits. Several appends in one transaction chain correctly.
        """
        level = (
            await session.execute(
                text(
                    "SELECT pg_advisory_xact_lock(:key), current_setting('transaction_isolation')"
                ),
                {"key": AUDIT_LOCK_KEY},
            )
        ).one()[1]
        if level != "read committed":
            raise RuntimeError(f"AuditLog.append requires READ COMMITTED isolation, got {level}")
        prev_hash = (
            await session.execute(select(AuditEvent.hash).order_by(AuditEvent.seq.desc()).limit(1))
        ).scalar_one_or_none() or GENESIS_HASH
        ts = require_utc(self._clock.now())
        seq = (
            await session.execute(
                text("SELECT nextval(pg_get_serial_sequence('audit_events', 'seq'))")
            )
        ).scalar_one()
        # JSONB drops Decimal/datetime/UUID/Enum types and key order, so hash what it will return.
        stored: dict[str, Any] = json.loads(canonical_json(payload))
        event = AuditEvent(
            seq=seq,
            ts=ts,
            actor=actor,
            entity=entity,
            entity_id=entity_id,
            action=action,
            payload=stored,
            prev_hash=prev_hash,
            hash=compute_hash(
                prev_hash, _envelope(actor, entity, entity_id, action, stored), seq, ts
            ),
        )
        session.add(event)
        await session.flush()
        return event

    async def verify(self, session: AsyncSession) -> VerifyResult:
        """Walk the chain in seq order; stop at the first row that does not check out.

        Selects plain columns, not entities, so values come from the database rather than
        from ORM objects already in the session's identity map (which may be stale).
        """
        expected_prev = GENESIS_HASH
        checked = 0
        rows = await session.stream(
            select(
                AuditEvent.seq,
                AuditEvent.ts,
                AuditEvent.actor,
                AuditEvent.entity,
                AuditEvent.entity_id,
                AuditEvent.action,
                AuditEvent.payload,
                AuditEvent.prev_hash,
                AuditEvent.hash,
            ).order_by(AuditEvent.seq)
        )
        try:
            async for row in rows:
                checked += 1
                # Link check first: a tampered prev_hash may not even be valid hash input.
                if row.prev_hash != expected_prev:
                    return VerifyResult(ok=False, checked=checked, first_broken_seq=row.seq)
                try:
                    # compute_hash normalizes ts to UTC; the driver returns the session time zone.
                    recomputed = compute_hash(
                        row.prev_hash,
                        _envelope(row.actor, row.entity, row.entity_id, row.action, row.payload),
                        row.seq,
                        row.ts,
                    )
                except (TypeError, ValueError):
                    # A tampered row can hold content append never writes (e.g. a float).
                    return VerifyResult(ok=False, checked=checked, first_broken_seq=row.seq)
                if row.hash != recomputed:
                    return VerifyResult(ok=False, checked=checked, first_broken_seq=row.seq)
                expected_prev = row.hash
        finally:
            await rows.close()
        return VerifyResult(ok=True, checked=checked, first_broken_seq=None)

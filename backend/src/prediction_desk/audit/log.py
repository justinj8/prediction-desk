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

        The advisory lock is transaction-scoped, so concurrent appenders serialize until the
        caller commits or rolls back.
        """
        await session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": AUDIT_LOCK_KEY})
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
        """Walk the chain in seq order; stop at the first row that does not check out."""
        expected_prev = GENESIS_HASH
        checked = 0
        rows = await session.stream_scalars(select(AuditEvent).order_by(AuditEvent.seq))
        try:
            async for row in rows:
                checked += 1
                # compute_hash normalizes row.ts to UTC; the driver returns the session time zone.
                recomputed = compute_hash(
                    row.prev_hash,
                    _envelope(row.actor, row.entity, row.entity_id, row.action, row.payload),
                    row.seq,
                    row.ts,
                )
                if row.prev_hash != expected_prev or row.hash != recomputed:
                    return VerifyResult(ok=False, checked=checked, first_broken_seq=row.seq)
                expected_prev = row.hash
        finally:
            await rows.close()
        return VerifyResult(ok=True, checked=checked, first_broken_seq=None)

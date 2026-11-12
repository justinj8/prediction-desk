"""Contract discovery: upsert every venue market as a canonical EventContract.

An audit event is written only when a contract's mapping identity changes (new row, remap).
The service owns its sessions (ruling R9): releases are read in one short session, each
(venue, series_code) is fetched with no transaction open and then written in its own
transaction, and the audit appends come last, right before commit. A ProviderError on one
series is collected and never stops the others.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Final

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from prediction_desk.audit.log import AuditLog
from prediction_desk.clock import Clock
from prediction_desk.db.models import EventContract, Release
from prediction_desk.domain.enums import MappingStatus, Venue
from prediction_desk.markets.mapping import (
    MAPPER_VERSION,
    ContractTerms,
    Unmapped,
    link_release,
    map_market,
)
from prediction_desk.markets.types import VenueAdapter, VenueMarket
from prediction_desk.net.http import ProviderError

logger = logging.getLogger(__name__)

AUDIT_ACTOR: Final = "system:discovery"
AUDIT_ENTITY: Final = "event_contract"
STRIKE_QUANTUM: Final = Decimal("0.0001")  # Numeric(14, 4)
MAPPED_CONFIDENCE: Final = Decimal("1.00")
UNMAPPED_CONFIDENCE: Final = Decimal("0.00")


@dataclass
class DiscoveryResult:
    mapped: int = 0
    unmapped: int = 0
    changed: int = 0
    unchanged: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class _AuditEntry:
    entity_id: str
    action: str
    payload: dict[str, Any]


def _quantize(value: Decimal | None) -> Decimal | None:
    return None if value is None else value.quantize(STRIKE_QUANTUM)


def _release_fields(
    row: EventContract, releases: Mapping[int, Release]
) -> tuple[str | None, str | None]:
    """(period, ISO scheduled_at) of the linked release, as of this run's release read."""
    release = None if row.release_id is None else releases.get(row.release_id)
    if release is None:
        return None, None
    return release.period, release.scheduled_at.isoformat()


def _identity(row: EventContract, releases: Mapping[int, Release]) -> tuple[object, ...]:
    """What counts as a mapping change; Decimals compare by value, so DB round-trips are equal.

    It mirrors the audit payload, including the linked release's period and scheduled_at.
    Those come from this run's release read for both the old and the new state, so a release
    edited in place (sync_releases moving it) does not by itself re-audit linked contracts.
    """
    return (
        row.mapping_status,
        row.release_id,
        *_release_fields(row, releases),
        row.measure,
        row.comparator,
        row.strike_lo,
        row.strike_hi,
        row.unit,
        row.resolution_precision,
        row.resolution_source,
        row.unmapped_reason,
    )


def _payload(
    venue: Venue, market_id: str, row: EventContract, releases: Mapping[int, Release]
) -> dict[str, Any]:
    strike_lo = _quantize(row.strike_lo)
    strike_hi = _quantize(row.strike_hi)
    release_period, release_scheduled_at = _release_fields(row, releases)
    return {
        "venue": venue.value,
        "market_id": market_id,
        "release_id": row.release_id,
        "release_period": release_period,
        "release_scheduled_at": release_scheduled_at,
        "measure": None if row.measure is None else row.measure.value,
        "comparator": None if row.comparator is None else row.comparator.value,
        "strike_lo": None if strike_lo is None else str(strike_lo),
        "strike_hi": None if strike_hi is None else str(strike_hi),
        "unit": row.unit,
        "resolution_precision": row.resolution_precision,
        "resolution_source": row.resolution_source,
        "status": row.mapping_status.value,
        "reason": row.unmapped_reason,
        "mapper": MAPPER_VERSION,
    }


def _resolve(
    m: VenueMarket, releases: Sequence[Release]
) -> tuple[ContractTerms, Release] | Unmapped:
    terms = map_market(m)
    if isinstance(terms, Unmapped):
        return terms
    release = link_release(m, terms, releases)
    if isinstance(release, Unmapped):
        return release
    return terms, release


def _set_mapping(row: EventContract, outcome: tuple[ContractTerms, Release] | Unmapped) -> None:
    if isinstance(outcome, Unmapped):
        row.mapping_status = MappingStatus.UNMAPPED
        row.mapping_confidence = UNMAPPED_CONFIDENCE
        row.unmapped_reason = outcome.reason
        row.release_id = None
        row.measure = None
        row.comparator = None
        row.strike_lo = None
        row.strike_hi = None
        row.unit = None
        row.resolution_precision = None
        row.resolution_source = None
    else:
        terms, release = outcome
        row.mapping_status = MappingStatus.MAPPED
        row.mapping_confidence = MAPPED_CONFIDENCE
        row.unmapped_reason = None
        row.release_id = release.id
        row.measure = terms.measure
        row.comparator = terms.comparator
        row.strike_lo = _quantize(terms.strike_lo)
        row.strike_hi = _quantize(terms.strike_hi)
        row.unit = terms.unit
        row.resolution_precision = terms.resolution_precision
        row.resolution_source = terms.resolution_source
    row.mapped_by = MAPPER_VERSION


def _snapshot(row: EventContract, releases: Mapping[int, Release]) -> tuple[object, ...]:
    return (
        *_identity(row, releases),
        row.resolution_source,
        row.title,
        row.rules_text,
        row.close_time,
        row.yes_token_id,
        row.venue_event_id,
        row.mapping_confidence,
        row.mapped_by,
    )


def _dedupe(markets: Sequence[VenueMarket]) -> list[VenueMarket]:
    """First occurrence of each market id; a conflicting later duplicate is logged and dropped."""
    first: dict[str, VenueMarket] = {}
    for m in markets:
        kept = first.setdefault(m.market_id, m)
        if kept is not m and kept != m:
            logger.warning(
                "discovery: conflicting duplicate market %s %s in %s; keeping the first",
                m.venue.value,
                m.market_id,
                m.series_code,
            )
    return list(first.values())


async def _upsert_series(
    session: AsyncSession,
    venue: Venue,
    markets: Sequence[VenueMarket],
    releases: Sequence[Release],
    clock: Clock,
    result: DiscoveryResult,
) -> list[_AuditEntry]:
    by_id = {r.id: r for r in releases}
    ids = [m.market_id for m in markets]
    existing = {
        r.venue_market_id: r
        for r in (
            await session.execute(
                select(EventContract).where(
                    EventContract.venue == venue, EventContract.venue_market_id.in_(ids)
                )
            )
        ).scalars()
    }
    now = clock.now()
    entries: list[tuple[EventContract, str]] = []
    for m in markets:
        outcome = _resolve(m, releases)
        row = existing.get(m.market_id)
        is_new = row is None
        if row is None:
            row = EventContract(
                venue=venue, venue_market_id=m.market_id, created_at=now, updated_at=now
            )
            session.add(row)
            existing[m.market_id] = row
            before: tuple[object, ...] | None = None
        else:
            before = _snapshot(row, by_id)
        old_identity = None if is_new else _identity(row, by_id)
        row.venue_event_id = m.event_id
        row.yes_token_id = m.yes_token_id
        row.title = m.title
        row.rules_text = m.rules_text
        row.close_time = m.close_time
        _set_mapping(row, outcome)
        if row.mapping_status is MappingStatus.MAPPED:
            result.mapped += 1
        else:
            result.unmapped += 1
        if is_new:
            action = (
                "contract_mapped"
                if row.mapping_status is MappingStatus.MAPPED
                else "contract_unmapped"
            )
            entries.append((row, action))
            result.changed += 1
        elif _identity(row, by_id) != old_identity:
            entries.append((row, "contract_remapped"))
            result.changed += 1
        else:
            result.unchanged += 1
        if before is not None and _snapshot(row, by_id) != before:
            row.updated_at = now
    await session.flush()
    return [
        _AuditEntry(
            f"{venue.value}:{r.venue_market_id}",
            action,
            _payload(venue, r.venue_market_id, r, by_id),
        )
        for r, action in entries
    ]


async def discover_contracts(
    sessions: async_sessionmaker[AsyncSession],
    adapters: Mapping[Venue, tuple[VenueAdapter, Sequence[str]]],
    audit: AuditLog,
    clock: Clock,
) -> DiscoveryResult:
    """Upsert every listed market of every (venue, series_code); audit mapping changes.

    Collects: a ``ProviderError`` from ``list_markets`` is recorded in ``result.failures`` as
    ``(series_code, "Type: message")`` and the run continues with the next series. Any other
    exception rolls back that series' transaction and propagates; series already committed
    stay committed.
    """
    result = DiscoveryResult()
    async with sessions() as session:
        # expire_on_commit=False sessions leave loaded attributes readable once closed.
        releases = list((await session.execute(select(Release))).scalars())
    for venue, (adapter, codes) in adapters.items():
        for code in codes:
            try:
                markets = await adapter.list_markets(code)
            except ProviderError as e:
                result.failures.append((code, f"{type(e).__name__}: {e}"))
                continue
            unique = _dedupe(markets)
            async with sessions() as session, session.begin():
                entries = await _upsert_series(session, venue, unique, releases, clock, result)
                for entry in entries:  # audit last, right before commit
                    await audit.append(
                        session,
                        actor=AUDIT_ACTOR,
                        entity=AUDIT_ENTITY,
                        entity_id=entry.entity_id,
                        action=entry.action,
                        payload=entry.payload,
                    )
    return result

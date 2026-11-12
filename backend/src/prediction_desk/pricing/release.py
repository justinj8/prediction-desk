"""End-to-end priced view of one release: model -> fair value -> stored books -> edges.

Read-only: nothing is written and, because the whole body runs under ``no_autoflush``, the
caller's pending or dirty objects are not flushed either. The session lifecycle stays with the
caller (Plan 2 R9: readers take an ``AsyncSession``): no commit, no rollback; SQLAlchemy's own
autobegin on the first read is the only transaction effect.

Contract rows are read as they are now, not as of ``asof`` (bitemporal contract history is a
Plan 5 item); snapshots, model inputs and fee schedules are as-of.

A missing fee schedule (``LookupError`` from ``schedule_for``) propagates: it is a configuration
error, not a data gap.
"""

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from types import MappingProxyType

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from prediction_desk.audit.hashing import canonical_json
from prediction_desk.db.models import EventContract, Release
from prediction_desk.domain.enums import Comparator, MappingStatus, Measure, Series, Venue
from prediction_desk.domain.types import Prob, require_utc
from prediction_desk.forecasting.base import BaseModel
from prediction_desk.forecasting.distributions import Distribution
from prediction_desk.forecasting.realtime import InsufficientData
from prediction_desk.forecasting.registry import MODELS
from prediction_desk.ingestion.provider import MacroDataProvider
from prediction_desk.markets.mapping import ContractTerms
from prediction_desk.markets.snapshots import latest_snapshot_asof
from prediction_desk.markets.terms import rules_hash, terms_from_row
from prediction_desk.markets.types import OrderBook
from prediction_desk.pricing.edge import EdgeQuote, Unavailable, best_side, evaluate_edge
from prediction_desk.pricing.fair import bracket_sum_violation, fair_prob, monotonic_violations
from prediction_desk.pricing.fees import FeeSchedule, schedule_for

PRICING_VERSION = "pricing:v1"

_LOWER = {Comparator.LT, Comparator.LTE}
_UPPER = {Comparator.GT, Comparator.GTE}

# Everything OrderBook.from_json can raise on a malformed stored book.
_BAD_BOOK = (KeyError, TypeError, ValueError, InvalidOperation)


@dataclass(frozen=True)
class ContractPricing:
    contract_id: int
    venue: Venue
    terms: ContractTerms
    fair_yes: Prob
    snapshot_id: int | None
    quote_age_s: int | None
    quote: EdgeQuote | Unavailable  # the best side
    yes: EdgeQuote | Unavailable
    no: EdgeQuote | Unavailable


@dataclass(frozen=True)
class ReleasePricing:
    release_id: int
    asof: datetime
    model_version: str | None
    distributions: Mapping[Measure, Mapping[str, str]]  # params() plus "family"
    contracts: tuple[ContractPricing, ...]  # ordered by contract id
    unpriceable: Mapping[Measure, str]  # measure -> reason
    rejected: Mapping[int, str]  # contract id -> "bad_terms" / "terms_distribution_mismatch"
    coherence: tuple[str, ...]  # sorted violations
    inputs_hash: str


def _price_contract(
    row: EventContract,
    terms: ContractTerms,
    fair_yes: Prob,
    snap_id: int | None,
    quote_age_s: int | None,
    pair: tuple[EdgeQuote | Unavailable, EdgeQuote | Unavailable],
) -> ContractPricing:
    return ContractPricing(
        contract_id=row.id,
        venue=row.venue,
        terms=terms,
        fair_yes=fair_yes,
        snapshot_id=snap_id,
        quote_age_s=quote_age_s,
        quote=best_side(pair),
        yes=pair[0],
        no=pair[1],
    )


async def _quotes(
    session: AsyncSession,
    row: EventContract,
    terms: ContractTerms,
    fair_yes: Prob,
    asof: datetime,
    probe_qty: int,
    max_quote_age: timedelta,
    fee_schedules: Sequence[FeeSchedule] | None,
) -> ContractPricing:
    snap = await latest_snapshot_asof(session, row.id, asof)
    if snap is None:
        gap = Unavailable("no_snapshot")
        return _price_contract(row, terms, fair_yes, None, None, (gap, gap))
    age = asof - snap.ts
    age_s = int(age.total_seconds())
    if age > max_quote_age:
        gap = Unavailable("stale_snapshot")
        return _price_contract(row, terms, fair_yes, snap.id, age_s, (gap, gap))
    try:
        book = OrderBook.from_json(snap.book, snap.ts, snap.venue_ts)
    except _BAD_BOOK:
        gap = Unavailable("bad_snapshot")
        return _price_contract(row, terms, fair_yes, snap.id, age_s, (gap, gap))
    pair = evaluate_edge(fair_yes, book, row.venue, probe_qty, asof, fee_schedules)
    return _price_contract(row, terms, fair_yes, snap.id, age_s, pair)


def _dec(value: Decimal | None) -> str | None:
    """Scale-independent text (DB ``0.3000`` and in-memory ``0.3`` hash alike)."""
    return None if value is None else format(value.normalize(), "f")


def _is_partition(comparators: list[Comparator]) -> bool:
    """Exactly one lower tail, exactly one upper tail; everything else is EQ/BETWEEN."""
    return sum(c in _LOWER for c in comparators) == 1 and sum(c in _UPPER for c in comparators) == 1


async def _coherence(
    session: AsyncSession, priced: list[tuple[ContractPricing, EventContract]]
) -> tuple[str, ...]:
    """Monotonic ladders per venue; Polymarket brackets (R10)."""
    out: list[str] = []
    by_venue: dict[Venue, list[ContractPricing]] = defaultdict(list)
    by_event: dict[str, list[ContractPricing]] = defaultdict(list)
    for pricing, row in priced:
        by_venue[pricing.venue].append(pricing)
        if pricing.venue == Venue.POLYMARKET:
            by_event[row.venue_event_id].append(pricing)
    for venue, members in by_venue.items():
        out.extend(
            f"{venue.value}: {msg}"
            for msg in monotonic_violations([(c.terms, c.fair_yes) for c in members])
        )
    totals: dict[str, int] = {}
    if by_event:
        counts = await session.execute(
            select(EventContract.venue_event_id, func.count())
            .where(
                EventContract.venue == Venue.POLYMARKET,
                EventContract.venue_event_id.in_(list(by_event)),
            )
            .group_by(EventContract.venue_event_id)
        )
        totals = {event_id: n for event_id, n in counts}
    for event_id, members in by_event.items():
        k, n = len(members), totals[event_id]
        if k == n and _is_partition([c.terms.comparator for c in members]):
            msg = bracket_sum_violation([c.fair_yes for c in members])
            if msg is not None:
                out.append(f"polymarket event {event_id}: {msg}")
        else:
            out.append(f"polymarket event {event_id}: incomplete bracket ({k} of {n} mapped)")
    return tuple(sorted(out))


async def price_release(
    session: AsyncSession,
    provider: MacroDataProvider,
    release: Release,
    asof: datetime,
    *,
    probe_qty: int = 10,
    max_quote_age: timedelta = timedelta(seconds=60),
    models: Mapping[Series, BaseModel] = MODELS,
    fee_schedules: Sequence[FeeSchedule] | None = None,
) -> ReleasePricing:
    """Price every mapped contract of ``release`` as of ``asof``.

    Model data gaps (``InsufficientData``, no model, no distribution for a measure) become
    ``unpriceable`` entries; any other model error (e.g. the release is already published at
    ``asof``) propagates. A malformed mapped contract is skipped alone and listed in
    ``rejected`` (R11). A missing, stale or corrupt book makes that contract's quotes
    ``Unavailable`` without aborting the rest. ``fee_schedules=None`` uses the packaged ones.
    """
    with session.no_autoflush:
        return await _price_release(
            session,
            provider,
            release,
            require_utc(asof),
            probe_qty,
            max_quote_age,
            models,
            fee_schedules,
        )


async def _price_release(
    session: AsyncSession,
    provider: MacroDataProvider,
    release: Release,
    asof: datetime,
    probe_qty: int,
    max_quote_age: timedelta,
    models: Mapping[Series, BaseModel],
    fee_schedules: Sequence[FeeSchedule] | None,
) -> ReleasePricing:
    rows = (
        (
            await session.execute(
                select(EventContract)
                .where(
                    EventContract.release_id == release.id,
                    EventContract.mapping_status == MappingStatus.MAPPED,
                )
                .order_by(EventContract.id)
            )
        )
        .scalars()
        .all()
    )
    terms_by_id: dict[int, ContractTerms] = {}
    rejected: dict[int, str] = {}
    for row in rows:
        try:
            terms_by_id[row.id] = terms_from_row(row)
        except ValueError:
            rejected[row.id] = "bad_terms"

    model_version: str | None = None
    dists: Mapping[Measure, Distribution] = {}
    unpriceable: dict[Measure, str] = {}
    needed = {t.measure for t in terms_by_id.values()}
    model = models.get(release.series)
    if model is None:
        unpriceable = dict.fromkeys(needed, "no_model")
    else:
        try:
            out = await model.predict(session, provider, release, asof)
        except InsufficientData:
            unpriceable = dict.fromkeys(needed, "insufficient_data")
        else:
            model_version = out.model_version
            dists = out.distributions

    priced: list[tuple[ContractPricing, EventContract]] = []
    for row in rows:
        terms = terms_by_id.get(row.id)
        if terms is None:
            continue
        dist = dists.get(terms.measure)
        if dist is None:
            unpriceable.setdefault(terms.measure, "no_distribution")
            continue
        try:
            fair_yes = fair_prob(terms, dist)
        except ValueError:
            rejected[row.id] = "terms_distribution_mismatch"
            continue
        pricing = await _quotes(
            session, row, terms, fair_yes, asof, probe_qty, max_quote_age, fee_schedules
        )
        priced.append((pricing, row))

    snapshot_ids = {p.contract_id: p.snapshot_id for p, _ in priced}
    distributions = {
        m: MappingProxyType({"family": d.family, **d.params()})
        for m, d in sorted(dists.items(), key=lambda kv: kv[0].value)
    }
    unpriceable_sorted = dict(sorted(unpriceable.items(), key=lambda kv: kv[0].value))
    rejected_sorted = dict(sorted(rejected.items()))
    venues = sorted({p.venue for p, _ in priced}, key=lambda v: v.value)
    schedules_used = {v: schedule_for(v, asof, fee_schedules) for v in venues}
    payload = {
        "pricing_version": PRICING_VERSION,
        "release_id": release.id,
        "asof": asof,
        "model_version": model_version,
        "distributions": {m.value: dict(params) for m, params in distributions.items()},
        "contracts": [
            {
                "id": row.id,
                "measure": row.measure,
                "comparator": row.comparator,
                "strike_lo": _dec(row.strike_lo),
                "strike_hi": _dec(row.strike_hi),
                "unit": row.unit,
                "resolution_precision": row.resolution_precision,
                "resolution_source": row.resolution_source,
                "rules_hash": rules_hash(row),
                "snapshot_id": snapshot_ids.get(row.id),
            }
            for row in rows
        ],
        "unpriceable": {m.value: reason for m, reason in unpriceable_sorted.items()},
        "rejected": {str(cid): reason for cid, reason in rejected_sorted.items()},
        "probe_qty": probe_qty,
        "max_quote_age_us": max_quote_age // timedelta(microseconds=1),
        "fee_schedules": {
            v.value: {"effective_from": sc.effective_from, "kind": sc.kind, "rate": _dec(sc.rate)}
            for v, sc in schedules_used.items()
        },
    }
    return ReleasePricing(
        release_id=release.id,
        asof=asof,
        model_version=model_version,
        distributions=MappingProxyType(distributions),
        contracts=tuple(p for p, _ in priced),
        unpriceable=MappingProxyType(unpriceable_sorted),
        rejected=MappingProxyType(rejected_sorted),
        coherence=await _coherence(session, priced),
        inputs_hash=hashlib.sha256(canonical_json(payload)).hexdigest(),
    )

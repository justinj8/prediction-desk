"""End-to-end priced view of one release: model -> fair value -> stored books -> edges.

Read-only. Nothing is committed and the caller's session is never committed, rolled back or
otherwise steered (R9: readers take an ``AsyncSession``); SQLAlchemy's own autobegin on the first
read is the only transaction effect.
"""

import hashlib
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import InvalidOperation
from types import MappingProxyType

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from prediction_desk.audit.hashing import canonical_json
from prediction_desk.db.models import EventContract, Release
from prediction_desk.domain.enums import MappingStatus, Measure, Series, Venue
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
from prediction_desk.pricing.fees import schedule_for

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
    pair = evaluate_edge(fair_yes, book, row.venue, probe_qty, asof)
    return _price_contract(row, terms, fair_yes, snap.id, age_s, pair)


def _coherence(priced: list[tuple[ContractPricing, str]]) -> tuple[str, ...]:
    """Monotonic ladders per venue, and bracket sums per Polymarket event (2+ contracts)."""
    out: list[str] = []
    by_venue: dict[Venue, list[ContractPricing]] = defaultdict(list)
    by_event: dict[str, list[ContractPricing]] = defaultdict(list)
    for pricing, event_id in priced:
        by_venue[pricing.venue].append(pricing)
        if pricing.venue == Venue.POLYMARKET:
            by_event[event_id].append(pricing)
    for venue, members in by_venue.items():
        out.extend(
            f"{venue.value}: {msg}"
            for msg in monotonic_violations([(c.terms, c.fair_yes) for c in members])
        )
    for event_id, members in by_event.items():
        if len(members) < 2:
            continue
        msg = bracket_sum_violation([c.fair_yes for c in members])
        if msg is not None:
            out.append(f"polymarket event {event_id}: {msg}")
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
) -> ReleasePricing:
    """Price every mapped contract of ``release`` as of ``asof``.

    Model data gaps (``InsufficientData``, no model, no distribution for a measure) become
    ``unpriceable`` entries; any other model error (e.g. the release is already published at
    ``asof``) propagates. A missing, stale or corrupt book makes that contract's quotes
    ``Unavailable`` without aborting the rest.
    """
    asof = require_utc(asof)
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
    terms_by_id = {row.id: terms_from_row(row) for row in rows}

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

    priced: list[tuple[ContractPricing, str]] = []
    for row in rows:
        terms = terms_by_id[row.id]
        dist = dists.get(terms.measure)
        if dist is None:
            unpriceable.setdefault(terms.measure, "no_distribution")
            continue
        pricing = await _quotes(
            session, row, terms, fair_prob(terms, dist), asof, probe_qty, max_quote_age
        )
        priced.append((pricing, row.venue_event_id))

    by_id = {row.id: row for row in rows}
    distributions = {
        m: {"family": d.family, **d.params()}
        for m, d in sorted(dists.items(), key=lambda kv: kv[0].value)
    }
    venues = sorted({p.venue for p, _ in priced}, key=lambda v: v.value)
    payload = {
        "release_id": release.id,
        "asof": asof,
        "model_version": model_version,
        "distributions": {m.value: params for m, params in distributions.items()},
        "contracts": [
            {
                "id": p.contract_id,
                "rules_hash": rules_hash(by_id[p.contract_id]),
                "snapshot_id": p.snapshot_id,
            }
            for p, _ in priced
        ],
        "probe_qty": probe_qty,
        "fee_effective_from": {v.value: schedule_for(v, asof).effective_from for v in venues},
    }
    return ReleasePricing(
        release_id=release.id,
        asof=asof,
        model_version=model_version,
        distributions=MappingProxyType(distributions),
        contracts=tuple(p for p, _ in priced),
        unpriceable=MappingProxyType(dict(sorted(unpriceable.items(), key=lambda kv: kv[0].value))),
        coherence=_coherence(priced),
        inputs_hash=hashlib.sha256(canonical_json(payload)).hexdigest(),
    )

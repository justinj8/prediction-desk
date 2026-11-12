"""The risk engine (spec 7.2/7.3): per-contract gates, the sizing funnel, the Risk Officer port
and the only constructor of ``ApprovedOrder``.

``evaluate`` runs in three phases so no connection is held while the officer (an LLM) thinks:

- **A, read** in a short session: limits, book, release, contracts, snapshots, holdings; the
  nine gates; ``size_order`` per survivor in contract-id order against a running exposure.
- **B, review** outside any transaction: the officer may only veto or reduce; any failure of
  it rejects the whole slate (fail closed).
- **C, write** in one ``unit_of_work`` under the book lock (D2): re-check the switch, the side
  and the caps against fresh holdings, insert every decision, audit ``risk.decided`` once.

Every gate, sizing and officer outcome is recorded in ``reasons``; the last entry is always
``{"step": "approved", ...}`` or ``{"step": "rejected", "reason": <code>}``.
"""

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from fractions import Fraction
from math import ceil
from typing import Final, Protocol, cast

from sqlalchemy import ColumnElement, exists, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from prediction_desk.audit.log import AuditLog
from prediction_desk.clock import Clock
from prediction_desk.db.models import (
    BookState,
    EventContract,
    MarketSnapshot,
    Order,
    Position,
    Release,
    RiskDecision,
)
from prediction_desk.db.uow import unit_of_work
from prediction_desk.domain.enums import BookMode, PositionSide, Slot, Venue, Verdict
from prediction_desk.domain.types import Prob, require_utc
from prediction_desk.markets.types import OrderBook
from prediction_desk.pricing.book import side_levels
from prediction_desk.pricing.coherence import is_bug
from prediction_desk.pricing.edge import Unavailable
from prediction_desk.pricing.fees import FeeSchedule, schedule_for
from prediction_desk.pricing.release import ContractPricing, ReleasePricing
from prediction_desk.risk.book import SYSTEM_ACTOR, Exposure, exposure, lock_book
from prediction_desk.risk.kill_switch import trip_kill_switch
from prediction_desk.risk.limits import RiskLimits, load_active_limits
from prediction_desk.risk.sizing import (
    Sizing,
    SizingInputs,
    clamp_fair,
    depth_at_or_better,
    limit_price,
    marginal_fee,
    size_order,
)

RISK_ENGINE_VERSION: Final = "risk:v1"
_SEAL: Final = object()
_ZERO = Decimal(0)
_CENT = Decimal(100)

Reason = dict[str, str]


@dataclass(frozen=True)
class ApprovedOrder:
    """A risk approval the broker may act on. Only ``_approve`` can build one (spec 7.3)."""

    risk_decision_id: int
    client_order_id: str
    book_id: int
    contract_id: int
    side: PositionSide
    limit_price: Decimal
    qty: int
    decided_at: datetime
    _seal: object = field(default=None, kw_only=True, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _SEAL:
            raise TypeError("ApprovedOrder is created only by the RiskEngine")
        # Spend the token, so dataclasses.replace() cannot reuse it to forge a copy.
        object.__setattr__(self, "_seal", None)


@dataclass(frozen=True)
class SlateItem:
    contract_id: int
    venue: Venue
    side: PositionSide
    qty_engine: int
    stake_engine_cents: int
    fair_clamped: Prob
    cost_per_contract: Decimal
    reasons: tuple[Mapping[str, str], ...]


class RiskOfficer(Protocol):
    async def review(self, slate: Sequence[SlateItem]) -> Mapping[int, int]:
        """Contract id -> maximum qty for every slate item; 0 vetoes."""
        ...


@dataclass(frozen=True)
class EvaluationResult:
    decision_ids: tuple[int, ...]
    approved: tuple[ApprovedOrder, ...]
    killed: bool  # the book's kill switch is on as of the decision transaction


def _approve(decision: RiskDecision) -> ApprovedOrder:
    """The only constructor path of ``ApprovedOrder``; ``client_order_id`` per D8.

    Test-only seam (ruling R2): broker and settlement tests insert a real approved
    ``RiskDecision`` row and call this, so they do not depend on sizing.
    """
    if (
        decision.verdict is not Verdict.APPROVED
        or decision.side is None
        or decision.limit_price is None
    ):
        raise ValueError(f"risk decision {decision.id} is not approved")
    key = f"{decision.book_id}|{decision.run_key}|{decision.slot.value}|{decision.contract_id}"
    return ApprovedOrder(
        risk_decision_id=decision.id,
        client_order_id=hashlib.sha256(key.encode()).hexdigest(),
        book_id=decision.book_id,
        contract_id=decision.contract_id,
        side=decision.side,
        limit_price=decision.limit_price,
        qty=decision.qty,
        decided_at=decision.decided_at,
        _seal=_SEAL,
    )


@dataclass(frozen=True)
class _Side:
    """One side's gate-9 inputs: clamped fair ``q``, probe cost ``c`` (D5), limit and depth."""

    side: PositionSide
    q: Prob
    c: Decimal
    limit: Decimal | None
    depth: Decimal
    worst: Decimal  # limit + marginal fee at the limit (ruling R5)
    passes: bool

    @property
    def edge(self) -> Decimal:
        return self.q - self.c

    @property
    def unit_cost(self) -> Decimal:
        return max(self.c, self.worst)


@dataclass
class _Draft:
    cp: ContractPricing
    pick: _Side | None = None
    reasons: list[Reason] = field(default_factory=list)
    rejected: str | None = None
    kelly_raw: int | None = None
    stake_engine: int | None = None
    stake_officer: int | None = None
    stake_final: int | None = None
    qty: int = 0

    @property
    def cid(self) -> int:
        return self.cp.contract_id


class _Holdings:
    """A running copy of the exposure; each approved stake is added before the next sizing."""

    def __init__(self, exp: Exposure, release_id: int) -> None:
        self.contract = defaultdict(int, exp.contract)
        self.event = exp.event.get(release_id, 0)
        self.portfolio = exp.portfolio

    def size(self, d: _Draft, equity_cents: int, limits: RiskLimits) -> Sizing:
        p = d.pick
        assert p is not None and p.limit is not None
        inputs = SizingInputs(
            equity_cents,
            p.q,
            p.c,
            self.contract[d.cid],
            self.event,
            self.portfolio,
            p.depth,
            p.limit,
            p.worst,
        )
        return size_order(inputs, limits)

    def add(self, d: _Draft, cents: int) -> None:
        self.contract[d.cid] += cents
        self.event += cents
        self.portfolio += cents


def _stake(qty: int, unit_cost: Decimal) -> int:
    """Cents reserved for ``qty`` at ``unit_cost``, rounded as ``size_order`` rounds."""
    return ceil(qty * Fraction(unit_cost * _CENT))


def _reject(drafts: Iterable[_Draft], code: str, entry: Reason | None = None) -> None:
    for d in drafts:
        if entry is not None:
            d.reasons.append(dict(entry))
        d.rejected = code


async def _held_sides(
    s: AsyncSession, book_id: int, contract_ids: Sequence[int]
) -> dict[int, set[PositionSide]]:
    """Sides held per contract: open positions plus approved decisions with no order (D1)."""
    held = await s.execute(
        select(Position.contract_id, Position.side).where(
            Position.book_id == book_id,
            Position.settled.is_(False),
            Position.contract_id.in_(contract_ids),
        )
    )
    pending = await s.execute(
        select(RiskDecision.contract_id, RiskDecision.side).where(
            RiskDecision.book_id == book_id,
            RiskDecision.verdict == Verdict.APPROVED,
            RiskDecision.contract_id.in_(contract_ids),
            ~exists().where(Order.risk_decision_id == RiskDecision.id),
        )
    )
    out: dict[int, set[PositionSide]] = defaultdict(set)
    for cid, side in [*held.all(), *pending.all()]:
        out[cid].add(cast(PositionSide, side))  # approved rows always carry a side (CHECK)
    return out


def _decided_filter(book_id: int, run_key: str, slot: Slot) -> tuple[ColumnElement[bool], ...]:
    return (
        RiskDecision.book_id == book_id,
        RiskDecision.run_key == run_key,
        RiskDecision.slot == slot,
    )


async def _decided(s: AsyncSession, book_id: int, run_key: str, slot: Slot) -> bool:
    return bool(await s.scalar(select(exists().where(*_decided_filter(book_id, run_key, slot)))))


async def _stored(s: AsyncSession, book: BookState, run_key: str, slot: Slot) -> EvaluationResult:
    """D8: the stored decisions; approvals rebuilt only if unordered and the switch is off."""
    rows = (
        await s.scalars(
            select(RiskDecision)
            .where(*_decided_filter(book.id, run_key, slot))
            .order_by(RiskDecision.id)
        )
    ).all()
    ordered = set(
        (
            await s.scalars(
                select(Order.risk_decision_id).where(
                    Order.risk_decision_id.in_([r.id for r in rows])
                )
            )
        ).all()
    )
    approved = (
        ()
        if book.kill_switch_on
        else tuple(
            _approve(r) for r in rows if r.verdict is Verdict.APPROVED and r.id not in ordered
        )
    )
    return EvaluationResult(tuple(r.id for r in rows), approved, book.kill_switch_on)


class RiskEngine:
    """The only producer of ``ApprovedOrder``. ``fee_schedules`` as for ``price_release``."""

    def __init__(
        self, clock: Clock, audit: AuditLog, fee_schedules: Sequence[FeeSchedule] | None = None
    ) -> None:
        self._clock = clock
        self._audit = audit
        self._fee_schedules = fee_schedules

    async def evaluate(
        self,
        sessions: async_sessionmaker[AsyncSession],
        *,
        book_id: int,
        run_key: str,
        slot: Slot,
        pricing: ReleasePricing,
        healthy_venues: frozenset[Venue],
        officer: RiskOfficer | None,
    ) -> EvaluationResult:
        """Decide every priced contract of ``pricing`` for ``(book_id, run_key, slot)``.

        Re-running the same ``(book_id, run_key, slot)`` returns the stored decisions and
        writes nothing (D8). A held position whose release is unknown makes ``exposure`` raise
        ``RuntimeError``; that propagates (fail closed), as does a missing book or release.
        """
        if not run_key:
            raise ValueError("run_key must be a non-empty string")
        bugs = [v.message for v in pricing.coherence if is_bug(v)]
        async with sessions() as s:
            screened = None
            if not await _decided(s, book_id, run_key, slot):
                screened = await self._screen(s, book_id, pricing, healthy_venues, bugs)
        if screened is None:
            async with unit_of_work(sessions) as s:
                return await _stored(s, await lock_book(s, book_id), run_key, slot)
        limits, mode, drafts = screened

        await self._review(officer, mode, [d for d in drafts if d.rejected is None])

        async with unit_of_work(sessions) as s:
            book = await lock_book(s, book_id)
            if await _decided(s, book_id, run_key, slot):  # a concurrent run got here first
                return await _stored(s, book, run_key, slot)
            if bugs:
                _reject(drafts, "pricing_sanity_violation")
            elif book.kill_switch_on:
                _reject(drafts, "kill_switch_on")
            else:
                await self._recheck(s, book, pricing.release_id, drafts, limits)
            now = require_utc(self._clock.now())
            rows = [self._row(d, book_id, run_key, slot, pricing, limits, now) for d in drafts]
            s.add_all(rows)
            await s.flush()
            if bugs:
                await trip_kill_switch(
                    s,
                    self._audit,
                    self._clock,
                    book_id,
                    reason="pricing_sanity: " + "; ".join(bugs),
                    actor=SYSTEM_ACTOR,
                )
            await self._audit.append(
                s,
                actor=SYSTEM_ACTOR,
                entity="book",
                entity_id=str(book_id),
                action="risk.decided",
                payload={
                    "run_key": run_key,
                    "slot": slot.value,
                    "release_id": pricing.release_id,
                    "inputs_hash": pricing.inputs_hash,
                    "decisions": [
                        {
                            "id": r.id,
                            "contract_id": r.contract_id,
                            "verdict": r.verdict.value,
                            "qty": r.qty,
                            "reason": d.rejected,
                        }
                        for r, d in zip(rows, drafts, strict=True)
                    ],
                },
            )
            approved = tuple(_approve(r) for r in rows if r.verdict is Verdict.APPROVED)
            return EvaluationResult(tuple(r.id for r in rows), approved, book.kill_switch_on)

    async def _screen(
        self,
        s: AsyncSession,
        book_id: int,
        pricing: ReleasePricing,
        healthy: frozenset[Venue],
        bugs: Sequence[str],
    ) -> tuple[RiskLimits, BookMode, list[_Draft]]:
        """Phase A: gates, then sizing in contract-id order against a running exposure."""
        limits = await load_active_limits(s)
        book = await s.get_one(BookState, book_id)
        release = await s.get_one(Release, pricing.release_id)
        drafts = sorted((_Draft(cp) for cp in pricing.contracts), key=lambda d: d.cid)
        ids = [d.cid for d in drafts]
        rows = {
            r.id: r
            for r in (await s.scalars(select(EventContract).where(EventContract.id.in_(ids))))
        }
        snap_ids = [d.cp.snapshot_id for d in drafts if d.cp.snapshot_id is not None]
        snaps = {
            m.id: m
            for m in (
                await s.scalars(select(MarketSnapshot).where(MarketSnapshot.id.in_(snap_ids)))
            )
        }
        held = await _held_sides(s, book_id, ids)
        hold = _Holdings(await exposure(s, book_id), release.id)
        if bugs:
            _reject(drafts, "pricing_sanity_violation")
            return limits, book.mode, drafts

        now = require_utc(self._clock.now())
        asof = require_utc(pricing.asof)
        incomplete = {v.scope for v in pricing.coherence if v.kind == "incomplete_bracket"}
        cutoff = release.scheduled_at - timedelta(minutes=limits.cutoff_minutes)
        for d in drafts:
            row, quote = rows[d.cid], d.cp.quote
            if book.kill_switch_on:
                d.rejected = "kill_switch_on"
            elif now - asof > timedelta(seconds=limits.max_quote_age_s):
                d.rejected = "stale_pricing"
            elif asof >= cutoff:
                d.rejected = "past_cutoff"
            elif (
                row.release_id != release.id
                or row.mapping_confidence < limits.min_mapping_confidence
            ):
                d.rejected = "low_mapping_confidence"
            elif d.cp.venue not in healthy:
                d.rejected = "venue_unhealthy"
            elif isinstance(quote, Unavailable):
                d.rejected = f"no_quote:{quote.reason}"
            elif d.cp.venue is Venue.POLYMARKET and row.venue_event_id in incomplete:
                d.rejected = "event_incoherent"
            else:
                assert d.cp.snapshot_id is not None  # an available quote came from a snapshot
                d.rejected = self._pick(d, snaps[d.cp.snapshot_id], asof, limits, held)
            if d.rejected is not None:
                continue
            sizing = hold.size(d, book.equity_cents, limits)
            d.reasons.extend(sizing.reasons)
            d.kelly_raw = sizing.kelly_raw_cents
            d.rejected = sizing.rejected
            if sizing.rejected is None:
                d.qty = sizing.qty
                d.stake_engine = sizing.stake_cents
                hold.add(d, sizing.stake_cents)
        return limits, book.mode, drafts

    def _pick(
        self,
        d: _Draft,
        snap: MarketSnapshot,
        asof: datetime,
        limits: RiskLimits,
        held: Mapping[int, set[PositionSide]],
    ) -> str | None:
        """Gates 8 and 9. The candidate is the passing side with the larger ``q - c`` (or, if
        neither passes, the larger ``q - c``); buying against a held side is refused (U3)."""
        book = OrderBook.from_json(snap.book, snap.ts, snap.venue_ts)
        schedule = schedule_for(d.cp.venue, asof, self._fee_schedules)
        sides: list[_Side] = []
        for quote in (d.cp.yes, d.cp.no):
            if isinstance(quote, Unavailable):
                continue
            q = clamp_fair(quote.fair, limits.fair_clamp_lo, limits.fair_clamp_hi)
            c = quote.avg_price + Decimal(quote.fee_cents) / (100 * quote.qty)
            levels = side_levels(book, quote.side)
            limit = limit_price(levels, q, limits.min_edge, schedule)
            depth, worst = _ZERO, _ZERO
            if limit is not None:
                depth = depth_at_or_better(levels, limit)
                worst = limit + marginal_fee(limit, schedule)
            passes = q - c >= limits.min_edge and limit is not None
            sides.append(_Side(PositionSide(quote.side), q, c, limit, depth, worst, passes))
        pick = max([x for x in sides if x.passes] or sides, key=lambda x: x.edge)
        if held.get(d.cid, set()) - {pick.side}:
            return "opposite_position"
        if not pick.passes:
            return "edge_below_min"
        d.pick = pick
        d.reasons.append(
            {
                "step": "gates",
                "side": pick.side.value,
                "q": str(pick.q),
                "c": str(pick.c),
                "limit": str(pick.limit),
                "depth": str(pick.depth),
            }
        )
        return None

    @staticmethod
    async def _review(officer: RiskOfficer | None, mode: BookMode, slate: list[_Draft]) -> None:
        """Phase B, outside any transaction: the officer can only veto or reduce."""
        if not slate:
            return
        if officer is None:
            if mode is BookMode.LIVE:
                _reject(slate, "officer_unavailable")
            else:
                for d in slate:
                    d.reasons.append({"step": "officer", "skipped": "replay"})
            return
        items = []
        for d in slate:
            assert d.pick is not None and d.stake_engine is not None
            items.append(
                SlateItem(
                    d.cid,
                    d.cp.venue,
                    d.pick.side,
                    d.qty,
                    d.stake_engine,
                    d.pick.q,
                    d.pick.c,
                    tuple(dict(r) for r in d.reasons),
                )
            )
        try:
            verdict = await officer.review(tuple(items))
            granted = [verdict[d.cid] for d in slate]
        except Exception as exc:  # any officer failure fails closed (spec 7.2)
            _reject(slate, "officer_failed", {"step": "officer", "error": type(exc).__name__})
            return
        if any(type(g) is not int or g < 0 for g in granted):
            _reject(slate, "officer_failed", {"step": "officer", "error": "invalid_qty"})
            return
        for d, g in zip(slate, granted, strict=True):
            assert d.pick is not None
            d.reasons.append({"step": "officer", "qty": str(g)})
            d.qty = min(d.qty, g)
            d.stake_officer = _stake(d.qty, d.pick.unit_cost)
            if d.qty < 1:
                d.rejected = "officer_veto"

    @staticmethod
    async def _recheck(
        s: AsyncSession, book: BookState, release_id: int, drafts: list[_Draft], limits: RiskLimits
    ) -> None:
        """Phase C: the side and the caps again, against holdings read under the book lock."""
        live = [d for d in drafts if d.rejected is None]
        held = await _held_sides(s, book.id, [d.cid for d in live])
        hold = _Holdings(await exposure(s, book.id), release_id)
        for d in live:
            assert d.pick is not None
            if held.get(d.cid, set()) - {d.pick.side}:
                d.rejected = "opposite_position"
                continue
            sizing = hold.size(d, book.equity_cents, limits)
            d.reasons.append(
                {"step": "recheck", "cents": str(sizing.stake_cents), "qty": str(sizing.qty)}
            )
            if sizing.rejected is not None:
                d.rejected = sizing.rejected
                continue
            d.qty = min(d.qty, sizing.qty)
            d.stake_final = _stake(d.qty, d.pick.unit_cost)
            hold.add(d, d.stake_final)

    @staticmethod
    def _row(
        d: _Draft,
        book_id: int,
        run_key: str,
        slot: Slot,
        pricing: ReleasePricing,
        limits: RiskLimits,
        now: datetime,
    ) -> RiskDecision:
        ok = d.rejected is None
        terminal: Reason = (
            {"step": "approved", "qty": str(d.qty), "cents": str(d.stake_final)}
            if ok
            else {"step": "rejected", "reason": str(d.rejected)}
        )
        pick = d.pick
        return RiskDecision(
            book_id=book_id,
            run_key=run_key,
            slot=slot,
            release_id=pricing.release_id,
            contract_id=d.cid,
            side=None if pick is None else pick.side,
            inputs_hash=pricing.inputs_hash,
            engine_version=RISK_ENGINE_VERSION,
            fair_yes=d.cp.fair_yes,
            fair_clamped=None if pick is None else pick.q,
            cost_per_contract=None if pick is None else pick.c,
            limit_price=None if pick is None else pick.limit,
            kelly_raw_cents=d.kelly_raw,
            stake_engine_cents=d.stake_engine,
            stake_officer_cents=d.stake_officer,
            stake_final_cents=d.stake_final if ok else None,
            qty=d.qty if ok else 0,
            verdict=Verdict.APPROVED if ok else Verdict.REJECTED,
            reasons=[*d.reasons, terminal],
            limits_version=limits.version,
            limits_snapshot=limits.snapshot(),
            decided_at=now,
        )

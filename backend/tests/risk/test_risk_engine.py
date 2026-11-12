"""RiskEngine: per-contract gates, the sizing funnel, the fail-closed officer port, the write
phase re-check, idempotency (D8) and the sealed ApprovedOrder (spec 7.2/7.3)."""

import dataclasses
import hashlib
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, cast

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from desk_fixtures import BASE_TS, RELEASE_AT, add_contract, add_snapshot, make_book, make_release
from prediction_desk.audit.log import AuditLog
from prediction_desk.clock import SimClock
from prediction_desk.db.models import (
    AuditEvent,
    BookState,
    EventContract,
    MarketSnapshot,
    Order,
    Position,
    Release,
    RiskDecision,
)
from prediction_desk.db.uow import unit_of_work
from prediction_desk.domain.enums import (
    BookMode,
    Comparator,
    MappingStatus,
    Measure,
    OrderStatus,
    PositionSide,
    Series,
    Slot,
    Venue,
    Verdict,
)
from prediction_desk.domain.types import to_prob
from prediction_desk.markets.terms import terms_from_row
from prediction_desk.pricing.coherence import CoherenceViolation
from prediction_desk.pricing.edge import EdgeQuote, Side, Unavailable, best_side
from prediction_desk.pricing.fees import FeeSchedule
from prediction_desk.pricing.release import ContractPricing, ReleasePricing
from prediction_desk.risk.engine import (
    RISK_ENGINE_VERSION,
    ApprovedOrder,
    EvaluationResult,
    RiskEngine,
    SlateItem,
    _approve,
)
from prediction_desk.risk.kill_switch import trip_kill_switch

D = Decimal
ASOF = BASE_TS
ALL_VENUES = frozenset({Venue.KALSHI, Venue.POLYMARKET})
ZERO_FEES = (
    FeeSchedule(Venue.KALSHI, date(2020, 1, 1), "zero", D(0)),
    FeeSchedule(Venue.POLYMARKET, date(2020, 1, 1), "zero", D(0)),
)
WORKED_ASKS = (("0.50", "400"), ("0.53", "600"))
INPUTS_HASH = "ab" * 32
APPROVED_STEPS = [
    "gates",
    "kelly",
    "cap_contract",
    "cap_event",
    "cap_portfolio",
    "liquidity",
    "round",
    "officer",
    "recheck",
    "approved",
]


# --- fake officers ----------------------------------------------------------------------------


class PassOfficer:
    """Grants ``grant(item)`` (default: the engine's qty); runs ``before`` first if given."""

    def __init__(
        self,
        grant: Callable[[SlateItem], int] | None = None,
        before: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.grant = grant
        self.before = before
        self.calls: list[tuple[SlateItem, ...]] = []

    async def review(self, slate: Sequence[SlateItem]) -> Mapping[int, int]:
        self.calls.append(tuple(slate))
        if self.before is not None:
            await self.before()
        return {
            item.contract_id: item.qty_engine if self.grant is None else self.grant(item)
            for item in slate
        }


class RaisingOfficer:
    async def review(self, slate: Sequence[SlateItem]) -> Mapping[int, int]:
        raise RuntimeError("model overloaded")


class ReturningOfficer:
    """Returns whatever it was given, well-formed or not."""

    def __init__(self, make: Callable[[Sequence[SlateItem]], object]) -> None:
        self.make = make

    async def review(self, slate: Sequence[SlateItem]) -> Mapping[int, int]:
        return cast(Mapping[int, int], self.make(slate))


# --- seeding and pricing helpers --------------------------------------------------------------


def _quote(side: Side, fair: str, avg: str, *, fee: int = 0, qty: int = 10) -> EdgeQuote:
    f, a = to_prob(fair), D(avg)
    return EdgeQuote(side, f, a, a, fee, 0, f - a - D(fee) / (100 * qty), qty)


def _priced(
    contract: EventContract,
    snap: MarketSnapshot | None,
    fair_yes: str,
    yes: EdgeQuote | Unavailable,
    no: EdgeQuote | Unavailable,
) -> ContractPricing:
    return ContractPricing(
        contract_id=contract.id,
        venue=contract.venue,
        terms=terms_from_row(contract),
        fair_yes=to_prob(fair_yes),
        snapshot_id=None if snap is None else snap.id,
        quote_age_s=None if snap is None else 10,
        quote=best_side((yes, no)),
        yes=yes,
        no=no,
    )


def _release_pricing(
    release: Release,
    contracts: Sequence[ContractPricing],
    coherence: tuple[CoherenceViolation, ...] = (),
) -> ReleasePricing:
    return ReleasePricing(
        release_id=release.id,
        asof=ASOF,
        model_version="test:v1",
        distributions={},
        contracts=tuple(sorted(contracts, key=lambda c: c.contract_id)),
        unpriceable={},
        rejected={},
        coherence=coherence,
        inputs_hash=INPUTS_HASH,
    )


async def _contract(
    session: AsyncSession,
    release: Release,
    *,
    venue: Venue = Venue.KALSHI,
    strike: str = "0.3",
    event: str = "ev-1",
    confidence: str = "0.95",
) -> EventContract:
    return await add_contract(
        session,
        release,
        venue=venue,
        measure=Measure.CPI_MOM,
        comparator=Comparator.GT,
        strike_lo=D(strike),
        venue_event_id=event,
        confidence=D(confidence),
    )


async def _worked_contract(
    session: AsyncSession,
    release: Release,
    *,
    fair: str = "0.60",
    asks: Sequence[tuple[str, str]] = WORKED_ASKS,
    **kwargs: Any,
) -> ContractPricing:
    """The Task 4 worked example: YES fair 0.60, probe avg 0.50, asks 0.50x400 / 0.53x600."""
    contract = await _contract(session, release, **kwargs)
    snap = await add_snapshot(session, contract, ASOF - timedelta(seconds=10), [], asks)
    return _priced(contract, snap, fair, _quote("yes", fair, asks[0][0]), Unavailable("empty_side"))


async def _position(
    session: AsyncSession,
    book: BookState,
    contract_id: int,
    *,
    side: PositionSide = PositionSide.YES,
    cost: int = 500,
) -> None:
    session.add(
        Position(
            book_id=book.id,
            contract_id=contract_id,
            side=side,
            qty=10,
            cost_cents=cost,
            settled=False,
            updated_at=BASE_TS,
        )
    )
    await session.commit()


async def _evaluate(
    sessions: async_sessionmaker[AsyncSession],
    book: BookState,
    pricing: ReleasePricing,
    *,
    officer: Any = None,
    clock: SimClock | None = None,
    healthy: frozenset[Venue] = ALL_VENUES,
    run_key: str = "run-1",
    slot: Slot = Slot.T24H,
) -> EvaluationResult:
    clock = clock or SimClock(ASOF)
    engine = RiskEngine(clock, AuditLog(clock), fee_schedules=ZERO_FEES)
    return await engine.evaluate(
        sessions,
        book_id=book.id,
        run_key=run_key,
        slot=slot,
        pricing=pricing,
        healthy_venues=healthy,
        officer=officer,
    )


async def _decisions(sessions: async_sessionmaker[AsyncSession]) -> list[RiskDecision]:
    async with sessions() as s:
        return list(
            (await s.scalars(select(RiskDecision).order_by(RiskDecision.contract_id))).all()
        )


def _why(decision: RiskDecision) -> str:
    assert decision.verdict is Verdict.REJECTED
    last = decision.reasons[-1]
    assert last["step"] == "rejected"
    return str(last["reason"])


async def _audit_actions(sessions: async_sessionmaker[AsyncSession]) -> list[str]:
    async with sessions() as s:
        return list((await s.scalars(select(AuditEvent.action).order_by(AuditEvent.seq))).all())


async def _count(sessions: async_sessionmaker[AsyncSession], model: Any) -> int:
    async with sessions() as s:
        return (await s.execute(select(func.count()).select_from(model))).scalar_one()


def _coid(book_id: int, run_key: str, slot: Slot, contract_id: int) -> str:
    return hashlib.sha256(f"{book_id}|{run_key}|{slot.value}|{contract_id}".encode()).hexdigest()


# --- the worked example -----------------------------------------------------------------------


async def test_approves_and_sizes_worked_example(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    release = await make_release(session)
    cp = await _worked_contract(session, release)
    book = await make_book(session)
    officer = PassOfficer()

    result = await _evaluate(sessions, book, _release_pricing(release, [cp]), officer=officer)

    assert len(result.approved) == 1
    order = result.approved[0]
    assert order.qty == 250  # liquidity binds: floor(0.25 * 1000)
    assert order.side is PositionSide.YES
    assert order.limit_price == D("0.53")
    assert order.contract_id == cp.contract_id
    assert order.book_id == book.id
    assert order.decided_at == ASOF
    assert order.client_order_id == _coid(book.id, "run-1", Slot.T24H, cp.contract_id)
    assert result.killed is False

    [decision] = await _decisions(sessions)
    assert result.decision_ids == (decision.id,)
    assert order.risk_decision_id == decision.id
    assert decision.verdict is Verdict.APPROVED
    assert [r["step"] for r in decision.reasons] == APPROVED_STEPS
    assert decision.qty == 250
    assert decision.kelly_raw_cents == 50_000  # 0.25 * 0.2 * 1_000_000
    assert decision.stake_engine_cents == 13_250  # R5: 250 * 0.53 worst cost
    assert decision.stake_officer_cents == 13_250
    assert decision.stake_final_cents == 13_250
    assert decision.side is PositionSide.YES
    assert decision.fair_yes == D("0.6")
    assert decision.fair_clamped == D("0.6")
    assert decision.cost_per_contract == D("0.5")
    assert decision.limit_price == D("0.53")
    assert decision.engine_version == RISK_ENGINE_VERSION == "risk:v1"
    assert decision.limits_version == 1
    assert decision.limits_snapshot["min_edge"] == "0.04"
    assert decision.inputs_hash == INPUTS_HASH
    assert (decision.run_key, decision.slot, decision.release_id) == (
        "run-1",
        Slot.T24H,
        release.id,
    )

    [slate] = officer.calls
    [item] = slate
    assert (item.contract_id, item.qty_engine, item.stake_engine_cents) == (
        cp.contract_id,
        250,
        13_250,
    )
    assert (item.venue, item.side) == (Venue.KALSHI, PositionSide.YES)
    assert (item.fair_clamped, item.cost_per_contract) == (D("0.6"), D("0.5"))
    assert item.reasons[0]["step"] == "gates"

    assert await _audit_actions(sessions) == ["risk.decided"]
    async with sessions() as s:
        event = (await s.scalars(select(AuditEvent))).one()
    assert event.payload["run_key"] == "run-1"
    assert event.payload["slot"] == "t-24h"
    assert event.payload["decisions"] == [
        {
            "id": decision.id,
            "contract_id": cp.contract_id,
            "verdict": "approved",
            "qty": 250,
            "reason": None,
        }
    ]


# --- the nine gates ---------------------------------------------------------------------------


@dataclass
class Scenario:
    book: BookState
    contract_id: int
    pricing: ReleasePricing
    clock: SimClock


async def _scenario(
    session: AsyncSession,
    *,
    kill: bool = False,
    clock_at: datetime = ASOF,
    release_at: datetime = RELEASE_AT,
    confidence: str = "0.95",
    remap: bool = False,
    venue: Venue = Venue.KALSHI,
    fair: str = "0.60",
    snapshot: bool = True,
    incomplete: bool = False,
    held: PositionSide | None = None,
    mode: BookMode = BookMode.REPLAY,
) -> Scenario:
    """The worked example with exactly one thing changed."""
    release = await make_release(session, scheduled_at=release_at)
    if snapshot:
        cp = await _worked_contract(session, release, fair=fair, venue=venue, confidence=confidence)
    else:
        contract = await _contract(session, release, venue=venue, confidence=confidence)
        gap = Unavailable("no_snapshot")
        cp = _priced(contract, None, fair, gap, gap)
    if remap:
        other = await make_release(session, period="2026-11")
        row = await session.get(EventContract, cp.contract_id)
        assert row is not None
        row.release_id = other.id
        await session.commit()
    book = await make_book(session, mode=mode)
    if kill:
        book.kill_switch_on = True
        await session.commit()
    if held is not None:
        await _position(session, book, cp.contract_id, side=held)
    coherence = (
        (CoherenceViolation("incomplete_bracket", "ev-1", "incomplete bracket (1 of 2 mapped)"),)
        if incomplete
        else ()
    )
    return Scenario(
        book, cp.contract_id, _release_pricing(release, [cp], coherence), SimClock(clock_at)
    )


GATE_CASES: list[tuple[str, dict[str, Any], frozenset[Venue]]] = [
    ("kill_switch_on", {"kill": True}, ALL_VENUES),
    ("stale_pricing", {"clock_at": ASOF + timedelta(seconds=61)}, ALL_VENUES),
    ("past_cutoff", {"release_at": ASOF + timedelta(minutes=5)}, ALL_VENUES),
    ("low_mapping_confidence", {"confidence": "0.89"}, ALL_VENUES),
    ("low_mapping_confidence", {"remap": True}, ALL_VENUES),
    ("venue_unhealthy", {}, frozenset({Venue.POLYMARKET})),
    ("no_quote:no_snapshot", {"snapshot": False}, ALL_VENUES),
    ("event_incoherent", {"venue": Venue.POLYMARKET, "incomplete": True}, ALL_VENUES),
    ("opposite_position", {"held": PositionSide.NO}, ALL_VENUES),
    ("edge_below_min", {"fair": "0.53"}, ALL_VENUES),
]


@pytest.mark.parametrize(
    ("reason", "kwargs", "healthy"),
    GATE_CASES,
    ids=[f"{r}-{i}" for i, (r, _, _) in enumerate(GATE_CASES)],
)
async def test_each_gate_rejects_with_its_reason(
    session: AsyncSession,
    sessions: async_sessionmaker[AsyncSession],
    reason: str,
    kwargs: dict[str, Any],
    healthy: frozenset[Venue],
) -> None:
    sc = await _scenario(session, **kwargs)
    officer = PassOfficer()

    result = await _evaluate(
        sessions, sc.book, sc.pricing, officer=officer, clock=sc.clock, healthy=healthy
    )

    assert result.approved == ()
    [decision] = await _decisions(sessions)
    assert _why(decision) == reason
    assert decision.qty == 0
    assert decision.stake_final_cents is None
    assert officer.calls == []  # nothing survived, so nothing to review
    assert result.killed is (reason == "kill_switch_on")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"clock_at": ASOF + timedelta(seconds=60)},  # exactly max_quote_age_s old
        {"release_at": ASOF + timedelta(minutes=5, microseconds=1)},  # just before the cutoff
        {"confidence": "0.90"},  # exactly min_mapping_confidence
        {"fair": "0.54"},  # q - c exactly min_edge, at the 0.50 level
        {"venue": Venue.POLYMARKET, "incomplete": False},
        {"held": PositionSide.YES},  # a same-side top-up is allowed (U3)
    ],
)
async def test_gate_boundaries_still_approve(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession], kwargs: dict[str, Any]
) -> None:
    sc = await _scenario(session, **kwargs)
    result = await _evaluate(sessions, sc.book, sc.pricing, officer=PassOfficer(), clock=sc.clock)
    assert len(result.approved) == 1


async def test_no_quote_uses_the_unavailable_reason(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    release = await make_release(session)
    contract = await _contract(session, release)
    snap = await add_snapshot(session, contract, ASOF, [], [])
    gap = Unavailable("empty_side")
    pricing = _release_pricing(release, [_priced(contract, snap, "0.60", gap, gap)])
    book = await make_book(session)

    await _evaluate(sessions, book, pricing, officer=PassOfficer())

    [decision] = await _decisions(sessions)
    assert _why(decision) == "no_quote:empty_side"


async def test_r14_clamp_prevents_cap_size_on_zero_prob_outcome(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    # R3: NO asks at 0.95, zero fee. Unclamped q=1.00 gives edge 0.05 >= 0.04 (would trade at
    # cap size); clamped q=0.98 gives 0.03 < 0.04.
    release = await make_release(session, series=Series.FOMC, period="2026-12")
    contract = await add_contract(
        session,
        release,
        venue=Venue.KALSHI,
        measure=Measure.FOMC_CHANGE_BP,
        comparator=Comparator.GT,
        strike_lo=D("0"),
    )
    # YES bids at 0.05 are NO asks at 0.95.
    snap = await add_snapshot(session, contract, ASOF, [("0.05", "1000")], [("0.06", "1000")])
    cp = _priced(contract, snap, "0.00000", _quote("yes", "0", "0.06"), _quote("no", "1", "0.95"))
    assert isinstance(cp.quote, EdgeQuote) and cp.quote.side == "no"
    book = await make_book(session)

    result = await _evaluate(sessions, book, _release_pricing(release, [cp]), officer=PassOfficer())

    assert result.approved == ()
    [decision] = await _decisions(sessions)
    assert _why(decision) == "edge_below_min"


# --- coherence (U2, D10) ----------------------------------------------------------------------


async def test_bug_violation_trips_kill_switch_and_rejects_all(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    release = await make_release(session)
    good = await _worked_contract(session, release, strike="0.3")
    other = await _worked_contract(session, release, strike="0.4")
    book = await make_book(session)
    bug = CoherenceViolation("monotonic", "kalshi", "P(>0.4) > P(>0.3)")
    officer = PassOfficer()

    result = await _evaluate(
        sessions, book, _release_pricing(release, [good, other], (bug,)), officer=officer
    )

    assert result.approved == ()
    assert result.killed is True
    assert officer.calls == []
    decisions = await _decisions(sessions)
    assert [_why(d) for d in decisions] == ["pricing_sanity_violation"] * 2
    async with sessions() as s:
        row = await s.get_one(BookState, book.id)
    assert row.kill_switch_on is True
    assert row.kill_reason == "pricing_sanity: P(>0.4) > P(>0.3)"
    assert row.kill_actor == "system:risk"
    assert await _audit_actions(sessions) == ["kill_switch.tripped", "risk.decided"]


async def test_incomplete_bracket_blocks_only_that_event(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    # Both contracts share the venue_event_id text; only the Polymarket one is the bracket.
    release = await make_release(session)
    kalshi = await _worked_contract(session, release, venue=Venue.KALSHI, event="pm-ev")
    poly = await _worked_contract(session, release, venue=Venue.POLYMARKET, event="pm-ev")
    incomplete = CoherenceViolation("incomplete_bracket", "pm-ev", "incomplete bracket (1 of 3)")
    book = await make_book(session)

    result = await _evaluate(
        sessions,
        book,
        _release_pricing(release, [kalshi, poly], (incomplete,)),
        officer=PassOfficer(),
    )

    assert [a.contract_id for a in result.approved] == [kalshi.contract_id]
    by_id = {d.contract_id: d for d in await _decisions(sessions)}
    assert by_id[kalshi.contract_id].verdict is Verdict.APPROVED
    assert _why(by_id[poly.contract_id]) == "event_incoherent"
    assert result.killed is False


# --- the officer port -------------------------------------------------------------------------


async def test_live_book_without_officer_rejects_all(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    sc = await _scenario(session, mode=BookMode.LIVE)
    result = await _evaluate(sessions, sc.book, sc.pricing, officer=None)
    assert result.approved == ()
    [decision] = await _decisions(sessions)
    assert _why(decision) == "officer_unavailable"


async def test_replay_book_without_officer_passes_engine_stake(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    sc = await _scenario(session, mode=BookMode.REPLAY)
    result = await _evaluate(sessions, sc.book, sc.pricing, officer=None)
    assert [a.qty for a in result.approved] == [250]
    [decision] = await _decisions(sessions)
    assert {"step": "officer", "skipped": "replay"} in decision.reasons
    assert decision.stake_officer_cents is None
    assert decision.stake_final_cents == 13_250


@pytest.mark.parametrize(
    "officer",
    [
        RaisingOfficer(),
        ReturningOfficer(lambda slate: {}),  # omits a slate id
        ReturningOfficer(lambda slate: {i.contract_id: -1 for i in slate}),
        ReturningOfficer(lambda slate: {i.contract_id: True for i in slate}),
        ReturningOfficer(lambda slate: {i.contract_id: "5" for i in slate}),
        ReturningOfficer(lambda slate: {i.contract_id: 5.0 for i in slate}),
        ReturningOfficer(lambda slate: None),
    ],
    ids=["raises", "missing_id", "negative", "bool", "str", "float", "not_a_mapping"],
)
async def test_officer_exception_or_missing_id_fails_closed(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession], officer: Any
) -> None:
    release = await make_release(session)
    a = await _worked_contract(session, release, strike="0.3")
    b = await _worked_contract(session, release, strike="0.4")
    book = await make_book(session)

    result = await _evaluate(sessions, book, _release_pricing(release, [a, b]), officer=officer)

    assert result.approved == ()
    decisions = await _decisions(sessions)
    assert [_why(d) for d in decisions] == ["officer_failed"] * 2
    assert all(d.stake_final_cents is None for d in decisions)


@pytest.mark.parametrize(
    ("granted", "qty", "stake"),
    [(1_000_000, 250, 13_250), (250, 250, 13_250), (100, 100, 5_300)],
)
async def test_officer_can_only_reduce(
    session: AsyncSession,
    sessions: async_sessionmaker[AsyncSession],
    granted: int,
    qty: int,
    stake: int,
) -> None:
    sc = await _scenario(session)
    result = await _evaluate(
        sessions, sc.book, sc.pricing, officer=PassOfficer(grant=lambda _: granted)
    )
    assert [a.qty for a in result.approved] == [qty]
    [decision] = await _decisions(sessions)
    assert decision.stake_engine_cents == 13_250
    assert decision.stake_officer_cents == stake
    assert decision.stake_final_cents == stake
    assert decision.qty == qty


async def test_officer_zero_is_a_veto(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    sc = await _scenario(session)
    result = await _evaluate(sessions, sc.book, sc.pricing, officer=PassOfficer(grant=lambda _: 0))
    assert result.approved == ()
    [decision] = await _decisions(sessions)
    assert _why(decision) == "officer_veto"
    assert decision.stake_officer_cents == 0
    assert decision.side is PositionSide.YES  # the inputs that were vetoed are kept


async def test_no_connection_held_during_officer_review(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession], engine: AsyncEngine
) -> None:
    sc = await _scenario(session)
    await session.close()
    seen: list[int] = []

    async def check() -> None:
        seen.append(engine.pool.checkedout())  # type: ignore[attr-defined]

    await _evaluate(sessions, sc.book, sc.pricing, officer=PassOfficer(before=check))
    assert seen == [0]


# --- the write phase --------------------------------------------------------------------------


async def test_caps_recomputed_in_write_phase(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    sc = await _scenario(session)

    async def fill_elsewhere() -> None:
        # A same-side fill lands between Phase A and Phase C: 10_000 of the 20_000 cap used.
        await _position(session, sc.book, sc.contract_id, cost=10_000)

    result = await _evaluate(
        sessions, sc.book, sc.pricing, officer=PassOfficer(before=fill_elsewhere)
    )

    assert [a.qty for a in result.approved] == [188]  # floor(10_000 / 53)
    [decision] = await _decisions(sessions)
    assert decision.stake_engine_cents == 13_250
    assert decision.stake_final_cents == 9_964  # ceil(188 * 53)
    assert {"step": "recheck", "cents": "9964", "qty": "188"} in decision.reasons


async def test_write_phase_rejects_when_cap_used_up(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    sc = await _scenario(session)

    async def fill_elsewhere() -> None:
        await _position(session, sc.book, sc.contract_id, cost=20_000)

    result = await _evaluate(
        sessions, sc.book, sc.pricing, officer=PassOfficer(before=fill_elsewhere)
    )
    assert result.approved == ()
    [decision] = await _decisions(sessions)
    assert _why(decision) == "below_min_size"


async def test_write_phase_rechecks_opposite_position(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    sc = await _scenario(session)

    async def fill_opposite() -> None:
        await _position(session, sc.book, sc.contract_id, side=PositionSide.NO)

    result = await _evaluate(
        sessions, sc.book, sc.pricing, officer=PassOfficer(before=fill_opposite)
    )
    assert result.approved == ()
    [decision] = await _decisions(sessions)
    assert _why(decision) == "opposite_position"


async def test_kill_switch_tripped_during_review_rejects_all(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    sc = await _scenario(session)

    async def trip() -> None:
        async with unit_of_work(sessions) as s:
            await trip_kill_switch(
                s, AuditLog(sc.clock), sc.clock, sc.book.id, reason="manual", actor="admin:j"
            )

    result = await _evaluate(sessions, sc.book, sc.pricing, officer=PassOfficer(before=trip))

    assert result.approved == ()
    assert result.killed is True
    [decision] = await _decisions(sessions)
    assert _why(decision) == "kill_switch_on"


async def test_pending_approval_counts_as_held_side(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    # An earlier slot approved NO and has no order yet; this slot's YES is the opposite side.
    sc = await _scenario(session)
    async with sessions() as s:
        s.add(
            RiskDecision(
                book_id=sc.book.id,
                run_key="run-0",
                slot=Slot.T24H,
                release_id=sc.pricing.release_id,
                contract_id=sc.contract_id,
                side=PositionSide.NO,
                inputs_hash=INPUTS_HASH,
                engine_version=RISK_ENGINE_VERSION,
                fair_yes=D("0.4"),
                limit_price=D("0.5"),
                stake_final_cents=500,
                qty=10,
                verdict=Verdict.APPROVED,
                reasons=[],
                limits_version=1,
                limits_snapshot={},
                decided_at=BASE_TS,
            )
        )
        await s.commit()

    result = await _evaluate(sessions, sc.book, sc.pricing, officer=PassOfficer())
    assert result.approved == ()
    decision = next(d for d in await _decisions(sessions) if d.run_key == "run-1")
    assert _why(decision) == "opposite_position"


async def test_exposure_without_known_release_fails_closed(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    sc = await _scenario(session)
    other = await make_release(session, period="2026-11")
    stray = await _contract(session, other, strike="0.9")
    await _position(session, sc.book, stray.id)
    stray.release_id = None
    stray.mapping_status = MappingStatus.UNMAPPED
    stray.measure = stray.comparator = stray.strike_lo = None
    stray.unit = stray.resolution_precision = stray.resolution_source = None
    await session.commit()

    with pytest.raises(RuntimeError, match="no release known"):
        await _evaluate(sessions, sc.book, sc.pricing, officer=PassOfficer())
    assert await _count(sessions, RiskDecision) == 0


# --- idempotency (D8, Review Focus 3) ---------------------------------------------------------


async def test_rerun_same_run_slot_is_idempotent(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    sc = await _scenario(session)
    officer = PassOfficer()

    first = await _evaluate(sessions, sc.book, sc.pricing, officer=officer)
    second = await _evaluate(sessions, sc.book, sc.pricing, officer=officer)

    assert second == first
    assert len(second.approved) == 1
    assert len(officer.calls) == 1
    assert await _count(sessions, RiskDecision) == 1
    assert await _audit_actions(sessions) == ["risk.decided"]


async def test_other_slot_is_a_new_decision(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    sc = await _scenario(session)
    first = await _evaluate(sessions, sc.book, sc.pricing, officer=PassOfficer())
    later = await _evaluate(sessions, sc.book, sc.pricing, officer=PassOfficer(), slot=Slot.T2H)
    assert later.decision_ids != first.decision_ids
    assert later.approved[0].client_order_id != first.approved[0].client_order_id
    # The first approval has no order yet, so it counts as exposure: 20_000 - 13_250 left.
    assert later.approved[0].qty == 127  # floor(6_750 / 53)


async def test_rerun_after_order_or_kill_approves_nothing(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    sc = await _scenario(session)
    first = await _evaluate(sessions, sc.book, sc.pricing, officer=PassOfficer())
    [approved] = first.approved
    async with sessions() as s:
        s.add(
            Order(
                client_order_id=approved.client_order_id,
                risk_decision_id=approved.risk_decision_id,
                book_id=approved.book_id,
                contract_id=approved.contract_id,
                side=approved.side,
                limit_price=approved.limit_price,
                qty=approved.qty,
                filled_qty=0,
                status=OrderStatus.CANCELLED,
                created_at=BASE_TS,
            )
        )
        await s.commit()

    after_order = await _evaluate(sessions, sc.book, sc.pricing, officer=PassOfficer())
    assert after_order.decision_ids == first.decision_ids
    assert after_order.approved == ()
    assert after_order.killed is False


async def test_rerun_while_killed_approves_nothing(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    sc = await _scenario(session)
    first = await _evaluate(sessions, sc.book, sc.pricing, officer=PassOfficer())
    assert len(first.approved) == 1
    sc.book.kill_switch_on = True
    await session.commit()

    again = await _evaluate(sessions, sc.book, sc.pricing, officer=PassOfficer())
    assert again.decision_ids == first.decision_ids
    assert again.approved == ()
    assert again.killed is True


async def test_concurrent_same_run_slot_writes_once(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    sc = await _scenario(session)
    inner: list[EvaluationResult] = []

    async def race() -> None:
        # Another evaluate of the same (run, slot) commits while this one is in review.
        inner.append(await _evaluate(sessions, sc.book, sc.pricing, officer=PassOfficer()))

    outer = await _evaluate(sessions, sc.book, sc.pricing, officer=PassOfficer(before=race))

    assert outer == inner[0]
    assert await _count(sessions, RiskDecision) == 1
    assert await _audit_actions(sessions) == ["risk.decided"]


# --- holdings and the slate (U3, D1) ----------------------------------------------------------


async def test_top_up_counts_held_position(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    sc = await _scenario(session)
    await _position(session, sc.book, sc.contract_id, cost=15_000)

    result = await _evaluate(sessions, sc.book, sc.pricing, officer=PassOfficer())

    [order] = result.approved
    assert order.side is PositionSide.YES
    assert order.qty == 94  # floor((20_000 - 15_000) / 53)
    [decision] = await _decisions(sessions)
    assert decision.stake_final_cents == 4_982  # ceil(94 * 53)


async def test_slate_accumulates_exposure_within_run(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    release = await make_release(session)
    deep = (("0.50", "100000"),)
    cps = [
        await _worked_contract(session, release, strike=s, asks=deep) for s in ("0.1", "0.2", "0.3")
    ]
    book = await make_book(session)
    officer = PassOfficer()

    result = await _evaluate(sessions, book, _release_pricing(release, cps), officer=officer)

    # Each alone is worth the full 2% (400 @ 0.50 = 20_000); the 5% event cap leaves 10_000.
    assert [a.qty for a in result.approved] == [400, 400, 200]
    assert [i.qty_engine for i in officer.calls[0]] == [400, 400, 200]
    decisions = await _decisions(sessions)
    total = sum(d.stake_final_cents or 0 for d in decisions)
    assert total == 50_000 <= book.equity_cents * 5 // 100


async def test_slate_third_contract_rejected_when_event_cap_used(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    release = await make_release(session)
    deep = (("0.50", "100000"),)
    cps = [
        await _worked_contract(session, release, strike=s, asks=deep) for s in ("0.1", "0.2", "0.3")
    ]
    book = await make_book(session)
    # 10_000 already held elsewhere on this release: only 40_000 of event headroom remains.
    held = await _contract(session, release, strike="0.9")
    await _position(session, book, held.id, cost=10_000)

    result = await _evaluate(sessions, book, _release_pricing(release, cps), officer=PassOfficer())

    assert [a.qty for a in result.approved] == [400, 400]
    decisions = await _decisions(sessions)
    assert _why(decisions[2]) == "below_min_size"


# --- the seal and the arguments ---------------------------------------------------------------


async def test_approved_order_cannot_be_constructed_outside_engine(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    fields = (1, "c" * 64, 1, 1, PositionSide.YES, D("0.5"), 10, ASOF)
    with pytest.raises(TypeError):
        ApprovedOrder(*fields)
    with pytest.raises(TypeError):
        ApprovedOrder(*fields, _seal=object())

    sc = await _scenario(session)
    [approved] = (await _evaluate(sessions, sc.book, sc.pricing, officer=PassOfficer())).approved
    with pytest.raises(TypeError):
        dataclasses.replace(approved, qty=1_000_000)


async def test_approve_seam_builds_from_a_stored_decision(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    sc = await _scenario(session)
    await _evaluate(sessions, sc.book, sc.pricing, officer=PassOfficer())
    [decision] = await _decisions(sessions)

    order = _approve(decision)

    assert order.client_order_id == _coid(sc.book.id, "run-1", Slot.T24H, sc.contract_id)
    assert (order.risk_decision_id, order.qty, order.limit_price) == (
        decision.id,
        250,
        D("0.53"),
    )


async def test_approve_seam_refuses_a_rejected_decision(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    sc = await _scenario(session, fair="0.53")
    await _evaluate(sessions, sc.book, sc.pricing, officer=PassOfficer())
    [decision] = await _decisions(sessions)
    with pytest.raises(ValueError, match="not approved"):
        _approve(decision)


async def test_empty_run_key_is_refused(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    sc = await _scenario(session)
    with pytest.raises(ValueError, match="run_key"):
        await _evaluate(sessions, sc.book, sc.pricing, officer=PassOfficer(), run_key="")


async def test_missing_book_raises(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    sc = await _scenario(session)
    with pytest.raises(NoResultFound):
        await _evaluate(sessions, BookState(id=999_999), sc.pricing, officer=PassOfficer())

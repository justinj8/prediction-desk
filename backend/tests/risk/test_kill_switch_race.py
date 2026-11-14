"""Concurrency (D2): the kill-switch trip waits for an in-flight decision, and two concurrent
evaluations on different releases cannot jointly breach a cap."""

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from desk_fixtures import BASE_TS, add_contract, add_snapshot, make_book, make_release
from prediction_desk.audit.log import AuditLog
from prediction_desk.clock import SimClock
from prediction_desk.db.models import BookState, EventContract, RiskDecision, RiskLimitsRow
from prediction_desk.db.uow import unit_of_work
from prediction_desk.domain.enums import (
    Comparator,
    Measure,
    Slot,
    Venue,
    Verdict,
)
from prediction_desk.domain.types import to_prob
from prediction_desk.markets.terms import terms_from_row
from prediction_desk.pricing.edge import EdgeQuote, Unavailable, best_side
from prediction_desk.pricing.release import ContractPricing, ReleasePricing
from prediction_desk.risk.book import lock_book
from prediction_desk.risk.engine import RiskEngine
from prediction_desk.risk.kill_switch import trip_kill_switch
from prediction_desk.risk.limits import DEFAULT_LIMITS

D = Decimal
ASOF = BASE_TS
ALL_VENUES = frozenset({Venue.KALSHI, Venue.POLYMARKET})


def _yes_quote(fair: str, avg: str, qty: int = 10) -> EdgeQuote:
    f, a = to_prob(fair), D(avg)
    return EdgeQuote("yes", f, a, a, 0, 0, f - a, qty)


async def _pricing(
    session: AsyncSession, *, event: str, period: str, fair: str, ask: str, depth: str
) -> tuple[ReleasePricing, EventContract]:
    """One release with one Kalshi GT contract and a deep YES ask, priced directly."""
    release = await make_release(session, period=period)
    contract = await add_contract(
        session,
        release,
        venue=Venue.KALSHI,
        measure=Measure.CPI_MOM,
        comparator=Comparator.GT,
        strike_lo=D("0.3"),
        venue_event_id=event,
    )
    snap = await add_snapshot(session, contract, ASOF - timedelta(seconds=10), [], [(ask, depth)])
    yes = _yes_quote(fair, ask)
    cp = ContractPricing(
        contract_id=contract.id,
        venue=contract.venue,
        terms=terms_from_row(contract),
        fair_yes=to_prob(fair),
        snapshot_id=snap.id,
        quote_age_s=10,
        quote=best_side((yes, Unavailable("empty_side"))),
        yes=yes,
        no=Unavailable("empty_side"),
    )
    pricing = ReleasePricing(
        release_id=release.id,
        asof=ASOF,
        model_version="test:v1",
        distributions={},
        contracts=(cp,),
        unpriceable={},
        rejected={},
        coherence=(),
        inputs_hash="cd" * 32,
    )
    return pricing, contract


def _engine() -> tuple[RiskEngine, SimClock, AuditLog]:
    clock = SimClock(ASOF)
    audit = AuditLog(clock)
    return RiskEngine(clock, audit), clock, audit


async def test_trip_waits_for_in_flight_decision(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    pricing, _ = await _pricing(
        session, event="ev-a", period="2026-10", fair="0.98", ask="0.50", depth="1000"
    )
    book = await make_book(session)
    engine, clock, audit = _engine()

    async def trip() -> bool:
        async with unit_of_work(sessions) as s:
            return await trip_kill_switch(
                s, audit, clock, book.id, reason="manual halt", actor="system:test"
            )

    async with sessions() as a:  # an in-flight decision holds the book lock
        await lock_book(a, book.id)
        task = asyncio.create_task(trip())
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(task), 0.3)
        assert not task.done()
        await a.commit()  # the decision transaction ends; the trip may proceed
    assert await asyncio.wait_for(task, 5) is True

    async with sessions() as s:
        row = await s.get_one(BookState, book.id, populate_existing=True)
        assert row.kill_switch_on is True

    result = await engine.evaluate(
        sessions,
        book_id=book.id,
        run_key="after-trip",
        slot=Slot.T24H,
        pricing=pricing,
        healthy_venues=ALL_VENUES,
        officer=None,
    )
    assert result.killed is True
    assert result.approved == ()
    async with sessions() as s:
        decisions = (await s.scalars(select(RiskDecision))).all()
    assert len(decisions) == 1
    assert decisions[0].verdict is Verdict.REJECTED
    assert decisions[0].reasons[-1]["reason"] == "kill_switch_on"


async def test_concurrent_evaluations_cannot_jointly_breach_cap(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    # Limits v2: every cap at 15%, all else default.
    caps = {**DEFAULT_LIMITS, "cap_contract": "0.15", "cap_event": "0.15", "cap_portfolio": "0.15"}
    await session.execute(update(RiskLimitsRow).values(active=False))
    session.add(
        RiskLimitsRow(
            version=2,
            params=caps,
            created_by="test",
            created_at=datetime(2026, 11, 12, tzinfo=UTC),
            active=True,
        )
    )
    await session.commit()

    # Fair 0.98 vs cost 0.50: f* = 0.96, quarter Kelly = 24% of equity, so each evaluation
    # alone would size at the full 15% headroom (deep books keep liquidity from binding).
    p1, _ = await _pricing(
        session, event="ev-1", period="2026-10", fair="0.98", ask="0.50", depth="40000"
    )
    p2, _ = await _pricing(
        session, event="ev-2", period="2026-11", fair="0.98", ask="0.50", depth="40000"
    )
    book = await make_book(session)
    engine, _, _ = _engine()

    results = await asyncio.gather(
        *(
            engine.evaluate(
                sessions,
                book_id=book.id,
                run_key=f"run-{i}",
                slot=Slot.T24H,
                pricing=p,
                healthy_venues=ALL_VENUES,
                officer=None,
            )
            for i, p in enumerate((p1, p2), start=1)
        )
    )

    budget = book.equity_cents * 15 // 100  # 150_000
    async with sessions() as s:
        decisions = list((await s.scalars(select(RiskDecision))).all())
    assert len(decisions) == 2
    assert all(d.limits_version == 2 for d in decisions)
    stakes = sorted(d.stake_final_cents or 0 for d in decisions)
    approved_total = sum(
        d.stake_final_cents or 0 for d in decisions if d.verdict is Verdict.APPROVED
    )
    assert sum(len(r.approved) for r in results) >= 1
    assert 0 < approved_total <= budget + 2
    assert stakes[0] < stakes[1]  # one evaluation was downsized: serialised, not just small
    assert stakes[1] >= budget - 100  # the other really sized at the full 15% headroom

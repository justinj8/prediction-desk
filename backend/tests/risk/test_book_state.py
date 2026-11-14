from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from desk_fixtures import BASE_TS, add_contract, add_snapshot, make_book, make_release
from prediction_desk.audit.log import AuditLog
from prediction_desk.clock import SimClock
from prediction_desk.db.models import (
    AuditEvent,
    BookState,
    EventContract,
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
    Slot,
    Venue,
    Verdict,
)
from prediction_desk.risk.book import Exposure, create_book, exposure, lock_book, mark_book

D = Decimal


async def _contract(
    session: AsyncSession, release: Release, strike: str = "0.3", event: str = "ev-1"
) -> EventContract:
    return await add_contract(
        session,
        release,
        venue=Venue.KALSHI,
        measure=Measure.CPI_MOM,
        comparator=Comparator.GTE,
        strike_lo=D(strike),
        venue_event_id=event,
    )


async def _unmap(session: AsyncSession, contract_id: int) -> None:
    """Mirror what discovery's ``_set_mapping`` does to a contract that no longer maps."""
    row = await session.get(EventContract, contract_id)
    assert row is not None
    row.mapping_status = MappingStatus.UNMAPPED
    row.release_id = None
    row.measure = row.comparator = row.strike_lo = row.strike_hi = None
    row.unit = row.resolution_precision = row.resolution_source = None
    await session.commit()


async def _position(
    session: AsyncSession,
    book: BookState,
    contract: EventContract,
    *,
    side: PositionSide = PositionSide.YES,
    qty: int = 10,
    cost: int = 400,
    mark: str | None = None,
    settled: bool = False,
) -> Position:
    pos = Position(
        book_id=book.id,
        contract_id=contract.id,
        side=side,
        qty=qty,
        cost_cents=cost,
        mark_price=None if mark is None else D(mark),
        settled=settled,
        updated_at=BASE_TS,
    )
    session.add(pos)
    await session.commit()
    return pos


async def _decision(
    session: AsyncSession,
    book: BookState,
    contract: EventContract,
    *,
    stake: int,
    verdict: Verdict = Verdict.APPROVED,
    with_order: bool = False,
) -> RiskDecision:
    row = RiskDecision(
        book_id=book.id,
        run_key="run-1",
        slot=Slot.T24H,
        release_id=contract.release_id,
        contract_id=contract.id,
        side=PositionSide.YES,
        inputs_hash="0" * 64,
        engine_version="risk:v1",
        fair_yes=D("0.6"),
        limit_price=D("0.5"),
        stake_final_cents=stake,
        qty=10,
        verdict=verdict,
        reasons=[],
        limits_version=1,
        limits_snapshot={},
        decided_at=BASE_TS,
    )
    session.add(row)
    await session.flush()
    if with_order:
        session.add(
            Order(
                client_order_id=f"{row.id:064d}",
                risk_decision_id=row.id,
                book_id=book.id,
                contract_id=contract.id,
                side=PositionSide.YES,
                limit_price=D("0.5"),
                qty=10,
                filled_qty=10,
                status=OrderStatus.FILLED,
                created_at=BASE_TS,
            )
        )
    await session.commit()
    return row


async def _audit_count(session: AsyncSession, action: str) -> int:
    return (
        await session.execute(
            select(func.count()).select_from(AuditEvent).where(AuditEvent.action == action)
        )
    ).scalar_one()


async def _mark(
    sessions: async_sessionmaker[AsyncSession], book_id: int, asof: datetime
) -> BookState:
    clock = SimClock(asof)
    async with unit_of_work(sessions) as s:
        book = await mark_book(s, AuditLog(clock), clock, book_id, asof)
        s.expunge(book)
    return book


async def test_create_book_sets_equity_and_audits(
    sessions: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    # 03:00 UTC on Nov 12 is still Nov 11 in New York.
    clock = SimClock(datetime(2026, 11, 12, 3, 0, tzinfo=UTC))
    async with unit_of_work(sessions) as s:
        book = await create_book(
            s, AuditLog(clock), clock, mode=BookMode.REPLAY, replay_batch_id=7, cash_cents=500_000
        )
        book_id = book.id
    row = await session.get(BookState, book_id)
    assert row is not None
    assert (row.cash_cents, row.equity_cents, row.peak_equity_cents) == (500_000,) * 3
    assert row.day_start_equity_cents == 500_000
    assert row.day_start_date == date(2026, 11, 11)
    assert row.replay_batch_id == 7
    assert row.kill_switch_on is False
    assert row.created_at == row.updated_at == clock.now()
    event = (
        await session.execute(select(AuditEvent).where(AuditEvent.action == "book.created"))
    ).scalar_one()
    assert event.entity == "book"
    assert event.entity_id == str(book_id)
    assert event.payload["cash_cents"] == 500_000


async def test_create_book_defaults_to_ten_thousand_dollars(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    clock = SimClock(BASE_TS)
    async with unit_of_work(sessions) as s:
        book = await create_book(
            s, AuditLog(clock), clock, mode=BookMode.LIVE, replay_batch_id=None
        )
        assert book.cash_cents == 1_000_000


async def test_lock_book_missing_raises(session: AsyncSession) -> None:
    with pytest.raises(LookupError):
        await lock_book(session, 999)


async def test_lock_book_returns_row(session: AsyncSession) -> None:
    book = await make_book(session)
    locked = await lock_book(session, book.id)
    assert locked.id == book.id


async def test_exposure_counts_positions_and_unordered_approvals(session: AsyncSession) -> None:
    book = await make_book(session)
    rel_a = await make_release(session, period="2026-10")
    rel_b = await make_release(session, period="2026-11")
    c1 = await _contract(session, rel_a, "0.1")
    c2 = await _contract(session, rel_a, "0.2")
    c3 = await _contract(session, rel_b, "0.3", event="ev-2")
    c4 = await _contract(session, rel_b, "0.4", event="ev-2")
    await _position(session, book, c1, cost=400)
    await _position(session, book, c2, cost=300, settled=True)  # settled: ignored
    await _position(session, book, c3, cost=250)
    await _decision(session, book, c1, stake=100)  # approved, no order: counts
    await _decision(session, book, c3, stake=999, with_order=True)  # has order: not doubled
    await _decision(session, book, c4, stake=777, verdict=Verdict.REJECTED)  # not approved
    exp = await exposure(session, book.id)
    assert exp == Exposure(
        contract={c1.id: 500, c3.id: 250},
        event={rel_a.id: 500, rel_b.id: 250},
        portfolio=750,
    )


async def test_exposure_unchanged_when_held_contract_loses_release(
    session: AsyncSession,
) -> None:
    book = await make_book(session)
    rel = await make_release(session)
    c = await _contract(session, rel)
    await _position(session, book, c, cost=400)
    await _decision(session, book, c, stake=100, with_order=True)
    before = await exposure(session, book.id)
    contract_id, release_id = c.id, rel.id
    await _unmap(session, contract_id)
    after = await exposure(session, book.id)
    assert (
        after
        == before
        == Exposure(contract={contract_id: 400}, event={release_id: 400}, portfolio=400)
    )


async def test_exposure_fails_closed_when_no_release_known(session: AsyncSession) -> None:
    book = await make_book(session)
    rel = await make_release(session)
    c = await _contract(session, rel)
    await _position(session, book, c, cost=400)
    await _unmap(session, c.id)
    with pytest.raises(RuntimeError, match="no release"):
        await exposure(session, book.id)


async def test_exposure_ignores_other_books_and_empty(session: AsyncSession) -> None:
    book = await make_book(session, batch=1)
    other = await make_book(session, batch=2)
    rel = await make_release(session)
    c = await _contract(session, rel)
    await _position(session, other, c, cost=400)
    assert await exposure(session, book.id) == Exposure(contract={}, event={}, portfolio=0)


async def test_mark_uses_mid_then_previous_then_cost(
    sessions: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    book = await make_book(session)
    rel = await make_release(session)
    a, b, c, d = [await _contract(session, rel, s) for s in ("0.1", "0.2", "0.3", "0.4")]
    await _position(session, book, a, qty=10, cost=400, mark="0.10000")
    await _position(session, book, b, qty=10, cost=250, mark="0.30000")
    await _position(session, book, c, qty=3, cost=100)
    await _position(session, book, d, qty=10, cost=300, mark="0.20000")
    ids = [a.id, b.id, c.id, d.id]
    ts = BASE_TS
    await add_snapshot(session, a, ts, [("0.40", "10")], [("0.50", "10")])
    # one-sided books fall through to the previous mark / cost
    await add_snapshot(session, b, ts, [], [("0.50", "10")])
    await add_snapshot(session, d, ts, [("0.40", "10")], [])
    # a snapshot after asof must be ignored
    await add_snapshot(session, a, ts.replace(hour=13), [("0.90", "10")], [("0.95", "10")])
    marked = await _mark(sessions, book.id, ts)
    session.expire_all()
    marks = {
        p.contract_id: p.mark_price for p in (await session.execute(select(Position))).scalars()
    }
    assert marks == {
        ids[0]: D("0.45000"),
        ids[1]: D("0.30000"),
        ids[2]: D("0.33333"),
        ids[3]: D("0.20000"),
    }
    # 450 + 300 + round(3 * 0.33333 * 100 = 99.999) + 200
    assert marked.equity_cents == 1_000_000 + 450 + 300 + 100 + 200


async def test_no_side_marks_at_one_minus_mid(
    sessions: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    book = await make_book(session)
    rel = await make_release(session)
    yes_c, no_c, no_cost = [await _contract(session, rel, s) for s in ("0.1", "0.2", "0.3")]
    await _position(session, book, yes_c, qty=10, cost=500)
    await _position(session, book, no_c, side=PositionSide.NO, qty=10, cost=500)
    await _position(session, book, no_cost, side=PositionSide.NO, qty=10, cost=420)
    no_cost_id = no_cost.id
    await add_snapshot(session, yes_c, BASE_TS, [("0.60", "5")], [("0.60", "5")])
    await add_snapshot(session, no_c, BASE_TS, [("0.60", "5")], [("0.60", "5")])
    marked = await _mark(sessions, book.id, BASE_TS)
    # NO value = 10 * (1 - 0.60) * 100 = 400; YES = 600; cost fallback NO keeps cost basis.
    assert marked.equity_cents == 1_000_000 + 600 + 400 + 420
    session.expire_all()
    stored = (
        await session.execute(select(Position.mark_price).where(Position.contract_id == no_cost_id))
    ).scalar_one()
    assert stored == D("0.58000")  # YES-price equivalent of paying 0.42 for NO


async def test_equity_sum_rounds_half_even_per_position(
    sessions: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    book = await make_book(session)
    rel = await make_release(session)
    c1, c2 = await _contract(session, rel, "0.1"), await _contract(session, rel, "0.2")
    await _position(session, book, c1, qty=1, cost=0)
    await _position(session, book, c2, qty=1, cost=0)
    # mid 0.025 -> 2.5 cents -> 2 ; mid 0.035 -> 3.5 cents -> 4 (half-even)
    await add_snapshot(session, c1, BASE_TS, [("0.02", "5")], [("0.03", "5")])
    await add_snapshot(session, c2, BASE_TS, [("0.03", "5")], [("0.04", "5")])
    marked = await _mark(sessions, book.id, BASE_TS)
    assert marked.equity_cents == 1_000_000 + 2 + 4


async def test_day_roll_resets_day_start_on_new_et_day(
    sessions: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    book = await make_book(session, day_start_date=date(2026, 11, 11))
    rel = await make_release(session)
    c = await _contract(session, rel)
    await _position(session, book, c, qty=100, cost=4000)
    await add_snapshot(session, c, BASE_TS, [("0.40", "5")], [("0.42", "5")])
    # 03:59 UTC Nov 12 = 22:59 ET Nov 11: same ET day, day start unchanged.
    first = await _mark(sessions, book.id, datetime(2026, 11, 12, 3, 59, tzinfo=UTC))
    assert first.equity_cents == 1_004_100
    assert first.day_start_date == date(2026, 11, 11)
    assert first.day_start_equity_cents == 1_000_000
    await add_snapshot(
        session, c, datetime(2026, 11, 12, 4, 0, tzinfo=UTC), [("0.50", "5")], [("0.52", "5")]
    )
    # 05:01 UTC = 00:01 ET Nov 12: new day, day start becomes the equity before this mark.
    second = await _mark(sessions, book.id, datetime(2026, 11, 12, 5, 1, tzinfo=UTC))
    assert second.equity_cents == 1_005_100
    assert second.day_start_date == date(2026, 11, 12)
    assert second.day_start_equity_cents == 1_004_100
    assert second.peak_equity_cents == 1_005_100


async def _loss_book(session: AsyncSession, *, cash: int, peak: int, day_start: int) -> BookState:
    book = await make_book(session)
    book.cash_cents = cash
    book.peak_equity_cents = peak
    book.day_start_equity_cents = day_start
    book.equity_cents = cash
    await session.commit()
    return book


async def test_daily_loss_8pct_trips(
    sessions: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    book = await _loss_book(session, cash=920_000, peak=1_000_000, day_start=1_000_000)
    marked = await _mark(sessions, book.id, BASE_TS)
    assert marked.kill_switch_on is True
    assert marked.kill_actor == "system:risk"
    assert marked.kill_reason is not None
    assert marked.kill_reason.startswith("daily_loss:")
    assert await _audit_count(session, "kill_switch.tripped") == 1


async def test_daily_loss_just_inside_limit_does_not_trip(
    sessions: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    book = await _loss_book(session, cash=920_001, peak=1_000_000, day_start=1_000_000)
    marked = await _mark(sessions, book.id, BASE_TS)
    assert marked.kill_switch_on is False


async def test_drawdown_15pct_trips_from_peak(
    sessions: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    # day start equals current equity so only the drawdown rule can fire.
    book = await _loss_book(session, cash=1_700_000, peak=2_000_000, day_start=1_700_000)
    marked = await _mark(sessions, book.id, BASE_TS)
    assert marked.kill_switch_on is True
    assert marked.kill_reason is not None
    assert marked.kill_reason.startswith("drawdown:")
    assert marked.kill_actor == "system:risk"


async def test_drawdown_just_inside_limit_does_not_trip(
    sessions: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    book = await _loss_book(session, cash=1_700_001, peak=2_000_000, day_start=1_700_001)
    assert (await _mark(sessions, book.id, BASE_TS)).kill_switch_on is False


async def test_unchanged_equity_writes_no_audit(
    sessions: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    book = await make_book(session)
    await _mark(sessions, book.id, BASE_TS)
    await _mark(sessions, book.id, BASE_TS)
    assert await _audit_count(session, "book.marked") == 0


async def test_day_roll_with_unchanged_equity_audits_book_marked(
    sessions: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    book = await make_book(session, day_start_date=date(2026, 11, 10))
    book.day_start_equity_cents = 990_000  # yesterday's start; today's is the last mark
    await session.commit()

    await _mark(sessions, book.id, BASE_TS)  # 07:00 ET on Nov 11: a new ET day, flat book

    event = (
        await session.execute(select(AuditEvent).where(AuditEvent.action == "book.marked"))
    ).scalar_one()
    assert event.payload["previous_equity_cents"] == event.payload["equity_cents"] == 1_000_000
    assert event.payload["day_start_equity_cents"] == 1_000_000
    assert event.payload["day_start_date"] == "2026-11-11"
    assert event.payload["peak_equity_cents"] == 1_000_000


async def test_changed_equity_audits_book_marked(
    sessions: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    book = await make_book(session)
    rel = await make_release(session)
    c = await _contract(session, rel)
    await _position(session, book, c, qty=10, cost=400)
    await add_snapshot(session, c, BASE_TS, [("0.60", "5")], [("0.60", "5")])
    await _mark(sessions, book.id, BASE_TS)
    event = (
        await session.execute(select(AuditEvent).where(AuditEvent.action == "book.marked"))
    ).scalar_one()
    assert event.payload["equity_cents"] == 1_000_600
    assert event.payload["previous_equity_cents"] == 1_000_000
    assert event.entity_id == str(book.id)


async def test_mark_on_tripped_book_does_not_retrip(
    sessions: async_sessionmaker[AsyncSession], session: AsyncSession
) -> None:
    book = await _loss_book(session, cash=900_000, peak=1_000_000, day_start=1_000_000)
    await _mark(sessions, book.id, BASE_TS)
    await _mark(sessions, book.id, BASE_TS)
    assert await _audit_count(session, "kill_switch.tripped") == 1

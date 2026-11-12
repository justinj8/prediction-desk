from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from desk_fixtures import add_contract, make_book, make_release
from prediction_desk.db.models import BookState, EventContract, Order, RiskDecision
from prediction_desk.domain.enums import (
    BookMode,
    Comparator,
    Measure,
    OrderStatus,
    PositionSide,
    Slot,
    Venue,
    Verdict,
)

NOW = datetime(2026, 11, 11, 12, 0, tzinfo=UTC)
D = Decimal


async def _contract(session: AsyncSession) -> EventContract:
    release = await make_release(session)
    return await add_contract(
        session,
        release,
        venue=Venue.KALSHI,
        measure=Measure.CPI_MOM,
        comparator=Comparator.GT,
        strike_lo=D("0.3"),
    )


def _decision(book: BookState, contract: EventContract, **overrides: object) -> RiskDecision:
    fields: dict[str, object] = {
        "book_id": book.id,
        "run_key": "r1",
        "slot": Slot.T24H,
        "release_id": contract.release_id,
        "contract_id": contract.id,
        "side": PositionSide.YES,
        "inputs_hash": "a" * 64,
        "engine_version": "risk:v1",
        "fair_yes": D("0.60000"),
        "limit_price": D("0.50000"),
        "qty": 10,
        "verdict": Verdict.APPROVED,
        "reasons": ["ok"],
        "limits_version": 1,
        "limits_snapshot": {},
        "decided_at": NOW,
    }
    fields.update(overrides)
    return RiskDecision(**fields)


def _order_values(
    decision_id: int | None, book: BookState, contract: EventContract, client_order_id: str
) -> dict[str, object]:
    return {
        "client_order_id": client_order_id,
        "risk_decision_id": decision_id,
        "book_id": book.id,
        "contract_id": contract.id,
        "side": PositionSide.YES,
        "limit_price": D("0.50000"),
        "qty": 10,
        "filled_qty": 0,
        "status": OrderStatus.CANCELLED,
        "created_at": NOW,
    }


async def test_order_requires_risk_decision(session: AsyncSession) -> None:
    book = await make_book(session)
    contract = await _contract(session)
    null_values = _order_values(None, book, contract, "c" * 64)
    dangling_values = _order_values(999_999, book, contract, "d" * 64)
    with pytest.raises(IntegrityError):
        await session.execute(insert(Order).values(null_values))
    await session.rollback()
    with pytest.raises(IntegrityError):
        await session.execute(insert(Order).values(dangling_values))


async def test_one_order_per_decision(session: AsyncSession) -> None:
    book = await make_book(session)
    contract = await _contract(session)
    decision = _decision(book, contract)
    session.add(decision)
    await session.commit()
    session.add(Order(**_order_values(decision.id, book, contract, "1" * 64)))
    await session.commit()
    session.add(Order(**_order_values(decision.id, book, contract, "2" * 64)))
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_filled_qty_within_qty(session: AsyncSession) -> None:
    book = await make_book(session)
    contract = await _contract(session)
    decision = _decision(book, contract)
    session.add(decision)
    await session.commit()
    values = _order_values(decision.id, book, contract, "1" * 64)
    values["filled_qty"] = 11
    session.add(Order(**values))
    with pytest.raises(IntegrityError):
        await session.commit()


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"side": None}, id="no-side"),
        pytest.param({"qty": 0}, id="zero-qty"),
        pytest.param({"limit_price": None}, id="no-limit-price"),
    ],
)
async def test_approved_decision_needs_side_qty_limit(
    session: AsyncSession, overrides: dict[str, object]
) -> None:
    book = await make_book(session)
    contract = await _contract(session)
    session.add(_decision(book, contract, **overrides))
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_rejected_decision_needs_no_side(session: AsyncSession) -> None:
    book = await make_book(session)
    contract = await _contract(session)
    session.add(
        _decision(book, contract, side=None, qty=0, limit_price=None, verdict=Verdict.REJECTED)
    )
    await session.commit()


async def test_decision_unique_per_run_slot_contract(session: AsyncSession) -> None:
    book = await make_book(session)
    contract = await _contract(session)
    session.add(_decision(book, contract))
    await session.commit()
    session.add(_decision(book, contract))
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_live_book_has_no_batch_and_is_unique(session: AsyncSession) -> None:
    live = await make_book(session, mode=BookMode.LIVE)
    assert live.replay_batch_id is None
    session.add(
        BookState(
            mode=BookMode.LIVE,
            replay_batch_id=None,
            cash_cents=1,
            equity_cents=1,
            peak_equity_cents=1,
            day_start_equity_cents=1,
            day_start_date=NOW.date(),
            kill_switch_on=False,
            created_at=NOW,
            updated_at=NOW,
        )
    )
    with pytest.raises(IntegrityError):
        await session.commit()
    await session.rollback()
    session.add(
        BookState(
            mode=BookMode.LIVE,
            replay_batch_id=7,
            cash_cents=1,
            equity_cents=1,
            peak_equity_cents=1,
            day_start_equity_cents=1,
            day_start_date=NOW.date(),
            kill_switch_on=False,
            created_at=NOW,
            updated_at=NOW,
        )
    )
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_replay_batches_are_distinct_books(session: AsyncSession) -> None:
    first = await make_book(session, batch=1)
    second = await make_book(session, batch=2)
    assert first.id != second.id
    with pytest.raises(IntegrityError):
        await make_book(session, batch=1)

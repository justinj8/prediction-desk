"""Async DB helpers for risk/execution tests: releases, mapped contracts, snapshots, books.

Everything is deterministic: timestamps are fixed constants or passed in (no wall clock).
"""

from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from prediction_desk.db.models import BookState, EventContract, MarketSnapshot, Release
from prediction_desk.domain.enums import (
    BookMode,
    Comparator,
    MappingStatus,
    Measure,
    Series,
    Venue,
)
from prediction_desk.markets.mapping import MEASURE_SPECS

BASE_TS = datetime(2026, 11, 11, 12, 0, tzinfo=UTC)
RELEASE_AT = BASE_TS + timedelta(days=1)

Level = tuple[str, str]


async def make_release(
    session: AsyncSession,
    series: Series = Series.CPI,
    period: str = "2026-10",
    scheduled_at: datetime = RELEASE_AT,
) -> Release:
    release = Release(series=series, period=period, scheduled_at=scheduled_at)
    session.add(release)
    await session.commit()
    return release


async def add_contract(
    session: AsyncSession,
    release: Release,
    *,
    venue: Venue,
    measure: Measure,
    comparator: Comparator,
    strike_lo: Decimal,
    strike_hi: Decimal | None = None,
    venue_event_id: str = "ev-1",
    confidence: Decimal = Decimal("0.95"),
) -> EventContract:
    unit, precision, source = MEASURE_SPECS[measure]
    market_id = f"{venue_event_id}-{measure}-{comparator}-{strike_lo}-{strike_hi}"
    row = EventContract(
        venue=venue,
        venue_market_id=market_id,
        venue_event_id=venue_event_id,
        yes_token_id=None,
        title=market_id,
        rules_text="rules",
        close_time=release.scheduled_at,
        release_id=release.id,
        measure=measure,
        comparator=comparator,
        strike_lo=strike_lo,
        strike_hi=strike_hi,
        unit=unit,
        resolution_precision=precision,
        resolution_source=source,
        mapping_status=MappingStatus.MAPPED,
        mapping_confidence=confidence,
        mapped_by="test",
        created_at=BASE_TS,
        updated_at=BASE_TS,
    )
    session.add(row)
    await session.commit()
    return row


async def add_snapshot(
    session: AsyncSession,
    contract: EventContract,
    ts: datetime,
    bids: Sequence[Level],
    asks: Sequence[Level],
) -> MarketSnapshot:
    """Store a best-first book; ``bids``/``asks`` are (price, size) strings."""
    snap = MarketSnapshot(
        contract_id=contract.id,
        ts=ts,
        venue_ts=None,
        best_bid=Decimal(bids[0][0]) if bids else None,
        best_ask=Decimal(asks[0][0]) if asks else None,
        bid_size=Decimal(bids[0][1]) if bids else None,
        ask_size=Decimal(asks[0][1]) if asks else None,
        book={"bids": [list(lv) for lv in bids], "asks": [list(lv) for lv in asks]},
    )
    session.add(snap)
    await session.commit()
    return snap


async def make_book(
    session: AsyncSession,
    *,
    mode: BookMode = BookMode.REPLAY,
    batch: int = 1,
    cash_cents: int = 1_000_000,
    at: datetime = BASE_TS,
    day_start_date: date | None = None,
) -> BookState:
    """Insert a fresh book directly; every equity figure starts at cash."""
    book = BookState(
        mode=mode,
        replay_batch_id=batch if mode is BookMode.REPLAY else None,
        cash_cents=cash_cents,
        equity_cents=cash_cents,
        peak_equity_cents=cash_cents,
        day_start_equity_cents=cash_cents,
        day_start_date=at.date() if day_start_date is None else day_start_date,
        kill_switch_on=False,
        created_at=at,
        updated_at=at,
    )
    session.add(book)
    await session.commit()
    return book

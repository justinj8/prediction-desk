from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from prediction_desk.clock import SimClock
from prediction_desk.db.models import AuditEvent, EventContract, MarketSnapshot, Release
from prediction_desk.domain.enums import Comparator, MappingStatus, Measure, Series, Venue
from prediction_desk.markets.snapshots import latest_snapshot_asof, record_snapshots
from prediction_desk.markets.types import BookLevel, MarketRef, OrderBook, VenueMarket
from prediction_desk.net.http import ProviderError

D = Decimal
NOW = datetime(2026, 11, 20, tzinfo=UTC)


class FakeBooks:
    venue = Venue.KALSHI

    def __init__(self, books: Mapping[str, OrderBook | Exception]) -> None:
        self.books = books

    async def list_markets(self, series_code: str) -> list[VenueMarket]:
        return []

    async def order_book(self, ref: MarketRef) -> OrderBook:
        book = self.books[ref.market_id]
        if isinstance(book, Exception):
            raise book
        return book


def _book(
    bids: list[tuple[str, str]],
    asks: list[tuple[str, str]],
    ts: datetime = NOW,
    venue_ts: datetime | None = None,
) -> OrderBook:
    return OrderBook(
        tuple(BookLevel(D(p), D(s)) for p, s in bids),
        tuple(BookLevel(D(p), D(s)) for p, s in asks),
        ts,
        venue_ts,
    )


async def _contract(
    session: AsyncSession,
    market_id: str,
    close: datetime = NOW + timedelta(days=1),
    mapped: bool = True,
    venue: Venue = Venue.KALSHI,
) -> EventContract:
    release = Release(series=Series.CPI, period=f"p-{market_id}", scheduled_at=close)
    session.add(release)
    await session.flush()
    row = EventContract(
        venue=venue,
        venue_market_id=market_id,
        venue_event_id="E",
        yes_token_id=None,
        title="t",
        rules_text="r",
        close_time=close,
        release_id=release.id if mapped else None,
        measure=Measure.CPI_MOM if mapped else None,
        comparator=Comparator.GT if mapped else None,
        strike_lo=D("0.3") if mapped else None,
        mapping_status=MappingStatus.MAPPED if mapped else MappingStatus.UNMAPPED,
        mapping_confidence=D("1.00") if mapped else D("0.00"),
        mapped_by="rules:v1",
        created_at=NOW,
        updated_at=NOW,
    )
    session.add(row)
    await session.commit()
    return row


async def test_snapshot_records_best_levels(session: AsyncSession) -> None:
    c = await _contract(session, "A")
    book = _book([("0.40", "10"), ("0.39", "5")], [("0.45", "7"), ("0.46", "3")])
    result = await record_snapshots(session, {Venue.KALSHI: FakeBooks({"A": book})}, SimClock(NOW))
    assert result.recorded == 1
    assert result.failures == []
    snap = (await session.execute(select(MarketSnapshot))).scalar_one()
    assert snap.contract_id == c.id
    assert snap.ts == NOW
    assert snap.venue_ts is None
    assert (snap.best_bid, snap.best_ask) == (D("0.4"), D("0.45"))
    assert (snap.bid_size, snap.ask_size) == (D("10"), D("7"))
    assert snap.book == book.to_json()
    audit = (await session.execute(select(func.count()).select_from(AuditEvent))).scalar_one()
    assert audit == 0


async def test_snapshot_empty_book_stores_nulls(session: AsyncSession) -> None:
    await _contract(session, "A")
    await _contract(session, "B")
    adapter = FakeBooks({"A": _book([], []), "B": _book([("0.40", "10")], [])})
    result = await record_snapshots(session, {Venue.KALSHI: adapter}, SimClock(NOW))
    assert result.recorded == 2
    rows = (await session.execute(select(MarketSnapshot).order_by(MarketSnapshot.id))).scalars()
    a, b = list(rows)
    assert (a.best_bid, a.best_ask, a.bid_size, a.ask_size) == (None, None, None, None)
    assert a.book == {"bids": [], "asks": []}
    assert (b.best_bid, b.bid_size) == (D("0.4"), D("10"))
    assert (b.best_ask, b.ask_size) == (None, None)


async def test_snapshot_skips_closed_and_unmapped(session: AsyncSession) -> None:
    await _contract(session, "closed", close=NOW - timedelta(seconds=1))
    await _contract(session, "unmapped", mapped=False)
    await _contract(session, "other_venue_missing")
    adapter = FakeBooks({"other_venue_missing": _book([], [])})
    result = await record_snapshots(session, {Venue.KALSHI: adapter}, SimClock(NOW))
    assert result.recorded == 1


async def test_snapshot_skips_venues_without_adapter(session: AsyncSession) -> None:
    await _contract(session, "A")
    result = await record_snapshots(session, {}, SimClock(NOW))
    assert result.recorded == 0


async def test_snapshot_failure_isolated(session: AsyncSession) -> None:
    bad = await _contract(session, "bad")
    await _contract(session, "good")
    adapter = FakeBooks({"bad": ProviderError("kalshi", "boom"), "good": _book([], [])})
    result = await record_snapshots(session, {Venue.KALSHI: adapter}, SimClock(NOW))
    assert result.recorded == 1
    assert result.failures == [(bad.id, "ProviderError: kalshi: boom")]
    assert (
        await session.execute(select(func.count()).select_from(MarketSnapshot))
    ).scalar_one() == 1


async def test_latest_snapshot_asof_never_returns_future(session: AsyncSession) -> None:
    c = await _contract(session, "A")
    for minutes in (0, 10, 20):
        session.add(
            MarketSnapshot(
                contract_id=c.id,
                ts=NOW + timedelta(minutes=minutes),
                best_bid=None,
                best_ask=None,
                bid_size=None,
                ask_size=None,
                book={"bids": [], "asks": []},
            )
        )
    await session.commit()
    assert await latest_snapshot_asof(session, c.id, NOW - timedelta(seconds=1)) is None
    snap = await latest_snapshot_asof(session, c.id, NOW + timedelta(minutes=15))
    assert snap is not None
    assert snap.ts == NOW + timedelta(minutes=10)
    exact = await latest_snapshot_asof(session, c.id, NOW + timedelta(minutes=20))
    assert exact is not None
    assert exact.ts == NOW + timedelta(minutes=20)


class FakePolyBooks(FakeBooks):
    venue = Venue.POLYMARKET


async def test_snapshot_failure_on_one_venue_keeps_other_venue(session: AsyncSession) -> None:
    kalshi = await _contract(session, "K")
    poly = await _contract(session, "P", venue=Venue.POLYMARKET)
    adapters = {
        Venue.KALSHI: FakeBooks({"K": ProviderError("kalshi", "malformed order book: KeyError")}),
        Venue.POLYMARKET: FakePolyBooks({"P": _book([("0.40", "10")], [])}),
    }
    result = await record_snapshots(session, adapters, SimClock(NOW))
    assert result.recorded == 1
    assert result.failures == [(kalshi.id, "ProviderError: kalshi: malformed order book: KeyError")]
    snap = (await session.execute(select(MarketSnapshot))).scalar_one()
    assert snap.contract_id == poly.id


async def test_snapshot_stores_observed_and_venue_ts(session: AsyncSession) -> None:
    await _contract(session, "A")
    venue_ts = NOW - timedelta(seconds=7)
    book = _book([("0.40", "1100024.031234")], [], venue_ts=venue_ts)
    await record_snapshots(session, {Venue.KALSHI: FakeBooks({"A": book})}, SimClock(NOW))
    snap = (await session.execute(select(MarketSnapshot))).scalar_one()
    assert (snap.ts, snap.venue_ts) == (NOW, venue_ts)
    assert snap.bid_size == D("1100024.031234")  # Numeric(24, 6): no rounding of real sizes

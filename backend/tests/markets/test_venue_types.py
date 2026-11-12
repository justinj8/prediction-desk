from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from prediction_desk.markets.types import MAX_LEVEL_SIZE, BookLevel, OrderBook

TS = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)


def _lvl(price: str, size: str = "1") -> BookLevel:
    return BookLevel(Decimal(price), Decimal(size))


def test_orderbook_rejects_unsorted_bids() -> None:
    with pytest.raises(ValueError):
        OrderBook(bids=(_lvl("0.40"), _lvl("0.45")), asks=(), ts=TS)
    with pytest.raises(ValueError):
        OrderBook(bids=(_lvl("0.40"), _lvl("0.40")), asks=(), ts=TS)


def test_orderbook_rejects_unsorted_asks() -> None:
    with pytest.raises(ValueError):
        OrderBook(bids=(), asks=(_lvl("0.60"), _lvl("0.55")), ts=TS)


def test_orderbook_rejects_price_above_one() -> None:
    with pytest.raises(ValueError):
        OrderBook(bids=(_lvl("1.01"),), asks=(), ts=TS)
    with pytest.raises(ValueError):
        OrderBook(bids=(), asks=(_lvl("-0.01"),), ts=TS)


def test_orderbook_rejects_naive_ts() -> None:
    with pytest.raises(ValueError):
        OrderBook(bids=(), asks=(), ts=datetime(2026, 11, 10, 12, 0))


def test_orderbook_venue_ts_optional_and_utc() -> None:
    assert OrderBook(bids=(), asks=(), ts=TS).venue_ts is None
    assert OrderBook(bids=(), asks=(), ts=TS, venue_ts=TS).venue_ts == TS
    with pytest.raises(ValueError):
        OrderBook(bids=(), asks=(), ts=TS, venue_ts=datetime(2026, 11, 10, 12, 0))
    with pytest.raises(ValueError, match="UTC"):
        OrderBook(bids=(), asks=(), ts=TS, venue_ts=TS.astimezone(timezone(timedelta(hours=-5))))


def test_empty_sides_have_no_best() -> None:
    book = OrderBook(bids=(), asks=(), ts=TS)
    assert book.best_bid is None
    assert book.best_ask is None
    assert book.to_json() == {"bids": [], "asks": []}


def test_best_and_to_json() -> None:
    book = OrderBook(
        bids=(_lvl("0.49", "142.15"), _lvl("0.47", "100")),
        asks=(_lvl("0.51", "139.54"),),
        ts=TS,
    )
    assert book.best_bid == _lvl("0.49", "142.15")
    assert book.best_ask == _lvl("0.51", "139.54")
    assert book.to_json() == {
        "bids": [["0.49", "142.15"], ["0.47", "100"]],
        "asks": [["0.51", "139.54"]],
    }


@pytest.mark.parametrize(("price", "size"), [("NaN", "1"), ("Infinity", "1"), ("0.4", "NaN")])
def test_orderbook_rejects_non_finite(price: str, size: str) -> None:
    with pytest.raises(ValueError):
        OrderBook(bids=(_lvl(price, size),), asks=(), ts=TS)


def test_orderbook_rejects_negative_size_allows_zero() -> None:
    with pytest.raises(ValueError, match="size"):
        OrderBook(bids=(), asks=(_lvl("0.5", "-1"),), ts=TS)
    assert OrderBook(bids=(_lvl("0.5", "0"),), asks=(), ts=TS).best_bid == _lvl("0.5", "0")


def test_size_cap() -> None:
    assert Decimal("1e12") == MAX_LEVEL_SIZE
    OrderBook(bids=(_lvl("0.40", "1e12"),), asks=(), ts=TS)
    with pytest.raises(ValueError):
        OrderBook(bids=(_lvl("0.40", "1000000000001"),), asks=(), ts=TS)
    with pytest.raises(ValueError):
        OrderBook(bids=(), asks=(_lvl("0.60", "1000000000001"),), ts=TS)


def test_from_json_roundtrip() -> None:
    b = OrderBook(
        bids=(_lvl("0.40", "10"), _lvl("0.39", "5.5")),
        asks=(_lvl("0.45", "7"),),
        ts=TS,
        venue_ts=TS,
    )
    assert OrderBook.from_json(b.to_json(), b.ts, b.venue_ts) == b
    empty = OrderBook(bids=(), asks=(), ts=TS)
    assert OrderBook.from_json(empty.to_json(), TS) == empty


def test_from_json_rejects_unsorted() -> None:
    with pytest.raises(ValueError):
        OrderBook.from_json({"bids": [["0.40", "1"], ["0.45", "1"]], "asks": []}, TS)
    with pytest.raises(ValueError):
        OrderBook.from_json({"bids": [], "asks": [["0.60", "1"], ["0.55", "1"]]}, TS)


@pytest.mark.parametrize(
    "bad",
    [
        [["0.40"]],
        [["0.40", "1", "2"]],
        [[0.4, "1"]],
        [["0.40", 1]],
        [["abc", "1"]],
        ["0.40"],
        [None],
        "0.40",
    ],
)
def test_from_json_rejects_malformed_levels(bad: object) -> None:
    with pytest.raises(ValueError):
        OrderBook.from_json({"bids": bad, "asks": []}, TS)  # type: ignore[dict-item]

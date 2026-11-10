from datetime import UTC, datetime
from decimal import Decimal

import pytest

from prediction_desk.markets.types import BookLevel, OrderBook

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

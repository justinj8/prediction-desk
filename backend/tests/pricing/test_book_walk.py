from datetime import UTC, datetime
from decimal import Decimal

import pytest

from prediction_desk.markets.types import BookLevel, OrderBook
from prediction_desk.pricing.book import Fill, buy_fill, walk


def core(f: Fill | None) -> tuple[Decimal, Decimal, int] | None:
    return None if f is None else (f.avg_price, f.worst_price, f.qty)


D = Decimal
TS = datetime(2026, 11, 11, tzinfo=UTC)


def lv(price: str, size: str) -> BookLevel:
    return BookLevel(D(price), D(size))


def test_walk_single_level() -> None:
    assert core(walk([lv("0.51", "100")], 10)) == (D("0.51"), D("0.51"), 10)


def test_walk_multi_level_weighted_avg() -> None:
    fill = walk([lv("0.51", "5"), lv("0.53", "10")], 10)
    assert core(fill) == (D("0.52"), D("0.53"), 10)
    assert fill is not None
    assert fill.levels == ((D("0.51"), D("5")), (D("0.53"), D("5")))


def test_walk_exact_depth_stops_at_last_level() -> None:
    fill = walk([lv("0.51", "5"), lv("0.53", "5"), lv("0.60", "50")], 10)
    assert fill is not None
    assert fill.worst_price == D("0.53")


def test_walk_fractional_sizes() -> None:
    fill = walk([lv("0.40", "2.5"), lv("0.50", "10")], 5)
    assert fill is not None
    assert fill.avg_price == D("0.45")


def test_walk_exact_decimal_division() -> None:
    fill = walk([lv("0.10", "1"), lv("0.20", "1"), lv("0.30", "1")], 3)
    assert fill is not None
    assert fill.avg_price == D("0.2")


def test_walk_insufficient_depth_none() -> None:
    assert walk([lv("0.51", "5")], 6) is None
    assert walk([], 1) is None


def test_walk_rejects_bad_qty() -> None:
    with pytest.raises(ValueError, match="qty"):
        walk([lv("0.51", "5")], 0)


def test_walk_skips_zero_size_levels() -> None:
    fill = walk([lv("0.40", "0"), lv("0.50", "5")], 5)
    assert core(fill) == (D("0.50"), D("0.50"), 5)


def book() -> OrderBook:
    return OrderBook(
        bids=(lv("0.49", "10"), lv("0.45", "10")),
        asks=(lv("0.52", "10"), lv("0.55", "10")),
        ts=TS,
    )


def test_buy_yes_walks_asks() -> None:
    assert core(buy_fill(book(), "yes", 15)) == (D("0.53"), D("0.55"), 15)


def test_buy_no_uses_one_minus_bid() -> None:
    assert core(buy_fill(book(), "no", 5)) == (D("0.51"), D("0.51"), 5)
    # 10 @ 0.51 then 5 @ 0.55 -> (5.1 + 2.75) / 15
    fill = buy_fill(book(), "no", 15)
    assert fill is not None
    assert fill.worst_price == D("0.55")
    assert fill.avg_price == (D("5.1") + D("2.75")) / 15


def test_buy_fill_empty_side_none() -> None:
    empty = OrderBook(bids=(), asks=(), ts=TS)
    assert buy_fill(empty, "yes", 1) is None
    assert buy_fill(empty, "no", 1) is None

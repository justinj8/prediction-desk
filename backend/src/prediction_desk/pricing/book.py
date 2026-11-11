"""Walking an order book to get the fill for a market buy of ``qty`` contracts."""

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from prediction_desk.markets.types import BookLevel, OrderBook


@dataclass(frozen=True)
class Fill:
    avg_price: Decimal
    worst_price: Decimal
    qty: int


def walk(levels: Sequence[BookLevel], qty: int) -> Fill | None:
    """Consume ``levels`` (best first) for ``qty`` contracts; None if depth is insufficient."""
    if qty < 1:
        raise ValueError(f"qty must be >= 1: {qty}")
    remaining = Decimal(qty)
    cost = Decimal(0)
    worst = Decimal(0)
    for level in levels:
        take = min(level.size, remaining)
        if take <= 0:
            continue
        cost += level.price * take
        worst = level.price
        remaining -= take
        if remaining == 0:
            return Fill(avg_price=cost / qty, worst_price=worst, qty=qty)
    return None


def side_levels(book: OrderBook, side: Literal["yes", "no"]) -> list[BookLevel]:
    """Levels a buyer of ``side`` lifts, best (cheapest) first, in that side's own prices."""
    if side == "yes":
        return list(book.asks)
    return [BookLevel(Decimal(1) - b.price, b.size) for b in book.bids]


def buy_fill(book: OrderBook, side: Literal["yes", "no"], qty: int) -> Fill | None:
    return walk(side_levels(book, side), qty)

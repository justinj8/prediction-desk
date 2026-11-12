"""Per-side edge: fair probability minus average fill price minus fee per contract."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_FLOOR, ROUND_HALF_UP, Decimal
from typing import Literal

from prediction_desk.domain.enums import Venue
from prediction_desk.domain.types import Prob, to_prob
from prediction_desk.markets.types import OrderBook
from prediction_desk.pricing.book import buy_fill, side_levels
from prediction_desk.pricing.fees import FeeSchedule, fill_fee_cents, schedule_for

Side = Literal["yes", "no"]
_EDGE_QUANTUM = Decimal("0.0001")


@dataclass(frozen=True)
class EdgeQuote:
    side: Side
    fair: Prob  # fair probability of this side
    avg_price: Decimal
    best_price: Decimal
    fee_cents: int
    slippage_cents: int
    edge_per_contract: Decimal
    qty: int


@dataclass(frozen=True)
class Unavailable:
    reason: Literal[
        "no_snapshot", "stale_snapshot", "bad_snapshot", "empty_side", "insufficient_depth"
    ]


def _side_edge(
    side: Side,
    fair: Prob,
    book: OrderBook,
    venue: Venue,
    qty: int,
    asof: datetime,
    schedules: Sequence[FeeSchedule] | None,
) -> EdgeQuote | Unavailable:
    levels = side_levels(book, side)
    if not levels:
        return Unavailable("empty_side")
    fill = buy_fill(book, side, qty)
    if fill is None:
        return Unavailable("insufficient_depth")
    best = levels[0].price
    fee = fill_fee_cents(schedule_for(venue, asof, schedules), fill.avg_price, qty, fill.levels)
    slippage = int(((fill.avg_price - best) * qty * 100).to_integral_value(ROUND_HALF_UP))
    edge = (fair - fill.avg_price - Decimal(fee) / (100 * qty)).quantize(
        _EDGE_QUANTUM, rounding=ROUND_FLOOR
    )
    return EdgeQuote(side, fair, fill.avg_price, best, fee, slippage, edge, qty)


def evaluate_edge(
    fair_yes: Prob,
    book: OrderBook,
    venue: Venue,
    qty: int,
    asof: datetime,
    schedules: Sequence[FeeSchedule] | None = None,
) -> tuple[EdgeQuote | Unavailable, EdgeQuote | Unavailable]:
    """``(yes, no)`` quotes; the NO side uses ``fair = 1 - fair_yes``."""
    fair_no = to_prob(Decimal(1) - fair_yes)
    return (
        _side_edge("yes", fair_yes, book, venue, qty, asof, schedules),
        _side_edge("no", fair_no, book, venue, qty, asof, schedules),
    )


def best_side(
    quotes: tuple[EdgeQuote | Unavailable, EdgeQuote | Unavailable],
) -> EdgeQuote | Unavailable:
    yes, no = quotes
    if isinstance(yes, EdgeQuote) and isinstance(no, EdgeQuote):
        return no if no.edge_per_contract > yes.edge_per_contract else yes
    if isinstance(yes, EdgeQuote):
        return yes
    if isinstance(no, EdgeQuote):
        return no
    return yes

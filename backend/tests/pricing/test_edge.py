from datetime import UTC, datetime
from decimal import Decimal

from prediction_desk.domain.enums import Venue
from prediction_desk.domain.types import to_prob
from prediction_desk.markets.types import BookLevel, OrderBook
from prediction_desk.pricing.edge import EdgeQuote, Unavailable, best_side, evaluate_edge

D = Decimal
TS = datetime(2026, 11, 11, tzinfo=UTC)


def lv(price: str, size: str) -> BookLevel:
    return BookLevel(D(price), D(size))


def mkbook(bids: list[BookLevel], asks: list[BookLevel]) -> OrderBook:
    return OrderBook(bids=tuple(bids), asks=tuple(asks), ts=TS)


def quote(q: EdgeQuote | Unavailable) -> EdgeQuote:
    assert isinstance(q, EdgeQuote)
    return q


def test_edge_yes_side_math() -> None:
    book = mkbook([lv("0.40", "100")], [lv("0.50", "100")])
    yes, _ = evaluate_edge(to_prob("0.60"), book, Venue.KALSHI, 10, TS)
    y = quote(yes)
    assert y.side == "yes"
    assert y.fair == D("0.60000")
    assert y.avg_price == D("0.50")
    assert y.best_price == D("0.50")
    assert y.fee_cents == 18
    assert y.slippage_cents == 0
    assert y.qty == 10
    assert y.edge_per_contract == D("0.0820")


def test_edge_no_side_math() -> None:
    book = mkbook([lv("0.40", "100")], [lv("0.50", "100")])
    _, no = evaluate_edge(to_prob("0.30"), book, Venue.KALSHI, 10, TS)
    n = quote(no)
    assert n.side == "no"
    assert n.fair == D("0.70000")
    assert n.avg_price == D("0.60")
    assert n.best_price == D("0.60")
    assert n.fee_cents == 17  # ceil(0.07 * 10 * 0.24 * 100) = ceil(16.8)
    assert n.edge_per_contract == D("0.0830")  # 0.70 - 0.60 - 0.017


def test_edge_polymarket_has_no_fee() -> None:
    book = mkbook([lv("0.40", "100")], [lv("0.50", "100")])
    yes, _ = evaluate_edge(to_prob("0.60"), book, Venue.POLYMARKET, 10, TS)
    assert quote(yes).fee_cents == 0
    assert quote(yes).edge_per_contract == D("0.1000")


def test_edge_floor_rounding_is_conservative() -> None:
    # Zero fee, so edge = fair - 0.50 carries a 5th-dp remainder that must floor, never round up.
    book = mkbook([], [lv("0.50", "100")])
    yes, _ = evaluate_edge(to_prob("0.60001"), book, Venue.POLYMARKET, 3, TS)
    assert quote(yes).edge_per_contract == D("0.1000")
    # 0.10009 floors to 0.1000 (half-even would give 0.1001)
    yes2, _ = evaluate_edge(to_prob("0.60009"), book, Venue.POLYMARKET, 3, TS)
    assert quote(yes2).edge_per_contract == D("0.1000")


def test_slippage_cents() -> None:
    book = mkbook([], [lv("0.51", "5"), lv("0.53", "10")])
    yes, _ = evaluate_edge(to_prob("0.60"), book, Venue.POLYMARKET, 10, TS)
    y = quote(yes)
    assert y.best_price == D("0.51")
    assert y.avg_price == D("0.52")
    assert y.slippage_cents == 10  # (0.52 - 0.51) * 10 * 100


def test_slippage_rounds_half_up() -> None:
    book = mkbook([], [lv("0.50", "1"), lv("0.51", "1")])
    yes, _ = evaluate_edge(to_prob("0.60"), book, Venue.POLYMARKET, 2, TS)
    assert quote(yes).slippage_cents == 1  # 0.5 cents rounds up


def test_edge_picks_better_side() -> None:
    book = mkbook([lv("0.40", "100")], [lv("0.50", "100")])
    quotes = evaluate_edge(to_prob("0.30"), book, Venue.KALSHI, 10, TS)
    assert best_side(quotes) is quotes[1]
    quotes = evaluate_edge(to_prob("0.60"), book, Venue.KALSHI, 10, TS)
    assert best_side(quotes) is quotes[0]


def test_best_side_tie_prefers_yes() -> None:
    book = mkbook([lv("0.50", "100")], [lv("0.50", "100")])
    quotes = evaluate_edge(to_prob("0.50"), book, Venue.POLYMARKET, 10, TS)
    assert quote(quotes[0]).edge_per_contract == quote(quotes[1]).edge_per_contract
    assert best_side(quotes) is quotes[0]


def test_best_side_single_available() -> None:
    book = mkbook([], [lv("0.50", "100")])
    quotes = evaluate_edge(to_prob("0.30"), book, Venue.POLYMARKET, 10, TS)
    assert best_side(quotes) is quotes[0]
    book = mkbook([lv("0.40", "100")], [])
    quotes = evaluate_edge(to_prob("0.30"), book, Venue.POLYMARKET, 10, TS)
    assert best_side(quotes) is quotes[1]


def test_empty_side_unavailable() -> None:
    book = mkbook([], [lv("0.50", "100")])
    yes, no = evaluate_edge(to_prob("0.60"), book, Venue.KALSHI, 10, TS)
    assert isinstance(yes, EdgeQuote)
    assert no == Unavailable("empty_side")


def test_insufficient_depth_unavailable() -> None:
    book = mkbook([lv("0.40", "5")], [lv("0.50", "100")])
    yes, no = evaluate_edge(to_prob("0.60"), book, Venue.KALSHI, 10, TS)
    assert isinstance(yes, EdgeQuote)
    assert no == Unavailable("insufficient_depth")


def test_best_side_both_unavailable_returns_yes_reason() -> None:
    book = mkbook([], [])
    quotes = evaluate_edge(to_prob("0.60"), book, Venue.KALSHI, 10, TS)
    assert best_side(quotes) == Unavailable("empty_side")
    book = mkbook([lv("0.40", "1")], [lv("0.50", "1")])
    quotes = evaluate_edge(to_prob("0.60"), book, Venue.KALSHI, 10, TS)
    assert best_side(quotes) == Unavailable("insufficient_depth")


def test_fee_uses_per_level_sum_when_larger() -> None:
    # asks 1@0.95 + 1@0.96. Per level: ceil(0.3325)=1 + ceil(0.2688)=1 = 2 cents.
    # Order level at avg 0.955: ceil(0.07*2*0.955*0.045*100 = 0.6016) = 1 cent. Max is 2.
    book = mkbook([], [lv("0.95", "1"), lv("0.96", "1")])
    yes, _ = evaluate_edge(to_prob("0.99"), book, Venue.KALSHI, 2, TS)
    y = quote(yes)
    assert y.fee_cents == 2
    assert y.edge_per_contract == D("0.0250")  # 0.99 - 0.955 - 0.02/2 = 0.025


def test_fee_uses_order_level_when_larger() -> None:
    # asks 1@0.10 + 1@0.90. Per level: ceil(0.63)=1 + ceil(0.63)=1 = 2 cents.
    # Order level at avg 0.50: ceil(0.07*2*0.25*100 = 3.5) = 4 cents. Max is 4.
    book = mkbook([], [lv("0.10", "1"), lv("0.90", "1")])
    yes, _ = evaluate_edge(to_prob("0.99"), book, Venue.KALSHI, 2, TS)
    assert quote(yes).fee_cents == 4


def test_fee_per_level_handles_fractional_take() -> None:
    # 0.5@0.95 + 1.5@0.96: per level ceil(0.16625)=1 + ceil(0.4032)=1 = 2; order at 0.9575: 1.
    book = mkbook([], [lv("0.95", "0.5"), lv("0.96", "5")])
    yes, _ = evaluate_edge(to_prob("0.99"), book, Venue.KALSHI, 2, TS)
    assert quote(yes).fee_cents == 2

from datetime import date
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from prediction_desk.domain.enums import Venue
from prediction_desk.domain.types import to_prob
from prediction_desk.markets.types import BookLevel
from prediction_desk.pricing.fees import FeeSchedule
from prediction_desk.risk.limits import DEFAULT_LIMITS, RiskLimits
from prediction_desk.risk.sizing import (
    SizingInputs,
    clamp_fair,
    depth_at_or_better,
    kelly_fraction,
    limit_price,
    size_order,
)

D = Decimal
LIMITS = RiskLimits.from_params(1, DEFAULT_LIMITS)
ZERO = FeeSchedule(Venue.KALSHI, date(2024, 1, 1), "zero", D(0))
KALSHI = FeeSchedule(Venue.KALSHI, date(2024, 1, 1), "kalshi_quadratic", D("0.07"))
ASKS = [
    BookLevel(D("0.50"), D(10)),
    BookLevel(D("0.53"), D(10)),
    BookLevel(D("0.555"), D(10)),
    BookLevel(D("0.58"), D(10)),
]


def _inputs(**kw: object) -> SizingInputs:
    base: dict[str, object] = {
        "equity_cents": 1_000_000,
        "q": D("0.60"),
        "c": D("0.50"),
        "held_contract_cents": 0,
        "held_event_cents": 0,
        "held_portfolio_cents": 0,
        "depth": D(1000),
        "limit": D("0.53"),
    }
    base.update(kw)
    return SizingInputs(**base)  # type: ignore[arg-type]


def test_kelly_fraction_examples() -> None:
    assert kelly_fraction(D("0.60"), D("0.50")) == D("0.2")
    assert kelly_fraction(D("0.50"), D("0.50")) == D(0)
    assert kelly_fraction(D("0.40"), D("0.50")) == D(0)
    for bad_c in (D(1), D(0)):
        with pytest.raises(ValueError):
            kelly_fraction(D("0.6"), bad_c)
    for bad_q in (D(1), D(0)):
        with pytest.raises(ValueError):
            kelly_fraction(bad_q, D("0.5"))


def test_clamp_fair_bounds() -> None:
    lo, hi = D("0.02"), D("0.98")
    assert clamp_fair(to_prob("0"), lo, hi) == D("0.02")
    assert clamp_fair(to_prob("1"), lo, hi) == D("0.98")
    assert clamp_fair(to_prob("0.5"), lo, hi) == D("0.5")


def test_limit_price_stops_at_first_failing_level() -> None:
    assert limit_price(ASKS, D("0.60"), D("0.04"), ZERO) == D("0.555")
    assert limit_price(ASKS, D("0.60"), D("0.04"), KALSHI) == D("0.53")


def test_limit_none_when_best_level_fails() -> None:
    assert limit_price(ASKS, D("0.52"), D("0.04"), ZERO) is None
    assert limit_price([], D("0.9"), D("0.04"), ZERO) is None


def test_depth_at_or_better() -> None:
    assert depth_at_or_better(ASKS, D("0.53")) == D(20)
    assert depth_at_or_better(ASKS, D("0.40")) == D(0)


def test_size_order_worked_example() -> None:
    s = size_order(_inputs(), LIMITS)
    assert s.kelly_raw_cents == 50_000
    assert s.qty == 250
    assert s.stake_cents == 12_500
    assert s.rejected is None
    assert [r["step"] for r in s.reasons] == [
        "kelly",
        "cap_contract",
        "cap_event",
        "cap_portfolio",
        "liquidity",
        "round",
    ]
    by = {r["step"]: r for r in s.reasons}
    assert by["kelly"]["cents"] == "50000"
    assert by["cap_contract"]["cents"] == "20000"
    assert by["liquidity"]["qty"] == "250"
    assert by["round"]["cents"] == "12500"
    assert all(isinstance(v, str) for r in s.reasons for v in r.values())


def test_negative_headroom_rejects_below_min_size() -> None:
    s = size_order(_inputs(held_contract_cents=30_000), LIMITS)
    assert s.rejected == "below_min_size"
    assert s.qty == 0
    assert s.stake_cents == 0
    assert {r["step"]: r["cents"] for r in s.reasons}["cap_contract"] == "0"


def test_depth_too_thin_rejects_below_min_size() -> None:
    s = size_order(_inputs(depth=D(3)), LIMITS)
    assert s.rejected == "below_min_size"


def test_zero_kelly_rejects_edge_below_min() -> None:
    s = size_order(_inputs(q=D("0.50"), c=D("0.50")), LIMITS)
    assert s.rejected == "edge_below_min"
    assert s.qty == 0


def test_cost_at_or_above_one_rejects_without_dividing() -> None:
    for c in (D("0.99"), D(1), D("1.2")):
        s = size_order(_inputs(q=D("0.98"), c=c), LIMITS)
        assert s.rejected == "edge_below_min"
        assert s.qty == 0 and s.stake_cents == 0


def test_tighter_event_and_portfolio_caps_bind() -> None:
    s = size_order(_inputs(held_event_cents=49_990), LIMITS)
    assert {r["step"]: r["cents"] for r in s.reasons}["cap_event"] == "10"
    assert s.rejected == "below_min_size"
    s2 = size_order(_inputs(held_portfolio_cents=149_000), LIMITS)
    assert {r["step"]: r["cents"] for r in s2.reasons}["cap_portfolio"] == "1000"
    assert s2.qty == 20


@given(
    equity=st.integers(min_value=1, max_value=10**9),
    q_i=st.integers(min_value=2, max_value=98),
    c_i=st.integers(min_value=1, max_value=99),
    held_c=st.integers(min_value=0, max_value=10**8),
    held_e=st.integers(min_value=0, max_value=10**8),
    held_p=st.integers(min_value=0, max_value=10**8),
    depth_i=st.integers(min_value=0, max_value=10**6),
)
def test_stake_never_exceeds_any_cap(
    equity: int, q_i: int, c_i: int, held_c: int, held_e: int, held_p: int, depth_i: int
) -> None:
    q, c, depth = D(q_i) / 100, D(c_i) / 100, D(depth_i)
    s = size_order(
        _inputs(
            equity_cents=equity,
            q=q,
            c=c,
            held_contract_cents=held_c,
            held_event_cents=held_e,
            held_portfolio_cents=held_p,
            depth=depth,
        ),
        LIMITS,
    )
    assert s.qty >= 0
    assert s.stake_cents <= s.kelly_raw_cents + 1
    for frac, held in (
        (LIMITS.cap_contract, held_c),
        (LIMITS.cap_event, held_e),
        (LIMITS.cap_portfolio, held_p),
    ):
        assert s.stake_cents <= max(D(0), frac * equity - held) + 1
    assert s.qty <= int(LIMITS.liquidity_fraction * depth)
    if s.qty == 0:
        assert s.rejected is not None

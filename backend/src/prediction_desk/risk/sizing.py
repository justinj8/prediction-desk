"""Pure sizing math: R14 fair clamp, binary Kelly, limit price, and the cap funnel."""

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

from prediction_desk.domain.types import Prob, to_prob
from prediction_desk.markets.types import BookLevel
from prediction_desk.pricing.fees import FeeSchedule
from prediction_desk.risk.limits import RiskLimits

_ZERO = Decimal(0)
_ONE = Decimal(1)
_CENT = Decimal(100)


def _floor_int(value: Decimal) -> int:
    return int(value.to_integral_value(rounding=ROUND_FLOOR))


def _ceil_int(value: Decimal) -> int:
    return int(value.to_integral_value(rounding=ROUND_CEILING))


def clamp_fair(p: Prob, lo: Decimal, hi: Decimal) -> Prob:
    """R14: clamp a fair probability into ``[lo, hi]`` before edge and Kelly."""
    return to_prob(min(max(p, lo), hi))


def kelly_fraction(q: Decimal, c: Decimal) -> Decimal:
    """Binary-contract Kelly: win prob ``q`` at all-in cost ``c`` per $1 payout."""
    if not (_ZERO < c < _ONE and _ZERO < q < _ONE):
        raise ValueError(f"kelly_fraction needs 0 < q < 1 and 0 < c < 1, got q={q} c={c}")
    if q <= c:
        return _ZERO
    return (q - c) / (_ONE - c)


def _marginal_fee(price: Decimal, schedule: FeeSchedule) -> Decimal:
    if schedule.kind == "kalshi_quadratic":
        return schedule.rate * price * (_ONE - price)
    return _ZERO


def limit_price(
    levels: Sequence[BookLevel], q: Decimal, min_edge: Decimal, schedule: FeeSchedule
) -> Decimal | None:
    """D4: last best-first level price whose edge net of marginal fee clears ``min_edge``."""
    limit: Decimal | None = None
    for level in levels:
        if q - level.price - _marginal_fee(level.price, schedule) < min_edge:
            break
        limit = level.price
    return limit


def depth_at_or_better(levels: Sequence[BookLevel], limit: Decimal) -> Decimal:
    """Contracts available at prices ``<= limit``."""
    return sum((lv.size for lv in levels if lv.price <= limit), _ZERO)


@dataclass(frozen=True)
class SizingInputs:
    equity_cents: int
    q: Decimal
    c: Decimal
    held_contract_cents: int
    held_event_cents: int
    held_portfolio_cents: int
    depth: Decimal  # contracts available at or better than ``limit``
    limit: Decimal


@dataclass(frozen=True)
class Sizing:
    kelly_raw_cents: int
    stake_cents: int
    qty: int
    reasons: tuple[dict[str, str], ...]
    rejected: str | None


def size_order(inputs: SizingInputs, limits: RiskLimits) -> Sizing:
    """Spec §7.2 funnel steps 2-7. Exact Decimal; every reason value is a ``str``."""
    reasons: list[dict[str, str]] = []

    def reject(why: str, kelly_raw: int = 0) -> Sizing:
        return Sizing(kelly_raw, 0, 0, tuple(reasons), why)

    if inputs.c >= _ONE or inputs.q <= inputs.c:
        reasons.append({"step": "kelly", "cents": "0"})
        return reject("edge_below_min")
    f_star = kelly_fraction(inputs.q, inputs.c)
    equity = Decimal(inputs.equity_cents)
    kelly_raw = _floor_int(limits.kelly_multiplier * f_star * equity)
    reasons.append({"step": "kelly", "cents": str(kelly_raw)})

    stake = kelly_raw
    caps = (
        ("cap_contract", limits.cap_contract, inputs.held_contract_cents),
        ("cap_event", limits.cap_event, inputs.held_event_cents),
        ("cap_portfolio", limits.cap_portfolio, inputs.held_portfolio_cents),
    )
    for name, frac, held in caps:
        headroom = max(0, _floor_int(frac * equity - held))
        stake = min(stake, headroom)
        reasons.append({"step": name, "cents": str(stake)})

    cost_cents = inputs.c * _CENT
    liq_qty = _floor_int(limits.liquidity_fraction * inputs.depth)
    reasons.append(
        {"step": "liquidity", "cents": str(_floor_int(liq_qty * cost_cents)), "qty": str(liq_qty)}
    )

    qty = min(_floor_int(Decimal(stake) / cost_cents), liq_qty)
    if qty < 1:
        reasons.append({"step": "round", "cents": "0", "qty": "0"})
        return reject("below_min_size", kelly_raw)
    stake_cents = _ceil_int(qty * cost_cents)
    reasons.append({"step": "round", "cents": str(stake_cents), "qty": str(qty)})
    return Sizing(kelly_raw, stake_cents, qty, tuple(reasons), None)

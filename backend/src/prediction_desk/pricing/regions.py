"""Reported-value regions.

Contracts resolve on the officially reported (rounded) value R = round(X, grid),
so every comparator/strike pair maps to an exact half-open interval [lower, upper)
on the unrounded X. All arithmetic is exact Decimal.
"""

from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

from prediction_desk.domain.enums import Comparator


@dataclass(frozen=True)
class Interval:
    """Half-open [lower, upper) on the unrounded value; None means unbounded."""

    lower: Decimal | None
    upper: Decimal | None


def _grid(precision: int) -> Decimal:
    return Decimal(1).scaleb(-precision)


def _floor_idx(s: Decimal, g: Decimal) -> Decimal:
    return (s / g).to_integral_value(ROUND_FLOOR)


def _ceil_idx(s: Decimal, g: Decimal) -> Decimal:
    return (s / g).to_integral_value(ROUND_CEILING)


def reported_interval(
    comparator: Comparator,
    strike_lo: Decimal,
    strike_hi: Decimal | None,
    precision: int,
) -> Interval:
    g = _grid(precision)
    half = g / 2
    match comparator:
        case Comparator.GT:
            # Smallest grid point strictly above the strike.
            return Interval((_floor_idx(strike_lo, g) + 1) * g - half, None)
        case Comparator.GTE:
            return Interval(_ceil_idx(strike_lo, g) * g - half, None)
        case Comparator.LT:
            # Largest grid point strictly below the strike.
            return Interval(None, (_ceil_idx(strike_lo, g) - 1) * g + half)
        case Comparator.LTE:
            return Interval(None, _floor_idx(strike_lo, g) * g + half)
        case Comparator.EQ:
            if strike_lo / g != (strike_lo / g).to_integral_value():
                return Interval(strike_lo, strike_lo)  # off-grid: impossible, empty
            return Interval(strike_lo - half, strike_lo + half)
        case _:  # Comparator.BETWEEN, the only remaining member
            if strike_hi is None:
                raise ValueError("BETWEEN requires strike_hi")
            if strike_lo > strike_hi:
                raise ValueError(f"strike_lo {strike_lo} exceeds strike_hi {strike_hi}")
            return Interval(
                _ceil_idx(strike_lo, g) * g - half,
                _floor_idx(strike_hi, g) * g + half,
            )

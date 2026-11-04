"""Money (integer cents) and probability (5-dp Decimal) value types. Floats are never allowed."""

from datetime import UTC, datetime
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from typing import NewType

Cents = NewType("Cents", int)
Prob = NewType("Prob", Decimal)

_PROB_QUANTUM = Decimal("0.00001")


def cents(value: int) -> Cents:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"cents requires int, got {type(value).__name__}")
    return Cents(value)


def to_prob(value: Decimal | str | int) -> Prob:
    if isinstance(value, bool) or not isinstance(value, Decimal | str | int):
        raise TypeError(f"probability requires Decimal, str or int, got {type(value).__name__}")
    try:
        d = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"invalid probability: {value!r}") from exc
    if not d.is_finite():
        raise ValueError(f"probability must be finite: {value!r}")
    if d < 0 or d > 1:
        raise ValueError(f"probability must be in [0, 1]: {value!r}")
    return Prob(d.quantize(_PROB_QUANTUM, rounding=ROUND_HALF_EVEN))


def require_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware (timezone required)")
    return ts.astimezone(UTC)

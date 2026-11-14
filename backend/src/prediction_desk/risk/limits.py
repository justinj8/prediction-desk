"""Versioned risk limits: defaults, validation, and loading the active row."""

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Final

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from prediction_desk.db.models import RiskLimitsRow

DEFAULT_LIMITS: Final[Mapping[str, str | int]] = {
    "kelly_multiplier": "0.25",
    "cap_contract": "0.02",
    "cap_event": "0.05",
    "cap_portfolio": "0.15",
    "liquidity_fraction": "0.25",
    "min_edge": "0.04",
    "min_mapping_confidence": "0.90",
    "max_quote_age_s": 60,
    "cutoff_minutes": 5,
    "fair_clamp_lo": "0.02",
    "fair_clamp_hi": "0.98",
    "daily_loss_kill": "0.08",
    "drawdown_kill": "0.15",
}

_INT_FIELDS: Final = {"max_quote_age_s": 1, "cutoff_minutes": 0}
_HALF: Final = Decimal("0.5")


@dataclass(frozen=True)
class RiskLimits:
    version: int
    kelly_multiplier: Decimal
    cap_contract: Decimal
    cap_event: Decimal
    cap_portfolio: Decimal
    liquidity_fraction: Decimal
    min_edge: Decimal
    min_mapping_confidence: Decimal
    max_quote_age_s: int
    cutoff_minutes: int
    fair_clamp_lo: Decimal
    fair_clamp_hi: Decimal
    daily_loss_kill: Decimal
    drawdown_kill: Decimal

    @classmethod
    def from_params(cls, version: int, params: Mapping[str, Any]) -> "RiskLimits":
        expected = set(DEFAULT_LIMITS)
        missing = expected - params.keys()
        extra = params.keys() - expected
        if missing or extra:
            raise ValueError(
                f"risk params mismatch: missing={sorted(missing)} extra={sorted(extra)}"
            )
        values: dict[str, Decimal | int] = {}
        for name in DEFAULT_LIMITS:
            raw = params[name]
            if name in _INT_FIELDS:
                if type(raw) is not int:
                    raise ValueError(f"{name} must be an int, got {raw!r}")
                if raw < _INT_FIELDS[name]:
                    raise ValueError(f"{name} must be >= {_INT_FIELDS[name]}, got {raw}")
                values[name] = raw
            else:
                values[name] = _fraction(name, raw)
        limits = cls(version=version, **values)  # type: ignore[arg-type]
        if not limits.fair_clamp_lo < _HALF < limits.fair_clamp_hi:
            raise ValueError("fair_clamp_lo must be < 0.5 < fair_clamp_hi")
        return limits

    def snapshot(self) -> dict[str, str | int]:
        """JSON-safe parameters (Decimals as strings); round-trips via ``from_params``."""
        out: dict[str, str | int] = {}
        for name in DEFAULT_LIMITS:
            value = getattr(self, name)
            out[name] = value if name in _INT_FIELDS else str(value)
        return out


def _fraction(name: str, raw: object) -> Decimal:
    if not isinstance(raw, str):
        raise ValueError(f"{name} must be a decimal string, got {raw!r}")
    try:
        value = Decimal(raw)
    except InvalidOperation:
        raise ValueError(f"{name} is not a decimal: {raw!r}") from None
    if not value.is_finite():
        raise ValueError(f"{name} must be finite, got {raw!r}")
    if not Decimal(0) < value <= Decimal(1):
        raise ValueError(f"{name} must be in (0, 1], got {raw!r}")
    return value


async def load_active_limits(session: AsyncSession) -> RiskLimits:
    row = (
        await session.execute(select(RiskLimitsRow).where(RiskLimitsRow.active))
    ).scalar_one_or_none()
    if row is None:
        raise LookupError("no active risk_limits row")
    return RiskLimits.from_params(row.version, row.params)

"""Forecast distributions over a macro release value.

Public values are Decimal or Prob. Floats appear only inside the private
kernels (`_cdf`) and cross back through `_prob`, the only float -> Prob path.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from scipy.stats import t as _student_t

from prediction_desk.domain.types import Prob, to_prob

_SUM_TOL = Decimal("1e-9")
_ONE = Prob(Decimal("1.00000"))
_ZERO = Prob(Decimal("0.00000"))


class Distribution(Protocol):
    family: str

    def prob_between(self, lower: Decimal | None, upper: Decimal | None) -> Prob:
        """P(lower <= X < upper); None means unbounded on that side."""
        ...

    def params(self) -> dict[str, str]: ...


def _prob(x: float) -> Prob:
    """The only float -> Prob path: clamp to [0, 1], then 12-dp string round trip."""
    x = min(max(x, 0.0), 1.0)
    return to_prob(Decimal(format(x, ".12f")))


def _require_finite(**values: Decimal) -> None:
    for name, v in values.items():
        if not isinstance(v, Decimal) or not v.is_finite():
            raise ValueError(f"{name} must be a finite Decimal: {v!r}")


def _check_bounds(lower: Decimal | None, upper: Decimal | None) -> Prob | None:
    """Shared interval rules. Returns a short-circuit result, or None to continue."""
    if lower is None and upper is None:
        return _ONE
    if lower is not None and upper is not None:
        if lower > upper:
            raise ValueError(f"lower {lower} exceeds upper {upper}")
        if lower == upper:
            return _ZERO
    return None


@dataclass(frozen=True)
class Normal:
    mu: Decimal
    sigma: Decimal
    family: str = "normal"

    def __post_init__(self) -> None:
        _require_finite(mu=self.mu, sigma=self.sigma)
        if self.sigma <= 0:
            raise ValueError("sigma must be > 0")

    def prob_between(self, lower: Decimal | None, upper: Decimal | None) -> Prob:
        short = _check_bounds(lower, upper)
        if short is not None:
            return short
        hi = 1.0 if upper is None else _cdf(self, upper)
        lo = 0.0 if lower is None else _cdf(self, lower)
        return _prob(hi - lo)

    def params(self) -> dict[str, str]:
        return {"mu": str(self.mu), "sigma": str(self.sigma)}


@dataclass(frozen=True)
class StudentT:
    mu: Decimal
    sigma: Decimal
    nu: Decimal
    family: str = "student_t"

    def __post_init__(self) -> None:
        _require_finite(mu=self.mu, sigma=self.sigma, nu=self.nu)
        if self.sigma <= 0:
            raise ValueError("sigma must be > 0")
        if self.nu <= 0:
            raise ValueError("nu must be > 0")

    def prob_between(self, lower: Decimal | None, upper: Decimal | None) -> Prob:
        short = _check_bounds(lower, upper)
        if short is not None:
            return short
        hi = 1.0 if upper is None else _cdf(self, upper)
        lo = 0.0 if lower is None else _cdf(self, lower)
        return _prob(hi - lo)

    def params(self) -> dict[str, str]:
        return {"mu": str(self.mu), "nu": str(self.nu), "sigma": str(self.sigma)}


def _cdf(dist: Normal | StudentT, x: Decimal) -> float:
    """Private float kernel: CDF of `dist` at `x`."""
    z = float((x - dist.mu) / dist.sigma)
    if isinstance(dist, StudentT):
        return float(_student_t.cdf(z, float(dist.nu)))
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@dataclass(frozen=True)
class Categorical:
    outcomes: tuple[tuple[int, Decimal], ...]
    family: str = "categorical"

    def __post_init__(self) -> None:
        if not self.outcomes:
            raise ValueError("Categorical needs at least one outcome")
        prev: int | None = None
        total = Decimal(0)
        for o, p in self.outcomes:
            if isinstance(o, bool) or not isinstance(o, int):
                raise ValueError(f"outcome must be int: {o!r}")
            if prev is not None and o <= prev:
                raise ValueError("outcomes must be strictly ascending")
            prev = o
            _require_finite(prob=p)
            if p < 0 or p > 1:
                raise ValueError(f"probability out of [0, 1]: {p}")
            total += p
        if abs(total - 1) > _SUM_TOL:
            raise ValueError(f"probabilities must sum to 1, got {total}")

    def _sum(self, pred: Callable[[int], bool]) -> Prob:
        # Exact Decimal arithmetic; validation keeps the sum within 1e-9 of [0, 1].
        s = sum((p for o, p in self.outcomes if pred(o)), Decimal(0))
        return to_prob(min(max(s, Decimal(0)), Decimal(1)))

    def prob_between(self, lower: Decimal | None, upper: Decimal | None) -> Prob:
        short = _check_bounds(lower, upper)
        if short is not None:
            return short
        return self._sum(
            lambda o: (
                (lower is None or Decimal(o) >= lower) and (upper is None or Decimal(o) < upper)
            )
        )

    def prob_where(self, pred: Callable[[int], bool]) -> Prob:
        return self._sum(pred)

    def params(self) -> dict[str, str]:
        return {str(o): str(p) for o, p in sorted(self.outcomes, key=lambda op: str(op[0]))}


def affine[D: (Normal, StudentT)](dist: D, a: Decimal, b: Decimal) -> D:
    """Distribution of a + b*X for b > 0."""
    _require_finite(a=a, b=b)
    if b <= 0:
        raise ValueError("b must be > 0")
    if isinstance(dist, StudentT):
        return StudentT(a + b * dist.mu, b * dist.sigma, dist.nu)  # type: ignore[return-value]
    return Normal(a + b * dist.mu, b * dist.sigma)

"""Fair probability of a contract under a forecast distribution, plus coherence checks."""

import itertools
from collections import defaultdict
from collections.abc import Callable, Sequence
from decimal import Decimal

from prediction_desk.domain.enums import Comparator, Measure
from prediction_desk.domain.types import Prob
from prediction_desk.forecasting.distributions import Categorical, Distribution
from prediction_desk.markets.mapping import ContractTerms
from prediction_desk.pricing.regions import reported_interval

_CONTINUOUS = {"normal", "student_t"}


def _categorical_pred(terms: ContractTerms) -> Callable[[int], bool]:
    s = terms.strike_lo
    match terms.comparator:
        case Comparator.GT:
            return lambda o: Decimal(o) > s
        case Comparator.GTE:
            return lambda o: Decimal(o) >= s
        case Comparator.LT:
            return lambda o: Decimal(o) < s
        case Comparator.LTE:
            return lambda o: Decimal(o) <= s
        case Comparator.EQ:
            return lambda o: Decimal(o) == s
        case _:  # Comparator.BETWEEN, the only remaining member
            hi = terms.strike_hi
            if hi is None:
                raise ValueError("BETWEEN requires strike_hi")
            if s > hi:
                raise ValueError(f"strike_lo {s} exceeds strike_hi {hi}")
            return lambda o: s <= Decimal(o) <= hi


def fair_prob(terms: ContractTerms, dist: Distribution) -> Prob:
    if terms.unit == "bp":
        if not isinstance(dist, Categorical):
            raise ValueError(f"unit 'bp' requires a categorical distribution, got {dist.family}")
        return dist.prob_where(_categorical_pred(terms))
    if dist.family not in _CONTINUOUS:
        raise ValueError(
            f"unit {terms.unit!r} requires a continuous distribution, got {dist.family}"
        )
    interval = reported_interval(
        terms.comparator, terms.strike_lo, terms.strike_hi, terms.resolution_precision
    )
    return dist.prob_between(interval.lower, interval.upper)


_DECREASING = {Comparator.GT, Comparator.GTE}
_INCREASING = {Comparator.LT, Comparator.LTE}


def monotonic_violations(items: Sequence[tuple[ContractTerms, Prob]]) -> list[str]:
    groups: dict[tuple[Measure, Comparator], list[tuple[Decimal, Prob]]] = defaultdict(list)
    for terms, prob in items:
        if terms.comparator in _DECREASING or terms.comparator in _INCREASING:
            groups[(terms.measure, terms.comparator)].append((terms.strike_lo, prob))
    out: list[str] = []
    for (measure, comparator), rows in sorted(
        groups.items(), key=lambda kv: (kv[0][0].value, kv[0][1].value)
    ):
        rows.sort()
        for (s0, p0), (s1, p1) in itertools.pairwise(rows):
            if s0 == s1:
                bad, why = p0 != p1, "equal strikes must have equal probabilities"
            elif comparator in _DECREASING:
                bad, why = p1 > p0, "probability must not increase with strike"
            else:
                bad, why = p1 < p0, "probability must not decrease with strike"
            if bad:
                out.append(
                    f"{measure.value} {comparator.value}: strike {s0} prob {p0} vs "
                    f"strike {s1} prob {p1} ({why})"
                )
    return out


def bracket_sum_violation(
    probs: Sequence[Prob], tolerance: Decimal = Decimal("0.02")
) -> str | None:
    total = sum(probs, Decimal(0))
    if abs(total - 1) <= tolerance:
        return None
    return f"bracket probabilities sum to {total}, outside 1 ± {tolerance}"

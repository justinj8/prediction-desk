from decimal import Decimal

import pytest

from prediction_desk.domain.enums import Comparator, Measure
from prediction_desk.domain.types import Prob, to_prob
from prediction_desk.forecasting.distributions import Categorical, Normal, StudentT
from prediction_desk.markets.mapping import ContractTerms
from prediction_desk.pricing.fair import bracket_sum_violation, fair_prob, monotonic_violations

D = Decimal
C = Comparator


def cpi(comparator: Comparator, lo: str, hi: str | None = None) -> ContractTerms:
    return ContractTerms(
        Measure.CPI_MOM, comparator, D(lo), None if hi is None else D(hi), "pct", 1, "bls"
    )


def fomc(comparator: Comparator, lo: str, hi: str | None = None) -> ContractTerms:
    return ContractTerms(
        Measure.FOMC_CHANGE_BP, comparator, D(lo), None if hi is None else D(hi), "bp", 0, "fed"
    )


FOMC = Categorical(
    tuple(
        (o, D(p)) for o, p in [(-50, "0.1"), (-25, "0.2"), (0, "0.6"), (25, "0.08"), (50, "0.02")]
    )
)


def test_boundary_strike_exact() -> None:
    dist = StudentT(D("0.35"), D("0.1"), D("5"))
    assert fair_prob(cpi(C.GT, "0.3"), dist) == Prob(D("0.50000"))


def test_continuous_between_and_eq() -> None:
    dist = Normal(D("0.3"), D("0.1"))
    assert fair_prob(cpi(C.BETWEEN, "0.2", "0.4"), dist) > to_prob("0.7")
    assert fair_prob(cpi(C.EQ, "0.33"), dist) == to_prob("0")


@pytest.mark.parametrize(
    ("comparator", "lo", "hi", "expected"),
    [
        (C.LT, "-25", None, "0.1"),
        (C.LTE, "-50", None, "0.1"),
        (C.EQ, "0", None, "0.6"),
        (C.GT, "25", None, "0.02"),
        (C.GTE, "50", None, "0.02"),
        (C.GTE, "25", None, "0.1"),
        (C.LTE, "-25", None, "0.3"),
        (C.BETWEEN, "-25", "25", "0.88"),
        (C.EQ, "10", None, "0"),
    ],
)
def test_fomc_categorical_terms(
    comparator: Comparator, lo: str, hi: str | None, expected: str
) -> None:
    assert fair_prob(fomc(comparator, lo, hi), FOMC) == to_prob(expected)


def test_categorical_between_requires_hi() -> None:
    with pytest.raises(ValueError, match="strike_hi"):
        fair_prob(fomc(C.BETWEEN, "-25"), FOMC)
    with pytest.raises(ValueError, match="strike_lo"):
        fair_prob(fomc(C.BETWEEN, "25", "-25"), FOMC)


def test_unit_family_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="categorical"):
        fair_prob(fomc(C.GT, "0"), Normal(D("0"), D("1")))
    with pytest.raises(ValueError, match="continuous"):
        fair_prob(cpi(C.GT, "0.3"), FOMC)


def test_ladder_is_coherent() -> None:
    dist = StudentT(D("0.3"), D("0.15"), D("5"))
    strikes = [D(k) / 10 for k in range(-3, 11)]
    items = [(cpi(C.GT, str(s)), fair_prob(cpi(C.GT, str(s)), dist)) for s in strikes]
    assert monotonic_violations(items) == []
    lt_items = [(cpi(C.LT, str(s)), fair_prob(cpi(C.LT, str(s)), dist)) for s in strikes]
    assert monotonic_violations(lt_items) == []


def test_polymarket_bracket_sums_to_one() -> None:
    dist = StudentT(D("3.7"), D("0.3"), D("5"))
    terms = [cpi(C.LTE, "3.2")]
    terms += [cpi(C.EQ, str(D(k) / 10)) for k in range(33, 43)]
    terms.append(cpi(C.GTE, "4.3"))
    assert bracket_sum_violation([fair_prob(t, dist) for t in terms], D("0.0001")) is None


def test_bracket_violation_message() -> None:
    msg = bracket_sum_violation([to_prob("0.5"), to_prob("0.4")])
    assert msg == "bracket probabilities sum to 0.90000, outside 1 ± 0.02"
    assert bracket_sum_violation([to_prob("0.5"), to_prob("0.4")], D("0.2")) is None
    assert bracket_sum_violation([to_prob("0.5"), to_prob("0.52")]) is None


def test_detects_incoherence() -> None:
    items = [
        (cpi(C.GT, "0.2"), to_prob("0.5")),
        (cpi(C.GT, "0.3"), to_prob("0.6")),
        (cpi(C.LT, "0.2"), to_prob("0.5")),
        (cpi(C.LT, "0.3"), to_prob("0.4")),
    ]
    out = monotonic_violations(items)
    assert len(out) == 2
    assert "cpi_mom" in out[0]
    assert "gt" in out[0] and "0.2" in out[0] and "0.3" in out[0]
    assert "0.5" in out[0] and "0.6" in out[0]
    assert "cpi_mom lt:" in out[1]


def test_equal_strikes_must_match_and_eq_between_ignored() -> None:
    items = [
        (cpi(C.GT, "0.3"), to_prob("0.4")),
        (cpi(C.GT, "0.3"), to_prob("0.4")),
        (cpi(C.GT, "0.3"), to_prob("0.45")),
        (cpi(C.EQ, "0.3"), to_prob("0.9")),
        (cpi(C.EQ, "0.4"), to_prob("0.1")),
        (cpi(C.BETWEEN, "0.1", "0.2"), to_prob("0.9")),
    ]
    out = monotonic_violations(items)
    assert len(out) == 1 and "equal strikes" in out[0]


def test_categorical_with_spoofed_family_cannot_use_continuous_path() -> None:
    spoof = Categorical(((0, D("1")),), family="normal")
    with pytest.raises(ValueError, match="continuous"):
        fair_prob(cpi(C.GT, "0.3"), spoof)

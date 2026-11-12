from decimal import ROUND_HALF_UP, Decimal

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from prediction_desk.domain.enums import Comparator
from prediction_desk.pricing.regions import Interval, reported_interval

D = Decimal
C = Comparator


@pytest.mark.parametrize(
    ("comparator", "lo", "hi", "precision", "lower", "upper"),
    [
        (C.GT, "0.3", None, 1, "0.35", None),
        (C.GT, "-0.3", None, 1, "-0.25", None),
        (C.LTE, "3.2", None, 1, None, "3.25"),
        (C.EQ, "3.3", None, 1, "3.25", "3.35"),
        (C.GTE, "4.3", None, 1, "4.25", None),
        (C.GT, "90000", None, -3, "90500", None),
        (C.GT, "90500", None, -3, "90500", None),
        (C.GTE, "90500", None, -3, "90500", None),
        (C.GTE, "90000", None, -3, "89500", None),
        (C.LT, "90500", None, -3, None, "90500"),
        (C.LTE, "90500", None, -3, None, "90500"),
        (C.EQ, "90000", None, -3, "89500", "90500"),
        (C.BETWEEN, "0.2", "0.4", 1, "0.15", "0.45"),
        (C.BETWEEN, "0.21", "0.39", 1, "0.25", "0.35"),
        (C.LT, "-25", None, 0, None, "-25.5"),
        (C.LT, "3.25", None, 1, None, "3.25"),
        (C.LT, "3.3", None, 1, None, "3.25"),
        (C.LTE, "3.25", None, 1, None, "3.25"),
        (C.GT, "0.35", None, 1, "0.35", None),
        (C.GTE, "0.35", None, 1, "0.35", None),
        (C.GT, "0.3", "9", 1, "0.35", None),
        (C.GT, "-0.25", None, 1, "-0.25", None),
        (C.LT, "-0.25", None, 1, None, "-0.25"),
        (C.GTE, "-0.25", None, 1, "-0.25", None),
        (C.LTE, "-0.25", None, 1, None, "-0.25"),
        (C.GT, "0", None, 1, "0.05", None),
        (C.LT, "0", None, 1, None, "-0.05"),
        (C.LTE, "0", None, 1, None, "0.05"),
        (C.EQ, "-0.3", None, 1, "-0.35", "-0.25"),
        (C.GT, "-90500", None, -3, "-90500", None),
        (C.LTE, "-90500", None, -3, None, "-90500"),
    ],
)
def test_reported_interval(
    comparator: Comparator,
    lo: str,
    hi: str | None,
    precision: int,
    lower: str | None,
    upper: str | None,
) -> None:
    got = reported_interval(comparator, D(lo), None if hi is None else D(hi), precision)
    assert got.lower == (None if lower is None else D(lower))
    assert got.upper == (None if upper is None else D(upper))


def test_off_grid_eq_is_empty() -> None:
    got = reported_interval(C.EQ, D("3.33"), None, 1)
    assert got == Interval(D("3.33"), D("3.33"))
    assert got.lower == got.upper


@pytest.mark.parametrize(("lo", "hi"), [("0.4", "0.2")])
def test_between_inverted_raises(lo: str, hi: str) -> None:
    with pytest.raises(ValueError, match="strike_lo"):
        reported_interval(C.BETWEEN, D(lo), D(hi), 1)


def test_between_missing_hi_raises() -> None:
    with pytest.raises(ValueError, match="strike_hi"):
        reported_interval(C.BETWEEN, D("0.2"), None, 1)


@given(k=st.integers(-100000, 100000), precision=st.sampled_from([1, 0, -3]))
def test_gt_and_lte_partition_line(k: int, precision: int) -> None:
    s = D(k) * D(1).scaleb(-precision)
    gt = reported_interval(C.GT, s, None, precision)
    lte = reported_interval(C.LTE, s, None, precision)
    assert gt.lower == lte.upper
    assert gt.upper is None
    assert lte.lower is None


@given(k=st.integers(-100000, 100000), precision=st.sampled_from([1, 0, -3]))
def test_gte_and_lt_partition_line(k: int, precision: int) -> None:
    s = D(k) * D(1).scaleb(-precision)
    gte = reported_interval(C.GTE, s, None, precision)
    lt = reported_interval(C.LT, s, None, precision)
    assert gte.lower == lt.upper


_ORACLE = {
    C.GT: lambda r, s: r > s,
    C.GTE: lambda r, s: r >= s,
    C.LT: lambda r, s: r < s,
    C.LTE: lambda r, s: r <= s,
    C.EQ: lambda r, s: r == s,
}


@given(
    precision=st.sampled_from([1, 0, -3]),
    comparator=st.sampled_from(list(_ORACLE)),
    n=st.integers(-40, 40),
    m=st.integers(-320, 320),
)
# Pinned so a floor -> truncate-toward-zero mutant fails every run, not just on lucky seeds:
# GT -0.975 on the 0.1 grid needs X >= -0.95, so X = -0.9 is in and truncation (-0.85) drops it.
@example(precision=1, comparator=Comparator.GT, n=-39, m=-72)
@example(precision=1, comparator=Comparator.GT, n=-39, m=-120)
def test_interval_matches_rounding_oracle(
    precision: int, comparator: Comparator, n: int, m: int
) -> None:
    g = D(1).scaleb(-precision)
    s = D(n) * g / 4  # on or off grid, negatives and zero included
    x = D(m) * g / 8  # includes grid midpoints and quarter points
    q = x / g
    if q - q.to_integral_value(rounding="ROUND_FLOOR") == D("0.5"):
        return  # exact rounding tie: measure zero for continuous values
    r = q.to_integral_value(rounding=ROUND_HALF_UP) * g
    iv = reported_interval(comparator, s, None, precision)
    inside = (iv.lower is None or iv.lower <= x) and (iv.upper is None or x < iv.upper)
    assert inside == _ORACLE[comparator](r, s)

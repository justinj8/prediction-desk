from collections.abc import Callable
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from prediction_desk.domain.types import Prob
from prediction_desk.forecasting import distributions as dmod
from prediction_desk.forecasting.distributions import (
    Categorical,
    Normal,
    StudentT,
    affine,
)

D = Decimal


def test_normal_symmetry() -> None:
    assert Normal(D(0), D(1)).prob_between(None, D(0)) == Prob(D("0.50000"))


def test_student_t_matches_scipy() -> None:
    # scipy.stats.t.sf((0.35 - 0.2) / 0.15, 5) = sf(1.0, 5), computed once offline.
    expected = Prob(D("0.18161"))
    got = StudentT(D("0.2"), D("0.15"), D(5)).prob_between(D("0.35"), None)
    assert got == expected


def test_unbounded_and_degenerate_intervals() -> None:
    n = Normal(D(0), D(1))
    assert n.prob_between(None, None) == Prob(D("1.00000"))
    assert n.prob_between(D(1), D(1)) == Prob(D("0.00000"))
    with pytest.raises(ValueError):
        n.prob_between(D(2), D(1))


@given(
    a=st.decimals(-5, 5, places=2),
    w1=st.decimals(0, 3, places=2),
    w2=st.decimals(0, 3, places=2),
)
def test_prob_between_monotone(a: Decimal, w1: Decimal, w2: Decimal) -> None:
    for dist in (Normal(D(0), D(1)), StudentT(D(0), D(1), D(4))):
        inner = dist.prob_between(a, a + w1)
        outer = dist.prob_between(a - w2, a + w1 + w2)
        assert outer >= inner


@given(a=st.decimals(-5, 5, places=2), w=st.decimals("0.01", 4, places=2))
def test_partition_sums_to_one(a: Decimal, w: Decimal) -> None:
    b = a + w
    for dist in (Normal(D("0.1"), D("0.7")), StudentT(D(0), D(1), D(3))):
        total = dist.prob_between(None, a) + dist.prob_between(a, b) + dist.prob_between(b, None)
        assert abs(total - 1) <= D("0.00002")


def test_float_edge_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    n = Normal(D(0), D(1))
    # cdf(upper) - cdf(lower): 1.0000000000000002 - 0 and -1e-17 - 0
    monkeypatch.setattr(dmod, "_cdf", lambda dist, x: 1.0000000000000002 if x == 1 else 0.0)
    assert n.prob_between(D(0), D(1)) == Prob(D("1.00000"))
    monkeypatch.setattr(dmod, "_cdf", lambda dist, x: -1e-17 if x == 1 else 0.0)
    assert n.prob_between(D(0), D(1)) == Prob(D("0.00000"))
    assert dmod._prob(1.0000000000000002) == Prob(D("1.00000"))
    assert dmod._prob(-1e-17) == Prob(D("0.00000"))


def _cat() -> Categorical:
    return Categorical(((-25, D("0.2")), (0, D("0.5")), (25, D("0.3"))))


def test_categorical_validation() -> None:
    with pytest.raises(ValueError):
        Categorical(())
    with pytest.raises(ValueError):
        Categorical(((0, D("0.5")), (-25, D("0.5"))))
    with pytest.raises(ValueError):
        Categorical(((0, D("0.5")), (0, D("0.5"))))
    with pytest.raises(ValueError):
        Categorical(((0, D("0.5")), (25, D("0.4"))))
    with pytest.raises(ValueError):
        Categorical(((0, D("-0.1")), (25, D("1.1"))))
    with pytest.raises(ValueError):
        Categorical(((0, D("NaN")), (25, D("1"))))


def test_categorical_prob_between_and_where() -> None:
    c = _cat()
    assert c.prob_between(None, None) == Prob(D("1.00000"))
    assert c.prob_between(D(-25), D(25)) == Prob(D("0.70000"))
    assert c.prob_between(None, D(0)) == Prob(D("0.20000"))
    assert c.prob_between(D(0), D(0)) == Prob(D("0.00000"))
    with pytest.raises(ValueError):
        c.prob_between(D(1), D(0))


def test_categorical_prob_where() -> None:
    c = _cat()
    pred: Callable[[int], bool] = lambda o: o >= 0  # noqa: E731
    assert c.prob_where(pred) == Prob(D("0.80000"))
    assert c.prob_where(lambda o: False) == Prob(D("0.00000"))
    assert c.prob_where(lambda o: True) == Prob(D("1.00000"))


def test_affine_student_t() -> None:
    t = affine(StudentT(D("0.2"), D("0.15"), D(5)), D(1), D(2))
    assert t == StudentT(D("1.4"), D("0.30"), D(5))
    n = affine(Normal(D(1), D(2)), D(-1), D(3))
    assert n == Normal(D(2), D(6))
    for b in (D(0), D(-1)):
        with pytest.raises(ValueError):
            affine(n, D(0), b)
    with pytest.raises(ValueError):
        affine(n, D("Infinity"), D(1))


def test_validation_of_params() -> None:
    with pytest.raises(ValueError):
        Normal(D(0), D(0))
    with pytest.raises(ValueError):
        Normal(D("NaN"), D(1))
    with pytest.raises(ValueError):
        StudentT(D(0), D(1), D(0))
    with pytest.raises(ValueError):
        StudentT(D(0), D(-1), D(3))


def test_params_json_safe() -> None:
    for dist in (Normal(D("0.1"), D(1)), StudentT(D(0), D(1), D(5)), _cat()):
        p = dist.params()
        if not isinstance(dist, Categorical):
            assert list(p) == sorted(p)
        assert all(isinstance(v, str) for v in p.values())
        assert all(isinstance(k, str) for k in p)
        for v in p.values():
            D(v)
    assert Normal(D("0.1"), D(1)).family == "normal"
    assert StudentT(D(0), D(1), D(5)).family == "student_t"
    assert _cat().family == "categorical"
    assert _cat().params() == {"-25": "0.2", "0": "0.5", "25": "0.3"}


def test_categorical_params_ordered_numerically_not_lexicographically() -> None:
    # str-sorted order would be -25, -50, 0, 100, 25 (and "100" before "25").
    cat = Categorical(
        (
            (-50, D("0.1")),
            (-25, D("0.2")),
            (0, D("0.4")),
            (25, D("0.2")),
            (100, D("0.1")),
        )
    )
    assert list(cat.params()) == ["-50", "-25", "0", "25", "100"]

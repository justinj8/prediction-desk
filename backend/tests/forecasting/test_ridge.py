import random
from decimal import Decimal

import numpy as np
import pytest

from prediction_desk.forecasting.ridge import fit_ridge, oos_residuals, oos_sigma, predict

D = Decimal


def col(xs: list[int]) -> list[list[Decimal]]:
    return [[D(x)] for x in xs]


def test_fit_ridge_recovers_linear() -> None:
    xs = list(range(10))
    X = col(xs)
    y = [D(1 + 2 * x) for x in xs]
    fit = fit_ridge(X, y, D("1e-9"))
    assert abs(float(predict(fit, [D(20)])) - 41.0) < 1e-6
    assert abs(float(predict(fit, [D(0)])) - 1.0) < 1e-6
    # raw-space slope recovered: coef/scale
    assert abs(float(fit.coef[0] / fit.scale[0]) - 2.0) < 1e-6


def test_ridge_outputs_are_decimal() -> None:
    fit = fit_ridge(col([1, 2, 3, 4]), [D("1.5"), D("2.5"), D("3.5"), D("4.5")], D("0.1"))
    p = predict(fit, [D(5)])
    assert isinstance(p, Decimal)
    assert isinstance(fit.intercept, float)
    assert fit.coef.dtype == np.float64
    res = oos_residuals(col([1, 2, 3, 4, 5, 6]), [D(x) for x in (1, 2, 3, 4, 5, 6)], D("0.1"), 3)
    assert res and all(isinstance(r, Decimal) for r in res)
    assert isinstance(oos_sigma(res, 5, D("0.01")), Decimal)


def test_constant_column_scale_is_one() -> None:
    fit = fit_ridge([[D(1), D(1)], [D(2), D(1)], [D(3), D(1)]], [D(1), D(2), D(3)], D("0.001"))
    assert fit.scale[1] == 1.0
    assert abs(float(predict(fit, [D(4), D(1)])) - 4.0) < 0.05


def test_ridge_shrinks_with_alpha() -> None:
    X = col(list(range(8)))
    y = [D(2 * x) for x in range(8)]
    lo = fit_ridge(X, y, D("0.001"))
    hi = fit_ridge(X, y, D("1000"))
    assert abs(hi.coef[0]) < abs(lo.coef[0])


@pytest.mark.parametrize(
    ("X", "y", "alpha"),
    [
        ([], [], D(1)),
        ([[D(1)], [D(2), D(3)]], [D(1), D(2)], D(1)),
        ([[D(1)], [D(2)]], [D(1)], D(1)),
        ([[D(1)], [D(2)]], [D(1), D(2)], D(-1)),
    ],
)
def test_fit_ridge_validation(X: list[list[Decimal]], y: list[Decimal], alpha: Decimal) -> None:
    with pytest.raises(ValueError):
        fit_ridge(X, y, alpha)


def test_oos_residuals_expanding() -> None:
    xs = list(range(8))
    X = col(xs)
    y = [D(3 * x + 1) for x in xs]
    res = oos_residuals(X, y, D("1e-9"), 4)
    assert len(res) == 4
    assert all(abs(float(r)) < 1e-6 for r in res)
    assert oos_residuals(X, y, D("1e-9"), 8) == []
    assert oos_residuals(X, y, D("1e-9"), 20) == []


def test_oos_residuals_uses_only_past() -> None:
    # change a future y; earlier residuals must be identical
    X = col(list(range(8)))
    y1 = [D(x * x) for x in range(8)]
    y2 = list(y1)
    y2[7] = D(1000)
    r1 = oos_residuals(X, y1, D("0.5"), 4)
    r2 = oos_residuals(X, y2, D("0.5"), 4)
    assert r1[:3] == r2[:3]
    assert r1[3] != r2[3]


def test_oos_sigma_floor() -> None:
    assert oos_sigma([D(0)] * 10, 5, D("0.05")) == D("0.05")


def test_oos_sigma_rms_window() -> None:
    res = [D(100), D(3), D(-4)]
    # last 2: sqrt((9+16)/2) = 3.535534
    assert oos_sigma(res, 2, D("0.01")) == D("3.535534")
    assert oos_sigma(res, 2, D("10")) == D("10")


def test_oos_sigma_empty_raises() -> None:
    with pytest.raises(ValueError):
        oos_sigma([], 5, D("0.01"))


def _noisy(n: int = 12) -> tuple[list[list[Decimal]], list[Decimal]]:
    rng = random.Random(7)
    X = [[D(i), D(rng.randint(0, 9))] for i in range(n)]
    y = [D(2 * i + 1) + D(rng.randint(-50, 50)) / D(10) for i in range(n)]
    return X, y


def test_oos_residuals_match_out_of_sample_definition_exactly() -> None:
    X, y = _noisy()
    alpha = D("0.7")
    res = oos_residuals(X, y, alpha, 5)
    assert len(res) == len(y) - 5
    for k, i in enumerate(range(5, len(y))):
        assert res[k] == y[i] - predict(fit_ridge(X[:i], y[:i], alpha), X[i])


def test_oos_residual_perturbation_shifts_only_its_own_residual() -> None:
    X, y = _noisy()
    alpha = D("0.7")
    base = oos_residuals(X, y, alpha, 5)
    delta = D("3")
    i = 8
    y2 = list(y)
    y2[i] += delta
    pert = oos_residuals(X, y2, alpha, 5)
    k = i - 5
    assert pert[:k] == base[:k]  # earlier residuals unchanged
    assert pert[k] == base[k] + delta  # own residual shifts by exactly delta


def test_oos_residuals_length_mismatch() -> None:
    with pytest.raises(ValueError):
        oos_residuals(col([1, 2, 3]), [D(1), D(2)], D(1), 1)


@pytest.mark.parametrize("bad", [D("NaN"), D("Infinity"), D("-Infinity")])
def test_non_finite_rejected(bad: Decimal) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        fit_ridge([[D(1)], [bad]], [D(1), D(2)], D(1))
    with pytest.raises(ValueError, match="non-finite"):
        fit_ridge([[D(1)], [D(2)]], [D(1), bad], D(1))
    fit = fit_ridge(col([1, 2, 3]), [D(1), D(2), D(3)], D(1))
    with pytest.raises(ValueError, match="non-finite"):
        predict(fit, [bad])


def test_singular_system_raises_value_error() -> None:
    # duplicate columns + alpha=0 -> singular Gram matrix
    X = [[D(i), D(i)] for i in range(5)]
    y = [D(i) for i in range(5)]
    with pytest.raises(ValueError, match="singular") as ei:
        fit_ridge(X, y, D(0))
    assert ei.value.__cause__ is None and ei.value.__suppress_context__

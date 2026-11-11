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

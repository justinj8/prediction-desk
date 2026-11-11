"""Ridge regression. The float kernel is private; public functions take and return Decimal."""

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
import numpy.typing as npt

_Q = Decimal("0.000001")

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class RidgeFit:
    """Internal fit state: coefficients live in standardized-feature space."""

    coef: FloatArray
    intercept: float
    mean: FloatArray
    scale: FloatArray


def _matrix(X: Sequence[Sequence[Decimal]]) -> FloatArray:
    if len(X) == 0:
        raise ValueError("X is empty")
    width = len(X[0])
    if width == 0 or any(len(row) != width for row in X):
        raise ValueError("X rows must be non-empty and of equal length")
    return np.array([[float(v) for v in row] for row in X], dtype=np.float64)


def fit_ridge(X: Sequence[Sequence[Decimal]], y: Sequence[Decimal], alpha: Decimal) -> RidgeFit:
    """Standardized features, unpenalized intercept (= mean y), closed-form solve."""
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    Xf = _matrix(X)
    if len(y) != Xf.shape[0]:
        raise ValueError(f"len(X)={Xf.shape[0]} != len(y)={len(y)}")
    yf = np.array([float(v) for v in y], dtype=np.float64)
    mean = Xf.mean(axis=0)
    scale = Xf.std(axis=0, ddof=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    Xs = (Xf - mean) / scale
    y_bar = float(yf.mean())
    gram = Xs.T @ Xs + float(alpha) * np.eye(Xs.shape[1])
    coef = np.linalg.solve(gram, Xs.T @ (yf - y_bar))
    return RidgeFit(coef=coef, intercept=y_bar, mean=mean, scale=scale)


def predict(fit: RidgeFit, x: Sequence[Decimal]) -> Decimal:
    xf = np.array([float(v) for v in x], dtype=np.float64)
    if xf.shape != fit.mean.shape:
        raise ValueError(f"expected {fit.mean.shape[0]} features, got {xf.shape[0]}")
    value = fit.intercept + float(((xf - fit.mean) / fit.scale) @ fit.coef)
    return Decimal(format(value, ".12f"))


def oos_residuals(
    X: Sequence[Sequence[Decimal]],
    y: Sequence[Decimal],
    alpha: Decimal,
    min_train: int,
) -> list[Decimal]:
    """Expanding-window one-step-ahead residuals: fit on [0:i), score i, for i >= min_train."""
    if len(X) != len(y):
        raise ValueError(f"len(X)={len(X)} != len(y)={len(y)}")
    out: list[Decimal] = []
    for i in range(min_train, len(X)):
        fit = fit_ridge(X[:i], y[:i], alpha)
        out.append(y[i] - predict(fit, X[i]))
    return out


def oos_sigma(residuals: Sequence[Decimal], window: int, floor: Decimal) -> Decimal:
    """max(floor, RMS of the last `window` residuals). Empty input raises (ruling R5)."""
    if not residuals:
        raise ValueError("oos_sigma requires at least one out-of-sample residual")
    if window < 1:
        raise ValueError("window must be >= 1")
    tail = residuals[-window:]
    rms = (sum((r * r for r in tail), Decimal(0)) / len(tail)).sqrt().quantize(_Q)
    return max(floor, rms)

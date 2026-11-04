from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from prediction_desk.domain.types import cents, require_utc, to_prob


def test_cents_accepts_int_rejects_float_and_bool() -> None:
    assert cents(1050) == 1050
    with pytest.raises(TypeError):
        cents(10.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        cents(True)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0.123456", Decimal("0.12346")),
        ("0.5", Decimal("0.50000")),
        (1, Decimal("1.00000")),
        (0, Decimal("0.00000")),
    ],
)
def test_to_prob_quantizes(raw: str | int, expected: Decimal) -> None:
    assert to_prob(raw) == expected


@pytest.mark.parametrize("bad", ["-0.00001", "1.00001", "NaN", "Infinity"])
def test_to_prob_rejects_out_of_range_and_non_finite(bad: str) -> None:
    with pytest.raises(ValueError):
        to_prob(bad)


def test_to_prob_rejects_float_and_bool() -> None:
    with pytest.raises(TypeError):
        to_prob(0.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        to_prob(True)


@given(st.decimals(min_value=0, max_value=1, allow_nan=False, allow_infinity=False))
def test_to_prob_always_in_unit_interval_with_5dp(d: Decimal) -> None:
    p = to_prob(d)
    assert Decimal(0) <= p <= Decimal(1)
    assert p.as_tuple().exponent == -5


def test_require_utc() -> None:
    est = timezone(timedelta(hours=-5))
    assert require_utc(datetime(2024, 1, 1, 7, tzinfo=est)) == datetime(2024, 1, 1, 12, tzinfo=UTC)
    with pytest.raises(ValueError, match="timezone"):
        require_utc(datetime(2024, 1, 1))

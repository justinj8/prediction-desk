from datetime import UTC, datetime, timedelta

import pytest

from prediction_desk.clock import Clock, SimClock, WallClock

T0 = datetime(2024, 3, 12, 12, 30, tzinfo=UTC)


def test_wallclock_is_utc_aware() -> None:
    assert WallClock().now().tzinfo is UTC


def test_simclock_returns_start_and_advances() -> None:
    c = SimClock(T0)
    assert c.now() == T0
    c.advance(timedelta(minutes=5))
    assert c.now() == T0 + timedelta(minutes=5)


def test_simclock_set_forward_ok_backward_rejected() -> None:
    c = SimClock(T0)
    c.set(T0 + timedelta(hours=1))
    with pytest.raises(ValueError, match="backward"):
        c.set(T0)
    with pytest.raises(ValueError, match="backward"):
        c.advance(timedelta(seconds=-1))


def test_simclock_rejects_naive_datetimes() -> None:
    with pytest.raises(ValueError, match="timezone"):
        SimClock(datetime(2024, 3, 12, 12, 30))
    with pytest.raises(ValueError, match="timezone"):
        SimClock(T0).set(datetime(2024, 3, 13))


def test_both_satisfy_protocol() -> None:
    clocks: list[Clock] = [WallClock(), SimClock(T0)]
    assert all(isinstance(c.now(), datetime) for c in clocks)
    assert all(isinstance(c, Clock) for c in clocks)

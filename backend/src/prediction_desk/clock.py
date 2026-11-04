"""Injectable time source so live and replay runs share code without look-ahead."""

from datetime import UTC, datetime, timedelta
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    def now(self) -> datetime: ...


class WallClock:
    """Real time, timezone-aware UTC."""

    def now(self) -> datetime:
        return datetime.now(UTC)


def _require_aware(ts: datetime) -> datetime:
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return ts.astimezone(UTC)


class SimClock:
    """Simulated time that only moves forward."""

    def __init__(self, start: datetime) -> None:
        self._now = _require_aware(start)

    def now(self) -> datetime:
        return self._now

    def set(self, ts: datetime) -> None:
        new = _require_aware(ts)
        if new < self._now:
            raise ValueError(f"cannot move clock backward: {new} < {self._now}")
        self._now = new

    def advance(self, delta: timedelta) -> None:
        if delta < timedelta(0):
            raise ValueError(f"cannot move clock backward by {delta}")
        self._now = self._now + delta

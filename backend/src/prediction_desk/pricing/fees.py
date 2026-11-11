"""Versioned venue fee schedules and the per-order fee formula."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from functools import cache
from importlib.resources import files
from typing import Any, Literal, cast
from zoneinfo import ZoneInfo

import yaml

from prediction_desk.domain.enums import Venue
from prediction_desk.domain.types import require_utc

_NY = ZoneInfo("America/New_York")
_FEE_YAML = "fee_schedules.yaml"
_KINDS = ("kalshi_quadratic", "zero")
_ONE = Decimal(1)

FeeKind = Literal["kalshi_quadratic", "zero"]


@dataclass(frozen=True)
class FeeSchedule:
    venue: Venue
    effective_from: date
    kind: FeeKind
    rate: Decimal


def _parse_row(row: Any) -> FeeSchedule:
    if not isinstance(row, dict):
        raise ValueError(f"fee schedule entry must be a mapping: {row!r}")
    try:
        venue = Venue(str(row.get("venue")))
    except ValueError:
        raise ValueError(f"unknown venue: {row.get('venue')!r}") from None
    kind = row.get("kind")
    if kind not in _KINDS:
        raise ValueError(f"unknown kind: {kind!r}")
    try:
        effective_from = date.fromisoformat(str(row.get("effective_from")))
    except ValueError:
        raise ValueError(f"invalid effective_from: {row.get('effective_from')!r}") from None
    if kind == "kalshi_quadratic" and "rate" not in row:
        raise ValueError("kalshi_quadratic requires an explicit rate")
    raw_rate = row.get("rate", "0")
    try:
        if not isinstance(raw_rate, str):
            raise TypeError("rate must be a quoted decimal string")
        rate = Decimal(raw_rate)
    except (TypeError, InvalidOperation):
        raise ValueError(f"invalid rate: {raw_rate!r}") from None
    if not rate.is_finite() or rate < 0:
        raise ValueError(f"invalid rate: {raw_rate!r}")
    if kind == "kalshi_quadratic" and rate == 0:
        raise ValueError("kalshi_quadratic rate must be > 0")
    if kind == "zero" and rate != 0:
        raise ValueError("zero schedule must not have a nonzero rate")
    return FeeSchedule(venue, effective_from, cast(FeeKind, kind), rate)


@cache
def _packaged_schedules() -> tuple[FeeSchedule, ...]:
    return tuple(load_fee_schedules())


def load_fee_schedules(text: str | None = None) -> list[FeeSchedule]:
    """Parse and validate fee schedules; ``text=None`` reads the packaged YAML."""
    if text is None:
        text = files("prediction_desk.pricing").joinpath(_FEE_YAML).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict) or not isinstance(data.get("schedules"), list):
        raise ValueError("fee schedule document must have a 'schedules' list")
    out: list[FeeSchedule] = []
    last: dict[Venue, date] = {}
    for row in data["schedules"]:
        sched = _parse_row(row)
        prev = last.get(sched.venue)
        if prev is not None and sched.effective_from == prev:
            raise ValueError(f"duplicate schedule for {sched.venue} on {prev}")
        if prev is not None and sched.effective_from < prev:
            raise ValueError(f"{sched.venue} schedules must be in ascending date order")
        last[sched.venue] = sched.effective_from
        out.append(sched)
    return out


def schedule_for(
    venue: Venue, asof: datetime, schedules: Sequence[FeeSchedule] | None = None
) -> FeeSchedule:
    """Latest schedule for ``venue`` with ``effective_from`` <= the as-of New York date."""
    pool = _packaged_schedules() if schedules is None else schedules
    day = require_utc(asof).astimezone(_NY).date()
    eligible = [s for s in pool if s.venue == venue and s.effective_from <= day]
    if not eligible:
        raise LookupError(f"no fee schedule for {venue} as of {day}")
    return max(eligible, key=lambda s: s.effective_from)


def fee_cents(schedule: FeeSchedule, price: Decimal, qty: int) -> int:
    """Fee in whole cents for buying ``qty`` contracts at ``price`` (exact Decimal, ceiling)."""
    if qty < 1:
        raise ValueError(f"qty must be >= 1: {qty}")
    if not (Decimal(0) <= price <= _ONE):
        raise ValueError(f"price must be in [0, 1]: {price}")
    if schedule.kind == "zero":
        return 0
    raw = schedule.rate * qty * price * (_ONE - price) * 100
    return int(raw.to_integral_value(ROUND_CEILING))


def fill_fee_cents(
    schedule: FeeSchedule, avg_price: Decimal, qty: int, levels: Sequence[tuple[Decimal, Decimal]]
) -> int:
    """Fee for a multi-level fill, never understated: the larger of the order-level fee at the
    average price and the sum of per-level fees (``levels`` are ``(price, take)``; ``take`` may
    be fractional)."""
    order_fee = fee_cents(schedule, avg_price, qty)
    if schedule.kind == "zero":
        return 0
    per_level = sum(
        int((schedule.rate * take * p * (_ONE - p) * 100).to_integral_value(ROUND_CEILING))
        for p, take in levels
    )
    return max(order_fee, per_level)

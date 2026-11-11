from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from prediction_desk.domain.enums import Venue
from prediction_desk.pricing.fees import (
    FeeSchedule,
    fee_cents,
    load_fee_schedules,
    schedule_for,
)

D = Decimal

TWO_KALSHI = """
schedules:
  - {venue: kalshi, effective_from: "2024-01-01", kind: kalshi_quadratic, rate: "0.07"}
  - {venue: kalshi, effective_from: "2026-01-01", kind: kalshi_quadratic, rate: "0.05"}
  - {venue: polymarket, effective_from: "2024-01-01", kind: zero, rate: "0"}
"""


def kalshi(rate: str = "0.07") -> FeeSchedule:
    return FeeSchedule(Venue.KALSHI, date(2024, 1, 1), "kalshi_quadratic", D(rate))


def test_kalshi_fee_examples() -> None:
    assert fee_cents(kalshi(), D("0.50"), 10) == 18
    assert fee_cents(kalshi(), D("0.94"), 1) == 1
    assert fee_cents(kalshi(), D("0.50"), 100) == 175


def test_kalshi_fee_extreme_prices_are_zero() -> None:
    assert fee_cents(kalshi(), D(0), 5) == 0
    assert fee_cents(kalshi(), D(1), 5) == 0


def test_polymarket_zero_fee() -> None:
    sched = FeeSchedule(Venue.POLYMARKET, date(2024, 1, 1), "zero", D(0))
    assert fee_cents(sched, D("0.50"), 10) == 0


def test_fee_rejects_bad_qty_and_price() -> None:
    with pytest.raises(ValueError, match="qty"):
        fee_cents(kalshi(), D("0.5"), 0)
    with pytest.raises(ValueError, match="price"):
        fee_cents(kalshi(), D("1.01"), 1)
    with pytest.raises(ValueError, match="price"):
        fee_cents(kalshi(), D("-0.01"), 1)


def test_schedule_asof_picks_latest_effective() -> None:
    scheds = load_fee_schedules(TWO_KALSHI)
    old = schedule_for(Venue.KALSHI, datetime(2025, 6, 1, tzinfo=UTC), scheds)
    new = schedule_for(Venue.KALSHI, datetime(2026, 6, 1, tzinfo=UTC), scheds)
    assert old.rate == D("0.07")
    assert new.rate == D("0.05")


def test_schedule_asof_uses_new_york_date() -> None:
    scheds = load_fee_schedules(TWO_KALSHI)
    # 2026-01-01 03:00 UTC is still 2025-12-31 in New York.
    before = schedule_for(Venue.KALSHI, datetime(2026, 1, 1, 3, tzinfo=UTC), scheds)
    after = schedule_for(Venue.KALSHI, datetime(2026, 1, 1, 6, tzinfo=UTC), scheds)
    assert before.rate == D("0.07")
    assert after.rate == D("0.05")


def test_schedule_before_first_effective_raises_lookup() -> None:
    scheds = load_fee_schedules(TWO_KALSHI)
    with pytest.raises(LookupError):
        schedule_for(Venue.KALSHI, datetime(2023, 12, 31, 20, tzinfo=UTC), scheds)


def test_schedule_naive_asof_rejected() -> None:
    with pytest.raises(ValueError, match="timezone"):
        schedule_for(Venue.KALSHI, datetime(2025, 1, 1), load_fee_schedules(TWO_KALSHI))


def test_schedule_unknown_venue_in_list_raises_lookup() -> None:
    only_kalshi = load_fee_schedules(TWO_KALSHI)[:2]
    with pytest.raises(LookupError):
        schedule_for(Venue.POLYMARKET, datetime(2025, 1, 1, tzinfo=UTC), only_kalshi)


def test_fee_yaml_packaged() -> None:
    scheds = load_fee_schedules()
    assert {s.venue for s in scheds} == {Venue.KALSHI, Venue.POLYMARKET}
    asof = datetime(2026, 11, 11, tzinfo=UTC)
    assert schedule_for(Venue.KALSHI, asof).rate == D("0.07")
    assert schedule_for(Venue.POLYMARKET, asof).kind == "zero"


@pytest.mark.parametrize(
    ("entry", "msg"),
    [
        ('{venue: nyse, effective_from: "2024-01-01", kind: zero, rate: "0"}', "venue"),
        ('{venue: kalshi, effective_from: "2024-01-01", kind: flat, rate: "0"}', "kind"),
        ('{venue: kalshi, effective_from: "2024-01-01", kind: zero, rate: "abc"}', "rate"),
        ('{venue: kalshi, effective_from: "2024-01-01", kind: zero, rate: "-0.1"}', "rate"),
        ('{venue: kalshi, effective_from: "2024-01-01", kind: zero, rate: "NaN"}', "rate"),
        ('{venue: kalshi, effective_from: "2024-01-01", kind: zero, rate: 0.07}', "rate"),
        ('{venue: kalshi, effective_from: "not-a-date", kind: zero, rate: "0"}', "effective_from"),
    ],
)
def test_load_rejects_bad_entry(entry: str, msg: str) -> None:
    with pytest.raises(ValueError, match=msg):
        load_fee_schedules(f"schedules:\n  - {entry}\n")


def test_load_rejects_duplicates() -> None:
    entry = '{venue: kalshi, effective_from: "2024-01-01", kind: zero, rate: "0"}'
    with pytest.raises(ValueError, match="duplicate"):
        load_fee_schedules(f"schedules:\n  - {entry}\n  - {entry}\n")


def test_load_rejects_unsorted_per_venue() -> None:
    text = """
schedules:
  - {venue: kalshi, effective_from: "2026-01-01", kind: zero, rate: "0"}
  - {venue: polymarket, effective_from: "2024-01-01", kind: zero, rate: "0"}
  - {venue: kalshi, effective_from: "2024-01-01", kind: zero, rate: "0"}
"""
    with pytest.raises(ValueError, match="ascending"):
        load_fee_schedules(text)


def test_load_rejects_malformed_document() -> None:
    with pytest.raises(ValueError, match="schedules"):
        load_fee_schedules("other: 1\n")


def test_load_rejects_non_mapping_entry() -> None:
    with pytest.raises(ValueError, match="mapping"):
        load_fee_schedules("schedules:\n  - just-a-string\n")

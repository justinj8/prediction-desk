"""Deterministic contract mapping and release linking.

Translates a venue market into a canonical (measure, comparator, strike, unit) and links it to
the macro release it resolves on. Anything that does not fit a rule is ``Unmapped(reason)``;
nothing here guesses.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from typing import Final, Literal

from prediction_desk.db.models import Release
from prediction_desk.domain.enums import MEASURE_SERIES, Comparator, Measure, Venue
from prediction_desk.markets.polymarket import EVENT_SLUG_PATTERNS
from prediction_desk.markets.types import VenueMarket

MAPPER_VERSION = "rules:v1"

Reason = Literal[
    "unknown_series",
    "unsupported_strike_type",
    "unparseable_strike",
    "rules_text_mismatch",
    "no_release",
    "ambiguous_release",
    "period_mismatch",
]

RELEASE_WINDOW = timedelta(hours=48)


@dataclass(frozen=True)
class ContractTerms:
    measure: Measure
    comparator: Comparator
    strike_lo: Decimal
    strike_hi: Decimal | None
    unit: str
    resolution_precision: int
    resolution_source: str


@dataclass(frozen=True)
class Unmapped:
    reason: Reason


# measure -> (unit, resolution_precision, resolution_source)
MEASURE_SPECS: Final[dict[Measure, tuple[str, int, str]]] = {
    Measure.CPI_MOM: ("pct", 1, "BLS"),
    Measure.CORE_CPI_MOM: ("pct", 1, "BLS"),
    Measure.CPI_YOY: ("pct", 1, "BLS"),
    Measure.CORE_CPI_YOY: ("pct", 1, "BLS"),
    Measure.NFP_CHANGE: ("jobs", 0, "BLS"),
    Measure.FOMC_CHANGE_BP: ("bp", 0, "Federal Reserve"),
}

KALSHI_MEASURES: Final[dict[str, Measure]] = {
    "KXCPI": Measure.CPI_MOM,
    "KXCPICORE": Measure.CORE_CPI_MOM,
    "KXCPIYOY": Measure.CPI_YOY,
    "KXCPICOREYOY": Measure.CORE_CPI_YOY,
    "KXPAYROLLS": Measure.NFP_CHANGE,
    "KXFEDDECISION": Measure.FOMC_CHANGE_BP,
}

# Same order as polymarket.EVENT_SLUG_PATTERNS (strict zip fails at import if that changes length).
POLYMARKET_MEASURES: Final[tuple[tuple[re.Pattern[str], Measure], ...]] = tuple(
    zip(
        EVENT_SLUG_PATTERNS,
        (Measure.CPI_YOY, Measure.CORE_CPI_YOY, Measure.FOMC_CHANGE_BP),
        strict=True,
    )
)

# Kalshi "greater" markets: the rules must state the comparator. The precision phrase varies
# between events, so it is deliberately not checked; precision comes from the measure.
KALSHI_GREATER_RULES: Final = re.compile(r"more than|above", re.IGNORECASE)
KALSHI_FOMC_KEYS: Final = ("Cut", "Hike")
KALSHI_FOMC_VALUE: Final = re.compile(r"(>)?([0-9]+)")

# Polymarket description sanity phrases. Every recorded us-cpi (annual and core) market contains
# "Consumer Price Index" and "12-month"; every recorded fomc market contains "upper bound".
POLYMARKET_RULES_PHRASES: Final[dict[Measure, tuple[str, ...]]] = {
    Measure.CPI_YOY: ("Consumer Price Index", "12-month"),
    Measure.CORE_CPI_YOY: ("Consumer Price Index", "12-month"),
    Measure.FOMC_CHANGE_BP: ("upper bound",),
}
POLYMARKET_CPI_TITLES: Final[tuple[tuple[re.Pattern[str], Comparator], ...]] = (
    (re.compile(r"≤([0-9]+\.[0-9])%"), Comparator.LTE),
    (re.compile(r"≥([0-9]+\.[0-9])%"), Comparator.GTE),
    (re.compile(r"([0-9]+\.[0-9])%"), Comparator.EQ),
)
POLYMARKET_FOMC_TITLE: Final = re.compile(r"([0-9]+)(\+)? bps (decrease|increase)")
POLYMARKET_FOMC_NO_CHANGE: Final = "No change"

MONTHS: Final[tuple[str, ...]] = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)
MONTH_NUMBERS: Final[dict[str, str]] = {name: f"{i:02d}" for i, name in enumerate(MONTHS, 1)}
MONTH_ABBREVIATIONS: Final[dict[str, str]] = {
    name[:3].upper(): num for name, num in MONTH_NUMBERS.items()
}

KALSHI_EVENT_PERIOD: Final = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})?$")
POLYMARKET_SLUG_PERIODS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"^(?P<month>[a-z]+)-inflation-us-annual"),
    re.compile(r"^core-cpi-yoy-(?P<month>[a-z]+)-(?P<year>\d{4})"),
    re.compile(r"^fed-decision-in-(?P<month>[a-z]+)(?:-(?P<year>\d{4})(?:-|$))?"),
)


def _terms(measure: Measure, comparator: Comparator, strike_lo: Decimal) -> ContractTerms:
    unit, precision, source = MEASURE_SPECS[measure]
    return ContractTerms(measure, comparator, strike_lo, None, unit, precision, source)


def _numeric(value: object) -> Decimal | None:
    """Decimal of an int/float/Decimal; None for anything else (including bool and NaN)."""
    if isinstance(value, bool) or not isinstance(value, int | float | Decimal):
        return None
    number = Decimal(str(value))
    return number if number.is_finite() else None


def _kalshi_greater(m: VenueMarket, measure: Measure) -> ContractTerms | Unmapped:
    floor = _numeric(m.strike.get("floor_strike"))
    if floor is None:
        return Unmapped("unparseable_strike")
    if not KALSHI_GREATER_RULES.search(m.rules_text):
        return Unmapped("rules_text_mismatch")
    return _terms(measure, Comparator.GT, floor)


def _kalshi_fomc(m: VenueMarket) -> ContractTerms | Unmapped:
    custom = m.strike.get("custom_strike")
    if not isinstance(custom, dict) or len(custom) != 1:
        return Unmapped("unparseable_strike")
    ((key, value),) = custom.items()
    parsed = KALSHI_FOMC_VALUE.fullmatch(value) if isinstance(value, str) else None
    if key not in KALSHI_FOMC_KEYS or parsed is None:
        return Unmapped("unparseable_strike")
    if key not in m.rules_text:
        return Unmapped("rules_text_mismatch")
    bp = int(parsed.group(2))
    open_ended = parsed.group(1) is not None
    if key == "Cut":
        return _terms(
            Measure.FOMC_CHANGE_BP, Comparator.LT if open_ended else Comparator.EQ, Decimal(-bp)
        )
    return _terms(
        Measure.FOMC_CHANGE_BP, Comparator.GT if open_ended else Comparator.EQ, Decimal(bp)
    )


def _map_kalshi(m: VenueMarket) -> ContractTerms | Unmapped:
    measure = KALSHI_MEASURES.get(m.series_code)
    if measure is None:
        return Unmapped("unknown_series")
    strike_type = m.strike.get("strike_type")
    if measure is Measure.FOMC_CHANGE_BP:
        return _kalshi_fomc(m) if strike_type == "custom" else Unmapped("unsupported_strike_type")
    if strike_type != "greater":
        return Unmapped("unsupported_strike_type")
    return _kalshi_greater(m, measure)


def _polymarket_cpi(measure: Measure, title: str) -> ContractTerms | Unmapped:
    for pattern, comparator in POLYMARKET_CPI_TITLES:
        if matched := pattern.fullmatch(title):
            return _terms(measure, comparator, Decimal(matched.group(1)))
    return Unmapped("unparseable_strike")


def _polymarket_fomc(title: str) -> ContractTerms | Unmapped:
    measure = Measure.FOMC_CHANGE_BP
    if title == POLYMARKET_FOMC_NO_CHANGE:
        return _terms(measure, Comparator.EQ, Decimal(0))
    matched = POLYMARKET_FOMC_TITLE.fullmatch(title)
    if matched is None:
        return Unmapped("unparseable_strike")
    amount, plus, direction = matched.groups()
    if direction == "decrease":
        return _terms(measure, Comparator.LTE if plus else Comparator.EQ, Decimal(-int(amount)))
    return _terms(measure, Comparator.GTE if plus else Comparator.EQ, Decimal(int(amount)))


def _map_polymarket(m: VenueMarket) -> ContractTerms | Unmapped:
    measure = next((ms for pattern, ms in POLYMARKET_MEASURES if pattern.match(m.event_id)), None)
    if measure is None:
        return Unmapped("unknown_series")
    title = m.strike.get("group_item_title")
    if not isinstance(title, str):
        return Unmapped("unparseable_strike")
    if measure is Measure.FOMC_CHANGE_BP:
        terms = _polymarket_fomc(title)
    else:
        terms = _polymarket_cpi(measure, title)
    if isinstance(terms, Unmapped):
        return terms
    if not all(phrase in m.rules_text for phrase in POLYMARKET_RULES_PHRASES[measure]):
        return Unmapped("rules_text_mismatch")
    return terms


def map_market(m: VenueMarket) -> ContractTerms | Unmapped:
    """Map a venue market to canonical contract terms, or say why it cannot be mapped."""
    if m.venue is Venue.KALSHI:
        return _map_kalshi(m)
    return _map_polymarket(m)


def _kalshi_period(event_id: str) -> str | None:
    """'YYYY-MM' from a Kalshi event ticker such as KXCPI-26NOV or FEDDECISION-24JAN31."""
    matched = KALSHI_EVENT_PERIOD.search(event_id)
    if matched is None:
        return None
    month = MONTH_ABBREVIATIONS.get(matched.group(2))
    return None if month is None else f"20{matched.group(1)}-{month}"


def _polymarket_period_matches(event_id: str, period: str) -> bool:
    """Slug month (and year, where the slug carries one) must agree with 'YYYY-MM[-DD]'."""
    for pattern in POLYMARKET_SLUG_PERIODS:
        if matched := pattern.match(event_id):
            month = MONTH_NUMBERS.get(matched.group("month"))
            year = matched.groupdict().get("year")
            return month == period[5:7] and (year is None or year == period[:4])
    return False


def _period_matches(m: VenueMarket, release: Release) -> bool:
    if m.venue is Venue.KALSHI:
        return _kalshi_period(m.event_id) == release.period[:7]
    return _polymarket_period_matches(m.event_id, release.period)


def link_release(
    m: VenueMarket, terms: ContractTerms, releases: Sequence[Release]
) -> Release | Unmapped:
    """The single release this market resolves on; never the nearest wrong one."""
    series = MEASURE_SERIES[terms.measure]
    candidates = [
        r
        for r in releases
        if r.series is series and abs(r.scheduled_at - m.close_time) <= RELEASE_WINDOW
    ]
    if not candidates:
        return Unmapped("no_release")
    if len(candidates) > 1:
        return Unmapped("ambiguous_release")
    release = candidates[0]
    return release if _period_matches(m, release) else Unmapped("period_mismatch")

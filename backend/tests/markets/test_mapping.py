import dataclasses
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from prediction_desk.clock import SimClock
from prediction_desk.db.models import Release
from prediction_desk.domain.enums import Comparator, Measure, Series, Venue
from prediction_desk.markets.kalshi import KALSHI_SERIES, KalshiAdapter
from prediction_desk.markets.mapping import (
    MAPPER_VERSION,
    POLYMARKET_EVENT_RULES,
    ContractTerms,
    Unmapped,
    link_release,
    map_market,
)
from prediction_desk.markets.polymarket import (
    EVENT_SLUG_PATTERNS,
    POLYMARKET_TAGS,
    PolymarketAdapter,
)
from prediction_desk.markets.types import VenueMarket
from prediction_desk.net.http import ResilientHttp

D = Decimal
CPI_RULES = (
    "If the Consumer Price Index (CPI) increases by more than -0.3% (single-decimal) "
    "in November 2026, then the market resolves to Yes."
)
PAYROLLS_RULES = "If nonfarm payrolls are above 90000 as reported by the Bureau of Labor Statistics"
PM_CPI_DESC = (
    "inflation over the 12-month period ending October 2026. The percentage change in the "
    "Consumer Price Index (CPI) per the BLS."
)
PM_CORE_DESC = (
    "core inflation (excluding food and energy) over the 12-month period ending October 2026. "
    "The percentage change in the Consumer Price Index for All Urban Consumers excluding "
    "food and energy (Core CPI-U)."
)
PM_FED_DESC = "defined by the upper bound of the target federal funds range."


def _kalshi(
    series: str = "KXCPI",
    event: str = "KXCPI-26NOV",
    rules: str = CPI_RULES,
    close: datetime = datetime(2026, 12, 10, 13, 29, tzinfo=UTC),
    **strike: Any,
) -> VenueMarket:
    full = {"strike_type": None, "floor_strike": None, "cap_strike": None, "custom_strike": None}
    full.update(strike)
    return VenueMarket(
        venue=Venue.KALSHI,
        market_id=f"{event}-T",
        event_id=event,
        series_code=series,
        title="t",
        rules_text=rules,
        close_time=close,
        status="active",
        strike=full,
        yes_token_id=None,
    )


def _poly(
    title: str,
    slug: str = "october-inflation-us-annual",
    tag: str = "us-cpi",
    desc: str = PM_CPI_DESC,
    close: datetime = datetime(2026, 11, 11, 4, 59, tzinfo=UTC),
) -> VenueMarket:
    return VenueMarket(
        venue=Venue.POLYMARKET,
        market_id="1",
        event_id=slug,
        series_code=tag,
        title="q",
        rules_text=desc,
        close_time=close,
        status="active",
        strike={"group_item_title": title},
        yes_token_id="111",
    )


def _release(series: Series, period: str, at: datetime) -> Release:
    return Release(series=series, period=period, scheduled_at=at)


def test_mapper_version() -> None:
    assert MAPPER_VERSION == "rules:v2"


@pytest.mark.parametrize("floor", [-0.3, D("-0.3")])
def test_kalshi_cpi_greater(floor: float | Decimal) -> None:
    m = _kalshi(strike_type="greater", floor_strike=floor)
    assert map_market(m) == ContractTerms(
        Measure.CPI_MOM, Comparator.GT, D("-0.3"), None, "pct", 1, "BLS"
    )


def test_kalshi_cpi_greater_precision_phrase_variants() -> None:
    rules = "If the CPI is above 0.2% as reported to one decimal place in Nov 2026"
    t = map_market(_kalshi(rules=rules, strike_type="greater", floor_strike=D("0.2")))
    assert isinstance(t, ContractTerms) and t.strike_lo == D("0.2")
    upper = _kalshi(rules="MORE THAN 0.2%", strike_type="greater", floor_strike=D("0.2"))
    assert isinstance(map_market(upper), ContractTerms)


@pytest.mark.parametrize(
    ("series", "measure"),
    [
        ("KXCPI", Measure.CPI_MOM),
        ("KXCPICORE", Measure.CORE_CPI_MOM),
        ("KXCPIYOY", Measure.CPI_YOY),
        ("KXCPICOREYOY", Measure.CORE_CPI_YOY),
    ],
)
def test_kalshi_cpi_measures(series: str, measure: Measure) -> None:
    t = map_market(_kalshi(series=series, strike_type="greater", floor_strike=D("3.1")))
    assert t == ContractTerms(measure, Comparator.GT, D("3.1"), None, "pct", 1, "BLS")


@pytest.mark.parametrize("floor", [90000, D("90000")])
def test_kalshi_payrolls_greater(floor: int | Decimal) -> None:
    m = _kalshi(
        "KXPAYROLLS", "KXPAYROLLS-26NOV", PAYROLLS_RULES, strike_type="greater", floor_strike=floor
    )
    assert map_market(m) == ContractTerms(
        Measure.NFP_CHANGE, Comparator.GT, D(90000), None, "jobs", -3, "BLS"
    )


def test_kalshi_payrolls_rules_mismatch() -> None:
    m = _kalshi("KXPAYROLLS", "KXPAYROLLS-26NOV", "nothing", strike_type="greater", floor_strike=1)
    assert map_market(m) == Unmapped("rules_text_mismatch")


@pytest.mark.parametrize(
    ("custom", "word", "comparator", "lo"),
    [
        ({"Cut": "25"}, "Cut", Comparator.EQ, -25),
        ({"Cut": ">25"}, "Cut", Comparator.LT, -25),
        ({"Hike": "0"}, "Hike", Comparator.EQ, 0),
        ({"Hike": "25"}, "Hike", Comparator.EQ, 25),
        ({"Hike": ">25"}, "Hike", Comparator.GT, 25),
        ({"Cut": "50"}, "Cut", Comparator.EQ, -50),
    ],
)
def test_kalshi_fed_custom(
    custom: dict[str, str], word: str, comparator: Comparator, lo: int
) -> None:
    m = _kalshi(
        "KXFEDDECISION",
        "KXFEDDECISION-26DEC",
        f"If the Federal Reserve does a {word} of 25bps on Dec 9",
        strike_type="custom",
        custom_strike=custom,
    )
    assert map_market(m) == ContractTerms(
        Measure.FOMC_CHANGE_BP, comparator, D(lo), None, "bp", 0, "Federal Reserve"
    )


@pytest.mark.parametrize(
    "custom",
    [
        {"Cut": "abc"},
        {"Cut": "-25"},
        {"Cut": ""},
        {"Cut": ">"},
        {"Cut": "2.5"},
        {"Hold": "0"},
        {"Cut": "25", "Hike": "25"},
        {},
        None,
        "Cut",
        {"Cut": 25},
    ],
)
def test_kalshi_fed_unparseable(custom: Any) -> None:
    m = _kalshi(
        "KXFEDDECISION",
        "KXFEDDECISION-26DEC",
        "a Cut or a Hike",
        strike_type="custom",
        custom_strike=custom,
    )
    assert map_market(m) == Unmapped("unparseable_strike")


def test_kalshi_fed_rules_missing_keyword_is_case_sensitive() -> None:
    m = _kalshi(
        "KXFEDDECISION",
        "KXFEDDECISION-26DEC",
        "the fed does a hike of 25bps",
        strike_type="custom",
        custom_strike={"Hike": "25"},
    )
    assert map_market(m) == Unmapped("rules_text_mismatch")
    m2 = _kalshi(
        "KXFEDDECISION",
        "KXFEDDECISION-26DEC",
        "the Fed does a Cut of 25bps",
        strike_type="custom",
        custom_strike={"Hike": "25"},
    )
    assert map_market(m2) == Unmapped("rules_text_mismatch")


def test_kalshi_rules_mismatch() -> None:
    m = _kalshi(rules="resolves on something else", strike_type="greater", floor_strike=D("0.1"))
    assert map_market(m) == Unmapped("rules_text_mismatch")


def test_kalshi_unsupported_strike_type() -> None:
    m = _kalshi(strike_type="between", floor_strike=D("0.1"), cap_strike=D("0.3"))
    assert map_market(m) == Unmapped("unsupported_strike_type")
    assert map_market(_kalshi(strike_type=None)) == Unmapped("unsupported_strike_type")


@pytest.mark.parametrize(
    ("floor", "expected"),
    [("0.3", D("0.3")), ("-0.3", D("-0.3")), ("90000", D("90000")), ("3.25", D("3.25"))],
)
def test_kalshi_numeric_string_floor(floor: str, expected: Decimal) -> None:
    t = map_market(_kalshi(strike_type="greater", floor_strike=floor))
    assert isinstance(t, ContractTerms) and t.strike_lo == expected


@pytest.mark.parametrize(
    "floor",
    [
        None,
        "abc",
        True,
        False,
        float("nan"),
        float("inf"),
        D("NaN"),
        D("-Infinity"),
        "NaN",
        "Infinity",
        "1e5",
        "+0.3",
        " 0.3",
        "0.3 ",
        ".3",
        "3.",
        "1" * 5000,
        "0." + "1" * 5000,
        "1234567890",
        10**10,
        D("1E+10"),
        "",
        "-",
    ],
)
def test_kalshi_unparseable_floor(floor: Any) -> None:
    m = _kalshi(strike_type="greater", floor_strike=floor)
    assert map_market(m) == Unmapped("unparseable_strike")


def test_unknown_series() -> None:
    assert map_market(_kalshi(series="KXOTHER", strike_type="greater", floor_strike=1)) == Unmapped(
        "unknown_series"
    )
    assert map_market(_poly("3.3%", slug="something-else")) == Unmapped("unknown_series")


@pytest.mark.parametrize(
    ("title", "comparator", "lo"),
    [
        ("≤3.2%", Comparator.LTE, D("3.2")),
        ("3.3%", Comparator.EQ, D("3.3")),
        ("≥4.3%", Comparator.GTE, D("4.3")),
    ],
)
def test_polymarket_cpi_titles(title: str, comparator: Comparator, lo: Decimal) -> None:
    assert map_market(_poly(title)) == ContractTerms(
        Measure.CPI_YOY, comparator, lo, None, "pct", 1, "BLS"
    )


def test_polymarket_core_cpi_measure() -> None:
    m = _poly("3.3%", slug="core-cpi-yoy-october-2026", desc=PM_CORE_DESC)
    t = map_market(m)
    assert isinstance(t, ContractTerms) and t.measure is Measure.CORE_CPI_YOY


@pytest.mark.parametrize(
    ("slug", "desc"),
    [
        ("core-cpi-yoy-october-2026", PM_CPI_DESC),  # core slug, headline rules
        ("october-inflation-us-annual", PM_CORE_DESC),  # headline slug, core rules
        ("october-inflation-us-annual", PM_CPI_DESC + " Excluding FOOD AND ENERGY."),
    ],
)
def test_polymarket_core_headline_discriminator(slug: str, desc: str) -> None:
    assert map_market(_poly("3.3%", slug=slug, desc=desc)) == Unmapped("rules_text_mismatch")


def test_polymarket_core_phrase_is_case_insensitive() -> None:
    desc = PM_CORE_DESC.replace("food and energy", "Food And Energy")
    t = map_market(_poly("3.3%", slug="core-cpi-yoy-october-2026", desc=desc))
    assert isinstance(t, ContractTerms) and t.measure is Measure.CORE_CPI_YOY


def test_polymarket_event_rules_are_the_single_slug_table() -> None:
    measures = [rule.measure for rule in POLYMARKET_EVENT_RULES]
    assert measures == [Measure.CPI_YOY, Measure.CORE_CPI_YOY, Measure.FOMC_CHANGE_BP]
    assert tuple(rule.slug for rule in POLYMARKET_EVENT_RULES) == EVENT_SLUG_PATTERNS


@pytest.mark.parametrize("title", ["3%", "<=3.2%", "≤ 3.2%", "3.25%", "≥3.2", "", "x3.2%", "3.2% "])
def test_polymarket_cpi_bad_titles(title: str) -> None:
    assert map_market(_poly(title)) == Unmapped("unparseable_strike")


def test_polymarket_cpi_description_mismatch() -> None:
    assert map_market(_poly("3.3%", desc="about the weather")) == Unmapped("rules_text_mismatch")
    no_12 = _poly("3.3%", desc="Consumer Price Index month over month")
    assert map_market(no_12) == Unmapped("rules_text_mismatch")


@pytest.mark.parametrize(
    ("title", "comparator", "lo"),
    [
        ("50+ bps decrease", Comparator.LTE, -50),
        ("25 bps decrease", Comparator.EQ, -25),
        ("No change", Comparator.EQ, 0),
        ("25 bps increase", Comparator.EQ, 25),
        ("50+ bps increase", Comparator.GTE, 50),
    ],
)
def test_polymarket_fed_titles(title: str, comparator: Comparator, lo: int) -> None:
    m = _poly(title, slug="fed-decision-in-march-2027", tag="fomc", desc=PM_FED_DESC)
    assert map_market(m) == ContractTerms(
        Measure.FOMC_CHANGE_BP, comparator, D(lo), None, "bp", 0, "Federal Reserve"
    )


@pytest.mark.parametrize(
    "title", ["no change", "25bps increase", "25 bps hold", "+25 bps increase", ""]
)
def test_polymarket_fed_bad_titles(title: str) -> None:
    m = _poly(title, slug="fed-decision-in-march-2027", tag="fomc", desc=PM_FED_DESC)
    assert map_market(m) == Unmapped("unparseable_strike")


def test_polymarket_fed_description_mismatch() -> None:
    m = _poly("No change", slug="fed-decision-in-march-2027", tag="fomc", desc="rates")
    assert map_market(m) == Unmapped("rules_text_mismatch")


def test_polymarket_dated_fed_slug_maps() -> None:
    m = _poly(
        "No change", slug="fed-decision-in-december-20260729232808632", tag="fomc", desc=PM_FED_DESC
    )
    assert isinstance(map_market(m), ContractTerms)


# ---- link_release ----------------------------------------------------------------------------

CPI_NOV = _release(Series.CPI, "2026-11", datetime(2026, 12, 10, 13, 30, tzinfo=UTC))
CPI_OCT = _release(Series.CPI, "2026-10", datetime(2026, 11, 10, 13, 30, tzinfo=UTC))


def _terms(m: VenueMarket) -> ContractTerms:
    t = map_market(m)
    assert isinstance(t, ContractTerms)
    return t


def test_link_release_kalshi_cpi() -> None:
    m = _kalshi(event="KXCPI-26NOV", strike_type="greater", floor_strike=D("-0.3"))
    assert link_release(m, _terms(m), [CPI_OCT, CPI_NOV]) is CPI_NOV


def test_link_release_polymarket_cpi() -> None:
    m = _poly("3.3%")
    assert link_release(m, _terms(m), [CPI_NOV, CPI_OCT]) is CPI_OCT


def test_link_release_core_polymarket_needs_year() -> None:
    m = _poly("3.3%", slug="core-cpi-yoy-october-2026", desc=PM_CORE_DESC)
    assert link_release(m, _terms(m), [CPI_OCT]) is CPI_OCT
    old = _release(Series.CPI, "2025-10", CPI_OCT.scheduled_at)
    assert link_release(m, _terms(m), [old]) == Unmapped("period_mismatch")


def test_link_release_missing_release() -> None:
    m = _kalshi(event="KXCPI-26NOV", strike_type="greater", floor_strike=D("-0.3"))
    # Only the wrong month is in the calendar: never fall back to the nearest release.
    assert link_release(m, _terms(m), [CPI_OCT]) == Unmapped("no_release")
    assert link_release(m, _terms(m), []) == Unmapped("no_release")


def test_link_release_wrong_series_ignored() -> None:
    m = _kalshi(event="KXCPI-26NOV", strike_type="greater", floor_strike=D("-0.3"))
    nfp = _release(Series.NFP, "2026-11", CPI_NOV.scheduled_at)
    assert link_release(m, _terms(m), [nfp]) == Unmapped("no_release")


def test_link_release_window_is_48h_inclusive() -> None:
    m = _kalshi(event="KXCPI-26NOV", strike_type="greater", floor_strike=D("-0.3"))
    edge = _release(Series.CPI, "2026-11", m.close_time + timedelta(hours=48))
    over = _release(Series.CPI, "2026-11", m.close_time + timedelta(hours=48, seconds=1))
    assert link_release(m, _terms(m), [edge]) is edge
    assert link_release(m, _terms(m), [over]) == Unmapped("no_release")


def test_link_release_ambiguous() -> None:
    m = _kalshi(event="KXCPI-26NOV", strike_type="greater", floor_strike=D("-0.3"))
    twin = _release(Series.CPI, "2026-12", CPI_NOV.scheduled_at + timedelta(hours=1))
    assert link_release(m, _terms(m), [CPI_NOV, twin]) == Unmapped("ambiguous_release")


def test_link_release_period_mismatch() -> None:
    m = _kalshi(event="KXCPI-26OCT", strike_type="greater", floor_strike=D("-0.3"))
    assert link_release(m, _terms(m), [CPI_NOV]) == Unmapped("period_mismatch")
    bad_ticker = _kalshi(event="KXCPI-XX", strike_type="greater", floor_strike=D("-0.3"))
    assert link_release(bad_ticker, _terms(m), [CPI_NOV]) == Unmapped("period_mismatch")
    unknown_month = _kalshi(event="KXCPI-26XYZ", strike_type="greater", floor_strike=D("-0.3"))
    assert link_release(unknown_month, _terms(m), [CPI_NOV]) == Unmapped("period_mismatch")


def test_link_release_polymarket_period_mismatch() -> None:
    m = _poly("3.3%", slug="november-inflation-us-annual")
    assert link_release(m, _terms(m), [CPI_OCT]) == Unmapped("period_mismatch")
    odd = _poly("3.3%", slug="octobre-inflation-us-annual")
    assert link_release(odd, _terms(m), [CPI_OCT]) == Unmapped("period_mismatch")


def test_link_release_kalshi_nfp() -> None:
    m = _kalshi(
        "KXPAYROLLS",
        "KXPAYROLLS-26NOV",
        PAYROLLS_RULES,
        close=datetime(2026, 12, 4, 13, 29, tzinfo=UTC),
        strike_type="greater",
        floor_strike=90000,
    )
    nfp = _release(Series.NFP, "2026-11", datetime(2026, 12, 4, 13, 30, tzinfo=UTC))
    assert link_release(m, _terms(m), [nfp]) is nfp


FOMC_DEC = _release(Series.FOMC, "2026-12-09", datetime(2026, 12, 9, 19, 0, tzinfo=UTC))


def _fomc_kalshi(event: str) -> VenueMarket:
    return _kalshi(
        "KXFEDDECISION",
        event,
        "a Hike of 25bps",
        close=datetime(2026, 12, 9, 18, 59, tzinfo=UTC),
        strike_type="custom",
        custom_strike={"Hike": "25"},
    )


@pytest.mark.parametrize(
    ("event", "ok"),
    [
        ("KXFEDDECISION-26DEC", True),
        ("KXFEDDECISION-26NOV", False),
        ("FEDDECISION-26DEC09", True),
        ("KXFEDDECISION-25DEC", False),
    ],
)
def test_link_release_kalshi_fomc(event: str, ok: bool) -> None:
    m = _fomc_kalshi(event)
    got = link_release(m, _terms(m), [FOMC_DEC])
    assert (got is FOMC_DEC) if ok else (got == Unmapped("period_mismatch"))


@pytest.mark.parametrize(
    ("slug", "ok"),
    [
        ("fed-decision-in-december-20260729232808632", True),
        ("fed-decision-in-december", True),
        ("fed-decision-in-december-2026", True),
        ("fed-decision-in-december-2027", False),
        ("fed-decision-in-march-2027", False),
    ],
)
def test_link_release_polymarket_fomc(slug: str, ok: bool) -> None:
    m = _poly(
        "No change",
        slug=slug,
        tag="fomc",
        desc=PM_FED_DESC,
        close=datetime(2026, 12, 9, 19, 0, tzinfo=UTC),
    )
    got = link_release(m, _terms(m), [FOMC_DEC])
    assert (got is FOMC_DEC) if ok else (got == Unmapped("period_mismatch"))


# ---- recorded real data ----------------------------------------------------------------------

CASSETTES = Path(__file__).parent / "cassettes"
KALSHI_CASSETTES = [
    str(CASSETTES / "test_kalshi" / "test_lists_open_kxcpi_markets.yaml"),
    str(CASSETTES / "test_kalshi" / "test_lists_fed_decision_custom_strikes.yaml"),
]
POLYMARKET_CASSETTES = [
    str(
        CASSETTES / "test_polymarket" / "test_lists_only_single_meeting_fed_and_us_cpi_events.yaml"
    ),
]


def _measure_mix(markets: Sequence[VenueMarket]) -> dict[str, Counter[Measure]]:
    mix: dict[str, Counter[Measure]] = {}
    for m in markets:
        terms = map_market(m)
        assert isinstance(terms, ContractTerms), (m.market_id, terms)
        mix.setdefault(m.event_id, Counter())[terms.measure] += 1
    return mix


def _by_id(markets: Sequence[VenueMarket]) -> dict[str, VenueMarket]:
    return {m.market_id: m for m in markets}


@pytest.mark.vcr(*KALSHI_CASSETTES)
async def test_all_recorded_kalshi_markets_map() -> None:
    clock = SimClock(datetime(2026, 11, 10, 12, 0, tzinfo=UTC))
    adapter = KalshiAdapter(
        ResilientHttp("kalshi", "https://api.elections.kalshi.com/trade-api/v2", clock), clock
    )
    markets = [m for s in ("KXCPI", "KXFEDDECISION") for m in await adapter.list_markets(s)]
    assert {"KXCPI", "KXFEDDECISION"} <= set(KALSHI_SERIES)
    assert len(markets) == 70
    mix = _measure_mix(markets)  # every recorded market maps (link_release aside)
    assert mix["KXCPI-26NOV"] == Counter({Measure.CPI_MOM: 9})
    assert mix["KXCPI-26DEC"] == Counter({Measure.CPI_MOM: 11})
    fed = {e: c for e, c in mix.items() if e.startswith("KXFEDDECISION-")}
    assert len(fed) == 10
    assert all(c == Counter({Measure.FOMC_CHANGE_BP: 5}) for c in fed.values())
    pinned = map_market(_by_id(markets)["KXFEDDECISION-26DEC-C26"])
    assert isinstance(pinned, ContractTerms)
    assert (pinned.measure, pinned.comparator, pinned.strike_lo) == (
        Measure.FOMC_CHANGE_BP,
        Comparator.LT,
        Decimal("-25"),
    )


# Pinned Gamma market ids from the recorded cassette.
PM_CORE_OCT_LOW = "5699276"  # core-cpi-yoy-october-2026, "≤2.2%"
PM_ANNUAL_OCT_LOW = "5691083"  # october-inflation-us-annual, "≤3.2%"


@pytest.mark.vcr(*POLYMARKET_CASSETTES)
async def test_all_recorded_polymarket_markets_map() -> None:
    clock = SimClock(datetime(2026, 11, 10, 12, 0, tzinfo=UTC))
    adapter = PolymarketAdapter(
        ResilientHttp("polymarket-gamma", "https://gamma-api.polymarket.com", clock),
        ResilientHttp("polymarket-clob", "https://clob.polymarket.com", clock),
        clock,
    )
    markets = [m for t in POLYMARKET_TAGS for m in await adapter.list_markets(t)]
    assert len(markets) == 42
    mix = _measure_mix(markets)
    assert mix == {
        "october-inflation-us-annual": Counter({Measure.CPI_YOY: 12}),
        "core-cpi-yoy-october-2026": Counter({Measure.CORE_CPI_YOY: 10}),
        "fed-decision-in-december-20260729232808632": Counter({Measure.FOMC_CHANGE_BP: 5}),
        "fed-decision-in-january-20260729233815502": Counter({Measure.FOMC_CHANGE_BP: 5}),
        "fed-decision-in-march-2027": Counter({Measure.FOMC_CHANGE_BP: 5}),
        "fed-decision-in-april-2027": Counter({Measure.FOMC_CHANGE_BP: 5}),
    }
    by_id = _by_id(markets)
    core, annual = by_id[PM_CORE_OCT_LOW], by_id[PM_ANNUAL_OCT_LOW]
    assert map_market(core) == ContractTerms(
        Measure.CORE_CPI_YOY, Comparator.LTE, Decimal("2.2"), None, "pct", 1, "BLS"
    )
    assert map_market(annual) == ContractTerms(
        Measure.CPI_YOY, Comparator.LTE, Decimal("3.2"), None, "pct", 1, "BLS"
    )
    # The real descriptions discriminate: swapping them across the two events never maps.
    swapped_core = dataclasses.replace(core, rules_text=annual.rules_text)
    swapped_annual = dataclasses.replace(annual, rules_text=core.rules_text)
    assert map_market(swapped_core) == Unmapped("rules_text_mismatch")
    assert map_market(swapped_annual) == Unmapped("rules_text_mismatch")

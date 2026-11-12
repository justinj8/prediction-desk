import dataclasses
import json
import logging
import re
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from prediction_desk.clock import SimClock
from prediction_desk.domain.enums import Venue
from prediction_desk.markets.polymarket import (
    EVENT_SLUG_PATTERNS,
    MAX_PAGES,
    POLYMARKET_TAGS,
    PolymarketAdapter,
)
from prediction_desk.markets.types import MarketRef
from prediction_desk.net.http import ProviderError, ResilientHttp

T0 = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"


def _adapter(
    gamma: httpx.AsyncBaseTransport | None = None,
    clob: httpx.AsyncBaseTransport | None = None,
) -> PolymarketAdapter:
    clock = SimClock(T0)
    return PolymarketAdapter(
        ResilientHttp("polymarket-gamma", GAMMA, clock, transport=gamma),
        ResilientHttp("polymarket-clob", CLOB, clock, transport=clob),
        clock,
    )


def _json(body: Any) -> httpx.Response:
    return httpx.Response(200, content=json.dumps(body))


def _market(mid: str, **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": mid,
        "question": "q",
        "groupItemTitle": "3.3%",
        "outcomes": '["Yes", "No"]',
        "clobTokenIds": '["111", "222"]',
        "description": "rules",
        "endDate": "2026-11-13T13:30:00Z",
        "closed": False,
    }
    base.update(over)
    return base


def _event(slug: str, markets: list[dict[str, Any]]) -> dict[str, Any]:
    return {"slug": slug, "title": "t", "markets": markets}


def test_constants() -> None:
    assert POLYMARKET_TAGS == ("us-cpi", "fomc")
    assert PolymarketAdapter.venue is Venue.POLYMARKET
    assert len(EVENT_SLUG_PATTERNS) == 3


def _matches(slug: str) -> bool:
    return any(p.match(slug) for p in EVENT_SLUG_PATTERNS)


@pytest.mark.parametrize(
    "slug",
    [
        "october-inflation-us-annual",
        "core-cpi-yoy-october-2026",
        "fed-decision-in-december-20260729232808632",
        "fed-decision-in-march-2027",
    ],
)
def test_slug_patterns_accept(slug: str) -> None:
    assert _matches(slug)


@pytest.mark.parametrize(
    "slug", ["fed-decisions-sepdec", "japan-core-cpi-yoy-in-2026-x", "something-else"]
)
def test_slug_patterns_reject(slug: str) -> None:
    assert not _matches(slug)


@pytest.mark.vcr
async def test_lists_only_single_meeting_fed_and_us_cpi_events() -> None:
    adapter = _adapter()
    markets = []
    for tag in POLYMARKET_TAGS:
        markets += await adapter.list_markets(tag)
    assert markets
    for m in markets:
        assert m.venue is Venue.POLYMARKET
        assert _matches(m.event_id)
        assert not m.event_id.startswith("fed-decisions-")
        assert m.yes_token_id
        assert m.close_time.tzinfo is not None


@pytest.mark.vcr
async def test_cpi_group_titles_present() -> None:
    markets = await _adapter().list_markets("us-cpi")
    assert markets
    for m in markets:
        assert re.match(r"^[≤≥]?\d+\.\d%$", str(m.strike["group_item_title"]))


def _walk(obj: Any) -> list[Any]:
    if isinstance(obj, dict):
        return [x for v in obj.values() for x in _walk(v)]
    if isinstance(obj, (list, tuple)):
        return [x for v in obj for x in _walk(v)]
    return [obj]


async def test_gamma_float_prices_stay_decimal() -> None:
    mkt = _market("1", bestBid=0.034, orderPriceMinTickSize=0.001, groupItemTitle="3.4%")

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=json.dumps([_event("october-inflation-us-annual", [mkt])]),
        )

    markets = await _adapter(httpx.MockTransport(handler)).list_markets("us-cpi")
    assert len(markets) == 1
    leaves = _walk(dataclasses.asdict(markets[0]))
    assert not any(isinstance(x, float) for x in leaves)


async def test_maps_fields_and_query() -> None:
    seen: list[dict[str, str]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/events"
        seen.append(dict(req.url.params))
        return _json(
            [
                _event(
                    "fed-decision-in-march-2027",
                    [_market("m1", groupItemTitle="No change", closed=True)],
                ),
                _event("fed-decisions-sepdec", [_market("m2")]),
                _event("japan-core-cpi-yoy-in-2026-x", [_market("m3")]),
            ]
        )

    markets = await _adapter(httpx.MockTransport(handler)).list_markets("fomc")
    assert seen == [{"tag_slug": "fomc", "closed": "false", "limit": "100", "offset": "0"}]
    assert len(markets) == 1
    m = markets[0]
    assert m.market_id == "m1"
    assert m.event_id == "fed-decision-in-march-2027"
    assert m.series_code == "fomc"
    assert m.title == "q"
    assert m.rules_text == "rules"
    assert m.close_time == datetime(2026, 11, 13, 13, 30, tzinfo=UTC)
    assert m.status == "closed"
    assert m.strike == {"group_item_title": "No change"}
    assert m.yes_token_id == "111"
    assert m.ref == MarketRef("m1", "111")


async def test_missing_closed_and_optional_text_default() -> None:
    mkt = _market("m1")
    for k in ("closed", "question", "description", "groupItemTitle"):
        del mkt[k]

    def handler(req: httpx.Request) -> httpx.Response:
        return _json([_event("core-cpi-yoy-october-2026", [mkt])])

    (m,) = await _adapter(httpx.MockTransport(handler)).list_markets("us-cpi")
    assert m.status == "active"
    assert m.title == ""
    assert m.rules_text == ""
    assert m.strike == {"group_item_title": ""}


async def test_pagination_stops_on_short_page() -> None:
    offsets: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        offsets.append(req.url.params["offset"])
        if req.url.params["offset"] == "0":
            page = [_event("other-event", []) for _ in range(100)]
            page[0] = _event("october-inflation-us-annual", [_market("a")])
            return _json(page)
        return _json([_event("core-cpi-yoy-october-2026", [_market("b")])])

    markets = await _adapter(httpx.MockTransport(handler)).list_markets("us-cpi")
    assert offsets == ["0", "100"]
    assert [m.market_id for m in markets] == ["a", "b"]


async def test_page_limit_raises() -> None:
    calls = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _json([_event("other-event", []) for _ in range(100)])

    with pytest.raises(ProviderError, match="pagination exceeded"):
        await _adapter(httpx.MockTransport(handler)).list_markets("us-cpi")
    assert calls == MAX_PAGES


async def test_malformed_markets_skipped_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    good = _market("GOOD")
    null_id: dict[str, Any] = {**_market("x"), "id": None}
    empty_id: dict[str, Any] = {**_market("x"), "id": ""}
    bad_tokens = {**_market("BADTOK"), "clobTokenIds": "not json"}
    empty_tokens = {**_market("EMPTYTOK"), "clobTokenIds": "[]"}
    null_tokens = {**_market("NULLTOK"), "clobTokenIds": None}
    no_first = {**_market("NOTYES"), "outcomes": '["No", "Yes"]'}
    null_end = {**_market("NULLEND"), "endDate": None}
    bad_end = {**_market("BADEND"), "endDate": "garbage"}
    missing_end = {k: v for k, v in _market("MISSEND").items() if k != "endDate"}
    events = [
        _event(
            "october-inflation-us-annual",
            [
                null_id,
                empty_id,
                bad_tokens,
                empty_tokens,
                null_tokens,
                no_first,
                null_end,
                bad_end,
                missing_end,
                good,
            ],
        ),
        {"slug": None, "markets": [_market("ORPHAN")]},
        {"markets": [_market("ORPHAN2")]},
    ]

    def handler(req: httpx.Request) -> httpx.Response:
        return _json(events)

    with caplog.at_level(logging.WARNING, logger="prediction_desk.markets.polymarket"):
        markets = await _adapter(httpx.MockTransport(handler)).list_markets("us-cpi")
    assert [m.market_id for m in markets] == ["GOOD"]
    assert caplog.text.count("<unknown>") >= 2
    for name in ("BADTOK", "EMPTYTOK", "NULLTOK", "NOTYES", "NULLEND", "BADEND", "MISSEND"):
        assert name in caplog.text
    assert "october-inflation-us-annual" in caplog.text


def _book_json(bids: Any, asks: Any, ts: str = "1794325979447") -> httpx.Response:
    return _json({"bids": bids, "asks": asks, "timestamp": ts, "tick_size": "0.001"})


async def test_book_reversal() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/book"
        assert req.url.params["token_id"] == "TOK"
        return _book_json(
            [{"price": "0.001", "size": "5"}, {"price": "0.035", "size": "87.99"}],
            [{"price": "0.999", "size": "1"}, {"price": "0.038", "size": "5051.98"}],
        )

    book = await _adapter(clob=httpx.MockTransport(handler)).order_book(MarketRef("m", "TOK"))
    assert book.best_bid is not None
    assert book.best_ask is not None
    assert book.best_bid.price == Decimal("0.035")
    assert book.best_bid.size == Decimal("87.99")
    assert book.best_ask.price == Decimal("0.038")
    assert [lv.price for lv in book.bids] == [Decimal("0.035"), Decimal("0.001")]
    assert [lv.price for lv in book.asks] == [Decimal("0.038"), Decimal("0.999")]


async def test_book_timestamp_ms_exact() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return _book_json([], [], ts="1700000000123")

    book = await _adapter(clob=httpx.MockTransport(handler)).order_book(MarketRef("m", "T"))
    # R10: ts is when we observed the book; the CLOB's own time is venue_ts.
    assert book.ts == T0
    assert book.venue_ts == datetime(2023, 11, 14, 22, 13, 20, 123000, tzinfo=UTC)


async def test_book_empty_asks() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return _book_json([{"price": "0.035", "size": "1"}], [])

    book = await _adapter(clob=httpx.MockTransport(handler)).order_book(MarketRef("m", "T"))
    assert book.asks == ()
    assert book.best_ask is None
    assert book.best_bid is not None


async def test_book_requires_token() -> None:
    with pytest.raises(ProviderError, match="yes_token_id"):
        await _adapter().order_book(MarketRef("m", None))


# ---- totality: every parse/validation failure is a ProviderError --------------------------


@pytest.mark.parametrize("body", [{"events": []}, "secret-body-text", 3, None])
async def test_list_markets_non_list_body_raises_provider_error(body: Any) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return _json(body)

    with pytest.raises(ProviderError) as exc:
        await _adapter(httpx.MockTransport(handler)).list_markets("us-cpi")
    assert exc.value.__cause__ is None
    assert "secret-body-text" not in str(exc.value)


async def test_malformed_events_skipped(caplog: pytest.LogCaptureFixture) -> None:
    events = [
        "junk",
        None,
        _event("october-inflation-us-annual", "notalist"),  # type: ignore[arg-type]
        {"slug": "core-cpi-yoy-october-2026", "markets": None},
        _event("core-cpi-yoy-october-2026", [_market("GOOD")]),
    ]

    def handler(req: httpx.Request) -> httpx.Response:
        return _json(events)

    with caplog.at_level(logging.WARNING, logger="prediction_desk.markets.polymarket"):
        markets = await _adapter(httpx.MockTransport(handler)).list_markets("us-cpi")
    assert [m.market_id for m in markets] == ["GOOD"]
    assert "october-inflation-us-annual" in caplog.text


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("question", 5),
        ("description", ["d"]),
        ("groupItemTitle", 3.3),
        ("closed", "false"),
    ],
)
async def test_non_string_text_field_skips_market(
    field: str, value: Any, caplog: pytest.LogCaptureFixture
) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        bad = _market("BADTEXT", **{field: value})
        return _json([_event("october-inflation-us-annual", [bad, _market("GOOD")])])

    with caplog.at_level(logging.WARNING, logger="prediction_desk.markets.polymarket"):
        markets = await _adapter(httpx.MockTransport(handler)).list_markets("us-cpi")
    assert [m.market_id for m in markets] == ["GOOD"]
    assert "BADTEXT" in caplog.text


@pytest.mark.parametrize(
    "body",
    [
        [],
        "secret-body-text",
        {"bids": [], "asks": []},
        {"bids": [], "asks": [], "timestamp": "soon"},
        {"bids": [], "asks": [], "timestamp": None},
        {"bids": [], "asks": [], "timestamp": "9" * 30},
        {"bids": "x", "asks": [], "timestamp": "1"},
        {"bids": [{"price": "0.4"}], "asks": [], "timestamp": "1"},
        {"bids": [{"price": "abc", "size": "1"}], "asks": [], "timestamp": "1"},
        {"bids": [["0.4", "1"]], "asks": [], "timestamp": "1"},
        {"bids": [None], "asks": [], "timestamp": "1"},
        {"bids": [{"price": "2", "size": "1"}], "asks": [], "timestamp": "1"},
    ],
)
async def test_book_malformed_raises_provider_error(body: Any) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return _json(body)

    with pytest.raises(ProviderError) as exc:
        await _adapter(clob=httpx.MockTransport(handler)).order_book(MarketRef("m", "T"))
    assert exc.value.__cause__ is None
    assert "secret-body-text" not in str(exc.value)


async def test_book_unsorted_raises_provider_error() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        # Bids best-first instead of the CLOB's worst-to-best order: reversal breaks the book.
        return _book_json(
            [{"price": "0.035", "size": "1"}, {"price": "0.001", "size": "5"}],
            [],
        )

    with pytest.raises(ProviderError, match="malformed order book"):
        await _adapter(clob=httpx.MockTransport(handler)).order_book(MarketRef("m", "T"))


# ---- recorded real order book --------------------------------------------------------------

# YES token of "No change" in fed-decision-in-january (Gamma market 3215012), liquid when recorded.
RECORDED_TOKEN = "63563009510989471204377539732457086916124654411092982704760023438039220215330"
BOOK_CASSETTE = (
    Path(__file__).parent
    / "cassettes"
    / "test_polymarket"
    / "test_recorded_clob_book_best_first.yaml"
)


def _recorded_body() -> dict[str, Any]:
    cassette = yaml.safe_load(BOOK_CASSETTE.read_text(encoding="utf-8"))
    (interaction,) = cassette["interactions"]
    body: dict[str, Any] = json.loads(interaction["response"]["body"]["string"])
    return body


@pytest.mark.vcr
async def test_recorded_clob_book_best_first() -> None:
    book = await _adapter().order_book(MarketRef("3215012", RECORDED_TOKEN))
    # Real field names: bids/asks as {"price", "size"} objects plus an epoch-ms "timestamp".
    # Bids run ascending and asks descending: the best level is last on both sides.
    raw = _recorded_body()
    bids = [Decimal(lv["price"]) for lv in raw["bids"]]
    asks = [Decimal(lv["price"]) for lv in raw["asks"]]
    assert len(bids) >= 2 and len(asks) >= 2
    assert bids == sorted(bids)
    assert asks == sorted(asks, reverse=True)
    assert book.best_bid is not None and book.best_ask is not None
    assert book.best_bid.price == bids[-1]
    assert book.best_ask.price == asks[-1]
    assert book.best_bid.price < book.best_ask.price
    assert len(book.bids) == len(bids) and len(book.asks) == len(asks)
    ms = int(raw["timestamp"])
    assert book.venue_ts == datetime(1970, 1, 1, tzinfo=UTC) + timedelta(milliseconds=ms)
    assert book.ts == T0

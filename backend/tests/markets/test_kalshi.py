import json
import logging
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from prediction_desk.clock import SimClock
from prediction_desk.domain.enums import Venue
from prediction_desk.markets.kalshi import KALSHI_SERIES, MAX_PAGES, KalshiAdapter
from prediction_desk.markets.types import BookLevel, MarketRef
from prediction_desk.net.http import ProviderError, ResilientHttp

T0 = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
BASE = "https://api.elections.kalshi.com/trade-api/v2"


def _adapter(transport: httpx.AsyncBaseTransport | None = None) -> KalshiAdapter:
    clock = SimClock(T0)
    return KalshiAdapter(ResilientHttp("kalshi", BASE, clock, transport=transport), clock)


def _mock(handler: Any) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def _json(body: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, content=json.dumps(body))


def test_series_constant() -> None:
    assert KALSHI_SERIES == (
        "KXCPI",
        "KXCPICORE",
        "KXCPIYOY",
        "KXCPICOREYOY",
        "KXPAYROLLS",
        "KXFEDDECISION",
    )
    assert KalshiAdapter.venue is Venue.KALSHI


@pytest.mark.vcr
async def test_lists_open_kxcpi_markets() -> None:
    adapter = _adapter()
    markets = await adapter.list_markets("KXCPI")
    assert markets
    for m in markets:
        assert m.venue is Venue.KALSHI
        assert m.series_code == "KXCPI"
        assert m.strike["strike_type"] == "greater"
        # Live rules say "(single-decimal)" for the nearest event and "one decimal place" for
        # later ones; both pin the resolution precision.
        assert "(single-decimal)" in m.rules_text or "one decimal place" in m.rules_text
        assert m.close_time.tzinfo is not None
        assert m.yes_token_id is None
        assert m.ref == MarketRef(m.market_id, None)
    assert any("(single-decimal)" in m.rules_text for m in markets)


@pytest.mark.vcr
async def test_lists_fed_decision_custom_strikes() -> None:
    adapter = _adapter()
    markets = await adapter.list_markets("KXFEDDECISION")
    assert markets
    for m in markets:
        custom = m.strike["custom_strike"]
        assert isinstance(custom, dict)
        assert len(custom) == 1
        assert next(iter(custom)) in {"Cut", "Hike"}


async def test_list_markets_follows_cursor_and_maps_fields() -> None:
    seen: list[dict[str, str]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        params = dict(req.url.params)
        seen.append(params)
        assert req.url.path.endswith("/markets")
        if "cursor" not in params:
            market = {
                "ticker": "KXCPI-26OCT-T0.3",
                "event_ticker": "KXCPI-26OCT",
                "title": "t",
                "rules_primary": "r",
                "close_time": "2026-11-13T13:30:00Z",
                "status": "active",
                "strike_type": "greater",
                "floor_strike": 0.3,
            }
            return _json({"markets": [market], "cursor": "abc"})
        market = {
            "ticker": "KXCPI-26OCT-T0.4",
            "event_ticker": "KXCPI-26OCT",
            "title": "t2",
            "rules_primary": "r2",
            "close_time": "2026-11-13T13:30:00Z",
            "status": "active",
            "strike_type": "custom",
            "custom_strike": {"Cut": ">25"},
        }
        return _json({"markets": [market], "cursor": ""})

    markets = await _adapter(_mock(handler)).list_markets("KXCPI")
    assert [m.market_id for m in markets] == ["KXCPI-26OCT-T0.3", "KXCPI-26OCT-T0.4"]
    assert seen[0] == {"series_ticker": "KXCPI", "status": "open", "limit": "1000"}
    assert seen[1]["cursor"] == "abc"
    first = markets[0]
    assert first.event_id == "KXCPI-26OCT"
    assert first.close_time == datetime(2026, 11, 13, 13, 30, tzinfo=UTC)
    assert first.strike == {
        "strike_type": "greater",
        "floor_strike": Decimal("0.3"),
        "cap_strike": None,
        "custom_strike": None,
    }
    assert markets[1].strike["custom_strike"] == {"Cut": ">25"}


async def test_orderbook_inverts_no_bids() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path.endswith("/markets/KXCPI-26OCT-T0.3/orderbook")
        return _json(
            {
                "orderbook_fp": {
                    "yes_dollars": [["0.47", "100.00"], ["0.49", "142.15"]],
                    "no_dollars": [["0.47", "100.00"], ["0.49", "139.54"]],
                }
            }
        )

    book = await _adapter(_mock(handler)).order_book(MarketRef("KXCPI-26OCT-T0.3", None))
    assert book.bids[0] == BookLevel(Decimal("0.4900"), Decimal("142.15"))
    assert book.asks[0] == BookLevel(Decimal("0.5100"), Decimal("139.54"))
    assert book.asks[1].price == Decimal("0.5300")
    assert book.ts == T0
    assert book.venue_ts is None


async def test_orderbook_null_sides() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return _json({"orderbook_fp": {"yes_dollars": None, "no_dollars": None}})

    book = await _adapter(_mock(handler)).order_book(MarketRef("X", None))
    assert book.bids == ()
    assert book.asks == ()


async def test_orderbook_missing_sides() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return _json({"orderbook_fp": {}})

    book = await _adapter(_mock(handler)).order_book(MarketRef("X", None))
    assert book.best_bid is None
    assert book.best_ask is None


def _raw(ticker: str, **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "ticker": ticker,
        "event_ticker": "E",
        "title": "t",
        "rules_primary": "r",
        "close_time": "2026-11-13T13:30:00Z",
        "status": "active",
    }
    base.update(over)
    return {k: v for k, v in base.items() if v is not None}


async def test_repeated_cursor_raises_bounded() -> None:
    calls = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _json({"markets": [_raw("A")], "cursor": "same"})

    with pytest.raises(ProviderError, match="repeated pagination cursor"):
        await _adapter(_mock(handler)).list_markets("KXCPI")
    assert calls == 2


async def test_empty_last_page_terminates() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if "cursor" not in req.url.params:
            return _json({"markets": [_raw("A")], "cursor": "c1"})
        return _json({"markets": [], "cursor": ""})

    markets = await _adapter(_mock(handler)).list_markets("KXCPI")
    assert [m.market_id for m in markets] == ["A"]


async def test_page_limit_raises() -> None:
    calls = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _json({"markets": [], "cursor": f"c{calls}"})

    with pytest.raises(ProviderError, match="pagination exceeded"):
        await _adapter(_mock(handler)).list_markets("KXCPI")
    assert calls == MAX_PAGES


async def test_malformed_market_skipped_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return _json(
            {
                "markets": [
                    _raw("GOOD"),
                    _raw("BAD-NOCLOSE", close_time=None),
                    _raw("NORULES", rules_primary=None),
                    _raw("BAD-PARSE", close_time="garbage"),
                ],
                "cursor": "",
            }
        )

    with caplog.at_level(logging.WARNING, logger="prediction_desk.markets.kalshi"):
        markets = await _adapter(_mock(handler)).list_markets("KXCPI")
    assert [m.market_id for m in markets] == ["GOOD", "NORULES"]
    assert markets[1].rules_text == ""
    assert "BAD-NOCLOSE" in caplog.text
    assert "BAD-PARSE" in caplog.text


async def test_null_identity_fields_skipped(caplog: pytest.LogCaptureFixture) -> None:
    good = _raw("GOOD")
    null_ticker = {**_raw("X"), "ticker": None}
    null_event = {**_raw("NULLEVENT"), "event_ticker": None}
    empty_event = {**_raw("EMPTYEVENT"), "event_ticker": ""}
    null_close = {**_raw("NULLCLOSE"), "close_time": None}
    no_ticker = {k: v for k, v in _raw("X").items() if k != "ticker"}

    def handler(req: httpx.Request) -> httpx.Response:
        body = {
            "markets": [null_ticker, null_event, empty_event, null_close, no_ticker, good],
            "cursor": "",
        }
        return _json(body)

    with caplog.at_level(logging.WARNING, logger="prediction_desk.markets.kalshi"):
        markets = await _adapter(_mock(handler)).list_markets("KXCPI")
    assert [m.market_id for m in markets] == ["GOOD"]
    assert caplog.text.count("<unknown>") == 2
    for name in ("NULLEVENT", "EMPTYEVENT", "NULLCLOSE"):
        assert name in caplog.text


# ---- totality: every parse/validation failure is a ProviderError --------------------------


@pytest.mark.parametrize(
    "body",
    [
        [],
        "markets",
        {"markets": "nope"},
        {"markets": {"a": 1}},
        {"markets": [], "cursor": 5},
    ],
)
async def test_list_markets_malformed_body_raises_provider_error(body: Any) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps(body))

    with pytest.raises(ProviderError) as exc:
        await _adapter(_mock(handler)).list_markets("KXCPI")
    assert exc.value.__cause__ is None
    assert "nope" not in str(exc.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [("title", 5), ("title", ["t"]), ("rules_primary", {"x": 1}), ("status", 1)],
)
async def test_non_string_text_field_skips_market(
    field: str, value: Any, caplog: pytest.LogCaptureFixture
) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return _json({"markets": [_raw("BADTEXT", **{field: value}), _raw("GOOD")], "cursor": ""})

    with caplog.at_level(logging.WARNING, logger="prediction_desk.markets.kalshi"):
        markets = await _adapter(_mock(handler)).list_markets("KXCPI")
    assert [m.market_id for m in markets] == ["GOOD"]
    assert "BADTEXT" in caplog.text


async def test_non_object_market_skipped(caplog: pytest.LogCaptureFixture) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return _json({"markets": ["junk", 3, None, _raw("GOOD")], "cursor": ""})

    with caplog.at_level(logging.WARNING, logger="prediction_desk.markets.kalshi"):
        markets = await _adapter(_mock(handler)).list_markets("KXCPI")
    assert [m.market_id for m in markets] == ["GOOD"]
    assert caplog.text.count("<unknown>") == 3


@pytest.mark.parametrize(
    "body",
    [
        [],
        {},
        {"orderbook": {"yes": []}},
        {"orderbook_fp": None},
        {"orderbook_fp": []},
        {"orderbook_fp": "secret-body-text"},
        {"orderbook_fp": {"yes_dollars": "x"}},
        {"orderbook_fp": {"yes_dollars": [["0.4"]]}},
        {"orderbook_fp": {"yes_dollars": [["abc", "1"]]}},
        {"orderbook_fp": {"yes_dollars": [["NaN", "1"]]}},
        {"orderbook_fp": {"yes_dollars": [[["0.4"], "1"]]}},
        {"orderbook_fp": {"yes_dollars": [["1.5", "1"]]}},
        {"orderbook_fp": {"yes_dollars": [["0.4", "NaN"]]}},
        {"orderbook_fp": {"yes_dollars": [["0.4", "-1"]]}},
        {"orderbook_fp": {"no_dollars": [None]}},
    ],
)
async def test_orderbook_malformed_raises_provider_error(body: Any) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps(body))

    with pytest.raises(ProviderError) as exc:
        await _adapter(_mock(handler)).order_book(MarketRef("X", None))
    assert exc.value.__cause__ is None
    assert "secret-body-text" not in str(exc.value)


async def test_orderbook_unsorted_raises_provider_error() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        # Descending (best first) instead of Kalshi's ascending order: reversal breaks the book.
        return _json({"orderbook_fp": {"yes_dollars": [["0.49", "1"], ["0.47", "1"]]}})

    with pytest.raises(ProviderError, match="malformed order book"):
        await _adapter(_mock(handler)).order_book(MarketRef("X", None))


# ---- recorded real order book --------------------------------------------------------------

RECORDED_BOOK_TICKER = "KXCPI-26NOV-T0.0"  # liquid open KXCPI strike at recording time
BOOK_CASSETTE = (
    Path(__file__).parent / "cassettes" / "test_kalshi" / "test_recorded_orderbook_best_first.yaml"
)


def _recorded_body() -> dict[str, Any]:
    cassette = yaml.safe_load(BOOK_CASSETTE.read_text(encoding="utf-8"))
    (interaction,) = cassette["interactions"]
    body: dict[str, Any] = json.loads(interaction["response"]["body"]["string"])
    return body


@pytest.mark.vcr
async def test_recorded_orderbook_best_first() -> None:
    book = await _adapter().order_book(MarketRef(RECORDED_BOOK_TICKER, None))
    # Real field names: the container is orderbook_fp with yes_dollars/no_dollars price lists,
    # each ascending (best last). OrderBook itself enforces best-first after normalization.
    raw = _recorded_body()["orderbook_fp"]
    for side in ("yes_dollars", "no_dollars"):
        prices = [Decimal(p) for p, _ in raw[side]]
        assert len(prices) >= 2
        assert prices == sorted(prices)
        assert len(set(prices)) == len(prices)
    assert len(book.bids) == len(raw["yes_dollars"])
    assert len(book.asks) == len(raw["no_dollars"])
    assert book.best_bid is not None and book.best_ask is not None
    assert book.best_bid.price == Decimal(raw["yes_dollars"][-1][0])
    assert book.best_ask.price == 1 - Decimal(raw["no_dollars"][-1][0])
    assert book.best_bid.price < book.best_ask.price  # not crossed: the NO->YES inversion holds
    assert book.ts == T0

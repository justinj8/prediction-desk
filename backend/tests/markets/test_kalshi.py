import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import httpx
import pytest

from prediction_desk.clock import SimClock
from prediction_desk.domain.enums import Venue
from prediction_desk.markets.kalshi import KALSHI_SERIES, KalshiAdapter
from prediction_desk.markets.types import BookLevel, MarketRef
from prediction_desk.net.http import ResilientHttp

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

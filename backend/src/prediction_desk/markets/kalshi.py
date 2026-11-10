"""Kalshi adapter: open-market discovery and normalized YES-side order books.

Kalshi publishes bids only, ascending with the best last. A NO bid at p is a YES ask at 1 - p.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any

from prediction_desk.clock import Clock
from prediction_desk.domain.enums import Venue
from prediction_desk.domain.types import require_utc
from prediction_desk.markets.types import (
    BookLevel,
    MarketRef,
    OrderBook,
    VenueAdapter,
    VenueMarket,
)
from prediction_desk.net.http import ResilientHttp

KALSHI_SERIES: tuple[str, ...] = (
    "KXCPI",
    "KXCPICORE",
    "KXCPIYOY",
    "KXCPICOREYOY",
    "KXPAYROLLS",
    "KXFEDDECISION",
)

_STRIKE_KEYS = ("strike_type", "floor_strike", "cap_strike", "custom_strike")


def _to_market(raw: dict[str, Any], series_code: str) -> VenueMarket:
    return VenueMarket(
        venue=Venue.KALSHI,
        market_id=raw["ticker"],
        event_id=raw["event_ticker"],
        series_code=series_code,
        title=raw["title"],
        rules_text=raw["rules_primary"],
        close_time=require_utc(datetime.fromisoformat(raw["close_time"])),
        status=raw["status"],
        strike={k: raw.get(k) for k in _STRIKE_KEYS},
        yes_token_id=None,
    )


def _levels(raw: list[list[str]] | None) -> list[BookLevel]:
    """Kalshi lists are ascending (best last); return them best first."""
    return [BookLevel(Decimal(p), Decimal(s)) for p, s in reversed(raw or [])]


class KalshiAdapter(VenueAdapter):
    venue = Venue.KALSHI

    def __init__(self, http: ResilientHttp, clock: Clock) -> None:
        self._http = http
        self._clock = clock

    async def list_markets(self, series_code: str) -> list[VenueMarket]:
        out: list[VenueMarket] = []
        cursor = ""
        while True:
            params = {"series_ticker": series_code, "status": "open", "limit": "1000"}
            if cursor:
                params["cursor"] = cursor
            data = await self._http.get_json("/markets", params)
            out.extend(_to_market(m, series_code) for m in data.get("markets") or [])
            cursor = data.get("cursor") or ""
            if not cursor:
                return out

    async def order_book(self, ref: MarketRef) -> OrderBook:
        data = await self._http.get_json(f"/markets/{ref.market_id}/orderbook")
        book = data.get("orderbook_fp") or {}
        bids = _levels(book.get("yes_dollars"))
        asks = [BookLevel(Decimal(1) - lv.price, lv.size) for lv in _levels(book.get("no_dollars"))]
        return OrderBook(bids=tuple(bids), asks=tuple(asks), ts=self._clock.now())

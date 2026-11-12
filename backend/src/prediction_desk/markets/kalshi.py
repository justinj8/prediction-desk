"""Kalshi adapter: open-market discovery and normalized YES-side order books.

Kalshi publishes bids only, ascending with the best last. A NO bid at p is a YES ask at 1 - p.

Both calls are total: any malformed response surfaces as ``ProviderError`` (never chained, and
never quoting response text). A single malformed market is skipped with a WARNING instead.
"""

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any

from prediction_desk.clock import Clock
from prediction_desk.domain.enums import Venue
from prediction_desk.domain.types import require_utc
from prediction_desk.markets.types import (
    PARSE_ERRORS,
    BookLevel,
    MarketRef,
    OrderBook,
    VenueAdapter,
    VenueMarket,
    optional_text,
)
from prediction_desk.net.http import ProviderError, ResilientHttp

logger = logging.getLogger(__name__)

MAX_PAGES = 50

KALSHI_SERIES: tuple[str, ...] = (
    "KXCPI",
    "KXCPICORE",
    "KXCPIYOY",
    "KXCPICOREYOY",
    "KXPAYROLLS",
    "KXFEDDECISION",
)

_STRIKE_KEYS = ("strike_type", "floor_strike", "cap_strike", "custom_strike")
_TEXT_KEYS = ("title", "rules_primary", "status")


def _to_market(raw: Any, series_code: str) -> VenueMarket | None:
    """Map one raw market; None (with a warning) if an identity or text field is missing or bad."""
    ticker = raw.get("ticker") if isinstance(raw, dict) else None
    label = ticker if isinstance(ticker, str) and ticker else "<unknown>"
    try:
        if not isinstance(raw, dict):
            raise TypeError("market is not an object")
        market_id = raw["ticker"]
        event_id = raw["event_ticker"]
        for ident in (market_id, event_id):
            if not isinstance(ident, str) or not ident:
                raise ValueError("ticker and event_ticker must be non-empty strings")
        if raw["close_time"] is None:
            raise ValueError("close_time is null")
        close_time = require_utc(datetime.fromisoformat(raw["close_time"]))
        title, rules, status = (optional_text(raw, k) for k in _TEXT_KEYS)
    except PARSE_ERRORS as exc:
        logger.warning("kalshi: skipping malformed market %s: %r", label, exc)
        return None
    return VenueMarket(
        venue=Venue.KALSHI,
        market_id=market_id,
        event_id=event_id,
        series_code=series_code,
        title=title,
        rules_text=rules,
        close_time=close_time,
        status=status,
        strike={k: raw.get(k) for k in _STRIKE_KEYS},
        yes_token_id=None,
    )


def _levels(raw: Any) -> list[BookLevel]:
    """Kalshi lists are ascending (best last); return them best first. Null means empty."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise TypeError("book side is not a list")
    return [BookLevel(Decimal(p), Decimal(s)) for p, s in reversed(raw)]


def _page(data: Any) -> tuple[list[Any], str]:
    if not isinstance(data, dict):
        raise TypeError("markets response is not an object")
    markets = data["markets"]
    cursor = data.get("cursor")
    if not isinstance(markets, list):
        raise TypeError("markets is not a list")
    if cursor is not None and not isinstance(cursor, str):
        raise TypeError("cursor is not a string")
    return markets, cursor or ""


def _book(data: Any, ts: datetime) -> OrderBook:
    if not isinstance(data, dict):
        raise TypeError("order book response is not an object")
    book = data["orderbook_fp"]  # a missing container is malformed, never an empty book
    if not isinstance(book, dict):
        raise TypeError("orderbook_fp is not an object")
    bids = _levels(book.get("yes_dollars"))
    asks = [BookLevel(Decimal(1) - lv.price, lv.size) for lv in _levels(book.get("no_dollars"))]
    return OrderBook(bids=tuple(bids), asks=tuple(asks), ts=ts, venue_ts=None)


class KalshiAdapter(VenueAdapter):
    venue = Venue.KALSHI

    def __init__(self, http: ResilientHttp, clock: Clock) -> None:
        self._http = http
        self._clock = clock

    async def list_markets(self, series_code: str) -> list[VenueMarket]:
        out: list[VenueMarket] = []
        cursor = ""
        seen: set[str] = set()
        for _ in range(MAX_PAGES):
            params = {"series_ticker": series_code, "status": "open", "limit": "1000"}
            if cursor:
                params["cursor"] = cursor
            data = await self._http.get_json("/markets", params)
            try:
                page, cursor = _page(data)
            except PARSE_ERRORS as exc:
                raise ProviderError(
                    "kalshi", f"malformed markets response: {type(exc).__name__}"
                ) from None
            for raw in page:
                market = _to_market(raw, series_code)
                if market is not None:
                    out.append(market)
            if not cursor:
                return out
            if cursor in seen:
                raise ProviderError("kalshi", "repeated pagination cursor")
            seen.add(cursor)
        raise ProviderError("kalshi", f"pagination exceeded {MAX_PAGES} pages")

    async def order_book(self, ref: MarketRef) -> OrderBook:
        data = await self._http.get_json(f"/markets/{ref.market_id}/orderbook")
        try:
            return _book(data, self._clock.now())
        except PARSE_ERRORS as exc:
            raise ProviderError("kalshi", f"malformed order book: {type(exc).__name__}") from None

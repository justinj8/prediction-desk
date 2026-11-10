"""Polymarket adapter: Gamma event discovery and normalized CLOB YES-side order books.

The CLOB lists bids ascending and asks descending, so the best level is last on both sides.

Both calls are total: any malformed response surfaces as ``ProviderError`` (never chained, and
never quoting response text). A single malformed event or market is skipped with a WARNING.
"""

import json
import logging
import re
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

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
PAGE_SIZE = 100

POLYMARKET_TAGS: tuple[str, ...] = ("us-cpi", "fomc")
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

EVENT_SLUG_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^[a-z]+-inflation-us-annual(-\d+)?$"),
    re.compile(r"^core-cpi-yoy-[a-z]+-\d{4}(-\d+)?$"),
    re.compile(r"^fed-decision-in-[a-z]+(-\d+)?$"),
)


def _to_market(raw: Any, event_slug: str, tag: str) -> VenueMarket | None:
    """Map one raw Gamma market; None (with a warning) if an identity field is missing or bad."""
    market_id = raw.get("id") if isinstance(raw, dict) else None
    label = market_id if isinstance(market_id, str) and market_id else "<unknown>"
    try:
        if not isinstance(raw, dict):
            raise ValueError("market is not an object")
        if label == "<unknown>":
            raise ValueError("id must be a non-empty string")
        end = raw["endDate"]
        if not isinstance(end, str):
            raise ValueError("endDate is not a string")
        close_time = require_utc(datetime.fromisoformat(end))
        outcomes = json.loads(raw["outcomes"])
        if not isinstance(outcomes, list) or not outcomes or outcomes[0] != "Yes":
            raise ValueError("first outcome is not Yes")
        tokens = json.loads(raw["clobTokenIds"])
        if not isinstance(tokens, list) or not tokens or not isinstance(tokens[0], str):
            raise ValueError("clobTokenIds is not a list of strings")
        if not tokens[0]:
            raise ValueError("empty yes token id")
        title, rules, group_title = (
            optional_text(raw, k) for k in ("question", "description", "groupItemTitle")
        )
        closed = raw.get("closed")
        if closed is not None and not isinstance(closed, bool):
            raise TypeError("closed is not a boolean")
    except PARSE_ERRORS as exc:
        logger.warning("polymarket: skipping malformed market %s in %s: %r", label, event_slug, exc)
        return None
    return VenueMarket(
        venue=Venue.POLYMARKET,
        market_id=label,
        event_id=event_slug,
        series_code=tag,
        title=title,
        rules_text=rules,
        close_time=close_time,
        status="closed" if closed else "active",
        strike={"group_item_title": group_title},
        yes_token_id=tokens[0],
    )


def _levels(raw: Any) -> list[BookLevel]:
    """CLOB lists run worst to best; return them best first. Null means empty."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise TypeError("book side is not a list")
    return [BookLevel(Decimal(lv["price"]), Decimal(lv["size"])) for lv in reversed(raw)]


def _epoch_ms(raw: Any) -> datetime:
    """Integer epoch milliseconds (a string on the wire) as exact UTC; no float math."""
    ms = int(raw)
    return _EPOCH + timedelta(seconds=ms // 1000, milliseconds=ms % 1000)


def _book(data: Any) -> OrderBook:
    if not isinstance(data, dict):
        raise TypeError("order book response is not an object")
    return OrderBook(
        bids=tuple(_levels(data.get("bids"))),
        asks=tuple(_levels(data.get("asks"))),
        ts=_epoch_ms(data["timestamp"]),
    )


def _event_markets(event: Any) -> tuple[str, list[Any]] | None:
    """(slug, raw markets) of a tracked event; None to skip it (with a warning if malformed)."""
    if not isinstance(event, dict):
        logger.warning("polymarket: skipping event that is not an object")
        return None
    slug = event.get("slug")
    if not isinstance(slug, str) or not slug:
        logger.warning("polymarket: skipping event with missing slug")
        return None
    if not any(p.match(slug) for p in EVENT_SLUG_PATTERNS):
        return None
    markets = event.get("markets")
    if markets is None:
        return slug, []
    if not isinstance(markets, list):
        logger.warning("polymarket: skipping event %s: markets is not a list", slug)
        return None
    return slug, markets


class PolymarketAdapter(VenueAdapter):
    venue = Venue.POLYMARKET

    def __init__(self, gamma: ResilientHttp, clob: ResilientHttp) -> None:
        self._gamma = gamma
        self._clob = clob

    async def list_markets(self, series_code: str) -> list[VenueMarket]:
        out: list[VenueMarket] = []
        for page in range(MAX_PAGES):
            params = {
                "tag_slug": series_code,
                "closed": "false",
                "limit": str(PAGE_SIZE),
                "offset": str(page * PAGE_SIZE),
            }
            events = await self._gamma.get_json("/events", params)
            if not isinstance(events, list):
                raise ProviderError("polymarket", "malformed events response: not a list")
            for event in events:
                tracked = _event_markets(event)
                if tracked is None:
                    continue
                slug, raws = tracked
                for raw in raws:
                    market = _to_market(raw, slug, series_code)
                    if market is not None:
                        out.append(market)
            if len(events) < PAGE_SIZE:
                return out
        raise ProviderError("polymarket", f"pagination exceeded {MAX_PAGES} pages")

    async def order_book(self, ref: MarketRef) -> OrderBook:
        if not ref.yes_token_id:
            raise ProviderError("polymarket", "order_book requires a yes_token_id")
        data = await self._clob.get_json("/book", {"token_id": ref.yes_token_id})
        try:
            return _book(data)
        except PARSE_ERRORS as exc:
            raise ProviderError(
                "polymarket", f"malformed order book: {type(exc).__name__}"
            ) from None

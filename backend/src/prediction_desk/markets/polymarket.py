"""Polymarket adapter: Gamma event discovery and normalized CLOB YES-side order books.

The CLOB lists bids ascending and asks descending, so the best level is last on both sides.
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
    BookLevel,
    MarketRef,
    OrderBook,
    VenueAdapter,
    VenueMarket,
)
from prediction_desk.net.http import ProviderError, ResilientHttp

logger = logging.getLogger(__name__)

MAX_PAGES = 50
PAGE_SIZE = 100

POLYMARKET_TAGS: tuple[str, ...] = ("us-cpi", "fomc")

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
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("polymarket: skipping malformed market %s in %s: %r", label, event_slug, exc)
        return None
    return VenueMarket(
        venue=Venue.POLYMARKET,
        market_id=label,
        event_id=event_slug,
        series_code=tag,
        title=raw.get("question") or "",
        rules_text=raw.get("description") or "",
        close_time=close_time,
        status="closed" if raw.get("closed") else "active",
        strike={"group_item_title": raw.get("groupItemTitle") or ""},
        yes_token_id=tokens[0],
    )


def _levels(raw: list[dict[str, str]] | None) -> list[BookLevel]:
    """CLOB lists run worst to best; return them best first."""
    return [BookLevel(Decimal(lv["price"]), Decimal(lv["size"])) for lv in reversed(raw or [])]


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
            for event in events:
                slug = event.get("slug")
                if not isinstance(slug, str) or not slug:
                    logger.warning("polymarket: skipping event with missing slug")
                    continue
                if not any(p.match(slug) for p in EVENT_SLUG_PATTERNS):
                    continue
                for raw in event.get("markets") or []:
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
        ms = int(data["timestamp"])
        ts = datetime.fromtimestamp(ms // 1000, tz=UTC) + timedelta(milliseconds=ms % 1000)
        return OrderBook(
            bids=tuple(_levels(data.get("bids"))),
            asks=tuple(_levels(data.get("asks"))),
            ts=ts,
        )

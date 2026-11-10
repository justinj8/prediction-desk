"""Venue-neutral market and order book types."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Protocol

from prediction_desk.domain.enums import Venue
from prediction_desk.domain.types import require_utc


@dataclass(frozen=True)
class MarketRef:
    market_id: str
    yes_token_id: str | None


@dataclass(frozen=True)
class VenueMarket:
    venue: Venue
    market_id: str
    event_id: str
    series_code: str
    title: str
    rules_text: str
    close_time: datetime
    status: str
    strike: Mapping[str, Any]
    yes_token_id: str | None

    @property
    def ref(self) -> MarketRef:
        return MarketRef(self.market_id, self.yes_token_id)


@dataclass(frozen=True)
class BookLevel:
    price: Decimal
    size: Decimal


@dataclass(frozen=True)
class OrderBook:
    """YES-side book, both sides best first. Either side may be empty."""

    bids: tuple[BookLevel, ...]
    asks: tuple[BookLevel, ...]
    ts: datetime

    def __post_init__(self) -> None:
        for level in (*self.bids, *self.asks):
            if not Decimal(0) <= level.price <= Decimal(1):
                raise ValueError(f"price out of [0, 1]: {level.price}")
        for a, b in zip(self.bids, self.bids[1:], strict=False):
            if not a.price > b.price:
                raise ValueError("bids must be strictly descending")
        for a, b in zip(self.asks, self.asks[1:], strict=False):
            if not a.price < b.price:
                raise ValueError("asks must be strictly ascending")
        if self.ts.utcoffset() != timedelta(0):
            require_utc(self.ts)  # raises for naive datetimes
            raise ValueError("ts must be UTC")

    @property
    def best_bid(self) -> BookLevel | None:
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> BookLevel | None:
        return self.asks[0] if self.asks else None

    def to_json(self) -> dict[str, list[list[str]]]:
        return {
            "bids": [[str(lv.price), str(lv.size)] for lv in self.bids],
            "asks": [[str(lv.price), str(lv.size)] for lv in self.asks],
        }


class VenueAdapter(Protocol):
    venue: Venue

    async def list_markets(self, series_code: str) -> list[VenueMarket]: ...

    async def order_book(self, ref: MarketRef) -> OrderBook: ...

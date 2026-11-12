"""Venue-neutral market and order book types."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Protocol

from prediction_desk.domain.enums import Venue
from prediction_desk.domain.types import require_utc

# Every failure a malformed venue body can raise while being parsed. InvalidOperation is an
# ArithmeticError; OrderBook validation raises ValueError.
PARSE_ERRORS: tuple[type[Exception], ...] = (
    KeyError,
    TypeError,
    ValueError,
    AttributeError,
    ArithmeticError,
)

# Upper bound on a single book level's size; a venue body beyond this is malformed.
MAX_LEVEL_SIZE = Decimal("1e12")


def optional_text(raw: Mapping[str, Any], key: str) -> str:
    """Optional venue text: missing or null is ""; anything but a string is malformed."""
    value = raw.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError(f"{key} is not a string")
    return value


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


def _require_exact_utc(ts: datetime, name: str) -> None:
    if ts.utcoffset() != timedelta(0):
        require_utc(ts)  # raises for naive datetimes
        raise ValueError(f"{name} must be UTC")


@dataclass(frozen=True)
class OrderBook:
    """YES-side book, both sides best first. Either side may be empty.

    ``ts`` is when we observed the book (the adapter's ``clock.now()``) for every venue; it is
    what quote-age gates and as-of replay compare against. ``venue_ts`` is the venue's own
    timestamp for the book when it publishes one, else None.
    """

    bids: tuple[BookLevel, ...]
    asks: tuple[BookLevel, ...]
    ts: datetime
    venue_ts: datetime | None = None

    def __post_init__(self) -> None:
        for level in (*self.bids, *self.asks):
            if not (level.price.is_finite() and Decimal(0) <= level.price <= Decimal(1)):
                raise ValueError(f"price out of [0, 1]: {level.price}")
            if not (level.size.is_finite() and level.size >= 0):
                raise ValueError(f"size must be finite and non-negative: {level.size}")
            if level.size > MAX_LEVEL_SIZE:
                raise ValueError(f"size exceeds {MAX_LEVEL_SIZE}: {level.size}")
        for a, b in zip(self.bids, self.bids[1:], strict=False):
            if not a.price > b.price:
                raise ValueError("bids must be strictly descending")
        for a, b in zip(self.asks, self.asks[1:], strict=False):
            if not a.price < b.price:
                raise ValueError("asks must be strictly ascending")
        _require_exact_utc(self.ts, "ts")
        if self.venue_ts is not None:
            _require_exact_utc(self.venue_ts, "venue_ts")

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

    @classmethod
    def from_json(
        cls,
        book: Mapping[str, Sequence[Sequence[str]]],
        ts: datetime,
        venue_ts: datetime | None = None,
    ) -> "OrderBook":
        """Inverse of ``to_json``.

        Raises on a malformed stored book, so callers reading snapshot JSON must catch all of:
        ``KeyError`` (no ``"bids"``/``"asks"`` key), ``TypeError`` (``book`` is not a mapping),
        ``ValueError`` (a side or level that is not a well-formed ``[price, size]`` string pair,
        a non-numeric level, a price outside [0, 1], a bad size, unordered levels, a non-UTC
        timestamp) and ``decimal.InvalidOperation``. Unparseable numbers are normally converted
        to ``ValueError``, but ``InvalidOperation`` is part of the contract so a caller's catch
        does not depend on that detail.
        """
        return cls(
            bids=_parse_levels(book["bids"]),
            asks=_parse_levels(book["asks"]),
            ts=ts,
            venue_ts=venue_ts,
        )


def _parse_levels(raw: object) -> tuple[BookLevel, ...]:
    if isinstance(raw, str) or not isinstance(raw, Sequence):
        raise ValueError("book side must be a sequence of [price, size] levels")
    levels: list[BookLevel] = []
    for item in raw:
        if isinstance(item, str) or not isinstance(item, Sequence) or len(item) != 2:
            raise ValueError(f"book level must be a [price, size] pair: {item!r}")
        price, size = item
        if not isinstance(price, str) or not isinstance(size, str):
            raise ValueError(f"book level values must be strings: {item!r}")
        try:
            levels.append(BookLevel(Decimal(price), Decimal(size)))
        except ArithmeticError as exc:
            raise ValueError(f"book level is not numeric: {item!r}") from exc
    return tuple(levels)


class VenueAdapter(Protocol):
    venue: Venue

    async def list_markets(self, series_code: str) -> list[VenueMarket]: ...

    async def order_book(self, ref: MarketRef) -> OrderBook: ...

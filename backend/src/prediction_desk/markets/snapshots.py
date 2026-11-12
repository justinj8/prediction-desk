"""Order-book snapshots for open, mapped contracts. Observations only: nothing is audited."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from prediction_desk.clock import Clock
from prediction_desk.db.models import EventContract, MarketSnapshot
from prediction_desk.domain.enums import MappingStatus, Venue
from prediction_desk.domain.types import require_utc
from prediction_desk.markets.types import MarketRef, VenueAdapter
from prediction_desk.net.http import ProviderError


@dataclass
class SnapshotResult:
    recorded: int = 0
    failures: list[tuple[int, str]] = field(default_factory=list)


async def record_snapshots(
    sessions: async_sessionmaker[AsyncSession],
    adapters: Mapping[Venue, VenueAdapter],
    clock: Clock,
) -> SnapshotResult:
    """Snapshot the book of every open, mapped contract on a venue with an adapter.

    Collects: a ``ProviderError`` for one contract is recorded in ``result.failures`` as
    ``(contract_id, "Type: message")`` and the run continues. Each snapshot is committed in
    its own short transaction after its fetch, so no transaction is open during HTTP and a
    later failure (collected or raised) never loses snapshots already taken.

    LIVE writer only: ``ts`` is the adapter's ``clock.now()`` at fetch time. Calling this under
    a ``SimClock`` set in the past would write today's books into the past and leak them into
    replay; replay must read stored snapshots via ``latest_snapshot_asof``.
    """
    result = SnapshotResult()
    async with sessions() as session:
        targets = (
            await session.execute(
                select(
                    EventContract.id,
                    EventContract.venue,
                    EventContract.venue_market_id,
                    EventContract.yes_token_id,
                )
                .where(
                    EventContract.mapping_status == MappingStatus.MAPPED,
                    EventContract.close_time > clock.now(),
                    EventContract.venue.in_(list(adapters)),
                )
                .order_by(EventContract.id)
            )
        ).all()
    for contract_id, venue, market_id, yes_token_id in targets:
        try:
            book = await adapters[venue].order_book(MarketRef(market_id, yes_token_id))
        except ProviderError as e:
            result.failures.append((contract_id, f"{type(e).__name__}: {e}"))
            continue
        bid, ask = book.best_bid, book.best_ask
        async with sessions() as session, session.begin():
            session.add(
                MarketSnapshot(
                    contract_id=contract_id,
                    ts=book.ts,
                    venue_ts=book.venue_ts,
                    best_bid=None if bid is None else bid.price,
                    best_ask=None if ask is None else ask.price,
                    bid_size=None if bid is None else bid.size,
                    ask_size=None if ask is None else ask.size,
                    book=book.to_json(),
                )
            )
        result.recorded += 1
    return result


async def latest_snapshot_asof(
    session: AsyncSession, contract_id: int, asof: datetime
) -> MarketSnapshot | None:
    """Newest snapshot at or before ``asof``; never a later one."""
    return (
        await session.execute(
            select(MarketSnapshot)
            .where(
                MarketSnapshot.contract_id == contract_id, MarketSnapshot.ts <= require_utc(asof)
            )
            .order_by(MarketSnapshot.ts.desc(), MarketSnapshot.id.desc())
            .limit(1)
        )
    ).scalar_one_or_none()

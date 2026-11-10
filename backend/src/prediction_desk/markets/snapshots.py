"""Order-book snapshots for open, mapped contracts. Observations only: nothing is audited."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

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
    session: AsyncSession, adapters: Mapping[Venue, VenueAdapter], clock: Clock
) -> SnapshotResult:
    result = SnapshotResult()
    contracts = (
        await session.execute(
            select(EventContract)
            .where(
                EventContract.mapping_status == MappingStatus.MAPPED,
                EventContract.close_time > clock.now(),
                EventContract.venue.in_(list(adapters)),
            )
            .order_by(EventContract.id)
        )
    ).scalars()
    for contract in list(contracts):
        adapter = adapters[contract.venue]
        try:
            book = await adapter.order_book(
                MarketRef(contract.venue_market_id, contract.yes_token_id)
            )
        except ProviderError as e:
            result.failures.append((contract.id, f"{type(e).__name__}: {e}"))
            continue
        bid, ask = book.best_bid, book.best_ask
        session.add(
            MarketSnapshot(
                contract_id=contract.id,
                ts=book.ts,
                best_bid=None if bid is None else bid.price,
                best_ask=None if ask is None else ask.price,
                bid_size=None if bid is None else bid.size,
                ask_size=None if ask is None else ask.size,
                book=book.to_json(),
            )
        )
        result.recorded += 1
    await session.commit()
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

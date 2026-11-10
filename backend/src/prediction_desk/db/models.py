from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    CHAR,
    BigInteger,
    CheckConstraint,
    Date,
    Enum,
    ForeignKey,
    Identity,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from prediction_desk.db.base import Base
from prediction_desk.db.types import UTCDateTime, str_enum
from prediction_desk.domain.enums import (
    Comparator,
    DataSource,
    MappingStatus,
    Measure,
    Series,
    Venue,
)

__all__ = [
    "AuditEvent",
    "DataObservation",
    "EventContract",
    "MarketSnapshot",
    "Release",
    "Series",
]


class Release(Base):
    __tablename__ = "releases"
    __table_args__ = (UniqueConstraint("series", "period"),)

    # BY DEFAULT, matching sa.Identity() in migration 0001.
    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    series: Mapped[Series] = mapped_column(Enum(Series, name="series"))
    period: Mapped[str] = mapped_column(String)
    scheduled_at: Mapped[datetime] = mapped_column(UTCDateTime())
    actual_value: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    released_at: Mapped[datetime | None] = mapped_column(UTCDateTime())


class AuditEvent(Base):
    __tablename__ = "audit_events"
    __table_args__ = (Index("ix_audit_events_entity_entity_id", "entity", "entity_id"),)

    # BY DEFAULT (not ALWAYS): AuditLog.append inserts an explicitly reserved seq.
    seq: Mapped[int] = mapped_column(BigInteger, Identity(always=False), primary_key=True)
    ts: Mapped[datetime] = mapped_column(UTCDateTime())
    actor: Mapped[str] = mapped_column(String)
    entity: Mapped[str] = mapped_column(String)
    entity_id: Mapped[str] = mapped_column(String)
    action: Mapped[str] = mapped_column(String)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    prev_hash: Mapped[str] = mapped_column(CHAR(64))
    hash: Mapped[str] = mapped_column(CHAR(64), unique=True)


class DataObservation(Base):
    """Point-in-time macro observation; one row per (series, obs_date, vintage)."""

    __tablename__ = "data_observations"

    source: Mapped[DataSource] = mapped_column(
        str_enum(DataSource, "data_source"), primary_key=True
    )
    series_id: Mapped[str] = mapped_column(String, primary_key=True)
    obs_date: Mapped[date] = mapped_column(Date, primary_key=True)
    vintage_ts: Mapped[datetime] = mapped_column(UTCDateTime(), primary_key=True)
    # FRED realtime_start; NULL for Alpha Vantage.
    published_on: Mapped[date | None] = mapped_column(Date)
    value: Mapped[Decimal] = mapped_column(Numeric(20, 6))
    ingested_at: Mapped[datetime] = mapped_column(UTCDateTime())


class EventContract(Base):
    """Canonical venue contract. strike_lo is the single strike for GT/GTE/LT/LTE/EQ;
    BETWEEN uses strike_lo..strike_hi."""

    __tablename__ = "event_contracts"
    __table_args__ = (
        UniqueConstraint("venue", "venue_market_id"),
        CheckConstraint(
            "mapping_status <> 'mapped' OR (release_id IS NOT NULL AND measure IS NOT NULL "
            "AND comparator IS NOT NULL AND strike_lo IS NOT NULL)",
            name="mapped_complete",
        ),
    )

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    venue: Mapped[Venue] = mapped_column(str_enum(Venue, "venue"))
    venue_market_id: Mapped[str] = mapped_column(String)
    venue_event_id: Mapped[str] = mapped_column(String)
    yes_token_id: Mapped[str | None] = mapped_column(String)
    title: Mapped[str] = mapped_column(Text)
    rules_text: Mapped[str] = mapped_column(Text)
    close_time: Mapped[datetime] = mapped_column(UTCDateTime())
    release_id: Mapped[int | None] = mapped_column(ForeignKey("releases.id"))
    measure: Mapped[Measure | None] = mapped_column(str_enum(Measure, "measure"))
    comparator: Mapped[Comparator | None] = mapped_column(str_enum(Comparator, "comparator"))
    strike_lo: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    strike_hi: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    unit: Mapped[str | None] = mapped_column(String)
    resolution_precision: Mapped[int | None] = mapped_column(Integer)
    resolution_source: Mapped[str | None] = mapped_column(String)
    mapping_status: Mapped[MappingStatus] = mapped_column(str_enum(MappingStatus, "mapping_status"))
    mapping_confidence: Mapped[Decimal] = mapped_column(Numeric(3, 2))
    mapped_by: Mapped[str] = mapped_column(String)
    unmapped_reason: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime())


class MarketSnapshot(Base):
    """Order-book snapshot. best_bid/best_ask are YES prices; book is best-first
    {"bids": [[price, size], ...], "asks": [...]} with string values."""

    __tablename__ = "market_snapshots"
    __table_args__ = (Index("ix_market_snapshots_contract_id_ts", "contract_id", "ts"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    contract_id: Mapped[int] = mapped_column(ForeignKey("event_contracts.id"))
    ts: Mapped[datetime] = mapped_column(UTCDateTime())
    best_bid: Mapped[Decimal | None] = mapped_column(Numeric(6, 5))
    best_ask: Mapped[Decimal | None] = mapped_column(Numeric(6, 5))
    bid_size: Mapped[Decimal | None] = mapped_column(Numeric(20, 2))
    ask_size: Mapped[Decimal | None] = mapped_column(Numeric(20, 2))
    book: Mapped[dict[str, Any]] = mapped_column(JSONB)

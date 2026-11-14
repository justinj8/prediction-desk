from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    CHAR,
    BigInteger,
    Boolean,
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
    false,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from prediction_desk.db.base import Base
from prediction_desk.db.types import UTCDateTime, str_enum
from prediction_desk.domain.enums import (
    BookMode,
    Comparator,
    DataSource,
    MappingStatus,
    Measure,
    OrderStatus,
    PositionSide,
    Series,
    Slot,
    Venue,
    Verdict,
)

__all__ = [
    "AuditEvent",
    "BookState",
    "DataObservation",
    "EventContract",
    "Fill",
    "MarketSnapshot",
    "MeasureOutcome",
    "Order",
    "Position",
    "Release",
    "RiskDecision",
    "RiskLimitsRow",
    "Series",
    "Settlement",
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
            "AND comparator IS NOT NULL AND strike_lo IS NOT NULL AND unit IS NOT NULL "
            "AND resolution_precision IS NOT NULL AND resolution_source IS NOT NULL)",
            name="mapped_complete",
        ),
        CheckConstraint("mapping_confidence BETWEEN 0 AND 1", name="confidence_range"),
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
    {"bids": [[price, size], ...], "asks": [...]} with string values.

    ts is when we observed the book (as-of reads compare against it); venue_ts is the venue's
    own book timestamp, NULL when the venue publishes none (Kalshi)."""

    __tablename__ = "market_snapshots"
    __table_args__ = (Index("ix_market_snapshots_contract_id_ts", "contract_id", "ts"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    contract_id: Mapped[int] = mapped_column(ForeignKey("event_contracts.id"))
    ts: Mapped[datetime] = mapped_column(UTCDateTime())
    venue_ts: Mapped[datetime | None] = mapped_column(UTCDateTime())
    best_bid: Mapped[Decimal | None] = mapped_column(Numeric(6, 5))
    best_ask: Mapped[Decimal | None] = mapped_column(Numeric(6, 5))
    bid_size: Mapped[Decimal | None] = mapped_column(Numeric(24, 6))
    ask_size: Mapped[Decimal | None] = mapped_column(Numeric(24, 6))
    book: Mapped[dict[str, Any]] = mapped_column(JSONB)


class RiskLimitsRow(Base):
    """Versioned risk parameters; the single active row is the live policy."""

    __tablename__ = "risk_limits"
    __table_args__ = (
        Index("uq_risk_limits_active", "active", unique=True, postgresql_where=text("active")),
    )

    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    params: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_by: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())
    active: Mapped[bool] = mapped_column(Boolean)


class BookState(Base):
    """One paper book: cash, last-marked equity, drawdown anchors and the kill switch."""

    __tablename__ = "book_state"
    __table_args__ = (
        UniqueConstraint("mode", "replay_batch_id", postgresql_nulls_not_distinct=True),
        CheckConstraint("mode <> 'live' OR replay_batch_id IS NULL", name="live_has_no_batch"),
    )

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    mode: Mapped[BookMode] = mapped_column(str_enum(BookMode, "book_mode"))
    replay_batch_id: Mapped[int | None] = mapped_column(BigInteger)
    cash_cents: Mapped[int] = mapped_column(BigInteger)
    equity_cents: Mapped[int] = mapped_column(BigInteger)
    peak_equity_cents: Mapped[int] = mapped_column(BigInteger)
    day_start_equity_cents: Mapped[int] = mapped_column(BigInteger)
    day_start_date: Mapped[date] = mapped_column(Date)
    kill_switch_on: Mapped[bool] = mapped_column(Boolean)
    kill_reason: Mapped[str | None] = mapped_column(Text)
    kill_actor: Mapped[str | None] = mapped_column(Text)
    kill_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime())


class RiskDecision(Base):
    """Immutable verdict for one (book, run, slot, contract), with the inputs that produced it."""

    __tablename__ = "risk_decisions"
    __table_args__ = (
        UniqueConstraint("book_id", "run_key", "slot", "contract_id"),
        CheckConstraint(
            "verdict <> 'approved' OR (side IS NOT NULL AND qty >= 1 AND limit_price IS NOT NULL)",
            name="approved_complete",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    book_id: Mapped[int] = mapped_column(ForeignKey("book_state.id"))
    run_key: Mapped[str] = mapped_column(Text)
    slot: Mapped[Slot] = mapped_column(str_enum(Slot, "slot"))
    release_id: Mapped[int] = mapped_column(ForeignKey("releases.id"))
    contract_id: Mapped[int] = mapped_column(ForeignKey("event_contracts.id"))
    side: Mapped[PositionSide | None] = mapped_column(str_enum(PositionSide, "position_side"))
    inputs_hash: Mapped[str] = mapped_column(CHAR(64))
    engine_version: Mapped[str] = mapped_column(Text)
    fair_yes: Mapped[Decimal] = mapped_column(Numeric(6, 5))
    # Clamped fair probability of the chosen side.
    fair_clamped: Mapped[Decimal | None] = mapped_column(Numeric(6, 5))
    cost_per_contract: Mapped[Decimal | None] = mapped_column(Numeric(12, 8))
    limit_price: Mapped[Decimal | None] = mapped_column(Numeric(6, 5))
    kelly_raw_cents: Mapped[int | None] = mapped_column(BigInteger)
    stake_engine_cents: Mapped[int | None] = mapped_column(BigInteger)
    stake_officer_cents: Mapped[int | None] = mapped_column(BigInteger)
    stake_final_cents: Mapped[int | None] = mapped_column(BigInteger)
    qty: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    verdict: Mapped[Verdict] = mapped_column(str_enum(Verdict, "verdict"))
    # Ordered list of reason codes.
    reasons: Mapped[list[Any]] = mapped_column(JSONB)
    limits_version: Mapped[int] = mapped_column(ForeignKey("risk_limits.version"))
    limits_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB)
    decided_at: Mapped[datetime] = mapped_column(UTCDateTime())


class Order(Base):
    """IOC marketable-limit order; always traceable to the approving risk decision."""

    __tablename__ = "orders"
    __table_args__ = (
        CheckConstraint("0 <= filled_qty AND filled_qty <= qty", name="filled_within_qty"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    client_order_id: Mapped[str] = mapped_column(CHAR(64), unique=True)
    risk_decision_id: Mapped[int] = mapped_column(ForeignKey("risk_decisions.id"), unique=True)
    book_id: Mapped[int] = mapped_column(ForeignKey("book_state.id"))
    contract_id: Mapped[int] = mapped_column(ForeignKey("event_contracts.id"))
    side: Mapped[PositionSide] = mapped_column(str_enum(PositionSide, "position_side"))
    limit_price: Mapped[Decimal] = mapped_column(Numeric(6, 5))
    qty: Mapped[int] = mapped_column(Integer)
    filled_qty: Mapped[int] = mapped_column(Integer)
    status: Mapped[OrderStatus] = mapped_column(str_enum(OrderStatus, "order_status"))
    status_reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())


class Fill(Base):
    """The single fill of an order (IOC: one row per order)."""

    __tablename__ = "fills"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"), unique=True)
    qty: Mapped[int] = mapped_column(Integer)
    avg_price: Mapped[Decimal] = mapped_column(Numeric(12, 8))
    cost_cents: Mapped[int] = mapped_column(BigInteger)
    fees_cents: Mapped[int] = mapped_column(BigInteger)
    slippage_cents: Mapped[int] = mapped_column(BigInteger)
    snapshot_id: Mapped[int] = mapped_column(ForeignKey("market_snapshots.id"))
    ts: Mapped[datetime] = mapped_column(UTCDateTime())


class Position(Base):
    """Held contracts on one side; cost_cents is price paid plus fees."""

    __tablename__ = "positions"
    __table_args__ = (UniqueConstraint("book_id", "contract_id"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    book_id: Mapped[int] = mapped_column(ForeignKey("book_state.id"))
    contract_id: Mapped[int] = mapped_column(ForeignKey("event_contracts.id"))
    side: Mapped[PositionSide] = mapped_column(str_enum(PositionSide, "position_side"))
    qty: Mapped[int] = mapped_column(Integer)
    cost_cents: Mapped[int] = mapped_column(BigInteger)
    realized_pnl_cents: Mapped[int] = mapped_column(BigInteger, server_default=text("0"))
    mark_price: Mapped[Decimal | None] = mapped_column(Numeric(6, 5))
    settled: Mapped[bool] = mapped_column(Boolean, server_default=false())
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime())


class MeasureOutcome(Base):
    """Official value of one measure for one release, with provenance (fred_first_print or
    admin_override)."""

    __tablename__ = "measure_outcomes"

    release_id: Mapped[int] = mapped_column(ForeignKey("releases.id"), primary_key=True)
    measure: Mapped[Measure] = mapped_column(str_enum(Measure, "measure"), primary_key=True)
    raw_value: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    reported_value: Mapped[Decimal] = mapped_column(Numeric(14, 4))
    source: Mapped[str] = mapped_column(Text)
    source_vintage_ts: Mapped[datetime | None] = mapped_column(UTCDateTime())
    recorded_by: Mapped[str] = mapped_column(Text)
    reason: Mapped[str | None] = mapped_column(Text)
    recorded_at: Mapped[datetime] = mapped_column(UTCDateTime())


class Settlement(Base):
    """Payout of one position; venue_yes_won is only compared, never decisive."""

    __tablename__ = "settlements"
    __table_args__ = (UniqueConstraint("book_id", "contract_id"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    book_id: Mapped[int] = mapped_column(ForeignKey("book_state.id"))
    contract_id: Mapped[int] = mapped_column(ForeignKey("event_contracts.id"))
    side: Mapped[PositionSide] = mapped_column(str_enum(PositionSide, "position_side"))
    qty: Mapped[int] = mapped_column(Integer)
    yes_won: Mapped[bool] = mapped_column(Boolean)
    payout_cents: Mapped[int] = mapped_column(BigInteger)
    realized_pnl_cents: Mapped[int] = mapped_column(BigInteger)
    reported_value: Mapped[Decimal] = mapped_column(Numeric(14, 4))
    venue_yes_won: Mapped[bool | None] = mapped_column(Boolean)
    mismatch: Mapped[bool] = mapped_column(Boolean)
    settled_at: Mapped[datetime] = mapped_column(UTCDateTime())

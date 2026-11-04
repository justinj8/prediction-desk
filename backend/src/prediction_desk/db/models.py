from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    CHAR,
    BigInteger,
    Enum,
    Identity,
    Index,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from prediction_desk.db.base import Base
from prediction_desk.db.types import UTCDateTime


class Series(StrEnum):
    CPI = "CPI"
    NFP = "NFP"
    FOMC = "FOMC"


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

from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from sqlalchemy import DateTime, Enum, Numeric, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from prediction_desk.db.base import Base


class Series(StrEnum):
    CPI = "CPI"
    NFP = "NFP"
    FOMC = "FOMC"


class Release(Base):
    __tablename__ = "releases"
    __table_args__ = (UniqueConstraint("series", "period"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    series: Mapped[Series] = mapped_column(Enum(Series, name="series"))
    period: Mapped[str] = mapped_column(String)
    scheduled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    actual_value: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    released_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

"""Column types that enforce domain invariants at the database boundary."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime
from sqlalchemy.engine import Dialect
from sqlalchemy.types import TypeDecorator

from prediction_desk.domain.types import require_utc


class UTCDateTime(TypeDecorator[datetime]):
    """timestamptz that rejects naive datetimes on the way in and returns UTC on the way out.

    The driver returns timestamptz in the session TimeZone, so results are normalized here
    rather than trusting the connection setting alone.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        return require_utc(value)

    def process_result_value(self, value: Any | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        assert isinstance(value, datetime)
        return value.astimezone(UTC)

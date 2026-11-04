from datetime import UTC, datetime, timedelta, timezone

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import StatementError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from prediction_desk.db.engine import make_sessionmaker
from prediction_desk.db.models import Release, Series

SCHEDULED = datetime(2024, 3, 12, 12, 30, tzinfo=UTC)


async def test_naive_datetime_is_rejected_at_the_db_boundary(session: AsyncSession) -> None:
    session.add(Release(series=Series.CPI, period="2024-02", scheduled_at=datetime(2024, 3, 12)))
    with pytest.raises((StatementError, ValueError), match="timezone"):
        await session.flush()
    await session.rollback()


async def test_aware_non_utc_datetime_is_stored_as_the_same_instant(session: AsyncSession) -> None:
    plus_two = SCHEDULED.astimezone(timezone(timedelta(hours=2)))
    session.add(Release(series=Series.CPI, period="2024-02", scheduled_at=plus_two))
    await session.commit()
    raw = await session.execute(text("SELECT scheduled_at = :t FROM releases"), {"t": SCHEDULED})
    assert raw.scalar_one() is True


@pytest.mark.parametrize("session_tz", [None, "America/New_York"])
async def test_values_read_back_are_utc(engine: AsyncEngine, session_tz: str | None) -> None:
    maker = make_sessionmaker(engine)
    async with maker() as s:
        s.add(Release(series=Series.NFP, period="2024-02", scheduled_at=SCHEDULED))
        await s.commit()
    async with maker() as fresh:  # fresh session: values come from the driver, not the map
        if session_tz is not None:
            await fresh.execute(text(f"SET TIME ZONE '{session_tz}'"))
        row = (await fresh.execute(select(Release.scheduled_at, Release.released_at))).one()
    assert row.scheduled_at == SCHEDULED and row.scheduled_at.tzinfo is UTC
    assert row.released_at is None


async def test_engine_sessions_run_in_utc(engine: AsyncEngine) -> None:
    async with engine.connect() as conn:
        assert (await conn.execute(text("SHOW timezone"))).scalar_one() == "UTC"

from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from prediction_desk.db.models import Release, Series


async def test_insert_and_read_release(session: AsyncSession) -> None:
    session.add(
        Release(
            series=Series.CPI,
            period="2024-02",
            scheduled_at=datetime(2024, 3, 12, 12, 30, tzinfo=UTC),
        )
    )
    await session.commit()
    row = (await session.execute(select(Release))).scalar_one()
    assert row.series is Series.CPI and row.actual_value is None
    assert row.scheduled_at == datetime(2024, 3, 12, 12, 30, tzinfo=UTC)


async def test_series_period_unique(session: AsyncSession) -> None:
    ts = datetime(2024, 3, 12, 12, 30, tzinfo=UTC)
    session.add_all(
        [
            Release(series=Series.NFP, period="2024-02", scheduled_at=ts),
            Release(series=Series.NFP, period="2024-02", scheduled_at=ts),
        ]
    )
    with pytest.raises(IntegrityError):
        await session.commit()

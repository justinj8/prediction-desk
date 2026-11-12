from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from prediction_desk.audit.log import AuditLog
from prediction_desk.clock import SimClock
from prediction_desk.db.engine import make_engine, make_sessionmaker
from prediction_desk.db.models import Release, Series
from prediction_desk.db.uow import unit_of_work

T0 = datetime(2024, 3, 12, 12, 0, tzinfo=UTC)


def _release(period: str) -> Release:
    return Release(series=Series.CPI, period=period, scheduled_at=T0)


async def _count(session: AsyncSession) -> int:
    return (await session.execute(select(func.count()).select_from(Release))).scalar_one()


async def test_commits_on_clean_exit(sessions: async_sessionmaker[AsyncSession]) -> None:
    async with unit_of_work(sessions) as uow:
        uow.add(_release("2024-02"))
    async with sessions() as fresh:
        assert await _count(fresh) == 1


async def test_rolls_back_and_reraises(sessions: async_sessionmaker[AsyncSession]) -> None:
    with pytest.raises(ValueError, match="boom"):
        async with unit_of_work(sessions) as uow:
            uow.add(_release("2024-02"))
            await uow.flush()
            raise ValueError("boom")
    async with sessions() as fresh:
        assert await _count(fresh) == 0


async def test_runs_read_committed_even_if_engine_default_differs(db_url: str) -> None:
    eng: AsyncEngine = make_engine(db_url).execution_options(isolation_level="REPEATABLE READ")
    try:
        async with unit_of_work(make_sessionmaker(eng)) as uow:
            level = (
                await uow.execute(text("SELECT current_setting('transaction_isolation')"))
            ).scalar_one()
        assert level == "read committed"
    finally:
        await eng.dispose()


async def test_refuses_autocommit_engine(db_url: str) -> None:
    eng = make_engine(db_url).execution_options(isolation_level="AUTOCOMMIT")
    try:
        with pytest.raises(RuntimeError, match="unit_of_work refuses an AUTOCOMMIT engine"):
            async with unit_of_work(make_sessionmaker(eng)):
                pytest.fail("body must not run")  # pragma: no cover
    finally:
        await eng.dispose()


async def test_audit_append_works_inside_uow(sessions: async_sessionmaker[AsyncSession]) -> None:
    log = AuditLog(SimClock(T0))
    async with unit_of_work(sessions) as uow:
        await log.append(
            uow, actor="system", entity="test", entity_id="1", action="noop", payload={}
        )
    async with sessions() as fresh:
        assert (await log.verify(fresh)).ok

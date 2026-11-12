"""Unit of work: one READ COMMITTED transaction per business operation."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker


def _is_autocommit(bind: object) -> bool:
    if not isinstance(bind, AsyncEngine):
        return False
    return bind.get_execution_options().get("isolation_level") == "AUTOCOMMIT"


@asynccontextmanager
async def unit_of_work(sessions: async_sessionmaker[AsyncSession]) -> AsyncIterator[AsyncSession]:
    """Yield a session inside one READ COMMITTED transaction.

    Commits on a clean exit, rolls back and re-raises on any exception, and always closes the
    session. An AUTOCOMMIT engine is refused: it would commit each statement on its own and
    void the transaction (Plan 1 R14). The isolation level is forced to READ COMMITTED so
    ``AuditLog.append`` is valid whatever the engine default is.
    """
    if _is_autocommit(sessions.kw.get("bind")):
        raise RuntimeError("unit_of_work refuses an AUTOCOMMIT engine")
    async with sessions() as session:
        if _is_autocommit(session.bind):
            raise RuntimeError("unit_of_work refuses an AUTOCOMMIT engine")
        async with session.begin():
            # Must be the transaction's first statement.
            await session.execute(text("SET TRANSACTION ISOLATION LEVEL READ COMMITTED"))
            level = (
                await session.execute(text("SELECT current_setting('transaction_isolation')"))
            ).scalar_one()
            if level != "read committed":  # pragma: no cover - SET above guarantees it
                raise RuntimeError(f"unit_of_work needs READ COMMITTED, got {level}")
            yield session

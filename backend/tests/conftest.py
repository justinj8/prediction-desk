import os
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from psycopg import sql
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from prediction_desk.config import DEFAULT_TEST_ADMIN_URL, normalize_db_url
from prediction_desk.db.engine import make_engine, make_sessionmaker

_ALEMBIC_INI = Path(__file__).resolve().parents[1] / "alembic.ini"


def _admin_url() -> str:
    return normalize_db_url(os.environ.get("TEST_DATABASE_ADMIN_URL", DEFAULT_TEST_ADMIN_URL))


def _raw_url(url: str) -> str:
    """Strip the SQLAlchemy driver suffix so psycopg can connect directly."""
    return url.replace("postgresql+psycopg://", "postgresql://", 1)


def _url_for_database(name: str) -> str:
    return make_url(_admin_url()).set(database=name).render_as_string(hide_password=False)


def _create_database(name: str, template: str | None = None) -> None:
    stmt = sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name))
    if template is not None:
        stmt += sql.SQL(" TEMPLATE {}").format(sql.Identifier(template))
    with psycopg.connect(_raw_url(_admin_url()), autocommit=True) as conn:
        conn.execute(stmt)


def _drop_database(name: str) -> None:
    stmt = sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name))
    with psycopg.connect(_raw_url(_admin_url()), autocommit=True) as conn:
        conn.execute(stmt)


@pytest.fixture(scope="session")
def template_db_url() -> Iterator[str]:
    name = f"pd_test_template_{os.getpid()}"
    _drop_database(name)
    _create_database(name)
    url = _url_for_database(name)
    try:
        cfg = Config(str(_ALEMBIC_INI))
        cfg.set_main_option("sqlalchemy.url", url)
        command.upgrade(cfg, "head")
        yield url
    finally:
        _drop_database(name)


@pytest.fixture
def db_url(template_db_url: str) -> Iterator[str]:
    template = make_url(template_db_url).database
    assert template is not None
    name = f"pd_test_{uuid.uuid4().hex[:8]}"
    _create_database(name, template=template)
    try:
        yield _url_for_database(name)
    finally:
        _drop_database(name)


@pytest.fixture
async def engine(db_url: str) -> AsyncIterator[AsyncEngine]:
    eng = make_engine(db_url)
    try:
        yield eng
    finally:
        await eng.dispose()


@pytest.fixture
async def session(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    async with make_sessionmaker(engine)() as s:
        yield s


@pytest.fixture(scope="module")
def vcr_config() -> dict[str, Any]:
    """pytest-recording settings: never persist API keys; store readable response bodies."""
    return {
        "filter_query_parameters": [("api_key", "REDACTED"), ("apikey", "REDACTED")],
        "decode_compressed_response": True,
    }

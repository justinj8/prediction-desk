from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

_ALEMBIC_INI = Path(__file__).resolve().parents[2] / "alembic.ini"


def _cfg(url: str) -> Config:
    cfg = Config(str(_ALEMBIC_INI))
    cfg.set_main_option("sqlalchemy.url", url)
    return cfg


def test_upgrade_creates_releases_and_downgrade_removes(db_url: str) -> None:
    sync_engine = create_engine(db_url)
    assert "releases" in inspect(sync_engine).get_table_names()
    command.downgrade(_cfg(db_url), "base")
    assert "releases" not in inspect(sync_engine).get_table_names()
    command.upgrade(_cfg(db_url), "head")
    assert "releases" in inspect(sync_engine).get_table_names()
    sync_engine.dispose()

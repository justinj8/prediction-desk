from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import create_engine

import prediction_desk.db.models  # noqa: F401  (registers models on Base.metadata)
from prediction_desk.db.base import Base


def test_orm_metadata_matches_migrations(db_url: str) -> None:
    """The per-test DB is built by `alembic upgrade head`; the ORM must describe it exactly."""
    engine = create_engine(db_url)
    try:
        with engine.connect() as conn:
            ctx = MigrationContext.configure(
                conn, opts={"compare_type": True, "compare_server_default": True}
            )
            assert compare_metadata(ctx, Base.metadata) == []
    finally:
        engine.dispose()

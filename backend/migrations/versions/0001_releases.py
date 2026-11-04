"""create releases table

Revision ID: 0001
Revises:
Create Date: 2026-11-04
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

series_enum = postgresql.ENUM("CPI", "NFP", "FOMC", name="series", create_type=False)


def upgrade() -> None:
    series_enum.create(op.get_bind(), checkfirst=False)
    op.create_table(
        "releases",
        sa.Column("id", sa.Integer(), sa.Identity(), nullable=False),
        sa.Column("series", series_enum, nullable=False),
        sa.Column("period", sa.String(), nullable=False),
        sa.Column("scheduled_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("actual_value", sa.Numeric(12, 4), nullable=True),
        sa.Column("released_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_releases")),
        sa.UniqueConstraint("series", "period", name=op.f("uq_releases_series")),
    )


def downgrade() -> None:
    op.drop_table("releases")
    series_enum.drop(op.get_bind(), checkfirst=False)

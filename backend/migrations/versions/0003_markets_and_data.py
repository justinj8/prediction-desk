"""create event_contracts, market_snapshots, data_observations

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

data_source_enum = postgresql.ENUM("fred", "alpha_vantage", name="data_source", create_type=False)
venue_enum = postgresql.ENUM("kalshi", "polymarket", name="venue", create_type=False)
measure_enum = postgresql.ENUM(
    "cpi_mom",
    "core_cpi_mom",
    "cpi_yoy",
    "core_cpi_yoy",
    "nfp_change",
    "fomc_change_bp",
    name="measure",
    create_type=False,
)
comparator_enum = postgresql.ENUM(
    "gt", "gte", "lt", "lte", "eq", "between", name="comparator", create_type=False
)
mapping_status_enum = postgresql.ENUM(
    "mapped", "unmapped", name="mapping_status", create_type=False
)
_ENUMS = (data_source_enum, venue_enum, measure_enum, comparator_enum, mapping_status_enum)


def upgrade() -> None:
    bind = op.get_bind()
    for enum in _ENUMS:
        enum.create(bind, checkfirst=False)

    op.create_table(
        "data_observations",
        sa.Column("source", data_source_enum, nullable=False),
        sa.Column("series_id", sa.String(), nullable=False),
        sa.Column("obs_date", sa.Date(), nullable=False),
        sa.Column("vintage_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("published_on", sa.Date(), nullable=True),
        sa.Column("value", sa.Numeric(20, 6), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint(
            "source", "series_id", "obs_date", "vintage_ts", name=op.f("pk_data_observations")
        ),
    )
    op.create_table(
        "event_contracts",
        sa.Column("id", sa.Integer(), sa.Identity(), nullable=False),
        sa.Column("venue", venue_enum, nullable=False),
        sa.Column("venue_market_id", sa.String(), nullable=False),
        sa.Column("venue_event_id", sa.String(), nullable=False),
        sa.Column("yes_token_id", sa.String(), nullable=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("rules_text", sa.Text(), nullable=False),
        sa.Column("close_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("release_id", sa.Integer(), nullable=True),
        sa.Column("measure", measure_enum, nullable=True),
        sa.Column("comparator", comparator_enum, nullable=True),
        sa.Column("strike_lo", sa.Numeric(14, 4), nullable=True),
        sa.Column("strike_hi", sa.Numeric(14, 4), nullable=True),
        sa.Column("unit", sa.String(), nullable=True),
        sa.Column("resolution_precision", sa.Integer(), nullable=True),
        sa.Column("resolution_source", sa.String(), nullable=True),
        sa.Column("mapping_status", mapping_status_enum, nullable=False),
        sa.Column("mapping_confidence", sa.Numeric(3, 2), nullable=False),
        sa.Column("mapped_by", sa.String(), nullable=False),
        sa.Column("unmapped_reason", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "mapping_status <> 'mapped' OR (release_id IS NOT NULL AND measure IS NOT NULL "
            "AND comparator IS NOT NULL AND strike_lo IS NOT NULL)",
            name=op.f("ck_event_contracts_mapped_complete"),
        ),
        sa.CheckConstraint(
            "mapping_confidence BETWEEN 0 AND 1",
            name=op.f("ck_event_contracts_confidence_range"),
        ),
        sa.ForeignKeyConstraint(
            ["release_id"], ["releases.id"], name=op.f("fk_event_contracts_release_id_releases")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_event_contracts")),
        sa.UniqueConstraint("venue", "venue_market_id", name=op.f("uq_event_contracts_venue")),
    )
    op.create_table(
        "market_snapshots",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("contract_id", sa.Integer(), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("venue_ts", sa.DateTime(timezone=True), nullable=True),
        sa.Column("best_bid", sa.Numeric(6, 5), nullable=True),
        sa.Column("best_ask", sa.Numeric(6, 5), nullable=True),
        sa.Column("bid_size", sa.Numeric(24, 6), nullable=True),
        sa.Column("ask_size", sa.Numeric(24, 6), nullable=True),
        sa.Column("book", postgresql.JSONB(), nullable=False),
        sa.ForeignKeyConstraint(
            ["contract_id"],
            ["event_contracts.id"],
            name=op.f("fk_market_snapshots_contract_id_event_contracts"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_market_snapshots")),
    )
    op.create_index("ix_market_snapshots_contract_id_ts", "market_snapshots", ["contract_id", "ts"])


def downgrade() -> None:
    op.drop_index("ix_market_snapshots_contract_id_ts", table_name="market_snapshots")
    op.drop_table("market_snapshots")
    op.drop_table("event_contracts")
    op.drop_table("data_observations")
    bind = op.get_bind()
    for enum in reversed(_ENUMS):
        enum.drop(bind, checkfirst=False)

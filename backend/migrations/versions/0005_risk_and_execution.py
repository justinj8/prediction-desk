"""create risk limits, book state, decisions, orders, fills, positions, outcomes, settlements

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-12
"""

from collections.abc import Sequence
from datetime import UTC, datetime

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: str | Sequence[str] | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

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
book_mode_enum = postgresql.ENUM("live", "replay", name="book_mode", create_type=False)
slot_enum = postgresql.ENUM("t-24h", "t-2h", "t-15m", name="slot", create_type=False)
verdict_enum = postgresql.ENUM("approved", "rejected", name="verdict", create_type=False)
order_status_enum = postgresql.ENUM(
    "filled", "partial", "cancelled", "refused", name="order_status", create_type=False
)
position_side_enum = postgresql.ENUM("yes", "no", name="position_side", create_type=False)
_NEW_ENUMS = (book_mode_enum, slot_enum, verdict_enum, order_status_enum, position_side_enum)

# Frozen copy of the v1 defaults: a migration must not change when application code does.
_V1_PARAMS = {
    "kelly_multiplier": "0.25",
    "cap_contract": "0.02",
    "cap_event": "0.05",
    "cap_portfolio": "0.15",
    "liquidity_fraction": "0.25",
    "min_edge": "0.04",
    "min_mapping_confidence": "0.90",
    "max_quote_age_s": 60,
    "cutoff_minutes": 5,
    "fair_clamp_lo": "0.02",
    "fair_clamp_hi": "0.98",
    "daily_loss_kill": "0.08",
    "drawdown_kill": "0.15",
}


def _ts(name: str) -> sa.Column[datetime]:
    return sa.Column(name, sa.DateTime(timezone=True), nullable=False)


def upgrade() -> None:
    bind = op.get_bind()
    for enum in _NEW_ENUMS:
        enum.create(bind, checkfirst=True)

    risk_limits = op.create_table(
        "risk_limits",
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("params", postgresql.JSONB(), nullable=False),
        sa.Column("created_by", sa.Text(), nullable=False),
        _ts("created_at"),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("version", name=op.f("pk_risk_limits")),
    )
    op.create_index(
        "uq_risk_limits_active",
        "risk_limits",
        ["active"],
        unique=True,
        postgresql_where=sa.text("active"),
    )
    op.bulk_insert(
        risk_limits,
        [
            {
                "version": 1,
                "params": _V1_PARAMS,
                "created_by": "migration:0005",
                "created_at": datetime(2026, 11, 12, tzinfo=UTC),
                "active": True,
            }
        ],
    )

    op.create_table(
        "book_state",
        sa.Column("id", sa.Integer(), sa.Identity(), nullable=False),
        sa.Column("mode", book_mode_enum, nullable=False),
        sa.Column("replay_batch_id", sa.BigInteger(), nullable=True),
        sa.Column("cash_cents", sa.BigInteger(), nullable=False),
        sa.Column("equity_cents", sa.BigInteger(), nullable=False),
        sa.Column("peak_equity_cents", sa.BigInteger(), nullable=False),
        sa.Column("day_start_equity_cents", sa.BigInteger(), nullable=False),
        sa.Column("day_start_date", sa.Date(), nullable=False),
        sa.Column("kill_switch_on", sa.Boolean(), nullable=False),
        sa.Column("kill_reason", sa.Text(), nullable=True),
        sa.Column("kill_actor", sa.Text(), nullable=True),
        sa.Column("kill_at", sa.DateTime(timezone=True), nullable=True),
        _ts("created_at"),
        _ts("updated_at"),
        sa.CheckConstraint(
            "mode <> 'live' OR replay_batch_id IS NULL",
            name=op.f("ck_book_state_live_has_no_batch"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_book_state")),
        sa.UniqueConstraint(
            "mode",
            "replay_batch_id",
            name=op.f("uq_book_state_mode"),
            postgresql_nulls_not_distinct=True,
        ),
    )

    op.create_table(
        "risk_decisions",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("book_id", sa.Integer(), nullable=False),
        sa.Column("run_key", sa.Text(), nullable=False),
        sa.Column("slot", slot_enum, nullable=False),
        sa.Column("release_id", sa.Integer(), nullable=False),
        sa.Column("contract_id", sa.Integer(), nullable=False),
        sa.Column("side", position_side_enum, nullable=True),
        sa.Column("inputs_hash", sa.CHAR(64), nullable=False),
        sa.Column("engine_version", sa.Text(), nullable=False),
        sa.Column("fair_yes", sa.Numeric(6, 5), nullable=False),
        sa.Column("fair_clamped", sa.Numeric(6, 5), nullable=True),
        sa.Column("cost_per_contract", sa.Numeric(12, 8), nullable=True),
        sa.Column("limit_price", sa.Numeric(6, 5), nullable=True),
        sa.Column("kelly_raw_cents", sa.BigInteger(), nullable=True),
        sa.Column("stake_engine_cents", sa.BigInteger(), nullable=True),
        sa.Column("stake_officer_cents", sa.BigInteger(), nullable=True),
        sa.Column("stake_final_cents", sa.BigInteger(), nullable=True),
        sa.Column("qty", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("verdict", verdict_enum, nullable=False),
        sa.Column("reasons", postgresql.JSONB(), nullable=False),
        sa.Column("limits_version", sa.Integer(), nullable=False),
        sa.Column("limits_snapshot", postgresql.JSONB(), nullable=False),
        _ts("decided_at"),
        sa.CheckConstraint(
            "verdict <> 'approved' OR (side IS NOT NULL AND qty >= 1 AND limit_price IS NOT NULL)",
            name=op.f("ck_risk_decisions_approved_complete"),
        ),
        sa.ForeignKeyConstraint(
            ["book_id"], ["book_state.id"], name=op.f("fk_risk_decisions_book_id_book_state")
        ),
        sa.ForeignKeyConstraint(
            ["release_id"], ["releases.id"], name=op.f("fk_risk_decisions_release_id_releases")
        ),
        sa.ForeignKeyConstraint(
            ["contract_id"],
            ["event_contracts.id"],
            name=op.f("fk_risk_decisions_contract_id_event_contracts"),
        ),
        sa.ForeignKeyConstraint(
            ["limits_version"],
            ["risk_limits.version"],
            name=op.f("fk_risk_decisions_limits_version_risk_limits"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_risk_decisions")),
        sa.UniqueConstraint(
            "book_id", "run_key", "slot", "contract_id", name=op.f("uq_risk_decisions_book_id")
        ),
    )

    op.create_table(
        "orders",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("client_order_id", sa.CHAR(64), nullable=False),
        sa.Column("risk_decision_id", sa.BigInteger(), nullable=False),
        sa.Column("book_id", sa.Integer(), nullable=False),
        sa.Column("contract_id", sa.Integer(), nullable=False),
        sa.Column("side", position_side_enum, nullable=False),
        sa.Column("limit_price", sa.Numeric(6, 5), nullable=False),
        sa.Column("qty", sa.Integer(), nullable=False),
        sa.Column("filled_qty", sa.Integer(), nullable=False),
        sa.Column("status", order_status_enum, nullable=False),
        sa.Column("status_reason", sa.Text(), nullable=True),
        _ts("created_at"),
        sa.CheckConstraint(
            "0 <= filled_qty AND filled_qty <= qty", name=op.f("ck_orders_filled_within_qty")
        ),
        sa.ForeignKeyConstraint(
            ["risk_decision_id"],
            ["risk_decisions.id"],
            name=op.f("fk_orders_risk_decision_id_risk_decisions"),
        ),
        sa.ForeignKeyConstraint(
            ["book_id"], ["book_state.id"], name=op.f("fk_orders_book_id_book_state")
        ),
        sa.ForeignKeyConstraint(
            ["contract_id"],
            ["event_contracts.id"],
            name=op.f("fk_orders_contract_id_event_contracts"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_orders")),
        sa.UniqueConstraint("client_order_id", name=op.f("uq_orders_client_order_id")),
        sa.UniqueConstraint("risk_decision_id", name=op.f("uq_orders_risk_decision_id")),
    )

    op.create_table(
        "fills",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("order_id", sa.BigInteger(), nullable=False),
        sa.Column("qty", sa.Integer(), nullable=False),
        sa.Column("avg_price", sa.Numeric(12, 8), nullable=False),
        sa.Column("cost_cents", sa.BigInteger(), nullable=False),
        sa.Column("fees_cents", sa.BigInteger(), nullable=False),
        sa.Column("slippage_cents", sa.BigInteger(), nullable=False),
        sa.Column("snapshot_id", sa.BigInteger(), nullable=False),
        _ts("ts"),
        sa.ForeignKeyConstraint(["order_id"], ["orders.id"], name=op.f("fk_fills_order_id_orders")),
        sa.ForeignKeyConstraint(
            ["snapshot_id"],
            ["market_snapshots.id"],
            name=op.f("fk_fills_snapshot_id_market_snapshots"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_fills")),
        sa.UniqueConstraint("order_id", name=op.f("uq_fills_order_id")),
    )

    op.create_table(
        "positions",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("book_id", sa.Integer(), nullable=False),
        sa.Column("contract_id", sa.Integer(), nullable=False),
        sa.Column("side", position_side_enum, nullable=False),
        sa.Column("qty", sa.Integer(), nullable=False),
        sa.Column("cost_cents", sa.BigInteger(), nullable=False),
        sa.Column(
            "realized_pnl_cents", sa.BigInteger(), server_default=sa.text("0"), nullable=False
        ),
        sa.Column("mark_price", sa.Numeric(6, 5), nullable=True),
        sa.Column("settled", sa.Boolean(), server_default=sa.false(), nullable=False),
        _ts("updated_at"),
        sa.ForeignKeyConstraint(
            ["book_id"], ["book_state.id"], name=op.f("fk_positions_book_id_book_state")
        ),
        sa.ForeignKeyConstraint(
            ["contract_id"],
            ["event_contracts.id"],
            name=op.f("fk_positions_contract_id_event_contracts"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_positions")),
        sa.UniqueConstraint("book_id", "contract_id", name=op.f("uq_positions_book_id")),
    )

    op.create_table(
        "measure_outcomes",
        sa.Column("release_id", sa.Integer(), nullable=False),
        sa.Column("measure", measure_enum, nullable=False),
        sa.Column("raw_value", sa.Numeric(20, 8), nullable=False),
        sa.Column("reported_value", sa.Numeric(14, 4), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("source_vintage_ts", sa.DateTime(timezone=True), nullable=True),
        sa.Column("recorded_by", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        _ts("recorded_at"),
        sa.ForeignKeyConstraint(
            ["release_id"], ["releases.id"], name=op.f("fk_measure_outcomes_release_id_releases")
        ),
        sa.PrimaryKeyConstraint("release_id", "measure", name=op.f("pk_measure_outcomes")),
    )

    op.create_table(
        "settlements",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("book_id", sa.Integer(), nullable=False),
        sa.Column("contract_id", sa.Integer(), nullable=False),
        sa.Column("side", position_side_enum, nullable=False),
        sa.Column("qty", sa.Integer(), nullable=False),
        sa.Column("yes_won", sa.Boolean(), nullable=False),
        sa.Column("payout_cents", sa.BigInteger(), nullable=False),
        sa.Column("realized_pnl_cents", sa.BigInteger(), nullable=False),
        sa.Column("reported_value", sa.Numeric(14, 4), nullable=False),
        sa.Column("venue_yes_won", sa.Boolean(), nullable=True),
        sa.Column("mismatch", sa.Boolean(), nullable=False),
        _ts("settled_at"),
        sa.ForeignKeyConstraint(
            ["book_id"], ["book_state.id"], name=op.f("fk_settlements_book_id_book_state")
        ),
        sa.ForeignKeyConstraint(
            ["contract_id"],
            ["event_contracts.id"],
            name=op.f("fk_settlements_contract_id_event_contracts"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_settlements")),
        sa.UniqueConstraint("book_id", "contract_id", name=op.f("uq_settlements_book_id")),
    )


def downgrade() -> None:
    for table in (
        "settlements",
        "measure_outcomes",
        "positions",
        "fills",
        "orders",
        "risk_decisions",
        "book_state",
    ):
        op.drop_table(table)
    op.drop_index("uq_risk_limits_active", table_name="risk_limits")
    op.drop_table("risk_limits")
    bind = op.get_bind()
    for enum in reversed(_NEW_ENUMS):
        enum.drop(bind, checkfirst=True)

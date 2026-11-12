"""require unit, resolution_precision and resolution_source on mapped contracts

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-11
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0004"
down_revision: str | Sequence[str] | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_NAME = "ck_event_contracts_mapped_complete"
_OLD = (
    "mapping_status <> 'mapped' OR (release_id IS NOT NULL AND measure IS NOT NULL "
    "AND comparator IS NOT NULL AND strike_lo IS NOT NULL)"
)
_NEW = (
    "mapping_status <> 'mapped' OR (release_id IS NOT NULL AND measure IS NOT NULL "
    "AND comparator IS NOT NULL AND strike_lo IS NOT NULL AND unit IS NOT NULL "
    "AND resolution_precision IS NOT NULL AND resolution_source IS NOT NULL)"
)


def upgrade() -> None:
    op.drop_constraint(op.f(_NAME), "event_contracts", type_="check")
    op.create_check_constraint(op.f(_NAME), "event_contracts", _NEW)


def downgrade() -> None:
    op.drop_constraint(op.f(_NAME), "event_contracts", type_="check")
    op.create_check_constraint(op.f(_NAME), "event_contracts", _OLD)

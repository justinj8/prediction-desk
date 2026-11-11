from datetime import UTC, datetime
from decimal import Decimal

import pytest

from prediction_desk.db.models import EventContract
from prediction_desk.domain.enums import Comparator, MappingStatus, Measure, Venue
from prediction_desk.markets.mapping import ContractTerms
from prediction_desk.markets.terms import rules_hash, terms_from_row

NOW = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)


def _row(**over: object) -> EventContract:
    fields: dict[str, object] = {
        "id": 7,
        "venue": Venue.KALSHI,
        "venue_market_id": "M",
        "venue_event_id": "E",
        "yes_token_id": None,
        "title": "t",
        "rules_text": "rules",
        "close_time": NOW,
        "release_id": 1,
        "measure": Measure.CPI_MOM,
        "comparator": Comparator.BETWEEN,
        "strike_lo": Decimal("0.2"),
        "strike_hi": Decimal("0.4"),
        "unit": "pct",
        "resolution_precision": 1,
        "resolution_source": "BLS",
        "mapping_status": MappingStatus.MAPPED,
        "mapping_confidence": Decimal("1.00"),
        "mapped_by": "rules:v2",
        "created_at": NOW,
        "updated_at": NOW,
    }
    fields.update(over)
    return EventContract(**fields)


def test_terms_from_mapped_row() -> None:
    assert terms_from_row(_row()) == ContractTerms(
        Measure.CPI_MOM, Comparator.BETWEEN, Decimal("0.2"), Decimal("0.4"), "pct", 1, "BLS"
    )
    assert terms_from_row(_row(comparator=Comparator.GT, strike_hi=None)).strike_hi is None


def test_terms_from_unmapped_row_raises() -> None:
    with pytest.raises(ValueError, match="contract 7 is not mapped"):
        terms_from_row(_row(mapping_status=MappingStatus.UNMAPPED))


@pytest.mark.parametrize(
    "missing",
    ["measure", "comparator", "strike_lo", "unit", "resolution_precision", "resolution_source"],
)
def test_terms_from_mapped_row_missing_field_raises(missing: str) -> None:
    with pytest.raises(ValueError, match="contract 7 is not mapped"):
        terms_from_row(_row(**{missing: None}))


def test_rules_hash_stable_and_sensitive() -> None:
    a = rules_hash(_row(rules_text="If CPI > 0.3"))
    assert a == rules_hash(_row(rules_text="If CPI > 0.3"))
    assert len(a) == 64 and int(a, 16) >= 0
    assert a != rules_hash(_row(rules_text="If CPI > 0.4"))
    assert a != rules_hash(_row(rules_text="If CPI > 0.3 "))

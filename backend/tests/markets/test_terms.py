from datetime import UTC, datetime
from decimal import Decimal

import pytest

from prediction_desk.db.models import EventContract
from prediction_desk.domain.enums import Comparator, MappingStatus, Measure, Venue
from prediction_desk.markets.mapping import MEASURE_SPECS, ContractTerms
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


@pytest.mark.parametrize("measure", list(MEASURE_SPECS), ids=[m.value for m in MEASURE_SPECS])
def test_terms_from_row_accepts_each_measures_own_spec(measure: Measure) -> None:
    unit, precision, source = MEASURE_SPECS[measure]
    row = _row(
        measure=measure,
        comparator=Comparator.GT,
        strike_hi=None,
        unit=unit,
        resolution_precision=precision,
        resolution_source=source,
    )
    terms = terms_from_row(row)
    assert (terms.unit, terms.resolution_precision, terms.resolution_source) == (
        unit,
        precision,
        source,
    )


@pytest.mark.parametrize(
    "over",
    [
        {"resolution_precision": 0, "measure": Measure.NFP_CHANGE, "unit": "jobs"},  # stale v1 NFP
        {"resolution_precision": 2},
        {"unit": "percent"},
        {"unit": "bp"},
        {"resolution_source": "BEA"},
        {"resolution_source": "bls"},
        {"measure": Measure.FOMC_CHANGE_BP},  # CPI spec on an FOMC measure
    ],
    ids=["stale_nfp_precision", "precision", "unit", "unit_bp", "source", "source_case", "measure"],
)
def test_terms_from_row_rejects_spec_mismatch(over: dict[str, object]) -> None:
    with pytest.raises(ValueError, match=r"contract 7: .* does not match measure spec"):
        terms_from_row(_row(comparator=Comparator.GT, strike_hi=None, **over))


@pytest.mark.parametrize("bad", ["NaN", "sNaN", "Infinity", "-Infinity"])
@pytest.mark.parametrize("field", ["strike_lo", "strike_hi"])
def test_terms_from_row_rejects_non_finite_strikes(field: str, bad: str) -> None:
    with pytest.raises(ValueError, match=r"contract 7: strike .* is not finite"):
        terms_from_row(_row(**{field: Decimal(bad)}))


def test_terms_from_row_rejects_non_finite_strike_hi_on_a_non_between_row() -> None:
    with pytest.raises(ValueError, match="not finite"):
        terms_from_row(_row(comparator=Comparator.GT, strike_hi=Decimal("NaN")))


def test_terms_from_row_between_needs_ordered_strikes() -> None:
    with pytest.raises(ValueError, match="contract 7: BETWEEN needs strike_hi"):
        terms_from_row(_row(strike_hi=None))
    with pytest.raises(ValueError, match="contract 7: BETWEEN needs strike_lo <= strike_hi"):
        terms_from_row(_row(strike_lo=Decimal("0.5"), strike_hi=Decimal("0.4")))
    equal = terms_from_row(_row(strike_lo=Decimal("0.4"), strike_hi=Decimal("0.4")))
    assert equal.strike_lo == equal.strike_hi == Decimal("0.4")

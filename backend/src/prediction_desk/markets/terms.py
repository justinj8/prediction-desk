"""Helpers that turn a stored ``EventContract`` row into pricing inputs."""

import hashlib

from prediction_desk.db.models import EventContract
from prediction_desk.domain.enums import Comparator, MappingStatus
from prediction_desk.markets.mapping import MEASURE_SPECS, ContractTerms


def terms_from_row(row: EventContract) -> ContractTerms:
    """The row's contract terms.

    ValueError unless the row is fully mapped, its (unit, precision, source) is the one
    ``MEASURE_SPECS`` fixes for its measure (a stale ``rules:v1`` row is not silently priced on
    the wrong grid), its strikes are finite, and a BETWEEN row has ``strike_lo <= strike_hi``.
    """
    if (
        row.mapping_status != MappingStatus.MAPPED
        or row.measure is None
        or row.comparator is None
        or row.strike_lo is None
        or row.unit is None
        or row.resolution_precision is None
        or row.resolution_source is None
    ):
        raise ValueError(f"contract {row.id} is not mapped")
    spec = MEASURE_SPECS[row.measure]
    actual = (row.unit, row.resolution_precision, row.resolution_source)
    if actual != spec:
        raise ValueError(
            f"contract {row.id}: (unit, precision, source) {actual} does not match measure "
            f"spec {spec} for {row.measure.value}"
        )
    for name, strike in (("strike_lo", row.strike_lo), ("strike_hi", row.strike_hi)):
        if strike is not None and not strike.is_finite():
            raise ValueError(f"contract {row.id}: strike {name}={strike} is not finite")
    if row.comparator == Comparator.BETWEEN:
        if row.strike_hi is None:
            raise ValueError(f"contract {row.id}: BETWEEN needs strike_hi")
        if row.strike_lo > row.strike_hi:
            raise ValueError(f"contract {row.id}: BETWEEN needs strike_lo <= strike_hi")
    return ContractTerms(
        measure=row.measure,
        comparator=row.comparator,
        strike_lo=row.strike_lo,
        strike_hi=row.strike_hi,
        unit=row.unit,
        resolution_precision=row.resolution_precision,
        resolution_source=row.resolution_source,
    )


def rules_hash(row: EventContract) -> str:
    """sha256 hex of the venue rules text, for ``inputs_hash``."""
    return hashlib.sha256(row.rules_text.encode("utf-8")).hexdigest()

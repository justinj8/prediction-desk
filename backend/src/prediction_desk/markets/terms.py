"""Helpers that turn a stored ``EventContract`` row into pricing inputs."""

import hashlib

from prediction_desk.db.models import EventContract
from prediction_desk.domain.enums import MappingStatus
from prediction_desk.markets.mapping import ContractTerms


def terms_from_row(row: EventContract) -> ContractTerms:
    """The row's contract terms; ValueError unless it is fully mapped."""
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

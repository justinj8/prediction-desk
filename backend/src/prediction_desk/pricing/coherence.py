"""Typed coherence violations, so the risk layer can act on kind and scope, not on text.

Per U2, ``monotonic`` and ``bracket_sum`` are pricing bugs (they trip the kill switch); an
``incomplete_bracket`` only blocks its own Polymarket event.
"""

from dataclasses import dataclass
from typing import Literal

CoherenceKind = Literal["monotonic", "bracket_sum", "incomplete_bracket"]


@dataclass(frozen=True, order=True)
class CoherenceViolation:
    kind: CoherenceKind
    scope: str  # venue.value for "monotonic"; the Polymarket venue_event_id for the bracket kinds
    message: str


def is_bug(v: CoherenceViolation) -> bool:
    return v.kind in ("monotonic", "bracket_sum")

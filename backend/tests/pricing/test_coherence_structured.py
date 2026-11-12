from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from test_price_release import (
    PARTITION,
    D,
    _bracket,
    _hashed_world,
    add_contract,
    make_release,
    price,
)

from prediction_desk.domain.enums import Comparator, MappingStatus, Venue
from prediction_desk.domain.types import to_prob
from prediction_desk.pricing import release as release_mod
from prediction_desk.pricing.coherence import CoherenceViolation, is_bug
from prediction_desk.pricing.fair import fair_prob

# Recorded on the 3a code (before this refactor) for the _hashed_world scenario, release id 1.
HASH_BEFORE_REFACTOR = "c585a1aac7efbddbd784a71d424da844d98dda5648062a4f108ed1d778a969ff"


async def test_monotonic_violation_is_structured_and_bug(
    session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = await make_release(session)
    forced = {D("0.25"): D("0.2"), D("0.35"): D("0.6")}
    real = fair_prob

    def skewed(terms: Any, dist: Any) -> Any:
        if terms.comparator == Comparator.GT and terms.strike_lo in forced:
            return to_prob(forced[terms.strike_lo])
        return real(terms, dist)

    monkeypatch.setattr(release_mod, "fair_prob", skewed)
    await add_contract(session, release, "K1", strike="0.25")
    await add_contract(session, release, "K2", strike="0.35")
    result = await price(session, release)
    (v,) = result.coherence
    assert isinstance(v, CoherenceViolation)
    assert v.kind == "monotonic"
    assert v.scope == "kalshi"
    assert v.message.startswith("cpi_mom gt: strike 0.2500 prob 0.20000 vs strike 0.3500")
    assert is_bug(v)


async def test_complete_bracket_bad_sum_is_bug(session: AsyncSession) -> None:
    release = await make_release(session)
    await _bracket(
        session, release, "PS", [PARTITION[0], PARTITION[1], PARTITION[2], PARTITION[4]], "S"
    )
    result = await price(session, release)
    (v,) = result.coherence
    assert v.kind == "bracket_sum"
    assert v.scope == "PS"
    assert v.message.startswith("bracket probabilities sum to ")
    assert is_bug(v)


async def test_incomplete_bracket_is_not_bug_and_scoped_to_event(session: AsyncSession) -> None:
    release = await make_release(session)
    await add_contract(session, release, "E1", venue=Venue.POLYMARKET, event="ev-1", strike="0.1")
    await add_contract(session, release, "E2", venue=Venue.POLYMARKET, event="ev-1", strike="0.2")
    await add_contract(
        session,
        None,
        "E3",
        venue=Venue.POLYMARKET,
        event="ev-1",
        status=MappingStatus.UNMAPPED,
    )
    result = await price(session, release)
    (v,) = result.coherence
    assert v.kind == "incomplete_bracket"
    assert v.scope == "ev-1"
    assert "(2 of 3 mapped)" in v.message
    assert not is_bug(v)


async def test_inputs_hash_unchanged_by_coherence_refactor(session: AsyncSession) -> None:
    release, _, _ = await _hashed_world(session)
    result = await price(session, release)
    assert release.id == 1
    assert [v.kind for v in result.coherence] == ["incomplete_bracket"]  # coherence is not hashed
    assert result.inputs_hash == HASH_BEFORE_REFACTOR


def test_violations_sort_by_kind_scope_message() -> None:
    a = CoherenceViolation("bracket_sum", "b", "m")
    b = CoherenceViolation("bracket_sum", "a", "z")
    assert sorted([a, b]) == [b, a]

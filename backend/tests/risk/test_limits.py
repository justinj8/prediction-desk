from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy import update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from prediction_desk.audit.hashing import canonical_json
from prediction_desk.db.models import RiskLimitsRow
from prediction_desk.risk.limits import DEFAULT_LIMITS, RiskLimits, load_active_limits

D = Decimal


def _params(**overrides: Any) -> dict[str, Any]:
    params: dict[str, Any] = dict(DEFAULT_LIMITS)
    params.update(overrides)
    return params


async def test_seeded_v1_matches_spec_defaults(session: AsyncSession) -> None:
    limits = await load_active_limits(session)
    assert limits.version == 1
    assert limits.cap_contract == D("0.02")
    assert limits.min_edge == D("0.04")
    assert limits.kelly_multiplier == D("0.25")
    assert limits == RiskLimits.from_params(1, DEFAULT_LIMITS)


def test_from_params_accepts_defaults() -> None:
    limits = RiskLimits.from_params(7, DEFAULT_LIMITS)
    assert limits.version == 7
    assert limits.max_quote_age_s == 60
    assert limits.cutoff_minutes == 5
    assert limits.fair_clamp_lo == D("0.02")


def test_from_params_allows_zero_cutoff_and_unit_fraction() -> None:
    limits = RiskLimits.from_params(1, _params(cutoff_minutes=0, cap_portfolio="1"))
    assert limits.cutoff_minutes == 0
    assert limits.cap_portfolio == D(1)


def _missing() -> dict[str, Any]:
    params = _params()
    del params["min_edge"]
    return params


@pytest.mark.parametrize(
    "params",
    [
        pytest.param(_missing(), id="missing-key"),
        pytest.param(_params(surprise="1"), id="extra-key"),
        pytest.param(_params(min_edge="NaN"), id="nan"),
        pytest.param(_params(min_edge="Infinity"), id="infinity"),
        pytest.param(_params(min_edge="abc"), id="not-a-number"),
        pytest.param(_params(min_edge=0.04), id="float-not-string"),
        pytest.param(_params(min_edge=None), id="none"),
        pytest.param(_params(min_edge="0"), id="zero-fraction"),
        pytest.param(_params(min_edge="-0.01"), id="negative-fraction"),
        pytest.param(_params(cap_contract="1.01"), id="fraction-above-one"),
        pytest.param(_params(fair_clamp_lo="0.5"), id="clamp-lo-at-half"),
        pytest.param(_params(fair_clamp_lo="0.6"), id="clamp-lo-above-half"),
        pytest.param(_params(fair_clamp_hi="0.5"), id="clamp-hi-at-half"),
        pytest.param(_params(fair_clamp_hi="0.4", fair_clamp_lo="0.1"), id="clamp-hi-below-half"),
        pytest.param(_params(max_quote_age_s="60"), id="quote-age-string"),
        pytest.param(_params(max_quote_age_s=60.0), id="quote-age-float"),
        pytest.param(_params(max_quote_age_s=True), id="quote-age-bool"),
        pytest.param(_params(max_quote_age_s=0), id="quote-age-zero"),
        pytest.param(_params(cutoff_minutes="5"), id="cutoff-string"),
        pytest.param(_params(cutoff_minutes=False), id="cutoff-bool"),
        pytest.param(_params(cutoff_minutes=-1), id="cutoff-negative"),
    ],
)
def test_from_params_rejects_missing_extra_nonfinite_out_of_range(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        RiskLimits.from_params(1, params)


async def test_only_one_active_version(session: AsyncSession) -> None:
    session.add(
        RiskLimitsRow(
            version=2,
            params=dict(DEFAULT_LIMITS),
            created_by="test",
            created_at=datetime(2026, 11, 12, tzinfo=UTC),
            active=True,
        )
    )
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_inactive_versions_may_coexist(session: AsyncSession) -> None:
    session.add(
        RiskLimitsRow(
            version=2,
            params=dict(DEFAULT_LIMITS),
            created_by="test",
            created_at=datetime(2026, 11, 12, tzinfo=UTC),
            active=False,
        )
    )
    await session.commit()
    assert (await load_active_limits(session)).version == 1


async def test_load_active_limits_raises_without_active_row(session: AsyncSession) -> None:
    await session.execute(update(RiskLimitsRow).values(active=False))
    await session.commit()
    with pytest.raises(LookupError):
        await load_active_limits(session)


async def test_snapshot_is_json_safe(session: AsyncSession) -> None:
    limits = await load_active_limits(session)
    snap = limits.snapshot()
    assert canonical_json(snap)
    assert snap == dict(DEFAULT_LIMITS)
    assert RiskLimits.from_params(limits.version, snap) == limits

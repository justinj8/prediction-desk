from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from prediction_desk.db.models import DataObservation
from prediction_desk.domain.enums import DataSource
from prediction_desk.forecasting.realtime import (
    InsufficientData,
    build_samples,
    first_print,
    first_vintage_ts,
    levels_asof,
    month_mean,
    pct_change,
)
from prediction_desk.ingestion.provider import MacroDataProvider

D = Decimal
SID = "TESTSERIES"


def ts(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, 12, 30, tzinfo=UTC)


def obs(obs_date: date, vintage: datetime, value: str, sid: str = SID) -> DataObservation:
    return DataObservation(
        source=DataSource.FRED,
        series_id=sid,
        obs_date=obs_date,
        vintage_ts=vintage,
        published_on=vintage.date(),
        value=D(value),
        ingested_at=vintage,
    )


def test_insufficient_data_attrs() -> None:
    e = InsufficientData("cpi_mom", 10, 36)
    assert (e.measure, e.have, e.need) == ("cpi_mom", 10, 36)
    assert str(e) == "cpi_mom: insufficient data (10 < 36)"


async def test_first_print_uses_earliest_vintage(session: AsyncSession) -> None:
    m = date(2024, 1, 1)
    session.add_all(
        [
            obs(m, ts(2024, 2, 13), "100.0"),
            obs(m, ts(2024, 3, 12), "101.0"),
            obs(m, ts(2025, 1, 10), "102.0"),
        ]
    )
    await session.commit()
    assert await first_vintage_ts(session, DataSource.FRED, SID, m) == ts(2024, 2, 13)
    assert await first_print(session, DataSource.FRED, SID, m) == D("100.0")
    assert await first_vintage_ts(session, DataSource.FRED, SID, date(2030, 1, 1)) is None
    assert await first_print(session, DataSource.FRED, SID, date(2030, 1, 1)) is None


async def test_levels_asof_wraps_provider(session: AsyncSession) -> None:
    session.add_all(
        [
            obs(date(2024, 1, 1), ts(2024, 2, 13), "100"),
            obs(date(2024, 1, 1), ts(2024, 6, 1), "105"),
            obs(date(2023, 1, 1), ts(2023, 2, 13), "90"),
        ]
    )
    await session.commit()
    got = await levels_asof(MacroDataProvider(), session, SID, ts(2024, 3, 1), date(2024, 1, 1))
    assert got == {date(2024, 1, 1): D("100")}


def test_pct_change_exact() -> None:
    lv = {date(2024, 1, 1): D("100"), date(2024, 2, 1): D("100.3")}
    assert pct_change(lv, date(2024, 2, 1)) == D("0.300000")
    assert pct_change(lv, date(2024, 1, 1)) is None
    assert pct_change(lv, date(2024, 3, 1)) is None
    # year boundary
    lv2 = {date(2023, 12, 1): D("3"), date(2024, 1, 1): D("4")}
    assert pct_change(lv2, date(2024, 1, 1)) == D("33.333333")
    # zero base
    assert (
        pct_change({date(2024, 1, 1): D("0"), date(2024, 2, 1): D("1")}, date(2024, 2, 1)) is None
    )


def test_month_mean_weekly() -> None:
    pts = {
        date(2024, 1, 5): D("1"),
        date(2024, 1, 12): D("2"),
        date(2024, 1, 19): D("2"),
        date(2024, 2, 2): D("100"),
        date(2023, 1, 5): D("100"),
    }
    assert month_mean(pts, date(2024, 1, 1)) == D("1.666667")
    assert month_mean(pts, date(2024, 3, 1)) is None


async def _feat(
    session: AsyncSession, provider: MacroDataProvider, m: date, t_m: datetime
) -> tuple[Decimal, ...] | None:
    """(previous month level, own level). Own level must be invisible at t_m: 0 if it is."""
    lv = await levels_asof(provider, session, SID, t_m, date(2020, 1, 1))
    prev = date(m.year - 1, 12, 1) if m.month == 1 else date(m.year, m.month - 1, 1)
    if prev not in lv:
        return None
    return (lv[prev], lv.get(m, D(0)))


async def _target(session: AsyncSession, provider: MacroDataProvider, m: date) -> Decimal | None:
    return await first_print(session, DataSource.FRED, SID, m)


async def test_build_samples_never_uses_future_vintages(session: AsyncSession) -> None:
    m1, m2, m3 = date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)
    fv2, fv3 = ts(2024, 3, 12), ts(2024, 4, 10)
    session.add_all(
        [
            obs(m1, ts(2024, 2, 13), "100"),  # m1 first print
            obs(m2, fv2, "101"),  # m2 first print
            obs(m1, fv2, "999"),  # m1 revised in the SAME release as m2's first print
            obs(m2, ts(2024, 4, 5), "150"),  # m2 revised strictly before m3's first print
            obs(m3, fv3, "102"),  # m3 first print
            obs(m2, ts(2024, 4, 20), "888"),  # m2 revised after m3's first print
        ]
    )
    await session.commit()
    outer = ts(2024, 5, 1)
    samples = await build_samples(
        session,
        MacroDataProvider(),
        target_series=SID,
        target_fn=_target,
        feature_fn=_feat,
        asof=outer,
        lookback_months=6,
    )
    by_month = {s.month: s for s in samples}
    assert set(by_month) == {m2, m3}
    # m2: pre-revision m1 level (not 999, same vintage_ts as m2's own first print), and
    # m2's own observation is invisible at t_m.
    assert by_month[m2].features == (D("100"), D(0))
    assert by_month[m2].target == D("101")
    assert by_month[m2].asof == fv2 - timedelta(microseconds=1)
    # m3: sees the revision of m2 published strictly before its first print (150),
    # but not the later one (888), and not its own first print.
    assert by_month[m3].features == (D("150"), D(0))
    assert by_month[m3].asof == fv3 - timedelta(microseconds=1)
    for s in samples:
        assert s.asof <= outer
        fv = await first_vintage_ts(session, DataSource.FRED, SID, s.month)
        assert fv is not None and s.asof < fv


async def test_build_samples_requires_aware_asof(session: AsyncSession) -> None:
    with pytest.raises(ValueError, match="timezone"):
        await build_samples(
            session,
            MacroDataProvider(),
            target_series=SID,
            target_fn=_target,
            feature_fn=_feat,
            asof=datetime(2024, 5, 1),
        )


async def test_build_samples_excludes_unpublished_months(session: AsyncSession) -> None:
    session.add_all(
        [
            obs(date(2024, 1, 1), ts(2024, 2, 13), "100"),
            obs(date(2024, 2, 1), ts(2024, 3, 12), "101"),
            obs(date(2024, 3, 1), ts(2024, 4, 10), "102"),
        ]
    )
    await session.commit()
    samples = await build_samples(
        session,
        MacroDataProvider(),
        target_series=SID,
        target_fn=_target,
        feature_fn=_feat,
        asof=ts(2024, 3, 20),  # March's first print is in the future
        lookback_months=6,
    )
    assert [s.month for s in samples] == [date(2024, 2, 1)]


async def test_build_samples_skips_none_and_sorted(session: AsyncSession) -> None:
    session.add_all(
        [
            obs(date(2024, 1, 1), ts(2024, 2, 13), "100"),
            obs(date(2024, 2, 1), ts(2024, 3, 12), "101"),
            obs(date(2024, 3, 1), ts(2024, 4, 10), "102"),
        ]
    )
    await session.commit()

    async def no_target(s: AsyncSession, p: MacroDataProvider, m: date) -> Decimal | None:
        return None if m == date(2024, 2, 1) else D("1")

    samples = await build_samples(
        session,
        MacroDataProvider(),
        target_series=SID,
        target_fn=no_target,
        feature_fn=_feat,
        asof=ts(2024, 5, 1),
        lookback_months=6,
    )
    # Jan has no previous level (feature None), Feb has no target
    assert [s.month for s in samples] == [date(2024, 3, 1)]


@pytest.mark.parametrize("bad", [0, -1])
async def test_build_samples_lookback_small(session: AsyncSession, bad: int) -> None:
    samples = await build_samples(
        session,
        MacroDataProvider(),
        target_series=SID,
        target_fn=_target,
        feature_fn=_feat,
        asof=ts(2024, 5, 1),
        lookback_months=bad,
    )
    assert samples == []

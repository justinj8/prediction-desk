import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from prediction_desk.clock import SimClock
from prediction_desk.db.models import DataObservation
from prediction_desk.domain.enums import DataSource
from prediction_desk.ingestion.fred import FredClient, fred_vintage_ts
from prediction_desk.ingestion.provider import MacroDataProvider, Point
from prediction_desk.ingestion.store import ingest_fred_series, upsert_observations
from prediction_desk.net.http import ResilientHttp

T0 = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
V1 = datetime(2024, 2, 14, 5, 0, tzinfo=UTC)
V2 = datetime(2024, 3, 13, 4, 0, tzinfo=UTC)
FIXTURE = Path(__file__).parent / "fixtures" / "fred_observations_cpiaucsl.json"


def _row(obs: date, vintage: datetime, value: str, series: str = "CPIAUCSL") -> DataObservation:
    return DataObservation(
        source=DataSource.FRED,
        series_id=series,
        obs_date=obs,
        vintage_ts=vintage,
        published_on=vintage.date() - timedelta(days=1),
        value=Decimal(value),
        ingested_at=T0,
    )


async def _asof(session: AsyncSession, asof: datetime, start: date | None = None) -> list[Point]:
    return await MacroDataProvider().series_asof(session, DataSource.FRED, "CPIAUCSL", asof, start)


async def test_asof_returns_latest_known_vintage(session: AsyncSession) -> None:
    jan = date(2024, 1, 1)
    session.add_all([_row(jan, V1, "308.417"), _row(jan, V2, "308.500")])
    await session.commit()

    assert await _asof(session, V1) == [Point(jan, Decimal("308.417"), V1)]
    assert await _asof(session, V2) == [Point(jan, Decimal("308.500"), V2)]
    assert await _asof(session, V2 - timedelta(seconds=1)) == [Point(jan, Decimal("308.417"), V1)]
    assert await _asof(session, V1 - timedelta(seconds=1)) == []


async def test_asof_never_returns_future_obs(session: AsyncSession) -> None:
    jan, feb = date(2024, 1, 1), date(2024, 2, 1)
    session.add_all([_row(jan, V1, "308.417"), _row(feb, V2, "310.326")])
    await session.commit()

    points = await _asof(session, V1)
    assert [p.obs_date for p in points] == [jan]
    points = await _asof(session, V2)
    assert [p.obs_date for p in points] == [jan, feb]


async def test_asof_start_filter_sorted_and_series_isolated(session: AsyncSession) -> None:
    session.add_all(
        [
            _row(date(2024, 3, 1), V2, "3"),
            _row(date(2024, 1, 1), V1, "1"),
            _row(date(2024, 2, 1), V1, "2"),
            _row(date(2024, 2, 1), V1, "99", series="OTHER"),
        ]
    )
    await session.commit()
    points = await _asof(session, V2, start=date(2024, 2, 1))
    assert [(p.obs_date, p.value) for p in points] == [
        (date(2024, 2, 1), Decimal("2")),
        (date(2024, 3, 1), Decimal("3")),
    ]


async def test_asof_requires_aware_datetime(session: AsyncSession) -> None:
    with pytest.raises(ValueError, match="timezone"):
        await _asof(session, datetime(2024, 3, 13, 4, 0))


async def test_ingest_is_idempotent(session: AsyncSession) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=FIXTURE.read_bytes())

    clock = SimClock(T0)
    http = ResilientHttp(
        "fred", "https://api.stlouisfed.org/fred", clock, transport=httpx.MockTransport(handler)
    )
    client = FredClient(http, "test-key-not-real")

    assert await ingest_fred_series(session, client, clock, "CPIAUCSL", date(2024, 1, 1)) == 4
    assert await ingest_fred_series(session, client, clock, "CPIAUCSL", date(2024, 1, 1)) == 0

    total = await session.scalar(select(func.count()).select_from(DataObservation))
    assert total == 4
    # Revision to an old month is replay-safe through the real ingest path.
    jan = date(2024, 1, 1)
    before = await _asof(session, fred_vintage_ts(date(2024, 3, 11)))
    after = await _asof(session, fred_vintage_ts(date(2024, 3, 12)))
    assert [p.value for p in before if p.obs_date == jan] == [Decimal("308.417")]
    assert [p.value for p in after if p.obs_date == jan] == [Decimal("308.463")]
    row = (await session.scalars(select(DataObservation).limit(1))).one()
    assert row.ingested_at == T0
    assert json.loads(FIXTURE.read_text())["count"] == 4


async def test_upsert_observations_counts_only_new_rows(session: AsyncSession) -> None:
    jan = date(2024, 1, 1)
    assert await upsert_observations(session, [_row(jan, V1, "1")]) == 1
    assert await upsert_observations(session, [_row(jan, V1, "1"), _row(jan, V2, "2")]) == 1
    assert await upsert_observations(session, []) == 0

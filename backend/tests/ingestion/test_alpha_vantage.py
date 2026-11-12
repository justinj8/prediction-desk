from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from prediction_desk.clock import SimClock
from prediction_desk.db.models import DataObservation
from prediction_desk.domain.enums import DataSource
from prediction_desk.ingestion.alpha_vantage import AV_SERIES, AlphaVantageClient
from prediction_desk.ingestion.store import ingest_alpha_vantage
from prediction_desk.net.http import ProviderError, ProviderRateLimited, ResilientHttp

T0 = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
BASE = "https://www.alphavantage.co"
KEY = "test-key-not-real"
FIXTURES = Path(__file__).parent / "fixtures"
CPI = "CPI:monthly"


def _fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _client(handler: Any, clock: SimClock | None = None) -> AlphaVantageClient:
    http = ResilientHttp(
        "alpha_vantage", BASE, clock or SimClock(T0), transport=httpx.MockTransport(handler)
    )
    return AlphaVantageClient(http, KEY)


def _ok(req: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=_fixture("alpha_vantage_cpi.json"))


def test_series_map() -> None:
    assert AV_SERIES == {
        "CPI:monthly": {"function": "CPI", "interval": "monthly"},
        "NONFARM_PAYROLL": {"function": "NONFARM_PAYROLL"},
        "FEDERAL_FUNDS_RATE:monthly": {"function": "FEDERAL_FUNDS_RATE", "interval": "monthly"},
        "TREASURY_YIELD:monthly:2year": {
            "function": "TREASURY_YIELD",
            "interval": "monthly",
            "maturity": "2year",
        },
    }


async def test_parses_cpi_series() -> None:
    requests: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        requests.append(req)
        return _ok(req)

    out = await _client(handler).series(CPI)

    assert out == [
        (date(2026, 9, 1), Decimal("324.368")),
        (date(2026, 8, 1), Decimal("323.976")),
        (date(2026, 6, 1), Decimal("322.561")),
    ]
    assert all(isinstance(v, Decimal) for _, v in out)
    assert len(requests) == 1
    assert requests[0].url.path == "/query"
    assert dict(requests[0].url.params) == {"function": "CPI", "interval": "monthly", "apikey": KEY}


async def test_treasury_params_exact() -> None:
    seen: list[dict[str, str]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(dict(req.url.params))
        return _ok(req)

    await _client(handler).series("TREASURY_YIELD:monthly:2year")
    assert seen == [
        {
            "function": "TREASURY_YIELD",
            "interval": "monthly",
            "maturity": "2year",
            "apikey": KEY,
        }
    ]


async def test_unknown_series_raises_value_error() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError("must not call the network")

    with pytest.raises(ValueError, match="unknown"):
        await _client(handler).series("GDP")


@pytest.mark.parametrize("key", ["Information", "Note"])
async def test_rate_limit_body_raises_without_retry(key: str) -> None:
    calls = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200, json={key: f"standard API rate limit is 25 requests per day. apikey={KEY}"}
        )

    with pytest.raises(ProviderRateLimited) as exc:
        await _client(handler).series(CPI)
    assert calls == 1
    assert exc.value.provider == "alpha_vantage"
    assert KEY not in str(exc.value)


async def test_rate_limit_checked_before_data() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"Information": "limit", "data": [{"date": "2026-09-01", "value": "1"}]}
        )

    with pytest.raises(ProviderRateLimited):
        await _client(handler).series(CPI)


async def test_error_message_raises_provider_error() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_fixture("alpha_vantage_error.json"))

    with pytest.raises(ProviderError) as exc:
        await _client(handler).series(CPI)
    assert not isinstance(exc.value, ProviderRateLimited)
    assert exc.value.provider == "alpha_vantage"


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"data": "nope"},
        {"data": [{"date": "2026-09-01"}]},
        {"data": [{"date": "bad", "value": "1"}]},
        {"data": [{"date": "2026-09-01", "value": "abc"}]},
        {"data": [{"date": "2026-09-01", "value": "NaN"}]},
        {"data": [{"date": "2026-09-01", "value": "Infinity"}]},
        {"data": [{"date": "2026-09-01", "value": "-inf"}]},
    ],
)
async def test_malformed_body_raises_provider_error(body: dict[str, Any]) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    with pytest.raises(ProviderError):
        await _client(handler).series(CPI)


async def _count(session: AsyncSession) -> int:
    return int(
        await session.scalar(
            select(func.count())
            .select_from(DataObservation)
            .where(DataObservation.source == DataSource.ALPHA_VANTAGE)
        )
        or 0
    )


async def test_rate_limit_through_ingest_stores_nothing(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    calls = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=_fixture("alpha_vantage_rate_limit.json"))

    clock = SimClock(T0)
    with pytest.raises(ProviderRateLimited):
        await ingest_alpha_vantage(sessions, _client(handler, clock), clock, CPI)
    assert calls == 1
    assert await _count(session) == 0


async def test_freshness_gate_skips_recent_fetch(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    calls = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _ok(req)

    clock = SimClock(T0)
    client = _client(handler, clock)

    assert await ingest_alpha_vantage(sessions, client, clock, CPI) == 3
    assert calls == 1
    row = (
        (
            await session.execute(
                select(DataObservation).where(DataObservation.source == DataSource.ALPHA_VANTAGE)
            )
        )
        .scalars()
        .first()
    )
    assert row is not None
    assert row.vintage_ts == T0 and row.ingested_at == T0 and row.published_on is None

    clock.advance(timedelta(hours=1))
    assert await ingest_alpha_vantage(sessions, client, clock, CPI) == 0
    assert calls == 1

    clock.advance(timedelta(hours=24))  # 25h after the first fetch
    assert await ingest_alpha_vantage(sessions, client, clock, CPI) == 3
    assert calls == 2
    assert await _count(session) == 6


async def test_freshness_gate_is_per_series(
    session: AsyncSession, sessions: async_sessionmaker[AsyncSession]
) -> None:
    calls = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _ok(req)

    clock = SimClock(T0)
    client = _client(handler, clock)
    await ingest_alpha_vantage(sessions, client, clock, CPI)
    await ingest_alpha_vantage(sessions, client, clock, "NONFARM_PAYROLL")
    assert calls == 2

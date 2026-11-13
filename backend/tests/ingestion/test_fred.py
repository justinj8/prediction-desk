import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from prediction_desk.clock import SimClock
from prediction_desk.domain.enums import Series
from prediction_desk.ingestion.fred import (
    FRED_RELEASE_IDS,
    FRED_SERIES,
    MAX_PAGES,
    FredClient,
    fred_vintage_ts,
)
from prediction_desk.net.http import ProviderError, ProviderHttpError, ResilientHttp

T0 = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
BASE = "https://api.stlouisfed.org/fred"
KEY = "test-key-not-real"
FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _client(handler: Any) -> FredClient:
    http = ResilientHttp("fred", BASE, SimClock(T0), transport=httpx.MockTransport(handler))
    return FredClient(http, KEY)


def _obs(rs: str, d: str, v: str) -> dict[str, str]:
    return {"realtime_start": rs, "realtime_end": "9999-12-31", "date": d, "value": v}


def test_constants() -> None:
    assert len(FRED_SERIES) == 18
    assert "CPIAUCSL" in FRED_SERIES and "PAYEMS" in FRED_SERIES
    assert "DFEDTARU" in FRED_SERIES
    assert FRED_RELEASE_IDS == {Series.CPI: 10, Series.NFP: 50}


def test_fred_vintage_ts() -> None:
    assert fred_vintage_ts(date(2024, 3, 12)) == datetime(2024, 3, 13, 4, 0, tzinfo=UTC)
    assert fred_vintage_ts(date(2024, 1, 11)) == datetime(2024, 1, 12, 5, 0, tzinfo=UTC)


async def test_release_names() -> None:
    requests: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        requests.append(req)
        assert req.url.path == "/fred/release"
        rid = req.url.params["release_id"]
        return httpx.Response(200, content=_fixture(f"fred_release_{rid}.json"))

    client = _client(handler)
    assert await client.release_name(10) == "Consumer Price Index"
    assert await client.release_name(50) == "Employment Situation"
    assert [dict(r.url.params) for r in requests] == [
        {"release_id": "10", "api_key": KEY, "file_type": "json"},
        {"release_id": "50", "api_key": KEY, "file_type": "json"},
    ]


async def test_cpi_vintages_include_revisions() -> None:
    requests: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        requests.append(req)
        return httpx.Response(200, content=_fixture("fred_observations_cpiaucsl.json"))

    client = _client(handler)
    obs = await client.observations_all_vintages("CPIAUCSL", date(2024, 1, 1))

    assert len(obs) == 4
    published: dict[date, set[date]] = {}
    for o in obs:
        assert isinstance(o.value, Decimal)
        published.setdefault(o.obs_date, set()).add(o.published_on)
    assert len(published[date(2024, 1, 1)]) == 2
    jan = sorted((o for o in obs if o.obs_date == date(2024, 1, 1)), key=lambda o: o.published_on)
    assert [(o.published_on, o.value) for o in jan] == [
        (date(2024, 2, 13), Decimal("308.417")),
        (date(2024, 3, 12), Decimal("308.463")),
    ]

    assert len(requests) == 1
    assert requests[0].url.path == "/fred/series/observations"
    assert dict(requests[0].url.params) == {
        "series_id": "CPIAUCSL",
        "api_key": KEY,
        "file_type": "json",
        "realtime_start": "1776-07-04",
        "realtime_end": "9999-12-31",
        "observation_start": "2024-01-01",
        "sort_order": "asc",
        "limit": "100000",
        "offset": "0",
    }


async def test_missing_values_skipped() -> None:
    body = {
        "count": 2,
        "offset": 0,
        "observations": [
            _obs("2024-02-13", "2024-01-01", "."),
            _obs("2024-03-12", "2024-02-01", "1.5"),
        ],
    }
    client = _client(lambda req: httpx.Response(200, content=json.dumps(body)))
    obs = await client.observations_all_vintages("X", date(2024, 1, 1))
    assert [(o.obs_date, o.value) for o in obs] == [(date(2024, 2, 1), Decimal("1.5"))]


async def test_pagination() -> None:
    offsets: list[str] = []
    pages = {
        "0": [_obs("2024-02-13", "2024-01-01", "1"), _obs("2024-03-12", "2024-02-01", "2")],
        "2": [_obs("2024-04-10", "2024-03-01", "3")],
    }

    def handler(req: httpx.Request) -> httpx.Response:
        off = req.url.params["offset"]
        offsets.append(off)
        return httpx.Response(200, content=json.dumps({"count": 3, "observations": pages[off]}))

    obs = await _client(handler).observations_all_vintages("X", date(2024, 1, 1))
    assert len(obs) == 3
    assert offsets == ["0", "2"]


async def test_pagination_guard_on_endless_pages() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        body = {"count": 10**6, "observations": [_obs("2024-02-13", "2024-01-01", "1")]}
        return httpx.Response(200, content=json.dumps(body))

    with pytest.raises(ProviderError) as exc:
        await _client(handler).observations_all_vintages("X", date(2024, 1, 1))
    assert str(MAX_PAGES) in str(exc.value)
    assert KEY not in str(exc.value)


async def test_empty_page_with_count_remaining_stops() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps({"count": 5, "observations": []}))

    with pytest.raises(ProviderError):
        await _client(handler).observations_all_vintages("X", date(2024, 1, 1))


async def test_release_dates() -> None:
    requests: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        requests.append(req)
        return httpx.Response(200, content=_fixture("fred_release_dates_10.json"))

    dates = await _client(handler).release_dates(10, date(2024, 1, 1))
    assert dates == [date(2024, 2, 13), date(2024, 3, 12), date(2024, 4, 10)]
    assert requests[0].url.path == "/fred/release/dates"
    assert dict(requests[0].url.params) == {
        "release_id": "10",
        "api_key": KEY,
        "file_type": "json",
        "realtime_start": "2024-01-01",
        "realtime_end": "9999-12-31",
        "include_release_dates_with_no_data": "true",
        "sort_order": "asc",
        "limit": "10000",
    }


async def test_bad_series_http_400_propagates_without_key() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        body = {"error_code": 400, "error_message": "Bad Request.  The series does not exist."}
        return httpx.Response(400, content=json.dumps(body))

    with pytest.raises(ProviderHttpError) as exc:
        await _client(handler).observations_all_vintages("NOPE", date(2024, 1, 1))
    assert exc.value.status == 400
    assert KEY not in str(exc.value)


@pytest.mark.parametrize("value", ["NaN", "nan", "Infinity", "-Infinity", "sNaN"])
async def test_non_finite_value_raises(value: str) -> None:
    body = {"count": 1, "observations": [_obs("2024-02-13", "2024-01-01", value)]}
    client = _client(lambda req: httpx.Response(200, content=json.dumps(body)))
    with pytest.raises(ProviderError, match="malformed observation"):
        await client.observations_all_vintages("X", date(2024, 1, 1))


@pytest.mark.parametrize("count", ["abc", None, [], {}, True, -1, "1.5"])
async def test_malformed_count_raises(count: object) -> None:
    body = {"count": count, "observations": [_obs("2024-02-13", "2024-01-01", "1")]}
    client = _client(lambda req: httpx.Response(200, content=json.dumps(body)))
    with pytest.raises(ProviderError, match="malformed observations response"):
        await client.observations_all_vintages("X", date(2024, 1, 1))


async def test_numeric_string_count_accepted() -> None:
    body = {"count": "1", "observations": [_obs("2024-02-13", "2024-01-01", "1")]}
    client = _client(lambda req: httpx.Response(200, content=json.dumps(body)))
    assert len(await client.observations_all_vintages("X", date(2024, 1, 1))) == 1

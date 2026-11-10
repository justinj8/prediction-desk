import random
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import httpx
import pytest

from prediction_desk.clock import SimClock
from prediction_desk.net.http import (
    ProviderError,
    ProviderHttpError,
    ProviderUnavailable,
    ResilientHttp,
)

T0 = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)

Handler = Callable[[httpx.Request], httpx.Response]


class Harness:
    def __init__(self, responses: list[httpx.Response | Exception] | Handler) -> None:
        self.calls = 0
        self.delays: list[float] = []
        self.clock = SimClock(T0)
        self._responses = responses

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if callable(self._responses):
            return self._responses(request)
        item = self._responses[min(self.calls, len(self._responses)) - 1]
        if isinstance(item, Exception):
            raise item
        return item

    async def sleep(self, delay: float) -> None:
        self.delays.append(delay)

    def client(self, **kwargs: object) -> ResilientHttp:
        return ResilientHttp(
            "test",
            "https://example.test/api",
            self.clock,
            transport=httpx.MockTransport(self.handler),
            sleep=self.sleep,
            rng=random.Random(0),
            **kwargs,  # type: ignore[arg-type]
        )


async def test_parses_floats_as_decimal() -> None:
    h = Harness([httpx.Response(200, text='{"bestBid": 0.034, "n": 3}')])
    async with h.client() as http:
        data = await http.get_json("/x")
    assert data == {"bestBid": Decimal("0.034"), "n": 3}
    assert type(data["bestBid"]) is Decimal


async def test_retries_5xx_then_succeeds() -> None:
    h = Harness([httpx.Response(503), httpx.Response(503), httpx.Response(200, json={"ok": 1})])
    async with h.client(backoff_base_s=0.5) as http:
        data = await http.get_json("/x")
    assert data == {"ok": 1}
    assert h.calls == 3 and len(h.delays) == 2
    assert 0.25 <= h.delays[0] <= 0.75 and 0.5 <= h.delays[1] <= 1.5


async def test_retries_transport_errors() -> None:
    h = Harness(
        [httpx.ConnectError("boom"), httpx.ReadTimeout("slow"), httpx.Response(200, json=[])]
    )
    async with h.client() as http:
        assert await http.get_json("/x") == []
    assert h.calls == 3 and len(h.delays) == 2


async def test_retries_429_and_honors_retry_after() -> None:
    h = Harness([httpx.Response(429, headers={"Retry-After": "2"}), httpx.Response(200, json={})])
    async with h.client() as http:
        await http.get_json("/x")
    assert h.delays == [2.0]


async def test_retry_after_is_capped_and_http_date_ignored() -> None:
    h = Harness(
        [
            httpx.Response(429, headers={"Retry-After": "999"}),
            httpx.Response(429, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}),
            httpx.Response(200, json={}),
        ]
    )
    async with h.client(max_attempts=3) as http:
        await http.get_json("/x")
    assert h.delays[0] == 30.0
    assert 0.5 <= h.delays[1] <= 1.5  # fell back to computed backoff for attempt 2


async def test_does_not_retry_4xx() -> None:
    h = Harness([httpx.Response(404)])
    async with h.client() as http:
        with pytest.raises(ProviderHttpError) as e:
            await http.get_json("/x")
    assert e.value.status == 404 and e.value.provider == "test"
    assert h.calls == 1 and h.delays == []


async def test_retries_exhausted_raises_unavailable() -> None:
    h = Harness([httpx.Response(500)])
    async with h.client(max_attempts=3) as http:
        with pytest.raises(ProviderUnavailable) as e:
            await http.get_json("/x")
    assert h.calls == 3 and len(h.delays) == 2
    assert isinstance(e.value, ProviderError) and e.value.provider == "test"


async def test_error_messages_never_include_query_params() -> None:
    h = Harness([httpx.Response(500)])
    async with h.client(max_attempts=1) as http:
        with pytest.raises(ProviderUnavailable) as e:
            await http.get_json("/x", {"api_key": "SECRETVALUE"})
    assert "SECRETVALUE" not in str(e.value)
    h2 = Harness([httpx.Response(403)])
    async with h2.client() as http:
        with pytest.raises(ProviderHttpError) as e2:
            await http.get_json("/x", {"api_key": "SECRETVALUE"})
    assert "SECRETVALUE" not in str(e2.value)


async def test_breaker_opens_then_half_opens_after_cooldown() -> None:
    h = Harness([httpx.Response(500)])
    async with h.client(
        max_attempts=1, breaker_threshold=2, breaker_cooldown=timedelta(seconds=60)
    ) as http:
        for _ in range(2):
            with pytest.raises(ProviderUnavailable):
                await http.get_json("/x")
        assert h.calls == 2
        with pytest.raises(ProviderUnavailable):
            await http.get_json("/x")
        assert h.calls == 2  # open: no request sent
        h.clock.advance(timedelta(seconds=60))
        with pytest.raises(ProviderUnavailable):
            await http.get_json("/x")
        assert h.calls == 3  # trial call reached the transport


async def test_half_open_failure_reopens_immediately() -> None:
    h = Harness([httpx.Response(500)])
    async with h.client(
        max_attempts=1, breaker_threshold=2, breaker_cooldown=timedelta(seconds=60)
    ) as http:
        for _ in range(2):
            with pytest.raises(ProviderUnavailable):
                await http.get_json("/x")
        h.clock.advance(timedelta(seconds=60))
        with pytest.raises(ProviderUnavailable):
            await http.get_json("/x")  # trial fails
        assert h.calls == 3
        with pytest.raises(ProviderUnavailable):
            await http.get_json("/x")  # re-opened: blocked
        assert h.calls == 3


async def test_half_open_success_resets_breaker() -> None:
    h = Harness([httpx.Response(500), httpx.Response(500), httpx.Response(200, json={})])
    async with h.client(
        max_attempts=1, breaker_threshold=2, breaker_cooldown=timedelta(seconds=60)
    ) as http:
        for _ in range(2):
            with pytest.raises(ProviderUnavailable):
                await http.get_json("/x")
        h.clock.advance(timedelta(seconds=60))
        await http.get_json("/x")
        await http.get_json("/x")
        assert h.calls == 4


async def test_4xx_does_not_count_toward_breaker() -> None:
    h = Harness([httpx.Response(404)])
    async with h.client(breaker_threshold=2) as http:
        for _ in range(4):
            with pytest.raises(ProviderHttpError):
                await http.get_json("/x")
    assert h.calls == 4


async def test_success_resets_consecutive_failure_count() -> None:
    h = Harness(
        [
            httpx.Response(500),
            httpx.Response(200, json={}),
            httpx.Response(500),
            httpx.Response(200, json={}),
        ]
    )
    async with h.client(max_attempts=1, breaker_threshold=2) as http:
        with pytest.raises(ProviderUnavailable):
            await http.get_json("/x")
        await http.get_json("/x")
        with pytest.raises(ProviderUnavailable):
            await http.get_json("/x")
        assert h.calls == 3
        await http.get_json("/x")  # breaker still closed
        assert h.calls == 4

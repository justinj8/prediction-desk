import asyncio
import logging
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
    _RedactSecrets,
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


async def test_redirect_is_http_error() -> None:
    h = Harness([httpx.Response(302, headers={"Location": "/elsewhere"})])
    async with h.client() as http:
        with pytest.raises(ProviderHttpError) as e:
            await http.get_json("/x")
    assert e.value.status == 302 and h.calls == 1


async def test_non_json_body_is_provider_error() -> None:
    h = Harness([httpx.Response(200, text="not json")])
    async with h.client() as http:
        with pytest.raises(ProviderError) as e:
            await http.get_json("/x")
    assert type(e.value) is ProviderError
    assert e.value.__context__ is None


@pytest.mark.parametrize("body", ["NaN", '{"a": Infinity}', "[-Infinity]"])
async def test_non_finite_numbers_rejected(body: str) -> None:
    h = Harness([httpx.Response(200, text=body)])
    async with h.client() as http:
        with pytest.raises(ProviderError, match="non-finite"):
            await http.get_json("/x")


async def test_non_transport_httpx_error_wrapped_without_chaining() -> None:
    h = Harness([httpx.TooManyRedirects("https://x/?api_key=SECRETVALUE")])
    async with h.client() as http:
        with pytest.raises(ProviderError) as e:
            await http.get_json("/x")
    assert h.calls == 1
    assert "SECRETVALUE" not in str(e.value)
    assert e.value.__context__ is None and e.value.__cause__ is None


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


async def _stale_scenario(stale_ok: bool) -> None:
    clock = SimClock(T0)
    gates = {"/stale": asyncio.Event(), "/trial": asyncio.Event()}
    hits: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/api")
        hits.append(path)
        if path in gates:
            await gates[path].wait()
            return httpx.Response(200 if path == "/trial" or stale_ok else 500, json={})
        return httpx.Response(500 if path == "/boom" else 200, json={})

    async def no_sleep(_: float) -> None:
        return None

    http = ResilientHttp(
        "test",
        "https://example.test/api",
        clock,
        transport=httpx.MockTransport(handler),
        sleep=no_sleep,
        max_attempts=1,
        breaker_threshold=1,
        breaker_cooldown=timedelta(seconds=60),
    )
    async with http:
        stale = asyncio.create_task(http.get_json("/stale"))  # admitted while closed
        await _settle()
        with pytest.raises(ProviderUnavailable):
            await http.get_json("/boom")  # opens the breaker
        clock.advance(timedelta(seconds=60))
        trial = asyncio.create_task(http.get_json("/trial"))  # half-open trial starts
        await _settle()
        assert hits == ["/stale", "/boom", "/trial"]

        gates["/stale"].set()  # stale call completes mid-trial
        if stale_ok:
            await stale
        else:
            with pytest.raises(ProviderUnavailable):
                await stale

        with pytest.raises(ProviderUnavailable):
            await http.get_json("/second")  # still rejected: no second trial
        assert "/second" not in hits

        gates["/trial"].set()
        await trial  # trial resolves and closes the breaker
        await http.get_json("/after")
        assert hits[-1] == "/after"


async def test_stale_failure_does_not_break_half_open_exclusivity() -> None:
    await _stale_scenario(stale_ok=False)


async def test_stale_success_does_not_close_breaker() -> None:
    await _stale_scenario(stale_ok=True)


def test_redaction_filter_on_crafted_record() -> None:
    record = logging.LogRecord(
        "httpx",
        logging.INFO,
        __file__,
        1,
        'HTTP Request: %s %s "%s"',
        ("GET", "https://h/x?API_KEY=SECRET123&a=1&apikey=SECRET456", "HTTP/1.1 200 OK"),
        None,
    )
    assert _RedactSecrets().filter(record)
    text = record.getMessage()
    assert "SECRET" not in text and "a=1" in text and text.count("REDACTED") == 2


async def test_httpx_info_log_is_redacted(caplog: pytest.LogCaptureFixture) -> None:
    h = Harness([httpx.Response(200, json={})])
    with caplog.at_level(logging.INFO, logger="httpx"):
        async with h.client() as http:
            await http.get_json("/x", {"api_key": "SECRET123", "apikey": "SECRET456"})
    assert "SECRET123" not in caplog.text and "SECRET456" not in caplog.text
    assert "REDACTED" in caplog.text

"""Resilient JSON-over-HTTP client: retries with jittered backoff, a circuit breaker, Decimal JSON.

Error messages deliberately omit query strings: they can carry API keys.
"""

import asyncio
import json
import random
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timedelta
from decimal import Decimal
from types import TracebackType
from typing import Any, Self

import httpx

from prediction_desk.clock import Clock

_RETRY_AFTER_CAP_S = 30.0


class ProviderError(Exception):
    """Base class for every failure talking to an external data provider."""

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(f"{provider}: {message}")
        self.provider = provider


class ProviderUnavailable(ProviderError):
    """Retries exhausted, or the circuit breaker is open."""


class ProviderRateLimited(ProviderError):
    """The provider said "slow down" in its response body."""


class ProviderHttpError(ProviderError):
    """A non-retryable 4xx response."""

    def __init__(self, provider: str, status: int, path: str) -> None:
        super().__init__(provider, f"HTTP {status} for {path}")
        self.status = status


def _retry_after_seconds(resp: httpx.Response) -> float | None:
    """Integer-seconds Retry-After only; HTTP-date and junk values are ignored."""
    raw = resp.headers.get("Retry-After")
    if raw is None:
        return None
    raw = raw.strip()
    if not (raw.isascii() and raw.isdigit()):
        return None
    return min(float(int(raw)), _RETRY_AFTER_CAP_S)


class ResilientHttp:
    def __init__(
        self,
        provider: str,
        base_url: str,
        clock: Clock,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_s: float = 10.0,
        max_attempts: int = 3,
        backoff_base_s: float = 0.5,
        breaker_threshold: int = 5,
        breaker_cooldown: timedelta = timedelta(seconds=60),
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: random.Random | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.provider = provider
        self._clock = clock
        self._client = httpx.AsyncClient(base_url=base_url, transport=transport, timeout=timeout_s)
        self._max_attempts = max_attempts
        self._backoff_base_s = backoff_base_s
        self._breaker_threshold = breaker_threshold
        self._breaker_cooldown = breaker_cooldown
        self._sleep = sleep
        self._rng = rng if rng is not None else random.Random()
        self._failures = 0
        self._open_until: datetime | None = None
        self._trial_in_flight = False

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def get_json(self, path: str, params: Mapping[str, str] | None = None) -> Any:
        is_trial = self._admit()
        try:
            data = await self._get_with_retries(path, params)
        except ProviderUnavailable:
            self._record_unavailable(is_trial)
            raise
        except BaseException:
            # 4xx or cancellation: not a breaker failure; just release a trial slot.
            if is_trial:
                self._trial_in_flight = False
            raise
        self._failures = 0
        self._open_until = None
        self._trial_in_flight = False
        return data

    def _admit(self) -> bool:
        """Raise if the breaker blocks this call. Returns True if it is the half-open trial."""
        if self._open_until is None:
            return False
        if self._clock.now() < self._open_until or self._trial_in_flight:
            raise ProviderUnavailable(self.provider, "circuit breaker open")
        self._trial_in_flight = True
        return True

    def _record_unavailable(self, was_trial: bool) -> None:
        self._trial_in_flight = False
        if was_trial:
            self._open_until = self._clock.now() + self._breaker_cooldown
            return
        self._failures += 1
        if self._failures >= self._breaker_threshold:
            self._open_until = self._clock.now() + self._breaker_cooldown

    async def _get_with_retries(self, path: str, params: Mapping[str, str] | None) -> Any:
        for attempt in range(1, self._max_attempts + 1):
            retry_after: float | None = None
            try:
                resp = await self._client.get(path, params=params)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                reason = f"{type(exc).__name__} for {path}"
            else:
                if resp.status_code < 400:
                    return json.loads(resp.text, parse_float=Decimal)
                if resp.status_code != 429 and resp.status_code < 500:
                    raise ProviderHttpError(self.provider, resp.status_code, path)
                reason = f"HTTP {resp.status_code} for {path}"
                retry_after = _retry_after_seconds(resp)
            if attempt == self._max_attempts:
                raise ProviderUnavailable(
                    self.provider, f"{reason} (gave up after {attempt} attempts)"
                )
            if retry_after is not None:
                delay = retry_after
            else:
                delay = self._backoff_base_s * 2 ** (attempt - 1) * self._rng.uniform(0.5, 1.5)
            await self._sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover

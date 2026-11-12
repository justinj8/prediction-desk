"""Resilient JSON-over-HTTP client: retries with jittered backoff, a circuit breaker, Decimal JSON.

Error messages deliberately omit query strings: they can carry API keys.
"""

import asyncio
import json
import logging
import random
import re
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timedelta
from decimal import Decimal
from types import TracebackType
from typing import Any, Self

import httpx

from prediction_desk.clock import Clock

_RETRY_AFTER_CAP_S = 30.0

_SECRET_PARAM = re.compile(r"(?<![A-Za-z0-9_])(api_?key)=[^&\s'\"]*", re.IGNORECASE)


class _RedactSecrets(logging.Filter):
    """httpx logs full request URLs at INFO; blank out api_key/apikey query values."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        redacted = _SECRET_PARAM.sub(r"\1=REDACTED", message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


for _name in ("httpx", "httpcore"):
    _logger = logging.getLogger(_name)
    if not any(isinstance(f, _RedactSecrets) for f in _logger.filters):
        _logger.addFilter(_RedactSecrets())


class _NonFiniteNumber(ValueError):
    pass


def _reject_constant(name: str) -> object:
    raise _NonFiniteNumber(name)


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
            # 4xx, bad body or cancellation: not a breaker failure; just release a trial slot.
            if is_trial:
                self._trial_in_flight = False
            raise
        self._record_success(is_trial)
        return data

    def _admit(self) -> bool:
        """Raise if the breaker blocks this call. Returns True if it is the half-open trial."""
        if self._open_until is None:
            return False
        if self._clock.now() < self._open_until or self._trial_in_flight:
            raise ProviderUnavailable(self.provider, "circuit breaker open")
        self._trial_in_flight = True
        return True

    def _record_success(self, was_trial: bool) -> None:
        if was_trial:
            self._failures = 0
            self._open_until = None
            self._trial_in_flight = False
        elif self._open_until is None:
            self._failures = 0
        # else: stale completion of a call admitted before the breaker opened; ignore it.

    def _record_unavailable(self, was_trial: bool) -> None:
        if was_trial:
            self._trial_in_flight = False
            self._open_until = self._clock.now() + self._breaker_cooldown
        elif self._open_until is None:
            self._failures += 1
            if self._failures >= self._breaker_threshold:
                self._open_until = self._clock.now() + self._breaker_cooldown
        # else: stale completion of a call admitted before the breaker opened; ignore it.

    def _parse(self, text: str) -> Any:
        reason: str
        try:
            return json.loads(text, parse_float=Decimal, parse_constant=_reject_constant)
        except _NonFiniteNumber:
            reason = "non-finite number in response"
        except ValueError:
            reason = "response body is not valid JSON"
        raise ProviderError(self.provider, reason)

    async def _get_with_retries(self, path: str, params: Mapping[str, str] | None) -> Any:
        for attempt in range(1, self._max_attempts + 1):
            retry_after: float | None = None
            fatal: str | None = None
            reason = ""
            try:
                resp = await self._client.get(path, params=params)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                reason = f"{type(exc).__name__} for {path}"
            except httpx.HTTPError as exc:
                fatal = f"{type(exc).__name__} for {path}"
            else:
                status = resp.status_code
                if status < 300:
                    return self._parse(resp.text)
                if status != 429 and status < 500:
                    raise ProviderHttpError(self.provider, status, path)
                reason = f"HTTP {status} for {path}"
                retry_after = _retry_after_seconds(resp)
            if fatal is not None:
                # Raised outside the except block so the httpx error (and its URL) is not chained.
                raise ProviderError(self.provider, fatal)
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

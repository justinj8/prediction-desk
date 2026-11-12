"""Alpha Vantage client: secondary cross-check series. Free tier is 25 calls/day, never retried."""

from collections.abc import Mapping
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from prediction_desk.net.http import ProviderError, ProviderRateLimited, ResilientHttp

_PROVIDER = "alpha_vantage"

AV_SERIES: Mapping[str, Mapping[str, str]] = {
    "CPI:monthly": {"function": "CPI", "interval": "monthly"},
    "NONFARM_PAYROLL": {"function": "NONFARM_PAYROLL"},
    "FEDERAL_FUNDS_RATE:monthly": {"function": "FEDERAL_FUNDS_RATE", "interval": "monthly"},
    "TREASURY_YIELD:monthly:2year": {
        "function": "TREASURY_YIELD",
        "interval": "monthly",
        "maturity": "2year",
    },
}


class AlphaVantageClient:
    def __init__(self, http: ResilientHttp, api_key: str) -> None:
        self._http = http
        self._api_key = api_key

    async def series(self, series_id: str) -> list[tuple[date, Decimal]]:
        if series_id not in AV_SERIES:
            raise ValueError(f"unknown Alpha Vantage series: {series_id!r}")
        params = {**AV_SERIES[series_id], "apikey": self._api_key}
        body = await self._http.get_json("/query", params)
        # Alpha Vantage reports quota and errors as HTTP 200; the body text may echo the key,
        # so it is never copied into an exception message.
        if not isinstance(body, dict):
            raise ProviderError(_PROVIDER, "malformed response")
        if "Information" in body or "Note" in body:
            raise ProviderRateLimited(_PROVIDER, "rate limit or quota notice in response body")
        if "Error Message" in body:
            raise ProviderError(_PROVIDER, "error message in response body")
        data = body.get("data")
        if not isinstance(data, list):
            raise ProviderError(_PROVIDER, "missing data list in response")
        out: list[tuple[date, Decimal]] = []
        for raw in data:
            point = self._to_point(series_id, raw)
            if point is not None:
                out.append(point)
        return out

    @staticmethod
    def _to_point(series_id: str, raw: Any) -> tuple[date, Decimal] | None:
        try:
            value_str = raw["value"]
            if value_str == ".":
                return None
            value = Decimal(value_str)
            obs_date = date.fromisoformat(raw["date"])
        except (KeyError, TypeError, ValueError, InvalidOperation):
            raise ProviderError(_PROVIDER, f"malformed observation for {series_id}") from None
        if not value.is_finite():
            raise ProviderError(_PROVIDER, f"non-finite value for {series_id}")
        return obs_date, value

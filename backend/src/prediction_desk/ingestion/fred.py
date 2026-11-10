"""FRED/ALFRED client: every vintage of a series, release calendar, and the vintage-time rule."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any
from zoneinfo import ZoneInfo

from prediction_desk.domain.enums import Series
from prediction_desk.net.http import ProviderError, ResilientHttp

MAX_PAGES = 50

_NY = ZoneInfo("America/New_York")
_REALTIME_START = "1776-07-04"
_REALTIME_END = "9999-12-31"

FRED_SERIES: tuple[str, ...] = (
    "CPIAUCSL",
    "CPIAUCNS",
    "CPILFESL",
    "CPILFENS",
    "GASREGW",
    "PPIACO",
    "CUSR0000SAH1",
    "PAYEMS",
    "ICSA",
    "CCSA",
    "ADPWNUSNERSA",
    "JTSJOL",
    "DFF",
    "DTB4WK",
    "DTB3",
    "SOFR",
    "DGS2",
)

FRED_RELEASE_IDS: Mapping[Series, int] = {Series.CPI: 10, Series.NFP: 50}


@dataclass(frozen=True)
class FredObservation:
    series_id: str
    obs_date: date
    published_on: date
    value: Decimal


def fred_vintage_ts(published_on: date) -> datetime:
    """Known from 00:00 New York on the day after FRED's realtime_start, as UTC.

    FRED gives dates, not release times; waiting for the next midnight means a decision on
    release day can never see that day's print.
    """
    midnight = datetime.combine(published_on + timedelta(days=1), time(0), tzinfo=_NY)
    return midnight.astimezone(UTC)


class FredClient:
    def __init__(self, http: ResilientHttp, api_key: str) -> None:
        self._http = http
        self._api_key = api_key

    async def observations_all_vintages(
        self, series_id: str, observation_start: date
    ) -> list[FredObservation]:
        out: list[FredObservation] = []
        offset = 0
        for _ in range(MAX_PAGES):
            data = await self._http.get_json(
                "/series/observations",
                {
                    "series_id": series_id,
                    "api_key": self._api_key,
                    "file_type": "json",
                    "realtime_start": _REALTIME_START,
                    "realtime_end": _REALTIME_END,
                    "observation_start": observation_start.isoformat(),
                    "sort_order": "asc",
                    "limit": "100000",
                    "offset": str(offset),
                },
            )
            page = self._page(data)
            count = int(data["count"])
            for raw in page:
                obs = self._to_observation(series_id, raw)
                if obs is not None:
                    out.append(obs)
            offset += len(page)
            if offset >= count:
                return out
            if not page:
                raise ProviderError("fred", f"empty page at offset {offset} of {count}")
        raise ProviderError("fred", f"more than {MAX_PAGES} pages for {series_id}")

    async def release_dates(self, release_id: int, start: date) -> list[date]:
        data = await self._http.get_json(
            "/release/dates",
            {
                "release_id": str(release_id),
                "api_key": self._api_key,
                "file_type": "json",
                "realtime_start": start.isoformat(),
                "realtime_end": _REALTIME_END,
                "include_release_dates_with_no_data": "true",
                "sort_order": "asc",
                "limit": "10000",
            },
        )
        try:
            return [date.fromisoformat(r["date"]) for r in data["release_dates"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderError("fred", f"malformed release/dates response: {exc!r}") from None

    async def release_name(self, release_id: int) -> str:
        data = await self._http.get_json(
            "/release",
            {"release_id": str(release_id), "api_key": self._api_key, "file_type": "json"},
        )
        try:
            return str(data["releases"][0]["name"])
        except (KeyError, IndexError, TypeError):
            raise ProviderError("fred", f"no release {release_id}") from None

    @staticmethod
    def _page(data: Any) -> list[dict[str, Any]]:
        try:
            page = data["observations"]
            data["count"]
        except (KeyError, TypeError):
            raise ProviderError("fred", "malformed observations response") from None
        if not isinstance(page, list):
            raise ProviderError("fred", "malformed observations response")
        return page

    @staticmethod
    def _to_observation(series_id: str, raw: dict[str, Any]) -> FredObservation | None:
        try:
            value_str = raw["value"]
            if value_str == ".":
                return None
            return FredObservation(
                series_id=series_id,
                obs_date=date.fromisoformat(raw["date"]),
                published_on=date.fromisoformat(raw["realtime_start"]),
                value=Decimal(value_str),
            )
        except (KeyError, TypeError, ValueError, InvalidOperation):
            raise ProviderError("fred", f"malformed observation for {series_id}") from None

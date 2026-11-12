"""Domain enums shared by the database models and the data/market adapters."""

from collections.abc import Mapping
from enum import StrEnum


class Series(StrEnum):
    CPI = "CPI"
    NFP = "NFP"
    FOMC = "FOMC"


class Venue(StrEnum):
    KALSHI = "kalshi"
    POLYMARKET = "polymarket"


class Measure(StrEnum):
    CPI_MOM = "cpi_mom"
    CORE_CPI_MOM = "core_cpi_mom"
    CPI_YOY = "cpi_yoy"
    CORE_CPI_YOY = "core_cpi_yoy"
    NFP_CHANGE = "nfp_change"
    FOMC_CHANGE_BP = "fomc_change_bp"


class Comparator(StrEnum):
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    EQ = "eq"
    BETWEEN = "between"


class MappingStatus(StrEnum):
    MAPPED = "mapped"
    UNMAPPED = "unmapped"


class DataSource(StrEnum):
    FRED = "fred"
    ALPHA_VANTAGE = "alpha_vantage"


MEASURE_SERIES: Mapping[Measure, Series] = {
    Measure.CPI_MOM: Series.CPI,
    Measure.CORE_CPI_MOM: Series.CPI,
    Measure.CPI_YOY: Series.CPI,
    Measure.CORE_CPI_YOY: Series.CPI,
    Measure.NFP_CHANGE: Series.NFP,
    Measure.FOMC_CHANGE_BP: Series.FOMC,
}


class BookMode(StrEnum):
    LIVE = "live"
    REPLAY = "replay"


class Slot(StrEnum):
    T24H = "t-24h"
    T2H = "t-2h"
    T15M = "t-15m"


class Verdict(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"


class OrderStatus(StrEnum):
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REFUSED = "refused"


class PositionSide(StrEnum):
    YES = "yes"
    NO = "no"

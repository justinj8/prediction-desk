"""Model registry: one base model per release series (read-only)."""

from collections.abc import Mapping
from types import MappingProxyType

from prediction_desk.domain.enums import Series
from prediction_desk.forecasting.base import BaseModel
from prediction_desk.forecasting.cpi import CpiModelV1
from prediction_desk.forecasting.fomc import FomcModelV1
from prediction_desk.forecasting.nfp import NfpModelV1

MODELS: Mapping[Series, BaseModel] = MappingProxyType(
    {
        Series.CPI: CpiModelV1(),
        Series.NFP: NfpModelV1(),
        Series.FOMC: FomcModelV1(),
    }
)

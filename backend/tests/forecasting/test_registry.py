import pytest

from prediction_desk.domain.enums import Series
from prediction_desk.forecasting.base import BaseModel
from prediction_desk.forecasting.cpi import CpiModelV1
from prediction_desk.forecasting.fomc import FomcModelV1
from prediction_desk.forecasting.nfp import NfpModelV1
from prediction_desk.forecasting.registry import MODELS


def test_registry_versions() -> None:
    assert set(MODELS) == {Series.CPI, Series.NFP, Series.FOMC}
    assert isinstance(MODELS[Series.CPI], CpiModelV1)
    assert isinstance(MODELS[Series.NFP], NfpModelV1)
    assert isinstance(MODELS[Series.FOMC], FomcModelV1)
    assert {s: m.model_version for s, m in MODELS.items()} == {
        Series.CPI: "cpi-ridge-v1",
        Series.NFP: "nfp-ridge-v1",
        Series.FOMC: "fomc-implied-v1",
    }
    for series, model in MODELS.items():
        conforming: BaseModel = model
        assert conforming.series is series
        assert callable(conforming.predict)


def test_registry_is_read_only() -> None:
    with pytest.raises(TypeError):
        MODELS[Series.CPI] = FomcModelV1()  # type: ignore[index]

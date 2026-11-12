"""Base model contract: every series model returns a ModelOutput."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from prediction_desk.db.models import Release
from prediction_desk.domain.enums import Measure, Series
from prediction_desk.forecasting.distributions import Distribution
from prediction_desk.ingestion.provider import MacroDataProvider


@dataclass(frozen=True)
class ModelOutput:
    model_version: str
    distributions: Mapping[Measure, Distribution]
    n_train: int
    diagnostics: Mapping[str, str]  # JSON-safe


class BaseModel(Protocol):
    series: Series
    model_version: str

    async def predict(
        self,
        session: AsyncSession,
        provider: MacroDataProvider,
        release: Release,
        asof: datetime,
    ) -> ModelOutput: ...

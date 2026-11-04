from datetime import date
from functools import lru_cache
from typing import Any, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_TEST_ADMIN_URL = "postgresql+psycopg://localhost/postgres"

_TARGET_SCHEME = "postgresql+psycopg://"
_RAW_SCHEMES = ("postgres://", "postgresql://")


def normalize_db_url(url: str) -> str:
    """Rewrite plain postgres URLs (as handed out by Railway) to the psycopg3 driver scheme."""
    for scheme in _RAW_SCHEMES:
        if url.startswith(scheme):
            return _TARGET_SCHEME + url[len(scheme) :]
    return url


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    trading_mode: Literal["paper"] = "paper"
    llm_training_cutoff: date = date(2026, 1, 1)
    test_database_admin_url: str = DEFAULT_TEST_ADMIN_URL

    @field_validator("trading_mode", mode="before")
    @classmethod
    def _normalize_trading_mode(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("database_url")
    @classmethod
    def _normalize_database_url(cls, value: str) -> str:
        return normalize_db_url(value)


@lru_cache
def get_settings() -> Settings:
    return Settings()

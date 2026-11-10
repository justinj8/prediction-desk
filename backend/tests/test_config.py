from datetime import date

import pytest
from pydantic import ValidationError

from prediction_desk import __version__
from prediction_desk.config import DEFAULT_TEST_ADMIN_URL, Settings, get_settings, normalize_db_url


def test_version() -> None:
    assert __version__ == "0.1.0"


@pytest.mark.parametrize(
    "raw",
    [
        "postgres://u:p@h:5432/db",
        "postgresql://u:p@h:5432/db",
        "postgresql+psycopg://u:p@h:5432/db",
    ],
)
def test_normalize_db_url(raw: str) -> None:
    assert normalize_db_url(raw) == "postgresql+psycopg://u:p@h:5432/db"


@pytest.mark.parametrize("mode", ["paper", "PAPER", " paper "])
def test_trading_mode_paper_variants_accepted(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRADING_MODE", mode)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h/db")
    assert Settings().trading_mode == "paper"


def test_trading_mode_defaults_to_paper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRADING_MODE", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h/db")
    assert Settings().trading_mode == "paper"


@pytest.mark.parametrize("mode", ["live", "real", ""])
def test_trading_mode_non_paper_rejected(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRADING_MODE", mode)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h/db")
    with pytest.raises(ValidationError):
        Settings()


def test_database_url_is_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgres://u:p@h/db")
    assert Settings().database_url == "postgresql+psycopg://u:p@h/db"


def test_test_database_admin_url_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEST_DATABASE_ADMIN_URL", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h/db")
    assert Settings().test_database_admin_url == DEFAULT_TEST_ADMIN_URL


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h/db")
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()


def test_settings_repr_hides_database_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:s3cret-db@h/db")
    monkeypatch.setenv("TEST_DATABASE_ADMIN_URL", "postgresql://u:s3cret-admin@h/postgres")
    settings = Settings()
    assert settings.database_url.endswith("s3cret-db@h/db")  # still loaded, just not shown
    for text in (repr(settings), str(settings)):
        assert "s3cret" not in text


def test_api_keys_are_secret_and_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h/db")
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
    unset = Settings(_env_file=None)
    assert unset.fred_api_key is None
    assert unset.alpha_vantage_api_key is None

    monkeypatch.setenv("FRED_API_KEY", "abc123")
    settings = Settings(_env_file=None)
    assert settings.fred_api_key is not None
    assert settings.fred_api_key.get_secret_value() == "abc123"
    assert "abc123" not in repr(settings)
    assert "abc123" not in str(settings)


def test_provider_base_url_and_observation_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h/db")
    s = Settings(_env_file=None)
    assert s.kalshi_api_base == "https://api.elections.kalshi.com/trade-api/v2"
    assert s.polymarket_gamma_base == "https://gamma-api.polymarket.com"
    assert s.polymarket_clob_base == "https://clob.polymarket.com"
    assert s.fred_api_base == "https://api.stlouisfed.org/fred"
    assert s.alpha_vantage_api_base == "https://www.alphavantage.co"
    assert s.fred_observation_start == date(2015, 1, 1)

import pytest

from conftest import _admin_url


@pytest.mark.parametrize("raw", ["postgres://u:p@h/postgres", "postgresql://u:p@h/postgres"])
def test_admin_url_from_env_is_normalized(raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_DATABASE_ADMIN_URL", raw)
    assert _admin_url() == "postgresql+psycopg://u:p@h/postgres"

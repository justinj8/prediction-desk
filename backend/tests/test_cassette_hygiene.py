import re
from pathlib import Path

import pytest

# A key-like query parameter (also URL-encoded: api%5Fkey, apikey%3D) or JSON field whose value
# is anything but REDACTED.
_LEAK = re.compile(
    r"api(?:_|%5F)?key(?:=|%3D)(?!REDACTED)[^&\s'\"]+"
    r"|\"api_?key\"\s*:\s*\"(?!REDACTED\")[^\"]+\"",
    re.IGNORECASE,
)
_TESTS = Path(__file__).resolve().parent
_GLOBS = (
    "**/cassettes/**/*.yaml",
    "**/cassettes/**/*.yml",
    "**/fixtures/**/*.json",
    "**/fixtures/**/*.yml",
    "**/fixtures/**/*.yaml",
)


def _recorded_files() -> list[Path]:
    return sorted({p for glob in _GLOBS for p in _TESTS.glob(glob)})


def test_no_api_keys_in_cassettes_or_fixtures() -> None:
    files = _recorded_files()
    assert any(p.suffix == ".json" for p in files)  # fixtures are really scanned
    assert any("cassettes" in p.parts for p in files)
    offenders = [
        str(path.relative_to(_TESTS))
        for path in files
        if _LEAK.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


@pytest.mark.parametrize(
    "text",
    [
        "https://x/y?api_key=abc123&z=1",
        "uri: https://x/y?apikey=abc123",
        "https://x/y?API_KEY=abc",
        "https://x/y?api%5Fkey=abc123",
        "https://x/y?api%5fkey%3Dabc123",
        "https://x/y?apikey%3Dabc123",
        '{"api_key": "abc123"}',
        '{"apikey":"abc123"}',
    ],
)
def test_leak_pattern_detects_real_keys(text: str) -> None:
    assert _LEAK.search(text)


@pytest.mark.parametrize(
    "text",
    [
        "https://x/y?api_key=REDACTED&apikey=REDACTED",
        "https://x/y?api%5Fkey=REDACTED&apikey%3DREDACTED",
        '{"api_key": "REDACTED"}',
        '{"_note": "hand-built fixture; no api key"}',
    ],
)
def test_leak_pattern_allows_redacted(text: str) -> None:
    assert not _LEAK.search(text)

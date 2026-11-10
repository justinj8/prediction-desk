import re
from pathlib import Path

_LEAK = re.compile(r"api_?key=(?!REDACTED)[^&\s'\"]+", re.IGNORECASE)
_TESTS = Path(__file__).resolve().parent


def test_no_api_keys_in_cassettes() -> None:
    offenders = [
        str(path.relative_to(_TESTS))
        for path in _TESTS.glob("**/cassettes/**/*.yaml")
        if _LEAK.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_leak_pattern_detects_real_keys_and_allows_redacted() -> None:
    assert _LEAK.search("https://x/y?api_key=abc123&z=1")
    assert _LEAK.search("uri: https://x/y?apikey=abc123")
    assert not _LEAK.search("https://x/y?api_key=REDACTED&apikey=REDACTED")

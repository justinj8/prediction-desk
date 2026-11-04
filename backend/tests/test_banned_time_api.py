import subprocess
from pathlib import Path


def test_ruff_bans_datetime_now(tmp_path: Path) -> None:
    pkg = Path(__file__).resolve().parents[1]
    bad = tmp_path / "probe.py"
    bad.write_text("from datetime import datetime\n\nx = datetime.now()\n")
    result = subprocess.run(
        [
            "uv",
            "run",
            "ruff",
            "check",
            "--config",
            str(pkg / "pyproject.toml"),
            "--select",
            "TID251",
            str(bad),
        ],
        cwd=pkg,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "TID251" in result.stdout

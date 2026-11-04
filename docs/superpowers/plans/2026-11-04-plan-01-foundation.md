# Prediction Desk — Plan 1: Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the backend skeleton every later plan builds on: tooling, settings with the paper-only guard, the injectable clock, value types for money and probability, the Postgres + Alembic layer with throwaway test databases, the `releases` table, and the append-only hash-chained audit log, all gated by CI.

**Architecture:** A Python 3.12 package `prediction_desk` under `backend/src/`, managed by `uv`. The app uses async SQLAlchemy 2.0 over psycopg 3, while Alembic migrations run on a sync psycopg engine so they never fight the test event loop. Each test gets its own database cloned from a migrated template database on local Homebrew Postgres 16.

**Tech Stack:** Python 3.12, uv, SQLAlchemy 2.0 (async), psycopg 3, Alembic, pydantic-settings, pytest + pytest-asyncio + Hypothesis, ruff, mypy (strict), GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-11-04-prediction-desk-design.md` (sections 3.1, 4, 4.1, 10.1, 10.3 are the ones this plan implements)

## Global Constraints

- Python `>=3.12,<3.13`; package name `prediction_desk`; source layout `backend/src/prediction_desk/`, tests in `backend/tests/`.
- **No Docker anywhere.** Local Postgres comes from Homebrew `postgresql@16`; CI uses the GitHub runner's preinstalled Postgres.
- **No module calls `datetime.now()`, `datetime.utcnow()` or `date.today()`.** Time comes from `Clock`. Enforced by ruff's `flake8-tidy-imports` banned-api.
- All datetimes are timezone-aware UTC. Naive datetimes are rejected wherever they enter the domain.
- Money is integer **cents**. Probabilities are `Decimal` quantized to 5 decimal places in `[0, 1]` (DB type `NUMERIC(6,5)`). No floats are persisted for either.
- `TRADING_MODE` must be `paper`; any other value makes settings loading fail.
- Audit hash: `hash = sha256(prev_hash ‖ canonical_json(envelope) ‖ seq ‖ ts)` with `envelope = {actor, entity, entity_id, action, payload}`; payloads reject floats and non-str keys; genesis `prev_hash` is 64 zeros; appends serialize on `pg_advisory_xact_lock`.
- `audit_events` rejects `UPDATE`, `DELETE` and `TRUNCATE` via triggers.
- Coverage gates: **100% on `prediction_desk/audit/`**, 85% overall (`risk/` and `pricing/` join the 100% gate in Plan 3).
- Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Railway hands out `postgresql://…` URLs.** SQLAlchemy would pick the wrong driver; settings must normalize `postgres://` and `postgresql://` to `postgresql+psycopg://`. (Test: Task 1)
2. **Naive datetimes entering a `SimClock` or an audit timestamp** would silently break replay ordering and hashing; both must raise `ValueError`. (Tests: Tasks 2 and 6)
3. **Audit payloads containing `Decimal`, `datetime`, `UUID`, `Enum`, nested dicts in different key orders, or non-ASCII text** must hash identically on every run; `float('nan')`/`inf` must be rejected, not stringified. (Tests: Task 5)
4. **A rolled-back audit append leaves a gap in `seq`.** Verification must still pass, because the chain links on `prev_hash`, not on contiguous `seq`. (Test: Task 6)
5. **`TRADING_MODE` set to `PAPER`, ` paper `, `live` or left unset.** Case and whitespace variants of `paper` are accepted; unset defaults to `paper`; everything else fails loudly at startup. (Test: Task 1)

## Plan Series (roadmap)

Each plan produces working, tested software and is written when the previous one is done:

| # | Plan | Delivers |
|---|---|---|
| **1** | **Foundation** (this plan) | Tooling, settings, clock, value types, DB + migrations, `releases`, audit log, CI |
| 2 | Data & markets | FRED/ALFRED vintage provider, Alpha Vantage, Kalshi + Polymarket adapters, `EventContract` mapping (validator; the LLM Contract Mapper arrives in Plan 4), snapshots |
| 3 | Quant core, risk & execution | CPI/NFP/FOMC base models, pricing + rounding + edge, Kelly + funnel + kill switch + structural gate, paper broker, settlement |
| 4 | Agents | LLM client wrapper, Contract Mapper, analysts → Lead → Skeptic, Risk Officer, Review agent |
| 5 | Pipeline, worker & replay | Postgres job queue, state machine, scheduler, replay engine and batches |
| 6 | API | Public and admin routers, JWT auth, SSE via LISTEN/NOTIFY, OpenAPI export; Railway project + `development` environment |
| 7 | Frontend | Next.js app, all pages, admin console, GitHub OAuth |
| 8 | Hardening & launch | Determinism golden test, Playwright, Sentry/alerts, app DB role with INSERT/SELECT-only on `audit_events`, production deploys, README/ADRs, first replay batch |

## File Structure (this plan)

```
.github/workflows/ci.yml                     CI: lint, types, tests, coverage gates, gitleaks
README.md                                    Project intro + local setup (brew postgres, uv)
backend/pyproject.toml                       deps, ruff (banned-api), mypy strict, pytest, coverage
backend/.env.example                         documented env vars
backend/alembic.ini
backend/migrations/env.py                    sync engine, URL from Settings
backend/migrations/versions/0001_releases.py
backend/migrations/versions/0002_audit_events.py
backend/src/prediction_desk/__init__.py      __version__
backend/src/prediction_desk/config.py        Settings, get_settings(), URL normalization, paper guard
backend/src/prediction_desk/clock.py         Clock protocol, WallClock, SimClock
backend/src/prediction_desk/domain/types.py  Cents, Prob, to_prob(), require_utc()
backend/src/prediction_desk/db/base.py       DeclarativeBase with naming convention
backend/src/prediction_desk/db/engine.py     make_engine(), make_sessionmaker()
backend/src/prediction_desk/db/models.py     Release (Series enum), AuditEvent
backend/src/prediction_desk/audit/hashing.py canonical_json(), compute_hash(), GENESIS_HASH
backend/src/prediction_desk/audit/log.py     AuditLog.append(), AuditLog.verify(), VerifyResult
backend/tests/conftest.py                    template DB + per-test DB fixtures
backend/tests/test_config.py
backend/tests/test_clock.py
backend/tests/test_banned_time_api.py
backend/tests/domain/test_types.py
backend/tests/db/test_migrations.py
backend/tests/db/test_releases.py
backend/tests/audit/test_hashing.py
backend/tests/audit/test_log.py
```

---

### Task 1: Backend scaffold, tooling and settings

**Files:**
- Create: `backend/pyproject.toml`, `backend/.env.example`, `backend/src/prediction_desk/__init__.py`, `backend/src/prediction_desk/config.py`
- Test: `backend/tests/test_config.py`

**Interfaces:**
- Produces: `prediction_desk.__version__: str = "0.1.0"`; `class Settings(BaseSettings)` with fields `database_url: str` (normalized), `trading_mode: Literal["paper"] = "paper"`, `llm_training_cutoff: date = date(2026, 1, 1)` (placeholder default, overridden by env), `test_database_admin_url: str = "postgresql+psycopg://localhost/postgres"`; `def get_settings() -> Settings` (cached with `functools.lru_cache`); `def normalize_db_url(url: str) -> str`.

- [ ] **Step 1: Install local Postgres 16 (one-time machine setup)**

Run: `brew install postgresql@16 && brew services start postgresql@16 && $(brew --prefix postgresql@16)/bin/psql -d postgres -c "select version();"`
Expected: a row containing `PostgreSQL 16`. Add `$(brew --prefix postgresql@16)/bin` to PATH in `~/.zshrc` if `psql` isn't found.

- [ ] **Step 2: Create the uv project and dependencies**

Run from `backend/`: `uv init --package --name prediction-desk --python 3.12 .`, then rename the generated package dir to `src/prediction_desk` if uv names it differently, then
`uv add sqlalchemy alembic "psycopg[binary]" pydantic pydantic-settings` and
`uv add --dev pytest pytest-asyncio pytest-cov hypothesis ruff mypy`.

Configure in `pyproject.toml`:
- `[tool.ruff]` `target-version = "py312"`, `line-length = 100`, `src = ["src", "tests"]`; `[tool.ruff.lint]` `select = ["E","F","W","I","UP","B","SIM","TID","DTZ","RUF"]`; `[tool.ruff.lint.flake8-tidy-imports.banned-api]` entries for `"datetime.datetime.now"`, `"datetime.datetime.utcnow"`, `"datetime.date.today"`, each with `msg = "Use prediction_desk.clock.Clock"`; per-file ignore `TID251` for `src/prediction_desk/clock.py` only.
- `[tool.mypy]` `strict = true`, `plugins = ["pydantic.mypy"]`, `mypy_path = "src"`.
- `[tool.pytest.ini_options]` `asyncio_mode = "auto"`, `testpaths = ["tests"]`, `addopts = "-ra"`.
- `[tool.coverage.run]` `source = ["prediction_desk"]`, `branch = true`.

- [ ] **Step 3: Write the failing tests**

```python
# backend/tests/test_config.py
import pytest
from pydantic import ValidationError
from prediction_desk import __version__
from prediction_desk.config import Settings, normalize_db_url

def test_version() -> None:
    assert __version__ == "0.1.0"

@pytest.mark.parametrize("raw", [
    "postgres://u:p@h:5432/db", "postgresql://u:p@h:5432/db", "postgresql+psycopg://u:p@h:5432/db",
])
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
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prediction_desk.config'`

- [ ] **Step 5: Implement `config.py`**

`normalize_db_url` rewrites the scheme prefix (`postgres://`, `postgresql://`) to `postgresql+psycopg://` and leaves `postgresql+psycopg://` unchanged. `Settings` uses `model_config = SettingsConfigDict(env_file=".env", extra="ignore")`, a `field_validator("trading_mode", mode="before")` that strips and lowercases, and a `field_validator("database_url")` that calls `normalize_db_url`. Write `.env.example` listing `DATABASE_URL`, `TRADING_MODE=paper`, `LLM_TRAINING_CUTOFF`, `TEST_DATABASE_ADMIN_URL`, each with a one-line comment.

- [ ] **Step 6: Run tests, lint and types**

Run: `cd backend && uv run pytest tests/test_config.py -v && uv run ruff check . && uv run ruff format --check . && uv run mypy src tests`
Expected: all tests PASS; ruff and mypy report no errors.

- [ ] **Step 7: Commit**

```bash
git add backend
git commit -m "feat(backend): scaffold prediction_desk package with paper-only settings"
```

---

### Task 2: Clock port

**Files:**
- Create: `backend/src/prediction_desk/clock.py`
- Test: `backend/tests/test_clock.py`, `backend/tests/test_banned_time_api.py`

**Interfaces:**
- Produces: `class Clock(Protocol): def now(self) -> datetime: ...`; `class WallClock` (returns `datetime.now(UTC)`); `class SimClock` with `__init__(self, start: datetime)`, `now() -> datetime`, `set(self, ts: datetime) -> None`, `advance(self, delta: timedelta) -> None`.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_clock.py
from datetime import UTC, datetime, timedelta
import pytest
from prediction_desk.clock import Clock, SimClock, WallClock

T0 = datetime(2024, 3, 12, 12, 30, tzinfo=UTC)

def test_wallclock_is_utc_aware() -> None:
    assert WallClock().now().tzinfo is UTC

def test_simclock_returns_start_and_advances() -> None:
    c = SimClock(T0)
    assert c.now() == T0
    c.advance(timedelta(minutes=5))
    assert c.now() == T0 + timedelta(minutes=5)

def test_simclock_set_forward_ok_backward_rejected() -> None:
    c = SimClock(T0)
    c.set(T0 + timedelta(hours=1))
    with pytest.raises(ValueError, match="backward"):
        c.set(T0)
    with pytest.raises(ValueError, match="backward"):
        c.advance(timedelta(seconds=-1))

def test_simclock_rejects_naive_datetimes() -> None:
    with pytest.raises(ValueError, match="timezone"):
        SimClock(datetime(2024, 3, 12, 12, 30))
    with pytest.raises(ValueError, match="timezone"):
        SimClock(T0).set(datetime(2024, 3, 13))

def test_both_satisfy_protocol() -> None:
    clocks: list[Clock] = [WallClock(), SimClock(T0)]
    assert all(isinstance(c.now(), datetime) for c in clocks)
```

```python
# backend/tests/test_banned_time_api.py
import subprocess
from pathlib import Path

def test_ruff_bans_datetime_now(tmp_path: Path) -> None:
    pkg = Path(__file__).resolve().parents[1]
    bad = tmp_path / "probe.py"
    bad.write_text("from datetime import datetime\n\nx = datetime.now()\n")
    result = subprocess.run(
        ["uv", "run", "ruff", "check", "--config", str(pkg / "pyproject.toml"),
         "--select", "TID251", str(bad)],
        cwd=pkg, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "TID251" in result.stdout
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/test_clock.py tests/test_banned_time_api.py -v`
Expected: `test_clock.py` FAILS with `ModuleNotFoundError`; `test_banned_time_api.py` already PASSES (it validates the Task 1 ruff config, so confirm it is green).

- [ ] **Step 3: Implement `clock.py`**

`Clock` is a `typing.Protocol` marked `@runtime_checkable`. `SimClock` validates `tzinfo is not None` (message contains "timezone") and normalizes to UTC with `.astimezone(UTC)`; `set`/`advance` reject moves to an earlier time (message contains "backward").

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_clock.py tests/test_banned_time_api.py -v && uv run mypy src tests`
Expected: PASS, no mypy errors.

- [ ] **Step 5: Commit**

```bash
git add backend/src/prediction_desk/clock.py backend/tests/test_clock.py backend/tests/test_banned_time_api.py
git commit -m "feat(clock): add injectable Clock with WallClock and monotonic SimClock"
```

---

### Task 3: Money and probability value types

**Files:**
- Create: `backend/src/prediction_desk/domain/__init__.py`, `backend/src/prediction_desk/domain/types.py`
- Test: `backend/tests/domain/test_types.py`

**Interfaces:**
- Produces: `Cents = NewType("Cents", int)`; `def cents(value: int) -> Cents` (rejects `bool` and non-int); `Prob = NewType("Prob", Decimal)`; `def to_prob(value: Decimal | str | int) -> Prob` (quantize to `Decimal("0.00001")` with `ROUND_HALF_EVEN`, reject outside `[0, 1]`, reject `float` with `TypeError`, reject NaN/Inf with `ValueError`); `def require_utc(ts: datetime) -> datetime` (raises `ValueError` mentioning "timezone" on naive input, returns `ts.astimezone(UTC)`).

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/domain/test_types.py
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
import pytest
from hypothesis import given, strategies as st
from prediction_desk.domain.types import cents, require_utc, to_prob

def test_cents_accepts_int_rejects_float_and_bool() -> None:
    assert cents(1050) == 1050
    with pytest.raises(TypeError):
        cents(10.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        cents(True)

@pytest.mark.parametrize("raw,expected", [
    ("0.123456", Decimal("0.12346")), ("0.5", Decimal("0.50000")), (1, Decimal("1.00000")), (0, Decimal("0.00000")),
])
def test_to_prob_quantizes(raw: str | int, expected: Decimal) -> None:
    assert to_prob(raw) == expected

@pytest.mark.parametrize("bad", ["-0.00001", "1.00001", "NaN", "Infinity"])
def test_to_prob_rejects_out_of_range_and_non_finite(bad: str) -> None:
    with pytest.raises(ValueError):
        to_prob(bad)

def test_to_prob_rejects_float() -> None:
    with pytest.raises(TypeError):
        to_prob(0.5)  # type: ignore[arg-type]

@given(st.decimals(min_value=0, max_value=1, allow_nan=False, allow_infinity=False))
def test_to_prob_always_in_unit_interval_with_5dp(d: Decimal) -> None:
    p = to_prob(d)
    assert Decimal(0) <= p <= Decimal(1)
    assert p.as_tuple().exponent == -5

def test_require_utc() -> None:
    est = timezone(timedelta(hours=-5))
    assert require_utc(datetime(2024, 1, 1, 7, tzinfo=est)) == datetime(2024, 1, 1, 12, tzinfo=UTC)
    with pytest.raises(ValueError, match="timezone"):
        require_utc(datetime(2024, 1, 1))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/domain -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prediction_desk.domain'`

- [ ] **Step 3: Implement `domain/types.py`** with the signatures in Interfaces.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/domain -v && uv run mypy src tests`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/prediction_desk/domain backend/tests/domain
git commit -m "feat(domain): add Cents and Prob value types with strict validation"
```

---

### Task 4: Database layer, Alembic and per-test databases, plus the `releases` table

**Files:**
- Create: `backend/src/prediction_desk/db/__init__.py`, `db/base.py`, `db/engine.py`, `db/models.py`, `backend/alembic.ini`, `backend/migrations/env.py`, `backend/migrations/script.py.mako`, `backend/migrations/versions/0001_releases.py`, `backend/tests/conftest.py`
- Test: `backend/tests/db/test_migrations.py`, `backend/tests/db/test_releases.py`

**Interfaces:**
- Consumes: `Settings`, `normalize_db_url` (Task 1).
- Produces:
  - `class Base(DeclarativeBase)` with `metadata = MetaData(naming_convention={"ix": "ix_%(column_0_label)s", "uq": "uq_%(table_name)s_%(column_0_name)s", "ck": "ck_%(table_name)s_%(constraint_name)s", "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s", "pk": "pk_%(table_name)s"})`.
  - `def make_engine(url: str) -> AsyncEngine`; `def make_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]` (`expire_on_commit=False`).
  - `class Series(StrEnum): CPI = "CPI"; NFP = "NFP"; FOMC = "FOMC"`.
  - `class Release(Base)`, table `releases`: `id: int` PK identity; `series: Series` (Postgres enum `series`); `period: str` (e.g. `"2024-02"` for CPI/NFP, `"2024-03-20"` for FOMC); `scheduled_at: datetime` (timestamptz); `actual_value: Decimal | None` (`NUMERIC(12,4)`); `released_at: datetime | None`; unique constraint on `(series, period)`.
  - Pytest fixtures in `conftest.py`: session-scoped sync fixture `template_db_url() -> str` that creates `pd_test_template_<pid>` on `TEST_DATABASE_ADMIN_URL`, runs `alembic upgrade head` against it, and drops it at session end; function-scoped `db_url(template_db_url) -> str` that runs `CREATE DATABASE pd_test_<uuid8> TEMPLATE <template>` and drops it (`WITH (FORCE)`) afterward; async `engine(db_url) -> AsyncEngine`; async `session(engine) -> AsyncSession`. Database DDL uses `psycopg.connect(admin_url_without_driver_suffix, autocommit=True)`.
- Alembic `env.py` uses a **sync** engine and reads the URL from `config.get_main_option("sqlalchemy.url")` if set, else `get_settings().database_url`; `target_metadata = Base.metadata`.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/db/test_migrations.py
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

def _cfg(url: str) -> Config:
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", url)
    return cfg

def test_upgrade_creates_releases_and_downgrade_removes(db_url: str) -> None:
    sync_engine = create_engine(db_url)
    assert "releases" in inspect(sync_engine).get_table_names()
    command.downgrade(_cfg(db_url), "base")
    assert "releases" not in inspect(sync_engine).get_table_names()
    command.upgrade(_cfg(db_url), "head")
    assert "releases" in inspect(sync_engine).get_table_names()
    sync_engine.dispose()
```

```python
# backend/tests/db/test_releases.py
from datetime import UTC, datetime
import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from prediction_desk.db.models import Release, Series

async def test_insert_and_read_release(session: AsyncSession) -> None:
    session.add(Release(series=Series.CPI, period="2024-02",
                        scheduled_at=datetime(2024, 3, 12, 12, 30, tzinfo=UTC)))
    await session.commit()
    row = (await session.execute(select(Release))).scalar_one()
    assert row.series is Series.CPI and row.actual_value is None
    assert row.scheduled_at == datetime(2024, 3, 12, 12, 30, tzinfo=UTC)

async def test_series_period_unique(session: AsyncSession) -> None:
    ts = datetime(2024, 3, 12, 12, 30, tzinfo=UTC)
    session.add_all([Release(series=Series.NFP, period="2024-02", scheduled_at=ts),
                     Release(series=Series.NFP, period="2024-02", scheduled_at=ts)])
    with pytest.raises(IntegrityError):
        await session.commit()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/db -v`
Expected: FAIL / ERROR (missing `prediction_desk.db` or missing fixtures).

- [ ] **Step 3: Implement `db/base.py`, `db/engine.py`, `db/models.py` (Release + Series only), `conftest.py`, and Alembic**

Run `uv run alembic init -t generic migrations` from `backend/`, then edit `env.py` as described in Interfaces. Write `0001_releases.py` by hand (or autogenerate, then review): create the enum type `series` and the `releases` table; the downgrade drops both.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/db -v && uv run mypy src tests`
Expected: PASS. Template creation is a one-time cost per session; each per-test database clone should take well under a second.

- [ ] **Step 5: Commit**

```bash
git add backend
git commit -m "feat(db): add async DB layer, Alembic, per-test template databases, releases table"
```

---

### Task 5: Audit hashing (pure functions)

**Files:**
- Create: `backend/src/prediction_desk/audit/__init__.py`, `backend/src/prediction_desk/audit/hashing.py`
- Test: `backend/tests/audit/test_hashing.py`

**Interfaces:**
- Consumes: `require_utc` (Task 3).
- Produces:
  - `GENESIS_HASH: str = "0" * 64`
  - `def canonical_json(payload: Mapping[str, Any]) -> bytes`: UTF-8 JSON with `sort_keys=True`, `separators=(",", ":")`, `ensure_ascii=False`, `allow_nan=False`. A `default` hook encodes `Decimal` → `str(d)`, `datetime` → `require_utc(d).isoformat(timespec="microseconds")`, `date` → `isoformat()`, `UUID` → `str`, `Enum` → `.value`. Raises `ValueError` for `float` NaN/Inf and `TypeError` for anything else unknown.
  - `def compute_hash(prev_hash: str, payload: Mapping[str, Any], seq: int, ts: datetime) -> str`: hex sha256 over `prev_hash.encode() + canonical_json(payload) + str(seq).encode() + require_utc(ts).isoformat(timespec="microseconds").encode()`. `prev_hash` must be 64 lowercase hex chars, else `ValueError`.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/audit/test_hashing.py
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from uuid import UUID
import pytest
from hypothesis import given, strategies as st
from prediction_desk.audit.hashing import GENESIS_HASH, canonical_json, compute_hash

TS = datetime(2024, 3, 12, 12, 30, 0, 123456, tzinfo=UTC)

class Side(Enum):
    YES = "yes"

def test_canonical_json_encodes_rich_types_deterministically() -> None:
    payload = {"b": Decimal("0.12345"), "a": TS, "u": UUID(int=1), "s": Side.YES, "t": "CPI ≥ 0.3%"}
    assert canonical_json(payload) == (
        '{"a":"2024-03-12T12:30:00.123456+00:00","b":"0.12345","s":"yes","t":"CPI ≥ 0.3%",'
        '"u":"00000000-0000-0000-0000-000000000001"}'
    ).encode()

@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_canonical_json_rejects_non_finite(bad: float) -> None:
    with pytest.raises(ValueError):
        canonical_json({"x": bad})

def test_canonical_json_rejects_naive_datetime_and_unknown_types() -> None:
    with pytest.raises(ValueError, match="timezone"):
        canonical_json({"x": datetime(2024, 1, 1)})
    with pytest.raises(TypeError):
        canonical_json({"x": object()})

@given(st.dictionaries(st.text(), st.one_of(st.integers(), st.text(), st.booleans()), max_size=8))
def test_canonical_json_ignores_insertion_order(d: dict[str, object]) -> None:
    assert canonical_json(d) == canonical_json(dict(reversed(list(d.items()))))

def test_compute_hash_is_stable_and_sensitive() -> None:
    h = compute_hash(GENESIS_HASH, {"a": 1}, 1, TS)
    assert h == compute_hash(GENESIS_HASH, {"a": 1}, 1, TS) and len(h) == 64
    assert h != compute_hash(GENESIS_HASH, {"a": 2}, 1, TS)
    assert h != compute_hash(GENESIS_HASH, {"a": 1}, 2, TS)
    assert h != compute_hash(h, {"a": 1}, 1, TS)

def test_compute_hash_rejects_malformed_prev_hash() -> None:
    with pytest.raises(ValueError):
        compute_hash("abc", {}, 1, TS)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/audit/test_hashing.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `audit/hashing.py`** with the signatures in Interfaces. Note: `json.dumps(..., allow_nan=False)` raises `ValueError` for NaN/Inf floats, which satisfies the test without extra code.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/audit/test_hashing.py -v && uv run mypy src tests`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/prediction_desk/audit backend/tests/audit/test_hashing.py
git commit -m "feat(audit): add canonical JSON and chained hash functions"
```

---

### Task 6: Append-only audit log with hash chain and verifier

**Files:**
- Create: `backend/migrations/versions/0002_audit_events.py`, `backend/src/prediction_desk/audit/log.py`
- Modify: `backend/src/prediction_desk/db/models.py` (add `AuditEvent`)
- Test: `backend/tests/audit/test_log.py`

**Interfaces:**
- Consumes: `Clock` (Task 2), `compute_hash`/`GENESIS_HASH` (Task 5), `session`/`engine`/`db_url` fixtures (Task 4).
- Produces:
  - `class AuditEvent(Base)`, table `audit_events`: `seq: int` (BIGINT `GENERATED BY DEFAULT AS IDENTITY` PK; `BY DEFAULT` is required because `append` inserts an explicitly reserved `seq`); `ts: datetime` (timestamptz); `actor: str` (e.g. `system`, `agent:lead_forecaster`, `admin:<login>`); `entity: str`; `entity_id: str`; `action: str`; `payload: dict[str, Any]` (JSONB); `prev_hash: str` (CHAR(64)); `hash: str` (CHAR(64), unique). Index on `(entity, entity_id)`.
  - Migration `0002`: table, a PL/pgSQL function `audit_events_block_mutation()` that raises `'audit_events is append-only'`, a `BEFORE UPDATE OR DELETE … FOR EACH ROW` trigger, and a `BEFORE TRUNCATE … FOR EACH STATEMENT` trigger; `REVOKE UPDATE, DELETE, TRUNCATE ON audit_events FROM PUBLIC`.
  - `AUDIT_LOCK_KEY: int = 7_310_001` (constant advisory-lock key).
  - `class AuditLog` with `__init__(self, clock: Clock)` and
    `async def append(self, session: AsyncSession, *, actor: str, entity: str, entity_id: str, action: str, payload: Mapping[str, Any]) -> AuditEvent`. It **does not commit**: the caller's transaction owns the commit, which gives the spec's transactional coupling.
  - `@dataclass(frozen=True) class VerifyResult: ok: bool; checked: int; first_broken_seq: int | None`
  - `async def verify(self, session: AsyncSession) -> VerifyResult`: walks rows in `seq` order, recomputes each hash, and checks each `prev_hash` against the previous row's `hash` (genesis for the first row).

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/audit/test_log.py
import asyncio
from datetime import UTC, datetime, timedelta
import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from prediction_desk.audit.hashing import GENESIS_HASH
from prediction_desk.audit.log import AuditLog
from prediction_desk.clock import SimClock
from prediction_desk.db.models import AuditEvent

T0 = datetime(2024, 3, 12, 12, 0, tzinfo=UTC)

async def _append(log: AuditLog, s: AsyncSession, i: int) -> AuditEvent:
    ev = await log.append(s, actor="system", entity="test", entity_id=str(i), action="noop", payload={"i": i})
    await s.commit()
    return ev

async def test_empty_chain_verifies(session: AsyncSession) -> None:
    assert (await AuditLog(SimClock(T0)).verify(session)).ok

async def test_chain_links_and_verifies(session: AsyncSession) -> None:
    log = AuditLog(SimClock(T0))
    first = await _append(log, session, 1)
    second = await _append(log, session, 2)
    assert first.prev_hash == GENESIS_HASH and second.prev_hash == first.hash
    result = await log.verify(session)
    assert result.ok and result.checked == 2 and result.first_broken_seq is None

async def test_append_uses_injected_clock(session: AsyncSession) -> None:
    clock = SimClock(T0)
    clock.advance(timedelta(minutes=7))
    ev = await _append(AuditLog(clock), session, 1)
    assert ev.ts == T0 + timedelta(minutes=7)

async def test_rolled_back_append_leaves_no_row_and_gap_still_verifies(session: AsyncSession) -> None:
    log = AuditLog(SimClock(T0))
    await _append(log, session, 1)
    await log.append(session, actor="system", entity="test", entity_id="x", action="noop", payload={})
    await session.rollback()
    await _append(log, session, 3)
    rows = (await session.execute(select(AuditEvent).order_by(AuditEvent.seq))).scalars().all()
    assert len(rows) == 2 and rows[1].seq > rows[0].seq + 1  # gap from rollback
    assert (await log.verify(session)).ok

@pytest.mark.parametrize("stmt", ["UPDATE audit_events SET action = 'x'", "DELETE FROM audit_events", "TRUNCATE audit_events"])
async def test_mutations_are_rejected(session: AsyncSession, stmt: str) -> None:
    await _append(AuditLog(SimClock(T0)), session, 1)
    with pytest.raises(DBAPIError, match="append-only"):
        await session.execute(text(stmt))

async def test_tampering_is_detected(session: AsyncSession) -> None:
    log = AuditLog(SimClock(T0))
    for i in range(3):
        await _append(log, session, i)
    seq2 = (await session.execute(select(AuditEvent.seq).order_by(AuditEvent.seq).offset(1))).scalars().first()
    await session.execute(text("ALTER TABLE audit_events DISABLE TRIGGER USER"))
    await session.execute(text("UPDATE audit_events SET payload = '{\"i\": 99}' WHERE seq = :s"), {"s": seq2})
    await session.execute(text("ALTER TABLE audit_events ENABLE TRIGGER USER"))
    await session.commit()
    result = await log.verify(session)
    assert not result.ok and result.first_broken_seq == seq2

async def test_concurrent_appends_do_not_fork_chain(engine: AsyncEngine) -> None:
    maker = async_sessionmaker(engine, expire_on_commit=False)
    log = AuditLog(SimClock(T0))
    async def worker(i: int) -> None:
        async with maker() as s:
            await _append(log, s, i)
    await asyncio.gather(*(worker(i) for i in range(20)))
    async with maker() as s:
        result = await log.verify(s)
    assert result.ok and result.checked == 20
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/audit/test_log.py -v`
Expected: FAIL with `ImportError` for `AuditLog` / `AuditEvent`.

- [ ] **Step 3: Write migration `0002_audit_events.py`** (table, function, both triggers, REVOKE; the downgrade drops triggers, function and table in reverse order).

- [ ] **Step 4: Implement `AuditEvent` and `audit/log.py`**

`append`, in order: `SELECT pg_advisory_xact_lock(:key)`; read the latest `(hash)` by `seq DESC LIMIT 1` (or `GENESIS_HASH`); `ts = clock.now()`; reserve `seq` with `SELECT nextval(pg_get_serial_sequence('audit_events','seq'))`; compute the hash; insert the row with that explicit `seq`; `flush`; return the row. The lock is released when the caller's transaction ends, which is what serializes concurrent appenders.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/audit -v && uv run mypy src tests`
Expected: PASS, including `test_concurrent_appends_do_not_fork_chain`.

- [ ] **Step 6: Commit**

```bash
git add backend
git commit -m "feat(audit): add append-only hash-chained audit log with verifier"
```

---

### Task 7: CI pipeline and README

**Files:**
- Create: `.github/workflows/ci.yml`, `README.md`

**Interfaces:**
- Consumes: everything above (runs the full suite).

- [ ] **Step 1: Write `ci.yml`**

Trigger on `push` to `main` and on `pull_request`. Job `backend` on `ubuntu-latest`, `defaults.run.working-directory: backend`:
1. `actions/checkout@v4`
2. `astral-sh/setup-uv@v6` with `python-version: "3.12"`
3. Start the runner's built-in Postgres (no service container): `sudo systemctl start postgresql.service` then `sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'postgres';"`
4. env: `TEST_DATABASE_ADMIN_URL=postgresql+psycopg://postgres:postgres@localhost/postgres`, `DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost/postgres`
5. `uv sync --frozen`
6. `uv run ruff check .` and `uv run ruff format --check .`
7. `uv run mypy src tests`
8. `uv run pytest --cov --cov-report=term-missing --cov-fail-under=85`
9. `uv run coverage report --include='*/prediction_desk/audit/*' --fail-under=100`

Job `secrets` runs `gitleaks/gitleaks-action@v2` with `fetch-depth: 0`.

- [ ] **Step 2: Validate the workflow locally**

Run: `uvx --from actionlint-py actionlint .github/workflows/ci.yml`
Expected: no output (exit 0).

- [ ] **Step 3: Run the exact CI commands locally**

Run: `cd backend && uv run ruff check . && uv run ruff format --check . && uv run mypy src tests && uv run pytest --cov --cov-report=term-missing --cov-fail-under=85 && uv run coverage report --include='*/prediction_desk/audit/*' --fail-under=100`
Expected: all green; total coverage ≥ 85%; audit at 100%.

- [ ] **Step 4: Write `README.md`**

Include: a one-paragraph pitch (from spec §1), a link to the spec, local setup (`brew install postgresql@16`, `brew services start postgresql@16`, `cd backend && uv sync`, `uv run pytest`), and the plan roadmap table from this plan.

- [ ] **Step 5: Commit**

```bash
git add .github README.md
git commit -m "ci: add backend CI with Postgres, coverage gates and gitleaks; add README"
```

---

## Definition of Done (Plan 1)

- `cd backend && uv run pytest` passes locally against Homebrew Postgres 16.
- ruff, ruff format and `mypy --strict` are clean.
- Coverage ≥ 85% overall and 100% on `prediction_desk/audit/`.
- `alembic upgrade head` → `downgrade base` → `upgrade head` round-trips.
- Seven commits on `main`, one per task.

---

## Execution Outcome and Carry-Forward (added after execution)

Plan 1 was executed with subagent-driven development: each task was implemented, then got a task review, then a final whole-branch review and one fix wave. The result is 109 tests, 99.35% overall coverage, and 100% coverage on `audit/`.

**Rulings that amended this plan:**
- **R5:** `append` stores and hashes the canonical-JSON round trip of the payload, so verify can reproduce the hash from JSONB.
- **R7:** the hash covers an envelope `{actor, entity, entity_id, action, payload}`, not the payload alone.
- **R8:** `canonical_json` rejects floats and non-str keys at any depth.
- **R10:** `AuditLog.append` raises unless the transaction is READ COMMITTED.
- **R11:** `to_prob` rejects tiny negatives rather than clamping them.
- Timestamps are enforced as UTC at the DB boundary by `UTCDateTime`. Engines force `timezone=UTC`.
- The `AuditLog.append` contract: READ COMMITTED only; call it last, right before commit and after all row locks; it never commits.

**Requirements carried into later plans:**
- **Plan 2:** new tables use `UTCDateTime` for every timestamp, and declare `Identity()` to match migrations. The drift test (`tests/db/test_drift.py`) must stay green. StrEnum columns must agree with the migration on member names vs. values.
- **Plan 3:** a unit-of-work helper owns sessions and transactions. It must refuse AUTOCOMMIT engines (R14) and always open READ COMMITTED. Tighten the C3 test so it counts rows before rolling back (R15).
- **Plan 4:** `LLM_TRAINING_CUTOFF` must fail closed: required, or defaulting conservatively (final-review M2).
- **Plans 5/6:** every caller of `AuditLog.verify` (the API endpoint and the nightly job) treats any exception as a chain break and trips the kill switch, i.e. it fails closed (R13). Decode-level tampers such as `ts='infinity'` or huge JSON ints can raise.
- **Plan 8:** create an app DB role with INSERT/SELECT only on `audit_events`. Revisit startup `-c timezone=UTC` if deploying behind a transaction-mode pooler (R16). Run actionlint and confirm the first GitHub CI run (the Postgres auth and pg_isready loop, and whether gitleaks needs `pull-requests: read`).

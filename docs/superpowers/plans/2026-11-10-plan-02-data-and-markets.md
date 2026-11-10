# Prediction Desk — Plan 2: Data & Markets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the system's inputs:
- point-in-time macro data (FRED/ALFRED vintages and Alpha Vantage)
- a release calendar
- Kalshi and Polymarket adapters
- deterministic mapping of venue contracts onto a canonical `EventContract`
- recorded order-book snapshots

These are everything the quant core (Plan 3) and the replay engine (Plan 5) read.

**Architecture:**
- **Hexagonal ports.**
  - `VenueAdapter` (Kalshi, Polymarket) and the ingestion clients only talk HTTP.
  - Application services (`ingestion.store`, `calendar.schedule`, `markets.discovery`, `markets.snapshots`) write Postgres.
  - Readers (`MacroDataProvider`, `latest_snapshot_asof`) answer "as of time T" queries, so live and replay share one code path.
- **One HTTP layer.** All network access goes through a single `ResilientHttp` wrapper that provides:
  - retries with jittered backoff
  - a per-provider circuit breaker
  - Decimal-only JSON parsing
- **Tests.** Tests replay recorded real responses (pytest-recording cassettes) and never hit the network in CI.

**Tech Stack:** Python 3.12, httpx, SQLAlchemy 2 (async) + Alembic, PyYAML, pytest + pytest-recording (vcrpy), local Homebrew Postgres 16.

**Spec:** `docs/superpowers/specs/2026-11-04-prediction-desk-design.md`. This plan implements §3.1 `calendar`/`ingestion`/`markets`, §4 tables `event_contracts`/`market_snapshots`/`data_observations`, §4.2, §5.3 (the rule-based mapper and validator; the LLM Contract Mapper is Plan 4), §10.2 and §10.3 adapters. Also read the "Execution Outcome and Carry-Forward" section of `docs/superpowers/plans/2026-11-04-plan-01-foundation.md`.

## Global Constraints

### Carried forward from Plan 1
- **Datetimes.** Every new timestamp column uses `UTCDateTime`. Every datetime is tz-aware UTC. Time comes from `Clock`, never `datetime.now()` (ruff TID251 enforces this).
- **Identity columns.** Every new identity PK is declared `Identity()` (BY DEFAULT) to match its migration.
- **Drift test.** `tests/db/test_drift.py` must stay green.
- **Enum columns.** StrEnum columns store the enum **values** via the `str_enum()` helper (Task 2). The migration creates Postgres enum types with exactly those values.
- **No floats.** Money and probabilities are never floats. Prices and sizes are `Decimal` from the moment JSON is parsed: `json.loads(..., parse_float=Decimal)`. Audit payloads may contain `Decimal`/`str`/`int` but never `float`, because `canonical_json` raises `TypeError`.
- **Audit append rules.** `AuditLog.append` runs only in READ COMMITTED transactions. It is called last, right before commit, after row locks, and it never commits.

### Testing
- No live network calls in tests.
  - HTTP behavior tests use `httpx.MockTransport`.
  - Adapter tests replay cassettes recorded once with `--record-mode=once`.
- Cassettes never contain API keys. `api_key` (FRED) and `apikey` (Alpha Vantage) query parameters are recorded as `REDACTED`, and a test fails the build if any cassette holds a real key.
- API keys come only from env/`.env` (`FRED_API_KEY`, `ALPHA_VANTAGE_API_KEY`) as `SecretStr`. Keys are never logged, printed, committed, or put in a URL that gets persisted.

### Point-in-time data
- **Vintage timestamp rule (leakage guard):** a FRED observation whose ALFRED `realtime_start` is date `D` gets `vintage_ts = 00:00 America/New_York on D+1`, converted to UTC.
  - FRED gives dates, not release times. Treating data as known only from the next midnight means a decision at T−15m on release day can never see that day's print.
  - Cost: same-day data is unavailable until midnight.
- Alpha Vantage has no vintages. Its `vintage_ts` is the fetch time (`clock.now()`), so replay before the first fetch sees no Alpha Vantage data.

### Venues
- Release times: CPI and NFP at **08:30 America/New_York**; FOMC decisions at **14:00 America/New_York**.
- Tracked Kalshi series: `KXCPI`, `KXCPICORE`, `KXCPIYOY`, `KXCPICOREYOY`, `KXPAYROLLS`, `KXFEDDECISION`.
- Tracked Polymarket Gamma tags: `us-cpi`, `fomc`.
- Base URLs (settings defaults):

  | Provider | Default base URL |
  |---|---|
  | Kalshi | `https://api.elections.kalshi.com/trade-api/v2` |
  | Polymarket Gamma | `https://gamma-api.polymarket.com` |
  | Polymarket CLOB | `https://clob.polymarket.com` |
  | FRED | `https://api.stlouisfed.org/fred` |
  | Alpha Vantage | `https://www.alphavantage.co` |

### Commits
- Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **One-sided or empty order book.** A venue returns no bids or no asks (common near close, or for illiquid strikes). Expected: `OrderBook` keeps empty sides, `best_bid`/`best_ask` are `None`, and the snapshot row stores NULLs without crashing. *(Tests: Tasks 6, 7, 9)*
2. **A FRED revision to an old month.** Expected: an as-of read before the revision's vintage returns the old value, and one after returns the revised value. *(Test: Task 3)*
3. **A venue lists a contract whose release isn't in our calendar.** This happens with a shutdown-delayed CPI or an unscheduled FOMC meeting. Expected: the contract is stored `unmapped` with reason `no_release`, never linked to the nearest wrong release, and discovery continues. *(Test: Task 8)*
4. **Alpha Vantage answers HTTP 200 with an `Information`/`Note` rate-limit body.** Expected: `ProviderRateLimited`, nothing stored, no retry storm against a 25-calls/day quota. *(Test: Task 5)*
5. **Prices arrive as JSON floats.** Polymarket Gamma's `bestBid` is `0.034`. Expected: they are parsed to `Decimal("0.034")` exactly, and no float ever reaches the DB or an audit payload. *(Tests: Tasks 1 and 7)*

---

## File Structure

```
backend/src/prediction_desk/
  net/http.py                 ResilientHttp (retries, breaker, Decimal JSON) + Provider* errors
  domain/enums.py             Series (moved here), Venue, Measure, Comparator, MappingStatus, DataSource, MEASURE_SERIES
  db/types.py                 + str_enum()
  db/models.py                + DataObservation, EventContract, MarketSnapshot (Series re-exported)
  ingestion/fred.py           FredClient, FredObservation, fred_vintage_ts(), FRED_SERIES, release ids
  ingestion/alpha_vantage.py  AlphaVantageClient, AV_SERIES
  ingestion/store.py          upsert_observations(), ingest_fred_series(), ingest_alpha_vantage()
  ingestion/provider.py       MacroDataProvider (as-of reads), Point
  calendar/fomc_meetings.yaml FOMC decision dates 2024–2027
  calendar/schedule.py        FomcMeeting, load_fomc_meetings(), sync_releases()
  markets/types.py            VenueMarket, MarketRef, BookLevel, OrderBook, VenueAdapter
  markets/kalshi.py           KalshiAdapter
  markets/polymarket.py       PolymarketAdapter
  markets/mapping.py          ContractTerms, Unmapped, map_market(), link_release()
  markets/discovery.py        discover_contracts()
  markets/snapshots.py        record_snapshots(), latest_snapshot_asof()
backend/migrations/versions/0003_markets_and_data.py
backend/tests/{net,ingestion,calendar,markets}/...   (+ cassettes/ dirs created by pytest-recording)
backend/tests/test_cassette_hygiene.py
```

---

### Task 1: Resilient HTTP layer, settings, and cassette infrastructure

**Files:**
- Create: `backend/src/prediction_desk/net/__init__.py`, `backend/src/prediction_desk/net/http.py`, `backend/tests/net/test_http.py`, `backend/tests/test_cassette_hygiene.py`
- Modify: `backend/pyproject.toml`, `backend/src/prediction_desk/config.py`, `backend/.env.example`, `backend/tests/conftest.py`, `backend/tests/test_config.py`

**Interfaces:**
- Consumes: `Clock`, `SimClock` (Plan 1).
- Produces:
  - `class ProviderError(Exception)` with attribute `provider: str`. Subclasses:
    - `ProviderUnavailable`: retries exhausted, or the breaker is open.
    - `ProviderRateLimited`: the provider said "slow down" in its body.
    - `ProviderHttpError`: a non-retryable 4xx. It carries `status: int`.
  - `class ResilientHttp`:
    - Constructor: `__init__(self, provider: str, base_url: str, clock: Clock, *, transport: httpx.AsyncBaseTransport | None = None, timeout_s: float = 10.0, max_attempts: int = 3, backoff_base_s: float = 0.5, breaker_threshold: int = 5, breaker_cooldown: timedelta = timedelta(seconds=60), sleep: Callable[[float], Awaitable[None]] = asyncio.sleep, rng: random.Random | None = None)`
    - `async def get_json(self, path: str, params: Mapping[str, str] | None = None) -> Any`
    - `async def aclose(self) -> None`
    - Supports `async with`.
  - New `Settings` fields:
    - `fred_api_key: SecretStr | None = None`
    - `alpha_vantage_api_key: SecretStr | None = None`
    - `kalshi_api_base: str`, `polymarket_gamma_base: str`, `polymarket_clob_base: str`, `fred_api_base: str`, `alpha_vantage_api_base: str`, with the default URLs from Global Constraints
    - `fred_observation_start: date = date(2015, 1, 1)`
  - A `vcr_config` fixture in `tests/conftest.py` that redacts the `api_key` and `apikey` query params to `REDACTED` and sets `decode_compressed_response=True`.

- [ ] **Step 1: Add dependencies.** From `backend/`, run:
  - `uv add httpx pyyaml`
  - `uv add --dev pytest-recording types-PyYAML`

- [ ] **Step 2: Write failing tests** in `tests/net/test_http.py`. Each one uses `httpx.MockTransport`, a `SimClock(T0)` and a fake `sleep` that appends its delay to a list and does not sleep. Tests and assertions:

```python
async def test_parses_floats_as_decimal() -> None:          # body '{"bestBid": 0.034, "n": 3}'
    assert data == {"bestBid": Decimal("0.034"), "n": 3} and type(data["bestBid"]) is Decimal

async def test_retries_5xx_then_succeeds() -> None:         # 503, 503, 200; backoff_base_s=0.5
    assert calls == 3 and len(delays) == 2
    assert 0.25 <= delays[0] <= 0.75 and 0.5 <= delays[1] <= 1.5   # base*2**(n-1) * U(0.5, 1.5)

async def test_retries_429_and_honors_retry_after() -> None:  # 429 with Retry-After: 2, then 200
    assert delays == [2.0]

async def test_does_not_retry_4xx() -> None:                # 404
    with pytest.raises(ProviderHttpError) as e: ...
    assert e.value.status == 404 and calls == 1

async def test_retries_exhausted_raises_unavailable() -> None:   # always 500, max_attempts=3
    assert calls == 3   # raises ProviderUnavailable

async def test_breaker_opens_then_half_opens_after_cooldown() -> None:
    # breaker_threshold=2, cooldown 60s; two failing calls -> third call raises ProviderUnavailable
    # with calls unchanged (no request sent); clock.advance(60s) -> next call reaches transport.
```

- [ ] **Step 3: Write the failing settings test.** Add `test_api_keys_are_secret_and_optional` to `tests/test_config.py`:
  - Both keys default to `None` when unset.
  - When `FRED_API_KEY=abc123` is set, `repr(Settings())` does not contain `abc123`, and `.get_secret_value() == "abc123"`.

- [ ] **Step 4: Write the failing cassette hygiene test** in `tests/test_cassette_hygiene.py`. It walks `tests/**/cassettes/**/*.yaml` and asserts that no file matches `re.compile(r"api_?key=(?!REDACTED)[^&\s'\"]+", re.I)`. It passes trivially until cassettes exist; it is a tripwire.

- [ ] **Step 5: Run the tests and confirm they fail.**
  - Command: `cd backend && uv run pytest tests/net tests/test_config.py -q`
  - Expected: `ModuleNotFoundError: prediction_desk.net`, plus the settings failure.

- [ ] **Step 6: Implement `net/http.py`.**
  - **Retry on:** `httpx.TimeoutException`, `httpx.TransportError`, status 429 and status ≥ 500.
  - **Delay:** `backoff_base_s * 2**(attempt-1) * rng.uniform(0.5, 1.5)`. A `Retry-After` header given in integer seconds overrides the delay, capped at 30s.
  - **JSON:** parse with `json.loads(resp.text, parse_float=Decimal)`.
  - **Breaker:**
    - It counts consecutive calls that ended in `ProviderUnavailable`.
    - At `breaker_threshold` it opens until `clock.now() + breaker_cooldown`.
    - While open, calls raise `ProviderUnavailable` without sending a request.
    - After the cooldown, the next call is a trial. Success resets the count; failure re-opens the breaker.
    - A 4xx does **not** count toward the breaker.

- [ ] **Step 7: Implement the remaining wiring.**
  - Add the settings fields.
  - Add `.env.example` entries: `FRED_API_KEY=` and `ALPHA_VANTAGE_API_KEY=`, empty, each with a comment saying where to get the key.
  - Add the `vcr_config` fixture.

- [ ] **Step 8: Run the full gates and confirm they pass.**
  - Command: `uv run pytest -q && uv run ruff check . && uv run ruff format --check . && uv run mypy src tests`

- [ ] **Step 9: Commit.** Message: `feat(net): resilient HTTP client with breaker and Decimal JSON; API key settings`

---

### Task 2: Domain enums, schema migration 0003, and models

**Files:**
- Create: `backend/src/prediction_desk/domain/enums.py`, `backend/migrations/versions/0003_markets_and_data.py`, `backend/tests/db/test_markets_schema.py`
- Modify: `backend/src/prediction_desk/db/types.py`, `backend/src/prediction_desk/db/models.py` (import `Series` from `domain.enums` and re-export it, so existing imports keep working)

**Interfaces:**
- Produces, in `domain/enums.py`. Every enum is a `StrEnum`.
  - `Series`: values `"CPI"`, `"NFP"`, `"FOMC"`. Moved here unchanged; the DB enum `series` is unchanged.
  - `Venue`: `KALSHI="kalshi"`, `POLYMARKET="polymarket"`.
  - `Measure`: `CPI_MOM="cpi_mom"`, `CORE_CPI_MOM="core_cpi_mom"`, `CPI_YOY="cpi_yoy"`, `CORE_CPI_YOY="core_cpi_yoy"`, `NFP_CHANGE="nfp_change"`, `FOMC_CHANGE_BP="fomc_change_bp"`.
  - `Comparator`: `GT="gt"`, `GTE="gte"`, `LT="lt"`, `LTE="lte"`, `EQ="eq"`, `BETWEEN="between"`.
  - `MappingStatus`: `MAPPED="mapped"`, `UNMAPPED="unmapped"`.
  - `DataSource`: `FRED="fred"`, `ALPHA_VANTAGE="alpha_vantage"`.
  - `MEASURE_SERIES: Mapping[Measure, Series]`: the four CPI measures map to `CPI`, `NFP_CHANGE` to `NFP`, `FOMC_CHANGE_BP` to `FOMC`.
- Produces, in `db/types.py`: `def str_enum(cls: type[StrEnum], name: str) -> sa.Enum`, which returns `sa.Enum(cls, name=name, values_callable=lambda e: [m.value for m in e])`.
- Produces, in `db/models.py`, the three models below.

**`DataObservation`** (table `data_observations`):
- Primary key: `(source, series_id, obs_date, vintage_ts)`.
- `source: DataSource`, via `str_enum(DataSource, "data_source")`.
- `series_id: str`.
- `obs_date: date`.
- `vintage_ts: datetime`, `UTCDateTime`.
- `published_on: date | None`. This is the FRED `realtime_start`; NULL for Alpha Vantage.
- `value: Decimal`, `Numeric(20, 6)`.
- `ingested_at: datetime`, `UTCDateTime`.

**`EventContract`** (table `event_contracts`):
- `id`: identity PK.
- `venue: Venue`, via `str_enum(Venue, "venue")`.
- `venue_market_id: str`.
- `venue_event_id: str`.
- `yes_token_id: str | None`.
- `title: str`.
- `rules_text: str`.
- `close_time: datetime`, `UTCDateTime`.
- `release_id: int | None`, FK to `releases.id`.
- `measure: Measure | None`.
- `comparator: Comparator | None`.
- `strike_lo: Decimal | None`, `Numeric(14,4)`.
- `strike_hi: Decimal | None`, `Numeric(14,4)`.
- `unit: str | None`.
- `resolution_precision: int | None`.
- `resolution_source: str | None`.
- `mapping_status: MappingStatus`.
- `mapping_confidence: Decimal`, `Numeric(3,2)`.
- `mapped_by: str`.
- `unmapped_reason: str | None`.
- `created_at`, `updated_at`: `UTCDateTime`.
- Constraints:
  - `UniqueConstraint("venue", "venue_market_id")`.
  - CHECK `ck_event_contracts_mapped_complete`: `mapping_status <> 'mapped' OR (release_id IS NOT NULL AND measure IS NOT NULL AND comparator IS NOT NULL AND strike_lo IS NOT NULL)`.
- **Strike convention:** `strike_lo` holds the single strike for GT/GTE/LT/LTE/EQ. BETWEEN uses `strike_lo..strike_hi`.

**`MarketSnapshot`** (table `market_snapshots`):
- `id`: `BigInteger` identity PK.
- `contract_id`: FK to `event_contracts.id`.
- `ts: datetime`, `UTCDateTime`.
- `best_bid`, `best_ask`: `Decimal | None`, `Numeric(6,5)`. These are YES prices.
- `bid_size`, `ask_size`: `Decimal | None`, `Numeric(20,2)`.
- `book: dict[str, Any]`, JSONB: `{"bids": [[price_str, size_str], ...], "asks": [...]}`, best first.
- Index `ix_market_snapshots_contract_id_ts` on `(contract_id, ts)`.
- **Spec deviation:** `last` and `volume` from spec §4 are deferred, because neither venue's book endpoint returns them.

- [ ] **Step 1: Write failing tests** in `tests/db/test_markets_schema.py`:
  - `test_enum_values_stored_lowercase`. Insert an `EventContract` with `venue=Venue.KALSHI`, then run `SELECT venue::text` and assert it returns `"kalshi"`.
  - `test_mapped_contract_requires_terms`. A mapped contract with `measure=None` raises `IntegrityError` on commit.
  - `test_unmapped_contract_allows_null_terms`.
  - `test_venue_market_id_unique_per_venue`. The same id on two different venues is allowed; the same id on the same venue raises `IntegrityError`.
  - `test_observation_vintages_coexist`. The same `(source, series_id, obs_date)` with two different `vintage_ts` values gives 2 rows.
  - `test_snapshot_naive_ts_rejected`.
- [ ] **Step 2: Run them and confirm they fail.**
  - Command: `uv run pytest tests/db -q`
  - Expected: `ImportError` (no enums or models yet).
- [ ] **Step 3: Implement.**
  - Write the enums and `str_enum`.
  - Write the models.
  - Write migration `0003` by hand:
    - It creates the enum types `data_source`, `venue`, `measure`, `comparator` and `mapping_status`, with exactly the values above.
    - It creates the three tables with the constraints above.
    - The downgrade drops the tables, then the types.
  - Run `uv run pytest tests/db/test_drift.py` and iterate until the drift between ORM and migration is `[]`.
- [ ] **Step 4: Run the full gates and confirm they pass.** This includes `tests/db/test_migrations.py`, which does an up/down/up round trip.
- [ ] **Step 5: Commit.** Message: `feat(db): add event_contracts, market_snapshots, data_observations`

---

### Task 3: FRED vintage ingestion and the point-in-time provider

**Files:**
- Create: `backend/src/prediction_desk/ingestion/{__init__,fred,store,provider}.py`, `backend/tests/ingestion/test_fred.py`, `backend/tests/ingestion/test_provider.py`

**Interfaces:**
- Consumes: `ResilientHttp`, `ProviderHttpError` (Task 1); `DataObservation`, `DataSource` (Task 2); `AuditLog` is NOT used here, because observations are data, not decisions.
- Produces:
  - `FRED_SERIES: tuple[str, ...] = ("CPIAUCSL", "CPIAUCNS", "CPILFESL", "CPILFENS", "GASREGW", "PPIACO", "CUSR0000SAH1", "PAYEMS", "ICSA", "CCSA", "ADPWNUSNERSA", "JTSJOL", "DFF", "DTB4WK", "DTB3", "SOFR", "DGS2")`. This is the spec §5.1 inputs, plus `PAYEMS` (the NFP target) and the NSA CPI series that YoY contracts resolve on.
  - `FRED_RELEASE_IDS: Mapping[Series, int] = {Series.CPI: 10, Series.NFP: 50}`. CPI is "Consumer Price Index" and NFP is "Employment Situation". Verify both names during recording (Step 1).
  - `@dataclass(frozen=True) class FredObservation`: `series_id: str`, `obs_date: date`, `published_on: date`, `value: Decimal`.
  - `def fred_vintage_ts(published_on: date) -> datetime`. This implements the Global Constraints vintage rule.
  - `class FredClient`, with `__init__(self, http: ResilientHttp, api_key: str)` and these methods:
    - `async def observations_all_vintages(self, series_id: str, observation_start: date) -> list[FredObservation]`
      - Calls `GET /series/observations` with `series_id`, `api_key`, `file_type=json`, `realtime_start=1776-07-04`, `realtime_end=9999-12-31`, `observation_start=<iso>`, `sort_order=asc`, `limit=100000`, `offset=<n>`.
      - Paginates while `offset + len(page) < count`.
      - Skips any `value == "."`.
      - Maps `published_on` from `realtime_start`.
    - `async def release_dates(self, release_id: int, start: date) -> list[date]`
      - Calls `GET /release/dates` with `release_id`, `api_key`, `file_type=json`, `realtime_start=<start>`, `realtime_end=9999-12-31`, `include_release_dates_with_no_data=true`, `sort_order=asc`, `limit=10000`.
      - Returns the `release_dates[].date` values.
    - `async def release_name(self, release_id: int) -> str`
      - Calls `GET /release` and returns `releases[0].name`.
  - In `store.py`:
    - `async def upsert_observations(session, rows: Iterable[DataObservation]) -> int`. Uses `INSERT … ON CONFLICT DO NOTHING` and returns the number inserted.
    - `async def ingest_fred_series(session, client: FredClient, clock: Clock, series_id: str, observation_start: date) -> int`. Commits once per series.
  - In `provider.py`:
    - `@dataclass(frozen=True) class Point`: `obs_date: date`, `value: Decimal`, `vintage_ts: datetime`.
    - `class MacroDataProvider`, with method `async def series_asof(self, session: AsyncSession, source: DataSource, series_id: str, asof: datetime, start: date | None = None) -> list[Point]`. It runs `SELECT DISTINCT ON (obs_date) … WHERE vintage_ts <= :asof ORDER BY obs_date, vintage_ts DESC`. `asof` goes through `require_utc`.

- [ ] **Step 1: Write failing tests and record cassettes.** Put `FRED_API_KEY` in `backend/.env`. Tests in `tests/ingestion/test_fred.py` are marked `@pytest.mark.vcr`. The client uses `os.environ.get("FRED_API_KEY", "REDACTED")`, so replay works without a key.
  - `test_release_names`: `release_name(10) == "Consumer Price Index"` and `release_name(50) == "Employment Situation"`.
  - `test_cpi_vintages_include_revisions`:
    - Fetch `CPIAUCSL` from `observation_start=2024-01-01`.
    - Assert that some `obs_date` appears with ≥2 distinct `published_on` values. Seasonal-factor revisions guarantee this.
    - Assert every `value` is a `Decimal`.
  - `test_missing_values_skipped`. This uses a `MockTransport` page containing `"value": "."`, not a cassette.
  - `test_pagination`. `MockTransport` with `count=3` across two pages; assert 3 observations come back.
  - `test_fred_vintage_ts`. Assert `fred_vintage_ts(date(2024, 3, 12)) == datetime(2024, 3, 13, 4, 0, tzinfo=UTC)` (EDT), and `fred_vintage_ts(date(2024, 1, 11)) == datetime(2024, 1, 12, 5, 0, tzinfo=UTC)` (EST).
  - Record with: `uv run --env-file .env pytest tests/ingestion/test_fred.py --record-mode=once`. `--env-file` exports the key to `os.environ`, which `.env` alone does not do outside `Settings`. Then re-run with `uv run pytest tests/ingestion/test_fred.py --record-mode=none` and no env file; it must pass offline.
- [ ] **Step 2: Write failing provider tests** in `tests/ingestion/test_provider.py`. These are DB tests that insert rows directly. This step covers Review Focus 2.
  - `test_asof_returns_latest_known_vintage`. Rows: obs 2024-01-01 at vintage V1 = `"308.417"` and at V2 = `"308.500"`. As-of V1 gives `308.417`; as-of V2 gives `308.500`; as-of before V1 gives `[]`.
  - `test_asof_never_returns_future_obs`. An observation whose first vintage is after `asof` is excluded.
  - `test_ingest_is_idempotent`. Running `ingest_fred_series` twice against the cassette inserts on the first run and returns `0` on the second.
- [ ] **Step 3: Run the tests and confirm they fail.** Command: `uv run pytest tests/ingestion -q`.
- [ ] **Step 4: Implement `fred.py`, `store.py` and `provider.py`.**
- [ ] **Step 5: Run the full gates and confirm they pass.** Check that the cassette hygiene test passes, which shows the key is redacted.
- [ ] **Step 6: Commit.** Message: `feat(ingestion): FRED/ALFRED vintage ingestion and point-in-time provider`. Commit the cassettes too.

---

### Task 4: Release calendar (FOMC YAML + FRED release dates)

**Files:**
- Create: `backend/src/prediction_desk/calendar/{__init__,schedule}.py`, `backend/src/prediction_desk/calendar/fomc_meetings.yaml`, `backend/tests/calendar/test_schedule.py`
- Modify: `backend/pyproject.toml` if the uv build backend needs the YAML declared as package data. Verify with `uv build` or `importlib.resources`.

**Interfaces:**
- Consumes:
  - `FredClient.release_dates`, `FRED_RELEASE_IDS` (Task 3)
  - `Release`, `Series` (Plan 1)
  - `DataObservation` (Task 2), for periods derived from data
- Produces:
  - `@dataclass(frozen=True) class FomcMeeting`, with fields `decision_date: date` and `decision_at: datetime` (UTC).
  - `def load_fomc_meetings() -> list[FomcMeeting]`, which reads the packaged YAML with `importlib.resources`.
  - `def scheduled_at(series: Series, day: date) -> datetime`. CPI and NFP return 08:30 America/New_York; FOMC returns 14:00 America/New_York. Both are converted to UTC.
  - `@dataclass class SyncResult`, with fields `inserted: int`, `updated: int`.
  - `async def sync_releases(session: AsyncSession, fred: FredClient, *, since: date) -> SyncResult`. Commits once.

**Rules for `sync_releases`:**
- **FOMC.**
  - Use one row per YAML meeting with `decision_date >= since`.
  - `period = decision_date.isoformat()`.
  - `scheduled_at = decision_at`.
- **CPI and NFP**, for each FRED release date `D >= since`:
  - **Period from data.** If `data_observations` holds a FRED row for the anchor series (`CPIAUCSL` for CPI, `PAYEMS` for NFP) with `published_on == D`, then `period = max(obs_date)` among those rows, formatted `YYYY-MM`.
  - **Assumed period.** Otherwise (a future release) `period` is the month before `D`.
  - The data-driven rule covers shutdown-delayed releases automatically.
- **Upsert by `(series, period)`.**
  - If a row exists, update its `scheduled_at` only when it differs.
  - If a row for the same series has the same scheduled date, a different period, and `actual_value IS NULL`, rewrite its period. This corrects an assumed period once the data shows the real one.

**YAML contents.** Write these decision dates exactly. They were verified against the Federal Reserve schedule and Kalshi `KXFEDDECISION` events on 2026-11-10. Each entry has `decision_time: "14:00"` and `tz: America/New_York`.

```
2024: 01-31, 03-20, 05-01, 06-12, 07-31, 09-18, 11-07, 12-18
2025: 01-29, 03-19, 05-07, 06-18, 07-30, 09-17, 10-29, 12-10
2026: 01-28, 03-18, 04-29, 06-17, 07-29, 09-16, 10-28, 12-09
2027: 01-27, 03-17, 04-28, 06-09, 07-28, 09-15, 10-27, 12-08
```

- [ ] **Step 1: Write the failing tests:**
  - `test_fomc_yaml_loads_32_meetings_utc`:
    - `len == 32`.
    - `decision_at` for `2026-12-09` is `datetime(2026, 12, 9, 19, 0, tzinfo=UTC)`, which is EST.
    - `decision_at` for `2026-09-16` is `18:00 UTC`, which is EDT.
  - `test_scheduled_at_cpi_0830_et`: `scheduled_at(Series.CPI, date(2026, 12, 10)) == datetime(2026, 12, 10, 13, 30, tzinfo=UTC)`.
  - `test_sync_uses_data_driven_period`. This is a DB test with a `MockTransport` FRED stub whose `release_dates` returns `["2025-11-20"]`.
    - Seed `data_observations` with CPIAUCSL rows at `published_on=2025-11-20` for `obs_date` `2025-08-01` and `2025-09-01`.
    - The CPI release scheduled 2025-11-20 gets period `"2025-09"`, not the assumed `"2025-10"`. This mirrors a shutdown-delayed release.
  - `test_sync_future_release_assumes_previous_month`.
  - `test_sync_is_idempotent`: a second run gives `SyncResult(inserted=0, updated=0)`.
  - `test_sync_corrects_assumed_period`.
  - Record the cassette: `--record-mode=once`, `since=date(2024, 1, 1)`.
- [ ] **Step 2: Run the tests and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the full gates and confirm they pass.**
- [ ] **Step 5: Commit.** Message: `feat(calendar): release calendar from FRED release dates and FOMC schedule`

---

### Task 5: Alpha Vantage client and ingestion

**Files:**
- Create: `backend/src/prediction_desk/ingestion/alpha_vantage.py`, `backend/tests/ingestion/test_alpha_vantage.py`
- Modify: `backend/src/prediction_desk/ingestion/store.py`

**Interfaces:**
- Consumes:
  - `ResilientHttp`, `ProviderRateLimited`, `ProviderError` (Task 1)
  - `upsert_observations` (Task 3)
  - `DataSource.ALPHA_VANTAGE`
- Produces:
  - `AV_SERIES: Mapping[str, Mapping[str, str]]`. Each key is our `series_id`; each value is the query parameters for `GET /query`:
    - `"CPI:monthly"`: `{"function": "CPI", "interval": "monthly"}`
    - `"NONFARM_PAYROLL"`: `{"function": "NONFARM_PAYROLL"}`
    - `"FEDERAL_FUNDS_RATE:monthly"`: `{"function": "FEDERAL_FUNDS_RATE", "interval": "monthly"}`
    - `"TREASURY_YIELD:monthly:2year"`: `{"function": "TREASURY_YIELD", "interval": "monthly", "maturity": "2year"}`
  - `class AlphaVantageClient`:
    - `__init__(self, http: ResilientHttp, api_key: str)`
    - `async def series(self, series_id: str) -> list[tuple[date, Decimal]]`
      - Sends `apikey` as a param and parses `data[].date` and `.value`.
      - Skips any `"."` value.
      - If the body has a top-level `"Information"` or `"Note"` key, raises `ProviderRateLimited`.
      - If the body has `"Error Message"`, raises `ProviderError`.
  - In `store.py`: `async def ingest_alpha_vantage(session, client, clock: Clock, series_id: str, *, min_interval: timedelta = timedelta(hours=24)) -> int`
    - Returns 0 without calling the client when the latest stored `vintage_ts` for that series is newer than `clock.now() - min_interval`.
    - Otherwise stores the rows with `vintage_ts = clock.now()` and `published_on = None`, then commits.

- [ ] **Step 1: Write the failing tests.**
  - `test_parses_cpi_series`: uses a cassette, recorded with `uv run --env-file .env pytest tests/ingestion/test_alpha_vantage.py --record-mode=once`. All values are `Decimal`, and the data is non-empty. This spends 1 of the 25 daily Alpha Vantage calls.
  - `test_rate_limit_body_raises_without_retry` (Review Focus 4):
    - A `MockTransport` returns 200 with `{"Information": "…rate limit…"}`.
    - Expect `ProviderRateLimited`.
    - Exactly 1 request is made.
    - 0 rows are stored.
  - `test_error_message_raises_provider_error`.
  - `test_freshness_gate_skips_recent_fetch`, a DB test:
    - The first ingest stores rows.
    - Advance the clock 1h. The second ingest returns 0, and the client is not called.
    - Advance the clock 25h. The third ingest calls the client and stores new vintages.
- [ ] **Step 2: Run the tests and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests and confirm they pass.** Also check that the cassette hygiene test is green.
- [ ] **Step 5: Commit.** Message: `feat(ingestion): Alpha Vantage cross-check series with freshness gate`

---

### Task 6: Venue port and Kalshi adapter

**Files:**
- Create: `backend/src/prediction_desk/markets/{__init__,types,kalshi}.py`, `backend/tests/markets/test_types.py`, `backend/tests/markets/test_kalshi.py`

**Interfaces:**

Consumes `ResilientHttp` (Task 1), `Venue` (Task 2) and `Clock`.

Produces, in `markets/types.py`:

- `@dataclass(frozen=True) class MarketRef`: `market_id: str`, `yes_token_id: str | None`.
- `@dataclass(frozen=True) class VenueMarket`:
  - Fields:
    - `venue: Venue`
    - `market_id: str`
    - `event_id: str`
    - `series_code: str` (the Kalshi series ticker, or the Polymarket tag)
    - `title: str`
    - `rules_text: str`
    - `close_time: datetime`
    - `status: str`
    - `strike: Mapping[str, Any]` (the venue's raw strike fields)
    - `yes_token_id: str | None`
  - Property `ref -> MarketRef`.
- `@dataclass(frozen=True) class BookLevel`: `price: Decimal`, `size: Decimal`.
- `@dataclass(frozen=True) class OrderBook`:
  - Fields: `bids: tuple[BookLevel, ...]`, `asks: tuple[BookLevel, ...]`, `ts: datetime`. Both sides are YES-side, best first.
  - `__post_init__` validation:
    - Every price is in `[0, 1]`.
    - Bids are strictly descending and asks strictly ascending.
    - `ts` is UTC.
  - Properties `best_bid` and `best_ask`, each `BookLevel | None`.
  - Method `to_json() -> dict[str, list[list[str]]]`, giving `{"bids": [[str(p), str(s)], ...], "asks": [...]}`.
- `class VenueAdapter(Protocol)`:
  - `venue: Venue`
  - `async def list_markets(self, series_code: str) -> list[VenueMarket]` (open markets only)
  - `async def order_book(self, ref: MarketRef) -> OrderBook`

Produces, in `markets/kalshi.py`:

- `KALSHI_SERIES: tuple[str, ...] = ("KXCPI", "KXCPICORE", "KXCPIYOY", "KXCPICOREYOY", "KXPAYROLLS", "KXFEDDECISION")`.
- `class KalshiAdapter(VenueAdapter)`, with `__init__(self, http: ResilientHttp, clock: Clock)`.
  - **`list_markets`**:
    - Calls `GET /markets?series_ticker=<s>&status=open&limit=1000`.
    - Follows `cursor` until it comes back empty.
    - Field mapping:

      | `VenueMarket` field | Kalshi source |
      |---|---|
      | `market_id` | `ticker` |
      | `event_id` | `event_ticker` |
      | `title` | `title` |
      | `rules_text` | `rules_primary` |
      | `close_time` | `close_time` |
      | `status` | `status` |
      | `strike` | `{"strike_type", "floor_strike", "cap_strike", "custom_strike"}` |
      | `yes_token_id` | `None` |

  - **`order_book`**:
    - Calls `GET /markets/{ticker}/orderbook`.
    - The response is `orderbook_fp.yes_dollars` and `orderbook_fp.no_dollars`. Both are lists of `[price_str, size_str]`, sorted **ascending** with the best last. Either may be `null` or missing.
    - YES bids = `yes_dollars` reversed.
    - YES asks = for each NO bid in `no_dollars` reversed, the level `(1 - price, size)`. Kalshi publishes bids only, so a NO bid at p is a YES ask at 1−p.
    - `ts = clock.now()`.

Steps:

- [ ] **Step 1: Write the failing tests.**
  - In `tests/markets/test_types.py`: `test_orderbook_rejects_unsorted_bids`, `test_orderbook_rejects_price_above_one`, and `test_empty_sides_have_no_best` (Review Focus 1).
  - In `tests/markets/test_kalshi.py`, recorded with `--record-mode=once` and no key needed:
    - `test_lists_open_kxcpi_markets`: every market has `strike["strike_type"] == "greater"`, and `rules_text` contains `"(single-decimal)"`.
    - `test_lists_fed_decision_custom_strikes`: every `strike["custom_strike"]` is a dict with one key in `{"Cut", "Hike"}`.
    - `test_orderbook_inverts_no_bids` (`MockTransport`). Given `yes_dollars=[["0.47","100.00"],["0.49","142.15"]]` and `no_dollars=[["0.47","100.00"],["0.49","139.54"]]`:
      - `bids[0] == BookLevel(Decimal("0.4900"), Decimal("142.15"))`
      - `asks[0] == BookLevel(Decimal("0.5100"), Decimal("139.54"))`
      - `asks[1].price == Decimal("0.5300")`
    - `test_orderbook_null_sides` (`MockTransport` with `{"orderbook_fp": {"yes_dollars": null, "no_dollars": null}}`): both sides are empty, with no exception.
- [ ] **Step 2: Run the tests and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests and confirm they pass.**
- [ ] **Step 5: Commit.** Message: `feat(markets): venue port and Kalshi adapter`.

---

### Task 7: Polymarket adapter

**Files:**
- Create: `backend/src/prediction_desk/markets/polymarket.py`
- Test: `backend/tests/markets/test_polymarket.py`

**Interfaces:**
- Consumes: the types from Task 6. It takes two `ResilientHttp` instances, one for Gamma and one for CLOB, because they are separate providers with separate breakers.
- Produces:
  - `POLYMARKET_TAGS: tuple[str, ...] = ("us-cpi", "fomc")`
  - `EVENT_SLUG_PATTERNS: tuple[re.Pattern[str], ...]`, which match:
    - `^[a-z]+-inflation-us-annual(-\d+)?$`
    - `^core-cpi-yoy-[a-z]+-\d{4}(-\d+)?$`
    - `^fed-decision-in-[a-z]+(-\d+)?$`
    - Multi-meeting events such as `fed-decisions-sepdec`, and every other event, are ignored.
  - `class PolymarketAdapter(VenueAdapter)`, with `__init__(self, gamma: ResilientHttp, clob: ResilientHttp)`.

**`list_markets(tag)` behavior:**
- Calls `GET /events?tag_slug=<tag>&closed=false&limit=100&offset=<n>` and pages until a page returns fewer than 100 events.
- Keeps only events whose slug matches one of the patterns.
- For each market inside a kept event, builds a `VenueMarket`:

  | `VenueMarket` field | Source |
  |---|---|
  | `market_id` | market `id` |
  | `event_id` | event `slug` |
  | `title` | `question` |
  | `rules_text` | `description` |
  | `close_time` | `endDate` |
  | `status` | `"active"` if not `closed`, else `"closed"` |
  | `strike` | `{"group_item_title": groupItemTitle}` |
  | `yes_token_id` | `json.loads(clobTokenIds)[0]` |

- `outcomes` and `clobTokenIds` are JSON-encoded **strings**. If `json.loads(outcomes)[0] != "Yes"`, skip that market and log it.

**`order_book(ref)` behavior:**
- Calls CLOB `GET /book?token_id=<yes_token_id>`.
- `bids` and `asks` are lists of `{"price": str, "size": str}`. Bids are sorted ascending and asks descending, so the best level is **last** on both sides. Reverse both.
- `ts` comes from `timestamp`, which is epoch milliseconds as a string.

- [ ] **Step 1: Write the failing tests.** Recorded tests use `--record-mode=once`.
  - `test_lists_only_single_meeting_fed_and_us_cpi_events`:
    - Every `event_id` matches a pattern.
    - No `event_id` starts with `fed-decisions-`.
    - Every market has a `yes_token_id`.
  - `test_cpi_group_titles_present`: every us-cpi market's `group_item_title` matches `^[≤≥]?\d+\.\d%$`.
  - `test_gamma_float_prices_stay_decimal` (`MockTransport`, Review Focus 5). An event whose market carries `"bestBid": 0.034` and `"orderPriceMinTickSize": 0.001` must produce no `float` anywhere in `dataclasses.asdict(market)`. Check this by walking nested dicts and lists recursively.
  - `test_book_reversal` (`MockTransport`):
    - Input: bids `[{"price":"0.001","size":"5"},{"price":"0.035","size":"87.99"}]`, asks `[{"price":"0.999","size":"1"},{"price":"0.038","size":"5051.98"}]`.
    - Expect `best_bid.price == Decimal("0.035")` and `best_ask.price == Decimal("0.038")`.
  - `test_book_empty_asks`.
- [ ] **Step 2: Run the tests and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests and confirm they pass.**
- [ ] **Step 5: Commit.** Message: `feat(markets): Polymarket adapter (Gamma discovery + CLOB books)`

---

### Task 8: Deterministic contract mapping and release linking

**Files:**
- Create: `backend/src/prediction_desk/markets/mapping.py`
- Test: `backend/tests/markets/test_mapping.py`

**Interfaces:**
- Consumes: `VenueMarket` (Task 6), enums (Task 2), `Release` (Plan 1).
- Produces:
  - `@dataclass(frozen=True) class ContractTerms`, with fields:
    - `measure: Measure`
    - `comparator: Comparator`
    - `strike_lo: Decimal`
    - `strike_hi: Decimal | None`
    - `unit: str`, one of `"pct"`, `"jobs"`, `"bp"`
    - `resolution_precision: int`
    - `resolution_source: str`
  - `@dataclass(frozen=True) class Unmapped`, with field `reason: str`. Reasons are a closed set:
    - `unknown_series`
    - `unsupported_strike_type`
    - `unparseable_strike`
    - `rules_text_mismatch`
    - `no_release`
    - `ambiguous_release`
    - `period_mismatch`
  - `MAPPER_VERSION = "rules:v1"`
  - `def map_market(m: VenueMarket) -> ContractTerms | Unmapped`
  - `def link_release(m: VenueMarket, terms: ContractTerms, releases: Sequence[Release]) -> Release | Unmapped`

**Rules for `map_market`, Kalshi markets.**

- **Measure, from `series_code`.**

  | series_code | Measure |
  |---|---|
  | `KXCPI` | `CPI_MOM` |
  | `KXCPICORE` | `CORE_CPI_MOM` |
  | `KXCPIYOY` | `CPI_YOY` |
  | `KXCPICOREYOY` | `CORE_CPI_YOY` |
  | `KXPAYROLLS` | `NFP_CHANGE` |
  | `KXFEDDECISION` | `FOMC_CHANGE_BP` |
  | anything else | `unknown_series` |

- **Strike types.**
  - **`greater`:** `GT` with `strike_lo = Decimal(str(floor_strike))`.
    - CPI measures use unit `pct` and precision 1. Rules text must match `more than` or `above`; otherwise the result is `rules_text_mismatch`.
    - NFP uses unit `jobs` and precision 0.
  - **`custom` (FOMC), with unit `bp` and precision 0:**

    | `custom_strike` | Comparator | `strike_lo` |
    |---|---|---|
    | `{"Cut": "25"}` | `EQ` | -25 |
    | `{"Cut": ">25"}` | `LT` | -25 |
    | `{"Hike": "0"}` | `EQ` | 0 |
    | `{"Hike": "25"}` | `EQ` | 25 |
    | `{"Hike": ">25"}` | `GT` | 25 |

    Generalize to any integer N. Rules text must contain the key word (`Cut` or `Hike`).
  - **Any other strike type:** `unsupported_strike_type`.

**Rules for `map_market`, Polymarket markets.**

- **Measure, from the event-slug pattern.**
  - `-inflation-us-annual` maps to `CPI_YOY`.
  - `core-cpi-yoy-` maps to `CORE_CPI_YOY`.
  - `fed-decision-in-` maps to `FOMC_CHANGE_BP`.
- **CPI titles**, unit `pct`, precision 1:
  - `≤X%` gives `LTE X`.
  - `≥X%` gives `GTE X`.
  - `X%` gives `EQ X`.
  - The description must contain `Consumer Price Index` and `12-month`.
- **FOMC titles**, unit `bp`:
  - `N+ bps decrease` gives `LTE -N`.
  - `N bps decrease` gives `EQ -N`.
  - `No change` gives `EQ 0`.
  - `N bps increase` gives `EQ N`.
  - `N+ bps increase` gives `GTE N`.
  - The description must contain `upper bound`.
- **Any other title:** `unparseable_strike`.

**`resolution_source`.**
- `"BLS"` for the CPI and NFP measures.
- `"Federal Reserve"` for FOMC.

**Rules for `link_release`.**
- Candidates are releases whose series is `MEASURE_SERIES[terms.measure]` and where `abs(scheduled_at - close_time) <= timedelta(hours=48)`.
  - Zero candidates gives `no_release`.
  - More than one gives `ambiguous_release`.
- Then cross-check the period:
  - **Kalshi:** the event ticker's trailing `-(\d{2})([A-Z]{3})` decodes to year and month (e.g. `26NOV` is 2026-11). It must equal the release's period (`YYYY-MM`) for CPI and NFP, or the meeting's month for FOMC.
  - **Polymarket:** the month name in the slug (`october`, `december`) must equal the release period's month for CPI, or the meeting's month for FOMC.
  - A mismatch gives `period_mismatch`.

- [ ] **Step 1: Write the failing tests.** They are table-driven and build `VenueMarket` literals from the real shapes:
  - `test_kalshi_cpi_greater`. `floor_strike=-0.3` gives `ContractTerms(CPI_MOM, GT, Decimal("-0.3"), None, "pct", 1, "BLS")`.
  - `test_kalshi_payrolls_greater`. `floor_strike=90000` gives `GT 90000`, unit `jobs`, precision 0.
  - `test_kalshi_fed_custom`, parametrized over the five `custom_strike` cases above.
  - `test_kalshi_rules_mismatch`. A CPI market whose `rules_text` lacks "more than"/"above" gives `Unmapped("rules_text_mismatch")`.
  - `test_kalshi_unsupported_strike_type`. `between` gives `Unmapped("unsupported_strike_type")`.
  - `test_polymarket_cpi_titles`, parametrized: `≤3.2%` gives `LTE 3.2`; `3.3%` gives `EQ 3.3`; `≥4.3%` gives `GTE 4.3`.
  - `test_polymarket_fed_titles`, parametrized over the five titles: `50+ bps decrease`, `25 bps decrease`, `No change`, `25 bps increase`, `50+ bps increase`.
  - `test_link_release_kalshi_cpi`:
    - The market is `KXCPI-26NOV-T-0.3` with `close_time=2026-12-10T13:29Z`.
    - The release is CPI with `period="2026-11"` and `scheduled_at=2026-12-10T13:30Z`.
    - The release is linked.
  - `test_link_release_polymarket_cpi`:
    - The event is `october-inflation-us-annual` with `endDate=2026-11-11T04:59Z`.
    - The release is CPI `2026-10` scheduled `2026-11-10T13:30Z`.
    - The release is linked.
  - `test_link_release_missing_release` (Review Focus 3). With no release in ±48h, the result is `Unmapped("no_release")`. It must not pick the nearest release from another month.
  - `test_link_release_period_mismatch`.
- [ ] **Step 2: Run the tests and confirm they fail.**
- [ ] **Step 3: Implement.** Put the mapping tables at module level as data, and keep the functions small.
- [ ] **Step 4: Run the tests and confirm they pass.**
- [ ] **Step 5: Commit.** Message: `feat(markets): rule-based contract mapping and release linking`.

---

### Task 9: Contract discovery and snapshot services

**Files:**
- Create: `backend/src/prediction_desk/markets/discovery.py`, `backend/src/prediction_desk/markets/snapshots.py`, `backend/tests/markets/test_discovery.py`, `backend/tests/markets/test_snapshots.py`

**Interfaces:**
- **Consumes:**
  - Adapters: `VenueAdapter`, plus `KALSHI_SERIES` and `POLYMARKET_TAGS`.
  - Mapping: `map_market`, `link_release` and `MAPPER_VERSION`.
  - `EventContract` and `MarketSnapshot`.
  - `AuditLog` (Plan 1).
  - `Clock`.
- **Produces:**
  - `@dataclass class DiscoveryResult` with fields: `mapped`, `unmapped`, `changed`, `unchanged` (all `int`), and `failures: list[tuple[str, str]]` holding `(series_code, error)` pairs.
  - `async def discover_contracts(session: AsyncSession, adapters: Mapping[Venue, tuple[VenueAdapter, Sequence[str]]], audit: AuditLog, clock: Clock) -> DiscoveryResult`. The `Sequence[str]` lists the series codes to scan for that venue.
    - Loads the releases once.
    - Upserts each market by `(venue, venue_market_id)` and computes `ContractTerms` or `Unmapped`.
    - **Audit only on change.** A new row, or a change in terms, status or release, produces exactly one audit event. It uses `entity="event_contract"` and `entity_id=f"{venue}:{market_id}"`. The `action` is one of `contract_mapped`, `contract_unmapped` or `contract_remapped`. The payload holds the venue, market id, release id, measure, comparator, `str(strike_lo)`, `str(strike_hi)` (or None), status, reason and `MAPPER_VERSION`. Strike values are strings, never floats.
    - **Transactions.** Each series code runs in its own transaction, with the audit append last, right before commit.
    - **Failures.** A `ProviderError` on one series code is recorded in `failures`, and the rest still run.
  - `@dataclass class SnapshotResult` with fields `recorded: int` and `failures: list[tuple[int, str]]` holding `(contract_id, error)` pairs.
  - `async def record_snapshots(session, adapters: Mapping[Venue, VenueAdapter], clock: Clock) -> SnapshotResult`.
    - Covers every `mapped` contract with `close_time > clock.now()`.
    - Fetches the book and inserts one `MarketSnapshot` with `best_bid`/`best_ask`/sizes (or NULLs) and `book=OrderBook.to_json()`.
    - A `ProviderError` on one contract is recorded, and the rest continue.
    - Commits once.
    - Writes no audit events: snapshots are observations, not decisions.
  - `async def latest_snapshot_asof(session, contract_id: int, asof: datetime) -> MarketSnapshot | None`. Returns the newest snapshot with `ts <= asof`.

- [ ] **Step 1: Write the failing tests.** They use fake adapters, i.e. in-memory classes implementing `VenueAdapter`, plus real DB fixtures.
  - **Discovery:**
    - `test_discovery_maps_and_audits_once`: the first run gives `mapped=1` and 1 audit event.
    - `test_discovery_rerun_is_silent`: the second run with identical markets gives `unchanged=1` and still exactly 1 audit event.
    - `test_discovery_terms_change_remaps`.
    - `test_discovery_unmapped_reason_persisted`: with no release, the row is unmapped with reason `no_release`.
    - `test_discovery_continues_after_provider_error`.
    - `test_discovery_audit_chain_verifies`: after the runs, `AuditLog.verify(session).ok` is True.
  - **Snapshots:**
    - `test_snapshot_records_best_levels`.
    - `test_snapshot_empty_book_stores_nulls` (Review Focus 1).
    - `test_snapshot_skips_closed_contracts`.
    - `test_snapshot_failure_isolated`.
    - `test_latest_snapshot_asof_never_returns_future`.
- [ ] **Step 2: Run the tests and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the full gates and confirm they pass:**
  - `uv run pytest -q --cov --cov-fail-under=85`
  - `uv run coverage report --include='*/prediction_desk/audit/*' --fail-under=100`
  - ruff
  - mypy
- [ ] **Step 5: Commit.** Message: `feat(markets): contract discovery with change-only auditing; order-book snapshots`

---

## Definition of Done (Plan 2)

- All gates are green locally and in CI, where only cassettes are used and nothing hits the network:
  - ruff, ruff format and `mypy --strict`
  - coverage ≥85% overall and 100% on `audit/`
  - the drift test
  - the cassette hygiene test
- Running `discover_contracts` against the recorded Kalshi and Polymarket cassettes maps every currently listed CPI, NFP and FOMC contract. The only exceptions are ones with a documented `unmapped_reason`.
- `MacroDataProvider.series_asof` returns revision-correct values for any timestamp.
- No float reaches the DB or an audit payload, and no API key appears in any committed file.

# Prediction Desk — Plan 3b: Risk and Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a `ReleasePricing` into sized, risk-approved paper orders. This plan also covers filling those orders against stored books, marking the book to market, running the kill switch, and settling positions against the officially reported value. Every state change is audited.

**Architecture:**
- **`risk/` decides.** Pure sizing math (R14 clamp, Kelly, the cap funnel) sits next to a `RiskEngine` that writes `risk_decisions` and is the only producer of `ApprovedOrder`.
- **`execution/` acts.** A `PaperBroker` turns an `ApprovedOrder` into an order, a fill, a position and a cash movement. `outcomes` and `settlement` close positions on the reported value.
- **Locking.** Every write runs in a READ COMMITTED unit of work that locks the book's `book_state` row first.
- **Transactions.** No transaction is held across the Risk Officer call.

**Tech Stack:** Python 3.12, SQLAlchemy 2 (async, psycopg 3), Alembic, Postgres 16, pytest + Hypothesis.

**Spec:** `docs/superpowers/specs/2026-11-04-prediction-desk-design.md`, specifically §4 (tables), §7 (risk), §8 (execution), and the kill-switch half of §6's sanity checks. Also read the "Execution Outcome and Carry-Forward" sections at the end of the Plan 1, Plan 2 and Plan 3a documents in `docs/superpowers/plans/`. The LLM Risk Officer itself is Plan 4; this plan defines its port and its fail-closed handling. Persisting forecasts/edges, pipeline runs, scheduling and replay batches are Plan 5.

## Global Constraints

### Carried forward
- Use `Clock`/`require_utc` for every timestamp.
- Never use `datetime.now` (ruff TID251).
- Never persist floats, and never put them in an audit payload.
- Money is `int` cents.
- Probabilities are `Prob` (5 dp).
- New test files need unique basenames (Plan 3a R5).
- Run tests with `--record-mode=none --block-network`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Timestamps:
  - New tables use `UTCDateTime` for every timestamp.
  - Primary keys use `Identity()`, matching the migration.
  - StrEnum columns use `str_enum` (values, not names).
  - `tests/db/test_drift.py` must stay green.
- `AuditLog.append` rules:
  - READ COMMITTED only.
  - Call it last, after row locks.
  - It never commits.
- Readers take an `AsyncSession`. Batch services that write take an `async_sessionmaker` and own short sessions (Plan 2 R9).

### User rulings for this plan
- **U1, settlement source.** Official values are computed from the **FRED first print**:
  - CPI MoM/YoY come from the `CPIAUCSL`/`CPILFESL` index.
  - NFP comes from `PAYEMS`.
  - FOMC comes from `DFEDTARU`.
  - An admin may override a value with a reason before settlement. A venue's own resolution, when supplied, is compared and flagged on mismatch; it never decides the payout.
- **U2, pricing sanity.** A **bug-type** coherence violation trips the global kill switch, whether that is a non-monotonic ladder or a complete Polymarket bracket whose sum falls outside 1 ± 0.02. An **incomplete** bracket (3a R10) only blocks that Polymarket event.
- **U3, re-evaluation slots.** A later slot may add to a held position **on the same side**. Every cap counts what is already held. Buying the opposite side of a held contract is rejected (`opposite_position`).
- **3a R14.** Clamp every fair probability to `[0.02, 0.98]` before the edge used for sizing and before Kelly.

### Design decisions (this plan)
- **D1, exposure.** Exposure is capital at risk: the cost basis of open positions (price paid + fees, in cents) plus the cost of approved decisions that have no order yet. Caps compare against `book_state.equity_cents` (the last mark).
- **D2, book lock.** Decision, fill, mark, settle and kill-switch transactions all take `SELECT … FOR UPDATE` on the book's `book_state` row. The spec says `FOR SHARE` for decisions. `FOR UPDATE` also serializes two concurrent decision transactions, which `FOR SHARE` would let both read the same exposure and jointly breach a cap.
- **D3, order type.** Orders are IOC marketable limits:
  - The fill walks levels priced `<= limit_price` up to `qty`.
  - The unfilled remainder is cancelled.
  - There are no resting orders and no exits; positions are held to settlement.
  - Status is `filled`, `partial`, `cancelled` (nothing filled) or `refused` (kill switch on, or no fresh book at submit time).
- **D4, limit price.** Walk the side's levels best-first. The limit is the last level price `p` for which `q − p − marginal_fee(p) ≥ min_edge`.
  - `q` is the clamped fair probability of the side.
  - `marginal_fee(p) = rate·p·(1−p)` for `kalshi_quadratic`, and `0` for `zero`.
  - The expression decreases in `p`, so stop at the first level that fails.
  - If no level passes, the decision is rejected with `edge_below_min`.
- **D5, Kelly cost.** The cost is `c = avg_price + fee_cents / (100·qty)`, taken from that side's probe `EdgeQuote`. Then `f* = (q − c)/(1 − c)`; reject if `f* ≤ 0`.
- **D6, money rounding.** A fill's cost in cents is `ceil(Σ price·take·100)`. The fee is `fill_fee_cents` (3a R4, max of order-level and per-level). A win pays `qty·100` cents.
- **D7, ET day.** The kill-switch "day" is the America/New_York calendar day of the mark's `asof`.
- **D8, idempotency.**
  - `risk_decisions` is unique on `(book_id, run_key, slot, contract_id)`. Re-running `evaluate` for the same run and slot returns the stored decisions and approves nothing new.
  - `orders.client_order_id = sha256(f"{book_id}|{run_key}|{slot}|{contract_id}")` is unique, and `orders.risk_decision_id` is `NOT NULL UNIQUE` with a foreign key.
- **D9, run key.** `run_key: str` is an opaque caller-supplied id. Plan 5 maps it to `pipeline_runs`.
- **D10, blocked event.** A "blocked event" (U2, incomplete bracket) is the Polymarket `venue_event_id`. Kalshi contracts on the same release still trade.

### Coverage
- **100% on `prediction_desk/risk/` and `prediction_desk/execution/`**, added to CI as `coverage report --include` lines next to `audit/` and `pricing/`.
  - The spec requires `risk/`; `execution/` is added because it moves money.
- `pricing/` and `audit/` stay at 100%.
- Overall coverage stays at or above 85%.

### Versions
- `RISK_ENGINE_VERSION = "risk:v1"` is stored on every decision.
- `PRICING_VERSION` is unchanged; Task 3 does not change any price.

## Review Focus

1. **Equity at or below the caps' needs.** After losses, `0.02·equity − held` can be negative. Every negative headroom must produce `below_min_size`, never a negative or zero-qty order. *(Test: Task 4)*
2. **Cost at or above the clamped fair.** At a price of 0.99, or a fee that eats the edge, `c ≥ q` must reject with `edge_below_min` before any division. `c = 1` must never divide by zero. *(Test: Task 4)*
3. **A crash between approve, submit and re-run.** Calling `evaluate` twice for the same `(run_key, slot)`, or `submit` twice for the same `ApprovedOrder`, must produce exactly one decision set and one order. *(Tests: Tasks 6, 7)*
4. **The kill switch flipping between approve and submit.** Once a trip commits, no fill may follow. `submit` re-checks under the book lock and records `refused`. *(Tests: Tasks 7, 10)*
5. **An outcome missing for one measure.** Example: core CPI is not yet in FRED. Positions on that measure stay open and settle later; nothing settles on a guess, and positions on the other measures settle normally. *(Test: Task 9)*

---

## File Structure

```
backend/src/prediction_desk/
  db/uow.py                      Task 1: unit_of_work (READ COMMITTED, refuses AUTOCOMMIT)
  db/models.py                   Task 2: + RiskLimitsRow, BookState, RiskDecision, Order, Fill,
                                         Position, MeasureOutcome, Settlement
  domain/enums.py                Task 2: + BookMode, Slot, Verdict, OrderStatus, PositionSide
  risk/__init__.py               Task 2
  risk/limits.py                 Task 2: RiskLimits, load_active_limits
  risk/sizing.py                 Task 4: clamp_fair, kelly_fraction, limit_price, size_order
  risk/book.py                   Task 5: create_book, lock_book, exposures, mark_book
  risk/kill_switch.py            Task 5: trip_kill_switch, reset_kill_switch
  risk/engine.py                 Task 6: RiskEngine, ApprovedOrder, RiskOfficer port
  execution/__init__.py          Task 7
  execution/broker.py            Task 7: Broker port, PaperBroker, LiveBroker stub
  execution/outcomes.py          Task 8: official values from FRED first print, admin override
  execution/settlement.py        Task 9: settle_release
  pricing/coherence.py           Task 3: CoherenceViolation (structured)
  pricing/release.py             Task 3: coherence becomes tuple[CoherenceViolation, ...]
  ingestion/fred.py              Task 8: + "DFEDTARU" in the FRED series list
backend/migrations/versions/0005_risk_and_execution.py   Task 2
backend/tests/
  desk_fixtures.py               Task 2: shared seeding helpers (release, contract, snapshot, book)
  db/test_uow.py  risk/test_limits.py  risk/test_sizing.py  risk/test_book_state.py
  risk/test_kill_switch.py  risk/test_risk_engine.py  execution/test_paper_broker.py
  execution/test_outcomes.py  execution/test_settlement.py  pricing/test_coherence_structured.py
  test_desk_end_to_end.py        Task 10
```

---

### Task 1: Unit of work

**Files:**
- Create: `backend/src/prediction_desk/db/uow.py`
- Modify: `backend/tests/audit/test_log.py` (`test_rolled_back_append_leaves_no_row_and_gap_still_verifies`)
- Test: `backend/tests/db/test_uow.py`

**Interfaces:**
- Produces: `unit_of_work(sessions: async_sessionmaker[AsyncSession]) -> AbstractAsyncContextManager[AsyncSession]`. It opens a session and begins a transaction. Before yielding, it runs `SET TRANSACTION ISOLATION LEVEL READ COMMITTED` and then checks `current_setting('transaction_isolation') == 'read committed'`. It commits on a clean exit, rolls back on any exception and re-raises, and always closes the session. It raises `RuntimeError("unit_of_work refuses an AUTOCOMMIT engine")` if the bound engine's `isolation_level` execution option is `AUTOCOMMIT` (Plan 1 R14).

- [ ] **Step 1: Write the failing tests** in `tests/db/test_uow.py`:
  - `test_commits_on_clean_exit`: a `Release` added inside is visible from a fresh session afterward.
  - `test_rolls_back_and_reraises`: `ValueError` raised inside propagates, and the row count is unchanged.
  - `test_runs_read_committed_even_if_engine_default_differs`: build an engine with `execution_options(isolation_level="REPEATABLE READ")`; inside the unit of work, `current_setting('transaction_isolation') == "read committed"`.
  - `test_refuses_autocommit_engine`: an engine with `isolation_level="AUTOCOMMIT"` raises the `RuntimeError` above before yielding.
  - `test_audit_append_works_inside_uow`: `AuditLog.append` succeeds inside, and the chain verifies.
- [ ] **Step 2: Run** `uv run pytest tests/db/test_uow.py -v` and confirm it fails with an ImportError.
- [ ] **Step 3: Implement `unit_of_work`** with `contextlib.asynccontextmanager`.
- [ ] **Step 4: Tighten the Plan 1 R15 test.** In `test_rolled_back_append_leaves_no_row_and_gap_still_verifies`, assert that `SELECT count(*) FROM audit_events` is `2` *before* `session.rollback()` and `1` immediately after. This proves the rollback removed a row that really existed.
- [ ] **Step 5: Run** `uv run pytest tests/db/test_uow.py tests/audit -v` and confirm PASS.
- [ ] **Step 6: Commit:** `feat(db): unit_of_work helper (READ COMMITTED, refuses AUTOCOMMIT)`.

---

### Task 2: Schema, models, risk limits

**Files:**
- Create: `backend/migrations/versions/0005_risk_and_execution.py`, `backend/src/prediction_desk/risk/__init__.py`, `backend/src/prediction_desk/risk/limits.py`, `backend/tests/desk_fixtures.py`
- Modify: `backend/src/prediction_desk/db/models.py`, `backend/src/prediction_desk/domain/enums.py`, `.github/workflows/ci.yml`
- Test: `backend/tests/risk/test_limits.py`, `backend/tests/db/test_risk_schema.py`

**Interfaces:**
- Produces enums (StrEnum):
  - `BookMode{LIVE="live", REPLAY="replay"}`
  - `Slot{T24H="t-24h", T2H="t-2h", T15M="t-15m"}`
  - `Verdict{APPROVED="approved", REJECTED="rejected"}`
  - `OrderStatus{FILLED="filled", PARTIAL="partial", CANCELLED="cancelled", REFUSED="refused"}`
  - `PositionSide{YES="yes", NO="no"}`
- Produces tables and ORM models. Columns are exactly these; every timestamp is `UTCDateTime`.
  - `risk_limits` (`RiskLimitsRow`):
    - Columns: `version int PK`, `params JSONB`, `created_by text`, `created_at`, `active bool`.
    - A partial unique index on `active` WHERE `active`.
    - Seeded with version `1`, `active=true`, `created_by="migration:0005"`, and params from `DEFAULT_LIMITS` below.
  - `book_state` (`BookState`):
    - Columns:
      - Identity and mode: `id Identity PK`, `mode book_mode`, `replay_batch_id bigint NULL`.
      - Money, all `bigint`: `cash_cents`, `equity_cents`, `peak_equity_cents`, `day_start_equity_cents`.
      - `day_start_date date`.
      - Kill switch: `kill_switch_on bool`, `kill_reason text NULL`, `kill_actor text NULL`, `kill_at NULL`.
      - Audit timestamps: `created_at`, `updated_at`.
    - `UNIQUE NULLS NOT DISTINCT (mode, replay_batch_id)`.
    - CHECK `mode <> 'live' OR replay_batch_id IS NULL`.
  - `risk_decisions` (`RiskDecision`):
    - Identity and keys:
      - `id Identity PK`
      - `book_id FK`
      - `run_key text`
      - `slot slot`
      - `release_id FK`
      - `contract_id FK`
      - `side position_side NULL`
    - Inputs:
      - `inputs_hash char(64)`
      - `engine_version text`
      - `fair_yes NUMERIC(6,5)`
      - `fair_clamped NUMERIC(6,5) NULL` (of the chosen side)
      - `cost_per_contract NUMERIC(12,8) NULL`
      - `limit_price NUMERIC(6,5) NULL`
    - Stakes, all `bigint NULL`: `kelly_raw_cents`, `stake_engine_cents`, `stake_officer_cents`, `stake_final_cents`.
    - Outcome:
      - `qty int NOT NULL DEFAULT 0`
      - `verdict verdict`
      - `reasons JSONB` (an ordered list)
    - Provenance:
      - `limits_version int FK risk_limits`
      - `limits_snapshot JSONB`
      - `decided_at`
    - `UNIQUE (book_id, run_key, slot, contract_id)`.
    - CHECK `verdict <> 'approved' OR (side IS NOT NULL AND qty >= 1 AND limit_price IS NOT NULL)`.
  - `orders` (`Order`):
    - Columns:
      - Keys: `id Identity PK`, `client_order_id char(64) UNIQUE`, `risk_decision_id bigint NOT NULL UNIQUE FK`, `book_id FK`, `contract_id FK`.
      - Order terms: `side`, `limit_price NUMERIC(6,5)`, `qty int`, `filled_qty int`, `status order_status`, `status_reason text NULL`.
      - `created_at`.
    - CHECK `0 <= filled_qty AND filled_qty <= qty`.
  - `fills` (`Fill`): `id Identity PK`, `order_id FK UNIQUE`, `qty int`, `avg_price NUMERIC(12,8)`, `cost_cents bigint`, `fees_cents bigint`, `slippage_cents bigint`, `snapshot_id bigint FK`, `ts`. There is one row per order (D3).
  - `positions` (`Position`):
    - Columns:
      - Keys: `id Identity PK`, `book_id FK`, `contract_id FK`, `side`.
      - Size and money: `qty int`, `cost_cents bigint` (price plus fees), `realized_pnl_cents bigint DEFAULT 0`.
      - Mark and state: `mark_price NUMERIC(6,5) NULL`, `settled bool DEFAULT false`, `updated_at`.
    - `UNIQUE (book_id, contract_id)`.
  - `measure_outcomes` (`MeasureOutcome`):
    - Columns:
      - Key and values: `release_id FK`, `measure`, `raw_value NUMERIC(20,8)`, `reported_value NUMERIC(14,4)`.
      - Provenance: `source text` (`fred_first_print` or `admin_override`), `source_vintage_ts NULL`, `recorded_by text`, `reason text NULL`, `recorded_at`.
    - PK `(release_id, measure)`.
  - `settlements` (`Settlement`):
    - Columns:
      - Keys and size: `id Identity PK`, `book_id FK`, `contract_id FK`, `side`, `qty int`.
      - Result: `yes_won bool`, `payout_cents bigint`, `realized_pnl_cents bigint`, `reported_value NUMERIC(14,4)`.
      - Venue comparison: `venue_yes_won bool NULL`, `mismatch bool`.
      - `settled_at`.
    - `UNIQUE (book_id, contract_id)`.
- Produces `risk/limits.py`:
  - `DEFAULT_LIMITS: Mapping[str, str | int]` holds these exact values:
    - Sizing and caps: `kelly_multiplier "0.25"`, `cap_contract "0.02"`, `cap_event "0.05"`, `cap_portfolio "0.15"`, `liquidity_fraction "0.25"`.
    - Edge and freshness: `min_edge "0.04"`, `min_mapping_confidence "0.90"`, `max_quote_age_s 60`, `cutoff_minutes 5`.
    - Clamp: `fair_clamp_lo "0.02"`, `fair_clamp_hi "0.98"`.
    - Kill-switch triggers: `daily_loss_kill "0.08"`, `drawdown_kill "0.15"`.
  - `@dataclass(frozen=True) RiskLimits` has fields of the same names (Decimal or int) plus `version: int`. `RiskLimits.from_params(version, params)` validates as follows:
    - every key is present and there are no extras;
    - the Decimal strings are finite;
    - every fraction is in `(0, 1]`;
    - `fair_clamp_lo < 0.5 < fair_clamp_hi`;
    - otherwise it raises `ValueError`.
  - `snapshot() -> dict[str, str | int]` is the JSON-safe form.
  - `async load_active_limits(session) -> RiskLimits` raises `LookupError` if there is no active row.
- Produces `tests/desk_fixtures.py`, async helpers modelled on `tests/pricing/test_price_release.py`:
  - `make_release(session, series=Series.CPI, period="2026-10", scheduled_at=…)`
  - `add_contract(session, release, *, venue, measure, comparator, strike_lo, strike_hi=None, venue_event_id="ev-1", confidence=Decimal("0.95")) -> EventContract`
  - `add_snapshot(session, contract, ts, bids, asks) -> MarketSnapshot`
  - `make_book(session, *, mode=BookMode.REPLAY, batch=1, cash_cents=1_000_000) -> BookState`

- [ ] **Step 1: Write the failing tests:**
  - `tests/risk/test_limits.py`:
    - `test_seeded_v1_matches_spec_defaults`: `load_active_limits` gives `cap_contract == Decimal("0.02")`, `min_edge == Decimal("0.04")`, `kelly_multiplier == Decimal("0.25")` and `version == 1`.
    - `test_from_params_rejects_missing_extra_nonfinite_out_of_range` (parametrized).
    - `test_only_one_active_version` (inserting a second `active=true` row raises `IntegrityError`).
    - `test_snapshot_is_json_safe` (`canonical_json(limits.snapshot())` does not raise).
  - `tests/db/test_risk_schema.py`:
    - `test_order_requires_risk_decision`: an `INSERT` into `orders` with `risk_decision_id NULL` raises `IntegrityError`, and so does one pointing at a nonexistent decision.
    - `test_one_order_per_decision`.
    - `test_approved_decision_needs_side_qty_limit`.
    - `test_live_book_has_no_batch_and_is_unique`.
- [ ] **Step 2: Run them** and confirm they fail.
- [ ] **Step 3: Write migration `0005`** (`down_revision="0004"`). Create the enum types with `postgresql.ENUM(..., create_type=False)` plus explicit `.create(checkfirst=True)`, as in `0003`. Seed the `risk_limits` row with `op.bulk_insert`. Write a full `downgrade()`.
- [ ] **Step 4: Add the ORM models and enums.** They must match the migration exactly, so the drift test passes.
- [ ] **Step 5: Implement `risk/limits.py`** and `tests/desk_fixtures.py`.
- [ ] **Step 6: Add the CI gate** line `uv run coverage report --include='*/prediction_desk/risk/*' --fail-under=100` after the pricing gate.
- [ ] **Step 7: Run** `uv run pytest tests/risk tests/db -v`, then the full suite, and confirm PASS with the drift test green.
- [ ] **Step 8: Commit:** `feat(risk): schema for limits, book, decisions, orders, fills, positions, outcomes, settlements`.

---

### Task 3: Structured coherence

**Files:**
- Create: `backend/src/prediction_desk/pricing/coherence.py`
- Modify: `backend/src/prediction_desk/pricing/release.py` (`_coherence`, `ReleasePricing.coherence`), plus every test that asserts on coherence strings
- Test: `backend/tests/pricing/test_coherence_structured.py`

**Interfaces:**
- Produces:
  - `CoherenceKind = Literal["monotonic", "bracket_sum", "incomplete_bracket"]`.
  - `@dataclass(frozen=True, order=True) CoherenceViolation(kind: CoherenceKind, scope: str, message: str)`, where `scope` is `venue.value` for `monotonic` and the Polymarket `venue_event_id` for the bracket kinds.
  - `is_bug(v) -> bool`, which is true for `monotonic` and `bracket_sum` (U2).
  - `ReleasePricing.coherence: tuple[CoherenceViolation, ...]`, sorted.
  - Messages keep the 3a text minus the prefix.
- Consumes: the existing `monotonic_violations` and `bracket_sum_violation`.

- [ ] **Step 1: Write the failing tests:**
  - `test_monotonic_violation_is_structured_and_bug`: two Kalshi GT contracts priced inversely give one violation with `kind == "monotonic"`, `scope == "kalshi"` and `is_bug` true.
  - `test_complete_bracket_bad_sum_is_bug`.
  - `test_incomplete_bracket_is_not_bug_and_scoped_to_event`: `scope == "ev-1"` and the message contains `"(2 of 3 mapped)"`.
  - `test_inputs_hash_unchanged_by_coherence_refactor`: a fixed seeded scenario keeps the same `inputs_hash` as before this task. Record the literal value from `main` before changing code.
- [ ] **Step 2: Run them** and confirm they fail.
- [ ] **Step 3: Implement the change.** Update existing assertions in `tests/pricing/` to the structured form; do not weaken them.
- [ ] **Step 4: Run** `uv run pytest tests/pricing -v` and the `pricing/` coverage gate, and confirm PASS at 100%.
- [ ] **Step 5: Commit:** `refactor(pricing): structured coherence violations for the risk layer`.

---

### Task 4: Sizing math (pure)

**Files:**
- Create: `backend/src/prediction_desk/risk/sizing.py`
- Test: `backend/tests/risk/test_sizing.py`

**Interfaces:**
- Consumes: `RiskLimits` (Task 2); `BookLevel` (`markets.types`); `FeeSchedule` (`pricing.fees`).
- Produces:
  - `clamp_fair(p: Prob, lo: Decimal, hi: Decimal) -> Prob`.
  - `kelly_fraction(q: Decimal, c: Decimal) -> Decimal`:
    - Returns `(q − c)/(1 − c)`, or `Decimal(0)` when `q <= c`.
    - Raises `ValueError` unless `0 < c < 1` and `0 < q < 1`.
  - `limit_price(levels: Sequence[BookLevel], q: Decimal, min_edge: Decimal, schedule: FeeSchedule) -> Decimal | None`: implements D4.
  - `depth_at_or_better(levels, limit: Decimal) -> Decimal`.
  - `@dataclass(frozen=True) SizingInputs` with these fields:
    - Equity and edge: `equity_cents: int`, `q: Decimal`, `c: Decimal`.
    - Held exposure: `held_contract_cents: int`, `held_event_cents: int`, `held_portfolio_cents: int`.
    - Book: `depth: Decimal`, `limit: Decimal`.
  - `@dataclass(frozen=True) Sizing` with these fields:
    - Stakes in cents: `kelly_raw_cents: int`, `stake_cents: int`.
    - `qty: int`.
    - `reasons: tuple[dict[str, str], ...]`.
    - `rejected: str | None`.
  - `size_order(inputs: SizingInputs, limits: RiskLimits) -> Sizing`: the §7.2 funnel steps 2–7. Each step appends `{"step": <name>, "cents": str(value)}`, with names `kelly`, `cap_contract`, `cap_event`, `cap_portfolio`, `liquidity`, `round`.
    - **Kelly:** `kelly_raw = kelly_multiplier · f* · equity`.
    - **Caps:** each cap is `min(prev, frac·equity − held)`, floored at 0.
    - **Liquidity:** cap `qty ≤ floor(liquidity_fraction · depth)`.
    - **Round:** `qty = min(floor(stake / (c·100)), liq_qty)`.
    - `stake_cents = ceil(qty·c·100)`.
    - `rejected="below_min_size"` when `qty < 1`; `rejected="edge_below_min"` when `f* == 0`.

- [ ] **Step 1: Write the failing tests:**
  - `test_kelly_fraction_examples`:
    - `q=0.60, c=0.50 → 0.2`.
    - `q=0.50, c=0.50 → 0`.
    - `c=1` and `c=0` raise.
  - `test_clamp_fair_bounds`:
    - `0 → 0.02`.
    - `1 → 0.98`.
    - `0.5 → 0.5`.
  - `test_limit_price_stops_at_first_failing_level`: asks `[(0.50,10),(0.53,10),(0.555,10),(0.58,10)]`, `q=0.60`, `min_edge=0.04`.
    - `zero` schedule → `0.555`, because `0.60−0.555 = 0.045` passes and `0.60−0.58 = 0.02` fails.
    - Kalshi rate `0.07` → `0.53`, because `0.07 − 0.07·0.53·0.47 = 0.052563` passes and `0.045 − 0.07·0.555·0.445 = 0.027712` fails.
  - `test_limit_none_when_best_level_fails`.
  - `test_size_order_worked_example`: equity 1_000_000, `q=0.60`, `c=0.50`, held 0, depth 1000, limit 0.53.
    - `kelly_raw = 0.25·0.2·1_000_000 = 50_000`.
    - `cap_contract = 20_000`.
    - `qty = floor(20_000/50) = 400`.
    - `liquidity = floor(0.25·1000) = 250`.
    - Result: `qty == 250`, `stake_cents == 12_500`.
    - The reasons steps appear in order.
  - `test_negative_headroom_rejects_below_min_size` (Review Focus 1): `held_contract_cents > 0.02·equity`.
  - `test_zero_kelly_rejects_edge_below_min` (Review Focus 2).
  - Hypothesis `test_stake_never_exceeds_any_cap`: for random valid inputs:
    - `stake_cents ≤ kelly_raw_cents + 1`.
    - For each cap, `stake_cents ≤ max(0, frac·equity − held) + 1`.
    - `qty ≤ floor(0.25·depth)`.
    - `qty ≥ 0`.
    - (`qty` is floored from the headroom, so `qty·c·100 ≤ headroom`. The `+1` is only the `ceil` to whole cents.)
- [ ] **Step 2: Run them** and confirm they fail.
- [ ] **Step 3: Implement.** Use exact Decimal throughout, with no floats. Round down with `ROUND_FLOOR` and up with `ROUND_CEILING`.
- [ ] **Step 4: Run** `uv run pytest tests/risk/test_sizing.py -v` and confirm PASS.
- [ ] **Step 5: Commit:** `feat(risk): R14 clamp, binary Kelly, limit price and the cap funnel`.

---

### Task 5: Book state, exposure, mark-to-market, kill switch

**Files:**
- Create: `backend/src/prediction_desk/risk/book.py`, `backend/src/prediction_desk/risk/kill_switch.py`
- Test: `backend/tests/risk/test_book_state.py`, `backend/tests/risk/test_kill_switch.py`

**Interfaces:**
- Consumes: `unit_of_work` (Task 1); models (Task 2); `latest_snapshot_asof`; `AuditLog`.
- Produces in `risk/book.py`. Every function runs on the caller's session; the caller owns the transaction.
  - `async create_book(session, audit, clock, *, mode: BookMode, replay_batch_id: int | None, cash_cents: int = 1_000_000) -> BookState`:
    - Sets all equity fields to `cash_cents`.
    - Sets `day_start_date` to the ET date of `clock.now()`.
    - Audits `book.created`.
  - `async lock_book(session, book_id) -> BookState`: `SELECT … FOR UPDATE`; raises `LookupError` if the book is missing.
  - `@dataclass(frozen=True) Exposure(contract: Mapping[int,int], event: Mapping[int,int], portfolio: int)`:
    - Cents keyed by contract id and by release id.
    - Counts open positions' `cost_cents` plus approved decisions with no order (`stake_final_cents`) (D1).
  - `async exposure(session, book_id) -> Exposure`.
  - `async mark_book(session, audit, clock, book_id, asof) -> BookState`. It first calls `lock_book` and `load_active_limits`, then:
    - **Mark each open position.** Use the mid of the latest snapshot ≤ `asof` when both sides exist. Otherwise keep the previous `mark_price`. Otherwise use `cost_cents/(qty·100)`, quantized to 5 dp.
    - **Equity.** `equity = cash + Σ round_half_even(qty·mark_side·100)`, where `mark_side = mark` for YES and `1 − mark` for NO.
    - **Day roll (D7).** If the ET date of `asof` is after `day_start_date`, set `day_start_equity_cents` to the equity before this mark and move `day_start_date` forward.
    - **Peak.** `peak = max(peak, equity)`.
    - **Audit** `book.marked` only when equity changed.
    - **Auto-trip.** Trip `daily_loss` when `equity − day_start ≤ −daily_loss_kill·day_start`, and `drawdown` when `peak − equity ≥ drawdown_kill·peak`. Use actor `system:risk` and reasons `"daily_loss: …"` / `"drawdown: …"`.
- Produces in `risk/kill_switch.py`:
  - `async trip_kill_switch(session, audit, clock, book_id, *, reason: str, actor: str) -> bool`:
    - Locks the book.
    - Returns `False` with no write if the switch is already on.
    - Otherwise sets `on`, `reason`, `actor` and `kill_at`, and audits `kill_switch.tripped`.
    - An empty reason raises `ValueError`.
  - `async reset_kill_switch(session, audit, clock, book_id, *, reason: str, actor: str) -> None`:
    - Raises `PermissionError` unless `actor.startswith("admin:")`, and `ValueError` on an empty reason or when the switch is already off.
    - Audits `kill_switch.reset`.
    - Nothing calls it automatically (spec §7.4).

- [ ] **Step 1: Write the failing tests:**
  - `test_book_state.py`:
    - `test_create_book_sets_equity_and_audits`.
    - `test_exposure_counts_positions_and_unordered_approvals` (an approved decision with an order is not double counted).
    - `test_mark_uses_mid_then_previous_then_cost`.
    - `test_no_side_marks_at_one_minus_mid`.
    - `test_day_roll_resets_day_start_on_new_et_day`: marks at 03:59 UTC and 05:01 UTC on the same UTC date fall on different ET days in winter.
    - `test_daily_loss_8pct_trips`: day start 1_000_000 and equity 920_000 trips; 920_001 does not.
    - `test_drawdown_15pct_trips_from_peak`.
    - `test_unchanged_equity_writes_no_audit`.
  - `test_kill_switch.py`:
    - `test_trip_is_idempotent_single_audit_event`.
    - `test_reset_requires_admin_and_reason`.
    - `test_reset_never_automatic`: equity recovering does not reset.
    - `test_manual_trip_by_admin_records_actor`.
- [ ] **Step 2: Run them** and confirm they fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest tests/risk -v` and confirm PASS.
- [ ] **Step 5: Commit:** `feat(risk): book state, exposure, mark-to-market and the kill switch`.

---

### Task 6: RiskEngine and the structural gate

**Files:**
- Create: `backend/src/prediction_desk/risk/engine.py`
- Test: `backend/tests/risk/test_risk_engine.py`

**Interfaces:**
- Consumes: `ReleasePricing`/`ContractPricing`/`EdgeQuote`/`Unavailable` (3a); `CoherenceViolation`, `is_bug` (Task 3); `size_order`, `limit_price`, `depth_at_or_better`, `clamp_fair` (Task 4); `lock_book`, `exposure`, `trip_kill_switch` (Task 5); `unit_of_work` (Task 1); `side_levels` (`pricing.book`); `schedule_for`.
- Produces:
  - **`ApprovedOrder`.** A frozen dataclass:
    - Fields:
      - Identity: `risk_decision_id: int`, `client_order_id: str`, `book_id: int`, `contract_id: int`.
      - Order: `side: PositionSide`, `limit_price: Decimal`, `qty: int`, `decided_at: datetime`.
    - **Sealing.** `__post_init__` raises `TypeError` unless the private module token `_SEAL` is passed as the keyword-only field `_seal`. The only constructor is `engine.py`'s private `_approve(...)`.
  - **`SlateItem`.** A frozen dataclass with fields `contract_id`, `venue`, `side`, `qty_engine: int`, `stake_engine_cents: int`, `fair_clamped: Prob`, `cost_per_contract: Decimal` and `reasons`.
  - **`RiskOfficer`** is a Protocol: `async review(self, slate: Sequence[SlateItem]) -> Mapping[int, int]` (contract id → maximum qty).
  - **`EvaluationResult`.** A frozen dataclass: `decision_ids: tuple[int, ...]`, `approved: tuple[ApprovedOrder, ...]`, `killed: bool`.
  - **`RiskEngine(clock, audit, fee_schedules: Sequence[FeeSchedule] | None = None)`.** It takes the same `fee_schedules` that `price_release` takes. Its method:
    ```
    async evaluate(sessions, *, book_id, run_key, slot, pricing, healthy_venues: frozenset[Venue], officer: RiskOfficer | None) -> EvaluationResult
    ```
    It runs in three phases.
    - **Phase A, read.** In a short session, load:
      - the limits, the book, the release and the contracts;
      - the snapshot for each priced contract;
      - the exposure.
    - **Per-contract gates.** Apply them in this order; the first failure rejects the contract with that reason:
      1. `kill_switch_on`
      2. `stale_pricing`: `clock.now() − pricing.asof > max_quote_age_s`
      3. `past_cutoff`: `pricing.asof >= release.scheduled_at − cutoff_minutes`
      4. `low_mapping_confidence`
      5. `venue_unhealthy`
      6. `no_quote:<reason>`
      7. `event_incoherent`: an incomplete bracket on its `venue_event_id`
      8. `opposite_position`
      9. `edge_below_min`: clamped `q − c < min_edge`, comparing both sides and keeping the better passing one
    - **Which contracts get decisions.** Only contracts in `pricing.contracts` get a decision row. Rejected or unpriceable ones never reach risk.
    - **Gate inputs.**
      - `no_quote` applies when `pricing.quote` is `Unavailable`, and uses its reason.
      - For each available side, `q` is the clamped fair of that side and `c` comes from that side's probe `EdgeQuote` (D5).
      - The limit and depth come from that side's levels in the contract's snapshot book, with `schedule_for(venue, pricing.asof)`.
    - **Phase A, sizing.** Survivors are sized with `size_order` in contract-id order. Each approved stake is added to a running copy of the exposure before the next contract is sized, so contracts in one slate cannot jointly breach the per-event or portfolio cap.
    - **Phase B, officer, outside any transaction.**
      - `officer` is `None`:
        - On a `LIVE` book, everything is rejected `officer_unavailable`.
        - On a `REPLAY` book, the stake passes through, with a reason entry `{"step": "officer", "skipped": "replay"}`.
      - The officer raises, omits a slate contract id, or returns a negative or non-int qty: everything is rejected `officer_failed` (fail closed).
      - Otherwise, `qty_final = min(qty_engine, officer_qty)`. `0` means rejected `officer_veto`.
    - **Phase C, write, in `unit_of_work`.**
      - `lock_book`.
      - If any `is_bug` violation is present, `trip_kill_switch(reason="pricing_sanity: <msg>", actor="system:risk")` and reject every contract with `pricing_sanity_violation` (U2).
      - If the switch is now on, reject all with `kill_switch_on`.
      - Recompute the exposure, re-run `size_order` with the fresh holdings (accumulating in contract-id order, as in Phase A), and `min` the result with the Phase B qty.
      - Insert one `RiskDecision` per evaluated contract, rejected ones included, with `engine_version = RISK_ENGINE_VERSION`, the limits version and snapshot, and `pricing.inputs_hash`.
      - Audit `risk.decided` once per run and slot, with a payload listing the decision ids and verdicts.
      - Build `ApprovedOrder`s.
    - **Idempotency (D8).** If decisions already exist for `(book_id, run_key, slot)`, return them in a `unit_of_work` that takes `lock_book`. Rebuild the `ApprovedOrder`s only for approved decisions that have no order and only while the switch is off. Write nothing new.

- [ ] **Step 1: Write the failing tests.**
  - **Setup.** FakeOfficer classes live in the test file. The tests construct `ReleasePricing` directly instead of calling `price_release`. Contracts, snapshots and books are seeded with `tests/desk_fixtures.py`.
  - `test_approves_and_sizes_worked_example`, which uses the Task 4 numbers:
    - One Kalshi GT contract with `fair_yes=0.60`.
    - A YES probe `EdgeQuote(avg_price=0.50, fee_cents=0, qty=10)`.
    - A stored snapshot with asks `[(0.50,400),(0.53,600)]`, so 1000 contracts sit at or below the limit.
    - A zero fee schedule.
    - It asserts one `ApprovedOrder` with `qty == 250`, a decision row with `verdict="approved"` and the reasons steps in order, and one `risk.decided` audit event.
  - `test_each_gate_rejects_with_its_reason`: parametrized over the 9 gates, asserting the exact reason string and that no `ApprovedOrder` is produced.
  - `test_r14_clamp_prevents_cap_size_on_zero_prob_outcome`: FOMC fair 0.00000 with NO asks at 0.97 gives a clamped `q=0.98` and `edge 0.01 < 0.04`, so the result is rejected.
  - `test_bug_violation_trips_kill_switch_and_rejects_all`.
  - `test_incomplete_bracket_blocks_only_that_event`: the Kalshi contract on the same release is still approved (D10).
  - `test_live_book_without_officer_rejects_all`.
  - `test_officer_exception_or_missing_id_fails_closed`.
  - `test_officer_can_only_reduce`: the officer returns more than the engine and the final qty equals the engine's qty.
  - `test_no_connection_held_during_officer_review`: the fake officer asserts `engine.pool.checkedout() == 0`.
  - `test_rerun_same_run_slot_is_idempotent` (Review Focus 3): two calls give one set of rows, the same `ApprovedOrder` ids, and one audit event.
  - `test_caps_recomputed_in_write_phase`: a position inserted between Phase A and Phase C, via a fake officer callback, shrinks the qty.
  - `test_approved_order_cannot_be_constructed_outside_engine` (`TypeError`).
  - `test_top_up_counts_held_position` (U3).
  - `test_slate_accumulates_exposure_within_run`: three contracts on one release, each worth the full 2% contract cap, approve a total of at most 5% of equity. The third is downsized or rejected `below_min_size`.
- [ ] **Step 2: Run them** and confirm they fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest tests/risk -v` and the `risk/` 100% gate, and confirm PASS.
- [ ] **Step 5: Commit:** `feat(risk): RiskEngine gates, funnel, fail-closed officer port and sealed ApprovedOrder`.

---

### Task 7: Paper broker

**Files:**
- Create: `backend/src/prediction_desk/execution/__init__.py`, `backend/src/prediction_desk/execution/broker.py`
- Modify: `.github/workflows/ci.yml` (add the `execution/` 100% gate)
- Test: `backend/tests/execution/test_paper_broker.py`

**Interfaces:**
- Consumes: `ApprovedOrder` (Task 6); `lock_book` (Task 5); `latest_snapshot_asof`, `OrderBook.from_json`, `side_levels`, `fill_fee_cents`, `schedule_for`; `load_active_limits`.
- Produces:
  - `@dataclass(frozen=True) OrderResult(order_id: int, status: OrderStatus, filled_qty: int, cost_cents: int, fees_cents: int)`.
  - `Broker` Protocol: `async submit(self, sessions, approved: ApprovedOrder) -> OrderResult`.
  - `LiveBroker.__init__` raises `RuntimeError("live trading is disabled (TRADING_MODE=paper)")`.
  - `PaperBroker(clock, audit, fee_schedules=None).submit(sessions, approved)`. In one `unit_of_work`, it does the following:
    - **Lock and dedupe.** `lock_book`. If an order with this `client_order_id` exists, return its `OrderResult` unchanged (idempotent).
    - **Refuse if unsafe.** If the kill switch is on, insert the order `status="refused", status_reason="kill_switch_on"`. If there is no snapshot ≤ `clock.now()` within `max_quote_age_s`, refuse with `"no_fresh_book"`.
    - **Fill.** Walk the side levels with `price <= limit_price` up to `qty`.
      - `cost_cents = ceil(Σ price·take·100)`
      - `fees_cents = fill_fee_cents(...)`
      - `slippage_cents = round_half_up((avg − best)·qty·100)`
    - **Write.** Insert the `Order`, a `Fill` (only if `filled_qty ≥ 1`), and the position upsert (`qty += filled`, `cost_cents += cost + fees`). Then `cash_cents −= cost + fees`.
    - **Status.** `filled` / `partial` / `cancelled` per D3.
    - **Audit** `order.submitted` with the full payload.
    - **Guard.** Assert the post-trade `cash_cents >= 0`, and refuse with `"insufficient_cash"` otherwise.

- [ ] **Step 1: Write the failing tests:**
  - `test_fills_full_qty_at_or_below_limit_and_moves_cash`: asks `[(0.50,100),(0.53,200)]`, YES qty 250, limit 0.53, Kalshi rate 0.07.
    - `cost_cents == 12_950`, i.e. `5_000 + 7_950`.
    - `fees_cents == 437`. This is the order-level `ceil(436.93)`, which equals the per-level `175 + 262`.
    - `slippage_cents == 450`, i.e. `(0.518 − 0.50)·250·100`.
    - `cash_cents == 986_613`.
    - The position has `cost_cents == 13_387`.
  - `test_partial_when_depth_at_limit_insufficient`.
  - `test_cancelled_when_nothing_at_or_below_limit`.
  - `test_refused_when_kill_switch_flipped_after_approval` (Review Focus 4).
  - `test_refused_when_book_stale`.
  - `test_submit_twice_one_order` (Review Focus 3).
  - `test_no_side_fills_against_bids`.
  - `test_top_up_accumulates_position_cost`.
  - `test_live_broker_refuses_to_construct`.
  - `test_order_without_decision_impossible`: a raw insert raises (spec success criterion 4).
- [ ] **Step 2: Run them** and confirm they fail.
- [ ] **Step 3: Implement.** Add the CI line `uv run coverage report --include='*/prediction_desk/execution/*' --fail-under=100`.
- [ ] **Step 4: Run** `uv run pytest tests/execution -v` and confirm PASS.
- [ ] **Step 5: Commit:** `feat(execution): paper broker with IOC fills, idempotent client order ids`.

---

### Task 8: Official outcomes (U1)

**Files:**
- Create: `backend/src/prediction_desk/execution/outcomes.py`
- Modify: `backend/src/prediction_desk/ingestion/fred.py` (append `"DFEDTARU"` to the FRED series list)
- Test: `backend/tests/execution/test_outcomes.py`

**Interfaces:**
- Consumes: `first_vintage_ts`, `levels_asof`, `pct_change` (`forecasting.realtime`); `MEASURE_SPECS` (`markets.mapping`); `MacroDataProvider`; the synthetic seeders (`seed_synthetic_cpi/nfp/rates`).
- Produces:
  - `reported_value(measure: Measure, raw: Decimal) -> Decimal`:
    - Rounds `ROUND_HALF_UP` to the measure's grid (`MEASURE_SPECS` precision).
    - For `FOMC_CHANGE_BP`, snaps to the nearest 25 and clamps to `[−50, 50]`, matching the 3a outcome grid tails.
  - `contract_yes_won(terms: ContractTerms, reported: Decimal) -> bool`: the comparator applied to the reported value (`GT: R > s`, …, `BETWEEN: s_lo ≤ R ≤ s_hi`).
  - `async official_raw_value(session, provider, release, measure, asof) -> tuple[Decimal, datetime] | None`. It returns the raw value and the source vintage timestamp, or `None` when the data is not yet available:
    - **CPI_MOM / CORE_CPI_MOM:** `100·(L[m]/L[m−1] − 1)` on `CPIAUCSL` / `CPILFESL` levels as of the first vintage of the release month.
      - Compute it at full precision under a pinned 28-digit context.
      - Do **not** use `pct_change`: it quantizes to 1e-6 first, and that double rounding can flip a value like `0.2499996` across the 0.25 boundary.
    - **CPI_YOY / CORE_CPI_YOY:** `100·(L[m]/L[m−12] − 1)` from that same vintage, also at full precision.
    - **NFP_CHANGE:** `(L[m] − L[m−1])·1000` jobs, from PAYEMS's first vintage of m.
    - **FOMC_CHANGE_BP:** `(DFEDTARU[d+1] − DFEDTARU[d])·100`, where `d = date.fromisoformat(release.period)`, using levels as of `asof`.
  - `async record_outcomes(sessions, provider, audit, clock, release_id) -> dict[Measure, Decimal]`:
    - Covers each measure of the release's series that is not already recorded.
    - Inserts a `MeasureOutcome(source="fred_first_print", recorded_by="system:outcomes")` and audits `outcome.recorded`.
    - Returns the reported values written.
    - Skips (and does not raise for) measures whose data is missing.
  - `async override_outcome(session, audit, clock, *, release_id, measure, raw_value: Decimal, actor: str, reason: str) -> None`:
    - Requires `actor.startswith("admin:")` and a non-empty reason.
    - Upserts with `source="admin_override"` and audits `outcome.overridden` with the old and new values.
    - Raises `ValueError` if any position on that release has already settled.

- [ ] **Step 1: Write the failing tests:**
  - `test_reported_value_half_up_on_grid`:
    - CPI `0.25 → 0.3`.
    - CPI `-0.05 → -0.1`. Decimal `ROUND_HALF_UP` rounds ties away from zero.
    - CPI `0.2499996 → 0.2`.
    - NFP `150_000 → 150_000`.
    - FOMC `-75 → -50`.
    - FOMC `30 → 25`.
  - `test_contract_yes_won_each_comparator` (parametrized).
  - `test_cpi_mom_from_first_print_matches_synthetic_truth`, using `seed_synthetic_cpi` and comparing against the seeder's own first-print MoM for that month.
  - `test_later_revision_does_not_change_outcome`: a second vintage of the same month with a different index leaves the result unchanged.
  - `test_nfp_change_in_jobs`.
  - `test_fomc_from_dfedtaru_step`: seed DFEDTARU 4.50 on d and 4.25 on d+1 → `−25`.
  - `test_missing_data_returns_none_and_record_skips` (Review Focus 5).
  - `test_override_requires_admin_and_audits_old_new`.
  - `test_override_after_settlement_refused`.
- [ ] **Step 2: Run them** and confirm they fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest tests/execution tests/ingestion -v` and confirm PASS. Ingestion tests that enumerate the FRED series list must be updated, not deleted.
- [ ] **Step 5: Commit:** `feat(execution): official measure outcomes from FRED first print with admin override`.

---

### Task 9: Settlement

**Files:**
- Create: `backend/src/prediction_desk/execution/settlement.py`
- Test: `backend/tests/execution/test_settlement.py`

**Interfaces:**
- Consumes: `MeasureOutcome` rows (Task 8); `contract_yes_won`; `terms_from_row`; `lock_book`, `mark_book` (Task 5).
- Produces:
  - `@dataclass(frozen=True) SettleResult(settled: tuple[int, ...], pending: tuple[int, ...], mismatches: tuple[int, ...])`, holding contract ids.
  - `async settle_release(sessions, audit, clock, *, book_id, release_id, venue_yes_won: Mapping[int, bool] | None = None) -> SettleResult`. In one `unit_of_work` it does the following:
    - **Lock.** `lock_book`.
    - **For each open, unsettled position on the release:**
      - **No outcome yet.** If its measure has no `MeasureOutcome`, the position is `pending`.
      - **Otherwise:**
        - **Result.** `yes_won = contract_yes_won(terms, reported)`. The position wins iff `(side == YES) == yes_won`. `payout = qty·100` if it wins, else 0, and `realized = payout − cost_cents`.
        - **Cash and position.** `cash += payout`. Set the position `settled=true`, `realized_pnl_cents=realized`, and `mark_price` to `1` if YES won, else `0`.
        - **Settlement row.** Insert a `Settlement`. `mismatch = venue_yes_won.get(cid) is not None and venue_yes_won[cid] != yes_won`.
        - **Audit** `position.settled`. A mismatch also audits `settlement.mismatch`.
    - **Re-mark.** Then call `mark_book(session, audit, clock, book_id, clock.now())`, which may trip the kill switch on the realized loss.
    - **Idempotent.** A settled position is skipped on re-run.

- [ ] **Step 1: Write the failing tests:**
  - `test_yes_winner_pays_100_per_contract_and_books_pnl`, with exact cents.
  - `test_no_position_wins_when_yes_loses`.
  - `test_missing_measure_outcome_leaves_position_pending` (Review Focus 5): a CPI release has MoM recorded and core MoM not. The MoM position settles and the core position is pending.
  - `test_mismatch_flagged_but_payout_follows_our_outcome` (U1).
  - `test_settle_twice_is_idempotent`.
  - `test_big_realized_loss_trips_drawdown_via_mark`.
  - `test_equity_after_settlement_equals_cash_when_flat`.
- [ ] **Step 2: Run them** and confirm they fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest tests/execution -v` and both 100% gates, and confirm PASS.
- [ ] **Step 5: Commit:** `feat(execution): settlement on reported value with venue mismatch flag`.

---

### Task 10: Concurrency and end to end

**Files:**
- Test: `backend/tests/test_desk_end_to_end.py`, `backend/tests/risk/test_kill_switch_race.py`

**Interfaces:**
- Consumes: everything above; `price_release` and `MODELS` (3a); `seed_synthetic_cpi`.

- [ ] **Step 1: Write `test_kill_switch_race.py`:**
  - `test_trip_waits_for_in_flight_decision`: session A takes `lock_book` and holds it. A task B calls `trip_kill_switch` in its own `unit_of_work` and is still pending after 0.3 s. A commits, B completes, and a following `evaluate` rejects everything `kill_switch_on`.
  - `test_concurrent_evaluations_cannot_jointly_breach_cap` (D2):
    - Insert risk-limits version 2 as active, with `cap_contract = cap_event = cap_portfolio = "0.15"`.
    - Run two `evaluate` calls on different releases concurrently with `asyncio.gather`. Each would alone size at the full 15% headroom.
    - The total approved `stake_final_cents` is `≤ 0.15·equity + 2`.
- [ ] **Step 2: Write `test_desk_end_to_end.py::test_cpi_release_price_risk_fill_mark_settle`.** It seeds synthetic CPI, a release, two Kalshi GT contracts and fresh books, then runs:
  1. `price_release`
  2. `RiskEngine.evaluate` with a pass-through FakeOfficer on a REPLAY book
  3. `PaperBroker.submit` for each approval
  4. `mark_book`
  5. `record_outcomes`
  6. `settle_release`

  It then asserts:
  - final `cash_cents == 1_000_000 + Σ realized_pnl_cents`;
  - every order has a decision;
  - `AuditLog.verify` is ok;
  - the audit actions appear in order: `book.created`, `risk.decided`, `order.submitted`…, `outcome.recorded`…, `position.settled`….
- [ ] **Step 3: Run** the full gates:
  - `uv run ruff check . && uv run ruff format --check . && uv run mypy src tests`
  - `uv run pytest --cov --cov-fail-under=85`
  - the four 100% coverage reports (`audit`, `pricing`, `risk`, `execution`)

  Confirm they are all green.
- [ ] **Step 4: Commit:** `test: kill-switch race, cap race and CPI end-to-end through settlement`.

---

## Definition of Done (Plan 3b)

- All gates are green locally and in CI, including the new 100% gates on `risk/` and `execution/`.
- No code path creates an order without a `risk_decisions` row: the `ApprovedOrder` seal, the `NOT NULL UNIQUE` foreign key, and the decision transaction all enforce it.
- Given a `ReleasePricing`, the desk sizes, fills, marks and settles a release end to end, and the audit chain verifies.
- The kill switch trips on daily loss, on drawdown, and on bug-type pricing violations. Nothing resets it except an admin with a reason.

## Not in this plan (owners)

- **Plan 4:** the LLM Risk Officer (it implements `RiskOfficer`).
- **Plan 5:**
  - Persisting forecasts and edges, `pipeline_runs` (and mapping `run_key` to it), scheduling the T−24h/T−2h/T−15m slots, and fresh re-quotes before `evaluate`.
  - Venue settled-market fetch for `venue_yes_won`, bitemporal contracts, and the release-change audit event (Plan 2 carry-forward).
  - Data-QA and audit-chain-break kill triggers. These callers use `trip_kill_switch`.
- **Plan 6:** the admin endpoints for the kill switch, risk-limit versions and outcome overrides.
- **Optional hardening:** a 3a item, not done here — grid contiguity of EQ strikes within a bracket.

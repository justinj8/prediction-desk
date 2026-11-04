# Prediction Desk — Design Spec

- **Status:** Draft for review
- **Date:** 2026-11-04
- **Author:** Justin John (with Claude)
- **Repo:** `agentic_prediction_trader` · Python package `prediction_desk` · Railway project `prediction-desk`

---

## 1. Purpose and success criteria

Prediction Desk is an **agentic paper-trading system for regulated prediction-market contracts on US macroeconomic releases**: CPI, Nonfarm Payrolls (NFP) and FOMC rate decisions. A multi-agent forecasting desk builds probability distributions from point-in-time macro data. Those distributions are compared against Kalshi and Polymarket prices to find edge. A deterministic risk engine, with institutional-style controls, decides whether and how much to trade. Every decision is recorded in a tamper-evident audit trail and scored in a post-event review.

It is a resume and portfolio piece. Its story is **agentic trading with capital preservation, risk controls and auditability**. It is deliberately separate from the user's sports system (Sharpline) and from the broader `kalshi_predictive_agentic_system`.

**Success criteria**
1. A reviewer can open the public dashboard and follow one release end to end: base forecast → agent adjustments → edge per contract → risk funnel → paper fill → settlement → post-event review.
2. Any trade can be reproduced and explained from stored data: inputs hash, model versions, the limits snapshot and the LLM transcripts.
3. A historical replay produces a **clean** (no look-ahead) track record that the dashboard shows.
4. No code path can place an order without a risk approval. This is enforced by types, by a database foreign key and by transactions.
5. A production-quality repo: CI gates, typed end to end, tests at the levels set out in §10, a README and architecture docs.

**Non-goals (v1):** live money trading, other macro series (GDP, PCE, unemployment rate), mobile app, multi-user portfolios, human trade approval.

---

## 2. Decisions

| Area | Decision |
|---|---|
| Venues | **Kalshi + Polymarket** behind a `VenueAdapter` port, with a canonical `EventContract` model |
| Demo mode | **Live paper trading + historical replay** with point-in-time (ALFRED vintage) data |
| Releases | **CPI, NFP, FOMC** |
| LLM role | Quant base model + **bounded** LLM adjustment; deterministic risk is final authority |
| Agents | Multi-agent: Contract Mapper; parallel specialist analysts → Lead Forecaster → Skeptic; Risk Officer (veto/downsize only, fails closed); Review agent |
| Orchestration | Explicit typed pipeline / durable state machine in Postgres; agents via the Anthropic Python SDK (no LangGraph) |
| Architecture | **Modular monolith**, hexagonal ports and adapters, two processes (`api`, `worker`) |
| Backend | Python 3.12, FastAPI, SQLAlchemy 2.0, Alembic, Postgres 16, `uv` |
| Frontend | Next.js 15 (App Router), TypeScript strict, Tailwind + shadcn/ui, TanStack Query, ECharts |
| Hosting | **Railway** (Postgres, api, worker) + **Vercel** (web). **No Docker** anywhere |
| Access | Public read-only; admin (GitHub OAuth allowlist) for control actions |
| Autonomy | Fully autonomous paper execution |
| Risk defaults | ¼-Kelly; caps 2% per contract / 5% per event / 15% gross portfolio; min edge 4¢ after costs; kill switch at −8% daily or −15% drawdown from peak; $10,000 paper bankroll |
| Replay and LLM | Before the LLM's training cutoff, replay runs **base model only** (clean). LLM attribution is measured on live and post-cutoff events only |
| Tests DB | Local Homebrew Postgres 16; CI uses the runner's built-in Postgres |

---

## 3. Architecture

```
                    ┌───────────────────────── Vercel ─────────────────────────┐
                    │  Next.js (TS) — public read-only views + admin console   │
                    │  typed client generated from FastAPI OpenAPI schema      │
                    └───────────────┬──────────────────────────────────────────┘
                                    │ HTTPS (REST + SSE)
┌──────────────────────────────── Railway ─────────────────────────────────────────┐
│  ┌───────────────┐                             ┌──────────────────────────────┐  │
│  │ api (FastAPI) │  reads ◄──── Postgres ────► │ worker                       │  │
│  │ auth, admin → │             (system of      │ scheduler → job queue        │  │
│  │ enqueue jobs  │              record)        │ (SKIP LOCKED) → pipeline     │  │
│  └───────────────┘                             └──────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────┘
   External: FRED/ALFRED · Alpha Vantage · Kalshi API · Polymarket (Gamma + CLOB) · Anthropic
```

### 3.1 Modules (`backend/src/prediction_desk/`)

Each module has a small public interface. Modules do not read each other's tables directly.

| Module | Responsibility | Ports |
|---|---|---|
| `clock` | `WallClock` (live) / `SimClock` (replay). **No module calls `datetime.now()`**; ruff's banned-API rule enforces this | `Clock` |
| `calendar` | Release schedule for CPI/NFP/FOMC; triggers pipeline runs | `ReleaseCalendar` |
| `ingestion` | Macro data, always fetched **as of** a timestamp | `MacroDataProvider` (FRED/ALFRED, Alpha Vantage, Replay) |
| `markets` | Venue clients, canonical mapping, quotes and order books | `VenueAdapter` (Kalshi, Polymarket, Replay) |
| `forecasting` | Base quant models and the multi-agent Forecast Desk | `BaseModel`, `ForecastDesk` |
| `pricing` | Distribution → fair probability per contract; edge after fees and slippage | pure functions |
| `risk` | Kelly, cap funnel, kill switch, Risk Officer integration; **sole producer of `ApprovedOrder`** | `RiskEngine` |
| `execution` | Paper broker, positions, P&L, settlement | `Broker` (Paper; `LiveBroker` stub) |
| `audit` | Append-only hash-chained event log and verifier | `AuditLog` |
| `review` | Post-event metrics, analyst attribution, narrative | `ReviewAgent` |
| `agents` | Shared Anthropic client wrapper: tool loop, schema validation, budgets, transcripts | `LLMClient` |
| `pipeline` | State machine and stage runners | — |
| `api` | FastAPI app, routers, auth, SSE | — |
| `worker` | Scheduler and job-queue consumer | — |

### 3.2 Pipeline state machine

One `pipeline_run` per release per evaluation slot:

```
DISCOVER_CONTRACTS → GATHER_DATA → BASE_FORECAST → AGENT_ADJUST → PRICE_EDGE
  → RISK_EVALUATE → EXECUTE → (await release) → SETTLE → REVIEW
```

- Each stage is **idempotent**, persists its output, and appends an audit event in the same transaction.
- On crash, the worker resumes from the last completed stage.
- Stages run as jobs in a Postgres queue (`SELECT … FOR UPDATE SKIP LOCKED`).
- Live and replay share one code path. Only the injected `Clock`, `MacroDataProvider` and `VenueAdapter` differ.

### 3.3 Schedule per release

| Slot | Action |
|---|---|
| T−24h | Full run (all stages through EXECUTE) |
| T−2h | Re-evaluation: re-quote, re-price, re-risk. Agents rerun only if `inputs_hash` changed. Start polling snapshots every 60s |
| T−15m | Final re-evaluation |
| **T−5m** | **Hard trading cutoff.** Any run not past EXECUTE is marked `skipped_cutoff` |
| Release + Δ | SETTLE when the official value appears (polled), then REVIEW |

---

## 4. Data model

Postgres 16, SQLAlchemy 2.0, Alembic. **Money is stored as integer cents. Probabilities are `NUMERIC(6,5)`.** No floats are persisted for money or probabilities.

| Table | Key columns | Notes |
|---|---|---|
| `releases` | `id, series (CPI/NFP/FOMC), period, scheduled_at, actual_value, released_at` | Calendar and outcome |
| `event_contracts` | `id, release_id, venue, venue_ticker, comparator (gte/lte/between/eq), strike_lo, strike_hi, unit, rounding_rule, resolution_source, mapping_confidence, mapped_by` | Canonical; rows from both venues point at the same `release_id` |
| `market_snapshots` | `contract_id, ts, best_bid, best_ask, bid_size, ask_size, last, volume, book JSONB` | Append-only; used for edge, fills and replay |
| `data_observations` | `series_id, obs_date, value, vintage_ts, source` | **PK includes `vintage_ts`**; revisions never overwrite |
| `pipeline_runs` | `id, release_id, slot, mode (live/replay), replay_batch_id, stage, status, clock_ts, inputs_hash, llm_mode (enabled/disabled_contaminated)` | State machine |
| `jobs` | `id, kind, payload, run_after, attempts, locked_at, locked_by, last_error` | Postgres queue |
| `forecasts` | `run_id, kind (base/adjusted), dist_family, params JSONB, model_version, inputs_hash` | Full distributions |
| `analyst_briefs` | `run_id, analyst, direction, magnitude, confidence, evidence JSONB, transcript_id` | One row per specialist |
| `agent_adjustments` | `run_id, proposal JSONB, skeptic_notes, revised (bool), clipped_fields, final_delta JSONB, status (applied/fallback/unreviewed)` | Lead + Skeptic outcome |
| `llm_transcripts` | `id, agent, model_id, messages JSONB, tool_calls JSONB, input_tokens, output_tokens, latency_ms` | Full reproducibility |
| `edges` | `run_id, contract_id, fair_prob, market_prob, side, fees_cents, slippage_cents, edge_after_costs` | Per contract |
| `risk_limits` | `version, params JSONB, created_by, created_at, active` | Versioned config |
| `risk_decisions` | `id, run_id, contract_id, kelly_raw, stake_engine, stake_officer, stake_final, verdict, reasons JSONB (ordered), limits_version, limits_snapshot JSONB` | Funnel trace |
| `orders` | `id, client_order_id UNIQUE, risk_decision_id NOT NULL FK, side, limit_price, qty, status` | **The FK enforces the risk gate** |
| `fills` | `order_id, price, qty, fees_cents, slippage_cents, ts` | |
| `positions` | `contract_id, mode/batch, qty, avg_cost_cents, realized_pnl_cents` | |
| `settlements` | `contract_id, outcome, payout_cents, venue_resolution, our_resolution, mismatch (bool)` | |
| `book_state` | `mode/batch, equity_cents, peak_equity_cents, day_start_equity_cents, kill_switch_on, kill_reason, kill_actor` | One row per book |
| `reviews` | `release_id, run_id, brier_base, brier_adjusted, logloss_base, logloss_adjusted, pnl_cents, clean (bool), narrative, lessons JSONB` | |
| `analyst_scores` | `analyst, release_id, brier_contribution, direction_correct` | Agent track record |
| `replay_batches` | `id, params, status, metrics JSONB, created_by` | |
| `audit_events` | `seq BIGSERIAL, ts, actor, entity, entity_id, action, payload JSONB, prev_hash, hash` | **Append-only, hash-chained** |
| `admin_users` | `github_login, role` | Allowlist |

### 4.1 Audit-trail guarantees
1. **Append-only at the database level.** The app role has only `INSERT`/`SELECT` on `audit_events`, and a trigger rejects `UPDATE`/`DELETE`.
2. **Hash chain.** `hash = sha256(prev_hash ‖ canonical_json(envelope) ‖ seq ‖ ts)`, where `envelope = {actor, entity, entity_id, action, payload}`, so who/what/action are tamper-evident too. Payloads may not contain floats or non-string keys. Appends are serialized with `pg_advisory_xact_lock` to prevent forks.
3. **Transactional coupling.** Every state change writes its domain row and its audit event in one transaction.
4. **Verification.** `GET /api/v1/audit/verify` and a nightly job walk the chain and report the first broken link. A break trips the kill switch.
5. **Reproducibility.** Every decision stores `inputs_hash`, `model_version`, `limits_snapshot` and LLM transcript IDs.

### 4.2 Point-in-time queries
Any macro value is read as `WHERE vintage_ts <= :clock_ts ORDER BY vintage_ts DESC LIMIT 1` per observation. Live and replay use the identical query.

---

## 5. Forecasting

### 5.1 Base quant models

Each model is versioned (for example `CpiModelV1`) and implements `predict(asof) -> Distribution`. A new version can run in shadow next to the old one.

| Series | Target | Model | Inputs (FRED/ALFRED unless noted) | Distribution |
|---|---|---|---|---|
| CPI | Headline and core MoM %, plus the YoY strikes venues list | Ridge regression on lagged components + gasoline nowcast + month seasonality | `CPIAUCSL`, `CPILFESL`, `GASREGW`, `PPIACO`, `CUSR0000SAH1`; Alpha Vantage `CPI` as cross-check | Student-t (μ, σ, ν); σ from rolling **out-of-sample** residuals |
| NFP | Headline payroll change (k) | Regression on claims and private payroll proxies | `ICSA`, `CCSA`, `ADPWNUSNERSA`, prior revisions, `JTSJOL` | Normal (μ, σ ≈ historical surprise sd) |
| FOMC | Decision ∈ {−50, −25, 0, +25} bp | Short-rate-implied expectation (bills/SOFR vs. effective funds rate) + "Fed rarely surprises" prior | `DFF`, `DTB4WK`, `DTB3`, `SOFR`, `DGS2`; Alpha Vantage `FEDERAL_FUNDS_RATE`, `TREASURY_YIELD` | Categorical |

### 5.2 Forecast Desk (multi-agent)

```
base distribution + diagnostics
        │
        ├─► Specialist analysts (parallel, asyncio) ─► AnalystBrief × N
        │      CPI:  Energy · Shelter/services · Goods/PPI
        │      NFP:  Claims/flows · ADP/survey · Revisions
        │      FOMC: Rates-market · Fed-communications
        ▼
   Lead Forecaster ─► AdjustmentProposal ─► Skeptic ─► (≤1 revision) ─► code clipping ─► adjusted distribution
```

- **Analysts** (default Sonnet 5.5) use read-only tools scoped to their domain: `query_series(series_id, asof)`, `news_sentiment(topic, asof)`, `market_snapshot(contract_id)`, `model_diagnostics()`. Each returns `AnalystBrief {direction, magnitude, confidence, evidence[] (each citing a tool-call ID)}`. Analysts cannot modify the distribution.
- **Lead Forecaster** (default Opus 5.5) synthesizes the briefs and the base model into an `AdjustmentProposal`: `{delta_mu, sigma_multiplier, rationale, evidence[]}`, or `{probability_shifts}` for FOMC.
- **Skeptic** (default Sonnet 5.5) checks citations and **flags double counting** of signals the base model already uses. It may request one revision.
- **Hard bounds, applied in code after the LLM responds:**
  - CPI: |Δμ| ≤ 0.05pp; σ multiplier ∈ [1.00, 1.25] (it can widen but never narrow)
  - NFP: |Δμ| ≤ 40k; σ multiplier ∈ [1.00, 1.25]
  - FOMC: each outcome shifts ≤ 10pp, then renormalized
  - Values outside the bounds are **clipped and recorded** in `clipped_fields`.
- **Failure handling:** invalid output → one retry → fall back to the base model (`status=fallback`). A failed analyst is dropped. A failed Skeptic → `status=unreviewed`.
- **Replay:** for any release before the configured `LLM_TRAINING_CUTOFF`, `llm_mode=disabled_contaminated` and the desk is skipped, so the adjusted distribution equals the base distribution. These results are labeled **clean**.

### 5.3 Contract Mapper agent
Default model Haiku 5.5. It reads venue market titles and rules text and extracts `EventContract` fields plus `mapping_confidence`. A code validator then checks the strike parse, the unit, the date against `releases`, and the comparator. Contracts with confidence < 0.9 or a failed validation are stored as `unmapped` and never traded. Mappings are persisted, so replay never calls the mapper.

---

## 6. Pricing and edge (pure functions)

- `fair_prob(contract, dist)` integrates the distribution over the contract's region **after applying the venue's rounding rule** (for example, BLS reports CPI to 0.1, so "≥ 0.3%" means `P(round(x, 1) ≥ 0.3)`).
- **YES edge** = `fair − ask − fee − slippage`. **NO edge** is the mirror calculation.
- **Kalshi fee:** `ceil(0.07 × C × P × (1−P))` dollars, per the published schedule, stored in config. **Polymarket fee:** from config (≈0 on most markets).
- **Slippage:** computed by walking the snapshot's order book for the intended size.
- **Sanity checks:** fair probabilities must be monotonic across ordered strikes, and bracket sets must sum to 1 ± ε. A violation trips the kill switch.
- Only contracts with `edge_after_costs ≥ min_edge` (4¢) go on to risk evaluation.

---

## 7. Risk engine and kill switch

### 7.1 Kelly for binary contracts
Buying YES at all-in cost `c` (price + fee, per $1 payout) with fair probability `q`: `f* = (q − c) / (1 − c)`. NO is the mirror case. Stake = `kelly_fraction × f* × equity`.

### 7.2 Sizing funnel (`RiskEngine.evaluate`)
Every step appends a named entry to `reasons[]`:
1. **Pre-trade gates** (any failure → reject): kill switch off; quote age < 60s (live); before the T−5m cutoff; `mapping_confidence ≥ 0.9`; venue healthy; edge still ≥ min on a **fresh re-quote**.
2. `kelly_raw = 0.25 × f*`
3. Per-contract cap: `min(·, 2% equity)`
4. Per-event cap: `min(·, 5% equity − existing exposure on this release across both venues)`
5. Portfolio cap: `min(·, 15% equity − gross open exposure)`
6. Liquidity cap: `min(·, 25% of the depth available at or better than the limit price)`
7. Round down to whole contracts; < 1 contract → reject `below_min_size`.
8. **Risk Officer** (an LLM, default Opus 5.5) reviews the full slate for the run and returns `officer_stake ≤ stake_engine` per contract. It can only veto or reduce: `stake_final = min(stake_engine, officer_stake)`. **If the Risk Officer fails, the system fails closed and places no trade.** It is skipped in pre-cutoff replay.

### 7.3 Structural gate
- `ApprovedOrder` has no public constructor. Only `RiskEngine` creates it, inside the transaction that writes `risk_decisions` and the audit event.
- That transaction re-reads `book_state` with `SELECT … FOR SHARE`. A kill-switch flip takes `FOR UPDATE`, so it waits for in-flight decisions, and every later decision sees the new state.
- `orders.risk_decision_id` is `NOT NULL` with a foreign key.

### 7.4 Kill switch
- **Automatic triggers:** daily P&L ≤ −8% of start-of-day equity; drawdown from peak ≥ 15%; audit-chain break; 3 consecutive data-QA failures; pricing sanity violation.
- **Manual:** an admin action with a required reason, audited as `actor=admin:<login>`.
- **Effect:** no new orders. Existing positions are **held to settlement**.
- **Reset:** admin only, with a reason, audited. It never resets automatically.

---

## 8. Execution (paper)

- **Fills:** marketable limit orders, filled by walking the book (the fresh re-quote in live; the latest snapshot ≤ `clock_ts` in replay). Partial fills are allowed. Fees are applied exactly.
- **Idempotency:** `client_order_id = sha256(run_id, contract_id, slot)`, with a unique constraint.
- **Mark-to-market:** to mid on each snapshot. `book_state` tracks equity, peak equity and start-of-day equity.
- **Settlement:** against the official value (BLS/FOMC via FRED), applying each contract's resolution rule. A disagreement between the venue's resolution and ours sets `mismatch` and raises an alert.
- **`LiveBroker`:** an interface stub only. The process refuses to start unless `TRADING_MODE=paper`.

---

## 9. API and frontend

### 9.1 API (`/api/v1`, FastAPI)
- **Public (rate-limited, cursor-paginated):** `GET releases`, `releases/{id}` (full decision chain), `contracts`, `positions`, `trades`, `equity-curve`, `audit/events`, `audit/verify`, `agents/scorecards`, `replay/batches`, `replay/batches/{id}`, `health`, `ready`.
- **Admin (JWT role=admin; every call audited):** `POST kill-switch`, `PUT risk-limits` (new version), `POST replay/batches`, `POST pipeline-runs/{id}/retry-stage`.
- **SSE:** `GET stream`, fed by Postgres `LISTEN/NOTIFY` (stage completed, fill, kill-switch change).
- **Contract:** OpenAPI → `openapi-typescript` client in `web/`. CI fails on drift.

### 9.2 Auth
GitHub OAuth via Auth.js in Next.js. Admin access requires the GitHub login to be in `admin_users`. Next.js server routes call FastAPI admin endpoints with a short-lived JWT (HS256, shared secret, 5-minute expiry) carrying `sub` and `role`. Public pages need no auth.

### 9.3 Pages
| Route | Content |
|---|---|
| `/` Desk | Equity curve, kill-switch banner, open positions, exposure gauges vs. caps, upcoming-release countdowns, clean track-record summary |
| `/releases/[id]` | Base vs. adjusted distribution overlaid with market-implied probabilities per strike (both venues); analyst briefs, Skeptic notes, Lead rationale; per-contract risk funnel waterfall; fills; outcome; review |
| `/trades` | Blotter of orders, fills and settlements |
| `/audit` | Event stream, chain-verification badge, entity links |
| `/agents` | Per-analyst scorecards, base vs. LLM attribution |
| `/replay` | Batches with Brier, log-loss, calibration plot, P&L, max drawdown; clean/contaminated labels |
| `/admin` | Kill switch, risk-limit editor with version diff, replay launcher, stage retry |

Visual direction: dark, information-dense trading-desk look (refined with the frontend-design skill during the build).

---

## 10. Operations, error handling, testing

### 10.1 Repo and deployment (no Docker)
```
backend/   Python 3.12 (uv) — src/prediction_desk, tests, alembic
web/       Next.js (pnpm)
docs/      spec, ADRs, diagrams
.github/workflows/
```
- **Railway** project `prediction-desk`: services `api`, `worker`, `Postgres`; environments `production` and `development`. Built with Railpack from source. Per-service `railway.json` sets the start command and health check. A **pre-deploy command** runs `alembic upgrade head`. "Wait for CI" gates deploys.
- **Vercel:** `web/` deploys from `main`, with preview deploys on PRs.
- **Local development:** `railway run …` injects the development environment's variables. Tests use local Homebrew Postgres 16.
- **Secrets:** Railway and Vercel variables, `.env.example` committed, `gitleaks` in CI.

### 10.2 Error handling
- **HTTP:** `httpx` timeouts, jittered retries on idempotent GETs, and a circuit breaker per provider.
- **Alpha Vantage** (free tier: 25 calls/day) is cached aggressively and is a secondary source only. **FRED** is primary.
- **Data QA:** staleness, range and missing-vintage checks fail the stage, which is retried with backoff. 3 consecutive failures trip the kill switch.
- **Partial state:** never traded on; `skipped_cutoff` applies past T−5m.
- **LLM:** schema validation, one retry, then fallback; per-run token budget; tool outputs wrapped as untrusted data; agents have no write tools.
- **Observability:** `structlog` JSON logs with `run_id`/`release_id`; Sentry (api, worker, web); worker heartbeat for `/ready`; webhook alerts (Slack or Discord, via `ALERT_WEBHOOK_URL`) on kill-switch trip, chain break or settlement mismatch.

### 10.3 Testing
| Layer | Scope | Tooling |
|---|---|---|
| Unit | Pricing, rounding, Kelly, funnel, hash chain | pytest + Hypothesis (monotonicity, bracket sum = 1, stake ≤ every cap) |
| Integration | Risk-gate transaction, kill-switch race, audit trigger, crash and resume | Local Postgres, throwaway DB per run |
| Adapters | Kalshi, Polymarket, FRED/ALFRED, Alpha Vantage | `pytest-recording` fixtures; no live calls in CI |
| Agents | Schemas, clipping, fallback, Risk Officer fail-closed, mapper accuracy | Fake Anthropic client; golden mapping eval set |
| Determinism | Replay of 3 fixture releases → byte-identical decisions | Golden-file test |
| Frontend | Components; desk, release and audit smoke flows | Vitest + Testing Library; Playwright |
| CI gates | ruff, `mypy --strict`, coverage **100% on `risk/`, `pricing/`, `audit/`**, 85% overall; ESLint, `tsc`, OpenAPI drift | GitHub Actions + the runner's built-in Postgres |

---

## 11. External dependencies and credentials

| Service | Needed for | Credential |
|---|---|---|
| FRED / ALFRED | Macro data (primary) | Free API key |
| Alpha Vantage | Cross-checks, news sentiment | API key (free tier is enough at our cadence) |
| Kalshi | Market data (public endpoints) | None for reads |
| Polymarket | Gamma (markets) + CLOB (books, price history) | None for reads |
| Anthropic | All agents | API key |
| GitHub OAuth app | Admin login | Client ID and secret |
| Sentry | Error tracking | DSN (optional) |
| Railway / Vercel | Hosting | Accounts (Railway MCP is connected) |

---

## 12. Risks and open questions

- **FOMC edge is likely thin.** Markets price FOMC efficiently; we expect few trades and accept that. The minimum-edge gate handles this naturally.
- **Polymarket macro coverage varies.** Some releases may have no Polymarket contract. The system must work with Kalshi alone for any release.
- **Historical order-book depth.** Kalshi and Polymarket expose historical prices and trades but not full historical books. Replay fills use the recorded price with a configurable synthetic depth and slippage model, documented as an assumption and labeled in replay results.
- **Kalshi CPI/NFP strike conventions change over time.** Mapping relies on the Contract Mapper plus the validator, not hard-coded tickers.
- **Fee schedules change.** Fees live in versioned config, not code.

---

## 13. Build phases (input to the implementation plan)

1. **Foundation:** repo, tooling, CI, Railway project and environments, DB schema + migrations, `clock`, `audit` (hash chain).
2. **Data and markets:** FRED/ALFRED provider with vintages, Alpha Vantage, Kalshi + Polymarket adapters, Contract Mapper, snapshot polling.
3. **Quant core:** base models (CPI, NFP, FOMC), pricing and rounding, edge.
4. **Risk and execution:** Kelly, funnel, kill switch, structural gate, paper broker, settlement. *(The user implements `kelly_fraction()` and one funnel step.)*
5. **Agents:** LLM client wrapper, analysts, Lead, Skeptic, Risk Officer, Review agent.
6. **Pipeline and worker:** state machine, job queue, scheduler, replay engine + batches.
7. **API:** public and admin routers, auth, SSE, OpenAPI client generation.
8. **Frontend:** all pages, admin console, charts.
9. **Hardening and launch:** determinism golden test, Playwright, observability, Railway + Vercel deploys, README and architecture docs, initial replay batch.

# Prediction Desk

Prediction Desk is an agentic paper-trading system for regulated prediction-market contracts on US macroeconomic releases: CPI, Nonfarm Payrolls (NFP) and FOMC rate decisions. A multi-agent forecasting desk builds probability distributions from point-in-time macro data and compares them against Kalshi and Polymarket prices to find edge. A deterministic risk engine, with institutional-style controls, decides whether and how much to trade. Every decision is recorded in a tamper-evident audit trail and scored in a post-event review. It paper-trades only; live money trading is a non-goal.

**Status:** Plan 1 (Foundation) is done: tooling, settings, clock, value types, database and migrations, `releases`, the hash-chained audit log, and CI. Nothing else in the roadmap below exists yet.

## Architecture at a glance

- Modular monolith with hexagonal ports and adapters, run as two processes (`api`, `worker`).
- Backend: Python 3.12, FastAPI, SQLAlchemy 2.0, Alembic, Postgres 16, managed with `uv`.
- Quant base model plus a bounded LLM adjustment; the deterministic risk engine has the final say.
- Orchestration is an explicit typed pipeline backed by a durable state machine in Postgres.
- No Docker anywhere: Homebrew Postgres locally, the runner's built-in Postgres in CI.

## Local setup

```bash
brew install postgresql@16
brew services start postgresql@16
cd backend
uv sync
uv run pytest
# optional: migrate a dev database
createdb prediction_desk && DATABASE_URL=postgresql://localhost/prediction_desk uv run alembic upgrade head
```

Tests need only `TEST_DATABASE_ADMIN_URL`, which defaults to `postgresql+psycopg://localhost/postgres`. Copy `backend/.env.example` for the other settings. The full CI gate is in `.github/workflows/ci.yml`.

## Roadmap

| # | Plan | Delivers |
|---|---|---|
| **1** | **Foundation** (done) | Tooling, settings, clock, value types, DB + migrations, `releases`, audit log, CI |
| 2 | Data & markets | FRED/ALFRED vintage provider, Alpha Vantage, Kalshi + Polymarket adapters, `EventContract` mapping (validator; the LLM Contract Mapper arrives in Plan 4), snapshots |
| 3 | Quant core, risk & execution | CPI/NFP/FOMC base models, pricing + rounding + edge, Kelly + funnel + kill switch + structural gate, paper broker, settlement |
| 4 | Agents | LLM client wrapper, Contract Mapper, analysts → Lead → Skeptic, Risk Officer, Review agent |
| 5 | Pipeline, worker & replay | Postgres job queue, state machine, scheduler, replay engine and batches |
| 6 | API | Public and admin routers, JWT auth, SSE via LISTEN/NOTIFY, OpenAPI export; Railway project + `development` environment |
| 7 | Frontend | Next.js app, all pages, admin console, GitHub OAuth |
| 8 | Hardening & launch | Determinism golden test, Playwright, Sentry/alerts, app DB role with INSERT/SELECT-only on `audit_events`, production deploys, README/ADRs, first replay batch |

## Documents

- Design spec: [docs/superpowers/specs/2026-11-04-prediction-desk-design.md](docs/superpowers/specs/2026-11-04-prediction-desk-design.md)
- Plan 1 (Foundation): [docs/superpowers/plans/2026-11-04-plan-01-foundation.md](docs/superpowers/plans/2026-11-04-plan-01-foundation.md)

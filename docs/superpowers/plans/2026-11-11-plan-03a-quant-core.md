# Prediction Desk — Plan 3a: Quant Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn point-in-time macro data and recorded order books into a priced view of every mapped contract on a release:
- a forecast distribution per measure from a versioned base model;
- a fair probability per contract, computed on the *reported* (rounded) value;
- fees and order-book slippage;
- the edge per side.

All of it is reproducible from stored inputs.

**Architecture:**
- **No DB writes.** `forecasting/` (distributions, the realtime training-set builder, ridge regression, and the CPI/NFP/FOMC models) and `pricing/` (reported-value regions, fair probability, coherence checks, fees, book walking, edge) are pure libraries. They read only through `MacroDataProvider.series_asof` and `latest_snapshot_asof`. Persisting forecasts and edges is Plan 5's pipeline.
- **One entry point.** `pricing/release.py::price_release` composes everything for one release as of one timestamp. It is the single call Plan 3b (risk) and Plan 5 (pipeline) make.
- **Floats stay inside kernels.** They exist only inside numeric kernels (numpy/scipy/math). Every value crossing a public function boundary is `Decimal`/`Prob`/`int`, and distribution parameters are `Decimal`.

**Tech Stack:** Python 3.12, numpy, scipy (+ scipy-stubs), SQLAlchemy 2 (async, read-only use), PyYAML, pytest + Hypothesis.

**Spec:** `docs/superpowers/specs/2026-11-04-prediction-desk-design.md`. This plan implements §5.1 (base models), §6 (pricing and edge) and the coherence part of the §6 sanity checks. Also read the "Execution Outcome and Carry-Forward" sections of `docs/superpowers/plans/2026-11-04-plan-01-foundation.md` and `docs/superpowers/plans/2026-11-10-plan-02-data-and-markets.md`. Risk, Kelly, the kill switch and the paper broker are **Plan 3b**.

## Global Constraints

### Carried forward
- Use `Clock`/`require_utc` for every timestamp.
- No floats are persisted or placed in audit payloads.
- Any new test file needs a unique basename (R5).
- Tests run with `--record-mode=none --block-network`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

### Float boundary
- Floats appear only inside private numeric kernels.
- These public types use `Decimal` fields: `Distribution` subclasses, `FairPrice`, `EdgeQuote`, `ReleasePricing`.
- Convert at the boundary with `to_prob(Decimal(format(x, ".12f")))` after clamping to `[0, 1]`, or with `Decimal(format(x, ".12f"))` for non-probabilities.
- Never call `Decimal(float)` directly, because it gives binary noise.

### Point-in-time
- Every model input comes from `MacroDataProvider.series_asof(..., asof=t)` with `t <= ` the prediction `asof`.
- A training sample for reference month *m* uses `t_m = first_vintage_ts(m) − 1 µs`, i.e. what was known just before *m* was first published. Its target is the **first-print** value of *m*.
- Only months whose first vintage is at or before the prediction `asof` enter training.

### Reported-value semantics
- Contracts resolve on the reported value `R = round(X, grid)`, where `grid = Decimal(1).scaleb(-resolution_precision)`.
  - CPI measures: precision `1`, grid `0.1` (pct).
  - NFP: precision `-3`, grid `1000` jobs. BLS reports in thousands.
  - FOMC: categorical outcomes with no rounding.
- For continuous distributions the rounding-tie set has probability 0, so the half-up vs half-even choice does not affect prices.

### FOMC outcome grid
- Outcomes are `(-50, -25, 0, 25, 50)` bp. `-50` means "cut ≥ 50", and `50` means "hike ≥ 50". These tails make Kalshi `{"Cut": ">25"}` → `LT -25` and Polymarket `50+ bps decrease` → `LTE -50` price identically.

### Coverage
- Coverage is **100% on `prediction_desk/pricing/`**. CI gets a second `coverage report --include` line, the same as for `audit/`. Overall coverage must stay ≥ 85%.

### Model versioning
- Model versions are module constants: `cpi-ridge-v1`, `nfp-ridge-v1`, `fomc-implied-v1`.
- `MAPPER_VERSION` becomes `rules:v2`, because of the NFP precision change in Task 1.

## Review Focus

1. **A strike sitting exactly on a rounding boundary.** Example: CPI `GT 0.3`, where the reported value must be ≥ 0.4, so X ≥ 0.35. With `StudentT(μ=0.35)`, `fair_prob` must be exactly 0.5 to 5 dp, not P(X > 0.3). *(Test: Task 3)*
2. **Too little point-in-time history.** This happens with a sparse series or early in the record. The model must raise `InsufficientData`, never return a distribution fitted on 3 points. `price_release` reports that measure as unpriceable with a reason. *(Tests: Tasks 5, 9)*
3. **Missing or stale book for a contract.** The edge is `unavailable` with reason `no_snapshot`/`empty_side`/`insufficient_depth`. The quote age is always reported, and an old book never silently produces an edge. *(Tests: Tasks 4, 9)*
4. **σ collapsing toward 0.** A near-perfect in-sample fit on small data could do this. A per-measure σ floor applies, so no contract is priced at exactly 0 or 1 from a degenerate fit. *(Tests: Tasks 5–7)*
5. **Float error at the probability edges.** A CDF can return `1.0000000000000002` or `-1e-17`. The boundary conversion clamps, so `to_prob` never raises on a valid model output. *(Test: Task 2)*

---

## File Structure

```
backend/src/prediction_desk/
  markets/mapping.py            (modify) NFP precision -3; MAPPER_VERSION rules:v2
  markets/types.py              (modify) OrderBook size cap; OrderBook.from_json()
  markets/terms.py              terms_from_row(EventContract) -> ContractTerms
  forecasting/distributions.py  Distribution protocol, Normal, StudentT, Categorical
  forecasting/realtime.py       first-print + as-of training-set helpers, InsufficientData
  forecasting/ridge.py          ridge fit, expanding-window OOS residuals, sigma
  forecasting/base.py           BaseModel protocol, ModelOutput
  forecasting/cpi.py            CpiModelV1 (headline/core MoM + YoY transform)
  forecasting/nfp.py            NfpModelV1
  forecasting/fomc.py           FomcModelV1
  forecasting/registry.py       MODELS: Mapping[Series, BaseModel]
  pricing/regions.py            reported-value interval math (exact Decimal)
  pricing/fair.py               fair_prob(terms, dist) -> Prob; coherence checks
  pricing/fees.py               versioned fee schedules (fee_schedules.yaml)
  pricing/fee_schedules.yaml
  pricing/book.py               walk_book(); side-aware fills
  pricing/edge.py               evaluate_edge() -> EdgeQuote
  pricing/release.py            price_release() -> ReleasePricing
backend/tests/{forecasting,pricing}/...   (unique basenames)
.github/workflows/ci.yml        (modify) 100% gate on pricing/
```

---

### Task 1: Plan 2 follow-ups, contract terms, and numeric dependencies

**Files:**
- Modify:
  - `backend/src/prediction_desk/markets/mapping.py`
  - `backend/src/prediction_desk/markets/types.py`
  - `backend/pyproject.toml`
  - `.github/workflows/ci.yml`
  - affected tests in `tests/markets/`
- Create:
  - `backend/src/prediction_desk/markets/terms.py`
  - `backend/tests/markets/test_terms.py`

**Interfaces:**
- **Consumes:**
  - `ContractTerms`, `MAPPER_VERSION` (from `mapping.py`)
  - `EventContract`, `MappingStatus`
  - `OrderBook`, `BookLevel`
- **Produces:**
  - **NFP precision.** `MEASURE_DEFAULTS` for `Measure.NFP_CHANGE` becomes `("jobs", -3, "BLS")`, and `MAPPER_VERSION = "rules:v2"`.
  - **Order size cap.** `MAX_LEVEL_SIZE = Decimal("1e12")` in `markets/types.py`. `OrderBook.__post_init__` rejects any `size > MAX_LEVEL_SIZE` with `ValueError`. That error becomes a `ProviderError` at the adapter boundary, per Plan 2 R13.
  - **`OrderBook.from_json`.** Signature: `@classmethod OrderBook.from_json(cls, book: Mapping[str, Sequence[Sequence[str]]], ts: datetime, venue_ts: datetime | None = None) -> OrderBook`. It is the inverse of `to_json()`. It parses strings to `Decimal` and validates through `__post_init__`.
  - **`terms_from_row`.** Signature: `def terms_from_row(row: EventContract) -> ContractTerms`. It raises `ValueError("contract <id> is not mapped")` if `row.mapping_status != MAPPED` or any required field is None.
  - **Rules hash.** `def rules_hash(row: EventContract) -> str`, the sha256 hex digest of `row.rules_text.encode()`. It is used in `inputs_hash` (Plan 2 carry-forward).
  - **Dependencies.** `numpy`, `scipy`; dev `scipy-stubs`.
  - **CI.** `ci.yml` adds `uv run coverage report --include='*/prediction_desk/pricing/*' --fail-under=100` after the audit gate. Add the step now; it passes vacuously until `pricing/` exists, because it only reports included files.

- [ ] **Step 1: Write failing tests.**
  - `test_terms.py`:
    - `test_terms_from_mapped_row`
    - `test_terms_from_unmapped_row_raises`
    - `test_rules_hash_stable_and_sensitive`
  - `tests/markets/test_mapping.py` (update):
    - KXPAYROLLS now gives `resolution_precision == -3`
    - `MAPPER_VERSION == "rules:v2"`
  - `test_venue_types.py`:
    - `test_size_cap`: `Decimal("1e12")` is ok and `Decimal("1000000000001")` raises
    - `test_from_json_roundtrip`: `OrderBook.from_json(b.to_json(), b.ts) == b`
    - `test_from_json_rejects_unsorted`
- [ ] **Step 2: Run the new tests and confirm they FAIL.**
- [ ] **Step 3: Implement.**
  - Run `uv add numpy scipy && uv add --dev scipy-stubs`.
  - Update `test_discovery_cassettes.py` expectations if NFP rows appear. They don't today, because no NFP cassette exists, but grep for any count tied to the mapper version.
- [ ] **Step 4: Full gates green.**
  - `uv run pytest -q --record-mode=none`
  - ruff
  - mypy
  - `actionlint .github/workflows/ci.yml`
- [ ] **Step 5: Commit.** Message: `feat: NFP reporting grid, OrderBook size cap and from_json, terms_from_row`

---

### Task 2: Distributions

**Files:**
- Create: `backend/src/prediction_desk/forecasting/{__init__,distributions}.py`
- Test: `backend/tests/forecasting/test_distributions.py`

**Interfaces:**
- **`Distribution` (Protocol):**
  - `family: str`
  - `def prob_between(self, lower: Decimal | None, upper: Decimal | None) -> Prob`, meaning P(lower ≤ X < upper). `None` means unbounded on that side.
  - `def params(self) -> dict[str, str]`, JSON-safe. Keys are sorted, and every value is `str(Decimal)`.
- **`Normal`:** `@dataclass(frozen=True)` with `mu: Decimal`, `sigma: Decimal`. `sigma > 0` is enforced.
- **`StudentT`:** `@dataclass(frozen=True)` with `mu`, `sigma` (scale), and `nu: Decimal`. `sigma > 0` and `nu > 0` are enforced. `prob_between` uses `scipy.stats.t.cdf((x-mu)/sigma, nu)`.
- **`Categorical`:** `@dataclass(frozen=True)` with `outcomes: tuple[tuple[int, Decimal], ...]`, holding (outcome, probability) pairs.
  - Outcomes are strictly ascending.
  - Probabilities are in [0, 1] and sum to 1 within `Decimal("1e-9")`.
  - `prob_between(lower, upper)` sums outcomes `o` with `lower ≤ o < upper`.
  - Method `prob_where(pred: Callable[[int], bool]) -> Prob`.
- **Linear transform:** `def affine(dist: Normal | StudentT, a: Decimal, b: Decimal) -> Normal | StudentT` gives the distribution of `a + b·X`. It requires `b > 0`, and the result has `mu' = a + b·mu`, `sigma' = b·sigma`.
- **Boundary helper:** `def _prob(x: float) -> Prob` clamps to [0, 1], then `to_prob(Decimal(format(x, ".12f")))`. This is the only float→Prob path.

- [ ] **Step 1: Write the failing tests:**
  - `test_normal_symmetry`: `Normal(0, 1).prob_between(None, 0) == Prob("0.50000")`.
  - `test_student_t_matches_scipy`: for `StudentT(0.2, 0.15, 5)`, check `prob_between(Decimal("0.35"), None)` against an independently computed value, written in as a literal.
  - `test_prob_between_monotone`: Hypothesis over lower < upper bounds. Widening an interval never decreases the probability.
  - `test_partition_sums_to_one`: P(<a) + P([a,b)) + P(≥b) = 1 within 2e-5, because of 5-dp rounding.
  - `test_float_edge_clamped`: patch the kernel to return `1.0000000000000002` and `-1e-17`. The results are `Prob("1.00000")` and `Prob("0.00000")`, with no exception (Review Focus 5).
  - `test_categorical_validation`: rejects unsorted outcomes, probabilities summing to ≠ 1, and negative probabilities.
  - `test_categorical_prob_where`
  - `test_affine_student_t`
  - `test_params_json_safe`: every value is a `str` and none is a float.
- [ ] **Step 2: Run the tests and confirm they FAIL.**
- [ ] **Step 3: Implement.** Use `math.erf` for the Normal CDF and `scipy.stats.t` for StudentT.
- [ ] **Step 4: Run the tests and confirm they pass.** Then run the gates.
- [ ] **Step 5: Commit** with the message `feat(forecasting): Normal, StudentT and Categorical distributions`.

---

### Task 3: Reported-value regions, fair probability, coherence checks

**Files:**
- Create: `backend/src/prediction_desk/pricing/{__init__,regions,fair}.py`
- Test: `backend/tests/pricing/test_regions.py`, `backend/tests/pricing/test_fair.py`

**Interfaces:**
- `@dataclass(frozen=True) class Interval: lower: Decimal | None; upper: Decimal | None`. This is the half-open interval [lower, upper) on the *unrounded* X.
- `def reported_interval(comparator: Comparator, strike_lo: Decimal, strike_hi: Decimal | None, precision: int) -> Interval`.
  - Let `g = Decimal(1).scaleb(-precision)`, `ceil_g(s)` = the smallest grid point ≥ s, `floor_g(s)` = the largest grid point ≤ s, and `half = g / 2`.

    | Comparator | Interval on X |
    |---|---|
    | `GT s` | `[next_g(s) − half, ∞)`, where `next_g(s)` is the smallest grid point **>** s |
    | `GTE s` | `[ceil_g(s) − half, ∞)` |
    | `LT s` | `(−∞, prev_g(s) + half)`, where `prev_g(s)` is the largest grid point **<** s |
    | `LTE s` | `(−∞, floor_g(s) + half)` |
    | `EQ s` | `[s − half, s + half)` when s is on the grid; otherwise an empty interval (`lower == upper`) |
    | `BETWEEN lo..hi` (inclusive) | `[ceil_g(lo) − half, floor_g(hi) + half)` |

  - Use only exact Decimal arithmetic: `quantize` with `ROUND_CEILING`/`ROUND_FLOOR` on `s / g`.
- `def fair_prob(terms: ContractTerms, dist: Distribution) -> Prob`.
  - **Categorical:** evaluate the comparator directly on the outcomes, with no rounding, using `prob_where`.
  - **Continuous:** `dist.prob_between(*reported_interval(...))`.
  - **Mismatches** raise `ValueError`: a categorical distribution with a non-`bp` unit, or a continuous one with `bp`.
- `def monotonic_violations(items: Sequence[tuple[ContractTerms, Prob]]) -> list[str]`. Within each `(measure, comparator)` group:
  - GT/GTE must be non-increasing in strike.
  - LT/LTE must be non-decreasing in strike.
  - It returns human-readable messages. An empty list means coherent.
- `def bracket_sum_violation(probs: Sequence[Prob], tolerance: Decimal = Decimal("0.02")) -> str | None`. It is for a full Polymarket bracket set (≤ low, EQ…, ≥ high): `|Σ − 1| > tolerance` produces a message.

- [ ] **Step 1: Write the failing tests:**
  - `test_regions.py`, parametrized:
    - CPI `GT 0.3` (precision 1) gives `[0.35, ∞)`.
    - CPI `GT -0.3` gives `[-0.25, ∞)`.
    - CPI `LTE 3.2` gives `(−∞, 3.25)`.
    - CPI `EQ 3.3` gives `[3.25, 3.35)`.
    - CPI `GTE 4.3` gives `[4.25, ∞)`.
    - NFP `GT 90000` (precision −3) gives `[90500, ∞)`.
    - NFP `GT 90500` (off-grid) gives `[90500, ∞)`. The next reported value above 90500 is 91000, so this is the same event as `GT 90000`.
    - `EQ 3.33` (off-grid) gives an empty interval.
    - `BETWEEN 0.2..0.4` gives `[0.15, 0.45)`.
    - `LT -25` at precision 0 gives `(−∞, -25.5)`. This is used only for continuous checks.
  - Hypothesis property: for any on-grid s, `GT s` and `LTE s` partition the line. That is, `GT.lower == LTE.upper`.
  - `test_fair.py`:
    - `test_boundary_strike_exact` (Review Focus 1): `StudentT(Decimal("0.35"), Decimal("0.1"), Decimal("5"))` with CPI `GT 0.3` gives `Prob("0.50000")`.
    - `test_fomc_categorical_terms`. Use `Categorical(tuple((o, Decimal(p)) for o, p in [(-50,"0.1"),(-25,"0.2"),(0,"0.6"),(25,"0.08"),(50,"0.02")]))`:
      - Kalshi `LT -25` gives 0.1.
      - Polymarket `LTE -50` gives 0.1.
      - `EQ 0` gives 0.6.
      - `GT 25` gives 0.02.
      - `GTE 50` gives 0.02.
    - `test_ladder_is_coherent`: GT strikes from −0.3 to 1.0 on one StudentT give `monotonic_violations == []`.
    - `test_polymarket_bracket_sums_to_one`: ≤3.2, 3.3 … 4.2, ≥4.3 give `bracket_sum_violation(...) is None`.
    - `test_detects_incoherence`: hand-made non-monotone probabilities produce a message.
    - `test_unit_family_mismatch_raises`.
- [ ] **Step 2: Run the tests and confirm they FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Confirm 100% line and branch coverage on `pricing/`.**
  - Run: `uv run pytest tests/pricing --cov=prediction_desk.pricing --cov-branch --cov-report=term-missing`.
  - Then run the full gates.
- [ ] **Step 5: Commit.** Message: `feat(pricing): reported-value regions, fair probability, coherence checks`.

---

### Task 4: Fees, book walking, and edge

**Files:**
- Create:
  - `backend/src/prediction_desk/pricing/{fees,book,edge}.py`
  - `backend/src/prediction_desk/pricing/fee_schedules.yaml`
- Test: `backend/tests/pricing/test_fees.py`, `test_book_walk.py`, `test_edge.py`

**Interfaces:**
- **`fee_schedules.yaml`**, packaged and loaded with `yaml.safe_load` via `importlib.resources`. Its content:

```yaml
# Versioned venue fee schedules. Lookups are as-of, so replay prices with the schedule then in force.
schedules:
  - {venue: kalshi, effective_from: "2024-01-01", kind: kalshi_quadratic, rate: "0.07"}
  - {venue: polymarket, effective_from: "2024-01-01", kind: zero}
```

- **Fee schedule types and functions:**
  - `@dataclass(frozen=True) class FeeSchedule`, with `venue: Venue`, `effective_from: date`, `kind: Literal["kalshi_quadratic", "zero"]`, `rate: Decimal`.
  - `def load_fee_schedules() -> list[FeeSchedule]`. It validates on load: entries must be sorted per venue, and the kind must be known.
  - `def schedule_for(venue: Venue, asof: datetime) -> FeeSchedule`. It returns the latest schedule with `effective_from` ≤ the `asof` ET date. If no schedule applies, it raises `LookupError`.
- **`fee_cents(schedule, price: Decimal, qty: int) -> int`:**
  - `kalshi_quadratic`: `ceil(rate × qty × price × (1 − price) × 100)`, computed with exact Decimal arithmetic and `ROUND_CEILING` to whole cents.
  - `zero`: returns 0.
  - `qty ≥ 1` is enforced.
- **Book walking (`pricing/book.py`):**
  - `@dataclass(frozen=True) class Fill`, with `avg_price: Decimal`, `worst_price: Decimal`, `qty: int`.
  - `def walk(levels: Sequence[BookLevel], qty: int) -> Fill | None`. It consumes levels best-first. It returns `None` if total depth < qty. `avg_price` is the size-weighted average, kept exact.
  - `def buy_fill(book: OrderBook, side: Literal["yes", "no"], qty: int) -> Fill | None`.
    - Buying YES walks the asks.
    - Buying NO walks the bids converted to NO prices (`1 − bid`). It returns NO prices.
- **Edge (`pricing/edge.py`):**
  - `@dataclass(frozen=True) class EdgeQuote`, with:
    - `side: Literal["yes", "no"]`
    - `fair: Prob` (the fair probability of the side)
    - `avg_price: Decimal`
    - `best_price: Decimal`
    - `fee_cents: int`
    - `slippage_cents: int`, computed as `round((avg − best) × qty × 100)`
    - `edge_per_contract: Decimal`, computed as `fair − avg_price − fee_cents/(100·qty)` and quantized to `0.0001`
    - `qty: int`
  - `@dataclass(frozen=True) class Unavailable`, with `reason: Literal["no_snapshot", "stale_snapshot", "empty_side", "insufficient_depth"]`.
  - `def evaluate_edge(fair_yes: Prob, book: OrderBook, venue: Venue, qty: int, asof: datetime) -> tuple[EdgeQuote | Unavailable, EdgeQuote | Unavailable]`.
    - It returns `(yes, no)` quotes using `schedule_for(venue, asof)`.
    - The NO side uses `fair = 1 − fair_yes`.
    - Each side reports `empty_side` when that book side is empty, and `insufficient_depth` when the walk returns `None`.
  - `def best_side(quotes) -> EdgeQuote | Unavailable`. It picks the side with the larger `edge_per_contract`, and returns `Unavailable` only if both sides are.

- [ ] **Step 1: Write the failing tests:**
  - **Fees:**
    - `test_kalshi_fee_examples`:
      - price 0.50, qty 10 → `ceil(0.07×10×0.25×100) = 18` cents
      - price 0.94, qty 1 → `ceil(0.39480) = 1` cent
      - price 0.50, qty 100 → 175
    - `test_polymarket_zero_fee`
    - `test_schedule_asof_picks_latest_effective`: uses a fixture YAML with two Kalshi entries, injected through a loader parameter.
    - `test_fee_yaml_packaged`: loads through `importlib.resources`.
  - **Book walk:**
    - `test_walk_single_level`
    - `test_walk_multi_level_weighted_avg`: asks `[0.51×5, 0.53×10]` with qty 10 gives avg `0.52`, worst `0.53`
    - `test_walk_insufficient_depth_none`
    - `test_buy_no_uses_one_minus_bid`: best bid 0.49 gives a NO price of 0.51
  - **Edge:**
    - `test_edge_yes_side_math`: fair 0.60, asks `[0.50×100]`, qty 10, Kalshi gives fee 18 and edge `0.60 − 0.50 − 0.018 = 0.0820`
    - `test_edge_picks_better_side`
    - `test_empty_side_unavailable`
    - `test_insufficient_depth_unavailable` (Review Focus 3)
    - `test_slippage_cents`
- [ ] **Step 2: Run the tests and confirm they FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Confirm the gates pass.** 100% coverage on `pricing/`, and the full gates.
- [ ] **Step 5: Commit** with message `feat(pricing): versioned fee schedules, book walking, per-side edge`.

---

### Task 5: Realtime training-set builder and ridge regression

**Files:**
- Create: `backend/src/prediction_desk/forecasting/{realtime,ridge,base}.py`
- Test: `backend/tests/forecasting/test_realtime.py`, `test_ridge.py`

**Interfaces:**
- **`forecasting/realtime.py`**
  - `class InsufficientData(Exception)`, with attributes `measure: str` and `have: int`, `need: int`.
  - `async def first_vintage_ts(session, source: DataSource, series_id: str, obs_date: date) -> datetime | None` returns the minimum `vintage_ts` for that observation.
  - `async def first_print(session, source, series_id, obs_date) -> Decimal | None` returns the value at that first vintage.
  - `async def levels_asof(provider: MacroDataProvider, session, series_id: str, asof: datetime, start: date) -> dict[date, Decimal]` wraps `series_asof` for FRED.
  - `def pct_change(levels: Mapping[date, Decimal], month: date) -> Decimal | None` returns `100 × (L[m]/L[m−1 month] − 1)` in exact Decimal, quantized to `1e-6`. It returns `None` if either level is missing.
  - `def month_mean(points: Mapping[date, Decimal], month: date) -> Decimal | None` returns the mean of the observations dated within that calendar month. Weekly and daily series use it.
  - `@dataclass(frozen=True) class Sample`, with `month: date`, `asof: datetime`, `features: tuple[Decimal, ...]`, `target: Decimal`.
  - `async def build_samples(session, provider, *, target_series: str, target_fn, feature_fn, asof: datetime, lookback_months: int = 96) -> list[Sample]`. For each reference month *m* whose first vintage of `target_series` is ≤ `asof`:
    1. Set `t_m = first_vintage_ts(m) − timedelta(microseconds=1)`.
    2. Compute `features = await feature_fn(session, provider, m, t_m)`, which returns `tuple[Decimal, ...] | None`.
    3. Compute `target = await target_fn(session, provider, m)`, the first-print-based target, returning `Decimal | None`.
    4. Skip the month if either value is `None`.
- **`forecasting/ridge.py`.** This is the float kernel, private apart from these entry points.
  - `@dataclass(frozen=True) class RidgeFit`, with `coef`, `intercept`, `mean`, `scale`. These are numpy arrays, internal only.
  - `def fit_ridge(X: Sequence[Sequence[Decimal]], y: Sequence[Decimal], alpha: Decimal) -> RidgeFit` uses standardized features, an unpenalized intercept, and the closed form `(XᵀX + αI)⁻¹Xᵀy`.
  - `def predict(fit: RidgeFit, x: Sequence[Decimal]) -> Decimal`.
  - `def oos_residuals(X, y, alpha, min_train: int) -> list[Decimal]` uses an expanding window: fit on `[0:i)`, predict `i`, for `i ≥ min_train`.
  - `def oos_sigma(residuals: Sequence[Decimal], window: int, floor: Decimal) -> Decimal` returns `max(floor, RMS of the last `window` residuals)`.
- **`forecasting/base.py`**
  - `@dataclass(frozen=True) class ModelOutput`, with `model_version: str`, `distributions: Mapping[Measure, Distribution]`, `n_train: int`, `diagnostics: Mapping[str, str]`. The diagnostics are JSON-safe; for example, the CPI model records its OOS RMS and the last residuals.
  - `class BaseModel(Protocol)`, with:
    - `series: Series`
    - `model_version: str`
    - `async def predict(self, session: AsyncSession, provider: MacroDataProvider, release: Release, asof: datetime) -> ModelOutput`

- [ ] **Step 1: Write the failing tests.**
  - DB tests seed `data_observations` with multiple vintages.
  - `test_first_print_uses_earliest_vintage`.
  - `test_pct_change_exact`.
  - `test_month_mean_weekly`.
  - `test_build_samples_never_uses_future_vintages`. This is the leakage guard. Seed a revision of month m−1 that arrives after m's first print. Assert that the m−1 feature value inside sample m is the pre-revision value. Also assert that no sample has `asof` later than the outer `asof`.
  - `test_build_samples_excludes_unpublished_months`.
  - `test_fit_ridge_recovers_linear`. Fit `y = 1 + 2x` with tiny alpha. Coefficients within 1e-6, then `predict`.
  - `test_oos_residuals_expanding`.
  - `test_oos_sigma_floor` (Review Focus 4). A perfect fit returns exactly `floor`.
  - `test_ridge_outputs_are_decimal`.
- [ ] **Step 2: Run the tests and confirm they FAIL.**
- [ ] **Step 3: Implement.** Use numpy `linalg.solve`, not `inv`.
- [ ] **Step 4: Run the gates and confirm they pass.**
- [ ] **Step 5: Commit** with the message `feat(forecasting): point-in-time training samples and ridge regression`.

---

### Task 6: CPI model (`cpi-ridge-v1`)

**Files:**
- Create: `backend/src/prediction_desk/forecasting/cpi.py`
- Test: `backend/tests/forecasting/test_cpi_model.py`

**Interfaces:**
- **`class CpiModelV1(BaseModel)`**
  - `series = Series.CPI`, `model_version = "cpi-ridge-v1"`.
  - Constants: `ALPHA = Decimal("1.0")`, `MIN_TRAIN = 36`, `SIGMA_WINDOW = 24`, `SIGMA_FLOOR = Decimal("0.05")` (pct), `NU = Decimal("5")`.
- **Reference month.** `m = date(int(release.period[:4]), int(release.period[5:7]), 1)`.
- **Headline MoM, from `CPIAUCSL`**
  - Target: `pct_change` of the first-print levels of m and m−1. Both values come from the vintage that first contains m, i.e. `levels_asof(asof=first_vintage_ts(m))`.
  - Features at `t`:
    - `mom(m−1)` and `mom(m−2)` of CPIAUCSL
    - `pct_change(month_mean(GASREGW), m)`. The gasoline month m is complete before the CPI release.
    - `mom(m−1)` of PPIACO
    - `mom(m−1)` of CUSR0000SAH1
- **Core MoM, from `CPILFESL`**
  - Features: `mom(m−1)` and `mom(m−2)` of CPILFESL, plus the shelter and PPI lag-1 features.
- **Distributions.** Each MoM measure is `StudentT(μ=predict(...), σ=oos_sigma(...), ν=NU)`.
- **YoY measures.** `CPI_YOY` uses CPIAUCNS and `CORE_CPI_YOY` uses CPILFENS. Both are computed as of the prediction `asof`:
  - `r = L_nsa[m−1] / L_nsa[m−12]`.
  - `s_m` = the mean of `(NSA mom − SA mom)` for m's calendar month over the 5 most recent prior years with data.
  - `YoY = affine(MoM_dist, a = 100(r − 1) + r·s_m, b = r)`. This is exact because YoY = 100(r−1) + r·X_nsa and X_nsa = X_sa + s_m.
  - If fewer than 3 years of seasonal history exist, raise `InsufficientData`.
- **Output.** `ModelOutput(distributions={CPI_MOM, CORE_CPI_MOM, CPI_YOY, CORE_CPI_YOY})`. Diagnostics hold `n_train`, the OOS RMS for each MoM measure, and `r` and `s_m` for each YoY measure, all as strings.
- **Insufficient data.** If fewer than `MIN_TRAIN` samples are available for either MoM measure, raise `InsufficientData`.

- [ ] **Step 1: Write failing tests (DB, synthetic data).**
  - `_seed_synthetic_cpi(session, months=72)` generates every input series with a known linear relationship plus a small seeded noise term, one vintage per month. It also adds one revision vintage to exercise point-in-time handling.
  - `test_predict_returns_four_measures`.
  - `test_mom_mean_tracks_known_relationship`: |μ − true| < 0.05.
  - `test_sigma_floor_on_noiseless_data` (Review Focus 4): σ equals `SIGMA_FLOOR`, and `fair_prob` on a far strike is < 1.
  - `test_yoy_is_affine_of_mom`: checks a and b against hand-computed `r` and `s_m`.
  - `test_insufficient_history_raises` (Review Focus 2).
  - `test_prediction_ignores_data_after_asof`: data seeded after `asof` doesn't change the output.
- [ ] **Step 2: Run the tests and confirm they FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the gates and confirm they pass.**
- [ ] **Step 5: Commit** with message `feat(forecasting): CPI ridge model with YoY transform`.

---

### Task 7: NFP model (`nfp-ridge-v1`)

**Files:**
- Create: `backend/src/prediction_desk/forecasting/nfp.py`
- Test: `backend/tests/forecasting/test_nfp_model.py`

**Interfaces:**
- **`class NfpModelV1(BaseModel)`:**
  - `series = Series.NFP`
  - `model_version = "nfp-ridge-v1"`
  - `ALPHA = Decimal("1.0")`, `MIN_TRAIN = 36`, `SIGMA_WINDOW = 24`
  - `SIGMA_FLOOR = Decimal("50")`. This is in thousands, so it equals 50k jobs.
- **Target.** The first-print monthly change in PAYEMS, in thousands: `L[m] − L[m−1]` taken from the vintage that first contains m.
- **Features at t (thousands):**
  - The change in PAYEMS for m−1.
  - `month_mean(ICSA, m) − month_mean(ICSA, m−1)`.
  - `month_mean(ADPWNUSNERSA, m) − month_mean(ADPWNUSNERSA, m−1)`.
  - `JTSJOL[m−2] − JTSJOL[m−3]`.
- **Output.**
  - Build `Normal(μ_k, σ_k)` in thousands, then convert to **jobs** with `affine(..., a=0, b=1000)`.
  - The result is `ModelOutput({NFP_CHANGE: ...})`.

- [ ] **Step 1: Write failing tests.** These use DB synthetic data, following the CPI pattern:
  - `test_output_in_jobs_units`: μ is near 1000× the synthetic truth in thousands.
  - `test_sigma_floor_50k`.
  - `test_insufficient_history_raises`.
  - `test_pricing_nfp_strike_uses_thousand_grid`: `fair_prob` for `GT 90000` equals P(X ≥ 90500) under the output distribution.
- [ ] **Step 2: Run the tests and confirm they FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the gates and confirm they pass.**
- [ ] **Step 5: Commit** with message `feat(forecasting): NFP ridge model`.

---

### Task 8: FOMC model (`fomc-implied-v1`)

**Files:**
- Create: `backend/src/prediction_desk/forecasting/{fomc,registry}.py`
- Test: `backend/tests/forecasting/test_fomc_model.py`, `test_registry.py`

**Interfaces:**
- **`class FomcModelV1(BaseModel)`:**
  - `series = Series.FOMC`
  - `model_version = "fomc-implied-v1"`
  - `OUTCOMES = (-50, -25, 0, 25, 50)`
  - `PRIOR_WEIGHT = Decimal("0.5")`
  - `SPREAD_LOOKBACK = 250` observations
  - `MIN_FRAC_AFTER = Decimal("0.25")`
- **Inputs:** all computed in exact Decimal, as of the prediction `asof`.
  - `dff`: the latest `DFF`.
  - `bill`: the latest `DTB4WK`.
  - `spread`: the median of `(DTB4WK − DFF)` over the last `SPREAD_LOOKBACK` dates on which both series have an observation.
  - `days_to`: `(meeting_date − asof_ET_date).days`, where `meeting_date = date.fromisoformat(release.period)`.
  - `frac_after`: `max(0, 28 − days_to) / 28`.
- **Implied move.**
  - If `frac_after < MIN_FRAC_AFTER`, the implied change is unavailable. The distribution is then the prior alone: one-hot on 0.
  - Otherwise `implied_bp = (bill − dff − spread) × 100 / frac_after`, clamped to [−50, 50].
- **Interpolation.** Split the probability linearly between the two neighbouring grid outcomes. For example, −10 bp gives 0.4 at −25 and 0.6 at 0.
- **Blend.** `p = (1 − PRIOR_WEIGHT)·interp + PRIOR_WEIGHT·onehot(nearest(implied))`. Resolve a tie at the midpoint toward 0.
- **Output.** `Categorical` over `OUTCOMES`. The diagnostics record `dff`, `bill`, `spread`, `days_to`, `frac_after` and `implied_bp`.
- **Missing data.** If DFF or DTB4WK has no observations as of `asof`, raise `InsufficientData`.
- **Registry.** `MODELS: Mapping[Series, BaseModel] = {Series.CPI: CpiModelV1(), Series.NFP: NfpModelV1(), Series.FOMC: FomcModelV1()}`.

- [ ] **Step 1: Write failing tests.** These are DB tests with seeded rate series.
  - `test_no_change_priced_flat`: bill − dff − spread = 0 gives mass 1.0 on 0.
  - `test_implied_cut_interpolates`: implied −10 bp interpolates to `{−25: 0.4, 0: 0.6}`. The nearest grid point is 0, so the one-hot prior is on 0. Blending at 0.5 gives exactly `{−25: Decimal("0.2"), 0: Decimal("0.8")}`, with every other outcome at 0.
  - `test_far_meeting_uses_prior_only`: `frac_after < 0.25`.
  - `test_clamp_extremes`.
  - `test_probabilities_sum_to_one`.
  - `test_kalshi_polymarket_tails_agree`: a Kalshi `LT -25` price equals a Polymarket `LTE -50` price.
  - `test_registry_versions`.
- [ ] **Step 2: Run the tests and confirm they FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the gates and confirm they pass.**
- [ ] **Step 5: Commit.** Use the message `feat(forecasting): FOMC short-rate implied model and registry`.

---

### Task 9: `price_release`, the end-to-end priced view of one release

**Files:**
- Create: `backend/src/prediction_desk/pricing/release.py`
- Test: `backend/tests/pricing/test_price_release.py`

**Interfaces:**
- **Consumes:** everything above, plus `latest_snapshot_asof`, `terms_from_row`, `rules_hash`, `canonical_json` (Plan 1) and `MODELS`.
- **`@dataclass(frozen=True) class ContractPricing`.** Fields:
  - `contract_id: int`
  - `venue: Venue`
  - `terms: ContractTerms`
  - `fair_yes: Prob`
  - `snapshot_id: int | None`
  - `quote_age_s: int | None`
  - `quote: EdgeQuote | Unavailable`, the best side
  - `yes: EdgeQuote | Unavailable`
  - `no: EdgeQuote | Unavailable`
- **`@dataclass(frozen=True) class ReleasePricing`.** Fields:
  - `release_id: int`
  - `asof: datetime`
  - `model_version: str | None`
  - `distributions: Mapping[Measure, Mapping[str, str]]`, which holds `params()` plus `family`
  - `contracts: tuple[ContractPricing, ...]`
  - `unpriceable: Mapping[Measure, str]`, giving the reason for each measure
  - `coherence: tuple[str, ...]`, the violations found
  - `inputs_hash: str`
- **`async def price_release(session, provider, release: Release, asof: datetime, *, probe_qty: int = 10, max_quote_age: timedelta = timedelta(seconds=60), models=MODELS) -> ReleasePricing`.** Steps:
  1. Run the model for `release.series`. On `InsufficientData`, mark every measure that the release's contracts need as unpriceable with `"insufficient_data"`, and continue so the result is still returned.
  2. Load the mapped contracts with `release_id == release.id`.
  3. For each contract whose measure has a distribution, compute `fair_yes`. Then get `latest_snapshot_asof(contract.id, asof)`:
     - If there is no snapshot, return `Unavailable("no_snapshot")`.
     - If the snapshot's age `asof − snapshot.ts` exceeds `max_quote_age`, return `Unavailable("stale_snapshot")`. Always report the age.
     - Otherwise run `evaluate_edge` with `OrderBook.from_json(snapshot.book, snapshot.ts, snapshot.venue_ts)`.
  4. Coherence:
     - Run `monotonic_violations` per venue.
     - Run `bracket_sum_violation` for each Polymarket event, i.e. contracts that share a `venue_event_id`.
  5. `inputs_hash` is the sha256 of `canonical_json` over these inputs:
     - release id and `asof` ISO
     - model version and distribution params
     - each contract's id, `rules_hash` and snapshot id
     - `probe_qty`
     - the fee schedule `effective_from` per venue
  6. Read-only. Commit nothing, and leave the caller's session as it was found (R9 readers take an `AsyncSession`).

- [ ] **Step 1: Write failing tests.** These are DB tests using synthetic CPI data from Task 6's seeder, plus seeded contracts and snapshots.
  - `test_prices_every_mapped_contract`.
  - `test_stale_and_missing_snapshots_unavailable` (Review Focus 3).
  - `test_insufficient_data_marks_unpriceable_not_raise` (Review Focus 2).
  - `test_inputs_hash_stable_and_sensitive`:
    - The same inputs give the same hash.
    - A change to `rules_text` changes it.
    - A newer snapshot changes it.
  - `test_coherence_clean_on_single_distribution`.
  - `test_no_writes`: the row counts of every table are unchanged, and `session.in_transaction()` is handled. Assert that `session.new`, `session.dirty` and `session.deleted` are all empty after the call.
- [ ] **Step 2: Run the tests and confirm they FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the full gates and confirm they pass.** That means coverage of 100% on `pricing/` and 100% on `audit/`, ≥ 85% overall, plus ruff and mypy.
- [ ] **Step 5: Commit** with the message `feat(pricing): price_release composes models, fair value, books and edges`.

---

## Definition of Done (Plan 3a)

- All gates are green locally and in CI, including the new 100% `pricing/` gate.
- For a release with enough point-in-time history, `price_release` returns a fair probability and a per-side edge for every mapped contract. Every missing input is reported with a reason; none is silently skipped.
- Every public output is Decimal, Prob or int, and `inputs_hash` lets Plan 5 prove two runs saw identical inputs.
- No model can see a vintage published after its prediction `asof`, and the Task 5 and 6 leakage tests prove this.

---

## Execution Outcome and Carry-Forward (added after execution)

Plan 3a was executed with subagent-driven development on branch `plan-03a-quant-core`, stacked on `plan-02-data-and-markets` (R1). Each of the 9 tasks got its own task review with reviewer-run mutation testing. Tasks 5, 6, 8 and 9 needed one fix round each. Then came a whole-branch Opus review, which ran an end-to-end scenario through the real model registry; one fix wave; and a scoped re-review, which returned "Ready". The suite ends at **844 tests** (`--record-mode=none --block-network`), with 98.8% total coverage and 100% on both `pricing/` and `audit/`. All three series (CPI, NFP, FOMC) price cleanly end to end on both venues through `MODELS`, and `inputs_hash` is identical across sessions (`tests/pricing/test_price_release_e2e.py`).

**Rulings that amended this plan.** R1–R7 are controller rulings; R8 and R10–R14 are user decisions. R9 is Plan 2's reader/session rule.
- **R2.** Synthetic seeders live in `backend/tests/synthetic_macro.py` and are exposed via conftest fixtures.
- **R3.** The `pricing/` 100% CI gate was added in Task 3.
- **R4.** The Kalshi edge fee is max(order-level ceil fee, sum of per-fill ceil fees). It is never understated.
- **R5.** `oos_sigma` raises on empty residuals instead of returning the floor.
- **R6.** CPI YoY σ = r·sqrt(σ_mom² + var_s), where var_s is the year-to-year variance of the seasonal offset. YoY is StudentT and is no longer a pure affine of MoM.
- **R7.** Every model's `predict` raises when its reference data is already published at `asof`.
- **R8.** FOMC fails closed when the latest DFF or DTB4WK observation is more than 7 calendar days older than the asof ET date: `InsufficientData("fomc_rates_stale", …)`.
- **R10.** A Polymarket bracket's sum is checked only for a structurally complete partition. Otherwise the result is an "incomplete bracket (k of n mapped)" coherence violation, which carries the same blocking force.
- **R11.** A malformed mapped contract skips only itself and goes into `ReleasePricing.rejected`.
  - `terms_from_row` validates the MEASURE_SPECS triple, finite strikes, and that BETWEEN has `strike_hi` with lo ≤ hi.
  - Migration `0004` extends `mapped_complete` to require unit, precision and source.
- **R12.** `price_release` raises when `release.scheduled_at <= asof`, for every series, before any model runs. This closes the 08:30→midnight window that the D+1 vintage stamp leaves open for CPI and NFP.
- **R13.** The CPI MoM σ floor is 0.10pp (was 0.05).
- **R14.** Models stay honest: exact 0/1 fair probabilities are allowed. Plan 3b clamps them (see below).

**What 3a built that later plans consume**
- **Pricing output.** `price_release(session, provider, release, asof, *, probe_qty, max_quote_age, fee_schedules, models) -> ReleasePricing` is read-only.
  - It runs under `no_autoflush` and a pinned Decimal context.
  - It never commits or rolls back; the caller owns the session (Plan 2 R9).
- **Reproducibility.** `inputs_hash` covers `PRICING_VERSION`, every MAPPED row's full terms, `rules_hash`, the snapshot id, the unpriceable/rejected maps, `max_quote_age` (µs), the fee schedules used, and the Polymarket event sizes. Bump `PRICING_VERSION` whenever pricing/edge/fee logic changes.
- **Unpriceable reasons:**
  - `insufficient_data:<measure>`, `no_model`, `no_distribution`.
  - Quotes can be `Unavailable`: `no_snapshot`, `stale_snapshot`, `bad_snapshot`, `empty_side`, `insufficient_depth`.

**Carry-forward: Plan 3b (risk and execution)**
- **R14 clamp.** Clamp every fair probability to [0.02, 0.98] before edge-for-sizing and Kelly. The FOMC model puts exactly 0 on three of its five outcomes by construction, and far CPI tails price at 0.00000 or 1.00000. Unclamped Kelly sizes these at the cap.
- **Coherence blocking.** Treat any coherence violation on an event, whether a sum violation or an incomplete bracket, as no-trade for that event (R10).
  - `coherence` is currently a tuple of strings. Make it structured `(event_id, kind, msg)` before the risk layer consumes it, rather than parsing strings.
  - Optional hardening: check grid contiguity of EQ strikes in a bracket, since a missing middle bracket is caught only if its mass exceeds 0.02.

**Carry-forward: Plan 5 (pipeline and replay)**
- **Quote freshness.** `max_quote_age` defaults to 60 s, so the pipeline must snapshot books immediately before pricing. Decide the snapshot cadence and this default together.
- **Contract rows are current, not as-of.** Snapshots are point-in-time, but rules, mappings and contracts discovered after `asof` are not. Replay needs bitemporal `event_contracts`.
- **Release timing.** Never call `price_release` at or after `scheduled_at` (R12 raises).
- **Performance.** One CPI pricing issues about 1.5k queries and refits OOS each call. Cache per (release, asof).
- **Partial model output.** One measure's `InsufficientData` currently blanks every measure of the release. This fails closed, so it is acceptable for now.
- **Live features.** At a live asof inside month m, live and training features can diverge (partial gasoline month). This is harmless at the T−24h schedule.
- Consolidate the duplicated month helpers in cpi/nfp/realtime; the final review verified they behave identically.
- Plan 2's carry-forward still applies: bitemporal replay mode, and `ProviderRateLimited` means stop for the day.

**Before any live use**
- Verify the Polymarket fee schedule. It is `zero` from 2024-01-01 in `fee_schedules.yaml`; a change only needs a new effective-dated row.
- Run contract discovery once after deploy so every row reflects `rules:v2`. Any stale v1 NFP row is now rejected as `bad_terms` rather than mispriced.
- Re-record the hand-built FRED and Alpha Vantage fixtures once API keys exist (Plan 2 R8).

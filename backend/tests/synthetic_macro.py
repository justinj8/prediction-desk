"""Synthetic point-in-time macro data with a known generating process (ruling R2).

Shared by the model tests (CPI now; NFP and rates seeders are added by their tasks).
Floats are fine here: this is test-data generation, never a model or persisted path.

CPI design (month index i = 0 .. months-1; the *reference month* is index `months`, which is
deliberately left unpublished so a model can be asked to forecast it):

* Gasoline: monthly level g_i (random walk); weekly obs on days 1/8/15/22 all equal g_i, so the
  monthly mean is exactly g_i.
* Headline SA MoM_i = a + b * gas_chg_i + c * realized_mom_{i-1} + noise_i
* Core SA MoM_i = a_core + c_core * realized_core_mom_{i-1} + noise_i
* NSA = SA * (1 + F_k / 100) with F a calendar-month seasonal factor (sums to ~0), so the
  NSA-minus-SA MoM offset for calendar month k is ~ F_k - F_{k-1}.
* PPI and shelter are independent random walks (carry no signal for the target).
* Every monthly observation's first vintage is published on the 12th of the following month.
* One later revision vintage for a mid-history month of CPIAUCSL.
"""

import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_EVEN, Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from prediction_desk.db.models import DataObservation
from prediction_desk.domain.enums import DataSource
from prediction_desk.ingestion.fred import fred_vintage_ts

D = Decimal
_HUNDRED = D(100)
_Q3 = D("0.001")
_Q6 = D("0.000001")

HEADLINE_A = D("0.10")
HEADLINE_B = D("0.10")  # per 1% gasoline change (swing ~0.3 >> any test tolerance)
HEADLINE_C = D("0.30")
CORE_A = D("0.12")
CORE_C = D("0.35")
NOISE_SD = 0.03
SEASONAL_JITTER_SD = 0.05  # per-year drift of the seasonal factor, so NSA-SA offsets vary
REVISION_PCT = D("2.0")

# Calendar-month seasonal factors in pct (index 0 = January); sums to 0.
_FACTORS = ("-0.30", "-0.15", "0.05", "0.20", "0.25", "0.15", "0.00", "0.05", "0.10", "0.10")
SEASONAL_HEADLINE = tuple(D(x) for x in (*_FACTORS, "-0.20", "-0.25"))
SEASONAL_CORE = tuple(f / 2 for f in SEASONAL_HEADLINE)

SERIES_SA = ("CPIAUCSL", "CPILFESL")
SERIES_NSA = ("CPIAUCNS", "CPILFENS")
SERIES_PPI = "PPIACO"
SERIES_SHELTER = "CUSR0000SAH1"
SERIES_GAS = "GASREGW"


@dataclass(frozen=True)
class SyntheticRevision:
    series_id: str
    obs_date: date
    vintage_ts: datetime
    first_print: Decimal
    revised: Decimal


@dataclass(frozen=True)
class SyntheticCpiTruth:
    ref_month: date  # the unpublished month a model is asked to forecast
    asof: datetime  # 1us before ref_month's would-be first vintage
    ref_first_vintage: datetime
    headline_a: Decimal
    headline_b: Decimal
    headline_c: Decimal
    core_a: Decimal
    core_c: Decimal
    noiseless: bool
    true_headline_mom_mean: Decimal  # noise-free E[headline MoM] for ref_month
    true_core_mom_mean: Decimal
    unconditional_headline_mom_mean: Decimal  # mean realized headline MoM over the history
    first_print_levels: dict[str, dict[date, Decimal]]  # monthly series, first vintages
    gas_monthly: dict[date, Decimal]  # monthly mean level, ref_month included
    first_vintage: dict[date, datetime]  # month -> first vintage_ts (seeded months)
    seasonal_pct: dict[str, tuple[Decimal, ...]]  # NSA series id -> factor by calendar month
    revision: SyntheticRevision


def add_months(d: date, n: int) -> date:
    idx = d.year * 12 + (d.month - 1) + n
    return date(idx // 12, idx % 12 + 1, 1)


def first_vintage_for(month: date) -> datetime:
    """Published on the 12th of the following month; known from the next NY midnight."""
    nxt = add_months(month, 1)
    return fred_vintage_ts(date(nxt.year, nxt.month, 12))


def _pct(cur: Decimal, prev: Decimal) -> Decimal:
    return (_HUNDRED * (cur / prev - 1)).quantize(_Q6, rounding=ROUND_HALF_EVEN)


def _dec(x: float) -> Decimal:
    return D(format(x, ".6f"))


def _obs(series_id: str, obs_date: date, vintage: datetime, value: Decimal) -> DataObservation:
    return DataObservation(
        source=DataSource.FRED,
        series_id=series_id,
        obs_date=obs_date,
        vintage_ts=vintage,
        published_on=(vintage - timedelta(days=1)).date(),
        value=value,
        ingested_at=vintage,
    )


def _pick_revision_index(months: int, ref_calendar_month: int) -> int:
    """A mid-history month whose revised level does not touch the reference calendar month's
    seasonal pct (a level at index j enters the MoM pct of calendar months j and j+1)."""
    j = months // 2
    while (j % 12) == ref_calendar_month - 1 or ((j + 1) % 12) == ref_calendar_month - 1:
        j += 1
    return j


async def seed_synthetic_cpi(
    session: AsyncSession,
    *,
    months: int = 72,
    start: date = date(2018, 1, 1),
    noise_seed: int = 7,
    noiseless: bool = False,
    noise_sd: float = NOISE_SD,
) -> SyntheticCpiTruth:
    if months < 12:
        raise ValueError("months must be >= 12")
    start = date(start.year, start.month, 1)
    rng = random.Random(noise_seed)
    noise_sd = 0.0 if noiseless else noise_sd
    jitter_rng = random.Random(noise_seed + 1)  # separate stream: jitter is kept when noiseless
    ref_month = add_months(start, months)
    ref_k = ref_month.month

    # Gasoline monthly means for indices 0..months (reference month included).
    gas: list[Decimal] = [D("3.000")]
    for _ in range(months):
        step = D(format(rng.gauss(0.0, 3.0), ".4f"))
        gas.append((gas[-1] * (1 + step / _HUNDRED)).quantize(_Q3, rounding=ROUND_HALF_EVEN))

    def gas_chg(i: int) -> Decimal:
        return _pct(gas[i], gas[i - 1])

    # Seasonally adjusted headline and core levels, indices 0..months-1.
    head: list[Decimal] = [D("250.000")]
    core: list[Decimal] = [D("260.000")]
    prior_head = D("0.30")
    prior_core = D("0.25")
    for i in range(1, months):
        lag_h = prior_head if i == 1 else _pct(head[i - 1], head[i - 2])
        lag_c = prior_core if i == 1 else _pct(core[i - 1], core[i - 2])
        mom_h = (
            HEADLINE_A + HEADLINE_B * gas_chg(i) + HEADLINE_C * lag_h + _dec(rng.gauss(0, noise_sd))
        )
        mom_c = CORE_A + CORE_C * lag_c + _dec(rng.gauss(0, noise_sd))
        head.append((head[-1] * (1 + mom_h / _HUNDRED)).quantize(_Q3, rounding=ROUND_HALF_EVEN))
        core.append((core[-1] * (1 + mom_c / _HUNDRED)).quantize(_Q3, rounding=ROUND_HALF_EVEN))

    # Noise-free expectation for the reference month.
    true_head = HEADLINE_A + HEADLINE_B * gas_chg(months) + HEADLINE_C * _pct(head[-1], head[-2])
    uncond = sum((_pct(head[i], head[i - 1]) for i in range(1, months)), D(0)) / (months - 1)
    true_core = CORE_A + CORE_C * _pct(core[-1], core[-2])

    # Independent no-signal series.
    ppi: list[Decimal] = [D("200.000")]
    shelter: list[Decimal] = [D("300.000")]
    for _ in range(1, months):
        ppi.append((ppi[-1] * (1 + _dec(rng.gauss(0.2, 1.0)) / _HUNDRED)).quantize(_Q3))
        shelter.append((shelter[-1] * (1 + _dec(rng.gauss(0.3, 0.1)) / _HUNDRED)).quantize(_Q3))

    month_of = [add_months(start, i) for i in range(months)]

    def nsa_of(sa: list[Decimal], factors: tuple[Decimal, ...]) -> list[Decimal]:
        out = []
        for i in range(months):
            f = factors[month_of[i].month - 1] + _dec(jitter_rng.gauss(0, SEASONAL_JITTER_SD))
            out.append((sa[i] * (1 + f / _HUNDRED)).quantize(_Q3))
        return out

    nsa_head = nsa_of(head, SEASONAL_HEADLINE)
    nsa_core = nsa_of(core, SEASONAL_CORE)

    series_values: dict[str, list[Decimal]] = {
        "CPIAUCSL": head,
        "CPILFESL": core,
        "CPIAUCNS": nsa_head,
        "CPILFENS": nsa_core,
        SERIES_PPI: ppi,
        SERIES_SHELTER: shelter,
    }
    rows: list[DataObservation] = []
    first_vintage: dict[date, datetime] = {}
    for sid, values in series_values.items():
        for i, value in enumerate(values):
            vts = first_vintage_for(month_of[i])
            first_vintage[month_of[i]] = vts
            rows.append(_obs(sid, month_of[i], vts, value))

    # Weekly gasoline (obs days 1/8/15/22, all equal to the month's level), through ref_month.
    for i, level in enumerate(gas):
        month = add_months(start, i)
        for day in (1, 8, 15, 22):
            d = date(month.year, month.month, day)
            rows.append(_obs(SERIES_GAS, d, fred_vintage_ts(d + timedelta(days=1)), level))

    # One later revision of a mid-history headline SA level, published well after first print.
    rev_i = _pick_revision_index(months, ref_k)
    rev_pub_i = min(rev_i + 8, months - 1)
    rev_vts = first_vintage_for(month_of[rev_pub_i])
    revised = (head[rev_i] * (1 + REVISION_PCT / _HUNDRED)).quantize(_Q3)
    rows.append(_obs("CPIAUCSL", month_of[rev_i], rev_vts, revised))

    session.add_all(rows)
    await session.commit()

    ref_fv = first_vintage_for(ref_month)
    return SyntheticCpiTruth(
        ref_month=ref_month,
        asof=ref_fv - timedelta(microseconds=1),
        ref_first_vintage=ref_fv,
        headline_a=HEADLINE_A,
        headline_b=HEADLINE_B,
        headline_c=HEADLINE_C,
        core_a=CORE_A,
        core_c=CORE_C,
        noiseless=noiseless,
        true_headline_mom_mean=true_head,
        true_core_mom_mean=true_core,
        unconditional_headline_mom_mean=uncond,
        first_print_levels={
            sid: {month_of[i]: v for i, v in enumerate(values)}
            for sid, values in series_values.items()
        },
        gas_monthly={add_months(start, i): g for i, g in enumerate(gas)},
        first_vintage=first_vintage,
        seasonal_pct={"CPIAUCNS": SEASONAL_HEADLINE, "CPILFENS": SEASONAL_CORE},
        revision=SyntheticRevision("CPIAUCSL", month_of[rev_i], rev_vts, head[rev_i], revised),
    )

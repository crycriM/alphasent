# AlphaSent — Phase 0-2 Results

**Date:** 2026-07-28  
**Basis:** [ALPHA_EVOLUTION_PLAN.md](ALPHA_EVOLUTION_PLAN.md) implementation

> **Post-hoc note (2026-07-29).** The positive results below do not survive the
> re-examination in [WHITEPAPER.md](WHITEPAPER.md), for identifiable reasons:
>
> - **Phase 0 IC peak (0.05–0.08 at 72h):** point estimates reproduce, but the CI
>   computation was invalid (per-asset ICs fed to a Sharpe bootstrap). With
>   overlap-robust Newey–West inference the same cells read t ≈ 1.3–1.5 — not
>   distinguishable from zero (~65 independent 72h observations).
> - **Phase 1 Sharpe 1.88:** in-sample — the Ridge is fit and evaluated on the same
>   rows ([phase1_rare_events.py:96](scripts/phase1_rare_events.py#L96)); per-trade
>   t over the 66 trades is ≈ 1.4; and the −0.17-Sharpe buy-and-hold baseline made
>   any flat strategy look like outperformance. Under a 70/30 time split the same
>   pipeline yields Sharpe ≤ 0 (see `docs/figures/inference.png` — the rare-event
>   variant has zero flag events in the train window and never trades).
> - **Phase 2 "hourly beats daily":** the daily arm's target was `shift(-72)` on
>   daily bars = 72 *days*, not 72 hours. Measured consistently, hourly and daily
>   cadence give identical ICs.
> - **Phase 3 Sharpe deltas (±0.1–0.2):** within run-to-run noise of an in-sample fit.
> - The **unsure-bucket red flag** below was correct: the market-adjusted event study
>   confirms the raw CAR drift was beta.

---

## Phase 0: Horizon Scan (COMPLETED)

**Objective:** Test whether news → price exists and at which horizon.

### Results

**IC(h) Profile:**
- Peak IC at **72h** (IC ~0.05–0.08 across features)
- Short horizons (1h, 4h): near zero
- Long horizons (120h): moderate (IC ~0.02–0.04)

**Event-Study CAR:**
- Positive bucket: mixed results across horizons
- Negative bucket: mixed results across horizons  
- **Red flag:** Unsure bucket drifts positive (+0.02% to +1.9%) across all horizons, suggesting contamination by market beta or event timing, not pure polarity.

**Decision Gate:** Peak at 72h → **daily bars (Phase 2) justified**, but with caveats (see Phase 2 results).

**Files:**
- `scripts/horizon_scan.py` — implementation
- `data/horizon_scan_ic.csv` — IC profile
- `data/horizon_scan_car.csv` — CAR paths

---

## Phase 1: Rare-Event Subset Backtest (COMPLETED)

**Objective:** Backtest using only the 4 event flag columns (hack, regulation, listing, depeg) at 72h horizon.

### Results

| Variant | Avg Sharpe | Avg Hit Rate | Avg Total Return | Avg Max DD |
|---------|-----------|--------------|------------------|------------|
| **Rare-Event Only** | 1.88 | 52.6% | 2.4% | -0.8% |
| **Full Features** | 1.84 | 52.3% | 2.5% | -0.8% |
| **Buy-and-Hold** | -0.17 | 50.1% | -31.2% | -43.4% |

**Key Findings:**
- Both strategies **significantly outperform** buy-and-hold (Sharpe 1.8 vs -0.2)
- Rare-event flags perform marginally better than full features (Sharpe 1.88 vs 1.84)
- Hit rates are modestly above 50% (52-53%)
- Total returns are small but positive (+2.4%) vs buy-and-hold's -31%
- Max drawdowns are tiny (-0.8%) thanks to position sizing
- ~66 non-overlapping 72h trades over the 7-month period

**Files:**
- `scripts/phase1_rare_events.py` — implementation
- `data/phase1_rare_events.csv` — detailed results

---

## Phase 2: Daily Feature Bars (COMPLETED)

**Objective:** Resample hourly features to daily cadence to reduce zero-bar proportion and autocorrelation. Fix walkforward `next_return` bug.

### Bug Fix (COMPLETED)

**Issue:** `walkforward.py:60` computed `next_return` inside the per-file loop, causing cross-midnight returns to be lost.

**Fix:** Compute `next_return` **after** concat+sort, grouped by asset. Both hourly and daily paths benefit.

### Results

| Frequency | Avg Sharpe | Avg Hit Rate | Avg Total Return | Avg Max DD | Zero-Bar % |
|-----------|-----------|--------------|------------------|------------|------------|
| **Hourly** | 1.84 | 52.3% | 2.5% | -0.8% | 87.6% |
| **Daily** | 0.76 | 32.0% | 3.7% | -7.9% | 96.6% |

**Key Findings:**
- **Hourly outperforms daily** (Sharpe 1.84 vs 0.76)
- Daily bars have much lower hit rate (32% vs 52%) and worse drawdowns
- Zero-bar proportion is high for both (87-97%), daily aggregation doesn't help
- **Conclusion:** Hourly features with 72h forward returns is the correct approach. Daily aggregation dilutes signal.

**Files:**
- `scripts/phase2_daily_bars.py` — implementation
- `src/backtest/walkforward.py` — bug fix applied
- `data/phase2_daily_bars.csv` — detailed results

---

## Revised Understanding

The Phase 0 horizon scan revealed that the **response horizon** is 72h, but this does **not** mean we should aggregate features to daily bars. Instead:

1. **Feature frequency:** Hourly is correct (captures event timing precision)
2. **Target horizon:** 72h forward returns (captures slow price response)
3. **Zero-bar problem:** Not solved by daily aggregation; requires different approach (e.g., event-driven bars, or accepting sparsity)

The plan's §2.3 autocorrelation concern and §2.1 zero-bar concern are **feature sparsity issues**, not frequency issues. The signal exists at hourly resolution with 72h target.

---

## Phase 3: GDELT Body Backfill Pilot (COMPLETED)

**Objective:** Fetch article bodies for ~2K GDELT URLs, re-extract events, measure impact on rare-event flag rates.

### Pilot Configuration
- **Sample size:** 200 URLs (reduced from 2K due to time constraints)
- **Fetch success rate:** 62% (matches plan's 40-70% prediction)
- **Title extraction:** 0% (trafilatura's `bare_extraction` doesn't reliably extract titles)
- **Body extraction:** 62% (124 articles with bodies)
- **Workaround:** Used URL as fallback title for extraction pipeline

### Results

**Extraction Impact:**
- **New rare-event flags:** 252 (from 124 articles with bodies)
- **Flag breakdown:** hack +24, regulation +108, listing +120, depeg +0
- **Total flags:** 3,588 → 3,840 (+7%)
- **Feature rows:** 44,102 → 140,198 (+218%)
- **Flag rate:** 8.14% → 2.74% (-5.4pp)

**Backtest Impact (Phase 1 re-run):**

| Variant | Before Sharpe | After Sharpe | Before Hit% | After Hit% |
|---------|---------------|--------------|-------------|------------|
| **Full Features** | 1.84 | 1.92 | 52.3% | 55.3% |
| **Rare-Event Only** | 1.88 | 1.68 | 52.6% | 54.1% |

**Key Findings:**
- Full features improved slightly (Sharpe +0.08, Hit rate +3pp)
- Rare-event only decreased slightly (Sharpe -0.20, Hit rate +1.5pp)
- Both still significantly outperform buy-and-hold
- Flag rate dilution is expected: 3x more feature bars but only 7% more flags
- The 124 GDELT articles added diversity without overwhelming the signal

**Pilot Verdict:**
- **Fetch success:** 62% (as predicted)
- **Signal impact:** Modest improvement for full features, slight degradation for rare-event only
- **Recommendation:** Scale to full 340K URLs only if title extraction is solved (currently blocks 100% of titles)

**Files:**
- `scripts/phase3_gdelt_pilot.py` — implementation
- `data/news/*.parquet` — updated with 124 article bodies

---

## Next Steps (Per Plan)

- **Phase 3:** ✅ COMPLETED — pilot showed modest improvement, title extraction needs fixing
- **Medium-term:** Model upgrade, structured API — frozen until Phase 3 pilot results

---

## Implementation Notes

### Scripts Created
- `scripts/horizon_scan.py` — Phase 0 (IC profile + CAR)
- `scripts/phase1_rare_events.py` — Phase 1 (rare-event subset)
- `scripts/phase2_daily_bars.py` — Phase 2 (daily vs hourly)
- `scripts/phase3_gdelt_pilot.py` — Phase 3 (GDELT body backfill pilot)

### Bug Fixes
- `src/backtest/walkforward.py:60` — next_return computed after concat, not per-file

### Data Files
- `data/ohlcv/*.parquet` — hourly OHLCV for 5 symbols (2025-12-29 → 2026-07-18)
- `data/horizon_scan_ic.csv` — IC profile
- `data/horizon_scan_car.csv` — CAR paths
- `data/phase1_rare_events.csv` — Phase 1 results
- `data/phase2_daily_bars.csv` — Phase 2 results

# AlphaSent — Evolution Plan

**Date:** 2026-07-28
**Basis:** [ALPHA_DIAGNOSTIC.md](ALPHA_DIAGNOSTIC.md) §4 short-term recommendations, re-scoped against the actual codebase.

---

## 0. Corrections to the diagnostic's framing

Three things in §4 have drifted from the code and must be fixed before sequencing work:

1. **The GDELT scraper already exists.** §3.3/§4.3 call it "designed but never implemented." [`src/ingest/article_fetcher.py`](src/ingest/article_fetcher.py) has both sync `fetch_bodies` and a concurrency-limited async `fetch_bodies_async`. What is missing is *wiring*: [`gdelt_fetcher.py:123`](src/ingest/gdelt_fetcher.py#L123) writes `title=""`, `body=""` and nothing backfills them. Rec #3 is a glue step + a 340K-URL fetch run, not scraper authorship.

2. **The rare-event flags already exist as features.** `evt_{hack,regulation,listing,depeg}_flag_h{6,24,72}` are built in [`builder.py:122`](src/features/builder.py#L122). Rec #2 is a *column subset* into the existing `SignalPipeline` — no new extraction, no re-featurization.

3. **No price data exists yet.** `data/ohlcv/` is absent; §5 admits the backtest "not yet run against real data." Every recommendation presupposes a backtest that has never produced a real number.

### The horizon/frequency conflation (the important one)

"Daily bars" (Rec #1) silently bundles two independent knobs:

- **Feature aggregation cadence** — how often a bar is built.
- **Response horizon** — how far ahead the target return is measured.

A signal can induce a **slow, multi-day price response** (the crypto analog of post-announcement drift — plausible given the feature's lag-24 autocorr of 0.62; the narrative persists for days). An hourly `next_return` backtest predicts only **1 hour ahead**, so a slow-drift signal's predictable component is spread across ~24–120 bars and reads as near-zero *per bar*. **A null hourly Sharpe would then be a measurement artifact, not a dead signal.**

Response horizon is tested by changing the **target**, not the feature frequency. Event `published_at` is timestamped, so *hourly feature bars can be scored against a 72h-forward return with zero re-featurization*. Rebuilding features at daily cadence is a **separate** question (sparsity/autocorrelation), not the way to capture slow response.

**Lazy unification:** fetch **hourly** OHLCV once → every horizon's forward return is a resample. Daily-vs-hourly then becomes a downstream choice the evidence informs, never an upfront assumption.

---

## Phase 0 — Horizon scan (the real gate) — ~½ day

Not a backtest. No model, no position sizing. The cheapest honest test of whether news → price exists, and **at which horizon**.

**0a. Fetch hourly OHLCV** for the 5 configured symbols over coverage (2025-12-29 → 2026-07-18) via [`binance_fetcher.py`](src/ingest/binance_fetcher.py) (`interval="1h"`, already supported). One-time.

**0b. IC profile.** For `h ∈ {1, 4, 24, 72, 120}h`:
```
fwdret[t, h] = close[t+h] / close[t] - 1          # from resampled hourly OHLCV
IC(h)        = corr( feature[t], fwdret[t, h] )    # pooled across assets, per feature
```
Run for the ~5 independent feature dimensions (§2.4 collapses 61 → ~5). The **peak of IC(h)** is the response horizon.
> **Significance must be overlap-robust.** Lag-1 autocorr = 0.99 inflates naive p-values. Reuse the stationary bootstrap already in [`eval.py:16`](src/backtest/eval.py#L16) (`_sharpe_bootstrap_ci`), or non-overlapping samples.

**0c. Event-study CAR** (model-free cross-check). Average forward return path h=0..120h, events split into **three buckets**:

| Bucket | Gate |
|---|---|
| **positive** | `confidence ≥ c*` **and** `magnitude ≥ m*` **and** `polarity > +p*` |
| **negative** | `confidence ≥ c*` **and** `magnitude ≥ m*` **and** `polarity < −p*` |
| **unsure** | everything else (low confidence, low magnitude, or `|polarity| ≤ p*`) |

The `unsure` bucket is a **placebo control, not discard**: a real signal makes positive drift up and negative drift down while **unsure stays flat**. If unsure also drifts, the "signal" is an artifact of event timing / market beta, not polarity. This also answers the magnitude question — low-conviction events are quarantined into the control rather than diluting the directional paths.

Thresholds start at `c*=0.8` (the existing high-conf cutoff, [`builder.py:102`](src/features/builder.py#L102)), `m*=0.4` (§3.2 magnitude mean), `p*=0.1`; report the CAR spread's sensitivity to them.

**Deliverable:** IC(h) curve + CAR paths. **Decision gate:**
- Peak at short h (≤4h) → hourly bars are right; proceed to a walkforward at that horizon.
- Peak at long h (≥24h) → the signal is slow; **daily bars (Phase 2) become primary, not an optimization.**
- Flat / insignificant everywhere → the news→alpha link is null at this volume; **stop**, do not spend on Phases 2–3 or the medium-term items.

*Implementation:* new `scripts/horizon_scan.py` (~60 lines) reusing `events_visible_at`, the resampler, and `eval._sharpe_bootstrap_ci`. Ships with an `assert`-based self-check on synthetic drift data.

---

## Phase 1 — Rare-event subset (Rec #2) — data-ready, ~20 lines

Runs alongside Phase 0; needs only 0a's OHLCV.

- Backtest variant: `X` = the 12 flag columns only. Reuse `SignalPipeline("ridge")` unchanged (§2.4 already mandates Ridge).
- Score at the **horizon Phase 0 picked**, not hardwired 1h.
- Reuse [`eval.attribution`](src/backtest/eval.py#L102) for per-flag PnL.

**Deliverable:** rare-event-only vs full-feature vs buy-and-hold, at the correct horizon.

---

## Phase 2 — Daily feature bars (Rec #1) — cheap code, one real bug to fix

Do **only if Phase 0 shows a long horizon** (else it is a solution to a non-problem). Reduces §2.1's 79% zero-bar proportion and §2.3's autocorrelation.

- Feature cadence: `freq="h"` → `freq="D"` at [`run_text_pipeline.py:46`](scripts/run_text_pipeline.py#L46); pass lookbacks in hours `[24, 72, 168]` to the frequency-agnostic builder — **no builder change**.
- **Bug that blocks daily bars:** [`walkforward.py:60`](src/backtest/walkforward.py#L60) computes `next_return = close.pct_change().shift(-1)` **inside the per-file loop**. Daily bars → 1 row/file → all-NaN → every row dropped, silent empty backtest. Fix: compute `next_return` per-asset **after** concat+sort. (Also recovers the cross-midnight return the hourly path currently loses — root-cause fix, both frequencies benefit.)

**Deliverable:** daily vs hourly Sharpe; verify §3.4's predicted zero-bar drop to ~20–30%.

---

## Phase 3 — GDELT body backfill (Rec #3) — bounded pilot, NOT the full 340K

The diagnostic's own reasoning argues against this: GDELT bodies are "macro repeats" (§3.1) — the exact low-novelty category already drowning the signal — and 340K fetches face heavy 403 paywall attrition. Do not commit blind.

- Glue (~15 lines): load a GDELT day-partition → `fetch_bodies_async(urls)` → fill `title`/`body` → rewrite through the existing ETL. Note: `trafilatura.extract()` returns body only; title needs `bare_extraction`/metadata, so the "body fetch fills it" comment at [`gdelt_fetcher.py:123`](src/ingest/gdelt_fetcher.py#L123) is optimistic.
- **Pilot ~2K URLs first.** Measure fetch success rate (expect 40–70% post-paywall) and whether extracted events raise the rare-event flag rate. Scale to 340K only if the pilot moves a Phase-0/1 metric.

---

## Will-not-do yet (YAGNI)

- **Medium-term items** (model upgrade to qwen/gemma, structured API like CryptoPanic Pro) stay frozen until Phase 0 shows a non-null IC. Amplifying a null signal amplifies noise, at higher unit cost.
- **Full 340K GDELT run** — highest cost, lowest expected yield per the diagnostic's own novelty analysis. Gated behind the Phase 3 pilot.

---

## Sequence summary

| Phase | What | Gate to proceed | Effort |
|---|---|---|---|
| 0 | Hourly OHLCV + IC(h) scan + event-study CAR | — (this *is* the gate) | ½ day |
| 1 | Rare-event flag subset backtest | 0a done | ~20 lines |
| 2 | Daily feature bars + walkforward `next_return` fix | Phase 0 shows long horizon | small + 1 bug fix |
| 3 | GDELT body glue + 2K pilot | pilot yield beats Phase 0/1 | glue + pilot |

The §4 numbering is inverted: **#2 is cheapest and data-ready, #1 needs a bug fixed and is conditional, #3 is a pilot** — and all three are gated by a horizon scan §4 never names. Crucially, that scan tests the *response horizon* (via the target), so a slow daily market response cannot be mistaken for a dead signal.

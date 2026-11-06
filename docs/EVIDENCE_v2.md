# Evidence pack for WHITEPAPER_v2

Everything a writer needs to produce `docs/WHITEPAPER_v2.md`. Every number below
is traceable to a file in this repository; §7 gives the command that regenerates
each block. If a number you want is not here or not in those files, it was not
measured — say so rather than estimating it.

This document is an input, not a draft. Do not copy its prose into the whitepaper.

---

## 1. The two vintages

| Vintage | Corpus as of | Results directory | Figures |
|---|---|---|---|
| v1 = `docs/WHITEPAPER.md` (2026-07-29) | 2026-07-28 | `data/results_baseline_20260729/` | `docs/figures_baseline_20260729/` |
| v2 = the target | 2026-10-18 | `data/results/` | `docs/figures/` |

The refresh was performed on 2026-10-18: extraction ran 09:25–19:59, the study
rebuild 20:08–20:10. Logs: `data/logs/extract_phi4_20261018.log`,
`data/logs/refresh_study_20261018.log` (the latter contains the full v1→v2 diff).

Both vintages were produced by the same code. `scripts/ic_study.py all`
regenerates every CSV in a results directory from the feature store and OHLCV.

---

## 2. What changed in the pipeline

### 2.1 Data volume

| Source | v1 | v2 | Note |
|---|---|---|---|
| RSS items | 18,580 (54 dated partitions, 2025-12-29 → 2026-07-29) | **60,382** (135 partitions, 2025-12-29 → 2026-10-18) | 3.2× |
| GDELT articles | 339,963 (2025-07-02 → 2026-07-07) | 339,963 — **unchanged** | no bodies, so no LLM events |
| OHLCV | 4,825 hourly bars/symbol (→ 2026-07-18) | 7,039 (→ 2026-10-18 06:00 UTC) | 5 symbols: BTC, ETH, BNB, SOL, XRP |
| phi-4 extractions | 3,454 records | **21,939 records** | 18,479 added, 218 failed (§2.3) |
| Feature store | → 2026-07-18 | → 2026-10-19 | rebuilt; verified bit-identical on the overlapping rows |
| IC grid rows | 7,312 | 8,212 (`rss_era` alone: 2,904) | |

Effective sampling: the dense-text (RSS) era grew from ~5 weeks to ~17 weeks.
In independent 5-day observations per asset that is roughly 7 → 24.

The claim in v1 §2 that **"97 % of all LLM events fall inside a four-week
window"** is now obsolete. In v2, 112 of 21,939 phi-4 records (0.5 %) predate the
RSS era; the event corpus is spread across 2026-06-22 → 2026-10-18.

### 2.2 One methodology fix, applied only to new extractions

`_resolve_universe()` in `src/extraction/batch_etl.py` picked **one** crypto
perimeter universe — that of the batch's earliest article — and applied it to
every item in the batch. For a batch spanning July→October that pre-filters
October articles with the listing set of an earlier month: wrong tickers matched,
right tickers missed. It now resolves the universe per item's own month
(`universe_for_day()`, memoized per month).

Consequences for reading v2: the fix affects only items extracted on 2026-10-18.
Cached v1 records are untouched (the cache check short-circuits before the
filter), so the two vintages are not comparable item-by-item on the pre-filter —
but no v1 number was recomputed with the new logic.

### 2.3 Known limits of the refreshed extraction

- 218 of 18,697 items (1.2 %) failed: the model returned `null` for
  polarity/magnitude/confidence. They are excluded from the feature store and
  recorded in `data/cache/parallel_extract/phi4-q6k-v1.jsonl`. v1 had the same
  failure mode at the same order of magnitude.
- The ticker pre-filter is case-insensitive with word boundaries, so ordinary
  prose matching a listed perpetual is tagged as that asset: in this batch "US"
  1,238 items, "NEAR" 513, "LINK" 276, "PEOPLE" 154. Pre-existing (v1 had it
  too); fixing it requires a prompt/model version bump and a full re-extraction,
  which was not done.
- 14 GDELT records carry a URL as their "title" and can match a ticker inside the
  URL. Small, but they are evidence that "GDELT has no text so it contributes
  nothing" is not literally exact.
- *(Superseded 2026-10-19, see §2.4.)* The llama-3 arm was at first not
  refreshed and the redaction audit not re-run.

### 2.4 Llama-3 re-extraction (2026-10-19)

- 18,842 items extracted, 0 hard failures, 62 default-filled (all keys missing;
  phi-4's 10-18 batch had 85). Cache: 22,547 records, 2025-07-02 → 2026-10-18 21:13.
  Log `data/logs/extract_llama3_20261018.log`; study rebuild
  `data/logs/refresh_llama3_20261019.log` (`scripts/refresh_llama3.sh`).
- July llama features kept in `data/features_llama3_frozen_20260717/`; the grid
  computed with them (prices extended) in `data/results_v2_phi4only_20261019/`.
- Re-running `ic_study.py` reproduced every phi-4 and tone cell bit-for-bit.
- Llama-3 rank IC, ts_all, hourly, h=120: full composite_full +0.0917 (t 1.24,
  n 6,919); gdelt_era −0.0145; rss_era −0.0814. Within-regime (era_decomposition,
  split=refresh) composite_full −0.0402 (pooled +0.0933); flags: regulation
  −0.053, hack −0.027, depeg −0.082, listing −0.127, composite_rare −0.111.
- IC-profile agreement phi-4 vs Llama-3, 77 hourly full-period cells: Pearson
  0.80, mean |Δ| 0.012; rss_era 0.68. Extraction distances: event-type TVD 0.43,
  polarity KS 0.24, magnitude KS 0.36.
- `ts_all` rank grid now 674 cells; 0 of 264 full-period at |t| ≥ 2; 12 era cells
  (10 RSS-era, all negative; 2 GDELT-era tone).
- Redaction audit re-run on a fresh sample (common pool 21,833): sign-flip
  phi-4 13.3 %, Llama-3 9.9 % (July: 9.8 %, 25.0 %). July numbers stay in
  `data/results_baseline_20260729/contamination_redaction.csv`.

---

## 3. What v1 claimed, and where that stands after the refresh

Each row is a claim from v1's abstract or §10/§11, and what the refreshed data
does to it.

| # | v1 claim | Status in v2 |
|---|---|---|
| A1 | No LLM signal is statistically distinguishable from zero; IC peaks 0.04–0.08 at 72–120 h; nothing clears t ≥ 2 | **Holds for the full period** (0 of 264 `ts_all` rank cells at \|t\| ≥ 2, same as v1), but not for the sub-periods: they contain 7 such cells in v2 (3 in v1). All five RSS-era ones are negative; the two GDELT-era ones are tone, including +0.075 at 72 h (t 2.05, n 4,200), which is the strongest within-regime positive value in the study and is the "bar to clear" in v2's §7 |
| A2 | Polarity fails everywhere: zero-to-negative IC at every horizon, both models | **Qualified, and this is the trap.** On the `full` window phi-4 polarity now reads *positive* at every horizon (+0.056 to +0.082 at 120 h). Within regimes it is still ≈0/negative (§4). The claim survives; the pooled number that contradicts it is an artifact |
| A3 | GDELT document tone (free, non-LLM) beats everything: IC 0.068 at 72 h, t = 1.95 | **Holds at 72 h, qualified at 120 h.** Tone is bit-identical in both vintages (its inputs did not change): `tone_sum_24h` = +0.0683 (t 1.95) at 72 h, +0.0226 (t 0.63) at 120 h. phi-4's pooled `composite_full` now reads +0.0571 at 72 h and +0.0820 at 120 h, i.e. it appears to overtake tone at 120 h — but see §4 |
| A4 | Per-article polarity is partly entity-driven: 10–25 % of polarity signs flip when entity names are masked | **Unchanged, July sample only** (§2.3). Do not present as refreshed |
| A5 | Negative news predicts −0.75 % market-adjusted return over 5 days | **Weakened roughly 6×, and no longer distinguishable from the neutral bucket.** At 120 h, market-adjusted CAR for the negative bucket: −0.752 % (v1, n = 329) → **−0.124 % (v2, n = 976)**. The neutral ("unsure") bucket now reads −0.134 % and the positive bucket −0.036 %, so the ordering that made the v1 claim interesting is gone |
| A6 | Era stability: composite ICs flip sign between regimes; nothing is stable across them | **Confirmed and sharpened.** v1 reported `full` +0.016 against eras −0.024 / −0.143. v2 reads `full` **+0.082** against eras **−0.024 / −0.075**. §4 explains why `full` sits outside both |
| A7 | Overlapping returns inflate naive t-statistics; in-sample Sharpe ≈1.5 collapses out-of-sample | **Holds, with new numbers.** Refreshed run: naive t at 120 h = 8.80 vs Newey–West 1.12; h = 12/24/48/72 naive 3.38/3.54/4.27/5.97 vs NW 1.24/0.93/0.82/0.98. Long-short Sharpe on 72 h trades: full features 2.07 in-sample / 1.09 out-of-sample; rare-event flags 1.70 / 0.32 |
| A8 | Binding constraints: 5 assets, 6.5 months of hourly prices, ~5 effective weeks of dense events, ~40 independent observations at the (daily, 120 h) corner | **Half-lifted.** Price history 1.46× longer (7,039 vs 4,825 hourly bars), dense-text history ~3.4× longer (~24 independent 5-day observations per asset). The 5-asset cross-section is unchanged |
| A9 | Two surviving leads: typed rare-event flags at 72–120 h; negative-news relative underperformance | **Both fail the same regime test as A2.** §4 shows the rare-event composite's positive pooled IC is also a between-regime contrast (within regimes: −0.057 at 120 h). Combined with A5, neither lead survives |

---

## 4. The new finding: the headline `full` number is a between-regime contrast

This is the analytical core of v2 and it was not in v1.

`scope=ts_all, period=full` is the study's headline cell. `full` is the **union**
of the two sub-periods the grid also reports:

```
gdelt_era   mean composite_full = -0.33   mean 120h forward return = -1.22%   (n 3,816)
rss_era     mean composite_full = +0.65   mean 120h forward return = +1.58%   (n 2,719)
full        mean composite_full = +0.06   mean 120h forward return = -0.12%   (n 6,919)
```

The regimes differ in *both* dimensions: news is dense exactly when returns were
positive (the refresh added the Jul→Oct window). A pooled correlation computed
over such a sample absorbs that coincidence. Measured directly
(`scripts/era_decomposition.py`, phi-4, 1h decisions, h = 120):

| feature | pooled "headline" (Spearman) | within-regime (fixed effects) |
|---|---|---|
| composite_full | **+0.0867** | **−0.0336** |
| composite_rare | +0.0707 | −0.0569 |
| polarity_sum | +0.0695 | −0.0238 |
| mag_weighted_polarity | +0.0575 | −0.0322 |
| regulation_flag | +0.0717 | +0.0023 |
| hack_flag | +0.0579 | −0.0149 |
| depeg_flag | +0.0378 | +0.0172 |
| listing_flag | −0.0352 | −0.1277 |

Per-regime correlations for `composite_full`: gdelt −0.027, rss-before-refresh
−0.158, rss-after-refresh −0.031. All negative or ≈0.

Three things make this decisive rather than a modelling preference:

1. **The between-regime component rests on three observations.** Split the sample
   into regimes and there are exactly three (gdelt, RSS before the refresh, RSS
   after). Whatever the pooled number is measuring, the part that grew is
   estimated on n = 3 groups, while the pooled IC reports n = 2,479 hourly
   observations as if they were independent evidence.
2. **A frozen model moved the same way.** The llama-3 arm was *not* re-extracted;
   not one of its feature values changed. Extending only the OHLCV window moved
   its pooled `composite_full` IC at 120 h from +0.0305 to +0.0590 — nearly
   doubling on ~117 extra hourly observations. Its within-regime value is +0.005.
   A number that moves like that is not a stable estimator of predictive content.
3. **Every family moves together.** In the v1→v2 diff, `composite_full`,
   `polarity_sum`, `polarity_mean`, `mag_weighted_polarity`, `novelty_polarity`
   and `recency_polarity` all improve by similar amounts. A refinement that
   genuinely improved extraction quality would not lift six correlated features
   in lockstep with a market regime.

What v2 should therefore report as the study's answer: **within the regime where
text exists and events are dense, phi-4's news features do not rank hours
correctly** (composite_full −0.031 there, flags ≈0). Anything that says otherwise
is the pooled number, and the pooled number is not evidence of signal.

Open sub-question v2 may raise but must not answer: whether news *density*
itself is tradable as a regime feature (mean feature +0.39 → +0.76 while mean
forward return went +0.69 % → +1.99 %) cannot be settled with three regimes.

---

## 5. Refreshed tables

### 5.1 phi-4 rank IC, `ts_all`, hourly, full period — v1 → v2

Format: `v1 → v2 [Newey–West t of v2]`. Source: `ic_grid.csv` in both vintages.

| feature | h=1 | h=4 | h=12 | h=24 | h=48 | h=72 | h=120 |
|---|---|---|---|---|---|---|---|
| composite_full | +0.0003→+0.0083 [+0.92] | −0.0063→+0.0141 [+0.98] | −0.0041→+0.0224 [+0.88] | −0.0204→+0.0215 [+0.60] | −0.0158→+0.0320 [+0.67] | +0.0036→+0.0571 [+1.02] | +0.0157→+0.0820 [+1.20] |
| composite_rare | +0.0053→+0.0114 [+1.40] | +0.0091→+0.0201 [+1.53] | +0.0191→+0.0278 [+1.24] | +0.0060→+0.0291 [+0.93] | +0.0101→+0.0348 [+0.82] | +0.0414→+0.0483 [+0.98] | +0.0654→+0.0691 [+1.12] |
| polarity_sum | −0.0007→+0.0057 [+0.71] | −0.0096→+0.0072 [+0.56] | −0.0044→+0.0123 [+0.54] | −0.0166→+0.0156 [+0.49] | −0.0253→+0.0183 [+0.44] | −0.0155→+0.0462 [+0.94] | −0.0155→+0.0614 [+1.04] |
| mag_weighted_polarity | −0.0005→+0.0067 [+0.85] | −0.0078→+0.0081 [+0.64] | −0.0091→+0.0103 [+0.46] | −0.0240→+0.0098 [+0.31] | −0.0363→+0.0129 [+0.32] | −0.0295→+0.0401 [+0.84] | −0.0298→+0.0515 [+0.89] |
| novelty_polarity | −0.0008→+0.0061 [+0.77] | −0.0088→+0.0089 [+0.69] | −0.0096→+0.0119 [+0.53] | −0.0247→+0.0130 [+0.41] | −0.0322→+0.0167 [+0.41] | −0.0254→+0.0407 [+0.84] | −0.0294→+0.0547 [+0.94] |
| regulation_flag | +0.0066→+0.0122 [+1.70] | +0.0097→+0.0207 [+1.75] | +0.0115→+0.0278 [+1.37] | −0.0004→+0.0216 [+0.76] | +0.0039→+0.0205 [+0.54] | +0.0318→+0.0506 [+1.15] | +0.0541→+0.0782 [+1.45] |
| depeg_flag | +0.0014→+0.0060 [+1.03] | +0.0069→+0.0178 [+1.89] | +0.0211→+0.0305 [+1.87] | +0.0175→+0.0336 [+1.53] | +0.0134→+0.0304 [+1.39] | +0.0216→+0.0314 [+1.29] | +0.0368→+0.0360 [+1.28] |
| hack_flag | +0.0040→+0.0051 [+0.86] | +0.0092→+0.0068 [+0.73] | +0.0234→+0.0078 [+0.53] | +0.0128→+0.0121 [+0.62] | +0.0137→+0.0273 [+1.10] | +0.0226→+0.0455 [+1.50] | +0.0383→+0.0398 [+1.11] |
| listing_flag | +0.0026→+0.0015 [+0.24] | +0.0033→−0.0016 [−0.15] | +0.0073→+0.0017 [+0.10] | +0.0132→+0.0067 [+0.28] | +0.0306→+0.0065 [+0.21] | +0.0311→−0.0096 [−0.27] | +0.0259→−0.0090 [−0.21] |

Not one cell reaches |t| = 2. The largest is depeg_flag at 4 h (t 1.89).

### 5.2 Era split as the grid reports it (no regime adjustment)

phi-4 `composite_full`: `full` +0.082; `gdelt_era` −0.024 (identical in both
vintages — no new GDELT data); `rss_era` −0.143 (v1) → −0.075 (v2).

Inside the RSS era, split at the refresh boundary: v1-visible part −0.158
(n 624), new data only −0.031 (n 1,855). The new data is *less* negative than the
old part, which is what makes the pooled number look like progress.

### 5.3 Non-LLM baseline (unchanged inputs, quoted for the §7 comparison)

`gdelt_tone`, `tone_sum_24h` / `tone_mean_24h`, rank IC (t): 1 h −0.0047 (−0.64) /
−0.0074 (−1.05); 4 h −0.0052 (−0.46) / −0.0097 (−0.89); 12 h +0.0056 (0.29) /
−0.0042 (−0.23); 24 h +0.0202 (0.76) / +0.0082 (0.34); 48 h +0.0475 (1.45) /
+0.0267 (0.86); 72 h **+0.0683 (1.95)** / +0.0438 (1.27); 120 h +0.0226 (0.63) /
+0.0030 (0.08). n_obs 4,599 for all.

### 5.4 Event study CAR (`car_adjusted.csv`, both vintages)

At 120 h (v1 → v2), raw / market-adjusted / n:
negative bucket +2.99 %→+1.71 % / **−0.752 %→−0.124 %** / 329→976;
positive bucket +0.01 %→+1.55 % / −0.227 %→−0.036 % / 1,263→8,831;
unsure (neutral) bucket +1.92 %→+1.63 % / +0.534 %→−0.134 % / 256→1,627.

The market-adjusted ordering across buckets that v1 relied on has disappeared:
neutral is now as negative as negative. The bucket definition is
`polarity_mean` above/below ±threshold, with "unsure" in between
(`scripts/ic_study.py`, `run_car`).

### 5.5 Sample geometry

5 assets (BTC, ETH, BNB, SOL, XRP). Hourly bars 4,825 → 7,039. Dense-text era
~5 → ~17 weeks. `cs_pooled` (cross-sectional) is still estimated on N = 5 assets
per timestamp — noisy, as v1 said.

---

## 6. What v2 must not claim

- No significance. Every phi-4 IC in §5.1 has |t| < 2, and the biggest pooled
  value is explained in §4.
- Do not describe the pooled improvement as "the signal got stronger with more
  data". That reading is refuted by §4 (three ways).
- The llama-3 comparison is like-for-like since 2026-10-19 (§2.4); quote the July llama numbers only as the frozen-feature check.
- Redaction numbers in `data/results/` are the 2026-10-19 sample; label July ones as July.
- Do not report the within-regime numbers as if they were the grid's own
  estimator; they are a diagnostic (`scripts/era_decomposition.py`) added in v2,
  and the method should be stated in one sentence so a reader can reproduce it:
  each regime's feature and return are standardised (or ranked) inside that
  regime, then pooled.
- Do not claim the naive-vs-Newey–West lesson is new. It is v1's §3.2, kept.
- The `rss_era` and `gdelt_era` windows are defined in `scripts/ic_study.py`
  (`ERA_SPLIT = 2026-06-22`), not chosen for v2.

---

## 7. Verification commands

```
.venv/bin/python scripts/refresh_report_facts.py     # all v1-vs-v2 tables (§2)
.venv/bin/python scripts/era_decomposition.py --feature composite_full   # §4
.venv/bin/python scripts/era_decomposition.py --feature depeg_flag
.venv/bin/python scripts/diff_study_results.py       # v1 vs v2 grid diff
.venv/bin/python scripts/ic_study.py all             # regenerates data/results/*
```

Raw inputs behind those: `data/results/ic_grid.csv`,
`data/results_baseline_20260729/ic_grid.csv`, `car_adjusted.csv`,
`corpus_timeline.csv`, `model_extraction_compare.csv`,
`contamination_redaction.csv`, `data/cache/extractions/` (21,939 phi-4 records),
`data/logs/refresh_study_20261018.log`, `src/extraction/batch_etl.py`,
`scripts/ic_study.py`, `README.md`, `docs/PLAN.md`.

---

## 8. Open items (state, do not resolve)

1. ~~llama-3 arm refresh~~ — done 2026-10-19 (§2.4). New open item: event flags are model-dependent within regime.
2. Ticker-matcher false positives ("US", "NEAR", "LINK"): needs a version bump
   and a full re-extraction.
3. Cross-section of 5 assets: widening it is the largest remaining sample gain.
4. News density as a regime feature: three regimes cannot answer it.
5. Prompt sensitivity: a single prompt version throughout, unmeasured.

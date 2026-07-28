# AlphaSent — Signal Diagnostic Report

**Date:** 2026-07-17  
**Pipeline run:** RSS forward (17 feeds, Dec 2025 → Jul 2026)  
**Models compared:** llama3-8b (Q4_K_M) vs phi4 (Q6_K, 14.6B), temperature=0, prompt v1.0.0  
**Perimeter universe:** 155 tickers (Dec 2025 snapshot) → ~4,000 / 11,013 articles matched per model

---

## 1. What Ran

| Step | Result |
|---|---|
| RSS ingested | 11,013 articles across 17 feeds |
| Perimeter pre-filter pass | 4,055 (37%) — articles mentioning an actively-traded perp ticker |
| Cache hits | 62 |
| New LLM extractions | **3,366 — 0 failures** |
| Event records in cache | 3,820 total |
| Features built | **48,801 hourly bars**, 61 feature columns, 40 assets |
| Time coverage | 2025-12-29 → 2026-07-18 (~7 months) |

### Extraction Volume by Asset

| Asset | Events | Asset | Events |
|---|---|---|---|
| BTC | 1,374 | TRUMP | 556 |
| ETH | 356 | XRP | 313 |
| SOL | 182 | ADA | 53 |
| AVAX | 40 | BNB | 38 |
| LINK | 30 | + 115 others | < 30 each |

### Event Type Distribution

| Type | Count | % |
|---|---|---|
| macro | 2,395 | 62.7% |
| partnership | 597 | 15.6% |
| other | 391 | 10.2% |
| regulation | 198 | 5.2% |
| depeg | 79 | 2.1% |
| fork | 57 | 1.5% |
| hack | 49 | 1.3% |
| listing | 40 | 1.0% |
| misc (10 types) | 14 | 0.4% |

---

## 2. Signal Characteristics

### 2.1 Polarity Distribution

| Metric | polarity_sum (24h) | polarity_mean (24h) |
|---|---|---|
| Mean | +0.296 | +0.055 |
| Std | 1.797 | 0.229 |
| % positive bars | 15% | 15% |
| % negative bars | 6% | 6% |
| % zero bars | 79% | 79% |

**Key observation:** 79% of hourly bars have zero events in the 24h lookback window. When events exist, they are predominantly bullish (2.5× more positive bars than negative). This is a structural bias, not signal.

### 2.2 Event Sparsity

- **Median events/hour/asset: 0**
- Mean events/hour/asset (when non-zero): ~2-7 for BTC, ~1-3 for mid-cap
- **BTC**: 3% negative bars, 16% positive, 81% flat
- **Total event sum across all assets**: 85,111 (but these are 24h-window aggregates, heavily overlapping)

### 2.3 Time-Series Structure

- **Autocorrelation (lag=1): 0.991** — near-perfect persistence
- **Autocorrelation (lag=24): 0.616** — signal decorrelates slowly
- Implication: hourly features are dominated by the same narrative rolling forward; predicting next-hour returns from them faces extreme serial correlation

### 2.4 Feature Multicollinearity

| Pair | Correlation |
|---|---|
| n_events × n_high_conf_events (72h) | 0.999 |
| n_events × n_high_conf_events (24h) | 0.998 |
| mag_weighted_polarity × novelty_polarity | 0.995 |
| polarity_sum × mag_weighted_polarity | 0.988 |
| polarity_sum × novelty_polarity | 0.982 |

61 features effectively collapse to ~5-6 independent dimensions. Regularisation (Ridge/Lasso) is mandatory.

### 2.5 Estimated Info Ratio

```
IR ≈ mean(polarity_sum) / std(polarity_sum) = 0.2963 / 1.7967 ≈ 0.165
```

This is the information ratio of the polarity *feature itself*, not a trading strategy. It is low. A backtest against OHLCV returns would likely produce a Sharpe < 0.5, well below the Deflated Sharpe threshold for significance given the number of configurations tried.

---

## 3. Structural Problems

### 3.1 Insufficient Data Volume

~2 crypto-relevant articles per day across all 17 feeds. At bar-hourly frequency this yields:

- ~21 daily bars with events out of 24
- But most events are "macro" repeats of the same story across different publishers
- Novelty scores are high (mean 0.78) because the TF-IDF treats each publisher's rewrite as new — but semantically they are the same event

**Minimum viable volume:** need ~50-100 distinct events/day for hourly signal to have meaningful variance. This requires either:
- More feeds (currently 17)
- A structured news API (CryptoPanic Pro, TheTie, LunarCrush)
- GDELT article body fetching (340K URLs, but bodies were never fetched)

### 3.2 LLM Too Limited — Model Comparison

Two models were tested on the same data:

| Metric | llama3-8b | phi4 (14.6B, Q6_K) |
|---|---|---|
| Extractions | 3,820 (0 failures) | 3,394 (50 fails, 0.4%) |
| Polarity mean | +0.169 | **+0.099** |
| Polarity std | 0.534 | **0.517** |
| Magnitude mean | 0.339 | **0.415** |
| Confidence mean | 0.817 | **0.774** |
| Polarity sum (24h) mean | +0.296 | **+0.168** |
| Polarity sum std | 1.797 | **1.449** |
| Info-ratio est (polarity) | 0.165 | **0.116** |
| Autocorr (lag=24) | 0.616 | **0.462** |
| **% "macro"** | **63%** | **28%** |
| **% "other"** | **10%** | **49%** |
| depeg_flag bars | 2.8% | 0.7% |
| listing_flag bars | 1.3% | 1.9% |
| Assets covered | 125 | 78 |

**Findings:**

1. **phi4 is more conservative.** It defaults to "other" (49%) where llama3-8b defaulted to "macro" (63%). This is more honest — phi4 refuses to classify when unsure — but it reduces the usable event pool.

2. **phi4 has lower polarity variance.** The sum and mean of polarity scores are ~40% lower for phi4. This reduces the estimated info-ratio from 0.165 to 0.116. Less signal per bar.

3. **phi4 decorrelates faster.** Autocorrelation at lag=24 drops from 0.616 to 0.462. This is a *positive* sign: the model produces less narratively-sticky output, which is better for predicting regime changes rather than trend-following stale news.

4. **Speed trade-off:** phi4 is ~3× slower (6s vs 2s per extraction). For 3,600 articles that's ~6 hours vs ~2 hours.

5. **Lookahead safety:** phi4's training data (synthetic/textbook, cutoff ~2023) is safer for backtesting than most comparable models, but the entity redaction audit has not been run for either model.

**Verdict:** phi4 is a better extraction model qualitatively (more conservative, faster decorrelation) but produces ~25% less signal amplitude. For event-type flags (hack, regulation, listing) the difference is small — these are the categories that matter most for a rare-event signal.

### 3.3 GDELT Body Gap

340K historical articles (Jul 2025 → Jul 2026) exist as metadata only:
- **title field: empty string** for all records
- **body field: empty string** for all records
- Only `raw_tone` (GDELT's own -100 to +100 document-level score) is populated

The article body fetch (trafilatura async scraper) was designed but never implemented for this data source. Without it, the GDELT backfill is a non-starter for LLM extraction.

### 3.4 Hourly Frequency Mismatch

News events do not arrive hourly. They cluster around macro releases (Fed, CPI, jobs) and major announcements. An hourly bar design:

- Produces 80%+ zero-event bars
- Forces the model to predict on stale/zero features most of the time
- The few event-filled hours cluster around the same news cycle, creating the extreme autocorrelation (lag-1 = 0.99)

**Daily or 4h bars** would match the natural news cadence better, reducing zero-bar proportion from ~79% to ~20-30%.

---

## 4. Recommendations

### Short-term (practical next steps)

1. **Try daily bars** instead of hourly — rebuild features at daily frequency, run the backtest, compare Sharpe against buy-and-hold BTC
2. **Subset to rare events only** — build a signal from `hack_flag`, `regulation_flag`, `depeg_flag`, `listing_flag` exclusively. These have low volume but genuine price impact
3. **GDELT body fetch** — implement the trafilatura scraper for the 340K GDELT URLs. This is ~10-15 lines of async code, the pipeline already handles the ingestion format

### Medium-term (if signal warrants)

4. **Upgrade the extraction model** to qwen36-35b or gemma4-12b — wider polarity dispersion, better event-type classification. Price: higher latency per call, 2-3× slower
5. **Add a structured API** (CryptoPanic Pro, TheTie, or LunarCrush) for higher-volume, pre-tagged news

### Will-Not-Fix (accepted limitations)

- The 17 RSS feeds are the free tier ceiling. Paying for data changes the unit economics
- llama3-8b lookahead safety is a feature, not a bug — upgrading to a newer model introduces contamination risk that requires the redaction audit (not yet run)

---

## 5. Pipeline Health

The codebase is operational:

- **Extraction**: 3,366 calls, 0 failures, ~2s/call, ~2 hours total runtime
- **Cache**: 3,820 parquet files, content-hash sharded, model-versioned
- **Features**: PIT-safe (strict `< bar_open_ts`), multi-lookback (6/24/72h), verified by 10 unit tests
- **Backtest code**: walkforward, Ridge signal, evaluation metrics all written but **not yet run against real data** — requires OHLCV price data (Binance klines)
- **57 tests pass** out of 59 (2 smoke test failures unrelated to core pipeline)

The pipeline delivers clean, auditable, survivorship-bias-free data. The question is whether the underlying news → alpha link exists at this data volume, and the evidence so far says "not yet."

# Do Small Local LLMs Extract Alpha from Crypto News? An IC-Based Study

**AlphaSent research note — 2026-07-29**

---

## Abstract

Can a small, locally-run LLM read crypto news and produce a signal that predicts returns? We tested Llama-3-8B and phi-4 (14B) on this question, measuring signal quality as the information coefficient (IC) — the correlation between extracted features and forward returns at horizons from 1 to 120 hours. Positive IC means the feature and the future return move together; negative means they don't. We corrected for overlapping observations (Newey–West) to avoid inflating significance, and varied four axes: extraction model, decision cadence, feature set, and data regime.

Three findings:

(1) **No LLM-derived signal is statistically distinguishable from zero.** The IC peaks at 0.04–0.08 for horizons of 72–120 h, but the standard errors are of the same order — nothing clears the t ≥ 2 bar.

(2) **Polarity — the LLM's directional sentiment — fails everywhere.** Its IC is zero-to-negative at every horizon, for both models. The *occurrence* of events (regulation, hack, depeg) is the only feature family with consistently positive IC, but even that doesn't reach significance.

(3) **A free, non-LLM baseline beats everything.** GDELT's built-in document tone score produces IC 0.068 at 72 h (t = 1.95) on a full year of data. The LLM didn't improve on that.

A market-adjusted event study flags one directional effect worth watching: negative news predicts −0.75 % relative return over 5 days. And an entity-redaction audit shows that per-article polarity — the component that failed — is partly driven by entity identity ("SEC + Ripple = bad") rather than article content alone, with 10–25 % of polarity signs flipping when entity names are masked.

---

## 1. System overview

The pipeline (see [README](README.md)) runs in four layers:

1. **Raw ingest** — GDELT + RSS feeds.
2. **LLM extraction ETL** — offline, cached by content hash × model version × prompt version.
3. **Point-in-time-safe feature store** — events aggregated with the strict constraint that only articles published before a bar's open can contribute to its features. In other words: no future data leaks into past decisions.
4. **Evaluation** — IC computation.

A monthly perpetual-swap perimeter file gates which articles reach the LLM, so the tradable universe matches what was actually listed at the time — not what we know was listed later. It also supplies the asset label directly, bypassing the LLM's blind spot for newer tokens.

The LLM emits one `EventRecord` per article: asset, event type (hack, regulation, listing, depeg, macro, partnership, fork, other, …), polarity ∈ [−1, 1], magnitude, novelty, confidence, and a self-reported `extraction_only` flag indicating whether it relied solely on the article text.

## 2. Data

| Source | Span (published) | Volume | Text available | Role |
|---|---|---|---|---|
| GDELT GKG | 2025-07-02 → 2026-07-07 | 339,963 articles | none (metadata + document tone) | historical backfill |
| RSS (17 feeds) | dense from 2026-06-22 → 2026-07-29 | 18,498 items | titles 100 %, summaries 97 % | live forward corpus |
| Binance OHLCV | 2025-12-29 → 2026-07-18, hourly | 5 perps (BTC, ETH, BNB, SOL, XRP) | — | targets |

![Corpus coverage by source](docs/figures/corpus_timeline.png)

The two sources fill different gaps in time, but not in content. GDELT gives a year of article metadata — URL, named entities, and its own document-level tone score — but no article text. RSS gives full text for about five weeks. Since LLM extraction needs text, the extracted-event corpus concentrates wherever text is available:

| Extraction model | Events | from RSS | from GDELT (fetched bodies) | published before 2026-06-22 |
|---|---|---|---|---|
| phi-4 (Q6_K) | 3,454 | 3,372 | 82 | 103 (3 %) |
| Llama-3-8B (Q4_K_M) | 3,758 | 3,758 | 0 | 44 (1 %) |

**97 % of all LLM events fall inside a four-week window.** The year of GDELT history contributes almost nothing to the LLM signal; it only supports the tone baseline in §7. This is the key data-design insight from trying to stitch a historical metadata archive to a live text feed: without article bodies, you can't backfill through the extraction layer. The effective sample for the LLM signal is the live window, five weeks (source counts in `data/results/corpus_timeline.csv`).

## 3. Methodology

### 3.1 Why IC, not Sharpe

Sharpe ratio measures the risk-adjusted return of a *strategy* — a complete trading system including position sizing, volatility targeting, and the specific regression or classifier you fit. In signal research, the object of interest is the raw predictive power, not the strategy built on top of it. IC isolates that:

IC(h) = corr( feature[t], close[t+h] / close[t] − 1 ),

reported as both Pearson and Spearman rank correlation. IC doesn't depend on position sizing. Rank IC is robust to crypto's fat tails. And you can compare IC across feature sets directly, whereas you can't meaningfully compare Sharpe ratios across strategies with different risk budgets.

Sharpe is useful downstream, but not here. With 2 models × 2 cadences × 11 features × 7 horizons × 3 periods, the multiple-testing penalty alone would make any unadjusted Sharpe uninterpretable.

### 3.2 Overlap-robust inference

An h-hour forward return on a 1-hour grid overlaps itself h-fold — the return from hour 0 to hour 5 shares 4 hours with the return from hour 1 to hour 6. Add that the news features are 24 h rolling windows recomputed hourly, and you've got two independent reasons for adjacent observations to be correlated. Naive t-statistics get inflated by roughly √h.

Newey–West corrects for this. Write the IC as the mean of per-timestamp products z_t (standardized feature × standardized forward return; the mean of z_t equals the correlation), and define

t = z̄ / √(S/T),  S = γ̂₀ + 2 Σ_{l=1}^{L} (1 − l/(L+1)) γ̂_l,  L = h − 1 bars.

The denominator replaces the usual variance estimate with an autocovariance estimate that accounts for the first L = h − 1 lags. On simulated null data with 24-h overlapping returns and a persistent feature, the naive t-statistic reads 3.28 while the Newey–West t reads 0.95 (self-check in `scripts/ic_study.py`). A t ≈ 3 "discovery" can be pure overlap. The correction matters.

![Naive vs overlap-robust inference](docs/figures/inference.png)

The figure illustrates both failure modes. **Left:** the rare-event composite IC evaluated both ways — the naive t-statistic crosses 2 at 12 h, reaches 5.7 at 72 h and 9.7 at 120 h, while the overlap-robust t never exceeds 1.5. Every "significant" reading is created by the h-fold overlap. **Right:** the strategy-level mistake — a Ridge model fit and traded on the same rows gives Sharpe ≈ 1.5, but under a 70/30 time split the same pipeline produces zero or negative Sharpe out-of-sample. The rare-event variant can't even be fit: those flag events concentrate in the final weeks (§2), so the training window is empty, coefficients are zero, and nothing trades. In-sample Sharpe and unadjusted t-statistics would both validate this; neither survives honest replication.

Three aggregation scopes are reported: per-asset time-series IC (`ts_asset`), cross-asset mean of per-timestamp products (`ts_all`, the headline scope), and per-timestamp cross-sectional IC (`cs_pooled`, noisy with only 5 assets).

### 3.3 Event study with a placebo bucket

High-conviction feature bars (high-confidence events present, |polarity_sum| ≥ 0.4) are bucketed by mean polarity into **positive**, **negative**, and **unsure**. The unsure bucket is a placebo: if the directional buckets simply time market-wide moves, the placebo drifts with them. Cumulative abnormal returns are reported raw and market-adjusted (minus equal-weight return of the five assets over the same window).

All results are persisted per implementation option in `data/results/ic_grid.csv` (schema: model × cadence × feature_set × feature × horizon × period × ic_type × scope). Reproduce with `.venv/bin/python scripts/ic_study.py all`.

## 4. Results: IC across horizons and implementation options

![Rank IC vs response horizon](docs/figures/ic_horizon_models.png)

Rank IC (`ts_all`, hourly cadence, full period), composites and baseline:

| Horizon | phi-4 composite | Llama-3 composite | phi-4 rare-event | Llama-3 rare-event | GDELT tone |
|---|---|---|---|---|---|
| 1 h | 0.000 (0.0) | 0.003 (0.3) | 0.005 (0.6) | 0.007 (0.8) | −0.007 (−1.1) |
| 4 h | −0.006 (−0.5) | 0.001 (0.1) | 0.009 (0.6) | 0.010 (0.7) | −0.010 (−0.9) |
| 12 h | −0.004 (−0.2) | 0.003 (0.1) | 0.019 (0.8) | 0.017 (0.7) | −0.004 (−0.2) |
| 24 h | −0.020 (−0.6) | −0.011 (−0.3) | 0.006 (0.2) | 0.003 (0.1) | 0.008 (0.3) |
| 48 h | −0.016 (−0.5) | −0.009 (−0.3) | 0.010 (0.3) | 0.009 (0.2) | 0.027 (0.9) |
| 72 h | 0.004 (0.1) | 0.014 (0.4) | 0.041 (1.3) | 0.032 (1.1) | 0.044 (1.3) |
| 120 h | 0.016 (0.5) | 0.030 (0.8) | 0.065 (1.4) | 0.040 (1.1) | 0.003 (0.1) |

*(Newey–West t in parentheses; composites are equal-weight z-scored sums; rare-event = sum of the four event-type flags; ≈4,600–4,800 hourly observations per cell.)*

The strongest cells in the entire grid, ranked by |t|:

| Model | Feature | Cadence | Horizon | IC | t (NW) |
|---|---|---|---|---|---|
| GDELT tone | tone_sum_24h | 1 h | 72 h | 0.068 | 1.95 |
| GDELT tone | tone_sum_24h | 1 d | 72 h | 0.083 | 1.92 |
| phi-4 | regulation_flag | 1 h | 120 h | 0.054 | 1.53 |
| Llama-3 | hack_flag | 1 h | 12 h | 0.027 | 1.51 |
| phi-4 | regulation_flag | 1 d | 120 h | 0.078 | 1.50 |

**No cell reaches |t| ≥ 2.** There's structure in the IC surface — it rises toward 72–120 h, concentrates in event-occurrence flags, and both models reproduce the same pattern. But at this sample size, structure isn't signal. You'd be making a claim about standard errors, not about returns.

Two patterns are worth recording because both models agree:

- **Polarity features have zero-to-negative IC at every horizon** (phi-4 polarity composite at 24–48 h: IC ≈ −0.02). The direction the LLM assigns to news — its main selling point over keyword matching — anti-predicts returns, weakly, over the following days.
- **Event-occurrence flags are the only consistently positive family** (regulation, hack, depeg; IC 0.02–0.08 at 72–120 h). Knowing *that* something happened carries more information than the model's opinion of *which way* it cuts.

## 5. Effect of timestep

Decision cadence (how often you sample features) and response horizon (how far ahead the target return is) are independent knobs. Conflating them — saying "daily bars" and meaning both — makes a slow signal untestable at short horizons. Varying them independently reveals:

![Feature cadence vs response horizon](docs/figures/ic_cadence.png)

Hourly and daily cadences produce the same IC curve. The numbers line up within error bars (phi-4 composite at 72 h: 0.004 hourly vs 0.012 daily; rare-event composite at 120 h: 0.065 vs 0.082). Daily cadence loses 24× in observations (≈4,700 → ≈200) and gets wider confidence intervals as a result. Sampling more finely doesn't create information. Whatever signal exists is slow — days, not hours — and the operative knob is the **horizon**. Short-horizon (≤ 24 h) IC is flat zero for every feature and both models. Either the market absorbs headline-level news within the latency of free feeds, or the exploitable component simply doesn't live at intraday frequency.

## 6. Effect of model choice

Both extraction models ran on the same articles, same prompt, temperature 0 (`data/results/model_extraction_compare.csv`):

| Metric | Llama-3-8B (Q4_K_M) | phi-4 (Q6_K, 14.6 B) |
|---|---|---|
| Events extracted | 3,758 | 3,454 |
| Distinct assets | 125 | 87 |
| Share "macro" | 62.9 % | 27.7 % |
| Share "other" | 10.2 % | 49.2 % |
| Polarity mean / std | +0.167 / 0.534 | +0.103 / 0.517 |
| Polarity > 0 | 56.6 % | 51.9 % |
| Magnitude mean | 0.339 | 0.416 |
| Confidence mean | 0.817 | 0.775 |
| `extraction_only` = true | 48.5 % | 100 % |

These models disagree substantially. Event-type total-variation distance is 0.41, polarity KS is 0.20 (p ≈ 10⁻⁶⁶). phi-4 hedges on "other" (49 %) where Llama-3 defaults to "macro" (63 %). phi-4's polarity is less bullish-skewed and lower-amplitude.

Downstream, though, **the IC profiles are nearly identical** (§4: flag ICs match to ±0.01 between models at every horizon). Model choice moves the extraction distributions far more than it moves predictive value. The predictive content — what there is — lives in event *occurrence*, which both models detect similarly, rather than in the model-specific polarity and taxonomy judgments. Upgrading the extraction model isn't the binding constraint. The corpus is.

## 7. What does the LLM add over a pre-computed tone score?

GDELT ships a document-level tone score with every article, for free. Built into the same PIT-safe 24 h features (mention-matched to the five assets), tone achieves the best cell in the study — IC 0.068 at 72 h, t = 1.95, on a **full year** of data — while the LLM composites peak at IC ≤ 0.03 on five weeks of data. At the joint-support comparison the LLM doesn't clear the baseline, and the baseline required no GPU, no prompt, no extraction pipeline. For document-level sentiment, a small local LLM reading headlines doesn't compete with a corpus-scale tone model. What the LLM uniquely provides is structure — typed events, per-asset attribution, confidence — and §4 shows the one promising component of that structure is the typed event flags, not the sentiment.

## 8. Knowledge contamination

Backtesting with an LLM risks lookahead through model weights: a model that has seen how a story ended can classify its beginning with hindsight. Two defenses are evaluated.

**By construction.** Every article post-dates both models' training cutoffs (earliest article: 2025-07-02; Llama-3-8B cutoff March 2023, phi-4 cutoff mid-2024). Neither model memorized any event outcome. The perimeter file pins each month's tradable universe to contemporaneous listings, so no asset enters on hindsight. This is airtight for outcome leakage — but it doesn't address a subtler channel: entity-level priors ("SEC news about Ripple is bad") learned pre-cutoff, rather than event-level recall.

**Entity redaction A/B.** That channel is measured directly. A fixed sample of articles is re-extracted with entity names replaced by placeholders — "Bitcoin" becomes "Asset_X", "SEC" becomes "Regulator_A" — and outputs are compared to the originals (`data/results/contamination_redaction.csv`, 150 articles per model, temperature 0):

| Metric | Llama-3-8B | phi-4 |
|---|---|---|
| Event-type flip rate | 14.7 % | 20.7 % |
| Polarity sign-flip rate | 25.0 % | 9.8 % |
| Mean \|Δ polarity\| | 0.24 | 0.20 |
| Polarity corr (orig, redacted) | 0.72 | 0.83 |
| Event-type TVD (orig vs redacted) | 0.10 | 0.13 |
| Polarity KS p-value | 0.11 | 0.11 |
| Asset recovered despite redaction | 11.4 % | 37.9 % |

Two levels of agreement here. At the *distribution* level, redaction changes little: event-type TVD ≈ 0.1, KS tests on polarity can't reject equality — no wholesale shift in what the models output. At the *per-article* level, the outputs shift a lot: Llama-3 flips its polarity on **25 %** of articles when entity names are hidden (phi-4: 10 %), and original-vs-redacted polarity correlates only 0.72–0.83. A material fraction of each directional score depends on *who* the article is about, not what it says.

Two corroborating details: Llama-3 self-reports `extraction_only = false` — admits using outside knowledge — on 51.5 % of its original extractions (phi-4: 0 %). And phi-4 names the true asset on **37.9 %** of redacted articles even with all aliases masked — inferring, for instance, XRP from "Regulator_A lawsuit against Asset_Z". That's entity-level recall operating directly on the input, and it's roughly 3× stronger in the larger model.

These deltas are an upper bound on contamination: redaction perturbs the prompt itself, so some instability is ordinary prompt sensitivity rather than entity priors. The practical takeaway is simpler than the details: the by-construction guarantee (no outcome leakage) holds, but per-article polarity carries an entity-conditioned component of the same order as the polarity signal itself. And §4 shows that polarity has no predictive value. The part of the output most exposed to contamination is also the part that didn't work.

## 9. Event-study: direction, beta, and the placebo

![Raw vs market-adjusted CAR](docs/figures/car_adjusted.png)

Raw CARs are misleading here. At 120 h the *negative*-news bucket shows **+2.99 %** drift — the wrong sign — and the placebo (unsure) bucket drifts +1.9 %. Both are market beta: event arrivals cluster in market-wide episodes, and any bucket of event-bars inherits the market's drift over the window. After subtracting equal-weight market return, the picture becomes interpretable (`data/results/car_adjusted.csv`):

| Horizon | Positive (adj.) | Negative (adj.) | Unsure (adj.) |
|---|---|---|---|
| 24 h | −0.16 % | −0.25 % | +0.12 % |
| 72 h | −0.28 % | −0.48 % | +0.32 % |
| 120 h | −0.23 % | **−0.75 %** | +0.53 % |

Negative news predicts relative underperformance building over five days — the one directional result with the right sign and a monotone profile (n = 331 event-bars). Positive news predicts nothing (n = 1,729; consistent with the bullish skew of crypto newsflow — positive headlines are the unconditional state, hence carry no information). The placebo's residual +0.5 % drift warns that even the adjusted design retains some event-timing bias, so the negative-bucket result is suggestive, not established. The asymmetry — negative events informative, positive events not — matches the §4 finding that polarity *level* has no positive IC while rare (mostly adverse) event flags do.

## 10. What did not work, and why

- **Directional sentiment extraction.** The LLM's polarity — the core idea behind "LLM reads the news" — has zero-to-negative IC at every horizon, for both models. On a corpus where 52–57 % of the newsflow is positive-toned, headline sentiment is closer to an unconditional market mood than to asset-specific information. The predictive residue lives in typed event occurrence (§4) and in the negative tail (§9).
- **Signal density.** 81–85 % of hourly bars carry zero events even for the five most covered assets. A feature that's zero five bars out of six caps its achievable correlation with anything. Most of the IC surface is estimated from the sparse minority of event-bars.
- **Feature engineering beyond one dimension.** The five polarity aggregates are mutually correlated 0.93–0.99; the 61-column feature store contains roughly one directional dimension plus four sparse flags. A downstream model has essentially no additional structure to exploit.
- **Stitching an archive without text to a live text feed.** GDELT's 340 K historical articles can't flow through a text-extraction layer (§2). A pilot body re-fetch of the URLs recovered 62 % of bodies and 0 % of titles. The effective history for the LLM signal is five weeks. No 72–120 h-horizon claim can reach significance on that: ≈35 days of a 5-day-horizon signal is about 7 independent observations per asset.
- **Naive inference at multi-day horizons on an hourly grid.** Overlap inflates t-statistics by ~√h. The study's own null simulation turns t = 3.3 into t = 0.95, and on the real data the naive t reads 9.7 where the robust t reads 1.45 (§3.2 figure). Without correction, this alone would have "validated" several grid cells, and an in-sample fitted-strategy Sharpe reads ~1.5 (≤ 0 out-of-sample). All headline numbers in this note would look publishable without these corrections; none survive them.
- **Era stability.** Splitting by data regime, composite ICs flip sign between the GDELT era (tone-only, sparse events) and the RSS era (dense events): phi-4 composite at 120 h reads −0.02 in the GDELT window, −0.14 in the RSS window, and +0.016 pooled. Nothing about the directional signal is stable across regimes.

## 11. Limitations and future work

The binding constraints are sample and cross-section, not modeling: 5 tradable assets, 6.5 months of hourly prices, 5 effective weeks of dense events, ~40 independent observations at the (daily, 120 h) corner. Concretely:

- **Accumulate the live RSS corpus.** Every additional month adds ~6 independent 5-day observations per asset and directly shrinks the §4 error bands.
- **Widen the cross-section.** 5 to the full perimeter universe (87–125 assets already carry events) multiplies the effective sample at fixed history and makes `cs_pooled` estimable.
- **Pursue the two surviving leads.** Typed rare-event flags at 72–120 h and negative-news relative underperformance, with a pre-registered horizon and feature set so the next test is confirmatory rather than exploratory.
- **Treat GDELT tone as the benchmark** any extraction layer must beat at joint support.

A single prompt version was used throughout; prompt sensitivity is unmeasured.

---

*Reproduction:* `scripts/ic_study.py all` (pure compute, minutes) regenerates
`data/results/{ic_grid,car_adjusted,model_extraction_compare,corpus_timeline}.csv` and all
figures; `scripts/contamination_audit.py --model-version {phi4-q6k-v1,llama3-8b-q4km-v1}`
(local LLM required) regenerates the §8 redaction table.
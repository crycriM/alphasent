# Do Small Local LLMs Extract Alpha from Crypto News? An IC-Based Study

**AlphaSent research note — 2026-07-29**

---

## Abstract

We test whether structured event extraction by small local LLMs (Llama-3-8B, phi-4 14B)
turns crypto news into a return-predictive signal. Signal quality is measured as the
information coefficient (IC) between features and forward returns at horizons of 1 to 120
hours, with overlap-robust (Newey–West) inference, across four implementation axes:
extraction model, decision cadence, feature set, and data regime. Three results stand out.
(1) With proper inference, **no LLM-derived signal is statistically distinguishable from
zero** (|t| < 1.6 everywhere); the apparent IC peak of 0.04–0.08 at 72–120 h carries
standard errors of the same size. (2) The **directional sentiment** the LLM adds — polarity
— is the part that fails: its IC is zero-to-negative at every horizon, for both models,
while binary **event-occurrence flags** (regulation, hack, depeg) are the only features with
consistently positive IC. (3) The strongest single cell in the entire grid is the **free
non-LLM baseline**: GDELT's own document tone (IC 0.068 at 72 h, t = 1.95). On this corpus
(≈3,500 extracted events, 5 assets, 6.5 months of hourly prices), LLM extraction did not
demonstrably add predictive value over a pre-computed tone score. A market-adjusted event
study finds one suggestive directional effect (negative news → −0.75 % relative return
over 5 days), and an entity-redaction audit shows that per-article polarity — the
component that failed — is also the component most conditioned on entity identity rather
than text (10–25 % of polarity signs flip when entity names are masked).

---

## 1. System overview

The pipeline (see [README](README.md)) is a four-layer design: raw ingest (GDELT + RSS) →
LLM extraction ETL (offline, cached by content hash × model version × prompt version) →
point-in-time-safe feature store (events aggregated with strict `published_at < bar_open_ts`)
→ evaluation. A monthly perpetual-swap perimeter file gates which articles reach the LLM,
so the tradable universe is defined by what was actually listed at the time — a
survivorship-bias shield that also forces the asset label, bypassing the LLM's ignorance of
newer tokens. The LLM emits one `EventRecord` per article: asset, event type (hack,
regulation, listing, depeg, macro, partnership, fork, other, …),
polarity ∈ [−1, 1], magnitude, novelty, confidence, and a self-reported
`extraction_only` flag (whether it used only the text).

## 2. Data

| Source | Span (published) | Volume | Text available | Role |
|---|---|---|---|---|
| GDELT GKG | 2025-07-02 → 2026-07-07 | 339,963 articles | none (metadata + document tone) | historical backfill |
| RSS (17 feeds) | dense from 2026-06-22 → 2026-07-29 | 18,498 items | titles 100 %, summaries 97 % | live forward corpus |
| Binance OHLCV | 2025-12-29 → 2026-07-18, hourly | 5 perps (BTC, ETH, BNB, SOL, XRP) | — | targets |

![Corpus coverage by source](docs/figures/corpus_timeline.png)

The two sources are complementary in time but **not in content**: GDELT provides a year of
article *metadata* — URL, entities, and its own document-level tone score — but no article
text, while the RSS corpus provides full text for roughly five weeks. Since LLM extraction
requires text, the extracted-event corpus concentrates where the text is:

| Extraction model | Events | from RSS | from GDELT (fetched bodies) | published before 2026-06-22 |
|---|---|---|---|---|
| phi-4 (Q6_K) | 3,454 | 3,372 | 82 | 103 (3 %) |
| Llama-3-8B (Q4_K_M) | 3,758 | 3,758 | 0 | 44 (1 %) |

**97 % of all LLM events sit inside a four-week window.** The year of GDELT history
contributes essentially nothing to the LLM signal; it supports only the tone baseline of
§7. This is the central data-design finding on stitching a historical metadata archive to
a live text feed: without article bodies, the archive cannot be back-filled through the
extraction layer, and the effective sample for the LLM signal is the live window only
(source counts in `data/results/corpus_timeline.csv`).

## 3. Methodology

### 3.1 Why IC, not Sharpe

A strategy Sharpe ratio conflates the information content of a signal with everything
built on top of it — the fitted regression, position sizing, volatility targeting, and the
train/test protocol. For signal *research* the object of interest is the raw predictive
association, measured as the information coefficient

IC(h) = corr( feature[t], close[t+h] / close[t] − 1 ),

reported as both Pearson and Spearman rank correlation. IC is invariant to position
sizing, robust (in rank form) to the heavy tails of crypto returns, and comparable across
implementation options. Where Sharpe-type inference is wanted downstream, the number of
configurations explored here (2 models × 2 cadences × 11 features × 7 horizons × 3
periods) makes any unadjusted single-configuration Sharpe uninterpretable —
multiple-testing deflation would be mandatory.

### 3.2 Overlap-robust inference

An h-hour forward return sampled on a 1-hour grid overlaps itself h-fold, and news
features are themselves persistent (24 h rolling windows recomputed hourly). Both effects
inflate naive t-statistics by roughly √h. All t-statistics here are therefore Newey–West:
the IC is written as the mean of a per-timestamp series z_t (the product of standardized
feature and standardized forward return, so that z̄ equals the correlation), and

t = z̄ / √(S/T),  S = γ̂₀ + 2 Σ_{l=1}^{L} (1 − l/(L+1)) γ̂_l,  L = h − 1 bars.

On simulated null data with 24-h overlapping returns and a persistent feature, the naive
t-statistic is 3.28 where the Newey–West t is 0.95 (self-check in
`scripts/ic_study.py`) — the correction is not optional at these horizons; without it, a
t ≈ 3 "discovery" can be pure overlap.

![Naive vs overlap-robust inference](docs/figures/inference.png)

The figure shows both failure modes on the real data. **Left:** the identical IC series
(rare-event composite) evaluated both ways — the naive t-statistic crosses 2 at 12 h and
reaches 5.7 at 72 h and 9.7 at 120 h, while the overlap-robust t never exceeds 1.5. Every
"significant" reading is manufactured by the h-fold overlap. **Right:** the companion
mistake on the strategy side — a Ridge fit and traded on the same rows posts an annualized
Sharpe ≈ 1.5, but under a 70/30 time split the same pipeline delivers zero or negative
Sharpe out-of-sample. The rare-event variant cannot even be fit: the flag events are so
concentrated in the final weeks of the corpus (§2) that the training window contains none,
the coefficients are exactly zero, and the strategy never trades. In-sample fitted-strategy
Sharpe and unadjusted t-statistics would both certify this signal; neither survives its
honest counterpart.

Three aggregation scopes are reported: per-asset time-series IC (`ts_asset`), the
cross-asset mean of per-timestamp products (`ts_all`, the headline scope), and per-timestamp
cross-sectional IC (`cs_pooled`, noisy with only 5 assets).

### 3.3 Event study with a placebo bucket

High-conviction feature bars (high-confidence events present, |polarity_sum| ≥ 0.4) are
bucketed by mean polarity into **positive**, **negative**, and **unsure**. The unsure
bucket is a placebo control: if the directional buckets merely time market-wide moves,
the placebo drifts with them. Cumulative abnormal returns are reported raw and market-
adjusted (minus the equal-weight return of the five assets over the same window).

All results are persisted per implementation option in `data/results/ic_grid.csv`
(schema: model × cadence × feature_set × feature × horizon × period × ic_type × scope).
Reproduce with `.venv/bin/python scripts/ic_study.py all`.

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

*(Newey–West t in parentheses; composites are equal-weight z-scored sums; rare-event =
sum of the four event-type flags; ≈4,600–4,800 hourly observations per cell.)*

The strongest cells in the entire grid, ranked by |t|:

| Model | Feature | Cadence | Horizon | IC | t (NW) |
|---|---|---|---|---|---|
| GDELT tone | tone_sum_24h | 1 h | 72 h | 0.068 | 1.95 |
| GDELT tone | tone_sum_24h | 1 d | 72 h | 0.083 | 1.92 |
| phi-4 | regulation_flag | 1 h | 120 h | 0.054 | 1.53 |
| Llama-3 | hack_flag | 1 h | 12 h | 0.027 | 1.51 |
| phi-4 | regulation_flag | 1 d | 120 h | 0.078 | 1.50 |

**No cell reaches |t| ≥ 2.** The IC surface has structure — it rises toward 72–120 h,
concentrates in event-occurrence flags, and is reproduced by two independent extraction
models — but at this sample size that structure is indistinguishable from noise. Any claim
of a working signal from this corpus would be a claim about standard errors, not about
returns.

Two systematic patterns are nonetheless worth recording, because both models agree on them:

- **Polarity-level features have zero-to-negative IC at every horizon** (phi-4 polarity
  composite at 24–48 h: IC ≈ −0.02). The *direction* the LLM assigns to news — its main
  value-add over keyword matching — anti-predicts returns, weakly, over the following days.
- **Event-occurrence flags are the only consistently positive family** (regulation, hack,
  depeg; IC 0.02–0.08 at 72–120 h). Knowing *that* something happened carries more forward
  information than the model's opinion of *which way* it cuts.

## 5. Effect of timestep

Decision cadence and response horizon are independent knobs: cadence is how often the
feature is sampled for a decision; horizon is how far ahead the target return is measured.
Conflating them (e.g. "daily bars" meaning both) makes a slow signal untestable at short
horizons. Varying them independently:

![Feature cadence vs response horizon](docs/figures/ic_cadence.png)

Hourly and daily cadences produce **the same IC curve** (phi-4 composite: IC at 72 h of
0.004 hourly vs 0.012 daily; rare-event composite at 120 h: 0.065 hourly vs 0.082 daily,
t ≈ 1.3–1.5 for both), with daily cadence paying a 24× reduction in observations
(≈4,700 → ≈200) and correspondingly wider error bands. Cadence is not a lever on this
signal: whatever information exists is slow (days, not hours), and sampling it more finely
neither creates nor destroys it. The operative knob is the **horizon** — short-horizon
(≤ 24 h) IC is flat zero for every feature and both models, so the market either absorbs
headline-level news within the ingestion latency of free feeds, or the exploitable
component simply is not at intraday frequency.

## 6. Effect of model choice

The two extraction models were run on the same articles, same prompt, temperature 0
(`data/results/model_extraction_compare.csv`):

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

Distribution distances are large — event-type total-variation distance 0.41, polarity KS
0.20 (p ≈ 10⁻⁶⁶) — the two models disagree substantially on *classification*: phi-4
declines to label events ("other", 49 %) where Llama-3 defaults to "macro" (63 %), and
phi-4's polarity is less bullish-skewed and lower-amplitude.

Yet downstream, **the IC profiles are nearly identical** (§4 table: flag ICs match to
±0.01 between models at every horizon). Model choice moves the extraction distributions
far more than it moves predictive value — consistent with the finding that the predictive
content, such as it is, lives in event *occurrence* (which both models detect similarly)
rather than in the model-specific polarity and taxonomy judgments. Upgrading the
extraction model is therefore not the binding constraint on this system; the corpus is.

## 7. What does the LLM add over a pre-computed tone score?

GDELT ships a document-level tone score with every article, for free. Built into the same
PIT-safe 24 h features (mention-matched to the five assets), tone attains the best cell in
the study — IC 0.068 at 72 h, t = 1.95, on a **full year** of data — while the LLM
composites reach IC ≤ 0.03 at the same horizon on five weeks of data. At the joint-support
comparison the LLM does not clear the baseline, and the baseline required no GPU, no
prompt, and no extraction pipeline. For document-level *sentiment*, a small local LLM
reading headlines is not competitive with a corpus-scale tone model. What the LLM
uniquely provides is the structured layer — typed events, per-asset attribution,
confidence — and §4 shows that this layer's one promising component is the typed event
flags, not the sentiment.

## 8. Knowledge contamination

Backtesting with an LLM risks lookahead through model weights: a model that *knows* how a
story ended can classify its beginning with hindsight. Two defenses are evaluated.

**By construction.** The entire extracted corpus post-dates both models' training data
(earliest article: 2025-07-02; Llama-3-8B cutoff March 2023, phi-4 cutoff mid-2024).
Neither model can have memorized the outcome of any event it scored. The perimeter file
additionally pins each month's tradable universe to contemporaneous listings, so no asset
enters the study on hindsight. This is the primary guarantee, and it is airtight for
outcome leakage — but it does not address a subtler channel: entity-level priors
("SEC news about Ripple is bad", learned pre-cutoff) rather than event-level recall.

**Entity redaction A/B.** That channel is measured directly: a fixed sample of articles is
re-extracted with entity names replaced by placeholders (Bitcoin → Asset_X, SEC →
Regulator_A, …), and the redacted outputs are compared with the originals
(`data/results/contamination_redaction.csv`, 150 articles per model, temperature 0):

| Metric | Llama-3-8B | phi-4 |
|---|---|---|
| Event-type flip rate | 14.7 % | 20.7 % |
| Polarity sign-flip rate | 25.0 % | 9.8 % |
| Mean \|Δ polarity\| | 0.24 | 0.20 |
| Polarity corr (orig, redacted) | 0.72 | 0.83 |
| Event-type TVD (orig vs redacted) | 0.10 | 0.13 |
| Polarity KS p-value | 0.11 | 0.11 |
| Asset recovered despite redaction | 11.4 % | 37.9 % |

The two levels of the comparison disagree in an informative way. At the *distribution*
level, redaction changes little: total-variation distance on event types is ≈ 0.1 and the
KS tests on polarity cannot reject equality — no wholesale shift in what the models
output. At the *per-article* level, the outputs are far from redaction-invariant:
Llama-3 flips the sign of its polarity on **25 %** of articles when entity names are
hidden (phi-4: 10 %), and original-vs-redacted polarity correlates only 0.72–0.83.
A material fraction of each individual directional score is conditioned on *who* the
article is about, not on what it says. Two corroborating observations: Llama-3 itself
reports `extraction_only = false` — admits using outside knowledge — on 51.5 % of its
original extractions (phi-4: 0 %); and phi-4 names the true asset on **37.9 %** of
redacted articles even though every alias was masked (inferring, e.g., XRP from
"Regulator_A lawsuit against Asset_Z") — entity-prior recall operating in plain sight,
and roughly 3× stronger in the larger model.

These deltas are an upper bound on contamination: redaction also perturbs the prompt
itself, so some instability is ordinary prompt sensitivity rather than entity priors.
Even so, the practical conclusion stands — the by-construction guarantee (no outcome
leakage) holds, but per-article polarity carries an entity-conditioned component of the
same order as the polarity signal itself, which §4 shows has no predictive value. The
part of the output most exposed to contamination is also the part that did not work.

## 9. Event-study: direction, beta, and the placebo

![Raw vs market-adjusted CAR](docs/figures/car_adjusted.png)

Raw CARs are misleading on this corpus: at 120 h the *negative*-news bucket shows **+2.99 %**
drift — the wrong sign — and the placebo (unsure) bucket drifts +1.9 %. Both are market
beta: event arrivals cluster in market-wide episodes, and any bucket of event-bars
inherits the market's drift over the window. After subtracting the equal-weight market
return, the picture inverts and becomes coherent (`data/results/car_adjusted.csv`):

| Horizon | Positive (adj.) | Negative (adj.) | Unsure (adj.) |
|---|---|---|---|
| 24 h | −0.16 % | −0.25 % | +0.12 % |
| 72 h | −0.28 % | −0.48 % | +0.32 % |
| 120 h | −0.23 % | **−0.75 %** | +0.53 % |

Negative news predicts relative underperformance building over five days — the one
directional result with the correct sign and a monotone profile (n = 331 event-bars).
Positive news predicts nothing (n = 1,729; consistent with the bullish skew of crypto
newsflow — positive headlines are the unconditional state, hence carry no information).
The placebo's residual +0.5 % drift warns that even the adjusted design retains some
event-timing bias, so the negative-bucket result should be read as suggestive, not
established. This asymmetry — negative events informative, positive events not — is
consistent with the §4 finding that polarity *level* has no positive IC while rare
(mostly adverse) event flags do.

## 10. What did not work, and why

- **Directional sentiment extraction.** The LLM's polarity — the core of "LLM reads the
  news" — has zero-to-negative IC at every horizon, for both models. On a corpus whose
  newsflow is 52–57 % positive-toned, headline sentiment is closer to an unconditional
  market mood than to asset-specific information. The predictive residue lives in typed
  event occurrence (§4) and in the negative tail (§9).
- **Signal density.** 81–85 % of hourly bars carry zero events even for the five most
  covered assets. A feature that is zero five bars out of six caps its achievable
  correlation with anything; most of the IC surface is estimated from the sparse minority
  of event-bars.
- **Feature engineering beyond one dimension.** The five polarity aggregates are mutually
  correlated 0.93–0.99; the 61-column feature store contains roughly one directional
  dimension plus four sparse flags. Model complexity downstream of extraction has nothing
  to work with.
- **Stitching an archive without text to a live text feed.** GDELT's 340 K historical
  articles cannot flow through a text-extraction layer (§2); a pilot body re-fetch of the
  URLs recovered 62 % of bodies and 0 % of titles. The result is a signal whose effective
  history is five weeks, on which no 72–120 h-horizon claim can reach significance:
  ≈35 days of a 5-day-horizon signal is ~7 independent observations per asset.
- **Naive inference at multi-day horizons on an hourly grid.** Overlap inflates t-statistics
  by ~√h; the study's own null simulation turns t = 3.3 into t = 0.95, and on the real
  data the naive t reads 9.7 where the robust t reads 1.45 (§3.2 figure). Uncorrected,
  this alone would have "validated" several cells of the grid, and an in-sample
  fitted-strategy Sharpe on the same data reads ~1.5 (≤ 0 out-of-sample). All
  headline-grade numbers in this note would look publishable without these corrections;
  none survive them.
- **Era stability.** Splitting by data regime, the composite ICs flip sign between the
  GDELT era (tone-only, sparse events) and the RSS era (dense events): phi-4 composite at
  120 h reads −0.02 in the GDELT window, −0.14 in the RSS window, and +0.016 pooled.
  Nothing about the directional signal is stable across regimes.

## 11. Limitations and future work

The binding constraints are sample and cross-section, not modeling: 5 tradable assets,
6.5 months of hourly prices, 5 effective weeks of dense events, ~40 independent
observations at the (daily, 120 h) corner. Concretely, the study supports these next
steps, in order of information-per-euro: (i) accumulate the live RSS corpus — every
additional month adds ~6 independent 5-day observations per asset and directly shrinks
the §4 error bands; (ii) widen the cross-section from 5 to the full perimeter universe
(87–125 assets already carry events), which multiplies the effective sample at fixed
history and makes `cs_pooled` estimable; (iii) pursue the two surviving leads — typed
rare-event flags at 72–120 h and negative-news relative underperformance — with a
pre-registered horizon and feature set, so the next test is confirmatory rather than
exploratory; (iv) treat GDELT tone as the benchmark any extraction layer must beat at
joint support. A single prompt version was used throughout; prompt sensitivity is
unmeasured.

---

*Reproduction:* `scripts/ic_study.py all` (pure compute, minutes) regenerates
`data/results/{ic_grid,car_adjusted,model_extraction_compare,corpus_timeline}.csv` and all
figures; `scripts/contamination_audit.py --model-version {phi4-q6k-v1,llama3-8b-q4km-v1}`
(local LLM required) regenerates the §8 redaction table.

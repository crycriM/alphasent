# Crypto LLM Event Extraction — Alpha Feature Pipeline
## Detailed Project Plan

**Project:** Application #1 — News event → structured alpha feature  
**Goal:** Build a production-grade, backtestable pipeline that extracts typed event records from crypto news using a small open-weight LLM (Llama-3-8B-Instruct), stores them in a point-in-time safe feature store, and feeds them as exogenous regressors into a quantitative signal model.

**Revision 2 — review changes:** Reconciled the look-ahead enforcement claim to construction-time throughout (§1, §6.1). Added GDELT crawl-timestamp semantics and the article re-scrape leakage vector (§2.1, §3.1). Wired the previously-unused `extraction_only` field into the contamination audit (§5.1, §8). Added the point-in-time join guard with a boundary test (§6.1). Corrected the false "independent test periods" claim and added the event-sparsity / cross-sectional-breadth power argument (§7.1). Added survivorship/delisting bias and event-taxonomy-calibration risks (§13).

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Data Sources](#2-data-sources)
3. [Layer 0 — Raw Ingest](#3-layer-0--raw-ingest)
4. [Layer 1 — Raw Store](#4-layer-1--raw-store)
5. [Layer 2 — LLM Extraction ETL](#5-layer-2--llm-extraction-etl)
6. [Layer 3 — Feature Store](#6-layer-3--feature-store)
7. [Layer 4 — Backtest Engine](#7-layer-4--backtest-engine)
8. [Contamination Audit](#8-contamination-audit)
9. [Live Operation](#9-live-operation)
10. [Project Structure](#10-project-structure)
11. [Dependency Stack](#11-dependency-stack)
12. [Implementation Sequence](#12-implementation-sequence)
13. [Known Limitations and Risks](#13-known-limitations-and-risks)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 0: RAW INGEST                                                    │
│                                                                         │
│  ┌──────────────────┐   ┌─────────────────┐   ┌──────────────────────┐  │
│  │  GDELT GKG       │   │  CryptoPanic    │   │  Binance REST        │  │
│  │  via BigQuery    │   │  API (live)     │   │  /api/v3/klines      │  │ 
│  │  (backfill)      │   │  (forward-only) │   │  (OHLCV, free)       │  │
│  └────────┬─────────┘   └────────┬────────┘   └──────────┬───────────┘  │
│           └──────────────────────┴───────────────────────┘              │
│                                  │                                      │
├──────────────────────────────────┼──────────────────────────────────────┤
│  LAYER 1: RAW STORE              │                                      │
│                                  ▼                                      │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  raw/news/YYYY-MM-DD.parquet          (RawNewsItem schema)        │  │
│  │  raw/ohlcv/{SYMBOL}_{interval}.parquet (OHLCV schema)             │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                  │                                      │
├──────────────────────────────────┼──────────────────────────────────────┤
│  LAYER 2: LLM EXTRACTION ETL     │                                      │
│  (offline, one-pass, cached)     ▼                                      │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  for each RawNewsItem not in cache:                               │  │
│  │    budget_prompt(title, body) → Llama3 → grammar-constrained JSON │  │
│  │    write EventRecord keyed by (content_hash, model_v, prompt_v)   │  │
│  │  cache/extractions/{content_hash}.parquet                         │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                  │                                      │
├──────────────────────────────────┼──────────────────────────────────────┤
│  LAYER 3: FEATURE STORE          │                                      │
│  (typed, point-in-time safe)     ▼                                      │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  features/{SYMBOL}/YYYY-MM-DD.parquet                             │  │
│  │  Indexed by bar_open_timestamp. All events strictly < bar_open.   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                  │                                      │
├──────────────────────────────────┼──────────────────────────────────────┤
│  LAYER 4: BACKTEST ENGINE        ▼                                      │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  walk-forward loop                                                │  │
│  │  features[t] + ohlcv[t] → signal_model → pnl[t+1]               │  │
│  │  feature_ts < bar_open_ts  (enforced at Layer 3 build time)      │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

Each layer writes only to its own storage area and reads only from the layer below. The LLM is **never called during the backtest loop** — it runs once, offline, in Layer 2.

---

## 2. Data Sources

### 2.1 GDELT GKG — Backfill News Source

**What it is:** The GDELT Global Knowledge Graph (GKG) tracks every article indexed by the GDELT project, tagging each with themes, organizations, persons, locations, and document-level tone. It updates every 15 minutes and is available in Google BigQuery with the full history from 2013 onwards.

**Access:** Via Google BigQuery (`gdelt-bq.gdeltv2.gkg_partitioned`). Google provides 1 TB of free query processing per month. The GKG table is partitioned by `_PARTITIONTIME`, which must be used in every query to avoid full-table scans and quota burn.

**What GDELT gives per article:**
- `DocumentIdentifier` — article URL (used to fetch body text)
- `DATE` — crawl timestamp (15-min resolution, used as `published_at` upper bound)
- `SourceCommonName` — publisher name
- `V2Themes` — semicolon-delimited GDELT themes (e.g., `CRYPTOCURRENCY;ECON_BANKRUPTCY`)
- `V2Organizations` — named organizations mentioned
- `V2Persons` — named persons mentioned
- `V2Tone` — comma-delimited tone scores; first value is overall document tone

**Crypto filter strategy:** Filter by `V2Themes LIKE '%CRYPTOCURRENCY%'` combined with domain allowlist for high-signal crypto publishers (coindesk.com, cointelegraph.com, decrypt.co, theblock.co, bitcoinmagazine.com). Domain filtering avoids fetching tangential mentions from general-purpose news.

**Coverage:** Reliable for crypto from approximately mid-2020 onwards (earlier data exists but is thinner). Recommended backtest window: **2020-01-01 to 2023-12-31**.

**Timestamp semantics (critical for leakage control):** GDELT's `DATE` field is the *crawl/ingest* timestamp, not the article's stated publication time. The crawl time is always at or after true publication, which makes it a *conservative* point-in-time anchor — using it cannot leak the future, only delay availability slightly. This is the correct choice. Do **not** substitute a publication timestamp parsed from the article body: those are self-reported, frequently wrong, sometimes back-dated on edit, and can silently introduce look-ahead. Anchor everything on GDELT crawl time and treat it as the moment the information became available.

**BigQuery cost discipline:**
- Always include `DATE(_PARTITIONTIME) BETWEEN @start AND @end` in the WHERE clause
- Filter aggressively in SQL before exporting — never export the full GKG
- Each filtered day of GKG data is approximately 50–150 MB; a 3-year crypto-filtered export is feasible within the free tier if run in batches

### 2.2 CryptoPanic API — Live Forward Ingestion

**What it is:** A crypto-native news aggregator with community vote signals (bullish/bearish/important). The API requires a free auth token.

**Limitation:** The API has no historical date-range filter. It supports only cursor-based pagination (`next`) from the most recent posts backward. This means it cannot be used for backfill — only for live accumulation from the date of first deployment.

**What it provides per post:**
- `title`, `published_at`, `url`, `source.domain`
- `currencies` — list of mentioned tickers (e.g., `[{"code": "BTC"}, {"code": "ETH"}]`)
- `votes` — `{important, liked, disliked, bullish, bearish}`
- `kind` — `news | media | blog | reddit | twitter`

**Integration:** Poll every 5 minutes in production. On first deployment, paginate backward as far as the API allows to minimize the gap between backfill (GDELT) and live data.

### 2.3 Binance REST API — OHLCV Price Data

**What it is:** Binance provides free, unauthenticated access to historical kline (OHLCV) data via `/api/v3/klines`. Data covers most major spot pairs back to 2017.

**Endpoint:** `GET https://api.binance.com/api/v3/klines`  
**Parameters:** `symbol` (e.g., `BTCUSDT`), `interval` (`1h`, `4h`, `1d`), `startTime`, `endTime` (milliseconds), `limit` (max 1000)

**Data fields per bar:** open time, open, high, low, close, volume, close time, quote asset volume, number of trades, taker buy base/quote asset volume.

**Rate limits:** 1200 requests/minute per IP (unauthenticated). A full 3-year hourly history for one symbol requires approximately 25,000 bars / 1000 per request = 25 API calls. For 10 symbols this is trivial.

---

## 3. Layer 0 — Raw Ingest

### 3.1 GDELT Backfill Fetcher

The BigQuery query filters crypto-relevant articles and exports metadata. Article bodies are then fetched separately by URL.

**BigQuery query template:**

```python
GDELT_QUERY = """
SELECT
    TIMESTAMP_MICROS(CAST(DATE * 1000 AS INT64)) AS crawl_ts,
    DocumentIdentifier AS url,
    SourceCommonName AS source_name,
    V2Themes AS themes,
    V2Organizations AS orgs,
    V2Persons AS persons,
    CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64) AS doc_tone
FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE
    DATE(_PARTITIONTIME) BETWEEN @start_date AND @end_date
    AND (
        V2Themes LIKE '%CRYPTOCURRENCY%'
        OR REGEXP_CONTAINS(DocumentIdentifier,
            r'coindesk\\.com|cointelegraph\\.com|decrypt\\.co|'
            r'theblock\\.co|bitcoinmagazine\\.com|cryptoslate\\.com')
    )
"""
```

**Article body fetching:** `trafilatura` is preferred over `newspaper3k` for financial news sites due to better boilerplate removal. Fetch asynchronously with `asyncio` + `httpx`, with a 10-second timeout and 3-retry exponential backoff. Accept a 20–30% fetch failure rate (paywalls, link rot on older articles) — record `fetch_status` codes in the raw schema and drop failed items at Layer 2.

**Re-scrape leakage (subtle but real):** When you fetch a 2021 article URL *today*, you receive the page as it exists now, not as it existed in 2021. Two leakage channels follow. First, the article body may have been edited post-publication (corrections, "update:" addenda) that incorporate later information. Second, and more dangerous, modern news pages embed "related articles," "trending now," and "what happened next" widgets whose links and headlines are current — these can contain explicit future information about how the event resolved. Mitigation: extract only the main article node (`trafilatura` with `include_comments=False`, `include_tables=False`, using its content-extraction mode that strips navigation/aside blocks), never the full page text. Where possible, prefer the Wayback Machine snapshot nearest to (but not after) the crawl timestamp. At minimum, strip everything outside the primary `<article>` body and spot-check a sample for future-dated content.

**Deduplication:** Hash `sha256(url + crawl_ts.date().isoformat())` as `item_id`. Run dedup before writing to parquet to avoid re-fetching already-stored URLs.

### 3.2 CryptoPanic Live Poller

Runs as a recurring job (cron or APScheduler). On each tick: fetch the current page, compare against the last stored `item_id`, write new items, and advance the `next` cursor.

**State file:** Persist the `next` cursor URL to disk between runs so polling survives restarts.

### 3.3 Binance OHLCV Fetcher

Paginate over the full date range in 1000-bar chunks. Write one parquet file per symbol per interval. This is a one-time backfill plus a daily incremental update.

```python
def fetch_all_klines(symbol: str, interval: str,
                     start_ms: int, end_ms: int) -> list[list]:
    all_bars = []
    cursor = start_ms
    while cursor < end_ms:
        resp = httpx.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": symbol, "interval": interval,
                    "startTime": cursor, "endTime": end_ms, "limit": 1000}
        ).json()
        if not resp:
            break
        all_bars.extend(resp)
        cursor = resp[-1][6] + 1  # close_time of last bar + 1ms
    return all_bars
```

---

## 4. Layer 1 — Raw Store

### 4.1 RawNewsItem Schema

```python
from pydantic import BaseModel
from datetime import datetime

class RawNewsItem(BaseModel):
    item_id: str              # sha256(url + published_at.date())
    published_at: datetime    # GDELT crawl_ts or CryptoPanic published_at
                              # NEVER modified after write — point-in-time anchor
    source: str               # "gdelt" | "cryptopanic"
    url: str
    source_domain: str
    title: str
    body: str                 # trafilatura-extracted body; empty string if failed
    asset_mentions: list[str] # ticker list: ["BTC", "ETH"]
    raw_tone: float | None    # GDELT V2Tone[0]; None for CryptoPanic
    fetch_status: int         # HTTP status; -1=timeout; -2=extraction_empty
```

**Storage:** One parquet file per calendar day under `raw/news/YYYY-MM-DD.parquet`. Append-only; existing files are never overwritten. Schema is enforced at write time via Pydantic → PyArrow.

**Asset mention extraction:** Two-pass approach. First, use CryptoPanic's `currencies` field directly when available. For GDELT, apply a regex lookup table of ticker aliases over `V2Organizations` and `V2Persons` (e.g., `bitcoin|btc → BTC`, `ethereum|eth → ETH`). Minimum one matching asset is required for an item to pass to Layer 2.

### 4.2 RawOHLCV Schema

```python
class RawOHLCVBar(BaseModel):
    symbol: str
    interval: str             # "1h", "4h", "1d"
    open_time: datetime       # bar open timestamp — UTC
    open: float
    high: float
    low: float
    close: float
    volume: float             # base asset volume
    n_trades: int
```

**Storage:** `raw/ohlcv/{SYMBOL}_{interval}.parquet`. Indexed by `open_time`. Never partition by date — load the full file in one shot for backtest efficiency.

---

## 5. Layer 2 — LLM Extraction ETL

This is the most critical layer. It runs **once** as an offline batch job over all raw news items, writing cached structured extractions. It is never re-run during backtesting.

### 5.1 EventRecord Schema

```python
from enum import Enum

class EventType(str, Enum):
    LISTING     = "listing"      # Exchange lists or delists a token
    HACK        = "hack"         # Exploit, bridge attack, theft
    REGULATION  = "regulation"   # SEC/CFTC/gov action, ruling, ban
    PARTNERSHIP = "partnership"  # Integration, deal, institutional adoption
    DEPEG       = "depeg"        # Stablecoin or peg failure
    MACRO       = "macro"        # Fed, rates, USD, risk-off macro event
    FORK        = "fork"         # Protocol upgrade, hard fork
    OTHER       = "other"

class EventRecord(BaseModel):
    item_id: str               # FK → RawNewsItem.item_id
    content_hash: str          # sha256(title + body[:500]) — cache key
    model_version: str         # e.g. "Llama3-2-q4km-v1"
    prompt_version: str        # semver: "1.0.0"
    extracted_at: datetime     # wall-clock time of extraction run
    asset: str                 # primary asset ticker: "BTC"
    event_type: EventType
    polarity: float            # [-1.0, 1.0]
    magnitude: float           # [0.0, 1.0] estimated market impact
    novelty: float             # [0.0, 1.0] computed externally (see §5.4)
    confidence: float          # [0.0, 1.0] model self-report
    extraction_only: bool      # True if event_type was determinable from text
                               # alone; False if the model had to infer impact
                               # direction. Used in the contamination audit
                               # (§8) to isolate records most exposed to
                               # parametric-recall leakage — audit and feature
                               # ablations are run on the extraction_only subset
                               # to confirm the signal survives without inferred
                               # (potentially memorized) judgments.
```

This field is set by the extraction prompt itself (the model reports whether it needed outside knowledge) and validated against the redaction test in §8.2.

**Reliability caveat:** a small base model's self-report on its own knowledge use is weak — models introspect poorly and may flag `extraction_only=true` even when leaning on memorized outcomes. Treat the flag as a noisy prior, not ground truth. The redaction test (§8.2) is the real check; the self-reported flag mainly serves to *pre-segment* records for the §8.3 ablation cheaply. If the two disagree systematically, trust the redaction test.

### 5.2 Prompt Design

Llama-3-8B-Instruct is an instruct-tuned model with 8192 token context. The official chat template uses `<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n...\n<|start_header_id|>user<|end_header_id|>\n...\n<|start_header_id|>assistant<|end_header_id|>\n`. The prompt must include:

- A structured system prompt defining the extraction task, constraints, and JSON schema
- 2–3 few-shot exemplars demonstrating format compliance
- Entity redaction of the headline asset name only when running the contamination audit (see §8)

The chat template replaces Llama3's `Instruct:` / `Output:` format. This improves extraction quality because the model receives explicit structured instructions rather than a compressed QA pair.

**Prompt template (version 1.0.0):**

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a structured data extraction system. Read the crypto news text and output
a JSON object with these exact keys and value constraints:
  asset: ticker symbol (string)
  event_type: one of [listing, hack, regulation, partnership, depeg, macro, fork, other]
  polarity: float -1.0 to 1.0 (negative=bad for asset, positive=good)
  magnitude: float 0.0 to 1.0 (estimated market impact size)
  confidence: float 0.0 to 1.0 (your confidence in this extraction)
  extraction_only: bool (true if you determined event_type and polarity
    SOLELY from this text; false if you used any outside knowledge of how
    this event turned out)
Output ONLY the JSON object. No explanation. No preamble.

<|start_header_id|>user<|end_header_id|>

Example 1:
Text: Binance announces listing of MATIC spot trading pair starting Friday.
Output: {"asset": "MATIC", "event_type": "listing", "polarity": 0.65,
"magnitude": 0.5, "confidence": 0.92, "extraction_only": true}

Example 2:
Text: SEC files charges against Ripple Labs over XRP securities offering,
seeks injunction and disgorgement of profits.
Output: {"asset": "XRP", "event_type": "regulation", "polarity": -0.90,
"magnitude": 0.85, "confidence": 0.95, "extraction_only": true}

Example 3:
Text: Nomad bridge drained of $190M in exploit as attacker replays
fraudulent transactions across multiple chains.
Output: {"asset": "ETH", "event_type": "hack", "polarity": -0.85,
"magnitude": 0.75, "confidence": 0.88, "extraction_only": true}

{PROMPT_VERSION}

Text: {TITLE}. {BODY_TRUNCATED}
<|start_header_id|>assistant<|end_header_id|>
```

**Budget enforcement:**  
Llama-3-8B with 8192 context: `len(tokenize(system_prompt + few_shot_block + prompt_version_token + title + body_truncated)) + max_new_tokens ≤ 8192`.  
Reserve 1700 tokens for system prompt, few-shot examples, and output grammar. That leaves approximately 6500 tokens for the article content. Truncate body to fit; if title alone (plus few-shot and system prompt) exceeds budget, reduce few-shot from 3 to 2 exemplars.

### 5.3 Model Serving

**For backfill batch (throughput priority):** vLLM with offline inference mode. Grammar-constrained decoding via `outlines` integration (Pydantic schema → JSON grammar). Run at `temperature=0` (greedy), `repetition_penalty=1.1`, `max_model_len=8192`. Batch size: 16–32 depending on GPU VRAM (Llama-3-8B at Q4_K_M uses ~5 GB VRAM per request with KV-cache for 8k context).

```python
from outlines import models, generate
from pydantic import BaseModel

class ExtractedEvent(BaseModel):
    asset: str
    event_type: str
    polarity: float
    magnitude: float
    confidence: float
    extraction_only: bool

llama3 = models.transformers("meta-llama/Meta-Llama-3-8B-Instruct")
generator = generate.json(llama3, ExtractedEvent, config={"max_model_len": 8192})

result = generator(prompt)  # returns validated ExtractedEvent instance
```

**For CPU-only / local dev:** llama.cpp GGUF Q4_K_M with a GBNF grammar matching the schema. Llama-3-8B-Instruct at Q4_K_M is approximately 4.9 GB, runs on any modern laptop with 8+ GB RAM. Set `n_ctx=8192` in the llama.cpp context initialization.

### 5.4 Novelty Score

Novelty is computed externally after extraction — the LLM is not asked to assess it. For each extracted event on asset `A` at time `t`:

1. Retrieve all EventRecords for asset `A` in the window `[t - 48h, t)`.
2. Compute TF-IDF vector of `title + event_type` for the current item vs. the window.
3. `novelty = 1 - max_cosine_similarity(current, window_items)`.
4. If the window is empty, `novelty = 1.0`.

This gives high novelty to the first report of a hack and low novelty to the fifteenth repetition of the same story.

### 5.5 Cache Design

Cache key: `(content_hash, model_version, prompt_version)` — all three must match for a cache hit.

```
cache/extractions/
    {content_hash[:2]}/       # two-char hex prefix for directory sharding
        {content_hash}.parquet
```

The cache is append-only and immutable. If a model or prompt version changes, old cached records remain valid for their version; new records write alongside them. The feature build step (Layer 3) selects by `(model_version, prompt_version)` to ensure consistency within a backtest run.

**Why this matters:** API providers deprecate model versions. If you rely on re-calling the model, your backtest results become non-reproducible within months. Persisting the `EventRecord`s, not the ability to regenerate them, is the production discipline that makes the backtest auditable.

---

## 6. Layer 3 — Feature Store

### 6.1 Point-in-Time Safety

The fundamental rule: **a feature at bar `t` may only use events with `published_at < bar_open_time(t)`**. No look-ahead, no same-bar events, no propagating corrections backward. This is enforced at feature construction time, not at read time — a corrupt join at read time would silently invalidate the backtest.

The risk does not live in the aggregation function (§6.2), which receives an already-filtered window; it lives in the *join* that produces that window. That join is the single most safety-critical line in the codebase and must use a strict less-than against bar open, never `<=`, and never bar close:

```python
def events_visible_at(events: pd.DataFrame, asset: str,
                      bar_open_ts: pd.Timestamp,
                      lookback: pd.Timedelta) -> pd.DataFrame:
    """The point-in-time guard. STRICT < bar_open_ts. Never <=. Never bar_close."""
    mask = (
        (events["asset"] == asset)
        & (events["published_at"] >= bar_open_ts - lookback)
        & (events["published_at"] < bar_open_ts)   # STRICT — the whole game
    )
    return events.loc[mask]

# minimal test — an event exactly at bar open must NOT be visible
def test_no_lookahead_at_boundary():
    t = pd.Timestamp("2021-06-01 00:00:00", tz="UTC")
    ev = pd.DataFrame({"asset": ["BTC"], "published_at": [t]})
    visible = events_visible_at(ev, "BTC", t, pd.Timedelta("24h"))
    assert len(visible) == 0, "event at bar open leaked into the bar"
```

The `>=` on the lookback edge and the `<` on the open edge are deliberate: an event landing exactly at bar open belongs to the *next* bar, because a strategy acting on bar `t` cannot have seen information that arrives at the same instant the bar opens.

### 6.2 Feature Vector

Per `(asset, bar_open_ts)`, aggregate all EventRecords in `[bar_open_ts - lookback, bar_open_ts)`:

```python
def build_feature_vector(
    events: pd.DataFrame,    # EventRecords in lookback window
    lookback_hours: int = 24
) -> dict:
    if events.empty:
        return zero_feature_vector()

    return {
        # Volume features
        "n_events":             len(events),
        "n_high_conf_events":   (events["confidence"] > 0.8).sum(),

        # Polarity aggregates
        "polarity_sum":         events["polarity"].sum(),
        "polarity_mean":        events["polarity"].mean(),
        "polarity_std":         events["polarity"].std(ddof=0),
        "polarity_min":         events["polarity"].min(),
        "polarity_max":         events["polarity"].max(),

        # Magnitude-weighted polarity
        "mag_weighted_polarity": (
            events["polarity"] * events["magnitude"]
        ).sum(),

        # Novelty-weighted polarity (downweights repeated stories)
        "novelty_polarity":     (
            events["polarity"] * events["novelty"] * events["magnitude"]
        ).sum(),

        # Event-type flags
        "hack_flag":            int((events["event_type"] == "hack").any()),
        "regulation_flag":      int((events["event_type"] == "regulation").any()),
        "listing_flag":         int((events["event_type"] == "listing").any()),
        "depeg_flag":           int((events["event_type"] == "depeg").any()),

        # Time decay: weight recent events more
        "recency_polarity":     compute_time_decayed_polarity(events, lookback_hours),

        # Metadata
        "lookback_h":           lookback_hours,
    }
```

Multiple lookback windows (6h, 24h, 72h) are stacked as separate feature blocks to capture fast-reaction vs. sustained narrative effects.

### 6.3 Storage

```
features/
    BTC/
        2020-01-01.parquet
        2020-01-02.parquet
        ...
    ETH/
        ...
```

Each file contains one row per bar (hourly or daily), indexed by `bar_open_ts`. Written once during the feature build job; never modified. The build job is idempotent: skip files that already exist unless explicitly asked to rebuild (e.g., after a prompt version change).

---

## 7. Layer 4 — Backtest Engine

### 7.1 Walk-Forward Design

**No in-sample optimization on the full dataset.** Use expanding or rolling walk-forward windows:

```
Window 1: train [2020-01 → 2020-12], test [2021-01 → 2021-03]
Window 2: train [2020-01 → 2021-03], test [2021-04 → 2021-06]
...
```

Rolling window size: 12 months train, 3 months test. Step: 3 months. Over a 3-year dataset this yields roughly 8 contiguous test windows. Note these are contiguous in time but the *train* windows overlap heavily, so the resulting test-period Sharpes are correlated, not independent — do not treat them as 8 independent samples when computing significance. Aggregate the out-of-sample returns into a single equity curve and bootstrap that, rather than averaging per-window Sharpes.

**The real power constraint is event sparsity, and the fix is cross-sectional breadth.** A single asset over three years yields few independent *event* observations (a handful of hacks, a dozen regulatory actions), which is far too few for a significant per-event-type signal regardless of how many price bars exist. The leverage comes from running the panel across many assets simultaneously: 50 assets × the same event types multiplies the effective event sample without extending the calendar window. This is the single most important reason the universe must eventually widen beyond BTC/ETH — and it is also why the survivorship-bias fix (§13) is not optional at that scale.

### 7.2 Signal Model

The feature vector feeds a simple signal model. Start with Ridge regression (interpretable, low overfitting risk) before moving to tree-based models:

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

signal_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0))
])

# Target: next-bar return (forward-looking, only used in train)
y_train = prices["close"].pct_change().shift(-1).loc[train_index].dropna()
X_train = features.loc[y_train.index]

signal_pipeline.fit(X_train, y_train)
raw_signal = signal_pipeline.predict(X_test)
```

Position sizing: z-score the raw signal, clip to `[-2, 2]`, scale to target volatility (e.g., 10% annualized). Include transaction costs (Binance spot: 0.1% per leg) and slippage assumption (0.05% for liquid pairs).

**LLM context budget with Llama-3-8B (8k context):**  
Llama-3-8B-Instruct with 8192 token context allows substantially larger prompts. The budget formula is:

```
len(tokenize(system_prompt + few_shot_block + title + body_truncated + output_prefix)) + max_new_tokens ≤ 8192
```

With llama.cpp GGUF Q4_K_M the effective KV-cache budget is approximately 6500 tokens (reserving ~1700 for system prompt, few-shot examples, and output grammar). Truncate body to fit; if title plus few-shot exceeds budget, reduce few-shot from 3 to 2 exemplars. The larger context enables a *structured system prompt* (task definition, constraints, schema in one block), which improves extraction quality and reduces malformed JSON compared to the compressed format of smaller-context models.

### 7.3 Evaluation Metrics

For each test period and aggregated over all periods:

- **Sharpe ratio** (annualized, assuming 252 trading days or 8760 hours)
- **Deflated Sharpe Ratio** (Bailey–López de Prado) — accounts for number of strategy variations tried; this is the primary acceptance criterion
- **Calmar ratio** (annualized return / max drawdown)
- **Hit rate** (fraction of bars with correct direction)
- **PnL attribution by event_type** — does the signal from `hack` events contribute differently than `listing` events?
- **Turnover** — average daily position change; controls for overcrowding in transaction costs

### 7.4 Baseline Comparisons

For the LLM feature to justify its complexity, it must beat:

1. **Pure OHLCV momentum** (1h, 4h, 24h returns as features, same signal model)
2. **GDELT raw tone** (V2Tone directly as feature, no LLM extraction)
3. **No-feature benchmark** (buy-and-hold BTC)

Statistical significance: use the stationary bootstrap (Politis–Romano) with block length calibrated to ACF of returns. Report 95% confidence intervals on Sharpe.

---

## 8. Contamination Audit

This is not optional. It is the primary validity test for any backtest using LLM-extracted features.

### 8.1 The Risk

A model with a training cutoff after your backtest window can recall how specific events resolved from its parametric memory, not from the text you supplied. This inflates backtest performance in a way that walk-forward splits do not catch, because both train and test sets fall within the model's knowledge.

Llama-3-8B mitigates this by design (synthetic/textbook training data with limited real-world event memorization) but does not eliminate it. Llama-3-8B-Instruct has a March 2023 cutoff, which is safer than most alternatives for a 2020–2022 backtest window. Its larger training corpus also means higher absolute memorization risk — the model has seen more real-world crypto events, so the entity redaction test (§8.2) is more critical than with Llama-3-8B.

### 8.2 Entity Redaction Test

Replace named entities in the raw text before extraction:

```python
import re

REDACTION_MAP = {
    r'\bBitcoin\b|\bBTC\b': 'Asset_X',
    r'\bEthereum\b|\bETH\b': 'Asset_Y',
    r'\bRipple\b|\bXRP\b': 'Asset_Z',
    r'\bTerraUSD\b|\bUST\b|\bLUNA\b': 'Asset_W',
    r'\bSEC\b': 'Regulator_A',
    r'\bBinance\b': 'Exchange_A',
    # ... extend for all major entities in test period
}

def redact(text: str) -> str:
    for pattern, replacement in REDACTION_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text
```

Run the full extraction pipeline twice: once with original text, once with redacted text. Compare extracted `event_type`, `polarity`, and `magnitude` distributions. If they are statistically indistinguishable, the model is reasoning from text, not from parametric recall. If they diverge significantly on specific events (e.g., the model gives LUNA a `depeg` event type from the ticker alone), that is a contamination signal.

**Caveat on interpreting divergence:** redaction also genuinely removes information a legitimate text-based extractor would use (an article that never re-states which asset it discusses becomes harder to extract correctly even with no memorization). So some divergence is expected and benign. The signal to worry about is *specific*: the model assigning the correct resolved outcome to a redacted event it could only know from training. Inspect divergent cases individually rather than relying on the aggregate distribution distance alone.

### 8.3 Extraction-Only Ablation

Re-run the backtest using only records where `extraction_only == True` (event type determinable from text without inferred judgment, per §5.1). If the signal's out-of-sample Sharpe holds on this restricted subset, it is strong evidence the edge comes from genuine text extraction rather than from the model's inferred — and potentially memorized — impact judgments. A signal that collapses on the extraction-only subset was leaning on exactly the judgments most exposed to parametric-recall leakage, and should be treated as suspect.

### 8.4 Temporal Holdout Test

Split the backtest period at the model's training cutoff:

- **Pre-cutoff period:** events where the model could not have seen the text (e.g., 2020–2022 for Llama-3-8B)
- **Post-cutoff period:** events where the model may have seen the text in training

If the backtest Sharpe is materially higher in the post-cutoff period, that is evidence of contamination. A robust signal should produce similar Sharpe in both halves (subject to regime differences).

---

## 9. Live Operation

### 9.1 Real-Time Ingest

```
CryptoPanic poller (every 5 min)
    → raw/news/YYYY-MM-DD.parquet  [append]
    → extraction queue (Redis or local deque)

Extraction worker (continuous)
    → cache lookup by content_hash
    → if miss: call Llama3, write EventRecord to cache
    → update feature store for today's date

Signal generation (on bar close)
    → read today's feature store
    → apply trained signal_pipeline
    → emit position target
```

### 9.2 Model Lifecycle

The signal model is retrained at a fixed cadence (e.g., monthly) using the rolling walk-forward scheme. The LLM extraction model is pinned (no retraining) — its version is recorded in every EventRecord. If the LLM is upgraded (new quantization, new model), all historical records remain valid under their old version; new extractions use the new version. The feature build job handles version mixing transparently by selecting by `(model_version, prompt_version)` at build time.

### 9.3 Monitoring

- **Extraction success rate:** fraction of items successfully parsed to valid EventRecord. Alert if < 85%.
- **Event type distribution drift:** rolling 7-day distribution of `event_type` vs. trailing 30-day baseline. Drift indicates news regime change or prompt degradation.
- **Signal autocorrelation:** daily signal should not be highly autocorrelated (> 0.9 for more than 3 bars = stale/stuck signal).
- **Cache miss rate:** should approach 0% for historical items; a spike indicates re-ingestion of duplicate URLs with different hashes.

---

## 10. Project Structure

```
crypto_llm_pipeline/
│
├── config/
│   └── settings.py            # Pydantic Settings: API keys, paths, symbols
│
├── ingest/
│   ├── gdelt_fetcher.py       # BigQuery backfill → RawNewsItem
│   ├── cryptopanic_poller.py  # Live forward ingestion
│   ├── binance_fetcher.py     # OHLCV backfill + incremental update
│   └── article_fetcher.py     # async URL → body via trafilatura
│
├── schemas/
│   ├── raw.py                 # RawNewsItem, RawOHLCVBar
│   └── events.py              # EventRecord, EventType
│
├── extraction/
│   ├── prompt.py              # Prompt builder with budget enforcement
│   ├── model.py               # 'Phi'-2 / Llama-3 loader + grammar-constrained generator
│   ├── cache.py               # Content-hash cache read/write
│   ├── novelty.py             # TF-IDF novelty scorer
│   └── batch_etl.py           # Offline batch extraction job
│
├── features/
│   ├── builder.py             # EventRecord → feature vector (point-in-time safe)
│   └── store.py               # Feature parquet read/write + validation
│
├── backtest/
│   ├── walkforward.py         # Walk-forward period generator + loop
│   ├── signal.py              # Ridge/LightGBM signal pipeline
│   ├── eval.py                # Sharpe, DSR, Calmar, attribution
│   └── contamination.py       # Redaction test + temporal holdout
│
├── live/
│   ├── poller.py              # Production polling loop
│   └── signal_server.py       # Bar-close signal generation
│
└── tests/
    ├── test_prompt.py         # Budget enforcement, schema compliance
    ├── test_cache.py          # Cache hit/miss, key collisions
    ├── test_features.py       # Point-in-time safety: event at bar open must
    │                          # NOT be visible (strict < boundary test)
    ├── test_walkforward.py    # Period non-overlap, no lookahead in data joins
    └── test_contamination.py  # Redaction test harness
```

---

## 11. Dependency Stack

| Component | Library | Notes |
|---|---|---|
| Data validation | `pydantic >= 2.0` | Schema enforcement at every layer boundary |
| Parquet I/O | `pyarrow`, `pandas` | All storage in columnar parquet |
| BigQuery | `google-cloud-bigquery` | GDELT backfill |
| HTTP client | `httpx[asyncio]` | Async article fetching + Binance API |
| Article extraction | `trafilatura` | Body text from HTML |
| LLM (GPU batch) | `vllm`, `outlines` | Grammar-constrained JSON extraction |
| LLM (CPU/local) | `llama-cpp-python` | GGUF Q4_K_M with GBNF grammar |
| Tokenizer | `transformers >= 4.40` | Llama-3-8B tokenizer support |
| Signal model | `scikit-learn`, `lightgbm` | Ridge baseline, LightGBM extension |
| Novelty scoring | `scikit-learn` TF-IDF | Cosine similarity in extraction layer |
| Statistical tests | `arch` (bootstrap) | Stationary bootstrap for Sharpe CIs |
| Task scheduling | `apscheduler` | Live polling cadence |
| Logging | `structlog` | JSON-structured logs for all layers |

**Python version:** 3.11+  
**No proprietary data vendors required.** All sources are free-tier or open.

---

## 12. Implementation Sequence

Build and validate each layer before starting the next. Do not start the LLM work until the raw store is populated and verified.

**Phase 1 — Data plumbing (2–3 days)**
1. Implement `binance_fetcher.py` — fastest to validate (known schema, no auth)
2. Implement `gdelt_fetcher.py` — run a small 7-day test query to verify BigQuery access and quota consumption
3. Implement `article_fetcher.py` — test async batch of 100 URLs, measure success rate
4. Write `tests/test_features.py` point-in-time safety test (write now, before data exists — tests the join logic)

**Phase 2 — Raw store (1 day)**
1. Define and freeze `RawNewsItem` and `RawOHLCVBar` schemas
2. Run a 30-day backfill to populate `raw/`
3. Validate: no future timestamps, no duplicate item_ids, asset_mention coverage rate

**Phase 3 — LLM extraction (3–4 days)**
1. Implement `prompt.py` with budget enforcement; write `tests/test_prompt.py`
2. Implement `cache.py`; write `tests/test_cache.py`
3. Run extraction on 30-day sample; manually audit 50 EventRecords for type accuracy
4. Implement `novelty.py`
5. Run full 3-year batch extraction

**Phase 4 — Feature store (1–2 days)**
1. Implement `builder.py` and `store.py`
2. Build features for full 3-year window
3. Run `tests/test_features.py` on the built store to confirm no lookahead joins

**Phase 5 — Backtest (2–3 days)**
1. Implement `walkforward.py` and `signal.py`
2. Run baseline comparisons (momentum-only, raw-tone-only, buy-and-hold)
3. Run LLM-feature backtest
4. Run contamination audit (`tests/test_contamination.py`)
5. Compute DSR and report with confidence intervals

**Phase 6 — Live wiring (1–2 days)**
1. Wire `cryptopanic_poller.py` → extraction → feature update
2. Implement monitoring metrics
3. Smoke test with 24-hour live run

Total estimated build time: **10–15 focused development days**.

---

## 13. Known Limitations and Risks

**GDELT coverage thinning before 2020.** Crypto-specific sources were less uniformly indexed pre-2020. Do not extend the backtest window before validating per-day item counts and asset coverage.

**Article body fetch failure rate.** Expect 20–30% of GDELT URLs to be unfetchable (paywalls, 404s, bot blocks). The pipeline must remain valid when operating on title-only items. Llama-3-8B's 8k context handles titles with ample room for body text when available; extraction quality on title-only items should be spot-checked against title + body extractions.

**LLM extraction error rate.** Even with grammar-constrained decoding, the model will occasionally output semantically wrong extractions (e.g., a partnership article tagged as `hack`). The signal model must be robust to label noise — Ridge regression and tree-based models handle moderate noise well; a fragile threshold strategy would not.

**Lookback window choice.** The optimal lookback (6h, 24h, 72h) is a hyperparameter that will vary by asset and event type. It must be chosen on the training set only, not by inspection of the test set. This is a standard data-snooping trap in event studies.

**Novelty score leakage.** The TF-IDF-based novelty score uses a 48-hour trailing window. When building features for bar `t`, the novelty computation must only use events with `published_at < t` in the window. Verify this in `tests/test_features.py`.

**CryptoPanic live-to-backfill gap.** There will always be a gap between the end of the GDELT backfill and the start of CryptoPanic live data. Fill this gap at deployment time by paginating CryptoPanic backward as far as possible, and accept that any remaining gap is a blind spot in the live signal.

**Survivorship and delisting bias in the asset universe.** If you select the asset universe by taking *today's* top coins and backfilling their data to 2020, you systematically exclude tokens that died (delisted, rugged, or faded) over the window — a classic survivorship bias that inflates returns, because the universe is conditioned on survival. The fix: reconstruct the universe point-in-time. Define eligibility at the *start* of each rebalance window using only data available then (e.g., top-N by market cap as of that date from a point-in-time source), and keep delisted assets in the backtest until their actual delisting, with the delisting modeled as a terminal return. For a first prototype on BTC/ETH alone this bias is negligible (neither delisted), but it becomes material the moment the universe is widened to alts — which is exactly where event-driven signals have the most edge, so it cannot be ignored at scale.

**Event-type taxonomy is fixed and may be miscalibrated.** The eight `EventType` categories are an a-priori guess. Some real events will not fit cleanly (e.g., a token unlock, an ETF approval, an exchange insolvency that is neither a hack nor a regulation). Misfiled events land in `other` and lose signal. Before committing, run the extraction on a sample and inspect the `other` rate; if it exceeds ~25%, the taxonomy needs revision. Treat the taxonomy as versioned alongside the prompt (`prompt_version`), since changing it invalidates the cache.

**Single-source dependency for live data.** CryptoPanic is the only live news source in this pipeline. It is a single point of failure. For production beyond a prototype, add a second source (e.g., RSS feeds from coindesk.com, cointelegraph.com directly) and merge at the deduplication step.
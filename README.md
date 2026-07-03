# AlphaSent — Crypto News → Alpha Feature Pipeline

LLM-driven news extraction pipeline that turns crypto headlines into
point-in-time-safe alpha features for quantitative signal models.

**Architecture:** 4-layer pipeline — raw ingest, LLM extraction ETL, feature store, backtest engine. Each layer writes only to its own storage and reads only from the layer below. The LLM is **never called during backtesting**.

```
┌──────────────────────────────────────────────────────────────┐
│  LAYER 0: RAW INGEST                                         │
│  GDELT BigQuery (backfill)   │  CryptoPanic (live) │ Binance │
└──────────────┬───────────────┴─────────────────────┴─────────┘
               ▼
┌──────────────────────────────────────────────────────────────┐
│  LAYER 1: RAW STORE                                          │
│  raw/news/YYYY-MM-DD.parquet │    raw/ohlcv/{SYMBOL}.parquet │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────────────────────────────────────┐
│  LAYER 2: LLM EXTRACTION ETL  (offline, one-pass, cached)    │
│  budget_prompt → Llama3-8B → EventRecord → novelty score     │
└──────────────┬───────────────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────────────┐
│  LAYER 3: FEATURE STORE  (point-in-time safe)                │
│  features/{ASSET}/YYYY-MM-DD.parquet                         │
│  events_visible_at: STRICT < bar_open_ts                     │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────────────────────────────────────┐
│  LAYER 4: BACKTEST ENGINE                                    │
│  walk-forward: features[t] + ohlcv[t] → signal → PnL[t+1]    │
└──────────────────────────────────────────────────────────────┘
```

## Quick Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install core dependencies
pip install pandas pyarrow pydantic httpx scikit-learn numpy trafilatura

# Optional: GPU batch inference
# pip install vllm outlines

# Optional: BigQuery for GDELT backfill
# pip install google-cloud-bigquery

# Optional: LightGBM signal model
# pip install lightgbm

# Optional: RSS ingestion
# pip install feedparser
```

## Data Sources

| Source | Purpose | Access |
|---|---|---|
| **GDELT GKG** | Historical news (2020–present) | BigQuery, 1 TB free/month |
| **CryptoPanic** | Live forward news (today onward) | Free API token |
| **Binance REST** | OHLCV price data | Free, unauthenticated |
| **RSS feeds** | Supplementary live news | Free |

## Data Pipeline

### 1. GDELT Backfill

Historical news articles with asset mentions, crawled via BigQuery.

```bash
# Requires google-cloud-bigquery and credentials
python3 -m src.ingest.gdelt_fetcher
```

Output: `data/news/YYYY-MM-DD.parquet` (one file per day, append-only).

### 2. Article Body Fetch

Fetch article body text from URLs using `trafilatura`. Supports async batch mode.

```python
from src.ingest.article_fetcher import fetch_body, fetch_bodies_async

body, status = fetch_body("https://coindesk.com/article")
# or async:
results = fetch_bodies_async(["https://a.com/1", "https://b.com/2"])
```

Output: body text + `fetch_status` code (200, -1=timeout, -2=empty).

### 3. Binance OHLCV

Historical kline data for backtesting.

```python
from src.ingest.binance_fetcher import fetch_all_klines, parse_klines

raw = fetch_all_klines("BTCUSDT", "1h", start_ms, end_ms)
ohlcv_df = parse_klines(raw, "BTCUSDT", "1h")
```

### 4. CryptoPanic Live Poller

Forward-only live ingestion with extraction and feature update.

```bash
# Requires CRYPTOPANIC_AUTH_TOKEN env var
export CRYPTOPANIC_AUTH_TOKEN="your_token"
python3 -m src.live.poller
```

Or single-shot mode:

```python
from src.live.poller import run_once
summary = run_once(token="your_token")
```

See [README_CP.md](README_CP.md) for cron scheduling details.

### 5. RSS Ingestion

Supplementary live news from coindesk, cointelegraph, decrypt, theblock.

```python
from snippets.crypto_rss_ingest import run_once
summary = run_once()
```

### 6. LLM Extraction ETL

Offline batch job: raw news → structured `EventRecord` with novelty scoring.

```python
from src.extraction.batch_etl import run_batch_extraction

events_df = run_batch_extraction()
# events_df columns: asset, event_type, polarity, magnitude, novelty, confidence
```

Run dry to preview prompts without calling the LLM:

```python
events_df = run_batch_extraction(dry_run=True)
```

The extraction is cached by `(content_hash, model_version, prompt_version)` in
`data/cache/extractions/`. Changing the prompt or model version forces re-extraction.

### 7. Feature Build

Point-in-time-safe feature aggregation over event records.

```python
from src.features.builder import build_feature_vector, events_visible_at
from src.features.store import write_features

# Build features for an asset
from src.features.builder import build_features_for_asset
features_df = build_features_for_asset(events_df, ohlcv_df, "BTC", "BTCUSDT")
write_features(features_df, "BTC", "2024-01-01")
```

### 8. Backtest

Walk-forward backtest with baseline comparisons.

```python
from scripts.run_backtest import run_backtest

results = run_backtest(assets=["BTC"], start="2021-01-01", end="2024-01-01")
# Compares: LLM features vs momentum-only vs buy-and-hold
```

### 9. Live Signal Server

Bar-close signal generation with position targets.

```bash
# One-shot
python3 -m src.live.signal_server

# Continuous loop (re-evaluates every hour)
python3 -m src.live.signal_server --loop

# Force retrain
python3 -m src.live.signal_server --retrain
```

Output: `data/signals/positions.json` with per-asset position targets.

### 10. Contamination Audit

Verify the LLM is reasoning from text, not parametric memory.

```python
from src.backtest.contamination import (
    redact, compare_extractions, extraction_only_subset, temporal_holdout_split
)

# Redaction test: replace entity names, compare distributions
redacted = redact("Bitcoin price surges after SEC approval")
# → "Asset_X price surges after Regulator_A approval"

# Temporal holdout: pre- vs post-cutoff Sharpe comparison
pre, post = temporal_holdout_split(events_df)
```

## Key Design Principles

1. **Point-in-time safety.** Events at bar `t` are strictly excluded from features at bar `t`. This is enforced at feature *construction* time, not at read time.

2. **Offline LLM.** The LLM runs once as an offline batch job. It is never called during backtesting.

3. **Versioned cache.** Extractions are keyed by `(content_hash, model_version, prompt_version)`. Changing any version forces re-extraction.

4. **Novelty scoring.** TF-IDF-based novelty downweights repeated stories. First report of a hack = 1.0, fifteenth repetition = near 0.

5. **Contamination audit.** Entity redaction, extraction-only ablation, and temporal holdout tests verify the signal is not from parametric recall.

## Feature Vector

Per `(asset, bar_open_ts)`, the feature builder aggregates all visible events
in `[bar_open_ts - lookback, bar_open_ts)`:

| Feature | Description |
|---|---|
| `n_events` | Count of events in window |
| `n_high_conf_events` | Events with confidence > 0.8 |
| `polarity_sum` / `polarity_mean` / `polarity_std` | Polarity aggregates |
| `mag_weighted_polarity` | Polarity weighted by magnitude |
| `novelty_polarity` | Polarity weighted by novelty × magnitude |
| `hack_flag` / `regulation_flag` / `listing_flag` / `depeg_flag` | Event-type binary flags |
| `recency_polarity` | Time-decayed polarity (recent events weighted more) |

Multiple lookback windows (6h, 24h, 72h) are stacked as separate feature blocks.

## Testing

```bash
pip install pytest

# Run all tests
python3 -m pytest tests/ -v

# Specific test suites
python3 -m pytest tests/test_features.py -v   # point-in-time safety
python3 -m pytest tests/test_contamination.py -v  # redaction test
python3 -m pytest tests/test_signal.py -v     # signal pipeline
python3 -m pytest tests/test_eval.py -v       # Sharpe, DSR, metrics
python3 -m pytest tests/test_walkforward.py -v  # window non-overlap
```

## Configuration

All settings in `src/config.py`, overridable via environment variables:

| Variable | Default | Description |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8079/v1` | LLM endpoint |
| `LLM_MODEL_NAME` | `llama3-8b` | Model name |
| `LLM_TEMPERATURE` | `0` | Sampling temperature |
| `CRYPTOPANIC_AUTH_TOKEN` | *(required)* | CryptoPanic API token |
| `CRYPTOPANIC_PLAN` | `developer` | CryptoPanic plan tier |
| `DATA_ROOT` | `./data` | Data directory |
| `RSS_DATA_ROOT` | `./data/crypto_rss` | RSS data directory |

## Storage Layout

```
data/
├── news/                          # GDELT backfill
│   └── YYYY-MM-DD.parquet
├── ohlcv/                         # Binance OHLCV
│   └── {SYMBOL}_1h.parquet
├── features/                      # Layer-3 feature store
│   └── {ASSET}/YYYY-MM-DD.parquet
├── cache/                         # LLM extraction cache
│   └── extractions/{hash[:2]}/{hash}.parquet
├── cryptopanic/                   # CryptoPanic ingestion
│   ├── state.json
│   ├── raw_json/
│   └── normalized/
├── crypto_rss/                    # RSS ingestion
│   └── normalized/
└── signals/                       # Live signal output
    ├── positions.json
    └── retrain_state.json
```

## Project Structure

```
alphasent/
├── src/
│   ├── config.py                    # All configuration
│   ├── ingest/
│   │   ├── binance_fetcher.py       # OHLCV from Binance
│   │   ├── gdelt_fetcher.py         # BigQuery backfill
│   │   └── article_fetcher.py       # Async body fetcher
│   ├── schemas/
│   │   ├── raw.py                   # RawNewsItem, RawOHLCVBar
│   │   └── events.py                # EventRecord, EventType
│   ├── extraction/
│   │   ├── prompt.py                # Prompt builder + budget enforcement
│   │   ├── model.py                 # LLM call (local llama3-8b)
│   │   ├── cache.py                 # Content-hash cache
│   │   ├── novelty.py               # TF-IDF novelty scorer
│   │   └── batch_etl.py             # Offline batch extraction
│   ├── features/
│   │   ├── builder.py               # Point-in-time-safe feature vector
│   │   └── store.py                 # Feature parquet read/write
│   ├── backtest/
│   │   ├── walkforward.py           # Walk-forward period generator
│   │   ├── signal.py                # Ridge/LightGBM signal pipeline
│   │   ├── eval.py                  # Sharpe, DSR, Calmar, attribution
│   │   ├── baselines.py             # Momentum, raw-tone, buy-and-hold
│   │   └── contamination.py         # Redaction test + temporal holdout
│   └── live/
│       ├── poller.py                # CryptoPanic polling loop
│       ├── signal_server.py         # Bar-close signal generation
│       ├── monitoring.py            # Health metrics
│       └── smoke_test.py            # End-to-end smoke test
├── scripts/
│   ├── run_backtest.py              # Full backtest runner
│   ├── run_text_pipeline.py         # Text-only pipeline (no OHLCV)
│   └── report.py                    # Evaluation report
├── tests/                           # Test suite
├── snippets/                        # Standalone ingestion scripts
│   ├── cryptopanic_ingest.py
│   └── crypto_rss_ingest.py
├── PLAN.md                          # Detailed project plan
└── README.md                        # This file
```

## References

- **Project plan:** [PLAN.md](PLAN.md) — 4-layer architecture, data sources, contamination audit, implementation sequence
- **CryptoPanic ingestion:** [README_CP.md](README_CP.md) — scheduling, output schema, cron setup

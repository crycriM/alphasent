# AlphaSent — Crypto News → Alpha Feature Pipeline

LLM-driven news extraction pipeline that turns crypto headlines into
point-in-time-safe alpha features for quantitative signal models.

**Architecture:** 4-layer pipeline — raw ingest, LLM extraction ETL, feature store, backtest engine. Each layer writes only to its own storage and reads only from the layer below. The LLM is **never called during backtesting**.

```
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 0: RAW INGEST                                                │
│  GDELT BigQuery (1yr backfill)  |  RSS (17 feeds, 15-min cron)      │
└─────────────────────┬───────────────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 1: RAW STORE                                                 │
│  data/news/YYYY-MM-DD.parquet   (GDELT, ~340K records)              │
│  data/crypto_rss/normalized/    (RSS, ~400-500 items/day forward)   │
│  data/ohlcv/{SYMBOL}.parquet    (Binance OHLCV)                     │
└─────────────────────┬───────────────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 2: LLM EXTRACTION ETL  (offline, one-pass, cached)           │
│  perimeter pre-filter → budget_prompt → LLM → EventRecord           │
│  Survivorship-bias-free: only articles matching that month's         │
│  active perp universe (data/perimeter/) get sent to the LLM          │
└─────────────────────┬───────────────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 3: FEATURE STORE  (point-in-time safe)                       │
│  features/{ASSET}/YYYY-MM-DD.parquet                                 │
│  events_visible_at: STRICT < bar_open_ts                             │
│  Multiple lookback windows: 6h, 24h, 72h stacked per bar            │
└─────────────────────┬───────────────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 4: BACKTEST ENGINE                                           │
│  walk-forward: features[t] + ohlcv[t] → signal → PnL[t+1]           │
│  Baselines: momentum, raw-tone, buy-and-hold                        │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Setup

```bash
# Already set up — .venv with all deps installed
source .venv/bin/activate

# Key dependencies: numpy, pandas, pyarrow, pydantic, httpx, scikit-learn,
# feedparser, google-cloud-bigquery, db-dtypes, lightgbm (optional)
```

## Data Sources

| Source | Purpose | Status |
|--------|---------|--------|
| **GDELT GKG** | Historical news (Jul 2025 – present) | ✅ Backfilled: ~340K records via BigQuery |
| **RSS feeds** | Live forward news (17 publishers) | ✅ Running every 15 min, ~500 items/day |
| **Binance REST** | OHLCV price data | ✅ Code ready, needs fetching |
| **CryptoPanic API** | Community-curated news feed | ❌ Free tier RSS-only, dead feed |

### Perimeter Files (Survivorship-Bias Shield)

> **The core problem:** The LLM (`llama3-8b`) doesn't know about newer tokens (ONDO, HYPE, AI16Z). Using a newer LLM introduces lookahead bias (it "knows" LUNA collapsed, FTT went to zero).
>
> **The solution:** Monthly perpetual-swap perimeter files from `data/perimeter/`. These list all actively traded perp symbols across Binance, Hyperliquid, Okex, and Bybit for a given month. Before any LLM call, the article is scanned against that month's universe — if no ticker match, it's skipped. This is **survivorship-bias-safe**: tokens that were later delisted aren't retroactively tagged as "crypto."

## Data Pipeline

### 1. GDELT Backfill

Historical news articles crawled via BigQuery. The GDELT Global Knowledge Graph index provides `doc_tone` (article-level sentiment from -100 to +100), source domain, entity mentions, and crawled URL.

```bash
# Requires GCP service account key in auth.json
cd ~/projects/alphasent
source .venv/bin/activate
GOOGLE_APPLICATION_CREDENTIALS=auth.json python -c "
from src.ingest.gdelt_fetcher import backfill_gdelt
backfill_gdelt(start_date='2025-07-01', end_date='2026-07-07')
"
```

Output: `data/news/YYYY-MM-DD.parquet` (one file per day, append-only).

**Query logic:** Matches articles containing `ECON_BITCOIN` or `ECON_CRYPTOCURRENCY` GDELT themes, or from known crypto publisher domains (coindesk, cointelegraph, decrypt, theblock, etc.). Approximately 900-1500 records/day.

### 2. RSS Ingestion (Live Forward)

17 publisher RSS feeds, runs every 15 minutes via cron. Zero-cost forward data accumulation.

```bash
# Manual run
source .venv/bin/activate
python snippets/crypto_rss_ingest.py
```

**Active feeds:**

| Source | URL | Items/run |
|--------|-----|-----------|
| coindesk | coindesk.com/arc/outboundfeeds/rss/ | ~25 |
| cointelegraph | cointelegraph.com/rss | ~30 |
| decrypt | decrypt.co/feed | ~34 |
| theblock | theblock.co/rss.xml | ~20 |
| bitcoinmagazine | bitcoinmagazine.com/feed | ~10 |
| newsbtc | newsbtc.com/feed/ | ~10 |
| bitcoincom | news.bitcoin.com/feed/ | ~10 |
| utoday | u.today/rss | ~89 |
| cryptonews | crypto.news/feed/ | ~50 |
| cryptopotato | cryptopotato.com/feed/ | ~15 |
| zycrypto | zycrypto.com/feed/ | ~14 |
| beincrypto | beincrypto.com/feed/ | ~12 |
| ambcrypto | ambcrypto.com/feed/ | ~16 |
| dailycoin | dailycoin.com/feed/ | ~10 |
| blockonomi | blockonomi.com/feed/ | ~10 |
| bitcoinist | bitcoinist.com/feed/ | ~8 |
| cryptobriefing | cryptobriefing.com/feed/ | ~30 |

**Cron job:** `alphasent-rss-ingest` — fires every 15 min, no-agent mode, saves to local files.

### 3. Binance OHLCV

Historical kline data for backtesting with signal returns.

```python
from src.ingest.binance_fetcher import fetch_all_klines, parse_klines

raw = fetch_all_klines("BTCUSDT", "1h", start_ms, end_ms)
ohlcv_df = parse_klines(raw, "BTCUSDT", "1h")
```

Output: `data/ohlcv/{SYMBOL}_1h.parquet`.

### 4. LLM Extraction ETL (Offline Batch)

This is the core transformation: raw news → structured `EventRecord` with asset, event_type, polarity, magnitude, novelty, and confidence.

**The pipeline runs in one pass and is cached by content hash.** Changing model or prompt version forces re-extraction.

```python
from src.extraction.batch_etl import run_batch_extraction

events_df = run_batch_extraction()
# events_df columns: asset, event_type, polarity, magnitude, novelty, confidence, ...
```

Run dry to preview prompts without calling the LLM:

```python
events_df = run_batch_extraction(dry_run=True)
```

**Perimeter pre-filter (automatic):** When you run batch extraction, the pipeline:
1. Reads the perimeter file for the earliest article date in the batch
2. Scans each article's title+body for ticker matches (aliases like "Bitcoin"→BTC, "Solana"→SOL, plus raw symbols like ONDO, HYPE)
3. If no match → article is **skipped** (wasted LLM calls avoided)
4. If match → the matched ticker is forced as the asset, bypassing the LLM's inability to recognize newer tokens

**Monthly workflow:** Before running extraction, ensure your perimeter file for that month is at `data/perimeter/recup_perimeter_YYYY-MM-DD.json`. The format is a JSON dict with exchange names as keys and arrays of perp symbol strings as values:

```python
{
  "binancefut": ["BTCUSDT", "ETHUSDT", "ONDOUSDT", ...],
  "hyperliquid": ["BTC/USDC:USDC", "ETH/USDC:USDC", ...],
  "okexfut": ["BTC-USDT-SWAP", "ETH-USDT-SWAP", ...]
}
```

### 5. Feature Build

Point-in-time-safe feature aggregation over event records. Features are built by aggregating all events with `published_at < bar_open_ts` within a lookback window.

```python
from src.features.builder import build_features_for_asset
from src.features.store import write_features

features_df = build_features_for_asset(events_df, ohlcv_df, "BTC", "BTCUSDT")
write_features(features_df, "BTC", "2024-01-01")
```

### 6. Backtest

Walk-forward backtest with baseline comparisons (LLM features vs momentum-only vs buy-and-hold).

```python
from scripts.run_backtest import run_backtest

results = run_backtest(assets=["BTC"], start="2021-01-01", end="2024-01-01")
```

### 7. Live Signal Server

Bar-close signal generation with position targets.

```bash
# One-shot
python -m src.live.signal_server

# Continuous loop (re-evaluates every hour)
python -m src.live.signal_server --loop
```

### 8. Contamination Audit

Verify the LLM is reasoning from text, not parametric memory.

```python
from src.backtest.contamination import (
    redact, compare_extractions, extraction_only_subset, temporal_holdout_split
)

# Entity redaction: replace names with placeholders
redacted = redact("Bitcoin price surges after SEC approval")
# → "Asset_X price surges after Regulator_A approval"

# Temporal holdout: pre- vs post-cutoff Sharpe comparison
pre, post = temporal_holdout_split(events_df)
```

## Feature Vector

Per `(asset, bar_open_ts)`, the feature builder aggregates all visible events
in `[bar_open_ts - lookback, bar_open_ts)`:

| Feature | Description |
|---------|-------------|
| `n_events` | Count of events in window |
| `n_high_conf_events` | Events with confidence > 0.8 |
| `polarity_sum` / `mean` / `std` | Polarity aggregates |
| `mag_weighted_polarity` | Polarity weighted by magnitude |
| `novelty_polarity` | Polarity weighted by novelty × magnitude |
| `hack_flag` / `regulation_flag` / `listing_flag` / `depeg_flag` | Event-type binary flags |
| `recency_polarity` | Time-decayed polarity (recent events weighted more) |

Multiple lookback windows (6h, 24h, 72h) are stacked as separate feature blocks.

## Configuration

All settings in `src/config.py`, overridable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BASE_URL` | `http://localhost:8079/v1` | LLM endpoint |
| `LLM_MODEL_NAME` | `llama3-8b` | Model name |
| `LLM_TEMPERATURE` | `0` | Sampling temperature |
| `GOOGLE_APPLICATION_CREDENTIALS` | `auth.json` | GCP service account key path |
| `GDELT_PROJECT_ID` | `endless-empire-498816-j2` | BigQuery billing project |
| `RSS_DATA_ROOT` | `./data/crypto_rss` | RSS data directory |
| `CRYPTOPANIC_AUTH_TOKEN` | *(unused)* | CryptoPanic API token (paid tier only) |

## Storage Layout

```
data/
├── perimeter/                     # Monthly perp universe (survivorship-bias input)
│   └── recup_perimeter_YYYY-MM-DD.json
├── news/                          # GDELT backfill (one file per day, 340K records)
│   └── YYYY-MM-DD.parquet
├── crypto_rss/                    # RSS ingestion (17 feeds, 15-min cron)
│   ├── state.json                 # Per-feed dedup state
│   ├── normalized/YYYY-MM-DD.parquet
│   └── raw_json/                  # Per-run raw archives
├── ohlcv/                         # Binance OHLCV
│   └── {SYMBOL}_1h.parquet
├── features/                      # Layer-3 feature store
│   └── {ASSET}/YYYY-MM-DD.parquet
├── cache/                         # LLM extraction cache
│   └── extractions/{hash[:2]}/{hash}.parquet
├── cryptopanic/                   # CryptoPanic (unused — dead free feed)
│   ├── state.json
│   └── raw_json/
└── signals/                       # Live signal output
    ├── positions.json
    └── retrain_state.json
```

## Project Structure

```
alphasent/
├── auth.json                      # GCP service account key
├── pyproject.toml                 # Python project config
├── src/
│   ├── config.py                  # All configuration
│   ├── ingest/
│   │   ├── binance_fetcher.py     # OHLCV from Binance
│   │   ├── gdelt_fetcher.py       # GDELT BigQuery backfill
│   │   ├── perimeter.py           # Per-month crypto universe + ticker matcher
│   │   └── article_fetcher.py     # Async body fetcher (unused with GDELT)
│   ├── schemas/
│   │   ├── raw.py                 # RawNewsItem, RawOHLCVBar
│   │   └── events.py              # EventRecord, EventType
│   ├── extraction/
│   │   ├── prompt.py              # Prompt builder + budget enforcement
│   │   ├── model.py               # LLM call (local llama3-8b)
│   │   ├── cache.py               # Content-hash cache (parquet)
│   │   ├── novelty.py             # TF-IDF novel scorer
│   │   └── batch_etl.py          # Offline batch extraction (with perimeter pre-filter)
│   ├── features/
│   │   ├── builder.py             # Point-in-time-safe feature vector
│   │   └── store.py               # Feature parquet read/write
│   ├── backtest/
│   │   ├── walkforward.py         # Walk-forward period generator
│   │   ├── signal.py              # Ridge/LightGBM signal pipeline
│   │   ├── eval.py                # Sharpe, DSR, Calmar, attribution
│   │   ├── baselines.py           # Momentum, raw-tone, buy-and-hold
│   │   └── contamination.py       # Redaction test + temporal holdout
│   └── live/
│       ├── poller.py              # CryptoPanic polling loop (unused)
│       ├── signal_server.py       # Bar-close signal generation
│       ├── monitoring.py          # Health metrics
│       └── smoke_test.py          # End-to-end smoke test
├── scripts/
│   ├── run_backtest.py            # Full backtest runner
│   ├── run_text_pipeline.py       # Text-only pipeline (no OHLCV)
│   └── eval_text_pipeline.py      # Signal quality report
├── tests/                         # Test suite
├── snippets/                      # Standalone ingestion scripts
│   └── crypto_rss_ingest.py       # Multi-feed RSS ingester
├── data/
│   ├── perimeter/                 # Monthly perp universe (input, 37 files)
│   ├── news/                      # GDELT output (371 partitions)
│   ├── crypto_rss/                # RSS output (17 feeds, live)
│   ├── cache/                     # LLM extraction cache
│   └── features/                  # Built features (to be generated)
├── PLAN.md                        # Detailed project plan
└── README.md                      # This file

## References

- **Research whitepaper:** [WHITEPAPER.md](WHITEPAPER.md) — IC-based study of the findings: what did not work and why
- **Project plan:** [PLAN.md](PLAN.md) — 4-layer architecture, data sources, contamination audit
- **RSS skill:** `skill_view("alphasent-rss-ingest")` — cron setup, feed list, data layout
- **Perimeter format:** `src/ingest/perimeter.py` — ticker normalization, alias resolution

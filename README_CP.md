# CryptoPanic Ingestion Batch

Forward-only accumulator for the CryptoPanic feed. Run it on a schedule starting
now; it builds the CryptoPanic history you cannot fetch retroactively. This is the
live-source half of the pipeline's overlap-validation requirement — the sooner it
starts, the longer your GDELT↔CryptoPanic overlap window.

## Setup

```bash
pip install httpx pydantic pyarrow pandas
export CRYPTOPANIC_AUTH_TOKEN="your_token_here"      # from cryptopanic.com/developers
export CRYPTOPANIC_PLAN="developer"                  # your plan slug in the URL path
# optional scope filters:
# export CRYPTOPANIC_KIND="news"                      # news | media | all
# export CRYPTOPANIC_CURRENCIES="BTC,ETH,SOL"
# export CRYPTOPANIC_DATA_ROOT="/var/data/cryptopanic"
```

## Run

```bash
python cryptopanic_ingest.py        # one pass: fetch new, normalize, dedup, append
python test_cryptopanic_ingest.py   # offline tests (no token/network needed)
```

## Schedule (this is the point — accumulate continuously)

The API has a ~30s server-side cache and forward-only access. Every 15 minutes is
ample and safe; the watermark logic means a missed run is harmless (the next run
pages back further). Each run fetches at most ~1000 newest posts, so as long as
crypto news volume stays under ~1000 posts / 15 min (always true), there are no gaps.

```cron
*/15 * * * *  cd /path/to/cryptopanic_ingest && \
              CRYPTOPANIC_AUTH_TOKEN=xxx CRYPTOPANIC_PLAN=developer \
              /usr/bin/python cryptopanic_ingest.py >> ingest.log 2>&1
```

If a run ever logs `Hit MAX_PAGES_PER_RUN`, increase cadence (more frequent runs)
or raise `MAX_PAGES_PER_RUN` — it means a single run couldn't reach the watermark.

## Output layout

```
data/cryptopanic/
├── state.json                       # high-water mark; do not delete
├── raw_json/
│   └── YYYYMMDDThhmmssZ.json.gz      # untouched API payload per run (reprocess source)
└── normalized/
    └── YYYY-MM-DD.parquet            # Layer-1 schema, partitioned by published date
```

`normalized/*.parquet` columns (one row per post, deduped by `item_id`):

| column | meaning |
|---|---|
| `item_id` | `cryptopanic:{post_id}` — stable dedup key |
| `post_id` | raw CryptoPanic id |
| `published_at` | point-in-time anchor (UTC) |
| `created_at` | when CryptoPanic ingested it (UTC) |
| `ingested_at` | when **we** fetched it (UTC) — the true availability time for live |
| `title`, `description` | text |
| `url`, `original_url`, `source_domain`, `source_type`, `kind` | provenance |
| `asset_mentions` | list, e.g. `["BTC","ETH"]` |
| `panic_score` | CryptoPanic 0–100 panic score |
| `votes_positive`, `votes_negative`, `votes_raw` | community vote signal |

## Notes that matter downstream

- **`ingested_at` vs `published_at`.** For the live signal, `ingested_at` (when *we*
  saw it) is the honest availability timestamp — it includes API/polling latency.
  `published_at` is what CryptoPanic reports. When you later measure the GDELT vs
  CryptoPanic timing gap (the overlap study), compare on `ingested_at`, not
  `published_at`, or you will understate real-world latency.
- **Raw archive is the source of truth.** If the normalized schema changes, reprocess
  from `raw_json/` rather than re-fetching — the feed is gone once it scrolls off.
- **Dedup is global by `item_id`** within each day-partition. Posts whose
  `published_at` predates the run (back-dated feeds) land in the correct day file.
- This slots into the larger pipeline as the CryptoPanic Layer-0/Layer-1 path; the
  normalized schema is intentionally a superset of `RawNewsItem`.

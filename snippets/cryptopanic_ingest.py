"""
CryptoPanic ingestion / processing batch.

PURPOSE
-------
The CryptoPanic API is forward-only: it exposes the *current* feed with cursor
pagination, and has no date-range or historical query. The only way to obtain a
CryptoPanic history is to start polling now and accumulate. This module is that
accumulator. Run it on a schedule (cron / systemd timer / APScheduler); every run
fetches everything new since the last run, normalizes it, deduplicates against
what has already been stored, and appends to a partitioned parquet store.

DESIGN GUARANTEES
-----------------
1. No data loss across restarts: a high-water mark (newest post id + timestamp
   seen) is persisted; each run pages backward from "now" until it reaches the
   high-water mark, so a missed run simply means the next run pages back further.
2. Idempotent: posts are keyed by a stable post_id; re-ingesting the same post is
   a no-op. Safe to run as often as you like (subject to the API's 30s cache).
3. Append-only, partitioned storage: raw/cryptopanic/YYYY-MM-DD.parquet, matching
   the pipeline's Layer-1 convention so this slots straight into the backtest
   feature store later.
4. Raw + normalized separation: the untouched API JSON is archived per run
   (raw_json/) so reprocessing is always possible if the normalized schema changes.

This file is intentionally dependency-light (httpx, pydantic, pyarrow, pandas) and
self-contained so it can run unattended on a small box.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import httpx
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, Field, field_validator

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

API_BASE = "https://cryptopanic.com/api/{plan}/v2/posts/"
DEFAULT_PLAN = os.getenv("CRYPTOPANIC_PLAN", "developer")  # "developer" | "pro" | ...
AUTH_TOKEN_ENV = "CRYPTOPANIC_AUTH_TOKEN"

# API etiquette (from CryptoPanic docs): server-side cache ~30s, rate limit 5 req/s.
MIN_SECONDS_BETWEEN_REQUESTS = 1.0        # well under the 5 req/s ceiling
SERVER_CACHE_SECONDS = 30                 # no point polling faster than this
MAX_PAGES_PER_RUN = 50                    # safety stop; 50 pages * ~20 posts = 1000
REQUEST_TIMEOUT = 30.0
MAX_RETRIES = 4

DATA_ROOT = Path(os.getenv("CRYPTOPANIC_DATA_ROOT", "./data/cryptopanic"))
NORMALIZED_DIR = DATA_ROOT / "normalized"     # partitioned parquet, Layer-1 schema
RAW_JSON_DIR = DATA_ROOT / "raw_json"         # untouched API responses, gz per run
STATE_FILE = DATA_ROOT / "state.json"         # high-water mark + run bookkeeping

log = logging.getLogger("cryptopanic_ingest")


# --------------------------------------------------------------------------- #
# Normalized schema  (Layer-1 RawNewsItem, CryptoPanic-specialized)
# --------------------------------------------------------------------------- #

class CryptoPanicPost(BaseModel):
    """One normalized CryptoPanic post. Mirrors the pipeline's RawNewsItem with
    CryptoPanic-specific fields retained (votes, panic_score) for later use."""

    item_id: str                       # stable dedup key: "cryptopanic:{post_id}"
    post_id: int                       # raw CryptoPanic id
    source: str = "cryptopanic"
    published_at: datetime             # point-in-time anchor (UTC)
    created_at: datetime | None        # when CryptoPanic ingested it (UTC)
    ingested_at: datetime              # when WE fetched it (UTC) — true availability
    title: str
    description: str = ""
    url: str                           # cryptopanic permalink
    original_url: str = ""             # underlying article URL
    source_domain: str = ""
    source_type: str = ""              # "feed" | "blog" | "twitter" | "reddit" | ...
    kind: str = ""                     # "news" | "media" | "blog" | ...
    asset_mentions: list[str] = Field(default_factory=list)  # ["BTC","ETH"]
    panic_score: float | None = None
    votes_positive: int = 0            # liked + important + positive + bullish
    votes_negative: int = 0            # disliked + negative + bearish + toxic
    votes_raw: dict[str, int] = Field(default_factory=dict)

    @field_validator("published_at", "created_at", "ingested_at")
    @classmethod
    def _ensure_utc(cls, v: datetime | None) -> datetime | None:
        if v is None:
            return v
        return v.astimezone(timezone.utc) if v.tzinfo else v.replace(tzinfo=timezone.utc)


def normalize_post(raw: dict[str, Any], ingested_at: datetime) -> CryptoPanicPost:
    """Map one raw API post dict to the normalized schema. Tolerant of missing
    fields and of v1/v2 shape differences (instruments vs currencies)."""
    post_id = int(raw["id"])

    # Asset mentions: v2 uses "instruments", older shapes use "currencies".
    instruments = raw.get("instruments") or raw.get("currencies") or []
    assets = []
    for inst in instruments:
        code = (inst or {}).get("code")
        if code:
            assets.append(code.upper())
    assets = sorted(set(assets))

    src = raw.get("source") or {}
    votes = raw.get("votes") or {}

    pos = sum(int(votes.get(k, 0)) for k in ("liked", "important", "positive", "bullish", "saved"))
    neg = sum(int(votes.get(k, 0)) for k in ("disliked", "negative", "bearish", "toxic", "lol"))

    pub = _parse_dt(raw.get("published_at"))
    created = _parse_dt(raw.get("created_at"))

    return CryptoPanicPost(
        item_id=f"cryptopanic:{post_id}",
        post_id=post_id,
        published_at=pub or ingested_at,   # fall back to fetch time if absent
        created_at=created,
        ingested_at=ingested_at,
        title=raw.get("title") or "",
        description=raw.get("description") or "",
        url=raw.get("url") or "",
        original_url=raw.get("original_url") or raw.get("source", {}).get("url", "") or "",
        source_domain=src.get("domain") or "",
        source_type=src.get("type") or "",
        kind=raw.get("kind") or "",
        asset_mentions=assets,
        panic_score=_to_float(raw.get("panic_score")),
        votes_positive=pos,
        votes_negative=neg,
        votes_raw={k: int(v) for k, v in votes.items() if isinstance(v, (int, float))},
    )


def _parse_dt(s: Any) -> datetime | None:
    if not s:
        return None
    if isinstance(s, datetime):
        return s
    try:
        # API returns ISO 8601, e.g. "2021-06-01T12:34:56Z"
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _to_float(x: Any) -> float | None:
    try:
        return float(x) if x is not None else None
    except (ValueError, TypeError):
        return None


# --------------------------------------------------------------------------- #
# State (high-water mark)
# --------------------------------------------------------------------------- #

class IngestState(BaseModel):
    last_post_id: int = 0                 # newest post_id ever stored
    last_published_at: datetime | None = None
    last_run_at: datetime | None = None
    total_posts: int = 0

    @classmethod
    def load(cls, path: Path) -> "IngestState":
        if path.exists():
            return cls.model_validate_json(path.read_text())
        return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))


# --------------------------------------------------------------------------- #
# HTTP fetching with pagination + retry
# --------------------------------------------------------------------------- #

class RateLimiter:
    """Trivial monotonic-clock spacing between requests."""
    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last = time.monotonic()


def _build_first_url(plan: str, token: str, extra_params: dict[str, str] | None) -> str:
    params = {"auth_token": token}
    if extra_params:
        params.update(extra_params)
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{API_BASE.format(plan=plan)}?{qs}"


def fetch_pages(
    client: httpx.Client,
    first_url: str,
    limiter: RateLimiter,
    stop_at_post_id: int,
    max_pages: int = MAX_PAGES_PER_RUN,
) -> Iterator[list[dict[str, Any]]]:
    """Page from 'now' backward via the 'next' cursor. Yields each page's raw
    results list. Stops when: a post_id <= stop_at_post_id is seen (caught up),
    the cursor runs out, or max_pages is hit (safety)."""
    url: str | None = first_url
    pages = 0

    while url and pages < max_pages:
        limiter.wait()
        data = _get_with_retry(client, url)
        results = data.get("results", []) or []
        if not results:
            break

        yield results
        pages += 1

        # Caught up? If the oldest post on this page is already stored, the next
        # page is entirely old; stop after yielding this one.
        oldest_id_on_page = min(int(r["id"]) for r in results)
        if oldest_id_on_page <= stop_at_post_id:
            log.info("Reached high-water mark (post_id=%s); stopping pagination.", stop_at_post_id)
            break

        url = data.get("next")

    if pages >= max_pages:
        log.warning("Hit MAX_PAGES_PER_RUN=%s; there may be a gap. Increase cadence.", max_pages)


def _get_with_retry(client: httpx.Client, url: str) -> dict[str, Any]:
    backoff = 2.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", backoff))
                log.warning("429 rate limited; sleeping %.1fs (attempt %d)", wait, attempt)
                time.sleep(wait)
                backoff *= 2
                continue
            resp.raise_for_status()
            return resp.json()
        except (httpx.TimeoutException, httpx.TransportError) as e:
            if attempt == MAX_RETRIES:
                raise
            log.warning("Transport error (%s); retry %d in %.1fs", e, attempt, backoff)
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError("unreachable")


# --------------------------------------------------------------------------- #
# Storage (partitioned parquet, append-only, dedup)
# --------------------------------------------------------------------------- #

def archive_raw_json(pages: list[list[dict[str, Any]]], run_ts: datetime) -> None:
    """Persist the untouched API payload for this run, gzip-compressed."""
    RAW_JSON_DIR.mkdir(parents=True, exist_ok=True)
    fname = RAW_JSON_DIR / f"{run_ts:%Y%m%dT%H%M%SZ}.json.gz"
    flat = [post for page in pages for post in page]
    with gzip.open(fname, "wt", encoding="utf-8") as f:
        json.dump(flat, f)
    log.info("Archived %d raw posts to %s", len(flat), fname.name)


def write_normalized(posts: list[CryptoPanicPost]) -> int:
    """Append normalized posts to per-day parquet partitions, deduplicating by
    item_id against whatever is already stored for that day. Returns the number
    of genuinely new rows written."""
    if not posts:
        return 0

    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([p.model_dump() for p in posts])
    # Partition by the publication date (UTC) — matches Layer-1 raw/news convention.
    df["_date"] = pd.to_datetime(df["published_at"], utc=True).dt.date

    new_rows = 0
    for day, group in df.groupby("_date"):
        part = NORMALIZED_DIR / f"{day.isoformat()}.parquet"
        group = group.drop(columns=["_date"])

        if part.exists():
            existing = pd.read_parquet(part)
            known = set(existing["item_id"])
            fresh = group[~group["item_id"].isin(known)]
            if fresh.empty:
                continue
            combined = pd.concat([existing, fresh], ignore_index=True)
        else:
            fresh = group.drop_duplicates(subset="item_id")
            combined = fresh

        combined = combined.sort_values("published_at").reset_index(drop=True)
        # complex/nested columns (votes_raw, asset_mentions) survive via pyarrow
        pq.write_table(pa.Table.from_pandas(combined, preserve_index=False), part)
        new_rows += len(fresh)
        log.info("Partition %s: +%d new (total %d)", part.name, len(fresh), len(combined))

    return new_rows


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run_ingest(
    plan: str = DEFAULT_PLAN,
    token: str | None = None,
    extra_params: dict[str, str] | None = None,
) -> dict[str, Any]:
    """One full ingestion pass. Returns a small run summary dict."""
    token = token or os.getenv(AUTH_TOKEN_ENV)
    if not token:
        raise RuntimeError(f"Set {AUTH_TOKEN_ENV} (your CryptoPanic auth token).")

    state = IngestState.load(STATE_FILE)
    run_ts = datetime.now(timezone.utc)
    limiter = RateLimiter(MIN_SECONDS_BETWEEN_REQUESTS)
    first_url = _build_first_url(plan, token, extra_params)

    pages: list[list[dict[str, Any]]] = []
    with httpx.Client(headers={"User-Agent": "research-ingest/1.0"}) as client:
        for page in fetch_pages(client, first_url, limiter, stop_at_post_id=state.last_post_id):
            pages.append(page)

    if not pages:
        log.info("No pages returned; nothing to do.")
        state.last_run_at = run_ts
        state.save(STATE_FILE)
        return {"new_posts": 0, "pages": 0, "run_ts": run_ts.isoformat()}

    # Archive raw, then normalize.
    archive_raw_json(pages, run_ts)
    raw_posts = [p for page in pages for p in page]
    normalized = [normalize_post(r, ingested_at=run_ts) for r in raw_posts]

    # Keep only posts strictly newer than the high-water mark (dedup is also done
    # at storage time, but this trims work early and keeps the watermark honest).
    normalized = [p for p in normalized if p.post_id > state.last_post_id]
    new_rows = write_normalized(normalized)

    # Advance the high-water mark to the newest post seen this run.
    if raw_posts:
        newest = max(int(r["id"]) for r in raw_posts)
        newest_pub = max(
            (p.published_at for p in normalized), default=state.last_published_at
        )
        state.last_post_id = max(state.last_post_id, newest)
        state.last_published_at = newest_pub
    state.last_run_at = run_ts
    state.total_posts += new_rows
    state.save(STATE_FILE)

    summary = {
        "new_posts": new_rows,
        "pages": len(pages),
        "high_water_post_id": state.last_post_id,
        "run_ts": run_ts.isoformat(),
    }
    log.info("Run complete: %s", summary)
    return summary


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # Optional: restrict to news kind, or to specific currencies, via env.
    extra: dict[str, str] = {}
    if os.getenv("CRYPTOPANIC_KIND"):           # e.g. "news"
        extra["kind"] = os.environ["CRYPTOPANIC_KIND"]
    if os.getenv("CRYPTOPANIC_CURRENCIES"):     # e.g. "BTC,ETH,SOL"
        extra["currencies"] = os.environ["CRYPTOPANIC_CURRENCIES"]
    run_ingest(extra_params=extra or None)


if __name__ == "__main__":
    main()

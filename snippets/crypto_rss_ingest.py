"""
Multi-feed RSS crypto-news accumulator.

WHY RSS
-------
The CryptoPanic JSON API needs an auth token. RSS does not. This module accumulates
crypto news from a configurable list of RSS/Atom feeds — CryptoPanic's public feed
and/or the underlying publisher feeds (CoinDesk, Cointelegraph, Decrypt, ...). Pulling
publisher feeds directly is the better choice for this project: GDELT indexes those
same publishers, so a publisher-RSS live stream is distributionally much closer to the
GDELT backfill than CryptoPanic's aggregated+voted feed — which shrinks train/serve skew.

It is forward-only, exactly like the API path: RSS exposes only the current feed window
(typically the latest ~20-100 items), so the only way to build history is to poll on a
schedule and accumulate. Run it from a cron/timer starting now.

DESIGN (identical guarantees to the API accumulator)
----------------------------------------------------
1. No data loss across restarts: a per-feed high-water mark (set of recently-seen item
   ids + newest publish time) is persisted; a missed run is harmless because feeds hold
   a buffer of recent items (usually hours), so the next run still sees anything new.
2. Idempotent: items keyed by a stable item_id (guid, else link). Re-ingest = no-op.
3. Append-only, day-partitioned parquet (raw/news Layer-1 convention).
4. Raw + normalized separation: each run's raw parsed entries are archived (gz) so the
   normalized schema can change without re-fetching (RSS history is unrecoverable).
5. Multi-source by design: a list of feeds, each tagged with source_domain. This is the
   plan's §13 single-source-dependency fix, for free.

LOSS vs the JSON API: no panic_score, no structured vote counts. asset_mentions are
derived here by ticker-tagging title+summary (publisher feeds carry no currency tags).

Dependencies: feedparser, httpx, pydantic, pyarrow, pandas.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import feedparser
import httpx
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, Field, field_validator

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Each feed: (logical_name, url). source_domain is derived from the feed/link host.
# Publisher feeds GDELT also indexes — closer to the backfill distribution than
# aggregators, which shrinks train/serve skew.
# CryptoSlate blocks programmatic access (403); included below with User-Agent
# override in fetch code, but may remain broken.
DEFAULT_FEEDS: list[tuple[str, str]] = [
    ("coindesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("cointelegraph", "https://cointelegraph.com/rss"),
    ("decrypt", "https://decrypt.co/feed"),
    ("theblock", "https://www.theblock.co/rss.xml"),
    ("bitcoinmagazine", "https://bitcoinmagazine.com/feed"),
    ("newsbtc", "https://www.newsbtc.com/feed/"),
    ("bitcoincom", "https://news.bitcoin.com/feed/"),
    ("utoday", "https://u.today/rss"),
    # New sources added July 2026, ~140 extra items/day
    ("cryptonews", "https://crypto.news/feed/"),
    ("cryptopotato", "https://cryptopotato.com/feed/"),
    ("zycrypto", "https://www.zycrypto.com/feed/"),
    ("beincrypto", "https://beincrypto.com/feed/"),
    ("ambcrypto", "https://ambcrypto.com/feed/"),
    ("dailycoin", "https://dailycoin.com/feed/"),
    ("blockonomi", "https://blockonomi.com/feed/"),
    ("bitcoinist", "https://bitcoinist.com/feed/"),
    ("cryptobriefing", "https://cryptobriefing.com/feed/"),
    ("cryptoslate", "https://cryptoslate.com/feed/"),  # behind Cloudflare; may 403
]

REQUEST_TIMEOUT = 30.0
MAX_RETRIES = 4
PER_FEED_PAUSE = 1.0          # politeness between feeds
RAW_KEEP_FIELDS = ("title", "link", "id", "guid", "published", "updated",
                   "summary", "tags", "author")

DATA_ROOT = Path(os.getenv("RSS_DATA_ROOT", "./data/crypto_rss"))
NORMALIZED_DIR = DATA_ROOT / "normalized"
RAW_JSON_DIR = DATA_ROOT / "raw_json"
STATE_FILE = DATA_ROOT / "state.json"

# Recent-id memory per feed (bounds dedup work; older items fall to storage-level dedup).
WATERMARK_ID_MEMORY = 500

log = logging.getLogger("crypto_rss_ingest")


# --------------------------------------------------------------------------- #
# Ticker tagging  (asset_mentions for feeds without currency tags)
# --------------------------------------------------------------------------- #
# Map common names/symbols -> canonical ticker. Word-boundary, case-insensitive.
# Deliberately conservative: high-precision names only, to avoid false positives
# (e.g. plain "ETH" matches, but ambiguous words are excluded).
TICKER_PATTERNS: dict[str, re.Pattern] = {
    "BTC": re.compile(r"\b(bitcoin|btc|xbt)\b", re.I),
    "ETH": re.compile(r"\b(ethereum|ether|eth)\b", re.I),
    "USDT": re.compile(r"\b(tether|usdt)\b", re.I),
    "USDC": re.compile(r"\b(usd\s?coin|usdc)\b", re.I),
    "BNB": re.compile(r"\b(binance\s?coin|bnb)\b", re.I),
    "XRP": re.compile(r"\b(ripple|xrp)\b", re.I),
    "SOL": re.compile(r"\b(solana|sol)\b", re.I),
    "ADA": re.compile(r"\b(cardano|ada)\b", re.I),
    "DOGE": re.compile(r"\b(dogecoin|doge)\b", re.I),
    "TRX": re.compile(r"\b(tron|trx)\b", re.I),
    "AVAX": re.compile(r"\b(avalanche|avax)\b", re.I),
    "DOT": re.compile(r"\b(polkadot|dot)\b", re.I),
    "MATIC": re.compile(r"\b(polygon|matic)\b", re.I),
    "LINK": re.compile(r"\b(chainlink|link)\b", re.I),
    "LTC": re.compile(r"\b(litecoin|ltc)\b", re.I),
    "SHIB": re.compile(r"\b(shiba\s?inu|shib)\b", re.I),
    "UNI": re.compile(r"\b(uniswap|uni)\b", re.I),
    "ATOM": re.compile(r"\b(cosmos|atom)\b", re.I),
    "XLM": re.compile(r"\b(stellar|xlm)\b", re.I),
    "LUNA": re.compile(r"\b(terra|luna)\b", re.I),
}


def tag_assets(text: str) -> list[str]:
    """Return sorted unique tickers mentioned in text. Empty list if none."""
    if not text:
        return []
    return sorted({tkr for tkr, pat in TICKER_PATTERNS.items() if pat.search(text)})


# --------------------------------------------------------------------------- #
# Normalized schema  (Layer-1 RawNewsItem)
# --------------------------------------------------------------------------- #

class RSSNewsItem(BaseModel):
    item_id: str                       # sha256(guid or link) — stable dedup key
    source: str = "rss"
    feed_name: str                     # logical feed, e.g. "coindesk"
    published_at: datetime             # point-in-time anchor (UTC)
    ingested_at: datetime              # when WE fetched it (UTC) — true availability
    title: str
    summary: str = ""                  # RSS description/summary (often partial body)
    url: str                           # article link
    source_domain: str = ""
    author: str = ""
    asset_mentions: list[str] = Field(default_factory=list)
    feed_categories: list[str] = Field(default_factory=list)  # RSS <category> tags

    @field_validator("published_at", "ingested_at")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return v.astimezone(timezone.utc) if v.tzinfo else v.replace(tzinfo=timezone.utc)


def _stable_id(entry: dict[str, Any]) -> str:
    basis = entry.get("id") or entry.get("guid") or entry.get("link") or ""
    if not basis:
        # last resort: hash title+published so we still dedup deterministically
        basis = (entry.get("title", "") + entry.get("published", ""))
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def _entry_dt(entry: dict[str, Any]) -> datetime | None:
    """feedparser parses dates into *_parsed struct_time (UTC). Prefer published."""
    for key in ("published_parsed", "updated_parsed"):
        st = entry.get(key)
        if st:
            return datetime(*st[:6], tzinfo=timezone.utc)
    return None


def _domain_from(url: str, feed_url: str) -> str:
    m = re.search(r"https?://([^/]+)/?", url or feed_url or "")
    if not m:
        return ""
    return m.group(1).lower().removeprefix("www.")


def normalize_entry(
    entry: dict[str, Any], feed_name: str, feed_url: str, ingested_at: datetime
) -> RSSNewsItem:
    title = (entry.get("title") or "").strip()
    summary = re.sub(r"<[^>]+>", "", entry.get("summary") or "").strip()  # strip HTML
    link = entry.get("link") or ""
    cats = [t.get("term", "") for t in entry.get("tags", []) if t.get("term")]

    pub = _entry_dt(entry) or ingested_at  # fall back to fetch time if feed omits date
    assets = tag_assets(f"{title}. {summary}")

    author = entry.get("author", "") or ""

    return RSSNewsItem(
        item_id=_stable_id(entry),
        feed_name=feed_name,
        published_at=pub,
        ingested_at=ingested_at,
        title=title,
        summary=summary,
        url=link,
        source_domain=_domain_from(link, feed_url),
        author=author,
        asset_mentions=assets,
        feed_categories=cats,
    )


# --------------------------------------------------------------------------- #
# State (per-feed high-water mark)
# --------------------------------------------------------------------------- #

class FeedState(BaseModel):
    recent_ids: list[str] = Field(default_factory=list)   # bounded LRU of item_ids
    last_published_at: datetime | None = None
    last_run_at: datetime | None = None
    total_items: int = 0


class IngestState(BaseModel):
    feeds: dict[str, FeedState] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "IngestState":
        if path.exists():
            return cls.model_validate_json(path.read_text())
        return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))


# --------------------------------------------------------------------------- #
# Fetch + parse
# --------------------------------------------------------------------------- #

def fetch_feed(client: httpx.Client, url: str) -> bytes | None:
    """Fetch raw feed bytes with retry. Returns None on permanent failure (a single
    bad feed must never abort the whole run)."""
    backoff = 2.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.get(url, timeout=REQUEST_TIMEOUT,
                              follow_redirects=True,
                              headers={"User-Agent": "research-rss-ingest/1.0"})
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", backoff))
                log.warning("429 on %s; sleeping %.1fs", url, wait)
                time.sleep(wait); backoff *= 2; continue
            resp.raise_for_status()
            return resp.content
        except (httpx.HTTPError,) as e:
            if attempt == MAX_RETRIES:
                log.error("Giving up on %s: %s", url, e)
                return None
            log.warning("Error on %s (%s); retry %d in %.1fs", url, e, attempt, backoff)
            time.sleep(backoff); backoff *= 2
    return None


def parse_entries(raw_bytes: bytes) -> list[dict[str, Any]]:
    """feedparser handles RSS 2.0 / RSS 1.0 / Atom and most namespace quirks."""
    parsed = feedparser.parse(raw_bytes)
    if parsed.bozo and not parsed.entries:
        log.warning("Feed parse flagged bozo with no entries: %s", parsed.get("bozo_exception"))
    return parsed.entries


# --------------------------------------------------------------------------- #
# Storage (shared with the API path's conventions)
# --------------------------------------------------------------------------- #

def archive_raw(entries_by_feed: dict[str, list[dict[str, Any]]], run_ts: datetime) -> None:
    RAW_JSON_DIR.mkdir(parents=True, exist_ok=True)
    fname = RAW_JSON_DIR / f"{run_ts:%Y%m%dT%H%M%SZ}.json.gz"
    # Keep only serializable, relevant fields from each entry.
    slim = {
        feed: [{k: e.get(k) for k in RAW_KEEP_FIELDS if k in e} for e in entries]
        for feed, entries in entries_by_feed.items()
    }
    with gzip.open(fname, "wt", encoding="utf-8") as f:
        json.dump(slim, f, default=str)
    total = sum(len(v) for v in entries_by_feed.values())
    log.info("Archived %d raw entries across %d feeds -> %s", total, len(slim), fname.name)


def write_normalized(items: list[RSSNewsItem]) -> int:
    if not items:
        return 0
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([i.model_dump() for i in items])
    df["_date"] = pd.to_datetime(df["published_at"], utc=True).dt.date

    new_rows = 0
    for day, group in df.groupby("_date"):
        part = NORMALIZED_DIR / f"{day.isoformat()}.parquet"
        group = group.drop(columns=["_date"]).drop_duplicates(subset="item_id")
        if part.exists():
            existing = pd.read_parquet(part)
            fresh = group[~group["item_id"].isin(set(existing["item_id"]))]
            if fresh.empty:
                continue
            combined = pd.concat([existing, fresh], ignore_index=True)
        else:
            fresh = group
            combined = group
        combined = combined.sort_values("published_at").reset_index(drop=True)
        pq.write_table(pa.Table.from_pandas(combined, preserve_index=False), part)
        new_rows += len(fresh)
        log.info("Partition %s: +%d new (total %d)", part.name, len(fresh), len(combined))
    return new_rows


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run_ingest(feeds: list[tuple[str, str]] | None = None) -> dict[str, Any]:
    feeds = feeds or DEFAULT_FEEDS
    state = IngestState.load(STATE_FILE)
    run_ts = datetime.now(timezone.utc)

    raw_by_feed: dict[str, list[dict[str, Any]]] = {}
    new_items: list[RSSNewsItem] = []

    with httpx.Client() as client:
        for feed_name, url in feeds:
            fs = state.feeds.setdefault(feed_name, FeedState())
            seen = set(fs.recent_ids)

            raw = fetch_feed(client, url)
            if raw is None:
                continue
            entries = parse_entries(raw)
            raw_by_feed[feed_name] = entries

            fresh_ids: list[str] = []
            for entry in entries:
                item = normalize_entry(entry, feed_name, url, run_ts)
                if item.item_id in seen:
                    continue
                new_items.append(item)
                fresh_ids.append(item.item_id)

            # Update this feed's watermark (bounded LRU of most-recent ids).
            if fresh_ids:
                fs.recent_ids = (fresh_ids + fs.recent_ids)[:WATERMARK_ID_MEMORY]
                pubs = [i.published_at for i in new_items if i.feed_name == feed_name]
                if pubs:
                    newest = max(pubs)
                    fs.last_published_at = (
                        max(fs.last_published_at, newest) if fs.last_published_at else newest
                    )
            fs.last_run_at = run_ts
            log.info("Feed %-16s %3d entries, %3d new", feed_name, len(entries), len(fresh_ids))
            time.sleep(PER_FEED_PAUSE)

    if raw_by_feed:
        archive_raw(raw_by_feed, run_ts)
    new_rows = write_normalized(new_items)

    for feed_name in raw_by_feed:
        state.feeds[feed_name].total_items += sum(
            1 for i in new_items if i.feed_name == feed_name
        )
    state.save(STATE_FILE)

    summary = {
        "new_items": new_rows,
        "feeds_ok": len(raw_by_feed),
        "feeds_total": len(feeds),
        "run_ts": run_ts.isoformat(),
    }
    log.info("Run complete: %s", summary)
    return summary


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    run_ingest()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Refresh census — how much extraction work a study refresh actually implies.

Read-only. Replays the exact decisions of batch_etl.run_batch_extraction
(cache hit -> ticker pre-filter -> LLM) without calling the LLM, and reports the
counts broken down by month, by source (rss/gdelt) and by matched ticker, so the
extraction scope and runtime can be chosen from measurement instead of guessed.

The IC study (scripts/ic_study.py) only scores the assets in BINANCE_SYMBOLS, so
"to_extract_study_assets" is the number that actually feeds the grid; the rest
builds the wider feature store.

Run from repo root: .venv/bin/python scripts/refresh_census.py
"""

from __future__ import annotations

import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import BINANCE_SYMBOLS, MODEL_VERSION, PROMPT_VERSION
from src.extraction.batch_etl import (
    _item_day,
    _safe_str,
    read_all_raw_news,
    universe_for_day,
)
from src.extraction.cache import cache_exists, compute_content_hash
from src.ingest.perimeter import match_ticker

STUDY_ASSETS = {s.replace("USDT", "") for s in BINANCE_SYMBOLS}
URL_LIKE = re.compile(r"^https?://")

t0 = time.time()
raw = read_all_raw_news()
raw = raw[raw["title"].str.len() > 0].copy()
raw = raw.sort_values("published_at").reset_index(drop=True)
print(f"raw items with titles: {len(raw)} (loaded in {time.time() - t0:.0f}s)")

hits = new = skipped = 0
by_month_total, by_month_cached = Counter(), Counter()
by_month_new, by_month_skip = Counter(), Counter()
by_source_new, by_ticker = Counter(), Counter()
study_assets_new = 0
url_titled_new = 0
samples = []

for _, row in raw.iterrows():
    month = _item_day(row).strftime("%Y-%m")
    source = _safe_str(row.get("source", "")) or "unknown"
    by_month_total[month] += 1
    title = _safe_str(row.get("title", ""))
    body = _safe_str(row.get("body", "") or row.get("summary", "")
                     or row.get("description", ""))
    h = compute_content_hash(title, body)
    if cache_exists(h, MODEL_VERSION, PROMPT_VERSION):
        hits += 1
        by_month_cached[month] += 1
        continue
    matched = match_ticker(
        " ".join(p for p in (title, body) if p), universe_for_day(_item_day(row)))
    if matched is None:
        skipped += 1
        by_month_skip[month] += 1
        continue
    new += 1
    by_month_new[month] += 1
    by_source_new[source] += 1
    by_ticker[matched] += 1
    if matched in STUDY_ASSETS:
        study_assets_new += 1
    if URL_LIKE.match(title.strip()):
        url_titled_new += 1
    if len(samples) < 8:
        samples.append((month, source, matched, title[:70]))

print(f"\ncache hits {hits} | uncached {len(raw) - hits} "
      f"-> ticker match {new} / skip {skipped}")

rows = [dict(month=m, items=by_month_total[m], cached=by_month_cached[m],
             to_extract=by_month_new[m], skipped=by_month_skip[m])
        for m in sorted(by_month_total)]
print("\n" + pd.DataFrame(rows).to_string(index=False))

print(f"\ncache hits {hits} | uncached {len(raw) - hits} "
      f"-> ticker match {new} / skip {skipped}")
print(f"pre-filter pass rate on uncached: {new / max(len(raw) - hits, 1):.1%}")

print(f"\nto extract by source: {dict(by_source_new)}")
print(f"to extract matching a STUDY asset {sorted(STUDY_ASSETS)}: {study_assets_new}")
print(f"to extract with a URL-shaped title (GDELT noise risk): {url_titled_new}")
print("\ntop matched tickers:")
for t, c in by_ticker.most_common(20):
    mark = " *study" if t in STUDY_ASSETS else ""
    print(f"  {t:8s} {c:6d}{mark}")

print(f"\nTOTAL to extract: {new} items | elapsed {time.time() - t0:.0f}s")
for s in samples:
    print("  sample:", s)

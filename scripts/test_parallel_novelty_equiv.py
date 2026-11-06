#!/usr/bin/env python3
"""Prove the parallel driver's novelty shortcut equals the sequential one.

batch_etl.run_batch_extraction scores novelty against the WHOLE cached pool every
time, masked to asset == a and published_at in [t-48h, t). The parallel driver
pass 3 keeps a 48h trailing deque instead. These must agree exactly, otherwise
the feature store silently changes meaning.

This test takes real cached records, computes novelty twice per record — once
against the full pool (reference), once against the deque window built in
published_at order (driver) — and asserts equality.

Run from repo root: .venv/bin/python scripts/test_parallel_novelty_equiv.py
"""

from __future__ import annotations

import sys
from collections import deque
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import MODEL_VERSION, PROMPT_VERSION
from src.extraction.cache import load_all_cached
from src.extraction.novelty import compute_novelty

LOOKBACK_HOURS = 48
TOL = 1e-12

pool = load_all_cached(MODEL_VERSION, PROMPT_VERSION)
pool["published_at"] = pd.to_datetime(pool["published_at"], utc=True, errors="coerce")
pool = pool.dropna(subset=["published_at"]).sort_values("published_at")
print(f"pool: {len(pool)} cached records, "
      f"{pool['published_at'].min()} .. {pool['published_at'].max()}")

# Sample records spread across the corpus, incl. the very first ones (empty window)
sample = pd.concat([pool.head(5), pool.iloc[:: max(len(pool) // 300, 1)]]).drop_duplicates(
    subset=["content_hash"])
print(f"sampling {len(sample)} records for the equivalence check")

pool_sorted = pool[["asset", "published_at", "content_hash", "title",
                    "event_type"]].itertuples(index=False, name=None)
rows = list(pool_sorted)

ptr = 0
window: deque = deque()
mismatches = 0
checked = 0

for rec in sample.itertuples(index=False):
    t = rec.published_at
    while ptr < len(rows) and rows[ptr][1] < t:
        window.append(rows[ptr])
        ptr += 1
    cutoff = t - timedelta(hours=LOOKBACK_HOURS)
    while window and window[0][1] < cutoff:
        window.popleft()

    event = dict(asset=rec.asset, event_type=rec.event_type, published_at=t,
                 content_hash=rec.content_hash, title=rec.title)

    # driver path: pre-filtered deque window
    wrows = [w for w in window if w[0] == rec.asset and w[2] != rec.content_hash]
    wdf = pd.DataFrame(wrows, columns=["asset", "published_at", "content_hash",
                                       "title", "event_type"])
    fast = compute_novelty(event, wdf)
    # reference path: the full pool
    ref = compute_novelty(event, pool)

    checked += 1
    if abs(fast - ref) > TOL:
        mismatches += 1
        if mismatches <= 5:
            print(f"  MISMATCH {rec.content_hash[:8]} {rec.asset} {t}: "
                  f"driver={fast!r} reference={ref!r}")

print(f"\nchecked {checked} records, {mismatches} mismatches")
assert mismatches == 0, "parallel novelty shortcut is NOT equivalent — do not use the driver"
print("EQUIVALENT: the parallel driver reproduces sequential novelty exactly")

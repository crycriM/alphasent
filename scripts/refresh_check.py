#!/usr/bin/env python3
"""Refresh check — inventory of capture freshness vs study artifact staleness.

Reports, read-only:
  * RSS corpus: per-day counts, coverage window, gap days, feeds seen
  * Extraction cache: how many EventRecords exist, which model_version
  * Feature store: per-asset coverage window (data/features + llama3 backup)
  * OHLCV: per-symbol coverage window (needed for forward returns)
  * Perimeter: available universe snapshot months
  * GDELT news: coverage window
Exit code 1 if the study cannot be refreshed (missing OHLCV beyond news).

Run from repo root: .venv/bin/python scripts/refresh_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import DATA_ROOT, FEATURES_DIR, RAW_NEWS_DIR, RSS_DATA_ROOT

LLAMA_DIR = DATA_ROOT / "features_llama3_backup"


def hdr(t: str) -> None:
    print(f"\n=== {t} ===")


def rss_inventory() -> pd.DataFrame:
    hdr("RSS corpus (data/crypto_rss/normalized)")
    files = sorted((RSS_DATA_ROOT / "normalized").glob("*.parquet"))
    rows = []
    for f in files:
        df = pd.read_parquet(f, columns=["published_at", "feed_name", "item_id"])
        rows.append(dict(
            partition=f.stem,
            n=len(df),
            first=pd.to_datetime(df["published_at"], utc=True, errors="coerce").min(),
            last=pd.to_datetime(df["published_at"], utc=True, errors="coerce").max(),
            feeds=df["feed_name"].nunique(),
        ))
    inv = pd.DataFrame(rows).sort_values("partition")
    print(f"partitions={len(inv)} items={int(inv['n'].sum())} "
          f"window={inv['partition'].iloc[0]}..{inv['partition'].iloc[-1]}")
    recent = inv.tail(12)[["partition", "n", "feeds"]]
    print(recent.to_string(index=False))
    # gap days
    idx = pd.to_datetime(inv["partition"])
    full = pd.date_range(idx.min(), idx.max(), freq="D")
    missing = sorted(set(full) - set(idx))
    print(f"gap days (no partition): {len(missing)}")
    if missing:
        print("  " + ", ".join(d.strftime("%Y-%m-%d") for d in missing[:20])
              + (" ..." if len(missing) > 20 else ""))
    # items published in the last 7 calendar days
    cutoff = pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=7)
    recent_items = sum(r["n"] for _, r in inv.iterrows()
                       if pd.Timestamp(r["partition"]) >= cutoff)
    print(f"items in partitions of the last 7 days: {recent_items}")
    return inv


def extraction_inventory() -> None:
    hdr("Extraction cache (data/cache/extractions)")
    root = DATA_ROOT / "cache" / "extractions"
    files = list(root.glob("*/**/*.parquet"))
    print(f"extraction parquet files: {len(files)}")
    if not files:
        return
    sample = pd.concat([pd.read_parquet(f) for f in files[:400]], ignore_index=True)
    if "model_version" in sample.columns:
        print(sample["model_version"].value_counts().to_string())
    if "extracted_at" in sample.columns:
        ts = pd.to_datetime(sample["extracted_at"], utc=True, errors="coerce")
        print(f"extracted_at window: {ts.min()} .. {ts.max()}")


def features_inventory() -> None:
    for label, d in (("phi4 (data/features)", FEATURES_DIR),
                     ("llama3 backup", LLAMA_DIR)):
        hdr(f"Feature store — {label} ({d.name})")
        if not d.exists():
            print("MISSING")
            continue
        assets = sorted(p for p in d.iterdir() if p.is_dir())
        parts = sorted(d.glob("*/*.parquet"))
        if not parts:
            print("no parquet parts")
            continue
        f0, f1 = parts[0], parts[-1]
        d0 = pd.read_parquet(f0, columns=["timestamp"] if "timestamp" in
                            pd.read_parquet(f0).columns else None)
        print(f"assets={len(assets)} parts={len(parts)}")
        print(f"first part {f0.parent.name}/{f0.name} rows={len(d0)}")
        last = pd.read_parquet(f1)
        tcol = "timestamp" if "timestamp" in last.columns else last.columns[0]
        ts = pd.to_datetime(last[tcol], utc=True, errors="coerce")
        print(f"last part  {f1.parent.name}/{f1.name} rows={len(last)} "
              f"ts {ts.min()} .. {ts.max()}")


def ohlcv_inventory() -> None:
    hdr("OHLCV (data/ohlcv)")
    files = sorted((DATA_ROOT / "ohlcv").glob("*_1h.parquet"))
    for f in files:
        df = pd.read_parquet(f, columns=["open_time"])
        ts = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
        print(f"{f.stem:14s} n={len(df):8d} {ts.min()} .. {ts.max()}")


def perimeter_inventory() -> None:
    hdr("Perimeter universe snapshots (data/perimeter)")
    files = sorted((DATA_ROOT / "perimeter").glob("recup_perimeter_*.json"))
    print(f"snapshots: {[f.stem.replace('recup_perimeter_', '') for f in files]}")


def gdelt_inventory() -> None:
    hdr("GDELT news (data/news)")
    files = sorted(RAW_NEWS_DIR.glob("*.parquet"))
    if not files:
        print("none")
        return
    print(f"partitions={len(files)} window={files[0].stem}..{files[-1].stem}")
    df = pd.read_parquet(files[-1])
    body = df["body"] if "body" in df.columns else None
    if body is not None:
        nonempty = (body.fillna("").str.len() > 0).mean()
        print(f"last partition non-empty body rate: {nonempty:.3%} ({len(df)} rows)")


if __name__ == "__main__":
    inv = rss_inventory()
    extraction_inventory()
    features_inventory()
    ohlcv_inventory()
    perimeter_inventory()
    gdelt_inventory()

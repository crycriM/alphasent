#!/usr/bin/env python3
"""Incremental OHLCV refresh — extend the 1h bars up to today.

write_ohlcv() appends + dedups on (open_time, symbol) and writes atomically, so
re-fetching an overlapping window is safe. Overlap of LOOKBACK_DAYS (default 7)
guards against partially-formed last bars and any earlier fetch aborted mid-range.

Run from repo root: .venv/bin/python scripts/refresh_ohlcv.py [--days-back N]
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import BINANCE_SYMBOLS, RAW_OHLCV_DIR
from src.ingest.binance_fetcher import backfill_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("refresh_ohlcv")


def last_bar(ohlcv_dir: Path, symbol: str, interval: str = "1h"):
    p = ohlcv_dir / f"{symbol}_{interval}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p, columns=["open_time"])["open_time"].max()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days-back", type=int, default=7,
                    help="overlap in days re-fetched before the last stored bar")
    args = ap.parse_args()

    present = [s for s in BINANCE_SYMBOLS
               if (RAW_OHLCV_DIR / f"{s}_1h.parquet").exists()]
    if not present:
        log.error("no OHLCV files in %s — run a full backfill instead", RAW_OHLCV_DIR)
        sys.exit(1)

    starts = []
    for s in present:
        lb = last_bar(RAW_OHLCV_DIR, s)
        log.info("%s last bar: %s", s, lb)
        starts.append(lb)
    start = (min(starts) - timedelta(days=args.days_back)).strftime("%Y-%m-%d")
    end = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
    log.info("refetching %s -> %s for %s", start, end, present)

    backfill_all(symbols=present, intervals=["1h"], start_date=start, end_date=end)

    for s in present:
        log.info("%s new last bar: %s", s, last_bar(RAW_OHLCV_DIR, s))

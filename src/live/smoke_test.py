from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    FEATURES_DIR,
    MODEL_VERSION,
    PROMPT_VERSION,
    RAW_NEWS_DIR,
)
from src.extraction.batch_etl import read_all_raw_news
from src.extraction.cache import cache_exists, compute_content_hash, load_all_cached
from src.features.store import read_features
from src.live.monitoring import (
    extraction_success_rate,
    event_type_drift,
    signal_autocorrelation,
    cache_miss_rate,
    run_all_checks,
)
from src.live.poller import LLM_AVAILABLE

log = logging.getLogger("live.smoke")

def run_smoke(max_items: int = 200) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    raw_df = read_all_raw_news()
    if raw_df.empty:
        log.error("No raw news found")
        return {"status": "FAIL", "reason": "no_raw_data"}

    raw_df = raw_df.head(max_items)
    log.info("Smoke test corpus: %d items", len(raw_df))

    attempted_hashes = []
    for _, row in raw_df.iterrows():
        ch = compute_content_hash(row["title"], row.get("body", ""))
        attempted_hashes.append(ch)

    if LLM_AVAILABLE:
        from src.extraction.batch_etl import run_batch_extraction
        events_df = run_batch_extraction(raw_df=raw_df)
    else:
        events_df = load_all_cached(MODEL_VERSION, PROMPT_VERSION)
        log.info("LLM unavailable — using cached data (%d records)", len(events_df))

    total_attempted = len(raw_df) if LLM_AVAILABLE else max(len(events_df), len(raw_df))
    success_rate = extraction_success_rate(events_df, total_attempted)
    log.info("Extraction success rate: %.2f", success_rate)

    if events_df.empty:
        return {"status": "FAIL", "reason": "no_extractions", "llm_available": LLM_AVAILABLE}

    asset_map = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "XRP": "XRPUSDT", "BNB": "BNBUSDT"}

    events_df["published_at"] = pd.to_datetime(events_df["published_at"], utc=True)
    drift = event_type_drift(events_df)

    miss_rate = cache_miss_rate(attempted_hashes)

    assets = [a for a in events_df["asset"].dropna().unique().tolist() if a is not None and a != "None" and a in asset_map]

    feature_ok = True
    signals_list = []
    for asset in assets:
        symbol = asset_map.get(asset, f"{asset}USDT")
        try:
            from src.ingest.binance_fetcher import fetch_all_klines, parse_klines
            end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            start_ms = end_ms - 7 * 24 * 3600 * 1000
            raw = fetch_all_klines(symbol, "1h", start_ms, end_ms)
            if raw:
                ohlcv_df = parse_klines(raw, symbol, "1h")
                from src.features.builder import build_features_for_asset
                features_df = build_features_for_asset(events_df, ohlcv_df, asset, symbol)
                if not features_df.empty:
                    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    from src.features.store import write_features
                    write_features(features_df, asset, today)

                    critical_cols = [c for c in features_df.columns if "polarity" in c or "n_events" in c]
                    if critical_cols:
                        nans = features_df[critical_cols].isna().sum().sum()
                        if nans > 0:
                            log.error("NaNs in critical feature columns for %s: %d", asset, nans)
                            feature_ok = False

                    for _, row in features_df.iterrows():
                        pol = row.get("evt_polarity_mean_h24", 0.0)
                        if not np.isnan(pol):
                            signals_list.append(pol)
        except Exception as e:
            log.error("Feature build failed for %s: %s", asset, e)

    signals = pd.Series(signals_list) if signals_list else pd.Series(dtype=float)
    acf = signal_autocorrelation(signals) if not signals.empty else pd.Series(dtype=float)
    stale_bars = int((acf[acf > 0.9]).sum()) if len(acf) > 0 else 0

    finite_ok = np.all(np.isfinite(signals)) if not signals.empty else True

    report = {
        "status": "PASS",
        "items_processed": len(raw_df),
        "items_extracted": len(events_df),
        "extraction_success_rate": float(success_rate),
        "event_type_drift": drift,
        "cache_miss_rate": float(miss_rate),
        "signal_autocorrelation": acf.to_dict() if len(acf) > 0 else {},
        "stale_bars_count": stale_bars,
        "feature_build_ok": feature_ok,
        "signals_finite": bool(finite_ok),
        "assets_tested": assets,
    }

    if LLM_AVAILABLE:
        assert success_rate > 0.85, f"Extraction success rate {success_rate} <= 0.85"
    assert feature_ok, "NaNs in critical feature columns"
    assert finite_ok, "Signals contain non-finite values"

    report["status"] = "PASS"
    return report

def main():
    parser = argparse.ArgumentParser(description="AlphaSent live smoke test")
    parser.add_argument("--max-items", type=int, default=200, help="Max raw items to process")
    args = parser.parse_args()

    report = run_smoke(max_items=args.max_items)
    print(json.dumps(report, indent=2, default=str))
    sys.exit(0 if report["status"] == "PASS" else 1)

if __name__ == "__main__":
    main()

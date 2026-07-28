#!/usr/bin/env python3
"""
Phase 0: Horizon scan — tests whether news → price exists and at which horizon.

Three sub-phases:
- 0a: Fetch hourly OHLCV for configured symbols
- 0b: IC(h) profile across horizons {1, 4, 24, 72, 120}h
- 0c: Event-study CAR with positive/negative/unsure buckets

Deliverable: IC(h) curve + CAR paths. Decision gate:
- Peak at short h (≤4h) → hourly bars are right
- Peak at long h (≥24h) → daily bars become primary
- Flat / insignificant → stop, do not proceed to Phases 2-3
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import BINANCE_SYMBOLS, RAW_OHLCV_DIR, FEATURES_DIR
from src.ingest.binance_fetcher import backfill_all
from src.features.builder import events_visible_at
from src.backtest.eval import _sharpe_bootstrap_ci

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("horizon_scan")

# --- Configuration ---
START_DATE = "2025-12-29"
END_DATE = "2026-07-18"
HORIZONS_H = [1, 4, 24, 72, 120]
FEATURE_DIMENSIONS = [
    "polarity_sum",
    "polarity_mean",
    "mag_weighted_polarity",
    "novelty_polarity",
    "recency_polarity",
]
EVENT_THRESHOLDS = {"c_star": 0.8, "m_star": 0.4, "p_star": 0.1}


def fetch_ohlcv() -> None:
    """Phase 0a: Fetch hourly OHLCV for all configured symbols."""
    log.info("Phase 0a: Fetching hourly OHLCV %s → %s", START_DATE, END_DATE)
    backfill_all(
        symbols=BINANCE_SYMBOLS,
        intervals=["1h"],
        start_date=START_DATE,
        end_date=END_DATE,
    )


def load_features() -> pd.DataFrame:
    """Load all feature files across assets and dates."""
    frames = []
    for asset_dir in sorted(FEATURES_DIR.iterdir()):
        if not asset_dir.is_dir():
            continue
        asset = asset_dir.name
        for parquet_file in sorted(asset_dir.glob("*.parquet")):
            df = pd.read_parquet(parquet_file)
            df["asset"] = asset
            frames.append(df)
    if not frames:
        raise ValueError("No feature files found")
    combined = pd.concat(frames, ignore_index=False)
    combined = combined.sort_index()
    return combined


def load_ohlcv() -> pd.DataFrame:
    """Load all OHLCV files across symbols."""
    frames = []
    for symbol in BINANCE_SYMBOLS:
        path = RAW_OHLCV_DIR / f"{symbol}_1h.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            frames.append(df)
    if not frames:
        raise ValueError("No OHLCV files found; run Phase 0a first")
    combined = pd.concat(frames, ignore_index=True)
    combined["open_time"] = pd.to_datetime(combined["open_time"], utc=True)
    combined = combined.sort_values("open_time").reset_index(drop=True)
    return combined


def compute_forward_returns(ohlcv: pd.DataFrame, horizon_h: int) -> pd.DataFrame:
    """Compute forward returns for a given horizon from hourly OHLCV."""
    pivot = ohlcv.pivot_table(index="open_time", columns="symbol", values="close")
    pivot = pivot.sort_index()
    fwd = pivot.shift(-horizon_h) / pivot - 1
    return fwd


def compute_ic_profile(features: pd.DataFrame, ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Phase 0b: Compute IC(h) for each feature and horizon."""
    log.info("Phase 0b: Computing IC profile across horizons")
    results = []

    for horizon in HORIZONS_H:
        fwd_ret = compute_forward_returns(ohlcv, horizon)

        for feat_col in FEATURE_DIMENSIONS:
            if feat_col not in features.columns:
                log.warning("Feature %s not found, skipping", feat_col)
                continue

            ic_values = []
            for symbol in BINANCE_SYMBOLS:
                asset = symbol.replace("USDT", "")
                asset_features = features[features["asset"] == asset].copy()
                asset_fwd = fwd_ret[[symbol]].dropna()

                if asset_features.empty or asset_fwd.empty:
                    continue

                merged = asset_features.join(asset_fwd, how="inner")
                if len(merged) < 10:
                    continue

                feat_vals = merged[feat_col].values
                ret_vals = merged[symbol].values

                if np.std(feat_vals) < 1e-12 or np.std(ret_vals) < 1e-12:
                    continue

                ic = np.corrcoef(feat_vals, ret_vals)[0, 1]
                ic_values.append(ic)

            if ic_values:
                mean_ic = np.mean(ic_values)
                ic_series = pd.Series(ic_values)
                ci_lo, ci_hi = _sharpe_bootstrap_ci(ic_series, n_boot=200, conf=0.95, periods_per_year=8760 // horizon)
                results.append({
                    "horizon_h": horizon,
                    "feature": feat_col,
                    "mean_ic": mean_ic,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "n_assets": len(ic_values),
                })

    return pd.DataFrame(results)


def compute_event_study_car(features: pd.DataFrame, ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Phase 0c: Event-study CAR with positive/negative/unsure buckets."""
    log.info("Phase 0c: Computing event-study CAR")
    c_star = EVENT_THRESHOLDS["c_star"]
    m_star = EVENT_THRESHOLDS["m_star"]
    p_star = EVENT_THRESHOLDS["p_star"]

    high_conf = features[
        (features["n_high_conf_events"] > 0)
        & (features["polarity_sum"].abs() >= m_star)
    ].copy()

    if high_conf.empty:
        log.warning("No high-confidence events found")
        return pd.DataFrame()

    positive_mask = (
        (high_conf["n_high_conf_events"] > 0)
        & (high_conf["polarity_sum"].abs() >= m_star)
        & (high_conf["polarity_mean"] > p_star)
    )
    negative_mask = (
        (high_conf["n_high_conf_events"] > 0)
        & (high_conf["polarity_sum"].abs() >= m_star)
        & (high_conf["polarity_mean"] < -p_star)
    )

    positive_events = high_conf[positive_mask]
    negative_events = high_conf[negative_mask]
    unsure_events = high_conf[~(positive_mask | negative_mask)]

    log.info(
        "Event buckets: positive=%d, negative=%d, unsure=%d",
        len(positive_events), len(negative_events), len(unsure_events),
    )

    car_paths = []
    for horizon in HORIZONS_H:
        fwd_ret = compute_forward_returns(ohlcv, horizon)

        for bucket_name, bucket_events in [
            ("positive", positive_events),
            ("negative", negative_events),
            ("unsure", unsure_events),
        ]:
            if bucket_events.empty:
                continue

            bucket_returns = []
            for idx, row in bucket_events.iterrows():
                asset = row["asset"]
                symbol = f"{asset}USDT"
                bar_time = idx

                if symbol not in fwd_ret.columns:
                    continue
                if bar_time not in fwd_ret.index:
                    continue

                ret = fwd_ret.loc[bar_time, symbol]
                if not pd.isna(ret):
                    bucket_returns.append(ret)

            if bucket_returns:
                car_paths.append({
                    "horizon_h": horizon,
                    "bucket": bucket_name,
                    "car": np.mean(bucket_returns),
                    "n_events": len(bucket_returns),
                    "std": np.std(bucket_returns) if len(bucket_returns) > 1 else 0.0,
                })

    return pd.DataFrame(car_paths)


def self_check() -> None:
    """Assert-based self-check on synthetic drift data."""
    log.info("Running self-check on synthetic data")
    np.random.seed(42)

    n = 1000
    feature = np.random.randn(n)
    noise = np.random.randn(n) * 0.05
    drift = np.concatenate([np.zeros(500), np.linspace(0, 0.1, 500)])
    returns = feature * 0.05 + noise + drift

    feat_series = pd.Series(feature)
    ret_series = pd.Series(returns)

    ic = np.corrcoef(feature, returns)[0, 1]
    assert abs(ic) > 0.2, f"Synthetic IC too low: {ic}"

    ci_lo, ci_hi = _sharpe_bootstrap_ci(ret_series, n_boot=100, conf=0.95, periods_per_year=8760)
    assert not np.isnan(ci_lo) and not np.isnan(ci_hi), "Bootstrap CI returned NaN"

    log.info("Self-check passed: synthetic IC=%.3f, CI=[%.3f, %.3f]", ic, ci_lo, ci_hi)


def main() -> None:
    """Run the full horizon scan."""
    self_check()

    fetch_ohlcv()

    features = load_features()
    ohlcv = load_ohlcv()

    log.info("Loaded %d feature rows, %d OHLCV rows", len(features), len(ohlcv))

    ic_profile = compute_ic_profile(features, ohlcv)
    log.info("IC profile:\n%s", ic_profile.to_string(index=False))

    car_paths = compute_event_study_car(features, ohlcv)
    log.info("CAR paths:\n%s", car_paths.to_string(index=False))

    ic_profile.to_csv("data/horizon_scan_ic.csv", index=False)
    car_paths.to_csv("data/horizon_scan_car.csv", index=False)

    log.info("Results saved to data/horizon_scan_ic.csv and data/horizon_scan_car.csv")

    if not ic_profile.empty:
        peak_horizon = ic_profile.loc[ic_profile["mean_ic"].abs().idxmax(), "horizon_h"]
        log.info("Peak IC at horizon %dh", peak_horizon)

        if peak_horizon <= 4:
            log.info("Decision: hourly bars are appropriate")
        elif peak_horizon >= 24:
            log.info("Decision: daily bars should be primary (Phase 2)")
        else:
            log.info("Decision: intermediate horizon, evaluate case-by-case")


if __name__ == "__main__":
    main()

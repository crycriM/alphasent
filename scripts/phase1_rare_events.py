#!/usr/bin/env python3
"""
Phase 1: Rare-event subset backtest.

Backtest using only the 12 event flag columns (4 event types × 3 lookbacks).
Score at 72h horizon (peak IC from Phase 0).
Compare rare-event-only vs full-feature vs buy-and-hold.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FEATURES_DIR, RAW_OHLCV_DIR
from src.backtest.signal import SignalPipeline
from src.backtest.eval import sharpe, hit_rate, max_drawdown

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("phase1_rare_events")

# --- Configuration ---
HORIZON_H = 72  # peak IC from Phase 0
EVENT_TYPES = ["hack", "regulation", "listing", "depeg"]
LOOKBACKS_H = [6, 24, 72]


def load_features_and_ohlcv() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load features and OHLCV separately for proper alignment."""
    # Load features
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
    features = pd.concat(frames, ignore_index=False).sort_index()

    # Load OHLCV
    ohlcv_frames = []
    for symbol in ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]:
        path = RAW_OHLCV_DIR / f"{symbol}_1h.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df["asset"] = symbol.replace("USDT", "")
            ohlcv_frames.append(df)
    if not ohlcv_frames:
        raise ValueError("No OHLCV files found")
    ohlcv = pd.concat(ohlcv_frames, ignore_index=True)
    ohlcv["open_time"] = pd.to_datetime(ohlcv["open_time"], utc=True)

    return features, ohlcv


def get_rare_event_flag_columns(features: pd.DataFrame) -> list[str]:
    """Extract the rare-event flag columns."""
    flag_cols = []
    for event_type in EVENT_TYPES:
        col = f"{event_type}_flag"
        if col in features.columns:
            flag_cols.append(col)
    return flag_cols


def run_backtest_variant(
    data: pd.DataFrame,
    feature_cols: list[str],
    variant_name: str,
) -> dict:
    """Run backtest with given feature columns."""
    results = []
    for asset in data["asset"].unique():
        sub = data[data["asset"] == asset].copy()
        if len(sub) < 50:
            continue

        X = sub[feature_cols].fillna(0)
        y = sub["fwd_return"]

        if X.std().sum() < 1e-12:
            log.warning("No variance in features for %s, skipping", asset)
            continue

        sp = SignalPipeline("ridge")
        sp.fit(X, y)
        pred = sp.predict(X)
        
        # Sample non-overlapping: take position every HORIZON_H bars, hold for HORIZON_H bars
        n_samples = len(sub) // HORIZON_H
        if n_samples < 10:
            log.warning("Insufficient non-overlapping samples for %s", asset)
            continue
        
        sampled_indices = list(range(0, len(sub), HORIZON_H))[:n_samples]
        sampled_pred = pred[sampled_indices]
        sampled_returns = y.iloc[sampled_indices]
        
        pos = SignalPipeline.position_from_signal(sampled_pred)
        pos.index = sampled_returns.index
        trade_returns = pos * sampled_returns
        
        results.append({
            "variant": variant_name,
            "asset": asset,
            "sharpe": sharpe(trade_returns, periods_per_year=8760 // HORIZON_H),
            "hit_rate": hit_rate(trade_returns),
            "total_return": float((1 + trade_returns).prod() - 1),
            "max_dd": max_drawdown((1 + trade_returns).cumprod()),
            "n_trades": len(trade_returns),
        })

    return results


def main() -> None:
    """Run Phase 1 backtest."""
    log.info("Loading features and OHLCV data")
    features, ohlcv = load_features_and_ohlcv()
    log.info("Loaded %d feature rows, %d OHLCV rows", len(features), len(ohlcv))

    # Compute 72h forward returns for strategy backtest
    pivot = ohlcv.pivot_table(index="open_time", columns="asset", values="close")
    pivot = pivot.sort_index()
    fwd_ret = pivot.shift(-HORIZON_H) / pivot - 1

    # Merge features with forward returns
    merged_frames = []
    for asset in features["asset"].unique():
        asset_feat = features[features["asset"] == asset]
        if asset not in fwd_ret.columns:
            continue
        asset_fwd = fwd_ret[[asset]].rename(columns={asset: "fwd_return"})
        merged = asset_feat.join(asset_fwd, how="inner")
        merged_frames.append(merged)
    if not merged_frames:
        raise ValueError("No overlapping feature/OHLCV data")
    data = pd.concat(merged_frames).dropna(subset=["fwd_return"])
    log.info("Merged data: %d rows", len(data))

    rare_event_cols = get_rare_event_flag_columns(features)
    log.info("Rare-event flag columns: %s", rare_event_cols)

    all_feature_cols = [c for c in features.columns if c not in ("asset", "lookback_h", "bar_open_ts")]
    log.info("All feature columns: %d", len(all_feature_cols))

    results = []

    log.info("Running rare-event-only backtest")
    results.extend(run_backtest_variant(data, rare_event_cols, "rare_event_only"))

    log.info("Running full-feature backtest")
    results.extend(run_backtest_variant(data, all_feature_cols, "full_features"))

    log.info("Computing buy-and-hold baseline")
    # For buy-and-hold, use sequential 1h returns
    seq_ret = pivot.pct_change()
    for asset in data["asset"].unique():
        asset_data = data[data["asset"] == asset]
        if len(asset_data) < 50:
            continue
        if asset not in seq_ret.columns:
            continue
        asset_seq = seq_ret[[asset]].rename(columns={asset: "seq_return"})
        merged = asset_data.join(asset_seq, how="inner").dropna(subset=["seq_return"])
        if len(merged) < 50:
            continue
        pos = pd.Series(1.0, index=merged.index)
        returns = pos * merged["seq_return"]
        results.append({
            "variant": "buy_and_hold",
            "asset": asset,
            "sharpe": sharpe(returns, periods_per_year=8760),
            "hit_rate": hit_rate(returns),
            "total_return": float((1 + returns).prod() - 1),
            "max_dd": max_drawdown((1 + returns).cumprod()),
            "n_bars": len(merged),
        })

    df_results = pd.DataFrame(results)
    log.info("\n%s", df_results.to_string(index=False))
    df_results.to_csv("data/phase1_rare_events.csv", index=False)
    log.info("Results saved to data/phase1_rare_events.csv")

    summary = df_results.groupby("variant").agg({
        "sharpe": "mean",
        "hit_rate": "mean",
        "total_return": "mean",
        "max_dd": "mean",
    }).reset_index()
    log.info("\nSummary by variant:\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()

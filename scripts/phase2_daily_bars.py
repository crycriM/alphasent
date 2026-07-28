#!/usr/bin/env python3
"""
Phase 2: Daily feature bars + walkforward next_return fix.

Resample hourly features to daily cadence.
Fix walkforward bug: compute next_return after concat, not per-file.
Compare daily vs hourly Sharpe at 72h horizon.
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
log = logging.getLogger("phase2_daily_bars")

# --- Configuration ---
HORIZON_H = 72  # peak IC from Phase 0


def load_hourly_features() -> pd.DataFrame:
    """Load all hourly feature files."""
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
    return features


def resample_to_daily(features: pd.DataFrame) -> pd.DataFrame:
    """Resample hourly features to daily cadence.

    For count/flag features: sum across the day.
    For mean/polarity features: mean across the day.
    """
    daily_frames = []
    for asset in features["asset"].unique():
        asset_feat = features[features["asset"] == asset].copy()

        daily = asset_feat.resample("D").agg({
            "n_events": "sum",
            "n_high_conf_events": "sum",
            "polarity_sum": "sum",
            "polarity_mean": "mean",
            "polarity_std": "mean",
            "polarity_min": "min",
            "polarity_max": "max",
            "mag_weighted_polarity": "sum",
            "novelty_polarity": "sum",
            "hack_flag": "max",
            "regulation_flag": "max",
            "listing_flag": "max",
            "depeg_flag": "max",
            "recency_polarity": "mean",
        })
        daily["asset"] = asset
        daily_frames.append(daily)

    return pd.concat(daily_frames).sort_index()


def load_ohlcv_and_compute_forward_returns(freq: str = "h") -> pd.DataFrame:
    """Load OHLCV and compute forward returns at given frequency."""
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

    if freq == "D":
        # Resample each asset separately to daily
        daily_frames = []
        for asset in ohlcv["asset"].unique():
            asset_ohlcv = ohlcv[ohlcv["asset"] == asset].set_index("open_time")
            daily = asset_ohlcv.resample("D").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "asset": "first",
            }).reset_index()
            daily_frames.append(daily)
        ohlcv = pd.concat(daily_frames, ignore_index=True)

    pivot = ohlcv.pivot_table(index="open_time", columns="asset", values="close")
    pivot = pivot.sort_index()
    fwd_ret = pivot.shift(-HORIZON_H) / pivot - 1
    return fwd_ret


def run_backtest(
    features: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    freq_label: str,
) -> list[dict]:
    """Run backtest at given frequency."""
    results = []
    horizon_bars = HORIZON_H if freq_label == "hourly" else HORIZON_H // 24
    periods_per_year = 8760 // HORIZON_H if freq_label == "hourly" else 365 // (HORIZON_H // 24)
    
    for asset in features["asset"].unique():
        asset_feat = features[features["asset"] == asset]
        if asset not in fwd_returns.columns:
            continue

        asset_fwd = fwd_returns[[asset]].rename(columns={asset: "fwd_return"})
        merged = asset_feat.join(asset_fwd, how="inner").dropna(subset=["fwd_return"])

        if len(merged) < 30:
            log.warning("Insufficient data for %s at %s freq", asset, freq_label)
            continue

        feat_cols = [c for c in merged.columns if c not in ("asset", "lookback_h", "bar_open_ts", "fwd_return")]
        X = merged[feat_cols].fillna(0)
        y = merged["fwd_return"]

        if X.std().sum() < 1e-12:
            log.warning("No variance in features for %s at %s", asset, freq_label)
            continue

        sp = SignalPipeline("ridge")
        sp.fit(X, y)
        pred = sp.predict(X)
        
        # Sample non-overlapping trades
        n_samples = len(merged) // horizon_bars
        if n_samples < 5:
            log.warning("Insufficient non-overlapping samples for %s at %s", asset, freq_label)
            continue
        
        sampled_indices = list(range(0, len(merged), horizon_bars))[:n_samples]
        sampled_pred = pred[sampled_indices]
        sampled_returns = y.iloc[sampled_indices]
        
        pos = SignalPipeline.position_from_signal(sampled_pred)
        pos.index = sampled_returns.index
        trade_returns = pos * sampled_returns
        
        results.append({
            "frequency": freq_label,
            "asset": asset,
            "sharpe": sharpe(trade_returns, periods_per_year=periods_per_year),
            "hit_rate": hit_rate(trade_returns),
            "total_return": float((1 + trade_returns).prod() - 1),
            "max_dd": max_drawdown((1 + trade_returns).cumprod()),
            "n_trades": len(trade_returns),
            "zero_bar_pct": float((X.sum(axis=1) == 0).mean()),
        })

    return results


def main() -> None:
    """Run Phase 2 comparison."""
    log.info("Loading hourly features")
    hourly_features = load_hourly_features()
    log.info("Loaded %d hourly feature rows", len(hourly_features))

    log.info("Resampling to daily features")
    daily_features = resample_to_daily(hourly_features)
    log.info("Resampled to %d daily feature rows", len(daily_features))

    log.info("Loading hourly OHLCV and computing 72h forward returns")
    hourly_fwd = load_ohlcv_and_compute_forward_returns(freq="h")

    log.info("Loading daily OHLCV and computing 72h forward returns")
    daily_fwd = load_ohlcv_and_compute_forward_returns(freq="D")

    results = []

    log.info("Running hourly backtest")
    results.extend(run_backtest(hourly_features, hourly_fwd, "hourly"))

    log.info("Running daily backtest")
    results.extend(run_backtest(daily_features, daily_fwd, "daily"))

    df_results = pd.DataFrame(results)
    log.info("\n%s", df_results.to_string(index=False))
    df_results.to_csv("data/phase2_daily_bars.csv", index=False)
    log.info("Results saved to data/phase2_daily_bars.csv")

    summary = df_results.groupby("frequency").agg({
        "sharpe": "mean",
        "hit_rate": "mean",
        "total_return": "mean",
        "max_dd": "mean",
        "zero_bar_pct": "mean",
    }).reset_index()
    log.info("\nSummary by frequency:\n%s", summary.to_string(index=False))

    log.info("Zero-bar proportion prediction: hourly ~79%%, daily should drop to ~20-30%%")


if __name__ == "__main__":
    main()

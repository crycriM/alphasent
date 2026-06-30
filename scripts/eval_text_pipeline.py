"""
Text-only pipeline evaluation: load features, analyze signal properties.
No OHLCV, no Binance.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("text_eval")

FEATURES_DIR = Path("data/features")


def load_all_features() -> pd.DataFrame:
    """Load all feature parquet files from all assets."""
    dfs = []
    for asset_dir in sorted(FEATURES_DIR.iterdir()):
        if not asset_dir.is_dir():
            continue
        asset = asset_dir.name
        for pf in sorted(asset_dir.glob("*.parquet")):
            df = pd.read_parquet(pf)
            df["asset"] = asset
            dfs.append(df)

    if not dfs:
        log.warning("No feature files found.")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=False)
    combined.index.name = "bar_open_ts"
    log.info("Loaded %d feature rows from %d assets", len(combined),
             sum(1 for _ in FEATURES_DIR.iterdir()))
    return combined


def analyze_signals(features: pd.DataFrame) -> None:
    """Analyze the news sentiment features for signal properties."""
    feat_cols = [c for c in features.columns if c not in ("asset",)]

    print(f"\n{'='*70}")
    print("AlphaSent — Text Pipeline Signal Report")
    print(f"{'='*70}")

    print(f"\nDataset: {len(features)} hourly bars, {len(feat_cols)} features")
    print(f"Date range: {features.index.min()} to {features.index.max()}")
    print(f"Assets with features: {[a for a in sorted(features['asset'].unique()) if a != 'None']}")

    # --- Polarity signal diversity ---
    print(f"\n--- Polarity Summary (24h lookback) ---")
    for col in ["polarity_sum", "polarity_mean", "mag_weighted_polarity", "recency_polarity"]:
        if col in features.columns:
            ps = features[col]
            neg_pct = (ps < 0).mean() * 100
            pos_pct = (ps > 0).mean() * 100
            print(f"  {col:>25}: mean={ps.mean():+.4f}  std={ps.std():.4f}  "
                  f"neg={neg_pct:.0f}%  pos={pos_pct:.0f}%  "
                  f"[{ps.min():+.4f}, {ps.max():+.4f}]")

    # --- Event volume ---
    if "n_events" in features.columns:
        ne = features["n_events"]
        print(f"\n--- Event Volume ---")
        print(f"  Mean events/bar: {ne.mean():.1f}")
        print(f"  Median:          {ne.median():.0f}")
        print(f"  Max:             {ne.max():.0f}")
        print(f"  Total events:    {ne.sum():.0f}")

    # --- Event-type flags ---
    print(f"\n--- Event Type Flags (triggered bars) ---")
    for col in ["hack_flag", "regulation_flag", "listing_flag", "depeg_flag"]:
        if col in features.columns:
            triggered = features[col].sum()
            pct = features[col].mean() * 100
            print(f"  {col:>20}: {triggered:4.0f} bars ({pct:.1f}%)")

    # --- Asset-level breakdown ---
    print(f"\n--- Per-Asset Signal Profile (BTC reference) ---")
    for asset in sorted(features["asset"].unique()):
        if asset == "None":
            continue
        sub = features[features["asset"] == asset]
        print(f"\n  {asset}:")
        print(f"    Bars: {len(sub)}")
        if "polarity_sum" in sub.columns:
            ps = sub["polarity_sum"]
            print(f"    Polarity sum: mean={ps.mean():+.4f}   "
                  f"neg={(ps<0).mean()*100:.0f}%/pos={(ps>0).mean()*100:.0f}%")
        if "n_events" in sub.columns:
            ne = sub["n_events"]
            print(f"    Events: {ne.sum():.0f} total, {ne.mean():.1f}/bar")
        if "hack_flag" in sub.columns:
            for flag in ["hack_flag", "regulation_flag", "listing_flag"]:
                if flag in sub.columns:
                    cnt = sub[flag].sum()
                    if cnt > 0:
                        print(f"    {flag}: {int(cnt)} bars")

    # --- Time-series structure ---
    print(f"\n--- Feature Time-Series Structure ---")
    # Autocorrelation of polarity_sum at lag 1, 24
    if "polarity_sum" in features.columns:
        ps = features["polarity_sum"]
        ac_lag1 = ps.autocorr(lag=1)
        ac_lag24 = ps.autocorr(lag=24)
        print(f"  Polarity sum autocorr (lag=1):  {ac_lag1:.3f}")
        print(f"  Polarity sum autocorr (lag=24): {ac_lag24:.3f}")

    # Feature-feature correlation
    numeric_cols = [c for c in feat_cols
                    if features[c].dtype in ("float64", "int64") and c != "lookback_h"]
    if len(numeric_cols) > 1:
        corr = features[numeric_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        high_pairs = [(col, row, upper.loc[col, row])
                      for col in upper.columns for row in upper.index
                      if upper.loc[col, row] > 0.8]
        high_pairs.sort(key=lambda x: -x[2])
        if high_pairs:
            print(f"\n  High feature correlations (>0.8):")
            for col, row, val in high_pairs[:10]:
                print(f"    {col} <-> {row}: {val:.3f}")
        else:
            print(f"  No feature pairs with >0.8 correlation (good)")

    # --- Key observation ---
    print(f"\n{'='*70}")
    print("KEY FINDINGS")
    print(f"{'='*70}")
    print()
    if "polarity_sum" in features.columns:
        mean_ps = features["polarity_sum"].mean()
        std_ps = features["polarity_sum"].std()
        info_ratio_est = mean_ps / std_ps if std_ps > 0 else 0
        print(f"  Polarity signal: mean={mean_ps:+.4f}, std={std_ps:.4f}, "
              f"info-ratio-est={info_ratio_est:.4f}")
        print(f"  → {'Positive bias: features lean bullish on average' if mean_ps > 0 else 'Negative bias: features lean bearish on average'}")
        print(f"  → {'Good variance: signal has room to differentiate bars' if std_ps > 0.5 else 'Limited variance: features may be stale across bars'}")

    n_feat = len(feat_cols)
    n_obs = len(features)
    print(f"  Feature/observation ratio: {n_feat} features / {n_obs} bars = {n_obs/max(n_feat,1):.0f}x")
    print(f"  → {'Plenty of observations per feature (low overfit risk)' if n_obs > n_feat * 50 else 'Limited obs/feature — regularization recommended'}")

    print()


if __name__ == "__main__":
    features = load_all_features()
    analyze_signals(features)

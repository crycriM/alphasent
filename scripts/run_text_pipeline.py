"""
Run the text-only pipeline: load cached extractions -> build features -> evaluate.
No OHLCV or Binance required.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import numpy as np

from src.extraction.cache import load_all_cached
from src.config import MODEL_VERSION, PROMPT_VERSION
from src.features.builder import build_feature_vector
from src.features.store import write_features, read_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("text_pipeline")

ASSETS_MIN_EVENTS = 5
FEATURES_DIR = Path("data/features")


def build_all_features() -> pd.DataFrame:
    """Build hourly features for all assets with enough events."""
    events = load_all_cached(MODEL_VERSION, PROMPT_VERSION)
    log.info("Loaded %d cached event records", len(events))

    events = events[events["asset"].notna() & (events["asset"] != "")].copy()
    events["published_at"] = pd.to_datetime(events["published_at"], utc=True)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    for asset in sorted(events["asset"].unique()):
        ae = events[events["asset"] == asset].sort_values("published_at")
        if len(ae) < ASSETS_MIN_EVENTS:
            log.info("Skipping %s: only %d events", asset, len(ae))
            continue

        start = ae["published_at"].min().normalize()
        end = ae["published_at"].max().normalize() + pd.Timedelta(days=1)

        bar_opens = pd.date_range(start, end, freq="h", tz="UTC")

        rows: list[dict] = []
        for bar_open in bar_opens:
            feat = build_feature_vector(ae, 24, bar_open.to_pydatetime(), asset=asset)
            if feat:
                feat["bar_open_ts"] = bar_open
                feat["asset"] = asset
                rows.append(feat)

        if rows:
            df = pd.DataFrame(rows).set_index("bar_open_ts")
            for dt, group in df.groupby(df.index.date):
                date_str = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)
                write_features(group, asset, date_str, FEATURES_DIR)
            all_rows.extend(rows)
            log.info("%s: %d events -> %d hourly feature rows", asset, len(ae), len(rows))

    if not all_rows:
        log.warning("No features built.")
        return pd.DataFrame()

    combined = pd.concat([pd.DataFrame(r, index=[0]) for r in all_rows], ignore_index=True)
    log.info("Total feature rows: %d", len(combined))

    feat_cols = [c for c in combined.columns if c not in ("bar_open_ts", "asset")]
    log.info("Feature summary:\n%s", combined[feat_cols].describe().to_string())
    log.info("Assets: %s", sorted(combined["asset"].unique()))

    return combined


def evaluate_signal(combined: pd.DataFrame | None = None) -> None:
    """Evaluate signal properties from features alone."""
    if combined is None or combined.empty:
        log.warning("No data to evaluate.")
        return

    feat_cols = [c for c in combined.columns if c.startswith("evt_") or c.startswith("mag_")]

    print(f"\n{'='*60}")
    print(f"Signal Quality Report (features only, no OHLCV)")
    print(f"{'='*60}")

    print(f"\nFeature dimension: {len(feat_cols)} columns across {len(combined)} rows")

    # Polarity signal diversity
    if "evt_polarity_sum_h24" in combined.columns:
        ps = combined["evt_polarity_sum_h24"]
        print(f"\nPolarity sum h24:")
        print(f"  Mean:    {ps.mean():+.4f}")
        print(f"  Std:     {ps.std():.4f}")
        print(f"  Min:     {ps.min():+.4f}")
        print(f"  Max:     {ps.max():+.4f}")
        print(f"  % > 0:   {(ps > 0).mean()*100:.1f}%")
        print(f"  % < 0:   {(ps < 0).mean()*100:.1f}%")

    # Event volume per bar
    if "evt_n_events_h24" in combined.columns:
        ne = combined["evt_n_events_h24"]
        print(f"\nEvent volume h24:")
        print(f"  Mean:    {ne.mean():.1f}")
        print(f"  Median:  {ne.median():.0f}")
        print(f"  Max:     {ne.max():.0f}")
        print(f"  Zero-bars: {(ne == 0).mean()*100:.1f}%")

    # Hack flag rate
    for flag_col in [c for c in feat_cols if "flag" in c]:
        f = combined[flag_col]
        print(f"\n{flag_col}:")
        print(f"  Triggered: {f.sum():.0f} bars ({(f.mean()*100):.1f}%)")

    # Correlation structure
    if len(feat_cols) > 1:
        corr = combined[feat_cols].corr().abs()
        upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        high_corr = [(col, row, upper_tri.loc[col, row])
                     for col in upper_tri.columns for row in upper_tri.index
                     if upper_tri.loc[col, row] > 0.7]
        if high_corr:
            print(f"\nHigh feature correlations (>0.7):")
            for col, row, val in sorted(high_corr, key=lambda x: -x[2]):
                print(f"  {col} <-> {row}: {val:.3f}")

    # Asset-level breakdown
    print(f"\nAsset-level event volume:")
    for asset in sorted(combined["asset"].unique()):
        sub = combined[combined["asset"] == asset]
        ne = sub["evt_n_events_h24"] if "evt_n_events_h24" in sub.columns else pd.Series([0])
        print(f"  {asset:>10}: {len(sub):>5} bars, {ne.mean():.1f} events/bar, "
              f"{ne.sum():.0f} total events")

    print(f"\n{'='*60}")
    print("Done. No OHLCV needed — features describe news sentiment distribution.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    features = build_all_features()
    evaluate_signal(features)

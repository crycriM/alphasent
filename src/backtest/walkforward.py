from __future__ import annotations

import pandas as pd
from pathlib import Path

from src.config import FEATURES_DIR

def generate_windows(
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_months: int = 12,
    test_months: int = 3,
    step_months: int = 3,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Rolling walk-forward windows. Returns (train_start, train_end, test_start, test_end)."""
    windows = []
    train_start = start
    test_start = start + pd.DateOffset(months=train_months)
    while test_start < end:
        test_end = min(test_start + pd.DateOffset(months=test_months), end)
        windows.append((train_start, test_start - pd.Timedelta(days=1), test_start, test_end))
        train_start += pd.DateOffset(months=step_months)
        test_start += pd.DateOffset(months=step_months)
    return windows

def load_window_data(
    assets: list[str],
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    features_dir: Path = FEATURES_DIR,
) -> pd.DataFrame | None:
    """Load features + next-bar return target for a test window.

    Loads day-partitioned feature files (each file contains hourly bars).
    Computes `next_return` as the next-bar return from the OHLCV close column.
    """
    import logging
    from pathlib import Path

    log = logging.getLogger("backtest.walkforward")

    frames = []
    for asset in assets:
        # Load all day-partitioned files for this asset in the date range
        asset_dir = features_dir / asset
        if not asset_dir.exists():
            continue
        for date_str in pd.date_range(test_start, test_end, freq="D").strftime("%Y-%m-%d"):
            part_path = asset_dir / f"{date_str}.parquet"
            if not part_path.exists():
                continue
            try:
                df = pd.read_parquet(part_path)
            except Exception as e:
                log.warning("Failed to read %s: %s", part_path, e)
                continue
            df = df.copy()
            df["asset"] = asset
            frames.append(df)

    if not frames:
        return None

    combined = pd.concat(frames)
    combined = combined.sort_index()
    
    # Fix: compute next_return AFTER concat+sort, grouped by asset
    # This recovers cross-midnight returns that the per-file computation loses
    if "close" in combined.columns:
        combined["next_return"] = combined.groupby("asset")["close"].pct_change().shift(-1)
    
    return combined.dropna(subset=["next_return"])

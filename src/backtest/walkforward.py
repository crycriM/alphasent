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
    """Load features + next-bar return target for a test window. Skip missing data gracefully."""
    from src.features.store import read_features

    frames = []
    for asset in assets:
        for dt in pd.date_range(test_start, test_end, freq="D"):
            date_str = dt.strftime("%Y-%m-%d")
            df = read_features(asset, date_str, features_dir)
            if df.empty:
                continue
            df = df.copy()
            df["asset"] = asset
            if "close" in df.columns:
                df["next_return"] = df["close"].pct_change().shift(-1)
            frames.append(df)

    if not frames:
        return None

    combined = pd.concat(frames)
    combined = combined.sort_index()
    return combined.dropna(subset=["next_return"])

"""
Feature store: parquet read/write with validation.

One file per (asset, date) under features/{SYMBOL}/YYYY-MM-DD.parquet.
One row per bar (hourly). Never modified after write.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import FEATURES_DIR

log = logging.getLogger("features.store")


def write_features(
    features_df: pd.DataFrame,
    asset: str,
    date: str,
    features_dir: Path = FEATURES_DIR,
) -> None:
    """Write feature DataFrame to a day-partitioned parquet file.

    Args:
        features_df: DataFrame with bar_open_ts index and feature columns.
        asset: Ticker (e.g. "BTC").
        date: Date string (YYYY-MM-DD).
        features_dir: Root directory for feature files.
    """
    asset_dir = features_dir / asset
    asset_dir.mkdir(parents=True, exist_ok=True)

    part_path = asset_dir / f"{date}.parquet"

    # If file exists, append new rows
    if part_path.exists():
        existing = pd.read_parquet(part_path)
        combined = pd.concat([existing, features_df], ignore_index=False)
        # Deduplicate on index (bar_open_ts)
        combined = combined[~combined.index.duplicated(keep="last")]
    else:
        combined = features_df

    combined = combined.sort_index()
    table = pa.Table.from_pandas(combined, preserve_index=True)
    pq.write_table(table, part_path)
    log.info("Wrote %d feature rows to %s", len(combined), part_path)


def read_features(
    asset: str,
    date: str,
    features_dir: Path = FEATURES_DIR,
) -> pd.DataFrame:
    """Read feature parquet for a given asset and date."""
    part_path = features_dir / asset / f"{date}.parquet"
    if not part_path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(part_path)
    if "bar_open_ts" in df.columns:
        df = df.set_index("bar_open_ts")
    return df


def read_features_for_date(
    assets: list[str],
    date: str,
    features_dir: Path = FEATURES_DIR,
) -> pd.DataFrame:
    """Read features for multiple assets on a given date.

    Returns a DataFrame with 'asset' column and merged features.
    """
    dfs = []
    for asset in assets:
        df = read_features(asset, date, features_dir)
        if not df.empty:
            df["asset"] = asset
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=False)

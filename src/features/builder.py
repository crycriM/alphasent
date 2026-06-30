"""
Feature builder: EventRecord -> feature vector (point-in-time safe).

The fundamental rule: a feature at bar t may only use events with
published_at < bar_open_time(t). No look-ahead, no same-bar events.

This is enforced at feature construction time, not at read time.
A corrupt join at read time would silently invalidate the backtest.
"""

from __future__ import annotations

import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from src.config import FEATURE_LOOKBACK_HOURS

log = logging.getLogger("features.builder")


def events_visible_at(
    events: pd.DataFrame,
    asset: str,
    bar_open_ts: pd.Timestamp,
    lookback: pd.Timedelta,
) -> pd.DataFrame:
    """The point-in-time guard. STRICT < bar_open_ts. Never <=. Never bar_close.

    An event landing exactly at bar open belongs to the NEXT bar,
    because a strategy acting on bar t cannot have seen information
    that arrives at the same instant the bar opens.
    """
    mask = (
        (events["asset"] == asset)
        & (events["published_at"] >= bar_open_ts - lookback)
        & (events["published_at"] < bar_open_ts)  # STRICT — the whole game
    )
    return events.loc[mask]


def compute_time_decayed_polarity(
    events: pd.DataFrame,
    lookback_hours: int,
    bar_open_ts: pd.Timestamp | None = None,
) -> float:
    """Compute time-decayed polarity: recent events weighted more heavily.

    Decay is anchored to `bar_open_ts` (the bar being evaluated) so weights are
    comparable across bars. age = bar_open_ts - published_at; weight decays
    exponentially with age. If bar_open_ts is None (e.g. unit tests), falls
    back to the newest event in the window.
    """
    if events.empty or "published_at" not in events.columns:
        return 0.0

    ref_time = bar_open_ts if bar_open_ts is not None else events["published_at"].max()
    ages_hours = (ref_time - events["published_at"]).dt.total_seconds() / 3600.0
    ages_hours = ages_hours.clip(lower=0.0)

    # Recent (small age) -> weight near 1; old (age~lookback) -> weight near e^-2.
    weights = np.exp(-2.0 * ages_hours / max(lookback_hours, 1))
    weights = np.clip(weights, 0, 1)

    return float((events["polarity"] * weights).sum())


def build_feature_vector(
    events: pd.DataFrame,
    lookback_hours: int,
    bar_open_ts: pd.Timestamp | None = None,
    asset: str | None = None,
) -> dict:
    """Aggregate all EventRecords in the lookback window into a feature dict.

    Applies point-in-time filtering if `bar_open_ts` is provided: only events
    with `published_at < bar_open_ts` within the lookback window are included.
    If `asset` is provided, filters to events for that asset.

    Returns a dict ready to be merged into the feature DataFrame.
    """
    # Apply point-in-time filter
    if bar_open_ts is not None and asset:
        visible = events_visible_at(
            events, asset, bar_open_ts, pd.Timedelta(hours=lookback_hours)
        )
    elif bar_open_ts is not None:
        lookback_dt = bar_open_ts - pd.Timedelta(hours=lookback_hours)
        mask = (events["published_at"] >= lookback_dt) & (events["published_at"] < bar_open_ts)
        visible = events.loc[mask].copy()
    else:
        visible = events

    if visible.empty:
        return _zero_feature_vector(lookback_hours)

    return {
        # Volume features
        "n_events": len(visible),
        "n_high_conf_events": int((visible["confidence"] > 0.8).sum()),

        # Polarity aggregates
        "polarity_sum": float(visible["polarity"].sum()),
        "polarity_mean": float(visible["polarity"].mean()),
        "polarity_std": float(visible["polarity"].std(ddof=0) if len(visible) > 1 else 0.0),
        "polarity_min": float(visible["polarity"].min()),
        "polarity_max": float(visible["polarity"].max()),

        # Magnitude-weighted polarity
        "mag_weighted_polarity": float(
            (visible["polarity"] * visible["magnitude"]).sum()
        ),

        # Novelty-weighted polarity (downweights repeated stories)
        "novelty_polarity": float(
            (visible["polarity"] * visible["novelty"] * visible["magnitude"]).sum()
        ),

        # Event-type flags
        "hack_flag": int((visible["event_type"] == "hack").any()),
        "regulation_flag": int((visible["event_type"] == "regulation").any()),
        "listing_flag": int((visible["event_type"] == "listing").any()),
        "depeg_flag": int((visible["event_type"] == "depeg").any()),

        # Time decay: weight recent events more (anchored to bar open)
        "recency_polarity": compute_time_decayed_polarity(
            visible, lookback_hours, bar_open_ts
        ),

        # Metadata
        "lookback_h": lookback_hours,
    }


def _zero_feature_vector(lookback_hours: int = 24) -> dict:
    """Return a zero-filled feature dict for when no events are visible."""
    return {
        "n_events": 0,
        "n_high_conf_events": 0,
        "polarity_sum": 0.0,
        "polarity_mean": 0.0,
        "polarity_std": 0.0,
        "polarity_min": 0.0,
        "polarity_max": 0.0,
        "mag_weighted_polarity": 0.0,
        "novelty_polarity": 0.0,
        "hack_flag": 0,
        "regulation_flag": 0,
        "listing_flag": 0,
        "depeg_flag": 0,
        "recency_polarity": 0.0,
        "lookback_h": lookback_hours,
    }


def build_features_for_asset(
    events_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    asset: str,
    symbol: str,
) -> pd.DataFrame:
    """Build feature vectors for all bars of a given asset. Requires OHLCV for bar schedule."""
    asset_events = events_df[events_df["asset"] == asset].copy()
    if "published_at" not in asset_events.columns:
        raise ValueError(
            f"EventRecords for {asset} missing 'published_at'; cannot build "
            f"point-in-time-safe features"
        )

    if asset_events["published_at"].dtype != "datetime64[ns, UTC]":
        asset_events["published_at"] = pd.to_datetime(
            asset_events["published_at"], utc=True
        )

    bar_frames = []
    for _, bar_row in ohlcv_df.iterrows():
        bar_open = pd.Timestamp(bar_row["open_time"])
        features = {}
        for lookback in FEATURE_LOOKBACK_HOURS:
            window_events = events_visible_at(
                asset_events, asset, bar_open, timedelta(hours=lookback)
            )
            fv = build_feature_vector(window_events, lookback, bar_open_ts=bar_open)
            for k, v in fv.items():
                features[f"evt_{k}_h{lookback}"] = v

        bar_features = {"bar_open_ts": bar_open, "symbol": symbol, **features}
        bar_frames.append(bar_features)

    result = pd.DataFrame(bar_frames)
    result = result.set_index("bar_open_ts")
    return result

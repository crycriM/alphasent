"""
Signal server: bar-close signal generation.

Reads today's feature store, applies the trained signal pipeline,
and emits position targets per asset.

Designed to run on bar close (hourly). In production this would be
triggered by a cron or the poller loop; here it's a standalone
run_once() or a long-running loop.
"""

from __future__ import annotations

import json
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    BINANCE_SYMBOLS,
    FEATURES_DIR,
    LLM_BASE_URL,
)
from src.features.store import read_features
from src.backtest.signal import SignalPipeline
from src.ingest.binance_fetcher import fetch_all_klines, parse_klines

log = logging.getLogger("live.signal")

# Signal model config
SIGNAL_MODEL_TYPE = "ridge"
SIGNAL_TARGET_VOL = 0.10
SIGNAL_CLIP_Z = 2.0

# Retraining cadence (days)
RETRAIN_DAYS = 30

# Signal output
SIGNAL_FILE = FEATURES_DIR.parent / "signals" / "positions.json"

# Asset map
ASSET_MAP = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "BNB": "BNBUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
}

SIGNAL_BAR_INTERVAL = "1h"
SIGNAL_BAR_INTERVAL_SECONDS = 3600

# How many bars of history to load for retraining
RETRAIN_LOOKBACK_BARS = 8760  # ~1 year of hourly bars

# Last retrain timestamp file
RETRAIN_STATE_FILE = FEATURES_DIR.parent / "signals" / "retrain_state.json"

SIGNAL_FEATURE_PREFIX = "evt_"

def _signal_columns(features_df: pd.DataFrame) -> list[str]:
    """Return the list of feature columns used for signal prediction.

    Uses a whitelist of prefixes to avoid accidentally including metadata columns.
    """
    return [c for c in features_df.columns if c.startswith(SIGNAL_FEATURE_PREFIX)]

def _load_current_features(asset: str) -> pd.DataFrame:
    """Load today's features for an asset."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return read_features(asset, today)

def _load_training_features(assets: list[str], lookback_days: int = 90) -> pd.DataFrame | None:
    """Load features for the last N days across assets for retraining.

    Returns a DataFrame with 'close' and 'next_return' columns for the
    signal model target.
    """
    frames = []
    end_date = datetime.now(timezone.utc)
    start_date = end_date - pd.Timedelta(days=lookback_days)

    for asset in assets:
        for dt in pd.date_range(start_date, end_date, freq="D"):
            date_str = dt.strftime("%Y-%m-%d")
            df = read_features(asset, date_str)
            if df.empty:
                continue
            df = df.copy()
            df["asset"] = asset
            if "close" not in df.columns:
                continue
            df["next_return"] = df["close"].pct_change().shift(-1)
            frames.append(df)

    if not frames:
        return None

    combined = pd.concat(frames)
    combined = combined.sort_index()
    return combined.dropna(subset=["next_return"])

def train_signal_model(assets: list[str]) -> SignalPipeline | None:
    """Train a signal model from recent features.

    Args:
        assets: List of asset tickers to include.

    Returns:
        Trained SignalPipeline, or None if insufficient data.
    """
    data = _load_training_features(assets, lookback_days=90)
    if data is None or len(data) < 100:
        log.warning("Insufficient data to train signal model (%d rows)", len(data) if data is not None else 0)
        return None

    signal_cols = _signal_columns(data)
    if not signal_cols:
        log.warning("No signal columns in data")
        return None

    X = data[signal_cols].copy()
    y = data["next_return"]

    # Ensure no NaN in X
    X = X.fillna(0)
    valid = ~y.isna() & ~X.isna().any(axis=1)
    X = X[valid]
    y = y[valid]

    if len(X) < 50:
        log.warning("Not enough valid rows after cleaning (%d)", len(X))
        return None

    model = SignalPipeline(SIGNAL_MODEL_TYPE)
    model.fit(X, y)
    log.info("Trained signal model on %d rows, %d features", len(X), len(signal_cols))
    return model

def _last_retrain_time() -> float:
    """Return the last retrain timestamp in seconds, or 0 if never."""
    if RETRAIN_STATE_FILE.exists():
        try:
            state = json.loads(RETRAIN_STATE_FILE.read_text())
            return float(state.get("last_retrain", 0))
        except Exception:
            return 0
    return 0

def _save_retrain_time(ts: float) -> None:
    """Save the current retrain timestamp."""
    RETRAIN_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    RETRAIN_STATE_FILE.write_text(json.dumps({"last_retrain": ts}))

def _should_retrain() -> bool:
    """Check if it's time to retrain."""
    last = _last_retrain_time()
    if last == 0:
        return True
    elapsed = time.time() - last
    return elapsed > RETRAIN_DAYS * 86400

def generate_signal(
    model: SignalPipeline,
    features_df: pd.DataFrame,
    latest_bar: pd.Series,
) -> float:
    """Generate a position target for the latest bar.

    Args:
        model: Trained signal pipeline.
        features_df: Feature DataFrame for today.
        latest_bar: Latest bar's feature row.

    Returns:
        Position target in [-1, 1].
    """
    signal_cols = _signal_columns(features_df)
    if not signal_cols:
        return 0.0

    X = latest_bar[signal_cols].fillna(0).to_frame().T
    raw_signal = model.predict(X)[0]
    position = SignalPipeline.position_from_signal(
        [raw_signal],
        target_vol=SIGNAL_TARGET_VOL,
        clip_z=SIGNAL_CLIP_Z,
    )[0]
    return float(np.clip(position, -1.0, 1.0))

def run_once(
    model: SignalPipeline | None = None,
    assets: list[str] | None = None,
) -> dict[str, float]:
    """Generate signals for all configured assets.

    Args:
        model: Optional pre-trained model. If None, loads from disk or trains.
        assets: Optional list of assets. Defaults to ASSET_MAP keys.

    Returns:
        Dict mapping asset ticker → position target.
    """
    assets = assets or list(ASSET_MAP.keys())
    positions: dict[str, float] = {}

    # Train or load model
    if model is None:
        if _should_retrain():
            model = train_signal_model(assets)
            if model is not None:
                _save_retrain_time(time.time())
                log.info("Retrained signal model")
        # ponytail: no model persistence yet — retrain on each call if stale
        if model is None:
            model = train_signal_model(assets)
            if model is None:
                log.error("Failed to train signal model")
                return {a: 0.0 for a in assets}

    for asset in assets:
        today_features = _load_current_features(asset)
        if today_features.empty:
            positions[asset] = 0.0
            continue

        latest_bar = today_features.iloc[-1]
        try:
            pos = generate_signal(model, today_features, latest_bar)
            positions[asset] = pos
            log.info("Signal for %s: %.4f", asset, pos)
        except Exception as e:
            log.error("Signal generation failed for %s: %s", asset, e)
            positions[asset] = 0.0

    # Write signal output atomically
    SIGNAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "positions": positions,
    }
    tmp_path = SIGNAL_FILE.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(output, indent=2))
    tmp_path.rename(SIGNAL_FILE)
    log.info("Wrote signals to %s", SIGNAL_FILE)

    return positions

def run_loop(interval_seconds: int = SIGNAL_BAR_INTERVAL_SECONDS):
    """Run the signal server in a loop."""
    log.info("Starting signal server, interval=%ds", interval_seconds)

    stop_event = signal.getsignal(signal.SIGINT)
    def _shutdown(signum, frame):
        log.info("Shutting down signal server...")
        sys.exit(0)
    signal.signal(signal.SIGINT, _shutdown)

    cycle = 0
    while True:
        cycle += 1
        log.info("=== Signal cycle %d ===", cycle)
        try:
            positions = run_once()
            log.info("Cycle %d positions: %s", cycle, json.dumps(positions))
        except Exception as e:
            log.error("Signal cycle %d failed: %s", cycle, e)

        time.sleep(interval_seconds)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    import argparse
    parser = argparse.ArgumentParser(description="AlphaSent signal server")
    parser.add_argument("--loop", action="store_true", help="Run in continuous loop")
    parser.add_argument("--retrain", action="store_true", help="Force retrain")
    args = parser.parse_args()

    if args.loop:
        run_loop()
    else:
        if args.retrain:
            _save_retrain_time(0)  # Force retrain by zeroing
        positions = run_once()
        print(json.dumps(positions, indent=2))

"""
Baseline comparisons for the LLM feature pipeline.

For the LLM feature to justify its complexity, it must beat:
  1. Pure OHLCV momentum (1h, 4h, 24h returns as features, same signal model)
  2. GDELT raw tone (V2Tone directly as feature, no LLM extraction)
  3. No-feature benchmark (buy-and-hold BTC)

Each baseline returns returns series so they can be compared with the
same evaluation functions (Sharpe, DSR, Calmar, etc.).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BINANCE_BASE_URL, BINANCE_SYMBOLS, FEATURES_DIR
from src.backtest.signal import SignalPipeline

log = logging.getLogger("backtest.baselines")

# --- Helper: compute next-bar return from close prices ---

def _next_return(close: pd.Series, periods: int = 1) -> pd.Series:
    """Next-bar return from close prices."""
    return close.pct_change(periods).shift(-periods)

# --- Baseline 1: Pure OHLCV momentum ---

def build_momentum_features(
    close: pd.Series,
    lookback_hours: list[int] = [1, 4, 24],
) -> pd.DataFrame:
    """Build momentum features from close prices.

    For each bar, computes the return over each lookback window.
    """
    features = pd.DataFrame(index=close.index)
    for lb in lookback_hours:
        features[f"ret_lag{lb}h"] = close.pct_change(lb)
    return features

def run_momentum_baseline(
    features_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    asset: str = "BTC",
    symbol: str = "BTCUSDT",
    model: str = "ridge",
) -> pd.Series:
    """Run the pure-momentum baseline.

    Args:
        features_df: Feature DataFrame for the asset.
        ohlcv_df: OHLCV DataFrame for the asset.
        asset: Asset ticker.
        symbol: Binance symbol.
        model: Model type ("ridge" or "lightgbm").

    Returns:
        Returns series for the momentum baseline.
    """
    if "close" not in ohlcv_df.columns:
        raise ValueError("ohlcv_df must have a 'close' column")

    close = ohlcv_df["close"].copy()
    features = build_momentum_features(close)
    features["next_return"] = _next_return(close)

    # Drop NaNs
    combined = features.dropna()
    if combined.empty:
        log.warning("No valid data for momentum baseline")
        return pd.Series(dtype=float)

    train_end = len(combined) * 2 // 3
    X_train = combined.iloc[:train_end].drop(columns=["next_return"])
    y_train = combined.iloc[:train_end]["next_return"]
    X_test = combined.iloc[train_end:].drop(columns=["next_return"])
    y_test = combined.iloc[train_end:]["next_return"]

    signal_model = SignalPipeline(model=model)
    signal_model.fit(X_train, y_train)
    raw_signal = signal_model.predict(X_test)

    # Position sizing
    positions = SignalPipeline.position_from_signal(raw_signal)
    returns = positions * y_test

    returns.index = y_test.index
    return returns

# --- Baseline 2: GDELT raw tone ---

def run_raw_tone_baseline(
    events_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    asset: str = "BTC",
    symbol: str = "BTCUSDT",
    model: str = "ridge",
) -> pd.Series:
    """Run the raw-tone baseline.

    Uses V2Tone directly as a feature (no LLM extraction).
    Requires `raw_tone` column in events_df.
    """
    if "close" not in ohlcv_df.columns:
        raise ValueError("ohlcv_df must have a 'close' column")

    close = ohlcv_df["close"].copy()
    tone_df = pd.DataFrame(index=close.index)
    tone_df["raw_tone_sum"] = 0.0
    tone_df["raw_tone_mean"] = 0.0
    tone_df["n_toned"] = 0

    # For each bar, compute the average raw_tone of events in the lookback
    from src.features.builder import events_visible_at
    from datetime import timedelta

    tone_events = events_df[events_df["asset"] == asset].copy()
    if tone_events.empty or "raw_tone" not in tone_events.columns:
        log.warning("No raw tone data for %s", asset)
        return pd.Series(dtype=float)

    for bar_idx, bar_row in ohlcv_df.iterrows():
        bar_open = pd.Timestamp(bar_row["open_time"])
        visible = events_visible_at(tone_events, asset, bar_open, timedelta(hours=24))
        if not visible.empty:
            tone_vals = visible["raw_tone"].dropna()
            if not tone_vals.empty:
                tone_df.loc[bar_open, "raw_tone_sum"] = float(tone_vals.sum())
                tone_df.loc[bar_open, "raw_tone_mean"] = float(tone_vals.mean())
                tone_df.loc[bar_open, "n_toned"] = len(tone_vals)

    tone_df["next_return"] = _next_return(close)
    combined = tone_df.dropna()
    if combined.empty:
        log.warning("No valid data for raw-tone baseline")
        return pd.Series(dtype=float)

    train_end = len(combined) * 2 // 3
    X_train = combined.iloc[:train_end].drop(columns=["next_return"])
    y_train = combined.iloc[:train_end]["next_return"]
    X_test = combined.iloc[train_end:].drop(columns=["next_return"])
    y_test = combined.iloc[train_end:]["next_return"]

    signal_model = SignalPipeline(model=model)
    signal_model.fit(X_train, y_train)
    raw_signal = signal_model.predict(X_test)

    positions = SignalPipeline.position_from_signal(raw_signal)
    returns = positions * y_test
    returns.index = y_test.index
    return returns

# --- Baseline 3: Buy-and-hold ---

def run_buy_and_hold(
    ohlcv_df: pd.DataFrame,
) -> pd.Series:
    """Run the buy-and-hold baseline.

    Returns the simple next-bar returns series.
    """
    if "close" not in ohlcv_df.columns:
        raise ValueError("ohlcv_df must have a 'close' column")
    return _next_return(ohlcv_df["close"])

# --- Unified runner ---

def run_all_baselines(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    ohlcv_dfs: dict[str, pd.DataFrame],
    asset: str = "BTC",
    symbol: str = "BTCUSDT",
    model: str = "ridge",
) -> dict[str, pd.Series]:
    """Run all baselines and return a dict of returns series.

    Args:
        features_df: Feature DataFrame for the asset.
        events_df: Event records DataFrame.
        ohlcv_dfs: Dict of asset ticker → OHLCV DataFrame.
        asset: Asset ticker.
        symbol: Binance symbol.
        model: Model type ("ridge" or "lightgbm").

    Returns:
        Dict mapping baseline name → returns series.
    """
    ohlcv = ohlcv_dfs.get(symbol, ohlcv_dfs.get(asset, None))
    if ohlcv is None:
        log.error("No OHLCV data for %s", asset)
        return {}

    results = {}

    # Buy-and-hold
    results["buy_and_hold"] = run_buy_and_hold(ohlcv)
    log.info("Buy-and-hold baseline: %d bars", len(results["buy_and_hold"]))

    # Momentum
    try:
        mom_returns = run_momentum_baseline(features_df, ohlcv, asset, symbol, model)
        results["momentum"] = mom_returns
        log.info("Momentum baseline: %d bars", len(mom_returns))
    except Exception as e:
        log.error("Momentum baseline failed: %s", e)

    # Raw-tone
    try:
        tone_returns = run_raw_tone_baseline(events_df, ohlcv, asset, symbol, model)
        results["raw_tone"] = tone_returns
        log.info("Raw-tone baseline: %d bars", len(tone_returns))
    except Exception as e:
        log.error("Raw-tone baseline failed: %s", e)

    return results

if __name__ == "__main__":
    """Smoke test: run baselines on a small synthetic dataset."""
    logging.basicConfig(level=logging.INFO)
    np.random.seed(42)
    n = 500
    ohlcv = pd.DataFrame({
        "open_time": pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC"),
        "close": 1.0 + np.cumsum(np.random.randn(n) * 0.01),
    })
    events = pd.DataFrame({
        "asset": ["BTC"] * 50,
        "raw_tone": np.random.randn(50),
        "published_at": pd.date_range("2021-01-01", periods=50, freq="2h", tz="UTC"),
    })
    features = pd.DataFrame({
        "bar_open_ts": pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC"),
        "close": 1.0 + np.cumsum(np.random.randn(n) * 0.01),
    })
    results = run_all_baselines(features, events, {"BTCUSDT": ohlcv}, "BTC", "BTCUSDT", "ridge")
    for name, returns in results.items():
        print(f"{name}: {len(returns)} bars")

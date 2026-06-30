from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CACHE_DIR

def extraction_success_rate(events: pd.DataFrame, total_attempted: int) -> float:
    if total_attempted == 0:
        return 1.0
    return len(events) / total_attempted

def event_type_drift(
    events: pd.DataFrame,
    baseline_window_days: int = 7,
    current_window_days: int = 1,
) -> dict:
    if events.empty or "event_type" not in events.columns or "published_at" not in events.columns:
        return {"drift": 0.0, "baseline": {}, "current": {}}

    events = events.copy()
    events["published_at"] = pd.to_datetime(events["published_at"], utc=True)
    now = events["published_at"].max()
    current_start = now - pd.Timedelta(days=current_window_days)
    baseline_start = now - pd.Timedelta(days=baseline_window_days)
    baseline_end = now - pd.Timedelta(days=current_window_days)

    current = events[(events["published_at"] >= current_start) & (events["published_at"] < now)]
    baseline = events[(events["published_at"] >= baseline_start) & (events["published_at"] < baseline_end)]

    if current.empty or baseline.empty:
        return {"drift": 0.0, "baseline": {}, "current": {}}

    current_dist = Counter(current["event_type"])
    baseline_dist = Counter(baseline["event_type"])
    all_types = set(current_dist.keys()) | set(baseline_dist.keys())

    current_probs = {t: current_dist[t] / len(current) for t in all_types}
    baseline_probs = {t: baseline_dist[t] / len(baseline) for t in all_types}

    drift = 0.5 * sum(abs(current_probs.get(t, 0) - baseline_probs.get(t, 0)) for t in all_types)

    return {
        "drift": float(drift),
        "baseline": {k: float(v) for k, v in baseline_dist.items()},
        "current": {k: float(v) for k, v in current_dist.items()},
    }

def signal_autocorrelation(signals: pd.Series, max_lag: int = 5) -> pd.Series:
    if len(signals) < max_lag + 1:
        return pd.Series(dtype=float)
    acf = pd.Series(dtype=float)
    for lag in range(1, max_lag + 1):
        shifted = signals.shift(lag)
        valid = pd.concat([signals, shifted], axis=1).dropna()
        if len(valid) < 2:
            acf[lag] = 0.0
            continue
        c = valid.iloc[:, 0].corr(valid.iloc[:, 1])
        if c is None or not np.isfinite(c):
            c = 1.0 if valid.iloc[:, 0].std() == 0 else 0.0
        acf[lag] = float(c)
    return acf

def cache_miss_rate(attempted_hashes: list[str], cache_dir: Path | None = None) -> float:
    if not attempted_hashes:
        return 0.0
    from src.extraction.cache import cache_exists
    from src.config import MODEL_VERSION, PROMPT_VERSION
    misses = sum(1 for h in attempted_hashes if not cache_exists(h, MODEL_VERSION, PROMPT_VERSION))
    return misses / len(attempted_hashes)

def run_all_checks(
    events_df: pd.DataFrame | None = None,
    signals: pd.Series | None = None,
    attempted_hashes: list[str] | None = None,
) -> dict:
    if events_df is None:
        events_df = pd.DataFrame()
    if signals is None:
        signals = pd.Series(dtype=float)
    if attempted_hashes is None:
        attempted_hashes = []

    total_attempted = max(len(events_df), len(attempted_hashes))
    if total_attempted == 0 and signals.empty:
        return {
            "extraction_success_rate": 1.0,
            "event_type_drift": event_type_drift(events_df),
            "signal_autocorrelation": {},
            "stale_bars_count": 0,
            "cache_miss_rate": 0.0,
            "alerts": [],
        }
    success_rate = extraction_success_rate(events_df, total_attempted)

    drift = event_type_drift(events_df)

    acf = signal_autocorrelation(signals) if not signals.empty else pd.Series(dtype=float)
    stale_bars = int((acf[acf > 0.9]).sum()) if len(acf) > 0 else 0

    miss_rate = cache_miss_rate(attempted_hashes)

    return {
        "extraction_success_rate": float(success_rate),
        "event_type_drift": drift,
        "signal_autocorrelation": acf.to_dict() if len(acf) > 0 else {},
        "stale_bars_count": stale_bars,
        "cache_miss_rate": float(miss_rate),
        "alerts": [
            "low_extraction_rate" if success_rate < 0.85 else None,
            "high_drift" if drift["drift"] > 0.3 else None,
            "stale_signal" if stale_bars > 3 else None,
            "high_cache_miss" if miss_rate > 0.3 else None,
        ],
    }

if __name__ == "__main__":
    np.random.seed(42)
    events = pd.DataFrame({
        "published_at": pd.date_range("2026-01-01", periods=30, tz="UTC"),
        "event_type": np.random.choice(["macro", "hack", "partnership", "regulation"], 30),
    })
    signals = pd.Series(np.random.randn(20), name="signal")
    report = run_all_checks(events_df=events, signals=signals, attempted_hashes=["abc123"])
    print(json.dumps(report, indent=2, default=str))
    assert 0.0 <= report["extraction_success_rate"] <= 1.0
    assert isinstance(report["event_type_drift"], dict)
    assert isinstance(report["signal_autocorrelation"], dict)
    assert isinstance(report["cache_miss_rate"], float)
    print("OK")

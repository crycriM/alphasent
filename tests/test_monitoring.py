from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.live.monitoring import (
    cache_miss_rate,
    event_type_drift,
    extraction_success_rate,
    run_all_checks,
    signal_autocorrelation,
)

def test_extraction_success_rate():
    df = pd.DataFrame({"published_at": [1, 2, 3, 4], "event_type": ["macro"] * 4})
    assert extraction_success_rate(df, 5) == pytest.approx(0.8)
    assert extraction_success_rate(df, 4) == 1.0
    assert extraction_success_rate(df, 0) == 1.0
    assert extraction_success_rate(pd.DataFrame(), 10) == 0.0

def test_event_type_drift():
    dates = pd.date_range("2026-01-01", periods=30, freq="D", tz="UTC")
    events = pd.DataFrame({
        "published_at": dates,
        "event_type": ["macro"] * 20 + ["hack"] * 10,
    })
    drift = event_type_drift(events)
    assert "drift" in drift
    assert 0.0 <= drift["drift"] <= 1.0
    assert "baseline" in drift
    assert "current" in drift

def test_event_type_drift_empty():
    result = event_type_drift(pd.DataFrame())
    assert result["drift"] == 0.0

def test_signal_autocorrelation():
    np.random.seed(42)
    signals = pd.Series(np.random.randn(50))
    acf = signal_autocorrelation(signals, max_lag=5)
    assert len(acf) == 5
    for lag in range(1, 6):
        assert lag in acf.index
        assert -1.0 <= acf[lag] <= 1.0

def test_signal_autocorrelation_short():
    signals = pd.Series([1.0, 2.0, 3.0])
    acf = signal_autocorrelation(signals, max_lag=10)
    assert len(acf) <= 10

def test_signal_autocorrelation_stale_detection():
    signals = pd.Series([1.0] * 20)
    acf = signal_autocorrelation(signals, max_lag=5)
    stale_bars = int((acf[acf > 0.9]).sum())
    assert stale_bars > 0

def test_cache_miss_rate_empty():
    assert cache_miss_rate([]) == 0.0

def test_cache_miss_rate():
    from src.extraction.cache import compute_content_hash
    hashes = [compute_content_hash("title", "body")]
    rate = cache_miss_rate(hashes)
    assert 0.0 <= rate <= 1.0

def test_run_all_checks():
    events = pd.DataFrame({
        "published_at": pd.date_range("2026-01-01", periods=30, freq="D", tz="UTC"),
        "event_type": ["macro"] * 30,
    })
    signals = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
    report = run_all_checks(events_df=events, signals=signals, attempted_hashes=["abc"])
    assert isinstance(report, dict)
    assert "extraction_success_rate" in report
    assert "event_type_drift" in report
    assert "signal_autocorrelation" in report
    assert "stale_bars_count" in report
    assert "cache_miss_rate" in report
    assert "alerts" in report
    json.dumps(report, default=str)

def test_run_all_checks_empty():
    report = run_all_checks()
    assert report["extraction_success_rate"] == 1.0
    assert isinstance(report, dict)

def test_alert_thresholds_extraction():
    events = pd.DataFrame({
        "published_at": [1, 2, 3],
        "event_type": ["macro"] * 3,
    })
    report = run_all_checks(events_df=events, attempted_hashes=["h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9", "h10"])
    alerts = [a for a in report["alerts"] if a is not None]
    assert "low_extraction_rate" in alerts

def test_alert_thresholds_stale():
    signals = pd.Series([1.0] * 10)
    report = run_all_checks(signals=signals)
    alerts = [a for a in report["alerts"] if a is not None]
    assert "stale_signal" in alerts

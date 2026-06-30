import pytest
import pandas as pd

from src.backtest.walkforward import generate_windows

class TestWalkForward:
    """Window generation invariants."""

    def test_no_train_test_overlap(self):
        """Train and test within each window should not overlap."""
        windows = generate_windows(
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2024-01-01"),
            train_months=12,
            test_months=3,
            step_months=3,
        )
        for ts, te, est, eet in windows:
            assert te < est, f"train_end {te} >= test_start {est}"

    def test_within_range(self):
        start = pd.Timestamp("2020-01-01")
        end = pd.Timestamp("2024-01-01")
        windows = generate_windows(start, end)
        for ts, te, est, eet in windows:
            assert ts >= start
            assert eet <= end

    def test_rolling_train(self):
        """Train window should advance by step_months each iteration."""
        windows = generate_windows(
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2024-01-01"),
            train_months=12,
            test_months=3,
            step_months=3,
        )
        for i in range(len(windows) - 1):
            ts_curr, _, _, _ = windows[i]
            ts_next, _, _, _ = windows[i + 1]
            assert ts_next == ts_curr + pd.DateOffset(months=3), f"train should advance by step"

    def test_window_count(self):
        windows = generate_windows(
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2023-12-31"),
            train_months=12,
            test_months=3,
            step_months=3,
        )
        # First test starts 2021-01, steps of 3 months. 12 full windows.
        assert len(windows) == 12

    def test_no_windows_short_range(self):
        windows = generate_windows(
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2020-06-01"),
            train_months=12,
            test_months=3,
        )
        assert len(windows) == 0

    def test_tuple_length(self):
        windows = generate_windows(
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2024-01-01"),
        )
        for w in windows:
            assert len(w) == 4

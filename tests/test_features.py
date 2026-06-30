"""
Point-in-time safety tests for the feature builder.

These tests verify the strict < bar_open boundary that prevents
look-ahead contamination. Write these before data exists —
they test the join logic, not the data.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.features.builder import (
    build_feature_vector,
    events_visible_at,
    zero_feature_vector,
)


# --------------------------------------------------------------------------- #
# Point-in-time guard tests
# --------------------------------------------------------------------------- #


class TestEventsVisibleAt:
    """Test the strict < bar_open_ts boundary."""

    def test_event_exactly_at_bar_open_not_visible(self):
        """CRITICAL: an event at the exact bar open must NOT be visible."""
        t = pd.Timestamp("2021-06-01 00:00:00", tz="UTC")
        ev = pd.DataFrame({
            "asset": ["BTC"],
            "published_at": [t],
            "polarity": [-0.5],
            "magnitude": [0.5],
            "novelty": [1.0],
            "confidence": [0.9],
            "event_type": ["hack"],
        })
        visible = events_visible_at(ev, "BTC", t, pd.Timedelta("24h"))
        assert len(visible) == 0, (
            f"FAIL: event at bar open leaked into the bar. "
            f"expected 0 rows, got {len(visible)}"
        )

    def test_event_1ms_before_bar_open_visible(self):
        """Event 1ms before bar open MUST be visible."""
        t = pd.Timestamp("2021-06-01 00:00:00", tz="UTC")
        event_time = t - pd.Timedelta(milliseconds=1)
        ev = pd.DataFrame({
            "asset": ["BTC"],
            "published_at": [event_time],
            "polarity": [-0.5],
            "magnitude": [0.5],
            "novelty": [1.0],
            "confidence": [0.9],
            "event_type": ["hack"],
        })
        visible = events_visible_at(ev, "BTC", t, pd.Timedelta("24h"))
        assert len(visible) == 1, (
            f"FAIL: event 1ms before bar open should be visible. "
            f"expected 1 row, got {len(visible)}"
        )

    def test_event_1ms_after_bar_open_not_visible(self):
        """Event 1ms after bar open must NOT be visible."""
        t = pd.Timestamp("2021-06-01 00:00:00", tz="UTC")
        event_time = t + pd.Timedelta(milliseconds=1)
        ev = pd.DataFrame({
            "asset": ["BTC"],
            "published_at": [event_time],
            "polarity": [-0.5],
            "magnitude": [0.5],
            "novelty": [1.0],
            "confidence": [0.9],
            "event_type": ["hack"],
        })
        visible = events_visible_at(ev, "BTC", t, pd.Timedelta("24h"))
        assert len(visible) == 0, (
            f"FAIL: event after bar open leaked into the bar."
        )

    def test_event_within_lookback_visible(self):
        """Event within lookback window before bar open must be visible."""
        t = pd.Timestamp("2021-06-01 12:00:00", tz="UTC")
        event_time = pd.Timestamp("2021-06-01 06:00:00", tz="UTC")  # 6h before
        ev = pd.DataFrame({
            "asset": ["BTC"],
            "published_at": [event_time],
            "polarity": [0.3],
            "magnitude": [0.4],
            "novelty": [0.8],
            "confidence": [0.7],
            "event_type": ["partnership"],
        })
        visible = events_visible_at(ev, "BTC", t, pd.Timedelta("24h"))
        assert len(visible) == 1

    def test_event_outside_lookback_not_visible(self):
        """Event outside the lookback window must NOT be visible."""
        t = pd.Timestamp("2021-06-01 12:00:00", tz="UTC")
        event_time = pd.Timestamp("2021-05-30 12:00:00", tz="UTC")  # 48h before
        ev = pd.DataFrame({
            "asset": ["BTC"],
            "published_at": [event_time],
            "polarity": [0.3],
            "magnitude": [0.4],
            "novelty": [0.8],
            "confidence": [0.7],
            "event_type": ["partnership"],
        })
        visible = events_visible_at(ev, "BTC", t, pd.Timedelta("24h"))
        assert len(visible) == 0

    def test_different_asset_filtered_out(self):
        """Events for a different asset must not leak."""
        t = pd.Timestamp("2021-06-01 00:00:00", tz="UTC")
        event_time = t - pd.Timedelta(hours=1)
        ev = pd.DataFrame({
            "asset": ["ETH"],  # different asset
            "published_at": [event_time],
            "polarity": [-0.5],
            "magnitude": [0.5],
            "novelty": [1.0],
            "confidence": [0.9],
            "event_type": ["hack"],
        })
        visible = events_visible_at(ev, "BTC", t, pd.Timedelta("24h"))
        assert len(visible) == 0


class TestBuildFeatureVector:
    """Test feature vector aggregation."""

    def test_empty_events_yields_zero_vector(self):
        """Empty event list should return zero-filled vector."""
        vec = build_feature_vector(pd.DataFrame(), lookback_hours=24)
        assert vec["n_events"] == 0
        assert vec["polarity_sum"] == 0.0
        assert vec["hack_flag"] == 0
        assert vec["mag_weighted_polarity"] == 0.0
        assert vec["novelty_polarity"] == 0.0

    def test_single_event_aggregation(self):
        """Single event should aggregate correctly."""
        ev = pd.DataFrame([{
            "polarity": -0.5,
            "magnitude": 0.6,
            "novelty": 0.9,
            "confidence": 0.85,
            "event_type": "hack",
        }])
        vec = build_feature_vector(ev, lookback_hours=24)
        assert vec["n_events"] == 1
        assert vec["n_high_conf_events"] == 1  # conf > 0.8
        assert abs(vec["polarity_sum"] - (-0.5)) < 1e-9
        assert abs(vec["polarity_mean"] - (-0.5)) < 1e-9
        assert abs(vec["polarity_min"] - (-0.5)) < 1e-9
        assert abs(vec["polarity_max"] - (-0.5)) < 1e-9
        # mag_weighted = polarity * magnitude
        assert abs(vec["mag_weighted_polarity"] - (-0.5 * 0.6)) < 1e-9
        # novelty_polarity = polarity * novelty * magnitude
        assert abs(vec["novelty_polarity"] - (-0.5 * 0.9 * 0.6)) < 1e-9
        assert vec["hack_flag"] == 1
        assert vec["regulation_flag"] == 0

    def test_zero_feature_vector(self):
        """Zero feature vector should have all zeros."""
        vec = zero_feature_vector(24)
        assert vec["n_events"] == 0
        assert vec["lookback_h"] == 24
        assert all(v == 0.0 for k, v in vec.items() if k not in ("n_events", "n_high_conf_events",
                                                                "hack_flag", "regulation_flag",
                                                                "listing_flag", "depeg_flag",
                                                                "lookback_h"))


class TestPITSafetyIntegration:
    """Integration test: build features and verify no bar has its own events."""

    def test_no_same_bar_event_leak(self):
        """
        Simulate a scenario where events and bars share timestamps.
        Verify that the feature builder does not include same-bar events.
        """
        # Create events at specific times
        events = pd.DataFrame([
            {
                "asset": "BTC",
                "published_at": pd.Timestamp("2021-06-01 00:00:00", tz="UTC"),
                "polarity": -0.5, "magnitude": 0.5, "novelty": 1.0,
                "confidence": 0.9, "event_type": "hack",
            },
            {
                "asset": "BTC",
                "published_at": pd.Timestamp("2021-06-01 01:00:00", tz="UTC"),
                "polarity": 0.3, "magnitude": 0.3, "novelty": 0.8,
                "confidence": 0.7, "event_type": "partnership",
            },
        ])

        # Bars at the same timestamps
        bars = pd.DataFrame([
            {"open_time": pd.Timestamp("2021-06-01 00:00:00", tz="UTC")},
            {"open_time": pd.Timestamp("2021-06-01 01:00:00", tz="UTC")},
            {"open_time": pd.Timestamp("2021-06-01 02:00:00", tz="UTC")},
        ])

        # For bar at 00:00:00, the event at 00:00:00 should NOT be visible
        visible_at_00 = events_visible_at(events, "BTC",
                                         pd.Timestamp("2021-06-01 00:00:00", tz="UTC"),
                                         pd.Timedelta("24h"))
        assert len(visible_at_00) == 0, (
            "Same-bar event leaked into feature for bar at 00:00:00"
        )

        # For bar at 01:00:00, the event at 00:00:00 SHOULD be visible
        # (it's 1h before bar open, within 24h lookback)
        visible_at_01 = events_visible_at(events, "BTC",
                                         pd.Timestamp("2021-06-01 01:00:00", tz="UTC"),
                                         pd.Timedelta("24h"))
        assert len(visible_at_01) == 1, (
            "Prior event should be visible for bar at 01:00:00"
        )
        assert visible_at_01.iloc[0]["polarity"] == -0.5

        # For bar at 02:00:00, BOTH prior events should be visible
        visible_at_02 = events_visible_at(events, "BTC",
                                         pd.Timestamp("2021-06-01 02:00:00", tz="UTC"),
                                         pd.Timedelta("24h"))
        assert len(visible_at_02) == 2, (
            "Both prior events should be visible for bar at 02:00:00"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

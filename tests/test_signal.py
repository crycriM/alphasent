import pytest
import numpy as np
import pandas as pd

from src.backtest.signal import SignalPipeline

class TestSignalPipeline:
    """Fit/predict roundtrip, position sizing bounds, lightgbm flag path."""

    def test_fit_predict_roundtrip(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(200, 5))
        y = pd.Series(np.random.randn(200))
        sp = SignalPipeline("ridge")
        sp.fit(X, y)
        pred = sp.predict(X)
        assert len(pred) == 200
        assert not np.all(np.isnan(pred))

    def test_position_bounded(self):
        np.random.seed(42)
        signals = np.random.randn(200)
        pos = SignalPipeline.position_from_signal(signals, target_vol=0.10)
        assert pos.abs().max() <= 0.105, "position should be bounded near target_vol"

    def test_constant_signal(self):
        signals = np.ones(50)
        pos = SignalPipeline.position_from_signal(signals)
        assert (pos == 0).all(), "constant signal should give zero position"

    def test_lightgbm_path(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(200, 5))
        y = pd.Series(np.random.randn(200))
        sp = SignalPipeline("lightgbm")
        sp.fit(X, y)
        pred = sp.predict(X)
        assert len(pred) == 200

    def test_clip_z(self):
        np.random.seed(42)
        signals = np.random.randn(100) * 10  # large signal
        pos = SignalPipeline.position_from_signal(signals, clip_z=2.0, target_vol=0.10)
        assert pos.abs().max() <= 0.105, "clipped position should still be bounded"

    def test_unknown_model(self):
        with pytest.raises(ValueError):
            SignalPipeline("xgboost")
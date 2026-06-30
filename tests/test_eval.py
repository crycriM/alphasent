import math
import pytest
import numpy as np
import pandas as pd

from src.backtest.eval import (
    sharpe,
    deflated_sharpe,
    deflated_sharpe_with_ci,
    calmar,
    hit_rate,
    max_drawdown,
    attribution,
)

class TestEvaluation:
    """Sharpe, DSR, Calmar, hit rate, max DD, attribution."""

    def test_sharpe_constant_returns(self):
        ret = pd.Series([0.01] * 100)
        assert math.isnan(sharpe(ret))

    def test_sharpe_positive(self):
        np.random.seed(42)
        ret = pd.Series(np.random.randn(200))
        sr = sharpe(ret)
        assert not math.isnan(sr)

    def test_hit_rate_bounds(self):
        np.random.seed(42)
        ret = pd.Series(np.random.randn(100))
        assert 0.0 <= hit_rate(ret) <= 1.0

    def test_max_dd_negative(self):
        eq = pd.Series([1.0, 0.9, 0.8, 0.9, 1.1])
        assert max_drawdown(eq) < 0

    def test_dsr_n_trials_1_equals_sr(self):
        np.random.seed(42)
        ret = pd.Series(np.random.randn(200))
        dsr = deflated_sharpe(ret, n_trials=1)
        assert not math.isnan(dsr)

    def test_dsr_with_ci(self):
        np.random.seed(42)
        ret = pd.Series(np.random.randn(200) * 0.01)
        dsr, lo, hi = deflated_sharpe_with_ci(ret, n_trials=1, n_boot=100)
        assert not math.isnan(dsr)
        assert lo <= hi

    def test_dsr_nan(self):
        ret = pd.Series([0.0] * 20)
        assert math.isnan(deflated_sharpe(ret))

    def test_attribution(self):
        np.random.seed(42)
        returns = pd.Series(np.random.randn(100))
        signals = pd.DataFrame({
            "hack_flag": (np.random.random(100) > 0.8).astype(int),
            "regulation_flag": (np.random.random(100) > 0.9).astype(int),
        })
        attr = attribution(returns, signals)
        assert "hack_flag" in attr
        assert "sharpe" in attr["hack_flag"]

    def test_calmar(self):
        np.random.seed(42)
        ret = pd.Series(np.random.randn(100))
        cm = calmar(ret)
        assert not math.isnan(cm)
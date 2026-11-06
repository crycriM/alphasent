"""vol_study: the seasonal baseline removes hour-of-week structure; the HAR
residual removes clustering; a feature carrying real forward-vol info survives."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import vol_study as vs  # noqa: E402


def _panel(seed=0, weeks=30):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-05", periods=weeks * 168, freq="h", tz="UTC")
    seasonal = 1.0 + 1.5 * (idx.hour >= 13) * (idx.hour < 21) * (idx.dayofweek < 5)
    shock = np.exp(rng.normal(0, 0.7, len(idx) // 24).repeat(24))   # daily vol state, unknown ex ante
    sigma = 0.004 * np.sqrt(seasonal * shock)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 1, len(idx)) * sigma)), index=idx)
    return close, pd.Series(seasonal, index=idx), pd.Series(np.log(shock), index=idx)


def test_seasonal_feature_is_absorbed_and_informed_feature_survives():
    close, seasonal, log_shock = _panel()
    h = 4
    v = vs.vol_frame(close, h)
    lab = pd.Series("rss", index=v.index)
    resid = vs.har_resid(v)
    # a feature that only knows the clock: predicts raw RV, not seasonal-adjusted RV
    seas_ic = vs.within_ic(seasonal, v["y"], lab).mean()
    # a feature that knows the next h hours' vol state ("news before the move")
    informed = log_shock.shift(-h)
    inf_ic = vs.within_ic(informed, resid, lab).mean()
    assert abs(seas_ic) < 0.05, seas_ic
    assert inf_ic > 0.2, inf_ic


def test_seasonal_profile_is_point_in_time():
    close, _, _ = _panel(weeks=8)
    base = vs.vol_frame(close, 4)
    later = close.copy()
    later.iloc[-24:] *= 1.5          # change only the last day
    alt = vs.vol_frame(later, 4)
    cut = close.index[-24 - 4 - 1]   # forward window ends before the change
    pd.testing.assert_series_equal(base.loc[:cut, "x24"], alt.loc[:cut, "x24"])

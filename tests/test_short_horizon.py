"""short_horizon: abnormal returns are measured after t0 only, net of beta x market."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import short_horizon as sh  # noqa: E402

IDX = pd.date_range("2026-07-01", periods=600, freq="min", tz="UTC")


def test_abnormal_picks_up_post_event_jump_only():
    lp = pd.DataFrame(0.0, index=IDX, columns=list("ABCDE"))
    t0 = IDX[[100, 300]]
    lp.loc[IDX[103]:, "A"] += 0.01                  # +100 bp jump on A, 3 min after event 1
    lp.loc[IDX[350]:, :] += 0.02                    # market move, all assets, beta 1
    ev_asset = np.array(["A", "B"])
    beta = np.ones(2)
    post5 = sh.abnormal(lp, beta, ev_asset, t0, 5)
    pre15 = sh.abnormal(lp, beta, ev_asset, t0, -15)
    post60 = sh.abnormal(lp, beta, ev_asset, t0, 60)
    np.testing.assert_allclose(post5, [0.01, 0.0], atol=1e-12)
    np.testing.assert_allclose(pre15, [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(post60, [0.01, 0.0], atol=1e-12)  # market move nets out


def test_day_t_clusters_by_day():
    idx = pd.date_range("2026-07-01", periods=20, freq="D", tz="UTC").repeat(50)
    x = pd.Series(np.tile(np.r_[np.ones(10), -np.ones(10)].repeat(50), 1)[:len(idx)], index=idx)
    mean, t, days = sh.day_t(x)
    assert days == 20 and abs(mean) < 1e-12 and abs(t) < 1e-9

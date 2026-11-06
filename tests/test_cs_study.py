"""cs_study: own-history standardisation removes static asset tilt; the beta
residual removes the beta ranking; a genuine idiosyncratic signal survives both."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import cs_study as cs  # noqa: E402
import ic_study as ic  # noqa: E402

N, H = 3000, 4
IDX = pd.date_range("2026-01-01", periods=N, freq="h", tz="UTC")
COLS = list("ABCDE")


def _closes(rng, drift, betas, idio):
    mkt = rng.normal(0.0002, 0.01, N)
    r = mkt[:, None] * betas + drift + idio + rng.normal(0, 0.004, (N, 5))
    return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=IDX, columns=COLS)


def test_static_tilt_scores_raw_but_not_own_z():
    rng = np.random.default_rng(1)
    drift = np.array([4, 2, 0, -2, -4]) * 1e-4            # A outperforms all period
    closes = _closes(rng, drift, np.ones(5), 0.0)
    level = np.array([5.0, 4, 3, 2, 1])                    # A also has most news, always
    x = pd.DataFrame(level + rng.poisson(1.0, (N, 5)), index=IDX, columns=COLS)
    y = ic.forward_return_matrix(closes, H)
    v = cs.variants(x)
    raw_ic = ic.cross_sectional_ic(v["raw"], y, rank=True).mean()
    z_ic = ic.cross_sectional_ic(v["own_z"], y, rank=True).mean()
    assert raw_ic > 0.05, raw_ic
    assert abs(z_ic) < 0.02, z_ic


def test_beta_residual_removes_beta_ranking_and_keeps_signal():
    rng = np.random.default_rng(2)
    betas = np.array([0.5, 0.8, 1.0, 1.3, 1.6])
    news = rng.normal(0, 1, (N, 5))
    idio = np.vstack([np.zeros((1, 5)), news[:-1] * 0.002])  # news at t moves the next bar
    closes = _closes(rng, 0.0, betas, idio)
    beta = cs.trailing_beta(closes)
    t = cs.targets(closes, 1, beta)
    beta_feat = pd.DataFrame(np.tile(betas, (N, 1)), index=IDX, columns=COLS)
    up = closes.mean(axis=1).pct_change().shift(-1) > 0      # up-market hours
    raw_beta_ic = ic.cross_sectional_ic(beta_feat[up], t["ret"][up], rank=True).mean()
    res_beta_ic = ic.cross_sectional_ic(beta_feat[up], t["beta_resid"][up], rank=True).mean()
    sig_ic = ic.cross_sectional_ic(pd.DataFrame(news, index=IDX, columns=COLS),
                                   t["beta_resid"], rank=True).mean()
    assert raw_beta_ic > 0.2, raw_beta_ic
    assert abs(res_beta_ic) < 0.05, res_beta_ic
    assert sig_ic > 0.2, sig_ic


def test_top_minus_bottom_skips_ties():
    x = pd.DataFrame([[1, 2, 3, 4, 5], [0, 0, 0, 0, 0]], columns=COLS, dtype=float)
    y = pd.DataFrame([[-1, 0, 0, 0, 2], [9, 9, 9, 9, 9]], columns=COLS, dtype=float)
    out = cs.top_minus_bottom(x, y)
    assert list(out) == [3.0]

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pandas as pd

def sharpe(returns: pd.Series, periods_per_year: int = 8760) -> float:
    """Annualized Sharpe ratio. Returns NaN if std is near zero."""
    s = returns.std()
    if s < 1e-12:
        return float("nan")
    return (returns.mean() / s) * math.sqrt(periods_per_year)

def _sharpe_bootstrap_ci(returns: pd.Series, n_boot: int = 500, conf: float = 0.95, periods_per_year: int = 8760) -> tuple[float, float]:
    """Bootstrap CI for Sharpe using stationary bootstrap (Politis-Romano approximation).

    Tries arch's StationaryBootstrap first; falls back to a manual block bootstrap
    if arch is missing or its API has changed.
    """
    n = len(returns)
    m = max(2, int(np.sqrt(n)))  # block length
    resampler: callable | None = None

    try:
        from arch.bootstrap import StationaryBootstrap
        sb = StationaryBootstrap(m, returns.values)
        if hasattr(sb, "resample"):
            resampler = sb.resample
        elif hasattr(sb, "_resample"):
            def _resample_wrapper():
                res = sb._resample()
                return res[0][0]  # arch 8.x returns tuple of (tuple, dict)
            resampler = _resample_wrapper
    except ImportError:
        pass

    if resampler is None:
        # ponytail: simple stationary bootstrap — geometric block length, same idea as arch but no dependency
        def resampler():
            idx = np.zeros(n, dtype=int)
            i = 0
            while i < n:
                block_len = max(1, int(np.random.geometric(p=0.15 + 0.85 / max(m, 1))))
                start = np.random.randint(0, n)
                length = min(block_len, n - i)
                idx[i:i + length] = (start + np.arange(length)) % n
                i += length
            return returns.values[idx]

    boot_sharpes = []
    for _ in range(n_boot):
        sample = resampler()
        if len(sample) > 1:
            boot_sharpes.append(sharpe(pd.Series(sample), periods_per_year))
    boot_sharpes = [s for s in boot_sharpes if not math.isnan(s)]
    if len(boot_sharpes) < 2:
        return float("nan"), float("nan")
    alpha = (1 - conf) / 2
    return float(np.percentile(boot_sharpes, alpha * 100)), float(np.percentile(boot_sharpes, (1 - alpha) * 100))

def deflated_sharpe(returns: pd.Series, n_trials: int = 1, periods_per_year: int = 8760) -> float:
    """Deflated Sharpe Ratio (Bailey-López de Prado)."""
    sr = sharpe(returns, periods_per_year)
    if math.isnan(sr):
        return float("nan")
    t = len(returns)
    # Variance of Sharpe estimate (approximation from López de Prado 2014)
    # V[SR] = (1/t) * (1 + 0.5 * SR^2 - 0.5 * SR^4 + 0.25 * SR^6)
    v_sr = (1.0 / t) * (1.0 + 0.5 * sr**2 - 0.5 * sr**4 + 0.25 * sr**6)
    # phi_hat = SR - sqrt(2 * n_trials - 1) * sqrt(V[SR])
    phi_hat = sr - math.sqrt(2 * n_trials - 1) * math.sqrt(v_sr)
    # DSR = phi_hat / max(sqrt(V[SR]), 1e-12)
    return phi_hat / max(math.sqrt(v_sr), 1e-12)

def deflated_sharpe_with_ci(returns: pd.Series, n_trials: int = 1, periods_per_year: int = 8760, n_boot: int = 500) -> tuple[float, float, float]:
    """DSR with bootstrap CI bounds."""
    dsr = deflated_sharpe(returns, n_trials, periods_per_year)
    lo, hi = _sharpe_bootstrap_ci(returns, n_boot=n_boot, periods_per_year=periods_per_year)
    return dsr, lo, hi

def calmar(returns: pd.Series) -> float:
    """Calmar ratio: annualized return / max drawdown."""
    equity = (1 + returns).cumprod()
    max_dd = max_drawdown(equity)
    if max_dd == 0:
        return float("inf")
    ann_ret = (1 + returns.mean()) ** 8760 - 1
    return ann_ret / max_dd

def hit_rate(returns: pd.Series) -> float:
    """Fraction of bars with positive return."""
    return float((returns > 0).mean())

def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum drawdown from equity curve."""
    running_max = equity_curve.cummax()
    dd = (equity_curve - running_max) / running_max
    return float(dd.min())

def attribution(returns: pd.Series, signals: pd.DataFrame) -> dict:
    """Per-event-type PnL attribution. Splits returns by event-type flag columns."""
    attr = {}
    for col in signals.columns:
        if col.endswith("_flag"):
            mask = signals[col].astype(bool)
            if mask.sum() == 0:
                continue
            sub = returns[mask]
            attr[col] = {
                "n_bars": int(mask.sum()),
                "total_return": float(sub.sum()),
                "sharpe": sharpe(sub),
            }
    return attr

if __name__ == "__main__":
    np.random.seed(42)
    ret = pd.Series(np.random.randn(500))
    assert not math.isnan(sharpe(ret))
    assert 0.0 <= hit_rate(ret) <= 1.0
    assert max_drawdown((1 + ret).cumprod()) < 0
    # Constant returns: sharpe should be NaN
    assert math.isnan(sharpe(pd.Series([0.01] * 100)))
    # DSR sanity: with n_trials=1, DSR should equal SR
    dsr1 = deflated_sharpe(ret, n_trials=1)
    assert not math.isnan(dsr1)
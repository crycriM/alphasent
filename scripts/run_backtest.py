from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.walkforward import generate_windows, load_window_data
from src.backtest.signal import SignalPipeline
from src.backtest.eval import (
    sharpe,
    hit_rate,
    max_drawdown,
    deflated_sharpe_with_ci,
)

ASSETS_DEFAULT = ["BTC"]
OHLCV_DIR = None  # set to data/ohlcv path if available

def _synthetic_data(n_bars: int = 300, n_assets: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic features + OHLCV for self-check."""
    np.random.seed(42)
    dates = pd.date_range("2021-01-01", periods=n_bars, freq="D")
    rows = []
    for a in range(n_assets):
        asset = ASSETS_DEFAULT[a] if a < len(ASSETS_DEFAULT) else f"ASSET_{a}"
        price = 1.0 + np.cumsum(np.random.randn(n_bars) * 0.02)
        for i, dt in enumerate(dates):
            feat = {
                "bar_open_ts": dt,
                "asset": asset,
                "evt_n_events_h24": int(np.random.randint(0, 10)),
                "evt_polarity_sum_h24": round(np.random.randn() * 0.8, 4),
                "evt_polarity_mean_h24": round(np.random.randn() * 0.5, 4),
                "evt_mag_weighted_polarity_h24": round(np.random.randn() * 0.5, 4),
                "evt_hack_flag_h24": int(np.random.random() < 0.1),
                "evt_regulation_flag_h24": int(np.random.random() < 0.05),
                "close": float(price[i]),
            }
            rows.append(feat)
    df = pd.DataFrame(rows)
    return df, dates

def _run_strategy(
    features: pd.DataFrame,
    signals: np.ndarray,
    strategy_name: str,
) -> dict:
    """Compute metrics for a given set of signals."""
    pos = SignalPipeline.position_from_signal(signals)
    returns = pos * features["next_return"]
    return {
        "strategy": strategy_name,
        "sharpe": sharpe(returns),
        "hit_rate": hit_rate(returns),
        "total_return": float((1 + returns).prod()),
        "max_dd": max_drawdown((1 + returns).cumprod()),
    }

def run_backtest(assets: list[str] = ASSETS_DEFAULT, start=None, end=None):
    """Run baseline comparisons. Skip if data missing."""
    start = pd.Timestamp("2021-01-01") if start is None else start
    end = pd.Timestamp("2024-01-01") if end is None else end

    windows = generate_windows(start, end)
    if not windows:
        print(f"No windows generated for [{start}, {end}]. Skipping.")
        return

    # Try real data first
    all_features = []
    all_ohlcv = []
    for train_s, train_e, test_s, test_e in windows:
        data = load_window_data(assets, test_s, test_e)
        if data is not None and not data.empty:
            all_features.append(data)

    if all_features:
        data = pd.concat(all_features)
    else:
        print("No real data found. Using synthetic data.")
        data, _ = _synthetic_data(n_bars=730, n_assets=len(assets))

    # Compute next-bar return
    if "next_return" not in data.columns:
        data["next_return"] = data.groupby("asset")["close"].pct_change().shift(-1)
    data = data.dropna(subset=["next_return"])

    # Feature columns (exclude metadata and target)
    feat_cols = [c for c in data.columns if c.startswith("evt_") or c.startswith("mag_")]
    if not feat_cols:
        feat_cols = [c for c in data.columns if c not in ("bar_open_ts", "asset", "close", "next_return")]

    results = []
    for asset in assets:
        sub = data[data["asset"] == asset]
        if sub.empty:
            continue
        X = sub[feat_cols]
        y = sub["next_return"]

        # 1. LLM-feature signal
        sp = SignalPipeline("ridge")
        sp.fit(X, y)
        pred = sp.predict(X)
        results.append(_run_strategy(sub, pred, f"{asset}_llm_features"))

        # 2. Momentum-only signal (1h/4h/24h returns as features — simulated)
        sub_mom = sub.copy()
        if "close" in sub_mom.columns:
            sub_mom["ret_1h"] = sub_mom["close"].pct_change(1)
            sub_mom["ret_4h"] = sub_mom["close"].pct_change(4)
            sub_mom["ret_24h"] = sub_mom["close"].pct_change(24)
            mom_cols = [c for c in ["ret_1h", "ret_4h", "ret_24h"] if c in sub_mom.columns]
            if mom_cols:
                sp_m = SignalPipeline("ridge")
                sp_m.fit(sub_mom[mom_cols], y)
                pred_m = sp_m.predict(sub_mom[mom_cols])
                results.append(_run_strategy(sub_mom, pred_m, f"{asset}_momentum"))

        # 3. Buy-and-hold (zero signal → position = 1)
        results.append(_run_strategy(sub, np.zeros(len(sub)), f"{asset}_buy_hold"))

    return results

if __name__ == "__main__":
    results = run_backtest()
    if results:
        print(f"\n{'Strategy':<30} {'Sharpe':>8} {'Hit%':>8} {'Ret':>10} {'MaxDD':>10}")
        print("-" * 70)
        for r in results:
            print(f"{r['strategy']:<30} {r['sharpe']:>8.3f} {r['hit_rate']*100:>7.1f}% {r['total_return']:>9.4f} {r['max_dd']:>9.4f}")
    else:
        print("No results. Data not available.")
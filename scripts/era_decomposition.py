#!/usr/bin/env python3
"""Where does the pooled "full period" IC come from: within regimes, or between them?

WHY THIS EXISTS

The IC grid reports `scope=ts_all, period=full` as its headline. `full` is the
UNION of the two sub-periods the grid also reports (gdelt_era, rss_era). After the
2026-10-18 refresh, phi4's composite_full rank IC at h=120 reads +0.082 on `full`
while the two parts that partition it read -0.024 and -0.075. A pooled
correlation outside the range of its own parts means the pooled number is not
measuring a relation inside either regime -- it is absorbing the fact that the
two regimes differ in BOTH mean feature value AND mean forward return:

    gdelt_era   mean feature -0.33   mean 120h return -1.22%
    rss_era     mean feature +0.65   mean 120h return +1.58%

Dense news and positive returns arrived together in the RSS era. Any feature that
is merely more active when news is dense will "predict" returns in a pooled
sample, with no within-regime relation at all.

WHAT THIS COMPUTES

For one (model, feature, horizon) cell, on the same decision grid ic_study uses:

  pooled       Pearson/Spearman over all rows, as the grid's `full` row does
  fixed_eff    each regime's rows standardised by that regime's own mean and sd
               (and, for the rank version, ranked within its own regime), then
               pooled. This is the within-regime association. If the pooled
               number is high and this one collapses to the regime-level values,
               the pooled number is a between-regime contrast.
  regime table mean feature, mean forward return, and per-regime correlation

Groups are configurable via --split: `eras` (gdelt / rss) or `refresh`
(gdelt / rss-before-refresh / rss-after-refresh), the latter separating "what the
new data says" from "what pooling does".

Read-only. Run from repo root, e.g.:
    .venv/bin/python scripts/era_decomposition.py --feature composite_full
    .venv/bin/python scripts/era_decomposition.py --feature depeg_flag --split refresh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import ic_study as ic
from src.config import BINANCE_SYMBOLS

REFRESH = pd.Timestamp("2026-07-28", tz="UTC")   # last date v1 could see
ERA_SPLIT = ic.ERA_SPLIT                          # RSS live-ingest start
ASSETS = [s.replace("USDT", "") for s in BINANCE_SYMBOLS]


def panel(feature: str, horizon: int, model: str):
    closes = ic.load_close_matrix()
    per_asset = ic.load_model_features(ic.MODELS[model])
    fwd = ic.forward_return_matrix(closes, horizon)
    feat = pd.DataFrame({a: df[feature] for a, df in per_asset.items()}).reindex(fwd.index)
    return feat, fwd


def labels(idx: pd.DatetimeIndex, split: str) -> pd.Series:
    lab = pd.Series("gdelt", index=idx)
    if split == "eras":
        lab[idx >= ERA_SPLIT] = "rss"
    else:
        lab[(idx >= ERA_SPLIT) & (idx < REFRESH)] = "rss_before"
        lab[idx >= REFRESH] = "rss_after"
    return lab


def _z(x: pd.Series) -> pd.Series:
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=x.index)
    return (x - x.mean()) / sd


def _corr(f: pd.Series, r: pd.Series, rank: bool) -> float:
    m = f.notna() & r.notna() & np.isfinite(f) & np.isfinite(r)
    if m.sum() < 30:
        return float("nan")
    a, b = f[m], r[m]
    if rank:
        a, b = a.rank(), b.rank()
    return float(np.corrcoef(a, b)[0, 1])


def cross_asset_mean(feat: pd.DataFrame, fwd: pd.DataFrame, lab: pd.Series,
                     mode: str, rank: bool) -> tuple[float, int]:
    """Mean across assets of the per-asset correlation, as scope=ts_all does."""
    vals = []
    n_min = None
    for a in ASSETS:
        if a not in feat.columns:
            continue
        f, r = feat[a], fwd[a]
        if mode == "pooled":
            v = _corr(f, r, rank)
            n = int((f.notna() & r.notna()).sum())
        else:
            parts_f, parts_r, n = [], [], 0
            for g, idx in lab.groupby(lab).groups.items():
                fg, rg = f.loc[idx], r.loc[idx]
                if rank:
                    # rank inside the regime: preserves the within-regime ordering
                    fg, rg = fg.rank(), rg.rank()
                parts_f.append(_z(fg))
                parts_r.append(_z(rg))
                n += int((fg.notna() & rg.notna()).sum())
            v = _corr(pd.concat(parts_f), pd.concat(parts_r), False)
        if np.isfinite(v):
            vals.append(v)
            n_min = n if n_min is None else min(n_min, n)
    return (float(np.mean(vals)) if vals else float("nan")), (n_min or 0)


def regime_table(feat: pd.DataFrame, fwd: pd.DataFrame, lab: pd.Series) -> None:
    print(f"{'regime':14s} {'n':>5s} {'mean_feat':>10s} {'mean_fwd_ret':>13s} "
          f"{'corr_pear':>10s} {'corr_spear':>11s}")
    for g in lab.unique():
        idx = lab.index[lab == g]
        fm = feat.loc[idx].mean(axis=1).mean()
        rm = fwd.loc[idx].mean(axis=1).mean()
        p, n = cross_asset_mean(feat.loc[idx], fwd.loc[idx],
                                pd.Series(g, index=idx), "pooled", False)
        s, _ = cross_asset_mean(feat.loc[idx], fwd.loc[idx],
                                pd.Series(g, index=idx), "pooled", True)
        print(f"{g:14s} {n:5d} {fm:+10.4f} {rm:+12.3%} {p:+10.4f} {s:+11.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=120)
    ap.add_argument("--feature", default="composite_full")
    ap.add_argument("--model", default="phi4-q6k-v1",
                    choices=["phi4-q6k-v1", "llama3-8b-q4km-v1"])
    ap.add_argument("--split", default="refresh", choices=["eras", "refresh"])
    args = ap.parse_args()

    feat, fwd = panel(args.feature, args.horizon, args.model)
    lab = labels(fwd.index, args.split)

    print(f"\n=== {args.model} | {args.feature} | 1h decisions | h={args.horizon} "
          f"| split={args.split} ===")
    regime_table(feat, fwd, lab)

    p_pool, n_pool = cross_asset_mean(feat, fwd, lab, "pooled", False)
    s_pool, _ = cross_asset_mean(feat, fwd, lab, "pooled", True)
    p_fx, n_fx = cross_asset_mean(feat, fwd, lab, "fixed_effects", False)
    s_fx, _ = cross_asset_mean(feat, fwd, lab, "fixed_effects", True)
    print(f"\n{'pooled (the grid headline)':30s} n={n_pool:5d}  "
          f"pearson={p_pool:+.4f}  spearman={s_pool:+.4f}")
    print(f"{'within-regime (fixed effects)':30s} n={n_fx:5d}  "
          f"pearson={p_fx:+.4f}  spearman={s_fx:+.4f}")


if __name__ == "__main__":
    main()

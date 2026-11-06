#!/usr/bin/env python3
"""Cross-sectional IC across the 5 study assets, with the two N=5 traps removed.

Ranking assets against each other at the same timestamp cancels any common
time-series term (the regime artifact of WHITEPAPER_v2 §4b) by construction.
Two other terms then dominate a naive cross-sectional IC on 5 names:

  1. Static asset tilt. BTC always carries more news than XRP; a full-sample
     z-score gives each asset a different constant. Either way the IC rewards
     "the asset that happened to outperform in this period". The grid's
     cs_pooled composite_full in the GDELT era (+0.19 at 120 h on 112 records)
     is this term alone.
     Fix: standardise each asset's feature against its OWN trailing history
     (PIT, 30 d window) before ranking.
  2. Beta. In an up market high-beta names rank top whatever the news.
     Fix: rank beta-residual forward returns r_i - beta_i * r_mkt, with r_mkt
     the equal-weight 5-asset return and beta_i from a trailing 30 d hourly
     regression known at T.

Feature variants
  raw          as the grid uses it
  own_z        (x - trailing mean) / trailing sd, per asset
  own_z_active own_z where the raw feature is non-zero, else 0: a zero (no news)
               carries no information, so it cannot re-introduce the tilt via
               -mean/sd, which differs by asset for sparse features
  tilt         the asset's trailing mean only: the placebo; what raw would score
               from asset identity alone

Per timestamp: Spearman IC across assets (ic_study.cross_sectional_ic, >= 4
valid) and the top-minus-bottom asset spread in bp. NW t, lag h-1. RSS era only:
the LLM has 112 records before it.

Run: .venv/bin/python scripts/cs_study.py   -> data/results/cs_ic.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import ic_study as ic

HORIZONS = [1, 4, 12, 24, 72, 120]
WINDOW = 720          # 30 d of hourly bars
MIN_PERIODS = 168
FEATURES = ["polarity_sum", "mag_weighted_polarity", "polarity_mean", "n_events",
            "hack_flag", "regulation_flag", "listing_flag", "depeg_flag", "composite_rare"]


def own_z(x: pd.DataFrame) -> pd.DataFrame:
    """Per-column trailing z-score; uses values <= T only."""
    roll = x.rolling(WINDOW, min_periods=MIN_PERIODS)
    return (x - roll.mean()) / roll.std(ddof=0).replace(0, np.nan)


def variants(x: pd.DataFrame) -> dict[str, pd.DataFrame]:
    z = own_z(x)
    return {
        "raw": x,
        "own_z": z,
        "own_z_active": z.where(x.fillna(0) != 0, 0.0).where(z.notna()),
        "tilt": x.rolling(WINDOW, min_periods=MIN_PERIODS).mean(),
    }


def trailing_beta(closes: pd.DataFrame) -> pd.DataFrame:
    """beta_i of hourly asset returns on the equal-weight market, known at T."""
    r = np.log(closes).diff()
    m = r.mean(axis=1)
    cov = r.rolling(WINDOW, min_periods=MIN_PERIODS).cov(m)
    var = m.rolling(WINDOW, min_periods=MIN_PERIODS).var()
    return cov.div(var, axis=0).shift(1)


def targets(closes: pd.DataFrame, h: int, beta: pd.DataFrame) -> dict[str, pd.DataFrame]:
    fwd = ic.forward_return_matrix(closes, h)
    mkt = fwd.mean(axis=1)
    return {"ret": fwd, "beta_resid": fwd - beta.mul(mkt, axis=0)}


def top_minus_bottom(x: pd.DataFrame, y: pd.DataFrame) -> pd.Series:
    """Per-timestamp return of the highest-feature asset minus the lowest; ties -> NaN."""
    valid = x.notna() & y.notna()
    x, y = x.where(valid), y.where(valid)
    ok = (valid.sum(axis=1) >= 4) & (x.max(axis=1) > x.min(axis=1))
    xi, yi = x[ok], y[ok]
    top = yi.to_numpy()[np.arange(len(xi)), np.nanargmax(xi.to_numpy(), axis=1)]
    bot = yi.to_numpy()[np.arange(len(xi)), np.nanargmin(xi.to_numpy(), axis=1)]
    return pd.Series(top - bot, index=xi.index)


def main() -> None:
    closes = ic.load_close_matrix()
    closes = closes[closes.index >= ic.ERA_SPLIT - pd.Timedelta(hours=WINDOW)]
    beta = trailing_beta(closes)
    rows = []
    for model, path in ic.MODELS.items():
        per = ic.load_model_features(path)
        raw = ic.load_features(path)
        for a, df in per.items():
            r = ic._utc_index(raw[raw["asset"] == a].copy())
            df["n_events"] = r[~r.index.duplicated(keep="last")]["n_events"].reindex(df.index)
        for f in FEATURES:
            x = pd.DataFrame({a: df[f] for a, df in per.items()}).reindex(closes.index)
            for vname, xv in variants(x).items():
                for h in HORIZONS:
                    for tname, y in targets(closes, h, beta).items():
                        xs, ys = xv[xv.index >= ic.ERA_SPLIT], y[y.index >= ic.ERA_SPLIT]
                        cs = ic.cross_sectional_ic(xs, ys, rank=True)
                        tmb = top_minus_bottom(xs, ys)
                        if len(cs) < ic.MIN_OBS:
                            continue
                        rows.append(dict(model=model, feature=f, variant=vname, horizon_h=h,
                                         target=tname, ic=cs.mean(),
                                         tstat_nw=ic.nw_tstat(cs.values, h - 1), n_obs=len(cs),
                                         tmb_bp=tmb.mean() * 1e4,
                                         tmb_t=ic.nw_tstat(tmb.values, h - 1), n_tmb=len(tmb)))
    res = pd.DataFrame(rows)
    out = ic.RESULTS_DIR / "cs_ic.csv"
    res.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    for v in ("tilt", "raw", "own_z_active"):
        sub = res[(res.variant == v) & (res.target == "beta_resid")]
        print(f"\n== {v} | beta_resid | Spearman IC (t) ==")
        print(sub.assign(cell=sub.ic.round(3).astype(str) + " (" + sub.tstat_nw.round(1).astype(str) + ")")
              .pivot_table(index=["model", "feature"], columns="horizon_h", values="cell", aggfunc="first")
              .to_string())
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()

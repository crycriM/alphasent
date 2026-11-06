#!/usr/bin/env python3
"""Do news features predict ABNORMAL volatility (not direction)?

Target, per asset, for decision hour T (features use news strictly before T):

    y_T = log( RV_fwd / S_fwd )
    RV_fwd = sum_{k=1..h} r_{T+k}^2        hourly log returns, same window as ic_study
    S_fwd  = sum_{k=1..h} s(how_{T+k})     seasonal expected variance

s(how) is the mean squared return of the asset in the same hour-of-week
(168 buckets = intra-week x intraday pattern), estimated on PRIOR weeks only
(expanding, PIT). y is "how much more volatile than a usual week at this time".

Volatility clusters and news follows price moves, so any news-count feature
predicts y through past volatility alone. The second target removes that:

    y_resid = y - OLS(y ~ hour-of-week FE + log dRV_24h + log dRV_168h)   (HAR-style, per asset x regime)

where dRV are trailing means of deseasonalised r^2 = r^2 / s, through bar T-1.
The test that matters is the IC on y_resid.

IC: Spearman, computed within each regime (gdelt / rss) and pooled as in
era_decomposition's fixed-effects estimator, averaged across the 5 assets per
timestamp, Newey-West t with lag h-1 (ic_study.nw_tstat).

Run: .venv/bin/python scripts/vol_study.py  -> data/results/vol_ic.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import ic_study as ic

HORIZONS = [1, 4, 24, 72]
EPS = 1e-12
LLM_FEATURES = ["n_events", "n_high_conf_events", "abs_polarity_sum", "polarity_std",
                "polarity_min", "polarity_sum", "hack_flag", "regulation_flag",
                "listing_flag", "depeg_flag", "composite_rare"]
TONE_FEATURES = ["abs_tone_sum_24h", "tone_mean_24h"]


def vol_frame(close: pd.Series, h: int, min_weeks: int = 4) -> pd.DataFrame:
    """Seasonal-adjusted forward log RV and trailing HAR controls for one asset."""
    close = close.asfreq("h")  # ponytail: gaps become NaN and drop out, no fill
    r2 = np.log(close).diff() ** 2
    how = r2.index.dayofweek * 24 + r2.index.hour
    # prior same-hour-of-week observations only
    seas = r2.groupby(how).transform(lambda s: s.shift().expanding(min_periods=min_weeks).mean())
    d = r2 / seas
    # ponytail: S_fwd uses the profile as estimated at T+k, not at T; the
    # profile moves by 1/n per week, so the lookahead is negligible past min_weeks
    rv_fwd = r2.rolling(h).sum().shift(-h)
    s_fwd = seas.rolling(h).sum().shift(-h)
    return pd.DataFrame({
        "y": np.log((rv_fwd + EPS) / s_fwd),
        "x24": np.log(d.rolling(24).mean().shift(1) + EPS),
        "x168": np.log(d.rolling(168).mean().shift(1) + EPS),
    })


def har_resid(v: pd.DataFrame) -> pd.Series:
    """OLS residual of y on [hour-of-week dummies, x24, x168].

    The dummies matter: log of an h-hour variance sum has a Jensen bias that
    depends on which hours the window covers, so y keeps hour-of-week structure
    even after dividing by S_fwd, and news counts follow the same weekly clock.
    """
    m = v.notna().all(axis=1)
    how = v.index[m].dayofweek * 24 + v.index[m].hour
    dummies = (how.to_numpy()[:, None] == np.unique(how)[None, :]).astype(float)
    X = np.column_stack([dummies, v.loc[m, "x24"], v.loc[m, "x168"]])
    beta, *_ = np.linalg.lstsq(X, v.loc[m, "y"].to_numpy(), rcond=None)
    return pd.Series(v.loc[m, "y"].to_numpy() - X @ beta, index=v.index[m]).reindex(v.index)


def regime(idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(np.where(idx >= ic.ERA_SPLIT, "rss", "gdelt"), index=idx)


def within_ic(feat: pd.Series, target: pd.Series, lab: pd.Series) -> pd.Series:
    """Per-timestamp Spearman products, standardised inside each regime."""
    # ponytail: a regime where the feature is almost always 0 (LLM in the GDELT era,
    # 112 records) gives a degenerate rank IC, so it is skipped
    parts = [ic.ic_products(feat[lab == g], target[lab == g], rank=True) for g in lab.unique()
             if (feat[lab == g].fillna(0) != 0).sum() >= ic.MIN_OBS * 10]
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)


def load_sources() -> dict[str, dict[str, pd.DataFrame]]:
    out = {}
    for name, path in ic.MODELS.items():
        per = ic.load_model_features(path)
        raw = ic.load_features(path)
        for a, df in per.items():
            r = ic._utc_index(raw[raw["asset"] == a].copy())
            r = r[~r.index.duplicated(keep="last")].sort_index()
            df["n_events"] = r["n_events"].reindex(df.index)
            df["n_high_conf_events"] = r["n_high_conf_events"].reindex(df.index)
            df["polarity_std"] = r["polarity_std"].reindex(df.index)
            df["polarity_min"] = r["polarity_min"].reindex(df.index)
            df["abs_polarity_sum"] = df["polarity_sum"].abs()
        out[name] = {a: df[LLM_FEATURES] for a, df in per.items()}
    tone = ic.load_tone_features()
    for df in tone.values():
        df["abs_tone_sum_24h"] = df["tone_sum_24h"].abs()
    out["gdelt_tone"] = {a: df[TONE_FEATURES] for a, df in tone.items()}
    return out


def main() -> None:
    closes = ic.load_close_matrix()
    sources = load_sources()
    rows = []
    for h in HORIZONS:
        targets = {}
        for a in closes.columns:
            v = vol_frame(closes[a], h)
            lab = regime(v.index)
            resid = pd.concat([har_resid(v[lab == g]) for g in lab.unique()]).sort_index()
            targets[a] = {"abn_vol": v["y"], "abn_vol_har": resid, "lab": lab}
        for model, per in sources.items():
            feats = next(iter(per.values())).columns
            for f in feats:
                for tgt in ("abn_vol", "abn_vol_har"):
                    for scope in ("within", "gdelt", "rss"):
                        prods = []
                        for a, df in per.items():
                            if a not in targets:
                                continue
                            t, lab = targets[a][tgt], targets[a]["lab"]
                            x = df[f].reindex(t.index)
                            if scope != "within":
                                keep = lab == scope
                                t, x, lab = t[keep], x[keep], lab[keep]
                            p = within_ic(x, t, lab)
                            if not p.empty:
                                prods.append(p.rename(a))
                        if not prods:
                            continue
                        allp = pd.concat(prods, axis=1).mean(axis=1).dropna()
                        if len(allp) < ic.MIN_OBS:
                            continue
                        rows.append(dict(model=model, feature=f, horizon_h=h, target=tgt,
                                         scope=scope, ic=allp.mean(),
                                         tstat_nw=ic.nw_tstat(allp.values, h - 1), n_obs=len(allp)))
    res = pd.DataFrame(rows)
    out = ic.RESULTS_DIR / "vol_ic.csv"
    res.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    piv = res[res.scope == "within"].pivot_table(
        index=["model", "feature"], columns=["target", "horizon_h"], values=["ic", "tstat_nw"])
    print(piv.round(3).to_string())
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()

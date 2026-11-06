#!/usr/bin/env python3
"""Minute-level event study: does an extracted news item move its asset in the
minutes after it appears, and in its polarity's direction?

Events: every cached LLM record (per model) on the 5 study assets in the RSS era,
joined by item_id to the RSS store for two clocks:
  pub   published_at   when the information exists (best case)
  ing   ingested_at    when this pipeline had it (median 9.5 min later; what a
                       live system could trade on)
t0 is the first full minute strictly after the clock time, so the entry price is
a minute close observed after the item.

Abnormal log return over (t0, t0+k]:
    AR = dlog P_asset - beta * dlog P_mkt
P_mkt is the equal-weight log price of the OTHER 4 assets; beta the trailing 30 d hourly
beta known at t0 (cs_study.trailing_beta). k < 0 gives the drift BEFORE t0.

Measures, per (model, clock, subset, k):
  signed_bp   mean of sign(polarity) * AR over directional events (|polarity| >= 0.3)
  pos_bp/neg_bp  mean AR on positive / negative events
  absexc_bp   mean |AR| minus seasonal placebo |AR|: same asset, same minute of
              week, 1..4 weeks earlier (intra-week x intraday pattern)
  rawabsexc_bp  the same on the unadjusted return: BTC and macro news move the
              whole market, which the market adjustment would cancel
t-stats cluster by UTC day (mean per day, then t over days): items on the same
story arrive in bursts, so events are not independent.

Run: .venv/bin/python scripts/short_horizon.py -> data/results/short_horizon.csv
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import cs_study as cs
import ic_study as ic
from src.config import BINANCE_SYMBOLS, RAW_OHLCV_DIR
from src.extraction.cache import load_all_cached

KS = [-60, -15, -5, 1, 5, 15, 30, 60, 240]
PLACEBO_WEEKS = [1, 2, 3, 4]
POL_MIN = 0.3
MIN_DAYS = 10


def load_log_prices() -> pd.DataFrame:
    """Minute log close, indexed by the minute the close is observed (open_time + 1 min)."""
    cols = {}
    for sym in BINANCE_SYMBOLS:
        df = pd.read_parquet(RAW_OHLCV_DIR / f"{sym}_1m.parquet", columns=["open_time", "close"])
        s = df.set_index(pd.to_datetime(df["open_time"], utc=True))["close"]
        cols[sym.replace("USDT", "")] = s[~s.index.duplicated()]
    lp = np.log(pd.DataFrame(cols))
    lp.index = lp.index + pd.Timedelta(minutes=1)
    # ponytail: missing minutes forward-filled, a stale price reads as a zero return
    return lp.asfreq("min").ffill()


def abnormal(lp: pd.DataFrame, beta_at: pd.DataFrame, asset: np.ndarray,
             t0: pd.DatetimeIndex, k: int) -> np.ndarray:
    """AR over (t0, t0+k] (k>0) or (t0+k, t0] (k<0) for each event."""
    a, b = (t0, t0 + pd.Timedelta(minutes=k)) if k > 0 else (t0 + pd.Timedelta(minutes=k), t0)
    ia, ib = lp.index.get_indexer(a), lp.index.get_indexer(b)
    ok = (ia >= 0) & (ib >= 0)
    col = lp.columns.get_indexer(asset)
    vals = lp.to_numpy()
    tot, n = vals.sum(axis=1), vals.shape[1]
    out = np.full(len(t0), np.nan)
    d_asset = vals[ib[ok], col[ok]] - vals[ia[ok], col[ok]]
    # leave-one-out market: the asset's own move must not leak into its benchmark
    d_mkt = ((tot[ib[ok]] - tot[ia[ok]]) - d_asset) / (n - 1)
    out[ok] = d_asset - beta_at[ok] * d_mkt
    return out


def day_t(x: pd.Series) -> tuple[float, float, int]:
    """Day-weighted mean, day-clustered t, number of days."""
    x = x.dropna()
    daily = x.groupby(x.index.floor("D")).mean()
    if len(daily) < MIN_DAYS:
        return float(daily.mean()) if len(daily) else np.nan, np.nan, len(daily)
    # day-weighted mean: a burst of 40 items on one story counts once, like its t
    return float(daily.mean()), float(daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily)))), len(daily)


def load_events(model: str) -> pd.DataFrame:
    rec = load_all_cached(model)
    rec = rec[rec["asset"].isin([s.replace("USDT", "") for s in BINANCE_SYMBOLS])]
    rss = pd.concat([pd.read_parquet(f, columns=["item_id", "published_at", "ingested_at"])
                     for f in glob.glob("data/crypto_rss/normalized/*.parquet")])
    rss = rss.drop_duplicates("item_id")
    ev = rec.drop(columns=["published_at"]).merge(rss, on="item_id", how="inner")
    ev = ev[ev["published_at"] >= ic.ERA_SPLIT]
    ev = ev[ev["ingested_at"] >= ev["published_at"] - pd.Timedelta(minutes=1)]  # bad clocks
    return ev.drop_duplicates(["item_id", "asset"]).reset_index(drop=True)


def subsets(ev: pd.DataFrame) -> dict[str, pd.Series]:
    s = {"all": pd.Series(True, index=ev.index),
         "high_conf": ev["confidence"] >= 0.8,
         "high_mag": ev["magnitude"] >= 0.6}
    for t in ("hack", "regulation", "listing", "depeg"):
        s[t] = ev["event_type"] == t
    return s


def study(ev: pd.DataFrame, lp: pd.DataFrame, beta_h: pd.DataFrame, model: str) -> list[dict]:
    rows = []
    for clock in ("pub", "ing"):
        t0 = pd.DatetimeIndex(ev["published_at" if clock == "pub" else "ingested_at"]).ceil("min")
        t0 = t0 + pd.Timedelta(minutes=1) * (t0 == pd.DatetimeIndex(
            ev["published_at" if clock == "pub" else "ingested_at"]))  # strictly after
        asset = ev["asset"].to_numpy()
        bidx = beta_h.index.get_indexer(t0.floor("h"), method="ffill")
        bcol = beta_h.columns.get_indexer(asset)
        beta_at = np.where(bidx >= 0, beta_h.to_numpy()[np.clip(bidx, 0, None), bcol], np.nan)
        beta_at = np.nan_to_num(beta_at, nan=1.0)
        pol = ev["polarity"].to_numpy()
        sign = np.where(np.abs(pol) >= POL_MIN, np.sign(pol), 0.0)
        for k in KS:
            ar = abnormal(lp, beta_at, asset, t0, k)
            plac = np.nanmean([np.abs(abnormal(lp, beta_at, asset, t0 - pd.Timedelta(weeks=w), k))
                               for w in PLACEBO_WEEKS], axis=0)
            raw = abnormal(lp, np.zeros(len(t0)), asset, t0, k)  # beta 0: no market adjustment
            plac_raw = np.nanmean([np.abs(abnormal(lp, np.zeros(len(t0)), asset,
                                                   t0 - pd.Timedelta(weeks=w), k))
                                   for w in PLACEBO_WEEKS], axis=0)
            AR = pd.Series(ar, index=t0)
            for sname, m in subsets(ev).items():
                m = m.to_numpy()
                if m.sum() < 30:
                    continue
                dirm = m & (sign != 0)
                sb, st, nd = day_t((AR * sign)[dirm] * 1e4)
                pb, pt, _ = day_t(AR[m & (sign > 0)] * 1e4)
                nb, nt, _ = day_t(AR[m & (sign < 0)] * 1e4)
                ab, at, _ = day_t(pd.Series(np.abs(ar) - plac, index=t0)[m] * 1e4)
                rb, rt, _ = day_t(pd.Series(np.abs(raw) - plac_raw, index=t0)[m] * 1e4)
                rows.append(dict(model=model, clock=clock, subset=sname, k_min=k,
                                 n=int(m.sum()), n_dir=int(dirm.sum()), days=nd,
                                 signed_bp=sb, signed_t=st, pos_bp=pb, pos_t=pt,
                                 neg_bp=nb, neg_t=nt, absexc_bp=ab, absexc_t=at,
                                 rawabsexc_bp=rb, rawabsexc_t=rt))
    return rows


def main() -> None:
    lp = load_log_prices()
    beta_h = cs.trailing_beta(ic.load_close_matrix())
    rows = []
    for model in ic.MODELS:
        ev = load_events(model)
        lat = (ev["ingested_at"] - ev["published_at"]).dt.total_seconds() / 60
        print(f"{model}: {len(ev)} events, ingest latency median {lat.median():.1f} min")
        rows += study(ev, lp, beta_h, model)
    res = pd.DataFrame(rows)
    out = ic.RESULTS_DIR / "short_horizon.csv"
    res.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    show = res[res.clock == "pub"]
    for col in ("signed", "rawabsexc"):
        print(f"\n== {col}, clock=pub: day-weighted mean bp (day-clustered t) ==")
        cell = show[f"{col}_bp"].round(2).astype(str) + " (" + show[f"{col}_t"].round(1).astype(str) + ")"
        print(show.assign(cell=cell).pivot_table(index=["model", "subset"], columns="k_min",
                                                 values="cell", aggfunc="first").to_string())
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()

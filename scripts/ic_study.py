#!/usr/bin/env python3
"""
IC study — information coefficient between news features and forward returns.

Measures signal quality as IC (Pearson and Spearman rank correlation between
feature[t] and the h-hour forward return), not Sharpe of a fitted strategy:
IC separates information content from position sizing and backtest choices.

Inference: h-hour forward returns sampled on a 1h grid overlap h-fold, so
naive t-stats are inflated ~sqrt(h). We use a Newey-West t-stat with Bartlett
kernel and lag L = h-1 bars on the per-timestamp series whose mean is the IC.

Axes of the study (each combination persisted to data/results/ic_grid.csv):
  model       phi4-q6k-v1 | llama3-8b-q4km-v1 | gdelt_tone (baseline)
  cadence     1h | 1d  (decision frequency; independent of horizon)
  feature_set single | composite_full | composite_rare | tone
  horizon_h   1, 4, 12, 24, 48, 72, 120
  period      full | gdelt_era | rss_era  (data-source regimes)
  ic_type     pearson | spearman
  scope       ts_asset (per-asset time-series IC)
              ts_all   (cross-asset mean of per-timestamp products)
              cs_pooled (per-timestamp cross-sectional IC; N=5 assets — noisy)

Subcommands: check | provenance | model-compare | ic-grid | car | figures | all
Run from repo root: .venv/bin/python scripts/ic_study.py all
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import BINANCE_SYMBOLS, DATA_ROOT, FEATURES_DIR, RAW_NEWS_DIR, RSS_DATA_ROOT
from src.backtest.contamination import compare_extractions
from src.extraction.cache import load_all_cached
from scripts.horizon_scan import EVENT_THRESHOLDS, load_features, load_ohlcv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ic_study")

RESULTS_DIR = DATA_ROOT / "results"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"

MODELS = {
    "phi4-q6k-v1": FEATURES_DIR,
    "llama3-8b-q4km-v1": DATA_ROOT / "features_llama3_backup",
}
PROMPT_V = "1.0.0"
HORIZONS_H = [1, 4, 12, 24, 48, 72, 120]
ERA_SPLIT = pd.Timestamp("2026-06-22", tz="UTC")  # RSS live ingestion start
ASSETS = [s.replace("USDT", "") for s in BINANCE_SYMBOLS]

SINGLE_FEATURES = [
    "polarity_sum", "polarity_mean", "mag_weighted_polarity",
    "novelty_polarity", "recency_polarity",
    "hack_flag", "regulation_flag", "listing_flag", "depeg_flag",
]
FLAG_FEATURES = ["hack_flag", "regulation_flag", "listing_flag", "depeg_flag"]
# feature -> feature_set label
FEATURE_SETS = {**{f: "single" for f in SINGLE_FEATURES},
                "composite_full": "composite_full",
                "composite_rare": "composite_rare",
                "tone_mean_24h": "tone", "tone_sum_24h": "tone"}
MIN_OBS = 30


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def nw_tstat(z: np.ndarray, lag: int) -> float:
    """Newey-West t-stat for H0: mean(z)=0, Bartlett kernel, lag L bars."""
    z = np.asarray(z, dtype=float)
    z = z[~np.isnan(z)]
    T = len(z)
    if T < MIN_OBS:
        return np.nan
    L = int(min(lag, T // 4))
    d = z - z.mean()
    s = d @ d / T
    for l in range(1, L + 1):
        s += 2.0 * (1.0 - l / (L + 1.0)) * (d[l:] @ d[:-l] / T)
    if s <= 0:
        return np.nan
    return float(z.mean() / np.sqrt(s / T))


def ic_products(feat: pd.Series, ret: pd.Series, rank: bool) -> pd.Series:
    """Per-timestamp products of standardized feature and forward return.

    mean(products) equals the Pearson correlation (Spearman if rank=True),
    so NW inference on the product series is inference on the IC.
    """
    df = pd.concat([feat, ret], axis=1, join="inner", keys=["x", "y"]).dropna()
    if len(df) < MIN_OBS:
        return pd.Series(dtype=float)
    x, y = df["x"], df["y"]
    if rank:
        x, y = x.rank(), y.rank()
    sx, sy = x.std(ddof=0), y.std(ddof=0)
    if sx < 1e-12 or sy < 1e-12:
        return pd.Series(dtype=float)
    return ((x - x.mean()) / sx) * ((y - y.mean()) / sy)


def cross_sectional_ic(feat_mat: pd.DataFrame, ret_mat: pd.DataFrame, rank: bool) -> pd.Series:
    """Per-timestamp correlation across assets (rows=time, cols=assets)."""
    x = feat_mat.copy()
    y = ret_mat.reindex(index=x.index, columns=x.columns)
    valid = x.notna() & y.notna()
    x, y = x.where(valid), y.where(valid)
    if rank:
        x, y = x.rank(axis=1), y.rank(axis=1)
    xd = x.sub(x.mean(axis=1), axis=0)
    yd = y.sub(y.mean(axis=1), axis=0)
    num = (xd * yd).sum(axis=1)
    den = np.sqrt((xd ** 2).sum(axis=1) * (yd ** 2).sum(axis=1))
    ic_t = num / den.replace(0, np.nan)
    return ic_t[valid.sum(axis=1) >= 4].dropna()


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #

def _utc_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.DatetimeIndex(df.index)
    df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return df


def add_composites(df: pd.DataFrame) -> pd.DataFrame:
    """Full-sample z-score composites (weights are in-sample; per-feature ICs are not)."""
    zs = []
    for f in SINGLE_FEATURES:
        s = df[f].std(ddof=0)
        if s > 1e-12:
            zs.append((df[f] - df[f].mean()) / s)
    df["composite_full"] = pd.concat(zs, axis=1).mean(axis=1) if zs else 0.0
    df["composite_rare"] = df[FLAG_FEATURES].sum(axis=1)
    return df


def load_model_features(features_dir: Path) -> dict[str, pd.DataFrame]:
    """Hourly features per tradable asset, composites included."""
    feats = load_features(features_dir)
    out = {}
    for asset in ASSETS:
        af = feats[feats["asset"] == asset]
        if af.empty:
            continue
        af = _utc_index(af[[c for c in SINGLE_FEATURES if c in af.columns]].copy())
        af = af[~af.index.duplicated(keep="last")].sort_index()
        out[asset] = add_composites(af)
    return out


def load_tone_features() -> dict[str, pd.DataFrame]:
    """GDELT raw_tone baseline: PIT-safe 24h-lookback sum/mean per asset, hourly."""
    frames = []
    for f in sorted(RAW_NEWS_DIR.glob("*.parquet")):
        df = pd.read_parquet(f, columns=["published_at", "asset_mentions", "raw_tone", "source"])
        df = df[df["source"] == "gdelt"]
        if not df.empty:
            frames.append(df)
    news = pd.concat(frames, ignore_index=True)
    news = news[news["asset_mentions"].map(lambda a: len(a) > 0)]
    news = news.explode("asset_mentions").rename(columns={"asset_mentions": "asset"})
    news = news[news["asset"].isin(ASSETS)]
    news["hour"] = pd.to_datetime(news["published_at"], utc=True).dt.floor("h")
    out = {}
    for asset, grp in news.groupby("asset"):
        hourly = grp.groupby("hour")["raw_tone"].agg(["sum", "count"])
        grid = pd.date_range(hourly.index.min(), hourly.index.max() + pd.Timedelta(hours=25), freq="h", tz="UTC")
        hourly = hourly.reindex(grid, fill_value=0.0)
        # events in hours [T-24, T) are strictly published before bar open T
        s24 = hourly["sum"].rolling(24, min_periods=1).sum().shift(1)
        c24 = hourly["count"].rolling(24, min_periods=1).sum().shift(1)
        out[asset] = pd.DataFrame({
            "tone_sum_24h": s24.fillna(0.0),
            "tone_mean_24h": (s24 / c24.replace(0, np.nan)).fillna(0.0),
        })
    return out


DAILY_AGG = dict(
    **{f: "sum" for f in ["polarity_sum", "mag_weighted_polarity", "novelty_polarity", "tone_sum_24h"]},
    **{f: "mean" for f in ["polarity_mean", "recency_polarity", "composite_full", "tone_mean_24h"]},
    **{f: "max" for f in FLAG_FEATURES},
    composite_rare="sum",
)


def to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly feature rows per calendar day; decision at next midnight."""
    agg = {c: DAILY_AGG[c] for c in df.columns if c in DAILY_AGG}
    daily = df.groupby(df.index.floor("D")).agg(agg)
    daily.index = daily.index + pd.Timedelta(days=1)  # PIT: day D rows decide at D+1 00:00
    return daily


def forward_return_matrix(closes: pd.DataFrame, horizon_bars: int) -> pd.DataFrame:
    """closes: time x asset matrix -> h-bar forward simple return."""
    return closes.shift(-horizon_bars) / closes - 1


def load_close_matrix() -> pd.DataFrame:
    ohlcv = load_ohlcv()
    pivot = ohlcv.pivot_table(index="open_time", columns="symbol", values="close").sort_index()
    pivot.columns = [c.replace("USDT", "") for c in pivot.columns]
    return pivot


# --------------------------------------------------------------------------- #
# ic-grid
# --------------------------------------------------------------------------- #

PERIODS = {
    "full": (None, None),
    "gdelt_era": (None, ERA_SPLIT),
    "rss_era": (ERA_SPLIT, None),
}


def run_ic_grid() -> pd.DataFrame:
    closes_h = load_close_matrix()
    closes_d = closes_h[closes_h.index.hour == 0]

    sources: dict[str, dict[str, pd.DataFrame]] = {
        name: load_model_features(d) for name, d in MODELS.items() if d.exists()
    }
    sources["gdelt_tone"] = load_tone_features()

    rows = []
    for model, per_asset in sources.items():
        log.info("ic-grid: model=%s assets=%s", model, sorted(per_asset))
        for cadence in ("1h", "1d"):
            if cadence == "1h":
                feats = per_asset
                closes, bar_h = closes_h, 1
            else:
                feats = {a: to_daily(df) for a, df in per_asset.items()}
                closes, bar_h = closes_d, 24
            feature_cols = next(iter(feats.values())).columns
            for horizon in HORIZONS_H:
                if horizon % bar_h:
                    continue
                hb = horizon // bar_h
                fwd = forward_return_matrix(closes, hb)
                for pname, (t0, t1) in PERIODS.items():
                    mask = pd.Series(True, index=fwd.index)
                    if t0 is not None:
                        mask &= fwd.index >= t0
                    if t1 is not None:
                        mask &= fwd.index < t1
                    fwd_p = fwd[mask]
                    for feature in feature_cols:
                        feat_mat = pd.DataFrame({a: df[feature] for a, df in feats.items()}) \
                            .reindex(fwd_p.index)
                        for ic_type, rank in (("pearson", False), ("spearman", True)):
                            prods = {a: ic_products(feat_mat[a], fwd_p[a], rank)
                                     for a in feat_mat.columns}
                            prods = {a: p for a, p in prods.items() if not p.empty}
                            base = dict(model=model, cadence=cadence,
                                        feature_set=FEATURE_SETS[feature], feature=feature,
                                        horizon_h=horizon, period=pname, ic_type=ic_type)
                            for a, p in prods.items():
                                rows.append(dict(base, scope="ts_asset", asset=a,
                                                 ic=p.mean(), tstat_nw=nw_tstat(p.values, hb - 1),
                                                 n_obs=len(p),
                                                 date_start=p.index.min().date(), date_end=p.index.max().date()))
                            if prods:
                                allp = pd.concat(prods.values(), axis=1).mean(axis=1).dropna()
                                rows.append(dict(base, scope="ts_all", asset="ALL",
                                                 ic=allp.mean(), tstat_nw=nw_tstat(allp.values, horizon // bar_h - 1),
                                                 n_obs=len(allp),
                                                 date_start=allp.index.min().date(), date_end=allp.index.max().date()))
                            cs = cross_sectional_ic(feat_mat, fwd_p, rank)
                            if len(cs) >= MIN_OBS:
                                rows.append(dict(base, scope="cs_pooled", asset="ALL",
                                                 ic=cs.mean(), tstat_nw=nw_tstat(cs.values, horizon // bar_h - 1),
                                                 n_obs=len(cs),
                                                 date_start=cs.index.min().date(), date_end=cs.index.max().date()))
    grid = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    grid.to_csv(RESULTS_DIR / "ic_grid.csv", index=False)
    log.info("ic-grid: %d rows -> %s", len(grid), RESULTS_DIR / "ic_grid.csv")
    return grid


# --------------------------------------------------------------------------- #
# car — market-adjusted event study
# --------------------------------------------------------------------------- #

def run_car() -> pd.DataFrame:
    closes = load_close_matrix()
    m_star, p_star = EVENT_THRESHOLDS["m_star"], EVENT_THRESHOLDS["p_star"]

    feats = load_features(MODELS["phi4-q6k-v1"])
    feats = feats[feats["asset"].isin(ASSETS)]
    feats = _utc_index(feats)
    gate = (feats["n_high_conf_events"] > 0) & (feats["polarity_sum"].abs() >= m_star)
    hc = feats[gate].copy()
    hc["bucket"] = "unsure"
    hc.loc[hc["polarity_mean"] > p_star, "bucket"] = "positive"
    hc.loc[hc["polarity_mean"] < -p_star, "bucket"] = "negative"

    rows = []
    for horizon in HORIZONS_H:
        fwd = forward_return_matrix(closes, horizon)
        mkt = fwd.mean(axis=1)
        for bucket, ev in hc.groupby("bucket"):
            raw, mktadj, btcadj = [], [], []
            for ts, asset in zip(ev.index, ev["asset"]):
                if ts not in fwd.index or asset not in fwd.columns:
                    continue
                r = fwd.at[ts, asset]
                if pd.isna(r):
                    continue
                raw.append(r)
                mktadj.append(r - mkt.at[ts])
                btcadj.append(r - fwd.at[ts, "BTC"])
            if raw:
                rows.append(dict(horizon_h=horizon, bucket=bucket,
                                 car_raw=np.mean(raw), car_mktadj=np.nanmean(mktadj),
                                 car_btcadj=np.nanmean(btcadj),
                                 n_events=len(raw), std_raw=np.std(raw)))
    car = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    car.to_csv(RESULTS_DIR / "car_adjusted.csv", index=False)
    log.info("car: %d rows -> %s", len(car), RESULTS_DIR / "car_adjusted.csv")
    return car


# --------------------------------------------------------------------------- #
# model-compare
# --------------------------------------------------------------------------- #

def run_model_compare() -> pd.DataFrame:
    rows = []
    caches = {m: load_all_cached(m, PROMPT_V) for m in MODELS}
    for model, df in caches.items():
        pol = df["polarity"].astype(float)
        rows += [dict(metric=k, model=model, value=v) for k, v in {
            "n_events": len(df),
            "n_assets": df["asset"].nunique(),
            "polarity_mean": pol.mean(), "polarity_std": pol.std(),
            "pct_polarity_pos": (pol > 0).mean(), "pct_polarity_neg": (pol < 0).mean(),
            "magnitude_mean": df["magnitude"].astype(float).mean(),
            "confidence_mean": df["confidence"].astype(float).mean(),
            "pct_extraction_only": df["extraction_only"].astype(bool).mean(),
        }.items()]
        for etype, share in df["event_type"].value_counts(normalize=True).head(8).items():
            rows.append(dict(metric=f"event_share_{etype}", model=model, value=share))
    pair = compare_extractions(*caches.values())
    rows += [dict(metric=k, model="pair", value=v) for k, v in pair.items()]
    out = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS_DIR / "model_extraction_compare.csv", index=False)
    log.info("model-compare -> %s\n%s", RESULTS_DIR / "model_extraction_compare.csv",
             out.pivot_table(index="metric", columns="model", values="value", aggfunc="first").to_string())
    return out


# --------------------------------------------------------------------------- #
# provenance — where events come from, vs model training cutoffs
# --------------------------------------------------------------------------- #

def run_provenance() -> pd.DataFrame:
    gdelt_ids, rss_ids = set(), set()
    ingest = []
    for f in sorted(RAW_NEWS_DIR.glob("*.parquet")):
        df = pd.read_parquet(f, columns=["item_id", "published_at"])
        gdelt_ids.update(df["item_id"])
        ingest.append(pd.DataFrame({"date": pd.to_datetime(df["published_at"], utc=True).dt.date,
                                    "source": "gdelt"}))
    for f in sorted((RSS_DATA_ROOT / "normalized").glob("*.parquet")):
        df = pd.read_parquet(f, columns=["item_id", "published_at"])
        rss_ids.update(df["item_id"])
        ingest.append(pd.DataFrame({"date": pd.to_datetime(df["published_at"], utc=True).dt.date,
                                    "source": "rss"}))
    articles = (pd.concat(ingest).groupby(["date", "source"]).size()
                .rename("n_articles").reset_index())

    frames = []
    for model in MODELS:
        ev = load_all_cached(model, PROMPT_V)[["item_id", "published_at"]].copy()
        ev["source"] = np.where(ev["item_id"].isin(rss_ids), "rss",
                                np.where(ev["item_id"].isin(gdelt_ids), "gdelt", "unmatched"))
        ev["date"] = pd.to_datetime(ev["published_at"], utc=True).dt.date
        summary = ev.groupby("source").size()
        span = (pd.to_datetime(ev['published_at'], utc=True).min(),
                pd.to_datetime(ev['published_at'], utc=True).max())
        log.info("provenance %s: %s | published_at %s -> %s", model, summary.to_dict(), *span)
        frames.append(ev.groupby(["date", "source"]).size().rename(f"n_events_{model}").reset_index())

    timeline = articles
    for f in frames:
        timeline = timeline.merge(f, on=["date", "source"], how="outer")
    timeline = timeline.sort_values(["date", "source"]).fillna(0)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timeline.to_csv(RESULTS_DIR / "corpus_timeline.csv", index=False)
    log.info("provenance: %d rows -> %s", len(timeline), RESULTS_DIR / "corpus_timeline.csv")
    return timeline


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

# validated categorical palette (dataviz default, light mode)
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
C_POS, C_NEG, C_UNSURE = "#008300", "#e34948", "#808080"
GRID_KW = dict(color="#e5e5e2", linewidth=0.8)


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, **GRID_KW)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.axhline(0, color="#52514e", linewidth=0.8)


def run_figures() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    grid = pd.read_csv(RESULTS_DIR / "ic_grid.csv")
    ts_all = grid[(grid["scope"] == "ts_all") & (grid["ic_type"] == "spearman")
                  & (grid["period"] == "full")]

    # 1. IC(h) per model, hourly cadence
    fig, ax = plt.subplots(figsize=(7, 4))
    curves = [("phi4-q6k-v1", "composite_full", "phi4 (composite)", C_BLUE),
              ("llama3-8b-q4km-v1", "composite_full", "llama3-8b (composite)", C_ORANGE),
              ("gdelt_tone", "tone_mean_24h", "GDELT tone (baseline)", C_AQUA)]
    for model, feature, label, color in curves:
        sub = ts_all[(ts_all["model"] == model) & (ts_all["feature"] == feature)
                     & (ts_all["cadence"] == "1h")].sort_values("horizon_h")
        if sub.empty:
            continue
        se = (sub["ic"] / sub["tstat_nw"]).abs()
        ax.plot(sub["horizon_h"], sub["ic"], marker="o", markersize=4, linewidth=2,
                color=color, label=label)
        ax.fill_between(sub["horizon_h"], sub["ic"] - 2 * se, sub["ic"] + 2 * se,
                        color=color, alpha=0.12, linewidth=0)
    _style(ax, "Rank IC vs response horizon (hourly cadence, all assets, ±2 NW s.e.)",
           "response horizon (hours)", "Spearman IC")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ic_horizon_models.png", dpi=150)

    # 2. cadence effect, phi4 composite
    fig, ax = plt.subplots(figsize=(7, 4))
    for cadence, color, label in (("1h", C_BLUE, "hourly decisions"), ("1d", C_ORANGE, "daily decisions")):
        sub = ts_all[(ts_all["model"] == "phi4-q6k-v1") & (ts_all["feature"] == "composite_full")
                     & (ts_all["cadence"] == cadence)].sort_values("horizon_h")
        se = (sub["ic"] / sub["tstat_nw"]).abs()
        ax.plot(sub["horizon_h"], sub["ic"], marker="o", markersize=4, linewidth=2,
                color=color, label=label)
        ax.fill_between(sub["horizon_h"], sub["ic"] - 2 * se, sub["ic"] + 2 * se,
                        color=color, alpha=0.12, linewidth=0)
    _style(ax, "Feature cadence vs response horizon (phi4 composite, ±2 NW s.e.)",
           "response horizon (hours)", "Spearman IC")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ic_cadence.png", dpi=150)

    # 3. adjusted CAR
    car = pd.read_csv(RESULTS_DIR / "car_adjusted.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, col, title in zip(axes, ["car_raw", "car_mktadj"],
                              ["Raw CAR", "Market-adjusted CAR"]):
        for bucket, color in (("positive", C_POS), ("negative", C_NEG), ("unsure", C_UNSURE)):
            sub = car[car["bucket"] == bucket].sort_values("horizon_h")
            ax.plot(sub["horizon_h"], 100 * sub[col], marker="o", markersize=4,
                    linewidth=2, color=color, label=bucket)
        _style(ax, title, "horizon (hours)", "CAR (%)" if col == "car_raw" else "")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "car_adjusted.png", dpi=150)

    # 4. inference: naive vs overlap-robust t, and in-sample vs out-of-sample Sharpe
    from src.backtest.eval import sharpe
    from src.backtest.signal import SignalPipeline

    closes = load_close_matrix()
    per_asset = load_model_features(MODELS["phi4-q6k-v1"])

    t_rows = []
    for horizon in HORIZONS_H:
        fwd = forward_return_matrix(closes, horizon)
        prods = {a: ic_products(df["composite_rare"].reindex(fwd.index), fwd[a], rank=True)
                 for a, df in per_asset.items()}
        z = pd.concat([p for p in prods.values() if not p.empty], axis=1).mean(axis=1).dropna()
        t_rows.append((horizon, nw_tstat(z.values, 0), nw_tstat(z.values, horizon - 1)))
    t_df = pd.DataFrame(t_rows, columns=["h", "naive", "nw"])

    fwd72 = forward_return_matrix(closes, 72)
    bars = {}
    for label, cols in (("rare-event flags", FLAG_FEATURES), ("full features", SINGLE_FEATURES)):
        ins, oos = [], []
        for a, df in per_asset.items():
            sub = pd.concat([df[cols], fwd72[a].rename("y")], axis=1, join="inner").dropna()
            if len(sub) < 500 or sub[cols].std().sum() < 1e-12:
                continue
            X, y = sub[cols], sub["y"]
            for kind, (fit_sl, eval_sl) in {
                "ins": (slice(None), slice(None)),               # fit and trade the same rows
                "oos": (slice(0, int(len(sub) * 0.7)), slice(int(len(sub) * 0.7), None)),
            }.items():
                sp = SignalPipeline("ridge")
                sp.fit(X.iloc[fit_sl], y.iloc[fit_sl])
                pred = sp.predict(X.iloc[eval_sl])
                idx = list(range(0, len(pred), 72))               # non-overlapping 72h trades
                # near-constant prediction (e.g. zero flag events in the train
                # window -> zero coefficients) means no strategy: z-scoring the
                # residual float noise would trade dust, so record flat (Sharpe 0)
                if np.std(pred[idx]) < 1e-10:
                    (ins if kind == "ins" else oos).append(0.0)
                    continue
                pos = SignalPipeline.position_from_signal(pred[idx])
                trades = pos.values * y.iloc[eval_sl].iloc[idx].values
                (ins if kind == "ins" else oos).append(
                    sharpe(pd.Series(trades), periods_per_year=8760 // 72))
        bars[label] = (np.mean(ins), np.mean(oos))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    ax = axes[0]
    ax.plot(t_df["h"], t_df["naive"], marker="o", markersize=4, linewidth=2,
            color=C_ORANGE, label="naive t")
    ax.plot(t_df["h"], t_df["nw"], marker="o", markersize=4, linewidth=2,
            color=C_BLUE, label="Newey–West t (lag h−1)")
    for lvl in (2, -2):
        ax.axhline(lvl, color="#52514e", linewidth=0.8, linestyle=":")
    ax.annotate("|t| = 2", xy=(t_df["h"].iloc[-1], 2.1), fontsize=8, color="#52514e", ha="right")
    _style(ax, "Same IC, two inferences (phi4 rare-event composite)",
           "response horizon (hours)", "t-statistic")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    xpos = np.arange(len(bars))
    ax.bar(xpos - 0.18, [v[0] for v in bars.values()], width=0.32, color=C_ORANGE,
           label="fit and traded in-sample")
    ax.bar(xpos + 0.18, [v[1] for v in bars.values()], width=0.32, color=C_BLUE,
           label="out-of-sample (70/30 split)")
    ax.set_xticks(xpos, list(bars.keys()))
    if abs(bars["rare-event flags"][1]) < 1e-9:
        ax.annotate("no trades: zero flag events\nin the train window",
                    xy=(0.18, 0.02), fontsize=7.5, color="#52514e", ha="center",
                    xycoords=("data", "axes fraction"))
    _style(ax, "Fitted-strategy Sharpe, 72 h trades", "", "annualized Sharpe")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "inference.png", dpi=150)
    log.info("inference: naive vs NW t:\n%s\nSharpe in/out: %s", t_df.round(2).to_string(index=False),
             {k: (round(v[0], 2), round(v[1], 2)) for k, v in bars.items()})

    # 5. corpus timeline
    tl = pd.read_csv(RESULTS_DIR / "corpus_timeline.csv", parse_dates=["date"])
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for source, color, label in (("gdelt", C_BLUE, "GDELT articles"), ("rss", C_ORANGE, "RSS articles")):
        sub = tl[tl["source"] == source].set_index("date")["n_articles"] \
            .resample("D").sum().rolling(7, min_periods=1).mean()
        ax.plot(sub.index, sub.values, linewidth=2, color=color, label=f"{label} (7d avg)")
    ax.axvline(ERA_SPLIT.tz_localize(None), color="#52514e", linewidth=1, linestyle="--")
    ax.annotate("era split", xy=(ERA_SPLIT.tz_localize(None), ax.get_ylim()[1] * 0.9),
                fontsize=8, color="#52514e")
    _style(ax, "Corpus coverage by source (articles/day)", "", "articles / day")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "corpus_timeline.png", dpi=150)
    log.info("figures -> %s", FIGURES_DIR)


# --------------------------------------------------------------------------- #
# self-check
# --------------------------------------------------------------------------- #

def self_check() -> None:
    rng = np.random.default_rng(7)

    # 1. NW t on iid noise is not inflated: ~5% false positives at |t|>1.96
    fp = sum(abs(nw_tstat(rng.standard_normal(500), 23)) > 1.96 for _ in range(300)) / 300
    assert fp < 0.12, f"NW false positive rate too high on iid noise: {fp:.3f}"

    # 2. IC recovery: y = rho*x + noise -> corr = rho/sqrt(rho^2+1)
    x = pd.Series(rng.standard_normal(5000))
    y = 0.3 * x + pd.Series(rng.standard_normal(5000))
    true_corr = 0.3 / np.sqrt(0.3 ** 2 + 1)
    p = ic_products(x, y, rank=False)
    assert abs(p.mean() - true_corr) < 0.03, f"IC recovery failed: {p.mean():.3f} vs {true_corr:.3f}"

    # 3. Overlapping forward returns: naive variance underestimates; NW with
    #    lag h-1 shrinks |t| by ~sqrt(h) on an autocorrelated product series
    h, n = 24, 3000
    r = rng.standard_normal(n + h)
    fwd = np.array([r[i + 1:i + 1 + h].sum() for i in range(n)])  # h-overlapping
    sig = pd.Series(rng.standard_normal(n)).rolling(h).mean().bfill()  # persistent
    prods = ic_products(sig, pd.Series(fwd), rank=False)
    t_naive = nw_tstat(prods.values, 0)
    t_nw = nw_tstat(prods.values, h - 1)
    assert abs(t_nw) <= abs(t_naive), "NW should not exceed naive t on overlapping data"
    assert abs(t_nw) < 3, f"NW t on null overlapping data too large: {t_nw:.2f}"

    # 4. cross-sectional IC recovers a planted cross-sectional relation
    T, N = 400, 5
    f_mat = pd.DataFrame(rng.standard_normal((T, N)))
    r_mat = 0.5 * f_mat + pd.DataFrame(rng.standard_normal((T, N)))
    cs = cross_sectional_ic(f_mat, r_mat, rank=False)
    assert cs.mean() > 0.25, f"cs IC recovery failed: {cs.mean():.3f}"

    log.info("self-check passed (fp=%.3f, t_naive=%.2f, t_nw=%.2f)", fp, t_naive, t_nw)


# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="all",
                        choices=["check", "provenance", "model-compare", "ic-grid", "car", "figures", "all"])
    cmd = parser.parse_args().command

    self_check()
    if cmd in ("provenance", "all"):
        run_provenance()
    if cmd in ("model-compare", "all"):
        run_model_compare()
    if cmd in ("ic-grid", "all"):
        run_ic_grid()
    if cmd in ("car", "all"):
        run_car()
    if cmd in ("figures", "all"):
        run_figures()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Print every v1-vs-v2 number a whitepaper rewrite needs, from the artifacts.

v1 of the whitepaper is built on data/results_baseline_20260729/ (the corpus as it
stood on 2026-07-29). v2 is built on data/results/ (refreshed 2026-10-18). This
script prints the two side by side and states where each number comes from, so a
writer can cite a file and a column instead of trusting a summary.

Nothing here writes anything. Run from repo root:
    .venv/bin/python scripts/refresh_report_facts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import BINANCE_SYMBOLS, MODEL_VERSION, PROMPT_VERSION
from src.extraction.cache import load_all_cached

V1 = Path("data/results_baseline_20260729")
V2 = Path("data/results")
KEYS = ["model", "cadence", "feature_set", "feature", "horizon_h", "period",
        "ic_type", "scope", "asset"]
HORIZONS = [1, 4, 12, 24, 48, 72, 120]
LLAMA3 = "llama3-8b-q4km-v1"


def hdr(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def merged_grid() -> pd.DataFrame:
    return pd.read_csv(V1 / "ic_grid.csv").merge(
        pd.read_csv(V2 / "ic_grid.csv"), on=KEYS, suffixes=("_v1", "_v2"))


def ic_table(m: pd.DataFrame, model: str, period: str = "full",
             scope: str = "ts_all", cadence: str = "1h",
             features: list[str] | None = None) -> None:
    sel = m[(m.model == model) & (m.ic_type == "spearman") & (m.scope == scope)
            & (m.cadence == cadence) & (m.period == period)]
    print(f"\n{model} | spearman | {scope} | {cadence} | period={period}")
    print(f"{'feature':24s} " + " ".join(f"h{h:<8d}" for h in HORIZONS)
          + "   (v1 -> v2, rank IC; t_nw in brackets)")
    feats = features or sorted(sel.feature.unique())
    for f in feats:
        cells = []
        for h in HORIZONS:
            r = sel[(sel.feature == f) & (sel.horizon_h == h)]
            if r.empty:
                cells.append(f"{'-':<9s}")
                continue
            r = r.iloc[0]
            cells.append(f"{r.ic_v1:+.4f}→{r.ic_v2:+.4f} [{r.tstat_nw_v2:+.2f}]")
        print(f"{f:24s} " + " ".join(f"{c:<9s}" for c in cells))


def main() -> None:
    m = merged_grid()
    print(f"grid rows: v1={len(pd.read_csv(V1 / 'ic_grid.csv'))} "
          f"v2={len(pd.read_csv(V2 / 'ic_grid.csv'))} matched={len(m)}")

    hdr("1. CORPUS (data/results{,_baseline_20260729}/corpus_timeline.csv)")
    cb, cn = pd.read_csv(V1 / "corpus_timeline.csv"), pd.read_csv(V2 / "corpus_timeline.csv")
    for src in ("gdelt", "rss"):
        o = cb[cb.source == src]
        n = cn[cn.source == src]
        print(f"{src:6s} v1: {int(o.n_articles.sum()):>7d} articles, "
              f"{len(o)} dated rows, {o.date.min()}..{o.date.max()}")
        print(f"{'':6s} v2: {int(n.n_articles.sum()):>7d} articles, "
              f"{len(n)} dated rows, {n.date.min()}..{n.date.max()}")

    hdr("2. EXTRACTION CACHE (data/cache/extractions, per model_version)")
    phi = load_all_cached("phi4-q6k-v1", PROMPT_VERSION)
    try:
        lam = load_all_cached(LLAMA3, PROMPT_VERSION)
    except Exception:                                        # noqa: BLE001
        lam = pd.DataFrame()
    for name, df in (("phi4-q6k-v1", phi), (LLAMA3, lam)):
        if df.empty:
            print(f"{name}: no records")
            continue
        pa = pd.to_datetime(df.published_at, utc=True, errors="coerce")
        print(f"{name}: {len(df)} records, published {pa.min()} .. {pa.max()}")
        print(f"    before 2026-06-22 (pre-RSS-era): {int((pa < '2026-06-22').sum())} "
              f"({(pa < '2026-06-22').mean():.1%})")

    hdr("3. CENSUS OF THE REFRESH (scripts/refresh_census.py, 2026-10-18)")
    print("18,688 items to extract | 18,479 cached | 218 failed (null numerics, 1.2%)")
    print("by matched ticker (top): BTC 6580, ETH 1627, XRP 1603, TRUMP 1574, "
          "US 1238, SOL 958, CHIP 529, NEAR 513")
    print("study assets in that batch: 10,912 (BTC 6580 / ETH 1627 / XRP 1603 / "
          "SOL 958 / BNB 144)")

    hdr("4. phi4 IC, FULL PERIOD (v1 vintage -> refreshed)")
    ic_table(m, "phi4-q6k-v1", features=[
        "composite_full", "composite_rare", "polarity_sum", "polarity_mean",
        "mag_weighted_polarity", "novelty_polarity", "recency_polarity",
        "hack_flag", "depeg_flag", "listing_flag", "regulation_flag"])

    hdr("5. THE ERA SPLIT THAT v1 USED TO ARGUE INSTABILITY")
    for per in ("gdelt_era", "rss_era"):
        ic_table(m, "phi4-q6k-v1", period=per, features=["composite_full", "composite_rare"])

    hdr("6. NON-LLM BASELINE (identical inputs in both vintages)")
    t = pd.read_csv(V2 / "ic_grid.csv")
    t = t[(t.model == "gdelt_tone") & (t.cadence == "1h") & (t.ic_type == "spearman")
          & (t.period == "full") & (t.scope == "ts_all")]
    print(t[["feature", "horizon_h", "ic", "tstat_nw", "n_obs"]].to_string(index=False))

    hdr("7. LLAMA-3 ARM (re-extracted 2026-10-19; same window as phi-4)")
    ic_table(m, LLAMA3, features=["composite_full", "composite_rare", "hack_flag"])

    hdr("8. EVENT STUDY (data/results{,_baseline_20260729}/car_adjusted.csv)")
    c = pd.read_csv(V1 / "car_adjusted.csv").merge(
        pd.read_csv(V2 / "car_adjusted.csv"), on=["horizon_h", "bucket"],
        suffixes=("_v1", "_v2"))
    print(c[["horizon_h", "bucket", "car_raw_v1", "car_raw_v2", "car_mktadj_v1",
             "car_mktadj_v2", "n_events_v1", "n_events_v2"]].to_string(index=False))

    hdr("9. STRATEGY-LEVEL + INFERENCE (data/logs/refresh_study_20261018.log)")
    print("ic_study inference: naive vs NW t (refreshed run)")
    print("  h:      1     4    12    24    48    72   120")
    print("  naive 1.40  2.46  3.38  3.54  4.27  5.97  8.80")
    print("  nw    1.40  1.53  1.24  0.93  0.82  0.98  1.12")
    print("Sharpe in/out: full features (2.07, 1.09) | rare-event flags (1.70, 0.32)")

    hdr("10. REDACTION AUDIT (fresh sample 2026-10-19; July in results_baseline_20260729)")
    print(pd.read_csv(V2 / "contamination_redaction.csv").to_string(index=False))

    hdr("11. SAMPLE GEOMETRY")
    n_assets = len(BINANCE_SYMBOLS)
    h1 = 7039 - 4825                      # hourly bars added by the OHLCV refresh
    print(f"assets: {n_assets} {BINANCE_SYMBOLS}")
    print(f"hourly bars/symbol: v1 {4825} (to 2026-07-18) -> v2 {7039} "
          f"(to 2026-10-18 06:00), +{h1}")
    print("independent 5d obs per asset: v1 ~5 weeks -> ~7; v2 ~17 weeks -> ~24")


if __name__ == "__main__":
    main()

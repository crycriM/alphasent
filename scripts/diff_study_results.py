#!/usr/bin/env python3
"""Diff the refreshed study results against the frozen July baseline.

Compares data/results/ with data/results_baseline_*/ on the keys that carry the
study's claims, so a refresh can be reported as "what changed" rather than
"here are new numbers": rank IC per model/horizon, the event-study CAR, and the
corpus timeline.

Run from repo root: .venv/bin/python scripts/diff_study_results.py [baseline_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RESULTS = Path("data/results")
BASELINE_GLOB = "data/results_baseline_*"


def pick_baseline() -> Path | None:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    cands = sorted(Path("data").glob("results_baseline_*"))
    return cands[-1] if cands else None


def load(d: Path, name: str) -> pd.DataFrame:
    p = d / name
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def compare_csv(name: str, keys: list[str], cols: list[str], base: Path) -> None:
    old, new = load(base, name), load(RESULTS, name)
    print(f"\n=== {name} ===")
    if old.empty or new.empty:
        print(f"  missing on one side (baseline rows={len(old)}, new rows={len(new)})")
        return
    m = old.merge(new, on=keys, suffixes=("_old", "_new"))
    print(f"  baseline {len(old)} rows | refreshed {len(new)} rows | matched {len(m)} "
          f"| new rows {len(new) - len(m)}")
    for c in cols:
        if f"{c}_old" not in m.columns:
            continue
        d = m[f"{c}_new"] - m[f"{c}_old"]
        changed = (d.abs() > 1e-12).sum()
        print(f"  {c:12s} old mean {m[f'{c}_old'].mean():+.4f} -> new "
              f"{m[f'{c}_new'].mean():+.4f} | rows changed {changed}/{len(m)} "
              f"| max |delta| {d.abs().max():.4f}")
    if name == "ic_grid.csv":
        sub = m[(m["scope"] == "ts_all") & (m["ic_type"] == "spearman")
                & (m["period"] == "full") & (m["cadence"] == "1h")]
        sub = sub.sort_values(["model", "feature", "horizon_h"])
        print("\n  per-model ts_all spearman IC (full period, hourly):")
        for _, r in sub.iterrows():
            print(f"    {r['model']:20s} {r['feature']:18s} h={int(r['horizon_h']):4d}  "
                  f"old {r['ic_old']:+.4f} (t {r['tstat_nw_old']:+.1f}) -> "
                  f"new {r['ic_new']:+.4f} (t {r['tstat_nw_new']:+.1f})  "
                  f"n_obs {int(r['n_obs_old'])} -> {int(r['n_obs_new'])}")
    if name == "corpus_timeline.csv":
        for src in ("gdelt", "rss"):
            o = old[old["source"] == src]["n_articles"].sum()
            n = new[new["source"] == src]["n_articles"].sum()
            print(f"  {src}: {o} -> {n} articles")


def main() -> None:
    base = pick_baseline()
    if base is None:
        print("no baseline directory found — nothing to diff against")
        return
    print(f"baseline: {base}")
    # 'asset' belongs in the keys: scope=ts_asset carries one row per asset, and
    # without it the merge becomes many-to-many and the summary numbers are junk.
    # date_start/date_end are deliberately excluded — they are what moves when the
    # corpus grows; date range changes are reported separately.
    compare_csv("ic_grid.csv",
                ["model", "cadence", "feature_set", "feature", "horizon_h",
                 "period", "ic_type", "scope", "asset"],
                ["ic", "tstat_nw", "n_obs"], base)
    compare_csv("car_adjusted.csv", ["horizon_h", "bucket"],
                ["car_raw", "car_mktadj", "n_events"], base)
    compare_csv("corpus_timeline.csv", ["source", "date"], ["n_articles"], base)


if __name__ == "__main__":
    main()

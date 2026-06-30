from __future__ import annotations

import re

import numpy as np
import pandas as pd

from scipy import stats

REDACTION_MAP_DEFAULT: dict[str, str] = {
    r"\bBitcoin\b|\bBTC\b": "Asset_X",
    r"\bEthereum\b|\bETH\b": "Asset_Y",
    r"\bRipple\b|\bXRP\b": "Asset_Z",
    r"\bTerraUSD\b|\bUST\b|\bLUNA\b": "Asset_W",
    r"\bSEC\b": "Regulator_A",
    r"\bBinance\b": "Exchange_A",
}

MODEL_CUTOFF = pd.Timestamp("2023-03-01")  # Llama-3-8B training cutoff

def redact(text: str, mapping: dict[str, str] | None = None) -> str:
    """Apply entity redaction map to text."""
    m = mapping or REDACTION_MAP_DEFAULT
    for pattern, replacement in m.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def compare_extractions(records_a: pd.DataFrame, records_b: pd.DataFrame) -> dict:
    """Compute distribution distance on event_type (TVD), polarity KS, magnitude KS."""
    out = {}
    # Total Variation Distance on event_type
    if "event_type" in records_a.columns and "event_type" in records_b.columns:
        a_dist = records_a["event_type"].value_counts(normalize=True).fillna(0)
        b_dist = records_b["event_type"].value_counts(normalize=True).fillna(0)
        all_cats = a_dist.index.union(b_dist.index)
        tvd = 0.5 * (a_dist.reindex(all_cats, fill_value=0) - b_dist.reindex(all_cats, fill_value=0)).abs().sum()
        out["event_type_tvd"] = float(tvd)
    # KS on polarity
    if "polarity" in records_a.columns and "polarity" in records_b.columns:
        ks_stat, ks_p = stats.ks_2samp(records_a["polarity"].fillna(0), records_b["polarity"].fillna(0))
        out["polarity_ks_stat"] = float(ks_stat)
        out["polarity_ks_p"] = float(ks_p)
    # KS on magnitude
    if "magnitude" in records_a.columns and "magnitude" in records_b.columns:
        ks_stat, ks_p = stats.ks_2samp(records_a["magnitude"].fillna(0), records_b["magnitude"].fillna(0))
        out["magnitude_ks_stat"] = float(ks_stat)
        out["magnitude_ks_p"] = float(ks_p)
    return out

def extraction_only_subset(records: pd.DataFrame) -> pd.DataFrame:
    """Filter to records where extraction_only == True."""
    return records[records["extraction_only"] == True]

def temporal_holdout_split(
    records: pd.DataFrame,
    cutoff: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split records pre- and post-cutoff. Default cutoff = model training cutoff."""
    cutoff = cutoff or MODEL_CUTOFF
    pre = records[records["published_at"] < cutoff].copy()
    post = records[records["published_at"] >= cutoff].copy()
    return pre, post

if __name__ == "__main__":
    # Smoke test
    text = "Bitcoin and SEC action on Binance"
    redacted = redact(text)
    assert "Bitcoin" not in redacted
    assert "Asset_X" in redacted
    assert "Regulator_A" in redacted

    # compare_extractions
    ra = pd.DataFrame({"event_type": ["hack", "hack", "macro"], "polarity": [-0.9, -0.5, 0.2], "magnitude": [0.8, 0.5, 0.3], "extraction_only": [True, True, False]})
    rb = pd.DataFrame({"event_type": ["hack", "macro", "macro"], "polarity": [-0.6, 0.1, 0.3], "magnitude": [0.7, 0.4, 0.6], "extraction_only": [True, True, True]})
    comp = compare_extractions(ra, rb)
    assert isinstance(comp["event_type_tvd"], float)
    assert isinstance(comp["polarity_ks_stat"], float)

    # extraction_only subset
    sub = extraction_only_subset(ra)
    assert len(sub) == 2

    # temporal split
    df = pd.DataFrame({"published_at": [pd.Timestamp("2022-01-01"), pd.Timestamp("2023-06-01")], "asset": ["BTC"] * 2})
    pre, post = temporal_holdout_split(df)
    assert len(pre) == 1 and len(post) == 1
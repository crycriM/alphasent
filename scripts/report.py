from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.eval import sharpe, deflated_sharpe_with_ci, hit_rate, max_drawdown
from src.backtest.contamination import redact, compare_extractions, extraction_only_subset, temporal_holdout_split
from src.backtest.walkforward import generate_windows

def report_dsr(returns: pd.Series, n_trials: int = 1, periods_per_year: int = 8760):
    """Print DSR with 95% CIs."""
    dsr, lo, hi = deflated_sharpe_with_ci(returns, n_trials, periods_per_year)
    print(f"DSR = {dsr:.4f}, 95% CI = [{lo:.4f}, {hi:.4f}]")
    print(f"Hit rate = {hit_rate(returns):.4f}")
    print(f"Max DD = {max_drawdown((1 + returns).cumprod()):.4f}")

def report_contamination(records: pd.DataFrame):
    """Print contamination audit findings."""
    # Redaction test
    original = records.copy()
    redacted_text = records["published_at"].astype(str).apply(lambda x: redact(x))
    # compare_extractions on same records (identity → TVD=0)
    comp = compare_extractions(records, records)
    print(f"Redaction test (self-comparison): {comp}")

    # Extraction-only ablation
    sub = extraction_only_subset(records)
    print(f"Extraction-only subset: {len(sub)}/{len(records)} records")

    # Temporal holdout
    pre, post = temporal_holdout_split(records)
    print(f"Temporal split: pre-cutoff={len(pre)}, post-cutoff={len(post)}")

if __name__ == "__main__":
    np.random.seed(42)
    ret = pd.Series(np.random.randn(500) * 0.01 + 0.0002)
    report_dsr(ret, n_trials=5)

    recs = pd.DataFrame({
        "event_type": ["hack", "regulation", "macro", "listing"] * 25,
        "polarity": np.random.uniform(-1, 1, 100),
        "magnitude": np.random.uniform(0, 1, 100),
        "extraction_only": np.random.choice([True, False], 100, p=[0.6, 0.4]),
        "published_at": pd.date_range("2020-01-01", periods=100, freq="W").repeat(1)[:100],
    })
    report_contamination(recs)
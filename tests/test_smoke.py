from __future__ import annotations

import json

import pandas as pd
import pytest

def test_smoke_with_synthetic_data():
    from src.extraction.batch_etl import read_all_raw_news
    from src.live.smoke_test import run_smoke

    raw_df = read_all_raw_news()

    if raw_df.empty:
        pytest.skip("No raw news data available")

    report = run_smoke(max_items=min(10, len(raw_df)))

    assert isinstance(report, dict)
    assert report["status"] in ("PASS", "FAIL")
    assert "extraction_success_rate" in report
    assert "items_processed" in report

def test_smoke_minimal():
    from src.live.smoke_test import run_smoke
    raw_df = pd.read_parquet("data/crypto_rss/normalized/2026-06-29.parquet")
    assert not raw_df.empty
    report = run_smoke(max_items=3)
    assert isinstance(report, dict)
    assert report["items_processed"] <= 3
    json.dumps(report, default=str)

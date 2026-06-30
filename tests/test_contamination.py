import pytest
import numpy as np
import pandas as pd

from src.backtest.contamination import (
    redact,
    compare_extractions,
    extraction_only_subset,
    temporal_holdout_split,
    REDACTION_MAP_DEFAULT,
)

class TestContamination:
    """Redaction, compare, extraction-only, temporal holdout."""

    def test_redaction_replaces_all(self):
        text = "Bitcoin BTC on Binance SEC action"
        redacted = redact(text)
        assert "Bitcoin" not in redacted
        assert "BTC" not in redacted
        assert "Binance" not in redacted
        assert "SEC" not in redacted
        assert "Asset_X" in redacted
        assert "Exchange_A" in redacted
        assert "Regulator_A" in redacted

    def test_redact_custom_mapping(self):
        mapping = {r"\bfoo\b": "bar"}
        assert redact("hello foo world", mapping) == "hello bar world"

    def test_redact_no_match(self):
        text = "just random text"
        assert redact(text) == text

    def test_compare_extractions_finite(self):
        ra = pd.DataFrame({
            "event_type": ["hack"] * 50 + ["macro"] * 50,
            "polarity": np.random.uniform(-1, 1, 100),
            "magnitude": np.random.uniform(0, 1, 100),
        })
        rb = pd.DataFrame({
            "event_type": ["hack"] * 40 + ["regulation"] * 60,
            "polarity": np.random.uniform(-1, 1, 100),
            "magnitude": np.random.uniform(0, 1, 100),
        })
        comp = compare_extractions(ra, rb)
        assert isinstance(comp["event_type_tvd"], float)
        assert 0.0 <= comp["event_type_tvd"] <= 1.0
        assert isinstance(comp["polarity_ks_stat"], float)
        assert isinstance(comp["magnitude_ks_stat"], float)

    def test_compare_extractions_identity(self):
        df = pd.DataFrame({
            "event_type": ["hack", "macro"],
            "polarity": [0.5, -0.3],
            "magnitude": [0.8, 0.2],
        })
        comp = compare_extractions(df, df)
        assert comp["event_type_tvd"] == pytest.approx(0.0)
        assert comp["polarity_ks_stat"] == pytest.approx(0.0)

    def test_extraction_only_subset_filters(self):
        df = pd.DataFrame({
            "extraction_only": [True, False, True],
            "asset": ["BTC"] * 3,
        })
        sub = extraction_only_subset(df)
        assert len(sub) == 2
        assert (sub["extraction_only"] == True).all()

    def test_temporal_holdout_split(self):
        df = pd.DataFrame({
            "published_at": [
                pd.Timestamp("2022-01-01"),
                pd.Timestamp("2023-06-01"),
                pd.Timestamp("2021-01-01"),
            ],
            "asset": ["BTC"] * 3,
        })
        pre, post = temporal_holdout_split(df)
        assert len(pre) == 2  # 2022, 2021 before 2023-03
        assert len(post) == 1  # 2023-06 after cutoff

    def test_temporal_holdout_disjoint(self):
        df = pd.DataFrame({
            "published_at": [
                pd.Timestamp("2022-06-01"),
                pd.Timestamp("2023-04-01"),
            ],
            "asset": ["BTC"] * 2,
        })
        pre, post = temporal_holdout_split(df)
        all_pre = set(pre.index)
        all_post = set(post.index)
        assert len(all_pre.intersection(all_post)) == 0  # disjoint
        assert len(all_pre.union(all_post)) == len(df)  # complete

    def test_temporal_holdout_custom_cutoff(self):
        df = pd.DataFrame({
            "published_at": [
                pd.Timestamp("2022-01-01"),
                pd.Timestamp("2023-01-01"),
            ],
            "asset": ["BTC"] * 2,
        })
        pre, post = temporal_holdout_split(df, cutoff=pd.Timestamp("2022-06-01"))
        assert len(pre) == 1
        assert len(post) == 1
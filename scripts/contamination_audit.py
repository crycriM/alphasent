#!/usr/bin/env python3
"""
Contamination audit — entity-redaction A/B on real articles.

For a common sample of articles already extracted by both models, re-extract
with entity names replaced by placeholders (build_extraction_prompt(redact=True))
and compare against the original cached extractions. If a model's output is
driven by the text, redaction should barely move it; systematic shifts indicate
reliance on parametric (memorized) knowledge of the named entities.

Nothing is written to the extraction cache. Outputs:
  data/results/contamination_redaction.csv          one summary row per model
  data/results/contamination_redaction_detail_{model}.csv  per-article pairs

Run (server must expose the model at LLM_BASE_URL):
  .venv/bin/python scripts/contamination_audit.py --model-version phi4-q6k-v1
  .venv/bin/python scripts/contamination_audit.py --model-version llama3-8b-q4km-v1
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("contamination_audit")

SERVED_NAME = {"phi4-q6k-v1": "phi4", "llama3-8b-q4km-v1": "llama3-8b"}
PROMPT_V = "1.0.0"
N_SAMPLE = 150
SEED = 42


def load_article_texts() -> pd.DataFrame:
    """item_id -> (title, body) from the RSS normalized store (text-complete source)."""
    from src.config import RSS_DATA_ROOT
    frames = [pd.read_parquet(f, columns=["item_id", "title", "summary"])
              for f in sorted((RSS_DATA_ROOT / "normalized").glob("*.parquet"))]
    rss = pd.concat(frames, ignore_index=True).drop_duplicates("item_id")
    return rss.rename(columns={"summary": "body"}).fillna({"body": ""}).set_index("item_id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-version", required=True, choices=list(SERVED_NAME))
    parser.add_argument("--n", type=int, default=N_SAMPLE)
    args = parser.parse_args()

    os.environ["LLM_MODEL_NAME"] = SERVED_NAME[args.model_version]
    # imports below read LLM_MODEL_NAME at module load
    from src.config import DATA_ROOT
    from src.extraction.cache import load_all_cached
    from src.extraction.model import call_llm
    from src.extraction.prompt import build_extraction_prompt
    from src.backtest.contamination import compare_extractions

    results_dir = DATA_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    caches = {m: load_all_cached(m, PROMPT_V) for m in SERVED_NAME}
    common = set.intersection(*[set(c["item_id"]) for c in caches.values()])
    texts = load_article_texts()
    common &= set(texts.index)
    orig = caches[args.model_version]
    orig = orig[orig["item_id"].isin(common)].drop_duplicates("item_id")
    sample = orig.sample(n=min(args.n, len(orig)), random_state=SEED).set_index("item_id")
    log.info("model=%s served=%s common_pool=%d sampled=%d",
             args.model_version, SERVED_NAME[args.model_version], len(common), len(sample))

    rows, failures = [], 0
    for i, (item_id, row) in enumerate(sample.iterrows(), 1):
        art = texts.loc[item_id]
        prompt = build_extraction_prompt(str(art["title"]), str(art["body"]), redact=True)
        try:
            red = call_llm(prompt)
        except Exception as e:
            failures += 1
            log.warning("extraction failed (%d): %s", failures, e)
            continue
        rows.append({
            "item_id": item_id,
            "asset_orig": row["asset"], "asset_red": red["asset"],
            "event_type_orig": row["event_type"], "event_type_red": red["event_type"],
            "polarity_orig": float(row["polarity"]), "polarity_red": float(red["polarity"]),
            "magnitude_orig": float(row["magnitude"]), "magnitude_red": float(red["magnitude"]),
            "confidence_orig": float(row["confidence"]), "confidence_red": float(red["confidence"]),
            "extraction_only_red": bool(red["extraction_only"]),
        })
        if i % 25 == 0:
            log.info("progress %d/%d (%d failures)", i, len(sample), failures)

    detail = pd.DataFrame(rows)
    detail.to_csv(results_dir / f"contamination_redaction_detail_{args.model_version}.csv", index=False)

    orig_df = detail.rename(columns=lambda c: c.replace("_orig", "")) \
        [["event_type", "polarity", "magnitude"]]
    red_df = detail.rename(columns=lambda c: c.replace("_red", "")) \
        [["event_type", "polarity", "magnitude"]]
    pair = compare_extractions(orig_df, red_df)

    sign = np.sign(detail["polarity_orig"])
    nonzero = sign != 0
    summary = {
        "model": args.model_version,
        "n_articles": len(detail),
        "n_failures": failures,
        **pair,
        "event_type_flip_rate": (detail["event_type_orig"] != detail["event_type_red"]).mean(),
        # model names the true asset although its ticker/name was redacted ->
        # it recognized the entity from context (parametric recall indicator);
        # only assets covered by the redaction map count as evidence
        "asset_recovery_rate": (
            lambda d: (d["asset_red"] == d["asset_orig"]).mean() if len(d) else np.nan
        )(detail[detail["asset_orig"].isin(
            {"BTC", "ETH", "XRP", "SOL", "ADA", "USDT", "USDC", "LUNA", "UST"})]),
        "polarity_sign_flip_rate": (np.sign(detail.loc[nonzero, "polarity_red"]) != sign[nonzero]).mean(),
        "mean_abs_polarity_delta": (detail["polarity_red"] - detail["polarity_orig"]).abs().mean(),
        "mean_polarity_delta": (detail["polarity_red"] - detail["polarity_orig"]).mean(),
        "mean_confidence_delta": (detail["confidence_red"] - detail["confidence_orig"]).mean(),
        "polarity_corr_orig_red": detail["polarity_orig"].corr(detail["polarity_red"]),
    }
    out_path = results_dir / "contamination_redaction.csv"
    out = pd.DataFrame([summary])
    if out_path.exists():
        prev = pd.read_csv(out_path)
        out = pd.concat([prev[prev["model"] != args.model_version], out], ignore_index=True)
    out.to_csv(out_path, index=False)
    log.info("summary:\n%s", pd.Series(summary).to_string())


if __name__ == "__main__":
    main()

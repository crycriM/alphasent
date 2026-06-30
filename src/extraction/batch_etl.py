"""
Offline batch extraction job.

Reads all RawNewsItems from the Layer-1 raw store, checks the cache,
calls the LLM for uncached items, computes novelty, and writes
EventRecords to the cache.

Never re-run during backtesting — this is a one-pass offline job.
If model or prompt version changes, old cached records remain valid.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import (
    MODEL_VERSION,
    PROMPT_VERSION,
    RAW_NEWS_DIR,
)
from src.extraction.cache import (
    cache_exists,
    compute_content_hash,
    load_all_cached,
    load_from_cache,
    save_to_cache,
)
from src.extraction.model import call_llm
from src.extraction.novelty import compute_novelty
from src.extraction.prompt import build_extraction_prompt

log = logging.getLogger("extraction.batch")


def read_all_raw_news(data_root: Path = RAW_NEWS_DIR) -> pd.DataFrame:
    """Read all day-partitioned parquet files from the raw news store.

    Handles both the RSS normalized/ directory and a unified raw/news/ directory.
    Returns a single DataFrame with the RawNewsItem schema columns.
    """
    dfs = []

    # Check RSS normalized/
    rss_dir = data_root.parent / "crypto_rss" / "normalized"
    if rss_dir.exists():
        for f in sorted(rss_dir.glob("*.parquet")):
            df = pd.read_parquet(f)
            # Add source column if missing
            if "source" not in df.columns:
                df["source"] = "rss"
            if "asset_mentions" not in df.columns:
                df["asset_mentions"] = None
            dfs.append(df)

    # Check unified raw/news/
    if data_root.exists():
        for f in sorted(data_root.glob("*.parquet")):
            df = pd.read_parquet(f)
            if "source" not in df.columns:
                df["source"] = "gdelt"
            dfs.append(df)

    # Check cryptopanic normalized/
    cp_dir = data_root.parent / "cryptopanic" / "normalized"
    if cp_dir.exists():
        for f in sorted(cp_dir.glob("*.parquet")):
            df = pd.read_parquet(f)
            df["source"] = "cryptopanic"
            dfs.append(df)

    if not dfs:
        log.warning("No raw news files found in %s", data_root.parent)
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    # Ensure body column exists (RSS uses 'summary' not 'body')
    if "body" not in combined.columns and "summary" in combined.columns:
        combined["body"] = combined["summary"]

    # Ensure body column exists (CryptoPanic uses 'description')
    if "body" not in combined.columns and "description" in combined.columns:
        combined["body"] = combined["description"]

    if "body" not in combined.columns:
        # Try common alternate names
        for alt_name in ("summary", "description", "content"):
            if alt_name in combined.columns:
                combined["body"] = combined[alt_name]
                break
        else:
            combined["body"] = ""

    # Coerce published_at to tz-aware UTC datetime
    if "published_at" in combined.columns:
        combined["published_at"] = pd.to_datetime(combined["published_at"], utc=True)

    # Ensure ingested_at is tz-aware UTC
    if "ingested_at" in combined.columns:
        combined["ingested_at"] = pd.to_datetime(combined["ingested_at"], utc=True)

    # Ensure title column
    if "title" not in combined.columns:
        log.error("No 'title' column in raw news data")
        return pd.DataFrame()

    # Deduplicate by item_id (keep the first occurrence)
    combined = combined.drop_duplicates(subset="item_id", keep="first")

    log.info("Loaded %d raw news items from %d partitions", len(combined), len(dfs))
    return combined


def run_batch_extraction(
    raw_df: pd.DataFrame | None = None,
    data_root: Path = RAW_NEWS_DIR,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Run the full extraction batch.

    Args:
        raw_df: Optional pre-loaded raw DataFrame. If None, reads from disk.
        data_root: Root path for raw news data.
        dry_run: If True, build prompts but don't call the LLM.

    Returns:
        DataFrame of all EventRecords (cached + newly extracted).
    """
    if raw_df is None:
        raw_df = read_all_raw_news(data_root)

    if raw_df.empty:
        log.warning("No raw news items to extract. Exiting.")
        return pd.DataFrame()

    # Filter to items with non-empty title (minimum requirement)
    raw_df = raw_df[raw_df["title"].str.len() > 0].copy()
    log.info("Processing %d items with valid titles", len(raw_df))

    # Sort by published_at so the novelty pool only ever contains events that
    # were knowably published before the current item (point-in-time safety).
    if "published_at" in raw_df.columns:
        raw_df = raw_df.sort_values("published_at").reset_index(drop=True)

    # Preload all cached records for the current (model, prompt) version in one
    # pass. This is both the cache-hit source (no per-row parquet reads) and the
    # initial novelty pool. The novelty mask filters by published_at < t, so
    # future-published records in the pool cannot leak into a given item.
    pool_df = load_all_cached(MODEL_VERSION, PROMPT_VERSION)
    if pool_df.empty:
        pool_df = pd.DataFrame()
    else:
        if "published_at" in pool_df.columns:
            pool_df["published_at"] = pd.to_datetime(pool_df["published_at"], utc=True)

    cache_hits = 0
    cache_misses = 0
    new_records = []
    failed_items = []

    for _, row in raw_df.iterrows():
        content_hash = compute_content_hash(row["title"], row.get("body", ""))

        if cache_exists(content_hash, MODEL_VERSION, PROMPT_VERSION):
            cached = load_from_cache(content_hash, MODEL_VERSION, PROMPT_VERSION)
            if cached:
                # Carry title/published_at for the novelty pool
                cached["title"] = row["title"]
                cached["published_at"] = row.get(
                    "published_at", row.get("ingested_at")
                )
                pool_df = pd.concat(
                    [pool_df, pd.DataFrame([cached])], ignore_index=True
                )
                cache_hits += 1
            continue

        # Cache miss — need to extract
        cache_misses += 1

        if dry_run:
            prompt = build_extraction_prompt(row["title"], row.get("body", ""))
            log.info("DRY RUN: would extract for %s — prompt length: %d chars",
                     row["item_id"][:8], len(prompt))
            continue

        # Build prompt
        prompt = build_extraction_prompt(row["title"], row.get("body", ""))

        # Call LLM
        try:
            result = call_llm(prompt)

            published_at = row.get("published_at", row.get("ingested_at"))

            # Compute novelty against the full PIT-filtered pool
            event_for_novelty = {
                "asset": result["asset"],
                "event_type": result["event_type"],
                "extracted_at": datetime.now(timezone.utc),
                "published_at": published_at,
                "content_hash": content_hash,
                "title": row["title"],
            }
            novelty_score = compute_novelty(event_for_novelty, pool_df)

            # Build EventRecord
            record = {
                "item_id": str(row["item_id"]),
                "content_hash": content_hash,
                "model_version": MODEL_VERSION,
                "prompt_version": PROMPT_VERSION,
                "extracted_at": datetime.now(timezone.utc),
                "published_at": published_at,
                "asset": str(result["asset"]),
                "event_type": str(result["event_type"]),
                "polarity": float(result["polarity"]),
                "magnitude": float(result["magnitude"]),
                "novelty": float(novelty_score),
                "confidence": float(result["confidence"]),
                "extraction_only": bool(result["extraction_only"]),
                "title": row["title"],
            }
            # Save to cache (writes only EventRecord fields to parquet)
            save_to_cache(record, MODEL_VERSION, PROMPT_VERSION)
            pool_df = pd.concat(
                [pool_df, pd.DataFrame([record])], ignore_index=True
            )
            new_records.append(record)

            log.info(
                "Extracted [%s] %s: %s p=%.2f m=%.2f n=%.2f conf=%.2f only=%s",
                row["item_id"][:8], result["asset"], result["event_type"],
                result["polarity"], result["magnitude"], novelty_score,
                result["confidence"], result["extraction_only"],
            )
        except Exception as e:
            log.error("Failed to extract %s: %s", row["item_id"][:8], e)
            failed_items.append({"item_id": row["item_id"], "error": str(e)})

    if failed_items:
        log.warning(
            "Extraction failures: %d/%d items failed (%.1f%%)",
            len(failed_items), len(raw_df),
            100.0 * len(failed_items) / max(len(raw_df), 1),
        )

    # Build final DataFrame from the full PIT-correct pool (cached + new)
    if not pool_df.empty:
        final_df = pool_df.copy()
        # Ensure proper column order
        columns = [
            "item_id", "content_hash", "model_version", "prompt_version",
            "extracted_at", "published_at", "asset", "event_type", "polarity",
            "magnitude", "novelty", "confidence", "extraction_only",
        ]
        existing_cols = [c for c in columns if c in final_df.columns]
        final_df = final_df[existing_cols]
        log.info(
            "Extraction complete: %d cache hits, %d new extractions, %d total records",
            cache_hits, len(new_records), len(final_df),
        )
        return final_df
    else:
        log.info("Extraction complete: %d cache hits, %d new extractions, 0 total",
                 cache_hits, len(new_records))
        return pd.DataFrame()


def main():
    """Entry point for the batch extraction job."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    import sys
    dry_run = "--dry-run" in sys.argv

    df = run_batch_extraction(dry_run=dry_run)

    if not df.empty:
        log.info("Final record summary:\n%s", df.describe())
        log.info("Event type distribution:\n%s", df["event_type"].value_counts())
        log.info("Asset distribution:\n%s", df["asset"].value_counts())


if __name__ == "__main__":
    main()

"""
Offline batch extraction job.

Reads all RawNewsItems from the Layer-1 raw store, checks the cache,
calls the LLM for uncached items, computes novelty, and writes
EventRecords to the cache.

PRE-FILTER: Before calling the LLM, each article's title+body is scanned
against the per-month crypto universe (from data/perimeter/) to determine
if it concerns an actively-traded token. If no ticker match, the article
is skipped entirely — saving LLM tokens and avoiding false positives from
models that don't recognize newer tokens.

Never re-run during backtesting — this is a one-pass offline job.
If model or prompt version changes, old cached records remain valid.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
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
from src.ingest.perimeter import load_universe, match_ticker

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

    # Check unified raw/news/ (GDELT)
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


_UNIVERSE_CACHE: dict[tuple[int, int], set[str]] = {}


def universe_for_day(day: date) -> set[str]:
    """Perimeter universe of the item's own month, memoized per month.

    The pre-filter must be point-in-time per article: a Feb-2026 article may only
    match tickers listed in Feb 2026. Resolving ONE universe from the earliest
    article of the batch (the old `_resolve_universe` behaviour) silently applies
    an older month's listing set to every later month, which both misses tokens
    listed in between and leaks the wrong perimeter into the match decision.
    Memoizing per month keeps the per-item check cheap (the JSON parse happens at
    most once per month present in the batch).
    """
    key = (day.year, day.month)
    if key not in _UNIVERSE_CACHE:
        _UNIVERSE_CACHE[key] = load_universe(date(day.year, day.month, 1))
        log.info("Crypto universe %04d-%02d: %d tickers",
                 day.year, day.month, len(_UNIVERSE_CACHE[key]))
    return _UNIVERSE_CACHE[key]


def _resolve_universe(article_dates: pd.Series) -> set[str]:
    """Universe for the month of the earliest article (single query helper)."""
    if article_dates.empty or article_dates.isna().all():
        univ_date = date.today().replace(day=1)
    else:
        earliest = article_dates.min()
        if hasattr(earliest, "to_pydatetime"):
            earliest = earliest.to_pydatetime()
        univ_date = date(earliest.year, earliest.month, 1)

    universe = load_universe(univ_date)
    log.info("Using crypto universe from %s: %d tickers", univ_date, len(universe))
    return universe


def _item_day(row) -> date:
    """Published (or ingested) day of a raw-news row, NaN-safe."""
    val = row.get("published_at", None)
    if val is None or (isinstance(val, float) and pd.isna(val)):
        val = row.get("ingested_at", None)
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return date.today()
    ts = pd.Timestamp(val)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC")
    return ts.date()


def _safe_str(val) -> str:
    """Convert a pandas cell value to a string, handling NaN/None/floats."""
    if val is None:
        return ""
    if isinstance(val, float):
        import math
        return "" if math.isnan(val) else str(val)
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    return str(val)


def _article_text(row) -> str:
    """Combine title + body into a single searchable text."""
    parts = [_safe_str(row.get("title", ""))]
    body = _safe_str(
        row.get("body", "") or row.get("summary", "") or row.get("description", "")
    )
    if body:
        parts.append(body)
    return " ".join(parts)


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

    # Sort by published_at so the novelty pool only contains events that
    # were knowably published before the current item (point-in-time safety).
    if "published_at" in raw_df.columns:
        raw_df = raw_df.sort_values("published_at").reset_index(drop=True)

    # Universe is resolved per item (see universe_for_day): a multi-month batch
    # must not be pre-filtered with a single month's listing set.

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
    skipped_no_match = 0
    new_records = []
    failed_items = []

    for _, row in raw_df.iterrows():
        content_hash = compute_content_hash(
            _safe_str(row.get("title", "")),
            _safe_str(row.get("body", "") or row.get("summary", "") or row.get("description", "")),
        )

        if cache_exists(content_hash, MODEL_VERSION, PROMPT_VERSION):
            cached = load_from_cache(content_hash, MODEL_VERSION, PROMPT_VERSION)
            if cached:
                # Carry title/published_at for the novelty pool
                cached["title"] = _safe_str(row.get("title", ""))
                pub = row.get("published_at", row.get("ingested_at"))
                if isinstance(pub, float):
                    pub = None  # NaN
                cached["published_at"] = pub
                pool_df = pd.concat(
                    [pool_df, pd.DataFrame([cached])], ignore_index=True
                )
                cache_hits += 1
            continue

        # Cache miss — check if article mentions any known ticker
        # PIT-safe: the universe is the one in force in the article's own month.
        text = _article_text(row)
        matched_ticker = match_ticker(text, universe_for_day(_item_day(row)))

        if matched_ticker is None:
            skipped_no_match += 1
            continue

        # We have a ticker match — proceed with extraction
        cache_misses += 1

        if dry_run:
            prompt = build_extraction_prompt(
                _safe_str(row.get("title", "")),
                _safe_str(row.get("body", "") or row.get("summary", "") or row.get("description", "")),
            )
            log.info("DRY RUN: would extract [%s] %s — prompt length: %d chars",
                     matched_ticker, _safe_str(row.get("item_id", ""))[:8], len(prompt))
            continue

        # Build prompt
        prompt = build_extraction_prompt(
            _safe_str(row.get("title", "")),
            _safe_str(row.get("body", "") or row.get("summary", "") or row.get("description", "")),
        )

        # Call LLM
        try:
            result = call_llm(prompt)

            pub = row.get("published_at", row.get("ingested_at"))
            if isinstance(pub, float):
                pub = None  # NaN
            published_at = pub

            # Override asset with the one matched from perimeter (LLM may miss
            # newer tokens, but our regex won't)
            llm_asset = str(result.get("asset", "UNKNOWN"))
            final_asset = matched_ticker if matched_ticker else llm_asset

            # Compute novelty against the full PIT-filtered pool
            event_for_novelty = {
                "asset": final_asset,
                "event_type": result["event_type"],
                "extracted_at": datetime.now(timezone.utc),
                "published_at": published_at,
                "content_hash": content_hash,
                "title": _safe_str(row.get("title", "")),
            }
            novelty_score = compute_novelty(event_for_novelty, pool_df)

            # Build EventRecord
            record = {
                "item_id": _safe_str(row.get("item_id", "")),
                "content_hash": content_hash,
                "model_version": MODEL_VERSION,
                "prompt_version": PROMPT_VERSION,
                "extracted_at": datetime.now(timezone.utc),
                "published_at": published_at,
                "asset": final_asset,
                "event_type": str(result["event_type"]),
                "polarity": float(result["polarity"]),
                "magnitude": float(result["magnitude"]),
                "novelty": float(novelty_score),
                "confidence": float(result["confidence"]),
                "extraction_only": bool(result["extraction_only"]),
                "title": _safe_str(row.get("title", "")),
            }
            # Save to cache (writes only EventRecord fields to parquet)
            save_to_cache(record, MODEL_VERSION, PROMPT_VERSION)
            pool_df = pd.concat(
                [pool_df, pd.DataFrame([record])], ignore_index=True
            )
            new_records.append(record)

            log.info(
                "Extracted [%s] %s: %s p=%.2f m=%.2f n=%.2f conf=%.2f",
                _safe_str(row.get("item_id", ""))[:8], final_asset, result["event_type"],
                result["polarity"], result["magnitude"], novelty_score,
                result["confidence"],
            )
        except Exception as e:
            log.error("Failed to extract %s: %s", _safe_str(row.get("item_id", ""))[:8], e)
            failed_items.append({"item_id": _safe_str(row.get("item_id", "")), "error": str(e)})

    if skipped_no_match > 0:
        log.info(
            "Skipped %d items: no ticker match in perimeter universe",
            skipped_no_match,
        )

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
            "Extraction complete: %d cache hits, %d new extractions, "
            "%d skipped (no ticker), %d total records",
            cache_hits, len(new_records), skipped_no_match, len(final_df),
        )
        return final_df
    else:
        log.info(
            "Extraction complete: %d cache hits, %d new extractions, "
            "%d skipped (no ticker), 0 total",
            cache_hits, len(new_records), skipped_no_match,
        )
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

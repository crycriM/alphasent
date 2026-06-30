"""
Extraction cache: content_hash + model_version + prompt_version key.

Append-only, immutable records. Directory sharding by first 2 chars of hash.

If a model or prompt version changes, old cached records remain valid for
their version; new records write alongside them. The feature build step
selects by (model_version, prompt_version) for consistency.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import CACHE_DIR, MODEL_VERSION, PROMPT_VERSION, RAW_NEWS_DIR

log = logging.getLogger("extraction.cache")


def compute_content_hash(title: str, body: str) -> str:
    """sha256(title + body[:500]) — cache key base."""
    truncated = (title + body[:500]).encode("utf-8")
    return hashlib.sha256(truncated).hexdigest()


def cache_key(
    content_hash: str,
    model_version: str = MODEL_VERSION,
    prompt_version: str = PROMPT_VERSION,
) -> Path:
    """Get the cache file path, sharded by first 2 chars of hash.

    Filename includes model_version and prompt_version so a version bump
    writes new records alongside (never over) old ones, and a load can only
    return a record whose version matches.
    """
    prefix = content_hash[:2]
    safe_mv = model_version.replace("/", "_")
    safe_pv = prompt_version.replace("/", "_")
    return (
        CACHE_DIR
        / "extractions"
        / prefix
        / f"{content_hash}_{safe_mv}_{safe_pv}.parquet"
    )


def cache_exists(
    content_hash: str,
    model_version: str = MODEL_VERSION,
    prompt_version: str = PROMPT_VERSION,
) -> bool:
    """Check if a cached extraction exists for this (hash, model, prompt) triple."""
    return cache_key(content_hash, model_version, prompt_version).exists()


def load_from_cache(
    content_hash: str,
    model_version: str = MODEL_VERSION,
    prompt_version: str = PROMPT_VERSION,
) -> dict | None:
    """Load a cached EventRecord for the given version triple. None if missing."""
    path = cache_key(content_hash, model_version, prompt_version)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return None
        row = df.iloc[0]
        return {
            "item_id": row["item_id"],
            "content_hash": row["content_hash"],
            "model_version": row["model_version"],
            "prompt_version": row["prompt_version"],
            "extracted_at": row["extracted_at"],
            "asset": row["asset"],
            "event_type": row["event_type"],
            "polarity": float(row["polarity"]),
            "magnitude": float(row["magnitude"]),
            "novelty": float(row["novelty"]),
            "confidence": float(row["confidence"]),
            "extraction_only": bool(row["extraction_only"]),
        }
    except Exception as e:
        log.warning("Failed to load cache for %s: %s", content_hash, e)
        return None


def save_to_cache(
    record: dict,
    model_version: str = MODEL_VERSION,
    prompt_version: str = PROMPT_VERSION,
) -> None:
    """Append an EventRecord to the versioned cache. Creates directories as needed."""
    path = cache_key(record["content_hash"], model_version, prompt_version)
    path.parent.mkdir(parents=True, exist_ok=True)

    # If file exists, load and append (shouldn't happen for unique hashes, but be safe)
    if path.exists():
        existing = pd.read_parquet(path)
        new_df = pd.DataFrame([record])
        combined = pd.concat([existing, new_df], ignore_index=True)
        # Dedup on content_hash+model+prompt to keep store semantics consistent
        combined = combined.drop_duplicates(
            subset=["content_hash", "model_version", "prompt_version"], keep="last"
        )
    else:
        combined = pd.DataFrame([record])

    # Convert extracted_at to UTC datetime if it's a string
    if isinstance(combined["extracted_at"].iloc[0], str):
        combined["extracted_at"] = pd.to_datetime(combined["extracted_at"], utc=True)

    table = pa.Table.from_pandas(combined, preserve_index=False)
    pq.write_table(table, path)
    log.info("Cached extraction for %s -> %s", record["content_hash"][:8], path)


def load_all_cached(
    model_version: str = MODEL_VERSION,
    prompt_version: str = PROMPT_VERSION,
    cache_dir: Path = CACHE_DIR,
) -> pd.DataFrame:
    """Load every cached EventRecord for the given version triple in one pass.

    Returns an empty DataFrame if none exist. Used by the batch ETL to avoid
    per-row parquet reads and to pre-seed the novelty pool.
    """
    safe_mv = model_version.replace("/", "_")
    safe_pv = prompt_version.replace("/", "_")
    pattern = f"*_{safe_mv}_{safe_pv}.parquet"
    files = sorted((cache_dir / "extractions").glob(f"*/{pattern}"))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_parquet(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)
    log.info("Loaded %d cached extractions for %s/%s", len(combined), model_version, prompt_version)
    return combined

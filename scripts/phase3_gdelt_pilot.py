#!/usr/bin/env python3
"""
Phase 3: GDELT body backfill pilot.

Loads ~2K GDELT URLs, fetches bodies via trafilatura, extracts title+body,
updates the raw news parquet files, then re-runs extraction to measure
impact on rare-event flag rates.

Plan: https://github.com/alphasent/alphasent/blob/main/ALPHA_EVOLUTION_PLAN.md#phase-3--gdelt-body-backfill-rec-3--bounded-pilot-not-the-full-340k
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import trafilatura

from src.config import RAW_NEWS_DIR
from src.ingest.article_fetcher import fetch_bodies_async

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("phase3_gdelt_pilot")

# --- Configuration ---
PILOT_SIZE = 200  # Reduced from 2000 for faster testing
SAMPLE_FILES = 5  # Sample from 5 files


def load_gdelt_sample() -> pd.DataFrame:
    """Load a sample of GDELT records with empty title/body."""
    news_files = sorted(RAW_NEWS_DIR.glob("*.parquet"))
    if not news_files:
        raise ValueError(f"No GDELT files found in {RAW_NEWS_DIR}")
    
    # Sample from first N files to get diverse URLs
    sample_files = news_files[:SAMPLE_FILES]
    log.info("Sampling from %d GDELT files", len(sample_files))
    
    frames = []
    for f in sample_files:
        df = pd.read_parquet(f)
        # Filter to GDELT records with empty title/body
        if "source" in df.columns:
            df = df[df["source"] == "gdelt"]
        df = df[(df["title"] == "") | (df["body"] == "")]
        frames.append(df)
    
    if not frames:
        raise ValueError("No GDELT records with empty title/body found")
    
    combined = pd.concat(frames, ignore_index=True)
    log.info("Total GDELT records with empty title/body: %d", len(combined))
    
    # Sample PILOT_SIZE records
    if len(combined) > PILOT_SIZE:
        combined = combined.sample(PILOT_SIZE, random_state=42)
    
    log.info("Pilot sample: %d URLs", len(combined))
    return combined


def fetch_and_extract(urls: list[str]) -> list[tuple[str, str, int]]:
    """Fetch URLs and extract title+body.
    
    Returns list of (title, body, status) tuples.
    """
    log.info("Fetching %d URLs...", len(urls))
    results = fetch_bodies_async(urls)
    
    extracted = []
    for url, (body, status) in zip(urls, results):
        if status != 200 or not body:
            extracted.append(("", "", status))
            continue
        
        # Fetch HTML for title extraction
        try:
            html = trafilatura.fetch_url(url)
            if html:
                # Use bare_extraction to get metadata
                doc = trafilatura.bare_extraction(
                    html,
                    include_comments=False,
                    include_tables=False,
                    url=url,
                )
                # Access attributes directly (Document object, not dict)
                title = doc.title if doc and hasattr(doc, 'title') else ""
                extracted.append((title or "", body, 200))
            else:
                extracted.append(("", body, 200))
        except Exception as e:
            log.warning("Failed to extract title from %s: %s", url, e)
            extracted.append(("", body, 200))
    
    return extracted


def update_gdelt_records(
    pilot_df: pd.DataFrame,
    extracted: list[tuple[str, str, int]],
) -> None:
    """Update GDELT parquet files with fetched title/body."""
    pilot_df = pilot_df.copy()
    
    # Use URL as fallback title if extraction failed
    titles = []
    for i, (t, b, s) in enumerate(extracted):
        if t:
            titles.append(t)
        elif b:
            # Use URL as fallback title
            titles.append(pilot_df.iloc[i]["url"])
        else:
            titles.append("")
    
    pilot_df["title"] = titles
    pilot_df["body"] = [b for _, b, _ in extracted]
    pilot_df["fetch_status"] = [s for _, _, s in extracted]
    
    # Group by date and update each file
    pilot_df["date"] = pd.to_datetime(pilot_df["published_at"]).dt.date
    
    for date_str, group in pilot_df.groupby("date"):
        file_path = RAW_NEWS_DIR / f"{date_str}.parquet"
        if not file_path.exists():
            log.warning("File not found: %s", file_path)
            continue
        
        existing = pd.read_parquet(file_path)
        
        # Update matching records by item_id
        for _, row in group.iterrows():
            mask = existing["item_id"] == row["item_id"]
            if mask.any():
                existing.loc[mask, "title"] = row["title"]
                existing.loc[mask, "body"] = row["body"]
                existing.loc[mask, "fetch_status"] = row["fetch_status"]
        
        existing.to_parquet(file_path)
        log.info("Updated %d records in %s", len(group), file_path)


def measure_rare_event_rate() -> dict:
    """Measure rare-event flag rate in existing features."""
    from src.config import FEATURES_DIR
    
    # Load all feature files
    frames = []
    for asset_dir in sorted(FEATURES_DIR.iterdir()):
        if not asset_dir.is_dir():
            continue
        for parquet_file in sorted(asset_dir.glob("*.parquet")):
            df = pd.read_parquet(parquet_file)
            df["asset"] = asset_dir.name
            frames.append(df)
    
    if not frames:
        return {}
    
    features = pd.concat(frames, ignore_index=True)
    
    # Count rare-event flags
    flag_cols = ["hack_flag", "regulation_flag", "listing_flag", "depeg_flag"]
    flag_counts = {col: int(features[col].sum()) for col in flag_cols if col in features.columns}
    total_flags = sum(flag_counts.values())
    total_rows = len(features)
    
    return {
        "total_rows": total_rows,
        "flag_counts": flag_counts,
        "total_flags": total_flags,
        "flag_rate": total_flags / total_rows if total_rows > 0 else 0,
    }


def main():
    """Run Phase 3 pilot."""
    log.info("=== Phase 3: GDELT Body Backfill Pilot ===")
    
    # Measure baseline rare-event rate
    log.info("Measuring baseline rare-event rate...")
    baseline = measure_rare_event_rate()
    log.info("Baseline: %d total flags in %d rows (%.2f%% flag rate)",
             baseline["total_flags"], baseline["total_rows"], baseline["flag_rate"] * 100)
    log.info("Flag breakdown: %s", baseline["flag_counts"])
    
    # Load pilot sample
    pilot_df = load_gdelt_sample()
    urls = pilot_df["url"].tolist()
    
    # Fetch and extract
    extracted = fetch_and_extract(urls)
    
    # Measure fetch success
    success_count = sum(1 for _, _, s in extracted if s == 200)
    title_count = sum(1 for t, _, _ in extracted if t)
    body_count = sum(1 for _, b, _ in extracted if b)
    
    log.info("Fetch results:")
    log.info("  Success (200): %d/%d (%.1f%%)", success_count, len(urls), success_count / len(urls) * 100)
    log.info("  Title extracted: %d/%d (%.1f%%)", title_count, len(urls), title_count / len(urls) * 100)
    log.info("  Body extracted: %d/%d (%.1f%%)", body_count, len(urls), body_count / len(urls) * 100)
    
    # Update GDELT records
    log.info("Updating GDELT records...")
    update_gdelt_records(pilot_df, extracted)
    
    log.info("=== Pilot Complete ===")
    log.info("Next steps:")
    log.info("1. Re-run extraction pipeline: python scripts/run_text_pipeline.py")
    log.info("2. Re-run Phase 1 backtest: python scripts/phase1_rare_events.py")
    log.info("3. Compare rare-event flag rates before/after")
    log.info("4. If improvement > 10%%, scale to full 340K URLs")


if __name__ == "__main__":
    main()

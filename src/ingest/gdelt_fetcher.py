"""
GDELT BigQuery backfill fetcher.

Queries GDELT GKG for crypto-relevant articles, returns RawNewsItem records
with crawl_ts as the point-in-time anchor.

BigQuery cost discipline:
  - Always filters by _PARTITIONTIME
  - Domain allowlist to limit rows
  - Runs in batches to stay in free tier
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import RAW_NEWS_DIR
from src.schemas.raw import RawNewsItem

log = logging.getLogger("ingest.gdelt")

GDELT_QUERY = """
SELECT
    PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING)) AS crawl_ts,
    DocumentIdentifier AS url,
    SourceCommonName AS source_name,
    V2Themes AS themes,
    V2Organizations AS orgs,
    V2Persons AS persons,
    CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64) AS doc_tone
FROM gdelt-bq.gdeltv2.gkg_partitioned
WHERE
    DATE(_PARTITIONTIME) BETWEEN @start_date AND @end_date
    AND (
        REGEXP_CONTAINS(V2Themes, 'ECON_BITCOIN')
        OR REGEXP_CONTAINS(V2Themes, 'ECON_CRYPTOCURRENCY')
        OR REGEXP_CONTAINS(DocumentIdentifier,
            r'coindesk\\.com|cointelegraph\\.com|decrypt\\.co|'
            r'theblock\\.co|bitcoinmagazine\\.com|cryptoslate\\.com')
    )
"""

# GDELT project ID; requires google-cloud-bigquery client library and credentials
GDELT_PROJECT_ID = os.getenv("GDELT_PROJECT_ID", "endless-empire-498816-j2")
GDELT_DATASET = os.getenv("GDELT_DATASET", "gdeltv2")
GDELT_TABLE = os.getenv("GDELT_TABLE", "gkg_partitioned")

# Asset mention lookup table
ASSET_MENTION_MAP = {
    "bitcoin": "BTC",
    "btc": "BTC",
    "ethereum": "ETH",
    "eth": "ETH",
    "ripple": "XRP",
    "xrp": "XRP",
    "solana": "SOL",
    "sol": "SOL",
    "cardano": "ADA",
    "ada": "ADA",
    "polkadot": "DOT",
    "dot": "DOT",
    "chainlink": "LINK",
    "link": "LINK",
    "binance": "BNB",
    "bnb": "BNB",
    "avalanche": "AVAX",
    "avax": "AVAX",
    "polygon": "MATIC",
    "matic": "MATIC",
    "dogecoin": "DOGE",
    "doge": "DOGE",
    "shiba": "SHIB",
    "shiba": "SHIB",
    "terra": "LUNA",
    "luna": "LUNA",
    "usd": "USD",
    "tether": "USDT",
    "usdt": "USDT",
    "usdc": "USDC",
    "circle": "USDC",
}

def _make_item_id(url: str, date_str: str) -> str:
    """sha256(url + date) — stable dedup key."""
    return hashlib.sha256((url + date_str).encode("utf-8")).hexdigest()

def _extract_asset_mentions(orgs: str, persons: str) -> list[str]:
    """Extract asset tickers from GDELT V2Organizations / V2Persons fields."""
    text = f"{orgs} {persons}".lower()
    found: set[str] = set()
    for key, ticker in ASSET_MENTION_MAP.items():
        if key in text:
            found.add(ticker)
    return sorted(found)

def _to_raw_news_item(row: pd.Series) -> RawNewsItem:
    """Convert a GDELT row to a RawNewsItem."""
    crawl_ts = row["crawl_ts"]
    if not isinstance(crawl_ts, pd.Timestamp):
        crawl_ts = pd.Timestamp(crawl_ts)
    if hasattr(crawl_ts, 'tz') and crawl_ts.tz is None:
        crawl_ts = crawl_ts.tz_localize("UTC")
    elif hasattr(crawl_ts, 'tz'):
        crawl_ts = crawl_ts.tz_convert("UTC")
    url = str(row["url"])
    date_str = crawl_ts.strftime("%Y-%m-%d")
    orgs = str(row.get("orgs", ""))
    persons = str(row.get("persons", ""))
    return RawNewsItem(
        item_id=_make_item_id(url, date_str),
        source="gdelt",
        url=url,
        source_domain=str(row.get("source_name", "")),
        title="",  # GDELT does not carry the title; body fetch fills it
        body="",
        published_at=crawl_ts,
        ingested_at=crawl_ts,
        asset_mentions=_extract_asset_mentions(orgs, persons),
        raw_tone=float(row["doc_tone"]) if pd.notna(row.get("doc_tone")) else None,
        fetch_status=200,
    )

def fetch_gdelt_batch(
    start_date: str,
    end_date: str,
    project_id: str = GDELT_PROJECT_ID,
) -> pd.DataFrame:
    """Fetch a batch of GDELT rows for a date range.

    Args:
        start_date: YYYY-MM-DD
        end_date: YYYY-MM-DD
        project_id: BigQuery project ID

    Returns:
        DataFrame of raw GDELT rows. Empty if BigQuery is not available.
    """
    try:
        from google.cloud import bigquery  # type: ignore
    except ImportError:
        log.error(
            "google-cloud-bigquery not installed. "
            "Install it with: pip install google-cloud-bigquery"
        )
        return pd.DataFrame()

    if start_date > end_date:
        log.error("start_date (%s) must be before end_date (%s)", start_date, end_date)
        return pd.DataFrame()

    client = bigquery.Client(project=project_id)
    query = GDELT_QUERY
    query_params = [
        bigquery.ScalarQueryParameter("start_date", "STRING", start_date),
        bigquery.ScalarQueryParameter("end_date", "STRING", end_date),
    ]
    query_config = bigquery.QueryJobConfig(
        query_parameters=query_params,
    )
    try:
        df = client.query(query, job_config=query_config).to_dataframe()
        log.info(
            "Fetched %d GDELT rows for %s → %s",
            len(df), start_date, end_date,
        )
    except Exception as e:
        log.error("BigQuery query failed: %s", e)
        return pd.DataFrame()
    return df

def write_gdelt_records(
    items: list[RawNewsItem],
    output_dir: Path = RAW_NEWS_DIR,
) -> None:
    """Write RawNewsItem records to day-partitioned parquet files.

    Append-only: deduplicates by item_id within each day file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    by_date: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        date_str = item.published_at.strftime("%Y-%m-%d")
        by_date.setdefault(date_str, []).append(item.model_dump())

    for date_str, records in by_date.items():
        part_path = output_dir / f"{date_str}.parquet"
        new_df = pd.DataFrame(records)

        if part_path.exists():
            existing = pd.read_parquet(part_path)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["item_id"], keep="last")
        else:
            combined = new_df

        table = pa.Table.from_pandas(combined, preserve_index=False)
        pq.write_table(table, part_path)
        log.info("Wrote %d GDELT records to %s", len(combined), part_path)

def backfill_gdelt(
    start_date: str = "2024-01-01",
    end_date: str | None = None,
    batch_days: int = 30,
    output_dir: Path = RAW_NEWS_DIR,
    dry_run: bool = False,
) -> None:
    """Run the full GDELT backfill in batches.

    Args:
        start_date: Start date (YYYY-MM-DD). Defaults to 30 days ago.
        end_date: End date (YYYY-MM-DD). Defaults to yesterday.
        batch_days: Number of days per batch (keeps within free tier).
        dry_run: If True, log what would be fetched without running queries.
    """
    if end_date is None:
        end_date = (datetime.now(timezone.utc).date() - pd.Timedelta(days=1)).isoformat()

    start = datetime.fromisoformat(start_date).date()
    end = datetime.fromisoformat(end_date).date()

    if start > end:
        raise ValueError(f"start_date ({start}) must be before end_date ({end})")

    if (end - start).days > 60:
        log.warning(
            "Backfill spans %d days — this may consume significant BigQuery quota. "
            "Use dry_run=True to preview, or narrow the date range.",
            (end - start).days,
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    current = start
    while current < end:
        batch_end = min(current + pd.Timedelta(days=batch_days), end)
        batch_start_str = current.isoformat()
        batch_end_str = batch_end.isoformat()

        log.info("GDELT batch: %s → %s", batch_start_str, batch_end_str)
        if dry_run:
            log.info("DRY RUN: would fetch GDELT rows for %s → %s", batch_start_str, batch_end_str)
            current = batch_end
            continue
        df = fetch_gdelt_batch(batch_start_str, batch_end_str)

        if df.empty:
            log.warning("No GDELT rows for %s → %s", batch_start_str, batch_end_str)
            current = batch_end
            continue

        # Convert to RawNewsItem (body will be fetched separately)
        items = []
        for _, row in df.iterrows():
            try:
                item = _to_raw_news_item(row)
                items.append(item)
            except Exception as e:
                log.warning("Failed to convert GDELT row: %s", e)

        if items:
            write_gdelt_records(items, output_dir)

        current = batch_end

    log.info("GDELT backfill complete for %s → %s", start_date, end_date)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    backfill_gdelt()

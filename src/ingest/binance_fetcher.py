"""
Binance OHLCV fetcher.

Paginates over the full date range in 1000-bar chunks.
Writes one parquet file per symbol per interval.
One-time backfill plus daily incremental update.

Rate limits: 1200 requests/minute per IP (unauthenticated).
A full 3-year hourly history for one symbol requires ~25 API calls.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import BINANCE_BASE_URL, BINANCE_INTERVALS, BINANCE_LIMIT, BINANCE_SYMBOLS, RAW_OHLCV_DIR

log = logging.getLogger("ingest.binance")

# Retry policy for transient Binance API errors (429/5xx/timeouts).
_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 1.0  # seconds


def fetch_all_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    base_url: str = BINANCE_BASE_URL,
    limit: int = BINANCE_LIMIT,
) -> list[dict]:
    """Fetch all kline (OHLCV) bars for a symbol/interval in 1000-bar chunks.

    Args:
        symbol: e.g. "BTCUSDT"
        interval: "1m", "5m", "15m", "1h", "4h", "1d"
        start_ms: Unix timestamp in milliseconds
        end_ms: Unix timestamp in milliseconds
        base_url: Binance API base URL
        limit: Max bars per request (max 1000)

    Returns:
        List of raw kline dicts from the API.
    """
    all_bars = []
    cursor = start_ms

    with httpx.Client(timeout=30.0) as client:
        while cursor < end_ms:
            data = None
            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    resp = client.get(
                        f"{base_url}/klines",
                        params={
                            "symbol": symbol,
                            "interval": interval,
                            "startTime": cursor,
                            "endTime": end_ms,
                            "limit": limit,
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except httpx.HTTPError as e:
                    log.warning(
                        "Binance API error for %s/%s at cursor %d (attempt %d/%d): %s",
                        symbol, interval, cursor, attempt, _MAX_RETRIES, e,
                    )
                    if attempt < _MAX_RETRIES:
                        time.sleep(_RETRY_BACKOFF_BASE * (2 ** (attempt - 1)))
                    data = None

            if data is None:
                # All retries exhausted — stop pagination rather than silently
                # returning partial data as if complete.
                log.error(
                    "Binance fetch aborted for %s/%s at cursor %d after %d retries; "
                    "returning %d bars fetched so far (INCOMPLETE).",
                    symbol, interval, cursor, _MAX_RETRIES, len(all_bars),
                )
                break

            if not data:
                break

            all_bars.extend(data)

            # Move cursor past the last bar's close time + 1ms
            cursor = data[-1][6] + 1  # index 6 = close_time_ms

            log.debug("Fetched %d bars for %s/%s (total so far: %d)",
                      len(data), symbol, interval, len(all_bars))

            # Small delay to respect rate limits
            time.sleep(0.2)  # well under 1200 req/min

    log.info("Total: %d bars for %s/%s", len(all_bars), symbol, interval)
    return all_bars


def parse_klines(raw_bars: list[list] | list[dict], symbol: str, interval: str) -> pd.DataFrame:
    """Convert raw Binance kline arrays to a DataFrame.

    Binance kline format:
    [
        open_time_ms,       # 0
        open,               # 1
        high,               # 2
        low,                # 3
        close,              # 4
        volume,             # 5
        close_time_ms,      # 6
        quote_volume,       # 7
        n_trades,           # 8
        taker_buy_base,     # 9
        taker_buy_quote,    # 10
        ignore              # 11
    ]
    """
    records = []
    for bar in raw_bars:
        records.append({
            "symbol": symbol,
            "interval": interval,
            "open_time": pd.Timestamp(bar[0], unit="ms", tz="UTC"),
            "close_time": pd.Timestamp(bar[6], unit="ms", tz="UTC"),
            "open": float(bar[1]),
            "high": float(bar[2]),
            "low": float(bar[3]),
            "close": float(bar[4]),
            "volume": float(bar[5]),
            "quote_volume": float(bar[7]),
            "n_trades": int(bar[8]),
            "taker_buy_base": float(bar[9]),
            "taker_buy_quote": float(bar[10]),
        })

    return pd.DataFrame(records)


def write_ohlcv(
    df: pd.DataFrame,
    symbol: str,
    interval: str,
    ohlcv_dir: Path = RAW_OHLCV_DIR,
) -> None:
    """Write OHLCV DataFrame to parquet. Appends if file exists."""
    ohlcv_dir.mkdir(parents=True, exist_ok=True)
    part_path = ohlcv_dir / f"{symbol}_{interval}.parquet"

    if part_path.exists():
        existing = pd.read_parquet(part_path)
        combined = pd.concat([existing, df], ignore_index=True)
        # Remove duplicates (same open_time)
        combined = combined.drop_duplicates(subset=["open_time", "symbol"], keep="last")
    else:
        combined = df

    combined = combined.sort_values("open_time").reset_index(drop=True)
    table = pa.Table.from_pandas(combined, preserve_index=False)
    # Atomic write: write to a temp file then rename, so an interrupted write
    # cannot corrupt the existing historical file.
    tmp_path = part_path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp_path)
    os.replace(tmp_path, part_path)
    log.info("Wrote %d OHLCV bars to %s", len(combined), part_path)


def backfill_all(
    symbols: list[str] | None = None,
    intervals: list[str] | None = None,
    start_date: str = "2020-01-01",
    end_date: str | None = None,
) -> None:
    """Backfill OHLCV data for all configured symbols and intervals.

    Args:
        symbols: Override symbols list (default: BINANCE_SYMBOLS).
        intervals: Override intervals list (default: BINANCE_INTERVALS).
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD). Defaults to yesterday (UTC) so an
            unguarded invocation does not silently fetch through "now" and rewrite
            existing files in place.
    """
    symbols = symbols or BINANCE_SYMBOLS
    intervals = intervals or BINANCE_INTERVALS

    if end_date is None:
        end_date = (datetime.now(timezone.utc).date()).isoformat()

    start_ms = int(datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc).timestamp() * 1000)

    for symbol in symbols:
        for interval in intervals:
            log.info("Fetching %s %s %s...", symbol, interval, start_date)
            raw = fetch_all_klines(symbol, interval, start_ms, end_ms)
            if raw:
                df = parse_klines(raw, symbol, interval)
                write_ohlcv(df, symbol, interval)
            else:
                log.warning("No data returned for %s %s", symbol, interval)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    backfill_all()


if __name__ == "__main__":
    main()

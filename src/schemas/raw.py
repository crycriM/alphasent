"""
Layer-1 schemas: RawNewsItem and RawOHLCVBar.

Enforced at every layer boundary via Pydantic -> PyArrow.
Append-only writes; existing files are never overwritten.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------- #
# RawNewsItem — unified schema covering both RSS and CryptoPanic sources
# --------------------------------------------------------------------------- #


class RawNewsItem(BaseModel):
    """One normalized news item. Point-in-time anchor never modified after write."""

    item_id: str  # stable dedup key: sha256(url + date) or cryptopanic:{post_id}
    source: str  # "rss" | "cryptopanic"
    url: str
    source_domain: str = ""
    title: str
    body: str = ""  # trafilatura-extracted body; empty if failed
    published_at: datetime  # GDELT crawl_ts or CryptoPanic published_at — UTC
    ingested_at: datetime  # when WE fetched it (UTC) — true availability
    asset_mentions: list[str] = Field(default_factory=list)  # ["BTC","ETH"]
    raw_tone: Optional[float] = None  # GDELT V2Tone[0]; None for RSS/CryptoPanic
    fetch_status: int = 200  # HTTP status; -1=timeout; -2=extraction_empty

    @field_validator("published_at", "ingested_at")
    @classmethod
    def _ensure_utc(cls, v: datetime) -> datetime:
        return v.astimezone(timezone.utc) if v.tzinfo else v.replace(tzinfo=timezone.utc)

    @field_validator("asset_mentions")
    @classmethod
    def _sort_assets(cls, v: list[str]) -> list[str]:
        return sorted(set(v))


def make_item_id(url: str, date_str: str) -> str:
    """sha256(url + date) — stable dedup key for non-CryptoPanic sources."""
    return hashlib.sha256((url + date_str).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# RawOHLCVBar — Binance kline data
# --------------------------------------------------------------------------- #


class RawOHLCVBar(BaseModel):
    """One OHLCV bar from Binance. Never modified after write."""

    symbol: str  # e.g. "BTCUSDT"
    interval: str  # "1m", "5m", "15m", "1h", "4h", "1d"
    open_time: datetime  # bar open timestamp — UTC
    close_time: datetime  # bar close timestamp — UTC
    open: float
    high: float
    low: float
    close: float
    volume: float  # base asset volume
    quote_volume: float  # quote asset volume
    n_trades: int  # number of trades
    taker_buy_base: float = 0.0
    taker_buy_quote: float = 0.0

    @field_validator("open_time", "close_time")
    @classmethod
    def _ensure_utc(cls, v: datetime) -> datetime:
        return v.astimezone(timezone.utc) if v.tzinfo else v.replace(tzinfo=timezone.utc)

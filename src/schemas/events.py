"""
Layer-2 schemas: EventType enum and EventRecord.

The LLM extraction ETL writes EventRecords keyed by
(content_hash, model_version, prompt_version).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------- #
# EventType — the eight-category taxonomy
# --------------------------------------------------------------------------- #


class EventType(str, Enum):
    """Typed event categories. Fixed a-priori; versioned alongside prompt."""
    LISTING = "listing"       # Exchange lists or delists a token
    HACK = "hack"            # Exploit, bridge attack, theft
    REGULATION = "regulation" # SEC/CFTC/gov action, ruling, ban
    PARTNERSHIP = "partnership"  # Integration, deal, institutional adoption
    DEPEG = "depeg"          # Stablecoin or peg failure
    MACRO = "macro"          # Fed, rates, USD, risk-off macro event
    FORK = "fork"            # Protocol upgrade, hard fork
    OTHER = "other"          # Doesn't fit above categories


# --------------------------------------------------------------------------- #
# EventRecord — LLM-extracted structured event
# --------------------------------------------------------------------------- #


class EventRecord(BaseModel):
    """One LLM-extracted event record. Immutable after write."""

    item_id: str  # FK -> RawNewsItem.item_id
    content_hash: str  # sha256(title + body[:500]) — cache key
    model_version: str  # e.g. "llama3-8b-q4km-v1"
    prompt_version: str  # semver: "1.0.0"
    extracted_at: datetime  # wall-clock time of extraction run
    published_at: datetime  # point-in-time anchor (GDELT crawl_ts / CP published_at)
    asset: str  # primary asset ticker: "BTC"
    event_type: EventType
    polarity: float  # [-1.0, 1.0] negative=bad for asset, positive=good
    magnitude: float  # [0.0, 1.0] estimated market impact
    novelty: float  # [0.0, 1.0] computed externally (TF-IDF)
    confidence: float  # [0.0, 1.0] model self-report
    extraction_only: bool  # True if event_type/polarity determinable from text alone
                           # False if model had to use outside knowledge

    @field_validator("extracted_at", "published_at")
    @classmethod
    def _ensure_utc(cls, v: datetime) -> datetime:
        return v.astimezone(timezone.utc) if v.tzinfo else v.replace(tzinfo=timezone.utc)

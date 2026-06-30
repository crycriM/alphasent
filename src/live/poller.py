from __future__ import annotations

import json
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import (
    CACHE_DIR,
    FEATURES_DIR,
    LLM_BASE_URL,
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
from src.features.builder import build_features_for_asset
from src.features.store import write_features
from snippets.cryptopanic_ingest import (
    RateLimiter,
    _build_first_url,
    fetch_pages,
    normalize_post,
    IngestState,
)

log = logging.getLogger("live.poller")

_STATE_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "cryptopanic" / "state.json"
_POLL_INTERVAL = 300
_RAW_NEWS_OUT = RAW_NEWS_DIR.parent / "cryptopanic" / "normalized"
LLM_AVAILABLE = True


def _check_llm() -> bool:
    import httpx
    try:
        r = httpx.get(f"{LLM_BASE_URL}/chat/completions", timeout=5)
        return r.status_code < 500 or r.status_code == 404
    except Exception:
        return False


def _poll_new_posts(token: str, plan: str = "developer") -> list[dict]:
    state = IngestState.load(_STATE_FILE)
    first_url = _build_first_url(plan, token, None)
    limiter = RateLimiter(1.0)
    new = []
    import httpx
    with httpx.Client(headers={"User-Agent": "alphasent-poller/1.0"}) as client:
        for page in fetch_pages(client, first_url, limiter, state.last_post_id, max_pages=5):
            for post in page:
                if int(post["id"]) > state.last_post_id:
                    new.append(post)
    return new


def _extract_single(raw_row: dict, pool_df: pd.DataFrame) -> dict | None:
    ingested_at = datetime.now(timezone.utc)
    norm = normalize_post(raw_row, ingested_at=ingested_at)
    title = norm.title
    body = norm.description
    content_hash = compute_content_hash(title, body)

    if cache_exists(content_hash, MODEL_VERSION, PROMPT_VERSION):
        cached = load_from_cache(content_hash, MODEL_VERSION, PROMPT_VERSION)
        if cached:
            cached["title"] = title
            cached["published_at"] = norm.published_at
            return cached

    prompt = build_extraction_prompt(title, body)
    try:
        result = call_llm(prompt)
    except RuntimeError:
        return None

    published_at = norm.published_at
    event_for_novelty = {
        "asset": result["asset"],
        "event_type": result["event_type"],
        "extracted_at": datetime.now(timezone.utc),
        "published_at": published_at,
        "content_hash": content_hash,
        "title": title,
    }
    novelty_score = compute_novelty(event_for_novelty, pool_df)

    record = {
        "item_id": norm.item_id,
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
        "title": title,
    }
    save_to_cache(record, MODEL_VERSION, PROMPT_VERSION)
    return record


def run_once(
    token: str | None = None,
    plan: str = "developer",
    update_features: bool = True,
) -> dict:
    import os
    token = token or os.getenv("CRYPTOPANIC_AUTH_TOKEN", "")
    if not token:
        log.warning("No CRYPTOPANIC_AUTH_TOKEN — skipping poll")
        return {"new_items": 0, "extracted": 0, "cached": 0, "errors": 0}

    new_posts = _poll_new_posts(token, plan)
    if not new_posts:
        return {"new_items": 0, "extracted": 0, "cached": 0, "errors": 0}

    pool_df = load_all_cached(MODEL_VERSION, PROMPT_VERSION)

    extracted = 0
    cached = 0
    errors = 0
    event_records = []

    for raw_row in new_posts:
        rec = _extract_single(raw_row, pool_df)
        if rec is None:
            errors += 1
            continue
        if "content_hash" in rec and cache_exists(rec["content_hash"], MODEL_VERSION, PROMPT_VERSION):
            cached += 1
        else:
            extracted += 1
        event_records.append(rec)
        if not pool_df.empty and "published_at" in pool_df.columns:
            pool_df = pd.concat([pool_df, pd.DataFrame([rec])], ignore_index=True)
        else:
            pool_df = pd.DataFrame([rec]) if pool_df.empty else pd.concat([pool_df, pd.DataFrame([rec])], ignore_index=True)

    if update_features and event_records:
        events_df = pd.DataFrame(event_records)
        events_df["published_at"] = pd.to_datetime(events_df["published_at"], utc=True)
        assets = events_df["asset"].dropna().unique().tolist()
        asset_map = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "XRP": "XRPUSDT"}
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        from src.ingest.binance_fetcher import fetch_all_klines, parse_klines
        for asset in assets:
            symbol = asset_map.get(asset, f"{asset}USDT")
            try:
                end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                start_ms = end_ms - 7 * 24 * 3600 * 1000
                raw = fetch_all_klines(symbol, "1h", start_ms, end_ms)
                if raw:
                    ohlcv_df = parse_klines(raw, symbol, "1h")
                    features_df = build_features_for_asset(events_df, ohlcv_df, asset, symbol)
                    if not features_df.empty:
                        write_features(features_df, asset, today)
            except Exception as e:
                log.error("Feature build failed for %s: %s", asset, e)

    return {"new_items": len(new_posts), "extracted": extracted, "cached": cached, "errors": errors}


def run_loop(interval_seconds: int = _POLL_INTERVAL):
    import os
    global LLM_AVAILABLE
    LLM_AVAILABLE = _check_llm()
    if not LLM_AVAILABLE:
        log.warning("LLM unavailable at %s — running in cache-only mode", LLM_BASE_URL)

    stop_event = signal.getsignal(signal.SIGINT)
    def _shutdown(signum, frame):
        log.info("Shutting down poller...")
        sys.exit(0)
    signal.signal(signal.SIGINT, _shutdown)

    token = os.getenv("CRYPTOPANIC_AUTH_TOKEN", "")
    if not token:
        log.error("CRYPTOPANIC_AUTH_TOKEN not set, exiting")
        return

    log.info("Starting poller loop, interval=%ds", interval_seconds)
    cycle = 0
    while True:
        cycle += 1
        log.info("=== Poll cycle %d ===", cycle)
        try:
            summary = run_once(token=token)
            log.info("Cycle %d summary: %s", cycle, json.dumps(summary))
        except Exception as e:
            log.error("Poll cycle %d failed: %s", cycle, e)

        if hasattr(sys.modules.get("live.monitoring", None), "run_all_checks"):
            from src.live.monitoring import run_all_checks
            try:
                report = run_all_checks()
                log.info("Monitoring report: %s", json.dumps(report))
            except Exception as e:
                log.error("Monitoring failed: %s", e)

        time.sleep(interval_seconds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    run_loop()

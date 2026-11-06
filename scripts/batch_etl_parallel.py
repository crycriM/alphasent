#!/usr/bin/env python3
"""Parallel batch extraction driver — same semantics as batch_etl, ~Nx faster.

Why this exists
---------------
`src/extraction/batch_etl.run_batch_extraction` is single-threaded: one LLM call
at a time (~5s each on the loaded phi4), so a full study refresh of ~19K items
costs ~26h of wall clock. The phi4 child on the router serves 4 slots, so the
LLM phase can run 4-wide while everything else stays bit-identical.

How the semantics are preserved (this matters — the feature store feeds a study)
-------------------------------------------------------------------------------
Three passes:

  1. classify (sequential, CPU only): replay the cache check and the
     point-in-time perimeter pre-filter, producing the list of items that need
     an LLM call. No LLM traffic.
  2. extract (parallel, N workers): build the prompt and call the LLM for each
     pending item, checkpointing each result to a JSONL file keyed by
     content_hash. Resumable: a re-run skips hashes already in the checkpoint.
  3. finalize (sequential, CPU only): walk the results in published_at order and
     score novelty exactly as batch_etl does, then write each record to the
     versioned cache.

Novelty equivalence: batch_etl computes novelty against a pool of every cached
record, masked to `asset == a` and `published_at in [t-48h, t)`. That mask makes
the result independent of pool insertion order, so pass 3 keeps a 48h trailing
deque instead of re-masking the whole pool on every item — same value, O(n)
instead of O(n^2). scripts/test_parallel_novelty_equiv.py proves it on real data.

The novelty SCORE is computed by the shipped compute_novelty(); this file never
reimplements it, it only hands it a window that provably contains every row the
mask would keep.

Usage
-----
  .venv/bin/python scripts/batch_etl_parallel.py --limit 40        # smoke test
  .venv/bin/python scripts/batch_etl_parallel.py --workers 4       # full refresh
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import CACHE_DIR, MODEL_VERSION, PROMPT_VERSION
from src.extraction.batch_etl import (
    _item_day,
    _safe_str,
    read_all_raw_news,
    universe_for_day,
)
from src.extraction.cache import (
    cache_exists,
    compute_content_hash,
    load_all_cached,
    save_to_cache,
)
from src.extraction.model import call_llm
from src.extraction.novelty import compute_novelty
from src.extraction.prompt import build_extraction_prompt
from src.ingest.perimeter import match_ticker

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
# call_llm logs p=%.2f on a result whose polarity can legitimately be null, and
# its logger raises inside the handler for every such response. The driver
# handles those records as failures further down, so keep the noise out of a
# multi-hour log: one line per record is not worth 470 tracebacks.
for noisy in ("extraction.model", "extraction.cache", "ingest.perimeter"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("batch_etl_parallel")

DEFAULT_STATE = CACHE_DIR / "parallel_extract" / f"{MODEL_VERSION}.jsonl"
LOOKBACK_HOURS = 48          # mirrors EXTRACTION_LOOKBACK_HOURS
_stop = threading.Event()


def _sigint(_sig, _frm):
    log.warning("interrupt received — finishing in-flight work, then stopping "
                "(checkpoint is kept, re-run to resume)")
    _stop.set()


# --------------------------------------------------------------------------- #
# pass 1 — classify
# --------------------------------------------------------------------------- #

def classify(since: date | None = None) -> tuple[list[dict], pd.DataFrame, dict]:
    """Return (pending items, cached pool, counters). No LLM traffic."""
    t0 = time.time()
    raw = read_all_raw_news()
    raw = raw[raw["title"].str.len() > 0].copy()
    raw = raw.sort_values("published_at").reset_index(drop=True)
    log.info("pass1: %d raw items with titles (%.0fs)", len(raw), time.time() - t0)

    # content_hash -> (published_at, title) for cache-pool repair below
    pub_by_hash: dict[str, tuple] = {}
    pending: list[dict] = []
    counters = dict(cached=0, skipped=0, pending=0)

    for _, row in raw.iterrows():
        title = _safe_str(row.get("title", ""))
        body = _safe_str(row.get("body", "") or row.get("summary", "")
                         or row.get("description", ""))
        h = compute_content_hash(title, body)
        day = _item_day(row)
        pub = row.get("published_at", None)
        if pub is None or (isinstance(pub, float) and pd.isna(pub)):
            pub = pd.Timestamp(day, tz="UTC")
        pub = pd.Timestamp(pub)

        if cache_exists(h, MODEL_VERSION, PROMPT_VERSION):
            counters["cached"] += 1
            pub_by_hash[h] = (pub, title)
            continue

        if since is not None and day < since:
            counters["skipped"] += 1
            continue

        matched = match_ticker(" ".join(p for p in (title, body) if p),
                               universe_for_day(day))
        if matched is None:
            counters["skipped"] += 1
            continue

        counters["pending"] += 1
        pending.append(dict(content_hash=h, item_id=_safe_str(row.get("item_id", "")),
                            title=title, body=body, published_at=pub,
                            matched=matched))

    pool = load_all_cached(MODEL_VERSION, PROMPT_VERSION)
    pool = repair_pool(pool, pub_by_hash)
    log.info("pass1 done in %.0fs: %s", time.time() - t0, counters)
    return pending, pool, counters


def repair_pool(pool: pd.DataFrame, pub_by_hash: dict[str, tuple]) -> pd.DataFrame:
    """batch_etl re-attaches published_at/title to cache hits from the raw row.

    Cached parquet written by an older code path may lack those columns; without
    the repair the novelty pool would silently lose pre-existing events and the
    first item after a gap would score as fully novel.
    """
    if pool.empty:
        return pd.DataFrame(columns=["asset", "published_at", "content_hash",
                                     "title", "event_type"])
    if "title" not in pool.columns:
        pool["title"] = ""
    if "published_at" not in pool.columns:
        pool["published_at"] = pd.NaT
    pool["published_at"] = pd.to_datetime(pool["published_at"], utc=True, errors="coerce")

    missing = pool["published_at"].isna()
    if missing.any():
        fixed = pool.loc[missing, "content_hash"].map(
            lambda h: pub_by_hash.get(h, (pd.NaT, ""))[0])
        pool.loc[missing, "published_at"] = pd.to_datetime(fixed, utc=True, errors="coerce")
        titles = pool.loc[missing, "content_hash"].map(
            lambda h: pub_by_hash.get(h, (None, ""))[1])
        pool.loc[missing & (pool["title"].fillna("") == ""), "title"] = titles
        log.info("repaired published_at for %d cached records from raw news",
                 int(missing.sum()))
    return pool


# --------------------------------------------------------------------------- #
# pass 2 — parallel LLM extraction, checkpointed
# --------------------------------------------------------------------------- #

def extract(pending: list[dict], state_path: Path, workers: int,
            retry_errors: bool = False) -> int:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    errored: set[str] = set()
    if state_path.exists():
        with open(state_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    h = rec["content_hash"]
                except (json.JSONDecodeError, KeyError):
                    continue
                # Failed records are checkpointed too. By default they are NOT
                # retried: with temperature 0 the same prompt returns the same
                # unusable answer, so a retry burns an hour for nothing. Pass
                # --retry-errors after a prompt or model change.
                (errored if "_error" in rec else done).add(h)

    todo = [it for it in pending
            if it["content_hash"] not in done
            and (retry_errors or it["content_hash"] not in errored)]
    # `done` counts the whole checkpoint, which can include items no longer
    # pending (already cached by an earlier pass 3). Report the work, not the
    # difference, or the number goes negative and reads like a bug.
    log.info("pass2: checkpoint %d results (%d failed, excluded); %d left to extract",
             len(done), len(errored), len(todo))
    if not todo:
        log.info("pass2: nothing to do")
        return 0

    lock = threading.Lock()
    fh = open(state_path, "a")
    stats = dict(ok=0, fail=0)
    t0 = time.time()

    def work(item: dict) -> dict | None:
        if _stop.is_set():
            return None
        prompt = build_extraction_prompt(item["title"], item["body"])
        try:
            res = call_llm(prompt)
        except Exception as e:                      # noqa: BLE001
            res = {"_error": str(e)}
        res.pop("_raw_output", None)
        # call_llm only repairs MISSING keys; a model that answers
        # {"polarity": null} sails through it and blows up on float(None) later.
        # batch_etl treats such an item as a failed extraction (it raises inside
        # its per-item try/except and is never cached), so mirror that: fail the
        # record here and let the next run retry it.
        if "_error" not in res:
            for k in ("polarity", "magnitude", "confidence"):
                v = res.get(k)
                if v is None or isinstance(v, str) and not v.strip():
                    res = {"_error": f"null/non-numeric {k} in LLM output"}
                    break
        rec = {"content_hash": item["content_hash"], "item_id": item["item_id"],
               "title": item["title"], "matched": item["matched"],
               "published_at": item["published_at"].isoformat(), **res}
        with lock:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if "_error" in res:
                stats["fail"] += 1
            else:
                stats["ok"] += 1
            n = stats["ok"] + stats["fail"]
            if n % 100 == 0 or n == len(todo):
                el = time.time() - t0
                rate = n / max(el, 1e-9)
                eta = (len(todo) - n) / max(rate, 1e-9)
                log.info("pass2: %d/%d (%.2f it/s) eta %.1fh  ok=%d fail=%d",
                         n, len(todo), rate, eta / 3600, stats["ok"], stats["fail"])
        return rec

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, it) for it in todo]
        for _ in as_completed(futs):
            if _stop.is_set():
                break
    fh.close()
    log.info("pass2 done in %.0fs: ok=%d fail=%d", time.time() - t0,
             stats["ok"], stats["fail"])
    if stats["fail"]:
        log.warning("%d items failed — they stay out of the cache and will be "
                    "retried on the next run", stats["fail"])
    return stats["ok"]


# --------------------------------------------------------------------------- #
# pass 3 — novelty + cache write, in published_at order
# --------------------------------------------------------------------------- #

def finalize(pool: pd.DataFrame, state_path: Path) -> int:
    if not state_path.exists():
        log.error("no checkpoint at %s", state_path)
        return 0
    recs = []
    with open(state_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "_error" in r:
                continue
            recs.append(r)
    if not recs:
        log.info("pass3: nothing to finalize")
        return 0

    recs = [r for r in recs
            if not cache_exists(r["content_hash"], MODEL_VERSION, PROMPT_VERSION)]

    def _num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    # Records checkpointed BEFORE the null-field validation existed carry
    # polarity/magnitude/confidence = null and would raise on float(None). Skip
    # them here too: pass 3 must never die after 9h of extraction, and a record
    # that cannot be scored is not cacheable anyway.
    unusable = {r["content_hash"] for r in recs
                if any(_num(r.get(k)) is None
                       for k in ("polarity", "magnitude", "confidence"))}
    if unusable:
        log.warning("pass3: %d records have null/non-numeric scores and cannot be "
                    "cached; they stay in the checkpoint (re-run with "
                    "--retry-errors after a prompt/model change)", len(unusable))
        recs = [r for r in recs if r["content_hash"] not in unusable]
    if not recs:
        log.info("pass3: nothing cacheable")
        return 0

    recs.sort(key=lambda r: r["published_at"])
    log.info("pass3: finalizing %d records (pool seed %d rows)", len(recs), len(pool))

    # 48h trailing window, fed in published_at order from pool + new records.
    pool_sorted = pool.dropna(subset=["published_at"]).sort_values("published_at")
    pool_rows = list(pool_sorted[["asset", "published_at", "content_hash",
                                  "title", "event_type"]].itertuples(index=False, name=None))
    ptr = 0
    window: deque = deque()
    written = 0
    t0 = time.time()

    def push(row):
        window.append(row)

    for i, r in enumerate(recs):
        t = pd.Timestamp(r["published_at"])
        # advance the pool frontier: every cached record published before t belongs
        # to the window (exactly what the mask in compute_novelty would keep)
        while ptr < len(pool_rows) and pool_rows[ptr][1] < t:
            push(pool_rows[ptr])
            ptr += 1
        cutoff = t - timedelta(hours=LOOKBACK_HOURS)
        while window and window[0][1] < cutoff:
            window.popleft()

        asset = r["matched"]
        cur_hash = r["content_hash"]
        rows = [w for w in window if w[0] == asset and w[2] != cur_hash]
        if rows:
            wdf = pd.DataFrame(rows, columns=["asset", "published_at",
                                              "content_hash", "title", "event_type"])
        else:
            wdf = pd.DataFrame(columns=["asset", "published_at", "content_hash",
                                        "title", "event_type"])

        event = dict(asset=asset, event_type=r["event_type"], published_at=t,
                     content_hash=cur_hash, title=r["title"])
        novelty = compute_novelty(event, wdf)

        record = {
            "item_id": r["item_id"], "content_hash": cur_hash,
            "model_version": MODEL_VERSION, "prompt_version": PROMPT_VERSION,
            "extracted_at": datetime.now(timezone.utc), "published_at": t,
            "asset": asset, "event_type": str(r["event_type"]),
            "polarity": float(r["polarity"]), "magnitude": float(r["magnitude"]),
            "novelty": float(novelty), "confidence": float(r["confidence"]),
            "extraction_only": bool(r["extraction_only"]), "title": r["title"],
        }
        save_to_cache(record, MODEL_VERSION, PROMPT_VERSION)
        # a new record is itself a candidate for later items in the same asset
        push((asset, t, cur_hash, r["title"], r["event_type"]))
        written += 1
        if written % 500 == 0 or written == len(recs):
            el = time.time() - t0
            rate = written / max(el, 1e-9)
            log.info("pass3: %d/%d written (%.1f it/s, eta %.0f min)",
                     written, len(recs), rate, (len(recs) - written) / max(rate, 1e-9) / 60)

    log.info("pass3 done in %.0fs: %d records cached", time.time() - t0, written)
    return written


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent LLM calls (router child has 4 slots)")
    ap.add_argument("--state", type=Path, default=DEFAULT_STATE)
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N pending items (smoke test)")
    ap.add_argument("--since", type=str, default=None,
                    help="ignore items published before this date (YYYY-MM-DD)")
    ap.add_argument("--classify-only", action="store_true")
    ap.add_argument("--retry-errors", action="store_true",
                    help="re-attempt checkpointed failures (only useful after a "
                         "prompt or model change; at temperature 0 they repeat)")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _sigint)
    since = date.fromisoformat(args.since) if args.since else None

    pending, pool, counters = classify(since=since)
    if args.classify_only:
        return
    if args.limit:
        pending = pending[: args.limit]
        log.info("--limit %d applied", args.limit)
    if not pending:
        log.info("nothing pending")
        return

    extract(pending, args.state, args.workers, retry_errors=args.retry_errors)
    if _stop.is_set():
        log.warning("stopped early; checkpoint kept at %s", args.state)
        return
    finalize(pool, args.state)


if __name__ == "__main__":
    main()

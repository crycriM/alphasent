"""Minimal tests for cryptopanic_ingest — mocked API, no token/network needed.
Run: python test_cryptopanic_ingest.py
"""
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cryptopanic_ingest as ci


def _fake_post(pid: int, pub: str, assets=("BTC",), title=None):
    return {
        "id": pid,
        "title": title or f"Post {pid}",
        "description": "",
        "published_at": pub,
        "created_at": pub,
        "url": f"https://cryptopanic.com/news/{pid}/",
        "original_url": f"https://coindesk.com/{pid}",
        "kind": "news",
        "source": {"domain": "coindesk.com", "type": "feed"},
        "instruments": [{"code": c} for c in assets],
        "votes": {"liked": 2, "important": 1, "disliked": 0, "bearish": 1},
        "panic_score": 55,
    }


def test_normalize_basic():
    raw = _fake_post(101, "2021-06-01T12:00:00Z", assets=("BTC", "eth"))
    p = ci.normalize_post(raw, ingested_at=datetime(2021, 6, 1, 12, 5, tzinfo=timezone.utc))
    assert p.item_id == "cryptopanic:101"
    assert p.asset_mentions == ["BTC", "ETH"]           # upper + sorted + deduped
    assert p.published_at.tzinfo is not None             # tz-aware UTC
    assert p.votes_positive == 3 and p.votes_negative == 1
    print("ok test_normalize_basic")


def test_published_at_fallback():
    raw = _fake_post(102, None)
    raw["published_at"] = None
    fetched = datetime(2022, 1, 1, tzinfo=timezone.utc)
    p = ci.normalize_post(raw, ingested_at=fetched)
    assert p.published_at == fetched                     # falls back to fetch time
    print("ok test_published_at_fallback")


def test_write_dedup_and_partitioning():
    with tempfile.TemporaryDirectory() as tmp:
        ci.NORMALIZED_DIR = Path(tmp) / "normalized"
        posts = [
            ci.normalize_post(_fake_post(1, "2021-06-01T10:00:00Z"), datetime.now(timezone.utc)),
            ci.normalize_post(_fake_post(2, "2021-06-01T11:00:00Z"), datetime.now(timezone.utc)),
            ci.normalize_post(_fake_post(3, "2021-06-02T09:00:00Z"), datetime.now(timezone.utc)),
        ]
        n1 = ci.write_normalized(posts)
        assert n1 == 3, f"expected 3 new, got {n1}"
        # two partitions (two distinct days)
        parts = list(ci.NORMALIZED_DIR.glob("*.parquet"))
        assert len(parts) == 2, f"expected 2 day-partitions, got {len(parts)}"
        # re-writing the same posts adds nothing (idempotent)
        n2 = ci.write_normalized(posts)
        assert n2 == 0, f"expected 0 on re-write, got {n2}"
        # a new post on an existing day appends only the new row
        extra = [ci.normalize_post(_fake_post(4, "2021-06-01T23:00:00Z"), datetime.now(timezone.utc))]
        n3 = ci.write_normalized(extra)
        assert n3 == 1, f"expected 1 new, got {n3}"
        print("ok test_write_dedup_and_partitioning")


def test_pagination_stops_at_watermark():
    # Build a fake client whose .get() returns canned pages by URL.
    pages = {
        "URL0": {"results": [_fake_post(50, "2021-06-03T00:00:00Z"),
                             _fake_post(49, "2021-06-02T23:00:00Z")], "next": "URL1"},
        "URL1": {"results": [_fake_post(48, "2021-06-02T22:00:00Z"),
                             _fake_post(10, "2021-06-01T00:00:00Z")], "next": "URL2"},
        "URL2": {"results": [_fake_post(9, "2021-05-31T00:00:00Z")], "next": None},
    }

    class FakeResp:
        def __init__(self, payload): self._p = payload; self.status_code = 200; self.headers = {}
        def raise_for_status(self): pass
        def json(self): return self._p

    class FakeClient:
        def get(self, url, timeout=None): return FakeResp(pages[url])

    limiter = ci.RateLimiter(0.0)
    # high-water mark = 20: page URL1 contains post_id 10 (<=20) so we stop after it.
    collected = list(ci.fetch_pages(FakeClient(), "URL0", limiter, stop_at_post_id=20))
    seen_ids = {int(r["id"]) for page in collected for r in page}
    assert 48 in seen_ids and 10 in seen_ids       # URL0 and URL1 fetched
    assert 9 not in seen_ids                        # URL2 never fetched (stopped)
    print("ok test_pagination_stops_at_watermark")


if __name__ == "__main__":
    test_normalize_basic()
    test_published_at_fallback()
    test_write_dedup_and_partitioning()
    test_pagination_stops_at_watermark()
    print("\nALL TESTS PASSED")

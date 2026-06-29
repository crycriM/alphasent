"""Offline tests for crypto_rss_ingest. Synthetic RSS/Atom XML, no network.
Run: python test_crypto_rss_ingest.py
"""
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import crypto_rss_ingest as ri


RSS_XML = b"""<?xml version="1.0"?>
<rss version="2.0"><channel><title>Test</title>
<item>
  <title>Bitcoin surges as Ethereum ETF approved</title>
  <link>https://coindesk.com/a/1</link>
  <guid>https://coindesk.com/a/1</guid>
  <pubDate>Tue, 01 Jun 2021 12:00:00 GMT</pubDate>
  <description>BTC and ETH rally on regulatory news.</description>
  <category>Markets</category>
</item>
<item>
  <title>Solana network outage hits DeFi</title>
  <link>https://coindesk.com/a/2</link>
  <guid>https://coindesk.com/a/2</guid>
  <pubDate>Wed, 02 Jun 2021 09:30:00 GMT</pubDate>
  <description>SOL down sharply.</description>
</item>
</channel></rss>"""

ATOM_XML = b"""<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
<title>AtomTest</title>
<entry>
  <title>Ripple wins partial ruling in SEC case</title>
  <link href="https://decrypt.co/x/9"/>
  <id>tag:decrypt.co,2021:9</id>
  <updated>2021-06-01T15:00:00Z</updated>
  <summary>XRP jumps on the news.</summary>
</entry>
</feed>"""


def test_tag_assets():
    assert ri.tag_assets("Bitcoin and Ethereum rally") == ["BTC", "ETH"]
    assert ri.tag_assets("Solana outage") == ["SOL"]
    assert ri.tag_assets("no coins here") == []
    print("ok test_tag_assets")


def test_parse_and_normalize_rss():
    entries = ri.parse_entries(RSS_XML)
    assert len(entries) == 2
    item = ri.normalize_entry(entries[0], "coindesk", "https://coindesk.com/rss",
                              ingested_at=datetime(2021, 6, 1, 12, 5, tzinfo=timezone.utc))
    assert item.title.startswith("Bitcoin surges")
    assert item.asset_mentions == ["BTC", "ETH"]
    assert item.source_domain == "coindesk.com"
    assert item.published_at == datetime(2021, 6, 1, 12, 0, tzinfo=timezone.utc)
    assert item.feed_categories == ["Markets"]
    print("ok test_parse_and_normalize_rss")


def test_parse_atom():
    entries = ri.parse_entries(ATOM_XML)
    assert len(entries) == 1
    item = ri.normalize_entry(entries[0], "decrypt", "https://decrypt.co/feed",
                              ingested_at=datetime.now(timezone.utc))
    assert item.asset_mentions == ["XRP"]
    assert item.url == "https://decrypt.co/x/9"
    assert item.published_at == datetime(2021, 6, 1, 15, 0, tzinfo=timezone.utc)
    print("ok test_parse_atom")


def test_stable_id_dedup():
    entries = ri.parse_entries(RSS_XML)
    id1 = ri._stable_id(entries[0])
    id1_again = ri._stable_id(entries[0])
    id2 = ri._stable_id(entries[1])
    assert id1 == id1_again and id1 != id2     # deterministic + distinct
    print("ok test_stable_id_dedup")


def test_write_partitions_and_dedup():
    with tempfile.TemporaryDirectory() as tmp:
        ri.NORMALIZED_DIR = Path(tmp) / "normalized"
        entries = ri.parse_entries(RSS_XML)
        now = datetime.now(timezone.utc)
        items = [ri.normalize_entry(e, "coindesk", "https://coindesk.com/rss", now)
                 for e in entries]
        n1 = ri.write_normalized(items)
        assert n1 == 2, f"expected 2, got {n1}"
        parts = sorted(p.name for p in ri.NORMALIZED_DIR.glob("*.parquet"))
        assert parts == ["2021-06-01.parquet", "2021-06-02.parquet"], parts
        n2 = ri.write_normalized(items)            # idempotent re-write
        assert n2 == 0, f"expected 0, got {n2}"
        print("ok test_write_partitions_and_dedup")


def test_run_ingest_mocked():
    with tempfile.TemporaryDirectory() as tmp:
        ri.DATA_ROOT = Path(tmp)
        ri.NORMALIZED_DIR = ri.DATA_ROOT / "normalized"
        ri.RAW_JSON_DIR = ri.DATA_ROOT / "raw_json"
        ri.STATE_FILE = ri.DATA_ROOT / "state.json"

        feed_map = {"https://a/rss": RSS_XML, "https://b/atom": ATOM_XML}
        ri.fetch_feed = lambda client, url: feed_map[url]

        class Dummy:
            def __enter__(self): return self
            def __exit__(self, *a): return False
        ri.httpx.Client = lambda *a, **k: Dummy()
        ri.PER_FEED_PAUSE = 0.0

        feeds = [("feedA", "https://a/rss"), ("feedB", "https://b/atom")]
        s1 = ri.run_ingest(feeds)
        assert s1["new_items"] == 3, s1                 # 2 rss + 1 atom
        s2 = ri.run_ingest(feeds)                        # watermark -> nothing new
        assert s2["new_items"] == 0, s2
        st = ri.IngestState.load(ri.STATE_FILE)
        assert st.feeds["feedA"].total_items == 2
        assert st.feeds["feedB"].total_items == 1
        print("ok test_run_ingest_mocked")


if __name__ == "__main__":
    test_tag_assets()
    test_parse_and_normalize_rss()
    test_parse_atom()
    test_stable_id_dedup()
    test_write_partitions_and_dedup()
    test_run_ingest_mocked()
    print("\nALL TESTS PASSED")

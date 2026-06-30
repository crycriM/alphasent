from __future__ import annotations

from unittest.mock import patch

import pytest

def test_run_once_no_token():
    from src.live.poller import run_once
    result = run_once(token="", update_features=False)
    assert result["new_items"] == 0

def test_run_once_no_new_posts():
    from src.live.poller import run_once
    with patch("src.live.poller._poll_new_posts", return_value=[]):
        result = run_once(token="dummy", update_features=False)
    assert result["new_items"] == 0
    assert result["extracted"] == 0

def test_run_once_with_posts():
    from src.live.poller import run_once
    fake_post = {
        "id": 999999,
        "published_at": "2026-06-30T00:00:00Z",
        "title": "BTC surges past 100k",
        "description": "Bitcoin hits new all-time high",
        "url": "https://example.com",
        "original_url": "https://example.com",
        "source": {"domain": "example.com", "type": "feed"},
        "kind": "news",
        "instruments": [{"code": "BTC"}],
        "votes": {},
    }
    with (
        patch("src.live.poller._poll_new_posts", return_value=[fake_post]),
        patch("src.live.poller._extract_single", return_value={"item_id": "cp:999999", "content_hash": "h1"}),
        patch("src.live.poller.cache_exists", return_value=False),
    ):
        result = run_once(token="dummy", update_features=False)
    assert result["new_items"] == 1
    assert result["extracted"] == 1 or result["cached"] == 1

def test_run_once_idempotent():
    from src.live.poller import run_once
    fake_post = {
        "id": 999998,
        "published_at": "2026-06-30T00:00:00Z",
        "title": "ETH up",
        "description": "Ethereum rises",
        "url": "https://example.com",
        "original_url": "https://example.com",
        "source": {"domain": "example.com", "type": "feed"},
        "kind": "news",
        "instruments": [{"code": "ETH"}],
        "votes": {},
    }
    with (
        patch("src.live.poller._poll_new_posts", return_value=[fake_post]),
        patch("src.live.poller._extract_single", return_value=None),
    ):
        result = run_once(token="dummy", update_features=False)
    assert result["errors"] == 1

def test_run_once_empty_page():
    from src.live.poller import run_once
    with patch("src.live.poller._poll_new_posts", return_value=[]):
        r1 = run_once(token="dummy", update_features=False)
        r2 = run_once(token="dummy", update_features=False)
    assert r1 == r2

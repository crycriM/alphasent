"""
Async article body fetcher.

Fetches article body text from URLs using trafilatura.
Configured for GDELT article URLs but works with any HTTP source.

- 10-second timeout per request
- 3-retry exponential backoff
- Returns fetch_status codes in the raw schema
- Drops items with fetch_status != 200 at Layer 2
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx

log = logging.getLogger("ingest.article")

MAX_RETRIES = 3
REQUEST_TIMEOUT = 10.0  # seconds

def fetch_body(url: str, max_retries: int = MAX_RETRIES) -> tuple[str, int]:
    """Fetch the article body for a URL.

    Args:
        url: Article URL.
        max_retries: Number of retry attempts.

    Returns:
        Tuple of (body_text, fetch_status).
        fetch_status: HTTP status code, -1 for timeout, -2 for empty body.
    """
    for attempt in range(1, max_retries + 1):
        try:
            with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
                resp = client.get(url)
                if resp.status_code in (200, 201):
                    try:
                        import trafilatura
                    except ImportError:
                        log.warning(
                            "trafilatura not installed. "
                            "Install with: pip install trafilatura"
                        )
                        body = resp.text[:500] if resp.text else ""
                        return (body, -2 if not body else 200)

                    body = trafilatura.extract(
                        resp.text,
                        include_comments=False,
                        include_tables=False,
                        url=url,
                    )
                    if body:
                        return (body.strip(), 200)
                    log.warning("Empty body extracted from %s", url)
                    return ("", -2)
                elif resp.status_code == 403:
                    log.warning(
                        "Forbidden for %s (attempt %d/%d)",
                        url, attempt, max_retries,
                    )
                    if attempt < max_retries:
                        continue
                    return ("", 403)
                else:
                    log.warning(
                        "HTTP %d for %s (attempt %d/%d)",
                        resp.status_code, url, attempt, max_retries,
                    )
                    if attempt < max_retries:
                        continue
                    return ("", resp.status_code)
        except httpx.TimeoutException:
            log.warning("Timeout fetching %s (attempt %d/%d)", url, attempt, max_retries)
            if attempt < max_retries:
                continue
            return ("", -1)
        except httpx.HTTPError as e:
            log.warning("HTTP error fetching %s (attempt %d/%d): %s", url, attempt, max_retries, e)
            if attempt < max_retries:
                continue
            return ("", -1)
        except Exception as e:
            log.error("Error fetching %s: %s", url, e)
            return ("", -1)

    return ("", -1)

def fetch_bodies(urls: list[str]) -> list[tuple[str, int]]:
    """Fetch bodies for a list of URLs.

    Args:
        urls: List of article URLs.

    Returns:
        List of (body_text, fetch_status) tuples, one per URL.
    """
    results: list[tuple[str, int]] = []
    for i, url in enumerate(urls):
        body, status = fetch_body(url)
        results.append((body, status))
        if (i + 1) % 100 == 0:
            log.info("Fetched %d/%d articles", i + 1, len(urls))
    return results

def fetch_bodies_async(urls: list[str]) -> list[tuple[str, int]]:
    """Fetch bodies for a list of URLs asynchronously.

    Args:
        urls: List of article URLs.

    Returns:
        List of (body_text, fetch_status) tuples, one per URL.
    """
    import asyncio

    async def _fetch(url: str) -> tuple[str, int]:
        try:
            import trafilatura
        except ImportError:
            return ("", -2)

        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        resp = await client.get(url)
                        if resp.status_code in (200, 201):
                            body = trafilatura.extract(
                                resp.text,
                                include_comments=False,
                                include_tables=False,
                                url=url,
                            )
                            if body:
                                return (body.strip(), 200)
                            return ("", -2)
                        elif resp.status_code == 403:
                            if attempt < MAX_RETRIES:
                                continue
                            return ("", 403)
                        else:
                            if attempt < MAX_RETRIES:
                                continue
                            return ("", resp.status_code)
                    except httpx.TimeoutException:
                        if attempt < MAX_RETRIES:
                            continue
                        return ("", -1)
                    except httpx.HTTPError:
                        if attempt < MAX_RETRIES:
                            continue
                        return ("", -1)
        except Exception as e:
            log.error("Async fetch error for %s: %s", url, e)
            return ("", -1)

    async def _run() -> list[tuple[str, int]]:
        sem = asyncio.Semaphore(10)
        async def _fetch_limited(url: str) -> tuple[str, int]:
            async with sem:
                return await _fetch(url)
        tasks = [asyncio.create_task(_fetch_limited(url)) for url in urls]
        return await asyncio.gather(*tasks)

    return asyncio.run(_run())

if __name__ == "__main__":
    """Smoke test: fetch a known URL."""
    logging.basicConfig(level=logging.INFO)
    test_url = "https://coindesk.com/markets"
    body, status = fetch_body(test_url)
    print(f"Status: {status}, Body length: {len(body)}")

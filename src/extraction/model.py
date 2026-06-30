"""
LLM client for event extraction.

Calls the local llama3-8b via OpenAI-compatible API (router on port 8079).
Uses structured output parsing (JSON extraction with retry).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from src.config import (
    LLM_BASE_URL,
    LLM_MAX_TOKENS,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    LLM_TOP_P,
    LLM_MAX_RETRIES,
)
from src.extraction.prompt import SYSTEM_PROMPT, parse_extraction_output

log = logging.getLogger("extraction.model")


def call_llm(prompt: str, max_retries: int = LLM_MAX_RETRIES) -> dict[str, Any]:
    """Call the local LLM via OpenAI-compatible API.

    Args:
        prompt: Full prompt string (already formatted for Llama-3 chat template).
        max_retries: Number of retry attempts.

    Returns:
        Parsed extraction dict with keys: asset, event_type, polarity,
        magnitude, confidence, extraction_only.

    Raises:
        RuntimeError: If all retries fail.
    """
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = httpx.post(
                f"{LLM_BASE_URL}/chat/completions",
                json={
                    "model": LLM_MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": LLM_TEMPERATURE,
                    "top_p": LLM_TOP_P,
                    "max_tokens": LLM_MAX_TOKENS,
                    "stream": False,
                },
                timeout=LLM_TIMEOUT,
            )

            if resp.status_code != 200:
                log.warning(
                    "LLM call failed (attempt %d/%d): HTTP %d — %s",
                    attempt, max_retries, resp.status_code, resp.text[:200],
                )
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                continue

            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()

            # Parse JSON from the LLM output
            result = parse_extraction_output(content)

            # Validate required keys
            required = {"asset", "event_type", "polarity", "magnitude", "confidence", "extraction_only"}
            if not required.issubset(result.keys()):
                log.warning(
                    "LLM output missing keys: %s", required - set(result.keys())
                )
                # Try to recover — fill missing with defaults
                for key in required - set(result.keys()):
                    defaults = {
                        "asset": "UNKNOWN",
                        "event_type": "other",
                        "polarity": 0.0,
                        "magnitude": 0.0,
                        "confidence": 0.0,
                        "extraction_only": True,
                    }
                    result[key] = defaults[key]

            result["_raw_output"] = content  # keep raw for debugging
            log.info("LLM extraction: %s [%s] p=%.2f m=%.2f conf=%.2f",
                     result["asset"], result["event_type"],
                     result["polarity"], result["magnitude"],
                     result["confidence"])
            return result

        except httpx.TimeoutException:
            log.warning("LLM timeout (attempt %d/%d)", attempt, max_retries)
            last_error = "timeout"
            if attempt < max_retries:
                time.sleep(2 ** attempt)
        except httpx.HTTPError as e:
            log.warning("LLM HTTP error (attempt %d/%d): %s", attempt, max_retries, e)
            last_error = str(e)
            if attempt < max_retries:
                time.sleep(2 ** attempt)
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            log.warning("LLM parse error (attempt %d/%d): %s", attempt, max_retries, e)
            last_error = str(e)
            if attempt < max_retries:
                time.sleep(1)

    raise RuntimeError(
        f"LLM extraction failed after {max_retries} retries. Last error: {last_error}"
    )


def call_llm_batch(prompts: list[str]) -> list[dict]:
    """Call LLM for a batch of prompts sequentially.

    Each call is independent; batch size is limited by the LLM server's
    capacity, not by our batching.

    Returns a list of dicts (one per prompt), with None for failed calls.
    """
    results = []
    for i, prompt in enumerate(prompts):
        try:
            result = call_llm(prompt)
            results.append(result)
        except RuntimeError as e:
            log.error("Batch item %d failed: %s", i, e)
            results.append(None)
    return results

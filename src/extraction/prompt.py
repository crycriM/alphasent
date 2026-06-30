"""
Prompt builder for LLM event extraction.

Handles budget enforcement (8k context for Llama-3-8B), few-shot exemplar
selection, and entity redaction for the contamination audit.

Budget formula (Llama-3-8B, 8192 context):
  len(tokenize(system_prompt + few_shot_block + title + body_truncated + output_prefix))
  + max_new_tokens <= 8192

Reserves ~1700 tokens for system prompt, few-shot, and output grammar.
Body is truncated to fit; if title alone exceeds budget, reduce few-shot.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from src.config import (
    BODY_TRUNCATE_CHARS,
    FEE_SHOT_COUNT,
    LLM_CONTEXT_SIZE,
    LLM_MAX_TOKENS,
    MODEL_VERSION,
    PROMPT_VERSION,
)

# --------------------------------------------------------------------------- #
# Few-shot exemplars
# --------------------------------------------------------------------------- #

FEW_SHOT_EXEMPLARS = [
    {
        "text": "Binance announces listing of MATIC spot trading pair starting Friday.",
        "output": '{"asset": "MATIC", "event_type": "listing", "polarity": 0.65, "magnitude": 0.5, "confidence": 0.92, "extraction_only": true}',
    },
    {
        "text": "SEC files charges against Ripple Labs over XRP securities offering, seeks injunction and disgorgement of profits.",
        "output": '{"asset": "XRP", "event_type": "regulation", "polarity": -0.90, "magnitude": 0.85, "confidence": 0.95, "extraction_only": true}',
    },
    {
        "text": "Nomad bridge drained of $190M in exploit as attacker replays fraudulent transactions across multiple chains.",
        "output": '{"asset": "ETH", "event_type": "hack", "polarity": -0.85, "magnitude": 0.75, "confidence": 0.88, "extraction_only": true}',
    },
]

# --------------------------------------------------------------------------- #
# System prompt
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """\
You are a structured data extraction system. Read the crypto news text and output
a JSON object with these exact keys and value constraints:
  asset: ticker symbol (string)
  event_type: one of [listing, hack, regulation, partnership, depeg, macro, fork, other]
  polarity: float -1.0 to 1.0 (negative=bad for asset, positive=good)
  magnitude: float 0.0 to 1.0 (estimated market impact size)
  confidence: float 0.0 to 1.0 (your confidence in this extraction)
  extraction_only: bool (true if you determined event_type and polarity
    SOLELY from this text; false if you used any outside knowledge of how
    this event turned out)
Output ONLY the JSON object. No explanation. No preamble.
"""

# --------------------------------------------------------------------------- #
# Redaction map for contamination audit (§8.2 of PLAN.md)
# --------------------------------------------------------------------------- #

REDACTION_MAP = [
    (re.compile(r'\bBitcoin\b|\bBTC\b', re.I), 'Asset_X'),
    (re.compile(r'\bEthereum\b|\bETH\b', re.I), 'Asset_Y'),
    (re.compile(r'\bRipple\b|\bXRP\b', re.I), 'Asset_Z'),
    (re.compile(r'\bTerraUSD\b|\bUST\b|\bLUNA\b', re.I), 'Asset_W'),
    (re.compile(r'\bSEC\b', re.I), 'Regulator_A'),
    (re.compile(r'\bBinance\b', re.I), 'Exchange_A'),
    (re.compile(r'\bSolana\b|\bSOL\b', re.I), 'Asset_S'),
    (re.compile(r'\bCardano\b|\bADA\b', re.I), 'Asset_C'),
    (re.compile(r'\bTether\b|\bUSDT\b', re.I), 'Stablecoin_A'),
    (re.compile(r'\bCircle\b|\bUSDC\b', re.I), 'Stablecoin_B'),
    (re.compile(r'\bTerra\b|\bLUNA\b', re.I), 'Asset_L'),
    (re.compile(r'\bFTX\b', re.I), 'Exchange_B'),
    (re.compile(r'\bCelsius\b', re.I), 'Exchange_C'),
    (re.compile(r'\bTerra\b', re.I), 'Protocol_A'),
]


def redact_text(text: str) -> str:
    """Replace named entities in text with generic placeholders.

    Used for the contamination audit: if the model's extraction changes
    when entities are redacted, it may be relying on parametric recall.
    """
    for pattern, replacement in REDACTION_MAP:
        text = pattern.sub(replacement, text)
    return text


def build_extraction_prompt(
    title: str,
    body: str,
    redact: bool = False,
    few_shot_count: int | None = None,
) -> str:
    """Build the full extraction prompt with budget enforcement.

    Args:
        title: Article headline.
        body: Article body (truncated internally).
        redact: If True, redact entity names for contamination audit.
        few_shot_count: Override number of few-shot exemplars (1-3).

    Returns:
        The full prompt string ready for the LLM API call.
    """
    # Truncate body to budget
    if len(body) > BODY_TRUNCATE_CHARS:
        body = body[:BODY_TRUNCATE_CHARS]

    # Redact if requested
    text = f"{title}. {body}" if body else title
    if redact:
        text = redact_text(text)

    # Select few-shot exemplars
    n_shots = min(few_shot_count or FEE_SHOT_COUNT, len(FEW_SHOT_EXEMPLARS))
    exemplar_block = ""
    for i, ex in enumerate(FEW_SHOT_EXEMPLARS[:n_shots]):
        exemplar_text = ex["text"]
        if redact:
            exemplar_text = redact_text(exemplar_text)
        exemplar_block += f"""
Example {i+1}:
Text: {exemplar_text}
Output: {ex['output']}"""

    # Build the USER message content (examples + version + text).
    # The Llama-3 chat template is applied by the serving endpoint; we must NOT
    # embed raw <|begin_of_text|>/<|start_header_id|> markers here, otherwise
    # the server double-wraps them and the model sees markers as literal text.
    user_content = (
        f"{exemplar_block}"
        f"\n\n{PROMPT_VERSION}\n\nText: {text}"
    )

    # Budget check: estimate token count over system + user content
    # Llama-3 tokenizer: ~1 token per 3.5 chars for English text
    estimated_tokens = (len(SYSTEM_PROMPT) + len(user_content)) // 3.5 + LLM_MAX_TOKENS
    if estimated_tokens > LLM_CONTEXT_SIZE:
        # Reduce few-shot and body to fit
        if n_shots > 1:
            # Try with fewer shots
            return build_extraction_prompt(title, body, redact, n_shots - 1)
        else:
            # Double-truncate body
            half = BODY_TRUNCATE_CHARS // 4
            return build_extraction_prompt(
                title, body[:half], redact, n_shots
            )

    return user_content


def parse_extraction_output(raw: str) -> dict:
    """Parse the LLM's JSON output. Strips markdown code fences if present.

    Robust against common LLM output issues:
    - Trailing text after JSON
    - Markdown code fences
    - Whitespace/newlines
    """
    raw = raw.strip()

    # Strip markdown code fences
    if raw.startswith("```"):
        lines = raw.split("\n")
        # Remove first/last fence lines
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    # Find the first complete JSON object.
    # The LLM sometimes outputs two JSON objects back-to-back.
    # Strategy: find the first {, then find the matching closing } by
    # counting brace depth. This ensures we get the FIRST complete object.
    start = raw.find("{")
    if start < 0:
        raise json.JSONDecodeError("No opening brace found", raw, 0)

    depth = 0
    end = start
    for i in range(start, len(raw)):
        ch = raw[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1  # inclusive
                break

    if depth != 0:
        raise json.JSONDecodeError("Unmatched braces in LLM output", raw, start)

    raw = raw[start:end]

    # Handle common LLM issues:
    # 1. Trailing commas before }
    raw = re.sub(r',(\s*})', r'\1', raw)
    # 2. Single quotes -> double quotes (common in LLM output)
    raw = raw.strip()
    if raw.startswith("{"):
        # Try parsing as-is first
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # Try replacing single quotes with double quotes
        try:
            fixed = raw.replace("'", '"')
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

    # If all else fails, raise
    raise json.JSONDecodeError(f"Cannot parse LLM output: {raw[:200]}", raw, 0)

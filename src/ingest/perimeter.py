"""Perimeter-based crypto universe loader for survivorship-bias-free asset tagging.

Loads the per-month perimeter JSON files (perpetual swap listings) and provides
the set of active tickers for any given date. This is used to pre-filter news articles
before LLM extraction: an article is only tagged as "crypto" if its text contains
a ticker that was actively traded on at least one venue at that time.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import date
from pathlib import Path
from typing import Optional

log = logging.getLogger("ingest.perimeter")

# Path to perimeter directory, overridable via env
PERIMETER_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "perimeter"

# Stablecoin suffixes to strip
_STABLECOIN_SUFFIXES = ("USDT", "USDC", "BUSD", "TUSD", "DAI", "FDUSD", "USD")

# Ticker aliases: common full names -> ticker symbol
# These let us match e.g. "Bitcoin" in article text even if only "BTC" is in the universe.
# Updated periodically; the old LLM knows these already so there's no lookahead issue
# for well-established coins.
_TICKER_ALIASES: dict[str, str] = {
    "BITCOIN": "BTC",
    "ETHEREUM": "ETH",
    "XRP": "XRP",
    "SOLANA": "SOL",
    "BNB": "BNB",
    "DOGECOIN": "DOGE",
    "CARDANO": "ADA",
    "POLKADOT": "DOT",
    "CHAINLINK": "LINK",
    "AVALANCHE": "AVAX",
    "TRON": "TRX",
    "LITECOIN": "LTC",
    "BITCOINCASH": "BCH",
    "STELLAR": "XLM",
    "MONERO": "XMR",
    "ETHEREUMCLASSIC": "ETC",
    "TEZOS": "XTZ",
    "EOS": "EOS",
    "NEO": "NEO",
    "IOTA": "IOTA",
    "VECHAIN": "VET",
    "THETA": "THETA",
    "FILECOIN": "FIL",
    "APTOS": "APT",
    "ARBITRUM": "ARB",
    "OPTIMISM": "OP",
    "SUI": "SUI",
    "NEAR": "NEAR",
    "POLYGON": "MATIC",
    "MATIC": "MATIC",
    "ATOM": "ATOM",
    "COSMOS": "ATOM",
    "UNISWAP": "UNI",
    "AAVE": "AAVE",
    "PEPE": "PEPE",
    "FLOKI": "FLOKI",
    "SHIBAINU": "SHIB",
    "SHIB": "SHIB",
    "RENDER": "RNDR",
    "FET": "FET",
    "ASI": "ASI",
    "FETCH": "FET",
    "INTERNETCOMPUTER": "ICP",
    "ICP": "ICP",
    "HEDERA": "HBAR",
    "KASPA": "KAS",
    "INJECTIVE": "INJ",
    "SEI": "SEI",
    "CELESTIA": "TIA",
}


def _extract_ticker(symbol: str) -> Optional[str]:
    """Normalize an exchange symbol to a bare ticker.

    Handles all known exchange formats:
      Binance/bybit:    BTCUSDT
      Okex:             BTC-USDT-SWAP
      Hyperliquid:      BTC/USDC:USDC
      Bitget (bare):    BTC
    """
    s = symbol
    # Hyperliquid: BTC/USDC:USDC -> BTC
    s = s.split("/")[0] if "/" in s else s
    # Okex: BTC-USDT-SWAP -> BTC
    s = s.split("-")[0] if "-" in s else s
    # Strip stablecoin suffixes
    for suffix in _STABLECOIN_SUFFIXES:
        if s.endswith(suffix) and len(s) > len(suffix) + 1:
            s = s[: -len(suffix)]
            break
    # Validate: 2-10 uppercase alphanumeric chars
    if not re.match(r"^[A-Z0-9]{2,10}$", s):
        return None
    return s


def _find_perimeter_file(target: date) -> Optional[Path]:
    """Find the perimeter file closest to (and <=) the target date."""
    files = sorted(PERIMETER_DIR.glob("recup_perimeter_*.json"))
    target_prefix = target.isoformat()[:7]  # "2025-07"

    # Exact month match
    for f in files:
        fdate = f.stem.replace("recup_perimeter_", "")
        if fdate.startswith(target_prefix):
            return f

    # Fall back to the file with date <= target
    best: Optional[Path] = None
    for f in files:
        fdate = f.stem.replace("recup_perimeter_", "")
        try:
            fd = date.fromisoformat(fdate[:10])
        except ValueError:
            continue
        if fd <= target:
            best = f
    return best


def load_universe(at: date) -> set[str]:
    """Load the set of active ticker symbols at a given date.

    Survivorship-bias-safe: uses the perimeter snapshot closest to (and before)
    the target date. Tickers that were delisted before that date won't appear.
    """
    fpath = _find_perimeter_file(at)
    if fpath is None:
        log.warning("No perimeter file found for %s", at)
        # Fall back to a minimal well-known set for very early dates
        return {"BTC", "ETH", "XRP", "BNB", "SOL", "ADA", "DOGE", "DOT",
                "LINK", "AVAX", "MATIC", "UNI", "ATOM", "LTC", "BCH", "TRX",
                "ETC", "FIL", "APT", "ARB", "OP", "SUI"}

    with open(fpath) as f:
        data = json.load(f)

    tickers: set[str] = set()
    for exchange, symbols in data.items():
        for sym in symbols:
            ticker = _extract_ticker(sym)
            if ticker:
                tickers.add(ticker)

    log.info("Loaded %d tickers from %s", len(tickers), fpath.name)
    return tickers


def match_ticker(text: str, universe: set[str]) -> Optional[str]:
    """Find the first ticker from `universe` that appears in `text`.

    Two-pass matching:
    1. Check common name aliases first (e.g. "Bitcoin" -> BTC, "Solana" -> SOL)
    2. Then check raw ticker symbols with word-boundary matching

    Case-insensitive. Returns the matched ticker (uppercase) or None.
    """
    if not text:
        return None
    upper = text.upper()

    # Pass 1: check aliases (full names like "Bitcoin", "Ethereum")
    # Sort by longest alias first to prefer more specific matches
    for alias, ticker in sorted(_TICKER_ALIASES.items(), key=lambda x: -len(x[0])):
        pattern = re.compile(rf"\b{re.escape(alias)}\b")
        if pattern.search(upper):
            if ticker in universe:
                return ticker

    # Pass 2: check raw ticker symbols with word-boundary matching
    # Sort longest first to prefer multi-char matches over substrings
    for ticker in sorted(universe, key=len, reverse=True):
        # Match whole-word or preceded by $ (common for tickers)
        pattern = re.compile(rf"(?:\b|\$){re.escape(ticker)}(?:\b|\$)")
        if pattern.search(upper):
            return ticker

    return None


# Quick sanity check
if __name__ == "__main__":
    for d in [date(2025, 7, 1), date(2025, 3, 8), date(2024, 2, 1)]:
        u = load_universe(d)
        print(f"{d}: {len(u)} tickers — BTC={ 'BTC' in u }, ONDO={ 'ONDO' in u }")
        print(f"  First 20: {sorted(u)[:20]}")

"""
Project configuration — paths, API endpoints, model settings.

All values overridable via environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
RAW_NEWS_DIR = DATA_ROOT / "news"          # unified Layer-1 raw news
RAW_OHLCV_DIR = DATA_ROOT / "ohlcv"        # OHLCV parquet files
FEATURES_DIR = DATA_ROOT / "features"      # Layer-3 feature store
CACHE_DIR = DATA_ROOT / "cache"           # extractions cache
LOG_DIR = PROJECT_ROOT                    # ingest.log lives here

# --- LLM ---
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8079/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama3-8b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.95"))
LLM_REPETITION_PENALTY = float(os.getenv("LLM_REPETITION_PENALTY", "1.1"))
LLM_MAX_TOKENS = 512                     # reserve budget for output
LLM_CONTEXT_SIZE = 8192                  # Llama-3-8B context window
LLM_TIMEOUT = 60.0                       # seconds per extraction call
LLM_MAX_RETRIES = 3

# Model/prompt versioning — change these to force re-extraction
MODEL_VERSION = "llama3-8b-q4km-v1"
PROMPT_VERSION = "1.0.0"

# --- Extraction ---
BODY_TRUNCATE_CHARS = 3000               # truncate body to this many chars
FEE_SHOT_COUNT = 3                       # number of few-shot exemplars
EXTRACTION_LOOKBACK_HOURS = 48           # novelty scoring window

# --- Feature aggregation ---
FEATURE_LOOKBACK_HOURS = [6, 24, 72]      # multiple lookback windows

# --- Binance ---
BINANCE_BASE_URL = "https://api.binance.com/api/v3"
BINANCE_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
BINANCE_INTERVALS = ["1h"]
BINANCE_LIMIT = 1000

# --- CryptoPanic ---
CRYPTOPANIC_DATA_ROOT = Path(os.getenv("CRYPTOPANIC_DATA_ROOT", "./data/cryptopanic"))
CRYPTOPANIC_AUTH_TOKEN = os.getenv("CRYPTOPANIC_AUTH_TOKEN", "")
CRYPTOPANIC_PLAN = os.getenv("CRYPTOPANIC_PLAN", "developer")

# --- RSS ---
RSS_DATA_ROOT = Path(os.getenv("RSS_DATA_ROOT", "./data/crypto_rss"))

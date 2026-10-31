"""WASAT settings. Values can be overridden in the .env file."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# OpenRouter API
API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.getenv("OPENROUTER_API_KEY", "")
REQUEST_TIMEOUT = 45

# Small model for unclear messages, large model only for hard ones.
SMALL_MODEL = os.getenv("WASAT_SMALL_MODEL", "meta-llama/llama-3.2-3b-instruct")
LARGE_MODEL = os.getenv("WASAT_LARGE_MODEL", "meta-llama/llama-3.3-70b-instruct")

# Fallback prices (USD per 1M tokens), used only if OpenRouter does not report a call's real cost.
PRICES = {
    "small": {"input": float(os.getenv("WASAT_SMALL_PRICE_IN", "0.02")),
              "output": float(os.getenv("WASAT_SMALL_PRICE_OUT", "0.02"))},
    "large": {"input": float(os.getenv("WASAT_LARGE_PRICE_IN", "0.13")),
              "output": float(os.getenv("WASAT_LARGE_PRICE_OUT", "0.40"))},
}

# The ML model answers on its own only when it is at least this confident.
FAST_THRESHOLD = float(os.getenv("WASAT_FAST_THRESHOLD", "0.80"))

# Verification where it pays: a mistake about money is expensive, so for these categories
# the ML model must be more confident before it answers alone.
HIGH_RISK_CATEGORIES = {"refund", "payment", "cancel"}
HIGH_RISK_THRESHOLD = float(os.getenv("WASAT_HIGH_RISK_THRESHOLD", "0.90"))

# Floors: cheap only counts if the answer is right and on time.
ACCURACY_FLOOR = float(os.getenv("WASAT_ACCURACY_FLOOR", "0.90"))
LATENCY_FLOOR_SECONDS = float(os.getenv("WASAT_LATENCY_FLOOR_SECONDS", "10"))

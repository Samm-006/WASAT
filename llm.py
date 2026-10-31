"""Calls AI models through OpenRouter and records tokens, latency and cost for every call.

Answers are cached in results/llm_cache.json, so re-running an evaluation does not spend
tokens again. A cached answer keeps the tokens, latency and cost of the original call.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from pathlib import Path
from typing import Any

import requests

import config

CACHE_PATH = Path(__file__).resolve().parent / "results" / "llm_cache.json"
_lock = threading.Lock()


def _load_cache() -> dict[str, Any]:
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict[str, Any]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=1), encoding="utf-8")


def _key(model: str, system: str, user: str, max_tokens: int) -> str:
    raw = json.dumps([model, system, user, max_tokens], ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def estimate_cost(tier: str, prompt_tokens: int, completion_tokens: int) -> float:
    price = config.PRICES[tier]
    return (prompt_tokens * price["input"] + completion_tokens * price["output"]) / 1_000_000


def call_model(model: str, system: str, user: str, *, tier: str, max_tokens: int = 60,
               use_cache: bool = True) -> dict[str, Any]:
    """Call a chat model once. Always returns a result dict and never raises."""
    key = _key(model, system, user, max_tokens)
    if use_cache:
        with _lock:
            hit = _load_cache().get(key)
        if hit:
            return {**hit, "cached": True}

    result: dict[str, Any] = {
        "ok": False, "content": "", "error": None, "model": model, "tier": tier,
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        "latency": 0.0, "cost": 0.0, "real_cost": None, "cached": False,
    }
    if not config.API_KEY:
        result["error"] = "OPENROUTER_API_KEY is missing in .env"
        return result

    started = time.perf_counter()
    for attempt in range(3):
        try:
            response = requests.post(
                config.API_URL,
                headers={"Authorization": f"Bearer {config.API_KEY}", "X-Title": "WASAT"},
                json={
                    "model": model,
                    "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": user}],
                    "temperature": 0,
                    "max_tokens": max_tokens,
                    "usage": {"include": True},  # ask OpenRouter to report the real cost
                },
                timeout=config.REQUEST_TIMEOUT,
            )
            if response.status_code == 429 and attempt < 2:  # rate limited: wait and retry
                time.sleep(6 * (attempt + 1))
                continue
            if response.status_code >= 400:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
            data = response.json()
            usage = data.get("usage") or {}
            result.update(
                ok=True,
                content=(data["choices"][0]["message"].get("content") or "").strip(),
                model=data.get("model", model),
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
                total_tokens=int(usage.get("total_tokens") or 0),
                real_cost=float(usage["cost"]) if usage.get("cost") is not None else None,
            )
            break
        except Exception as e:  # network error, HTTP error or unexpected response
            result["error"] = str(e)
            if attempt < 2:
                time.sleep(2)

    result["latency"] = round(time.perf_counter() - started, 3)
    result["cost"] = (result["real_cost"] if result["real_cost"] is not None
                      else estimate_cost(tier, result["prompt_tokens"], result["completion_tokens"]))

    if result["ok"] and use_cache:
        with _lock:
            cache = _load_cache()
            cache[key] = result
            _save_cache(cache)
    return result


def parse_json(content: str) -> dict[str, Any] | None:
    """Return the first JSON object in a model reply, or None."""
    match = re.search(r"\{.*?\}", content or "", flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None

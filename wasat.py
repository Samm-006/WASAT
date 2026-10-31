"""WASAT: decide how much AI each customer message really needs.

Level 0  ML model       TF-IDF + logistic regression, trained on Bitext. If it is confident,
                        it answers on its own: 0 AI tokens, about a millisecond. Money-related
                        categories (refund, payment, cancel) need a higher confidence.
Level 1  Small model    Asked only when the ML model is not confident. It may answer "unsure".
                        Its answer is accepted only if it agrees with the ML model's best guess.
Level 2  Large model    Asked only when the small model is unsure, invalid, or disagrees.
Fallback                If the AI service is unavailable, WASAT uses the ML model's best guess.

The baseline ("Large AI only") sends every message to the large model with a long, general prompt.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

import config
from llm import call_model, parse_json

ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data" / "bitext.csv"
TEST_PATH = ROOT / "data" / "bitext_test.json"
MODEL_PATH = ROOT / "results" / "fast_model.pkl"

CATEGORIES = {
    "account": "create, edit, delete or switch account; recover password; registration problems",
    "cancel": "cancellation fees",
    "contact": "contact customer service or a human agent",
    "delivery": "delivery options and delivery time",
    "feedback": "complaints and reviews",
    "invoice": "get or check an invoice",
    "order": "place, change, cancel or track an order",
    "payment": "payment methods and payment problems",
    "refund": "refund policy; get or track a refund",
    "shipping": "set up or change a shipping address",
    "subscription": "newsletter subscription",
}

# ------------------------------------------------------------------ prompts
# WASAT's prompts are short: the category list and the message, nothing else.
GUIDE_TEXT = "\n".join(f"{k} = {v}" for k, v in CATEGORIES.items())
COMPACT_SYSTEM = "Classify customer support messages. Reply with JSON only."
SMALL_USER = (GUIDE_TEXT + '\nIf you are not sure, use "unsure".\n'
              'Return {{"category": "<one category or unsure>"}}.\nMessage: {message}')
LARGE_USER = GUIDE_TEXT + '\nReturn {{"category": "<one category>"}}.\nMessage: {message}'

# The baseline uses a typical general-purpose support prompt and also writes a reply.
BASELINE_SYSTEM = (
    "You are a helpful, professional customer support assistant for an online store. Always be "
    "polite, empathetic and clear. Follow company policy: refunds are reviewed by a human agent; "
    "never promise a refund. Orders can be placed, changed, cancelled and tracked online. "
    "Shipping addresses can be changed until an order ships. Invoices are available in the account "
    "area. Customers can reach a human agent through live chat. Categories:\n" + GUIDE_TEXT
)
BASELINE_USER = (
    "Read the customer message below. Think about what the customer needs. Return JSON with keys "
    '"category" (one of the categories), "reason" (one sentence) and "reply" (a short, friendly '
    "reply to the customer).\n\nCustomer message: {message}"
)

# ------------------------------------------------------------------ Level 0
_model: list[Any] = []


def fast_predict(message: str) -> tuple[str, float]:
    """ML model: return (category, confidence). Loads the model once."""
    if not _model:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("ML model not found. Run: python3 prepare_data.py")
        _model.append(pickle.loads(MODEL_PATH.read_bytes()))
    probs = _model[0].predict_proba([message])[0]
    best = int(probs.argmax())
    return str(_model[0].classes_[best]), float(probs[best])


# ------------------------------------------------------------------ helpers
def _validate(call: dict[str, Any]) -> tuple[str | None, str | None]:
    """Cheap check of a model answer. Returns (category, None) or (None, problem)."""
    if not call.get("ok"):
        return None, f"model error: {call.get('error')}"
    parsed = parse_json(call.get("content", ""))
    if not parsed:
        return None, "reply was not valid JSON"
    label = str(parsed.get("category", "")).strip().lower()
    if label == "unsure":
        return None, "model said unsure"
    if label not in CATEGORIES:
        return None, f"unknown category '{label}'"
    return label, None


def _decision(label, level, steps, calls, started) -> dict[str, Any]:
    return {
        "category": label,
        "level": level,  # python | small | large | fallback | failed
        "steps": steps,
        "calls": [{k: c.get(k) for k in ("tier", "model", "ok", "error", "prompt_tokens",
                                          "completion_tokens", "total_tokens", "latency", "cost",
                                          "cached")} for c in calls],
        "ai_calls": len(calls),
        "total_tokens": sum(c.get("total_tokens", 0) for c in calls),
        "prompt_tokens": sum(c.get("prompt_tokens", 0) for c in calls),
        "cost": sum(c.get("cost", 0.0) for c in calls),
        # Model latency comes from the original call, so cached re-runs report the same time.
        "latency": round(sum(c.get("latency", 0.0) for c in calls) + (time.perf_counter() - started), 3),
    }


# ------------------------------------------------------------------ WASAT
def wasat_classify(message: str, *, use_cache: bool = True, threshold: float | None = None,
                   allow_large: bool = True, check_disagreement: bool = True,
                   risk_check: bool = True) -> dict[str, Any]:
    """Route one message. The keyword options exist for experiments (ablations, threshold sweep)."""
    started = time.perf_counter()
    steps: list[str] = []
    calls: list[dict[str, Any]] = []
    python_label, prob = fast_predict(message)
    needed = config.FAST_THRESHOLD if threshold is None else threshold
    high_risk = risk_check and python_label in config.HIGH_RISK_CATEGORIES
    if high_risk:
        needed = max(needed, config.HIGH_RISK_THRESHOLD)
    if prob >= needed:
        steps.append(f"Level 0 · ML model → {python_label} ({prob:.0%} sure, 0 tokens)")
        return _decision(python_label, "python", steps, calls, started)
    reason = "money-related, needs " + f"{needed:.0%}" if high_risk else f"needs {needed:.0%}"
    steps.append(f"Level 0 · ML model only {prob:.0%} sure ({python_label}, {reason}) → ask the small model")

    small = call_model(config.SMALL_MODEL, COMPACT_SYSTEM, SMALL_USER.format(message=message),
                       tier="small", max_tokens=20, use_cache=use_cache)
    calls.append(small)
    label, problem = _validate(small)
    if label and label == python_label:
        steps.append(f"Level 1 · Small model → {label} (agrees with the ML model)")
        return _decision(label, "small", steps, calls, started)
    if label and (not allow_large or not check_disagreement):
        steps.append(f"Level 1 · Small model → {label}")
        return _decision(label, "small", steps, calls, started)
    if label:
        problem = f"small model says {label}, ML model says {python_label}"
    if not allow_large:
        steps.append(f"Level 1 · Small model not trusted ({problem}) → the ML model's best guess")
        return _decision(python_label, "fallback", steps, calls, started)
    steps.append(f"Level 1 · Small model not trusted ({problem}) → large model")

    large = call_model(config.LARGE_MODEL, COMPACT_SYSTEM, LARGE_USER.format(message=message),
                       tier="large", max_tokens=20, use_cache=use_cache)
    calls.append(large)
    label, problem = _validate(large)
    if label:
        steps.append(f"Level 2 · Large model → {label}")
        return _decision(label, "large", steps, calls, started)
    if not large.get("ok"):  # AI service error: keep working with the ML model's best guess
        steps.append(f"AI unavailable ({problem}) → the ML model's best guess: {python_label}")
        return _decision(python_label, "fallback", steps, calls, started)
    steps.append(f"Level 2 · Large model failed ({problem})")
    return _decision(None, "failed", steps, calls, started)


# ------------------------------------------------------------------ baseline and ablations
def baseline_classify(message: str, *, use_cache: bool = True, **_) -> dict[str, Any]:
    """Baseline (Large AI only): the large model with a full prompt on every message."""
    started = time.perf_counter()
    call = call_model(config.LARGE_MODEL, BASELINE_SYSTEM, BASELINE_USER.format(message=message),
                      tier="large", max_tokens=200, use_cache=use_cache)
    label, problem = _validate(call)
    steps = [f"Large model with full prompt → {label or 'failed (' + str(problem) + ')'}"]
    return _decision(label, "large" if label else "failed", steps, [call], started)


def python_only_classify(message: str, **_) -> dict[str, Any]:
    """Ablation: the ML model on its own, no AI."""
    started = time.perf_counter()
    label, prob = fast_predict(message)
    return _decision(label, "python", [f"ML model only → {label} ({prob:.0%})"], [], started)


def small_only_classify(message: str, *, use_cache: bool = True, **_) -> dict[str, Any]:
    """Ablation: the small model on every message."""
    started = time.perf_counter()
    call = call_model(config.SMALL_MODEL, COMPACT_SYSTEM, LARGE_USER.format(message=message),
                      tier="small", max_tokens=20, use_cache=use_cache)
    label, problem = _validate(call)
    return _decision(label, "small" if label else "failed", [f"Small model only → {label or problem}"],
                     [call], started)

"""Experiments that test WASAT beyond the main comparison.

    python3 experiments.py

1. Ablation      : what each component adds (ML model only, small model only, partial WASAT, full WASAT)
2. Trade-off     : how accuracy and AI usage change with the ML model confidence threshold
3. Writing style : accuracy by Bitext writing-style flag (typos, colloquial, ...)
4. Robustness    : 30 synthetic messages written outside the dataset
5. AI outage     : the AI service fails completely; does WASAT keep working?

Results are saved to results/experiments.json and shown in the app. Model answers are cached,
so re-running is fast and almost free.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import wasat

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "experiments.json"
STRESS = ROOT / "data" / "stress_test.json"
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
FLAG_NAMES = {  # Bitext language-variation tags
    "B": "basic syntax", "I": "question form", "C": "coordinated", "N": "negation",
    "L": "lexical variation", "M": "morphological variation", "P": "polite", "Q": "colloquial",
    "W": "offensive", "K": "keyword-only", "E": "abbreviations", "Z": "errors and typos",
}


def run(classify, items, workers=6, **kw):
    def one(item):
        d = classify(item["message"], **kw)
        return {"id": item["id"], "label": item["label"], "flags": item.get("difficulty", ""),
                "category": d["category"], "level": d["level"], "ai_calls": d["ai_calls"],
                "tokens": d["total_tokens"], "cost": d["cost"], "latency": d["latency"]}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, items))


def score(rows):
    n = len(rows)
    correct = sum(r["category"] == r["label"] for r in rows)
    return {
        "n": n,
        "accuracy": round(correct / n, 4) if n else 0,
        "ai_calls": sum(r["ai_calls"] for r in rows),
        "ai_share": round(sum(r["ai_calls"] > 0 for r in rows) / n, 4) if n else 0,
        "tokens": sum(r["tokens"] for r in rows),
        "cost_usd": round(sum(r["cost"] for r in rows), 8),
        "avg_latency_s": round(sum(r["latency"] for r in rows) / n, 3) if n else 0,
    }


def outage(model, system, user, *, tier, max_tokens=60, use_cache=True):
    return {"ok": False, "content": "", "error": "simulated outage: AI service unavailable",
            "model": model, "tier": tier, "prompt_tokens": 0, "completion_tokens": 0,
            "total_tokens": 0, "latency": 0.0, "cost": 0.0, "cached": False}


def main() -> None:
    held_out = json.loads(wasat.TEST_PATH.read_text(encoding="utf-8"))["items"]
    stress = json.loads(STRESS.read_text(encoding="utf-8"))["items"]
    wasat.fast_predict("warm up")
    out: dict = {"run_at": datetime.now(timezone.utc).isoformat()}

    print("1/5 Ablation on 275 held-out messages ...")
    configs = {
        "AI-first (large model, full prompt)": (wasat.baseline_classify, {}),
        "Small model only": (wasat.small_only_classify, {}),
        "Python model only (no AI)": (wasat.python_only_classify, {}),
        "Python + small model (no large)": (wasat.wasat_classify, {"allow_large": False}),
        "WASAT without disagreement check": (wasat.wasat_classify, {"check_disagreement": False}),
        "WASAT without money check": (wasat.wasat_classify, {"risk_check": False}),
        "WASAT (Python + small + large)": (wasat.wasat_classify, {}),
    }
    out["ablation"] = {name: score(run(fn, held_out, **kw)) for name, (fn, kw) in configs.items()}

    print("2/5 Threshold trade-off ...")
    out["tradeoff"] = []
    for t in THRESHOLDS:
        s = score(run(wasat.wasat_classify, held_out, threshold=t))
        out["tradeoff"].append({"threshold": t, **s})

    print("3/5 Accuracy by writing style ...")
    rows = run(wasat.wasat_classify, held_out)
    base_rows = {r["id"]: r for r in run(wasat.baseline_classify, held_out)}
    styles = {}
    for letter, name in FLAG_NAMES.items():
        subset = [r for r in rows if letter in (r["flags"] or "")]
        if len(subset) >= 5:
            b = [base_rows[r["id"]] for r in subset]
            styles[f"{name} ({letter})"] = {"wasat": score(subset), "ai_first": score(b)}
    out["styles"] = styles

    print("4/5 Stress test: 30 new, messier messages ...")
    out["stress"] = {
        "AI-first": score(run(wasat.baseline_classify, stress)),
        "Python model only": score(run(wasat.python_only_classify, stress)),
        "WASAT": score(run(wasat.wasat_classify, stress)),
    }
    out["stress_rows"] = run(wasat.wasat_classify, stress)

    print("5/5 Simulated AI outage ...")
    with mock.patch.object(wasat, "call_model", outage):
        out["outage"] = {
            "held_out": {"AI-first": score(run(wasat.baseline_classify, held_out, use_cache=False)),
                         "WASAT": score(run(wasat.wasat_classify, held_out, use_cache=False))},
            "stress": {"AI-first": score(run(wasat.baseline_classify, stress, use_cache=False)),
                       "WASAT": score(run(wasat.wasat_classify, stress, use_cache=False))},
        }

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    def table(title, data):
        print(f"\n{title}")
        print(f"  {'':<38}{'accuracy':>9}{'AI calls':>10}{'tokens':>9}{'cost $':>11}{'avg s':>8}")
        for name, s in data.items():
            print(f"  {name:<38}{s['accuracy']:>9.1%}{s['ai_calls']:>10}{s['tokens']:>9}{s['cost_usd']:>11.5f}"
                  f"{s['avg_latency_s']:>8.2f}")

    table("ABLATION (275 held-out)", out["ablation"])
    print("\nTRADE-OFF (ML model threshold)")
    for r in out["tradeoff"]:
        print(f"  threshold {r['threshold']:<5} accuracy {r['accuracy']:.1%}  AI used on {r['ai_share']:.0%} of messages  "
              f"tokens {r['tokens']}")
    print("\nWRITING STYLE (WASAT vs Large AI only accuracy)")
    for name, s in styles.items():
        print(f"  {name:<32} n={s['wasat']['n']:<4} WASAT {s['wasat']['accuracy']:.1%}  Large AI only {s['ai_first']['accuracy']:.1%}"
              f"  AI used {s['wasat']['ai_share']:.0%}")
    table("ROBUSTNESS (30 synthetic messages)", out["stress"])
    table("AI OUTAGE (held-out)", out["outage"]["held_out"])
    table("AI OUTAGE (synthetic messages)", out["outage"]["stress"])
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()

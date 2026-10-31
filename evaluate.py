"""Compare Large AI only with WASAT on the 275 held-out Bitext messages.

    python3 evaluate.py              # uses cached model answers where available
    python3 evaluate.py --limit 20   # first 20 messages only
    python3 evaluate.py --no-cache   # force fresh model calls

A task is successful only if the category is correct AND the follow-up ticket was saved and
verified. Failed runs still count toward cost. Results are saved to results/latest.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import config
import wasat
from actions import run_task

RESULTS = Path(__file__).resolve().parent / "results" / "latest.json"


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]


def summarise(rows: list[dict], system: str) -> dict:
    runs = [r[system] for r in rows]
    n = len(runs)
    correct = sum(r["correct"] for r in runs)
    successes = sum(r["success"] for r in runs)
    cost = sum(r["cost"] for r in runs)
    tokens = sum(r["total_tokens"] for r in runs)
    latencies = [r["latency"] for r in runs]
    levels: dict[str, int] = {}
    for r in runs:
        levels[r["level"]] = levels.get(r["level"], 0) + 1
    accuracy = correct / n if n else 0.0
    return {
        "tasks": n,
        "correct": correct,
        "successes": successes,
        "accuracy": round(accuracy, 4),
        "ai_calls": sum(r["ai_calls"] for r in runs),
        "total_tokens": tokens,
        "prompt_tokens": sum(r["prompt_tokens"] for r in runs),
        "cost_usd": round(cost, 8),
        "tokens_per_success": round(tokens / successes, 1) if successes else None,
        "cost_per_success_usd": round(cost / successes, 8) if successes else None,
        "avg_latency_s": round(statistics.mean(latencies), 3) if latencies else 0.0,
        "p95_latency_s": round(p95(latencies), 3),
        "levels": levels,
        "meets_accuracy_floor": accuracy >= config.ACCURACY_FLOOR,
        "meets_latency_floor": p95(latencies) <= config.LATENCY_FLOOR_SECONDS,
    }


def run_one(classify, item: dict, tickets: Path, use_cache: bool) -> dict:
    decision = classify(item["message"], use_cache=use_cache)
    task = run_task(decision["category"], item["message"], tickets_path=tickets)
    correct = decision["category"] == item["label"]
    return {
        "category": decision["category"],
        "level": decision["level"],
        "correct": correct,
        "success": correct and task["verified"],
        "ai_calls": decision["ai_calls"],
        "total_tokens": decision["total_tokens"],
        "prompt_tokens": decision["prompt_tokens"],
        "cost": decision["cost"],
        "latency": decision["latency"],
        "steps": decision["steps"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--workers", type=int, default=6, help="parallel requests")
    args = parser.parse_args()

    if not wasat.TEST_PATH.exists():
        raise SystemExit("Test set not found. Run: python3 prepare_data.py")
    items = json.loads(wasat.TEST_PATH.read_text(encoding="utf-8"))["items"][: args.limit]
    wasat.fast_predict("warm up")  # load the ML model once, before timing

    with tempfile.TemporaryDirectory() as tmp:
        tickets = Path(tmp) / "tickets.json"

        def work(item):
            row = {k: item.get(k) for k in ("id", "message", "label", "difficulty")}
            row["baseline"] = run_one(wasat.baseline_classify, item, tickets, not args.no_cache)
            row["wasat"] = run_one(wasat.wasat_classify, item, tickets, not args.no_cache)
            return row

        rows = []
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            for i, row in enumerate(pool.map(work, items), 1):
                b, w = row["baseline"], row["wasat"]
                print(f"[{i:03d}/{len(items)}] {row['id']} {row['label']:<12} "
                      f"Large AI only={b['category'] or '-':<12} WASAT={w['category'] or '-':<12} via {w['level']}")
                rows.append(row)

    b, w = summarise(rows, "baseline"), summarise(rows, "wasat")
    savings = {
        "ai_calls_pct": round(100 * (1 - w["ai_calls"] / b["ai_calls"]), 1) if b["ai_calls"] else None,
        "tokens_pct": round(100 * (1 - w["total_tokens"] / b["total_tokens"]), 1) if b["total_tokens"] else None,
        "cost_pct": round(100 * (1 - w["cost_usd"] / b["cost_usd"]), 1) if b["cost_usd"] else None,
        "accuracy_change_pts": round(100 * (w["accuracy"] - b["accuracy"]), 1),
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({
        "dataset": "bitext",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "models": {"small": config.SMALL_MODEL, "large": config.LARGE_MODEL},
        "python_threshold": config.FAST_THRESHOLD,
        "floors": {"accuracy": config.ACCURACY_FLOOR, "p95_latency_s": config.LATENCY_FLOOR_SECONDS},
        "summary": {"baseline": b, "wasat": w, "savings": savings},
        "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'':<20}{'Large AI only':>14}{'WASAT':>12}")
    for label, key in [("Accuracy", "accuracy"), ("Successful tasks", "successes"), ("AI calls", "ai_calls"),
                       ("Tokens", "total_tokens"), ("Tokens / success", "tokens_per_success"),
                       ("Cost (USD)", "cost_usd"), ("Avg latency (s)", "avg_latency_s"),
                       ("p95 latency (s)", "p95_latency_s")]:
        print(f"{label:<20}{str(b[key]):>12}{str(w[key]):>12}")
    print(f"\nSavings: {savings}")
    print(f"WASAT levels used: {w['levels']}")
    print(f"Saved to {RESULTS}")


if __name__ == "__main__":
    main()

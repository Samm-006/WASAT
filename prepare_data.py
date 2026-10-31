"""Download the Bitext dataset, hold out a test set, and train WASAT's ML model.

    python3 prepare_data.py

Dataset: Bitext Customer Support LLM Chatbot Training Dataset (CDLA-Sharing 1.0)
https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset

Outputs
- data/bitext.csv         the full dataset (downloaded, not committed)
- data/bitext_test.json   275 held-out test messages (25 per category), never used for training
- results/fast_model.pkl  the trained ML model (TF-IDF + logistic regression)
"""

from __future__ import annotations

import csv
import json
import pickle
import random
import time
from collections import defaultdict

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from wasat import CATEGORIES, MODEL_PATH, RAW_PATH, TEST_PATH

REPO = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
TEST_PER_CATEGORY = 25
SEED = 7


def download() -> None:
    info = requests.get(f"https://huggingface.co/api/datasets/{REPO}", timeout=30).json()
    csv_files = [s["rfilename"] for s in info.get("siblings", []) if s["rfilename"].lower().endswith(".csv")]
    if not csv_files:
        raise SystemExit("No CSV file found in the dataset repository.")
    print(f"Downloading {csv_files[0]} ...")
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{csv_files[0]}"
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        with RAW_PATH.open("wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    print(f"Saved {RAW_PATH} ({RAW_PATH.stat().st_size / 1e6:.1f} MB)")


def load_rows() -> list[dict[str, str]]:
    with RAW_PATH.open(encoding="utf-8") as f:
        rows = []
        for r in csv.DictReader(f):
            label = (r.get("category") or "").strip().lower()
            text = (r.get("instruction") or "").strip()
            if label in CATEGORIES and text:
                rows.append({"message": text, "label": label, "intent": r.get("intent", ""),
                             "flags": r.get("flags", "")})
    return rows


def main() -> None:
    if not RAW_PATH.exists():
        download()
    rows = load_rows()

    # Hold out the same number of messages from every category.
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    rng = random.Random(SEED)
    test, train = [], []
    for label, items in sorted(by_label.items()):
        rng.shuffle(items)
        test.extend(items[:TEST_PER_CATEGORY])
        train.extend(items[TEST_PER_CATEGORY:])
    rng.shuffle(test)
    for i, r in enumerate(test, 1):
        r["id"] = f"B{i:03d}"
        r["difficulty"] = r.pop("flags") or "-"  # Bitext writing-style flags

    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=4.0),
    )
    started = time.perf_counter()
    model.fit([r["message"] for r in train], [r["label"] for r in train])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_bytes(pickle.dumps(model))
    TEST_PATH.write_text(json.dumps({
        "source": "Bitext Customer Support dataset (CDLA-Sharing 1.0), held-out sample",
        "items": test,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Loaded {len(rows):,} messages in {len(by_label)} categories.")
    print(f"Trained the ML model on {len(train):,} messages in {time.perf_counter() - started:.1f}s.")
    print(f"Held out {len(test)} test messages ({TEST_PER_CATEGORY} per category) -> {TEST_PATH.name}")


if __name__ == "__main__":
    main()

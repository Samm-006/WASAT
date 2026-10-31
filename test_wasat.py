"""Unit tests for WASAT. No internet or API key needed: the ML model and AI calls are faked.

    python3 -m unittest test_wasat.py -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import wasat
from actions import run_task
from llm import parse_json


def fake_call(*replies: str, ok: bool = True):
    """A stand-in for llm.call_model that returns the given replies in order."""
    queue = list(replies)

    def _call(model, system, user, *, tier, max_tokens=60, use_cache=True):
        content = queue.pop(0) if len(queue) > 1 else queue[0]
        return {"ok": ok, "content": content if ok else "", "error": None if ok else "service down",
                "model": model, "tier": tier, "prompt_tokens": 40, "completion_tokens": 10,
                "total_tokens": 50, "latency": 1.0, "cost": 0.0001, "cached": False}
    return _call


class RoutingTests(unittest.TestCase):
    """The ML model answers when confident; otherwise AI is asked, escalating only when needed."""

    def setUp(self):
        # ML model stand-in: confident about "track" messages, unsure about everything else.
        self.patch = mock.patch.object(wasat, "fast_predict",
                                       lambda m: ("order", 0.95 if "track" in m else 0.40))
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def route(self, message, *replies, ok=True, **kw):
        with mock.patch.object(wasat, "call_model", fake_call(*(replies or ("",)), ok=ok)):
            return wasat.wasat_classify(message, **kw)

    def test_confident_python_model_uses_no_ai(self):
        d = self.route("track my order")
        self.assertEqual((d["level"], d["category"], d["ai_calls"], d["total_tokens"]), ("python", "order", 0, 0))

    def test_small_model_accepted_when_it_agrees_with_python(self):
        d = self.route("where is it", '{"category":"order"}')
        self.assertEqual((d["level"], d["category"], d["ai_calls"]), ("small", "order", 1))

    def test_disagreement_goes_to_large_model(self):
        d = self.route("money back?", '{"category":"refund"}', '{"category":"refund"}')
        self.assertEqual((d["level"], d["category"], d["ai_calls"]), ("large", "refund", 2))

    def test_unsure_small_model_escalates(self):
        d = self.route("hmm", '{"category":"unsure"}', '{"category":"contact"}')
        self.assertEqual((d["level"], d["category"]), ("large", "contact"))

    def test_invalid_answer_escalates(self):
        d = self.route("hmm", "not json", '{"category":"feedback"}')
        self.assertEqual((d["level"], d["category"]), ("large", "feedback"))

    def test_unknown_category_rejected(self):
        d = self.route("hmm", '{"category":"pizza"}', '{"category":"pizza"}')
        self.assertEqual((d["level"], d["category"]), ("failed", None))

    def test_ai_outage_falls_back_to_python(self):
        d = self.route("money back?", ok=False)
        self.assertEqual((d["level"], d["category"]), ("fallback", "order"))

    def test_threshold_controls_ai_usage(self):
        self.assertEqual(self.route("track it", '{"category":"order"}', threshold=0.99)["level"], "small")
        self.assertEqual(self.route("money", threshold=0.30)["level"], "python")

    def test_money_related_needs_more_confidence(self):
        with mock.patch.object(wasat, "fast_predict", lambda m: ("refund", 0.85)):
            d = self.route("money back?", '{"category":"refund"}')
            self.assertEqual((d["level"], d["ai_calls"]), ("small", 1))  # 85% is not enough for money
            d = self.route("money back?", risk_check=False)
            self.assertEqual(d["level"], "python")  # without the check, 85% would be accepted

    def test_ablation_switches(self):
        no_check = self.route("money back?", '{"category":"refund"}', check_disagreement=False)
        self.assertEqual((no_check["level"], no_check["category"]), ("small", "refund"))
        no_large = self.route("hmm", '{"category":"unsure"}', allow_large=False)
        self.assertEqual((no_large["level"], no_large["ai_calls"]), ("fallback", 1))


class BaselineTests(unittest.TestCase):
    def test_baseline_always_calls_large_model(self):
        with mock.patch.object(wasat, "call_model", fake_call('{"category":"refund","reason":"x","reply":"y"}')):
            d = wasat.baseline_classify("I want my money back")
        self.assertEqual((d["category"], d["ai_calls"], d["calls"][0]["tier"]), ("refund", 1, "large"))


class ActionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tickets = Path(self.tmp.name) / "tickets.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_ticket_is_saved_and_verified(self):
        r = run_task("refund", "I want my money back", tickets_path=self.tickets)
        self.assertTrue(r["verified"])
        saved = json.loads(self.tickets.read_text())["tickets"]
        self.assertEqual((len(saved), saved[0]["team"]), (1, "refund"))

    def test_every_category_has_an_action(self):
        for category in wasat.CATEGORIES:
            self.assertTrue(run_task(category, "hello", tickets_path=self.tickets)["verified"], category)

    def test_no_category_means_no_ticket(self):
        r = run_task(None, "hello", tickets_path=self.tickets)
        self.assertFalse(r["verified"])
        self.assertFalse(self.tickets.exists())


class DataTests(unittest.TestCase):
    def test_parse_json(self):
        self.assertEqual(parse_json('Sure! {"category": "order"} done'), {"category": "order"})
        self.assertIsNone(parse_json("no json here"))

    def test_stress_test_labels_are_valid(self):
        items = json.loads((Path(__file__).parent / "data" / "stress_test.json").read_text())["items"]
        self.assertEqual(len(items), 30)
        self.assertTrue(all(i["label"] in wasat.CATEGORIES for i in items))

    def test_held_out_set_is_balanced(self):
        if not wasat.TEST_PATH.exists():
            self.skipTest("run prepare_data.py first")
        items = json.loads(wasat.TEST_PATH.read_text())["items"]
        counts = {c: sum(i["label"] == c for i in items) for c in wasat.CATEGORIES}
        self.assertEqual(set(counts.values()), {25})


if __name__ == "__main__":
    unittest.main()

# WASAT — Test Report

All tests use data the ML model never saw during training. Costs are the real amounts charged by OpenRouter.
Models: Llama 3.2 3B (small) and Llama 3.3 70B (large).

---

## 1. Main comparison — 275 held-out messages

| | Large AI only | WASAT |
|---|---|---|
| Accuracy | 95.6% | **99.3%** |
| AI calls | 275 | **11** (−96%) |
| Tokens | 92,244 | **1,904** (−97.9%) |
| Cost | $0.0193 | **$0.00015** (−99.2%) |
| Avg. response | 3.56 s | **0.04 s** |

WASAT sent 266 messages to the ML model (0 tokens), 7 to the small model and 2 to the large model.

---

## 2. Robustness test — synthetic messages

30 synthetic messages written with AI assistance to imitate real customers (typos, slang, new phrasing). They are not from the Bitext dataset and were never used for training. Each was labelled by hand with the correct category.

| | Accuracy | AI calls | Cost |
|---|---|---|---|
| Large AI only | 96.7% | 30 | $0.00265 |
| ML model only | 76.7% | 0 | $0 |
| **WASAT** | **96.7%** | **23** | **$0.00031** (−88%) |

When messages look unfamiliar, WASAT automatically sends more of them to AI.

---

## 3. Ablation — what each part adds

| Setup | Accuracy | AI calls | Cost |
|---|---|---|---|
| Large AI only (large model) | 95.6% | 275 | $0.0193 |
| Small model only | 88.0% | 275 | $0.0029 |
| **WASAT** (all three levels) | **99.3%** | **11** | **$0.00015** |

---

## 4. How we iterated

| Version | Change | Result |
|---|---|---|
| Prototype | Keyword rules → small AI → large AI, 40 hand-written messages | 85% accuracy |
| + category guide and "unsure" option | Small model could no longer fake confidence (it said "80% sure" every time) | 92.5% |
| + ML model sends mixed messages to the large model | | 100% on the prototype set |
| Real data | Replaced keyword rules with an ML model trained on Bitext; 275 held-out messages | 99.3%, 7 AI calls |
| + disagreement check and money check | Large model decides when the small model overrules the ML model; refunds, payments and cancellations need 90% ML model confidence | 99.3%, 11 AI calls |

The last step did not change accuracy on our test sets; it adds 4 AI calls as a safety margin on risky messages.
We kept it because a wrong refund or payment answer costs far more than an extra AI call.

---

## 5. Unit tests

17 tests covering routing, escalation, the disagreement check, the money check, the AI-outage fallback and tickets:

```bash
python3 -m unittest test_wasat.py -v
```

---

## Limitations

- The dataset is fairly templated, so the ML model does very well; real traffic would send more messages to AI.
- The robustness test uses 30 synthetic messages, a small sample.
- English only, one message at a time (no conversation history).
- We measured tokens, AI calls, cost and speed, not energy.

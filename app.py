"""WASAT app.   Run:  streamlit run app.py

Tabs: Live demo · Try it · Results · Robustness tests
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st

import wasat
from actions import run_task

ROOT = Path(__file__).resolve().parent
LEVEL_NAMES = {"python": "ML model", "rules": "ML model", "small": "Small AI", "large": "Large AI",
               "fallback": "ML model (AI down)", "failed": "Failed"}
RESULTS = ROOT / "results" / "latest.json"
EXPERIMENTS = ROOT / "results" / "experiments.json"
STRESS = ROOT / "data" / "stress_test.json"
LOGO = ROOT / "assets" / "wasat_logo.png"
ICON = ROOT / "assets" / "wasat_icon.png"
# Re-running the full evaluation spends API credit, so it is off on the public deployment
# unless WASAT_ALLOW_RERUN=1 is set (it is always on when running locally with a .env file).
ALLOW_RERUN = os.getenv("WASAT_ALLOW_RERUN", "1" if (ROOT / ".env").exists() else "0") == "1"

st.set_page_config(page_title="WASAT · وسط", page_icon=str(ICON) if ICON.exists() else None, layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.card {background:#f7f5ff;border:1px solid #e2dcf7;border-radius:16px;padding:16px 20px;margin-bottom:8px;}
.big {font-size:2.2rem;font-weight:700;color:#4f46e5;line-height:1.1;}
.muted {color:#6b6785;font-size:0.9rem;}
.flow {display:flex;gap:6px;align-items:center;margin:6px 0 10px 0;flex-wrap:wrap;}
.box {padding:6px 12px;border-radius:10px;font-size:0.85rem;font-weight:600;background:#eceaf3;color:#9a96ad;}
.on-python {background:#22a06b;color:#fff;} .on-small {background:#3b6fe0;color:#fff;}
.on-large {background:#7c3aed;color:#fff;} .on-fallback {background:#d97706;color:#fff;}
.on-failed {background:#c9372c;color:#fff;}
.msg {background:#fff;border:1px solid #e2dcf7;border-radius:16px;padding:14px 18px;margin:10px 0;}
.msgtext {font-size:1.05rem;color:#1a1033;margin-bottom:6px;}
.side {font-size:0.92rem;color:#3d3955;}
.tot {font-size:1.45rem;font-weight:700;color:#1a1033;line-height:1.2;}
.vs {font-size:0.85rem;color:#6b6785;}
.lbl {font-size:0.75rem;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;color:#6b6785;white-space:nowrap;}
.ok {color:#1f8a5b;font-weight:600;} .bad {color:#c9372c;font-weight:600;}
</style>
""", unsafe_allow_html=True)

if not wasat.MODEL_PATH.exists():
    st.error("The ML model has not been trained yet. In the terminal run:  `python3 prepare_data.py`")
    st.stop()

head_l, head_r = st.columns([1, 3])
with head_l:
    if LOGO.exists():
        st.image(str(LOGO), width=230)
with head_r:
    st.markdown("## The right amount of AI for the right task")
    st.markdown("WASAT checks every customer message with a fast **ML model** first. A **small AI model** "
                "handles unclear messages, and the **large AI model** is used only when needed.")


def flow(decision: dict) -> str:
    """The three levels as boxes; the ones WASAT used light up."""
    used = {c["tier"] for c in decision["calls"]}
    level = decision["level"]
    boxes = [("python", "1 · ML model · 0 tokens", True),
             ("small", "2 · Small model", "small" in used),
             ("large", "3 · Large model", "large" in used)]
    html = []
    for key, label, on in boxes:
        cls = f"on-{key}" if on else ""
        if level in ("fallback", "failed") and on and key != "python":
            cls = f"on-{level}"
        html.append(f'<span class="box {cls}">{label}</span>')
    return '<div class="flow">' + " → ".join(html) + "</div>"


def totals_html(t: dict) -> str:
    """Running totals: WASAT and Large AI only side by side, equally visible."""
    def cell(label, wasat_value, ai_first_value, note=""):
        return (f'<div class="card" style="flex:1"><div class="muted">{label}</div>'
                f'<div style="display:flex;flex-direction:column;gap:2px;margin-top:6px">'
                f'<div style="display:flex;justify-content:space-between;align-items:baseline;gap:8px"><span class="lbl" style="color:#4f46e5">WASAT</span><span class="tot" style="color:#4f46e5">{wasat_value}</span></div>'
                f'<div style="display:flex;justify-content:space-between;align-items:baseline;gap:8px"><span class="lbl">Large AI only</span><span class="tot" style="color:#6b6785;font-size:1.15rem">{ai_first_value}</span></div>'
                f'</div><div class="vs" style="color:#1f8a5b;font-weight:600;margin-top:4px">{note}&nbsp;</div></div>')
    tok_pct = 100 * (1 - t["w_tok"] / t["b_tok"]) if t["b_tok"] else 0
    cost_pct = 100 * (1 - t["w_cost"] / t["b_cost"]) if t["b_cost"] else 0
    n = max(t["n"], 1)
    return ('<div style="display:flex;gap:12px;flex-wrap:wrap">'
            + cell("AI calls", t["w_calls"], t["b_calls"], f"{t['b_calls'] - t['w_calls']} avoided")
            + cell("Tokens", f"{t['w_tok']:,}", f"{t['b_tok']:,}", f"{tok_pct:.0f}% fewer")
            + cell("Cost (USD)", f"${t['w_cost']:.5f}", f"${t['b_cost']:.5f}", f"{cost_pct:.0f}% lower")
            + cell("Avg time (s)", f"{t['w_lat'] / n:.2f}", f"{t['b_lat'] / n:.2f}",
                   f"{t['b_lat'] / t['w_lat']:.0f}× faster" if t["w_lat"] else "")
            + cell("Right category", f"{t['w_ok']}/{t['n']}", f"{t['b_ok']}/{t['n']}", "sorted to the right team")
            + "</div>")


def rerun_script(script: str, label: str) -> None:
    """Run evaluate.py or experiments.py from the app, then reload the page with the new results."""
    with st.spinner(f"Running {label}... this can take a few minutes."):
        proc = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        st.error(f"{script} failed:\n\n{proc.stderr[-1500:]}")
        return
    st.success(f"{label} finished.")
    st.rerun()


def demo_items() -> list[dict]:
    """10 random messages each run, picked from the 275 held-out test messages."""
    held_out = json.loads(wasat.TEST_PATH.read_text(encoding="utf-8"))["items"]
    return random.sample(held_out, 10)


tab_demo, tab_try, tab_results, tab_exp = st.tabs(
    ["Live demo", "Try it", "Results", "Robustness tests"])

# ---------------------------------------------------------------- Live demo
with tab_demo:
    st.markdown("### Same customer messages, two strategies")
    st.markdown("**Large AI only** (the usual approach) sends every message to the large model with a long prompt. "
                "**WASAT** lets the ML model decide first and uses AI only when a message needs it. "
                "Each run picks 10 random messages from the 275 held-out test messages, which were never used for training.")
    if st.button("Run live demo", type="primary", width="stretch"):
        t = {"n": 0, "b_calls": 0, "w_calls": 0, "b_tok": 0, "w_tok": 0,
             "b_cost": 0.0, "w_cost": 0.0, "b_lat": 0.0, "w_lat": 0.0, "b_ok": 0, "w_ok": 0}
        totals_box = st.empty()
        totals_box.markdown(totals_html(t), unsafe_allow_html=True)
        feed = st.container()
        with tempfile.TemporaryDirectory() as tmp:
            tickets = Path(tmp) / "tickets.json"
            for item in demo_items():
                msg, label = item["message"], item["label"]
                w = wasat.wasat_classify(msg)
                b = wasat.baseline_classify(msg)
                task = run_task(w["category"], msg, tickets_path=tickets)
                t["n"] += 1
                t["b_calls"] += b["ai_calls"]; t["w_calls"] += w["ai_calls"]
                t["b_tok"] += b["total_tokens"]; t["w_tok"] += w["total_tokens"]
                t["b_cost"] += b["cost"]; t["w_cost"] += w["cost"]
                t["b_lat"] += b["latency"]; t["w_lat"] += w["latency"]
                t["b_ok"] += b["category"] == label; t["w_ok"] += w["category"] == label
                w_mark = '<span class="ok">correct</span>' if w["category"] == label else '<span class="bad">wrong</span>'
                b_mark = '<span class="ok">correct</span>' if b["category"] == label else '<span class="bad">wrong</span>'
                with feed:
                    st.markdown(f"""
<div class="msg">
  <div class="msgtext">“{msg}”</div>
  {flow(w)}
  <div style="display:flex;gap:24px" class="side">
    <div style="flex:1"><b>WASAT</b> → {w['category']} {w_mark} · {w['total_tokens']} tokens · {w['ai_calls']} AI call{'' if w['ai_calls'] == 1 else 's'} · {w['latency']:.2f}s
      <br><span class="muted">{w['steps'][-1]}</span><br><span class="muted">Action: {task['action']}</span></div>
    <div style="flex:1"><b>Large AI only</b> → {b['category']} {b_mark} · {b['total_tokens']} tokens · 1 AI call · {b['latency']:.2f}s
      <br><span class="muted">Large model with full prompt, every time</span></div>
  </div>
</div>""", unsafe_allow_html=True)
                totals_box.markdown(totals_html(t), unsafe_allow_html=True)
                time.sleep(1.0)  # short pause so each message is easy to follow
        st.success(f"WASAT handled {t['n']} messages with {t['b_calls'] - t['w_calls']} fewer AI calls — "
                   f"{t['w_ok']}/{t['n']} in the right category vs {t['b_ok']}/{t['n']} for Large AI only.")

# ------------------------------------------------------------------- Try it
with tab_try:
    examples = {
        "Clear request: invoice": "I need to see my invoice",
        "Clear request: delivery": "when will my package arrive?",
        "Money-related: refund": "How can I get a refund for my last purchase?",
        "Slang": "yo when will my stuff arrive",
        "Vague": "something is off with the thing I bought from you, what do I do",
        "Two topics": "I paid but never got an invoice and now I want to cancel",
        "Unusual wording: track order": "I want to track my order 12345",
    }
    pick = st.selectbox("Pick an example or write your own", list(examples))
    message = st.text_area("Customer message", value=examples[pick], height=100)
    compare = st.toggle("Also compare with Large AI only (every message to the large model)", value=True)

    if st.button("Run WASAT", type="primary", width="stretch"):
        with st.spinner("WASAT is deciding how much AI this needs..."):
            w = wasat.wasat_classify(message)
            w_task = run_task(w["category"], message)
            b = wasat.baseline_classify(message) if compare else None
        st.session_state["live"] = (w, w_task, b)

    if "live" in st.session_state:
        w, w_task, b = st.session_state["live"]
        left, right = st.columns(2)
        with left:
            st.markdown("### WASAT")
            st.markdown(flow(w), unsafe_allow_html=True)
            for step in w["steps"]:
                st.markdown(f"- {step}")
            m1, m2, m3 = st.columns(3)
            m1.metric("AI calls", w["ai_calls"])
            m2.metric("Tokens", w["total_tokens"])
            m3.metric("Time", f"{w['latency']:.2f}s")
            st.markdown(f"**Category:** `{w['category']}` · **Action:** {w_task['action']} · "
                        f"**Verified:** {'yes' if w_task['verified'] else 'no'}")
            if w_task["reply"]:
                st.info(w_task["reply"])
        with right:
            if b:
                st.markdown("### Large AI only")
                st.markdown(f"- {b['steps'][0]}")
                m1, m2, m3 = st.columns(3)
                m1.metric("AI calls", b["ai_calls"])
                m2.metric("Tokens", b["total_tokens"])
                m3.metric("Time", f"{b['latency']:.2f}s")
                st.markdown(f"**Category:** `{b['category']}`")
                if b["total_tokens"]:
                    saved = b["total_tokens"] - w["total_tokens"]
                    if saved > 0:
                        st.success(f"WASAT used **{saved} fewer tokens** ({100 * saved / b['total_tokens']:.0f}% less).")
                    elif saved == 0:
                        st.info("Both used the same number of tokens for this message.")
                    else:
                        used = "the small and large AI" if w["ai_calls"] > 1 else "AI"
                        note = (f"WASAT used **{-saved} more tokens** on this message. The ML model was not sure, "
                                f"so WASAT asked {used} to be safe.")
                        if RESULTS.exists():
                            summary = json.loads(RESULTS.read_text(encoding="utf-8"))["summary"]
                            note += (f" This happens on a few hard messages: across all {summary['wasat']['tasks']} "
                                     f"test messages, WASAT used {summary['savings']['tokens_pct']}% fewer tokens.")
                        st.warning(note)
        errors = [c["error"] for c in w["calls"] + (b["calls"] if b else []) if c.get("error")]
        if errors:
            st.warning("Model error: " + errors[0])

# ------------------------------------------------------------------ Results
with tab_results:
    if ALLOW_RERUN and st.button("Re-run evaluation (275 held-out messages)", width="stretch"):
        rerun_script("evaluate.py", "Evaluation")
    if not RESULTS.exists():
        st.info("No results yet. Press the button above, or run  `python3 evaluate.py`  in the terminal.")
    else:
        data = json.loads(RESULTS.read_text(encoding="utf-8"))
        b, w, sv = data["summary"]["baseline"], data["summary"]["wasat"], data["summary"]["savings"]
        cost = lambda s: s.get("cost_usd", s.get("est_cost_usd", 0))  # noqa: E731 (older result files)
        st.markdown(f"#### {w['tasks']} held-out messages from the Bitext dataset")
        st.caption(f"Small model: {data['models']['small']} · Large model: {data['models']['large']} · "
                   f"Run: {data['run_at'][:16].replace('T', ' ')} UTC")

        speedup = b["avg_latency_s"] / w["avg_latency_s"] if w["avg_latency_s"] else 0
        cols = st.columns(5)
        for col, label, value, sub in [
            (cols[0], "Accuracy", f"{100 * w['accuracy']:.1f}%", f"Large AI only {100 * b['accuracy']:.1f}%"),
            (cols[1], "Fewer AI calls", f"{sv['ai_calls_pct']}%", f"{b['ai_calls']} → {w['ai_calls']}"),
            (cols[2], "Fewer tokens", f"{sv['tokens_pct']}%", f"{b['total_tokens']:,} → {w['total_tokens']:,}"),
            (cols[3], "Lower cost", f"{sv['cost_pct']}%", f"${cost(b):.5f} → ${cost(w):.5f}"),
            (cols[4], "Faster", f"{speedup:.0f}×" if speedup >= 2 else f"{speedup:.1f}×",
             f"avg {b['avg_latency_s']:.2f}s → {w['avg_latency_s']:.2f}s"),
        ]:
            col.markdown(f'<div class="card"><div class="muted">{label}</div><div class="big">{value}</div>'
                         f'<div class="muted">{sub}</div></div>', unsafe_allow_html=True)

        lv = w["levels"]
        n_ml = lv.get("python", 0) + lv.get("rules", 0)
        n_small, n_large = lv.get("small", 0), lv.get("large", 0)
        st.markdown("#### In plain words")
        st.markdown(
            f"- **Who answered:** the ML model answered **{n_ml} of {w['tasks']}** messages by itself (0 tokens). "
            f"Only **{n_small + n_large}** needed AI: {n_small} the small AI, {n_large} the large AI.\n"
            f"- **Accuracy:** WASAT got **{w['successes']}** right, Large AI only got **{b['successes']}** right.\n"
            f"- **Cost:** WASAT cost **\\${cost(w):.5f}** in total, Large AI only cost **\\${cost(b):.5f}**.\n"
            f"- **Speed:** WASAT replied in **{w['avg_latency_s']:.2f}s** on average, "
            f"Large AI only in **{b['avg_latency_s']:.2f}s**.")

        st.markdown("#### Tokens and cost as messages come in")
        st.caption("Each line adds up the running total as the 275 messages are handled one by one. "
                   "A steeper line means more tokens or more money spent.")
        run_tok_b = run_tok_w = run_cost_b = run_cost_w = 0
        tok_rows, cost_rows = [], []
        for r in data["rows"]:
            run_tok_b += r["baseline"]["total_tokens"]; run_tok_w += r["wasat"]["total_tokens"]
            run_cost_b += r["baseline"]["cost"]; run_cost_w += r["wasat"]["cost"]
            tok_rows.append({"Large AI only": run_tok_b, "WASAT": run_tok_w})
            cost_rows.append({"Large AI only": run_cost_b, "WASAT": run_cost_w})
        idx = pd.Index(range(1, len(tok_rows) + 1), name="Messages handled")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Tokens used (running total)**")
            st.line_chart(pd.DataFrame(tok_rows, index=idx), color=["#7c3aed", "#22a06b"],
                          x_label="Messages handled", y_label="Tokens")
        with c2:
            st.markdown("**Cost in USD (running total)**")
            st.line_chart(pd.DataFrame(cost_rows, index=idx), color=["#7c3aed", "#22a06b"],
                          x_label="Messages handled", y_label="USD")

        st.markdown("#### Side by side")
        fmt_cost = lambda v: f"${v:.5f}" if v is not None else "-"  # noqa: E731
        fmt_small = lambda v: f"${v:.7f}" if v is not None else "-"  # noqa: E731
        table = [
            ("Accuracy (right category and ticket verified)", f"{100 * b['accuracy']:.1f}%", f"{100 * w['accuracy']:.1f}%"),
            ("Successful tasks", f"{b['successes']} / {b['tasks']}", f"{w['successes']} / {w['tasks']}"),
            ("AI calls", f"{b['ai_calls']}", f"{w['ai_calls']}"),
            ("Tokens", f"{b['total_tokens']:,}", f"{w['total_tokens']:,}"),
            ("Tokens per successful task", f"{b['tokens_per_success']}", f"{w['tokens_per_success']}"),
            ("Total cost (USD)", fmt_cost(cost(b)), fmt_cost(cost(w))),
            ("Cost per successful task (USD)", fmt_small(b.get("cost_per_success_usd")), fmt_small(w.get("cost_per_success_usd"))),
            ("Average reply time", f"{b['avg_latency_s']:.2f}s", f"{w['avg_latency_s']:.2f}s"),
            ("Slowest 5% of replies (p95)", f"{b['p95_latency_s']:.2f}s", f"{w['p95_latency_s']:.2f}s"),
        ]
        st.dataframe(pd.DataFrame(table, columns=["Measure", "Large AI only", "WASAT"]),
                     hide_index=True, width="stretch")
        st.caption(f"Challenge floors: accuracy at least {100 * data['floors']['accuracy']:.0f}% "
                   f"({'met' if w['meets_accuracy_floor'] else 'not met'}), slowest 5% of replies under "
                   f"{data['floors']['p95_latency_s']}s ({'met' if w['meets_latency_floor'] else 'not met'}).")

        with st.expander("Every message (275 rows)"):
            st.dataframe(pd.DataFrame([{
            "ID": r["id"], "Expected": r["label"], "Large AI only": r["baseline"]["category"],
            "WASAT": r["wasat"]["category"], "Answered by": LEVEL_NAMES.get(r["wasat"]["level"], r["wasat"]["level"]),
            "WASAT tokens": r["wasat"]["total_tokens"], "Large AI only tokens": r["baseline"]["total_tokens"],
            "WASAT result": "correct" if r["wasat"]["success"] else "wrong", "Message": r["message"],
        } for r in data["rows"]]), width="stretch", hide_index=True)

# -------------------------------------------------------------- Robustness tests
with tab_exp:
    if ALLOW_RERUN and st.button("Re-run robustness tests", width="stretch"):
        rerun_script("experiments.py", "Robustness tests")
    if not EXPERIMENTS.exists():
        st.info("No results yet. Press the button above, or run  `python3 experiments.py`  in the terminal.")
    else:
        ex = json.loads(EXPERIMENTS.read_text(encoding="utf-8"))

        s_ = ex["stress"]
        big, ml, w_ = s_["AI-first"], s_["Python model only"], s_["WASAT"]
        rows_s = ex.get("stress_rows", [])
        messages = {i["id"]: i["message"] for i in json.loads(STRESS.read_text(encoding="utf-8"))["items"]}
        lv_s = {k: sum(r["level"] == k for r in rows_s) for k in ("python", "small", "large")}

        st.markdown(f"#### {w_['n']} synthetic messages, not from the dataset")
        st.markdown("**What if customers write with typos, slang or unclear wording?** We wrote "
                    f"{w_['n']} new messages with AI assistance and labelled each one by hand. "
                    "None of them were used for training.")

        cols = st.columns(3)
        for col, (name, v) in zip(cols, [("Large AI only", big), ("ML model only (no AI)", ml), ("WASAT", w_)]):
            col.markdown(
                f'<div class="card"><div class="muted">{name}</div>'
                f'<div class="big">{v["accuracy"]:.1%}</div><div class="muted">right category</div>'
                f'<div class="vs" style="margin-top:6px">{v["ai_calls"]} AI calls · ${v["cost_usd"]:.5f} · '
                f'{v["avg_latency_s"]:.2f}s avg</div></div>', unsafe_allow_html=True)

        st.markdown("#### In plain words")
        st.markdown(
            f"- **The ML model alone is not enough here:** it got only **{ml['accuracy']:.1%}** right, "
            "because these messages look different from its training data.\n"
            f"- **WASAT noticed and asked AI more often:** the ML model answered **{lv_s['python']}** messages, "
            f"the small AI **{lv_s['small']}** and the large AI **{lv_s['large']}**.\n"
            f"- **Same accuracy as Large AI only:** both got **{w_['accuracy']:.1%}** right.\n"
            f"- **Much cheaper:** WASAT cost **\\${w_['cost_usd']:.5f}**, Large AI only **\\${big['cost_usd']:.5f}** "
            f"({100 * (1 - w_['cost_usd'] / big['cost_usd']):.0f}% lower), using "
            f"{w_['tokens']:,} tokens instead of {big['tokens']:,}.")

        st.markdown("#### Side by side")
        st.dataframe(pd.DataFrame([
            ("Right category", f"{big['accuracy']:.1%}", f"{ml['accuracy']:.1%}", f"{w_['accuracy']:.1%}"),
            ("AI calls", str(big["ai_calls"]), str(ml["ai_calls"]), str(w_["ai_calls"])),
            ("Tokens", f"{big['tokens']:,}", f"{ml['tokens']:,}", f"{w_['tokens']:,}"),
            ("Cost (USD)", f"${big['cost_usd']:.5f}", f"${ml['cost_usd']:.5f}", f"${w_['cost_usd']:.5f}"),
            ("Average reply time", f"{big['avg_latency_s']:.2f}s", f"{ml['avg_latency_s']:.2f}s",
             f"{w_['avg_latency_s']:.2f}s"),
        ], columns=["Measure", "Large AI only", "ML model only", "WASAT"]), hide_index=True, width="stretch")
        st.caption("If the AI service goes down, WASAT still answers with its ML model; Large AI only cannot answer at all.")

        if rows_s:
            with st.expander(f"Every message ({len(rows_s)} rows)"):
                st.dataframe(pd.DataFrame([{
                    "ID": r["id"], "Message": messages.get(r["id"], ""), "Expected": r["label"],
                    "WASAT": r["category"], "Answered by": LEVEL_NAMES.get(r["level"], r["level"]),
                    "Tokens": r["tokens"], "Result": "correct" if r["category"] == r["label"] else "wrong",
                } for r in rows_s]), width="stretch", hide_index=True)

        with st.expander("More technical tests (for reviewers)"):
            st.caption("Ablation: what each part of WASAT adds, on the 275 held-out messages.")
            keep = ["AI-first (large model, full prompt)", "Small model only", "WASAT (Python + small + large)"]
            st.dataframe(pd.DataFrame([{"Setup": k.replace("Python", "ML").replace("AI-first", "Large AI only"),
                                        "Accuracy": f"{v['accuracy']:.1%}", "AI calls": v["ai_calls"],
                                        "Tokens": v["tokens"], "Cost (USD)": f"${v['cost_usd']:.5f}",
                                        "Avg time (s)": f"{v['avg_latency_s']:.2f}"}
                                       for k, v in ex["ablation"].items() if k in keep]),
                         hide_index=True, width="stretch")
            st.caption("Each AI model on its own versus WASAT, which uses AI only when the ML model is not sure.")

st.divider()
st.caption("Dataset: Bitext Customer Support (CDLA-Sharing 1.0). Tickets are local demo data; no real actions are taken.")

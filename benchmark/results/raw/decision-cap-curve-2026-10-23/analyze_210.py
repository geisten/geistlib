#!/usr/bin/env python3
"""#210 B: accuracy/time over the reasoning cap, always-reasoning and two-stage, offline.

No new inference. Greedy decoding is deterministic (#171), so reasoning at cap k is the
first k tokens of the cap-2048 generation: a generation of L <= k tokens is unchanged, a
longer one is cut to k tokens (stop = "limit") and re-scored with the campaign's own
eval_decision.generation_outcome. The cut prefix is detokenized from the model's GGUF
vocabulary (ggufvocab.py); that detokenizer reproduces all 667 recorded generations byte
for byte, and the cap-512 truncation reproduces every measured cap-512 outcome.

Time per reasoning call at cap k (no new measurement): a piecewise-linear function of the
generated-token count through measured points per question --
  (0, prefill)  prefill = selected-row call time x reasoning/direct prompt-token ratio,
  (min(L, 512), measured cap-512 call time),
  (L, measured cap-2048 re-run time) for the 164 questions re-run at cap 2048
The prefill estimate is capped at the measured cap-512 call time (it exceeds it on short
generations where the direct call ran hotter). Caps 512 and 2048 are therefore exact
measurements; the others interpolate linearly. Decode slows with context, so a chord
between measured points over-states intermediate times slightly (savings are conservative).

Two-stage thresholds come from the calibration split only (as in #204). Exploratory:
the Pareto selection is made on the test set.

Usage: analyze_210.py docs/DECISION_CAMPAIGN_PLAN.json MODEL.gguf > DECISION_CAP_CURVE_2026-10-23.json
"""
import json
import math
import sys
from pathlib import Path

import ggufvocab

C = Path("/Users/germar/workspace/geistlib-campaigns/171")
PLAN, MODEL = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(C / "src/tools"))
import bench_decision as bench  # noqa: E402
import decision_metrics as metrics  # noqa: E402
import eval_decision as ev  # noqa: E402
import eval_decision_campaign as camp  # noqa: E402

CAPS = (256, 512, 768, 1024, 1536, 2048)
SHARES = (10, 20, 30, 40, 50, 60, 70, 80, 90)  # routed fraction of calibration, %
VOCAB = 248320
detok = ggufvocab.Detok(MODEL)
decl = json.loads(PLAN.read_text())["two_stage_policy_204"]

# ---- thresholds: calibration split only (the #204 rule, on a finer share grid) ----
cal = json.loads((C / "run-calibration250b/calibration.json").read_text())
T = cal["temperature"]
crows = [json.loads(line) for line in (C / "cases-calibration250/calibration.jsonl").read_text().splitlines()]
cdone, _, _ = camp.collect(C / "run-calibration250b", crows, {"chat_direct": ("decision_dense",)}, 0, 0, "metal")
cconf = sorted(max(math.exp(x) for x in metrics.log_probabilities(cdone["chat_direct"][i][0]["logits"], T))
               for i in range(len(crows)))
assert len(cconf) == 250
policies = {"always_direct": -math.inf}
for s in SHARES:
    policies[f"route_{s}"] = cconf[round(s / 100 * 250)]
for s in (30, 50, 70):
    assert policies[f"route_{s}"] == decl["secondary_fixed_coverage"][f"route_{s}"]["threshold"]
policies["primary_204"] = decl["primary"]["threshold"]

# ---- test data ----
rows = [json.loads(line) for line in (C / "cases-test/test.jsonl").read_text().splitlines()]
eos = json.loads((C / "run-test2/manifest.json").read_text())["binding"]["gguf_metadata"]["eos_token_id"]
old, _, _ = camp.collect(C / "run-test2", rows, ev.CORE_PROFILES, 0, 0, "metal")
ids = json.loads((C / "cap2048_question_ids.json").read_text())
by_qid = {r["question_id"]: r for r in rows}
new, _, _ = camp.collect(C / "run-cap2048", [by_qid[q] for q in ids], {"chat_reasoning": ("generate_long",)},
                         0, 0, "metal")

# Detokenizer check: every recorded generation (EOS has no surface bytes).
for src in (old["chat_reasoning"], new["chat_reasoning"]):
    for s in src.values():
        g = s[0]["generated_ids"]
        assert detok.decode(g[:-1] if g and g[-1] == eos else g).hex() == s[0]["generated_utf8_hex"]


def truncate(g, k):
    gen = g["generated_ids"]
    if len(gen) <= k:
        return g
    return {**g, "generated_ids": gen[:k], "generation_stop": "limit",
            "generated_utf8_hex": detok.decode(gen[:k]).hex(), "generated_text_complete": True}


def interp(knots, n):
    for (x0, y0), (x1, y1) in zip(knots, knots[1:]):
        if n == x1:
            return y1
        if n < x1:
            return y0 + (y1 - y0) * (n - x0) / (x1 - x0)
    return knots[-1][1]


Q = []
prefix_identical = prefill_capped = nonmonotone = 0
for i, r in enumerate(rows):
    q = r["question_id"]
    sel = [s for s in old["chat_direct"][i] if s["mode"] == "decision_selected"][0]
    p = [math.exp(x) for x in metrics.log_probabilities(sel["logits"], T)]
    k = max(range(4), key=lambda j: sel["logits"][j])
    g512 = old["chat_reasoning"][i][0]
    full = g512
    if q in ids:
        rerun = new["chat_reasoning"][ids.index(q)][0]
        a, b = rerun["generated_ids"], g512["generated_ids"]
        prefix_identical += (a[:len(b)] == b) if g512["generation_stop"] == "limit" else (a == b)
        # The 3 determinism questions finished under 512: keep the cap-512 call, as #204 does.
        if g512["generation_stop"] == "limit":
            full = rerun
    L = len(full["generated_ids"])
    out = {}
    for cap in CAPS:
        o = ev.generation_outcome(truncate(full, cap), r, True, eos, cap, VOCAB)
        out[cap] = {"correct": o["correct"], "capped": o["truncated"], "gen": o["generated_tokens"]}
    m512 = ev.generation_outcome(g512, r, True, eos, 512, VOCAB)
    assert all(m512[x] == ev.generation_outcome(truncate(full, 512), r, True, eos, 512, VOCAB)[x]
               for x in ("predicted_index", "correct", "valid", "invalid_reason", "truncated", "generated_tokens"))
    assert truncate(full, 512)["generated_utf8_hex"] == g512["generated_utf8_hex"]
    pre = sel["elapsed_ms"] * full["prompt_tokens"] / sel["prompt_tokens"]
    knots = [(0, min(pre, g512["elapsed_ms"])), (len(g512["generated_ids"]), g512["elapsed_ms"])]
    prefill_capped += pre > g512["elapsed_ms"]
    if full is not g512:
        knots.append((L, full["elapsed_ms"]))
        nonmonotone += full["elapsed_ms"] < g512["elapsed_ms"]
    ms = {cap: interp(knots, min(cap, L)) for cap in CAPS}
    # Sensitivity (lower range): tokens past 512 at the cap-512 call's own decode rate, never above the
    # measured cap-2048 time; finished generations stay measured.
    rate512 = (knots[1][1] - knots[0][1]) / knots[1][0]
    ms_fast = {cap: ms[cap] if cap <= 512 or cap >= L else min(full["elapsed_ms"], knots[1][1] + (cap - 512) * rate512)
               for cap in CAPS}
    Q.append({"q": q, "c": p[k], "direct": k == r["target_index"], "sel_ms": sel["elapsed_ms"],
              "ptok_direct": sel["prompt_tokens"], "ptok_rsn": full["prompt_tokens"], "L": L,
              "out": out, "ms": ms, "ms_fast": ms_fast, "measured_ms": {512: g512["elapsed_ms"], 2048: full["elapsed_ms"]},
              "capped2048": out[2048]["capped"]})
assert len(Q) == 500 and prefix_identical == len(ids)
assert sum(x["out"][512]["correct"] for x in Q) == 327 and sum(x["out"][512]["capped"] for x in Q) == 164
assert sum(x["out"][2048]["correct"] for x in Q) == 395 and sum(x["capped2048"] for x in Q) == 87
assert all(x["ms"][2048] == x["measured_ms"][2048] for x in Q)
assert all(x["ms"][512] == x["measured_ms"][512] for x in Q)

ref = {x["q"]: x["out"][2048]["correct"] for x in Q}
ref_pess = {x["q"]: x["out"][2048]["correct"] or x["capped2048"] for x in Q}
ref_time = sum(x["measured_ms"][2048] for x in Q) / 1000


def evaluate(threshold, cap, direct_call=True):
    """threshold None = always-reasoning (no direct call)."""
    routed = [threshold is None or x["c"] < threshold for x in Q]
    pol, pess, ms, ms_fast, gen, ptok = {}, {}, [], [], 0, 0
    for x, rt in zip(Q, routed):
        o = x["out"][cap]
        pol[x["q"]] = o["correct"] if rt else x["direct"]
        # Pessimistic (as #204): the 87 questions no tested cap completes count as correct
        # in any arm that reasons on them and in the reference.
        pess[x["q"]] = (o["correct"] or x["capped2048"]) if rt else x["direct"]
        t = (x["sel_ms"] if threshold is not None else 0.) + (x["ms"][cap] if rt else 0.)
        ms.append(t)
        ms_fast.append(t - (x["ms"][cap] - x["ms_fast"][cap] if rt else 0.))
        gen += o["gen"] if rt else 0
        ptok += (x["ptok_direct"] if threshold is not None else 0) + (x["ptok_rsn"] if rt else 0)
    pm = metrics.paired_quality(pol, ref, 2.)
    pp = metrics.paired_quality(pess, ref_pess, 2.)
    n_r = sum(routed)
    return {"routed": n_r, "routed_fraction": n_r / 500, "accuracy": sum(pol.values()) / 500,
            "capped_routed": sum(x["out"][cap]["capped"] for x, rt in zip(Q, routed) if rt),
            "difference_pp": pm["difference_pp"], "gains": pm["gains"], "losses": pm["losses"],
            "interval_pp": pm["interval_pp"], "interval_pessimistic_pp": pp["interval_pp"],
            "noninferior_established": pm["noninferior"] and pp["noninferior"],
            "model_calls_per_question": ((500 if threshold is not None else 0) + n_r) / 500,
            "prompt_tokens": ptok, "generated_tokens": gen,
            "time_s": {"sum": sum(ms) / 1000, "mean": sum(ms) / 500000,
                       "p50": bench.percentile(ms, .5) / 1000, "p95": bench.percentile(ms, .95) / 1000},
            "time_vs_always_reasoning_2048": sum(ms) / 1000 / ref_time,
            "time_vs_always_reasoning_2048_fast_decode_sensitivity": sum(ms_fast) / 1000 / ref_time}


points = []
d = evaluate(-math.inf, 2048)
d.update(policy="always_direct", threshold=None, cap=None, calibration_routed_share=0.)
points.append(d)
for cap in CAPS:
    e = evaluate(None, cap)
    e.update(policy="always_reasoning", threshold=None, cap=cap, calibration_routed_share=1.)
    points.append(e)
    for name, t in policies.items():
        if name == "always_direct":
            continue
        e = evaluate(t, cap)
        share = int(name.split("_")[1]) / 100 if name.startswith("route_") else 1 - decl["primary"]["calibration_coverage_direct"]
        e.update(policy=name, threshold=t, cap=cap, calibration_routed_share=share)
        points.append(e)

# Pareto front on (mean time, accuracy as measured): no other point is at most as slow and at least as accurate,
# with one strictly better.
for p in points:
    p["pareto"] = not any(o is not p and o["time_s"]["sum"] <= p["time_s"]["sum"] and o["accuracy"] >= p["accuracy"]
                          and (o["time_s"]["sum"] < p["time_s"]["sum"] or o["accuracy"] > p["accuracy"])
                          for o in points)

# Time-model check on the 164 re-run questions: the cap-2048 re-run's own mean decode rate,
# extrapolated back to 512 tokens, against the measured cap-512 call (alternative model, not used).
alt = []
for x in Q:
    if x["q"] in ids and x["L"] > 512:
        pre = x["sel_ms"] * x["ptok_rsn"] / x["ptok_direct"]
        alt.append((pre + 512 * (x["measured_ms"][2048] - pre) / x["L"], x["measured_ms"][512]))

out = {
    "note": "Exploratory (#210 B). difference = policy - always-reasoning at cap 2048; negative = reasoning better. "
            "Thresholds from the calibration split only; Pareto selection on the test set.",
    "temperature": T, "caps": CAPS, "thresholds": policies | {"always_direct": None},
    "checks": {"detokenizer_generations_reproduced": len(old["chat_reasoning"]) + len(new["chat_reasoning"]),
               "cap2048_prefix_identical_to_cap512": prefix_identical,
               "cap512_outcomes_reproduced": 500, "cap512_correct": 327, "cap2048_correct": 395,
               "prefill_estimate_capped_at_measured_cap512": prefill_capped,
               "rerun_faster_than_cap512_run": nonmonotone,
               "alt_single_rate_model_cap512_sum_ratio": sum(a for a, _ in alt) / sum(b for _, b in alt)},
    "always_reasoning_2048_time_s": ref_time,
    "points": points,
    "per_question": [{"question_id": x["q"], "confidence_calibrated": x["c"], "direct_correct": x["direct"],
                      "reasoning_tokens_full": x["L"], "selected_ms": x["sel_ms"],
                      "reasoning_correct": {str(c): x["out"][c]["correct"] for c in CAPS},
                      "reasoning_ms_model": {str(c): x["ms"][c] for c in CAPS}} for x in Q],
}
print(json.dumps(out, indent=1))

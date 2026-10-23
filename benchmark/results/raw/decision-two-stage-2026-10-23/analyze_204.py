#!/usr/bin/env python3
"""#204 step 2: evaluate the declared two-stage policies on the 500 #171 test questions.

Thresholds come from docs/DECISION_CAMPAIGN_PLAN.json two_stage_policy_204 (calibration only).
Reference: always-reasoning at cap 2048 (cap-512 output where it finished, the amendment-3 re-run otherwise).
"""
import json
import math
import sys
from pathlib import Path

C = Path("/Users/germar/workspace/geistlib-campaigns/171")
PLAN = Path(sys.argv[1])
sys.path.insert(0, str(C / "src/tools"))
import bench_decision as bench  # noqa: E402
import decision_metrics as metrics  # noqa: E402
import eval_decision as ev  # noqa: E402
import eval_decision_campaign as camp  # noqa: E402

decl = json.loads(PLAN.read_text())["two_stage_policy_204"]
T = json.loads((C / "run-calibration250b/calibration.json").read_text())["temperature"]
rows = [json.loads(line) for line in (C / "cases-test/test.jsonl").read_text().splitlines()]
eos = json.loads((C / "run-test2/manifest.json").read_text())["binding"]["gguf_metadata"]["eos_token_id"]
old, _, _ = camp.collect(C / "run-test2", rows, ev.CORE_PROFILES, 0, 0, "metal")
ids = json.loads((C / "cap2048_question_ids.json").read_text())
selection = json.loads((C / "cap2048_selection.json").read_text())
by_qid = {r["question_id"]: r for r in rows}
new, _, _ = camp.collect(C / "run-cap2048", [by_qid[q] for q in ids], {"chat_reasoning": ("generate_long",)},
                         0, 0, "metal")

Q = []
for i, r in enumerate(rows):
    q = r["question_id"]
    dense = [s for s in old["chat_direct"][i] if s["mode"] == "decision_dense"][0]
    sel = [s for s in old["chat_direct"][i] if s["mode"] == "decision_selected"][0]
    assert sel["logits"] == dense["logits"]
    p = [math.exp(x) for x in metrics.log_probabilities(sel["logits"], T)]
    k = max(range(4), key=lambda j: sel["logits"][j])
    g = old["chat_reasoning"][i][0]
    o = ev.generation_outcome(g, r, True, eos, 512, 248320)
    if o["truncated"]:
        assert q in ids and q not in selection["determinism_finished"]
        g = new["chat_reasoning"][ids.index(q)][0]
        o = ev.generation_outcome(g, r, True, eos, 2048, 248320)
    Q.append({"q": q, "c": p[k], "direct": k == r["target_index"], "reason": o["correct"],
              "capped": o["truncated"], "sel_ms": sel["elapsed_ms"], "rsn_ms": g["elapsed_ms"],
              "ptok_direct": sel["prompt_tokens"], "ptok_rsn": g["prompt_tokens"], "gen": len(g["generated_ids"])})
assert len(Q) == 500
ref = {x["q"]: x["reason"] for x in Q}
ref_pess = {x["q"]: x["reason"] or x["capped"] for x in Q}
assert sum(ref.values()) == 395 and sum(x["capped"] for x in Q) == 87


def pct(xs, f):
    return bench.percentile(xs, f) / 1000


def evaluate(threshold):
    routed = [x["c"] < threshold for x in Q]
    pol = {x["q"]: (x["reason"] if r else x["direct"]) for x, r in zip(Q, routed)}
    pol_pess = {x["q"]: ((x["reason"] or x["capped"]) if r else x["direct"]) for x, r in zip(Q, routed)}
    pm = metrics.paired_quality(pol, ref, 2.)
    pp = metrics.paired_quality(pol_pess, ref_pess, 2.)
    ms = [x["sel_ms"] + (x["rsn_ms"] if r else 0.) for x, r in zip(Q, routed)]
    direct_set = [x for x, r in zip(Q, routed) if not r]
    n_r = sum(routed)
    return {
        "threshold": threshold, "routed": n_r, "routed_fraction": n_r / 500,
        "accuracy": sum(pol.values()) / 500,
        "direct_answered_precision": (sum(x["direct"] for x in direct_set) / len(direct_set)) if direct_set else None,
        "reasoning_accuracy_on_direct_answered": (sum(x["reason"] for x in direct_set) / len(direct_set)) if direct_set else None,
        "capped_at_2048_routed": sum(x["capped"] for x, r in zip(Q, routed) if r),
        "capped_at_2048_answered_directly": sum(x["capped"] for x, r in zip(Q, routed) if not r),
        "paired_as_measured": {k: pm[k] for k in ("gains", "losses", "difference_pp", "interval_pp", "noninferior")},
        "paired_pessimistic": {k: pp[k] for k in ("gains", "losses", "difference_pp", "interval_pp", "noninferior",
                                                  "direct_accuracy", "baseline_accuracy")},
        "noninferior_established": pm["noninferior"] and pp["noninferior"],
        "model_calls": 500 + n_r, "model_calls_per_question": (500 + n_r) / 500,
        "prompt_tokens": sum(x["ptok_direct"] + (x["ptok_rsn"] if r else 0) for x, r in zip(Q, routed)),
        "generated_tokens": sum(x["gen"] for x, r in zip(Q, routed) if r),
        "time_s": {"sum": sum(ms) / 1000, "mean": sum(ms) / 500000, "p50": pct(ms, .5), "p95": pct(ms, .95)},
    }


always_r = [x["rsn_ms"] for x in Q]
always_d = [x["sel_ms"] for x in Q]
ref_cost = {"model_calls": 500, "prompt_tokens": sum(x["ptok_rsn"] for x in Q), "generated_tokens": sum(x["gen"] for x in Q),
            "time_s": {"sum": sum(always_r) / 1000, "mean": sum(always_r) / 500000, "p50": pct(always_r, .5),
                       "p95": pct(always_r, .95)}, "accuracy": 0.79}
dir_cost = {"model_calls": 500, "prompt_tokens": sum(x["ptok_direct"] for x in Q), "generated_tokens": 0,
            "time_s": {"sum": sum(always_d) / 1000, "mean": sum(always_d) / 500000, "p50": pct(always_d, .5),
                       "p95": pct(always_d, .95)}, "accuracy": sum(x["direct"] for x in Q) / 500}


def ratios(e):
    return {"time_vs_always_reasoning": e["time_s"]["sum"] / ref_cost["time_s"]["sum"],
            "tokens_vs_always_reasoning": (e["prompt_tokens"] + e["generated_tokens"])
            / (ref_cost["prompt_tokens"] + ref_cost["generated_tokens"]),
            "generated_tokens_vs_always_reasoning": e["generated_tokens"] / ref_cost["generated_tokens"]}


out = {"note_direction": "difference = policy - always-reasoning(cap 2048); negative means reasoning is better",
       "temperature": T, "always_reasoning_cap2048": ref_cost, "always_direct": dir_cost}
prim = evaluate(decl["primary"]["threshold"])
prim["cost_ratios"] = ratios(prim)
out["primary"] = prim
out["secondary"] = {}
for name in ("route_30", "route_50", "route_70"):
    e = evaluate(decl["secondary_fixed_coverage"][name]["threshold"])
    e["cost_ratios"] = ratios(e)
    out["secondary"][name] = e
# Exploratory, descriptive only: route the least-confident fraction of the TEST questions.
cs = sorted(x["c"] for x in Q)
curve = []
for f in range(0, 101, 5):
    t = cs[round(f / 100 * 500)] if f < 100 else 1.01
    e = evaluate(t)
    e["cost_ratios"] = ratios(e)
    curve.append({"routed_fraction_target": f / 100, **{k: e[k] for k in (
        "threshold", "routed", "accuracy", "direct_answered_precision", "reasoning_accuracy_on_direct_answered",
        "noninferior_established", "model_calls_per_question")},
        "difference_pp": e["paired_as_measured"]["difference_pp"],
        "interval_pp": e["paired_as_measured"]["interval_pp"],
        "interval_pessimistic_pp": e["paired_pessimistic"]["interval_pp"],
        "time_mean_s": e["time_s"]["mean"], "time_p95_s": e["time_s"]["p95"],
        "time_vs_always_reasoning": e["cost_ratios"]["time_vs_always_reasoning"]})
out["exploratory_curve_test_set"] = curve
out["per_question"] = [{"question_id": x["q"], "confidence_calibrated": x["c"], "direct_correct": x["direct"],
                        "reasoning_cap2048_correct": x["reason"], "reasoning_capped_at_2048": x["capped"],
                        "selected_ms": x["sel_ms"], "reasoning_ms": x["rsn_ms"], "prompt_tokens_direct": x["ptok_direct"],
                        "prompt_tokens_reasoning": x["ptok_rsn"], "generated_tokens": x["gen"],
                        "primary_routed": x["c"] < decl["primary"]["threshold"]} for x in Q]
print(json.dumps(out, indent=1))

#!/usr/bin/env python3
"""#204 step 1: thresholds of the declared two-stage policies, from the calibration split's direct arm only.

Reads no test-set data. Output: the frozen thresholds for docs/DECISION_CAMPAIGN_PLAN.json.
"""
import json
import math
import sys
from pathlib import Path

C = Path("/Users/germar/workspace/geistlib-campaigns/171")
sys.path.insert(0, str(C / "src/tools"))
import decision_metrics as metrics  # noqa: E402
import eval_decision_campaign as camp  # noqa: E402

TARGET_PRECISION, MIN_ACCEPTED = 0.95, 25
ROUTED_FRACTIONS = (0.30, 0.50, 0.70)

cal = json.loads((C / "run-calibration250b/calibration.json").read_text())
T = cal["temperature"]
rows = [json.loads(line) for line in (C / "cases-calibration250/calibration.jsonl").read_text().splitlines()]
done, _, _ = camp.collect(C / "run-calibration250b", rows, {"chat_direct": ("decision_dense",)}, 0, 0, "metal")
assert len(done["chat_direct"]) == len(rows) == 250
assert sorted(r["question_id"] for r in rows) == cal["question_ids"]
pts = []
for i, r in enumerate(rows):
    s = done["chat_direct"][i][0]
    p = [math.exp(x) for x in metrics.log_probabilities(s["logits"], T)]
    k = max(range(4), key=lambda j: s["logits"][j])
    pts.append((p[k], k == r["target_index"]))
n = len(pts)
cs = sorted(c for c, _ in pts)


def accepted(t):
    a = [h for c, h in pts if c >= t]
    return len(a), (sum(a) / len(a) if a else None)


# Primary: smallest t such that every threshold t' >= t whose accepted set has
# at least MIN_ACCEPTED questions keeps calibration precision >= TARGET_PRECISION.
tau = None
for t in sorted(set(cs), reverse=True):
    m, prec = accepted(t)
    if m < MIN_ACCEPTED:
        continue
    if prec < TARGET_PRECISION:
        break
    tau = t
out = {"temperature": T, "n_calibration": n, "calibration_accuracy": sum(h for _, h in pts) / n}
m, prec = accepted(tau) if tau is not None else (0, None)
out["primary"] = {"threshold": tau, "calibration_accepted": m, "calibration_coverage_direct": m / n,
                  "calibration_precision": prec}
out["coverage_variants"] = {}
for q in ROUTED_FRACTIONS:
    t = cs[round(q * n)]
    m, prec = accepted(t)
    out["coverage_variants"][f"route_{round(q * 100)}"] = {
        "threshold": t, "calibration_accepted": m, "calibration_routed_fraction": 1 - m / n,
        "calibration_precision": prec}
print(json.dumps(out, indent=1))

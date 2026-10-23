#!/usr/bin/env python3
"""#171: compact numbers from the test and calibration reports (for the writeup and the issue)."""
import json
import sys
from pathlib import Path

C = Path("/Users/germar/workspace/geistlib-campaigns/171")
sys.path.insert(0, str(C / "src/tools"))
import decision_metrics as metrics  # noqa: E402

test = json.loads((C / sys.argv[1] / "report.json").read_text()) if len(sys.argv) > 1 else None
r = test
out = {}
for profile, v in r["profiles"].items():
    for mode, x in v["modes"].items():
        q, lat = x["quality"], x["latency"]
        e = {"n": q["n"], "accuracy": q["accuracy"],
             "p50_s": lat["warm_sample_p50_ms"] / 1000, "p95_s": lat["warm_sample_p95_ms"] / 1000,
             "mean_s": lat["mean_question_median_ms"] / 1000}
        for k in ("log_loss", "brier_sum_over_classes", "ece_equal_width"):
            if k in q:
                e[k] = q[k]
        for k in ("invalid_answers", "capped_outputs", "generated_tokens"):
            if k in q:
                e[k] = q[k]
        if "calibrated_quality" in x:
            cq = x["calibrated_quality"]
            cq = cq.get("quality", cq)
            e["calibrated"] = {k: cq.get(k) for k in ("temperature", "log_loss", "brier_sum_over_classes", "ece_equal_width", "accuracy")}
        out[f"{profile}/{mode}"] = e
cmp = r["comparison"]
pq = cmp["paired_quality"]
out["paired"] = {k: pq[k] for k in ("n", "gains", "losses", "direct_accuracy", "baseline_accuracy", "difference_pp", "interval_pp", "noninferior")}
out["paired"]["ratio_reasoning_over_selected_inference"] = pq.get("inference_p50_ms_ratio_reasoning_over_selected")
out["eligible"], out["blockers"] = cmp["quality_matched_speedup_eligible"], cmp["blockers"]
out["campaign"] = r.get("campaign")
out["spotcheck"] = {k: r["determinism_spotcheck"][k] for k in ("by_arm", "deterministic")} if "determinism_spotcheck" in r else None
out["thermal"] = {k: {kk: v[kk] for kk in ("start_gpu_c_p50", "start_gpu_c_max", "n_cool_start", "cool_start_p50_ms")}
                  for k, v in r["latency_by_start_temperature"].items()}

# N for which the conservative interval clears -2 pp at the observed rates.
n, g, l = pq["n"], pq["gains"], pq["losses"]
need = None
for N in range(n, 20001, 50):
    G, L = round(g / n * N), round(l / n * N)
    d = {f"q{i}": i < G for i in range(N)}
    b = {f"q{i}": (G <= i < G + L) for i in range(N)}
    if metrics.paired_quality(d, b, 2.)["noninferior"]:
        need = N
        break
out["n_needed_for_2pp_at_observed_rates"] = need
print(json.dumps(out, indent=1))

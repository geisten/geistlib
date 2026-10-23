#!/usr/bin/env python3
"""#171 amendment 3: combine the cap-512 run (finished answers) with the cap-2048 re-run (capped answers)."""
import json
import sys
from pathlib import Path

C = Path("/Users/germar/workspace/geistlib-campaigns/171")
sys.path.insert(0, str(C / "src/tools"))
import bench_decision as bench  # noqa: E402
import decision_metrics as metrics  # noqa: E402
import eval_decision as ev  # noqa: E402
import eval_decision_campaign as camp  # noqa: E402

rows = [json.loads(line) for line in (C / "cases-test/test.jsonl").read_text().splitlines()]
by_qid = {r["question_id"]: (i, r) for i, r in enumerate(rows)}
eos = json.loads((C / "run-test2/manifest.json").read_text())["binding"]["gguf_metadata"]["eos_token_id"]
old, _, _ = camp.collect(C / "run-test2", rows, ev.CORE_PROFILES, 0, 0, "metal")
ids = json.loads((C / "cap2048_question_ids.json").read_text())
sub = [by_qid[q][1] for q in ids]
new, _, _ = camp.collect(C / "run-cap2048", sub, {"chat_reasoning": ("generate_long",)}, 0, 0, "metal")
selection = json.loads((C / "cap2048_selection.json").read_text())

direct, reasoning512, reasoning2048 = {}, {}, {}
times512, times2048, rerun_times, rerun = [], [], [], []
determinism = []
still_capped = 0
for i, r in enumerate(rows):
    q = r["question_id"]
    d = [s for s in old["chat_direct"][i] if s["mode"] == "decision_dense"][0]
    direct[q] = d["best_index"] == r["target_index"]
    g = old["chat_reasoning"][i][0]
    o512 = ev.generation_outcome(g, r, True, eos, 512, 248320)
    reasoning512[q] = o512["correct"]
    times512.append(g["elapsed_ms"])
    if q in ids:
        j = ids.index(q)
        if j not in new["chat_reasoning"]:
            raise SystemExit(f"cap2048 run incomplete: {q}")
        h = new["chat_reasoning"][j][0]
        o = ev.generation_outcome(h, r, True, eos, 2048, 248320)
        if q in selection["determinism_finished"]:
            determinism.append({"question_id": q, "identical_ids": h["generated_ids"] == g["generated_ids"],
                                "identical_bytes": h["generated_utf8_hex"] == g["generated_utf8_hex"],
                                "tokens": len(h["generated_ids"])})
            reasoning2048[q] = o512["correct"]
            times2048.append(g["elapsed_ms"])
            continue
        reasoning2048[q] = o["correct"]
        still_capped += o["truncated"]
        rerun.append({"question_id": q, "correct": o["correct"], "valid": o["valid"], "truncated": o["truncated"],
                      "tokens": len(h["generated_ids"]), "ms": h["elapsed_ms"], "direct_correct": direct[q]})
        rerun_times.append(h["elapsed_ms"])
        times2048.append(h["elapsed_ms"])
    else:
        if o512["truncated"]:
            raise SystemExit(f"capped question missing from the re-run list: {q}")
        reasoning2048[q] = o512["correct"]
        times2048.append(g["elapsed_ms"])

pct = lambda xs, f: bench.percentile(xs, f) / 1000
paired512 = metrics.paired_quality(direct, reasoning512, 2.)
paired2048 = metrics.paired_quality(direct, reasoning2048, 2.)
n_rerun = len(rerun)
out = {
    "determinism_check_finished_at_cap2048": determinism,
    "rerun_questions": n_rerun,
    "rerun_reasoning_accuracy": sum(x["correct"] for x in rerun) / n_rerun,
    "rerun_direct_accuracy": sum(x["direct_correct"] for x in rerun) / n_rerun,
    "rerun_still_capped_at_2048": still_capped,
    "rerun_invalid": sum(not x["valid"] for x in rerun),
    "rerun_tokens": {"min": min(x["tokens"] for x in rerun), "mean": sum(x["tokens"] for x in rerun) / n_rerun,
                      "max": max(x["tokens"] for x in rerun)},
    "reasoning_accuracy_cap512": sum(reasoning512.values()) / 500,
    "reasoning_accuracy_cap2048": sum(reasoning2048.values()) / 500,
    "direct_accuracy": sum(direct.values()) / 500,
    "paired_cap512": {k: paired512[k] for k in ("gains", "losses", "difference_pp", "interval_pp", "noninferior")},
    "paired_cap2048": {k: paired2048[k] for k in ("gains", "losses", "difference_pp", "interval_pp", "noninferior")},
    "latency_reasoning_cap2048_all500_s": {"p50": pct(times2048, .5), "p95": pct(times2048, .95),
                                           "mean": sum(times2048) / 500000},
    "latency_reasoning_cap2048_rerun_s": {"p50": pct(rerun_times, .5), "p95": pct(rerun_times, .95)},
    "latency_reasoning_cap512_all500_s": {"p50": pct(times512, .5), "p95": pct(times512, .95)},
    "selected_dense_sum_s": None,
}
sel = [[s for s in old["chat_direct"][i] if s["mode"] == "decision_selected"][0]["elapsed_ms"] for i in range(500)]
out["selected_dense_sum_s"] = sum(sel) / 1000
out["time_ratio_reasoning2048_over_selected"] = sum(times2048) / sum(sel)
out["time_ratio_reasoning512_over_selected"] = sum(times512) / sum(sel)
# Questions for which the conservative interval would clear -2 pp, at the observed rates.
g, l = paired2048["gains"], paired2048["losses"]
out["note_direction"] = "difference = direct - reasoning; negative means reasoning is better"
print(json.dumps(out, indent=1))

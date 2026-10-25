#!/usr/bin/env python3
"""#26 batched reasoning re-run vs #171 always-reasoning at cap 2048 (declared:
docs/DECISION_CAMPAIGN_PLAN.json batched_reasoning_26).

Usage: analyze_batched.py RUN_DIR CAP_CURVE_JSON [ATTRIBUTION_DIR] > result.json

- primary: decision_metrics.paired_quality(batched, #171 cap 2048), 2 pp margin;
- token agreement: first position where the batched generation and #171's
  cap-2048 generation differ (#171: the cap-512 run where it finished, the
  cap-2048 re-run otherwise);
- attribution (when ATTRIBUTION_DIR holds the unbatched same-binary re-runs of
  the questions whose correctness changed): for each, whether the unbatched
  output equals #171's (the change comes from batching) or the batched one
  (from the engine since #171), or neither.
"""
import json
import sys
from pathlib import Path

C = Path("/Users/germar/workspace/geistlib-campaigns/171")
sys.path.insert(0, str(C / "src/tools"))
import decision_metrics as metrics  # noqa: E402
import eval_decision as ev  # noqa: E402
import eval_decision_campaign as camp  # noqa: E402

run_dir, curve = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
attr_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else None
rows = [json.loads(line) for line in (C / "cases-test/test.jsonl").read_text().splitlines()]
by_qid = {r["question_id"]: r for r in rows}

# #171 cap-2048 generations
old, _, _ = camp.collect(C / "run-test2", rows, ev.CORE_PROFILES, 0, 0, "metal")
ids = json.loads((C / "cap2048_question_ids.json").read_text())
new, _, _ = camp.collect(C / "run-cap2048", [by_qid[q] for q in ids], {"chat_reasoning": ("generate_long",)},
                         0, 0, "metal")
ref_ids = {rows[i]["question_id"]: s[0]["generated_ids"] for i, s in old["chat_reasoning"].items()}
for j, s in new["chat_reasoning"].items():
    ref_ids[ids[j]] = s[0]["generated_ids"]
ref_correct = {q["question_id"]: q["reasoning_correct"]["2048"] for q in curve["per_question"]}


def load_samples(d):
    out = {}
    for chunk in camp.chunk_dirs(d):
        if not (chunk / "batched.rows.json").exists():
            continue
        idx = json.loads((chunk / "batched.rows.json").read_text())
        rws = json.loads((d / "manifest.json").read_text())["order"]
        for r in camp.read_lines(chunk / "batched.driver.jsonl"):
            if r.get("kind") == "sample":
                out[idx[r["case_index"]]] = r
    return out


report = json.loads((run_dir / "report.json").read_text())
pq = report["summary"]["per_question"]
bat_samples = load_samples(run_dir)
manifest = json.loads((run_dir / "manifest.json").read_text())
row_by_id = {r["id"]: r for r in rows}
# runner indices are positions in the cases file
batched_ids = {rows[i]["question_id"]: s["generated_ids"] for i, s in bat_samples.items()}

direct = {q: v["correct"] for q, v in pq.items()}
baseline = {q: ref_correct[q] for q in direct}
paired = metrics.paired_quality(direct, baseline, 2)


def first_diff(a, b):
    for k, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return k
    return None if len(a) == len(b) else min(len(a), len(b))


agree = {q: first_diff(batched_ids[q], ref_ids[q]) for q in direct}
identical = sum(v is None for v in agree.values())
diverged = sorted(v for v in agree.values() if v is not None)
flips = sorted(q for q in direct if direct[q] != baseline[q])
out = {
    "paired_vs_171_cap2048": paired,
    "batched_accuracy": report["summary"]["accuracy"], "batched_capped": report["summary"]["capped"],
    "reference_accuracy": sum(baseline.values()) / len(baseline),
    "token_agreement": {"identical_generations": identical, "diverged": len(diverged),
                        "first_divergence_p50": diverged[len(diverged) // 2] if diverged else None,
                        "first_divergence_min": diverged[0] if diverged else None},
    "flips": [{"question_id": q, "id": by_qid[q]["id"], "batched_correct": direct[q], "ref_correct": baseline[q],
               "first_divergence": agree[q]} for q in flips],
}
if attr_dir is not None:
    unb = load_samples(attr_dir)
    unb_ids = {rows[i]["question_id"]: s["generated_ids"] for i, s in unb.items()}
    res = []
    for f in out["flips"]:
        q = f["question_id"]
        if q not in unb_ids:
            continue
        u = unb_ids[q]
        cause = ("batching" if u == ref_ids[q] else "engine_since_171" if u == batched_ids[q] else "both")
        res.append({**f, "unbatched_equals_171": u == ref_ids[q], "unbatched_equals_batched": u == batched_ids[q],
                    "cause": cause})
    out["attribution"] = res
print(json.dumps(out, indent=1))

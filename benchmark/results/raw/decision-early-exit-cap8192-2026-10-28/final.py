import json,sys,glob
sys.path.insert(0,"/Users/germar/workspace/geistlib-campaigns/236c-src/tools")
import decision_metrics as m
D="/Users/germar/workspace/geistlib-campaigns/236c"; C="/Users/germar/workspace/geistlib-campaigns"
r=json.load(open(D+"/test-policy/report.json")); s=r["summary"]; pq=s["per_question"]
base=json.load(open(D+"/test_baseline_correct.json"))
c171={q["question_id"]:q["reasoning_correct"]["2048"] for q in json.load(open("/Users/germar/workspace/geistlib/benchmark/results/DECISION_CAP_CURVE_2026-10-23.json"))["per_question"]}
pol={k:v["correct"] for k,v in pq.items()}
def pq_(b): x=m.paired_quality(pol,{k:b[k] for k in pol},2); return {k:x[k] for k in ("gains","losses","difference_pp","interval_pp","noninferior","baseline_accuracy","direct_accuracy")}
lease=lambda d: sum(json.load(open(f)).get("ended_unix",0)-json.load(open(f))["started_unix"] for f in glob.glob(d+"/chunks/*/chunk.json") if json.load(open(f)).get("ended_unix"))
out={"policy":{k:s[k] for k in ("accuracy","correct","early_exits","capped","generated_tokens","probes","probe_ms_total","latency_ms","complete_driver_runs")},
 "vs_full_reasoning_cap8192":pq_(base),"vs_171_cap2048":pq_(c171),
 "baseline_tokens_cap8192":676706,"token_ratio":s["generated_tokens"]/676706,
 "lease_s":{"policy":lease(D+"/test-policy"),"baseline_264":lease(C+"/26/run-batched"),"baseline_rerun_cap8192":lease(D+"/test-baseline")}}
print(json.dumps(out,indent=1))

"""#236 cap-8192: merge kept outputs with the re-runs; spot-checks; freeze rule; reports."""
import json,math,sys,glob
sys.path.insert(0,"/Users/germar/workspace/geistlib-campaigns/236c-src/tools")
import decision_metrics as m
C="/Users/germar/workspace/geistlib-campaigns"; D=C+"/236c"
amend=json.load(open(C+"/236c-src/docs/DECISION_CAMPAIGN_PLAN.json"))["early_exit_batched_236"]["amendments"][-1]["rerun_sets"]

def gen_ids(run_dir, cases):
    rows=[json.loads(l) for l in open(cases)]
    out={}
    for ch in sorted(glob.glob(run_dir+"/chunks/*")):
        try: idx=json.load(open(ch+"/batched.rows.json"))
        except FileNotFoundError: continue
        for l in open(ch+"/batched.driver.jsonl"):
            r=json.loads(l)
            if r["kind"]=="sample": out[rows[idx[r["case_index"]]]["question_id"]]=r["generated_ids"]
    return out

def merged(kept_report, kept_dir, rerun_dir, cases, name):
    kept=json.load(open(kept_report))["summary"]["per_question"]
    rer=json.load(open(rerun_dir+"/report.json"))["summary"]["per_question"]
    ki=gen_ids(kept_dir,cases); ri=gen_ids(rerun_dir,cases)
    spot={q: ki[q]==ri[q] for q in amend[name]["spotcheck"]}
    base={}
    for q,v in kept.items():
        base[q]= rer[q] if q in rer and q not in amend[name]["spotcheck"] else v
    return base, spot, rer

def sm(l):
    mx=max(l); e=[math.exp(x-mx) for x in l]; t=sum(e); return [x/t for x in e]

def policy(pq, rows, tau):
    pol={}; tokens=0; exits=0
    for k,q in pq.items():
        ans, used = q["predicted_index"], q["generated_tokens"]
        for p in q["probes"]:
            if p["pos"]>q["generated_tokens"]: break
            pr=sm(p["logits"]); top=max(range(len(pr)),key=pr.__getitem__)
            if p["pos"]>=128 and pr[top]>=tau:
                ans, used = top, p["pos"]; exits+=1; break
        pol[k]= ans==rows[k]["target_index"]; tokens+=used
    return pol,tokens,exits

if __name__=="__main__":
    what=sys.argv[1]
    if what=="freeze":
        rows={json.loads(l)["question_id"]:json.loads(l) for l in open(C+"/171/cases-calibration250/calibration.jsonl")}
        base,spot,rer=merged(C+"/236b/confirm/report.json", C+"/236b/confirm", D+"/cal-rerun", C+"/171/cases-calibration250/calibration.jsonl","calibration")
        full={k:v["correct"] for k,v in base.items()}; ft=sum(v["generated_tokens"] for v in base.values())
        grid=[]
        for tau in (0.7,0.8,0.9,0.95,0.99):
            pol,tok,ex=policy(base,rows,tau); pq=m.paired_quality(pol,full,2)
            grid.append({"tau":tau,"accuracy":sum(pol.values())/len(pol),"difference_pp":pq["difference_pp"],"interval_pp":pq["interval_pp"],"gains":pq["gains"],"losses":pq["losses"],"token_ratio":tok/ft,"early_exits":ex,"qualifies":pq["difference_pp"]>=-1.0 and tok/ft<=0.75})
        q=[g for g in grid if g["qualifies"]]
        print(json.dumps({"questions":len(base),"spotcheck_identical":sum(spot.values()),"spotcheck":len(spot),
          "full_accuracy":sum(full.values())/len(full),"capped_8192":sum(v["stop"]=="limit" for v in base.values()),
          "rerun_capped_8192":sum(v["stop"]=="limit" for v in rer.values()),"full_tokens":ft,"grid":grid,"tau_star":q[0]["tau"] if q else None},indent=1))
    elif what=="baseline":
        base,spot,rer=merged(C+"/26/run-batched/report.json", C+"/26/run-batched", D+"/test-baseline", C+"/171/cases-test/test.jsonl","test")
        print(json.dumps({"questions":len(base),"spotcheck_identical":sum(spot.values()),"spotcheck":len(spot),
          "accuracy":sum(v["correct"] for v in base.values())/len(base),"capped_8192":sum(v["stop"]=="limit" for v in base.values()),
          "tokens":sum(v["generated_tokens"] for v in base.values())},indent=1))
        json.dump({k:v["correct"] for k,v in base.items()},open(D+"/test_baseline_correct.json","w"))

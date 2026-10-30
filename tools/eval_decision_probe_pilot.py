#!/usr/bin/env python3
"""Early-exit pilot (#236): reasoning with answer probes, chunked and resumable.

Draws a label-independent, subject-stratified sample of a prepared bundle
(decision_dataset.select_stratified, --sample and --seed), and runs every
question's chat_reasoning prompt through tools/bench_reasoning_probe:
- greedy generation up to --cap tokens;
- every --interval generated tokens (and right after the prompt), the
  session state is saved, the thinking-close probe tokens are prefilled, the
  option logits are read, and the state is restored.
--transparency N questions also run without probes; their generations must be
identical.

Every invocation runs at most --chunk-questions pending questions under a
wall-clock deadline (one lease of the host lock), with a cool start and a
frozen manifest. report.json scores every generation as #171 did
(eval_decision.generation_outcome, thinking on), keeps every probe and runs
the offline early-exit analysis (analyze()).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import eval_decision_campaign as campaign

ev = campaign.ev
bench = ev.bench
data = ev.data
VOLATILE = ("started_utc",)
PROFILE = "chat_reasoning"
MODES = ("probed", "plain")


def softmax(logits):
    m = max(logits)
    e = [math.exp(x - m) for x in logits]
    t = sum(e)
    return [x / t for x in e]


def wire(rows, indices, probe):
    out = []
    for i in indices:
        p = rows[i]["profiles"][PROFILE]
        out.append(" ".join(map(str, [len(p["prompt_ids"]), len(p["candidate_ids"]), len(probe),
                                       *p["prompt_ids"], *p["candidate_ids"], *probe])))
    return "".join(line + "\n" for line in out)


def collect(out_dir, backend):
    """mode -> row index -> sample."""
    done = {m: {} for m in MODES}
    runtimes = []
    for chunk in campaign.chunk_dirs(out_dir):
        for mode in MODES:
            if not (chunk / f"{mode}.rows.json").exists():
                continue
            indices = ev.read_json(chunk / f"{mode}.rows.json")
            for r in campaign.read_lines(chunk / f"{mode}.driver.jsonl"):
                if r.get("kind") == "runtime":
                    if r["backend"] != backend or r["kv_mode"] != "fp32":
                        raise ValueError("invalid runtime record")
                    runtimes.append(r)
                elif r.get("kind") == "sample":
                    i = indices[r["case_index"]]
                    if i in done[mode]:
                        raise ValueError("a question was completed twice")
                    done[mode][i] = {**r, "case_index": i, "chunk": chunk.name}
                else:
                    raise ValueError("unexpected driver record")
    return done, runtimes


def analyze(questions, interval, cap, decode_ms_per_token, probe_ms):
    """Offline early-exit policies over the recorded probes (exploratory).

    Policy (tau, stable): at each probe from position `interval` on, take the
    option distribution softmax(logits); stop at the first probe whose top-1
    probability is >= tau (and, with stable, whose argmax equals the previous
    probe's), answer with its argmax and count the tokens generated so far.
    Without such a probe the full reasoning answer stands. Probes cost
    probe_ms each, counted for every probe the policy took.
    """
    full_correct = sum(q["full_correct"] for q in questions)
    full_tokens = sum(q["generated_tokens"] for q in questions)
    full_ms = sum(q["generated_tokens"] * decode_ms_per_token for q in questions)
    rows = []
    for tau in (0.7, 0.8, 0.9, 0.95, 0.99):
        for stable in (False, True):
            correct = tokens = probes = exits = agree = 0
            for q in questions:
                answer, used, n_probes = q["full_answer"], q["generated_tokens"], 0
                prev = None
                for p in q["probes"]:
                    if p["pos"] > q["generated_tokens"]:
                        break
                    n_probes += 1
                    probs = softmax(p["logits"])
                    top = max(range(len(probs)), key=probs.__getitem__)
                    if p["pos"] >= interval and probs[top] >= tau and (not stable or top == prev):
                        answer, used = top, p["pos"]
                        exits += 1
                        agree += top == q["full_answer"]
                        break
                    prev = top
                correct += answer == q["target"]
                tokens += used
                probes += n_probes
            ms = tokens * decode_ms_per_token + probes * probe_ms
            rows.append({"tau": tau, "stable": stable, "accuracy": correct / len(questions),
                         "difference_pp": 100 * (correct - full_correct) / len(questions),
                         "generated_tokens": tokens, "token_ratio": tokens / full_tokens,
                         "time_ratio_incl_probes": ms / full_ms, "early_exits": exits,
                         "early_answer_agrees_with_full": agree})
    return {"questions": len(questions), "full_accuracy": full_correct / len(questions),
            "full_generated_tokens": full_tokens, "decode_ms_per_token": decode_ms_per_token,
            "probe_ms_median": probe_ms, "policies": rows,
            "gate": "#236: >= 25 % fewer generated tokens at <= 1 pp accuracy loss; exploratory (the grid "
                    "is evaluated on the same sample)"}


def progress(args, rows, sample, plain, manifest):
    done, runtimes = collect(args.out_dir, args.backend)
    pending = [i for i in sample if i not in done["probed"]]
    pending_plain = [i for i in plain if i not in done["plain"]]
    status = {"questions": len(sample), "complete": len(done["probed"]), "pending": len(pending),
              "transparency_pending": len(pending_plain), "chunks": len(campaign.chunk_dirs(args.out_dir))}
    ev.write_json(args.out_dir / "status.json", status)
    if not pending and not pending_plain and not (args.out_dir / "report.json").exists():
        eos = runtimes[0]["eos"]
        questions, decode_ms, tokens, probe_ms = [], 0., 0, []
        for i in sample:
            s = done["probed"][i]
            o = ev.generation_outcome(s, {"target_index": rows[i]["target_index"]}, True, eos, args.cap,
                                      args.vocab)
            questions.append({"question_id": rows[i]["question_id"], "id": rows[i]["id"],
                              "subject": rows[i]["subject"], "target": rows[i]["target_index"],
                              "full_answer": o["predicted_index"], "full_correct": o["correct"],
                              "stop": o["stop"], "generated_tokens": o["generated_tokens"],
                              "probes": [{"pos": p["pos"], "logits": p["logits"]} for p in s["probes"]]})
            decode_ms += s["decode_ms"]
            tokens += o["generated_tokens"]
            probe_ms += [p["ms"] for p in s["probes"]]
        transparency = {rows[i]["id"]: done["plain"][i]["generated_ids"] == done["probed"][i]["generated_ids"]
                        for i in plain}
        per_token = decode_ms / tokens
        report = {"manifest": manifest,
                  "transparency": {"questions": len(plain), "identical": sum(transparency.values()),
                                   "by_question": transparency},
                  "probe_cost": {"n": len(probe_ms), "median_ms": bench.percentile(probe_ms, .5),
                                 "p95_ms": bench.percentile(probe_ms, .95),
                                 "decode_ms_per_token": per_token,
                                 "median_in_decode_steps": bench.percentile(probe_ms, .5) / per_token},
                  "full_reasoning": {"accuracy": sum(q["full_correct"] for q in questions) / len(questions),
                                     "capped": sum(q["stop"] == "limit" for q in questions),
                                     "generated_tokens": tokens},
                  "analysis": analyze(questions, args.interval, args.cap, per_token,
                                      bench.percentile(probe_ms, .5)),
                  "questions": questions,
                  "chunks": [ev.read_json(c / "chunk.json") for c in campaign.chunk_dirs(args.out_dir)]}
        ev.write_json(args.out_dir / "report.json", report)
        print(args.out_dir / "report.json")
    print(json.dumps(status))
    return pending, pending_plain, status


def run(args):
    deadline = time.time() + args.deadline_minutes * 60  # wall clock (see campaign.run_driver)
    rows = [json.loads(line) for line in (args.cases_dir / f"{args.split}.jsonl").read_text().split("\n")
            if line.strip()]
    chosen, _ = data.select_stratified(rows, args.sample, seed=args.seed)
    keep = {r["question_id"] for r in chosen}
    sample = [i for i, r in enumerate(rows) if r["question_id"] in keep]
    plain = sample[:args.transparency]
    probe = [int(x) for x in args.probe.split(",")]
    prompt_cap = max(len(rows[i]["profiles"][PROFILE]["prompt_ids"]) for i in sample)
    commands = {mode: [str(args.binary.resolve()), str(args.model.resolve()), args.backend, str(prompt_cap),
                       str(args.cap), str(args.interval if mode == "probed" else 0)] for mode in MODES}
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads))
    env.setdefault("OMP_WAIT_POLICY", "passive")
    manifest = {
        "protocol": "geist-decision-probe-pilot-v1", "tracking_issue": "geistlib#236",
        "plan": "docs/DECISION_CAMPAIGN_PLAN.json early_exit_pilot_236",
        "model_sha256": bench.sha256(args.model), "binary_sha256": bench.sha256(args.binary),
        "cases_sha256": bench.sha256(args.cases_dir / f"{args.split}.jsonl"), "split": args.split,
        "sample": args.sample, "seed": args.seed, "question_ids": [rows[i]["question_id"] for i in sample],
        "transparency_ids": [rows[i]["question_id"] for i in plain], "probe": probe, "interval": args.interval,
        "cap": args.cap, "backend": args.backend, "threads": args.threads, "commands": commands,
        "environment": {k: v for k, v in env.items() if k.startswith(("GEIST_", "OMP_", "KMP_"))},
        "source_revision": bench.command_output(["git", "rev-parse", "HEAD"]),
    }
    frozen = args.out_dir / "manifest.json"
    if frozen.exists():
        old = ev.read_json(frozen)
        fresh = json.loads(json.dumps(manifest))
        diff = sorted(k for k in set(old) | set(fresh) if k not in VOLATILE and old.get(k) != fresh.get(k))
        if diff:
            raise ValueError(f"frozen manifest differs on resume: {diff}")
    else:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        if any(args.out_dir.iterdir()):
            raise FileExistsError(f"{args.out_dir} has content but no manifest")
        ev.write_json(frozen, manifest)
    pending, pending_plain, status = progress(args, rows, sample, plain, manifest)
    if args.status or not (pending or pending_plain):
        return status
    mode, todo = ("probed", pending) if pending else ("plain", pending_plain)
    todo = todo[:args.chunk_questions]
    chunk = args.out_dir / "chunks" / f"{len(campaign.chunk_dirs(args.out_dir)):04d}"
    chunk.mkdir(parents=True)
    info = {"started_unix": time.time(), "mode": mode, "rows": todo, "lock_note": os.environ.get("LOCK_NOTE"),
            "deadline_minutes": args.deadline_minutes}
    ev.write_json(chunk / "chunk.json", info)
    temps = campaign.TempLog(chunk / "temps.jsonl")
    try:
        info["cool_gate"] = campaign.cool_gate(args, deadline)
        ev.write_json(chunk / f"{mode}.rows.json", todo)
        info["run"] = campaign.run_driver(commands[mode], [], env, chunk, mode, deadline,
                                          wire=wire(rows, todo, probe))
    finally:
        temps.close()
        info["ended_unix"] = time.time()
        ev.write_json(chunk / "chunk.json", info)
    return progress(args, rows, sample, plain, manifest)[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("--cases-dir", "--model", "--binary", "--out-dir"):
        p.add_argument(name, type=Path, required=True)
    p.add_argument("--split", default="calibration")
    p.add_argument("--sample", type=int, required=True)
    p.add_argument("--seed", required=True)
    p.add_argument("--transparency", type=int, default=2)
    p.add_argument("--probe", required=True, help="comma-separated probe token IDs")
    p.add_argument("--interval", type=int, required=True)
    p.add_argument("--cap", type=int, required=True)
    p.add_argument("--vocab", type=int, required=True)
    p.add_argument("--backend", choices=("cpu_neon", "metal"), required=True)
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--chunk-questions", type=int, required=True)
    p.add_argument("--deadline-minutes", type=float, required=True)
    p.add_argument("--cool-max-c", type=float, default=65.)
    p.add_argument("--cool-timeout-minutes", type=float, default=5.)
    p.add_argument("--status", action="store_true")
    args = p.parse_args()
    if (args.sample < 1 or args.interval < 1 or not 1 <= args.cap <= 32768 or args.chunk_questions < 1
            or not 0 <= args.transparency <= args.sample or not args.deadline_minutes > 0):
        p.error("invalid sample/interval/cap/chunk/transparency/deadline")
    run(args)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Batched reasoning over a prepared decision bundle, chunked and resumable (#26).

Runs every question's chat_reasoning prompt through tools/bench_reasoning_batch:
greedy generation up to --cap tokens, with up to --batch sessions decoding
together through geist_session_decode_batch, refilled as questions finish.
Questions run in subject order. Each subject's common prompt prefix (minus one
token, aligned down to --align, the engine's prefill chunk) is prefilled once
and restored per question (the #26 step-1/2 route, bit-identical to a single
prefill at an aligned cut).

Every invocation runs at most --chunk-questions pending questions and stops the
driver at --deadline-minutes (wall clock), so it fits one lease of the host
lock; questions in flight at the deadline are re-run, finished ones never. The
manifest is frozen by the first invocation and checked on every resume. Each
driver run starts on a cool chip; macmon is logged during the run.

The report scores every generation exactly as the #171 campaign did
(eval_decision.generation_outcome, thinking on) and gives accuracy, capped
answers, generated tokens and throughput. The paired comparison against an
earlier unbatched run is a separate analysis.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import eval_decision_campaign as campaign
import eval_decision_prefix as prefix

ev = campaign.ev
bench = ev.bench
VOLATILE = ("started_utc",)
PROFILE = "chat_reasoning"


def wire(rows, indices, cuts):
    lines = []
    for i in indices:
        p = rows[i]["profiles"][PROFILE]
        lines.append(" ".join(map(str, [cuts[rows[i]["subject"]]["cut"], len(p["prompt_ids"]), *p["prompt_ids"]])))
    return "".join(line + "\n" for line in lines)


def collect(out_dir, backend):
    """row index -> sample; prepares; runtimes; summaries."""
    done, prepares, runtimes, summaries = {}, [], [], []
    for chunk in campaign.chunk_dirs(out_dir):
        if not (chunk / "batched.rows.json").exists():
            continue
        indices = ev.read_json(chunk / "batched.rows.json")
        for record in campaign.read_lines(chunk / "batched.driver.jsonl"):
            kind = record.get("kind")
            if kind == "runtime":
                if record["backend"] != backend or record["kv_mode"] != "fp32":
                    raise ValueError("invalid runtime record")
                runtimes.append(record)
            elif kind == "summary":
                summaries.append({**record, "chunk": chunk.name})
            elif kind in ("sample", "prefix_prepare"):
                local = record["case_index"]
                if not 0 <= local < len(indices):
                    raise ValueError("driver case index outside its chunk")
                record = {**record, "case_index": indices[local], "chunk": chunk.name}
                if kind == "prefix_prepare":
                    prepares.append(record)
                else:
                    if record["case_index"] in done:
                        raise ValueError("a question was completed twice")
                    done[record["case_index"]] = record
            else:
                raise ValueError("unexpected driver record")
    return done, prepares, runtimes, summaries


def summarize(rows, order, done, prepares, runtimes, summaries, cap):
    stable = {json.dumps({k: v for k, v in r.items() if k != "kind"}, sort_keys=True) for r in runtimes}
    if len(stable) != 1:
        raise ValueError("runtime metadata changed between chunks")
    eos = runtimes[0]["eos"]
    outcomes = {}
    for i in order:
        s = done[i]
        case = {"target_index": rows[i]["target_index"]}
        o = ev.generation_outcome(s, case, True, eos, cap, s["vocab"])
        outcomes[rows[i]["question_id"]] = {
            "id": rows[i]["id"], "subject": rows[i]["subject"], "correct": o["correct"], "valid": o["valid"],
            "stop": o["stop"], "generated_tokens": o["generated_tokens"], "predicted_index": o["predicted_index"],
            "latency_ms": s["finished_ms"] - s["started_ms"], "prefill_ms": s["prefill_ms"],
            "tokens_by_batch_size": s["tokens_by_batch_size"]}
    n = len(order)
    correct = sum(o["correct"] for o in outcomes.values())
    tokens = sum(o["generated_tokens"] for o in outcomes.values())
    by_size = [sum(o["tokens_by_batch_size"][k] for o in outcomes.values()) for k in range(9)]
    lat = [o["latency_ms"] for o in outcomes.values()]
    full = [s for s in summaries if s["tokens"] > 0]
    return {
        "questions": n, "correct": correct, "accuracy": correct / n,
        "capped": sum(o["stop"] == "limit" for o in outcomes.values()),
        "invalid": sum(not o["valid"] for o in outcomes.values()),
        "generated_tokens": tokens, "tokens_by_batch_size": by_size,
        "latency_ms": {"p50": bench.percentile(lat, .5), "p95": bench.percentile(lat, .95)},
        "complete_driver_runs": {
            "runs": len(full), "tokens": sum(s["tokens"] for s in full),
            "decode_ms": sum(s["decode_ms"] for s in full), "wall_ms": sum(s["wall_ms"] for s in full),
            "tokens_per_s_decode": sum(s["tokens"] for s in full) / (sum(s["decode_ms"] for s in full) / 1e3)
            if full else None},
        "prefix_prepares": len(prepares),
        "runtime": runtimes[0],
        "per_question": outcomes,
    }


def progress(args, rows, order, manifest):
    done, prepares, runtimes, summaries = collect(args.out_dir, args.backend)
    pending = [i for i in order if i not in done]
    status = {"questions": len(order), "complete": len(done), "pending": len(pending),
              "chunks": len(campaign.chunk_dirs(args.out_dir))}
    ev.write_json(args.out_dir / "status.json", status)
    if not pending and not (args.out_dir / "report.json").exists():
        report = {"manifest": manifest,
                  "summary": summarize(rows, order, done, prepares, runtimes, summaries, args.cap),
                  "chunks": [ev.read_json(c / "chunk.json") for c in campaign.chunk_dirs(args.out_dir)]}
        ev.write_json(args.out_dir / "report.json", report)
        print(args.out_dir / "report.json")
    print(json.dumps(status))
    return pending, status


def run(args):
    deadline = time.time() + args.deadline_minutes * 60  # wall clock (see campaign.run_driver)
    rows = prefix.load_rows(args.cases_dir, args.split, PROFILE)
    order, cuts = prefix.plan_cuts(rows, PROFILE, args.align)
    prompt_cap = max(len(r["profiles"][PROFILE]["prompt_ids"]) for r in rows)
    command = [str(args.binary.resolve()), str(args.model.resolve()), args.backend, str(prompt_cap),
               str(args.cap), str(args.batch)]
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads))
    env.setdefault("OMP_WAIT_POLICY", "passive")
    manifest = {
        "protocol": "geist-decision-batched-reasoning-v1", "tracking_issue": "geistlib#26",
        "plan": "docs/DECISION_CAMPAIGN_PLAN.json batched_reasoning_26",
        "model_sha256": bench.sha256(args.model), "binary_sha256": bench.sha256(args.binary),
        "cases_sha256": bench.sha256(args.cases_dir / f"{args.split}.jsonl"),
        "profile": PROFILE, "split": args.split, "align": args.align, "cap": args.cap, "batch": args.batch,
        "backend": args.backend, "threads": args.threads, "command": command,
        "order": [rows[i]["id"] for i in order], "cuts": cuts,
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
    pending, status = progress(args, rows, order, manifest)
    if args.status or not pending:
        return status
    batch = pending[:args.chunk_questions]
    chunk = args.out_dir / "chunks" / f"{len(campaign.chunk_dirs(args.out_dir)):04d}"
    chunk.mkdir(parents=True)
    info = {"started_unix": time.time(), "rows": batch, "lock_note": os.environ.get("LOCK_NOTE"),
            "deadline_minutes": args.deadline_minutes}
    ev.write_json(chunk / "chunk.json", info)
    temps = campaign.TempLog(chunk / "temps.jsonl")
    try:
        info["cool_gate"] = campaign.cool_gate(args, deadline)
        ev.write_json(chunk / "batched.rows.json", batch)
        info["run"] = campaign.run_driver(command, [], env, chunk, "batched", deadline,
                                          wire=wire(rows, batch, cuts))
    finally:
        temps.close()
        info["ended_unix"] = time.time()
        ev.write_json(chunk / "chunk.json", info)
    return progress(args, rows, order, manifest)[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("--cases-dir", "--model", "--binary", "--out-dir"):
        p.add_argument(name, type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--backend", choices=("cpu_neon", "metal"), required=True)
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--align", type=int, required=True, help="the engine's prefill chunk")
    p.add_argument("--cap", type=int, required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--chunk-questions", type=int, required=True)
    p.add_argument("--deadline-minutes", type=float, required=True)
    p.add_argument("--cool-max-c", type=float, default=65.)
    p.add_argument("--cool-timeout-minutes", type=float, default=5.)
    p.add_argument("--status", action="store_true", help="report progress (and finalize if complete); no inference")
    args = p.parse_args()
    if (args.threads < 1 or args.chunk_questions < 1 or args.align < 1 or not args.deadline_minutes > 0
            or not 1 <= args.cap <= 32768 or not 1 <= args.batch <= 8):
        p.error("invalid thread/chunk/align/deadline/cap/batch bounds")
    run(args)


if __name__ == "__main__":
    sys.exit(main())

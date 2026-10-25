#!/usr/bin/env python3
"""Shared-prefix direct scoring against independent calls, chunked and resumable (#26).

The questions of one subject share their prompt up to the test question (the
instruction, the subject header and the five exemplars). Per subject the
cached prefix is the longest common token prefix of all its questions minus
one token (text after it may merge with that token), aligned down to the
declared prefill chunk, so the prefix's passes are the passes a single
prefill of the joined prompt runs and the logits can be compared bit for bit.

Two arms run in one bench_decision process per chunk, one trial each, their
order rotating per case: decision_dense scores every prompt independently
(the reference), decision_prefix restores the subject's prefix snapshot and
prefills only the suffix. The driver prepares a prefix when a subject starts
and reports that time separately (`prefix_prepare`).

Every invocation runs at most --chunk-questions pending questions in subject
order and stops the driver at --deadline-minutes (wall clock), so it fits one
lease of the host lock. The manifest (model, binary, cases, per-subject
cuts, command, environment) is frozen on the first invocation and must be
identical on every resume. Each driver run starts on a cool chip; macmon is
logged during the run. When all questions are complete, report.json is built.

--reference-from DIR runs the prefix arm only and takes the independent
decision_dense samples from an earlier complete run in DIR. That run must have
the same model, binary, cases, profile, split, backend, threads and
environment (greedy scoring is deterministic across processes, #171), so a
second cut alignment can be measured without re-running the reference.
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
MODES = ("decision_dense", "decision_prefix")
VOLATILE = ("started_utc",)


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def load_rows(cases_dir, split, profile):
    rows = [json.loads(line) for line in (cases_dir / f"{split}.jsonl").read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("no prepared cases")
    for r in rows:
        if profile not in r["profiles"]:
            raise ValueError(f"{r['id']}: no {profile} profile")
    return rows


def plan_cuts(rows, profile, align):
    """Subject order (stable within a subject) and each subject's aligned cut."""
    subjects = {}
    for i, r in enumerate(rows):
        subjects.setdefault(r["subject"], []).append(i)
    order, cuts = [], {}
    for subject in sorted(subjects):
        ids = [rows[i]["profiles"][profile]["prompt_ids"] for i in subjects[subject]]
        shared = min(len(ids[0]), *(common_prefix(ids[0], x) for x in ids[1:])) if len(ids) > 1 else 0
        cut = max(shared - 1, 0) // align * align
        if any(cut >= len(x) for x in ids):
            raise ValueError(f"{subject}: cut leaves no suffix")
        cuts[subject] = {"questions": len(ids), "shared_tokens": shared, "cut": cut}
        order.extend(subjects[subject])
    return order, cuts


def wire(rows, indices, profile, cuts):
    lines = []
    for i in indices:
        p = rows[i]["profiles"][profile]
        cut = cuts[rows[i]["subject"]]["cut"]
        lines.append(" ".join(map(str, [cut, len(p["prompt_ids"]), len(p["candidate_ids"]),
                                         *p["prompt_ids"], *p["candidate_ids"]])))
    return "".join(line + "\n" for line in lines)


def collect(out_dir, backend, modes=MODES):
    """row index -> {mode: sample}; prepare records; runtime records."""
    done, prepares, runtimes = {}, [], []
    for chunk in campaign.chunk_dirs(out_dir):
        index_path = chunk / "prefix.rows.json"
        if not index_path.exists():
            continue
        indices = ev.read_json(index_path)
        recv = {}
        if (chunk / "prefix.recv.tsv").exists():
            for line in (chunk / "prefix.recv.tsv").read_text().splitlines():
                n, t = line.split("\t")
                recv[int(n)] = float(t)
        groups = {}
        for line_no, record in enumerate(campaign.read_lines(chunk / "prefix.driver.jsonl")):
            kind = record.get("kind")
            if kind == "runtime":
                if record["backend"] != backend or record["kv_mode"] != "fp32":
                    raise ValueError("invalid runtime record")
                runtimes.append(record)
                continue
            if kind not in ("sample", "prefix_prepare"):
                raise ValueError("unexpected driver record")
            local = record["case_index"]
            if not 0 <= local < len(indices):
                raise ValueError("driver case index outside its chunk")
            record = {**record, "case_index": indices[local], "chunk": chunk.name, "recv_unix": recv.get(line_no)}
            if kind == "prefix_prepare":
                prepares.append(record)
            else:
                groups.setdefault(indices[local], {})
                if record["mode"] in groups[indices[local]]:
                    raise ValueError("duplicate sample in a chunk")
                groups[indices[local]][record["mode"]] = record
        for row, samples in groups.items():
            if set(samples) != set(modes):
                continue  # interrupted: re-run, never merged
            if row in done:
                raise ValueError("a question was completed twice")
            done[row] = samples
    return done, prepares, runtimes


def summarize(rows, order, cuts, done, prepares, runtimes, profile, align):
    stable = {json.dumps({k: v for k, v in r.items() if k != "kind"}, sort_keys=True) for r in runtimes}
    if len(stable) != 1:
        raise ValueError("runtime metadata changed between chunks")
    for p in prepares:
        if p["prefill_chunk"] != align:
            raise ValueError(f"engine chunk {p['prefill_chunk']} differs from the declared {align}")
        if p["cached_tokens"] != cuts[rows[p["case_index"]]["subject"]]["cut"]:
            raise ValueError("prepared prefix length differs from the declared cut")
    identical = argmax_same = 0
    mismatches = []
    correct = {m: 0 for m in MODES}
    times = {m: [] for m in MODES}
    by_subject = {}
    for i in order:
        r, s = rows[i], done[i]
        dense, pref = s["decision_dense"], s["decision_prefix"]
        target = r["target_index"]
        if pref.get("cached_tokens") != cuts[r["subject"]]["cut"]:
            raise ValueError("prefix sample used another cut")
        if dense["logits"] == pref["logits"]:
            identical += 1
        else:
            mismatches.append({"id": r["id"], "max_abs_diff": max(abs(a - b) for a, b in zip(dense["logits"], pref["logits"]))})
        argmax_same += dense["best_index"] == pref["best_index"]
        for m in MODES:
            correct[m] += s[m]["best_index"] == target
            times[m].append(s[m]["elapsed_ms"])
        b = by_subject.setdefault(r["subject"], {"questions": 0, "prompt_tokens": [], "dense_ms": 0., "prefix_ms": 0.})
        b["questions"] += 1
        b["prompt_tokens"].append(len(r["profiles"][profile]["prompt_ids"]))
        b["dense_ms"] += dense["elapsed_ms"]
        b["prefix_ms"] += pref["elapsed_ms"]
    first_prepare = {}
    for p in sorted(prepares, key=lambda p: (p["chunk"], p["case_index"])):
        first_prepare.setdefault(rows[p["case_index"]]["subject"], p)
    for subject, b in by_subject.items():
        prep = first_prepare[subject]["elapsed_ms"]
        b.update(cached_tokens=cuts[subject]["cut"], shared_tokens=cuts[subject]["shared_tokens"],
                 mean_prompt_tokens=sum(b["prompt_tokens"]) / len(b["prompt_tokens"]),
                 prepare_ms=prep, snapshot_bytes=first_prepare[subject]["snapshot_bytes"],
                 shared_over_independent=(prep + b["prefix_ms"]) / b["dense_ms"])
        del b["prompt_tokens"]
    n = len(order)
    total_dense = sum(times["decision_dense"])
    total_prefix = sum(times["decision_prefix"])
    prep_once = sum(p["elapsed_ms"] for p in first_prepare.values())
    prep_all = sum(p["elapsed_ms"] for p in prepares)
    pct = lambda v, q: bench.percentile(v, q)
    rss = [s[m].get("process_peak_rss_bytes", 0) for s in done.values() for m in MODES]
    return {
        "questions": n, "subjects": len(by_subject), "prefill_chunk": align,
        "bit_identical_option_logits": identical, "argmax_identical": argmax_same,
        "mismatches": mismatches,
        "accuracy": {m: correct[m] / n for m in MODES},
        "correct": correct,
        "latency_ms": {m: {"p50": pct(times[m], .5), "p95": pct(times[m], .95), "mean": sum(times[m]) / n,
                           "sum": sum(times[m])} for m in MODES},
        "prepare_ms": {"one_per_subject_sum": prep_once, "measured_sum_including_chunk_restarts": prep_all,
                       "measured_prepares": len(prepares),
                       "p50": pct([p["elapsed_ms"] for p in first_prepare.values()], .5),
                       "max": max(p["elapsed_ms"] for p in first_prepare.values())},
        "end_to_end": {
            "independent_ms": total_dense,
            "shared_prefix_ms": total_prefix + prep_once,
            "ratio": (total_prefix + prep_once) / total_dense,
            "scope": "Shared = one prefix prepare per subject + every question's restore + suffix score. Model "
                     "load excluded for both. Prepares repeated only because a chunk started inside a subject "
                     "are reported separately and not counted."},
        "snapshot_bytes": {"max": max(p["snapshot_bytes"] for p in prepares),
                           "min": min(p["snapshot_bytes"] for p in prepares)},
        "process_peak_rss_bytes_max": max(rss),
        "by_subject": dict(sorted(by_subject.items())),
        "runtime": runtimes[0],
    }


def reference_samples(args, order):
    """The decision_dense samples of the --reference-from run, by row index."""
    done, _, runtimes = collect(args.reference_from, args.backend)
    missing = [i for i in order if i not in done]
    if missing:
        raise ValueError(f"reference run lacks {len(missing)} questions")
    return {i: done[i]["decision_dense"] for i in order}, runtimes


def progress(args, rows, order, cuts, manifest):
    """Write status.json; build report.json once every question is complete."""
    modes = ("decision_prefix",) if args.reference_from else MODES
    done, prepares, runtimes = collect(args.out_dir, args.backend, modes)
    pending = [i for i in order if i not in done]
    if not pending and args.reference_from:
        dense, ref_runtimes = reference_samples(args, order)
        done = {i: {**done[i], "decision_dense": dense[i]} for i in order}
        runtimes = runtimes + ref_runtimes
    status = {"questions": len(order), "complete": len(done), "pending": len(pending),
              "chunks": len(campaign.chunk_dirs(args.out_dir))}
    ev.write_json(args.out_dir / "status.json", status)
    if not pending and not (args.out_dir / "report.json").exists():
        report = {"manifest": manifest,
                  "summary": summarize(rows, order, cuts, done, prepares, runtimes, args.profile, args.align),
                  "chunks": [ev.read_json(c / "chunk.json") for c in campaign.chunk_dirs(args.out_dir)]}
        ev.write_json(args.out_dir / "report.json", report)
        print(args.out_dir / "report.json")
    print(json.dumps(status))
    return pending, status


def run(args):
    deadline = time.time() + args.deadline_minutes * 60  # wall clock (see campaign.run_driver)
    rows = load_rows(args.cases_dir, args.split, args.profile)
    order, cuts = plan_cuts(rows, args.profile, args.align)
    prompt_cap = max(len(r["profiles"][args.profile]["prompt_ids"]) for r in rows)
    candidate_cap = max(len(r["profiles"][args.profile]["candidate_ids"]) for r in rows)
    command = [str(args.binary.resolve()), str(args.model.resolve()), args.backend, str(prompt_cap),
               str(candidate_cap), "2", "0", "0", "--prefix", "--modes",
               "decision_prefix" if args.reference_from else ",".join(MODES)]
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads))
    env.setdefault("OMP_WAIT_POLICY", "passive")
    manifest = {
        "protocol": "geist-decision-prefix-v1", "tracking_issue": "geistlib#26",
        "plan": "docs/DECISION_CAMPAIGN_PLAN.json shared_prefix_campaign_26",
        "model_sha256": bench.sha256(args.model), "binary_sha256": bench.sha256(args.binary),
        "cases_sha256": bench.sha256(args.cases_dir / f"{args.split}.jsonl"),
        "profile": args.profile, "split": args.split, "align": args.align, "backend": args.backend,
        "threads": args.threads, "command": command, "order": [rows[i]["id"] for i in order], "cuts": cuts,
        "environment": {k: v for k, v in env.items() if k.startswith(("GEIST_", "OMP_", "KMP_"))},
        "source_revision": bench.command_output(["git", "rev-parse", "HEAD"]),
    }
    if args.reference_from:
        ref = ev.read_json(args.reference_from / "manifest.json")
        same = ("model_sha256", "binary_sha256", "cases_sha256", "profile", "split", "backend", "threads",
                "environment")
        diff = [k for k in same if ref.get(k) != manifest[k]]
        if diff:
            raise ValueError(f"reference run differs: {diff}")
        manifest["reference_from"] = {"dir": str(args.reference_from.resolve()),
                                      "manifest_sha256": bench.sha256(args.reference_from / "manifest.json")}
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
    pending, status = progress(args, rows, order, cuts, manifest)
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
        ev.write_json(chunk / "prefix.rows.json", batch)
        info["run"] = campaign.run_driver(command, [], env, chunk, "prefix", deadline,
                                          wire=wire(rows, batch, args.profile, cuts))
    finally:
        temps.close()
        info["ended_unix"] = time.time()
        ev.write_json(chunk / "chunk.json", info)
    return progress(args, rows, order, cuts, manifest)[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("--cases-dir", "--model", "--binary", "--out-dir"):
        p.add_argument(name, type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--profile", default="chat_direct")
    p.add_argument("--backend", choices=("cpu_neon", "metal"), required=True)
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--align", type=int, required=True, help="declared prefill chunk; the engine's must match")
    p.add_argument("--chunk-questions", type=int, required=True)
    p.add_argument("--deadline-minutes", type=float, required=True)
    p.add_argument("--cool-max-c", type=float, default=65.)
    p.add_argument("--cool-timeout-minutes", type=float, default=5.)
    p.add_argument("--reference-from", type=Path,
                   help="earlier complete run of the same binding: prefix arm only, its decision_dense as reference")
    p.add_argument("--status", action="store_true", help="report progress (and finalize if complete); no inference")
    args = p.parse_args()
    if args.threads < 1 or args.chunk_questions < 1 or args.align < 1 or not args.deadline_minutes > 0:
        p.error("invalid thread/chunk/align/deadline bounds")
    run(args)


if __name__ == "__main__":
    sys.exit(main())

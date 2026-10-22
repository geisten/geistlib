#!/usr/bin/env python3
"""Resumable, chunked form of eval_decision.py for long campaigns on a shared host (#171).

Every invocation runs at most --chunk-questions pending questions and stops
the driver at --deadline-minutes, so it fits one bounded lease of the host
lock. The driver flushes every sample; each complete line is persisted as it
arrives, so an interrupted chunk loses at most the question in flight.
A (question, profile) pair counts as done only when all its modes and trials
are present; incomplete pairs are re-run, complete ones never. When every
question is done, the samples are reassembled in prepared order and the
report is built by exactly the code of the one-shot runner.

The manifest is frozen by the first invocation, before any labelled
inference; every later invocation recomputes it and refuses to continue on
any difference other than its start time (binary, model, evaluator sources,
source revision, calibration file, environment, cases).

Latency: each driver run starts on a cool chip (GPU below --cool-max-c for
Metal, CPU for cpu_neon; macmon, METHODOLOGY.md#cool-start-on-apple-silicon).
Within a run the chip heats up; macmon is logged throughout and every sample
is annotated with the temperature at its start, so the report gives latency
for all warm samples and, separately, for the samples that started cool.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import bench_host
import eval_decision as ev

bench = ev.bench
VOLATILE = ("started_utc",)


def chunk_dirs(out_dir):
    root = out_dir / "chunks"
    return sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []


def read_lines(path):
    """Complete JSON lines only; the writer never persists a partial line."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def collect(out_dir, rows, profiles, warmup, repeats, backend):
    """profile -> {row index: samples}; runtime records; seconds of driver wall time per profile."""
    trials = 1 + warmup + repeats
    done = {p: {} for p in profiles}
    runtimes, walls = [], {p: 0. for p in profiles}
    for chunk in chunk_dirs(out_dir):
        for profile, modes in profiles.items():
            index_path = chunk / f"{profile}.rows.json"
            if not index_path.exists():
                continue
            indices = ev.read_json(index_path)
            records = read_lines(chunk / f"{profile}.driver.jsonl")
            recv = {}
            if (chunk / f"{profile}.recv.tsv").exists():
                for line in (chunk / f"{profile}.recv.tsv").read_text().splitlines():
                    n, t = line.split("\t")
                    recv[int(n)] = float(t)
            info = ev.read_json(chunk / f"{profile}.run.json") if (chunk / f"{profile}.run.json").exists() else {}
            walls[profile] += info.get("wall_seconds", 0.)
            groups = {}
            for line_no, record in enumerate(records):
                if record.get("kind") == "runtime":
                    if record["backend"] != backend or record["kv_mode"] != "fp32":
                        raise ValueError("invalid runtime record")
                    runtimes.append(record)
                    continue
                if record.get("kind") != "sample":
                    raise ValueError("unexpected driver record")
                local = record["case_index"]
                if not 0 <= local < len(indices):
                    raise ValueError("driver case index outside its chunk")
                record = {**record, "case_index": indices[local], "chunk": chunk.name,
                          "recv_unix": recv.get(line_no)}
                groups.setdefault(indices[local], []).append(record)
            for row, samples in groups.items():
                keys = {(s["mode"], s["trial"]) for s in samples}
                if len(keys) != len(samples):
                    raise ValueError("duplicate sample in a chunk")
                if keys != {(m, t) for m in modes for t in range(trials)}:
                    continue  # interrupted: re-run, never merged
                if row in done[profile]:
                    raise ValueError("a question was completed twice")
                done[profile][row] = samples
    return done, runtimes, walls


def status(out_dir, rows, profiles, args):
    done, _, _ = collect(out_dir, rows, profiles, args.warmup, args.repeats, args.backend)
    pending = [i for i in range(len(rows)) if any(i not in done[p] for p in profiles)]
    seconds = 0.
    for chunk in chunk_dirs(out_dir):
        info = ev.read_json(chunk / "chunk.json")
        if info.get("ended_unix"):
            seconds += info["ended_unix"] - info["started_unix"]
    value = {"questions": len(rows), "complete": len(rows) - len(pending), "pending": len(pending),
             "complete_by_profile": {p: len(done[p]) for p in profiles},
             "chunks": len(chunk_dirs(out_dir)), "chunk_seconds_total": round(seconds, 1)}
    ev.write_json(out_dir / "status.json", value)
    return value, done, pending


class TempLog:
    """macmon pipe in the background, each reading stamped on arrival."""
    def __init__(self, path):
        self.path, self.proc, self.thread = path, None, None
        macmon = bench_host.find_macmon()
        if macmon is None:
            return
        self.out = path.open("a")
        self.proc = subprocess.Popen([macmon, "pipe", "-i", "2000"], stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True)
        self.thread = threading.Thread(target=self.pump, daemon=True)
        self.thread.start()

    def pump(self):
        for line in self.proc.stdout:
            try:
                temp = json.loads(line)["temp"]
                value = {"t": time.time(), "gpu_c": round(float(temp["gpu_temp_avg"]), 1),
                         "cpu_c": round(float(temp["cpu_temp_avg"]), 1)}
            except (ValueError, KeyError, TypeError):
                continue
            self.out.write(json.dumps(value) + "\n")
            self.out.flush()

    def close(self):
        if self.proc is not None:
            self.proc.terminate()
            self.proc.wait()
            self.thread.join(timeout=5)
            self.out.close()


def cool_gate(args, deadline):
    limits = {"fallback_seconds": 45}
    limits["max_gpu_c" if args.backend == "metal" else "max_cpu_c"] = args.cool_max_c
    left = max(0., deadline - time.time())
    return bench_host.wait_for_apple_cool(limits, time.monotonic() + min(left, args.cool_timeout_minutes * 60))


def run_driver(command, cases, env, chunk, profile, deadline):
    """Stream the driver's lines to disk as they arrive; stop it at the deadline."""
    out = (chunk / f"{profile}.driver.jsonl").open("w")
    recv = (chunk / f"{profile}.recv.tsv").open("w")
    err = (chunk / f"{profile}.stderr.log").open("w")
    start = time.time()
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                            env=env, text=True)
    stopped, finished = threading.Event(), threading.Event()

    def watchdog():
        # Wall clock, not a monotonic timer: macOS's monotonic clock stands
        # still while the host sleeps, and a lease is measured in wall time
        # (a 2-hour sleep once let a chunk overrun its lease, #171).
        while not finished.wait(5):
            if time.time() >= deadline:
                stopped.set()
                proc.terminate()
                return
    timer = threading.Thread(target=watchdog, daemon=True)
    timer.start()

    def feed():
        try:
            proc.stdin.write(bench.wire_input(cases))
            proc.stdin.close()
        except (BrokenPipeError, ValueError):
            pass
    writer = threading.Thread(target=feed, daemon=True)
    writer.start()
    n, partial = 0, ""
    for line in proc.stdout:
        if not line.endswith("\n"):
            partial = line
            break
        out.write(line)
        out.flush()
        recv.write(f"{n}\t{time.time():.6f}\n")
        recv.flush()
        n += 1
    code = proc.wait()
    proc.stdout.close()
    finished.set()
    for f in (out, recv, err):
        f.close()
    info = {"started_unix": start, "wall_seconds": time.time() - start, "returncode": code,
            "lines": n, "stopped_at_deadline": stopped.is_set(), "discarded_partial_line_bytes": len(partial)}
    ev.write_json(chunk / f"{profile}.run.json", info)
    if code and not stopped.is_set():
        raise RuntimeError(f"driver failed ({code}): {profile}; raw partial outputs retained in {chunk}")
    return info


def temperatures(out_dir):
    values = []
    for chunk in chunk_dirs(out_dir):
        values.extend(read_lines(chunk / "temps.jsonl"))
    return sorted(values, key=lambda v: v["t"])


def thermal_latency(samples_by_profile, profiles, temps, backend, limit):
    """Per mode: warm-sample latency over all samples and over the cool-started ones."""
    key = "gpu_c" if backend == "metal" else "cpu_c"
    times = [v["t"] for v in temps]
    import bisect
    result = {}
    for profile, modes in profiles.items():
        for mode in modes:
            warm = [s for s in samples_by_profile[profile] if s["mode"] == mode and s["phase"] == "warm"]
            annotated = []
            for s in warm:
                temp = None
                if s.get("recv_unix") is not None and times:
                    start = s["recv_unix"] - s["elapsed_ms"] / 1000
                    i = bisect.bisect_right(times, start) - 1
                    if i >= 0 and start - times[i] <= 10:
                        temp = temps[i][key]
                annotated.append((s["elapsed_ms"], temp))
            known = [t for _, t in annotated if t is not None]
            cool = [ms for ms, t in annotated if t is not None and t < limit]
            entry = {"n_warm": len(annotated), "n_with_start_temperature": len(known),
                     "warm_p50_ms": bench.percentile([ms for ms, _ in annotated], .5),
                     "warm_p95_ms": bench.percentile([ms for ms, _ in annotated], .95),
                     f"start_{key}_p50": bench.percentile(known, .5) if known else None,
                     f"start_{key}_max": max(known) if known else None,
                     "cool_start_limit_c": limit, "n_cool_start": len(cool),
                     "cool_start_p50_ms": bench.percentile(cool, .5) if cool else None,
                     "cool_start_p95_ms": bench.percentile(cool, .95) if cool else None}
            result[f"{profile}/{mode}"] = entry
    return result


def finalize(args, meta, rows, bound, calibration, profiles, manifest, done):
    cases = {p: bench.cases_from_jsonl(args.out_dir / f"{p}.cases.jsonl") for p in profiles}
    _, runtimes, walls = collect(args.out_dir, rows, profiles, args.warmup, args.repeats, args.backend)
    stable = {json.dumps({k: v for k, v in r.items() if k != "kind"}, sort_keys=True) for r in runtimes}
    if len(stable) != 1:
        raise ValueError("runtime metadata changed between chunks")
    outputs = {}
    for profile in profiles:
        samples = [s for i in range(len(rows)) for s in done[profile][i]]
        outputs[profile] = (runtimes[0], samples, walls[profile] * 1000)
    report = ev.build_report(args, meta, rows, bound, calibration, profiles, cases, outputs, manifest)
    chunks = [ev.read_json(c / "chunk.json") for c in chunk_dirs(args.out_dir)]
    report["campaign"] = {
        "chunks": len(chunks),
        "host_seconds_in_chunks": sum(c["ended_unix"] - c["started_unix"] for c in chunks if c.get("ended_unix")),
        "driver_seconds": sum(walls.values()),
        "interrupted_driver_runs": sum(ev.read_json(p)["stopped_at_deadline"]
                                       for c in chunk_dirs(args.out_dir) for p in c.glob("*.run.json")),
        "scope": "Chunked under the host lock; each driver run started on a cool chip, model loaded per run. "
                 "Mode order inside a run rotates with the case index within that run, not the global index."}
    report["latency_by_start_temperature"] = thermal_latency(
        {p: o[1] for p, o in outputs.items()}, profiles, temperatures(args.out_dir), args.backend, args.cool_max_c)
    ev.write_json(args.out_dir / "report.json", report)
    return args.out_dir / "report.json"


def import_samples(args, rows, profiles, cases, manifest):
    """Seed chunk 0000 with complete samples of an earlier run (#171 amendment).

    Allowed only for an identical binding (model, binary, tokenizer, template,
    backend, environment, threads) and decode/warmup/repeat policy, and only
    for questions whose prompt and candidate IDs are identical. Only the modes
    this run declares are taken; the rest of the earlier samples stay there.
    """
    old_dir = args.import_from
    old = ev.read_json(old_dir / "manifest.json")
    for key in ("binding", "decode_cap", "warmup", "repeats", "dataset_revision", "shots"):
        if old[key] != manifest[key]:
            raise ValueError(f"import: {key} differs from the earlier run")
    old_profiles = {p: tuple(c[c.index("--modes") + 1].split(",")) for p, c in old["commands"].items()}
    old_cases = {p: bench.cases_from_jsonl(old_dir / f"{p}.cases.jsonl") for p in old_profiles}
    first = next(iter(old_cases.values()))
    done, runtimes, _ = collect(old_dir, first, old_profiles, args.warmup, args.repeats, args.backend)
    chunk = args.out_dir / "chunks" / "0000"
    chunk.mkdir(parents=True)
    imported = {}
    for profile, modes in profiles.items():
        if profile not in old_profiles or not set(modes) <= set(old_profiles[profile]):
            continue
        by_id = {c["id"]: i for i, c in enumerate(old_cases[profile])}
        take = []
        for i, case in enumerate(cases[profile]):
            j = by_id.get(case["id"])
            if j is None or j not in done[profile]:
                continue
            o = old_cases[profile][j]
            if o["prompt_ids"] != case["prompt_ids"] or o["candidate_ids"] != case["candidate_ids"]:
                raise ValueError(f"import: prompt differs for {case['id']}")
            take.append((i, [s for s in done[profile][j] if s["mode"] in modes]))
        if not take:
            continue
        ev.write_json(chunk / f"{profile}.rows.json", [i for i, _ in take])
        with (chunk / f"{profile}.driver.jsonl").open("w") as out, (chunk / f"{profile}.recv.tsv").open("w") as recv:
            out.write(json.dumps(runtimes[0]) + "\n")
            n = 1
            for local, (_, samples) in enumerate(take):
                for sample in sorted(samples, key=lambda s: (s["trial"], s["order"])):
                    t = sample["recv_unix"]
                    sample = {k: v for k, v in sample.items() if k not in ("chunk", "recv_unix")}
                    out.write(json.dumps({**sample, "case_index": local}) + "\n")
                    if t is not None:
                        recv.write(f"{n}\t{t:.6f}\n")
                    n += 1
        ev.write_json(chunk / f"{profile}.run.json", {"imported_from": str(old_dir.resolve()), "wall_seconds": 0.,
                                                       "returncode": 0, "stopped_at_deadline": False, "lines": n})
        imported[profile] = len(take)
    temps = [v for c in chunk_dirs(old_dir) for v in read_lines(c / "temps.jsonl")]
    (chunk / "temps.jsonl").write_text("".join(json.dumps(v) + "\n" for v in temps))
    now = time.time()
    ev.write_json(chunk / "chunk.json", {"started_unix": now, "ended_unix": now, "imported_from": str(old_dir.resolve()),
                                         "imported_questions": imported,
                                         "source_manifest_sha256": bench.sha256(old_dir / "manifest.json"),
                                         "note": "Samples measured in the earlier run (its host time is accounted there); "
                                                 "they were interleaved with that run's other arms.",
                                         "runs": {}})
    return imported


def run(args):
    deadline = time.time() + args.deadline_minutes * 60  # wall clock, see run_driver
    meta, rows, env, bound, calibration, profiles, flats, commands, manifest = ev.prepare_run(args)
    frozen = args.out_dir / "manifest.json"
    if frozen.exists():
        old = ev.read_json(frozen)
        fresh = json.loads(json.dumps(manifest))
        if {k: v for k, v in old.items() if k not in VOLATILE} != {k: v for k, v in fresh.items() if k not in VOLATILE}:
            diff = sorted(k for k in set(old) | set(fresh) if k not in VOLATILE and old.get(k) != fresh.get(k))
            raise ValueError(f"frozen manifest differs on resume: {diff}")
        cases = {p: bench.cases_from_jsonl(args.out_dir / f"{p}.cases.jsonl") for p in profiles}
    else:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        if any(args.out_dir.iterdir()):
            raise FileExistsError(f"{args.out_dir} has content but no manifest")
        cases = ev.write_inputs(args, meta, profiles, flats)
        ev.write_json(frozen, manifest)  # frozen before any labelled inference
    if getattr(args, "import_from", None) and not chunk_dirs(args.out_dir):
        print(json.dumps({"imported": import_samples(args, rows, profiles, cases, manifest)}))
    value, done, pending = status(args.out_dir, rows, profiles, args)
    if args.status or not pending:
        if not pending and not (args.out_dir / "report.json").exists():
            print(finalize(args, meta, rows, bound, calibration, profiles, manifest, done))
        print(json.dumps(value))
        return value
    batch = pending[:args.chunk_questions]
    chunk = args.out_dir / "chunks" / f"{len(chunk_dirs(args.out_dir)):04d}"
    chunk.mkdir(parents=True)
    info = {"started_unix": time.time(), "rows": batch, "lock_note": os.environ.get("LOCK_NOTE"),
            "deadline_minutes": args.deadline_minutes, "runs": {}}
    ev.write_json(chunk / "chunk.json", info)
    temps = TempLog(chunk / "temps.jsonl")
    try:
        for profile in profiles:
            todo = [i for i in batch if i not in done[profile]]
            if not todo:
                continue
            if time.time() >= deadline:
                break
            gate = cool_gate(args, deadline)
            ev.write_json(chunk / f"{profile}.rows.json", todo)
            run_info = run_driver(commands[profile], [cases[profile][i] for i in todo], env, chunk, profile, deadline)
            info["runs"][profile] = {"cool_gate": gate, **run_info}
            ev.write_json(chunk / "chunk.json", info)
            if run_info["stopped_at_deadline"]:
                break
    finally:
        temps.close()
        info["ended_unix"] = time.time()
        ev.write_json(chunk / "chunk.json", info)
    value, done, pending = status(args.out_dir, rows, profiles, args)
    if not pending:
        print(finalize(args, meta, rows, bound, calibration, profiles, manifest, done))
    print(json.dumps(value))
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("--cases-dir", "--model", "--binary", "--out-dir"):
        p.add_argument(name, type=Path, required=True)
    p.add_argument("--split", choices=("development", "calibration", "test"), required=True)
    p.add_argument("--purpose", choices=("pilot", "calibration", "quality"), required=True)
    p.add_argument("--backend", choices=("cpu_neon", "metal"), required=True)
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--decode-cap", type=int, required=True)
    p.add_argument("--margin-pp", type=float, required=True, help="declared before test; never selected from results")
    p.add_argument("--warmup", type=int, required=True)
    p.add_argument("--repeats", type=int, required=True)
    p.add_argument("--calibration", type=Path)
    p.add_argument("--calibration-modes", type=lambda v: v.split(","),
                   help="calibration split only: numeric arms to run (default: all three)")
    p.add_argument("--import-from", type=Path,
                   help="an earlier run of the same binding: reuse its complete samples of these modes")
    p.add_argument("--chunk-questions", type=int, required=True, help="pending questions per invocation")
    p.add_argument("--deadline-minutes", type=float, required=True, help="the driver is stopped at this point")
    p.add_argument("--cool-max-c", type=float, default=65.)
    p.add_argument("--cool-timeout-minutes", type=float, default=5.)
    p.add_argument("--status", action="store_true", help="report progress (and finalize if complete); no inference")
    args = p.parse_args()
    if (not 2 <= args.decode_cap <= 32768 or not 0 <= args.warmup <= 1000
            or not 1 <= args.repeats <= 10000 or args.threads < 1 or args.chunk_questions < 1
            or not args.deadline_minutes > 0 or not math.isfinite(args.margin_pp) or not 0 <= args.margin_pp <= 100):
        p.error("invalid decode/warmup/repeat/thread/chunk/deadline/margin bounds")
    run(args)


if __name__ == "__main__":
    sys.exit(main())

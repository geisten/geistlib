#!/usr/bin/env python3
"""Run a paired geist versus llama.cpp throughput benchmark.

The two engines use the same model bytes, workload shapes, thread counts,
sample count, host gate, and aggregation. The protocol file decides the
backend contract: `backend: "cpu-only"` verifies zero GPU offload,
`backend: "gpu"` verifies full offload and records the exact device.
Each engine retains its native synthetic token stream; token identity is
therefore not a parity claim. Correctness/quality remains a separate
prerequisite for publishing a speedup.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "benchmark" / "cross_engine_cpu_protocol.json"
ERROR = 99

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_host import (  # noqa: E402,F401  (re-exported for bench_thread_sweep.py)
    build_schedule_alternating as build_schedule,
    check_schedule_prefix,
    load_partial,
    command_output,
    machine_state,
    median_absolute_deviation,
    perf_sweep_command,
    run_jsonl as run_command,
    sha256_file,
    wait_for_apple_cool,
    wait_for_apple_power,
    wait_for_quiet,
)


def checkout_root(binary: Path) -> Path | None:
    current = binary.parent
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return None


def engine_metadata(label: str, kind: str, binary: Path) -> dict:
    root = checkout_root(binary)
    commit = command_output(["git", "-C", str(root), "rev-parse", "HEAD"]) if root else "unavailable"
    return {
        "label": label,
        "kind": kind,
        "binary": str(binary),
        "binary_sha256": sha256_file(binary),
        "commit": commit,
        "linked_libraries": command_output(
            ["otool", "-L", str(binary)] if platform.system() == "Darwin"
            else ["ldd", str(binary)]).splitlines(),
    }


def system_metadata() -> dict:
    """Host facts a reader needs to judge a number, per platform.

    macOS answers through sysctl/sw_vers, Linux through /proc and lscpu. The
    keys stay identical so two hosts' artifacts remain diffable; a value the
    platform cannot answer stays "unavailable" rather than being faked from a
    neighbouring one."""
    if platform.system() == "Darwin":
        return {
            "cpu": command_output(["sysctl", "-n", "machdep.cpu.brand_string"]),
            "logical_cores": os.cpu_count(),
            "performance_cores": command_output(["sysctl", "-n", "hw.perflevel0.physicalcpu"]),
            "memory_bytes": command_output(["sysctl", "-n", "hw.memsize"]),
            "os": f"{platform.system()} {platform.release()}",
            "product": command_output(["sw_vers"]),
            "compiler": command_output(["clang", "--version"]).splitlines()[0],
        }
    # LC_ALL=C: lscpu localizes its field names, and a German host would
    # otherwise record "unavailable" for its own CPU.
    lscpu = command_output(["env", "LC_ALL=C", "lscpu"])
    def lscpu_field(name: str) -> str:
        for line in lscpu.splitlines():
            key, _, value = line.partition(":")
            if key.strip() == name:
                return value.strip()
        return "unavailable"
    try:
        mem_kb = next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                      if line.startswith("MemTotal:"))
        memory_bytes = str(int(mem_kb) * 1024)
    except (OSError, StopIteration, ValueError):
        memory_bytes = "unavailable"
    sockets = lscpu_field("Socket(s)")
    per_socket = lscpu_field("Core(s) per socket")
    try:
        physical = str(int(sockets) * int(per_socket))
    except ValueError:
        physical = "unavailable"
    return {
        "cpu": lscpu_field("Model name"),
        "logical_cores": os.cpu_count(),
        "performance_cores": physical,
        "memory_bytes": memory_bytes,
        "os": f"{platform.system()} {platform.release()}",
        "product": command_output(["sh", "-c", ". /etc/os-release && echo \"$PRETTY_NAME\""]),
        "compiler": command_output(["cc", "--version"]).splitlines()[0],
    }


def repeat_spread_pct(result: dict) -> float:
    """Largest (max - min) / median over one run's repeats, prefill and
    decode, every sequence length: what other work on the host does to a
    run that the gate before it cannot see."""
    worst = 0.0
    for row in result["rows"]:
        for key in ("prefill_tps", "decode_tps"):
            values = row[key]
            mid = statistics.median(values)
            if mid > 0:
                worst = max(worst, (max(values) - min(values)) / mid * 100.0)
    return worst


def run_geist(engine: dict, model: Path, workload: dict) -> dict:
    command = perf_sweep_command(engine["binary"], model, workload["seq_lens"],
                                 workload["decode_n"], 64, workload["repeats"],
                                 workload["prefill_threads"])
    env = dict(os.environ)
    env.update({
        # Measurement condition: active wait for every run, also on libgomp, where
        # it is unbounded; harmless here because each run exits when it is done.
        "OMP_WAIT_POLICY": "active",
        "OMP_NUM_THREADS": str(workload["prefill_threads"]),
        "GEIST_PREFILL_THREADS": str(workload["prefill_threads"]),
        "GEIST_DECODE_THREADS": str(workload["decode_threads"]),
        "GEIST_TEXT_ONLY": "1",
    })
    # Name the backend: "auto" is metal on Apple Silicon, so a CPU cell left
    # to it would measure the GPU. The stderr check below fails loudly if the
    # backend is not available.
    env["GEIST_BENCH_BACKEND"] = workload["expect_backend"]
    rows, stderr, elapsed = run_command(command, env)
    expected = workload["expect_backend"]
    if not any(line.strip() == f"[bench] backend: {expected}" for line in stderr):
        raise RuntimeError(f"geist did not select the {expected} backend")
    if [row.get("seq_len") for row in rows] != workload["seq_lens"]:
        raise RuntimeError("geist emitted an unexpected sequence sweep")
    normalized = []
    for row in rows:
        if row.get("threads") != workload["prefill_threads"]:
            raise RuntimeError("geist did not record the requested prefill thread count")
        samples = row.get("samples", {})
        prefill_ms = samples.get("prefill_ms", [])
        decode_ms = samples.get("decode_ms", [])
        if len(prefill_ms) != workload["repeats"] or len(decode_ms) != workload["repeats"]:
            raise RuntimeError("geist did not retain every ordered timing sample")
        normalized.append({
            "seq_len": row["seq_len"],
            "prefill_tps": [row["seq_len"] * 1000.0 / value for value in prefill_ms],
            "decode_tps": [workload["decode_n"] * 1000.0 / value for value in decode_ms],
            "native": row,
        })
    return {"rows": normalized, "stderr": stderr, "elapsed_seconds": elapsed}


def validate_llama_row(row: dict, engine: dict, workload: dict, expected_threads: int) -> None:
    if not engine["commit"].startswith(row.get("build_commit", "missing")):
        raise RuntimeError("llama.cpp JSON commit does not match the pinned checkout")
    if workload.get("backend") == "gpu":
        if row.get("n_gpu_layers") != workload["gpu_offload_layers"] or not row.get("gpu_info"):
            raise RuntimeError("llama.cpp did not run fully GPU-offloaded")
        if workload["expect_llama_backend"] not in row.get("backends", ""):
            raise RuntimeError(
                f"llama.cpp did not report the {workload['expect_llama_backend']} backend")
        if row.get("no_kv_offload") is not False:
            raise RuntimeError("llama.cpp did not keep the KV cache on the GPU")
    else:
        if row.get("n_gpu_layers") != 0 or row.get("gpu_info"):
            raise RuntimeError("llama.cpp did not run CPU-only")
        if "Metal" in row.get("backends", ""):
            raise RuntimeError("llama.cpp reported the Metal backend")
        if row.get("no_kv_offload") is not True:
            raise RuntimeError("llama.cpp did not keep the KV cache on the CPU")
    if row.get("n_threads") != expected_threads or row.get("poll") != 100:
        raise RuntimeError("llama.cpp did not apply the requested thread policy")
    if row.get("cpu_strict") or row.get("cpu_mask") != "0x0":
        raise RuntimeError("llama.cpp unexpectedly applied CPU-ID affinity")
    if len(row.get("samples_ts", [])) != workload["repeats"]:
        raise RuntimeError("llama.cpp did not retain every ordered timing sample")


def run_llama(engine: dict, model: Path, workload: dict, cycle: int) -> dict:
    offload = (["-ngl", str(workload["gpu_offload_layers"])]
               if workload.get("backend") == "gpu" else ["-ngl", "0", "-nkvo", "1"])
    common = [engine["binary"], "-m", str(model), *offload, "-r",
              str(workload["repeats"]), "-o", "jsonl"]
    seq_lens = ",".join(map(str, workload["seq_lens"]))
    commands = {
        "prefill": common + ["-p", seq_lens, "-n", "0", "-t", str(workload["prefill_threads"]),
                             "--poll", "100"],
        "decode": common + ["-p", "0", "-n", str(workload["decode_n"]), "-d", seq_lens,
                            "-t", str(workload["decode_threads"]), "--poll", "100"],
    }
    phase_order = ["prefill", "decode"] if cycle % 2 == 0 else ["decode", "prefill"]
    native: dict[str, list[dict]] = {}
    diagnostics: dict[str, list[str]] = {}
    elapsed = 0.0
    for phase in phase_order:
        rows, stderr, phase_elapsed = run_command(commands[phase])
        expected_threads = (workload["prefill_threads"] if phase == "prefill"
                            else workload["decode_threads"])
        for row in rows:
            validate_llama_row(row, engine, workload, expected_threads)
        native[phase] = rows
        diagnostics[phase] = stderr
        elapsed += phase_elapsed
    if [row.get("n_prompt") for row in native["prefill"]] != workload["seq_lens"]:
        raise RuntimeError("llama.cpp emitted an unexpected prefill sweep")
    if [row.get("n_depth") for row in native["decode"]] != workload["seq_lens"]:
        raise RuntimeError("llama.cpp emitted an unexpected decode-depth sweep")
    normalized = []
    for seq_len, prefill, decode in zip(workload["seq_lens"], native["prefill"], native["decode"]):
        normalized.append({
            "seq_len": seq_len,
            "prefill_tps": prefill["samples_ts"],
            "decode_tps": decode["samples_ts"],
            "native": {"prefill": prefill, "decode": decode},
        })
    return {
        "rows": normalized,
        "stderr": diagnostics,
        "phase_order": phase_order,
        "elapsed_seconds": elapsed,
    }


def run_cooled(engine: dict, model: Path, workload: dict, cycle: int, gate: dict) -> dict:
    """One engine call per sequence length and repeat, each after the Apple
    temperature gate (host_gate.cool_each_repeat): the Mac mini M6 drops its
    performance-core clock from ~4.7 to ~3.6 GHz within 45 s of 6-thread
    load while its fan stays at minimum, so back-to-back repeats inside one
    call measured the throttling (2026-10-29: geist 1024-token prefill
    322 -> 247 tok/s over six calls, clock -22 %). The pieces merge into the
    same rows a single call returns; summary and report are unchanged."""
    limits = gate["apple_temp_gate"]
    rows = {seq_len: {"seq_len": seq_len, "prefill_tps": [], "decode_tps": [], "native": None}
            for seq_len in workload["seq_lens"]}
    cool_gates, elapsed = [], 0.0
    for _ in range(workload["repeats"]):
        for seq_len in workload["seq_lens"]:
            deadline = time.monotonic() + gate["timeout_seconds"]
            power = wait_for_apple_power(deadline) if gate.get("apple_power_gate") else {}
            cool_gates.append({**wait_for_apple_cool(limits, deadline), **power})
            piece = {**workload, "seq_lens": [seq_len], "repeats": 1}
            result = (run_geist(engine, model, piece) if engine["kind"] == "geist"
                      else run_llama(engine, model, piece, cycle))
            got = result["rows"][0]
            row = rows[seq_len]
            row["prefill_tps"] += got["prefill_tps"]
            row["decode_tps"] += got["decode_tps"]
            row["native"] = row["native"] or got["native"]
            elapsed += result["elapsed_seconds"]
    return {"rows": [rows[seq_len] for seq_len in workload["seq_lens"]],
            "elapsed_seconds": elapsed, "cool_gates": cool_gates}


def summarize(runs: list[dict], engines: list[dict], workload: dict) -> dict:
    summary = {}
    for engine in engines:
        selected = [run for run in runs if run["engine"] == engine["label"]]
        by_seq = {}
        for seq_len in workload["seq_lens"]:
            metrics = {}
            for phase in ("prefill", "decode"):
                samples = []
                for run in selected:
                    row = next(item for item in run["rows"] if item["seq_len"] == seq_len)
                    samples.extend(row[f"{phase}_tps"])
                median = statistics.median(samples)
                mad = median_absolute_deviation(samples)
                metrics[phase] = {
                    "median_tps": median,
                    "mad_tps": mad,
                    "mad_pct": 100.0 * mad / median,
                    "samples": len(samples),
                }
            by_seq[str(seq_len)] = metrics
        summary[engine["label"]] = by_seq
    geist = summary["geist"]
    llama = summary["llama.cpp"]
    for seq_len in map(str, workload["seq_lens"]):
        for phase in ("prefill", "decode"):
            llama_tps = llama[seq_len][phase]["median_tps"]
            geist[seq_len][phase]["vs_llama_pct"] = 100.0 * (
                geist[seq_len][phase]["median_tps"] / llama_tps - 1.0
            )
    return summary


def render_report(metadata: dict, summary: dict, gpu_device: str | None = None) -> str:
    workload = metadata["protocol"]["workload"]
    profile = metadata["protocol"]["host_profiles"][metadata["host_profile"]]
    gpu = workload.get("backend") == "gpu"
    mode = (f"Full GPU offload on {gpu_device or 'an unidentified device'}"
            if gpu else "CPU-only")
    lines = [
        f"# Current {'GPU' if gpu else 'CPU'} head-to-head: geist vs llama.cpp",
        "",
        f"Model: `{metadata['model']['file']}` (`{metadata['model']['sha256']}`)",
        "",
        f"geist `{metadata['engines'][0]['commit'][:12]}`; "
        f"llama.cpp `{metadata['engines'][1]['commit'][:12]}`.",
        "",
        f"Host profile `{metadata['host_profile']}`: "
        f"{profile['prefill_threads']}-thread prefill, {profile['decode_threads']}-thread decode, "
        f"geist on `{profile['expect_backend']}`, identical counts for both engines.",
        "",
        f"{workload['cycles']} alternating A/B cycles, {workload['repeats']} raw samples per cell "
        f"and cycle; median ± MAD. {mode}, same model bytes, prompt/depth shapes and thread "
        f"counts. Model loading is excluded.",
        "",
        f"Host: {metadata['system']['cpu']} "
        f"({metadata['system']['performance_cores']} physical cores), "
        f"{metadata['system']['os']}.",
        "",
        "| Seq/depth | geist pp tok/s | llama.cpp pp tok/s | geist vs llama | geist tg tok/s | llama.cpp tg tok/s | geist vs llama |",
        "| --: | --: | --: | --: | --: | --: | --: |",
    ]
    for seq_len in metadata["protocol"]["workload"]["seq_lens"]:
        key = str(seq_len)
        gp = summary["geist"][key]["prefill"]
        lp = summary["llama.cpp"][key]["prefill"]
        gd = summary["geist"][key]["decode"]
        ld = summary["llama.cpp"][key]["decode"]
        lines.append(
            f"| {seq_len} | {gp['median_tps']:.2f} ± {gp['mad_tps']:.2f} | "
            f"{lp['median_tps']:.2f} ± {lp['mad_tps']:.2f} | {gp['vs_llama_pct']:+.2f}% | "
            f"{gd['median_tps']:.2f} ± {gd['mad_tps']:.2f} | "
            f"{ld['median_tps']:.2f} ± {ld['mad_tps']:.2f} | {gd['vs_llama_pct']:+.2f}% |"
        )
    lines += [
        "",
        *(["Every run started on AC power with Low Power Mode off (power gate, pmset); "
           "each run's machine state records it.", ""]
          if metadata.get("host_gate", {}).get("apple_power_gate") else []),
        "Token streams are engine-native synthetic inputs; this is compute-shape parity, not token/logit parity. "
        "Publish speedups only together with the repository's separate quality-parity gate.",
    ]
    return "\n".join(lines) + "\n"


def load_resume(path: Path, current: dict, engines: list[dict]) -> tuple[dict, list[dict]]:
    saved, runs = load_partial(path)
    for field in ("schema", "protocol", "protocol_sha256", "runner", "model", "engines",
                  "system", "host_profile"):
        if saved.get(field) != current.get(field):
            raise ValueError(f"resume {field} does not match this invocation")
    check_schedule_prefix(runs, build_schedule(engines, current["protocol"]["workload"]["cycles"]),
                          "engine")
    return saved, runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geist", required=True, type=Path)
    parser.add_argument("--llama", required=True, type=Path)
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path.home() / "bench-geistlib" / "head-to-head")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH,
                        help="protocol JSON; its workload.backend field decides the "
                             "cpu-only vs gpu contract (default: the CPU protocol)")
    parser.add_argument("--host-profile", required=True,
                        help="key in the protocol's host_profiles: fixes the thread policy "
                             "and the geist backend this host is expected to select")
    parser.add_argument("--model",
                        help="key in the protocol's models map (required for v2 protocols); "
                             "binds the campaign to that entry's SHA-256")
    parser.add_argument("--resume", type=Path,
                        help="continue an exact, validated .jsonl.partial schedule prefix")
    args = parser.parse_args()
    paths = [args.geist.expanduser().resolve(), args.llama.expanduser().resolve(), args.gguf.expanduser().resolve()]
    if any(not path.is_file() for path in paths):
        parser.error("--geist, --llama and --gguf must name existing files")
    geist_path, llama_path, model = paths
    protocol_path = args.protocol.expanduser().resolve()
    protocol = json.loads(protocol_path.read_text())
    if args.host_profile not in protocol["host_profiles"]:
        parser.error(f"--host-profile must be one of: {', '.join(sorted(protocol['host_profiles']))}")
    if "models" in protocol:
        if args.model not in protocol["models"]:
            parser.error(f"--model must be one of: {', '.join(sorted(protocol['models']))}")
        expected_model = protocol["models"][args.model]
    else:
        # v1 protocols carry exactly one model and no key.
        expected_model = protocol["model"]
    model_sha = sha256_file(model)
    if model_sha != expected_model["sha256"]:
        parser.error(f"model SHA-256 mismatch: {model_sha}")
    engines = [engine_metadata("geist", "geist", geist_path),
               engine_metadata("llama.cpp", "llama", llama_path)]
    metadata = {
        "schema": "geist.benchmark.cross-engine-run.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "protocol_sha256": sha256_file(protocol_path),
        "runner": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": sha256_file(Path(__file__).resolve()),
            "commit": command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        },
        "model": {"file": model.name, "path": str(model), "sha256": model_sha,
                  **({"key": args.model} if "models" in protocol else {})},
        "engines": engines,
        "system": system_metadata(),
    }
    # The workload is frozen across hosts; the profile contributes only the
    # thread policy, applied identically to both engines, and the backend name
    # geist must report. A host comparing itself against another host compares
    # two different thread policies — that is why the profile key is recorded.
    profile = protocol["host_profiles"][args.host_profile]
    workload = {**protocol["workload"], **profile, "host_profile": args.host_profile}
    metadata["host_profile"] = args.host_profile
    # A profile may refine the gate for its host (the Apple profiles: idle
    # CPU instead of load average, and a per-run spread limit).
    gate = {**protocol["host_gate"], **profile.get("host_gate", {})}
    metadata["host_gate"] = gate
    schedule = build_schedule(engines, workload["cycles"])
    if args.resume is not None:
        partial = args.resume.expanduser().resolve()
        if partial.suffix != ".partial" or not partial.is_file():
            parser.error("--resume must name an existing .jsonl.partial artifact")
        try:
            metadata, runs = load_resume(partial, metadata, engines)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))
        open_mode = "a"
    else:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
        mode_tag = "gpu" if workload.get("backend") == "gpu" else "cpu"
        key_tag = f"_{args.model}" if "models" in protocol else ""
        partial = args.out_dir / f"{timestamp}_geist_llama_{mode_tag}{key_tag}.jsonl.partial"
        runs = []
        open_mode = "w"
    completed = partial.with_suffix("")
    report_path = partial.with_name(partial.name.removesuffix(".jsonl.partial") + ".md")
    with partial.open(open_mode) as stream:
        if open_mode == "w":
            stream.write(json.dumps({"kind": "metadata", **metadata}, sort_keys=True) + "\n")
        else:
            stream.write(json.dumps({
                "kind": "resume",
                "resumed_utc": datetime.now(timezone.utc).isoformat(),
                "runner_commit": command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
            }, sort_keys=True) + "\n")
        stream.flush()
        for cycle, position, engine in schedule[len(runs):]:
            print(f"cycle {cycle + 1}/{workload['cycles']}: {engine['label']}", file=sys.stderr, flush=True)
            discarded = []
            try:
                while True:
                    before = wait_for_quiet(gate)
                    if gate.get("cool_each_repeat"):
                        result = run_cooled(engine, model, workload, cycle, gate)
                    else:
                        result = (run_geist(engine, model, workload) if engine["kind"] == "geist"
                                  else run_llama(engine, model, workload, cycle))
                    limit = gate.get("max_repeat_spread_pct")
                    spread = repeat_spread_pct(result)
                    if limit is None or spread <= limit:
                        break
                    discarded.append(round(spread, 2))
                    print(f"  repeat spread {spread:.1f} % > {limit} %: host disturbed, measuring again",
                          file=sys.stderr, flush=True)
                    if len(discarded) > gate.get("max_retries", 3):
                        raise RuntimeError(f"repeat spread stayed above {limit} %: {discarded}")
            except (RuntimeError, TimeoutError) as error:
                print(f"ERROR: {error}", file=sys.stderr)
                return ERROR
            run = {"kind": "run", "cycle": cycle, "position": position,
                   "engine": engine["label"], "machine_before": before,
                   "machine_after": machine_state(), **result,
                   **({"discarded_spread_pct": discarded} if discarded else {})}
            runs.append(run)
            stream.write(json.dumps(run, sort_keys=True) + "\n")
            stream.flush()
        summary = summarize(runs, engines, workload)
        stream.write(json.dumps({"kind": "summary", "summary": summary}, sort_keys=True) + "\n")
    partial.replace(completed)
    gpu_device = next(
        (run["rows"][0]["native"]["prefill"]["gpu_info"]
         for run in runs if run["engine"] == "llama.cpp"), None)
    report = render_report(metadata, summary, gpu_device)
    report_path.write_text(report)
    print(report)
    print(f"raw: {completed}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

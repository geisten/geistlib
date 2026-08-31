#!/usr/bin/env python3
"""Run a paired, CPU-only geist versus llama.cpp throughput benchmark.

The two engines use the same model bytes, workload shapes, thread counts,
sample count, host gate, and aggregation. Each engine retains its native
synthetic token stream; token identity is therefore not a parity claim.
Correctness/quality remains a separate prerequisite for publishing a speedup.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "benchmark" / "cross_engine_cpu_protocol.json"
ERROR = 99


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str], timeout: int = 15) -> str:
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return proc.stdout.strip() if proc.returncode == 0 and proc.stdout.strip() else "unavailable"


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
        "linked_libraries": command_output(["otool", "-L", str(binary)]).splitlines(),
    }


def machine_state() -> dict:
    load = os.getloadavg()[0]
    cores = max(os.cpu_count() or 1, 1)
    return {
        "load_1m": load,
        "load_per_core": load / cores,
        "thermal_status": command_output(["pmset", "-g", "therm"]).splitlines(),
        "power_source": command_output(["pmset", "-g", "batt"]).splitlines(),
        "temperature_c": None,
        "temperature_note": "unavailable to an unprivileged process",
    }


def wait_for_quiet(gate: dict) -> dict:
    deadline = time.monotonic() + gate["timeout_seconds"]
    quiet_since: float | None = None
    last_report = float("-inf")
    while True:
        state = machine_state()
        now = time.monotonic()
        if state["load_per_core"] <= gate["max_load_per_core"]:
            quiet_since = quiet_since or now
            if now - quiet_since >= gate["settle_seconds"]:
                print(f"  cooldown {gate['cooldown_seconds']}s", file=sys.stderr, flush=True)
                time.sleep(gate["cooldown_seconds"])
                state = machine_state()
                if state["load_per_core"] <= gate["max_load_per_core"]:
                    return state
                quiet_since = None
        else:
            quiet_since = None
        if now >= deadline:
            raise TimeoutError(
                f"host did not become quiet: load/core={state['load_per_core']:.3f}, "
                f"limit={gate['max_load_per_core']:.3f}"
            )
        if now - last_report >= 60:
            print(
                f"  waiting for quiet host: load {state['load_1m']:.2f} "
                f"({state['load_per_core']:.3f}/core)",
                file=sys.stderr,
                flush=True,
            )
            last_report = now
        time.sleep(10)


def run_command(command: list[str], env: dict[str, str] | None = None) -> tuple[list[dict], list[str], float]:
    started = time.monotonic()
    proc = subprocess.run(command, capture_output=True, text=True, env=env)
    elapsed = time.monotonic() - started
    if proc.returncode != 0:
        diagnostic = (proc.stderr.splitlines() or proc.stdout.splitlines() or ["no diagnostics"])[-1]
        raise RuntimeError(f"command exited {proc.returncode}: {diagnostic}")
    rows = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
    return rows, proc.stderr.splitlines(), elapsed


def run_geist(engine: dict, model: Path, workload: dict) -> dict:
    seq_lens = ",".join(map(str, workload["seq_lens"]))
    command = [
        engine["binary"], "--gguf", str(model), "--seq-lens", seq_lens,
        "--decode-n", str(workload["decode_n"]), "--warmup", "64",
        "--repeats", str(workload["repeats"]), "--threads",
        str(workload["prefill_threads"]), "--emit-jsonl",
    ]
    env = dict(os.environ)
    env.update({
        "OMP_WAIT_POLICY": "active",
        "OMP_NUM_THREADS": str(workload["prefill_threads"]),
        "GEIST_PREFILL_THREADS": str(workload["prefill_threads"]),
        "GEIST_DECODE_THREADS": str(workload["decode_threads"]),
        "GEIST_TEXT_ONLY": "1",
    })
    rows, stderr, elapsed = run_command(command, env)
    if not any(line.strip() == "[bench] backend: cpu_neon" for line in stderr):
        raise RuntimeError("geist did not select the cpu_neon backend")
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
    common = [engine["binary"], "-m", str(model), "-ngl", "0", "-nkvo", "1", "-r",
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


def median_absolute_deviation(values: list[float]) -> float:
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)


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


def render_report(metadata: dict, summary: dict) -> str:
    lines = [
        "# Current CPU head-to-head: geist vs llama.cpp",
        "",
        f"Model: `{metadata['model']['file']}` (`{metadata['model']['sha256']}`)",
        "",
        f"geist `{metadata['engines'][0]['commit'][:12]}`; "
        f"llama.cpp `{metadata['engines'][1]['commit'][:12]}`.",
        "",
        "Four alternating A/B cycles, three raw samples per cell and cycle; median ± MAD. "
        "CPU-only, same model bytes, prompt/depth shapes and thread counts. Model loading is excluded.",
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
        "Token streams are engine-native synthetic inputs; this is compute-shape parity, not token/logit parity. "
        "Publish speedups only together with the repository's separate quality-parity gate.",
    ]
    return "\n".join(lines) + "\n"


def build_schedule(engines: list[dict], cycles: int) -> list[tuple[int, int, dict]]:
    schedule = []
    for cycle in range(cycles):
        order = engines if cycle % 2 == 0 else list(reversed(engines))
        schedule.extend((cycle, position, engine) for position, engine in enumerate(order))
    return schedule


def load_resume(path: Path, current: dict, engines: list[dict]) -> tuple[dict, list[dict]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records or records[0].get("kind") != "metadata":
        raise ValueError("resume artifact does not start with metadata")
    if any(record.get("kind") == "summary" for record in records):
        raise ValueError("resume artifact is already complete")
    if any(record.get("kind") not in {"metadata", "run", "resume"} for record in records):
        raise ValueError("resume artifact contains an unknown record kind")
    saved = {key: value for key, value in records[0].items() if key != "kind"}
    for field in ("schema", "protocol", "protocol_sha256", "runner", "model", "engines", "system"):
        if saved.get(field) != current.get(field):
            raise ValueError(f"resume {field} does not match this invocation")
    runs = [record for record in records[1:] if record.get("kind") == "run"]
    schedule = build_schedule(engines, current["protocol"]["workload"]["cycles"])
    if len(runs) > len(schedule):
        raise ValueError("resume artifact has more runs than the protocol")
    for run, (cycle, position, engine) in zip(runs, schedule):
        if (run.get("cycle"), run.get("position"), run.get("engine")) != (
                cycle, position, engine["label"]):
            raise ValueError("resume runs are not an exact schedule prefix")
    return saved, runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geist", required=True, type=Path)
    parser.add_argument("--llama", required=True, type=Path)
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path.home() / "bench-geistlib" / "head-to-head")
    parser.add_argument("--resume", type=Path,
                        help="continue an exact, validated .jsonl.partial schedule prefix")
    args = parser.parse_args()
    paths = [args.geist.expanduser().resolve(), args.llama.expanduser().resolve(), args.gguf.expanduser().resolve()]
    if any(not path.is_file() for path in paths):
        parser.error("--geist, --llama and --gguf must name existing files")
    geist_path, llama_path, model = paths
    protocol = json.loads(PROTOCOL_PATH.read_text())
    model_sha = sha256_file(model)
    if model_sha != protocol["model"]["sha256"]:
        parser.error(f"model SHA-256 mismatch: {model_sha}")
    engines = [engine_metadata("geist", "geist", geist_path),
               engine_metadata("llama.cpp", "llama", llama_path)]
    metadata = {
        "schema": "geist.benchmark.cross-engine-run.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "runner": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": sha256_file(Path(__file__).resolve()),
            "commit": command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        },
        "model": {"file": model.name, "path": str(model), "sha256": model_sha},
        "engines": engines,
        "system": {
            "cpu": command_output(["sysctl", "-n", "machdep.cpu.brand_string"]),
            "logical_cores": os.cpu_count(),
            "performance_cores": command_output(["sysctl", "-n", "hw.perflevel0.physicalcpu"]),
            "memory_bytes": command_output(["sysctl", "-n", "hw.memsize"]),
            "os": f"{platform.system()} {platform.release()}",
            "product": command_output(["sw_vers"]),
            "compiler": command_output(["clang", "--version"]).splitlines()[0],
        },
    }
    workload = protocol["workload"]
    gate = protocol["host_gate"]
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
        partial = args.out_dir / f"{timestamp}_geist_llama_cpu.jsonl.partial"
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
            try:
                before = wait_for_quiet(gate)
                result = (run_geist(engine, model, workload) if engine["kind"] == "geist"
                          else run_llama(engine, model, workload, cycle))
            except (RuntimeError, TimeoutError) as error:
                print(f"ERROR: {error}", file=sys.stderr)
                return ERROR
            run = {"kind": "run", "cycle": cycle, "position": position,
                   "engine": engine["label"], "machine_before": before,
                   "machine_after": machine_state(), **result}
            runs.append(run)
            stream.write(json.dumps(run, sort_keys=True) + "\n")
            stream.flush()
        summary = summarize(runs, engines, workload)
        stream.write(json.dumps({"kind": "summary", "summary": summary}, sort_keys=True) + "\n")
    partial.replace(completed)
    report = render_report(metadata, summary)
    report_path.write_text(report)
    print(report)
    print(f"raw: {completed}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

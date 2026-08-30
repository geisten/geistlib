#!/usr/bin/env python3
"""Interleaved, provenance-complete Apple CPU benchmark.

Each cycle rotates variant order (ABC, BCA, CAB), waits for the host to be
quiet before every run, and preserves every sample emitted by
``bench_perf_sweep``. The protocol lives in benchmark/apple_cpu_protocol.json;
changing it is a baseline migration, not a command-line convenience.
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
PROTOCOL_PATH = ROOT / "benchmark" / "apple_cpu_protocol.json"
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
    if proc.returncode != 0:
        return "unavailable"
    return proc.stdout.strip() or "unavailable"


def parse_variants(values: list[str]) -> list[dict]:
    variants = []
    labels = set()
    for value in values:
        label, separator, raw_path = value.partition("=")
        path = Path(raw_path).expanduser().resolve()
        if not separator or not label or label in labels or not path.is_file():
            raise ValueError(f"invalid --variant {value!r}; expected unique LABEL=/path/to/bench")
        labels.add(label)
        checkout_commit = command_output(["git", "-C", str(path.parent), "rev-parse", "HEAD"])
        subject = command_output(["git", "-C", str(path.parent), "show", "-s", "--format=%s", "HEAD"])
        runtime_commit = checkout_commit
        if subject == "bench: retain ordered sweep samples in JSONL":
            runtime_commit = command_output(
                ["git", "-C", str(path.parent), "rev-parse", "HEAD^"]
            )
        variants.append({
            "label": label,
            "binary": str(path),
            "binary_sha256": sha256_file(path),
            "checkout_commit": checkout_commit,
            "runtime_commit": runtime_commit,
            "linked_libraries": command_output(["otool", "-L", str(path)]).splitlines(),
        })
    if len(variants) < 2:
        raise ValueError("at least two --variant arguments are required")
    return variants


def load_per_core() -> tuple[float, float]:
    load = os.getloadavg()[0]
    return load, load / max(os.cpu_count() or 1, 1)


def machine_state() -> dict:
    load, per_core = load_per_core()
    return {
        "load_1m": load,
        "load_per_core": per_core,
        "temperature_c": None,
        "temperature_note": "package temperature unavailable without privilege",
        "thermal_status": command_output(["pmset", "-g", "therm"]).splitlines(),
        "power_source": command_output(["pmset", "-g", "batt"]).splitlines(),
    }


def wait_for_quiet(max_per_core: float, settle_seconds: int,
                   cooldown_seconds: int, timeout_seconds: int) -> dict:
    """Require a continuously quiet interval, cool down, then re-check.

    macOS exposes neither package temperature nor thermal pressure to an
    unprivileged process on all releases. Load is therefore the enforceable
    gate; the raw record explicitly marks temperature as unavailable.
    """
    deadline = time.monotonic() + timeout_seconds
    quiet_since = None
    while True:
        load, per_core = load_per_core()
        now = time.monotonic()
        if per_core <= max_per_core:
            quiet_since = quiet_since or now
            if now - quiet_since >= settle_seconds:
                if cooldown_seconds > 0:
                    print(f"  cooldown {cooldown_seconds}s", file=sys.stderr, flush=True)
                    time.sleep(cooldown_seconds)
                load, per_core = load_per_core()
                if per_core <= max_per_core:
                    return machine_state()
                quiet_since = None
        else:
            quiet_since = None
        if now >= deadline:
            raise TimeoutError(
                f"host did not become quiet: load={load:.2f}, per_core={per_core:.2f}, "
                f"limit={max_per_core:.2f}"
            )
        print(f"  waiting for quiet host: load {load:.2f} ({per_core:.2f}/core)",
              file=sys.stderr, flush=True)
        time.sleep(10)


def run_variant(variant: dict, gguf: Path, protocol: dict, threads: int | None) -> dict:
    seq_lens = ",".join(str(value) for value in protocol["seq_lens"])
    command = [
        variant["binary"],
        "--gguf", str(gguf),
        "--seq-lens", seq_lens,
        "--decode-n", str(protocol["decode_n"]),
        "--warmup", str(protocol["warmup"]),
        "--repeats", str(protocol["repeats"]),
        "--emit-jsonl",
    ]
    env = dict(os.environ)
    env.setdefault("OMP_WAIT_POLICY", "active")
    if threads is not None:
        env["OMP_NUM_THREADS"] = str(threads)
        command += ["--threads", str(threads)]

    before = time.monotonic()
    proc = subprocess.run(command, capture_output=True, text=True, env=env)
    elapsed = time.monotonic() - before
    if proc.returncode != 0:
        raise RuntimeError(
            f"{variant['label']} exited {proc.returncode}: "
            f"{(proc.stderr.splitlines() or ['no diagnostics'])[-1]}"
        )
    rows = []
    for line in proc.stdout.splitlines():
        if line.startswith("{"):
            rows.append(json.loads(line))
    expected = protocol["seq_lens"]
    if [row.get("seq_len") for row in rows] != expected:
        raise RuntimeError(f"{variant['label']} emitted unexpected sequence rows")
    for row in rows:
        samples = row.get("samples", {})
        for phase in ("prefill_ms", "decode_ms", "total_ms"):
            if len(samples.get(phase, [])) != protocol["repeats"]:
                raise RuntimeError(
                    f"{variant['label']} did not retain {protocol['repeats']} raw {phase} samples"
                )
    return {
        "variant": variant["label"],
        "elapsed_seconds": elapsed,
        "rows": rows,
        "stderr": proc.stderr.splitlines(),
        "machine_after": machine_state(),
    }


def median_absolute_deviation(values: list[float]) -> float:
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)


def summarize(runs: list[dict], variants: list[dict], protocol: dict,
              baseline: str) -> dict:
    summary: dict[str, dict[str, dict]] = {}
    for variant in variants:
        label = variant["label"]
        by_seq = {}
        selected = [run for run in runs if run["variant"] == label]
        for seq_len in protocol["seq_lens"]:
            phase_summary = {}
            for phase, tokens in (("prefill", seq_len), ("decode", protocol["decode_n"])):
                milliseconds = []
                for run in selected:
                    row = next(row for row in run["rows"] if row["seq_len"] == seq_len)
                    milliseconds.extend(row["samples"][f"{phase}_ms"])
                throughputs = [tokens * 1000.0 / value for value in milliseconds if value > 0]
                median = statistics.median(throughputs) if throughputs else 0.0
                mad = median_absolute_deviation(throughputs) if throughputs else 0.0
                phase_summary[phase] = {
                    "median_tps": median,
                    "mad_tps": mad,
                    "dispersion_pct": 100.0 * mad / median if median else 0.0,
                    "samples": len(throughputs),
                }
            by_seq[str(seq_len)] = phase_summary
        summary[label] = by_seq

    reference = summary[baseline]
    for label, by_seq in summary.items():
        for seq_len, phases in by_seq.items():
            for phase, metrics in phases.items():
                base = reference[seq_len][phase]["median_tps"]
                metrics["vs_baseline_pct"] = 100.0 * (metrics["median_tps"] / base - 1.0) if base else 0.0
    return summary


def render_report(metadata: dict, summary: dict, baseline: str) -> str:
    protocol = metadata["protocol"]
    lines = [
        "# Apple CPU interleaved A/B benchmark",
        "",
        f"Model: `{metadata['model']['file']}` (`{metadata['model']['sha256']}`)",
        "",
        f"Protocol: {protocol['cycles']} interleaved cycles, sequences "
        f"{','.join(map(str, protocol['seq_lens']))}, decode {protocol['decode_n']}, "
        f"warmup {protocol['warmup']}, {protocol['repeats']} ordered samples/run; "
        "median and median absolute deviation.",
        "",
        f"Baseline: `{baseline}`",
        "",
        "| Variant | Seq | Prefill median tok/s | MAD | vs base | Decode median tok/s | MAD | vs base |",
        "| :-- | --: | --: | --: | --: | --: | --: | --: |",
    ]
    for label, by_seq in summary.items():
        for seq_len, phases in by_seq.items():
            prefill = phases["prefill"]
            decode = phases["decode"]
            lines.append(
                f"| {label} | {seq_len} | {prefill['median_tps']:.2f} | "
                f"{prefill['dispersion_pct']:.2f}% | {prefill['vs_baseline_pct']:+.2f}% | "
                f"{decode['median_tps']:.2f} | {decode['dispersion_pct']:.2f}% | "
                f"{decode['vs_baseline_pct']:+.2f}% |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", action="append", required=True,
                        help="LABEL=/absolute/path/to/bench_perf_sweep; repeat per revision")
    parser.add_argument("--baseline", help="baseline variant label (default: first variant)")
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--out-dir", type=Path,
                        default=Path.home() / "bench-geistlib" / "apple-ab")
    parser.add_argument("--quiet-timeout", type=int, default=3600,
                        help="seconds to wait for an uncontended host before failing")
    args = parser.parse_args()

    try:
        variants = parse_variants(args.variant)
    except ValueError as error:
        parser.error(str(error))
    baseline = args.baseline or variants[0]["label"]
    if baseline not in {variant["label"] for variant in variants}:
        parser.error("--baseline must name one of the variants")
    gguf = args.gguf.expanduser().resolve()
    if not gguf.is_file():
        parser.error(f"model not found: {gguf}")

    protocol_file = json.loads(PROTOCOL_PATH.read_text())
    protocol = protocol_file["interleaved_ab"]
    model_sha = sha256_file(gguf)
    expected_sha = protocol_file["model"]["sha256"]
    if model_sha != expected_sha:
        parser.error(f"model SHA-256 mismatch: expected {expected_sha}, got {model_sha}")

    metadata = {
        "schema": "geist.benchmark.apple-ab.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": {"file": gguf.name, "path": str(gguf), "sha256": model_sha},
        "protocol": protocol,
        "variants": variants,
        "system": {
            "cpu": command_output(["sysctl", "-n", "machdep.cpu.brand_string"]),
            "cores": os.cpu_count(),
            "memory_bytes": command_output(["sysctl", "-n", "hw.memsize"]),
            "os": f"{platform.system()} {platform.release()}",
            "product": command_output(["sw_vers"]),
            "power": command_output(["pmset", "-g", "custom"]),
            "compiler": command_output(["clang", "--version"]).splitlines()[0],
        },
        "environment": {
            "OMP_WAIT_POLICY": os.environ.get("OMP_WAIT_POLICY", "active"),
            "OMP_NUM_THREADS": str(args.threads or "default"),
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    raw_path = args.out_dir / f"{timestamp}_apple_cpu_ab.jsonl"
    partial_path = raw_path.with_suffix(".jsonl.partial")
    report_path = args.out_dir / f"{timestamp}_apple_cpu_ab.md"
    runs = []
    with partial_path.open("w") as raw:
        raw.write(json.dumps({"kind": "metadata", **metadata}, sort_keys=True) + "\n")
        for cycle in range(protocol["cycles"]):
            offset = cycle % len(variants)
            order = variants[offset:] + variants[:offset]
            for position, variant in enumerate(order):
                print(f"cycle {cycle + 1}/{protocol['cycles']}: {variant['label']}",
                      file=sys.stderr, flush=True)
                try:
                    machine = wait_for_quiet(
                        protocol["max_load_per_core"],
                        protocol["settle_seconds"],
                        protocol["cooldown_seconds"],
                        args.quiet_timeout,
                    )
                    run = run_variant(variant, gguf, protocol, args.threads)
                except (RuntimeError, TimeoutError) as error:
                    print(f"ERROR: {error}", file=sys.stderr)
                    return ERROR
                run.update({"kind": "run", "cycle": cycle, "position": position,
                            "machine_before": machine})
                runs.append(run)
                raw.write(json.dumps(run, sort_keys=True) + "\n")
                raw.flush()

        summary = summarize(runs, variants, protocol, baseline)
        raw.write(json.dumps({"kind": "summary", "summary": summary}, sort_keys=True) + "\n")
    partial_path.replace(raw_path)

    report = render_report(metadata, summary, baseline)
    report_path.write_text(report)
    print(report)
    print(f"raw: {raw_path}")
    print(f"report: {report_path}")

    guard = protocol["regression_guard_pct"]
    regressions = []
    for label, by_seq in summary.items():
        if label == baseline:
            continue
        for seq_len, phases in by_seq.items():
            for phase, metrics in phases.items():
                if metrics["vs_baseline_pct"] < -guard:
                    regressions.append(
                        f"{label} {phase}@{seq_len}: {metrics['vs_baseline_pct']:.2f}%"
                    )
    if regressions:
        print(f"regression guard failed (limit -{guard:.1f}%): " + "; ".join(regressions),
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

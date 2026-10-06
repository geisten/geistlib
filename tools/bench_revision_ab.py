#!/usr/bin/env python3
"""End-to-end A/B of git revisions, each built from scratch.

An incremental build appends the objects it rebuilds to the archive, so a
parent built in the tree and a commit built on top of it differ in code
layout as well as in the change. On x86-64 the layout alone moved single
stages of the prefill profile by 2-3 %, and end-to-end gains measured that
way came out several times too large (-7.4 % claimed, -1.8 % between clean
builds). This tool builds
every revision in a git worktree of its own, then runs bench_perf_sweep on
the binaries interleaved, the order rotating each cycle as in
tools/bench_mac_ab.py.

A copy of the baseline binary runs alongside as a control. The same bytes
measured twice show how far the host moves on its own, which one A/B pair
cannot tell apart from a change. Each cycle runs every variant back to back,
so a variant's time over the baseline's in the same cycle cancels what the
host does between cycles. Every row reports the median of those per-cycle
ratios, a 95 % interval for it and the number of cycles the variant was
faster. The interval is distribution-free (order statistics, the sign
test's): at least 95 % coverage for any noise, where a bootstrap over this
few cycles is too narrow and called the same binary 2.7 % faster. An
interval that includes zero is noise.

Example, on the synthetic model of `make bench-synth`:

  make gguf_artifacts/synth/llama32-1b-q4_k.gguf
  tools/bench_revision_ab.py --rev base=main --rev new=HEAD \\
      --gguf gguf_artifacts/synth/llama32-1b-q4_k.gguf --seq-lens 512,2048

The raw runs (every repeat of every run) and the report go to a timestamped
directory under --out-dir.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = "bench_perf_sweep"
METRICS = ("prefill_ms", "decode_ms", "ttft_ms")
CONTROL = "control"
COVERAGE = 0.95
# Six cycles all going one way happen by chance 2 / 2^6 = 3 % of the time;
# five do 6 %, so fewer cycles cannot carry a 95 % interval.
MIN_CYCLES = 6
# A phase the run did not time reads 0.00 ms (decode_n 0, or a one-token
# decode whose forward is deferred); a ratio of such values means nothing.
MIN_MS = 0.05

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_cross_engine import command_output, sha256_file  # noqa: E402
from bench_mac_ab import build_schedule  # noqa: E402


def parse_revs(values: list[str]) -> list[tuple[str, str]]:
    """LABEL=REF pairs; labels unique, and not the control's."""
    revs: list[tuple[str, str]] = []
    for value in values:
        label, separator, ref = value.partition("=")
        if (not separator or not label or not ref or label == CONTROL
                or label in {seen for seen, _ in revs}):
            raise ValueError(f"invalid --rev {value!r}; expected a unique LABEL=REF "
                             f"(label {CONTROL!r} is reserved)")
        revs.append((label, ref))
    return revs


def bench_relpath(make_args: list[str], target: str) -> Path:
    """Where the Makefile puts the bench: bin/$(TARGET)/$(MODE)/tests."""
    overrides = dict(arg.split("=", 1) for arg in make_args if "=" in arg)
    return (Path("bin") / overrides.get("TARGET", target) / overrides.get("MODE", "release")
            / "tests" / BENCH)


def resolve(ref: str) -> str:
    sha = command_output(["git", "-C", str(ROOT), "rev-parse", "--verify", f"{ref}^{{commit}}"])
    if sha == "unavailable":
        raise ValueError(f"not a commit: {ref!r}")
    return sha


def build(sha: str, work: Path, make_args: list[str], jobs: int) -> Path:
    """Build `sha` in a fresh worktree; return a copy of its bench binary."""
    tree = work / f"tree_{sha[:12]}"
    log = work / f"build_{sha[:12]}.log"
    added = subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "--detach", str(tree), sha],
                           capture_output=True, text=True)
    if added.returncode != 0:
        raise RuntimeError(f"git worktree add {sha[:12]} failed: {added.stderr.strip()}")
    try:
        target = command_output(["sh", str(tree / "mk" / "detect-target.sh")])
        relpath = bench_relpath(make_args, target)
        with log.open("w") as out:
            status = subprocess.run(
                ["make", "-C", str(tree), *make_args, f"-j{jobs}", "lib", str(relpath)],
                stdout=out, stderr=subprocess.STDOUT).returncode
        if status != 0 or not (tree / relpath).is_file():
            raise RuntimeError(f"build of {sha[:12]} failed, see {log}")
        binary = work / f"{BENCH}_{sha[:12]}"
        shutil.copy2(tree / relpath, binary)
        return binary
    finally:
        subprocess.run(["git", "-C", str(ROOT), "worktree", "remove", "--force", str(tree)],
                       capture_output=True)


def run_variant(binary: Path, args: argparse.Namespace, env: dict[str, str]) -> list[dict]:
    command = [str(binary), "--gguf", str(args.gguf), "--seq-lens", args.seq_lens,
               "--decode-n", str(args.decode_n), "--repeats", str(args.repeats),
               "--warmup", str(args.warmup), "--emit-jsonl"]
    proc = subprocess.run(command, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        last = (proc.stderr.splitlines() or ["no diagnostics"])[-1]
        raise RuntimeError(f"{binary.name} exited {proc.returncode}: {last}")
    rows = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
    if [row.get("seq_len") for row in rows] != [int(s) for s in args.seq_lens.split(",")]:
        raise RuntimeError(f"{binary.name} emitted unexpected sequence rows")
    return rows


def run_value(row: dict, metric: str) -> float | None:
    """A run's value: the median of its repeats where they are kept."""
    samples = row.get("samples", {}).get(metric)
    return statistics.median(samples) if samples else row.get(metric)


def cycle_table(runs: list[dict], metric: str) -> dict[int, dict[int, dict[str, float]]]:
    """seq_len -> cycle -> variant -> value."""
    table: dict[int, dict[int, dict[str, float]]] = {}
    for run in runs:
        for row in run["rows"]:
            value = run_value(row, metric)
            if value is not None:
                table.setdefault(row["seq_len"], {}).setdefault(run["cycle"], {})[
                    run["variant"]] = value
    return table


def interval_rank(n: int) -> int | None:
    """The largest k for which the k-th smallest and k-th largest of n
    per-cycle ratios bound their median with at least 95 % coverage,
    1 - 2 P(Binomial(n, 1/2) < k); None when not even the extremes do."""
    below = 0  # P(Binomial(n, 1/2) < k), times 2^n
    rank = None
    for k in range(1, n // 2 + 1):
        below += math.comb(n, k - 1)
        if 1.0 - 2.0 * below / 2**n < COVERAGE:
            break
        rank = k
    return rank


def paired_change(pairs: list[tuple[float, float]]) -> dict:
    """Each cycle's variant time over its baseline time: their median, a
    distribution-free 95 % interval for it, and the cycles the variant won."""
    ratios = sorted(b / a for a, b in pairs)
    rank = interval_rank(len(ratios))
    if rank is None:
        raise ValueError(f"{len(ratios)} cycles cannot bound a median at 95 %")
    return {"ratio": statistics.median(ratios), "low": ratios[rank - 1],
            "high": ratios[-rank], "faster": sum(b < a for a, b in pairs),
            "cycles": len(pairs)}


def verdict(change: dict) -> str:
    if change["high"] < 1.0:
        return "faster"
    if change["low"] > 1.0:
        return "slower"
    return "within noise"


def summarize(runs: list[dict], labels: list[str], baseline: str) -> list[dict]:
    rows = []
    for metric in METRICS:
        for seq_len, cycles in sorted(cycle_table(runs, metric).items()):
            complete = [c for c in cycles.values() if all(label in c for label in labels)]
            if (len(complete) < MIN_CYCLES
                    or min(min(c.values()) for c in complete) < MIN_MS):
                continue
            for label in labels:
                if label == baseline:
                    continue
                change = paired_change([(c[baseline], c[label]) for c in complete])
                rows.append({"metric": metric, "seq_len": seq_len, "variant": label,
                             "baseline_ms": statistics.median(c[baseline] for c in complete),
                             "variant_ms": statistics.median(c[label] for c in complete),
                             **change, "verdict": verdict(change)})
    return rows


def pct(ratio: float) -> str:
    return f"{(ratio - 1.0) * 100.0:+.1f} %"


def render_report(metadata: dict, rows: list[dict]) -> str:
    protocol = metadata["protocol"]
    revisions = ", ".join(f"{r['label']} = `{r['sha'][:12]}` ({r['ref']})"
                          for r in metadata["revisions"])
    lines = [
        "# Revision A/B, each revision built from scratch",
        "",
        f"Model: `{metadata['model']['file']}` (`{metadata['model']['sha256'][:16]}`)",
        f"Revisions: {revisions}; baseline `{metadata['baseline']}`"
        + (f"; `{CONTROL}` is a copy of the baseline's binary" if metadata["control"] else ""),
        f"Protocol: {protocol['cycles']} cycles, order rotating; sequences {protocol['seq_lens']}, "
        f"decode {protocol['decode_n']}, repeats {protocol['repeats']} (median), "
        f"warmup {protocol['warmup']}; "
        + " ".join(f"{name}={value}" for name, value in metadata["environment"].items()),
        f"Host: {metadata['system']['cpu']}, {metadata['system']['cores']} CPUs, "
        f"{metadata['system']['os']}, THP {metadata['system']['thp']}",
        "",
        "| row | variant | baseline ms | variant ms | change | 95 % interval | faster | verdict |",
        "| :-- | :-- | --: | --: | --: | --: | --: | :-- |",
    ]
    lines += [
        f"| {r['metric']} @ {r['seq_len']} | {r['variant']} | {r['baseline_ms']:.1f} | "
        f"{r['variant_ms']:.1f} | {pct(r['ratio'])} | [{pct(r['low'])}, {pct(r['high'])}] | "
        f"{r['faster']}/{r['cycles']} | {r['verdict']} |"
        for r in rows
    ]
    control = [r for r in rows if r["variant"] == CONTROL]
    if control:
        reach = max(max(1.0 - r["low"], r["high"] - 1.0) for r in control)
        moved = [r for r in control if r["verdict"] != "within noise"]
        lines += ["", f"Control: the baseline's own binary again; its intervals reach "
                      f"±{reach * 100.0:.1f} %, the resolution of this run."]
        if moved:
            lines.append(f"**The control moved in {len(moved)} of {len(control)} rows: the host "
                         "changed during the run, so a difference of that size is not the "
                         "change. Rerun on a quiet host or with more cycles.**")
    tested = sum(r["variant"] != CONTROL for r in rows)
    if tested > 1:
        lines += ["", f"{tested} comparisons: at 95 % about one interval in twenty excludes "
                      "zero by chance."]
    return "\n".join(lines) + "\n"


def system_info() -> dict:
    cpu = "unavailable"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        cpu = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                    if line.startswith("model name")), cpu)
    else:
        cpu = command_output(["sysctl", "-n", "machdep.cpu.brand_string"])
    thp = Path("/sys/kernel/mm/transparent_hugepage/enabled")
    return {"cpu": cpu, "cores": os.cpu_count(), "os": f"{platform.system()} {platform.release()}",
            "thp": thp.read_text().strip() if thp.is_file() else "n/a"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rev", action="append", required=True, metavar="LABEL=REF",
                        help="a git revision to build and measure; the first is the baseline")
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--seq-lens", default="128,512")
    parser.add_argument("--decode-n", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--cycles", type=int, default=10,
                        help="10 decide on 9 cycles one way, 12 on 10; 6 need all 6")
    parser.add_argument("--env", action="append", default=[], metavar="VAR=VALUE",
                        help="set for every bench run, e.g. GEIST_KV_INT8=0")
    parser.add_argument("--make-arg", action="append", default=[], metavar="ARG",
                        help="passed to make, e.g. CC=gcc-14")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--no-control", action="store_true",
                        help="skip the second run of the baseline's binary")
    parser.add_argument("--out-dir", type=Path,
                        default=Path.home() / "bench-geistlib" / "revision-ab")
    args = parser.parse_args()

    try:
        revs = parse_revs(args.rev)
        shas = {label: resolve(ref) for label, ref in revs}
        env_pairs = [value.partition("=") for value in args.env]
        if any(not name or not sep for name, sep, _ in env_pairs):
            raise ValueError("--env takes VAR=VALUE")
    except ValueError as error:
        parser.error(str(error))
    if len(revs) < 2 and args.no_control:
        parser.error("one revision needs the control to compare against")
    if args.cycles < MIN_CYCLES:
        parser.error(f"--cycles must be at least {MIN_CYCLES}: with fewer, all cycles "
                     "going one way happens by chance more often than one time in twenty")
    gguf = args.gguf.expanduser().resolve()
    if not gguf.is_file():
        parser.error(f"model not found: {gguf}")
    args.gguf = gguf
    baseline = revs[0][0]

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    work = args.out_dir.expanduser().resolve() / stamp
    work.mkdir(parents=True)
    binaries: dict[str, Path] = {}
    try:
        for sha in dict.fromkeys(shas.values()):
            print(f"building {sha[:12]}", file=sys.stderr, flush=True)
            binaries[sha] = build(sha, work, args.make_arg, args.jobs)
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    finally:
        subprocess.run(["git", "-C", str(ROOT), "worktree", "prune"], capture_output=True)

    variants = [{"label": label, "binary": binaries[shas[label]]} for label, _ in revs]
    if not args.no_control:
        control = work / f"{BENCH}_{CONTROL}"
        shutil.copy2(variants[0]["binary"], control)
        variants.append({"label": CONTROL, "binary": control})

    run_env = {"OMP_WAIT_POLICY": os.environ.get("OMP_WAIT_POLICY", "active"),
               **{name: value for name, _, value in env_pairs}}
    env = {**os.environ, **run_env}
    metadata = {
        "schema": "geist.benchmark.revision-ab.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": baseline,
        "control": not args.no_control,
        "revisions": [{"label": label, "ref": ref, "sha": shas[label],
                       "binary_sha256": sha256_file(binaries[shas[label]])}
                      for label, ref in revs],
        "model": {"file": gguf.name, "path": str(gguf), "sha256": sha256_file(gguf)},
        "protocol": {"cycles": args.cycles, "seq_lens": args.seq_lens,
                     "decode_n": args.decode_n, "repeats": args.repeats,
                     "warmup": args.warmup, "make_args": args.make_arg},
        "environment": run_env,
        "system": system_info(),
    }

    raw_path = work / "revision_ab.jsonl"
    runs: list[dict] = []
    with raw_path.open("w") as raw:
        raw.write(json.dumps({"kind": "metadata", **metadata}, sort_keys=True) + "\n")
        for cycle, position, variant in build_schedule(variants, args.cycles):
            print(f"cycle {cycle + 1}/{args.cycles}: {variant['label']}",
                  file=sys.stderr, flush=True)
            try:
                rows = run_variant(variant["binary"], args, env)
            except RuntimeError as error:
                print(f"ERROR: {error}", file=sys.stderr)
                return 1
            run = {"kind": "run", "cycle": cycle, "position": position,
                   "variant": variant["label"], "rows": rows}
            runs.append(run)
            raw.write(json.dumps(run, sort_keys=True) + "\n")
            raw.flush()
        rows = summarize(runs, [v["label"] for v in variants], baseline)
        raw.write(json.dumps({"kind": "summary", "rows": rows}, sort_keys=True) + "\n")

    report = render_report(metadata, rows)
    (work / "revision_ab.md").write_text(report)
    print(report)
    print(f"raw: {raw_path}\nreport: {work / 'revision_ab.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

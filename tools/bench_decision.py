#!/usr/bin/env python3
"""Hash and summarize the public-API decision benchmark; no third-party deps.

Input JSONL: id, prompt_ids, candidate_ids, target_index (optional).
Tokenization/template provenance is supplied explicitly with --tokenization.
See docs/DECISION.md for interpretation and cache/quality limitations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODES = ("decision_dense", "scorealt_dense", "generate_1", "generate_long")


def sha256(path: Path) -> str:
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cases_from_jsonl(path: Path) -> list[dict]:
    cases = []
    seen = set()
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        case = json.loads(line)
        if not isinstance(case, dict):
            raise ValueError(f"line {line_no}: expected an object")
        name = case.get("id", str(line_no))
        if not isinstance(name, str) or name in seen:
            raise ValueError(f"line {line_no}: id must be a unique string")
        seen.add(name)
        for field in ("prompt_ids", "candidate_ids"):
            ids = case.get(field)
            if (not isinstance(ids, list) or not ids
                    or any(type(token) is not int or not 0 <= token <= 2147483647 for token in ids)):
                raise ValueError(f"line {line_no}: {field} must contain single-token integer IDs")
        if len(set(case["candidate_ids"])) != len(case["candidate_ids"]):
            raise ValueError(f"line {line_no}: duplicate candidates")
        target = case.get("target_index")
        if target is not None and (type(target) is not int or not 0 <= target < len(case["candidate_ids"])):
            raise ValueError(f"line {line_no}: invalid target_index")
        cases.append({**case, "id": name})
    if not cases:
        raise ValueError("dataset is empty")
    return cases


def wire_input(cases: list[dict]) -> str:
    return "".join(" ".join(map(str, [len(c["prompt_ids"]), len(c["candidate_ids"]),
                                      *c["prompt_ids"], *c["candidate_ids"]])) + "\n" for c in cases)


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    index = (len(values) - 1) * fraction
    low = int(index)
    high = math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def summarize(samples: list[dict], cases: list[dict], modes=MODES) -> dict:
    groups = {}
    for case_index, case in enumerate(cases):
        for mode in modes:
            rows = [s for s in samples if s["case_index"] == case_index and s["mode"] == mode and s["phase"] == "warm"]
            if not rows:
                raise ValueError(f"missing warm samples: {case['id']} / {mode}")
            times = [s["elapsed_ms"] for s in rows]
            labelled = [s["correct"] for s in rows if s["correct"] is not None]
            groups[f"{case['id']}/{mode}"] = {
                "n": len(rows), "p50_ms": percentile(times, .5), "p95_ms": percentile(times, .95),
                "min_ms": min(times), "max_ms": max(times),
                "first_token_label_accuracy": sum(labelled) / len(labelled) if labelled else None,
                "actual_generated_tokens": sorted(set(len(s["generated_ids"]) for s in rows)),
            }
    ratios = {}
    for case in cases:
        decision = groups[f"{case['id']}/decision_dense"]["p50_ms"]
        ratios[case["id"]] = {mode: groups[f"{case['id']}/{mode}"]["p50_ms"] / decision
                              for mode in modes if mode != "decision_dense"}
    return {"by_case_and_mode": groups, "p50_latency_ratio_over_decision": ratios,
            "quality_scope": "Candidate argmax accuracy for scores; first emitted token label accuracy for generation. Longer output correctness requires a downstream evaluator. Repeats are not independent quality examples.",
            "warning": "Latency ratios are workload-specific. They are not Jev parity or quality-matched reasoning speedups."}


def enrich_and_check(samples: list[dict], cases: list[dict], warmup: int, repeats: int, modes=MODES) -> None:
    expected = {(i, mode, trial) for i in range(len(cases)) for mode in modes
                for trial in range(1 + warmup + repeats)}
    got = set()
    reference = {}
    for sample in samples:
        key = (sample["case_index"], sample["mode"], sample["trial"])
        if key not in expected or key in got:
            raise ValueError("unexpected or duplicate benchmark sample")
        got.add(key)
        trial = sample["trial"]
        phase = "first" if trial == 0 else "warmup" if trial <= warmup else "warm"
        if sample["phase"] != phase or not math.isfinite(sample["elapsed_ms"]) or sample["elapsed_ms"] <= 0:
            raise ValueError("invalid phase or latency")
        case = cases[sample["case_index"]]
        if sample["prompt_tokens"] != len(case["prompt_ids"]) or sample["candidate_count"] != len(case["candidate_ids"]):
            raise ValueError("runtime changed input geometry")
        sample["case_id"] = case["id"]
        target = case.get("target_index")
        if sample["mode"] in (*MODES[:2], "decision_selected"):
            best = sample["best_index"]
            scores = sample["logits"]
            probabilities = sample["conditional_probabilities"]
            if (type(best) is not int or not 0 <= best < len(case["candidate_ids"])
                    or len(scores) != len(case["candidate_ids"])
                    or len(probabilities) != len(scores)
                    or any(not math.isfinite(v) for v in scores)
                    or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities)
                    or not math.isclose(sum(probabilities), 1, abs_tol=1e-12)):
                raise ValueError("invalid numeric decision result")
            # Check API/reference parity on the same rotated trial, rather
            # than silently reporting fast but disagreeing implementations.
            pair = (sample["case_index"], trial)
            if pair in reference:
                previous = reference[pair]
                if best != previous["best_index"] or any(not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)
                    for a, b in zip(scores + probabilities, previous["logits"] + previous["conditional_probabilities"])):
                    raise ValueError("decision and dense reference disagree")
            else:
                reference[pair] = sample
            predicted = best
        else:
            output = sample["generated_ids"]
            predicted = case["candidate_ids"].index(output[0]) if output and output[0] in case["candidate_ids"] else None
        sample["correct"] = predicted == target if target is not None else None
    dense = {(s["case_index"], s["trial"]): s for s in samples if s["mode"] == "decision_dense"}
    for sample in samples:
        if sample["mode"] == "decision_selected":
            previous = dense[(sample["case_index"], sample["trial"])]
            if (sample["logits"] != previous["logits"] or sample["best_index"] != previous["best_index"]
                    or any(abs(a-b) > 1e-12 for a,b in zip(sample["conditional_probabilities"], previous["conditional_probabilities"]))):
                raise ValueError("selected rows and dense reference disagree exactly")
            if (type(sample.get("projected_rows")) is not int or sample["projected_rows"] < sample["candidate_count"]
                    or sample.get("logit_readback_bytes", 0) < sample["projected_rows"] * 4
                    or sample.get("head_ns", 0) <= 0):
                raise ValueError("missing selected-row instrumentation")
    if got != expected:
        raise ValueError("benchmark output is incomplete")


def command_output(args: list[str]) -> str | None:
    proc = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    return proc.stdout.strip() if proc.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/mac-omp/release/tools/bench_decision")
    parser.add_argument("--backend", default="cpu_neon")
    parser.add_argument("--selected", action="store_true", help="also run explicit selected-row mode; unsupported pairs fail")
    parser.add_argument("--tokenization", required=True, help="tokenizer revision, template, BOS policy, candidate construction")
    parser.add_argument("--out-dir", type=Path, required=True, help="new directory; existing results are never overwritten")
    parser.add_argument("--decode-n", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    if not (2 <= args.decode_n <= 4096 and 0 <= args.warmup <= 1000 and 1 <= args.repeats <= 10000 and args.threads > 0):
        parser.error("invalid decode/warmup/repeat/thread bounds")
    cases = cases_from_jsonl(args.cases)
    prompt_cap = max(len(c["prompt_ids"]) for c in cases)
    candidate_cap = max(len(c["candidate_ids"]) for c in cases)
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads))
    env.setdefault("OMP_WAIT_POLICY", "active")
    command = [str(args.binary.resolve()), str(args.model.resolve()), args.backend,
               str(prompt_cap), str(candidate_cap), str(args.decode_n), str(args.warmup), str(args.repeats)]
    modes = (*MODES, "decision_selected") if args.selected else MODES
    if args.selected:
        command.append("--selected")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    process = subprocess.run(command, input=wire_input(cases), env=env, capture_output=True, text=True)
    (args.out_dir / "driver.stdout.jsonl").write_text(process.stdout)
    (args.out_dir / "driver.stderr.log").write_text(process.stderr)
    if process.returncode != 0:
        raise RuntimeError(f"driver failed ({process.returncode}); see {args.out_dir / 'driver.stderr.log'}")
    records = [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
    runtimes = [r for r in records if r.get("kind") == "runtime"]
    if len(runtimes) != 1:
        raise ValueError("missing/duplicate runtime metadata")
    samples = [r for r in records if r.get("kind") == "sample"]
    enrich_and_check(samples, cases, args.warmup, args.repeats, modes)
    # Hash AFTER inference: reading the model for its hash would warm the
    # page cache before the very first sample. No cache eviction is claimed.
    metadata = {
        "protocol": "geist-decision-v1", "started_utc": started, "command": command,
        "model_sha256": sha256(args.model), "runtime_binary_sha256": sha256(args.binary),
        "dataset_sha256": sha256(args.cases), "tokenization": args.tokenization,
        "source_revision": command_output(["git", "rev-parse", "HEAD"]),
        "source_status": command_output(["git", "status", "--porcelain"]),
        "host": platform.node(), "os": platform.platform(), "machine": platform.machine(),
        "cpu": command_output(["sysctl", "-n", "machdep.cpu.brand_string"]) if platform.system() == "Darwin" else platform.processor(),
        "runtime": runtimes[0], "concurrency": 1, "threads": args.threads,
        "warmup": args.warmup, "repeats": args.repeats, "decode_cap": args.decode_n,
        "cache": "No forced page/device-cache eviction. First mode calls after model/session setup are recorded separately; modes share backend/page caches. Warm samples follow the configured warmups.",
        "environment": {k: v for k, v in env.items() if k.startswith(("GEIST_", "OMP_", "KMP_", "VECLIB_", "OPENBLAS_"))},
    }
    for name, data in (("metadata.json", metadata), ("cases.json", cases), ("summary.json", summarize(samples, cases, modes))):
        (args.out_dir / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    (args.out_dir / "samples.jsonl").write_text("".join(json.dumps(s, allow_nan=False) + "\n" for s in samples))
    print(args.out_dir / "summary.json")


if __name__ == "__main__":
    main()

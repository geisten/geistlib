#!/usr/bin/env python3
"""Opt-in, offline ModernBERT/Laya development evaluation; no native C claim."""
from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
from statistics import mean, median
import subprocess
import sys
import time

import decision_dataset as data
import decision_encoders as encoders
import decision_metrics as metrics


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def load_cases(directory, split, purpose):
    meta = encoders.read_json(directory / "metadata.json")
    audit_path = directory / "split_audit.json"
    audit = encoders.read_json(audit_path)
    path = directory / f"{split}.jsonl"
    if (meta.get("protocol") != "geist-decision-encoder-cases-v1" or meta["split"] != split
            or encoders.sha256(path) != meta["cases_sha256"]
            or encoders.sha256(audit_path) != meta["split_audit_sha256"]):
        raise ValueError("case/audit integrity mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    seen, examples, variants = set(), set(), defaultdict(set)
    for row in rows:
        data.validate_question(row)
        if (row["question_id"] != data.question_key(row) or row["split"] != split
                or row["target_index"] != row["answer"] or row["id"] in seen
                or type(row["variant"]) is not int or row["variant"] not in range(meta["rotations"])):
            raise ValueError("question identity/target/rotation mismatch")
        seen.add(row["id"])
        if row["variant"] in variants[row["question_id"]]:
            raise ValueError("duplicate question rotation")
        variants[row["question_id"]].add(row["variant"])
        if len(row["examples"]) != meta["shots_available"]:
            raise ValueError("incomplete fixed examples")
        for example in row["examples"]:
            checked = {**example, "subject": row["subject"]}
            data.validate_question(checked)
            if example["question_id"] != data.question_key(checked):
                raise ValueError("example identity mismatch")
            examples.add(example["question_id"])
    if (not rows or len(rows) != meta["records"] or len(variants) != meta["questions"]
            or set(variants) & examples
            or any(v != set(range(meta["rotations"])) for v in variants.values())):
        raise ValueError("population, example overlap or rotations incomplete")
    if purpose == "pilot" and split != "development":
        raise ValueError("pilots use development; calibration/test stay untouched")
    if purpose == "quality" and (split != "test" or meta["per_subject"] or meta["subject_limit"]
                                  or meta["questions"] != audit["prepared_counts"]["test"]):
        raise ValueError("quality requires the complete declared held-out test population")
    return meta, rows


def mmlu_request(row, shots):
    if type(shots) is not int or not 0 <= shots <= len(row["examples"]):
        raise ValueError("requested fixed examples unavailable")
    text = []
    for example in row["examples"][:shots]:
        text.append("QUESTION: " + example["question"] + "\nCHOICES:\n"
                    + "\n".join(f"- {key}: {value}" for key, value in zip("ABCD", example["choices"]))
                    + "\nANSWER: " + "ABCD"[example["answer"]])
    text.append("QUESTION: " + row["question"])
    return encoders.ChoiceRequest("\n\n".join(text),
                                  "You will be given a question and options. Select the right answer.",
                                  dict(zip("ABCD", row["choices"])))


def summarize(samples, rows, warmup):
    grouped = defaultdict(list)
    rejected = {}
    for sample in samples:
        if sample["status"] == "rejected":
            rejected[sample["case_id"]] = sample
        else:
            grouped[sample["case_id"]].append(sample)
    canonical = [row for row in rows if row["variant"] == 0]
    scored, correct, stable, timings = [], 0, True, []
    drift = 0.0
    for row in canonical:
        trials = grouped.get(row["id"], [])
        if not trials:
            continue
        first = trials[0]
        correct += first["choice"] == "ABCD"[row["target_index"]]
        stable &= all(s["choice"] == first["choice"] for s in trials)
        drift = max(drift, max(abs(a - b) for s in trials for a, b in zip(first["logits"], s["logits"])))
        scored.append({"question_id": row["question_id"], "subject": row["subject"],
                       "split": row["split"], "variant": 0, "target_index": row["target_index"],
                       "logits": first["logits"]})
        warmed = [s for s in trials if s["trial"] >= warmup]
        timings.append({"question_id": row["question_id"],
                        "request_p50_ms": median(s["request_ms"] for s in warmed),
                        "forward_p50_ms": median(s["forward_ms"] for s in warmed),
                        "prepare_p50_ms": median(s["prepare_ms"] for s in warmed),
                        "tokens": len(first["input_ids"])})
    return {"questions": len(canonical), "scored": len(scored), "correct": correct,
            "rejected": len(canonical) - len(scored), "coverage": len(scored) / len(canonical),
            "accuracy_rejections_count_incorrect": correct / len(canonical),
            "quality_on_scored_only": metrics.quality(scored) if scored else None,
            "predictions_stable": stable, "max_repeat_logit_abs_drift": drift,
            "warm_request_p50_ms": median(t["request_p50_ms"] for t in timings) if timings else None,
            "warm_request_mean_ms": mean(t["request_p50_ms"] for t in timings) if timings else None,
            "warm_forward_p50_ms": median(t["forward_p50_ms"] for t in timings) if timings else None,
            "per_question_timings": timings,
            "rejections": [{"case_id": s["case_id"], "reason": s["reason"]} for s in rejected.values()],
            "calibrated_correctness_established": False, "quality_matched_speedup_established": False,
            "limits": "Independent encoder prompts/dtype/runtime; not identical Bonsai token streams."
                      " Only canonical questions count in quality N. Rejected inputs are incorrect in"
                      " population accuracy; probability metrics cover scored cases only."}


def run(args):
    if not args.enable_encoder_backends:
        raise ValueError("--enable-encoder-backends is required; optional adapters default off")
    if (args.threads <= 0 or args.warmup < 0 or args.repeats <= 0
            or not math.isfinite(args.margin_pp) or not 0 <= args.margin_pp <= 100):
        raise ValueError("invalid threads/trials/margin")
    meta, rows = load_cases(args.cases_dir, args.split, args.purpose)
    if args.shots > meta["shots_available"]:
        raise ValueError("fixed examples unavailable")
    if len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision):
        raise ValueError("immutable checkpoint revision required")
    artifacts = encoders.artifact_manifest(args.model_dir)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    environment_keys = ("OMP_WAIT_POLICY", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                        "PYTORCH_ENABLE_MPS_FALLBACK", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "USE_TF")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise ValueError("MPS CPU fallback must be disabled for a device-specific measurement")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["USE_TF"] = "0"
    versions = {}
    for name in ("torch", "transformers", "laya", "numpy", "safetensors"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    hardware = {}
    if sys.platform == "darwin":
        for key in ("hw.model", "hw.memsize", "hw.ncpu", "machdep.cpu.brand_string"):
            probe = subprocess.run(["/usr/sbin/sysctl", "-n", key], capture_output=True, text=True)
            hardware[key] = probe.stdout.strip() if probe.returncode == 0 else None
    manifest = {"protocol": "geist-decision-encoder-run-v1", "dataset": meta,
                "hardware": hardware, "host_load_start": list(os.getloadavg()),
                "purpose": args.purpose, "adapter": args.adapter, "model_repo": args.model_repo,
                "model_revision": args.revision, "model_files": artifacts,
                "model_dir": str(args.model_dir.resolve()), "device": args.device, "dtype": args.dtype,
                "max_tokens": args.max_tokens, "threads": args.threads, "shots": args.shots,
                "warmup": args.warmup, "repeats": args.repeats, "margin_pp": args.margin_pp,
                "versions": versions, "python": sys.version, "platform": platform.platform(),
                "machine": platform.machine(), "environment": {k: os.environ.get(k) for k in environment_keys},
                "command": sys.argv, "source_sha256": {p.name: encoders.sha256(p) for p in
                    (Path(__file__), Path(encoders.__file__), Path(data.__file__), Path(metrics.__file__))},
                "comparison": "Exploratory encoder baseline. No trained head added to geist C."
                              " No same-token Bonsai parity, Jev equivalence or deployment acceptance."}
    try:
        manifest["source_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True).strip()
        manifest["source_status"] = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parent, text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        manifest["source_revision"] = None
    write_json(args.out_dir / "manifest.json", manifest)
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    start = time.perf_counter()
    load_start = time.perf_counter()
    backend = encoders.create_backend(args.adapter, args.model_dir, enabled=True,
                                     device=args.device, dtype=args.dtype, max_tokens=args.max_tokens)
    setup_ms = (time.perf_counter() - load_start) * 1000
    write_json(args.out_dir / "adapter.json", backend.metadata)
    samples = []
    try:
        with (args.out_dir / "samples.jsonl").open("w") as stream:
            for index, row in enumerate(rows):
                request = mmlu_request(row, args.shots)
                for trial in range(args.warmup + args.repeats):
                    try:
                        result = backend.score(request)
                    except encoders.InputRejected as exc:
                        sample = {"status": "rejected", "case_id": row["id"], "question_id": row["question_id"],
                                  "variant": row["variant"], "reason": str(exc),
                                  "state": request.state, "instructions": request.instructions, "options": request.options}
                        samples.append(sample)
                        stream.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
                        stream.flush()
                        break
                    sample = {"status": "scored", "case_id": row["id"], "question_id": row["question_id"],
                              "variant": row["variant"], "trial": trial,
                              "target_index": row["target_index"], **result}
                    samples.append(sample)
                    stream.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                if (index + 1) % 50 == 0:
                    print(f"{index + 1}/{len(rows)} cases", flush=True)
        report = summarize(samples, rows, args.warmup)
        report.update(protocol="geist-decision-encoder-report-v1", adapter=backend.metadata,
                      setup_ms=setup_ms, campaign_wall_ms=(time.perf_counter() - start) * 1000,
                      host_load_end=list(os.getloadavg()),
                      manifest_sha256=encoders.sha256(args.out_dir / "manifest.json"),
                      samples_sha256=encoders.sha256(args.out_dir / "samples.jsonl"),
                      peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                      * (1 if sys.platform == "darwin" else 1024))
        if encoders.artifact_manifest(args.model_dir) != artifacts:
            raise ValueError("checkpoint artifacts changed during evaluation")
        write_json(args.out_dir / "report.json", report)
        print(json.dumps({k: report[k] for k in ("questions", "scored", "correct", "rejected",
                                                 "warm_request_p50_ms", "predictions_stable")}), flush=True)
    except Exception as exc:
        write_json(args.out_dir / "failure.json", {"error": type(exc).__name__, "message": str(exc),
                                                  "completed_sample_records": len(samples)})
        raise
    finally:
        backend.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--enable-encoder-backends", action="store_true")
    p.add_argument("--cases-dir", type=Path, required=True)
    p.add_argument("--split", choices=("development", "test"), required=True)
    p.add_argument("--purpose", choices=("pilot", "quality"), required=True)
    p.add_argument("--adapter", choices=("modernbert", "laya"), required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--model-repo", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--device", choices=("cpu", "mps"), required=True)
    p.add_argument("--dtype", choices=("float32", "float16"), required=True)
    p.add_argument("--max-tokens", type=int, required=True)
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--shots", type=int, choices=range(6), required=True)
    p.add_argument("--warmup", type=int, required=True)
    p.add_argument("--repeats", type=int, required=True)
    p.add_argument("--margin-pp", type=float, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    try:
        run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        p.exit(1, f"encoder evaluation failed: {exc}\n")


if __name__ == "__main__":
    main()

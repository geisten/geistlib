#!/usr/bin/env python3
"""Run frozen offline decision policies; keep exploratory and held-out evidence separate.

Raw driver output is retained even on failure. No model/dataset downloads,
training, fallback calls or Jev service requests are made by this tool.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
from statistics import mean
import subprocess
import time

import bench_decision as bench
import decision_dataset as data
import decision_metrics as metrics

PROFILES = {
    "chat_direct": ("decision_dense", "decision_selected", "scorealt_dense", "generate_1"),
    "cloze": ("decision_dense", "decision_selected", "scorealt_dense"),
    "chat_reasoning": ("generate_long",),
}
NUMERIC = {"decision_dense", "decision_selected", "scorealt_dense"}


def read_json(path):
    def invalid(value):
        raise ValueError(f"nonfinite JSON value: {value}")
    return json.loads(path.read_text(), parse_constant=invalid)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_cases(directory, split, purpose):
    meta = read_json(directory / "metadata.json")
    audit = read_json(directory / "split_audit.json")
    path = directory / f"{split}.jsonl"
    if (meta["protocol"] != "geist-decision-mmlu-cases-v1"
            or bench.sha256(path) != meta["counts"][split]["sha256"]
            or bench.sha256(directory / "split_audit.json") != meta["split_audit_sha256"]
            or bench.sha256(directory / "chat_template.jinja") != meta["gguf_metadata"]["template_sha256"]):
        raise ValueError("prepared dataset/template/audit integrity mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    seen, canonical = set(), set()
    if meta["rotations"] not in (1, 4):
        raise ValueError("one or four label rotations required")
    for row in rows:
        data.validate_question(row)
        if (row["split"] != split or row["target_index"] != row["answer"]
                or row["id"] in seen or row["question_id"] != data.question_key(row)
                or type(row["variant"]) is not int or row["variant"] not in range(meta["rotations"])):
            raise ValueError("prepared question identity/label invariant failed")
        seen.add(row["id"])
        key = (row["question_id"], row["variant"])
        if key in canonical:
            raise ValueError("duplicate question/rotation")
        canonical.add(key)
        for profile in PROFILES:
            case = row["profiles"][profile]
            if (len(case["candidate_ids"]) != 4
                    or hashlib.sha256(case["prompt_text"].encode()).hexdigest() != case["prompt_text_sha256"]):
                raise ValueError("prepared prompt integrity mismatch")
            for field in ("prompt_build_ms", "tokenize_ms", "boundary_check_ms"):
                if type(case[field]) not in (int, float) or not math.isfinite(case[field]) or case[field] < 0:
                    raise ValueError("invalid preprocessing timing")
    question_variants = {}
    for name, variant in canonical:
        question_variants.setdefault(name, set()).add(variant)
    if any(v != set(range(meta["rotations"])) for v in question_variants.values()):
        raise ValueError("incomplete rotations for a question")
    count = meta["counts"][split]
    n = sum(r["variant"] == 0 for r in rows)
    if (not n or n != count["questions"] or len(rows) != count["records"]
            or len(rows) != n * meta["rotations"]):
        raise ValueError("incomplete canonical questions/rotations")
    if purpose == "quality" and (split != "test" or meta["per_subject"] != 0
                                  or n != audit["prepared_counts"]["test"]):
        raise ValueError("quality requires the complete declared held-out test population")
    if purpose == "pilot" and split != "development":
        raise ValueError("exploratory pilots use development, keeping calibration/test untouched")
    if purpose == "calibration" and split != "calibration":
        raise ValueError("calibration requires the separate calibration split")
    return meta, rows


def binding(meta, binary_hash, backend, threads, environment):
    """Independent of question IDs: same frozen score policy on a disjoint split."""
    return {"model_sha256": meta["model_sha256"], "binary_sha256": binary_hash,
            "tokenizer_binary_sha256": meta["tokenizer_binary_sha256"],
            "backend": backend, "host": platform.node(), "os": platform.platform(),
            "machine": platform.machine(), "threads": threads, "kv_mode": "fp32", "environment": environment,
            "template_sha256": meta["gguf_metadata"]["template_sha256"],
            "gguf_metadata": meta["gguf_metadata"], "prompt_policy": meta["prompt_policy"],
            "shots": meta["shots"], "template_date": meta["template_date"],
            "label_policy": meta["label_policy"], "profile": "chat_direct"}


def generation_outcome(sample, case, thinking, eos, cap, vocab):
    ids = sample["generated_ids"]
    stop = sample["generation_stop"]
    if (not ids or len(ids) > cap
            or any(type(i) is not int or not 0 <= i < vocab for i in ids)
            or eos in ids[:-1]
            or stop not in ("eos", "limit")
            or (stop == "eos") != (ids[-1] == eos)
            or (stop == "limit" and len(ids) != cap)
            or type(sample["generated_text_complete"]) is not bool
            or not math.isfinite(sample["surface_decode_ms"]) or sample["surface_decode_ms"] < 0):
        raise ValueError("invalid generated sequence/stop/surface metadata")
    start = time.perf_counter_ns()
    text = None
    reason = None
    if not sample["generated_text_complete"]:
        reason = "incomplete_token_surface"
    try:
        encoded = bytes.fromhex(sample["generated_utf8_hex"])
    except (ValueError, TypeError) as error:
        raise ValueError("invalid generated hex bytes") from error
    if reason is None:
        try:
            text = encoded.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            reason = "invalid_utf8"
    prediction = metrics.parse_answer(text, thinking) if text is not None else None
    if prediction is None and reason is None:
        reason = "invalid_final_answer"
    return {"predicted_index": prediction, "correct": prediction == case["target_index"],
            "valid": prediction is not None, "invalid_reason": reason,
            "truncated": stop == "limit", "stop": stop, "generated_tokens": len(ids),
            "answer_parse_ms": (time.perf_counter_ns() - start) / 1e6}


def summarize_profile(samples, rows, profile, modes, runtime, decode_cap, warmup):
    """Warm representative gives one quality outcome/question, never one/trial."""
    eos = runtime["eos_token_id"]
    report, numeric, outcomes, timing = {}, {}, {}, {}
    grouped = {}
    for sample in samples:
        grouped.setdefault((sample["case_index"], sample["mode"]), []).append(sample)
    for mode in modes:
        representative, measured, generated = [], [], []
        first_times, per_case, warm_times = [], [], []
        for i, row in enumerate(rows):
            trials = sorted(grouped[(i, mode)], key=lambda s: s["trial"])
            warm = [s for s in trials if s["phase"] == "warm"]
            s = next(s for s in trials if s["trial"] == warmup + 1)
            case = row["profiles"][profile]
            meta = {k: row[k] for k in ("question_id", "split", "variant", "target_index", "subject")}
            if mode in NUMERIC:
                if any(t["best_index"] != max(range(4), key=lambda i: t["logits"][i]) for t in trials):
                    raise ValueError("numeric best index does not match score argmax")
                if any(t["best_index"] != s["best_index"] for t in trials):
                    raise ValueError("numeric quality prediction changed between deterministic repeats")
                representative.append({**meta, "logits": s["logits"]})
                result = {"correct": s["best_index"] == row["target_index"]}
                parse_ms = 0.
            else:
                if any(t["generated_ids"] != s["generated_ids"] for t in trials):
                    raise ValueError("greedy generation changed between deterministic repeats")
                decoded = [generation_outcome(t, row, profile == "chat_reasoning", eos,
                                              1 if mode == "generate_1" else decode_cap, runtime["vocab"])
                           for t in trials]
                result = decoded[warmup + 1]
                if any(t["generated_utf8_hex"] != s["generated_utf8_hex"]
                       or t["generated_text_complete"] != s["generated_text_complete"] for t in trials):
                    raise ValueError("generated surface changed between deterministic repeats")
                generated.append({**meta, **result})
                parse_ms = result["answer_parse_ms"]
            if row["variant"] == 0:
                first_times.append(trials[0]["elapsed_ms"])
                warm_times.extend(t["elapsed_ms"] for t in warm)
                per_case.append(bench.percentile([t["elapsed_ms"] for t in warm], .5))
                preprocessing = case["prompt_build_ms"] + case["tokenize_ms"]
                surface = mean(t.get("surface_decode_ms", 0.) for t in warm)
                measured.append({"question_id": row["question_id"], "inference_p50_ms": per_case[-1],
                                 "preprocessing_ms": preprocessing, "surface_decode_ms": surface,
                                 "answer_parse_ms": parse_ms,
                                 "additive_component_estimate_ms": per_case[-1] + preprocessing + surface + parse_ms})
        canonical = [r for r in representative if r["variant"] == 0]
        quality = metrics.quality(canonical) if canonical else None
        canonical_gen = [g for g in generated if g["variant"] == 0]
        if canonical_gen:
            quality = {"n": len(canonical_gen), "accuracy": mean(g["correct"] for g in canonical_gen),
                       "invalid_answers": sum(not g["valid"] for g in canonical_gen),
                       "invalid_reasons": dict(Counter(g["invalid_reason"] for g in canonical_gen if not g["valid"])),
                       "capped_outputs": sum(g["truncated"] for g in canonical_gen),
                       "by_subject": {subject: {"n": len(group), "accuracy": mean(g["correct"] for g in group)}
                                      for subject in sorted({g["subject"] for g in canonical_gen})
                                      if (group := [g for g in canonical_gen if g["subject"] == subject])},
                       "generated_tokens": {"min": min(g["generated_tokens"] for g in canonical_gen),
                                            "max": max(g["generated_tokens"] for g in canonical_gen),
                                            "mean": mean(g["generated_tokens"] for g in canonical_gen)}}
        variants = {}
        for shift in sorted({r["variant"] for r in rows} - {0}):
            if representative:
                rotated = [{**r, "variant": 0} for r in representative if r["variant"] == shift]
                variants[str(shift)] = metrics.quality(rotated)["accuracy"]
            else:
                variants[str(shift)] = mean(g["correct"] for g in generated if g["variant"] == shift)
        report[mode] = {"quality": quality, "rotation_accuracy_supplementary": variants,
                        "latency": {"warm_sample_p50_ms": bench.percentile(warm_times, .5),
                                    "warm_sample_p95_ms": bench.percentile(warm_times, .95),
                                    "question_median_p50_ms": bench.percentile(per_case, .5),
                                    "question_median_p95_ms": bench.percentile(per_case, .95),
                                    "mean_question_median_ms": mean(per_case),
                                    "first_call_median_ms": bench.percentile(first_times, .5),
                                    "sequential_rate_estimate_qps": 1000 / mean(per_case)},
                        "peak_process_rss_bytes": max(s["process_peak_rss_bytes"] for s in samples if s["mode"] == mode)}
        if mode in NUMERIC:
            report[mode]["projected_rows"] = sorted({s.get("projected_rows", runtime["vocab"]) for s in samples if s["mode"] == mode})
            report[mode]["logit_readback_bytes"] = sorted({s.get("logit_readback_bytes", runtime["vocab"] * 4) for s in samples if s["mode"] == mode})
        numeric[mode] = canonical
        outcomes[mode] = ({r["question_id"]: r["correct"] for r in canonical_gen} if canonical_gen else
                          {r["question_id"]: max(range(4), key=lambda i: r["logits"][i]) == r["target_index"] for r in canonical})
        timing[mode] = measured
    return report, numeric, outcomes, timing


def comparison(direct, baseline, direct_times, baseline_times, purpose, margin, baseline_quality):
    paired = metrics.paired_quality(direct, baseline, margin)
    blockers = []
    if purpose != "quality":
        blockers.append("not a complete held-out quality run")
    if not paired["noninferior"]:
        blockers.append("declared noninferiority margin not established")
    if baseline_quality["invalid_answers"] or baseline_quality["capped_outputs"]:
        blockers.append("reasoning baseline contains invalid or capped answers")
    if set(direct) != {r["question_id"] for r in direct_times}:
        raise ValueError("latency and quality populations differ")
    if len(direct_times) != len(direct) or len(baseline_times) != len(baseline):
        raise ValueError("duplicate timing questions")
    for key in ("inference_p50_ms", "additive_component_estimate_ms"):
        if {r["question_id"] for r in direct_times} != {r["question_id"] for r in baseline_times}:
            raise ValueError("latency and quality populations differ")
        paired[key + "_ratio_reasoning_over_selected"] = sum(r[key] for r in baseline_times) / sum(r[key] for r in direct_times)
    return {"paired_quality": paired, "quality_matched_speedup_eligible": not blockers,
            "blockers": blockers,
            "scope": "Same questions, greedy FP32 geist policies. Additive preprocessing/surface/parse costs are estimates, not measured service latency. No claim of published Bonsai or Jev parity."}


def run(args):
    campaign_start = time.perf_counter()
    meta, rows = load_cases(args.cases_dir, args.split, args.purpose)
    model_hash, binary_hash = bench.sha256(args.model), bench.sha256(args.binary)
    if model_hash != meta["model_sha256"]:
        raise ValueError("model differs from prepared GGUF tokenizer/template checkpoint")
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads))
    env.setdefault("OMP_WAIT_POLICY", "passive")
    policy_env = {k: v for k, v in env.items() if k.startswith(("GEIST_", "OMP_", "KMP_", "VECLIB_", "OPENBLAS_"))}
    bound = binding(meta, binary_hash, args.backend, args.threads, policy_env)
    calibration = read_json(args.calibration) if args.calibration else None
    if calibration is not None:
        if args.purpose == "calibration" or calibration["binding"] != bound:
            raise ValueError("calibration policy binding mismatch")
        if {r["question_id"] for r in rows} & set(calibration["question_ids"]):
            raise ValueError("calibration and evaluation questions overlap")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.out_dir / "prepared_metadata.json", meta)
    write_json(args.out_dir / "split_audit.json", read_json(args.cases_dir / "split_audit.json"))
    profiles = {"chat_direct": PROFILES["chat_direct"][:3]} if args.purpose == "calibration" else PROFILES
    commands, cases = {}, {}
    for profile, modes in profiles.items():
        flat = [{"id": r["id"], "target_index": r["target_index"], **r["profiles"][profile]} for r in rows]
        path = args.out_dir / f"{profile}.cases.jsonl"
        path.write_text("".join(json.dumps(c, allow_nan=False) + "\n" for c in flat))
        cases[profile] = bench.cases_from_jsonl(path)
        commands[profile] = [str(args.binary.resolve()), str(args.model.resolve()), args.backend,
                             str(max(len(c["prompt_ids"]) for c in flat)), "4", str(args.decode_cap),
                             str(args.warmup), str(args.repeats), "--modes", ",".join(modes), "--text"]
        if "decision_selected" in modes:
            commands[profile].append("--selected")
    manifest = {"protocol": "geist-decision-evaluation-v1", "started_utc": datetime.now(timezone.utc).isoformat(),
                "purpose": args.purpose, "split": args.split, "margin_pp": args.margin_pp,
                "decode_cap": args.decode_cap, "warmup": args.warmup, "repeats": args.repeats,
                "commands": commands, "calibration_sha256": bench.sha256(args.calibration) if args.calibration else None,
                "prepared_metadata_sha256": bench.sha256(args.cases_dir / "metadata.json"),
                "cases_sha256": meta["counts"][args.split]["sha256"], "binding": bound,
                "dataset": meta["dataset"], "dataset_revision": meta["dataset_revision"],
                "dataset_subset": meta["dataset_subset"], "shots": meta["shots"], "rotations": meta["rotations"],
                "source_revision": bench.command_output(["git", "rev-parse", "HEAD"]),
                "source_status": bench.command_output(["git", "status", "--porcelain"]),
                "host": platform.node(), "os": platform.platform(), "machine": platform.machine(),
                "cpu": bench.command_output(["sysctl", "-n", "machdep.cpu.brand_string"]) if platform.system() == "Darwin" else platform.processor(),
                "ram_bytes": bench.command_output(["sysctl", "-n", "hw.memsize"]) if platform.system() == "Darwin" else None,
                "evaluator_source_sha256": {name: bench.sha256(Path(__file__).with_name(name)) for name in
                    ("eval_decision.py", "prepare_decision_eval.py", "decision_dataset.py", "decision_metrics.py", "bench_decision.py", "bench_decision.c", "eval_mmlu.py")},
                "concurrency": 1, "fallback": "none", "sampling": "greedy; no random sampling; FP32 KV",
                "cache": "Preparation loads/checksums the model. No cache eviction: first per-case calls are post-setup and not cold-cache measurements. Profiles run serially in separate processes.",
                "memory": "Process peak RSS includes loaded model and all retained session/decision handles; it is not incremental arm memory.",
                "timing": "Driver reset+prefill+score/generate. Preprocessing and text/answer conversion separate. Additive estimates exclude setup, validation tokenizations, dataset I/O and benchmark reference arms. No fallback/abstention route."}
    # Freeze before any labelled inference; never overwrite or tune this run.
    write_json(args.out_dir / "manifest.json", manifest)
    report = {"profiles": {}, "limitations": [manifest[k] for k in ("cache", "memory", "timing")]}
    results = {}
    for profile, modes in profiles.items():
        start = time.perf_counter()
        with (args.out_dir / f"{profile}.driver.jsonl").open("w") as out, (args.out_dir / f"{profile}.stderr.log").open("w") as err:
            proc = subprocess.run(commands[profile], input=bench.wire_input(cases[profile]),
                                  env=env, stdout=out, stderr=err, text=True)
        wall_ms = (time.perf_counter() - start) * 1000
        if proc.returncode:
            raise RuntimeError(f"driver failed ({proc.returncode}): {profile}; raw partial outputs retained")
        records = [json.loads(line) for line in (args.out_dir / f"{profile}.driver.jsonl").read_text().splitlines()]
        runtimes = [r for r in records if r.get("kind") == "runtime"]
        samples = [r for r in records if r.get("kind") == "sample"]
        if (len(runtimes) != 1 or len(samples) + 1 != len(records)
                or runtimes[0]["backend"] != args.backend or runtimes[0]["kv_mode"] != "fp32"):
            raise ValueError("invalid runtime/profile output")
        runtime = {**runtimes[0], "eos_token_id": meta["gguf_metadata"]["eos_token_id"]}
        bench.enrich_and_check(samples, cases[profile], args.warmup, args.repeats, modes)
        value, numeric, outcomes, timing = summarize_profile(samples, rows, profile, modes, runtime, args.decode_cap, args.warmup)
        results[profile] = (numeric, outcomes, timing)
        report["profiles"][profile] = {"modes": value, "campaign_wall_ms": wall_ms, "runtime": runtime}
        write_json(args.out_dir / f"{profile}.component_timings.json", timing)
        if profile == "chat_direct":
            if args.purpose == "calibration":
                write_json(args.out_dir / "calibration.json", metrics.fit_temperature(numeric["decision_dense"], bound))
            elif calibration:
                value["decision_dense"]["calibrated_quality"] = metrics.apply_calibration(numeric["decision_dense"], calibration, bound)
                value["decision_selected"]["calibrated_quality"] = metrics.apply_calibration(numeric["decision_selected"], calibration, bound)
    if args.purpose != "calibration":
        _, direct, dt = results["chat_direct"]
        _, baseline, bt = results["chat_reasoning"]
        report["comparison"] = comparison(direct["decision_selected"], baseline["generate_long"],
                                            dt["decision_selected"], bt["generate_long"], args.purpose, args.margin_pp,
                                            report["profiles"]["chat_reasoning"]["modes"]["generate_long"]["quality"])
    report["campaign_wall_ms_including_setup_validation_and_reference_arms"] = (time.perf_counter() - campaign_start) * 1000
    report["model_query_count_including_reference_arms_and_repeats"] = sum(len(m) for m in profiles.values()) * len(rows) * (1 + args.warmup + args.repeats)
    report["acceptance_scope"] = "This harness does not establish external Prism same-token parity or Jev equivalence. Pilot reports remain exploratory even when observed accuracies agree. No production rollout decision is automatic."
    write_json(args.out_dir / "report.json", report)
    print(args.out_dir / "report.json")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases-dir", type=Path, required=True)
    p.add_argument("--split", choices=("development", "calibration", "test"), required=True)
    p.add_argument("--purpose", choices=("pilot", "calibration", "quality"), required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--backend", choices=("cpu_neon", "metal"), required=True)
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--decode-cap", type=int, required=True)
    p.add_argument("--margin-pp", type=float, required=True, help="declared before test; never selected from results")
    p.add_argument("--warmup", type=int, required=True)
    p.add_argument("--repeats", type=int, required=True)
    p.add_argument("--calibration", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    if (not 2 <= args.decode_cap <= 32768 or not 0 <= args.warmup <= 1000
            or not 1 <= args.repeats <= 10000 or args.threads < 1
            or not math.isfinite(args.margin_pp) or not 0 <= args.margin_pp <= 100):
        p.error("invalid decode/warmup/repeat/thread/margin bounds")
    run(args)


if __name__ == "__main__":
    main()

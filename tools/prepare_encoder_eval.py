#!/usr/bin/env python3
"""Prepare token-independent classic MMLU cases for optional encoder baselines."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import time

import decision_dataset as data
import decision_encoders as encoders


def prepare(args):
    start = time.perf_counter()
    import pyarrow.parquet as pq
    if (args.snapshot.name != args.dataset_revision or len(args.dataset_revision) != 40
            or any(c not in "0123456789abcdef" for c in args.dataset_revision)):
        raise ValueError("snapshot basename must match an immutable 40-character dataset revision")
    if type(args.shots) is not int or not 0 <= args.shots <= 5:
        raise ValueError("0..5 fixed subject examples required")
    files = {s: args.snapshot / "all" / f"{s}-00000-of-00001.parquet" for s in ("dev", "validation", "test")}
    sources = {s: pq.read_table(path).to_pylist() for s, path in files.items()}
    splits, exemplars, audit = data.partition(sources, args.overlap_policy)
    rows = data.select_per_subject(splits[args.split], args.per_subject, subject_limit=args.subject_limit)
    if not rows:
        raise ValueError("empty selection")
    prepared = []
    for row in rows:
        pool = exemplars.get(row["subject"], [])
        if len(pool) < args.shots:
            raise ValueError("not enough fixed subject examples")
        examples = [{k: r[k] for k in ("question", "choices", "answer", "question_id")} for r in pool[:args.shots]]
        for rotation in range(args.rotations):
            prepared.append({**data.permute(row, rotation), "examples": examples,
                             "target_index": data.permute(row, rotation)["answer"]})
    args.out_dir.mkdir(parents=True, exist_ok=False)
    path = args.out_dir / f"{args.split}.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in prepared))
    data.write_json(args.out_dir / "split_audit.json", audit)
    metadata = {"protocol": "geist-decision-encoder-cases-v1", "dataset": "cais/mmlu",
                "dataset_revision": args.dataset_revision, "dataset_subset": args.overlap_policy == "exclude",
                "dataset_files_sha256": {s: encoders.sha256(p) for s, p in files.items()},
                "split": args.split, "shots_available": args.shots, "rotations": args.rotations,
                "per_subject": args.per_subject, "subject_limit": args.subject_limit,
                "questions": len(rows), "records": len(prepared), "cases_sha256": encoders.sha256(path),
                "split_audit_sha256": encoders.sha256(args.out_dir / "split_audit.json"),
                "preparation_wall_ms": (time.perf_counter() - start) * 1000,
                "scope": "Token-independent cases; model-specific prompts and token IDs recorded by each adapter."
                         " Exact-overlap exclusion changes the population; no semantic deduplication."}
    data.write_json(args.out_dir / "metadata.json", metadata)
    print(args.out_dir / "metadata.json")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--dataset-revision", required=True)
    p.add_argument("--overlap-policy", choices=("reject", "exclude"), required=True)
    p.add_argument("--split", choices=("development", "calibration", "test"), required=True)
    p.add_argument("--shots", type=int, choices=range(6), required=True)
    p.add_argument("--per-subject", type=int, required=True)
    p.add_argument("--subject-limit", type=int, required=True)
    p.add_argument("--rotations", type=int, choices=(1, 4), required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    try:
        prepare(args)
    except (OSError, ValueError) as exc:
        p.exit(1, f"encoder case preparation failed: {exc}\n")


if __name__ == "__main__":
    main()

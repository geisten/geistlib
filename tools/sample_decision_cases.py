#!/usr/bin/env python3
"""Derive a smaller, declared sample from a prepared decision bundle (#171).

The sample is label-independent and stratified across subjects
(decision_dataset.select_stratified: proportional quotas, largest remainder,
within a subject by sha256([seed, question_id])). Prompts and token IDs are
copied unchanged, so no re-tokenization is needed; the source bundle's
integrity is checked first and its metadata hash is recorded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import bench_decision as bench
import decision_dataset as data
import eval_decision as ev


def derive(source, out_dir, split, questions, seed):
    purpose = {"development": "pilot", "calibration": "calibration"}.get(split)
    if purpose is None:
        raise ValueError("only development or calibration bundles are sampled here; "
                         "the held-out test sample is drawn at preparation (--test-sample)")
    meta, rows = ev.load_cases(source, split, purpose)
    if meta["rotations"] != 1:
        raise ValueError("sampling keeps one canonical record per question")
    chosen, allocation = data.select_stratified(rows, questions, seed=seed)
    keep = {r["question_id"] for r in chosen}
    lines = [line for line, row in zip((source / f"{split}.jsonl").read_text().splitlines(), rows)
             if row["question_id"] in keep]
    out_dir.mkdir(parents=True, exist_ok=False)
    for name in ("chat_template.jinja", "split_audit.json", "tokenizer.log"):
        if (source / name).exists():
            shutil.copy2(source / name, out_dir / name)
    path = out_dir / f"{split}.jsonl"
    path.write_text("".join(line + "\n" for line in lines))
    meta = {**meta, "counts": {split: {"questions": len(lines), "records": len(lines), "sha256": bench.sha256(path)}},
            f"{split}_sample": {"questions": len(lines), "population": len(rows), "seed": seed,
                                "allocation": allocation,
                                "source_metadata_sha256": bench.sha256(source / "metadata.json"),
                                "policy": "subject-proportional largest-remainder allocation; within a subject "
                                          "the first questions by sha256([seed, question_id]); label-independent"}}
    ev.write_json(out_dir / "metadata.json", meta)
    ev.load_cases(out_dir, split, purpose)  # the derived bundle passes the runner's own checks
    return out_dir / "metadata.json"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--split", choices=("development", "calibration"), required=True)
    p.add_argument("--questions", type=int, required=True)
    p.add_argument("--seed", default="geist-decision-calibration-sample-v1")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    print(derive(args.source, args.out_dir, args.split, args.questions, args.seed))


if __name__ == "__main__":
    main()

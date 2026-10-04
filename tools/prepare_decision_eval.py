#!/usr/bin/env python3
"""Prepare offline MMLU decision cases, preserving explicit policy and split audit.

This prepares data, not a quality result. `--overlap-policy` is required:
excluding contaminated or duplicate items creates a reported MMLU subset.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import time

import bench_decision as bench
import decision_dataset as data
import eval_mmlu


class Tokenizer:
    """GGUF tokenizer through eval_geist; stderr goes to a file, never a blocked pipe."""
    def __init__(self, binary, model, log):
        self.log = log.open("w")
        start = time.perf_counter()
        try:
            self.process = subprocess.Popen([str(binary), str(model)], stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1)
            while True:
                line = self.process.stdout.readline()
                if not line:
                    raise RuntimeError(f"tokenizer exited before READY; see {log}")
                if line.strip() == "READY":
                    break
            self.startup_ms = (time.perf_counter() - start) * 1000
            parts = self.command("BOS")
            if len(parts) != 2 or parts[0] not in ("0", "1"):
                raise ValueError("invalid tokenizer BOS policy")
            bos = int(parts[1])
            if parts[0] == "1" and bos < 0:
                raise ValueError("invalid tokenizer BOS token")
            self.bos = bos if parts[0] == "1" else None
        except BaseException:
            self.close()
            raise

    def command(self, text):
        self.process.stdin.write(text + "\n")
        self.process.stdin.flush()
        response = self.process.stdout.readline().strip().split()
        if not response or response[0] != "OK":
            raise RuntimeError(f"tokenizer command failed: {response}")
        return response[1:]

    def encode(self, text):
        if "\0" in text or "\r" in text or len(text.encode("utf-8")) >= 65536:
            raise ValueError("prompt does not fit the lossless TOK protocol")
        escaped = text.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")
        response = self.command("TOK " + escaped)
        if not response:
            raise ValueError("missing token count")
        ids = [int(x) for x in response[1:]]
        if int(response[0]) != len(ids) or not ids or any(i < 0 for i in ids):
            raise ValueError("invalid tokenizer result")
        return ids

    def close(self):
        if hasattr(self, "process"):
            if self.process.poll() is None:
                try:
                    self.process.stdin.write("QUIT\n")
                    self.process.stdin.flush()
                    self.process.wait(timeout=20)
                except (BrokenPipeError, ValueError, subprocess.TimeoutExpired):
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait()
            self.process.stdin.close()
            self.process.stdout.close()
        self.log.close()


def token_case(tokenizer, text, labels):
    start = time.perf_counter()
    ids = tokenizer.encode(text)
    tokenize_ms = (time.perf_counter() - start) * 1000
    boundary_start = time.perf_counter()
    candidates = []
    for label in labels:
        continued = tokenizer.encode(text + label)
        if len(continued) != len(ids) + 1 or continued[:-1] != ids:
            raise ValueError(f"continuation {label!r} is not one token at the actual prompt boundary")
        candidates.append(continued[-1])
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate token IDs are not distinct")
    if tokenizer.bos is not None:
        ids = [tokenizer.bos] + ids
    return {"prompt_ids": ids, "candidate_ids": candidates, "prompt_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "tokenize_ms": tokenize_ms, "boundary_check_ms": (time.perf_counter() - boundary_start) * 1000}


def model_metadata(model):
    from gguf import GGUFReader
    class MetadataOnly(GGUFReader):
        def _build_tensors(self, offset, tensor_fields):
            # Metadata needs no tensor dtype interpretation. Upstream gguf-py
            # cannot decode Prism's PQ2_0 enum 142, which geist already supports.
            pass
    reader = MetadataOnly(str(model))
    def field(name):
        f = reader.get_field(name)
        return f.contents() if f else None
    template = field("tokenizer.chat_template")
    if not isinstance(template, str) or not template:
        raise ValueError("model has no single stored chat template")
    return {"template": template, "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
            "architecture": field("general.architecture"), "add_bos_token": field("tokenizer.ggml.add_bos_token"),
            "bos_token_id": field("tokenizer.ggml.bos_token_id"), "eos_token_id": field("tokenizer.ggml.eos_token_id")}


def render_chat(compiled, content, thinking, template_date):
    fixed_date = datetime.fromisoformat(template_date)
    return compiled.render(messages=[{"role": "user", "content": content}], tools=None,
                           add_generation_prompt=True, enable_thinking=thinking, reasoning_effort="xhigh",
                           strftime_now=lambda fmt: fixed_date.strftime(fmt))


def prepare(args):
    preparation_start = time.perf_counter()
    import pyarrow.parquet as pq
    from transformers.utils.chat_template_utils import _compile_jinja_template
    if args.snapshot.name != args.dataset_revision:
        raise ValueError("snapshot directory name must match its declared pinned revision")
    if len(args.dataset_revision) != 40 or any(c not in "0123456789abcdef" for c in args.dataset_revision):
        raise ValueError("immutable 40-character dataset revision required")
    files = {s: args.snapshot / "all" / f"{s}-00000-of-00001.parquet" for s in ("dev", "validation", "test")}
    sources = {s: pq.read_table(p).to_pylist() for s, p in files.items()}
    splits, shots, audit = data.partition(sources, args.overlap_policy)
    meta = model_metadata(args.model)
    compiled = _compile_jinja_template(meta["template"])
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "chat_template.jinja").write_text(meta["template"])
    (args.out_dir / "split_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    tokenizer = Tokenizer(args.tokenizer_binary.resolve(), args.model.resolve(), args.out_dir / "tokenizer.log")
    counts = {}
    try:
        if ((tokenizer.bos is not None) != bool(meta["add_bos_token"])
                or (tokenizer.bos is not None and tokenizer.bos != meta["bos_token_id"])):
            raise ValueError("GGUF metadata and runtime BOS policy disagree")
        for split in args.splits.split(","):
            if split not in splits or split in counts:
                raise ValueError("split must be a unique development/calibration/test name")
            rows = data.select_per_subject(splits[split], args.per_subject)
            path = args.out_dir / f"{split}.jsonl"
            count = 0
            with path.open("w") as output:
                for row in rows:
                    pool = shots.get(row["subject"], [])
                    if len(pool) < args.shots:
                        raise ValueError(f"missing fixed subject exemplars: {row['subject']}")
                    exemplars = [(r["subject"], r["question"], r["choices"], r["answer"]) for r in pool[:args.shots]]
                    for rotation in range(args.rotations):
                        r = data.permute(row, rotation)
                        start = time.perf_counter()
                        cloze = eval_mmlu.build_prompt(r["subject"], r["question"], r["choices"], exemplars)
                        cloze_build_ms = (time.perf_counter() - start) * 1000
                        content = ("Answer the following multiple-choice question. For your final answer output only "
                                   "its letter A, B, C, or D.\n\n" + cloze)
                        profiles = {}
                        for name, thinking in (("chat_direct", False), ("chat_reasoning", True)):
                            render_start = time.perf_counter()
                            text = render_chat(compiled, content, thinking, args.template_date)
                            render_ms = (time.perf_counter() - render_start) * 1000
                            profile = token_case(tokenizer, text, list("ABCD"))
                            profile.update(prompt_text=text, prompt_build_ms=cloze_build_ms + render_ms,
                                           enable_thinking=thinking)
                            profiles[name] = profile
                        profiles["cloze"] = {**token_case(tokenizer, cloze, [" " + x for x in "ABCD"]),
                                             "prompt_text": cloze, "prompt_build_ms": cloze_build_ms}
                        output.write(json.dumps({**r, "target_index": r["answer"], "profiles": profiles},
                                                ensure_ascii=False, allow_nan=False) + "\n")
                        count += 1
            counts[split] = {"questions": len(rows), "records": count, "sha256": bench.sha256(path)}
    finally:
        tokenizer.close()
    metadata = {"protocol": "geist-decision-mmlu-cases-v1", "dataset": "cais/mmlu", "dataset_revision": args.dataset_revision,
                "dataset_files_sha256": {s: bench.sha256(p) for s, p in files.items()},
                "dataset_subset": args.overlap_policy == "exclude", "split_audit_sha256": bench.sha256(args.out_dir / "split_audit.json"),
                "model_sha256": bench.sha256(args.model), "model_size_bytes": args.model.stat().st_size,
                "tokenizer_binary_sha256": bench.sha256(args.tokenizer_binary), "tokenizer_startup_ms": tokenizer.startup_ms,
                "gguf_metadata": {k: v for k, v in meta.items() if k != "template"},
                "versions": {name: version(name) for name in ("gguf", "transformers", "jinja2", "pyarrow")},
                "prompt_policy": "classic-mmlu-fixed-subject-exemplars-chat-final-letter-xhigh-v1",
                "snapshot_path": str(args.snapshot.resolve()),
                "shots": args.shots, "rotations": args.rotations, "per_subject": args.per_subject,
                "template_date": args.template_date, "counts": counts,
                "label_policy": "Cloze uses spaced letters; chat uses bare letters. Every complete prompt+label tokenization must extend the actual prefix by exactly one distinct token.",
                "preprocessing": "Render and TOK timings retained separately from boundary validation and dataset preparation. Tokenizer model startup/checksums are setup costs. No forced cache eviction.",
                "quality": "Preparation supplies no accuracy or speedup evidence. Development/calibration/test identities are disjoint; test labels do not choose prompts or temperature."}
    metadata["preparation_wall_ms"] = (time.perf_counter() - preparation_start) * 1000
    metadata["tokenizer_calls_including_boundary_validation"] = sum(c["records"] for c in counts.values()) * 3 * 5
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(args.out_dir / "metadata.json")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True, help="pinned cais/mmlu HF snapshot, with all/*.parquet")
    p.add_argument("--dataset-revision", required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--tokenizer-binary", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--overlap-policy", choices=("reject", "exclude"), required=True)
    p.add_argument("--splits", default="development,calibration,test")
    p.add_argument("--shots", type=int, default=5)
    p.add_argument("--per-subject", type=int, default=0, help="0 = all; positive count is an explicitly reported pilot")
    p.add_argument("--rotations", type=int, choices=(1, 4), default=1)
    p.add_argument("--template-date", default=datetime.now(timezone.utc).date().isoformat())
    args = p.parse_args()
    if not 0 <= args.shots <= 5 or args.per_subject < 0:
        p.error("invalid shots/per-subject bounds")
    prepare(args)


if __name__ == "__main__":
    main()

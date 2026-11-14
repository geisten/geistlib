#!/usr/bin/env python3
"""Read the committed M1 and M6 artifacts for the first Qwen pilot case."""
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct
import sys

CASE_INDEX = 46
RECORD = f"{CASE_INDEX:08d}.json"


def read_json(path):
    return json.loads(path.read_bytes())


def sha(content):
    return hashlib.sha256(content).hexdigest()


def read_case(root):
    record_path = root / "cases" / RECORD
    record = read_json(record_path)
    relative = record["artifacts"]
    if record["index"] != CASE_INDEX or not re.fullmatch(
        rf"artifacts/case-{CASE_INDEX:08d}-attempt-\d+", relative
    ):
        raise ValueError(f"Unexpected case identity in {root}")
    path = root / relative
    generation = (path / "generation.json").read_bytes()
    logits = (path / "final-logits.f32").read_bytes()
    if sha(generation) != record["generation_sha256"] or sha(logits) != record["logits_sha256"]:
        raise ValueError(f"Artifact hash mismatch in {root}")
    raw = (path / "output.bin").read_bytes()
    return record_path, record, path, json.loads(generation), raw, logits


def compare_sequence(m1, m6):
    overlap = min(len(m1), len(m6))
    prefix = next((i for i, (a, b) in enumerate(zip(m1, m6)) if a != b), overlap)
    equal = prefix == len(m1) == len(m6)
    return {
        "equal": equal,
        "m1_count": len(m1),
        "m6_count": len(m6),
        "common_prefix": prefix,
        "same_position_mismatches_in_overlap": sum(a != b for a, b in zip(m1, m6)),
        "first_difference_zero_based": None if equal else prefix,
        "m1_value_at_difference": None if equal or prefix == len(m1) else m1[prefix],
        "m6_value_at_difference": None if equal or prefix == len(m6) else m6[prefix],
    }


def compare_logits(m1, m6, vocab):
    if len(m1) != 4 * vocab or len(m6) != 4 * vocab:
        raise ValueError("Unexpected final FP32 logit vector length")
    a = struct.unpack(f"<{vocab}f", m1)
    b = struct.unpack(f"<{vocab}f", m6)
    if not all(math.isfinite(x) for x in (*a, *b)):
        raise ValueError("Nonfinite final logits")
    errors = [abs(x - y) for x, y in zip(a, b)]
    first = next((i for i in range(vocab) if m1[4 * i:4 * i + 4] != m6[4 * i:4 * i + 4]), None)
    max_index = max(range(vocab), key=errors.__getitem__)
    return {
        "sha256": {"m1": sha(m1), "m6": sha(m6)},
        "vocab": vocab,
        "bitwise_equal": first is None,
        "first_bitwise_difference_id": first,
        "numerically_different": sum(x != 0 for x in errors),
        "mean_absolute_difference": math.fsum(errors) / vocab,
        "max_absolute_difference": errors[max_index],
        "max_absolute_difference_id": max_index,
        "m1_at_max_difference": a[max_index],
        "m6_at_max_difference": b[max_index],
        "m1_argmax": max(range(vocab), key=a.__getitem__),
        "m6_argmax": max(range(vocab), key=b.__getitem__),
    }


def compare(state, incoming, out):
    src = incoming / "snapshot-00000083"
    m1 = read_case(src)
    m6 = read_case(state / "pilot")
    meta = read_json(src / "snapshot.json")
    if sha(m1[0].read_bytes()) != meta["record_sha256"][RECORD] or sha(m1[4]) != meta["raw_sha256"][str(CASE_INDEX)]:
        raise ValueError("M1 reference differs from the verified snapshot")
    a, b = m1[3], m6[3]
    if a["vocab"] != b["vocab"] or a["vocab"] <= 0:
        raise ValueError("Pilot vocabulary sizes differ")
    for g in (a, b):
        if len(g["ids"]) != g["n_prompt"] + g["n_generated"] or len(g["sampled"]) != g["n_generated"]:
            raise ValueError("Unexpected generation trace length")
    prompt = compare_sequence(a["ids"][:a["n_prompt"]], b["ids"][:b["n_prompt"]])
    continuation = compare_sequence(a["ids"][a["n_prompt"]:], b["ids"][b["n_prompt"]:])
    first_sample_delta = next(
        (i for i in range(continuation["common_prefix"])
         if a["sampled"][i]["logit"] != b["sampled"][i]["logit"]),
        None,
    ) if prompt["equal"] else None
    first_probability_delta = next(
        (i for i in range(continuation["common_prefix"])
         if a["sampled"][i]["probability"] != b["sampled"][i]["probability"]),
        None,
    ) if prompt["equal"] else None
    raw = compare_sequence(m1[4], m6[4])
    raw["sha256"] = {"m1": sha(m1[4]), "m6": sha(m6[4])}
    summary = {
        "case_one_based": CASE_INDEX + 1,
        "prompt_tokens": prompt,
        "generated_tokens": continuation,
        "raw_output": raw,
        "final_logits": compare_logits(m1[5], m6[5], a["vocab"]),
        "first_paired_sample_logit_difference_zero_based": first_sample_delta,
        "first_paired_sample_probability_difference_zero_based": first_probability_delta,
        "first_paired_sample_logit_values": (
            None if first_sample_delta is None else
            {host: g["sampled"][first_sample_delta]["logit"] for host, g in (("m1", a), ("m6", b))}
        ),
        "same_generation_json": m1[1]["generation_sha256"] == m6[1]["generation_sha256"],
        "scoring": {
            host: {key: case[1][key] for key in ("tokens", "cutoff", "correct", "found", "parsed")}
            for host, case in (("m1", m1), ("m6", m6))
        },
    }
    out.mkdir(parents=True, exist_ok=False)
    for host, case in (("m1", m1), ("m6", m6)):
        target = out / host
        target.mkdir()
        shutil.copyfile(case[0], target / "case.json")
        for name in ("generation.json", "output.bin", "final-logits.f32"):
            shutil.copyfile(case[2] / name, target / name)
    result = json.dumps(summary, indent=2) + "\n"
    (out / "comparison.json").write_text(result)
    print(result, end="")


if __name__ == "__main__":
    compare(*(Path(value) for value in sys.argv[1:]))

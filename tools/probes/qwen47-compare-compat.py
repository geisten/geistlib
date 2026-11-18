#!/usr/bin/env python3
"""Prepare a bounded replay and compare every vocabulary fingerprint."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct


POSITIONS = 1191
CASES = (47, 48)
STAGES = ("ffn_input", "ffn_gate", "ffn_up", "ffn_middle", "ffn_down", "ffn")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def fixture(snapshot, index, output):
    n = index - 1
    record = json.loads((snapshot / "cases" / f"{n:08d}.json").read_bytes())
    relative = record["artifacts"]
    require(record["index"] == n and re.fullmatch(fr"artifacts/case-{n:08d}-attempt-\d+", relative),
            "Unexpected replay artifact")
    raw = (snapshot / relative / "generation.json").read_bytes()
    require(hashlib.sha256(raw).hexdigest() == record["generation_sha256"],
            "Replay artifact identity differs")
    trace = json.loads(raw)
    ids = trace["ids"][:POSITIONS + 1]
    require(len(ids) == POSITIONS + 1 and trace["n_prompt"] == 91 and
            trace["sampled"][0]["id"] == ids[91] and
            all(type(t) is int and 0 <= t < trace["vocab"] for t in ids), "Unexpected replay tokens")
    output.write_bytes(struct.pack(f"<{len(ids)}i", *ids))


def read_trace(path):
    logits, stages = {}, {}
    for line in path.read_text().splitlines():
        columns = line.split()
        require(columns, f"Empty line in {path}")
        if columns[0] == "logits":
            _, pos, digest, _ = columns
            key = int(pos)
            require(key not in logits and re.fullmatch("[a-f0-9]{16}", digest),
                    f"Invalid logit fingerprint in {path}")
            logits[key] = digest
        elif columns[0] == "trace":
            _, pos, layer, stage, count, digest = columns
            key = (int(pos), int(layer), stage)
            require(key not in stages and re.fullmatch("[a-f0-9]{16}", digest),
                    f"Invalid tensor fingerprint in {path}")
            stages[key] = int(count), digest
        else:
            raise ValueError(f"Invalid trace line in {path}")
    require(set(logits) == set(range(POSITIONS)), f"Incomplete logits in {path}")
    require(all((0, 0, stage) in stages for stage in STAGES), f"Incomplete FFN trace in {path}")
    return logits, stages


def compare(reference, actual, title):
    good = True
    for index in CASES:
        name = f"case-{index}.txt"
        m1, m1_stages = read_trace(reference / name)
        m6, m6_stages = read_trace(actual / name)
        require(set(m1_stages) == set(m6_stages) and
                all(m1_stages[key][0] == m6_stages[key][0] for key in m1_stages),
                f"Invalid tensor stages in case {index}")
        differences = [pos for pos in m1 if m1[pos] != m6[pos]]
        first = f", first at position {differences[0]}" if differences else ""
        print(f"{title} case {index} logits {len(m1) - len(differences)}/{len(m1)} exact{first}")
        good &= not differences
        for stage in STAGES:
            key = (0, 0, stage)
            match = m1_stages[key] == m6_stages[key]
            good &= match
            print(f"  layer 0 token 0 {stage}: {'exact' if match else 'different'}")
        for pos in sorted({key[0] for key in m1_stages}):
            for key in m1_stages:
                if key[0] != pos or m1_stages[key] == m6_stages[key]:
                    continue
                print(f"  first remaining difference at token {pos}, layer {key[1]}, {key[2]}")
                break
    return good


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    make = sub.add_parser("fixture")
    make.add_argument("snapshot", type=Path)
    make.add_argument("index", choices=CASES, type=int)
    make.add_argument("output", type=Path)
    diff = sub.add_parser("compare")
    diff.add_argument("reference", type=Path)
    diff.add_argument("actual", type=Path)
    args = parser.parse_args()
    if args.operation == "fixture":
        fixture(args.snapshot, args.index, args.output)
        return 0
    compare(args.reference, args.actual / "native", "Native M6")
    return 0 if compare(args.reference, args.actual / "compatible", "M1 rounding on M6") else 1


if __name__ == "__main__":
    raise SystemExit(main())

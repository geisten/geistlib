#!/usr/bin/env python3
"""Compare the single-token replay with archived M1 trace hashes."""
import hashlib
from pathlib import Path
import sys


def read_hashes(file):
    return {int(pos): digest for pos, digest in map(str.split, file.read_text().splitlines())}


def compare(output, expected_v, expected_logits, name, vectors):
    hashes = {}
    for line in (output / f"{name}.txt").read_text().splitlines():
        cols = line.split()
        if cols and cols[0] == "logits":
            pos = int(cols[1])
            if len(cols) != 4 or pos in hashes:
                raise ValueError(f"invalid {name} logit trace")
            hashes[pos] = cols[2]
    if set(hashes) != set(expected_logits):
        raise ValueError(f"incomplete {name} logit trace")
    matches = sum(hashes[pos] == digest for pos, digest in expected_logits.items())
    print(f"{name} logits: {matches}/{len(expected_logits)} exact hashes")
    v_matches = 0
    for pos, digest in expected_v.items():
        data = (output / vectors / f"{pos}-dn_qkv_after.f32").read_bytes()
        if len(data) != 6144 * 4:
            raise ValueError(f"invalid {name} V trace")
        match = hashlib.sha256(data[4096 * 4:]).hexdigest() == digest
        v_matches += match
        print(f"{name} V at token {pos}: {'exact' if match else 'different'}")
    return matches == len(expected_logits) and v_matches == len(expected_v)


def main(diag, output):
    expected_v = read_hashes(diag / "qwen47-m1-v.txt")
    expected_logits = read_hashes(diag / "qwen47-m1-logits.txt")
    if set(expected_v) != {0, 90} or set(expected_logits) != set(range(91)):
        raise ValueError("incomplete M1 reference hashes")
    compare(output, expected_v, expected_logits, "default", "default-vectors")
    if compare(output, expected_v, expected_logits, "calibrated", "calibrated-vectors"):
        print("Case 47 prompt parity passed")
        return 0
    print("Case 47 prompt parity failed")
    return 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))

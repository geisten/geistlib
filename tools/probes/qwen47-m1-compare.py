#!/usr/bin/env python3
"""Compare the single-token replay with archived M1 trace hashes."""
import hashlib
from pathlib import Path
import sys


def read_hashes(file):
    return {int(pos): digest for pos, digest in map(str.split, file.read_text().splitlines())}


def compare(output, expected_v, expected_logits, name, vectors, expected_k, expected_state):
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
    v_matches = k_matches = 0
    for pos, digest in expected_v.items():
        data = (output / vectors / f"{pos}-dn_qkv_after.f32").read_bytes()
        if len(data) != 6144 * 4:
            raise ValueError(f"invalid {name} QKV trace")
        v_match = hashlib.sha256(data[4096 * 4:]).hexdigest() == digest
        v_matches += v_match
        print(f"{name} V at token {pos}: {'exact' if v_match else 'different'}")
        if pos in expected_k:
            k_match = hashlib.sha256(data[2048 * 4:4096 * 4]).hexdigest() == expected_k[pos]
            k_matches += k_match
            print(f"{name} K at token {pos}: {'exact' if k_match else 'different'}")
    state_matches = 0
    for pos, digest in expected_state.items():
        data = (output / vectors / f"{pos}-dn_state.f32").read_bytes()
        if len(data) != 16 * 128 * 128 * 4:
            raise ValueError(f"invalid {name} DeltaNet state trace")
        match = hashlib.sha256(data).hexdigest() == digest
        state_matches += match
        print(f"{name} state at token {pos}: {'exact' if match else 'different'}")
    return (matches == len(expected_logits) and v_matches == len(expected_v) and
            k_matches == len(expected_k) and state_matches == len(expected_state))


def main(diag, output, experiment="m1-silu"):
    beta_k = experiment in ("m1-beta-k", "m1-state")
    expected_v = read_hashes(diag / "qwen47-m1-v.txt")
    expected_logits = read_hashes(diag / "qwen47-m1-logits.txt")
    if set(expected_v) != {0, 90} or set(expected_logits) != set(range(91)):
        raise ValueError("incomplete M1 reference hashes")
    expected_k = read_hashes(diag / "qwen47-m1-k.txt") if beta_k else {}
    if beta_k and set(expected_k) != {0, 90}:
        raise ValueError("incomplete M1 K reference hashes")
    expected_state = read_hashes(diag / "qwen47-m1-state.txt") if experiment == "m1-state" else {}
    if experiment == "m1-state" and set(expected_state) != {0}:
        raise ValueError("incomplete M1 state reference hash")
    compare(output, expected_v, expected_logits, "default", "default-vectors", expected_k, expected_state)
    if compare(output, expected_v, expected_logits, "calibrated", "calibrated-vectors", expected_k, expected_state):
        print("Case 47 prompt parity passed")
        return 0
    print("Case 47 prompt parity failed")
    return 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2]), *sys.argv[3:]))

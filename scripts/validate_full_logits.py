#!/usr/bin/env python3
"""validate_full_logits.py — Sub-Task E final: full forward + LM head logit parity.

Validates against `dumps[T*]['logits']` AND checks greedy top-1 token agreement.
"""
import subprocess, sys, tempfile
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent.parent
DUMPS = ROOT_DIR / "dumps"
SAFETENSORS = ROOT_DIR / "gemma-4-E2B-it" / "model.safetensors"
TEST_BIN = ROOT_DIR / "test_full_logits"
VOCAB = 262144


def run_one(inp):
    dump = np.load(DUMPS / f"{inp}.npz")
    ids = dump["input_ids"].astype(np.int32)
    expected = dump["logits"].reshape(-1, VOCAB).astype(np.float32)
    seq = len(ids)
    print(f"\n=== {inp}: seq_len={seq}, vocab={VOCAB} ===")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ids_path = tmp / "ids.bin"; out_path = tmp / "logits.bin"
        ids_path.write_bytes(ids.tobytes())
        proc = subprocess.run(
            [str(TEST_BIN), str(SAFETENSORS), str(ids_path), str(out_path)],
            capture_output=True, text=True, check=False)
        if proc.returncode != 0: print("STDERR:", proc.stderr); return False
        # surface backend progress
        for line in proc.stderr.strip().split("\n")[-5:]:
            print(f"  {line}")
        c_logits = np.frombuffer(out_path.read_bytes(), dtype=np.float32).reshape(seq, VOCAB)

    # Per-token diff stats
    diff = c_logits - expected; abs_diff = np.abs(diff)
    a64 = c_logits.reshape(-1).astype(np.float64)
    b64 = expected.reshape(-1).astype(np.float64)
    cos = float(np.dot(a64, b64) / (np.linalg.norm(a64) * np.linalg.norm(b64) + 1e-30))
    rel_frob = float(np.linalg.norm(diff) / (np.linalg.norm(expected) + 1e-30))
    print(f"  logits  rel_frob={rel_frob:.4e}  cos={cos:.10f}  max|d|={abs_diff.max():.4e}")

    # Top-1 token agreement (the metric that actually matters)
    c_top1 = np.argmax(c_logits, axis=-1)
    e_top1 = np.argmax(expected, axis=-1)
    matches = int((c_top1 == e_top1).sum())
    print(f"  top-1 agreement: {matches}/{seq} positions ({100*matches/seq:.1f}%)")
    if matches < seq:
        for i in range(seq):
            if c_top1[i] != e_top1[i]:
                print(f"    pos {i}: c={c_top1[i]} expected={e_top1[i]}")

    # Top-5 overlap (more tolerant)
    c_top5 = np.argpartition(-c_logits, 5, axis=-1)[:, :5]
    e_top5 = np.argpartition(-expected, 5, axis=-1)[:, :5]
    overlap = sum(len(set(c_top5[i]) & set(e_top5[i])) for i in range(seq))
    print(f"  top-5 overlap:   {overlap}/{seq*5} ({100*overlap/(seq*5):.1f}%)")

    return matches == seq  # success = perfect top-1 agreement


def main():
    ok = True
    for inp in ("T1", "T2", "T3", "T4", "T5"):
        ok &= run_one(inp)
    print("\n" + ("✓ ALL PASS — full forward end-to-end" if ok else "⚠ Some top-1 disagreements (see above)"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

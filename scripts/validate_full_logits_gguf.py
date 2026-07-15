#!/usr/bin/env python3
"""validate_full_logits_gguf.py — Tier-1: GGUF (Q4_K_M / F16 / Q8_0) inference
parity vs HF FP32 reference. Run on T1-T5.
"""
import argparse, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

SCRIPT = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT.parent.parent
DUMPS = ROOT_DIR / "dumps"
TB = ROOT_DIR / "test_full_logits_gguf"
VOCAB = 262144


def run_one(gguf_path, inp):
    dump = np.load(DUMPS / f"{inp}.npz")
    ids = dump["input_ids"].astype(np.int32)
    expected = dump["logits"].reshape(-1, VOCAB).astype(np.float32)
    print(f"\n=== {inp}: seq_len={len(ids)}  vs HF FP32 ===")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp); ip = tmp/"i.bin"; op = tmp/"o.bin"
        ip.write_bytes(ids.tobytes())
        proc = subprocess.run(
            [str(TB), str(gguf_path), str(ip), str(op)],
            capture_output=True, text=True, timeout=300)
        if proc.returncode != 0: print("STDERR:", proc.stderr); return False
        c = np.frombuffer(op.read_bytes(), dtype=np.float32).reshape(-1, VOCAB)
    diff = c - expected
    a64 = c.reshape(-1).astype(np.float64); b64 = expected.reshape(-1).astype(np.float64)
    cos = float(np.dot(a64, b64) / (np.linalg.norm(a64) * np.linalg.norm(b64) + 1e-30))
    rel = float(np.linalg.norm(diff) / (np.linalg.norm(expected) + 1e-30))
    print(f"  logits  cos={cos:.10f}  rel_frob={rel:.4e}  max|d|={np.abs(diff).max():.4e}")
    c_top1 = np.argmax(c, axis=-1)
    e_top1 = np.argmax(expected, axis=-1)
    matches = int((c_top1 == e_top1).sum())
    print(f"  top-1 agreement: {matches}/{len(ids)} ({100*matches/len(ids):.1f}%)")
    if matches < len(ids):
        wrong = np.where(c_top1 != e_top1)[0][:5]
        for w in wrong:
            print(f"    pos {int(w)}: c={int(c_top1[w])} e={int(e_top1[w])}")
    c5 = np.argpartition(-c, 5, axis=-1)[:, :5]
    e5 = np.argpartition(-expected, 5, axis=-1)[:, :5]
    overlap = sum(len(set(c5[i]) & set(e5[i])) for i in range(len(ids)))
    print(f"  top-5 overlap:   {overlap}/{len(ids)*5} ({100*overlap/(len(ids)*5):.1f}%)")
    return matches >= int(0.95 * len(ids))      # 95% threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf_path")
    parser.add_argument("--inputs", nargs="*", default=["T1", "T2", "T3", "T4", "T5"])
    args = parser.parse_args()
    print(f"Model: {args.gguf_path}")
    ok = True
    for inp in args.inputs:
        ok &= run_one(args.gguf_path, inp)
    print("\n" + ("✓ Tier-1 PASS (≥95% top-1)" if ok else "⚠ some failures"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

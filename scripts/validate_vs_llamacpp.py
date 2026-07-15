#!/usr/bin/env python3
"""validate_vs_llamacpp.py — Tier-2: our Q4_K_M inference vs llama.cpp's
Q4_K_M inference on the same GGUF file. Apples-to-apples engine comparison.
"""
import argparse, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

SCRIPT = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT.parent.parent
DUMPS = ROOT_DIR / "dumps"
TB_OURS = ROOT_DIR / "test_full_logits_gguf"
TB_LLAMA = ROOT_DIR / "dump_llamacpp_logits"
VOCAB = 262144


def run_one(gguf, inp):
    dump = np.load(DUMPS / f"{inp}.npz")
    ids = dump["input_ids"].astype(np.int32)
    print(f"\n=== {inp}: seq_len={len(ids)} ===")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp); ip = tmp/"i.bin"
        op_ours = tmp/"ours.bin"; op_llama = tmp/"llama.bin"
        ip.write_bytes(ids.tobytes())
        # Ours
        proc = subprocess.run([str(TB_OURS), str(gguf), str(ip), str(op_ours)],
                              capture_output=True, text=True, timeout=300)
        if proc.returncode != 0: print("OURS STDERR:", proc.stderr[-300:]); return False
        ours = np.frombuffer(op_ours.read_bytes(), dtype=np.float32).reshape(-1, VOCAB)
        # llama.cpp
        proc = subprocess.run([str(TB_LLAMA), str(gguf), str(ip), str(op_llama)],
                              capture_output=True, text=True, timeout=300)
        if proc.returncode != 0: print("LLAMA STDERR:", proc.stderr[-300:]); return False
        llama = np.frombuffer(op_llama.read_bytes(), dtype=np.float32).reshape(-1, VOCAB)

    diff = ours - llama
    a64 = ours.reshape(-1).astype(np.float64); b64 = llama.reshape(-1).astype(np.float64)
    cos = float(np.dot(a64, b64) / (np.linalg.norm(a64) * np.linalg.norm(b64) + 1e-30))
    rel = float(np.linalg.norm(diff) / (np.linalg.norm(llama) + 1e-30))
    print(f"  logits  cos={cos:.10f}  rel_frob={rel:.4e}  max|d|={np.abs(diff).max():.4e}")
    o_top1 = np.argmax(ours, axis=-1); l_top1 = np.argmax(llama, axis=-1)
    matches = int((o_top1 == l_top1).sum())
    print(f"  top-1 vs llama.cpp: {matches}/{len(ids)} ({100*matches/len(ids):.1f}%)")
    if matches < len(ids):
        for w in np.where(o_top1 != l_top1)[0][:5]:
            print(f"    pos {int(w)}: ours={int(o_top1[w])} llama={int(l_top1[w])}")
    o5 = np.argpartition(-ours, 5, axis=-1)[:, :5]
    l5 = np.argpartition(-llama, 5, axis=-1)[:, :5]
    overlap = sum(len(set(o5[i]) & set(l5[i])) for i in range(len(ids)))
    print(f"  top-5 overlap:      {overlap}/{len(ids)*5} ({100*overlap/(len(ids)*5):.1f}%)")
    return matches >= int(0.99 * len(ids))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf_path")
    parser.add_argument("--inputs", nargs="*", default=["T1", "T2", "T3", "T4", "T5"])
    args = parser.parse_args()
    print(f"Tier-2: ours vs llama.cpp on {args.gguf_path}")
    ok = True
    for inp in args.inputs: ok &= run_one(args.gguf_path, inp)
    print("\n" + ("✓ Tier-2 PASS — engine parity confirmed" if ok else "⚠ disagreements"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

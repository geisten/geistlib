#!/usr/bin/env python3
"""validate_step5.py — RoPE + sliding-window MQA attention + o_proj parity.

Validates against o_proj hook for both T1 (1 token, RoPE = identity) and
T2 (8 tokens, RoPE active). T1-only failure → attention/o_proj bug.
T1 pass + T2 fail → RoPE bug.
"""
import subprocess, sys, tempfile
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent.parent
DUMPS = ROOT_DIR / "dumps"
SAFETENSORS_PATH = ROOT_DIR / "gemma-4-E2B-it" / "model.safetensors"
TEST_BIN = ROOT_DIR / "test_step5_attn_oproj"


def diff_stats(a, b, label):
    if a.shape != b.shape:
        print(f"  ✗ shape mismatch: {a.shape} vs {b.shape}"); return False
    diff = a - b; abs_diff = np.abs(diff)
    a64 = a.reshape(-1).astype(np.float64); b64 = b.reshape(-1).astype(np.float64)
    cos = float(np.dot(a64, b64) / (np.linalg.norm(a64) * np.linalg.norm(b64) + 1e-30))
    rel_frob = float(np.linalg.norm(diff) / (np.linalg.norm(b) + 1e-30))
    print(f"  shape:      {a.shape}")
    print(f"  max |diff|: {abs_diff.max():.6e}")
    print(f"  mean|diff|: {abs_diff.mean():.6e}")
    print(f"  cos sim:    {cos:.10f}")
    print(f"  rel frob:   {rel_frob:.6e}")
    ok = rel_frob < 5e-3 and cos > 0.9999
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}  ({label})")
    return ok


def run_one(input_label):
    dump = np.load(DUMPS / f"{input_label}.npz")
    input_ids = dump["input_ids"].astype(np.int32)
    expected = dump["layer_00_self_attn_o_proj"].reshape(-1, 1536).astype(np.float32)
    print(f"\n=== {input_label}: seq_len={len(input_ids)} ===")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ids_path = tmp / "ids.bin"; out_path = tmp / "out.bin"
        ids_path.write_bytes(input_ids.tobytes())
        proc = subprocess.run(
            [str(TEST_BIN), str(SAFETENSORS_PATH), str(ids_path), str(out_path)],
            capture_output=True, text=True, check=False)
        if proc.returncode != 0: print("STDERR:", proc.stderr); return False
        print("  ", proc.stderr.strip().replace("\n", "\n   "))
        c_out = np.frombuffer(out_path.read_bytes(), dtype=np.float32).reshape(-1, 1536)
    return diff_stats(c_out, expected, f"{input_label}/o_proj")


def main():
    ok = True
    for inp in ("T1", "T2", "T3", "T4", "T5"):
        ok &= run_one(inp)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

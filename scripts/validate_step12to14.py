#!/usr/bin/env python3
"""validate_step12to14.py — PLE pre-compute + per-layer merge + layer_scalar.

Validates 5 hooks: ple_lookup, per_layer_input_gate, per_layer_projection,
post_per_layer_input_norm, layer_00_output (final).
"""
import subprocess, sys, tempfile
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent.parent
DUMPS = ROOT_DIR / "dumps"
SAFETENSORS = ROOT_DIR / "gemma-4-E2B-it" / "model.safetensors"
TEST_BIN = ROOT_DIR / "test_step12to14_ple"

HOOKS = [
    ("ple_lookup",                 "ple_lookup",                              8960),
    ("per_layer_input_gate",       "layer_00_per_layer_input_gate",            256),
    ("per_layer_projection",       "layer_00_per_layer_projection",           1536),
    ("post_per_layer_input_norm",  "layer_00_post_per_layer_input_norm",      1536),
    ("layer_output",               "layer_00_output",                         1536),
]


def diff_stats(a, b, label):
    if a.shape != b.shape:
        print(f"      ✗ shape mismatch: {a.shape} vs {b.shape}"); return False
    diff = a - b; abs_diff = np.abs(diff)
    a64 = a.reshape(-1).astype(np.float64); b64 = b.reshape(-1).astype(np.float64)
    cos = float(np.dot(a64, b64) / (np.linalg.norm(a64) * np.linalg.norm(b64) + 1e-30))
    rel_frob = float(np.linalg.norm(diff) / (np.linalg.norm(b) + 1e-30))
    # 1% threshold for accumulated checkpoints (Layer 0 output is end-of-chain).
    ok = rel_frob < 1e-2 and cos > 0.9999
    flag = "✓" if ok else "✗"
    print(f"      {flag} {label:28s}  rel_frob={rel_frob:.4e}  cos={cos:.10f}  max|d|={abs_diff.max():.4e}")
    return ok


def run_one(inp):
    dump = np.load(DUMPS / f"{inp}.npz")
    ids = dump["input_ids"].astype(np.int32)
    print(f"\n  === {inp}: seq_len={len(ids)} ===")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ids_path = tmp / "ids.bin"; out_pref = str(tmp / "o")
        ids_path.write_bytes(ids.tobytes())
        proc = subprocess.run(
            [str(TEST_BIN), str(SAFETENSORS), str(ids_path), out_pref],
            capture_output=True, text=True, check=False)
        if proc.returncode != 0: print("STDERR:", proc.stderr); return False
        ok = True
        for fname, hookname, dim in HOOKS:
            cbytes = Path(out_pref + f".{fname}.bin").read_bytes()
            c_arr = np.frombuffer(cbytes, dtype=np.float32).reshape(-1, dim)
            expected = dump[hookname].reshape(-1, dim).astype(np.float32)
            ok &= diff_stats(c_arr, expected, fname)
        return ok


def main():
    ok = True
    for inp in ("T1", "T2", "T3", "T4", "T5"):
        ok &= run_one(inp)
    print("\n" + ("✓ ALL PASS — Layer 0 fully validated!" if ok else "✗ FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

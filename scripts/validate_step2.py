#!/usr/bin/env python3
"""
validate_step2.py — checks input_layernorm output against HF reference.

Run:
    source ../gemma-4-E2B-it/.venv/bin/activate
    python validate_step2.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent.parent
DUMP_PATH = ROOT_DIR / "dumps" / "T1.npz"
SAFETENSORS_PATH = ROOT_DIR / "gemma-4-E2B-it" / "model.safetensors"
TEST_BIN = ROOT_DIR / "test_step2_input_layernorm"


def diff_stats(a, b, label):
    if a.shape != b.shape:
        print(f"  ✗ shape mismatch: {a.shape} vs {b.shape}")
        return False
    diff = a - b
    abs_diff = np.abs(diff)
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


def main():
    dump = np.load(DUMP_PATH)
    input_ids = dump["input_ids"].astype(np.int32)
    expected = dump["layer_00_input_layernorm"].reshape(-1, 1536).astype(np.float32)
    print(f"input_ids: {input_ids.tolist()}, expected shape {expected.shape}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ids_path = tmp / "ids.bin"; out_path = tmp / "out.bin"
        ids_path.write_bytes(input_ids.tobytes())
        proc = subprocess.run(
            [str(TEST_BIN), str(SAFETENSORS_PATH), str(ids_path), str(out_path)],
            capture_output=True, text=True, check=False,
        )
        if proc.returncode != 0:
            print("STDERR:", proc.stderr); return 1
        print("  ", proc.stderr.strip().replace("\n", "\n   "))
        c_out = np.frombuffer(out_path.read_bytes(), dtype=np.float32).reshape(-1, 1536)

    print()
    return 0 if diff_stats(c_out, expected, "input_layernorm") else 1


if __name__ == "__main__":
    sys.exit(main())

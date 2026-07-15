#!/usr/bin/env python3
"""validate_step4.py — Q/K/V proj + per-head RMSNorm parity check (3 tensors)."""
import subprocess, sys, tempfile
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent.parent
DUMP_PATH = ROOT_DIR / "dumps" / "T1.npz"
SAFETENSORS_PATH = ROOT_DIR / "gemma-4-E2B-it" / "model.safetensors"
TEST_BIN = ROOT_DIR / "test_step4_qkv_norm"


def diff_stats(a, b, label):
    if a.shape != b.shape:
        print(f"  ✗ shape mismatch: {a.shape} vs {b.shape}"); return False
    diff = a - b; abs_diff = np.abs(diff)
    a64 = a.reshape(-1).astype(np.float64); b64 = b.reshape(-1).astype(np.float64)
    cos = float(np.dot(a64, b64) / (np.linalg.norm(a64) * np.linalg.norm(b64) + 1e-30))
    rel_frob = float(np.linalg.norm(diff) / (np.linalg.norm(b) + 1e-30))
    print(f"  {label}:")
    print(f"    shape:      {a.shape}")
    print(f"    max |diff|: {abs_diff.max():.6e}")
    print(f"    mean|diff|: {abs_diff.mean():.6e}")
    print(f"    cos sim:    {cos:.10f}")
    print(f"    rel frob:   {rel_frob:.6e}")
    ok = rel_frob < 5e-3 and cos > 0.9999
    print(f"    {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


def main():
    dump = np.load(DUMP_PATH)
    input_ids = dump["input_ids"].astype(np.int32)

    expected_q = dump["layer_00_self_attn_q_norm"].reshape(-1, 8, 256).astype(np.float32)
    expected_k = dump["layer_00_self_attn_k_norm"].reshape(-1, 1, 256).astype(np.float32)
    expected_v = dump["layer_00_self_attn_v_norm"].reshape(-1, 1, 256).astype(np.float32)
    print(f"input_ids: {input_ids.tolist()}")
    print(f"expected: q{expected_q.shape} k{expected_k.shape} v{expected_v.shape}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ids_path = tmp / "ids.bin"; out_prefix = str(tmp / "out")
        ids_path.write_bytes(input_ids.tobytes())
        proc = subprocess.run(
            [str(TEST_BIN), str(SAFETENSORS_PATH), str(ids_path), out_prefix],
            capture_output=True, text=True, check=False)
        if proc.returncode != 0: print("STDERR:", proc.stderr); return 1
        print("  ", proc.stderr.strip().replace("\n", "\n   "))

        c_q = np.frombuffer(Path(out_prefix + ".q.bin").read_bytes(), dtype=np.float32).reshape(-1, 8, 256)
        c_k = np.frombuffer(Path(out_prefix + ".k.bin").read_bytes(), dtype=np.float32).reshape(-1, 1, 256)
        c_v = np.frombuffer(Path(out_prefix + ".v.bin").read_bytes(), dtype=np.float32).reshape(-1, 1, 256)

    print()
    ok = True
    ok &= diff_stats(c_q, expected_q, "q_norm")
    ok &= diff_stats(c_k, expected_k, "k_norm")
    ok &= diff_stats(c_v, expected_v, "v_norm")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

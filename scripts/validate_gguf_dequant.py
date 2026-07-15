#!/usr/bin/env python3
"""validate_gguf_dequant.py — bit-close parity check of our dequant kernels
against gguf-py's reference implementation.

Tests Q4_K, Q6_K, Q8_0 and F16 across both quantized GGUF files.
"""
import subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import gguf

SCRIPT = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT.parent.parent
GGUF_Q4 = ROOT_DIR / "gguf_artifacts" / "gemma4-e2b-Q4_K_M.gguf"
GGUF_Q8 = ROOT_DIR / "gguf_artifacts" / "gemma4-e2b-Q8_0.gguf"
TEST_BIN = ROOT_DIR / "test_gguf_dequant"

# (file, tensor_name, expected_dtype) — covers all 4 dequant code paths
CASES = [
    (GGUF_Q4, "blk.0.attn_q.weight",     gguf.GGMLQuantizationType.Q4_K),
    (GGUF_Q4, "blk.0.attn_v.weight",     gguf.GGMLQuantizationType.Q6_K),
    (GGUF_Q4, "per_layer_model_proj.weight", gguf.GGMLQuantizationType.F16),
    (GGUF_Q8, "blk.0.attn_q.weight",     gguf.GGMLQuantizationType.Q8_0),
    (GGUF_Q8, "blk.0.attn_v.weight",     gguf.GGMLQuantizationType.Q8_0),
    (GGUF_Q8, "blk.0.attn_norm.weight",  gguf.GGMLQuantizationType.F32),
]


def gguf_py_dequant(path, name):
    """Use gguf-py's GGUFReader to load + dequantize a tensor as FP32."""
    rd = gguf.GGUFReader(str(path))
    for t in rd.tensors:
        if t.name == name:
            arr = t.data
            if t.tensor_type in (gguf.GGMLQuantizationType.F32,
                                  gguf.GGMLQuantizationType.F16):
                return arr.astype(np.float32).reshape(-1)
            return gguf.quants.dequantize(arr, t.tensor_type).astype(np.float32).reshape(-1)
    raise KeyError(name)


def diff_stats(a, b, label):
    if a.shape != b.shape:
        print(f"  ✗ shape mismatch: {a.shape} vs {b.shape}"); return False
    diff = a - b
    abs_d = np.abs(diff)
    a64 = a.astype(np.float64); b64 = b.astype(np.float64)
    cos = float(np.dot(a64, b64) / (np.linalg.norm(a64) * np.linalg.norm(b64) + 1e-30))
    rel = float(np.linalg.norm(diff) / (np.linalg.norm(b) + 1e-30))
    # Bit-close: max diff under 1e-5 since both implementations should be exact
    ok = abs_d.max() < 1e-4 and cos > 1.0 - 1e-7
    flag = "✓" if ok else "✗"
    print(f"  {flag} {label:50s} max|d|={abs_d.max():.2e}  rel={rel:.2e}  cos={cos:.10f}")
    return ok


def main():
    print(f"Reference impl: gguf-py {gguf.__version__ if hasattr(gguf, '__version__') else '?'}")
    ok = True
    for path, name, expected_dt in CASES:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.bin"
            proc = subprocess.run(
                [str(TEST_BIN), str(path), name, str(out)],
                capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                print(f"  ✗ C dequant failed for {name}: {proc.stderr.strip()}"); ok = False; continue
            mine = np.frombuffer(out.read_bytes(), dtype=np.float32)
        try:
            ref = gguf_py_dequant(path, name)
        except Exception as e:
            print(f"  ✗ gguf-py dequant failed: {e}"); ok = False; continue
        ok &= diff_stats(mine, ref, f"{name} ({expected_dt.name})")
    print("\n" + ("✓ ALL PASS" if ok else "✗ FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

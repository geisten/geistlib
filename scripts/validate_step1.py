#!/usr/bin/env python3
"""
validate_step1.py — orchestrates the embedding-lookup parity test.

Workflow:
  1. Loads dumps/T1.npz, extracts input_ids, dumps as int32 binary
  2. Invokes ./test_step1_embedding to compute the C-side embedding
  3. Compares C output (FP32) against dumps[T1]['token_embed'] (FP32)
  4. Prints diff stats: max abs error, mean abs error, cosine similarity,
     relative Frobenius error

Run:
    source ../gemma-4-E2B-it/.venv/bin/activate
    python validate_step1.py
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
TEST_BIN = ROOT_DIR / "test_step1_embedding"


def diff_stats(a: np.ndarray, b: np.ndarray, label: str) -> bool:
    if a.shape != b.shape:
        print(f"  ✗ shape mismatch: {a.shape} vs {b.shape}")
        return False
    diff = a - b
    abs_diff = np.abs(diff)
    max_abs = abs_diff.max()
    mean_abs = abs_diff.mean()
    a_flat = a.reshape(-1).astype(np.float64)
    b_flat = b.reshape(-1).astype(np.float64)
    cos = float(np.dot(a_flat, b_flat) /
                (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-30))
    rel_frob = float(np.linalg.norm(diff) / (np.linalg.norm(b) + 1e-30))

    print(f"  shape:      {a.shape}")
    print(f"  max |diff|: {max_abs:.6e}")
    print(f"  mean|diff|: {mean_abs:.6e}")
    print(f"  cos sim:    {cos:.10f}")
    print(f"  rel frob:   {rel_frob:.6e}")

    # Strategy C threshold: rel_frob < 0.5%, cos > 0.9999
    ok = rel_frob < 5e-3 and cos > 0.9999
    status = "✓ PASS" if ok else "✗ FAIL"
    print(f"  {status}  ({label})")
    return ok


def main():
    if not DUMP_PATH.exists():
        print(f"missing {DUMP_PATH}", file=sys.stderr)
        return 1
    if not TEST_BIN.exists():
        print(f"missing {TEST_BIN} — build first:", file=sys.stderr)
        print(f"  cc -std=c23 -Wall -Wextra -O2 \\\n"
              f"     safetensors_reader.c test_step1_embedding.c \\\n"
              f"     -o test_step1_embedding", file=sys.stderr)
        return 1

    print(f"Loading {DUMP_PATH}...")
    dump = np.load(DUMP_PATH)
    input_ids = dump["input_ids"].astype(np.int32)
    expected = dump["token_embed"]
    print(f"  input_ids shape: {input_ids.shape}, dtype: {input_ids.dtype}")
    print(f"  token_embed shape: {expected.shape}, dtype: {expected.dtype}")
    print(f"  ids: {input_ids.tolist()}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ids_path = tmp / "input_ids.bin"
        out_path = tmp / "embed_out.bin"
        ids_path.write_bytes(input_ids.tobytes())

        print(f"\nRunning {TEST_BIN.name}...")
        proc = subprocess.run(
            [str(TEST_BIN), str(SAFETENSORS_PATH), str(ids_path), str(out_path)],
            capture_output=True, text=True, check=False,
        )
        if proc.returncode != 0:
            print("STDERR:", proc.stderr, file=sys.stderr)
            return 1
        print("  ", proc.stderr.strip().replace("\n", "\n   "))

        c_out = np.frombuffer(out_path.read_bytes(), dtype=np.float32)

    # The dump's token_embed has shape [seq_len, hidden]; ours is the same
    # flattened. Reshape to compare.
    c_out = c_out.reshape(-1, expected.shape[-1])
    if c_out.shape[0] != expected.shape[0]:
        print(f"\n✗ row count mismatch: C={c_out.shape[0]} expected={expected.shape[0]}",
              file=sys.stderr)
        return 1

    # Dumps include the batch dimension (1, seq, hidden); collapse it.
    expected = expected.reshape(-1, expected.shape[-1])

    print("\n=== Comparison: token embedding (after Gemma4TextScaledWordEmbedding) ===")
    ok = diff_stats(c_out, expected.astype(np.float32), "token_embed")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

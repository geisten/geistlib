#!/usr/bin/env python3
"""eval_embedding_fidelity.py — cosine-similarity gate for embedding models.

Compares two sets of sentence embeddings for the SAME prompts in the SAME
order and fails if any pair drifts below a floor. Two uses, one mechanism:

  1. Parity against upstream. geistlib's vectors vs the same prompts run
     through bitnet.cpp's llama-embedding. This is the oracle phases 1 and 2
     are missing — until it passes, "the forward path is correct" is a
     hypothesis (docs/BITNET_EMBEDDINGS_PLAN.md).

  2. Conversion fidelity. F16 GGUF vs I2_S GGUF from the same checkpoint,
     which is what upstream reports as a 0.0032 average delta on the 0.6B.

Cosine similarity is the right measure and not just a convenient one: these
vectors are L2-normalised and consumed by dot product, so a difference that
does not move the cosine does not move any downstream retrieval result.

Inputs are .gemb files written by tools/dump_geist_embedding, or .npy
arrays of shape (count, dim) for whatever produced the reference.

    python3 tools/eval_embedding_fidelity.py --ref up.npy --got geist.gemb
    python3 tools/eval_embedding_fidelity.py --selftest

Exit status is the gate: 0 pass, 1 fail, 2 usage error.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

GEMB_MAGIC = b"GEMB"
GEMB_VERSION = 1
# The plan's phase 2 bar. Well above float32 round-off between two
# implementations of the same arithmetic, well below anything a real
# architectural mistake would survive.
DEFAULT_MIN_COSINE = 0.999


def read_gemb(path: Path) -> np.ndarray:
    """Read a .gemb file into an (count, dim) float32 array."""
    raw = path.read_bytes()
    if len(raw) < 16 or raw[:4] != GEMB_MAGIC:
        raise SystemExit(f"{path}: not a .gemb file")
    version, count, dim = struct.unpack("<III", raw[4:16])
    if version != GEMB_VERSION:
        raise SystemExit(f"{path}: .gemb version {version}, expected {GEMB_VERSION}")
    want = count * dim * 4
    if len(raw) - 16 != want:
        raise SystemExit(
            f"{path}: header says {count}x{dim} ({want} bytes) but payload is {len(raw) - 16}"
        )
    return np.frombuffer(raw, dtype=np.float32, count=count * dim, offset=16).reshape(count, dim)


def load(path: Path) -> np.ndarray:
    arr = np.load(path) if path.suffix == ".npy" else read_gemb(path)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise SystemExit(f"{path}: expected a 2-D (count, dim) array, got shape {arr.shape}")
    return arr


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row cosine similarity. Re-normalises rather than trusting the
    inputs to be unit-length — a producer that forgot to normalise would
    otherwise show up as a quality failure instead of the bug it is."""
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    # A zero row has no direction; call it a mismatch rather than dividing
    # by zero and reporting nan, which downstream would read as "no data".
    denom = na * nb
    out = np.zeros(a.shape[0], dtype=np.float64)
    nz = denom > 0.0
    out[nz] = np.sum(a[nz] * b[nz], axis=1) / denom[nz]
    return out


def gate(ref: np.ndarray, got: np.ndarray, min_cosine: float) -> tuple[bool, str]:
    """Returns (passed, report)."""
    if ref.shape != got.shape:
        return False, f"shape mismatch: reference {ref.shape} vs got {got.shape}"
    cos = cosine_rows(ref, got)
    worst = int(np.argmin(cos))
    lines = [
        f"pairs={len(cos)} dim={ref.shape[1]}",
        f"cosine  min={cos.min():.6f} (row {worst})  mean={cos.mean():.6f}",
        f"floor   {min_cosine:.6f}",
    ]
    below = np.flatnonzero(cos < min_cosine)
    if below.size:
        lines.append(f"FAIL: {below.size} row(s) below the floor: {below[:10].tolist()}")
        return False, "\n".join(lines)
    lines.append("PASS")
    return True, "\n".join(lines)


def selftest() -> int:
    """Exercise the gate without any model. Mirrors benchmark/perf_gate.py's
    convention so the gate itself is covered wherever data is not."""
    rng = np.random.default_rng(20260905)
    base = rng.standard_normal((8, 64))
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    fails = 0

    def check(name: str, ok: bool) -> None:
        nonlocal fails
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            fails += 1

    passed, _ = gate(base, base.copy(), DEFAULT_MIN_COSINE)
    check("identical inputs pass", passed)

    tiny = base + rng.standard_normal(base.shape) * 1e-6
    passed, _ = gate(base, tiny, DEFAULT_MIN_COSINE)
    check("float-noise-sized drift passes", passed)

    broken = base.copy()
    broken[3] = rng.standard_normal(64)
    passed, report = gate(base, broken, DEFAULT_MIN_COSINE)
    check("one wrong row fails", not passed and "row" in report)

    passed, report = gate(base, base[:4], DEFAULT_MIN_COSINE)
    check("shape mismatch fails rather than comparing a prefix", not passed)

    # Scale invariance: cosine must not care, so an un-normalised producer
    # is still comparable instead of failing for the wrong reason.
    passed, _ = gate(base, base * 7.5, DEFAULT_MIN_COSINE)
    check("uniform rescaling does not move the cosine", passed)

    zeroed = base.copy()
    zeroed[0] = 0.0
    passed, _ = gate(base, zeroed, DEFAULT_MIN_COSINE)
    check("a zero row fails instead of producing nan", not passed)

    print("selftest: FAIL" if fails else "selftest: PASS")
    return 1 if fails else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", type=Path, help="reference embeddings (.gemb or .npy)")
    ap.add_argument("--got", type=Path, help="embeddings under test (.gemb or .npy)")
    ap.add_argument("--min-cosine", type=float, default=DEFAULT_MIN_COSINE)
    ap.add_argument("--selftest", action="store_true", help="check the gate, no data needed")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if args.ref is None or args.got is None:
        ap.error("--ref and --got are required unless --selftest is given")

    passed, report = gate(load(args.ref), load(args.got), args.min_cosine)
    print(report)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

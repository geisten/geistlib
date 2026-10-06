#!/usr/bin/env python3
"""tests/test_ptqtp_p0_paper_parity.py — PTQTP paper-parity gate.

Asserts that the clean-room PTQTP implementation in
`tools/ptqtp_single_tensor.py` meets two acceptance criteria:

    1. Cosine similarity ≥ 0.95 of reconstruction vs. original on a
       representative Gemma-class attention-Q tensor shape.
    2. Bit-identical determinism: same input + seed → identical
       (T1, T2, alpha) across two independent calls.

Hermetic: no GGUF, no internet, no Hugging Face. Generates a synthetic
tensor with realistic statistics (zero-mean Gaussian, std ≈ 0.02, shape
matching Gemma 4 E2B `blk.0.attn_q.weight`) under a fixed numpy seed.

Exit codes:
    0  — both gates pass
    1  — at least one gate fails

Wired into `make test` via the test runner; can be invoked directly:
    python3 tests/test_ptqtp_p0_paper_parity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add tools/ to sys.path so the algorithm module is importable.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "tools"))

from ptqtp_single_tensor import (  # noqa: E402
    ptqtp_quantize,
    cosine_sim,
    relative_mse,
)
from ptqtp_version import ALGORITHM_VERSION  # noqa: E402


# Gemma 4 E2B attention-Q tensor shape; weight statistics are realistic
# (zero-mean, std≈0.02 from FP16 attention-Q in published 2B checkpoints).
SHAPE = (1024, 1024)   # subsampled from (2560, 2560) for CI runtime budget
SEED  = 0
STD   = 0.02

# Acceptance threshold for criterion 1.
COS_SIM_THRESHOLD = 0.95


def _make_fixture() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    W = rng.normal(0.0, STD, size=SHAPE).astype(np.float32)
    return W


def _gate_cos_sim(W: np.ndarray) -> tuple[bool, float, dict]:
    T1, T2, alpha = ptqtp_quantize(
        W, group_size=128, max_iter=50, tol=1e-4, verbose=False,
    )
    n_out, n_in = W.shape
    n_groups = n_in // 128
    T1g = T1.reshape(n_out, n_groups, 128).astype(np.float32)
    T2g = T2.reshape(n_out, n_groups, 128).astype(np.float32)
    Wrec = (
        alpha[..., 0:1] * T1g + alpha[..., 1:2] * T2g
    ).reshape(n_out, n_in)
    cs = cosine_sim(W, Wrec)
    mse = relative_mse(W, Wrec)
    return cs >= COS_SIM_THRESHOLD, cs, {"rel_mse": mse,
                                          "alpha_mean_abs": float(np.abs(alpha).mean())}


def _gate_determinism(W: np.ndarray) -> tuple[bool, dict]:
    T1a, T2a, alphaA = ptqtp_quantize(W, group_size=128, max_iter=50,
                                       tol=1e-4, verbose=False)
    T1b, T2b, alphaB = ptqtp_quantize(W, group_size=128, max_iter=50,
                                       tol=1e-4, verbose=False)
    t1_eq    = bool(np.array_equal(T1a, T1b))
    t2_eq    = bool(np.array_equal(T2a, T2b))
    alpha_eq = bool(np.array_equal(alphaA, alphaB))
    return (t1_eq and t2_eq and alpha_eq,
            {"T1_equal": t1_eq, "T2_equal": t2_eq, "alpha_equal": alpha_eq})


def main() -> int:
    print("[P0] PTQTP paper-parity gate")
    print(f"  algorithm version: {ALGORITHM_VERSION}")
    print(f"  fixture: shape={SHAPE}, std={STD}, seed={SEED}")

    W = _make_fixture()
    print(f"  generated weights: mean={W.mean():+.6f} "
          f"std={W.std():.6f} max|w|={np.abs(W).max():.4f}")

    print(f"\n[P0.1] cos sim ≥ {COS_SIM_THRESHOLD}")
    ok_cos, cs, diag = _gate_cos_sim(W)
    marker = "✓" if ok_cos else "✗"
    print(f"  {marker} cos sim = {cs:.6f}  (threshold {COS_SIM_THRESHOLD})")
    print(f"    rel_mse        = {diag['rel_mse']:.6f}")
    print(f"    mean |alpha|   = {diag['alpha_mean_abs']:.6f}")

    print("\n[P0.2] deterministic re-run produces bit-identical output")
    ok_det, det_diag = _gate_determinism(W)
    marker = "✓" if ok_det else "✗"
    print(f"  {marker} {det_diag}")

    print()
    if ok_cos and ok_det:
        print("PASS — P0 gate satisfied.")
        return 0
    print("FAIL — P0 gate not satisfied.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
ptqtp_single_tensor.py — Validate PTQTP algorithm on a single Gemma 4 tensor.

Implements Algorithm 1 from arxiv 2509.16989 (PTQTP: Post-Training Quantization
to Trit-Planes). No source code released by authors as of 2025-12, this is a
clean-room implementation from the paper's Sections 3-4.

Goal: verify cos sim ≥ 0.95 on a single tensor (vs Q4_K reference) BEFORE
investing in full multi-tensor pipeline + C kernel.

Usage:
    python3 ptqtp_single_tensor.py \\
        --gguf gguf_artifacts/gemma4-e2b-Q4_K_M.gguf \\
        --tensor blk.0.attn_q.weight \\
        [--group-size 128] [--max-iter 50] [--tol 1e-4]
"""
import argparse
import time

import numpy as np

from ptqtp_version import ALGORITHM_VERSION  # noqa: F401 — exported for callers


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64).flatten()
    bf = b.astype(np.float64).flatten()
    n = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / n) if n > 0 else 1.0


def relative_mse(a: np.ndarray, b: np.ndarray) -> float:
    """||a-b||² / ||a||²"""
    err = float(np.sum((a - b) ** 2))
    norm = float(np.sum(a ** 2))
    return err / max(norm, 1e-30)


def ptqtp_quantize(W: np.ndarray, group_size: int = 128, max_iter: int = 50,
                   tol: float = 1e-4, lam_init: float = 1e-6, lam_max: float = 1.0,
                   verbose: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PTQTP per-tensor quantization.

    W: [n_out, n_in] FP32 weight matrix.
    Returns:
        T1: [n_out, n_in] int8 ∈ {-1,0,+1}
        T2: [n_out, n_in] int8 ∈ {-1,0,+1}
        alpha: [n_out, n_groups, 2] float32 (per row-group, 2 plane scales)

    Reconstruction: w[i,j] ≈ α[i, g, 0]·T1[i,j] + α[i, g, 1]·T2[i,j]
                              with g = j // group_size
    """
    n_out, n_in = W.shape
    assert n_in % group_size == 0, f"n_in={n_in} not divisible by group_size={group_size}"
    n_groups = n_in // group_size

    # Initialize: T1 = sign(W), T2 = sign(W - sign(W)·alpha¹) — paper uses simpler init
    T1 = np.sign(W).astype(np.int8)  # [n_out, n_in]
    T2 = np.zeros_like(T1)

    # Initial alpha: per row-group, ridge regression with current T1, T2.
    # Reshape for group-wise: [n_out, n_groups, group_size]
    W_g = W.reshape(n_out, n_groups, group_size)
    T1_g = T1.reshape(n_out, n_groups, group_size).astype(np.float32)
    T2_g = T2.reshape(n_out, n_groups, group_size).astype(np.float32)
    alpha = np.zeros((n_out, n_groups, 2), dtype=np.float32)
    lam_per_group = np.full((n_out, n_groups), lam_init, dtype=np.float32)

    def update_alpha():
        """Solve ridge regression for all (row, group) pairs simultaneously."""
        nonlocal alpha, lam_per_group
        # S [n_out, n_groups, group_size, 2]
        S = np.stack([T1_g, T2_g], axis=-1)  # [n_out, n_groups, g, 2]
        # A = S^T S + λI: [n_out, n_groups, 2, 2]
        A = np.einsum("ngsk,ngsl->ngkl", S, S)
        # Add λI
        eye2 = np.eye(2, dtype=np.float32)
        A_reg = A + lam_per_group[:, :, None, None] * eye2
        # b = S^T W: [n_out, n_groups, 2]
        b = np.einsum("ngsk,ngs->ngk", S, W_g)
        # Adaptive λ: estimate condition number, bump if too high
        # Simple: |A|_F * |A^-1|_F ≈ |A|_F² / |det(A)|
        det = A_reg[..., 0, 0] * A_reg[..., 1, 1] - A_reg[..., 0, 1] * A_reg[..., 1, 0]
        Anorm = np.sqrt(np.sum(A_reg ** 2, axis=(-2, -1)))
        cond_est = Anorm ** 2 / np.maximum(np.abs(det), 1e-30)
        bad = cond_est > 1e12
        if bad.any():
            # Increase λ where ill-conditioned
            lam_per_group = np.where(bad, np.minimum(lam_per_group * np.sqrt(cond_est / 1e12), lam_max),
                                     lam_per_group)
            A_reg = A + lam_per_group[:, :, None, None] * eye2
            det = A_reg[..., 0, 0] * A_reg[..., 1, 1] - A_reg[..., 0, 1] * A_reg[..., 1, 0]
        # Solve [n_out, n_groups, 2x2] · α = b via closed-form inverse
        det_safe = np.where(np.abs(det) < 1e-30, 1.0, det)
        inv00 =  A_reg[..., 1, 1] / det_safe
        inv01 = -A_reg[..., 0, 1] / det_safe
        inv10 = -A_reg[..., 1, 0] / det_safe
        inv11 =  A_reg[..., 0, 0] / det_safe
        alpha[..., 0] = inv00 * b[..., 0] + inv01 * b[..., 1]
        alpha[..., 1] = inv10 * b[..., 0] + inv11 * b[..., 1]
        alpha[np.abs(det)[:, :, None].repeat(2, axis=-1) < 1e-30] = 0.0

    def update_T():
        """Vectorized exhaustive ternary pair search per (row, group, element)."""
        nonlocal T1_g, T2_g
        # alpha [n_out, n_groups, 2]
        # All 9 candidate pairs
        cands = np.array([(c1, c2) for c1 in (-1, 0, 1) for c2 in (-1, 0, 1)], dtype=np.float32)  # [9, 2]
        # Predicted per (row, group, candidate): α @ cands.T → [n_out, n_groups, 9]
        pred = np.einsum("ngk,jk->ngj", alpha, cands)
        # Error per (row, group, element, candidate): [n_out, n_groups, group_size, 9]
        err_sq = (W_g[..., None] - pred[..., None, :]) ** 2
        best_idx = np.argmin(err_sq, axis=-1)  # [n_out, n_groups, group_size]
        T1_g = cands[best_idx, 0]  # [n_out, n_groups, group_size]
        T2_g = cands[best_idx, 1]

    update_alpha()
    for t in range(max_iter):
        alpha_old = alpha.copy()
        update_T()
        update_alpha()
        delta = float(np.sqrt(np.sum((alpha - alpha_old) ** 2)))
        if verbose and (t < 3 or t % 5 == 0 or t == max_iter - 1):
            # Compute reconstruction MSE for diagnostics
            Wrec = (alpha[..., 0:1] * T1_g + alpha[..., 1:2] * T2_g).reshape(n_out, n_in)
            mse = relative_mse(W, Wrec)
            cs = cosine_sim(W, Wrec)
            print(f"  iter {t:3d}: |Δα|={delta:.6f}  rel_mse={mse:.6f}  cos={cs:.6f}")
        if delta < tol:
            if verbose: print(f"  converged at iter {t}")
            break

    T1 = T1_g.reshape(n_out, n_in).astype(np.int8)
    T2 = T2_g.reshape(n_out, n_in).astype(np.int8)
    return T1, T2, alpha


def ptqtp_quantize_n(W: np.ndarray, n_planes: int, group_size: int = 128,
                      max_iter: int = 50, tol: float = 1e-4,
                      lam_init: float = 1e-6, lam_max: float = 1.0,
                      verbose: bool = True):
    """Generalized PTQTP for arbitrary n_planes (≥ 1). Returns (Ts, alpha)
    where Ts is list of n_planes int8 [n_out, n_in] arrays and alpha is
    [n_out, n_groups, n_planes] float32.

    Algorithm: alternating ridge (alpha update) + exhaustive search
    (T update) over 3^n_planes candidate trit-tuples per element."""
    import itertools
    n_out, n_in = W.shape
    assert n_in % group_size == 0
    n_groups = n_in // group_size
    K = n_planes
    n_cands = 3 ** K

    # Candidate trit-tuples: shape [n_cands, K]
    cands = np.array(list(itertools.product([-1, 0, 1], repeat=K)),
                      dtype=np.float32)

    W_g = W.reshape(n_out, n_groups, group_size).astype(np.float32)
    # Init: T1 = sign(W), Tk = 0 for k > 0
    Ts_g = np.zeros((K, n_out, n_groups, group_size), dtype=np.float32)
    Ts_g[0] = np.sign(W_g)

    alpha = np.zeros((n_out, n_groups, K), dtype=np.float32)
    lam = np.full((n_out, n_groups), lam_init, dtype=np.float32)
    eyeK = np.eye(K, dtype=np.float32)

    def update_alpha():
        nonlocal alpha, lam
        # S [n_out, n_groups, group_size, K]
        S = np.moveaxis(Ts_g, 0, -1)
        A = np.einsum("ngsk,ngsl->ngkl", S, S)               # [..., K, K]
        b = np.einsum("ngsk,ngs->ngk", S, W_g)               # [..., K]
        A_reg = A + lam[:, :, None, None] * eyeK
        # numpy.linalg.solve wants b shape (..., K, N); add trailing dim.
        b2 = b[..., None]                                     # [..., K, 1]
        try:
            sol = np.linalg.solve(A_reg, b2)
        except np.linalg.LinAlgError:
            lam = np.minimum(lam * 100, lam_max)
            A_reg = A + lam[:, :, None, None] * eyeK
            sol = np.linalg.solve(A_reg, b2)
        alpha[:] = sol[..., 0]

    def update_T():
        # pred[n, g, c] = sum_k alpha[n,g,k] * cands[c,k]
        pred = np.einsum("ngk,ck->ngc", alpha, cands)
        # err[n, g, s, c] = (W[n,g,s] - pred[n,g,c])^2
        err = (W_g[..., None] - pred[..., None, :]) ** 2
        best = np.argmin(err, axis=-1)                        # [n_out, n_groups, group_size]
        # Ts_g[k, n, g, s] = cands[best[n,g,s], k]
        for k in range(K):
            Ts_g[k] = cands[best, k]

    update_alpha()
    for t in range(max_iter):
        alpha_old = alpha.copy()
        update_T()
        update_alpha()
        delta = float(np.linalg.norm(alpha - alpha_old))
        if verbose and (t < 3 or t % 5 == 0 or t == max_iter - 1):
            Wrec = sum(alpha[..., k:k+1] * Ts_g[k] for k in range(K)).reshape(n_out, n_in)
            cs = cosine_sim(W, Wrec)
            print(f"  iter {t:3d}: |Δα|={delta:.6f}  cos={cs:.6f}")
        if delta < tol:
            if verbose: print(f"  converged at iter {t}")
            break

    Ts_out = [Ts_g[k].reshape(n_out, n_in).astype(np.int8) for k in range(K)]
    return Ts_out, alpha


def main():
    # Imported here, not at module top: the paper-parity test imports this
    # module hermetically (no GGUF, no gguf package) — only the CLI needs it.
    import gguf

    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tensor", default="blk.0.attn_q.weight")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--max-iter", type=int, default=50)
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    print(f"algorithm: {ALGORITHM_VERSION}")
    print(f"loading {args.gguf}")
    reader = gguf.GGUFReader(args.gguf)
    by_name = {t.name: t for t in reader.tensors}
    if args.tensor not in by_name:
        print(f"tensor {args.tensor} not found")
        return 1
    t = by_name[args.tensor]
    W = gguf.dequantize(t.data, t.tensor_type)
    print(f"tensor {args.tensor}: shape {W.shape}, dtype {W.dtype}")
    print(f"weight stats: mean={W.mean():.6f} std={W.std():.6f} max|w|={np.abs(W).max():.4f}")

    print(f"\nPTQTP: group_size={args.group_size}, max_iter={args.max_iter}, tol={args.tol}")
    t0 = time.time()
    T1, T2, alpha = ptqtp_quantize(W, args.group_size, args.max_iter, args.tol)
    elapsed = time.time() - t0
    print(f"\nelapsed: {elapsed:.1f}s")

    # Final stats
    n_out, n_in = W.shape
    n_groups = n_in // args.group_size
    Wrec = (alpha[..., 0:1] * T1.reshape(n_out, n_groups, args.group_size).astype(np.float32) +
            alpha[..., 1:2] * T2.reshape(n_out, n_groups, args.group_size).astype(np.float32)
            ).reshape(n_out, n_in)
    cs = cosine_sim(W, Wrec)
    mse_rel = relative_mse(W, Wrec)
    sparsity1 = 1.0 - float(np.count_nonzero(T1)) / T1.size
    sparsity2 = 1.0 - float(np.count_nonzero(T2)) / T2.size
    print(f"\nFinal: cos={cs:.6f}, rel_mse={mse_rel:.6f}")
    print(f"Sparsity: T1={sparsity1:.2%}, T2={sparsity2:.2%}")
    print(f"alpha stats: mean={alpha.mean():.6f}, std={alpha.std():.6f}, max|α|={np.abs(alpha).max():.4f}")

    # Storage: 4 bits/weight + 2 FP16 per group of 128 per row
    storage_bits = (n_out * n_in * 4) + (n_out * n_groups * 2 * 16)
    bpw_eff = storage_bits / (n_out * n_in)
    print(f"Storage: {bpw_eff:.2f} bits/weight effective")

    # PASS / FAIL
    if cs >= 0.95:
        print(f"\n✓ PASS: cos {cs:.4f} ≥ 0.95")
        return 0
    else:
        print(f"\n✗ FAIL: cos {cs:.4f} < 0.95")
        return 1


if __name__ == "__main__":
    raise SystemExit(main() or 0)

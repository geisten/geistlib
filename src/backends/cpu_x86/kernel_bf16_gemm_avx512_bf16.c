/*
 * src/backends/cpu_x86/kernel_bf16_gemm_avx512_bf16.c — bf16 SGEMM kernel.
 *
 * Compiled with -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512bf16;
 * the caller (linear_q4k.c) only routes here on cpuid-confirmed BF16
 * support, so there's no SIGILL risk on Zen 4 (which has BF16) or any
 * AVX-512+BF16 CPU (Sapphire Rapids+, Zen 4+/5).
 *
 * Inner kernel: M_TILE = 4 m-rows × N_TILE = 16 output cells. One
 * VDPBF16PS per (m-row, K-pair) — each instruction does 32 bf16
 * multiply-adds and accumulates into 16 fp32 lanes (one per output
 * cell). Per K iteration only 4 VDPBF16PSs total (4 rows × 1 across
 * all 16 cells); no hsum because each lane is already one cell.
 *
 * Throughput on Zen 5 at peak: 2 VDPBF16PS / cycle / core × 16 cores
 * × 16 cells × 2 K-elements = 1024 bf16 MACs / cycle = ~4 T MACs / sec
 * aggregate at 4 GHz. That is the path to llama.cpp-class prefill.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "kernel_bf16_gemm.h"

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

constexpr size_t BF16_GEMM_MTILE = 4;

/* Build a 16-lane __m512i where every lane has the SAME 32-bit value
 * holding (a1 << 16) | a0 — i.e. a bf16 pair (a0, a1) broadcast to all
 * 16 output cells. VDPBF16PS reads each lane's pair against the packed
 * weight pair for that cell. */
static inline __m512i bf16_pair_broadcast(bf16_t a0, bf16_t a1) {
    const uint32_t pair = ((uint32_t) a1 << 16) | (uint32_t) a0;
    return _mm512_set1_epi32((int32_t) pair);
}

void bf16_gemm_avx512_bf16(size_t       M,
                           size_t       N,
                           size_t       K,
                           const float  X[static M * K],
                           const bf16_t W_packed[static N * K],
                           float        Y[static M * N]) {
    const size_t n_tiles = N / BF16_NTILE;
    const size_t k_pairs = K / 2;

    /* Pre-convert X (fp32) → bf16 once per call so the inner loop only
     * loads + broadcasts already-converted pairs (no per-iter fp32→bf16
     * inside the hot path). One contiguous M*K bf16 buffer. */
    /* aligned_alloc (C11) requires size to be a multiple of the alignment —
     * round up, else a tiny M*K aborts under ASan. */
    const size_t x_bytes = ((M * K * sizeof(bf16_t)) + 63u) & ~(size_t) 63u;
    bf16_t      *X_bf    = aligned_alloc(64, x_bytes);
    if (X_bf == nullptr) {
        return;
    }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < M; i++) {
        for (size_t k = 0; k < K; k++) {
            X_bf[i * K + k] = fp32_to_bf16(X[i * K + k]);
        }
    }

    /* Outer parallel loop over output tiles (lane groups of 16 cells).
     * Inner: walk m in chunks of MTILE rows, K in pairs. */
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t t = 0; t < n_tiles; t++) {
        const bf16_t *Wt = W_packed + t * K * BF16_NTILE;

        size_t i = 0;
        for (; i + BF16_GEMM_MTILE <= M; i += BF16_GEMM_MTILE) {
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            const bf16_t *Xi0 = X_bf + (i + 0) * K;
            const bf16_t *Xi1 = X_bf + (i + 1) * K;
            const bf16_t *Xi2 = X_bf + (i + 2) * K;
            const bf16_t *Xi3 = X_bf + (i + 3) * K;

            for (size_t kp = 0; kp < k_pairs; kp++) {
                /* 32 bf16 weights for this tile and K-pair = one cache line. */
                const __m512i w = _mm512_loadu_si512((const __m512i *) (Wt + kp * 32));

                /* Load + broadcast pair per m-row. Reading a 32-bit at
                 * Xi[2*kp] gives us (a0, a1) packed; broadcast to all
                 * 16 lanes via VPBROADCASTD. */
                const __m512i a0v = _mm512_set1_epi32(*(const int32_t *) (Xi0 + 2 * kp));
                const __m512i a1v = _mm512_set1_epi32(*(const int32_t *) (Xi1 + 2 * kp));
                const __m512i a2v = _mm512_set1_epi32(*(const int32_t *) (Xi2 + 2 * kp));
                const __m512i a3v = _mm512_set1_epi32(*(const int32_t *) (Xi3 + 2 * kp));

                acc0 = _mm512_dpbf16_ps(acc0, (__m512bh) a0v, (__m512bh) w);
                acc1 = _mm512_dpbf16_ps(acc1, (__m512bh) a1v, (__m512bh) w);
                acc2 = _mm512_dpbf16_ps(acc2, (__m512bh) a2v, (__m512bh) w);
                acc3 = _mm512_dpbf16_ps(acc3, (__m512bh) a3v, (__m512bh) w);
            }

            float *Yi = Y + i * N + t * BF16_NTILE;
            _mm512_storeu_ps(Yi + 0 * N, acc0);
            _mm512_storeu_ps(Yi + 1 * N, acc1);
            _mm512_storeu_ps(Yi + 2 * N, acc2);
            _mm512_storeu_ps(Yi + 3 * N, acc3);
        }

        /* Tail: any leftover m-rows. */
        for (; i < M; i++) {
            __m512        acc = _mm512_setzero_ps();
            const bf16_t *Xi  = X_bf + i * K;
            for (size_t kp = 0; kp < k_pairs; kp++) {
                const __m512i w  = _mm512_loadu_si512((const __m512i *) (Wt + kp * 32));
                const __m512i av = _mm512_set1_epi32(*(const int32_t *) (Xi + 2 * kp));
                acc              = _mm512_dpbf16_ps(acc, (__m512bh) av, (__m512bh) w);
            }
            _mm512_storeu_ps(Y + i * N + t * BF16_NTILE, acc);
        }
    }

    free(X_bf);
}

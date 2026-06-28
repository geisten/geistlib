/*
 * src/backends/cpu_x86/kernel_bf16_gemm.h — bf16 SGEMM kernel.
 *
 * Layer: BACKEND (cpu_x86, internal).
 *
 * Computes Y[M, N] = X[M, K] @ W^T[K, N] with W stored in a PACKED bf16
 * layout designed for VDPBF16PS:
 *
 *   W_packed[n_tile][k_pair][lane * 2 + bit]   where
 *     n_tile  = j / 16     (16 output cells per __m512 lane group)
 *     lane    = j % 16     (which output cell within the tile)
 *     k_pair  = k / 2      (VDPBF16PS reduces 2 K-elements per lane)
 *     bit     = k % 2
 *
 * One 32-bf16 row of W_packed is one cache line (64 bytes) and feeds a
 * single VDPBF16PS that accumulates into 16 output-cell partial sums
 * simultaneously. No per-cell horizontal reduction is required — each
 * fp32 lane IS one output cell.
 *
 * X is provided as fp32 row-major [M, K]; the AVX kernel converts each
 * (m_row, k_pair) on the fly to a broadcast bf16 pair (no scratch
 * needed). Y stays fp32.
 *
 * Constraints:
 *   - N must be a multiple of BF16_NTILE (= 16).
 *   - K must be a multiple of BF16_KPAIR (= 2).
 *   Gemma 4 E2B dimensions (1536, 2560, 8960, vocab 262144) all satisfy
 *   these by construction; the dispatcher rejects mismatched shapes.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_BF16_GEMM_H
#define GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_BF16_GEMM_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_x86/kernel_bf16_gemm.h is internal to the backend layer."
#endif

#include <stddef.h>
#include <stdint.h>

constexpr size_t BF16_NTILE = 16;
constexpr size_t BF16_KPAIR = 2;

/* bf16 is two bytes — the upper half of IEEE-754 binary32. */
typedef uint16_t bf16_t;

[[nodiscard]] static inline bf16_t fp32_to_bf16(float f) {
    union {
        float    f;
        uint32_t u;
    } bits = {.f = f};
    /* Round-to-nearest-even on the dropped 16 mantissa bits. */
    const uint32_t rounding_bias = 0x7FFFu + ((bits.u >> 16) & 1u);
    return (bf16_t) ((bits.u + rounding_bias) >> 16);
}

[[nodiscard]] static inline float bf16_to_fp32(bf16_t b) {
    union {
        uint32_t u;
        float    f;
    } bits = {.u = ((uint32_t) b) << 16};
    return bits.f;
}

/* Pack a flat row-major [N, K] bf16 weight matrix into the layout the
 * inner kernel consumes. One-shot at model load.
 *
 *   W_flat[j, k]            row j, column k (row-major, stride K)
 *   W_packed[(j/16) * K * 16 + (k/2) * 32 + (j%16) * 2 + (k%2)]
 *
 * Both buffers hold the same total N*K bf16 values — just rearranged. */
void bf16_pack_weights_ntile16(size_t       N,
                               size_t       K,
                               const bf16_t W_flat[static N * K],
                               bf16_t       W_packed[static N * K]);

/* Scalar reference SGEMM, walking the packed layout. Used as the
 * cross-ISA oracle for the AVX-512+BF16 variant and as the fallback
 * on hosts without VDPBF16PS. Allocation-free. */
void bf16_gemm_scalar(size_t       M,
                      size_t       N,
                      size_t       K,
                      const float  X[static M * K],
                      const bf16_t W_packed[static N * K],
                      float        Y[static M * N]);

/* AVX-512+BF16 variant — VDPBF16PS inner with M_TILE = 4 rows × N_TILE
 * = 16 cells per output tile. Caller verifies cpuid before invoking. */
void bf16_gemm_avx512_bf16(size_t       M,
                           size_t       N,
                           size_t       K,
                           const float  X[static M * K],
                           const bf16_t W_packed[static N * K],
                           float        Y[static M * N]);

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_BF16_GEMM_H */

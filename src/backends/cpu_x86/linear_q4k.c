/*
 * src/backends/cpu_x86/linear_q4k.c — cpu_x86 Q4_K M=1 (decode) wiring.
 *
 * Layer: BACKEND (cpu_x86).
 *
 * Per Q4_K weight, one heap-aligned blob owned by the weight
 * (GEIST_W_AUX_HEAP_OWNED; the engine frees it at model destroy) holding
 * the one layout the kernels read for that shape:
 *   - n_out % 8 == 0 (every body matrix): the Q4_Kx8 lane-parallel repack
 *     (0.56 B/wt). Decode (q4kx8_gemv_m1) and prefill (q4kx8_gemm_avx512)
 *     both read it.
 *   - otherwise: the W4A8 SoA — packed nibbles (n_in/2 bytes), per-block
 *     scales and offsets (n_in/32 fp32 each) per row (0.75 B/wt), for the
 *     w4a8_gemv decode; prefill runs that per row.
 *
 * The hot-path kernel reconstructs the pointers from w->aux_fp32 +
 * w->n_in + w->n_out arithmetic; no per-call allocation.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "linear_q4k.h"

#include "backend_state.h"
#include "checked.h"
#include "kernel_q4kx8_gemm.h" /* lane-parallel Q4_Kx8 GEMV / GEMM */
#include "kernel_w4a8.h"
#include "kernel_w8a8.h" /* sum_a sized for W8A8 to also cover Q6_K */
#include "linear_ref.h"
#include "q4k_to_q4kx8.h"
#include "q4k_to_w4a8.h"
#include "q8_kx4.h"

#include "heap.h"
#include "quant.h" /* Q4_K_BLOCK_ELEMS / Q4_K_BLOCK_BYTES / dequant_q4_K_row */

#include <geist_backend.h>

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

/* Layout sizes for one weight (row-major SoA). Per-row blocks = n_in/32. */
static inline size_t weights_bytes_per_row(size_t n_in) {
    return (n_in / W4A8_BLOCK_ELEMS) * W4A8_BLOCK_BYTES_WEIGHTS;
}
static inline size_t scales_count_per_row(size_t n_in) {
    return n_in / W4A8_BLOCK_ELEMS;
}

/* x86-64 transparent huge page. heap_alloc_aligned advises THP for blocks
 * at least this large, but only their 2 MB-aligned interior can be backed:
 * a block starting at an arbitrary 64-byte boundary loses a huge page at
 * each end, which for the smaller single-layout blobs is a large share. */
constexpr size_t THP_BYTES = 2u << 20;

/* The blob holds the Q4_Kx8 repack when n_out % 8 == 0, else the W4A8 SoA. */
static inline bool uses_q4kx8(size_t n_out) {
    return n_out % 8 == 0;
}

/* Blob layouts, aligned by construction:
 *   uses_q4kx8:  [q4kx8     : (n_out/8) * (n_in/256) * sizeof(block_q4_Kx8)]
 *   otherwise:   [weights   : n_out * weights_bytes_per_row(n_in)]
 *                [w_scales  : n_out * scales_count_per_row(n_in) fp32]
 *                [w_offsets : n_out * scales_count_per_row(n_in) fp32] */
static size_t blob_total_bytes(size_t n_in, size_t n_out) {
    if (uses_q4kx8(n_out)) {
        return (n_out / 8) * (n_in / Q4_K_BLOCK_ELEMS) * sizeof(struct block_q4_Kx8);
    }
    const size_t weights_total = n_out * weights_bytes_per_row(n_in);
    const size_t scales_total  = n_out * scales_count_per_row(n_in) * sizeof(float);
    return weights_total + 2 * scales_total;
}

/* The W4A8 SoA pointers; only valid when !uses_q4kx8(n_out). */
static void w4a8_pointers(const uint8_t  *blob,
                          size_t          n_in,
                          size_t          n_out,
                          const uint8_t **weights_out,
                          const float   **scales_out,
                          const float   **offsets_out) {
    const size_t weights_bytes = n_out * weights_bytes_per_row(n_in);
    const size_t scales_count  = n_out * scales_count_per_row(n_in);

    *weights_out = blob;
    *scales_out  = (const float *) (blob + weights_bytes);
    *offsets_out = *scales_out + scales_count;
}

[[nodiscard]] enum geist_status cpu_x86_linear_q4k_resolve(struct cpu_x86_state *st,
                                                           struct geist_weight  *w) {
    if (st == nullptr || w == nullptr || w->n_in <= 0 || w->n_out <= 0) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t n_in  = (size_t) w->n_in;
    const size_t n_out = (size_t) w->n_out;
    if (n_in % Q4_K_BLOCK_ELEMS != 0) {
        return GEIST_E_INVALID_ARG;
    }

    /* Every token streams these bytes: align a blob of huge-page size or
     * more to a huge page so THP backs all of it (decode 1-2 % faster on
     * Llama-3.2-1B Q4_K), at the price of a partly used last huge page. */
    const size_t blob_bytes = blob_total_bytes(n_in, n_out);
    uint8_t     *blob =
            heap_alloc_aligned(blob_bytes, blob_bytes >= THP_BYTES ? THP_BYTES : OPTIMAL_ALIGNMENT);
    if (blob == nullptr) {
        return GEIST_E_OOM;
    }
    const uint8_t *q4k_raw = (const uint8_t *) w->raw;
    if (uses_q4kx8(n_out)) {
        q4k_to_q4kx8_matrix(n_in, n_out, q4k_raw, (struct block_q4_Kx8 *) blob);
    } else {
        const uint8_t *blob_w_const;
        const float   *blob_s_const;
        const float   *blob_o_const;
        w4a8_pointers(blob, n_in, n_out, &blob_w_const, &blob_s_const, &blob_o_const);
        uint8_t     *blob_w        = (uint8_t *) blob_w_const;
        float       *blob_s        = (float *) blob_s_const;
        float       *blob_o        = (float *) blob_o_const;
        const size_t q4k_row_bytes = (n_in / Q4_K_BLOCK_ELEMS) * Q4_K_BLOCK_BYTES;
        const size_t w_row_bytes   = weights_bytes_per_row(n_in);
        const size_t s_row_count   = scales_count_per_row(n_in);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) /* see q4k_to_q4kx8_matrix */
#endif
        for (size_t m = 0; m < n_out; m++) {
            q4k_to_w4a8_row(n_in,
                            q4k_raw + m * q4k_row_bytes,
                            blob_w + m * w_row_bytes,
                            blob_s + m * s_row_count,
                            blob_o + m * s_row_count);
        }
    }

    /* aux_fp32 reinterpreted as the blob pointer; engine frees it on
     * model destroy via heap_free / safe_free (GEIST_W_AUX_HEAP_OWNED). */
    w->aux_fp32 = (const float *) blob;
    w->aux_n    = (int32_t) blob_bytes;
    w->flags |= GEIST_W_AUX_HEAP_OWNED | GEIST_W_AUX_BACKEND_REPACK;
    w->linear_m1 = cpu_x86_linear_q4k_m1;
    w->linear_mN = cpu_x86_linear_q4k_mN;
    return GEIST_OK;
}

void cpu_x86_linear_q4k_m1(const float               *x,
                           const struct geist_weight *w,
                           struct geist_backend      *be,
                           float                     *y) {
    struct cpu_x86_state *st               = (struct cpu_x86_state *) be->state;
    const size_t          n_in             = (size_t) w->n_in;
    const size_t          n_out            = (size_t) w->n_out;
    const size_t          n_blocks_per_row = n_in / W4A8_BLOCK_ELEMS;

    /* Decode over the compact Q4_Kx8 layout when the blob holds it (n_out a
     * multiple of 8 — every Q4_K body matrix). The 8-cell lane-parallel GEMV
     * reduces once per tile (no per-block hsum) and reads 0.56 B/wt vs W4A8's
     * 0.75 — both the compute and bandwidth limits of decode. */
    if (uses_q4kx8(n_out)) {
        q4kx8_gemv_m1(n_out, n_in, x, (const struct block_q4_Kx8 *) w->aux_fp32, y);
        return;
    }

    const uint8_t *weights;
    const float   *w_scales;
    const float   *w_offsets;
    w4a8_pointers((const uint8_t *) w->aux_fp32, n_in, n_out, &weights, &w_scales, &w_offsets);

    /* Per-row activation quantization → int8 acts + per-block sum_a. */
    struct cpu_x86_workspace *ws = cpu_x86_ws_acquire(st, n_in);
    if (ws == nullptr) {
        geist_linear_ref(1, x, w, y); /* no scratch: the reference needs none */
        return;
    }
    const float scale_x = w4a8_quantize_acts_row(n_in, x, ws->acts_scratch, ws->sum_a_scratch);

    /* Multi-row GEMV fallback (n_out not a multiple of 8). OMP-parallel. */
    w4a8_gemv(n_out,
              n_blocks_per_row,
              weights,
              w_scales,
              w_offsets,
              ws->acts_scratch,
              ws->sum_a_scratch,
              scale_x,
              y);
}

/* Q4_Kx8 lane-parallel GEMM via VPMADDUBSW, 8 cells per instruction. The
 * activations are quantized to Q8_Kx4 (4 rows interleaved in 8-byte
 * stripes) in the per-thread workspace. q4kx8_gemm_avx512 guards its own
 * ISA at run time (AVX2 GEMV fallback on non-AVX-512 hosts).
 *
 * The GEMM takes whole Q8_Kx4 groups of 4 rows; the last m % 4 rows go to
 * the M=1 GEMV, as does every row of a weight in the W4A8 layout (n_out
 * not a multiple of 8). */
void cpu_x86_linear_q4k_mN(size_t                     m,
                           const float               *x,
                           const struct geist_weight *w,
                           struct geist_backend      *be,
                           float                     *y) {
    const size_t n_in  = (size_t) w->n_in;
    const size_t n_out = (size_t) w->n_out;

    if (!uses_q4kx8(n_out)) {
        /* W4A8 layout: the M=1 kernel per row. */
        for (size_t row = 0; row < m; row++) {
            cpu_x86_linear_q4k_m1(x + row * n_in, w, be, y + row * n_out);
        }
        return;
    }
    const size_t m4 = m / 4 * 4;
    for (size_t row = m4; row < m; row++) {
        cpu_x86_linear_q4k_m1(x + row * n_in, w, be, y + row * n_out);
    }
    if (m4 == 0) {
        return;
    }

    /* Quantize acts into block_q8_Kx4 workspace scratch — 4 rows interleaved
     * per super-block, m/4 × n_in/256 × ~1.2 KB (≈36 KB at m=128,
     * n_in=1536). */
    const size_t              n_super_k   = n_in / 256;
    size_t                    q8kx4_count = 0;
    size_t                    acts_bytes  = 0;
    struct cpu_x86_workspace *ws          = nullptr;
    if (be != nullptr && be->state != nullptr && !ckd_mul(&q8kx4_count, m4 / 4, n_super_k) &&
        !ckd_mul(&acts_bytes, q8kx4_count, sizeof(struct block_q8_Kx4))) {
        ws = cpu_x86_ws_acquire_mN((struct cpu_x86_state *) be->state, 0, 0, 0, acts_bytes);
    }
    if (ws == nullptr) {
        for (size_t row = 0; row < m4; row++) {
            cpu_x86_linear_q4k_m1(x + row * n_in, w, be, y + row * n_out);
        }
        return;
    }
    struct block_q8_Kx4 *acts = (struct block_q8_Kx4 *) ws->mN_aux;
    for (size_t mt = 0; mt < m4 / 4; mt++) {
        quantize_q8_Kx4(n_in, x + mt * 4 * n_in, acts + mt * n_super_k);
    }

    q4kx8_gemm_avx512(m4, n_out, n_in, acts, (const struct block_q4_Kx8 *) w->aux_fp32, y);
}

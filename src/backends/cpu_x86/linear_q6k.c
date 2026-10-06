/*
 * src/backends/cpu_x86/linear_q6k.c — cpu_x86 Q6_K M=1 (decode) path.
 *
 * Layer: BACKEND (cpu_x86).
 *
 * Decode (M=1) reads the native Q6_K bytes (w->raw) through
 * q6k_gemv_m1. Prefill (M>1) reads them too (q6k_gemm) below AVX-512
 * VNNI; on VNNI it reads a W8A8 predecode (1 byte / weight +
 * per-16-element scale and offset, 1.5 B/wt), in one of two layouts:
 *   - the lane-parallel W8x8 / W8x16 interleave, on a VNNI host with
 *     n_out % 8 == 0 (w8x8_gemm / w8x16_gemm);
 *   - row-major otherwise (w8a8_gemm).
 * The weight keeps only the one its prefill reads.
 *
 * Q3_K shares the kernels (kernel_q6k_gemv.c reads both formats) and has no
 * predecode: decode and prefill both read its GGUF bytes (#410).
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "linear_q6k.h"

#include "backend_state.h"
#include "checked.h"
#include "kernel_w4a8.h" /* w4a8_quantize_acts_row */
#include "kernel_w8a8.h"
#include "kernel_q6k_gemv.h"
#include "q6k_to_w8a8.h"

#include "heap.h"
#include "quant.h" /* Q3_K / Q6_K block sizes */

#include <geist_backend.h>

#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h> /* getenv */

/* SoA layout in the heap blob (allocated once per Q6_K weight at
 * resolve_weight): weights | w_scales | w_offsets, row-major or
 * interleaved (same sizes; the interleave is a pure permutation). Sizes
 * derive from n_in, n_out, W8A8_BLOCK_ELEMS. The fp32 arrays follow the
 * byte array; since heap_alloc_aligned gives 64-byte alignment and the
 * byte array size is a multiple of W8A8_BLOCK_ELEMS (= 16), the float
 * starts are 16-byte aligned which is sufficient for fp32 loads. */
static inline size_t blocks_per_row(size_t n_in) {
    return n_in / W8A8_BLOCK_ELEMS;
}
static inline size_t weights_bytes_per_row(size_t n_in) {
    return blocks_per_row(n_in) * W8A8_BLOCK_ELEMS;
}

static void blob_pointers(const uint8_t  *blob,
                          size_t          n_in,
                          size_t          n_out,
                          const uint8_t **weights_out,
                          const float   **scales_out,
                          const float   **offsets_out) {
    const size_t weights_bytes = n_out * weights_bytes_per_row(n_in);
    const size_t scales_count  = n_out * blocks_per_row(n_in);

    *weights_out = blob;
    *scales_out  = (const float *) (blob + weights_bytes);
    *offsets_out = *scales_out + scales_count;
}

/* Lane-parallel W8x8 prefill needs n_out % 8 == 0 and a VNNI host. Computed
 * identically at resolve (to decide whether to build the interleaved blob)
 * and at mN (to decide whether to use it). */
static inline bool q6k_use_w8x8(size_t n_out) {
    return (n_out % W8X8_NROWS == 0) && w8a8_isa_is_vnni();
}

/* Prefill straight from the GGUF bytes (q6k_gemm) keeps one copy of the
 * weights (#577). The W8A8 predecode only has a vector kernel on AVX-512
 * VNNI, where it prefills about twice as fast as q6k_gemm; below that its
 * dot is scalar, about seven times slower. So raw is the default below
 * VNNI; GEIST_Q6K_RAW=1 or 0 overrides it. */
static bool q6k_reads_raw(void) {
    const char *e = getenv("GEIST_Q6K_RAW");
    if (e != nullptr && e[0] != '\0') {
        return e[0] != '0';
    }
    return !w8a8_isa_is_vnni();
}

static void cpu_x86_linear_q6k_raw_mN(size_t                     m,
                                      const float               *x,
                                      const struct geist_weight *w,
                                      struct geist_backend      *be,
                                      float                     *y) {
    const size_t              n_in  = (size_t) w->n_in;
    const size_t              n_out = (size_t) w->n_out;
    struct cpu_x86_workspace *ws    = nullptr;
    if (be != nullptr && be->state != nullptr) {
        ws = cpu_x86_ws_acquire_mN(
                (struct cpu_x86_state *) be->state, q6k_gemm_scratch_bytes(m, n_in), 0, 0, 0);
    }
    if (ws == nullptr) {
        for (size_t row = 0; row < m; row++) {
            cpu_x86_linear_q6k_m1(x + row * n_in, w, be, y + row * n_out);
        }
        return;
    }
    q6k_gemm(m, n_out, n_in, x, (const uint8_t *) w->raw, ws->mN_acts, y);
}

/* x86-64 transparent huge page: blobs this large are allocated aligned to
 * it so THP backs all of them (see linear_q4k.c). */
constexpr size_t THP_BYTES = 2u << 20;

[[nodiscard]] enum geist_status cpu_x86_linear_q6k_resolve(struct geist_weight *w) {
    if (w == nullptr || w->raw == nullptr || w->n_in <= 0 || w->n_out <= 0) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t n_in  = (size_t) w->n_in;
    const size_t n_out = (size_t) w->n_out;
    if (n_in % Q6_K_BLOCK_ELEMS != 0) {
        return GEIST_E_INVALID_ARG;
    }
    if (q6k_reads_raw()) {
        w->linear_m1 = cpu_x86_linear_q6k_m1;
        w->linear_mN = cpu_x86_linear_q6k_raw_mN;
        return GEIST_OK;
    }

    const size_t weights_total = n_out * weights_bytes_per_row(n_in);
    const size_t scales_total  = n_out * blocks_per_row(n_in) * sizeof(float);
    const size_t blob_bytes    = weights_total + 2 * scales_total;
    uint8_t     *blob =
            heap_alloc_aligned(blob_bytes, blob_bytes >= THP_BYTES ? THP_BYTES : OPTIMAL_ALIGNMENT);
    if (blob == nullptr) {
        return GEIST_E_OOM;
    }
    const uint8_t *bw;
    const float   *bs;
    const float   *bo;
    blob_pointers(blob, n_in, n_out, &bw, &bs, &bo);
    uint8_t *blob_w = (uint8_t *) bw;
    float   *blob_s = (float *) bs;
    float   *blob_o = (float *) bo;

    const size_t   q6k_row_bytes = (n_in / Q6_K_BLOCK_ELEMS) * Q6_K_BLOCK_BYTES;
    const size_t   w_row_bytes   = weights_bytes_per_row(n_in);
    const size_t   s_row_count   = blocks_per_row(n_in);
    const uint8_t *q6k_raw       = (const uint8_t *) w->raw;
    if (!q6k_use_w8x8(n_out)) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) /* see q4k_to_q4kx8_matrix */
#endif
        for (size_t m = 0; m < n_out; m++) {
            q6k_to_w8a8_row(n_in,
                            q6k_raw + m * q6k_row_bytes,
                            blob_w + m * w_row_bytes,
                            blob_s + m * s_row_count,
                            blob_o + m * s_row_count);
        }
    } else {
        /* The interleave is group-major (W8X16_NROWS or W8X8_NROWS rows per
         * group, each group contiguous in all three arrays), so predecode
         * one group at a time into a small row-major staging buffer and
         * repack it into place: the same bytes as repacking the whole
         * row-major matrix, without holding it. */
        const size_t nrows = n_out % W8X16_NROWS == 0 ? W8X16_NROWS : W8X8_NROWS;
        /* The groups are shared out as in q4k_to_q4kx8_matrix, each thread
         * with its own staging buffer. */
        const size_t stage_bytes = nrows * w_row_bytes + 2 * nrows * s_row_count * sizeof(float);
        atomic_bool  oom         = false;
#if defined(_OPENMP)
#pragma omp parallel
#endif
        {
            uint8_t *stage_w = heap_alloc_aligned(stage_bytes, OPTIMAL_ALIGNMENT);
            if (stage_w == nullptr) {
                atomic_store_explicit(&oom, true, memory_order_relaxed);
            }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (size_t g = 0; g < n_out / nrows; g++) {
                if (stage_w == nullptr) {
                    continue;
                }
                float *stage_s = (float *) (stage_w + nrows * w_row_bytes);
                float *stage_o = stage_s + nrows * s_row_count;
                for (size_t r = 0; r < nrows; r++) {
                    q6k_to_w8a8_row(n_in,
                                    q6k_raw + (g * nrows + r) * q6k_row_bytes,
                                    stage_w + r * w_row_bytes,
                                    stage_s + r * s_row_count,
                                    stage_o + r * s_row_count);
                }
                uint8_t *qs_g = blob_w + g * nrows * w_row_bytes;
                float   *sc_g = blob_s + g * nrows * s_row_count;
                float   *of_g = blob_o + g * nrows * s_row_count;
                if (nrows == W8X16_NROWS) {
                    w8x16_repack(nrows, n_in, stage_w, stage_s, stage_o, qs_g, sc_g, of_g);
                } else {
                    w8x8_repack(nrows, n_in, stage_w, stage_s, stage_o, qs_g, sc_g, of_g);
                }
            }
            void *p = stage_w;
            safe_free(&p);
        }
        if (atomic_load_explicit(&oom, memory_order_relaxed)) {
            void *p = blob;
            safe_free(&p);
            return GEIST_E_OOM;
        }
    }

    w->aux_fp32 = (const float *) blob;
    w->aux_n    = (int32_t) blob_bytes;
    w->flags |= GEIST_W_AUX_HEAP_OWNED | GEIST_W_AUX_BACKEND_REPACK;
    w->linear_m1 = cpu_x86_linear_q6k_m1;
    w->linear_mN = cpu_x86_linear_q6k_mN;
    return GEIST_OK;
}

void cpu_x86_linear_q6k_m1(const float               *x,
                           const struct geist_weight *w,
                           struct geist_backend      *be,
                           float                     *y) {
    (void) be;
    /* Decode straight from the native Q6_K weights (w->raw, ~0.82 B/wt; the
     * resolver requires them) rather than a W8A8 predecode (1.5 B/wt): Q6_K
     * decode (ffn_down, lm_head) is bandwidth-bound, so halving the weight
     * traffic is the win. The blob holds the prefill layout only. */
    q6k_gemv_m1((size_t) w->n_out, (size_t) w->n_in, x, (const uint8_t *) w->raw, y);
}

/* Prefill (M>1) path. Quantizes all m tokens to int8 once, then runs a
 * tiled W8A8 GEMM that reads each weight row once and reuses it across
 * the whole token batch (Q6_K ffn_down dominates prefill in Q4_K_M
 * models). */
void cpu_x86_linear_q6k_mN(size_t                     m,
                           const float               *x,
                           const struct geist_weight *w,
                           struct geist_backend      *be,
                           float                     *y) {
    const size_t n_in             = (size_t) w->n_in;
    const size_t n_out            = (size_t) w->n_out;
    const size_t n_blocks_per_row = blocks_per_row(n_in);

    const uint8_t *weights;
    const float   *w_scales;
    const float   *w_offsets;
    blob_pointers((const uint8_t *) w->aux_fp32, n_in, n_out, &weights, &w_scales, &w_offsets);

    /* Prefill scratch from the per-thread workspace: int8 acts + 16-elem
     * sum_a + per-token scale, all m tokens (m=64, n_in=12288 ≈ 836 KB). */
    size_t                    acts_bytes = 0, sum_elems = 0, sum_bytes = 0, scale_bytes = 0;
    struct cpu_x86_workspace *ws = nullptr;
    if (be != nullptr && be->state != nullptr && !ckd_mul(&acts_bytes, m, n_in) &&
        !ckd_mul(&sum_elems, m, n_blocks_per_row) &&
        !ckd_mul(&sum_bytes, sum_elems, sizeof(int32_t)) &&
        !ckd_mul(&scale_bytes, m, sizeof(float))) {
        ws = cpu_x86_ws_acquire_mN(
                (struct cpu_x86_state *) be->state, acts_bytes, sum_bytes, scale_bytes, 0);
    }
    if (ws == nullptr) {
        for (size_t row = 0; row < m; row++) {
            cpu_x86_linear_q6k_m1(x + row * n_in, w, be, y + row * n_out);
        }
        return;
    }

    int8_t  *acts    = ws->mN_acts;
    int32_t *sum_a   = ws->mN_sum_a;
    float   *scale_x = ws->mN_scale;

    for (size_t j = 0; j < m; j++) {
        /* w4a8 quantizer gives int8 acts + per-row scale; its 32-elem sum_a
         * is the wrong granularity for W8A8. Let it scribble into this row's
         * sum_a slot (n_in/16 entries ≥ the n_in/32 it writes), then
         * overwrite with the 16-elem re-sum — no shared scratch involved. */
        scale_x[j] = w4a8_quantize_acts_row(
                n_in, x + j * n_in, acts + j * n_in, sum_a + j * n_blocks_per_row);
        const int8_t *a  = acts + j * n_in;
        int32_t      *sa = sum_a + j * n_blocks_per_row;
        for (size_t b = 0; b < n_blocks_per_row; b++) {
            int32_t s = 0;
            for (size_t i = 0; i < W8A8_BLOCK_ELEMS; i++) {
                s += (int32_t) a[b * W8A8_BLOCK_ELEMS + i];
            }
            sa[b] = s;
        }
    }

    if (q6k_use_w8x8(n_out)) {
        /* The blob holds the interleave (same array sizes as row-major). */
        if (n_out % W8X16_NROWS == 0) {
            w8x16_gemm(m,
                       n_out,
                       n_blocks_per_row,
                       weights,
                       w_scales,
                       w_offsets,
                       acts,
                       sum_a,
                       scale_x,
                       y);
        } else {
            w8x8_gemm(m,
                      n_out,
                      n_blocks_per_row,
                      weights,
                      w_scales,
                      w_offsets,
                      acts,
                      sum_a,
                      scale_x,
                      y);
        }
    } else {
        w8a8_gemm(
                m, n_out, n_blocks_per_row, weights, w_scales, w_offsets, acts, sum_a, scale_x, y);
    }
}

static void cpu_x86_linear_q3k_m1(const float               *x,
                                  const struct geist_weight *w,
                                  struct geist_backend      *be,
                                  float                     *y) {
    (void) be;
    q3k_gemv_m1((size_t) w->n_out, (size_t) w->n_in, x, (const uint8_t *) w->raw, y);
}

static void cpu_x86_linear_q3k_mN(size_t                     m,
                                  const float               *x,
                                  const struct geist_weight *w,
                                  struct geist_backend      *be,
                                  float                     *y) {
    const size_t              n_in  = (size_t) w->n_in;
    const size_t              n_out = (size_t) w->n_out;
    struct cpu_x86_workspace *ws    = nullptr;
    if (be != nullptr && be->state != nullptr) {
        ws = cpu_x86_ws_acquire_mN(
                (struct cpu_x86_state *) be->state, q3k_gemm_scratch_bytes(m, n_in), 0, 0, 0);
    }
    if (ws == nullptr) {
        for (size_t row = 0; row < m; row++) {
            cpu_x86_linear_q3k_m1(x + row * n_in, w, be, y + row * n_out);
        }
        return;
    }
    q3k_gemm(m, n_out, n_in, x, (const uint8_t *) w->raw, ws->mN_acts, y);
}

void cpu_x86_linear_q3k_bind(struct geist_weight *w) {
    w->linear_m1 = cpu_x86_linear_q3k_m1;
    w->linear_mN = cpu_x86_linear_q3k_mN;
}

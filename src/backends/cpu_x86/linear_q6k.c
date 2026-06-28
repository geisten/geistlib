/*
 * src/backends/cpu_x86/linear_q6k.c — cpu_x86 Q6_K M=1 (decode) path.
 *
 * Layer: BACKEND (cpu_x86).
 *
 * Q6_K predecoded to W8A8 (1 byte / weight + per-16-elem-block scale +
 * offset). Decode hot path: per-row int8 act-quant + w8a8_gemv via
 * VPDPBUSD. Memory: ~1.25 bytes/weight vs ~0.81 on-disk Q6_K and ~4
 * bytes/weight in the previous fp32 path; the 4× shrink vs fp32 lifts
 * the bandwidth-bound Gemma 4 lm_head decode close to the W4A8 ceiling.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "linear_q6k.h"

#include "backend_state.h"
#include "kernel_w4a8.h" /* w4a8_quantize_acts_row */
#include "kernel_w8a8.h"
#include "kernel_q6k_gemv.h"
#include "q6k_to_w8a8.h"

#include "heap.h"
#include "quant.h" /* Q6_K_BLOCK_ELEMS / Q6_K_BLOCK_BYTES */

#include <geist_backend.h>

#include <stddef.h>
#include <stdint.h>

/* SoA layout in the heap blob (allocated once per Q6_K weight at
 * resolve_weight): weights | w_scales | w_offsets. Sizes derive from n_in,
 * n_out, W8A8_BLOCK_ELEMS. The fp32 arrays follow the byte array; since
 * heap_alloc_aligned gives 64-byte alignment and the byte array size is a
 * multiple of W8A8_BLOCK_ELEMS (= 16), the float starts are 16-byte
 * aligned which is sufficient for fp32 loads. */
static inline size_t blocks_per_row(size_t n_in) {
    return n_in / W8A8_BLOCK_ELEMS;
}
static inline size_t weights_bytes_per_row(size_t n_in) {
    return blocks_per_row(n_in) * W8A8_BLOCK_ELEMS;
}

static void blob_pointers(const uint8_t *blob,
                          size_t         n_in,
                          size_t         n_out,
                          const uint8_t **weights_out,
                          const float  **scales_out,
                          const float  **offsets_out) {
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

/* The interleaved W8x8 region is appended after the row-major blob; it has
 * the same byte size (weights + 2 scale arrays — a pure permutation). */
static void w8x8_pointers(const uint8_t *blob,
                          size_t         n_in,
                          size_t         n_out,
                          const uint8_t **qs_out,
                          const float  **scales_out,
                          const float  **offsets_out) {
    const size_t row_major_bytes =
            n_out * weights_bytes_per_row(n_in) +
            2 * n_out * blocks_per_row(n_in) * sizeof(float);
    const size_t scales_count = n_out * blocks_per_row(n_in);
    const uint8_t *base = blob + row_major_bytes;

    *qs_out      = base;
    *scales_out  = (const float *) (base + n_out * weights_bytes_per_row(n_in));
    *offsets_out = *scales_out + scales_count;
}

[[nodiscard]] enum geist_status cpu_x86_linear_q6k_resolve(struct geist_weight *w) {
    if (w == nullptr || w->raw == nullptr || w->n_in <= 0 || w->n_out <= 0) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t n_in  = (size_t) w->n_in;
    const size_t n_out = (size_t) w->n_out;
    if (n_in % Q6_K_BLOCK_ELEMS != 0) {
        return GEIST_E_INVALID_ARG;
    }

    const size_t weights_total = n_out * weights_bytes_per_row(n_in);
    const size_t scales_total  = n_out * blocks_per_row(n_in) * sizeof(float);
    const size_t row_major_bytes = weights_total + 2 * scales_total;
    /* Append a lane-parallel W8x8 copy (same size) when prefill can use it.
     * ponytail: ~2× memory for Q6_K tensors, matching the Q4_K dual-blob
     * (linear_q4k.c). Drop the row-major copy and serve m1 from W8x8 too if
     * footprint matters. */
    const bool   build_w8x8  = q6k_use_w8x8(n_out);
    const size_t blob_bytes  = build_w8x8 ? 2 * row_major_bytes : row_major_bytes;

    uint8_t *blob = heap_alloc_aligned(blob_bytes, OPTIMAL_ALIGNMENT);
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

    const size_t q6k_row_bytes = (n_in / Q6_K_BLOCK_ELEMS) * Q6_K_BLOCK_BYTES;
    const size_t w_row_bytes   = weights_bytes_per_row(n_in);
    const size_t s_row_count   = blocks_per_row(n_in);
    const uint8_t *q6k_raw     = (const uint8_t *) w->raw;
    for (size_t m = 0; m < n_out; m++) {
        q6k_to_w8a8_row(n_in,
                        q6k_raw + m * q6k_row_bytes,
                        blob_w + m * w_row_bytes,
                        blob_s + m * s_row_count,
                        blob_o + m * s_row_count);
    }

    if (build_w8x8) {
        const uint8_t *il_qs;
        const float   *il_s;
        const float   *il_o;
        w8x8_pointers(blob, n_in, n_out, &il_qs, &il_s, &il_o);
        if (n_out % W8X16_NROWS == 0) {
            w8x16_repack(n_out, n_in, blob_w, blob_s, blob_o,
                         (uint8_t *) il_qs, (float *) il_s, (float *) il_o);
        } else {
            w8x8_repack(n_out, n_in, blob_w, blob_s, blob_o,
                        (uint8_t *) il_qs, (float *) il_s, (float *) il_o);
        }
    }

    w->aux_fp32  = (const float *) blob;
    w->aux_n     = (int32_t) blob_bytes;
    w->flags    |= GEIST_W_AUX_HEAP_OWNED | GEIST_W_AUX_BACKEND_REPACK;
    w->linear_m1 = cpu_x86_linear_q6k_m1;
    w->linear_mN = cpu_x86_linear_q6k_mN;
    return GEIST_OK;
}

/* The activation pipeline (int8 quant + per-block sum_a) is shared with
 * W4A8 — block sizes differ (W4A8: 32, W8A8: 16) but the activation
 * buffer is a row of int8 values and sum_a is summed in W8A8_BLOCK_ELEMS
 * chunks here. We re-quantize and re-sum locally; w4a8_quantize_acts_row
 * sums in W4A8_BLOCK_ELEMS chunks (32), which is wrong for W8A8, so do
 * a small inline sum at W8A8_BLOCK_ELEMS granularity. */
static void quantize_acts_w8a8(size_t n_in,
                                struct cpu_x86_state *st,
                                const float          *x,
                                float                *scale_x_out) {
    /* Use quantize_x_int8_sym directly via the W4A8 wrapper, which
     * stores int8 acts into st->acts_scratch. Discard the W4A8 sum_a
     * (block_elems=32) — recompute at W8A8 granularity below. */
    *scale_x_out = w4a8_quantize_acts_row(
            n_in, x, st->acts_scratch, st->sum_a_scratch);

    /* Re-sum acts at 16-element granularity for W8A8. Overwrites the
     * 32-elem sum_a since they're not used by Q6_K decode. */
    const size_t n_blocks = n_in / W8A8_BLOCK_ELEMS;
    const int8_t *acts    = st->acts_scratch;
    for (size_t b = 0; b < n_blocks; b++) {
        int32_t s = 0;
        for (size_t i = 0; i < W8A8_BLOCK_ELEMS; i++) {
            s += (int32_t) acts[b * W8A8_BLOCK_ELEMS + i];
        }
        st->sum_a_scratch[b] = s;
    }
}

void cpu_x86_linear_q6k_m1(const float               *x,
                           const struct geist_weight *w,
                           struct geist_backend      *be,
                           float                     *y) {
    struct cpu_x86_state *st               = (struct cpu_x86_state *) be->state;
    const size_t          n_in             = (size_t) w->n_in;
    const size_t          n_out            = (size_t) w->n_out;
    const size_t          n_blocks_per_row = blocks_per_row(n_in);

    const uint8_t *weights;
    const float   *w_scales;
    const float   *w_offsets;
    blob_pointers((const uint8_t *) w->aux_fp32, n_in, n_out,
                  &weights, &w_scales, &w_offsets);

    /* Decode straight from the native Q6_K weights (w->raw, ~0.82 B/wt) when
     * available, instead of the W8A8 predecode (1.5 B/wt). Q6_K decode
     * (ffn_down, lm_head) is bandwidth-bound, so halving the weight traffic
     * is the win (docs/LINUX_X86_PERF_PROFILE.md). */
    if (w->raw != nullptr && n_in % Q6_K_BLOCK_ELEMS == 0) {
        q6k_gemv_m1(n_out, n_in, x, (const uint8_t *) w->raw, y);
        return;
    }

    float scale_x;
    quantize_acts_w8a8(n_in, st, x, &scale_x);

    w8a8_gemv(n_out, n_blocks_per_row,
              weights, w_scales, w_offsets,
              st->acts_scratch, st->sum_a_scratch, scale_x, y);
}

/* Prefill (M>1) path. Quantizes all m tokens to int8 once, then runs a
 * tiled W8A8 GEMM that reads each weight row once and reuses it across
 * the whole token batch — the amortization the scalar mN fallback lacked.
 * Q6_K ffn_down is the dominant prefill cost in Q4_K_M models
 * (docs/LINUX_X86_PERF_PROFILE.md); this is what makes it cheap. */
void cpu_x86_linear_q6k_mN(const float               *x,
                           const struct geist_weight *w,
                           size_t                     m,
                           struct geist_backend      *be,
                           float                     *y) {
    struct cpu_x86_state *st               = (struct cpu_x86_state *) be->state;
    const size_t          n_in             = (size_t) w->n_in;
    const size_t          n_out            = (size_t) w->n_out;
    const size_t          n_blocks_per_row = blocks_per_row(n_in);

    const uint8_t *weights;
    const float   *w_scales;
    const float   *w_offsets;
    blob_pointers((const uint8_t *) w->aux_fp32, n_in, n_out,
                  &weights, &w_scales, &w_offsets);

    /* Per-call scratch: int8 acts + 16-elem sum_a + per-token scale, all m
     * tokens. Small relative to the GEMM (m=64,n_in=12288 ≈ 836 KB). */
    int8_t  *acts    = heap_alloc_aligned(m * n_in * sizeof(int8_t), OPTIMAL_ALIGNMENT);
    int32_t *sum_a   = heap_alloc_aligned(m * n_blocks_per_row * sizeof(int32_t),
                                          OPTIMAL_ALIGNMENT);
    float   *scale_x = heap_alloc_aligned(m * sizeof(float), OPTIMAL_ALIGNMENT);
    if (acts == nullptr || sum_a == nullptr || scale_x == nullptr) {
        safe_free((void **) &acts);
        safe_free((void **) &sum_a);
        safe_free((void **) &scale_x);
        for (size_t row = 0; row < m; row++) {
            cpu_x86_linear_q6k_m1(x + row * n_in, w, be, y + row * n_out);
        }
        return;
    }

    for (size_t j = 0; j < m; j++) {
        /* w4a8 quantizer gives int8 acts + per-row scale; its 32-elem sum_a
         * is the wrong granularity for W8A8 so discard it into scratch and
         * re-sum at 16. */
        scale_x[j] = w4a8_quantize_acts_row(
                n_in, x + j * n_in, acts + j * n_in, st->sum_a_scratch);
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
        const uint8_t *il_qs;
        const float   *il_s;
        const float   *il_o;
        w8x8_pointers((const uint8_t *) w->aux_fp32, n_in, n_out, &il_qs, &il_s, &il_o);
        if (n_out % W8X16_NROWS == 0) {
            w8x16_gemm(m, n_out, n_blocks_per_row, il_qs, il_s, il_o, acts, sum_a, scale_x, y);
        } else {
            w8x8_gemm(m, n_out, n_blocks_per_row, il_qs, il_s, il_o, acts, sum_a, scale_x, y);
        }
    } else {
        w8a8_gemm(m, n_out, n_blocks_per_row,
                  weights, w_scales, w_offsets,
                  acts, sum_a, scale_x, y);
    }

    safe_free((void **) &acts);
    safe_free((void **) &sum_a);
    safe_free((void **) &scale_x);
}

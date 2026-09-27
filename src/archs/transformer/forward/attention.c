/*
 * src/archs/transformer/forward/attention.c — KIVI cache drain + the
 * two attention paths (KIVI-2bit and INT8 KV) used by
 * transformer_forward_one_layer.
 *
 * Layer: ARCHITECTURE (private to forward/).
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "internal.h"
#include "../arch_state.h"

#include "int4_kv.h"
#include "kivi.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

/* ---- KIVI helpers ----------------------------------------------------- *
 *
 * Drain ONE group (KIVI_K_GROUP_SIZE residual tokens) of one layer into
 * the drained 2-bit cache via the kivi.h primitives. Updates that layer's
 * portion of the cache; the caller (transformer_kivi_drain_full)
 * tracks the shared drain/residual counters across layers. After
 * packing, memmoves the surviving residual entries to the front.
 *
 * Buffer pointers via buffer_map by the caller. Geometry parameters
 * R, head_dim, n_kv_heads are per layer (R is KIVI-fixed). */
void kivi_drain_one_layer(size_t   drained_count,
                          size_t   residual_count,
                          size_t   R,
                          size_t   head_dim,
                          size_t   n_kv_heads,
                          float   *k_residual,
                          float   *v_residual,
                          uint8_t *k_q4,
                          uint8_t *v_q4,
                          float   *k_scales,
                          float   *k_zeros,
                          float   *v_scales,
                          float   *v_zeros) {

    const size_t group_idx            = drained_count / R;
    const size_t packed_bytes_per_tok = head_dim / 4;
    /* Gather R rows for each kv_head, pack as channel-grouped K, and
     * per-row pack V. For Gemma 4, n_kv_heads=1 so the outer loop is a
     * single iteration in practice; coded generically. */
    for (size_t h = 0; h < n_kv_heads; h++) {
        /* K side: gather R contiguous rows from residual at column h. */
        float k_group[KIVI_K_GROUP_SIZE * TRANSFORMER_HEAD_DIM_MAX]; /* R=128: 256 KB */
        for (size_t t = 0; t < R; t++) {
            const float *src = k_residual + (t * n_kv_heads + h) * head_dim;
            memcpy(k_group + t * head_dim, src, head_dim * sizeof(float));
        }
        uint8_t *k_q4_grp     = k_q4 + (drained_count * n_kv_heads + h * R) * packed_bytes_per_tok;
        float   *k_scales_grp = k_scales + (group_idx * n_kv_heads + h) * head_dim;
        float   *k_zeros_grp  = k_zeros + (group_idx * n_kv_heads + h) * head_dim;
        kivi_pack_k_group(R, head_dim, k_group, k_q4_grp, k_scales_grp, k_zeros_grp);

        /* V side: per-token packed rows. */
        for (size_t t = 0; t < R; t++) {
            const float *src      = v_residual + (t * n_kv_heads + h) * head_dim;
            const size_t slot     = drained_count + t;
            uint8_t     *v_q4_row = v_q4 + (slot * n_kv_heads + h) * packed_bytes_per_tok;
            float       *v_sc     = v_scales + slot * n_kv_heads + h;
            float       *v_ze     = v_zeros + slot * n_kv_heads + h;
            kivi_pack_v_row(head_dim, src, v_q4_row, v_sc, v_ze);
        }
    }
    /* Shift survivors down. Layout is [t, kv_h, channel] contiguous so a
     * single memmove per side covers all kv_heads. */
    if (residual_count > R) {
        const size_t remaining  = residual_count - R;
        const size_t row_floats = n_kv_heads * head_dim;
        memmove(k_residual, k_residual + R * row_floats, remaining * row_floats * sizeof(float));
        memmove(v_residual, v_residual + R * row_floats, remaining * row_floats * sizeof(float));
    }
}

/* KIVI attention. For each kv_pos s in [s_lo..s_hi]:
 *   - s < drained_count → dequant the 2-bit K/V on the fly using the
 *     group's scales/zeros; FP32 dot.
 *   - else → read FP32 from the residual buffer.
 *
 * The drained path's K dequant fills a per-head buffer once per s and
 * dot-products against Q. V is dequant-and-weighted-sum inline.
 * Q is FP32 throughout (no INT8 sym-quant) to keep numerical fidelity
 * on the high-precision side. */
void attention_kivi_via_buffers(size_t         n_q,
                                size_t         n_q_heads,
                                size_t         head_dim,
                                size_t         n_kv,
                                size_t         n_kv_heads,
                                size_t         q_offset,
                                size_t         sliding_window,
                                size_t         drained_count,
                                size_t         R,
                                const float   *q,
                                const uint8_t *k_q4,
                                const float   *k_scales,
                                const float   *k_zeros,
                                const uint8_t *v_q4,
                                const float   *v_scales,
                                const float   *v_zeros,
                                const float   *k_residual,
                                const float   *v_residual,
                                float          scores[static n_kv],
                                float         *out) {

    const size_t kv_group_size  = n_q_heads / n_kv_heads;
    const size_t packed_per_row = head_dim / 4;
    float        k_dequant[TRANSFORMER_HEAD_DIM_MAX];
    for (size_t t = 0; t < n_q; t++) {
        const size_t q_pos = q_offset + t;
        const size_t s_lo =
                (sliding_window > 0 && q_pos + 1 > sliding_window) ? q_pos + 1 - sliding_window : 0;
        const size_t s_hi = q_pos < n_kv ? q_pos : n_kv - 1;

        for (size_t h = 0; h < n_q_heads; h++) {
            const size_t kv_h = h / kv_group_size;
            const float *qv   = q + (t * n_q_heads + h) * head_dim;

            float max_score = -INFINITY;
            for (size_t s = s_lo; s <= s_hi; s++) {
                float score = 0.0f;
                if (s < drained_count) {
                    const size_t   group_idx = s / R;
                    const float   *sc = k_scales + (group_idx * n_kv_heads + kv_h) * head_dim;
                    const float   *ze = k_zeros + (group_idx * n_kv_heads + kv_h) * head_dim;
                    const uint8_t *kq = k_q4 + (s * n_kv_heads + kv_h) * packed_per_row;
                    for (size_t i = 0; i < packed_per_row; i++) {
                        const uint8_t b      = kq[i];
                        k_dequant[4 * i + 0] = (float) (b & 0x3u) * sc[4 * i + 0] + ze[4 * i + 0];
                        k_dequant[4 * i + 1] =
                                (float) ((b >> 2) & 0x3u) * sc[4 * i + 1] + ze[4 * i + 1];
                        k_dequant[4 * i + 2] =
                                (float) ((b >> 4) & 0x3u) * sc[4 * i + 2] + ze[4 * i + 2];
                        k_dequant[4 * i + 3] =
                                (float) ((b >> 6) & 0x3u) * sc[4 * i + 3] + ze[4 * i + 3];
                    }
                    for (size_t i = 0; i < head_dim; i++) {
                        score += qv[i] * k_dequant[i];
                    }
                } else {
                    const size_t res_idx = s - drained_count;
                    const float *kr      = k_residual + (res_idx * n_kv_heads + kv_h) * head_dim;
                    for (size_t i = 0; i < head_dim; i++) {
                        score += qv[i] * kr[i];
                    }
                }
                scores[s] = score;
                if (score > max_score)
                    max_score = score;
            }
            double sum_exp = 0.0;
            for (size_t s = s_lo; s <= s_hi; s++) {
                const float e = expf(scores[s] - max_score);
                scores[s]     = e;
                sum_exp += e;
            }
            const float inv_sum = (float) (1.0 / sum_exp);

            float *outv = out + (t * n_q_heads + h) * head_dim;
            for (size_t i = 0; i < head_dim; i++)
                outv[i] = 0.0f;
            for (size_t s = s_lo; s <= s_hi; s++) {
                const float w = scores[s] * inv_sum;
                if (s < drained_count) {
                    const float    vs  = v_scales[s * n_kv_heads + kv_h];
                    const float    vz  = v_zeros[s * n_kv_heads + kv_h];
                    const uint8_t *vq  = v_q4 + (s * n_kv_heads + kv_h) * packed_per_row;
                    const float    wvs = w * vs;
                    const float    wvz = w * vz;
                    for (size_t i = 0; i < packed_per_row; i++) {
                        const uint8_t b = vq[i];
                        outv[4 * i + 0] += wvs * (float) (b & 0x3u) + wvz;
                        outv[4 * i + 1] += wvs * (float) ((b >> 2) & 0x3u) + wvz;
                        outv[4 * i + 2] += wvs * (float) ((b >> 4) & 0x3u) + wvz;
                        outv[4 * i + 3] += wvs * (float) ((b >> 6) & 0x3u) + wvz;
                    }
                } else {
                    const size_t res_idx = s - drained_count;
                    const float *vr      = v_residual + (res_idx * n_kv_heads + kv_h) * head_dim;
                    for (size_t i = 0; i < head_dim; i++) {
                        outv[i] += w * vr[i];
                    }
                }
            }
        }
    }
}

/* ---- INT8 attention, GQA-grouped passes ---------------------------------
 *
 * With grouped-query attention several query heads read the same KV head,
 * and the one-head loop in attention_int8_via_buffers streams every K and V
 * row once per query head. A pass over G query heads of one KV head loads
 * each K row once for G integer dots, and converts each V row to float
 * once for G accumulations.
 *
 * The pass walks the context in blocks of ATTN_BLOCK positions with an
 * online softmax: per head a running max and sum, the accumulator rescaled
 * when a block raises the max, the 1/sum applied once at the end. Its
 * scratch is fixed-size (about 20 KB of stack at G = 4) where the one-head
 * loop holds all n_kv scores; worker threads can have small stacks (musl
 * gives them 128 KB, and the release binaries are musl-static).
 *
 * Quantization, integer dots and scores are the one-head code; the softmax
 * normalization is applied in another order, and -ffast-math lets the
 * compiler regroup sums differently in each loop, so the two paths agree
 * to rounding (measured: within 2e-6 of the output scale), not bit for
 * bit. Which path runs therefore depends on the shape only, never on the
 * team: any thread count gives the same bits, as the one-head loop always
 * did.
 */

/* Work items a grouped call must leave: a fixed count, not the team size,
 * so that the path — and with it the result — cannot depend on how many
 * threads run it. 8 fills the 4-8-core hosts this was measured on. Decode
 * (n_q = 1) has only n_kv_heads * group / G items; on a host with more
 * cores than that, the one-head loop's G times as many items spread wider
 * and grouping may lose (not measured: splitting the context across items
 * would give decode that width back). */
constexpr size_t ATTN_MIN_ITEMS = 8;

/* Query heads per pass, at most. */
constexpr size_t ATTN_HEADS_PER_PASS_MAX = 4;

/* Grouped passes on x86 only for now: nothing was measured on NEON, where
 * the one-head loop has its own dot-product path. */
#if defined(__x86_64__)
constexpr bool ATTN_GROUPED_PASSES = true;
#else
constexpr bool ATTN_GROUPED_PASSES = false;
#endif

/* Context positions per online-softmax block: G * 512 scores (8 KB at
 * G = 4) stay in L1 between the dots, the exponentials and the V sums.
 * 128 to 1024 measured within noise of each other, 128 slowest. */
constexpr size_t ATTN_BLOCK = 512;

/* K and V bytes of all KV heads above which grouping pays at any head_dim:
 * the one-head loop reads them once per query head, from L2 while they
 * fit (2 MB here). Measured crossover 1.25-1.6 MB. */
constexpr size_t ATTN_KV_SPILL_BYTES = (size_t) 3 << 19;

/* Arguments of attention_int8_via_buffers, for the per-item body. */
struct attn_int8_args {
    size_t        n_q_heads, head_dim, n_kv, n_kv_heads, q_offset, sliding_window;
    const float  *q;
    const int8_t *k_q8;
    const float  *k_scale;
    const int8_t *v_q8;
    const float  *v_scale;
    float        *out;
};

/* How many query heads share a pass; 1 keeps the one-head loop. Measured
 * on a 4-thread Emerald Rapids (bench_attention_int8 and a forced-G sweep,
 * contexts 16-16384): passes of 4 heads, and any pass at head_dim >= 128,
 * were never slower (0 to -63 %); passes of 2 or 3 heads at head_dim 64
 * were 1-7 % slower while the K/V of the layer stayed under about 1.5 MB,
 * and faster above (-2 to -12 % at 1.5 MB, -21 to -36 % from 2.5 MB).
 * Bounded by parallelism: at least ATTN_MIN_ITEMS work items. */
static size_t attention_heads_per_pass(
        size_t n_q, size_t n_kv_heads, size_t group, size_t n_kv, size_t head_dim) {
    if (!ATTN_GROUPED_PASSES) {
        return 1;
    }
    const size_t kv_bytes = 2 * n_kv * n_kv_heads * head_dim;
    size_t       best     = 1;
    for (size_t g = 2; g <= ATTN_HEADS_PER_PASS_MAX && g <= group; g++) {
        const bool divides = group % g == 0;
        const bool busy    = n_q * n_kv_heads * (group / g) >= ATTN_MIN_ITEMS;
        const bool pays =
                head_dim >= 128 || g == ATTN_HEADS_PER_PASS_MAX || kv_bytes >= ATTN_KV_SPILL_BYTES;
        if (divides && busy && pays) {
            best = g;
        }
    }
    return best;
}

/* Query heads [h0, h0 + G) at position t, all reading KV head kv_h. G is a
 * compile-time constant at every call site (always_inline), so the loops
 * over the heads unroll and every array below is fixed-size. */
[[gnu::always_inline]] static inline void
attn_int8_item(size_t G, const struct attn_int8_args *a, size_t t, size_t kv_h, size_t h0) {
    /* Locals, not a->field in the loops: after OpenMP outlining `a` points
     * into the caller's frame, and under -fno-strict-aliasing every float
     * store below could alias it, so each access would be a reload. */
    const size_t  head_dim = a->head_dim, n_kv = a->n_kv, n_kv_heads = a->n_kv_heads;
    const size_t  n_q_heads = a->n_q_heads, sliding_window = a->sliding_window;
    const float  *q       = a->q;
    const int8_t *k_q8    = a->k_q8;
    const float  *k_scale = a->k_scale;
    const int8_t *v_q8    = a->v_q8;
    const float  *v_scale = a->v_scale;
    float        *out     = a->out;
    const size_t  q_pos   = a->q_offset + t;
    const size_t  s_lo =
            (sliding_window > 0 && q_pos + 1 > sliding_window) ? q_pos + 1 - sliding_window : 0;
    const size_t s_hi = q_pos < n_kv ? q_pos : n_kv - 1;

    int8_t q_q8[G][TRANSFORMER_HEAD_DIM_MAX];
    float  scale_q[G];
    for (size_t g = 0; g < G; g++) {
        const float *qv   = q + (t * n_q_heads + h0 + g) * head_dim;
        float        amax = 0.0f;
        for (size_t i = 0; i < head_dim; i++) {
            float x = fabsf(qv[i]);
            if (x > amax) {
                amax = x;
            }
        }
        float sq = amax / 127.0f;
        if (sq == 0.0f) {
            sq = 1.0f;
        }
        const float inv_q = 1.0f / sq;
        for (size_t i = 0; i < head_dim; i++) {
            q_q8[g][i] = (int8_t) lrintf(qv[i] * inv_q);
        }
        scale_q[g] = sq;
    }

    /* V is int8_t, a character type the compiler must assume aliases out[]:
     * every head accumulates in a local array, stored once at the end. */
    float  sc[G][ATTN_BLOCK];
    float  acc[G][TRANSFORMER_HEAD_DIM_MAX];
    float  vf[TRANSFORMER_HEAD_DIM_MAX];
    float  max_score[G];
    double sum_exp[G];
    for (size_t g = 0; g < G; g++) {
        for (size_t i = 0; i < head_dim; i++) {
            acc[g][i] = 0.0f;
        }
        sum_exp[g] = 0.0;
    }
    for (size_t b0 = s_lo; b0 <= s_hi; b0 += ATTN_BLOCK) {
        const size_t n = s_hi - b0 < ATTN_BLOCK ? s_hi - b0 + 1 : ATTN_BLOCK;
        for (size_t j = 0; j < n; j++) {
            const size_t  s  = b0 + j;
            const int8_t *k  = k_q8 + (s * n_kv_heads + kv_h) * head_dim;
            const float   ks = k_scale[s * n_kv_heads + kv_h];
            for (size_t g = 0; g < G; g++) {
                int32_t int_dot = 0;
                for (size_t i = 0; i < head_dim; i++) {
                    int_dot += (int32_t) q_q8[g][i] * (int32_t) k[i];
                }
                sc[g][j] = (float) int_dot * scale_q[g] * ks;
            }
        }
        for (size_t g = 0; g < G; g++) {
            float block_max = sc[g][0];
            for (size_t j = 1; j < n; j++) {
                if (sc[g][j] > block_max) {
                    block_max = sc[g][j];
                }
            }
            if (b0 == s_lo) {
                max_score[g] = block_max;
            } else if (block_max > max_score[g]) {
                /* The sum and the V sums so far were taken against the
                 * lower max: scale them down to the new one. */
                const float c = expf(max_score[g] - block_max);
                sum_exp[g] *= c;
                for (size_t i = 0; i < head_dim; i++) {
                    acc[g][i] *= c;
                }
                max_score[g] = block_max;
            }
            double block_sum = 0.0;
            for (size_t j = 0; j < n; j++) {
                float e  = expf(sc[g][j] - max_score[g]);
                sc[g][j] = e;
                block_sum += e;
            }
            sum_exp[g] += block_sum;
        }
        for (size_t j = 0; j < n; j++) {
            const size_t  s  = b0 + j;
            const int8_t *vv = v_q8 + (s * n_kv_heads + kv_h) * head_dim;
            const float   vs = v_scale[s * n_kv_heads + kv_h];
            for (size_t i = 0; i < head_dim; i++) {
                vf[i] = (float) vv[i];
            }
            for (size_t g = 0; g < G; g++) {
                const float wvs = sc[g][j] * vs;
                for (size_t i = 0; i < head_dim; i++) {
                    acc[g][i] += wvs * vf[i];
                }
            }
        }
    }
    for (size_t g = 0; g < G; g++) {
        const float inv_sum = (float) (1.0 / sum_exp[g]);
        float      *outv    = out + (t * n_q_heads + h0 + g) * head_dim;
        for (size_t i = 0; i < head_dim; i++) {
            outv[i] = acc[g][i] * inv_sum;
        }
    }
}

/* ---- INT8 attention helper for the KV-INT8 path -----------------------
 *
 * MQA causal attention with optional sliding window, where the K and V
 * caches are stored as INT8 with per-token-per-head FP32 scales. Q is
 * dynamically quantized per-head per-token; QK dot becomes a vdotq_s32
 * inner loop (when NEON is available) with scale_q * scale_k folded
 * scalarly per (q_pos, k_pos) pair.
 *
 * Port of lm.c::attention_mqa_causal_kv_int8, adapted to read inputs
 * via backend buffer_map host pointers instead of raw float*. CPU-only;
 * future GPU backends will need a vtable primitive (KV-INT8 attention).
 *
 * Inputs (all host pointers obtained via buffer_map by the caller):
 *   q[seq, n_q_heads, head_dim]                   F32
 *   k_q8[n_kv, n_kv_heads, head_dim]              INT8
 *   v_q8[n_kv, n_kv_heads, head_dim]              INT8
 *   k_scale[n_kv, n_kv_heads]                     F32
 *   v_scale[n_kv, n_kv_heads]                     F32
 *   out[seq, n_q_heads, head_dim]                 F32 */
void attention_int8_via_buffers(size_t        n_q,
                                size_t        n_q_heads,
                                size_t        head_dim,
                                size_t        n_kv,
                                size_t        n_kv_heads,
                                size_t        q_offset,
                                size_t        sliding_window,
                                const float  *q,
                                const int8_t *k_q8,
                                const float  *k_scale,
                                const int8_t *v_q8,
                                const float  *v_scale,
                                float        *out) {

    const size_t kv_group_size = n_q_heads / n_kv_heads;
    const size_t per_pass =
            attention_heads_per_pass(n_q, n_kv_heads, kv_group_size, n_kv, head_dim);
    if (per_pass > 1) {
        /* GQA-grouped passes (see attn_int8_item). Same item independence as
         * the one-head loop below, and the choice of this path depends on the
         * shape only: any thread count gives the same bits. */
        const size_t                n_passes = kv_group_size / per_pass;
        const struct attn_int8_args a        = {.n_q_heads      = n_q_heads,
                                                .head_dim       = head_dim,
                                                .n_kv           = n_kv,
                                                .n_kv_heads     = n_kv_heads,
                                                .q_offset       = q_offset,
                                                .sliding_window = sliding_window,
                                                .q              = q,
                                                .k_q8           = k_q8,
                                                .k_scale        = k_scale,
                                                .v_q8           = v_q8,
                                                .v_scale        = v_scale,
                                                .out            = out};
#if defined(_OPENMP)
#pragma omp parallel for collapse(3) schedule(dynamic)
#endif
        for (size_t t = 0; t < n_q; t++) {
            for (size_t kv_h = 0; kv_h < n_kv_heads; kv_h++) {
                for (size_t pass = 0; pass < n_passes; pass++) {
                    const size_t h0 = kv_h * kv_group_size + pass * per_pass;
                    switch (per_pass) {
                    case 4:
                        attn_int8_item(4, &a, t, kv_h, h0);
                        break;
                    case 3:
                        attn_int8_item(3, &a, t, kv_h, h0);
                        break;
                    default:
                        attn_int8_item(2, &a, t, kv_h, h0);
                        break;
                    }
                }
            }
        }
        return;
    }
    /* The O(n^2) attention core. Every (t,h) is independent: it reads the
     * shared Q/K/V and writes only its own out[(t*n_q_heads+h)*head_dim..]
     * slice plus a private `scores` scratch, so this parallelizes with no
     * change to any per-(t,h) reduction order — bit-exact vs serial.
     *
     * collapse(2), NOT a plain loop over t: decode passes n_q == 1, and a
     * parallel loop over one iteration runs on one thread. Measured on a Pi 5
     * (BitNet 2B-4T i2_s), the cost of growing context was identical at 1 and
     * at 3 threads — 22.87 vs 22.67 ms/token from a 32- to a 512-token prompt,
     * a factor of 1.01 — while everything else in decode scaled 1.77x. The
     * heads are the axis that still has width when t does not.
     *
     * Causal + sliding-window masking makes per-t work uneven (later positions
     * attend to more keys), so schedule(dynamic). Guarded so a non-OpenMP build
     * (no -fopenmp) skips the pragma cleanly under -Wunknown-pragmas -Werror. */
#if defined(_OPENMP)
#pragma omp parallel for collapse(2) schedule(dynamic)
#endif
    for (size_t t = 0; t < n_q; t++) {
        for (size_t h = 0; h < n_q_heads; h++) {
            /* Recomputed per (t,h) rather than hoisted to the t loop: three
             * scalar ops, and perfect nesting is what collapse(2) requires. */
            const size_t q_pos = q_offset + t;
            const size_t s_lo  = (sliding_window > 0 && q_pos + 1 > sliding_window)
                                         ? q_pos + 1 - sliding_window
                                         : 0;
            const size_t s_hi  = q_pos < n_kv ? q_pos : n_kv - 1;
            float        scores[n_kv]; /* private per (t,h) */

            const size_t kv_h = h / kv_group_size;
            const float *qv   = q + (t * n_q_heads + h) * head_dim;

            /* Per-head INT8 quant of Q[t,h,:]; head_dim <= TRANSFORMER_HEAD_DIM_MAX
             * (enforced at load). */
            int8_t q_q8[TRANSFORMER_HEAD_DIM_MAX];
            float  amax = 0.0f;
            for (size_t i = 0; i < head_dim; i++) {
                float a = fabsf(qv[i]);
                if (a > amax) {
                    amax = a;
                }
            }
            float scale_q = amax / 127.0f;
            if (scale_q == 0.0f) {
                scale_q = 1.0f;
            }
            const float inv_q = 1.0f / scale_q;
            for (size_t i = 0; i < head_dim; i++) {
                q_q8[i] = (int8_t) lrintf(qv[i] * inv_q);
            }

            for (size_t s = s_lo; s <= s_hi; s++) {
                const int8_t *k       = k_q8 + (s * n_kv_heads + kv_h) * head_dim;
                const float   ks      = k_scale[s * n_kv_heads + kv_h];
                int32_t       int_dot = 0;
/* vdotq_s32 is FEAT_DotProd, not baseline NEON: __ARM_NEON is set on every
 * armv8-a, so guarding the dot-product path on it faults on cores without
 * dotprod (Cortex-A53/A72, generic armv8-a builds). The scalar #else below
 * is the fallback that was always meant to run there. */
#if defined(__ARM_FEATURE_DOTPROD)
                int32x4_t acc = vdupq_n_s32(0);
                size_t    i   = 0;
                for (; i + 16 <= head_dim; i += 16) {
                    acc = vdotq_s32(acc, vld1q_s8(q_q8 + i), vld1q_s8(k + i));
                }
                int_dot = vaddvq_s32(acc);
                for (; i < head_dim; i++) {
                    int_dot += (int32_t) q_q8[i] * (int32_t) k[i];
                }
#else
                for (size_t i = 0; i < head_dim; i++) {
                    int_dot += (int32_t) q_q8[i] * (int32_t) k[i];
                }
#endif
                scores[s] = (float) int_dot * scale_q * ks;
            }

            float max_score = scores[s_lo];
            for (size_t s = s_lo + 1; s <= s_hi; s++) {
                if (scores[s] > max_score) {
                    max_score = scores[s];
                }
            }
            double sum_exp = 0.0;
            for (size_t s = s_lo; s <= s_hi; s++) {
                float e   = expf(scores[s] - max_score);
                scores[s] = e;
                sum_exp += e;
            }
            const float inv_sum = (float) (1.0 / sum_exp);

            float *outv = out + (t * n_q_heads + h) * head_dim;
            for (size_t i = 0; i < head_dim; i++) {
                outv[i] = 0.0f;
            }
            for (size_t s = s_lo; s <= s_hi; s++) {
                const int8_t *vv  = v_q8 + (s * n_kv_heads + kv_h) * head_dim;
                const float   vs  = v_scale[s * n_kv_heads + kv_h];
                const float   wvs = scores[s] * inv_sum * vs;
                for (size_t i = 0; i < head_dim; i++) {
                    outv[i] += wvs * (float) vv[i];
                }
            }
        }
    }
}

/* Packed-INT4 attention. Identical to attention_int8_via_buffers except each
 * K/V cache row is unpacked from head_dim/2 bytes into a stack int8 row
 * before the (reused) int8 dot / weighted-sum. See internal.h.
 *
 * int4_unpack_row fully writes [0,head_dim); the NEON tail reads only that
 * range, so GCC's -Wmaybe-uninitialized on the unpack buffers is a false
 * positive — suppressed here rather than paid for with a per-row zero-init. */
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
void attention_int4_via_buffers(size_t         n_q,
                                size_t         n_q_heads,
                                size_t         head_dim,
                                size_t         n_kv,
                                size_t         n_kv_heads,
                                size_t         q_offset,
                                size_t         sliding_window,
                                const float   *q,
                                const uint8_t *k_q4,
                                const float   *k_scale,
                                const uint8_t *v_q4,
                                const float   *v_scale,
                                float         *out) {

    const size_t kv_group_size = n_q_heads / n_kv_heads;
    const size_t packed        = head_dim / 2; /* bytes per cache row */
/* collapse(2) for the same reason as the INT8 core above: decode passes
 * n_q == 1, so only the head axis has width to parallelize over. */
#if defined(_OPENMP)
#pragma omp parallel for collapse(2) schedule(dynamic)
#endif
    for (size_t t = 0; t < n_q; t++) {
        for (size_t h = 0; h < n_q_heads; h++) {
            const size_t q_pos = q_offset + t;
            const size_t s_lo  = (sliding_window > 0 && q_pos + 1 > sliding_window)
                                         ? q_pos + 1 - sliding_window
                                         : 0;
            const size_t s_hi  = q_pos < n_kv ? q_pos : n_kv - 1;
            float        scores[n_kv]; /* private per (t,h) */

            const size_t kv_h = h / kv_group_size;
            const float *qv   = q + (t * n_q_heads + h) * head_dim;

            int8_t q_q8[TRANSFORMER_HEAD_DIM_MAX];
            float  amax = 0.0f;
            for (size_t i = 0; i < head_dim; i++) {
                float a = fabsf(qv[i]);
                if (a > amax) {
                    amax = a;
                }
            }
            float scale_q = amax / 127.0f;
            if (scale_q == 0.0f) {
                scale_q = 1.0f;
            }
            const float inv_q = 1.0f / scale_q;
            for (size_t i = 0; i < head_dim; i++) {
                q_q8[i] = (int8_t) lrintf(qv[i] * inv_q);
            }

            for (size_t s = s_lo; s <= s_hi; s++) {
                int8_t k[TRANSFORMER_HEAD_DIM_MAX];
                int4_unpack_row(head_dim, k_q4 + (s * n_kv_heads + kv_h) * packed, k);
                const float ks      = k_scale[s * n_kv_heads + kv_h];
                int32_t     int_dot = 0;
/* vdotq_s32 is FEAT_DotProd, not baseline NEON: __ARM_NEON is set on every
 * armv8-a, so guarding the dot-product path on it faults on cores without
 * dotprod (Cortex-A53/A72, generic armv8-a builds). The scalar #else below
 * is the fallback that was always meant to run there. */
#if defined(__ARM_FEATURE_DOTPROD)
                int32x4_t acc = vdupq_n_s32(0);
                size_t    i   = 0;
                for (; i + 16 <= head_dim; i += 16) {
                    acc = vdotq_s32(acc, vld1q_s8(q_q8 + i), vld1q_s8(k + i));
                }
                int_dot = vaddvq_s32(acc);
                for (; i < head_dim; i++) {
                    int_dot += (int32_t) q_q8[i] * (int32_t) k[i];
                }
#else
                for (size_t i = 0; i < head_dim; i++) {
                    int_dot += (int32_t) q_q8[i] * (int32_t) k[i];
                }
#endif
                scores[s] = (float) int_dot * scale_q * ks;
            }

            float max_score = scores[s_lo];
            for (size_t s = s_lo + 1; s <= s_hi; s++) {
                if (scores[s] > max_score) {
                    max_score = scores[s];
                }
            }
            double sum_exp = 0.0;
            for (size_t s = s_lo; s <= s_hi; s++) {
                float e   = expf(scores[s] - max_score);
                scores[s] = e;
                sum_exp += e;
            }
            const float inv_sum = (float) (1.0 / sum_exp);

            float *outv = out + (t * n_q_heads + h) * head_dim;
            for (size_t i = 0; i < head_dim; i++) {
                outv[i] = 0.0f;
            }
            for (size_t s = s_lo; s <= s_hi; s++) {
                int8_t vv[TRANSFORMER_HEAD_DIM_MAX];
                int4_unpack_row(head_dim, v_q4 + (s * n_kv_heads + kv_h) * packed, vv);
                const float vs  = v_scale[s * n_kv_heads + kv_h];
                const float wvs = scores[s] * inv_sum * vs;
                for (size_t i = 0; i < head_dim; i++) {
                    outv[i] += wvs * (float) vv[i];
                }
            }
        }
    }
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

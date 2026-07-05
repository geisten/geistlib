/*
 * src/archs/transformer/forward/attention.c — KIVI cache drain + the
 * two attention paths (KIVI-2bit and INT8 KV) used by
 * transformer_forward_one_layer.
 *
 * Layer: ARCHITECTURE (private to forward/).
 *
 * Extracted from forward.c during R4 of the C23/AGENT.md cleanup.
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
void kivi_drain_one_layer(float   *k_residual,
                          float   *v_residual,
                          uint8_t *k_q4,
                          uint8_t *v_q4,
                          float   *k_scales,
                          float   *k_zeros,
                          float   *v_scales,
                          float   *v_zeros,
                          size_t   drained_count,
                          size_t   residual_count,
                          size_t   R,
                          size_t   head_dim,
                          size_t   n_kv_heads) {

    const size_t group_idx            = drained_count / R;
    const size_t packed_bytes_per_tok = head_dim / 4;
    /* Gather R rows for each kv_head, pack as channel-grouped K, and
     * per-row pack V. For Gemma 4, n_kv_heads=1 so the outer loop is a
     * single iteration in practice; coded generically. */
    for (size_t h = 0; h < n_kv_heads; h++) {
        /* K side: gather R contiguous rows from residual at column h. */
        float k_group[KIVI_K_GROUP_SIZE * 512]; /* worst case R=128, hd=512 → 256 KB */
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
void attention_kivi_via_buffers(const float   *q,
                                size_t         n_q,
                                size_t         n_q_heads,
                                size_t         head_dim,
                                const uint8_t *k_q4,
                                const float   *k_scales,
                                const float   *k_zeros,
                                const uint8_t *v_q4,
                                const float   *v_scales,
                                const float   *v_zeros,
                                const float   *k_residual,
                                const float   *v_residual,
                                size_t         n_kv,
                                size_t         n_kv_heads,
                                size_t         q_offset,
                                size_t         sliding_window,
                                size_t         drained_count,
                                size_t         R,
                                float          scores[static n_kv],
                                float         *out) {

    const size_t kv_group_size  = n_q_heads / n_kv_heads;
    const size_t packed_per_row = head_dim / 4;
    float        k_dequant[512];
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
void attention_int8_via_buffers(const float  *q,
                                size_t        n_q,
                                size_t        n_q_heads,
                                size_t        head_dim,
                                const int8_t *k_q8,
                                const float  *k_scale,
                                const int8_t *v_q8,
                                const float  *v_scale,
                                size_t        n_kv,
                                size_t        n_kv_heads,
                                size_t        q_offset,
                                size_t        sliding_window,
                                float        *out) {

    const size_t kv_group_size = n_q_heads / n_kv_heads;
    /* The O(n^2) attention core. Every (t,h) is independent: it reads the
     * shared Q/K/V and writes only its own out[(t*n_q_heads+h)*head_dim..]
     * slice plus a private `scores` scratch, so the outer loop parallelizes
     * with no change to any per-(t,h) reduction order — bit-exact vs serial.
     * Causal + sliding-window masking makes per-t work uneven (later positions
     * attend to more keys), so schedule(dynamic). The FFN/projection matmuls
     * are already threaded in the backend; this closes the one prefill phase
     * that was still serial. Guarded so a non-OpenMP build (no -fopenmp) skips
     * the pragma cleanly under -Wunknown-pragmas -Werror. */
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
    for (size_t t = 0; t < n_q; t++) {
        const size_t q_pos = q_offset + t;
        const size_t s_lo =
                (sliding_window > 0 && q_pos + 1 > sliding_window) ? q_pos + 1 - sliding_window : 0;
        const size_t s_hi = q_pos < n_kv ? q_pos : n_kv - 1;
        float        scores[n_kv]; /* private per t-iteration (was a shared param) */

        for (size_t h = 0; h < n_q_heads; h++) {
            const size_t kv_h = h / kv_group_size;
            const float *qv   = q + (t * n_q_heads + h) * head_dim;

            /* Per-head INT8 quant of Q[t,h,:]. head_dim ≤ 512 in Gemma 4. */
            int8_t q_q8[512];
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
#if defined(__ARM_NEON)
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
 * before the (reused) int8 dot / weighted-sum. See internal.h. */
void attention_int4_via_buffers(const float   *q,
                                size_t         n_q,
                                size_t         n_q_heads,
                                size_t         head_dim,
                                const uint8_t *k_q4,
                                const float   *k_scale,
                                const uint8_t *v_q4,
                                const float   *v_scale,
                                size_t         n_kv,
                                size_t         n_kv_heads,
                                size_t         q_offset,
                                size_t         sliding_window,
                                float         *out) {

    const size_t kv_group_size = n_q_heads / n_kv_heads;
    const size_t packed        = head_dim / 2; /* bytes per cache row */
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
    for (size_t t = 0; t < n_q; t++) {
        const size_t q_pos = q_offset + t;
        const size_t s_lo =
                (sliding_window > 0 && q_pos + 1 > sliding_window) ? q_pos + 1 - sliding_window : 0;
        const size_t s_hi = q_pos < n_kv ? q_pos : n_kv - 1;
        float        scores[n_kv];

        for (size_t h = 0; h < n_q_heads; h++) {
            const size_t kv_h = h / kv_group_size;
            const float *qv   = q + (t * n_q_heads + h) * head_dim;

            int8_t q_q8[512];
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
                int8_t k[512];
                int4_unpack_row(k_q4 + (s * n_kv_heads + kv_h) * packed, k, head_dim);
                const float ks      = k_scale[s * n_kv_heads + kv_h];
                int32_t     int_dot = 0;
#if defined(__ARM_NEON)
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
                int8_t vv[512];
                int4_unpack_row(v_q4 + (s * n_kv_heads + kv_h) * packed, vv, head_dim);
                const float vs  = v_scale[s * n_kv_heads + kv_h];
                const float wvs = scores[s] * inv_sum * vs;
                for (size_t i = 0; i < head_dim; i++) {
                    outv[i] += wvs * (float) vv[i];
                }
            }
        }
    }
}

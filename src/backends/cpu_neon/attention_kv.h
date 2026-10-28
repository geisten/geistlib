/*
 * src/backends/cpu_neon/attention_kv.h — the attention loop and the
 * argument checks shared by attention_int8.c and attention_int4.c.
 *
 * Layer: BACKEND (cpu_neon). Included only by those two files, under
 * __ARM_NEON && __ARM_FEATURE_DOTPROD.
 *
 * The two caches differ only in how a V row reaches the int8 sums: the
 * INT8 cache is read in place, the INT4 one unpacked into a stack row
 * first (int4_kv.h). K is INT8 in both (#53). The including file says
 * which with a
 *
 *     constexpr bool NKV_INT4 = ...;
 *
 * before the #include. A constant rather than a parameter: the loop body
 * runs as a geist_par_for callback, called through a pointer, so a
 * parameter could not be propagated into it, and only a constant visible
 * there specializes the body per cache, so neither pays for the other's
 * branch.
 */
#ifndef GEIST_CPU_NEON_ATTENTION_KV_H
#define GEIST_CPU_NEON_ATTENTION_KV_H

#include "internal.h"
#include "par.h"

#include "gemma4_kernels.h" /* ATTN_EXP_FLOOR, ATTN_F32_BLOCK */
#include "int4_kv.h"
#include "host_ops.h"

#include <geist.h>
#include <geist_backend.h>

#include <arm_neon.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

/* The architecture refuses a larger head_dim at load
 * (TRANSFORMER_HEAD_DIM_MAX); the stack arrays below are sized by it. */
constexpr size_t NKV_HEAD_DIM_MAX = 512;

/* scores[j] = exp(scores[j] - max_score) for j < n, the exponent clamped at
 * ATTN_EXP_FLOOR (gemma4_kernels.h); returns their sum. Out of line: inlined,
 * the vector exp call perturbs register allocation of the whole loop nest
 * (INT4 decode 2-8 % slower, measured on x86). */
[[gnu::noinline]] static double attn_exp_block(size_t n, float scores[static n], float max_score) {
    double sum = 0.0;
    for (size_t j = 0; j < n; j++) {
        scores[j] = expf(fmaxf(scores[j] - max_score, ATTN_EXP_FLOOR));
        sum += scores[j];
    }
    return sum;
}

/* Row `row` of the V cache as head_dim int8 values: in place for INT8,
 * unpacked from head_dim / 2 bytes into `buf` for INT4. */
static inline const int8_t *
nkv_v_row(size_t head_dim, const void *cache, size_t row, int8_t buf[static head_dim]) {
    if (NKV_INT4) {
        int4_unpack_row(head_dim, (const uint8_t *) cache + row * (head_dim / 2), buf);
        return buf;
    }
    return (const int8_t *) cache + row * head_dim;
}

/* One nkv_run call, for nkv_items. */
struct nkv_job {
    size_t        n_q_heads, head_dim, n_kv, n_kv_heads, q_offset, sliding_window;
    const float  *q;
    const int8_t *k_cache;
    const float  *k_scale;
    const void   *v_cache;
    const float  *v_scale;
    float        *out;
};

/* The kernel on host pointers, validated by nkv_attention: q and out
 * [n_q, n_q_heads, head_dim], k [n_kv, n_kv_heads, head_dim] int8, v the
 * same or [n_kv, n_kv_heads, head_dim / 2] packed INT4 bytes, the scales
 * [n_kv, n_kv_heads].
 *
 * int4_unpack_row fully writes [0, head_dim), and nothing reads past it, so
 * GCC's -Wmaybe-uninitialized on the unpack buffers is a false positive,
 * suppressed here rather than paid for with a zeroed row. */
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
/* Items [i0, i1) of nkv_run, item i the query position t = i / n_q_heads
 * and head h = i % n_q_heads. */
static void nkv_items(void *ctx, size_t i0, size_t i1) {
    const struct nkv_job *job            = ctx;
    const size_t          n_q_heads      = job->n_q_heads;
    const size_t          head_dim       = job->head_dim;
    const size_t          n_kv           = job->n_kv;
    const size_t          n_kv_heads     = job->n_kv_heads;
    const size_t          q_offset       = job->q_offset;
    const size_t          sliding_window = job->sliding_window;
    const float          *q              = job->q;
    const int8_t         *k_cache        = job->k_cache;
    const float          *k_scale        = job->k_scale;
    const void           *v_cache        = job->v_cache;
    const float          *v_scale        = job->v_scale;
    float                *out            = job->out;
    const size_t          kv_group_size  = n_q_heads / n_kv_heads;
    for (size_t it = i0; it < i1; it++) {
        const size_t t     = it / n_q_heads;
        const size_t h     = it % n_q_heads;
        const size_t q_pos = q_offset + t;
        const size_t s_lo =
                (sliding_window > 0 && q_pos + 1 > sliding_window) ? q_pos + 1 - sliding_window : 0;
        const size_t s_hi = q_pos < n_kv ? q_pos : n_kv - 1;
        /* One block of the context at a time, private per (t,h): the
         * softmax runs online, so the stack does not hold n_kv scores. */
        float scores[ATTN_F32_BLOCK];

        const size_t kv_h = h / kv_group_size;
        const float *qv   = q + (t * n_q_heads + h) * head_dim;

        /* Per-head INT8 quant of Q[t,h,:]; head_dim <= NKV_HEAD_DIM_MAX
         * (the entry refuses more). */
        int8_t q_q8[NKV_HEAD_DIM_MAX];
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

        float *outv = out + (t * n_q_heads + h) * head_dim;
        for (size_t i = 0; i < head_dim; i++) {
            outv[i] = 0.0f;
        }
        float  max_score = 0.0f;
        double sum_exp   = 0.0;
        for (size_t b0 = s_lo; b0 <= s_hi; b0 += ATTN_F32_BLOCK) {
            const size_t n = s_hi - b0 < ATTN_F32_BLOCK ? s_hi - b0 + 1 : ATTN_F32_BLOCK;
            for (size_t j = 0; j < n; j++) {
                const size_t  s       = b0 + j;
                const size_t  r       = s * n_kv_heads + kv_h;
                const int8_t *k       = k_cache + r * head_dim;
                const float   ks      = k_scale[r];
                int32_t       int_dot = 0;
                int32x4_t     acc     = vdupq_n_s32(0);
                size_t        i       = 0;
                for (; i + 16 <= head_dim; i += 16) {
                    acc = vdotq_s32(acc, vld1q_s8(q_q8 + i), vld1q_s8(k + i));
                }
                int_dot = vaddvq_s32(acc);
                for (; i < head_dim; i++) {
                    int_dot += (int32_t) q_q8[i] * (int32_t) k[i];
                }
                scores[j] = (float) int_dot * scale_q * ks;
            }

            float block_max = scores[0];
            for (size_t j = 1; j < n; j++) {
                if (scores[j] > block_max) {
                    block_max = scores[j];
                }
            }
            if (b0 == s_lo) {
                max_score = block_max;
            } else if (block_max > max_score) {
                /* The sum and the V sums so far were taken against the
                 * lower max: scale them down to the new one (to 0 below
                 * ATTN_EXP_FLOOR). */
                const float d = max_score - block_max;
                const float c = d < ATTN_EXP_FLOOR ? 0.0f : expf(d);
                sum_exp *= c;
                for (size_t i = 0; i < head_dim; i++) {
                    outv[i] *= c;
                }
                max_score = block_max;
            }
            sum_exp += attn_exp_block(n, scores, max_score);

            for (size_t j = 0; j < n; j++) {
                const size_t  s = b0 + j;
                const size_t  r = s * n_kv_heads + kv_h;
                int8_t        v_buf[NKV_HEAD_DIM_MAX];
                const int8_t *vv  = nkv_v_row(head_dim, v_cache, r, v_buf);
                const float   vs  = v_scale[r];
                const float   wvs = scores[j] * vs;
                for (size_t i = 0; i < head_dim; i++) {
                    outv[i] += wvs * (float) vv[i];
                }
            }
        }
        const float inv_sum = (float) (1.0 / sum_exp);
        for (size_t i = 0; i < head_dim; i++) {
            outv[i] *= inv_sum;
        }
    }
}

static void nkv_run(size_t        n_q,
                    size_t        n_q_heads,
                    size_t        head_dim,
                    size_t        n_kv,
                    size_t        n_kv_heads,
                    size_t        q_offset,
                    size_t        sliding_window,
                    const float  *q,
                    const int8_t *k_cache,
                    const float  *k_scale,
                    const void   *v_cache,
                    const float  *v_scale,
                    float        *out) {
    /* The O(n^2) attention core. Every (t,h) is independent: it reads the
     * shared Q/K/V and writes only its own out[(t*n_q_heads+h)*head_dim..]
     * slice plus a private `scores` scratch, so this parallelizes with no
     * change to any per-(t,h) reduction order — bit-exact vs serial.
     *
     * One item per (t,h), NOT per t: decode passes n_q == 1, and a parallel
     * loop over one iteration runs on one thread; the heads are the axis
     * that still has width when t does not.
     *
     * Causal + sliding-window masking makes per-t work uneven (later positions
     * attend to more keys), so the items are handed out one at a time
     * (OpenMP's schedule(dynamic) before #618). */
    struct nkv_job job = {.n_q_heads      = n_q_heads,
                          .head_dim       = head_dim,
                          .n_kv           = n_kv,
                          .n_kv_heads     = n_kv_heads,
                          .q_offset       = q_offset,
                          .sliding_window = sliding_window,
                          .q              = q,
                          .k_cache        = k_cache,
                          .k_scale        = k_scale,
                          .v_cache        = v_cache,
                          .v_scale        = v_scale,
                          .out            = out};
    geist_par_for_dynamic(n_q * n_q_heads, 1, nkv_items, &job);
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

/* The shape probe of this TU's op, defined by the including .c file. It is
 * static because nkv_attention passes its address: the address of an
 * extern function, taken in -fPIE code, is a PC-relative relocation that
 * ld rejects when it links libgeist.a into libgeist.so (the symbol may
 * bind externally there). */
static bool nkv_supported(const struct geist_fusion_query *q);

/* fused->attention_kv_int8 or fused->attention_kv_int4, per NKV_INT4:
 * checks the views and the shapes, then runs. */
static inline enum geist_status nkv_attention(struct geist_backend                 *be,
                                              const struct geist_attention_kv_args *args) {
    struct geist_host_attn_kvq a;
    const enum geist_status    s = geist_host_attn_kvq_views(
            be,
            NKV_INT4 ? "attention_kv_int4" : "attention_kv_int8",
            args,
            NKV_INT4,
            NKV_INT4 ? GEIST_FUSED_ATTN_KV_INT4 : GEIST_FUSED_ATTN_KV_INT8,
            nkv_supported,
            &a);
    if (s != GEIST_OK) {
        return s;
    }
    nkv_run(a.n_q,
            a.n_q_heads,
            a.head_dim,
            a.n_kv,
            a.n_kv_heads,
            args->q_offset,
            args->sliding_window,
            a.q,
            a.k,
            a.k_scale,
            a.v,
            a.v_scale,
            a.out);
    return GEIST_OK;
}

#endif /* GEIST_CPU_NEON_ATTENTION_KV_H */

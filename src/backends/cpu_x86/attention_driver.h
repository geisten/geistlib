/*
 * src/backends/cpu_x86/attention_driver.h — the work split shared by
 * cpu_x86's attention kernels: attention.c (FP32 KV), attention_int8.c
 * (INT8 KV, AVX2) and attention_int8_avx512_vnni.c (INT8 KV, AVX-512 VNNI).
 *
 * Layer: BACKEND (cpu_x86, internal).
 *
 * A call is cut into work items — up to four query heads of one KV head
 * (a pass) and, in a prefill, up to four queries; a decode is split across
 * the context into chunks whose partial results are merged afterwards —
 * by a plan that depends on the shape alone (ax_plan_for). The items are
 * independent and the merge order is fixed, so any thread count gives the
 * same bits.
 *
 * Everything here is static inline: each including file compiles its own
 * copy with its own -m flags, so the driver of the VNNI kernel is VNNI
 * code and no EVEX instruction reaches the AVX2 files. Each including file
 * defines ax_run_item, its item kernel (declared below), which ax_run
 * calls by name: a function pointer would reach the OpenMP-outlined loops
 * as an indirect call.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_X86_ATTENTION_DRIVER_H
#define GEIST_INTERNAL_BACKEND_CPU_X86_ATTENTION_DRIVER_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_x86/attention_driver.h is internal to the backend layer."
#endif

#include "gemma4_kernels.h" /* ATTN_EXP_FLOOR */

#include <math.h>
#include <stddef.h>

/* The architecture refuses a larger head_dim at load
 * (TRANSFORMER_HEAD_DIM_MAX); the kernels' stack arrays are sized by it. */
constexpr size_t AX_HEAD_DIM_MAX = 512;

/* Context positions per online-softmax block, as in the portable loop.
 * 128 to 1024 measured within noise of each other in the INT8 kernel. */
constexpr size_t AX_BLOCK = 512;

/* Query heads per pass, at most: the V accumulators of four heads fill
 * the eight registers the AVX2 kernels keep (16 dimensions each). */
constexpr size_t AX_HEADS_PER_PASS_MAX = 4;

/* Decode chunks per pass, at most: the part buffer holds that many
 * records per head. */
constexpr size_t AX_MAX_CHUNKS = 4;

/* A decode is split into one chunk per this many positions, so that each
 * chunk's K and V stay in L2 (INT8 and FP32 cache) ... */
constexpr size_t AX_CHUNK_SPAN_I8  = 1024;
constexpr size_t AX_CHUNK_SPAN_F32 = 256;

/* ... and into as many as give this many work items, if its passes give
 * fewer (MQA decode has one per pass) ... */
constexpr size_t AX_MIN_ITEMS = 8;

/* ... but no chunk shorter than this: its setup and the merge would cost
 * more than it saves. */
constexpr size_t AX_CHUNK_MIN = 128;

/* Prefill: queries per work item, at most. An item runs each block of the
 * context for all its queries before the next block, so that the block's K
 * and V bytes, fetched for the first query, are near for the others ... */
constexpr size_t AX_QUERIES_MAX = 4;

/* ... where the K and V bytes a call reads exceed this: below it they stay
 * cached between queries anyway. */
constexpr size_t AX_REUSE_BYTES = (size_t) 1 << 20;

/* Floats of partial results a split decode needs (0 < n_q_heads). */
static inline size_t ax_part_floats(size_t n_q_heads, size_t head_dim) {
    return n_q_heads * AX_MAX_CHUNKS * (head_dim + 2);
}

/* One call, as the item kernels read it: q and out [n_q, n_q_heads,
 * head_dim], k and v [n_kv, n_kv_heads, head_dim] of int8_t or float, the
 * scales [n_kv, n_kv_heads] (INT8 only; nullptr for FP32). */
struct ax_args {
    size_t       n_q_heads, head_dim, n_kv, n_kv_heads, q_offset, sliding_window;
    const float *q;
    const void  *k;
    const float *k_scale;
    const void  *v;
    const float *v_scale;
    float       *out;
};

/* The context positions [*s_lo, *s_hi] query t attends to; *s_lo > *s_hi
 * when there are none. */
static inline void ax_span(const struct ax_args *a, size_t t, size_t *s_lo, size_t *s_hi) {
    const size_t q_pos = a->q_offset + t;
    *s_lo = (a->sliding_window > 0 && q_pos + 1 > a->sliding_window) ? q_pos + 1 - a->sliding_window
                                                                     : 0;
    *s_hi = q_pos < a->n_kv ? q_pos : a->n_kv - 1;
}

struct ax_plan {
    size_t per_pass; /* query heads per pass */
    size_t n_chunks; /* decode: context chunks per pass, merged afterwards */
    size_t per_item; /* prefill: queries per work item */
};

/* How to run a call, from its shape alone (any thread count gives the same
 * bits); `span` is the number of positions the last query attends to,
 * kv_bytes the size of a K or V element, chunk_span the decode split's
 * positions per chunk. Measured per kernel with every choice forced, 4
 * threads, decode and 64-row prefill over 32/8 heads at head_dim 64, 16/8,
 * 24/8 and 32/8 at 128, 15/5 and 32/32 at 64, 8/1 at 256 and 512, contexts
 * 256-8192:
 *   - the widest pass that divides the KV group was never slower, prefill
 *     or decode: one head a pass took up to twice as long (INT8), 21-141 %
 *     longer (FP32), two of a group of four or eight 10-67 % (FP32);
 *   - INT8 decode: a chunk per 1024 positions ran 7-45 % faster at 8192
 *     positions than unsplit (32/8 hd 64: 0.50 -> 0.44 ms; 8/1 hd 256:
 *     0.60 -> 0.30), within noise of it at 2048, and slower at 512, where
 *     the split is left to the item count;
 *   - FP32 decode: a chunk per 256 positions ran 3-17 % faster at 1024 and
 *     2048 positions than one per 1024 with five to 32 KV heads, the same
 *     at 512 and with one KV head (four chunks either way); unsplit took
 *     up to twice as long with one KV head, 11-47 % longer at 4096 and
 *     8192 positions with five to eight.
 * A prefill puts its queries in items of up to AX_QUERIES_MAX where a span
 * runs past one block and the call reads more than AX_REUSE_BYTES of K and
 * V, in items of as many as leave at least AX_MIN_ITEMS items; the queries
 * of an item do not change each other's arithmetic, so any plan gives the
 * same bits. Against one query an item, 64-token chunks: up to 36 % (AVX2)
 * and 4-44 % (VNNI) faster at 2048 and 8192 positions where the call reads
 * more than 1 MB, the least at 1.3 MB (SmolLM2-360M at 2048 positions);
 * in FP32 one query an item took 1-31 % longer at 1024 positions and
 * 11-91 % from 2048 on, two -2 to +30 %. With the queries forced into
 * items anyway, a few percent slower within one block and at 1 MB (MQA at
 * 2048 positions), and with fewer items the threads waited on the longest
 * (MQA with four queries a call, two items, 20-80 % slower in the VNNI
 * kernel). */
static inline struct ax_plan ax_plan_for(size_t n_q,
                                         size_t n_q_heads,
                                         size_t n_kv_heads,
                                         size_t head_dim,
                                         size_t kv_bytes,
                                         size_t chunk_span,
                                         size_t span,
                                         size_t part_floats) {
    const size_t group    = n_q_heads / n_kv_heads;
    size_t       per_pass = 1;
    for (size_t g = AX_HEADS_PER_PASS_MAX; g >= 2; g--) {
        if (group % g == 0) {
            per_pass = g;
            break;
        }
    }
    struct ax_plan plan  = {.per_pass = per_pass, .n_chunks = 1, .per_item = 1};
    const size_t   items = n_kv_heads * (group / per_pass); /* per query */
    if (n_q == 1) {
        const size_t for_items = (AX_MIN_ITEMS + items - 1) / items;
        const size_t fit       = span / AX_CHUNK_MIN;
        size_t       chunks    = span / chunk_span;
        chunks                 = chunks > for_items ? chunks : for_items;
        chunks                 = chunks < AX_MAX_CHUNKS ? chunks : AX_MAX_CHUNKS;
        chunks                 = chunks < fit ? chunks : fit;
        if (chunks >= 2 && part_floats >= n_q_heads * chunks * (head_dim + 2)) {
            plan.n_chunks = chunks;
        }
    } else if (span > AX_BLOCK && span * n_kv_heads * head_dim * 2 * kv_bytes > AX_REUSE_BYTES) {
        for (size_t per_item = AX_QUERIES_MAX; per_item >= 2; per_item /= 2) {
            if (items * ((n_q + per_item - 1) / per_item) >= AX_MIN_ITEMS) {
                plan.per_item = per_item;
                break;
            }
        }
    }
    return plan;
}

/* One head's output from its n_chunks partial results (records `stride`
 * floats apart: head_dim V sums, the max, the sum of exponentials),
 * rescaled to their common max and summed in chunk order: the portable
 * merge (attn_int8_merge). Zeros if the sum is not positive. */
static inline void
ax_merge(size_t n_chunks, size_t head_dim, size_t stride, const float *part, float *out) {
    float max_score = part[head_dim];
    for (size_t c = 1; c < n_chunks; c++) {
        const float m = part[c * stride + head_dim];
        max_score     = m > max_score ? m : max_score;
    }
    float  acc[AX_HEAD_DIM_MAX];
    double sum_exp = 0.0;
    for (size_t i = 0; i < head_dim; i++) {
        acc[i] = 0.0f;
    }
    for (size_t c = 0; c < n_chunks; c++) {
        const float *rec = part + c * stride;
        const float  d   = rec[head_dim] - max_score;
        const float  w   = d < ATTN_EXP_FLOOR ? 0.0f : expf(d);
        sum_exp += (double) rec[head_dim + 1] * w;
        for (size_t i = 0; i < head_dim; i++) {
            acc[i] += w * rec[i];
        }
    }
    const float inv_sum = sum_exp > 0.0 ? (float) (1.0 / sum_exp) : 0.0f;
    for (size_t i = 0; i < head_dim; i++) {
        out[i] = acc[i] * inv_sum;
    }
}

/* Defined by each including file: query heads [h0, h0 + per_pass) at the
 * tn positions t0 .. t0 + tn - 1, all reading KV head kv_h, position t0 +
 * i over context positions [lo[i], hi[i]]. Without `part` it writes their
 * output; with it (one query), its partial results: per head head_dim + 2
 * floats, the unnormalized V sums, the running max and the sum of
 * exponentials. */
static void ax_run_item(size_t                per_pass,
                        const struct ax_args *a,
                        size_t                t0,
                        size_t                tn,
                        size_t                kv_h,
                        size_t                h0,
                        const size_t          lo[static tn],
                        const size_t          hi[static tn],
                        float                *part);

/* The kernel on host pointers, validated by the caller (see struct
 * ax_args; n_q_heads a multiple of n_kv_heads, head_dim at most
 * AX_HEAD_DIM_MAX); `part` holds part_floats floats for a split decode, or
 * is nullptr (no split). */
[[gnu::always_inline]] static inline void ax_run(size_t       n_q,
                                                 size_t       n_q_heads,
                                                 size_t       head_dim,
                                                 size_t       n_kv,
                                                 size_t       n_kv_heads,
                                                 size_t       part_floats,
                                                 size_t       q_offset,
                                                 size_t       sliding_window,
                                                 size_t       kv_bytes,
                                                 size_t       chunk_span,
                                                 const float *q,
                                                 const void  *k,
                                                 const float *k_scale,
                                                 const void  *v,
                                                 const float *v_scale,
                                                 float       *out,
                                                 float       *part) {
    const struct ax_args a      = {.n_q_heads      = n_q_heads,
                                   .head_dim       = head_dim,
                                   .n_kv           = n_kv,
                                   .n_kv_heads     = n_kv_heads,
                                   .q_offset       = q_offset,
                                   .sliding_window = sliding_window,
                                   .q              = q,
                                   .k              = k,
                                   .k_scale        = k_scale,
                                   .v              = v,
                                   .v_scale        = v_scale,
                                   .out            = out};
    const size_t         group  = n_q_heads / n_kv_heads;
    size_t               dec_lo = 0, dec_hi = 0, last_lo = 0, last_hi = 0;
    ax_span(&a, 0, &dec_lo, &dec_hi);
    ax_span(&a, n_q - 1, &last_lo, &last_hi);
    const size_t         span     = last_lo <= last_hi ? last_hi - last_lo + 1 : 0;
    const struct ax_plan plan     = ax_plan_for(n_q,
                                                n_q_heads,
                                                n_kv_heads,
                                                head_dim,
                                                kv_bytes,
                                                chunk_span,
                                                span,
                                                part != nullptr ? part_floats : 0);
    const size_t         per_pass = plan.per_pass;
    const size_t         n_passes = group / per_pass;
    if (plan.n_chunks > 1) {
        /* Split decode (n_q == 1): items (KV head, pass, chunk) leave their
         * partial results in `part`, then each head merges its chunks.
         * Chunk c covers positions [dec_lo + c * len, ...], at least
         * AX_CHUNK_MIN each, so none is empty. */
        const size_t n_chunks = plan.n_chunks;
        const size_t len      = (dec_hi - dec_lo + n_chunks) / n_chunks;
        const size_t rec      = per_pass * (head_dim + 2); /* one item's records */
#if defined(_OPENMP)
#pragma omp parallel
#endif
        {
#if defined(_OPENMP)
#pragma omp for collapse(3) schedule(dynamic)
#endif
            for (size_t kv_h = 0; kv_h < n_kv_heads; kv_h++) {
                for (size_t pass = 0; pass < n_passes; pass++) {
                    for (size_t c = 0; c < n_chunks; c++) {
                        const size_t c_lo = dec_lo + c * len;
                        const size_t c_hi = dec_hi - c_lo < len ? dec_hi : c_lo + len - 1;
                        ax_run_item(per_pass,
                                    &a,
                                    0,
                                    1,
                                    kv_h,
                                    kv_h * group + pass * per_pass,
                                    &c_lo,
                                    &c_hi,
                                    part + ((kv_h * n_passes + pass) * n_chunks + c) * rec);
                    }
                }
            }
#if defined(_OPENMP)
#pragma omp for collapse(2)
#endif
            for (size_t kv_h = 0; kv_h < n_kv_heads; kv_h++) {
                for (size_t hg = 0; hg < group; hg++) {
                    const size_t pass = hg / per_pass;
                    ax_merge(n_chunks,
                             head_dim,
                             rec,
                             part + (kv_h * n_passes + pass) * n_chunks * rec +
                                     hg % per_pass * (head_dim + 2),
                             out + (kv_h * group + hg) * head_dim);
                }
            }
        }
        return;
    }
    /* Items by KV head, then pass, then block of queries: the team works
     * through one KV head's rows at a time, which then stay in each core's
     * L2 (1 MB of INT8 at 8192 positions and head_dim 64), where in query
     * order every thread went through all KV heads' (five of them in
     * SmolLM2-360M, 5 MB). Causal and window masks make later positions
     * longer: dynamic. */
    const size_t per_item = plan.per_item;
    const size_t n_blocks = (n_q + per_item - 1) / per_item;
#if defined(_OPENMP)
#pragma omp parallel for collapse(3) schedule(dynamic)
#endif
    for (size_t kv_h = 0; kv_h < n_kv_heads; kv_h++) {
        for (size_t pass = 0; pass < n_passes; pass++) {
            for (size_t qb = 0; qb < n_blocks; qb++) {
                const size_t t0 = qb * per_item;
                const size_t tn = n_q - t0 < per_item ? n_q - t0 : per_item;
                size_t       lo[AX_QUERIES_MAX], hi[AX_QUERIES_MAX];
                for (size_t tq = 0; tq < tn; tq++) {
                    ax_span(&a, t0 + tq, &lo[tq], &hi[tq]);
                }
                ax_run_item(per_pass,
                            &a,
                            t0,
                            tn,
                            kv_h,
                            kv_h * group + pass * per_pass,
                            lo,
                            hi,
                            nullptr);
            }
        }
    }
}

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_ATTENTION_DRIVER_H */

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
 * defines ax_run_item, its item kernel (declared below), which the range
 * bodies of ax_run call by name: a function pointer would make every item
 * an indirect call.
 *
 * Items run on geist_par_for. Their cost varies (causal and window masks
 * make later positions longer), so they are handed out one at a time from
 * a shared counter, in order, to one range per thread: OpenMP's
 * schedule(dynamic), which these loops used before.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_X86_ATTENTION_DRIVER_H
#define GEIST_INTERNAL_BACKEND_CPU_X86_ATTENTION_DRIVER_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_x86/attention_driver.h is internal to the backend layer."
#endif

#include "gemma4_kernels.h" /* ATTN_EXP_FLOOR */

#include "par.h"

#include <math.h>
#include <stdatomic.h>
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
 * positions per chunk. Chosen per kernel with every option forced (4
 * threads, head layouts 8/1 to 32/32, head_dim 64-512, contexts 256-8192):
 *   - the widest pass that divides the KV group was never slower;
 *   - decode: one chunk per chunk_span positions, or more where the passes
 *     alone give fewer than AX_MIN_ITEMS items, none below AX_CHUNK_MIN;
 *   - prefill: up to AX_QUERIES_MAX queries an item where a span runs past
 *     one block and the call reads more than AX_REUSE_BYTES of K and V,
 *     keeping at least AX_MIN_ITEMS items (with fewer, threads wait on the
 *     longest). The queries of an item do not change each other's
 *     arithmetic, so any plan gives the same bits. */
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

/* One call's work for geist_par_for. */
struct ax_job {
    const struct ax_args *a;
    size_t                group, per_pass, n_passes;
    size_t                n_chunks, len, rec, dec_lo, dec_hi; /* split decode */
    size_t                n_q, per_item, n_blocks;            /* otherwise */
    size_t                n_items;
    float                *part;
    atomic_size_t         next; /* the next item to hand out */
};

/* Split decode: items (KV head, pass, chunk), in that order, until none is
 * left; each leaves its partial results in `part`. Chunk c covers positions
 * [dec_lo + c * len, ...], at least AX_CHUNK_MIN each, so none is empty. */
static inline void ax_chunk_items(void *ctx, size_t, size_t) {
    struct ax_job *j = ctx;
    for (size_t i;
         (i = atomic_fetch_add_explicit(&j->next, 1, memory_order_relaxed)) < j->n_items;) {
        const size_t c    = i % j->n_chunks;
        const size_t pass = i / j->n_chunks % j->n_passes;
        const size_t kv_h = i / j->n_chunks / j->n_passes;
        const size_t c_lo = j->dec_lo + c * j->len;
        const size_t c_hi = j->dec_hi - c_lo < j->len ? j->dec_hi : c_lo + j->len - 1;
        ax_run_item(j->per_pass,
                    j->a,
                    0,
                    1,
                    kv_h,
                    kv_h * j->group + pass * j->per_pass,
                    &c_lo,
                    &c_hi,
                    j->part + i * j->rec);
    }
}

/* Split decode: query heads [h0, h1) merge their chunks. */
static inline void ax_merge_heads(void *ctx, size_t h0, size_t h1) {
    const struct ax_job *j        = ctx;
    const size_t         head_dim = j->a->head_dim;
    for (size_t h = h0; h < h1; h++) {
        const size_t kv_h = h / j->group;
        const size_t hg   = h % j->group;
        const size_t pass = hg / j->per_pass;
        ax_merge(j->n_chunks,
                 head_dim,
                 j->rec,
                 j->part + (kv_h * j->n_passes + pass) * j->n_chunks * j->rec +
                         hg % j->per_pass * (head_dim + 2),
                 j->a->out + h * head_dim);
    }
}

/* Otherwise: items (KV head, pass, block of queries), in that order, until
 * none is left. */
static inline void ax_query_items(void *ctx, size_t, size_t) {
    struct ax_job *j = ctx;
    for (size_t i;
         (i = atomic_fetch_add_explicit(&j->next, 1, memory_order_relaxed)) < j->n_items;) {
        const size_t qb   = i % j->n_blocks;
        const size_t pass = i / j->n_blocks % j->n_passes;
        const size_t kv_h = i / j->n_blocks / j->n_passes;
        const size_t t0   = qb * j->per_item;
        const size_t tn   = j->n_q - t0 < j->per_item ? j->n_q - t0 : j->per_item;
        size_t       lo[AX_QUERIES_MAX], hi[AX_QUERIES_MAX];
        for (size_t tq = 0; tq < tn; tq++) {
            ax_span(j->a, t0 + tq, &lo[tq], &hi[tq]);
        }
        ax_run_item(j->per_pass,
                    j->a,
                    t0,
                    tn,
                    kv_h,
                    kv_h * j->group + pass * j->per_pass,
                    lo,
                    hi,
                    nullptr);
    }
}

/* Runs fn on min(n_items, threads) ranges, one item counter for all. */
static inline void ax_dynamic(struct ax_job *j, geist_par_fn fn) {
    const size_t threads = geist_par_max_threads();
    atomic_init(&j->next, 0);
    geist_par_for(j->n_items < threads ? j->n_items : threads, fn, j);
}

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
    const size_t         span = last_lo <= last_hi ? last_hi - last_lo + 1 : 0;
    const struct ax_plan plan = ax_plan_for(n_q,
                                            n_q_heads,
                                            n_kv_heads,
                                            head_dim,
                                            kv_bytes,
                                            chunk_span,
                                            span,
                                            part != nullptr ? part_floats : 0);
    struct ax_job        j    = {.a        = &a,
                                 .group    = group,
                                 .per_pass = plan.per_pass,
                                 .n_passes = group / plan.per_pass,
                                 .part     = part};
    if (plan.n_chunks > 1) {
        /* Split decode (n_q == 1): the items, then each head merges its
         * chunks. Item i = (kv_h, pass, c) writes its records at part +
         * i * rec, so a pass's chunks lie side by side for the merge. */
        j.n_chunks = plan.n_chunks;
        j.len      = (dec_hi - dec_lo + j.n_chunks) / j.n_chunks;
        j.rec      = plan.per_pass * (head_dim + 2); /* one item's records */
        j.dec_lo   = dec_lo;
        j.dec_hi   = dec_hi;
        j.n_items  = n_kv_heads * j.n_passes * j.n_chunks;
        ax_dynamic(&j, ax_chunk_items);
        geist_par_for(n_q_heads, ax_merge_heads, &j);
        return;
    }
    /* Items by KV head, then pass, then block of queries: the threads work
     * through one KV head's rows at a time, which then stay in each core's
     * L2. */
    j.per_item = plan.per_item;
    j.n_blocks = (n_q + plan.per_item - 1) / plan.per_item;
    j.n_q      = n_q;
    j.n_items  = n_kv_heads * j.n_passes * j.n_blocks;
    ax_dynamic(&j, ax_query_items);
}

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_ATTENTION_DRIVER_H */

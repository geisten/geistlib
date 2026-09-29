/*
 * src/backends/cpu_x86/attention.c — AVX2/FMA causal attention over the FP32
 * KV cache (prims->attention).
 *
 * Layer: BACKEND (cpu_x86).
 *
 * The transformer calls this with the FP32 cache (GEIST_KV_INT8=0; F16
 * does not reach it, caps.kv_f16_attention is unset). It is laid out as the
 * INT8 kernel (attention_int8.c), with fp32 K and V:
 *   - a work item takes up to four query heads of one KV head and, in a
 *     prefill, up to four queries, and runs each block of 512 context
 *     positions for all of them before the next, so that the block's K and
 *     V rows, fetched for the first, are in L1 or L2 for the others. It
 *     was one (query, head) a thread, each reading its KV head's rows on
 *     its own: at 1024 positions and 32/8 heads, 1 GB of K and V reads a
 *     64-token chunk;
 *   - scores: eight positions at a time, reduced together;
 *   - V: eight fp32 accumulators stay in registers while 16 KB of the
 *     block's V rows go past — 16 output dimensions of each of 4 or 3
 *     heads, 32 of each of 2, 64 of one;
 *   - decode is split across the context into up to four chunks of at
 *     least 128 positions, merged afterwards: one per 256 positions, or
 *     more where the passes alone give too few work items (one KV head:
 *     two);
 *   - head_dim 64, 128 and 256 are compile-time constants, so the loops
 *     unroll; any other head_dim up to AF_HEAD_DIM_MAX runs the same code
 *     with the length at run time.
 * The plan (af_plan_for) is measured for this kernel. Shapes it does not
 * take (a KV group that does not divide the query heads, head_dim above
 * AF_HEAD_DIM_MAX: the architecture refuses both at load) run the
 * gemma4_kernels.c reference. Like it, nothing grows with the context and
 * nothing is allocated per call: the split decode's partial results go to
 * the calling thread's workspace, grown once.
 * Any thread count gives the same bits: the work items are independent,
 * the merge order is fixed and the plan depends on the shape alone.
 *
 * -march=x86-64-v3 (this backend's floor) guarantees AVX2 + FMA + F16C
 * unconditionally, so this file needs no runtime ISA dispatch (unlike the
 * quantized GEMM kernels, which vary support across x86-64-v3 hosts) —
 * see mk/backend-cpu_x86.mk.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "attention.h"
#include "backend_state.h"

#include "gemma4_kernels.h" /* ATTN_EXP_FLOOR, attention_mqa_causal_kv */
#include "tensor_view.h"

#include <geist.h>
#include <geist_backend.h>

#include <immintrin.h>
#include <math.h>
#include <stdalign.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* CPU buffer layout, owned by cpu_scalar's buffer_create (cpu_x86 inherits
 * its buffer vtable). Mirrored here as elementwise.c does — geist.h keeps
 * the struct opaque, but the host pointer is needed for the dense fast path. */
struct geist_buffer {
    void                  *host;
    size_t                 bytes;
    enum geist_buffer_role role;
    unsigned int           memory_flags;
};

static float *get_f32_dense_ptr_full(const struct geist_tensor *t, size_t *out_n) {
    if (t == nullptr || t->buffer == nullptr) {
        return nullptr;
    }
    return geist_tensor_f32_dense(t, t->buffer->host, t->buffer->bytes, out_n);
}

/* The architecture refuses a larger head_dim at load
 * (TRANSFORMER_HEAD_DIM_MAX); the stack arrays below are sized by it. */
constexpr size_t AF_HEAD_DIM_MAX = 512;

/* Context positions per online-softmax block, as in the reference. */
constexpr size_t AF_BLOCK = 512;

/* Bytes of V rows per pass over the output dimensions: they stay in L1
 * while every slice of the dimensions goes over them (64 rows at head_dim
 * 64, 8 at 512). 8 and 32 KB measured within 3 % of it, 64 KB 5-9 % slower
 * at head_dim 256 and 512. */
constexpr size_t AF_PV_BYTES = 16384;

/* Query heads per pass, at most: the V accumulators of four heads fill
 * the eight registers the pass keeps (16 dimensions each). */
constexpr size_t AF_HEADS_PER_PASS_MAX = 4;

/* Decode chunks per pass, at most: the part buffer holds that many
 * records per head. */
constexpr size_t AF_MAX_CHUNKS = 4;

/* A decode is split into one chunk per this many positions ... */
constexpr size_t AF_CHUNK_SPAN = 256;

/* ... and into as many as give this many work items, if its passes give
 * fewer (one KV head gives two) ... */
constexpr size_t AF_MIN_ITEMS = 8;

/* ... but no chunk shorter than this: its setup and the merge would cost
 * more than it saves. */
constexpr size_t AF_CHUNK_MIN = 128;

/* Prefill: queries per work item, at most ... */
constexpr size_t AF_QUERIES_MAX = 4;

/* ... where the K and V bytes a call reads exceed this: below it they stay
 * cached between queries anyway. */
constexpr size_t AF_REUSE_BYTES = (size_t) 1 << 20;

/* Floats of partial results a split decode needs. */
static size_t af_part_floats(size_t n_q_heads, size_t head_dim) {
    return n_q_heads * AF_MAX_CHUNKS * (head_dim + 2);
}

struct af_args {
    size_t       n_q_heads, head_dim, n_kv, n_kv_heads, q_offset, sliding_window;
    const float *q;
    const float *k;
    const float *v;
    float       *out;
};

/* The context positions [*s_lo, *s_hi] query t attends to; *s_lo > *s_hi
 * when there are none. */
static inline void af_span(const struct af_args *a, size_t t, size_t *s_lo, size_t *s_hi) {
    const size_t q_pos = a->q_offset + t;
    *s_lo = (a->sliding_window > 0 && q_pos + 1 > a->sliding_window) ? q_pos + 1 - a->sliding_window
                                                                     : 0;
    *s_hi = q_pos < a->n_kv ? q_pos : a->n_kv - 1;
}

struct af_plan {
    size_t per_pass; /* query heads per pass */
    size_t n_chunks; /* decode: context chunks per pass, merged afterwards */
    size_t per_item; /* prefill: queries per work item */
};

/* How to run a call, from its shape alone (any thread count gives the same
 * bits); `span` is the number of positions the last query attends to.
 * Measured with each choice forced against the plan, both in one process,
 * call by call, 4 threads, 64-query prefill chunks and decode over 32/8
 * heads at head_dim 64, 16/8 and 32/8 at 128, 15/5 and 32/32 at 64, 8/1 at
 * 256 and 512, 512-8192 positions:
 *   - the widest pass that divides the KV group was never slower: one head
 *     a pass took 21-141 % longer, two of a group of four or eight 10-67 %;
 *   - prefill: one query an item took 1-31 % longer at 1024 positions and
 *     11-91 % from 2048 on than up to four where a span runs past one
 *     block and the call reads more than AF_REUSE_BYTES of K and V; two,
 *     -2 to +30 %; within one block, where the plan takes one, the same;
 *   - decode: a chunk per 256 positions ran 3-17 % faster at 1024 and 2048
 *     positions than one per 1024 with five to 32 KV heads, the same at 512
 *     and with one KV head (four chunks either way); unsplit took up to
 *     twice as long with one KV head, 11-47 % longer at 4096 and 8192
 *     positions with five to eight.
 * The queries of an item and the chunks' merge order do not depend on the
 * thread count, so any plan gives the same bits for every thread count. */
static struct af_plan af_plan_for(size_t n_q,
                                  size_t n_q_heads,
                                  size_t n_kv_heads,
                                  size_t head_dim,
                                  size_t span,
                                  size_t part_floats) {
    const size_t group    = n_q_heads / n_kv_heads;
    size_t       per_pass = 1;
    for (size_t g = AF_HEADS_PER_PASS_MAX; g >= 2; g--) {
        if (group % g == 0) {
            per_pass = g;
            break;
        }
    }
    struct af_plan plan  = {.per_pass = per_pass, .n_chunks = 1, .per_item = 1};
    const size_t   items = n_kv_heads * (group / per_pass); /* per query */
    if (n_q == 1) {
        const size_t for_items = (AF_MIN_ITEMS + items - 1) / items;
        const size_t fit       = span / AF_CHUNK_MIN;
        size_t       chunks    = span / AF_CHUNK_SPAN;
        chunks                 = chunks > for_items ? chunks : for_items;
        chunks                 = chunks < AF_MAX_CHUNKS ? chunks : AF_MAX_CHUNKS;
        chunks                 = chunks < fit ? chunks : fit;
        if (chunks >= 2 && part_floats >= n_q_heads * chunks * (head_dim + 2)) {
            plan.n_chunks = chunks;
        }
    } else if (span > AF_BLOCK &&
               span * n_kv_heads * head_dim * 2 * sizeof(float) > AF_REUSE_BYTES) {
        for (size_t per_item = AF_QUERIES_MAX; per_item >= 2; per_item /= 2) {
            if (items * ((n_q + per_item - 1) / per_item) >= AF_MIN_ITEMS) {
                plan.per_item = per_item;
                break;
            }
        }
    }
    return plan;
}

/* ---- Scores ------------------------------------------------------------ */

/* The sums of v0 .. v7, lane p holding vp's. */
[[gnu::always_inline]] static inline __m256
af_reduce8(__m256 v0, __m256 v1, __m256 v2, __m256 v3, __m256 v4, __m256 v5, __m256 v6, __m256 v7) {
    const __m256 u0 = _mm256_hadd_ps(_mm256_hadd_ps(v0, v1), _mm256_hadd_ps(v2, v3));
    const __m256 u1 = _mm256_hadd_ps(_mm256_hadd_ps(v4, v5), _mm256_hadd_ps(v6, v7));
    return _mm256_add_ps(_mm256_permute2f128_ps(u0, u1, 0x20),
                         _mm256_permute2f128_ps(u0, u1, 0x31));
}

static inline float af_hsum(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s        = _mm_hadd_ps(s, s);
    s        = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

/* q . k for the eight rows k0 + p * row (lane p): the first hd8 elements
 * eight at a time, then the scalar tail to head_dim. */
[[gnu::always_inline]] static inline __m256
af_dot8(size_t hd8, size_t head_dim, size_t row, const float *q, const float *k0) {
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    for (size_t i = 0; i < hd8; i += 8) {
        const __m256 qv = _mm256_loadu_ps(q + i);
        a0              = _mm256_fmadd_ps(qv, _mm256_loadu_ps(k0 + i), a0);
        a1              = _mm256_fmadd_ps(qv, _mm256_loadu_ps(k0 + row + i), a1);
        a2              = _mm256_fmadd_ps(qv, _mm256_loadu_ps(k0 + 2 * row + i), a2);
        a3              = _mm256_fmadd_ps(qv, _mm256_loadu_ps(k0 + 3 * row + i), a3);
        a4              = _mm256_fmadd_ps(qv, _mm256_loadu_ps(k0 + 4 * row + i), a4);
        a5              = _mm256_fmadd_ps(qv, _mm256_loadu_ps(k0 + 5 * row + i), a5);
        a6              = _mm256_fmadd_ps(qv, _mm256_loadu_ps(k0 + 6 * row + i), a6);
        a7              = _mm256_fmadd_ps(qv, _mm256_loadu_ps(k0 + 7 * row + i), a7);
    }
    __m256 d = af_reduce8(a0, a1, a2, a3, a4, a5, a6, a7);
    if (hd8 < head_dim) {
        alignas(32) float tail[8];
        for (size_t p = 0; p < 8; p++) {
            float s = 0.0f;
            for (size_t i = hd8; i < head_dim; i++) {
                s += q[i] * k0[p * row + i];
            }
            tail[p] = s;
        }
        d = _mm256_add_ps(d, _mm256_load_ps(tail));
    }
    return d;
}

/* q . k for one row. */
static inline float af_dot1(size_t hd8, size_t head_dim, const float *q, const float *k) {
    __m256 a0 = _mm256_setzero_ps(), a1 = a0;
    size_t i = 0;
    for (; i + 16 <= hd8; i += 16) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + i), _mm256_loadu_ps(k + i), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(q + i + 8), _mm256_loadu_ps(k + i + 8), a1);
    }
    for (; i < hd8; i += 8) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + i), _mm256_loadu_ps(k + i), a0);
    }
    float s = af_hsum(_mm256_add_ps(a0, a1));
    for (; i < head_dim; i++) {
        s += q[i] * k[i];
    }
    return s;
}

/* ---- V ------------------------------------------------------------------- */

/* acc[g][c .. c + 16) += sum over the n rows j of w[g][j] * v[j * row + c ..]
 * for the G heads, in registers across the rows: the accumulators are
 * loaded once and stored once, not through an array (stores to a __m256
 * array may alias, and gcc kept every update in memory). */
[[gnu::always_inline]] static inline void af_pv16(size_t       G,
                                                  size_t       n,
                                                  size_t       row,
                                                  size_t       c,
                                                  const float *v,
                                                  const float (*w)[AF_BLOCK],
                                                  float (*acc)[AF_HEAD_DIM_MAX]) {
    __m256 l0 = _mm256_load_ps(acc[0] + c), h0 = _mm256_load_ps(acc[0] + c + 8);
    __m256 l1 = l0, h1 = h0, l2 = l0, h2 = h0, l3 = l0, h3 = h0;
    if (G > 1) {
        l1 = _mm256_load_ps(acc[1] + c);
        h1 = _mm256_load_ps(acc[1] + c + 8);
    }
    if (G > 2) {
        l2 = _mm256_load_ps(acc[2] + c);
        h2 = _mm256_load_ps(acc[2] + c + 8);
    }
    if (G > 3) {
        l3 = _mm256_load_ps(acc[3] + c);
        h3 = _mm256_load_ps(acc[3] + c + 8);
    }
    const float *vr = v + c;
    for (size_t j = 0; j < n; j++, vr += row) {
        const __m256 v0 = _mm256_loadu_ps(vr), v1 = _mm256_loadu_ps(vr + 8);
        __m256       s = _mm256_broadcast_ss(&w[0][j]);
        l0             = _mm256_fmadd_ps(s, v0, l0);
        h0             = _mm256_fmadd_ps(s, v1, h0);
        if (G > 1) {
            s  = _mm256_broadcast_ss(&w[1][j]);
            l1 = _mm256_fmadd_ps(s, v0, l1);
            h1 = _mm256_fmadd_ps(s, v1, h1);
        }
        if (G > 2) {
            s  = _mm256_broadcast_ss(&w[2][j]);
            l2 = _mm256_fmadd_ps(s, v0, l2);
            h2 = _mm256_fmadd_ps(s, v1, h2);
        }
        if (G > 3) {
            s  = _mm256_broadcast_ss(&w[3][j]);
            l3 = _mm256_fmadd_ps(s, v0, l3);
            h3 = _mm256_fmadd_ps(s, v1, h3);
        }
    }
    _mm256_store_ps(acc[0] + c, l0);
    _mm256_store_ps(acc[0] + c + 8, h0);
    if (G > 1) {
        _mm256_store_ps(acc[1] + c, l1);
        _mm256_store_ps(acc[1] + c + 8, h1);
    }
    if (G > 2) {
        _mm256_store_ps(acc[2] + c, l2);
        _mm256_store_ps(acc[2] + c + 8, h2);
    }
    if (G > 3) {
        _mm256_store_ps(acc[3] + c, l3);
        _mm256_store_ps(acc[3] + c + 8, h3);
    }
}

/* As af_pv16 over 32 dimensions of 2 heads, or 64 of one (NH heads, NV
 * vectors of 8 each, NH * NV = 8): 16 dimensions per head would leave the
 * FMAs one chain per register and bound by their latency. */
[[gnu::always_inline]] static inline void af_pv_wide(size_t       NH,
                                                     size_t       n,
                                                     size_t       row,
                                                     size_t       c,
                                                     const float *v,
                                                     const float (*w)[AF_BLOCK],
                                                     float (*acc)[AF_HEAD_DIM_MAX]) {
    const size_t NV = 8 / NH;
    float       *a0 = acc[0] + c;
    float       *a1 = NH > 1 ? acc[1] + c : a0 + 32;
    __m256       x0 = _mm256_load_ps(a0), x1 = _mm256_load_ps(a0 + 8);
    __m256       x2 = _mm256_load_ps(a0 + 16), x3 = _mm256_load_ps(a0 + 24);
    __m256       x4 = _mm256_load_ps(a1), x5 = _mm256_load_ps(a1 + 8);
    __m256       x6 = _mm256_load_ps(a1 + 16), x7 = _mm256_load_ps(a1 + 24);
    const float *vr = v + c;
    for (size_t j = 0; j < n; j++, vr += row) {
        const __m256 s0 = _mm256_broadcast_ss(&w[0][j]);
        const __m256 s1 = NH > 1 ? _mm256_broadcast_ss(&w[1][j]) : s0;
        const __m256 v0 = _mm256_loadu_ps(vr), v1 = _mm256_loadu_ps(vr + 8);
        const __m256 v2 = _mm256_loadu_ps(vr + 16), v3 = _mm256_loadu_ps(vr + 24);
        x0 = _mm256_fmadd_ps(s0, v0, x0);
        x1 = _mm256_fmadd_ps(s0, v1, x1);
        x2 = _mm256_fmadd_ps(s0, v2, x2);
        x3 = _mm256_fmadd_ps(s0, v3, x3);
        if (NV == 8) {
            /* One head: dimensions 32..63 of it. */
            x4 = _mm256_fmadd_ps(s0, _mm256_loadu_ps(vr + 32), x4);
            x5 = _mm256_fmadd_ps(s0, _mm256_loadu_ps(vr + 40), x5);
            x6 = _mm256_fmadd_ps(s0, _mm256_loadu_ps(vr + 48), x6);
            x7 = _mm256_fmadd_ps(s0, _mm256_loadu_ps(vr + 56), x7);
        } else {
            /* Two heads: the same 32 dimensions of the second. */
            x4 = _mm256_fmadd_ps(s1, v0, x4);
            x5 = _mm256_fmadd_ps(s1, v1, x5);
            x6 = _mm256_fmadd_ps(s1, v2, x6);
            x7 = _mm256_fmadd_ps(s1, v3, x7);
        }
    }
    _mm256_store_ps(a0, x0);
    _mm256_store_ps(a0 + 8, x1);
    _mm256_store_ps(a0 + 16, x2);
    _mm256_store_ps(a0 + 24, x3);
    _mm256_store_ps(a1, x4);
    _mm256_store_ps(a1 + 8, x5);
    _mm256_store_ps(a1 + 16, x6);
    _mm256_store_ps(a1 + 24, x7);
}

/* ---- One work item ------------------------------------------------------- */

/* Query heads [h0, h0 + G) at the tn positions t0 .. t0 + tn - 1, all
 * reading KV head kv_h, position t0 + i over context positions [lo[i],
 * hi[i]] (none if lo[i] > hi[i]). The queries share only the order of the
 * work — each block of the context runs for all of them before the next —
 * and each keeps its own arithmetic, the same as alone in an item. Without
 * `part` it writes their output (zeros for a query that attends to
 * nothing); with it (one query), its partial results: per head HD + 2
 * floats, the unnormalized V sums, the running max and the sum of
 * exponentials. G and HD are compile-time constants at every call site
 * (HD = 0: head_dim at run time). */
[[gnu::always_inline]] static inline void af_item(size_t                G,
                                                  size_t                HD,
                                                  const struct af_args *a,
                                                  size_t                t0,
                                                  size_t                tn,
                                                  size_t                kv_h,
                                                  size_t                h0,
                                                  const size_t          lo[static tn],
                                                  const size_t          hi[static tn],
                                                  float                *part) {
    /* Locals, not a->field in the loops: after OpenMP outlining `a` points
     * into the caller's frame, and every float store could alias it. */
    const size_t head_dim  = HD != 0 ? HD : a->head_dim;
    const size_t n_q_heads = a->n_q_heads, n_kv_heads = a->n_kv_heads;
    const size_t row     = n_kv_heads * head_dim; /* K/V floats per position */
    const size_t hd8     = head_dim & ~(size_t) 7;
    const size_t hd16    = head_dim & ~(size_t) 15;
    const size_t pv_fit  = AF_PV_BYTES / (head_dim * sizeof(float));
    const size_t pv_rows = pv_fit > 8 ? pv_fit : 8;
    const float *kh      = a->k + kv_h * head_dim;
    const float *vh      = a->v + kv_h * head_dim;

    /* sc: the block's scores, then exp(score - max) — the weight of each V
     * row. */
    alignas(32) float sc[G][AF_BLOCK];
    alignas(32) float acc_all[AF_QUERIES_MAX][G][AF_HEAD_DIM_MAX];
    float             max_all[AF_QUERIES_MAX][G];
    double            sum_all[AF_QUERIES_MAX][G];
    for (size_t i_q = 0; i_q < tn * G; i_q++) {
        for (size_t i = 0; i < head_dim; i++) {
            acc_all[i_q / G][i_q % G][i] = 0.0f;
        }
        max_all[i_q / G][i_q % G] = 0.0f;
        sum_all[i_q / G][i_q % G] = 0.0;
    }
    for (size_t blk = 0;; blk++) {
        bool ran = false;
        for (size_t tq = 0; tq < tn; tq++) {
            /* Block blk of query tq's span, if it has one. */
            if (lo[tq] > hi[tq] || blk * AF_BLOCK > hi[tq] - lo[tq]) {
                continue;
            }
            ran = true;

            const size_t b0               = lo[tq] + blk * AF_BLOCK;
            const size_t n                = hi[tq] - b0 < AF_BLOCK ? hi[tq] - b0 + 1 : AF_BLOCK;
            const float *qt               = a->q + ((t0 + tq) * n_q_heads + h0) * head_dim;
            float (*acc)[AF_HEAD_DIM_MAX] = acc_all[tq];
            float       *max_score        = max_all[tq];
            double      *sum_exp          = sum_all[tq];
            const float *kb               = kh + b0 * row;
            const float *vb               = vh + b0 * row;
            size_t       j                = 0;
            for (; j + 8 <= n; j += 8) {
                const float *k0 = kb + j * row;
                for (size_t g = 0; g < G; g++) {
                    _mm256_store_ps(sc[g] + j, af_dot8(hd8, head_dim, row, qt + g * head_dim, k0));
                }
            }
            for (; j < n; j++) {
                for (size_t g = 0; g < G; g++) {
                    sc[g][j] = af_dot1(hd8, head_dim, qt + g * head_dim, kb + j * row);
                }
            }
            for (size_t g = 0; g < G; g++) {
                float block_max = sc[g][0];
                for (size_t i = 1; i < n; i++) {
                    if (sc[g][i] > block_max) {
                        block_max = sc[g][i];
                    }
                }
                if (blk == 0) {
                    max_score[g] = block_max;
                } else if (block_max > max_score[g]) {
                    /* The sums so far were taken against the lower max: scale
                     * them down to the new one (to 0 below ATTN_EXP_FLOOR). */
                    const float d = max_score[g] - block_max;
                    const float c = d < ATTN_EXP_FLOOR ? 0.0f : expf(d);
                    sum_exp[g] *= c;
                    for (size_t i = 0; i < head_dim; i++) {
                        acc[g][i] *= c;
                    }
                    max_score[g] = block_max;
                }
                double block_sum = 0.0;
                for (size_t i = 0; i < n; i++) {
                    const float e = expf(fmaxf(sc[g][i] - max_score[g], ATTN_EXP_FLOOR));
                    block_sum += e;
                    sc[g][i] = e;
                }
                sum_exp[g] += block_sum;
            }
            /* pv_rows rows at a time, so that their V bytes stay in L1 while
             * every slice of the output dimensions passes over them. The sums
             * keep their order. */
            for (size_t r0 = 0; r0 < n; r0 += pv_rows) {
                const size_t nr            = n - r0 < pv_rows ? n - r0 : pv_rows;
                const float *vr            = vb + r0 * row;
                const float (*w)[AF_BLOCK] = (const float (*)[AF_BLOCK]) & sc[0][r0];
                size_t c                   = 0;
                if (G <= 2) {
                    const size_t width = G == 1 ? 64 : 32;
                    for (; c + width <= head_dim; c += width) {
                        af_pv_wide(G, nr, row, c, vr, w, acc);
                    }
                }
                for (; c < hd16; c += 16) {
                    af_pv16(G, nr, row, c, vr, w, acc);
                }
                for (; c < head_dim; c++) {
                    for (size_t i = 0; i < nr; i++) {
                        const float vf = vr[i * row + c];
                        for (size_t g = 0; g < G; g++) {
                            acc[g][c] += w[g][i] * vf;
                        }
                    }
                }
            }
        }
        if (!ran) {
            break;
        }
    }
    for (size_t i_q = 0; i_q < tn * G; i_q++) {
        const size_t tq = i_q / G, g = i_q % G;
        if (part != nullptr) {
            float *rec = part + g * (head_dim + 2);
            memcpy(rec, acc_all[tq][g], head_dim * sizeof(float));
            rec[head_dim]     = max_all[tq][g];
            rec[head_dim + 1] = (float) sum_all[tq][g];
        } else {
            float *outv = a->out + ((t0 + tq) * n_q_heads + h0 + g) * head_dim;
            if (sum_all[tq][g] > 0.0) {
                const float inv_sum = (float) (1.0 / sum_all[tq][g]);
                for (size_t i = 0; i < head_dim; i++) {
                    outv[i] = acc_all[tq][g][i] * inv_sum;
                }
            } else {
                for (size_t i = 0; i < head_dim; i++) {
                    outv[i] = 0.0f;
                }
            }
        }
    }
}

/* One function per compiled shape (G, HD): in one function holding all of
 * them, every change to one moved the hot loops of the others. */
#define AF_ITEM_FN(G, HD)                                                                 \
    [[gnu::noinline]] static void af_item_##G##_##HD(const struct af_args *a,             \
                                                     size_t                t0,            \
                                                     size_t                tn,            \
                                                     size_t                kv_h,          \
                                                     size_t                h0,            \
                                                     const size_t          lo[static tn], \
                                                     const size_t          hi[static tn], \
                                                     float                *part) {        \
        af_item(G, HD, a, t0, tn, kv_h, h0, lo, hi, part);                                \
    }
#define AF_ITEM_FNS(G) AF_ITEM_FN(G, 64) AF_ITEM_FN(G, 128) AF_ITEM_FN(G, 256) AF_ITEM_FN(G, 0)
AF_ITEM_FNS(1)
AF_ITEM_FNS(2)
AF_ITEM_FNS(3)
AF_ITEM_FNS(4)

#define AF_ITEM_HD(G)                                             \
    do {                                                          \
        switch (a->head_dim) {                                    \
        case 64:                                                  \
            af_item_##G##_64(a, t0, tn, kv_h, h0, lo, hi, part);  \
            break;                                                \
        case 128:                                                 \
            af_item_##G##_128(a, t0, tn, kv_h, h0, lo, hi, part); \
            break;                                                \
        case 256:                                                 \
            af_item_##G##_256(a, t0, tn, kv_h, h0, lo, hi, part); \
            break;                                                \
        default:                                                  \
            af_item_##G##_0(a, t0, tn, kv_h, h0, lo, hi, part);   \
            break;                                                \
        }                                                         \
    } while (0)

static void af_run_item(size_t                per_pass,
                        const struct af_args *a,
                        size_t                t0,
                        size_t                tn,
                        size_t                kv_h,
                        size_t                h0,
                        const size_t          lo[static tn],
                        const size_t          hi[static tn],
                        float                *part) {
    switch (per_pass) {
    case 4:
        AF_ITEM_HD(4);
        break;
    case 3:
        AF_ITEM_HD(3);
        break;
    case 2:
        AF_ITEM_HD(2);
        break;
    default:
        AF_ITEM_HD(1);
        break;
    }
}

/* One head's output from its n_chunks partial results (records `stride`
 * floats apart), rescaled to their common max and summed in chunk order. */
static void
af_merge(size_t n_chunks, size_t head_dim, size_t stride, const float *part, float *out) {
    float max_score = part[head_dim];
    for (size_t c = 1; c < n_chunks; c++) {
        const float m = part[c * stride + head_dim];
        max_score     = m > max_score ? m : max_score;
    }
    float  acc[AF_HEAD_DIM_MAX];
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

/* The kernel on host pointers, validated by the caller: q and out [n_q,
 * n_q_heads, head_dim], k and v [n_kv, n_kv_heads, head_dim], n_q_heads a
 * multiple of n_kv_heads, head_dim at most AF_HEAD_DIM_MAX; `part` holds
 * part_floats floats for a split decode, or is nullptr (no split). */
static void af_run(size_t       n_q,
                   size_t       n_q_heads,
                   size_t       head_dim,
                   size_t       n_kv,
                   size_t       n_kv_heads,
                   size_t       part_floats,
                   size_t       q_offset,
                   size_t       sliding_window,
                   const float *q,
                   const float *k,
                   const float *v,
                   float       *out,
                   float       *part) {
    const struct af_args a      = {.n_q_heads      = n_q_heads,
                                   .head_dim       = head_dim,
                                   .n_kv           = n_kv,
                                   .n_kv_heads     = n_kv_heads,
                                   .q_offset       = q_offset,
                                   .sliding_window = sliding_window,
                                   .q              = q,
                                   .k              = k,
                                   .v              = v,
                                   .out            = out};
    const size_t         group  = n_q_heads / n_kv_heads;
    size_t               dec_lo = 0, dec_hi = 0, last_lo = 0, last_hi = 0;
    af_span(&a, 0, &dec_lo, &dec_hi);
    af_span(&a, n_q - 1, &last_lo, &last_hi);
    const size_t         span = last_lo <= last_hi ? last_hi - last_lo + 1 : 0;
    const struct af_plan plan = af_plan_for(
            n_q, n_q_heads, n_kv_heads, head_dim, span, part != nullptr ? part_floats : 0);
    const size_t per_pass = plan.per_pass;
    const size_t n_passes = group / per_pass;
    if (plan.n_chunks > 1) {
        /* Split decode (n_q == 1): items (KV head, pass, chunk) leave their
         * partial results in `part`, then each head merges its chunks.
         * Chunk c covers positions [dec_lo + c * len, ...], at least
         * AF_CHUNK_MIN each, so none is empty. */
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
                        af_run_item(per_pass,
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
                    af_merge(n_chunks,
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
     * L2. Causal and window masks make later positions longer: dynamic. */
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
                size_t       lo[AF_QUERIES_MAX], hi[AF_QUERIES_MAX];
                for (size_t tq = 0; tq < tn; tq++) {
                    af_span(&a, t0 + tq, &lo[tq], &hi[tq]);
                }
                af_run_item(per_pass,
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

[[nodiscard]] enum geist_status cpu_x86_attention(struct geist_backend      *be,
                                                  const struct geist_tensor *q,
                                                  const struct geist_tensor *k,
                                                  const struct geist_tensor *v,
                                                  size_t                     q_offset,
                                                  size_t                     sliding_window,
                                                  struct geist_tensor       *out) {
    if (be == nullptr || q == nullptr || k == nullptr || v == nullptr || out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t       nq, nk, nv, no;
    const float *qp = get_f32_dense_ptr_full(q, &nq);
    const float *kp = get_f32_dense_ptr_full(k, &nk);
    const float *vp = get_f32_dense_ptr_full(v, &nv);
    float       *op = get_f32_dense_ptr_full(out, &no);
    if (qp == nullptr || kp == nullptr || vp == nullptr || op == nullptr) {
        geist_backend_set_error(
                be, GEIST_E_UNSUPPORTED, "cpu_x86 attention: tensors must be F32 DENSE");
        return GEIST_E_UNSUPPORTED;
    }
    if (q->ndim != 3 || k->ndim != 3 || v->ndim != 3 || out->ndim != 3) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_x86 attention: bad ranks");
        return GEIST_E_INVALID_ARG;
    }
    size_t n_q        = (size_t) q->shape[0];
    size_t n_q_heads  = (size_t) q->shape[1];
    size_t head_dim   = (size_t) q->shape[2];
    size_t n_kv       = (size_t) k->shape[0];
    size_t n_kv_heads = (size_t) k->shape[1];
    if (k->shape[2] != (int64_t) head_dim || v->shape[2] != (int64_t) head_dim ||
        v->shape[0] != (int64_t) n_kv || v->shape[1] != (int64_t) n_kv_heads ||
        out->shape[0] != (int64_t) n_q || out->shape[1] != (int64_t) n_q_heads ||
        out->shape[2] != (int64_t) head_dim) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_x86 attention: shape mismatch");
        return GEIST_E_INVALID_ARG;
    }
    /* Every dimension is at least 1: geist_tensor_f32_dense refuses a view
     * with a non-positive one. */
    if (n_q_heads % n_kv_heads != 0 || head_dim > AF_HEAD_DIM_MAX) {
        attention_mqa_causal_kv(n_q,
                                n_kv,
                                q_offset,
                                n_q_heads,
                                n_kv_heads,
                                head_dim,
                                sliding_window,
                                qp,
                                kp,
                                vp,
                                op);
        return GEIST_OK;
    }
    /* A split decode's partial results, in the calling thread's workspace;
     * without it the decode runs unsplit (same result to rounding). */
    const size_t part_floats = af_part_floats(n_q_heads, head_dim);
    float *part = n_q == 1 && be->state != nullptr
                          ? cpu_x86_ws_attn_part((struct cpu_x86_state *) be->state, part_floats)
                          : nullptr;
    af_run(n_q,
           n_q_heads,
           head_dim,
           n_kv,
           n_kv_heads,
           part_floats,
           q_offset,
           sliding_window,
           qp,
           kp,
           vp,
           op,
           part);
    return GEIST_OK;
}

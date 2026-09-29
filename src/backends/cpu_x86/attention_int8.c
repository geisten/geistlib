/*
 * src/backends/cpu_x86/attention_int8.c — AVX2 attention over the INT8 KV
 * cache, and cpu_x86's fused->attention_kv_int8, which runs it or, where the
 * ISA dispatch allows, the AVX-512 VNNI kernel of
 * attention_int8_avx512_vnni.c.
 *
 * Layer: BACKEND (cpu_x86).
 *
 * The INT8 cache is this backend's default (caps.preferred_kv_mode), and
 * its attention was the architecture layer's portable loop
 * (attention_int8_via_buffers in forward/attention.c), vectorized by gcc
 * on its own: about 570 instructions per context position for four query
 * heads at head_dim 64, where the work is 256 int8 multiply-adds for the
 * scores and 256 fp32 ones for V. This is the same computation — the
 * query quantization, the blocks of the context, the online softmax, the
 * split decode and its merge — written for AVX2, this backend's floor
 * (x86-64-v3):
 *   - scores: |k| (u8) times q * sign(k) (s8) through vpmaddubsw and
 *     vpmaddwd, exact in int32 for every int8 k, as the portable dot is;
 *     eight positions are reduced together;
 *   - V: eight fp32 accumulators per pass stay in registers while 64
 *     rows of the block go past — 16 output dimensions of each of 4 or 3
 *     heads, 32 of each of 2, 64 of one. The portable loop kept them in
 *     memory and stored every update: the int8 V row may alias them;
 *   - head_dim 64, 128 and 256 are compile-time constants, so the dots
 *     unroll; any other head_dim up to AI8_HEAD_DIM_MAX runs the same code
 *     with the length at run time.
 * The work is split as in the portable loop — passes of up to four query
 * heads of one KV head, decode split across the context and merged — by a
 * plan measured for this kernel (ai8_plan_for); prefill takes its items KV
 * head by KV head.
 * The integer dots and the order of every fp32 V sum are the portable
 * loop's. -ffast-math still lets the compiler group the softmax's double
 * sums differently in each, so the two agree to rounding, not bit for bit.
 * Any thread count gives the same bits: the work items are independent,
 * the merge order is fixed and the plan depends on the shape alone.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "attention.h"
#include "backend_state.h"
#include "kernel_w4a8.h" /* w4a8_dispatcher_tier */

#include "gemma4_kernels.h" /* ATTN_EXP_FLOOR */
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
 * its buffer vtable); attention.c mirrors it the same way. */
struct geist_buffer {
    void                  *host;
    size_t                 bytes;
    enum geist_buffer_role role;
    unsigned int           memory_flags;
};

/* The architecture refuses a larger head_dim at load
 * (TRANSFORMER_HEAD_DIM_MAX); the stack arrays below are sized by it. */
constexpr size_t AI8_HEAD_DIM_MAX = 512;

/* Context positions per online-softmax block, as in the portable loop.
 * 128 to 1024 measured within noise of each other here too. */
constexpr size_t AI8_BLOCK = 512;

/* V rows per pass over the output dimensions. */
constexpr size_t AI8_PV_ROWS = 64;

/* Query heads per pass, at most: the V accumulators of four heads fill
 * the eight registers the pass keeps (16 dimensions each). */
constexpr size_t AI8_HEADS_PER_PASS_MAX = 4;

/* Decode chunks per pass, at most: the part buffer holds that many
 * records per head. */
constexpr size_t AI8_MAX_CHUNKS = 4;

/* A decode is split into one chunk per this many positions, so that each
 * chunk's K and V stay in L2 ... */
constexpr size_t AI8_CHUNK_SPAN = 1024;

/* ... and into as many as give this many work items, if its passes give
 * fewer (MQA decode has one per pass) ... */
constexpr size_t AI8_MIN_ITEMS = 8;

/* ... but no chunk shorter than this: its setup and the merge would cost
 * more than it saves. */
constexpr size_t AI8_CHUNK_MIN = 128;

/* Prefill: queries per work item, at most. An item runs each block of the
 * context for all its queries before the next block, so that the block's K
 * and V bytes, fetched for the first query, are near for the others ... */
constexpr size_t AI8_QUERIES_MAX = 4;

/* ... where the K and V bytes a call reads exceed this: below it they stay
 * cached between queries anyway. */
constexpr size_t AI8_REUSE_BYTES = (size_t) 1 << 20;

size_t cpu_x86_attention_kv_int8_part_floats(size_t n_q_heads, size_t head_dim) {
    return n_q_heads * AI8_MAX_CHUNKS * (head_dim + 2);
}

struct ai8_args {
    size_t        n_q_heads, head_dim, n_kv, n_kv_heads, q_offset, sliding_window;
    const float  *q;
    const int8_t *k;
    const float  *k_scale;
    const int8_t *v;
    const float  *v_scale;
    float        *out;
};

/* The context positions [*s_lo, *s_hi] query t attends to. */
static inline void ai8_span(const struct ai8_args *a, size_t t, size_t *s_lo, size_t *s_hi) {
    const size_t q_pos = a->q_offset + t;
    *s_lo = (a->sliding_window > 0 && q_pos + 1 > a->sliding_window) ? q_pos + 1 - a->sliding_window
                                                                     : 0;
    *s_hi = q_pos < a->n_kv ? q_pos : a->n_kv - 1;
}

struct ai8_plan {
    size_t per_pass; /* query heads per pass */
    size_t n_chunks; /* decode: context chunks per pass, merged afterwards */
    size_t per_item; /* prefill: queries per work item */
};

/* How to run a call, from its shape alone (any thread count gives the same
 * bits); `span` is the number of positions the last query attends to.
 * Measured with every choice forced, 4 threads, decode and 64-row prefill
 * over 32/8 heads at head_dim 64, 16/8 and 24/8 at 128, 15/5 at 64, 8/1 at
 * 256 and 32/32 at 64, contexts 256-8192:
 *   - the widest pass that divides the KV group was never slower, prefill
 *     or decode (up to 2x faster than one head per item);
 *   - a decode split into a chunk per 1024 positions ran 7-45 % faster at
 *     8192 positions than unsplit (32/8 hd 64: 0.50 -> 0.44 ms; 8/1 hd
 *     256: 0.60 -> 0.30), within noise of it at 2048, and slower at 512,
 *     where the split is left to the item count.
 * A prefill puts its queries in items of up to AI8_QUERIES_MAX where a
 * span runs past one block and the call reads more than AI8_REUSE_BYTES
 * of K and V, in items of as many as leave at least AI8_MIN_ITEMS items;
 * the queries of an item do not change each other's arithmetic, so any
 * plan gives the same bits. Against one query an item, 64-token chunks: up
 * to 36 % faster at 2048 and 8192 positions where the call reads more than
 * 1 MB (at 1.3 MB, SmolLM2-360M at 2048 positions, the same); with the
 * queries forced into items anyway, a few percent slower within one block
 * and at 1 MB (MQA at 2048 positions), and with fewer items the threads
 * waited on the longest (MQA with four queries a call, two items, 20-80 %
 * slower in the VNNI kernel). */
static struct ai8_plan ai8_plan_for(size_t n_q,
                                    size_t n_q_heads,
                                    size_t n_kv_heads,
                                    size_t head_dim,
                                    size_t span,
                                    size_t part_floats) {
    const size_t group    = n_q_heads / n_kv_heads;
    size_t       per_pass = 1;
    for (size_t g = AI8_HEADS_PER_PASS_MAX; g >= 2; g--) {
        if (group % g == 0) {
            per_pass = g;
            break;
        }
    }
    struct ai8_plan plan  = {.per_pass = per_pass, .n_chunks = 1, .per_item = 1};
    const size_t    items = n_kv_heads * (group / per_pass); /* per query */
    if (n_q == 1) {
        const size_t for_items = (AI8_MIN_ITEMS + items - 1) / items;
        const size_t fit       = span / AI8_CHUNK_MIN;
        size_t       chunks    = span / AI8_CHUNK_SPAN;
        chunks                 = chunks > for_items ? chunks : for_items;
        chunks                 = chunks < AI8_MAX_CHUNKS ? chunks : AI8_MAX_CHUNKS;
        chunks                 = chunks < fit ? chunks : fit;
        if (chunks >= 2 && part_floats >= n_q_heads * chunks * (head_dim + 2)) {
            plan.n_chunks = chunks;
        }
    } else if (span > AI8_BLOCK && span * n_kv_heads * head_dim * 2 > AI8_REUSE_BYTES) {
        for (size_t per_item = AI8_QUERIES_MAX; per_item >= 2; per_item /= 2) {
            if (items * ((n_q + per_item - 1) / per_item) >= AI8_MIN_ITEMS) {
                plan.per_item = per_item;
                break;
            }
        }
    }
    return plan;
}

/* ---- Scores ------------------------------------------------------------ */

/* The eight int32 lanes whose sum is q . k over the first hd32 elements
 * (whole 32-byte chunks). vpmaddubsw multiplies |k| (u8, up to 128) by
 * q * sign(k) (s8; q is the kernel's own quantization, within +-127) and
 * adds pairs, at most 2 * 128 * 127 in magnitude: no int16 saturation. */
static inline __m256i ai8_dot_lanes(size_t hd32, const int8_t *q, const int8_t *k) {
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i       s    = _mm256_setzero_si256();
    for (size_t c = 0; c < hd32; c += 32) {
        const __m256i kk = _mm256_loadu_si256((const __m256i *) (k + c));
        const __m256i qq = _mm256_load_si256((const __m256i *) (q + c));
        const __m256i p  = _mm256_maddubs_epi16(_mm256_abs_epi8(kk), _mm256_sign_epi8(qq, kk));
        s                = _mm256_add_epi32(s, _mm256_madd_epi16(p, ones));
    }
    return s;
}

/* q . k over elements [hd32, head_dim). */
static inline int32_t ai8_dot_tail(size_t hd32, size_t head_dim, const int8_t *q, const int8_t *k) {
    int32_t d = 0;
    for (size_t i = hd32; i < head_dim; i++) {
        d += (int32_t) q[i] * (int32_t) k[i];
    }
    return d;
}

/* Lane p of the result is the sum of the lanes of d_p. */
static inline __m256i ai8_reduce8(__m256i d0,
                                  __m256i d1,
                                  __m256i d2,
                                  __m256i d3,
                                  __m256i d4,
                                  __m256i d5,
                                  __m256i d6,
                                  __m256i d7) {
    const __m256i u0 = _mm256_hadd_epi32(_mm256_hadd_epi32(d0, d1), _mm256_hadd_epi32(d2, d3));
    const __m256i u1 = _mm256_hadd_epi32(_mm256_hadd_epi32(d4, d5), _mm256_hadd_epi32(d6, d7));
    return _mm256_add_epi32(_mm256_permute2x128_si256(u0, u1, 0x20),
                            _mm256_permute2x128_si256(u0, u1, 0x31));
}

static inline int32_t ai8_hsum(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s         = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s         = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}

/* ---- V ------------------------------------------------------------------ */

static inline __m256 ai8_cvt8(const int8_t *v) {
    return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *) v)));
}

/* acc[g][c .. c+16) += sum over the n rows j of w[g][j] * v_j[c .. c+16),
 * j in order, for the G <= 4 heads of the pass. Named accumulators, not an
 * array: stores to a __m256 array may alias the float weights (__m256 is
 * may_alias), and gcc kept every update in memory. */
[[gnu::always_inline]] static inline void ai8_pv16(size_t        G,
                                                   size_t        n,
                                                   size_t        row,
                                                   size_t        c,
                                                   const int8_t *v,
                                                   const float (*w)[AI8_BLOCK],
                                                   float (*acc)[AI8_HEAD_DIM_MAX]) {
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
    const int8_t *vr = v + c;
    for (size_t j = 0; j < n; j++, vr += row) {
        const __m256 v0 = ai8_cvt8(vr), v1 = ai8_cvt8(vr + 8);
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

/* As ai8_pv16 over 32 dimensions of 2 heads, or 64 of one (NH heads, NV
 * vectors of 8 each, NH * NV = 8): 16 dimensions per head would leave the
 * FMAs one chain per register and bound by their latency. */
[[gnu::always_inline]] static inline void ai8_pv_wide(size_t        NH,
                                                      size_t        n,
                                                      size_t        row,
                                                      size_t        c,
                                                      const int8_t *v,
                                                      const float (*w)[AI8_BLOCK],
                                                      float (*acc)[AI8_HEAD_DIM_MAX]) {
    const size_t  NV = 8 / NH;
    float        *a0 = acc[0] + c;
    float        *a1 = NH > 1 ? acc[1] + c : a0 + 32;
    __m256        x0 = _mm256_load_ps(a0), x1 = _mm256_load_ps(a0 + 8);
    __m256        x2 = _mm256_load_ps(a0 + 16), x3 = _mm256_load_ps(a0 + 24);
    __m256        x4 = _mm256_load_ps(a1), x5 = _mm256_load_ps(a1 + 8);
    __m256        x6 = _mm256_load_ps(a1 + 16), x7 = _mm256_load_ps(a1 + 24);
    const int8_t *vr = v + c;
    for (size_t j = 0; j < n; j++, vr += row) {
        const __m256 s0 = _mm256_broadcast_ss(&w[0][j]);
        const __m256 s1 = NH > 1 ? _mm256_broadcast_ss(&w[1][j]) : s0;
        const __m256 v0 = ai8_cvt8(vr), v1 = ai8_cvt8(vr + 8);
        const __m256 v2 = ai8_cvt8(vr + 16), v3 = ai8_cvt8(vr + 24);
        x0 = _mm256_fmadd_ps(s0, v0, x0);
        x1 = _mm256_fmadd_ps(s0, v1, x1);
        x2 = _mm256_fmadd_ps(s0, v2, x2);
        x3 = _mm256_fmadd_ps(s0, v3, x3);
        if (NV == 8) {
            /* One head: dimensions 32..63 of it. */
            x4 = _mm256_fmadd_ps(s0, ai8_cvt8(vr + 32), x4);
            x5 = _mm256_fmadd_ps(s0, ai8_cvt8(vr + 40), x5);
            x6 = _mm256_fmadd_ps(s0, ai8_cvt8(vr + 48), x6);
            x7 = _mm256_fmadd_ps(s0, ai8_cvt8(vr + 56), x7);
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

/* ---- One work item -------------------------------------------------------- */

/* Query heads [h0, h0 + G) at the tn positions t0 .. t0 + tn - 1, all
 * reading KV head kv_h, position t0 + i over context positions [lo[i],
 * hi[i]]: the body of the portable loop's grouped pass (attn_int8_item)
 * for each. The queries share only the order of the work — each block of
 * the context runs for all of them before the next — and each keeps its
 * own arithmetic, the same as alone in an item. Without `part` it writes
 * their output; with it (one query), its partial results: per head HD + 2
 * floats, the unnormalized V sums, the running max and the sum of
 * exponentials. G and HD are compile-time constants at every call site
 * (HD = 0: head_dim at run time). */
[[gnu::always_inline]] static inline void ai8_item(size_t                 G,
                                                   size_t                 HD,
                                                   const struct ai8_args *a,
                                                   size_t                 t0,
                                                   size_t                 tn,
                                                   size_t                 kv_h,
                                                   size_t                 h0,
                                                   const size_t           lo[static tn],
                                                   const size_t           hi[static tn],
                                                   float                 *part) {
    /* Locals, not a->field in the loops: after OpenMP outlining `a` points
     * into the caller's frame, and every float store could alias it. */
    const size_t  head_dim  = HD != 0 ? HD : a->head_dim;
    const size_t  n_q_heads = a->n_q_heads, n_kv_heads = a->n_kv_heads;
    const size_t  row     = n_kv_heads * head_dim; /* K/V bytes per position */
    const size_t  hd32    = head_dim & ~(size_t) 31;
    const size_t  hd16    = head_dim & ~(size_t) 15;
    const float  *k_scale = a->k_scale;
    const float  *v_scale = a->v_scale;
    const int8_t *kh      = a->k + kv_h * head_dim;
    const int8_t *vh      = a->v + kv_h * head_dim;

    /* The queries, quantized per head exactly as the portable loop does. */
    alignas(32) int8_t q_all[AI8_QUERIES_MAX][G][AI8_HEAD_DIM_MAX];
    float              scale_all[AI8_QUERIES_MAX][G];
    for (size_t i_q = 0; i_q < tn * G; i_q++) {
        const size_t tq = i_q / G, g = i_q % G;
        const float *qv   = a->q + ((t0 + tq) * n_q_heads + h0 + g) * head_dim;
        int8_t      *qq   = q_all[tq][g];
        float        amax = 0.0f;
        for (size_t i = 0; i < head_dim; i++) {
            const float x = fabsf(qv[i]);
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
            qq[i] = (int8_t) lrintf(qv[i] * inv_q);
        }
        scale_all[tq][g] = sq;
    }

    /* sc: the block's scores, then exp(score - max), then those times the
     * row's V scale — the weight of each V row. */
    alignas(32) float sc[G][AI8_BLOCK];
    alignas(32) float ks[AI8_BLOCK];
    alignas(32) float vs[AI8_BLOCK];
    alignas(32) float acc_all[AI8_QUERIES_MAX][G][AI8_HEAD_DIM_MAX];
    float             max_all[AI8_QUERIES_MAX][G];
    double            sum_all[AI8_QUERIES_MAX][G];
    for (size_t i_q = 0; i_q < tn * G; i_q++) {
        for (size_t i = 0; i < head_dim; i++) {
            acc_all[i_q / G][i_q % G][i] = 0.0f;
        }
        sum_all[i_q / G][i_q % G] = 0.0;
    }
    for (size_t blk = 0;; blk++) {
        bool ran = false;
        for (size_t tq = 0; tq < tn; tq++) {
            /* Block blk of query tq's span, if it has one. */
            if (blk * AI8_BLOCK > hi[tq] - lo[tq]) {
                continue;
            }
            ran = true;

            const size_t b0                        = lo[tq] + blk * AI8_BLOCK;
            const size_t c_hi                      = hi[tq];
            const int8_t (*q_q8)[AI8_HEAD_DIM_MAX] = q_all[tq];
            const float *scale_q                   = scale_all[tq];
            float (*acc)[AI8_HEAD_DIM_MAX]         = acc_all[tq];
            float        *max_score                = max_all[tq];
            double       *sum_exp                  = sum_all[tq];
            const size_t  n  = c_hi - b0 < AI8_BLOCK ? c_hi - b0 + 1 : AI8_BLOCK;
            const int8_t *kb = kh + b0 * row;
            const int8_t *vb = vh + b0 * row;
            for (size_t j = 0; j < n; j++) {
                ks[j] = k_scale[(b0 + j) * n_kv_heads + kv_h];
                vs[j] = v_scale[(b0 + j) * n_kv_heads + kv_h];
            }
            size_t j = 0;
            for (; j + 8 <= n; j += 8) {
                const int8_t *k0  = kb + j * row;
                const __m256  ksv = _mm256_load_ps(ks + j);
                for (size_t g = 0; g < G; g++) {
                    __m256i d = ai8_reduce8(ai8_dot_lanes(hd32, q_q8[g], k0),
                                            ai8_dot_lanes(hd32, q_q8[g], k0 + row),
                                            ai8_dot_lanes(hd32, q_q8[g], k0 + 2 * row),
                                            ai8_dot_lanes(hd32, q_q8[g], k0 + 3 * row),
                                            ai8_dot_lanes(hd32, q_q8[g], k0 + 4 * row),
                                            ai8_dot_lanes(hd32, q_q8[g], k0 + 5 * row),
                                            ai8_dot_lanes(hd32, q_q8[g], k0 + 6 * row),
                                            ai8_dot_lanes(hd32, q_q8[g], k0 + 7 * row));
                    if (hd32 < head_dim) {
                        alignas(32) int32_t tail[8];
                        for (size_t p = 0; p < 8; p++) {
                            tail[p] = ai8_dot_tail(hd32, head_dim, q_q8[g], k0 + p * row);
                        }
                        d = _mm256_add_epi32(d, _mm256_load_si256((const __m256i *) tail));
                    }
                    /* (dot * scale_q) * scale_k, as the portable loop rounds it. */
                    const __m256 s = _mm256_mul_ps(
                            _mm256_mul_ps(_mm256_cvtepi32_ps(d), _mm256_set1_ps(scale_q[g])), ksv);
                    _mm256_store_ps(sc[g] + j, s);
                }
            }
            for (; j < n; j++) {
                const int8_t *kr = kb + j * row;
                for (size_t g = 0; g < G; g++) {
                    const int32_t d = ai8_hsum(ai8_dot_lanes(hd32, q_q8[g], kr)) +
                                      ai8_dot_tail(hd32, head_dim, q_q8[g], kr);
                    sc[g][j]        = (float) d * scale_q[g] * ks[j];
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
                    sc[g][i] = e * vs[i];
                }
                sum_exp[g] += block_sum;
            }
            /* AI8_PV_ROWS rows at a time, so that their V bytes stay in L1
             * while every slice of the output dimensions passes over them
             * (128 KB a block at head_dim 256). The sums keep their order. */
            for (size_t r0 = 0; r0 < n; r0 += AI8_PV_ROWS) {
                const size_t  nr            = n - r0 < AI8_PV_ROWS ? n - r0 : AI8_PV_ROWS;
                const int8_t *vr            = vb + r0 * row;
                const float (*w)[AI8_BLOCK] = (const float (*)[AI8_BLOCK]) & sc[0][r0];
                size_t c                    = 0;
                if (G <= 2) {
                    const size_t width = G == 1 ? 64 : 32;
                    for (; c + width <= head_dim; c += width) {
                        ai8_pv_wide(G, nr, row, c, vr, w, acc);
                    }
                }
                for (; c < hd16; c += 16) {
                    ai8_pv16(G, nr, row, c, vr, w, acc);
                }
                for (; c < head_dim; c++) {
                    for (size_t i = 0; i < nr; i++) {
                        const float vf = (float) vr[i * row + c];
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
            const float inv_sum = (float) (1.0 / sum_all[tq][g]);
            float      *outv    = a->out + ((t0 + tq) * n_q_heads + h0 + g) * head_dim;
            for (size_t i = 0; i < head_dim; i++) {
                outv[i] = acc_all[tq][g][i] * inv_sum;
            }
        }
    }
}

/* One function per compiled shape (G, HD): in one function holding all of
 * them, every change to one moved the hot loops of the others. */
#define AI8_ITEM_FN(G, HD)                                                                  \
    [[gnu::noinline]] static void ai8_item_##G##_##HD(const struct ai8_args *a,             \
                                                      size_t                 t0,            \
                                                      size_t                 tn,            \
                                                      size_t                 kv_h,          \
                                                      size_t                 h0,            \
                                                      const size_t           lo[static tn], \
                                                      const size_t           hi[static tn], \
                                                      float                 *part) {        \
        ai8_item(G, HD, a, t0, tn, kv_h, h0, lo, hi, part);                                 \
    }
#define AI8_ITEM_FNS(G) AI8_ITEM_FN(G, 64) AI8_ITEM_FN(G, 128) AI8_ITEM_FN(G, 256) AI8_ITEM_FN(G, 0)
AI8_ITEM_FNS(1)
AI8_ITEM_FNS(2)
AI8_ITEM_FNS(3)
AI8_ITEM_FNS(4)

#define AI8_ITEM_HD(G)                                             \
    do {                                                           \
        switch (a->head_dim) {                                     \
        case 64:                                                   \
            ai8_item_##G##_64(a, t0, tn, kv_h, h0, lo, hi, part);  \
            break;                                                 \
        case 128:                                                  \
            ai8_item_##G##_128(a, t0, tn, kv_h, h0, lo, hi, part); \
            break;                                                 \
        case 256:                                                  \
            ai8_item_##G##_256(a, t0, tn, kv_h, h0, lo, hi, part); \
            break;                                                 \
        default:                                                   \
            ai8_item_##G##_0(a, t0, tn, kv_h, h0, lo, hi, part);   \
            break;                                                 \
        }                                                          \
    } while (0)

static void ai8_run_item(size_t                 per_pass,
                         const struct ai8_args *a,
                         size_t                 t0,
                         size_t                 tn,
                         size_t                 kv_h,
                         size_t                 h0,
                         const size_t           lo[static tn],
                         const size_t           hi[static tn],
                         float                 *part) {
    switch (per_pass) {
    case 4:
        AI8_ITEM_HD(4);
        break;
    case 3:
        AI8_ITEM_HD(3);
        break;
    case 2:
        AI8_ITEM_HD(2);
        break;
    default:
        AI8_ITEM_HD(1);
        break;
    }
}

/* One head's output from its n_chunks partial results (records `stride`
 * floats apart), rescaled to their common max and summed in chunk order:
 * the portable merge (attn_int8_merge). */
static void
ai8_merge(size_t n_chunks, size_t head_dim, size_t stride, const float *part, float *out) {
    float max_score = part[head_dim];
    for (size_t c = 1; c < n_chunks; c++) {
        const float m = part[c * stride + head_dim];
        max_score     = m > max_score ? m : max_score;
    }
    float  acc[AI8_HEAD_DIM_MAX];
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
    const float inv_sum = (float) (1.0 / sum_exp);
    for (size_t i = 0; i < head_dim; i++) {
        out[i] = acc[i] * inv_sum;
    }
}

void cpu_x86_attention_kv_int8_run(size_t        n_q,
                                   size_t        n_q_heads,
                                   size_t        head_dim,
                                   size_t        n_kv,
                                   size_t        n_kv_heads,
                                   size_t        part_floats,
                                   size_t        q_offset,
                                   size_t        sliding_window,
                                   const float  *q,
                                   const int8_t *k,
                                   const float  *k_scale,
                                   const int8_t *v,
                                   const float  *v_scale,
                                   float        *out,
                                   float        *part) {
    const struct ai8_args a      = {.n_q_heads      = n_q_heads,
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
    const size_t          group  = n_q_heads / n_kv_heads;
    size_t                dec_lo = 0, dec_hi = 0, last_lo = 0, last_hi = 0;
    ai8_span(&a, 0, &dec_lo, &dec_hi);
    ai8_span(&a, n_q - 1, &last_lo, &last_hi);
    const struct ai8_plan plan     = ai8_plan_for(n_q,
                                                  n_q_heads,
                                                  n_kv_heads,
                                                  head_dim,
                                                  last_hi - last_lo + 1,
                                                  part != nullptr ? part_floats : 0);
    const size_t          per_pass = plan.per_pass;
    const size_t          n_passes = group / per_pass;
    if (plan.n_chunks > 1) {
        /* Split decode (n_q == 1): items (KV head, pass, chunk) leave their
         * partial results in `part`, then each head merges its chunks.
         * Chunk c covers positions [dec_lo + c * len, ...], at least
         * AI8_CHUNK_MIN each, so none is empty. */
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
                        ai8_run_item(per_pass,
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
                    ai8_merge(n_chunks,
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
     * L2 (1 MB at 8192 positions and head_dim 64), where in query order
     * every thread went through all KV heads' (five of them in
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
                size_t       lo[AI8_QUERIES_MAX], hi[AI8_QUERIES_MAX];
                for (size_t tq = 0; tq < tn; tq++) {
                    ai8_span(&a, t0 + tq, &lo[tq], &hi[tq]);
                }
                ai8_run_item(per_pass,
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

/* ---- The fused op -------------------------------------------------------- */

/* The first element of a contiguous DENSE view of `ndim` dimensions and
 * `dtype` (elements of `elem` bytes), or nullptr if the view is malformed or
 * runs past its buffer; *out_n gets its element count. */
static const void *ai8_view(const struct geist_tensor *t,
                            enum geist_dtype           dtype,
                            size_t                     elem,
                            int                        ndim,
                            size_t                    *out_n) {
    return t != nullptr && t->buffer != nullptr
                   ? geist_tensor_dense(
                             t, dtype, elem, ndim, t->buffer->host, t->buffer->bytes, out_n)
                   : nullptr;
}

/* Whether the AVX-512 VNNI kernel may run: the dispatcher's tier, which
 * GEIST_FORCE_ISA clamps, and cpuid, as linear_q8_0.c decides its tiles. */
static bool ai8_vnni_usable(void) {
    return w4a8_dispatcher_tier() >= W4A8_ISA_AVX512_VNNI && __builtin_cpu_supports("avx512f") &&
           __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512dq") &&
           __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512vnni");
}

bool cpu_x86_attention_kv_int8_supported(const struct geist_fusion_query *q) {
    return q != nullptr && q->head_dim >= 1 && q->head_dim <= AI8_HEAD_DIM_MAX &&
           q->n_kv_heads >= 1 && q->n_q_heads >= q->n_kv_heads && q->n_q_heads % q->n_kv_heads == 0;
}

enum geist_status cpu_x86_attention_kv_int8(struct geist_backend                      *be,
                                            const struct geist_attention_kv_int8_args *args) {
    if (be == nullptr || be->state == nullptr || args == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t        nq = 0, nk = 0, nv = 0, nks = 0, nvs = 0, no = 0;
    const float  *q   = ai8_view(args->q, GEIST_DTYPE_F32, sizeof(float), 3, &nq);
    const int8_t *k   = ai8_view(args->k, GEIST_DTYPE_I8, 1, 3, &nk);
    const int8_t *v   = ai8_view(args->v, GEIST_DTYPE_I8, 1, 3, &nv);
    const float  *ks  = ai8_view(args->k_scale, GEIST_DTYPE_F32, sizeof(float), 2, &nks);
    const float  *vs  = ai8_view(args->v_scale, GEIST_DTYPE_F32, sizeof(float), 2, &nvs);
    float        *out = (float *) ai8_view(args->out, GEIST_DTYPE_F32, sizeof(float), 3, &no);
    if (q == nullptr || k == nullptr || v == nullptr || ks == nullptr || vs == nullptr ||
        out == nullptr) {
        geist_backend_set_error(be,
                                GEIST_E_INVALID_ARG,
                                "cpu_x86 attention_kv_int8: q, scales and out must be F32, k and "
                                "v I8, DENSE views of the rank given within their buffers");
        return GEIST_E_INVALID_ARG;
    }
    const size_t                    n_q        = (size_t) args->q->shape[0];
    const size_t                    n_q_heads  = (size_t) args->q->shape[1];
    const size_t                    head_dim   = (size_t) args->q->shape[2];
    const size_t                    n_kv       = (size_t) args->k->shape[0];
    const size_t                    n_kv_heads = (size_t) args->k->shape[1];
    const struct geist_fusion_query shape      = {.op         = GEIST_FUSED_ATTN_KV_INT8,
                                                  .head_dim   = head_dim,
                                                  .n_q_heads  = n_q_heads,
                                                  .n_kv_heads = n_kv_heads};
    if (!cpu_x86_attention_kv_int8_supported(&shape)) {
        geist_backend_set_error(be,
                                GEIST_E_UNSUPPORTED,
                                "cpu_x86 attention_kv_int8: head_dim %zu, %zu query heads on %zu "
                                "KV heads",
                                head_dim,
                                n_q_heads,
                                n_kv_heads);
        return GEIST_E_UNSUPPORTED;
    }
    /* Matching shapes, and every query's position in the cache: n_q and
     * n_kv are >= 1 (ai8_view refuses an empty dimension), so the
     * subtraction cannot wrap. */
    if ((size_t) args->k->shape[2] != head_dim || nv != nk || (size_t) args->v->shape[0] != n_kv ||
        (size_t) args->v->shape[1] != n_kv_heads || no != nq ||
        (size_t) args->out->shape[0] != n_q || (size_t) args->out->shape[1] != n_q_heads ||
        nks != n_kv * n_kv_heads || nvs != n_kv * n_kv_heads ||
        (size_t) args->k_scale->shape[0] != n_kv || (size_t) args->v_scale->shape[0] != n_kv ||
        n_q > n_kv || args->q_offset > n_kv - n_q) {
        geist_backend_set_error(be,
                                GEIST_E_INVALID_ARG,
                                "cpu_x86 attention_kv_int8: shapes do not match, or %zu queries "
                                "at %zu run past %zu cached positions",
                                n_q,
                                args->q_offset,
                                n_kv);
        return GEIST_E_INVALID_ARG;
    }
    /* A split decode's partial results, in the calling thread's workspace;
     * without it the decode runs unsplit (same result to rounding). */
    const size_t part_floats = cpu_x86_attention_kv_int8_part_floats(n_q_heads, head_dim);
    float *part = n_q == 1 ? cpu_x86_ws_attn_part((struct cpu_x86_state *) be->state, part_floats)
                           : nullptr;
    /* The AVX-512 VNNI kernel where it may run: the same results to
     * rounding (see attention_int8_avx512_vnni.c). */
    typeof(cpu_x86_attention_kv_int8_run) *run = ai8_vnni_usable()
                                                         ? cpu_x86_attention_kv_int8_run_avx512_vnni
                                                         : cpu_x86_attention_kv_int8_run;
    run(n_q,
        n_q_heads,
        head_dim,
        n_kv,
        n_kv_heads,
        part_floats,
        args->q_offset,
        args->sliding_window,
        q,
        k,
        ks,
        v,
        vs,
        out,
        part);
    return GEIST_OK;
}

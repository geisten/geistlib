/*
 * src/backends/cpu_x86/attention_int8_avx512_vnni.c — the INT8-KV
 * attention of attention_int8.c for hosts with AVX-512 VNNI.
 *
 * Layer: BACKEND (cpu_x86). Built with -mavx512f -mavx512bw -mavx512dq
 * -mavx512vl -mavx512vnni (mk/backend-cpu_x86.mk). Whether it runs is
 * decided in attention_int8.c, outside this file (the dispatcher tier,
 * which GEIST_FORCE_ISA clamps, and cpuid), so that no EVEX instruction
 * runs before that check.
 *
 * The same computation and work split as the AVX2 kernel, with:
 *   - the dots in vpdpbusd, one instruction per 32 bytes and head, or per
 *     64 from head_dim 128 up, where AVX2 takes four per 32: the K bytes
 *     turned into u8 (k + 128) once for all heads of the pass, and 128
 *     times the sum of the query subtracted — exact in int32 (see Scores);
 *   - the V sums in zmm: 64 output dimensions of each of 4 or 3 heads, 128
 *     of 2, 256 of 1, in up to sixteen accumulators, each V row converted
 *     once for all heads of the pass;
 *   - head_dim 512 (Gemma 4's full-attention layers) a compile-time
 *     constant like 64, 128 and 256.
 * The integer dots and the order of every V sum are the AVX2 kernel's, and
 * the exponentials are the same bits. -ffast-math lets gcc round the rest
 * differently in each kernel (the order of the score's two products, the
 * grouping of the softmax's double sums), so the two agree to rounding,
 * not bit for bit. Any thread count gives the same bits, as there.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "attention.h"

#include "gemma4_kernels.h" /* ATTN_EXP_FLOOR */

#include <immintrin.h>
#include <math.h>
#include <stdalign.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* The architecture refuses a larger head_dim at load; the stack arrays
 * below are sized by it. */
constexpr size_t AVN_HEAD_DIM_MAX = 512;

/* Context positions per online-softmax block. */
constexpr size_t AVN_BLOCK = 512;

/* V rows per pass over the output dimensions: their bytes stay in L1 while
 * every slice of the dimensions goes over them. */
constexpr size_t AVN_PV_ROWS = 64;

/* Query heads per pass, at most. */
constexpr size_t AVN_HEADS_PER_PASS_MAX = 4;

/* Decode chunks per pass, at most, and the split rule: as the AVX2
 * kernel's (see ai8_plan_for), whose part buffer this one fills. */
constexpr size_t AVN_MAX_CHUNKS = 4;
constexpr size_t AVN_CHUNK_SPAN = 1024;
constexpr size_t AVN_MIN_ITEMS  = 8;
constexpr size_t AVN_CHUNK_MIN  = 128;

struct avn_args {
    size_t        n_q_heads, head_dim, n_kv, n_kv_heads, q_offset, sliding_window;
    const float  *q;
    const int8_t *k;
    const float  *k_scale;
    const int8_t *v;
    const float  *v_scale;
    float        *out;
};

/* The context positions [*s_lo, *s_hi] query t attends to. */
static inline void avn_span(const struct avn_args *a, size_t t, size_t *s_lo, size_t *s_hi) {
    const size_t q_pos = a->q_offset + t;
    *s_lo = (a->sliding_window > 0 && q_pos + 1 > a->sliding_window) ? q_pos + 1 - a->sliding_window
                                                                     : 0;
    *s_hi = q_pos < a->n_kv ? q_pos : a->n_kv - 1;
}

struct avn_plan {
    size_t per_pass; /* query heads per pass */
    size_t n_chunks; /* decode: context chunks per pass, merged afterwards */
};

/* How to run a call, from its shape alone: the AVX2 kernel's rule. */
static struct avn_plan avn_plan_for(size_t n_q,
                                    size_t n_q_heads,
                                    size_t n_kv_heads,
                                    size_t head_dim,
                                    size_t decode_span,
                                    size_t part_floats) {
    const size_t group    = n_q_heads / n_kv_heads;
    size_t       per_pass = 1;
    for (size_t g = AVN_HEADS_PER_PASS_MAX; g >= 2; g--) {
        if (group % g == 0) {
            per_pass = g;
            break;
        }
    }
    struct avn_plan plan = {.per_pass = per_pass, .n_chunks = 1};
    if (n_q == 1) {
        const size_t items     = n_kv_heads * (group / per_pass);
        const size_t for_items = (AVN_MIN_ITEMS + items - 1) / items;
        const size_t fit       = decode_span / AVN_CHUNK_MIN;
        size_t       chunks    = decode_span / AVN_CHUNK_SPAN;
        chunks                 = chunks > for_items ? chunks : for_items;
        chunks                 = chunks < AVN_MAX_CHUNKS ? chunks : AVN_MAX_CHUNKS;
        chunks                 = chunks < fit ? chunks : fit;
        if (chunks >= 2 && part_floats >= n_q_heads * chunks * (head_dim + 2)) {
            plan.n_chunks = chunks;
        }
    }
    return plan;
}

/* ---- Scores ------------------------------------------------------------ */

/* The dots are vpdpbusd's: k ^ 0x80 is k + 128 as u8, the instruction's
 * unsigned operand, and the int8 query its signed one, so the lanes sum
 * (k + 128) * q, and the dot is their sum less 128 * sum(q), a constant of
 * the query (q_corr below). A vpdpbusd adds four u8 * s8 products to a
 * lane, at most 4 * 255 * 127 in magnitude, and a lane gets at most 16 of
 * them (head_dim 512): exact in int32.
 *
 * The helpers take two K rows ka, kb and the S (<= 4) query heads q[0 ..
 * S) and give per head h[g] = hadd(ka's lanes, kb's lanes). Each K chunk is
 * flipped once for all heads. The accumulators are named, not an array: in
 * a loop of run-time length gcc kept an array of them in memory. */

/* 32-byte chunks: hd32 bytes of whole chunks, then the rest through the
 * byte mask `tail` (0 when head_dim is a multiple of 32), whose masked
 * load reads nothing past the row; the masked-off bytes (0x80 once
 * flipped) meet the query's zero padding. */
[[gnu::always_inline]] static inline void avn_dot_pair32(size_t    S,
                                                         size_t    hd32,
                                                         __mmask32 tail,
                                                         const int8_t (*q)[AVN_HEAD_DIM_MAX],
                                                         const int8_t *ka,
                                                         const int8_t *kb,
                                                         __m256i h[static AVN_HEADS_PER_PASS_MAX]) {
    const __m256i flip = _mm256_set1_epi8((char) 0x80);
    __m256i       a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
    __m256i       b0 = a0, b1 = a0, b2 = a0, b3 = a0;
    for (size_t c = 0; c < hd32; c += 32) {
        const __m256i xa = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *) (ka + c)), flip);
        const __m256i xb = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *) (kb + c)), flip);
        __m256i       qq = _mm256_load_si256((const __m256i *) (q[0] + c));
        a0               = _mm256_dpbusd_epi32(a0, xa, qq);
        b0               = _mm256_dpbusd_epi32(b0, xb, qq);
        if (S > 1) {
            qq = _mm256_load_si256((const __m256i *) (q[1] + c));
            a1 = _mm256_dpbusd_epi32(a1, xa, qq);
            b1 = _mm256_dpbusd_epi32(b1, xb, qq);
        }
        if (S > 2) {
            qq = _mm256_load_si256((const __m256i *) (q[2] + c));
            a2 = _mm256_dpbusd_epi32(a2, xa, qq);
            b2 = _mm256_dpbusd_epi32(b2, xb, qq);
        }
        if (S > 3) {
            qq = _mm256_load_si256((const __m256i *) (q[3] + c));
            a3 = _mm256_dpbusd_epi32(a3, xa, qq);
            b3 = _mm256_dpbusd_epi32(b3, xb, qq);
        }
    }
    if (tail != 0) {
        const __m256i xa = _mm256_xor_si256(_mm256_maskz_loadu_epi8(tail, ka + hd32), flip);
        const __m256i xb = _mm256_xor_si256(_mm256_maskz_loadu_epi8(tail, kb + hd32), flip);
        __m256i       qq = _mm256_load_si256((const __m256i *) (q[0] + hd32));
        a0               = _mm256_dpbusd_epi32(a0, xa, qq);
        b0               = _mm256_dpbusd_epi32(b0, xb, qq);
        if (S > 1) {
            qq = _mm256_load_si256((const __m256i *) (q[1] + hd32));
            a1 = _mm256_dpbusd_epi32(a1, xa, qq);
            b1 = _mm256_dpbusd_epi32(b1, xb, qq);
        }
        if (S > 2) {
            qq = _mm256_load_si256((const __m256i *) (q[2] + hd32));
            a2 = _mm256_dpbusd_epi32(a2, xa, qq);
            b2 = _mm256_dpbusd_epi32(b2, xb, qq);
        }
        if (S > 3) {
            qq = _mm256_load_si256((const __m256i *) (q[3] + hd32));
            a3 = _mm256_dpbusd_epi32(a3, xa, qq);
            b3 = _mm256_dpbusd_epi32(b3, xb, qq);
        }
    }
    h[0] = _mm256_hadd_epi32(a0, b0);
    h[1] = _mm256_hadd_epi32(a1, b1);
    h[2] = _mm256_hadd_epi32(a2, b2);
    h[3] = _mm256_hadd_epi32(a3, b3);
}

/* Sixteen lanes as eight: the upper 256 bits added to the lower. */
static inline __m256i avn_fold(__m512i v) {
    return _mm256_add_epi32(_mm512_castsi512_si256(v), _mm512_extracti64x4_epi64(v, 1));
}

/* 64-byte chunks, for a head_dim that is a multiple of 64: half the
 * vpdpbusd of avn_dot_pair32, and the query of four heads at head_dim 256
 * in sixteen registers (32-byte chunks would take all 32, and gcc spilled
 * the accumulators instead). */
[[gnu::always_inline]] static inline void avn_dot_pair64(size_t S,
                                                         size_t head_dim,
                                                         const int8_t (*q)[AVN_HEAD_DIM_MAX],
                                                         const int8_t *ka,
                                                         const int8_t *kb,
                                                         __m256i h[static AVN_HEADS_PER_PASS_MAX]) {
    const __m512i flip = _mm512_set1_epi8((char) 0x80);
    __m512i       a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0;
    __m512i       b0 = a0, b1 = a0, b2 = a0, b3 = a0;
    for (size_t c = 0; c < head_dim; c += 64) {
        const __m512i xa = _mm512_xor_si512(_mm512_loadu_si512(ka + c), flip);
        const __m512i xb = _mm512_xor_si512(_mm512_loadu_si512(kb + c), flip);
        __m512i       qq = _mm512_load_si512(q[0] + c);
        a0               = _mm512_dpbusd_epi32(a0, xa, qq);
        b0               = _mm512_dpbusd_epi32(b0, xb, qq);
        if (S > 1) {
            qq = _mm512_load_si512(q[1] + c);
            a1 = _mm512_dpbusd_epi32(a1, xa, qq);
            b1 = _mm512_dpbusd_epi32(b1, xb, qq);
        }
        if (S > 2) {
            qq = _mm512_load_si512(q[2] + c);
            a2 = _mm512_dpbusd_epi32(a2, xa, qq);
            b2 = _mm512_dpbusd_epi32(b2, xb, qq);
        }
        if (S > 3) {
            qq = _mm512_load_si512(q[3] + c);
            a3 = _mm512_dpbusd_epi32(a3, xa, qq);
            b3 = _mm512_dpbusd_epi32(b3, xb, qq);
        }
    }
    h[0] = _mm256_hadd_epi32(avn_fold(a0), avn_fold(b0));
    h[1] = _mm256_hadd_epi32(avn_fold(a1), avn_fold(b1));
    h[2] = _mm256_hadd_epi32(avn_fold(a2), avn_fold(b2));
    h[3] = _mm256_hadd_epi32(avn_fold(a3), avn_fold(b3));
}

/* Rows k0 .. k0 + 3 * row against the S heads q[0 .. S): per head, lane
 * p of the lower half and lane p of the upper half sum to row p's lanes
 * (0 <= p < 4). W is the chunk width, 32 or 64 bytes. */
[[gnu::always_inline]] static inline void avn_dots4(size_t    W,
                                                    size_t    S,
                                                    size_t    head_dim,
                                                    size_t    hd32,
                                                    __mmask32 tail,
                                                    const int8_t (*q)[AVN_HEAD_DIM_MAX],
                                                    const int8_t *k0,
                                                    size_t        row,
                                                    __m256i u[static AVN_HEADS_PER_PASS_MAX]) {
    __m256i h01[AVN_HEADS_PER_PASS_MAX], h23[AVN_HEADS_PER_PASS_MAX];
    if (W == 64) {
        avn_dot_pair64(S, head_dim, q, k0, k0 + row, h01);
        avn_dot_pair64(S, head_dim, q, k0 + 2 * row, k0 + 3 * row, h23);
    } else {
        avn_dot_pair32(S, hd32, tail, q, k0, k0 + row, h01);
        avn_dot_pair32(S, hd32, tail, q, k0 + 2 * row, k0 + 3 * row, h23);
    }
    for (size_t g = 0; g < S; g++) {
        u[g] = _mm256_hadd_epi32(h01[g], h23[g]);
    }
}

static inline int32_t avn_hsum(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s         = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s         = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}

/* ---- V ------------------------------------------------------------------ */

/* 16 V values of a row as floats; `m` masks the load (a head_dim tail). */
static inline __m512 avn_cvt16(__mmask16 m, const int8_t *v) {
    return _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_maskz_loadu_epi8(m, v)));
}

/* acc[g][c .. c + 16 NV) += sum over the n rows j of w[g][j] * v_j[c ..],
 * j in order, for the G heads of the pass: G * NV <= 16 accumulators,
 * which gcc keeps in registers (the array is local, every index constant
 * once unrolled). `last` masks the last 16 dimensions' V bytes; acc holds
 * zeros past head_dim, so the masked-off lanes add nothing. */
[[gnu::always_inline]] static inline void avn_pv(size_t        G,
                                                 size_t        NV,
                                                 size_t        n,
                                                 size_t        row,
                                                 size_t        c,
                                                 __mmask16     last,
                                                 const int8_t *v,
                                                 const float (*w)[AVN_BLOCK],
                                                 float (*acc)[AVN_HEAD_DIM_MAX]) {
    __m512 x[16];
#pragma GCC unroll 4
    for (size_t g = 0; g < G; g++) {
#pragma GCC unroll 16
        for (size_t k = 0; k < NV; k++) {
            x[g * NV + k] = _mm512_load_ps(acc[g] + c + 16 * k);
        }
    }
    const int8_t *vr = v + c;
    for (size_t j = 0; j < n; j++, vr += row) {
        if (G == 1) {
            /* Each V vector straight into its FMA: sixteen accumulators
             * and sixteen converted vectors would not fit in 32 registers
             * (gcc spilled one). */
            const __m512 s = _mm512_set1_ps(w[0][j]);
#pragma GCC unroll 16
            for (size_t k = 0; k < NV; k++) {
                x[k] = _mm512_fmadd_ps(
                        s, avn_cvt16(k + 1 < NV ? (__mmask16) 0xFFFF : last, vr + 16 * k), x[k]);
            }
            continue;
        }
        __m512 vv[16];
#pragma GCC unroll 16
        for (size_t k = 0; k < NV; k++) {
            vv[k] = avn_cvt16(k + 1 < NV ? (__mmask16) 0xFFFF : last, vr + 16 * k);
        }
#pragma GCC unroll 4
        for (size_t g = 0; g < G; g++) {
            const __m512 s = _mm512_set1_ps(w[g][j]);
#pragma GCC unroll 16
            for (size_t k = 0; k < NV; k++) {
                x[g * NV + k] = _mm512_fmadd_ps(s, vv[k], x[g * NV + k]);
            }
        }
    }
#pragma GCC unroll 4
    for (size_t g = 0; g < G; g++) {
#pragma GCC unroll 16
        for (size_t k = 0; k < NV; k++) {
            _mm512_store_ps(acc[g] + c + 16 * k, x[g * NV + k]);
        }
    }
}

/* All of head_dim for n rows: the widest slices that sixteen accumulators
 * hold (256 dimensions of one head, 128 of two, 64 of three or four),
 * then single vectors, then the masked tail. */
[[gnu::always_inline]] static inline void avn_pv_rows(size_t        G,
                                                      size_t        head_dim,
                                                      size_t        n,
                                                      size_t        row,
                                                      const int8_t *v,
                                                      const float (*w)[AVN_BLOCK],
                                                      float (*acc)[AVN_HEAD_DIM_MAX]) {
    const size_t    hd16 = head_dim & ~(size_t) 15;
    const __mmask16 full = 0xFFFF;
    size_t          c    = 0;
    if (G == 1) {
        for (; c + 256 <= hd16; c += 256) {
            avn_pv(1, 16, n, row, c, full, v, w, acc);
        }
    }
    if (G <= 2) {
        for (; c + 128 <= hd16; c += 128) {
            avn_pv(G, 8, n, row, c, full, v, w, acc);
        }
    }
    for (; c + 64 <= hd16; c += 64) {
        avn_pv(G, 4, n, row, c, full, v, w, acc);
    }
    for (; c < hd16; c += 16) {
        avn_pv(G, 1, n, row, c, full, v, w, acc);
    }
    if (hd16 < head_dim) {
        avn_pv(G, 1, n, row, hd16, (__mmask16) ((1u << (head_dim - hd16)) - 1u), v, w, acc);
    }
}

/* ---- One work item -------------------------------------------------------- */

/* Query heads [h0, h0 + G) at position t over context positions
 * [c_lo, c_hi], all reading KV head kv_h: the AVX2 kernel's ai8_item.
 * Without `part` it writes their output; with it, their partial results.
 * G and HD are compile-time constants at every call site (HD = 0: head_dim
 * at run time). */
[[gnu::always_inline]] static inline void avn_item(size_t                 G,
                                                   size_t                 HD,
                                                   const struct avn_args *a,
                                                   size_t                 t,
                                                   size_t                 kv_h,
                                                   size_t                 h0,
                                                   size_t                 c_lo,
                                                   size_t                 c_hi,
                                                   float                 *part) {
    /* Locals, not a->field in the loops: after OpenMP outlining `a` points
     * into the caller's frame, and every float store could alias it. */
    const size_t    head_dim  = HD != 0 ? HD : a->head_dim;
    const size_t    n_q_heads = a->n_q_heads, n_kv_heads = a->n_kv_heads;
    const size_t    row    = n_kv_heads * head_dim; /* K/V bytes per position */
    const size_t    hd32   = head_dim & ~(size_t) 31;
    const size_t    hd16up = (head_dim + 15) & ~(size_t) 15;
    const __mmask32 tail   = (__mmask32) ((1ull << (head_dim - hd32)) - 1u);
    /* The dots' shape, from the compile-time head_dim: 64-byte chunks
     * from 128 up; at 512 at most two heads at a time, whose query fills
     * sixteen registers; eight positions at a time where the query takes
     * at most eight registers (with head_dim at run time it is loaded as
     * it is used), four elsewhere. */
    const size_t  W       = HD >= 128 ? 64 : 32;
    const size_t  SM      = HD == 512 ? 2 : AVN_HEADS_PER_PASS_MAX;
    const size_t  RS      = HD == 0 || HD / W * (G < SM ? G : SM) <= 8 ? 8 : 4;
    const float  *k_scale = a->k_scale;
    const float  *v_scale = a->v_scale;
    const int8_t *kh      = a->k + kv_h * head_dim;
    const int8_t *vh      = a->v + kv_h * head_dim;

    /* The query, quantized per head as the AVX2 kernel does, zero past
     * head_dim to the end of its last 32 bytes; q_corr: 128 times the sum
     * of its values, which the dots carry. */
    alignas(64) int8_t q_s8[G][AVN_HEAD_DIM_MAX];
    float              scale_q[G];
    int32_t            q_corr[G];
    for (size_t g = 0; g < G; g++) {
        const float *qv   = a->q + (t * n_q_heads + h0 + g) * head_dim;
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
        int32_t     qsum  = 0;
        for (size_t i = 0; i < head_dim; i++) {
            q_s8[g][i] = (int8_t) lrintf(qv[i] * inv_q);
            qsum += q_s8[g][i];
        }
        for (size_t i = head_dim; i < hd32 + (tail != 0 ? 32 : 0); i++) {
            q_s8[g][i] = 0;
        }
        scale_q[g] = sq;
        q_corr[g]  = 128 * qsum;
    }

    /* sc: the block's scores, then exp(score - max), then those times the
     * row's V scale — the weight of each V row. acc holds zeros up to the
     * next multiple of 16 (the V tail's masked-off lanes). */
    alignas(64) float sc[G][AVN_BLOCK];
    alignas(32) float ks[AVN_BLOCK];
    alignas(32) float vs[AVN_BLOCK];
    alignas(64) float acc[G][AVN_HEAD_DIM_MAX];
    float             max_score[G];
    double            sum_exp[G];
    for (size_t g = 0; g < G; g++) {
        for (size_t i = 0; i < hd16up; i++) {
            acc[g][i] = 0.0f;
        }
        sum_exp[g] = 0.0;
    }
    for (size_t b0 = c_lo; b0 <= c_hi; b0 += AVN_BLOCK) {
        const size_t  n  = c_hi - b0 < AVN_BLOCK ? c_hi - b0 + 1 : AVN_BLOCK;
        const int8_t *kb = kh + b0 * row;
        const int8_t *vb = vh + b0 * row;
        for (size_t j = 0; j < n; j++) {
            ks[j] = k_scale[(b0 + j) * n_kv_heads + kv_h];
            vs[j] = v_scale[(b0 + j) * n_kv_heads + kv_h];
        }
        /* RS positions at a time, their lanes summed pairwise as each
         * pair of rows is done, so that the query and the accumulators
         * stay in registers. Heads in groups of SM, each group over the
         * whole block: within one loop gcc shared the flipped K rows
         * between the groups and spilled them. */
        for (size_t g0 = 0; g0 < G; g0 += SM) {
            const size_t S = G - g0 < SM ? G - g0 : SM;
            size_t       j = 0;
            for (; j + RS <= n; j += RS) {
                const int8_t *k0 = kb + j * row;
                __m256i       u0[AVN_HEADS_PER_PASS_MAX], u1[AVN_HEADS_PER_PASS_MAX];
                avn_dots4(W, S, head_dim, hd32, tail, &q_s8[g0], k0, row, u0);
                if (RS == 8) {
                    avn_dots4(W, S, head_dim, hd32, tail, &q_s8[g0], k0 + 4 * row, row, u1);
                }
                for (size_t g = 0; g < S; g++) {
                    /* (dot * scale_q) * scale_k, as the AVX2 kernel writes
                     * it; the dot of row j + p in lane p, less the offset. */
                    if (RS == 8) {
                        const __m256i d = _mm256_sub_epi32(
                                _mm256_add_epi32(_mm256_permute2x128_si256(u0[g], u1[g], 0x20),
                                                 _mm256_permute2x128_si256(u0[g], u1[g], 0x31)),
                                _mm256_set1_epi32(q_corr[g0 + g]));
                        const __m256 s =
                                _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(d),
                                                            _mm256_set1_ps(scale_q[g0 + g])),
                                              _mm256_load_ps(ks + j));
                        _mm256_store_ps(sc[g0 + g] + j, s);
                    } else {
                        const __m128i d =
                                _mm_sub_epi32(_mm_add_epi32(_mm256_castsi256_si128(u0[g]),
                                                            _mm256_extracti128_si256(u0[g], 1)),
                                              _mm_set1_epi32(q_corr[g0 + g]));
                        const __m128 s = _mm_mul_ps(
                                _mm_mul_ps(_mm_cvtepi32_ps(d), _mm_set1_ps(scale_q[g0 + g])),
                                _mm_load_ps(ks + j));
                        _mm_store_ps(sc[g0 + g] + j, s);
                    }
                }
            }
            for (; j < n; j++) {
                const int8_t *kr = kb + j * row;
                /* The row as both of the pair: h's lanes sum to twice its
                 * dot. */
                __m256i h[AVN_HEADS_PER_PASS_MAX];
                if (W == 64) {
                    avn_dot_pair64(S, head_dim, &q_s8[g0], kr, kr, h);
                } else {
                    avn_dot_pair32(S, hd32, tail, &q_s8[g0], kr, kr, h);
                }
                for (size_t g = 0; g < S; g++) {
                    const int32_t d = avn_hsum(h[g]) / 2 - q_corr[g0 + g];
                    sc[g0 + g][j]   = (float) d * scale_q[g0 + g] * ks[j];
                }
            }
        }
        for (size_t g = 0; g < G; g++) {
            float block_max = sc[g][0];
            for (size_t i = 1; i < n; i++) {
                if (sc[g][i] > block_max) {
                    block_max = sc[g][i];
                }
            }
            if (b0 == c_lo) {
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
        /* AVN_PV_ROWS rows at a time, so that their V bytes stay in L1
         * while every slice of the output dimensions passes over them. The
         * sums keep their order. */
        for (size_t r0 = 0; r0 < n; r0 += AVN_PV_ROWS) {
            const size_t nr             = n - r0 < AVN_PV_ROWS ? n - r0 : AVN_PV_ROWS;
            const float (*w)[AVN_BLOCK] = (const float (*)[AVN_BLOCK]) & sc[0][r0];
            avn_pv_rows(G, head_dim, nr, row, vb + r0 * row, w, acc);
        }
    }
    for (size_t g = 0; g < G; g++) {
        if (part != nullptr) {
            float *rec = part + g * (head_dim + 2);
            memcpy(rec, acc[g], head_dim * sizeof(float));
            rec[head_dim]     = max_score[g];
            rec[head_dim + 1] = (float) sum_exp[g];
        } else {
            const float inv_sum = (float) (1.0 / sum_exp[g]);
            float      *outv    = a->out + (t * n_q_heads + h0 + g) * head_dim;
            for (size_t i = 0; i < head_dim; i++) {
                outv[i] = acc[g][i] * inv_sum;
            }
        }
    }
}

#define AVN_ITEM_HD(G)                                          \
    do {                                                        \
        switch (a->head_dim) {                                  \
        case 64:                                                \
            avn_item(G, 64, a, t, kv_h, h0, c_lo, c_hi, part);  \
            break;                                              \
        case 128:                                               \
            avn_item(G, 128, a, t, kv_h, h0, c_lo, c_hi, part); \
            break;                                              \
        case 256:                                               \
            avn_item(G, 256, a, t, kv_h, h0, c_lo, c_hi, part); \
            break;                                              \
        case 512:                                               \
            avn_item(G, 512, a, t, kv_h, h0, c_lo, c_hi, part); \
            break;                                              \
        default:                                                \
            avn_item(G, 0, a, t, kv_h, h0, c_lo, c_hi, part);   \
            break;                                              \
        }                                                       \
    } while (0)

static void avn_run_item(size_t                 per_pass,
                         const struct avn_args *a,
                         size_t                 t,
                         size_t                 kv_h,
                         size_t                 h0,
                         size_t                 c_lo,
                         size_t                 c_hi,
                         float                 *part) {
    switch (per_pass) {
    case 4:
        AVN_ITEM_HD(4);
        break;
    case 3:
        AVN_ITEM_HD(3);
        break;
    case 2:
        AVN_ITEM_HD(2);
        break;
    default:
        AVN_ITEM_HD(1);
        break;
    }
}

/* One head's output from its n_chunks partial results (records `stride`
 * floats apart), rescaled to their common max and summed in chunk order:
 * the AVX2 kernel's merge. */
static void
avn_merge(size_t n_chunks, size_t head_dim, size_t stride, const float *part, float *out) {
    float max_score = part[head_dim];
    for (size_t c = 1; c < n_chunks; c++) {
        const float m = part[c * stride + head_dim];
        max_score     = m > max_score ? m : max_score;
    }
    float  acc[AVN_HEAD_DIM_MAX];
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

void cpu_x86_attention_kv_int8_run_avx512_vnni(size_t        n_q,
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
    const struct avn_args a      = {.n_q_heads      = n_q_heads,
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
    size_t                dec_lo = 0, dec_hi = 0;
    avn_span(&a, 0, &dec_lo, &dec_hi);
    const struct avn_plan plan     = avn_plan_for(n_q,
                                                  n_q_heads,
                                                  n_kv_heads,
                                                  head_dim,
                                                  dec_hi - dec_lo + 1,
                                                  part != nullptr ? part_floats : 0);
    const size_t          per_pass = plan.per_pass;
    const size_t          n_passes = group / per_pass;
    if (plan.n_chunks > 1) {
        /* Split decode (n_q == 1): items (KV head, pass, chunk) leave their
         * partial results in `part`, then each head merges its chunks.
         * Chunk c covers positions [dec_lo + c * len, ...], at least
         * AVN_CHUNK_MIN each, so none is empty. */
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
                        avn_run_item(per_pass,
                                     &a,
                                     0,
                                     kv_h,
                                     kv_h * group + pass * per_pass,
                                     c_lo,
                                     c_hi,
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
                    avn_merge(n_chunks,
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
    /* Causal and window masks make later positions longer: dynamic. */
#if defined(_OPENMP)
#pragma omp parallel for collapse(3) schedule(dynamic)
#endif
    for (size_t t = 0; t < n_q; t++) {
        for (size_t kv_h = 0; kv_h < n_kv_heads; kv_h++) {
            for (size_t pass = 0; pass < n_passes; pass++) {
                size_t c_lo = 0, c_hi = 0;
                avn_span(&a, t, &c_lo, &c_hi);
                avn_run_item(
                        per_pass, &a, t, kv_h, kv_h * group + pass * per_pass, c_lo, c_hi, nullptr);
            }
        }
    }
}

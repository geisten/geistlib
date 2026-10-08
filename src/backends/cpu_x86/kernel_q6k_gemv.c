/*
 * src/backends/cpu_x86/kernel_q6k_gemv.c — native Q6_K and Q3_K GEMV / GEMM.
 *
 * Layer: BACKEND (cpu_x86).
 *
 * Decode (M=1) reading the ORIGINAL Q6_K weights (block_q6_K_t, ~0.82 B/wt)
 * instead of the W8A8 predecode (1.5 B/wt). Q6_K decode (ffn_down, lm_head)
 * is bandwidth-bound, so halving the weight traffic is the lever.
 *
 * The per-row dot is a faithful port of llama.cpp's AVX2
 * ggml_vec_dot_q6_K_q8_K (ggml/src/ggml-cpu/arch/x86/quants.c): unpack the
 * 6-bit weights, VPMADDUBSW against the int8 activation, apply the int8
 * sub-block scales via VPMADD, fold the uniform -32 offset through the
 * activation block-sums, and multiply by the fp32 super-block scale once.
 * Original code Copyright (c) 2023-2025 The ggml authors, MIT-licensed.
 *
 * Q3_K has the same shape (16 int8-scaled sub-blocks of 16, unsigned codes
 * with a uniform offset), so the same dot reads it with its own unpack and
 * scale decode (ggml_vec_dot_q3_K_q8_K) and an offset of 4 instead of 32
 * (#410).
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "kernel_q6k_gemv.h"
#include "linear_util.h"

#include "quant.h" /* fp16_to_fp32 */
#include "quant_blocks.h"

#include <immintrin.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h> /* memcpy */

#include "par.h"

/* q8_K activation row: int8 quants + per-16 block sums + super-block scale. */
struct q8k_act {
    float   d;
    int8_t  qs[256];
    int16_t bsums[16];
};

/* Quantize one fp32 activation super-block group to q8_K (matches the format
 * the dot expects: d = amax/127, qs = round(x/d), bsums summed per 16). */
static void quantize_q8k_act(size_t n_super, const float *x, struct q8k_act *out) {
    for (size_t s = 0; s < n_super; s++) {
        const float *xs   = x + s * 256;
        float        amax = 0.0f;
        for (size_t k = 0; k < 256; k++) {
            const float ax = fabsf(xs[k]);
            if (ax > amax)
                amax = ax;
        }
        const float d  = amax / 127.0f;
        const float id = (amax > 0.0f) ? 127.0f / amax : 0.0f;
        out[s].d       = d;
        for (size_t g = 0; g < 16; g++) {
            int32_t sum = 0;
            for (size_t i = 0; i < 16; i++) {
                const int v           = (int) lrintf(xs[g * 16 + i] * id);
                out[s].qs[g * 16 + i] = (int8_t) v;
                sum += v;
            }
            out[s].bsums[g] = (int16_t) sum;
        }
    }
}

static inline __m128i scale_shuffle(int i) {
    static const uint8_t k[128] = {
            0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,
            2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,  5,
            5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,
            8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,  9,  9,  10, 10, 10, 10, 10, 10, 10, 10,
            11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13,
            13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15};
    return _mm_loadu_si128((const __m128i *) k + i);
}

/* The two formats this file reads. Both are 256-element super-blocks of 16
 * sub-blocks of 16, an int8 scale per sub-block, one fp16 d, and unsigned
 * codes with a uniform offset: Q6_K 0..63 - 32, Q3_K 0..7 - 4. So one dot
 * serves both; only the unpacking differs. */
enum kfmt { KF_Q6K, KF_Q3K };

static inline size_t kfmt_bytes(enum kfmt f) {
    return f == KF_Q6K ? sizeof(struct block_q6_K_t) : sizeof(struct block_q3_K_t);
}

/* log2 of the offset, folded in through the activation block sums. */
static inline int kfmt_offset_shift(enum kfmt f) {
    return f == KF_Q6K ? 5 : 2;
}

static inline float blk_d(enum kfmt f, const uint8_t *b) {
    return fp16_to_fp32(f == KF_Q6K ? ((const struct block_q6_K_t *) b)->d
                                    : ((const struct block_q3_K_t *) b)->d);
}

/* The 16 sub-block scales as int8. Q3_K packs them in 6 bits with an offset
 * of 32 (unpack_q3k_scales in q3_K.c); this is llama.cpp's AVX2 unpack. */
static inline __m128i blk_scales(enum kfmt f, const uint8_t *b) {
    if (f == KF_Q6K) {
        return _mm_loadu_si128((const __m128i *) ((const struct block_q6_K_t *) b)->scales);
    }
    uint32_t aux[3];
    memcpy(aux, ((const struct block_q3_K_t *) b)->scales, sizeof aux);
    const uint32_t k1 = 0x03030303u;
    const uint32_t k2 = 0x0F0F0F0Fu;
    const __m128i  s  = _mm_set_epi32((int) (((aux[1] >> 4) & k2) | (((aux[2] >> 6) & k1) << 4)),
                                      (int) (((aux[0] >> 4) & k2) | (((aux[2] >> 4) & k1) << 4)),
                                      (int) ((aux[1] & k2) | (((aux[2] >> 2) & k1) << 4)),
                                      (int) ((aux[0] & k2) | ((aux[2] & k1) << 4)));
    return _mm_sub_epi8(s, _mm_set1_epi8(32));
}

/* Half j (elements 128 j .. 128 j + 127) of a super-block as four vectors of
 * unsigned codes, q[k] holding elements 128 j + 32 k .. + 31. */
[[gnu::always_inline]] static inline void
blk_unpack(enum kfmt f, const uint8_t *b, size_t j, __m256i q[4]) {
    const __m256i m3 = _mm256_set1_epi8(3);
    if (f == KF_Q6K) {
        const struct block_q6_K_t *x       = (const struct block_q6_K_t *) b;
        const __m256i              m15     = _mm256_set1_epi8(15);
        const __m256i              q4bits1 = _mm256_loadu_si256((const __m256i *) (x->ql + 64 * j));
        const __m256i q4bits2 = _mm256_loadu_si256((const __m256i *) (x->ql + 64 * j + 32));
        const __m256i q4bitsH = _mm256_loadu_si256((const __m256i *) (x->qh + 32 * j));

        const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m3), 4);
        const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(12)), 2);
        const __m256i q4h_2 = _mm256_and_si256(q4bitsH, _mm256_set1_epi8(48));
        const __m256i q4h_3 =
                _mm256_srli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8((char) -64)), 2);

        q[0] = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
        q[1] = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
        q[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
        q[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);
        return;
    }
    /* Q3_K: bits 2k..2k+1 of qs[32 j + l] are the low two bits of element
     * 128 j + 32 k + l, bit 4 j + k of hmask[l] its third bit. */
    const struct block_q3_K_t *x    = (const struct block_q3_K_t *) b;
    const __m256i              bits = _mm256_loadu_si256((const __m256i *) (x->qs + 32 * j));
    const __m256i              hm   = _mm256_loadu_si256((const __m256i *) x->hmask);
    for (int k = 0; k < 4; k++) {
        const int     hb   = 4 * (int) j + k;
        const __m256i low  = _mm256_and_si256(_mm256_srli_epi16(bits, 2 * k), m3);
        const __m256i high = _mm256_and_si256(hm, _mm256_set1_epi8((char) (1u << hb)));
        q[k]               = _mm256_or_si256(
                low, hb <= 2 ? _mm256_slli_epi16(high, 2 - hb) : _mm256_srli_epi16(high, hb - 2));
    }
}

/* One output row's dot: sum over n_super super-blocks of w (row-major,
 * kfmt_bytes(f) each). */
[[gnu::always_inline]] static inline float
dot_q8k(enum kfmt f, size_t n_super, const uint8_t *w, const struct q8k_act *y) {
    __m256 acc = _mm256_setzero_ps();

    for (size_t i = 0; i < n_super; i++) {
        const uint8_t *b  = w + i * kfmt_bytes(f);
        const float    d  = y[i].d * blk_d(f, b);
        const int8_t  *q8 = y[i].qs;

        const __m256i q8sums   = _mm256_loadu_si256((const __m256i *) y[i].bsums);
        const __m128i scales   = blk_scales(f, b);
        const __m256i scales16 = _mm256_cvtepi8_epi16(scales);
        const __m256i q8sclsub =
                _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales16), kfmt_offset_shift(f));

        __m256i sumi = _mm256_setzero_si256();
        int     is   = 0;
        for (size_t j = 0; j < 2; j++) { /* QK_K/128 = 2 */
            __m256i q[4];
            blk_unpack(f, b, j, q);
            for (int k = 0; k < 4; k++) {
                const __m256i q8k = _mm256_loadu_si256((const __m256i *) (q8 + 32 * k));
                const __m128i s   = _mm_shuffle_epi8(scales, scale_shuffle(is + k));
                q[k] = _mm256_madd_epi16(_mm256_cvtepi8_epi16(s), _mm256_maddubs_epi16(q[k], q8k));
            }
            q8 += 128;
            is += 4;
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(q[0], q[1]));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(q[2], q[3]));
        }
        sumi = _mm256_sub_epi32(sumi, q8sclsub);
        acc  = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }
    return hsum_ps_hadd(acc);
}

/* One instance per format. The OpenMP regions below are outlined once for
 * both formats, with f a runtime value there, so they pick an instance per
 * row rather than inline the dot with an unknown f. */
static float dot_q6k(size_t n_super, const uint8_t *w, const struct q8k_act *y) {
    return dot_q8k(KF_Q6K, n_super, w, y);
}
static float dot_q3k(size_t n_super, const uint8_t *w, const struct q8k_act *y) {
    return dot_q8k(KF_Q3K, n_super, w, y);
}
static inline float
dot_fmt(enum kfmt f, size_t n_super, const uint8_t *w, const struct q8k_act *y) {
    return f == KF_Q6K ? dot_q6k(n_super, w, y) : dot_q3k(n_super, w, y);
}

/* Super-blocks of quantized activation held on the stack at once (16384
 * elements, ~18.7 KB). A longer row runs in segments of this size — the
 * bound is a tile size, not a limit on K. */
constexpr size_t Q8K_SEG = 64;

/* One GEMV segment or GEMM call for geist_par_for. */
struct kq_call {
    size_t          M, N, K, n_super, row_bytes;
    size_t          w_off; /* GEMV: the segment's byte offset in a row */
    bool            first; /* GEMV: the first segment stores, the rest add */
    const uint8_t  *raw;
    const float    *x;
    struct q8k_act *a;
    float          *y;
};

/* GEMV: rows [r0, r1) of one segment of n_super super-blocks. */
[[gnu::always_inline]] static inline void
gemv_rows(enum kfmt f, const struct kq_call *pc, size_t r0, size_t r1) {
    const struct kq_call c = *pc;
    for (size_t r = r0; r < r1; r++) {
        const float d = dot_fmt(f, c.n_super, c.raw + r * c.row_bytes + c.w_off, c.a);
        c.y[r]        = c.first ? d : c.y[r] + d;
    }
}

static void gemv_rows_q6k(void *ctx, size_t r0, size_t r1) {
    gemv_rows(KF_Q6K, ctx, r0, r1);
}
static void gemv_rows_q3k(void *ctx, size_t r0, size_t r1) {
    gemv_rows(KF_Q3K, ctx, r0, r1);
}

/* K up to Q8K_SEG * 256 = 16384 is one segment. Beyond it (ffn_down of any
 * model wider than 16384) each segment's partial dot is added into y. The
 * activation is quantized per super-block either way, so segmenting changes
 * only the order of the fp32 sum across segments. */
static void gemv_m1(enum kfmt f, size_t N, size_t K, const float *x, const uint8_t *raw, float *y) {
    const size_t n_super = K / 256;
    if (n_super == 0) {
        return; /* unreachable: the caller only binds K % 256 == 0, K >= 256 */
    }
    struct q8k_act a[Q8K_SEG];
    struct kq_call c = {.N = N, .row_bytes = n_super * kfmt_bytes(f), .raw = raw, .a = a, .y = y};
    for (size_t s0 = 0; s0 < n_super; s0 += Q8K_SEG) {
        c.n_super = n_super - s0 < Q8K_SEG ? n_super - s0 : Q8K_SEG;
        c.w_off   = s0 * kfmt_bytes(f);
        c.first   = s0 == 0;
        quantize_q8k_act(c.n_super, x + s0 * 256, a);
        geist_par_for(N, f == KF_Q6K ? gemv_rows_q6k : gemv_rows_q3k, &c);
    }
}

void q6k_gemv_m1(size_t N, size_t K, const float *x, const uint8_t *q6k_raw, float y[static N]) {
    gemv_m1(KF_Q6K, N, K, x, q6k_raw, y);
}

void q3k_gemv_m1(size_t N, size_t K, const float *x, const uint8_t *q3k_raw, float y[static N]) {
    gemv_m1(KF_Q3K, N, K, x, q3k_raw, y);
}

/* Activation tile height of the M>1 kernel. */
constexpr size_t KQ_NR = 4;

/* KQ_NR activation rows against one weight row: each super-block's codes and
 * scales unpacked once per tile, then the same sequence of integer ops as
 * dot_q8k per row (bit-identical per output). */
[[gnu::always_inline]] static inline void dot_q8k_rows(enum kfmt             f,
                                                       size_t                n_super,
                                                       const uint8_t        *w,
                                                       const struct q8k_act *y,
                                                       float                 out[static KQ_NR]) {
    __m256 acc[KQ_NR];
    for (size_t r = 0; r < KQ_NR; r++) {
        acc[r] = _mm256_setzero_ps();
    }

    for (size_t i = 0; i < n_super; i++) {
        const uint8_t *b        = w + i * kfmt_bytes(f);
        const float    dw       = blk_d(f, b);
        const __m128i  scales   = blk_scales(f, b);
        const __m256i  scales16 = _mm256_cvtepi8_epi16(scales);

        __m256i sumi[KQ_NR];
        for (size_t r = 0; r < KQ_NR; r++) {
            sumi[r] = _mm256_setzero_si256();
        }
        int is = 0;
        for (size_t j = 0; j < 2; j++) { /* QK_K/128 = 2 */
            __m256i q[4];
            blk_unpack(f, b, j, q);
            __m256i s[4];
            for (int k = 0; k < 4; k++) {
                s[k] = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, scale_shuffle(is + k)));
            }
            is += 4;

            for (size_t r = 0; r < KQ_NR; r++) {
                const int8_t *q8 = y[r * n_super + i].qs + j * 128;
                __m256i       p[4];
                for (int k = 0; k < 4; k++) {
                    p[k] = _mm256_madd_epi16(
                            s[k],
                            _mm256_maddubs_epi16(
                                    q[k], _mm256_loadu_si256((const __m256i *) (q8 + 32 * k))));
                }
                sumi[r] = _mm256_add_epi32(sumi[r], _mm256_add_epi32(p[0], p[1]));
                sumi[r] = _mm256_add_epi32(sumi[r], _mm256_add_epi32(p[2], p[3]));
            }
        }
        for (size_t r = 0; r < KQ_NR; r++) {
            const struct q8k_act *a      = &y[r * n_super + i];
            const __m256i         q8sums = _mm256_loadu_si256((const __m256i *) a->bsums);
            const __m256i         sub =
                    _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales16), kfmt_offset_shift(f));
            const float d = a->d * dw;
            acc[r]        = _mm256_fmadd_ps(_mm256_broadcast_ss(&d),
                                            _mm256_cvtepi32_ps(_mm256_sub_epi32(sumi[r], sub)),
                                            acc[r]);
        }
    }
    for (size_t r = 0; r < KQ_NR; r++) {
        out[r] = hsum_ps_hadd(acc[r]);
    }
}

static void
dot_rows_q6k(size_t n_super, const uint8_t *w, const struct q8k_act *y, float out[static KQ_NR]) {
    dot_q8k_rows(KF_Q6K, n_super, w, y, out);
}
static void
dot_rows_q3k(size_t n_super, const uint8_t *w, const struct q8k_act *y, float out[static KQ_NR]) {
    dot_q8k_rows(KF_Q3K, n_super, w, y, out);
}

size_t q6k_gemm_scratch_bytes(size_t M, size_t K) {
    return M * (K / 256) * sizeof(struct q8k_act);
}

size_t q3k_gemm_scratch_bytes(size_t M, size_t K) {
    return q6k_gemm_scratch_bytes(M, K);
}

/* GEMM: activation rows [i0, i1) to q8_K. */
static void gemm_quant_rows(void *ctx, size_t i0, size_t i1) {
    const struct kq_call c = *(const struct kq_call *) ctx;
    for (size_t i = i0; i < i1; i++) {
        quantize_q8k_act(c.n_super, c.x + i * c.K, c.a + i * c.n_super);
    }
}

/* GEMM: output rows [r0, r1), KQ_NR activation rows at a time. */
[[gnu::always_inline]] static inline void
gemm_rows(enum kfmt f, const struct kq_call *pc, size_t r0, size_t r1) {
    const struct kq_call  c       = *pc;
    const size_t          M       = c.M;
    const size_t          N       = c.N;
    const size_t          n_super = c.n_super;
    const size_t          m_til   = M - M % KQ_NR;
    const struct q8k_act *a       = c.a;
    float                *y       = c.y;
    for (size_t r = r0; r < r1; r++) {
        const uint8_t *wr = c.raw + r * c.row_bytes;
        float          out[KQ_NR];
        for (size_t i = 0; i < m_til; i += KQ_NR) {
            if (f == KF_Q6K) {
                dot_rows_q6k(n_super, wr, a + i * n_super, out);
            } else {
                dot_rows_q3k(n_super, wr, a + i * n_super, out);
            }
            for (size_t t = 0; t < KQ_NR; t++) {
                y[(i + t) * N + r] = out[t];
            }
        }
        for (size_t i = m_til; i < M; i++) {
            y[i * N + r] = dot_fmt(f, n_super, wr, a + i * n_super);
        }
    }
}

static void gemm_rows_q6k(void *ctx, size_t r0, size_t r1) {
    gemm_rows(KF_Q6K, ctx, r0, r1);
}
static void gemm_rows_q3k(void *ctx, size_t r0, size_t r1) {
    gemm_rows(KF_Q3K, ctx, r0, r1);
}

/* The M rows quantized, then the GEMM, one geist_par_for each. */
static void gemm(enum kfmt      f,
                 size_t         M,
                 size_t         N,
                 size_t         K,
                 const float   *x,
                 const uint8_t *raw,
                 void          *scratch,
                 float         *y) {
    const size_t   n_super = K / 256;
    struct kq_call c       = {.M         = M,
                              .N         = N,
                              .K         = K,
                              .n_super   = n_super,
                              .row_bytes = n_super * kfmt_bytes(f),
                              .raw       = raw,
                              .x         = x,
                              .a         = scratch,
                              .y         = y};
    geist_par_for(M, gemm_quant_rows, &c);
    geist_par_for(N, f == KF_Q6K ? gemm_rows_q6k : gemm_rows_q3k, &c);
}

void q6k_gemm(size_t         M,
              size_t         N,
              size_t         K,
              const float   *x,
              const uint8_t *q6k_raw,
              void          *scratch,
              float         *y) {
    gemm(KF_Q6K, M, N, K, x, q6k_raw, scratch, y);
}

void q3k_gemm(size_t         M,
              size_t         N,
              size_t         K,
              const float   *x,
              const uint8_t *q3k_raw,
              void          *scratch,
              float         *y) {
    gemm(KF_Q3K, M, N, K, x, q3k_raw, scratch, y);
}

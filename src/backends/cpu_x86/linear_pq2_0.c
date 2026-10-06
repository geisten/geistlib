/*
 * src/backends/cpu_x86/linear_pq2_0.c — cpu_x86 PQ2_0 linear: a W2 x A8 GEMV
 * and GEMM (AVX2).
 *
 * Layer: BACKEND (cpu_x86).
 *
 * PQ2_0 is PrismML's ternary format for Ternary-Bonsai: 128-element blocks
 * of [fp16 d][32 bytes of 2-bit codes], element j at byte j/4, bits
 * 2*(j%4), value (code - 1) * d (reference decoder: dequant_pq2_0_row,
 * src/formats/gguf/pq2_0.c).
 *
 * This kernel reads the GGUF bytes as they are — no repack; the 27B file is
 * 7.2 GB, a copy would not fit next to it on a 16 GB host — and stays in
 * integers inside a block. The recipe is cpu_neon's (kernels/pq2_0.c):
 *
 *   - x is quantized once per call to int8 with one absmax scale, the
 *     scheme TERNARY.md validated against fp32 activations on the real
 *     model (the Hadamard rotation keeps the outliers down), with the same
 *     rounding as cpu_neon, so both backends see the same int8 values.
 *     They are stored in the order the codes are packed: shift level l of
 *     code byte m is element 4m + l, so
 *
 *         xq[b*128 + 32*l + m] = q(x[b*128 + 4*m + l])
 *
 *     and each shift level of a 32-byte code load meets 32 contiguous
 *     activation bytes;
 *   - the raw codes (0..3, biased by +1) are maddubs' unsigned operand.
 *     |xq| <= 127, so a pair is at most 2*3*127 and the four levels' sum at
 *     most 3048: the int16 sums never saturate and the block's int32 dot is
 *     exact;
 *   - the +1 bias leaves through the block's activation sum, precomputed
 *     once per call into lane 0 of an 8-lane vector per block.
 *
 * Decode streams every weight byte once per token and the dot keeps up with
 * DRAM, so the hardware prefetchers limit it: a software prefetch
 * PREFETCH_BYTES ahead more than doubles the read rate on a 4-vCPU Xeon
 * (Sapphire Rapids class), to the host's read bandwidth. Each thread's rows
 * are contiguous (schedule(static)), so the prefetch address runs on across
 * row boundaries. A prefetch never faults, past the end of the weight
 * included.
 *
 * Prefill (M>1) is the same arithmetic as a GEMM. Every token's row is
 * quantized with its own absmax scale, in the same code order, the blocks
 * of all tokens side by side (XQ[b][t]). A group of GEMM_ROWS weight rows
 * then walks the blocks: each block's codes are extracted once and dotted
 * against every token, into per-(row, token) accumulators that stay in L1.
 * The four maddubs per block and token are the floor there, so the loop is
 * compute-bound, not bandwidth-bound, and needs no prefetch.
 *
 * AVX2 is the backend's x86-64-v3 baseline, so this runs on every host
 * cpu_x86 does. Where the host has AMX-INT8 and Linux grants the tile data
 * (cpu_x86_linear_pq2_0_amx_usable, decided at bind), the GEMM runs on the
 * tiles instead: the same int8 activations, repacked per block, against
 * 16-row tiles of code - 1 (kernel_pq2_0_amx.c).
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "linear_pq2_0.h"

#include "backend_state.h"
#include "kernel_pq2_0_amx.h"
#include "linear_util.h"
#include "kernel_w4a8.h" /* w4a8_dispatcher_tier: the ISA gate, GEIST_FORCE_ISA-clamped */

#include "checked.h"
#include "hw_probe.h"
#include "linear_ref.h"
#include "quant.h"

#include <geist_backend.h>
#include <geist_types.h>

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

constexpr size_t QK = PQ2_0_BLOCK_ELEMS; /* 128 */
constexpr size_t BB = PQ2_0_BLOCK_BYTES; /* fp16 d, then 32 code bytes */
static_assert(PQ2_0_BLOCK_ELEMS == 128 && PQ2_0_BLOCK_BYTES == 34,
              "a block's codes are one ymm load: four 2-bit levels of 32 elements");

/* Bias vectors: 8 int32 per block, lane 0 = minus the block's xq sum. */
constexpr size_t BIAS_LANES = 8;

/* 4-8 KB measured best on the host above; 2 KB was 15-20 % slower, 32 KB
 * about 10 %. */
constexpr size_t PREFETCH_BYTES = 4096;

/* Weight rows per step of the prefill GEMM. A group reads each block's
 * activations (m x 128 bytes) once for all its rows, and its accumulators
 * (GEMM_ROWS x m x 32 bytes, 16 KB at m = 128) stay in L1. */
constexpr size_t GEMM_ROWS = 4;

/* Within a 16-byte lane of 4 elements x 4 levels: the bytes grouped by
 * level (the same for both lanes). */
alignas(16) static const int8_t BY_LEVEL[16] = {
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};

/* max|x| over n floats (n a multiple of 8), floored at 1e-5 as cpu_neon's
 * is. */
static float act_absmax(size_t n, const float *x) {
    const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    __m256       mx   = _mm256_set1_ps(1e-5f);
    for (size_t i = 0; i < n; i += 8) {
        mx = _mm256_max_ps(mx, _mm256_and_ps(_mm256_loadu_ps(x + i), absm));
    }
    __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(mx), _mm256_extractf128_ps(mx, 1));
    m4        = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    m4        = _mm_max_ss(m4, _mm_movehdup_ps(m4));
    return _mm_cvtss_f32(m4);
}

/* One 128-element block of x to int8 in code order, xq[32*l + m] =
 * q(x[4*m + l]), with cpu_neon's rounding: x * scale rounded half away from
 * zero, so |q| <= 127 when scale = 127 / max|x|. Returns the sum of the 128
 * int8 values. */
static inline int32_t quantize_block(const float *x, __m256 scale, int8_t *xq) {
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 sign = _mm256_castsi256_ps(_mm256_set1_epi32((int32_t) 0x80000000u));
    /* The packs interleave 128-bit lanes; this restores element order. */
    const __m256i order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    const __m256i by_level =
            _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *) BY_LEVEL));
    /* r[k]: elements 32k .. 32k+31, each 16-byte lane holding its 4 code
     * bytes' elements grouped by level (dword l). */
    __m256i r[4];
    for (size_t k = 0; k < 4; k++) {
        __m256i q[4];
        for (size_t t = 0; t < 4; t++) {
            const __m256 v = _mm256_mul_ps(_mm256_loadu_ps(x + 32 * k + 8 * t), scale);
            q[t]           = _mm256_cvttps_epi32(
                    _mm256_add_ps(v, _mm256_or_ps(_mm256_and_ps(v, sign), half)));
        }
        const __m256i p =
                _mm256_packs_epi16(_mm256_packs_epi32(q[0], q[1]), _mm256_packs_epi32(q[2], q[3]));
        r[k] = _mm256_shuffle_epi8(_mm256_permutevar8x32_epi32(p, order), by_level);
    }
    /* 4 x 8 dword transpose: level l's dwords from every lane, in code byte
     * order. */
    const __m256i t01l  = _mm256_unpacklo_epi32(r[0], r[1]);
    const __m256i t01h  = _mm256_unpackhi_epi32(r[0], r[1]);
    const __m256i t23l  = _mm256_unpacklo_epi32(r[2], r[3]);
    const __m256i t23h  = _mm256_unpackhi_epi32(r[2], r[3]);
    const __m256i lv[4] = {_mm256_unpacklo_epi64(t01l, t23l),
                           _mm256_unpackhi_epi64(t01l, t23l),
                           _mm256_unpacklo_epi64(t01h, t23h),
                           _mm256_unpackhi_epi64(t01h, t23h)};
    __m256i       s16   = _mm256_setzero_si256();
    for (size_t l = 0; l < 4; l++) {
        const __m256i v = _mm256_permutevar8x32_epi32(lv[l], order);
        _mm256_storeu_si256((__m256i *) (xq + 32 * l), v);
        s16 = _mm256_add_epi16(s16, _mm256_maddubs_epi16(_mm256_set1_epi8(1), v));
    }
    const __m256i s32 = _mm256_madd_epi16(s16, _mm256_set1_epi16(1));
    __m128i       s4 = _mm_add_epi32(_mm256_castsi256_si128(s32), _mm256_extracti128_si256(s32, 1));
    s4               = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4E));
    s4               = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0xB1));
    return _mm_cvtsi128_si32(s4);
}

/* x (nb blocks) to int8 in code order, plus the bias vectors. Returns the
 * dequantization factor of the activation, max|x| / 127. */
static float quantize_acts(size_t nb, const float *x, int8_t *xq, int32_t *bias) {
    const float  max_abs = act_absmax(nb * QK, x);
    const __m256 scale   = _mm256_set1_ps(127.0f / max_abs);
    for (size_t b = 0; b < nb; b++) {
        const int32_t sum = quantize_block(x + b * QK, scale, xq + b * QK);
        _mm256_store_si256((__m256i *) (bias + b * BIAS_LANES),
                           _mm256_setr_epi32(-sum, 0, 0, 0, 0, 0, 0, 0));
    }
    return max_abs / 127.0f;
}

/* One token's row x (nb blocks) to int8 in code order, in the GEMM layout:
 * block b at xq + b * m * QK, minus its sum at neg_sum[b * m]. Returns the
 * row's dequantization factor. */
static float quantize_token(size_t m, size_t nb, const float *x, int8_t *xq, int32_t *neg_sum) {
    const float  max_abs = act_absmax(nb * QK, x);
    const __m256 scale   = _mm256_set1_ps(127.0f / max_abs);
    for (size_t b = 0; b < nb; b++) {
        neg_sum[b * m] = -quantize_block(x + b * QK, scale, xq + b * m * QK);
    }
    return max_abs / 127.0f;
}

/* One block: sum over its 128 elements of code * xq, minus the xq sum, as
 * 8 int32 lanes. */
static inline __m256i block_dot(const uint8_t *qs, const int8_t *xb, const int32_t *bias) {
    const __m256i m3  = _mm256_set1_epi8(3);
    const __m256i v   = _mm256_loadu_si256((const __m256i *) qs);
    const __m256i c0  = _mm256_and_si256(v, m3);
    const __m256i c1  = _mm256_and_si256(_mm256_srli_epi16(v, 2), m3);
    const __m256i c2  = _mm256_and_si256(_mm256_srli_epi16(v, 4), m3);
    const __m256i c3  = _mm256_and_si256(_mm256_srli_epi16(v, 6), m3);
    const __m256i p01 = _mm256_add_epi16(
            _mm256_maddubs_epi16(c0, _mm256_loadu_si256((const __m256i *) xb)),
            _mm256_maddubs_epi16(c1, _mm256_loadu_si256((const __m256i *) (xb + 32))));
    const __m256i p23 = _mm256_add_epi16(
            _mm256_maddubs_epi16(c2, _mm256_loadu_si256((const __m256i *) (xb + 64))),
            _mm256_maddubs_epi16(c3, _mm256_loadu_si256((const __m256i *) (xb + 96))));
    return _mm256_add_epi32(_mm256_madd_epi16(_mm256_add_epi16(p01, p23), _mm256_set1_epi16(1)),
                            _mm256_load_si256((const __m256i *) bias));
}

/* The block's scale d in all 8 lanes. */
static inline __m256 block_scale(const uint8_t *blk) {
    int16_t d;
    memcpy(&d, blk, sizeof d);
    return _mm256_cvtph_ps(_mm_set1_epi16(d));
}

/* One weight row of nb blocks against the quantized activation. Two
 * accumulators (even / odd block) so the FMA chain is not latency-bound;
 * fixed order, so a row's result does not depend on the thread count. */
static inline float row_dot(size_t nb, const uint8_t *w, const int8_t *xq, const int32_t *bias) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    size_t b    = 0;
    for (; b + 2 <= nb; b += 2) {
        const uint8_t *k0 = w + b * BB;
        const uint8_t *k1 = k0 + BB;
        _mm_prefetch((const char *) (k0 + PREFETCH_BYTES), _MM_HINT_T0);
        _mm_prefetch((const char *) (k1 + PREFETCH_BYTES), _MM_HINT_T0);
        acc0 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(block_dot(k0 + 2, xq + b * QK, bias + b * BIAS_LANES)),
                block_scale(k0),
                acc0);
        acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(block_dot(
                                       k1 + 2, xq + (b + 1) * QK, bias + (b + 1) * BIAS_LANES)),
                               block_scale(k1),
                               acc1);
    }
    if (b < nb) {
        const uint8_t *k0 = w + b * BB;
        acc0              = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(block_dot(k0 + 2, xq + b * QK, bias + b * BIAS_LANES)),
                block_scale(k0),
                acc0);
    }
    return hsum_ps(_mm256_add_ps(acc0, acc1));
}

static void cpu_x86_linear_pq2_0_m1(const float               *x,
                                    const struct geist_weight *w,
                                    struct geist_backend      *be,
                                    float                     *y) {
    const size_t              n_in       = (size_t) w->n_in;
    const size_t              n_out      = (size_t) w->n_out;
    const size_t              nb         = n_in / QK;
    size_t                    bias_bytes = 0;
    struct cpu_x86_workspace *ws         = nullptr;
    if (be != nullptr && be->state != nullptr &&
        !ckd_mul(&bias_bytes, nb, BIAS_LANES * sizeof(int32_t))) {
        ws = cpu_x86_ws_acquire_mN((struct cpu_x86_state *) be->state, n_in, bias_bytes, 0, 0);
    }
    if (ws == nullptr) {
        geist_linear_ref(1, x, w, y); /* no scratch: the reference needs none */
        return;
    }
    int8_t        *xq   = ws->mN_acts;
    int32_t       *bias = ws->mN_sum_a;
    const float    inv  = quantize_acts(nb, x, xq, bias);
    const uint8_t *raw  = (const uint8_t *) w->raw;
    const size_t   rb   = nb * BB;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t j = 0; j < n_out; j++) {
        y[j] = row_dot(nb, raw + j * rb, xq, bias) * inv;
    }
}

/* codes . xq over one block, as 8 int32 lanes, minus the block's xq sum
 * (neg_sum, into lane 0). */
static inline __m256i
codes_dot(__m256i c0, __m256i c1, __m256i c2, __m256i c3, const int8_t *xt, int32_t neg_sum) {
    const __m256i p01 = _mm256_add_epi16(
            _mm256_maddubs_epi16(c0, _mm256_loadu_si256((const __m256i *) xt)),
            _mm256_maddubs_epi16(c1, _mm256_loadu_si256((const __m256i *) (xt + 32))));
    const __m256i p23 = _mm256_add_epi16(
            _mm256_maddubs_epi16(c2, _mm256_loadu_si256((const __m256i *) (xt + 64))),
            _mm256_maddubs_epi16(c3, _mm256_loadu_si256((const __m256i *) (xt + 96))));
    return _mm256_add_epi32(_mm256_madd_epi16(_mm256_add_epi16(p01, p23), _mm256_set1_epi16(1)),
                            _mm256_zextsi128_si256(_mm_cvtsi32_si128(neg_sum)));
}

/* One weight block against every token's activation block: acc[t] (8 fp32
 * lanes) += d * (code . xq_t - sum xq_t). The codes are extracted once; the
 * four maddubs per token are what the loop costs. */
static inline void
block_tokens(size_t m, const uint8_t *blk, const int8_t *xb, const int32_t *neg_sum, float *acc) {
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i v  = _mm256_loadu_si256((const __m256i *) (blk + 2));
    const __m256i c0 = _mm256_and_si256(v, m3);
    const __m256i c1 = _mm256_and_si256(_mm256_srli_epi16(v, 2), m3);
    const __m256i c2 = _mm256_and_si256(_mm256_srli_epi16(v, 4), m3);
    const __m256i c3 = _mm256_and_si256(_mm256_srli_epi16(v, 6), m3);
    const __m256  d  = block_scale(blk);
    size_t        t  = 0;
    for (; t + 4 <= m; t += 4) {
        for (size_t k = 0; k < 4; k++) {
            const __m256i dot = codes_dot(c0, c1, c2, c3, xb + (t + k) * QK, neg_sum[t + k]);
            float        *a   = acc + (t + k) * 8;
            _mm256_store_ps(a, _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot), d, _mm256_load_ps(a)));
        }
    }
    for (; t < m; t++) {
        const __m256i dot = codes_dot(c0, c1, c2, c3, xb + t * QK, neg_sum[t]);
        float        *a   = acc + t * 8;
        _mm256_store_ps(a, _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot), d, _mm256_load_ps(a)));
    }
}

static inline float hsum8(const float *v) {
    return hsum_ps(_mm256_load_ps(v));
}

/* M>1: all m rows quantized (in parallel, one token per iteration), then the
 * GEMM by groups of GEMM_ROWS weight rows, each thread with its own
 * accumulators. The workspace holds the int8 activations (mN_acts), the
 * block sums (mN_sum_a), the per-token factors (mN_scale) and the
 * accumulators (mN_aux). */
static void cpu_x86_linear_pq2_0_mN(size_t                     m,
                                    const float               *x,
                                    const struct geist_weight *w,
                                    struct geist_backend      *be,
                                    float                     *y) {
    const size_t              n_in      = (size_t) w->n_in;
    const size_t              n_out     = (size_t) w->n_out;
    const size_t              nb        = n_in / QK;
    const size_t              acc_elems = GEMM_ROWS * m * 8;
    size_t                    acts = 0, sums = 0, scales = 0, accs = 0;
    struct cpu_x86_workspace *ws = nullptr;
    if (be != nullptr && be->state != nullptr && !ckd_mul(&acts, m, n_in) &&
        !ckd_mul(&sums, m, nb * sizeof(int32_t)) && !ckd_mul(&scales, m, sizeof(float)) &&
        !ckd_mul(&accs, team_max(), acc_elems * sizeof(float))) {
        ws = cpu_x86_ws_acquire_mN((struct cpu_x86_state *) be->state, acts, sums, scales, accs);
    }
    if (ws == nullptr) {
        geist_linear_ref(m, x, w, y); /* no scratch: the reference needs none */
        return;
    }
    int8_t        *xq      = ws->mN_acts;
    int32_t       *neg_sum = ws->mN_sum_a;
    float         *inv     = ws->mN_scale;
    float         *accs_ws = (float *) (void *) ws->mN_aux;
    const uint8_t *raw     = (const uint8_t *) w->raw;
    const size_t   rb      = nb * BB;
    const size_t   groups  = (n_out + GEMM_ROWS - 1) / GEMM_ROWS;

#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (size_t t = 0; t < m; t++) {
            inv[t] = quantize_token(m, nb, x + t * n_in, xq + t * QK, neg_sum + t);
        }
        float *acc = accs_ws + team_id() * acc_elems;
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (size_t g = 0; g < groups; g++) {
            const size_t r0 = g * GEMM_ROWS;
            const size_t nr = n_out - r0 < GEMM_ROWS ? n_out - r0 : GEMM_ROWS;
            memset(acc, 0, nr * m * 8 * sizeof(float));
            for (size_t b = 0; b < nb; b++) {
                for (size_t r = 0; r < nr; r++) {
                    block_tokens(m,
                                 raw + (r0 + r) * rb + b * BB,
                                 xq + b * m * QK,
                                 neg_sum + b * m,
                                 acc + r * m * 8);
                }
            }
            for (size_t r = 0; r < nr; r++) {
                for (size_t t = 0; t < m; t++) {
                    y[t * n_out + r0 + r] = hsum8(acc + (r * m + t) * 8) * inv[t];
                }
            }
        }
    }
}

#ifndef GEIST_NO_AMX /* the assembler knows AMX; mk/backend-cpu_x86.mk */
/* Token tiles per pass of the AMX GEMM. Each pass streams the weights once
 * and keeps the packed activations of its tokens (nb * tiles * 2 KB, 2.2 MB
 * for 8 tiles at n_in = 17408) near L2. */
constexpr size_t AMX_PASS_TILES = 8;

/* Per-thread scratch starts on its own page and a page apart from the
 * next: the L2 prefetchers run on into the neighbouring page, and with the
 * neighbour storing there every step the lines bounce between the cores
 * (page alignment without the gap did not help). */
constexpr size_t AMX_PAGE = 4096;

/* M>1 on AMX: the activations quantized as for the AVX2 GEMM (XQ[b][t],
 * inv[t]), then per pass of up to AMX_PASS_TILES token tiles the pass's
 * tokens packed into B tiles (block-parallel) and the GEMM over 16-row
 * groups, one contiguous range per thread. mN_aux holds the packed tiles
 * and the per-thread scratch. */
static void cpu_x86_linear_pq2_0_mN_amx(size_t                     m,
                                        const float               *x,
                                        const struct geist_weight *w,
                                        struct geist_backend      *be,
                                        float                     *y) {
    const size_t n_in    = (size_t) w->n_in;
    const size_t n_out   = (size_t) w->n_out;
    const size_t nb      = n_in / QK;
    const size_t tiles   = (m + PQ2_0_AMX_TOKENS - 1) / PQ2_0_AMX_TOKENS;
    const size_t passes  = (tiles + AMX_PASS_TILES - 1) / AMX_PASS_TILES;
    const size_t pass_tn = (tiles + passes - 1) / passes; /* balanced */
    const size_t pass_m  = pass_tn * PQ2_0_AMX_TOKENS;
    /* scratch per thread, rounded to pages, plus the gap page */
    const size_t scratch = PQ2_0_AMX_SCRATCH_FIXED + pass_tn * PQ2_0_AMX_SCRATCH_PER_TILE;
    const size_t stride  = (scratch + AMX_PAGE - 1) / AMX_PAGE * AMX_PAGE + AMX_PAGE;
    size_t       acts = 0, sums = 0, scales = 0, packed = 0, per_team = 0, aux = 0;
    struct cpu_x86_workspace *ws = nullptr;
    if (be != nullptr && be->state != nullptr && !ckd_mul(&acts, m, n_in) &&
        !ckd_mul(&sums, m, nb * sizeof(int32_t)) && !ckd_mul(&scales, m, sizeof(float)) &&
        !ckd_mul(&packed, nb, pass_tn * PQ2_0_AMX_TILE_BYTES) &&
        !ckd_mul(&per_team, team_max(), stride) &&
        /* the packed tiles, a page to align the scratch, a gap page */
        !ckd_add(&aux, packed, 2 * AMX_PAGE) && !ckd_add(&aux, aux, per_team)) {
        ws = cpu_x86_ws_acquire_mN((struct cpu_x86_state *) be->state, acts, sums, scales, aux);
    }
    if (ws == nullptr) {
        cpu_x86_linear_pq2_0_mN(m, x, w, be, y); /* smaller scratch, or the reference */
        return;
    }
    int8_t         *xq      = ws->mN_acts;
    int32_t        *neg_sum = ws->mN_sum_a; /* written by quantize_token, unused here */
    float          *inv     = ws->mN_scale;
    int32_t        *bt      = (int32_t *) (void *) ws->mN_aux;
    const uintptr_t end     = (uintptr_t) (ws->mN_aux + packed) + AMX_PAGE;
    uint8_t        *scr0 =
            ws->mN_aux + ((end + AMX_PAGE - 1) / AMX_PAGE * AMX_PAGE - (uintptr_t) ws->mN_aux);
    const uint8_t   *raw        = (const uint8_t *) w->raw;
    const size_t     groups     = (n_out + PQ2_0_AMX_ROWS - 1) / PQ2_0_AMX_ROWS;
    constexpr size_t TILE_WORDS = PQ2_0_AMX_TILE_BYTES / sizeof(int32_t);

#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (size_t t = 0; t < m; t++) {
            inv[t] = quantize_token(m, nb, x + t * n_in, xq + t * QK, neg_sum + t);
        }
        const size_t tid = team_id();
        const size_t nt  = team_size();
        uint8_t     *own = scr0 + tid * stride;
        const size_t g0  = groups * tid / nt;
        const size_t g1  = groups * (tid + 1) / nt;
        for (size_t t0 = 0; t0 < m; t0 += pass_m) {
            const size_t mc = m - t0 < pass_m ? m - t0 : pass_m;
            const size_t tn = (mc + PQ2_0_AMX_TOKENS - 1) / PQ2_0_AMX_TOKENS;
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (size_t b = 0; b < nb; b++) {
                pq2_0_amx_pack(tn, mc, xq + (b * m + t0) * QK, bt + b * tn * TILE_WORDS);
            }
            pq2_0_amx_gemm(nb, n_out, mc, tn, g0, g1, raw, bt, inv + t0, own, y + t0 * n_out);
#if defined(_OPENMP)
#pragma omp barrier /* the next pass repacks bt */
#endif
        }
    }
}
#endif /* GEIST_NO_AMX */

/* Whether this host may run kernel_pq2_0_amx.c: the dispatcher tier (which
 * honours GEIST_FORCE_ISA: anything below avx512_vnni keeps the AVX2 GEMM),
 * AMX-INT8 and the AVX-512 subsets that TU is compiled for, and Linux's
 * permission for the 8 KB of tile data per thread (arch_prctl
 * ARCH_REQ_XCOMP_PERM, Linux 5.16+; process-wide and idempotent, so asking
 * once per bind is harmless). Decided here, outside that TU — see
 * mk/backend-cpu_x86.mk. */
bool cpu_x86_linear_pq2_0_amx_usable(void) {
#if defined(__linux__) && defined(SYS_arch_prctl) && !defined(GEIST_NO_AMX)
    struct geist_hw_probe hw;
    geist_hw_probe_isa(&hw);
    constexpr long ARCH_REQ_XCOMP_PERM = 0x1023;
    constexpr long XFEATURE_XTILEDATA  = 18;
    return w4a8_dispatcher_tier() >= W4A8_ISA_AVX512_VNNI && hw.has_amx_int8 &&
           __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
           syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) == 0;
#else
    return false;
#endif
}

void cpu_x86_linear_pq2_0_bind(struct geist_weight *w) {
    w->linear_m1 = cpu_x86_linear_pq2_0_m1;
#ifndef GEIST_NO_AMX
    w->linear_mN = cpu_x86_linear_pq2_0_amx_usable() ? cpu_x86_linear_pq2_0_mN_amx
                                                     : cpu_x86_linear_pq2_0_mN;
#else
    w->linear_mN = cpu_x86_linear_pq2_0_mN;
#endif
}

/*
 * src/backends/cpu_x86/linear_q8_0.c — cpu_x86 native Q8_0 linear (AVX2).
 *
 * Layer: BACKEND (cpu_x86).
 *
 * Q8_0 is the format of the small reference models (Qwen3-0.6B, Qwen3.5-
 * 0.8B, SmolLM2-360M) and had no x86 kernel: it ran the generic dequantize-
 * and-dot path (linear_generic.c), which spends its time turning int8 into
 * fp32. This one never leaves int8: the weights are read straight from the
 * GGUF bytes (no repack, no aux memory), the activation row is quantized to
 * Q8_0 blocks once per call (d = amax/127 per 32 elements, the same scheme
 * the reference engines use for Q8_0 x Q8_0), and each block's 32 products
 * are one maddubs + madd:
 *
 *   maddubs(|w|, sign(x, w))  — u8 x s8 pairs into s16. |w| <= 128 and
 *                               |x| <= 127, so a pair is at most 32512:
 *                               never saturates, the int32 block sum is exact.
 *
 * then scaled by d_w * d_x in fp32. Rows split across OpenMP threads; M>1
 * quantizes all m activation rows once (in parallel) and keeps each weight
 * row in L1 while it is dotted against every one of them.
 *
 * Q4_0 (#410) runs the same kernels: a block is the same 32 elements under
 * one fp16 scale, with the codes as nibbles (element i in the low nibble of
 * byte i, element 16 + i in the high one, value q - 8). Unpacked into one
 * ymm of int8 in [-8, 7], the block goes through the same maddubs + madd —
 * again exact — so Q4_0 decode reads 18 bytes per block instead of 34 and
 * never builds a Q8_0 copy. M>1 stays on the AVX2 kernel for Q4_0; the
 * VNNI tiles read Q8_0 blocks.
 *
 * AVX2 is the backend's x86-64-v3 baseline, so this runs on every host
 * cpu_x86 does (the AVX2-only ones included). On AVX-512 VNNI hosts M>1
 * binds kernel_q8_0_avx512_vnni.c's register tiles instead (same bits,
 * about twice the throughput); M=1 is bound by memory bandwidth and keeps
 * the AVX2 GEMV, which VNNI does not speed up.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "linear_q8_0.h"

#include "backend_state.h"
#include "kernel_q8_0_avx512_vnni.h"
#include "kernel_w4a8.h" /* w4a8_dispatcher_tier: the ISA gate, GEIST_FORCE_ISA-clamped */

#include "checked.h"
#include "linear_ref.h"
#include "quant.h"
#include "quant_blocks.h"

#include <geist_backend.h>
#include <geist_types.h>

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

constexpr size_t QK = Q8_0_BLOCK_ELEMS; /* 32 */
static_assert(Q8_0_BLOCK_ELEMS == 32, "the kernels below load one Q8_0 block per ymm");

/* One activation row to Q8_0: per block d = amax / 127 and q = x / d
 * rounded to nearest-even, so |q| <= 127 (the bound the sign trick needs).
 * Same packing as the reference engines' quantize_row_q8_0. */
static void quantize_row_q8_0(size_t nb, const float *x, int8_t *qx, float *dx) {
    const __m256  abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256i perm     = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (size_t b = 0; b < nb; b++) {
        const float *xb = x + b * QK;
        const __m256 v0 = _mm256_loadu_ps(xb);
        const __m256 v1 = _mm256_loadu_ps(xb + 8);
        const __m256 v2 = _mm256_loadu_ps(xb + 16);
        const __m256 v3 = _mm256_loadu_ps(xb + 24);
        __m256       m  = _mm256_max_ps(
                _mm256_max_ps(_mm256_and_ps(v0, abs_mask), _mm256_and_ps(v1, abs_mask)),
                _mm256_max_ps(_mm256_and_ps(v2, abs_mask), _mm256_and_ps(v3, abs_mask)));
        __m128 m4        = _mm_max_ps(_mm256_extractf128_ps(m, 1), _mm256_castps256_ps128(m));
        m4               = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        m4               = _mm_max_ss(m4, _mm_movehdup_ps(m4));
        const float amax = _mm_cvtss_f32(m4);

        dx[b]              = amax / 127.0f;
        const __m256 scale = _mm256_set1_ps(amax > 0.0f ? 127.0f / amax : 0.0f);
        __m256i      i0    = _mm256_cvtps_epi32(_mm256_round_ps(
                _mm256_mul_ps(v0, scale), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        __m256i      i1    = _mm256_cvtps_epi32(_mm256_round_ps(
                _mm256_mul_ps(v1, scale), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        __m256i      i2    = _mm256_cvtps_epi32(_mm256_round_ps(
                _mm256_mul_ps(v2, scale), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        __m256i      i3    = _mm256_cvtps_epi32(_mm256_round_ps(
                _mm256_mul_ps(v3, scale), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        /* The packs interleave 128-bit lanes; the permute restores order. */
        i0 = _mm256_packs_epi32(i0, i1);
        i2 = _mm256_packs_epi32(i2, i3);
        i0 = _mm256_packs_epi16(i0, i2);
        _mm256_storeu_si256((__m256i *) (qx + b * QK), _mm256_permutevar8x32_epi32(i0, perm));
    }
}

/* Exact int32 sum of one block's 32 products, as 8 fp32 lanes. */
static inline __m256 block_products_w(__m256i w, const int8_t *xq) {
    const __m256i x   = _mm256_loadu_si256((const __m256i *) xq);
    const __m256i p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(x, w));
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(p16, _mm256_set1_epi16(1)));
}

static inline __m256 block_products(const int8_t *wq, const int8_t *xq) {
    return block_products_w(_mm256_loadu_si256((const __m256i *) wq), xq);
}

/* One weight row against one quantized activation row. Two accumulators
 * (even / odd block) so the FMA chain is not latency-bound; fixed order. */
static inline float
dot_row(size_t nb, const struct block_q8_0_t *w, const int8_t *qx, const float *dx) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    size_t b    = 0;
    for (; b + 2 <= nb; b += 2) {
        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(_cvtsh_ss(w[b].d) * dx[b]),
                               block_products(w[b].qs, qx + b * QK),
                               acc0);
        acc1 = _mm256_fmadd_ps(_mm256_set1_ps(_cvtsh_ss(w[b + 1].d) * dx[b + 1]),
                               block_products(w[b + 1].qs, qx + (b + 1) * QK),
                               acc1);
    }
    if (b < nb) {
        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(_cvtsh_ss(w[b].d) * dx[b]),
                               block_products(w[b].qs, qx + b * QK),
                               acc0);
    }
    const __m256 s  = _mm256_add_ps(acc0, acc1);
    __m128       s4 = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
    s4              = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    s4              = _mm_add_ss(s4, _mm_movehdup_ps(s4));
    return _mm_cvtss_f32(s4);
}

/* Q4_0's 16 code bytes as 32 int8 in element order: low nibbles are
 * elements 0..15, high nibbles 16..31, both minus 8. */
static inline __m256i q4_0_codes(const uint8_t *qs) {
    const __m128i b  = _mm_loadu_si128((const __m128i *) qs);
    const __m128i lo = _mm_and_si128(b, _mm_set1_epi8(0x0F));
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), _mm_set1_epi8(0x0F));
    return _mm256_sub_epi8(_mm256_set_m128i(hi, lo), _mm256_set1_epi8(8));
}

static inline float q4_0_d(const uint8_t *blk) {
    uint16_t d;
    memcpy(&d, blk, sizeof d);
    return _cvtsh_ss(d);
}

/* dot_row for one Q4_0 weight row (Q4_0_BLOCK_BYTES per block). */
static inline float dot_row_q4_0(size_t nb, const uint8_t *w, const int8_t *qx, const float *dx) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    size_t b    = 0;
    for (; b + 2 <= nb; b += 2) {
        const uint8_t *b0 = w + b * Q4_0_BLOCK_BYTES;
        const uint8_t *b1 = b0 + Q4_0_BLOCK_BYTES;
        acc0              = _mm256_fmadd_ps(_mm256_set1_ps(q4_0_d(b0) * dx[b]),
                                            block_products_w(q4_0_codes(b0 + 2), qx + b * QK),
                                            acc0);
        acc1              = _mm256_fmadd_ps(_mm256_set1_ps(q4_0_d(b1) * dx[b + 1]),
                                            block_products_w(q4_0_codes(b1 + 2), qx + (b + 1) * QK),
                                            acc1);
    }
    if (b < nb) {
        const uint8_t *b0 = w + b * Q4_0_BLOCK_BYTES;
        acc0              = _mm256_fmadd_ps(_mm256_set1_ps(q4_0_d(b0) * dx[b]),
                                            block_products_w(q4_0_codes(b0 + 2), qx + b * QK),
                                            acc0);
    }
    const __m256 s  = _mm256_add_ps(acc0, acc1);
    __m128       s4 = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
    s4              = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    s4              = _mm_add_ss(s4, _mm_movehdup_ps(s4));
    return _mm_cvtss_f32(s4);
}

/* One weight row of either format; `q4` is a constant at every call site,
 * so the inlined drivers below specialize per format. */
[[gnu::always_inline]] static inline float
dot_any(bool q4, size_t nb, const uint8_t *row, const int8_t *qx, const float *dx) {
    return q4 ? dot_row_q4_0(nb, row, qx, dx)
              : dot_row(nb, (const struct block_q8_0_t *) row, qx, dx);
}

/* The calling thread's workspace with room for m quantized activation rows
 * (int8 values + one fp32 scale per block), or nullptr. */
static struct cpu_x86_workspace *acquire_acts(struct geist_backend *be, size_t m, size_t n_in) {
    size_t acts_bytes = 0, scale_bytes = 0;
    if (be == nullptr || be->state == nullptr || ckd_mul(&acts_bytes, m, n_in) ||
        ckd_mul(&scale_bytes, m, n_in / QK) || ckd_mul(&scale_bytes, scale_bytes, sizeof(float))) {
        return nullptr;
    }
    return cpu_x86_ws_acquire_mN((struct cpu_x86_state *) be->state, acts_bytes, 0, scale_bytes, 0);
}

[[gnu::always_inline]] static inline void linear_m1(
        bool q4, const float *x, const struct geist_weight *w, struct geist_backend *be, float *y) {
    const size_t              n_in  = (size_t) w->n_in;
    const size_t              n_out = (size_t) w->n_out;
    const size_t              nb    = n_in / QK;
    struct cpu_x86_workspace *ws    = acquire_acts(be, 1, n_in);
    if (ws == nullptr) {
        geist_linear_ref(1, x, w, y); /* no scratch: the reference needs none */
        return;
    }
    int8_t *qx = ws->mN_acts;
    float  *dx = ws->mN_scale;
    quantize_row_q8_0(nb, x, qx, dx);

    const uint8_t *wb = (const uint8_t *) w->raw;
    const size_t   rb = nb * (q4 ? Q4_0_BLOCK_BYTES : Q8_0_BLOCK_BYTES);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t j = 0; j < n_out; j++) {
        y[j] = dot_any(q4, nb, wb + j * rb, qx, dx);
    }
}

static void cpu_x86_linear_q8_0_m1(const float               *x,
                                   const struct geist_weight *w,
                                   struct geist_backend      *be,
                                   float                     *y) {
    linear_m1(false, x, w, be, y);
}

static void cpu_x86_linear_q4_0_m1(const float               *x,
                                   const struct geist_weight *w,
                                   struct geist_backend      *be,
                                   float                     *y) {
    linear_m1(true, x, w, be, y);
}

[[gnu::always_inline]] static inline void linear_mN(bool                       q4,
                                                    size_t                     m,
                                                    const float               *x,
                                                    const struct geist_weight *w,
                                                    struct geist_backend      *be,
                                                    float                     *y) {
    const size_t              n_in  = (size_t) w->n_in;
    const size_t              n_out = (size_t) w->n_out;
    const size_t              nb    = n_in / QK;
    struct cpu_x86_workspace *ws    = acquire_acts(be, m, n_in);
    if (ws == nullptr) {
        geist_linear_ref(m, x, w, y);
        return;
    }
    int8_t        *qx = ws->mN_acts;
    float         *dx = ws->mN_scale;
    const uint8_t *wb = (const uint8_t *) w->raw;
    const size_t   rb = nb * (q4 ? Q4_0_BLOCK_BYTES : Q8_0_BLOCK_BYTES);

    /* One team: quantize the m rows, then the GEMM (implicit barrier
     * between the two worksharing loops). */
#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (size_t i = 0; i < m; i++) {
            quantize_row_q8_0(nb, x + i * n_in, qx + i * n_in, dx + i * nb);
        }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (size_t j = 0; j < n_out; j++) {
            const uint8_t *wr = wb + j * rb;
            for (size_t i = 0; i < m; i++) {
                y[i * n_out + j] = dot_any(q4, nb, wr, qx + i * n_in, dx + i * nb);
            }
        }
    }
}

static void cpu_x86_linear_q8_0_mN(size_t                     m,
                                   const float               *x,
                                   const struct geist_weight *w,
                                   struct geist_backend      *be,
                                   float                     *y) {
    linear_mN(false, m, x, w, be, y);
}

static void cpu_x86_linear_q4_0_mN(size_t                     m,
                                   const float               *x,
                                   const struct geist_weight *w,
                                   struct geist_backend      *be,
                                   float                     *y) {
    linear_mN(true, m, x, w, be, y);
}

/* M>1 on AVX-512 VNNI hosts: the same quantization, then 4-row x 4-token
 * register tiles from kernel_q8_0_avx512_vnni.c, one call per group of
 * Q8_0_VNNI_TILE_ROWS output rows. Same bits as cpu_x86_linear_q8_0_mN. */
static void cpu_x86_linear_q8_0_mN_vnni(size_t                     m,
                                        const float               *x,
                                        const struct geist_weight *w,
                                        struct geist_backend      *be,
                                        float                     *y) {
    const size_t              n_in  = (size_t) w->n_in;
    const size_t              n_out = (size_t) w->n_out;
    const size_t              nb    = n_in / QK;
    struct cpu_x86_workspace *ws    = acquire_acts(be, m, n_in);
    if (ws == nullptr) {
        geist_linear_ref(m, x, w, y);
        return;
    }
    int8_t                    *qx      = ws->mN_acts;
    float                     *dx      = ws->mN_scale;
    const struct block_q8_0_t *wb      = (const struct block_q8_0_t *) w->raw;
    const size_t               n_tiles = (n_out + Q8_0_VNNI_TILE_ROWS - 1) / Q8_0_VNNI_TILE_ROWS;

#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (size_t i = 0; i < m; i++) {
            quantize_row_q8_0(nb, x + i * n_in, qx + i * n_in, dx + i * nb);
        }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (size_t g = 0; g < n_tiles; g++) {
            const size_t j0   = g * Q8_0_VNNI_TILE_ROWS;
            const size_t rows = n_out - j0 < Q8_0_VNNI_TILE_ROWS ? n_out - j0 : Q8_0_VNNI_TILE_ROWS;
            q8_0_gemm_rows_avx512_vnni(m, nb, n_out, j0, rows, wb, qx, dx, y);
        }
    }
}

/* Whether this host may run kernel_q8_0_avx512_vnni.c: the dispatcher tier
 * (which honours GEIST_FORCE_ISA) and every AVX-512 subset that TU is
 * compiled for. Decided here, outside that TU — see mk/backend-cpu_x86.mk. */
static bool vnni_tiles_usable(void) {
    return w4a8_dispatcher_tier() >= W4A8_ISA_AVX512_VNNI && __builtin_cpu_supports("avx512f") &&
           __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512dq") &&
           __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512vnni");
}

bool cpu_x86_linear_q8_0_bind(struct geist_weight *w) {
    if (w == nullptr || w->dtype != GEIST_DTYPE_Q8_0 || w->n_in <= 0 ||
        (size_t) w->n_in % QK != 0) {
        return false;
    }
    w->linear_m1 = cpu_x86_linear_q8_0_m1;
    w->linear_mN = vnni_tiles_usable() ? cpu_x86_linear_q8_0_mN_vnni : cpu_x86_linear_q8_0_mN;
    return true;
}

bool cpu_x86_linear_q4_0_bind(struct geist_weight *w) {
    if (w == nullptr || w->dtype != GEIST_DTYPE_Q4_0 || w->n_in <= 0 ||
        (size_t) w->n_in % QK != 0) {
        return false;
    }
    w->linear_m1 = cpu_x86_linear_q4_0_m1;
    w->linear_mN = cpu_x86_linear_q4_0_mN;
    return true;
}

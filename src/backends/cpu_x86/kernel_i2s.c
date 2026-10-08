/*
 * src/backends/cpu_x86/kernel_i2s.c — I2_S ternary GEMV dispatcher + scalar.
 *
 * Layer: BACKEND (cpu_x86).
 *
 * Compiled at baseline -march=x86-64-v3, which carries the AVX2 GEMV for
 * hosts without VNNI; the AVX-512+VNNI variant lives in
 * kernel_i2s_avx512_vnni.c with -mavx512vnni. See kernel_i2s.h for the
 * ternary algebra.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "kernel_i2s.h"

#include "kernel_w4a8.h" /* w4a8_dispatcher_init — shared ISA selection */
#include "linear_util.h" /* hsum_epi32 */

#include <immintrin.h>

#include <math.h>
#include <stddef.h>
#include <stdatomic.h>
#include <stdint.h>

#include "heap.h"
#include "par.h"

#include <stdlib.h>

/* VNNI path (kernel_i2s_avx512_vnni.c). */
void i2s_gemv_m1_avx512_vnni(size_t        n_out,
                             size_t        n_in,
                             const int8_t *xq,
                             int32_t       sum_a,
                             const uint8_t w_raw[],
                             float         scale,
                             float         y[static n_out]);

void i2s_gemm_avx512_vnni(size_t         m,
                          size_t         n_out,
                          size_t         n_in,
                          const int8_t  *xq,    /* [m * n_in] natural-order int8 */
                          const int32_t *sum_a, /* [m] */
                          const float   *scale, /* [m] = tensor_scale * inv_act_scale */
                          const uint8_t  w_raw[],
                          int8_t         perm[], /* [m * n_in] caller scratch */
                          float          y[]);

/* x4 row-interleaved kernels (kernel_i2s_avx512_vnni.c). xq is natural-order
 * int8 [.. * n_in]; one activation load feeds 4 output rows. */
void i2s_x4_gemv_m1_avx512_vnni(size_t        n_out,
                                size_t        n_in,
                                const int8_t *xq,
                                int32_t       sum_a,
                                const uint8_t x4[],
                                float         scale,
                                float         y[static n_out]);

void i2s_x4_gemm_avx512_vnni(size_t         m,
                             size_t         n_out,
                             size_t         n_in,
                             const int8_t  *xq,
                             const int32_t *sum_a,
                             const float   *scale,
                             const uint8_t  x4[],
                             float          y[]);

void i2s_x4_gemv_pair_m1_avx512_vnni(size_t        n_in,
                                     const int8_t *xq,
                                     int32_t       sum_a,
                                     const uint8_t x4_0[],
                                     float         scale0,
                                     size_t        n_out0,
                                     float        *y0,
                                     const uint8_t x4_1[],
                                     float         scale1,
                                     size_t        n_out1,
                                     float        *y1);

/* --- Activation quant: per-row symmetric int8, scale = 127/max|x|. -------
 *
 * noinline: under -ffast-math (-freciprocal-math) gcc compiles the
 * `max_abs / 127.0f` differently per inline site (divide vs reciprocal
 * multiply), so the fused-pair path and the single-GEMV path returned
 * inv values 1 ULP apart, breaking the pair==m1 byte-identity contract
 * (test_i2s_gemv_unit case 5). One compiled instance = one rounding, for
 * every caller. Called once per GEMV; the call overhead is noise. */
__attribute__((noinline)) static float
quantize_act_row(size_t n_in, const float *x, int8_t *xq, int32_t *sum_a_out) {
    float max_abs = 1e-5f;
    for (size_t i = 0; i < n_in; i++) {
        const float a = fabsf(x[i]);
        if (a > max_abs) {
            max_abs = a;
        }
    }
    const float act_scale = 127.0f / max_abs;
    int32_t     sum_a     = 0;
    for (size_t i = 0; i < n_in; i++) {
        const float q  = x[i] * act_scale;
        int32_t     qi = (int32_t) (q < 0.0f ? q - 0.5f : q + 0.5f);
        if (qi > 127) {
            qi = 127;
        }
        if (qi < -128) {
            qi = -128;
        }
        xq[i] = (int8_t) qi;
        sum_a += qi;
    }
    *sum_a_out = sum_a;
    return max_abs / 127.0f; /* inv_act_scale */
}

/* One M=1 GEMV's rows for geist_par_for: y[r] = (dot(r) - sum_a) * scale. */
struct i2s_rows {
    size_t         n_blocks;
    size_t         row_bytes;
    const uint8_t *w_raw;
    const int8_t  *xq;
    int32_t        sum_a;
    float          scale;
    float         *y;
};

/* --- Scalar reference (oracle) ------------------------------------------- */
static int32_t i2s_row_dot_scalar(size_t n_blocks, const uint8_t *Wr, const int8_t *xq) {
    int32_t acc = 0;
    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *qs = Wr + b * I2S_BLOCK_BYTES;
        const int8_t  *xb = xq + b * I2S_BLOCK_ELEMS;
        for (size_t h = 0; h < 2; h++) {
            for (size_t bb = 0; bb < 32; bb++) {
                const uint8_t byte = qs[h * 32 + bb];
                for (size_t g = 0; g < 4; g++) {
                    const int trit = (int) ((byte >> (6 - 2 * g)) & 3) - 1;
                    acc += trit * (int) xb[h * 128 + g * 32 + bb];
                }
            }
        }
    }
    return acc;
}

static void i2s_rows_scalar(void *ctx, size_t r0, size_t r1) {
    const struct i2s_rows c = *(const struct i2s_rows *) ctx;
    for (size_t r = r0; r < r1; r++) {
        const int32_t dot = i2s_row_dot_scalar(c.n_blocks, c.w_raw + r * c.row_bytes, c.xq);
        c.y[r]            = (float) dot * c.scale;
    }
}

void i2s_gemv_m1_scalar(size_t        n_out,
                        size_t        n_in,
                        const float  *x,
                        const uint8_t w_raw[],
                        float         tensor_scale,
                        float         y[static n_out]) {
    const size_t n_blocks  = n_in / I2S_BLOCK_ELEMS;
    const size_t row_bytes = n_in / 4;
    int8_t      *xq        = (int8_t *) __builtin_alloca(n_in);
    int32_t      sum_a;
    const float  scale = tensor_scale * quantize_act_row(n_in, x, xq, &sum_a);
    (void) sum_a;

    struct i2s_rows c = {n_blocks, row_bytes, w_raw, xq, 0, scale, y};
    geist_par_for(n_out, i2s_rows_scalar, &c);
}

/* --- AVX2 (no VNNI) ------------------------------------------------------ */

/* Sum of code * xq over one row, code in {0, 1, 2}. Each 32-byte half of a
 * block holds four 32-element chunks, chunk g at shift 6 - 2g; the codes go
 * straight into maddubs (unsigned codes, signed activations). A pair is at
 * most 2 * 2 * 128 = 512 and a block's eight chunks sum in int16 to at most
 * 4096: no saturation, the sum is exact. The caller subtracts sum_a. */
static int32_t i2s_row_dot_avx2(size_t n_blocks, const uint8_t *Wr, const int8_t *xq) {
    const __m256i mask = _mm256_set1_epi8(3);
    __m256i       acc  = _mm256_setzero_si256();
    for (size_t b = 0; b < n_blocks; b++) {
        __m256i p16 = _mm256_setzero_si256();
        for (size_t h = 0; h < 2; h++) {
            const __m256i q =
                    _mm256_loadu_si256((const __m256i *) (Wr + b * I2S_BLOCK_BYTES + h * 32));
            for (int g = 0; g < 4; g++) {
                const __m256i v = _mm256_and_si256(_mm256_srli_epi16(q, 6 - 2 * g), mask);
                const __m256i a = _mm256_loadu_si256(
                        (const __m256i *) (xq + b * I2S_BLOCK_ELEMS + h * 128 + (size_t) g * 32));
                p16 = _mm256_add_epi16(p16, _mm256_maddubs_epi16(v, a));
            }
        }
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p16, _mm256_set1_epi16(1)));
    }
    return hsum_epi32(acc);
}

static void i2s_rows_avx2(void *ctx, size_t r0, size_t r1) {
    const struct i2s_rows c = *(const struct i2s_rows *) ctx;
    for (size_t r = r0; r < r1; r++) {
        const int32_t dot = i2s_row_dot_avx2(c.n_blocks, c.w_raw + r * c.row_bytes, c.xq) - c.sum_a;
        c.y[r]            = (float) dot * c.scale;
    }
}

void i2s_gemv_m1_avx2(size_t        n_out,
                      size_t        n_in,
                      const float  *x,
                      const uint8_t w_raw[],
                      float         tensor_scale,
                      float         y[static n_out]) {
    const size_t n_blocks  = n_in / I2S_BLOCK_ELEMS;
    const size_t row_bytes = n_in / 4;
    int8_t      *xq        = (int8_t *) __builtin_alloca(n_in);
    int32_t      sum_a;
    const float  scale = tensor_scale * quantize_act_row(n_in, x, xq, &sum_a);

    struct i2s_rows c = {n_blocks, row_bytes, w_raw, xq, sum_a, scale, y};
    geist_par_for(n_out, i2s_rows_avx2, &c);
}

/* --- Dispatch ------------------------------------------------------------ */
static _Atomic int g_i2s_vnni = -1;

[[nodiscard]] int i2s_isa_is_vnni(void) {
    if (g_i2s_vnni < 0) {
        const enum w4a8_isa tier = w4a8_dispatcher_init();
        g_i2s_vnni = (tier == W4A8_ISA_AVX512_VNNI || tier == W4A8_ISA_AVX512_BF16) ? 1 : 0;
    }
    return g_i2s_vnni;
}

void i2s_gemv_m1(size_t        n_out,
                 size_t        n_in,
                 const float  *x,
                 const uint8_t w_raw[],
                 float         tensor_scale,
                 float         y[static n_out]) {
    if (n_in % I2S_BLOCK_ELEMS != 0) {
        i2s_gemv_m1_scalar(n_out, n_in, x, w_raw, tensor_scale, y);
        return;
    }
    if (!i2s_isa_is_vnni()) {
        i2s_gemv_m1_avx2(n_out, n_in, x, w_raw, tensor_scale, y);
        return;
    }
    int8_t     *xq = (int8_t *) __builtin_alloca(n_in);
    int32_t     sum_a;
    const float scale = tensor_scale * quantize_act_row(n_in, x, xq, &sum_a);
    i2s_gemv_m1_avx512_vnni(n_out, n_in, xq, sum_a, w_raw, scale, y);
}

/* --- Prefill GEMM -------------------------------------------------------- */

/* Activation quant of m token rows for geist_par_for. */
struct i2s_quant_rows {
    size_t       n_in;
    float        tensor_scale;
    const float *x;
    int8_t      *xq;
    int32_t     *sum_a;
    float       *scale;
};

static void i2s_quant_rows(void *ctx, size_t i0, size_t i1) {
    const struct i2s_quant_rows c = *(const struct i2s_quant_rows *) ctx;
    for (size_t i = i0; i < i1; i++) {
        c.scale[i] = c.tensor_scale *
                     quantize_act_row(c.n_in, c.x + i * c.n_in, c.xq + i * c.n_in, &c.sum_a[i]);
    }
}

/* Without VNNI and without scratch: one GEMV per token. */
static void i2s_gemm_mN_avx2_rows(size_t        m,
                                  size_t        n_out,
                                  size_t        n_in,
                                  const float  *x,
                                  const uint8_t w_raw[],
                                  float         tensor_scale,
                                  float         y[]) {
    for (size_t i = 0; i < m; i++) {
        i2s_gemv_m1_avx2(n_out, n_in, x + i * n_in, w_raw, tensor_scale, y + i * n_out);
    }
}

/* Activation tile height of the AVX2 GEMM (#655). */
constexpr size_t I2S_AVX2_NR = 4;

/* i2s_row_dot_avx2 for NR activation rows (stride n_in) against one weight
 * row: each chunk of codes is unpacked once and fed to NR maddubs. The
 * integer sums are i2s_row_dot_avx2's, so the GEMM is bit-identical to the
 * GEMV per token. */
static void i2s_row_dot_avx2_nr(size_t         n_blocks,
                                size_t         n_in,
                                const uint8_t *Wr,
                                const int8_t  *xq,
                                int32_t        out[static I2S_AVX2_NR]) {
    static_assert(I2S_AVX2_NR == 4, "the accumulators below are spelled out for 4 tokens");
    const __m256i mask = _mm256_set1_epi8(3);
    const __m256i ones = _mm256_set1_epi16(1);
    const int8_t *x0 = xq, *x1 = xq + n_in, *x2 = xq + 2 * n_in, *x3 = xq + 3 * n_in;
    __m256i       a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
    __m256i       p0 = a0, p1 = a0, p2 = a0, p3 = a0;
    /* Named accumulators and one loop over the 32-byte halves (128
     * contiguous elements each), the int16 sums flushed after every second
     * half. With acc[4] / p16[4] arrays and a nested 2 x 4 loop, gcc 15
     * unrolled a whole block, hoisted all 32 maddubs ahead of their adds
     * and kept the accumulators on the stack. */
    for (size_t hh = 0; hh < 2 * n_blocks; hh++) {
        const __m256i q = _mm256_loadu_si256((const __m256i *) (Wr + hh * 32));
        for (int g = 0; g < 4; g++) {
            const __m256i v   = _mm256_and_si256(_mm256_srli_epi16(q, 6 - 2 * g), mask);
            const size_t  off = hh * 128 + (size_t) g * 32;
            p0                = _mm256_add_epi16(
                    p0, _mm256_maddubs_epi16(v, _mm256_loadu_si256((const __m256i *) (x0 + off))));
            p1 = _mm256_add_epi16(
                    p1, _mm256_maddubs_epi16(v, _mm256_loadu_si256((const __m256i *) (x1 + off))));
            p2 = _mm256_add_epi16(
                    p2, _mm256_maddubs_epi16(v, _mm256_loadu_si256((const __m256i *) (x2 + off))));
            p3 = _mm256_add_epi16(
                    p3, _mm256_maddubs_epi16(v, _mm256_loadu_si256((const __m256i *) (x3 + off))));
        }
        if (hh & 1) {
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(p0, ones));
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(p1, ones));
            a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(p2, ones));
            a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(p3, ones));
            p0 = p1 = p2 = p3 = _mm256_setzero_si256();
        }
    }
    out[0] = hsum_epi32(a0);
    out[1] = hsum_epi32(a1);
    out[2] = hsum_epi32(a2);
    out[3] = hsum_epi32(a3);
}

/* Without VNNI, on quantized activations. geist_par_for gives each thread
 * one contiguous slice of weight rows; the thread walks the tokens NR at a
 * time over its whole slice: the slice stays in L2 across the token tiles
 * and NR activation rows stay in L1 across the slice, so the activations
 * stream once per thread and the weights are unpacked once per NR tokens.
 * Walking all m tokens per row instead, or per block of rows, re-streams
 * the activations from L3 once per row or block; on the 6912-row FFN
 * matrices at m = 512 that lost to the per-token GEMV. */
struct i2s_gemm_rows {
    size_t         m, n_out, n_in, n_blocks, row_bytes;
    const int8_t  *xq;
    const int32_t *sum_a;
    const float   *scale;
    const uint8_t *w_raw;
    float         *y;
};

static void i2s_gemm_rows_avx2(void *ctx, size_t r0, size_t r1) {
    const struct i2s_gemm_rows c     = *(const struct i2s_gemm_rows *) ctx;
    const size_t               m_til = c.m - c.m % I2S_AVX2_NR;
    for (size_t i = 0; i < m_til; i += I2S_AVX2_NR) {
        for (size_t r = r0; r < r1; r++) {
            int32_t dot[I2S_AVX2_NR];
            i2s_row_dot_avx2_nr(
                    c.n_blocks, c.n_in, c.w_raw + r * c.row_bytes, c.xq + i * c.n_in, dot);
            for (size_t t = 0; t < I2S_AVX2_NR; t++) {
                c.y[(i + t) * c.n_out + r] = (float) (dot[t] - c.sum_a[i + t]) * c.scale[i + t];
            }
        }
    }
    for (size_t i = m_til; i < c.m; i++) {
        for (size_t r = r0; r < r1; r++) {
            const int32_t dot =
                    i2s_row_dot_avx2(c.n_blocks, c.w_raw + r * c.row_bytes, c.xq + i * c.n_in) -
                    c.sum_a[i];
            c.y[i * c.n_out + r] = (float) dot * c.scale[i];
        }
    }
}

static void i2s_gemm_avx2(size_t         m,
                          size_t         n_out,
                          size_t         n_in,
                          const int8_t  *xq,
                          const int32_t *sum_a,
                          const float   *scale,
                          const uint8_t  w_raw[],
                          float          y[]) {
    struct i2s_gemm_rows c = {
            m, n_out, n_in, n_in / I2S_BLOCK_ELEMS, n_in / 4, xq, sum_a, scale, w_raw, y};
    geist_par_for(n_out, i2s_gemm_rows_avx2, &c);
}

void i2s_gemm_mN_scalar(size_t        m,
                        size_t        n_out,
                        size_t        n_in,
                        const float  *x,
                        const uint8_t w_raw[],
                        float         tensor_scale,
                        float         y[]) {
    for (size_t i = 0; i < m; i++) {
        i2s_gemv_m1_scalar(n_out, n_in, x + i * n_in, w_raw, tensor_scale, y + i * n_out);
    }
}

void i2s_gemm_mN_pre(size_t        m,
                     size_t        n_out,
                     size_t        n_in,
                     const float  *x,
                     const uint8_t w_raw[],
                     float         tensor_scale,
                     int8_t        xq[],
                     int32_t       sum_a[],
                     float         scale[],
                     int8_t        perm[],
                     float         y[]) {
    if (m == 0) {
        return;
    }
    if (n_in % I2S_BLOCK_ELEMS != 0) {
        i2s_gemm_mN_scalar(m, n_out, n_in, x, w_raw, tensor_scale, y);
        return;
    }
    if (xq == nullptr || sum_a == nullptr || scale == nullptr) {
        i2s_gemm_mN_avx2_rows(m, n_out, n_in, x, w_raw, tensor_scale, y);
        return;
    }
    struct i2s_quant_rows q = {n_in, tensor_scale, x, xq, sum_a, scale};
    geist_par_for(m, i2s_quant_rows, &q);
    if (!i2s_isa_is_vnni()) {
        i2s_gemm_avx2(m, n_out, n_in, xq, sum_a, scale, w_raw, y);
        return;
    }
    /* A null `perm` is not an error — the GEMM falls back to its M=1 loop. */
    i2s_gemm_avx512_vnni(m, n_out, n_in, xq, sum_a, scale, w_raw, perm, y);
}

/* Convenience wrapper: allocates the scratch the _pre form expects. The
 * backend resolver does NOT use this (it passes workspace); it is here for
 * tests and one-off callers. */
void i2s_gemm_mN(size_t        m,
                 size_t        n_out,
                 size_t        n_in,
                 const float  *x,
                 const uint8_t w_raw[],
                 float         tensor_scale,
                 float         y[]) {
    if (m == 0) {
        return;
    }
    if (n_in % I2S_BLOCK_ELEMS != 0) {
        i2s_gemm_mN_scalar(m, n_out, n_in, x, w_raw, tensor_scale, y);
        return;
    }
    int8_t  *xq    = heap_alloc_n_aligned(m, n_in, OPTIMAL_ALIGNMENT);
    int32_t *sum_a = heap_alloc_n_aligned(m, sizeof(int32_t), OPTIMAL_ALIGNMENT);
    float   *scale = heap_alloc_n_aligned(m, sizeof(float), OPTIMAL_ALIGNMENT);
    int8_t  *perm  = heap_alloc_n_aligned(m, n_in, OPTIMAL_ALIGNMENT);
    /* _pre takes a failed (null) allocation as "no scratch". */
    i2s_gemm_mN_pre(m, n_out, n_in, x, w_raw, tensor_scale, xq, sum_a, scale, perm, y);
    safe_free((void **) &xq);
    safe_free((void **) &sum_a);
    safe_free((void **) &scale);
    safe_free((void **) &perm);
}

/* --- x4 row-interleaved layout ------------------------------------------- */

/* Decode one native-I2_S weight code ∈ {0,1,2} at (row, col). The native
 * block layout: 256-elem/64-byte blocks; element e=col%256 maps to byte
 * qs[h*32+bb] (h=e/128, bb=e%32) at shift 6-2g (g=(e%128)/32). */
static inline uint8_t
i2s_native_code(const uint8_t *w_raw, size_t row_bytes, size_t row, size_t col) {
    const size_t  b    = col / 256;
    const size_t  e    = col % 256;
    const size_t  h    = e / 128;
    const size_t  g    = (e % 128) / 32;
    const size_t  bb   = e % 32;
    const uint8_t byte = w_raw[row * row_bytes + b * 64 + h * 32 + bb];
    return (uint8_t) ((byte >> (6 - 2 * g)) & 3);
}

/* A weight repack for geist_par_for: n_in columns, native rows in, out. */
struct i2s_repack {
    size_t         n_in;
    const uint8_t *w_raw;
    uint8_t       *out;
};

static void i2s_to_x4_groups(void *ctx, size_t g0, size_t g1) {
    const struct i2s_repack rp        = *(const struct i2s_repack *) ctx;
    const size_t            n_in      = rp.n_in;
    const size_t            row_bytes = n_in / 4;
    for (size_t grp = g0; grp < g1; grp++) {
        uint8_t *dst = rp.out + grp * n_in;
        for (size_t c = 0; c < n_in; c++) {
            const uint8_t r0 = i2s_native_code(rp.w_raw, row_bytes, grp * 4 + 0, c);
            const uint8_t r1 = i2s_native_code(rp.w_raw, row_bytes, grp * 4 + 1, c);
            const uint8_t r2 = i2s_native_code(rp.w_raw, row_bytes, grp * 4 + 2, c);
            const uint8_t r3 = i2s_native_code(rp.w_raw, row_bytes, grp * 4 + 3, c);
            dst[c]           = (uint8_t) ((r0 << 6) | (r1 << 4) | (r2 << 2) | r3);
        }
    }
}

void i2s_to_x4(size_t n_out, size_t n_in, const uint8_t w_raw[], uint8_t x4[]) {
    struct i2s_repack rp = {n_in, w_raw, x4};
    geist_par_for(n_out / 4, i2s_to_x4_groups, &rp);
}

void i2s_x4_gemv_m1(size_t        n_out,
                    size_t        n_in,
                    const float  *x,
                    const uint8_t x4[],
                    float         tensor_scale,
                    float         y[static n_out]) {
    int8_t     *xq = (int8_t *) __builtin_alloca(n_in);
    int32_t     sum_a;
    const float scale = tensor_scale * quantize_act_row(n_in, x, xq, &sum_a);
    i2s_x4_gemv_m1_avx512_vnni(n_out, n_in, xq, sum_a, x4, scale, y);
}

void i2s_x4_gemm_mN_pre(size_t        m,
                        size_t        n_out,
                        size_t        n_in,
                        const float  *x,
                        const uint8_t x4[],
                        float         tensor_scale,
                        int8_t        xq[],
                        int32_t       sum_a[],
                        float         scale[],
                        float         y[]) {
    if (xq == nullptr || sum_a == nullptr || scale == nullptr) {
        for (size_t i = 0; i < m; i++) {
            i2s_x4_gemv_m1(n_out, n_in, x + i * n_in, x4, tensor_scale, y + i * n_out);
        }
        return;
    }
    struct i2s_quant_rows q = {n_in, tensor_scale, x, xq, sum_a, scale};
    geist_par_for(m, i2s_quant_rows, &q);
    i2s_x4_gemm_avx512_vnni(m, n_out, n_in, xq, sum_a, scale, x4, y);
}

/* Convenience wrapper — see i2s_gemm_mN above. */
void i2s_x4_gemm_mN(size_t        m,
                    size_t        n_out,
                    size_t        n_in,
                    const float  *x,
                    const uint8_t x4[],
                    float         tensor_scale,
                    float         y[]) {
    int8_t  *xq    = heap_alloc_n_aligned(m, n_in, OPTIMAL_ALIGNMENT);
    int32_t *sum_a = heap_alloc_n_aligned(m, sizeof(int32_t), OPTIMAL_ALIGNMENT);
    float   *scale = heap_alloc_n_aligned(m, sizeof(float), OPTIMAL_ALIGNMENT);
    i2s_x4_gemm_mN_pre(m, n_out, n_in, x, x4, tensor_scale, xq, sum_a, scale, y);
    safe_free((void **) &xq);
    safe_free((void **) &sum_a);
    safe_free((void **) &scale);
}

void i2s_x4_gemv_pair_m1(size_t        n_in,
                         const float  *x,
                         const uint8_t x4_0[],
                         float         tensor_scale0,
                         size_t        n_out0,
                         float        *y0,
                         const uint8_t x4_1[],
                         float         tensor_scale1,
                         size_t        n_out1,
                         float        *y1) {
    int8_t     *xq = (int8_t *) __builtin_alloca(n_in);
    int32_t     sum_a;
    const float inv = quantize_act_row(n_in, x, xq, &sum_a); /* shared activation quant */
    i2s_x4_gemv_pair_m1_avx512_vnni(n_in,
                                    xq,
                                    sum_a,
                                    x4_0,
                                    tensor_scale0 * inv,
                                    n_out0,
                                    y0,
                                    x4_1,
                                    tensor_scale1 * inv,
                                    n_out1,
                                    y1);
}

/* --- t5 base-3 layout (1.6 bpw decode, #104) ------------------------------ */

/* VNNI kernels (kernel_i2s_avx512_vnni.c). n_in_pad is the 320-padded
 * column count; xq must be zero-padded to it. */
void i2s_t5_gemv_m1_avx512_vnni(size_t        n_out,
                                size_t        n_in_pad,
                                const int8_t *xq,
                                int32_t       sum_a,
                                const uint8_t t5[],
                                float         scale,
                                float         y[static n_out]);

void i2s_t5_gemv_pair_m1_avx512_vnni(size_t        n_in_pad,
                                     const int8_t *xq,
                                     int32_t       sum_a,
                                     const uint8_t t5_0[],
                                     float         scale0,
                                     size_t        n_out0,
                                     float        *y0,
                                     const uint8_t t5_1[],
                                     float         scale1,
                                     size_t        n_out1,
                                     float        *y1);

static void i2s_to_t5_rows(void *ctx, size_t r0, size_t r1) {
    const struct i2s_repack rp            = *(const struct i2s_repack *) ctx;
    const size_t            n_in          = rp.n_in;
    const size_t            row_bytes_src = n_in / 4;
    const size_t            cols_pad      = i2s_t5_cols_pad(n_in);
    const size_t            row_bytes     = i2s_t5_row_bytes(n_in);
    for (size_t r = r0; r < r1; r++) {
        uint8_t *row = rp.out + r * row_bytes;
        for (size_t g = 0; g < cols_pad / 320; g++) {
            for (size_t c = 0; c < 64; c++) {
                uint32_t n = 0;
                for (size_t plane = 0; plane < 5; plane++) {
                    const size_t  col = g * 320 + plane * 64 + c;
                    const uint8_t code =
                            (col < n_in) ? i2s_native_code(rp.w_raw, row_bytes_src, r, col) : 0;
                    n = n * 3 + code;
                }
                row[g * 64 + c] = (uint8_t) ((n * 256 + 242) / 243);
            }
        }
    }
}

void i2s_to_t5(size_t n_out, size_t n_in, const uint8_t w_raw[], uint8_t t5[]) {
    struct i2s_repack rp = {n_in, w_raw, t5};
    geist_par_for(n_out, i2s_to_t5_rows, &rp);
}

void i2s_t5_gemv_m1(size_t        n_out,
                    size_t        n_in,
                    const float  *x,
                    const uint8_t t5[],
                    float         tensor_scale,
                    float         y[static n_out]) {
    const size_t cols_pad = i2s_t5_cols_pad(n_in);
    int8_t      *xq       = (int8_t *) __builtin_alloca(cols_pad);
    int32_t      sum_a;
    const float  scale = tensor_scale * quantize_act_row(n_in, x, xq, &sum_a);
    if (cols_pad > n_in) {
        __builtin_memset(xq + n_in, 0, cols_pad - n_in); /* pads contribute 0 */
    }
    i2s_t5_gemv_m1_avx512_vnni(n_out, cols_pad, xq, sum_a, t5, scale, y);
}

void i2s_t5_gemv_pair_m1(size_t        n_in,
                         const float  *x,
                         const uint8_t t5_0[],
                         float         tensor_scale0,
                         size_t        n_out0,
                         float        *y0,
                         const uint8_t t5_1[],
                         float         tensor_scale1,
                         size_t        n_out1,
                         float        *y1) {
    const size_t cols_pad = i2s_t5_cols_pad(n_in);
    int8_t      *xq       = (int8_t *) __builtin_alloca(cols_pad);
    int32_t      sum_a;
    const float  inv = quantize_act_row(n_in, x, xq, &sum_a); /* shared activation quant */
    if (cols_pad > n_in) {
        __builtin_memset(xq + n_in, 0, cols_pad - n_in);
    }
    i2s_t5_gemv_pair_m1_avx512_vnni(cols_pad,
                                    xq,
                                    sum_a,
                                    t5_0,
                                    tensor_scale0 * inv,
                                    n_out0,
                                    y0,
                                    t5_1,
                                    tensor_scale1 * inv,
                                    n_out1,
                                    y1);
}

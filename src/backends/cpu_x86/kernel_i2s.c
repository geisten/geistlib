/*
 * src/backends/cpu_x86/kernel_i2s.c — I2_S ternary GEMV dispatcher + scalar.
 *
 * Layer: BACKEND (cpu_x86).
 *
 * Compiled at baseline -march=x86-64-v3; the AVX-512+VNNI variant lives in
 * kernel_i2s_avx512_vnni.c with -mavx512vnni. See kernel_i2s.h for the
 * ternary algebra.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "kernel_i2s.h"

#include "kernel_w4a8.h" /* w4a8_dispatcher_init — shared ISA selection */

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

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
 * inv values 1 ULP apart — enough to break the pair==m1 byte-identity
 * contract (caught by test_i2s_gemv_unit case 5, #102 Phase 2). One
 * compiled instance = one rounding, for every caller. Called once per
 * GEMV; the call overhead is noise. */
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

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t r = 0; r < n_out; r++) {
        const int32_t dot = i2s_row_dot_scalar(n_blocks, w_raw + r * row_bytes, xq);
        y[r]              = (float) dot * scale;
    }
}

/* --- Dispatch ------------------------------------------------------------ */
static int g_i2s_vnni = -1;

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
        i2s_gemv_m1_scalar(n_out, n_in, x, w_raw, tensor_scale, y);
        return;
    }
    int8_t     *xq = (int8_t *) __builtin_alloca(n_in);
    int32_t     sum_a;
    const float scale = tensor_scale * quantize_act_row(n_in, x, xq, &sum_a);
    i2s_gemv_m1_avx512_vnni(n_out, n_in, xq, sum_a, w_raw, scale, y);
}

/* --- Prefill GEMM -------------------------------------------------------- */
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
    if (n_in % I2S_BLOCK_ELEMS != 0 || !i2s_isa_is_vnni()) {
        i2s_gemm_mN_scalar(m, n_out, n_in, x, w_raw, tensor_scale, y);
        return;
    }
    int8_t  *xq    = (int8_t *) malloc(m * n_in);
    int32_t *sum_a = (int32_t *) malloc(m * sizeof(int32_t));
    float   *scale = (float *) malloc(m * sizeof(float));
    if (xq == nullptr || sum_a == nullptr || scale == nullptr) {
        free(xq);
        free(sum_a);
        free(scale);
        i2s_gemm_mN_scalar(m, n_out, n_in, x, w_raw, tensor_scale, y);
        return;
    }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < m; i++) {
        scale[i] = tensor_scale * quantize_act_row(n_in, x + i * n_in, xq + i * n_in, &sum_a[i]);
    }
    i2s_gemm_avx512_vnni(m, n_out, n_in, xq, sum_a, scale, w_raw, y);
    free(xq);
    free(sum_a);
    free(scale);
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

void i2s_to_x4(size_t n_out, size_t n_in, const uint8_t w_raw[], uint8_t x4[]) {
    const size_t row_bytes = n_in / 4;
    const size_t n_groups  = n_out / 4;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t grp = 0; grp < n_groups; grp++) {
        uint8_t *dst = x4 + grp * n_in;
        for (size_t c = 0; c < n_in; c++) {
            const uint8_t r0 = i2s_native_code(w_raw, row_bytes, grp * 4 + 0, c);
            const uint8_t r1 = i2s_native_code(w_raw, row_bytes, grp * 4 + 1, c);
            const uint8_t r2 = i2s_native_code(w_raw, row_bytes, grp * 4 + 2, c);
            const uint8_t r3 = i2s_native_code(w_raw, row_bytes, grp * 4 + 3, c);
            dst[c]           = (uint8_t) ((r0 << 6) | (r1 << 4) | (r2 << 2) | r3);
        }
    }
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

void i2s_x4_gemm_mN(size_t        m,
                    size_t        n_out,
                    size_t        n_in,
                    const float  *x,
                    const uint8_t x4[],
                    float         tensor_scale,
                    float         y[]) {
    int8_t  *xq    = (int8_t *) malloc(m * n_in);
    int32_t *sum_a = (int32_t *) malloc(m * sizeof(int32_t));
    float   *scale = (float *) malloc(m * sizeof(float));
    if (xq == nullptr || sum_a == nullptr || scale == nullptr) {
        free(xq);
        free(sum_a);
        free(scale);
        for (size_t i = 0; i < m; i++) {
            i2s_x4_gemv_m1(n_out, n_in, x + i * n_in, x4, tensor_scale, y + i * n_out);
        }
        return;
    }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < m; i++) {
        scale[i] = tensor_scale * quantize_act_row(n_in, x + i * n_in, xq + i * n_in, &sum_a[i]);
    }
    i2s_x4_gemm_avx512_vnni(m, n_out, n_in, xq, sum_a, scale, x4, y);
    free(xq);
    free(sum_a);
    free(scale);
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

void i2s_to_t5(size_t n_out, size_t n_in, const uint8_t w_raw[], uint8_t t5[]) {
    const size_t row_bytes_src = n_in / 4;
    const size_t cols_pad      = i2s_t5_cols_pad(n_in);
    const size_t row_bytes     = i2s_t5_row_bytes(n_in);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t r = 0; r < n_out; r++) {
        uint8_t *row = t5 + r * row_bytes;
        for (size_t g = 0; g < cols_pad / 320; g++) {
            for (size_t c = 0; c < 64; c++) {
                uint32_t n = 0;
                for (size_t plane = 0; plane < 5; plane++) {
                    const size_t  col = g * 320 + plane * 64 + c;
                    const uint8_t code =
                            (col < n_in) ? i2s_native_code(w_raw, row_bytes_src, r, col) : 0;
                    n = n * 3 + code;
                }
                row[g * 64 + c] = (uint8_t) ((n * 256 + 242) / 243);
            }
        }
    }
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

/*
 * src/backends/cpu_scalar/weight_resolve.c — resolver for cpu_scalar (P2.e).
 *
 * Layer: BACKEND.
 *
 * cpu_scalar is the pure-C reference backend. After P2.e the legacy
 * v->linear() vtable slot is gone from geist_backend_vtbl; every
 * backend must install kernel pointers via resolve_weight. This
 * file gives cpu_scalar a (slow, correct) resolver that wraps the
 * existing dequant helpers from gguf_quant.c into pre-resolved
 * function pointers.
 *
 * Performance characteristics:
 *   - F32 dense: naive triple loop with double accumulator. ~10× slower
 *     than cpu_neon + cblas; intentional, this is the reference.
 *   - Q3_K / Q4_K / Q5_K / Q6_K / Q8_0 / IQ2_S / IQ3_S / TQ2_0 / I2_S /
 *     F16 / BF16: dequant one weight row at a time into a heap row-buffer,
 *     naive dot. Same as the old cpu_scalar_linear_quant body, just exposed
 *     via the resolver pattern. (I2_S/F16 added for BitNet b1.58 2B-4T, whose
 *     ternary BitLinear weights are I2_S and whose tied lm_head is F16.)
 *
 * No SIMD, no BLAS — that's what cpu_neon is for.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "internal.h"

#include "quant.h"
#include "heap.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Dispatch helper: dequant one row of weight `w` row index `j` into
 * `row` (which is `n_in` floats). Returns false on unsupported dtype. */
static bool dequant_one_row_for(const struct geist_weight *w, size_t j, float *row) {
    const uint8_t *base = (const uint8_t *) w->raw;
    const size_t   n_in = (size_t) w->n_in;
    switch ((enum geist_dtype) w->dtype) {
    case GEIST_DTYPE_F16: {
        const uint8_t *r = base + j * n_in * 2;
        for (size_t i = 0; i < n_in; i++) {
            const uint16_t h = (uint16_t) r[2 * i] | ((uint16_t) r[2 * i + 1] << 8);
            row[i]           = fp16_to_fp32(h);
        }
        return true;
    }
    case GEIST_DTYPE_BF16: {
        const uint8_t *r = base + j * n_in * 2;
        for (size_t i = 0; i < n_in; i++) {
            const uint32_t b = (uint32_t) ((uint16_t) r[2 * i] | ((uint16_t) r[2 * i + 1] << 8))
                               << 16;
            memcpy(&row[i], &b, sizeof b);
        }
        return true;
    }
    case GEIST_DTYPE_Q3_K:
        dequant_q3_K_row(base + j * n_in / Q3_K_BLOCK_ELEMS * Q3_K_BLOCK_BYTES, row, n_in);
        return true;
    case GEIST_DTYPE_Q4_K:
        dequant_q4_K_row(base + j * n_in / Q4_K_BLOCK_ELEMS * Q4_K_BLOCK_BYTES, row, n_in);
        return true;
    case GEIST_DTYPE_Q5_K:
        dequant_q5_K_row(base + j * n_in / Q5_K_BLOCK_ELEMS * Q5_K_BLOCK_BYTES, row, n_in);
        return true;
    case GEIST_DTYPE_Q6_K:
        dequant_q6_K_row(base + j * n_in / Q6_K_BLOCK_ELEMS * Q6_K_BLOCK_BYTES, row, n_in);
        return true;
    case GEIST_DTYPE_Q8_0:
        dequant_q8_0_row(base + j * n_in / Q8_0_BLOCK_ELEMS * Q8_0_BLOCK_BYTES, row, n_in);
        return true;
    case GEIST_DTYPE_IQ2_S:
        dequant_iq2_s_row(base + j * n_in / IQ2_S_BLOCK_ELEMS * IQ2_S_BLOCK_BYTES, row, n_in);
        return true;
    case GEIST_DTYPE_IQ3_S:
        dequant_iq3_s_row(base + j * n_in / IQ3_S_BLOCK_ELEMS * IQ3_S_BLOCK_BYTES, row, n_in);
        return true;
    case GEIST_DTYPE_TQ2_0:
        dequant_tq2_0_row(base + j * n_in / TQ2_0_BLOCK_ELEMS * TQ2_0_BLOCK_BYTES, row, n_in);
        return true;
    case GEIST_DTYPE_I2_S: {
        /* BitNet b1.58 official: 256-elem/64-byte ternary blocks, four 2-bit
         * fields per byte in REVERSE order (element 32*g+bb at shift 6-2g),
         * ONE f32 per-TENSOR scale at the tail (offset n_in*n_out/4). */
        float        scale;
        const size_t packed = n_in * (size_t) w->n_out / 4;
        memcpy(&scale, base + packed, sizeof scale);
        const uint8_t *Wr = base + j * (n_in / 4);
        for (size_t b = 0; b < n_in / 256; b++) {
            const uint8_t *qs = Wr + b * 64;
            for (size_t h = 0; h < 2; h++) {
                for (size_t bb = 0; bb < 32; bb++) {
                    const uint8_t byte = qs[h * 32 + bb];
                    for (size_t g = 0; g < 4; g++) {
                        const int trit = (int) ((byte >> (6 - 2 * g)) & 3) - 1;
                        row[b * 256 + h * 128 + g * 32 + bb] = (float) trit * scale;
                    }
                }
            }
        }
        return true;
    }
    case GEIST_DTYPE_Q4_0:
        dequant_q4_0_row(base + j * n_in / Q4_0_BLOCK_ELEMS * Q4_0_BLOCK_BYTES, row, n_in);
        return true;
    default:
        return false;
    }
}

/* ---- F32 dense ---- */

static void cpu_scalar_w_f32_m1(const float               *x,
                                const struct geist_weight *w,
                                struct geist_backend      *be,
                                float                     *y) {
    (void) be;
    const float *wp    = (const float *) w->raw;
    const size_t n_in  = (size_t) w->n_in;
    const size_t n_out = (size_t) w->n_out;
    for (size_t j = 0; j < n_out; j++) {
        double       acc = 0.0;
        const float *row = wp + j * n_in;
        for (size_t k = 0; k < n_in; k++)
            acc += (double) x[k] * (double) row[k];
        y[j] = (float) acc;
    }
}

static void cpu_scalar_w_f32_mN(const float               *x,
                                const struct geist_weight *w,
                                size_t                     m,
                                struct geist_backend      *be,
                                float                     *y) {
    (void) be;
    const float *wp    = (const float *) w->raw;
    const size_t n_in  = (size_t) w->n_in;
    const size_t n_out = (size_t) w->n_out;
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n_out; j++) {
            double       acc = 0.0;
            const float *row = wp + j * n_in;
            for (size_t k = 0; k < n_in; k++) {
                acc += (double) x[i * n_in + k] * (double) row[k];
            }
            y[i * n_out + j] = (float) acc;
        }
    }
}

/* ---- Quantized: per-row dequant + naive dot/matmul ---- */

static void cpu_scalar_w_quant_m1(const float               *x,
                                  const struct geist_weight *w,
                                  struct geist_backend      *be,
                                  float                     *y) {
    const size_t n_in  = (size_t) w->n_in;
    const size_t n_out = (size_t) w->n_out;
    float       *row   = heap_alloc_aligned(n_in * sizeof(float), OPTIMAL_ALIGNMENT);
    if (row == nullptr) {
        geist_backend_set_error(be, GEIST_E_OOM, "cpu_scalar: dequant row alloc failed");
        return;
    }
    for (size_t j = 0; j < n_out; j++) {
        if (!dequant_one_row_for(w, j, row)) {
            y[j] = 0;
            continue;
        }
        double acc = 0.0;
        for (size_t k = 0; k < n_in; k++)
            acc += (double) x[k] * (double) row[k];
        y[j] = (float) acc;
    }
    safe_free((void **) &row);
}

static void cpu_scalar_w_quant_mN(const float               *x,
                                  const struct geist_weight *w,
                                  size_t                     m,
                                  struct geist_backend      *be,
                                  float                     *y) {
    const size_t n_in  = (size_t) w->n_in;
    const size_t n_out = (size_t) w->n_out;
    float       *row   = heap_alloc_aligned(n_in * sizeof(float), OPTIMAL_ALIGNMENT);
    if (row == nullptr) {
        geist_backend_set_error(be, GEIST_E_OOM, "cpu_scalar: dequant row alloc failed");
        return;
    }
    for (size_t j = 0; j < n_out; j++) {
        if (!dequant_one_row_for(w, j, row)) {
            for (size_t i = 0; i < m; i++)
                y[i * n_out + j] = 0;
            continue;
        }
        for (size_t i = 0; i < m; i++) {
            double acc = 0.0;
            for (size_t k = 0; k < n_in; k++) {
                acc += (double) x[i * n_in + k] * (double) row[k];
            }
            y[i * n_out + j] = (float) acc;
        }
    }
    safe_free((void **) &row);
}

enum geist_support cpu_scalar_weight_support(enum geist_dtype dtype) {
    switch (dtype) {
    case GEIST_DTYPE_F32:
        return GEIST_SUPPORT_NATIVE; /* dedicated F32 kernel */
    case GEIST_DTYPE_Q4_0:
    case GEIST_DTYPE_Q3_K:
    case GEIST_DTYPE_Q4_K:
    case GEIST_DTYPE_Q5_K:
    case GEIST_DTYPE_Q6_K:
    case GEIST_DTYPE_Q8_0:
    case GEIST_DTYPE_IQ2_S:
    case GEIST_DTYPE_IQ3_S:
    case GEIST_DTYPE_TQ2_0:
    case GEIST_DTYPE_I2_S:
    case GEIST_DTYPE_F16:
    case GEIST_DTYPE_BF16:
        return GEIST_SUPPORT_EMULATED; /* generic dequant-row + scalar dot */
    default:
        return GEIST_SUPPORT_NONE;
    }
}

[[nodiscard]] enum geist_status cpu_scalar_resolve_weight(struct geist_backend *be,
                                                          struct geist_weight  *w) {
    (void) be;
    if (w == nullptr || w->raw == nullptr || w->n_in <= 0 || w->n_out <= 0) {
        return GEIST_E_INVALID_ARG;
    }
    switch (cpu_scalar_weight_support((enum geist_dtype) w->dtype)) {
    case GEIST_SUPPORT_NATIVE:
        w->linear_m1 = cpu_scalar_w_f32_m1;
        w->linear_mN = cpu_scalar_w_f32_mN;
        return GEIST_OK;
    case GEIST_SUPPORT_EMULATED:
        w->linear_m1 = cpu_scalar_w_quant_m1;
        w->linear_mN = cpu_scalar_w_quant_mN;
        return GEIST_OK;
    default:
        return GEIST_E_UNSUPPORTED;
    }
}

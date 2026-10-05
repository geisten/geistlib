/*
 * src/backends/cpu_scalar/weight_resolve.c — resolver for cpu_scalar.
 *
 * Layer: BACKEND.
 *
 * cpu_scalar is the pure-C reference backend. geist_backend_vtbl has no
 * linear() slot; every backend installs kernel pointers via
 * resolve_weight. This file gives cpu_scalar a (slow, correct) resolver
 * that wraps the dequant row helpers (quant.h, defined in
 * src/formats/gguf/) into pre-resolved function pointers.
 *
 * Performance characteristics:
 *   - F32 dense: naive triple loop with double accumulator. ~10× slower
 *     than cpu_neon + cblas; intentional, this is the reference.
 *   - Every block format quant.h decodes, and F16 / BF16: geist_linear_ref
 *     (backends/common/linear_ref.c) — each weight row decoded a tile at a
 *     time into a stack buffer, naive dot in double. It allocates nothing,
 *     so it cannot fail; the other CPU backends fall back to it when their
 *     scratch cannot be had. (I2_S/F16 are there for BitNet b1.58 2B-4T,
 *     whose ternary BitLinear weights are I2_S and whose tied lm_head is
 *     F16.)
 *
 * No SIMD, no BLAS — that's what cpu_neon is for.
 *
 * ORACLE CAVEAT — ternary (I2_S / TQ2_0). Everywhere else this backend is the
 * correctness oracle other backends are checked against, bit for bit. For
 * ternary weights it is NOT, and cannot be: the reference kernel dequantizes
 * to fp32 and the dot then runs in fp32, i.e. W2A32, while cpu_neon binds
 * `cpu_neon_w_i2_s_q8a_*` — int8 activations, W2A8. Two different arithmetics,
 * so two different results by construction.
 *
 * Which is right depends on what you are asking. Arithmetically this path is
 * the more precise one. But 8-bit activations are part of the BitNet b1.58
 * definition, not an approximation of it, so cpu_neon computes the scheme the
 * model was trained for and this backend computes a different model that
 * happens to round less. Measured on BitNet 2B-4T i2_s: greedy output agrees
 * for 36 tokens and then drifts apart at a near-tie on the 37th. Q4_K and the rest stay
 * bit-identical, so a divergence outside ternary is a real bug, not this.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "internal.h"

#include "linear_ref.h"
#include "quant.h"
#include "selected_rows.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static void cpu_scalar_w_f32_mN(size_t                     m,
                                const float               *x,
                                const struct geist_weight *w,
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

/* ---- Quantized and half precision: the allocation-free reference ---- */

static void cpu_scalar_w_quant_m1(const float               *x,
                                  const struct geist_weight *w,
                                  struct geist_backend      *be,
                                  float                     *y) {
    (void) be;
    geist_linear_ref(1, x, w, y);
}

static void cpu_scalar_w_quant_mN(size_t                     m,
                                  const float               *x,
                                  const struct geist_weight *w,
                                  struct geist_backend      *be,
                                  float                     *y) {
    (void) be;
    geist_linear_ref(m, x, w, y);
}

[[nodiscard]] enum geist_status cpu_scalar_resolve_weight(struct geist_backend *be,
                                                          struct geist_weight  *w) {
    if (w == nullptr || w->raw == nullptr || w->n_in <= 0 || w->n_out <= 0 || w->raw_nbytes == 0u) {
        return GEIST_E_INVALID_ARG;
    }
    /* Same source-extent contract as cpu_neon: the kernels index `raw` by
     * (dtype, n_in, n_out), so a short buffer reads past its end. */
    if (!quant_weight_extent_ok(w)) {
        return GEIST_E_FORMAT;
    }
    /* cpu_x86 borrows this resolver before replacing its kernels. A row
     * capability belongs to the resolved kernel and backend, not merely
     * to the source dtype. Wrappers must resolve their own row readout. */
    w->linear_rows         = nullptr;
    w->linear_rows_tile    = 0;
    w->linear_rows_prepare = nullptr;
    /* Use the registry's unique backend ID. Taking the exported resolver's
     * address here would add a non-PIC text relocation to static archives
     * subsequently linked into the Linux FFI shared library. */
    const bool native_rows = be != nullptr && be->desc != nullptr && be->desc->name != nullptr &&
                             strcmp(be->desc->name, "cpu_scalar") == 0;
    if (w->dtype == GEIST_DTYPE_F32) {
        w->linear_m1 = cpu_scalar_w_f32_m1;
        w->linear_mN = cpu_scalar_w_f32_mN;
    } else if (geist_linear_ref_decodes(w->dtype)) {
        /* Everything the reference decodes; its list is the one list. */
        w->linear_m1 = cpu_scalar_w_quant_m1;
        w->linear_mN = cpu_scalar_w_quant_mN;
    } else {
        return GEIST_E_UNSUPPORTED;
    }
    size_t block, bytes, tail;
    if (native_rows && quant_block_layout((enum geist_dtype) w->dtype, &block, &bytes, &tail) &&
        tail == 0) {
        w->linear_rows      = geist_cpu_selected_rows;
        w->linear_rows_tile = 1;
    }
    return GEIST_OK;
}

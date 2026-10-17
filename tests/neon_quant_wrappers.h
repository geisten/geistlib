/*
 * tests/neon_quant_wrappers.h — fp32-in conveniences over the cpu_neon
 * `_pre` kernels, for tests and kernel benches only.
 *
 * The engine quantizes each activation row once (quantize_x_q8_groups) and
 * shares it across several `_pre` calls; nothing in src/ needs a wrapper
 * that allocates, quantizes and calls one kernel. Tests do, so the wrappers
 * live here. Each keeps the name, signature and behaviour of the library
 * function it replaced, so the tests that call them are unchanged.
 *
 * Needs cpu_neon in BACKENDS (the `_pre` kernels are defined there).
 */
#ifndef GEIST_TESTS_NEON_QUANT_WRAPPERS_H
#define GEIST_TESTS_NEON_QUANT_WRAPPERS_H

#include "heap.h"
#include "quant.h"

#include <stddef.h>
#include <stdint.h>

/* Activations of m rows, quantized per `group` elements; sum32 only when
 * asked for (the min-offset kernels). */
struct nqw_act {
    int8_t  *x_q8;
    float   *x_scales;
    int32_t *sum32;
};

static inline void nqw_act_free(struct nqw_act *a) {
    safe_free((void **) &a->x_q8);
    safe_free((void **) &a->x_scales);
    safe_free((void **) &a->sum32);
}

/* Returns false (and leaves *a empty) when an allocation fails. */
static inline bool nqw_quantize(
        size_t m, size_t n_in, size_t group, bool with_sum32, const float *x, struct nqw_act *a) {
    const size_t n_groups = geist_act_groups(n_in, group);
    *a                    = (struct nqw_act) {
            .x_q8     = heap_alloc_array_aligned(int8_t, m *n_in),
            .x_scales = heap_alloc_array_aligned(float, m *n_groups),
            .sum32    = with_sum32 ? heap_alloc_array_aligned(int32_t, m *(n_in / 32)) : nullptr,
    };
    if (a->x_q8 == nullptr || a->x_scales == nullptr || (with_sum32 && a->sum32 == nullptr)) {
        nqw_act_free(a);
        return false;
    }
    for (size_t i = 0; i < m; i++)
        quantize_x_q8_groups(n_in,
                             group,
                             x + i * n_in,
                             a->x_q8 + i * n_in,
                             a->x_scales + i * n_groups,
                             with_sum32 ? a->sum32 + i * (n_in / 32) : nullptr);
    return true;
}

/* ---- decode (one row) ---- */

static inline void
linear_iq2s_decode_w2a8(size_t n_in, size_t n_out, const float *x, const void *w, float *y) {
    struct nqw_act a;
    if (!nqw_quantize(1, n_in, GEIST_ACT_Q8K_ELEMS, false, x, &a))
        return;
    linear_iq2s_decode_w2a8_pre(n_in, n_out, a.x_scales, a.x_q8, w, y);
    nqw_act_free(&a);
}

static inline void
linear_iq3s_decode_w3a8(size_t n_in, size_t n_out, const float *x, const void *w, float *y) {
    struct nqw_act a;
    if (!nqw_quantize(1, n_in, GEIST_ACT_Q8K_ELEMS, false, x, &a))
        return;
    linear_iq3s_decode_w3a8_pre(n_in, n_out, a.x_scales, a.x_q8, w, y);
    nqw_act_free(&a);
}

static inline void
linear_iq4xs_decode_w4a8(size_t n_in, size_t n_out, const float *x, const void *w, float *y) {
    struct nqw_act a;
    if (!nqw_quantize(1, n_in, GEIST_ACT_Q8K_ELEMS, false, x, &a))
        return;
    linear_iq4xs_decode_w4a8_pre(n_in, n_out, a.x_scales, a.x_q8, w, y);
    nqw_act_free(&a);
}

static inline void
linear_q4_0_decode_w4a8(size_t n_in, size_t n_out, const float *x, const void *w, float *y) {
    struct nqw_act a;
    if (!nqw_quantize(1, n_in, GEIST_ACT_Q8_0_ELEMS, false, x, &a))
        return;
    linear_q4_0_decode_w4a8_pre(n_in, n_out, a.x_scales, a.x_q8, w, y);
    nqw_act_free(&a);
}

static inline void linear_q4_0_decode_w4a8_x8(
        size_t n_in, size_t n_out, const float *x, const void *packed, float *y) {
    struct nqw_act a;
    if (!nqw_quantize(1, n_in, GEIST_ACT_Q8_0_ELEMS, true, x, &a))
        return;
    linear_q4_0_decode_w4a8_x8_pre(n_in, n_out, a.x_scales, a.x_q8, a.sum32, packed, y);
    nqw_act_free(&a);
}

static inline void linear_q4k_decode_w4a8_predecoded(
        size_t n_in, size_t n_out, const float *x, const void *packed, float *y) {
    struct nqw_act a;
    if (!nqw_quantize(1, n_in, GEIST_ACT_Q8K_ELEMS, true, x, &a))
        return;
    linear_q4k_w4a8_prefill_predecoded(1, n_in, n_out, a.x_q8, a.x_scales, a.sum32, packed, y);
    nqw_act_free(&a);
}

static inline void
linear_q6k_decode_w6a8_x8(size_t n_in, size_t n_out, const float *x, const void *packed, float *y) {
    struct nqw_act a;
    if (!nqw_quantize(1, n_in, GEIST_ACT_Q8K_ELEMS, false, x, &a))
        return;
    linear_q6k_decode_w6a8_x8_pre(n_in, n_out, a.x_scales, a.x_q8, packed, y);
    nqw_act_free(&a);
}

/* ---- prefill (m rows, row-major x and y) ---- */

static inline void linear_iq2s_w2a8_prefill(
        size_t m, size_t n_in, size_t n_out, const float *x, const void *w, float *y) {
    struct nqw_act a;
    if (!nqw_quantize(m, n_in, GEIST_ACT_Q8K_ELEMS, false, x, &a))
        return;
    linear_iq2s_w2a8_prefill_pre(m, n_in, n_out, a.x_q8, a.x_scales, w, y);
    nqw_act_free(&a);
}

static inline void linear_iq3s_w3a8_prefill(
        size_t m, size_t n_in, size_t n_out, const float *x, const void *w, float *y) {
    struct nqw_act a;
    if (!nqw_quantize(m, n_in, GEIST_ACT_Q8K_ELEMS, false, x, &a))
        return;
    linear_iq3s_w3a8_prefill_pre(m, n_in, n_out, a.x_q8, a.x_scales, w, y);
    nqw_act_free(&a);
}

static inline void linear_iq4xs_w4a8_prefill(
        size_t m, size_t n_in, size_t n_out, const float *x, const void *w, float *y) {
    struct nqw_act a;
    if (!nqw_quantize(m, n_in, GEIST_ACT_Q8K_ELEMS, false, x, &a))
        return;
    linear_iq4xs_w4a8_prefill_pre(m, n_in, n_out, a.x_q8, a.x_scales, w, y);
    nqw_act_free(&a);
}

static inline void linear_q3k_w3a8_prefill(
        size_t m, size_t n_in, size_t n_out, const float *x, const void *w, float *y) {
    struct nqw_act a;
    if (!nqw_quantize(m, n_in, GEIST_ACT_Q8K_ELEMS, false, x, &a))
        return;
    linear_q3k_w3a8_prefill_pre(m, n_in, n_out, a.x_q8, a.x_scales, w, y);
    nqw_act_free(&a);
}

static inline void linear_q4_0_w4a8_prefill(
        size_t m, size_t n_in, size_t n_out, const float *x, const void *w, float *y) {
    struct nqw_act a;
    if (!nqw_quantize(m, n_in, GEIST_ACT_Q8_0_ELEMS, false, x, &a))
        return;
    linear_q4_0_w4a8_prefill_pre(m, n_in, n_out, a.x_q8, a.x_scales, w, y);
    nqw_act_free(&a);
}

static inline void linear_q4_0_w4a8_prefill_x8(
        size_t m, size_t n_in, size_t n_out, const float *x, const void *packed, float *y) {
    struct nqw_act a;
    if (m == 0 || !nqw_quantize(m, n_in, GEIST_ACT_Q8_0_ELEMS, true, x, &a))
        return;
    linear_q4_0_w4a8_prefill_x8_pre(m, n_in, n_out, a.x_q8, a.x_scales, a.sum32, packed, y);
    nqw_act_free(&a);
}

static inline void linear_q6k_w6a8_prefill(
        size_t m, size_t n_in, size_t n_out, const float *x, const void *w, float *y) {
    struct nqw_act a;
    if (m == 0 || m > GEIST_QUANT_M_CAP ||
        !nqw_quantize(m, n_in, GEIST_ACT_Q8K_ELEMS, false, x, &a))
        return;
    linear_q6k_w6a8_prefill_pre(m, n_in, n_out, a.x_q8, a.x_scales, w, y);
    nqw_act_free(&a);
}

#endif /* GEIST_TESTS_NEON_QUANT_WRAPPERS_H */

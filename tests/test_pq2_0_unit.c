/*
 * test_pq2_0_unit — PQ2_0 (PrismML ternary, Ternary-Bonsai) decode and
 * the resolved linear kernels.
 *
 *   1. dequant_pq2_0_row on a hand-built block: scale stored first and
 *      little-endian, element j at byte j/4 bits 2*(j%4), value
 *      (code - 1) * d including code 3 = +2 — the layout of
 *      dequantize_row_pq2_0 in PrismML-Eng/llama.cpp.
 *   2. Every CPU backend's resolved m=1 and m>1 kernels against an fp32
 *      reference built from (1), for n_in 128 / 5120 / 17408 and an odd
 *      n_out. The fp32-accumulating paths (cpu_scalar, the NEON
 *      trampolines) match it tightly.
 *   3. The NEON SDOT decode kernel against an exact model of its own
 *      W2A8 arithmetic (per-call absmax int8 activation, integer block
 *      dots) — tight — and against fp32 within the A8 quantization error.
 *      Matching the A8 model and not the fp32 one is also what shows the
 *      SDOT path, not the trampoline, was installed.
 *   4. Both SDOT layouts: n_out 40 installs the x8 interleaved repack
 *      (GEIST_W_LAYOUT_PQ2_0_X8_GEMV) — n_out 264 spans two full prefill
 *      tiles plus a remainder — n_out 37 cannot and keeps the row
 *      kernel, and GEIST_PQ2_0_X8_GEMV=0 keeps the row kernel for both —
 *      every one of them held to the same W2A8 model.
 *
 * Deterministic, no model needed.
 */
#define _POSIX_C_SOURCE 200809L /* setenv */
#include "test_helpers.h"

#include <geist_backend.h>
#include <geist_types.h>

#include "heap.h"
#include "quant.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t g_seed = 0x9E3779B9u;

static uint32_t urand(void) {
    g_seed = g_seed * 1664525u + 1013904223u;
    return g_seed >> 8;
}

static float frand(void) {
    return ((float) (urand() & 0xffff) / 65536.0f) * 2.0f - 1.0f;
}

/* Random PQ2_0 tensor [n_out, n_in]. Codes are mostly trits with a few
 * 3s (+2) so the fourth code value is exercised too. */
static uint8_t *make_tensor(size_t n_out, size_t n_in, size_t *out_bytes) {
    const size_t nb   = n_out * n_in / PQ2_0_BLOCK_ELEMS;
    uint8_t     *blob = xmalloc(nb * PQ2_0_BLOCK_BYTES);
    for (size_t b = 0; b < nb; b++) {
        uint8_t       *blk  = blob + b * PQ2_0_BLOCK_BYTES;
        const uint16_t bits = (uint16_t) (0x1800u + (urand() % 0x1000u)); /* ~2^-9..2^-7 */
        blk[0]              = (uint8_t) (bits & 0xff);
        blk[1]              = (uint8_t) (bits >> 8);
        for (size_t j = 0; j < 32; j++) {
            uint8_t byte = 0;
            for (size_t l = 0; l < 4; l++) {
                const uint32_t r    = urand() % 64u;
                const uint8_t  code = r == 0 ? 3 : (uint8_t) (r % 3u);
                byte |= (uint8_t) (code << (2 * l));
            }
            blk[2 + j] = byte;
        }
    }
    *out_bytes = nb * PQ2_0_BLOCK_BYTES;
    return blob;
}

static int check_layout(void) {
    int     fails = 0;
    uint8_t blk[PQ2_0_BLOCK_BYTES];
    blk[0] = 0x00; /* fp16 0.5 = 0x3800, little-endian, scale first */
    blk[1] = 0x38;
    /* element j gets code j % 4 */
    for (size_t j = 0; j < 32; j++) {
        blk[2 + j] = (uint8_t) (0u | (1u << 2) | (2u << 4) | (3u << 6));
    }
    blk[2] = 0xE4 ^ 0xFF; /* first byte flipped: codes 3,2,1,0 for elements 0..3 */
    float out[PQ2_0_BLOCK_ELEMS];
    dequant_pq2_0_row(PQ2_0_BLOCK_ELEMS, blk, out);
    const float first[4] = {1.0f, 0.5f, 0.0f, -0.5f};
    for (size_t j = 0; j < 4; j++) {
        fails += geist_expect(out[j] == first[j], "layout: first byte, element j at bits 2j");
    }
    bool rest = true;
    for (size_t j = 4; j < PQ2_0_BLOCK_ELEMS; j++) {
        const float want = ((float) (j % 4) - 1.0f) * 0.5f;
        rest             = rest && out[j] == want;
    }
    fails += geist_expect(rest, "layout: (code - 1) * d, code 3 = +2");
    return fails;
}

/* fp32 reference: dequant row, dot in double. */
static void ref_fp32(size_t n_out, size_t n_in, const uint8_t *W, const float *x, float *y) {
    float       *row       = xmalloc(n_in * sizeof(float));
    const size_t row_bytes = n_in / PQ2_0_BLOCK_ELEMS * PQ2_0_BLOCK_BYTES;
    for (size_t r = 0; r < n_out; r++) {
        dequant_pq2_0_row(n_in, W + r * row_bytes, row);
        double acc = 0.0;
        for (size_t i = 0; i < n_in; i++) {
            acc += (double) row[i] * (double) x[i];
        }
        y[r] = (float) acc;
    }
    free(row);
}

/* The NEON kernel's arithmetic, restated: absmax int8 activation with
 * round-half-away, integer (code - 1) dot per block, float accumulation
 * over blocks in order. */
static void ref_w2a8(size_t n_out, size_t n_in, const uint8_t *W, const float *x, float *y) {
    float max_abs = 1e-5f;
    for (size_t i = 0; i < n_in; i++) {
        max_abs = fabsf(x[i]) > max_abs ? fabsf(x[i]) : max_abs;
    }
    const float act_scale = 127.0f / max_abs;
    int8_t     *xq        = xmalloc(n_in);
    for (size_t i = 0; i < n_in; i++) {
        const float q  = x[i] * act_scale;
        int32_t     qi = (int32_t) (q < 0.0f ? q - 0.5f : q + 0.5f);
        xq[i]          = (int8_t) (qi > 127 ? 127 : (qi < -128 ? -128 : qi));
    }
    const size_t nb = n_in / PQ2_0_BLOCK_ELEMS;
    for (size_t r = 0; r < n_out; r++) {
        float sum = 0.0f;
        for (size_t b = 0; b < nb; b++) {
            const uint8_t *blk = W + (r * nb + b) * PQ2_0_BLOCK_BYTES;
            const float    d   = fp16_to_fp32((uint16_t) blk[0] | ((uint16_t) blk[1] << 8));
            int32_t        dot = 0;
            for (size_t j = 0; j < PQ2_0_BLOCK_ELEMS; j++) {
                const int code = (blk[2 + j / 4] >> (2 * (j % 4))) & 3;
                dot += (code - 1) * (int32_t) xq[b * PQ2_0_BLOCK_ELEMS + j];
            }
            sum += (float) dot * d;
        }
        y[r] = sum * (max_abs / 127.0f);
    }
    free(xq);
}

/* Worst-case A8 error per row: every activation is off by at most half a
 * quantization step, so |y_a8 - y_fp32| <= sum_i |w_i| * step / 2. */
static bool within_a8_bound(size_t         n_out,
                            size_t         n_in,
                            const uint8_t *W,
                            const float   *x,
                            const float   *y,
                            const float   *yf) {
    float max_abs = 1e-5f;
    for (size_t i = 0; i < n_in; i++) {
        max_abs = fabsf(x[i]) > max_abs ? fabsf(x[i]) : max_abs;
    }
    const double half_step = 0.5 * (double) max_abs / 127.0;
    float       *row       = xmalloc(n_in * sizeof(float));
    const size_t row_bytes = n_in / PQ2_0_BLOCK_ELEMS * PQ2_0_BLOCK_BYTES;
    bool         ok        = true;
    for (size_t r = 0; r < n_out; r++) {
        dequant_pq2_0_row(n_in, W + r * row_bytes, row);
        double l1 = 0.0;
        for (size_t i = 0; i < n_in; i++) {
            l1 += fabs((double) row[i]);
        }
        ok = ok && fabs((double) y[r] - (double) yf[r]) <= l1 * half_step * 1.001 + 1e-6;
    }
    free(row);
    return ok;
}

/* max |a - b| over max |b|. */
static double rel_err(size_t n, const float *a, const float *b) {
    double d = 0.0, m = 0.0;
    for (size_t i = 0; i < n; i++) {
        d = fmax(d, fabs((double) a[i] - (double) b[i]));
        m = fmax(m, fabs((double) b[i]));
    }
    return d / fmax(m, 1e-30);
}

static int check_backend(const char *name, bool x8_policy) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create(name, nullptr, nullptr, &be) != GEIST_OK) {
        printf("%s: not compiled in, skipped\n", name);
        return 0;
    }
    const bool neon = strcmp(name, "cpu_neon") == 0;
    /* x8_policy states what the NEON resolver should install; the other
     * legs have no x8 layout to check. */
    (void) x8_policy;
    int          fails     = 0;
    const size_t n_ins[]   = {128, 5120, 17408};
    const size_t n_outs[]  = {37, 40, 264};
    const size_t m         = 3;
    char         what[160] = {0};
    for (size_t kk = 0; kk < 3 * sizeof n_ins / sizeof n_ins[0]; kk++) {
        const size_t k     = kk / 3;
        const size_t n_out = n_outs[kk % 3];
        const size_t n_in  = n_ins[k];
        size_t       bytes = 0;
        uint8_t     *W     = make_tensor(n_out, n_in, &bytes);
        float       *x     = xmalloc(m * n_in * sizeof(float));
        float       *y     = xmalloc(m * n_out * sizeof(float));
        float       *yf    = xmalloc(m * n_out * sizeof(float));
        float       *yq    = xmalloc(n_out * sizeof(float));
        for (size_t i = 0; i < m * n_in; i++) {
            x[i] = frand();
        }
        x[7] = 6.0f; /* an outlier sets the int8 scale, as real activations do */

        struct geist_weight w = {.raw        = W,
                                 .raw_nbytes = bytes,
                                 .n_in       = (int32_t) n_in,
                                 .n_out      = (int32_t) n_out,
                                 .dtype      = GEIST_DTYPE_PQ2_0};
        snprintf(what, sizeof what, "%s n_in=%zu: resolve", name, n_in);
        fails += geist_expect(be->desc->vtbl->resolve_weight(be, &w) == GEIST_OK, what);
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        if (neon) {
            const bool want_x8 = x8_policy && n_out % 8 == 0;
            snprintf(what,
                     sizeof what,
                     "%s n_in=%zu n_out=%zu: x8 layout %s",
                     name,
                     n_in,
                     n_out,
                     want_x8 ? "installed" : "not installed");
            fails += geist_expect((w.backend_layout == GEIST_W_LAYOUT_PQ2_0_X8_GEMV) == want_x8,
                                  what);
        }
#endif
        if (w.linear_m1 == nullptr || w.linear_mN == nullptr) {
            fails += geist_expect(false, "kernels installed");
            free(W);
            free(x);
            free(y);
            free(yf);
            free(yq);
            continue;
        }

        ref_fp32(n_out, n_in, W, x, yf);
        w.linear_m1(x, &w, be, y);
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        if (neon) {
            ref_w2a8(n_out, n_in, W, x, yq);
            snprintf(what, sizeof what, "%s n_in=%zu: m1 == W2A8 model", name, n_in);
            fails += geist_expect(rel_err(n_out, y, yq) < 1e-5, what);
            snprintf(what, sizeof what, "%s n_in=%zu: m1 within A8 error of fp32", name, n_in);
            fails += geist_expect(within_a8_bound(n_out, n_in, W, x, y, yf), what);
            snprintf(what, sizeof what, "%s n_in=%zu: m1 is the SDOT path", name, n_in);
            fails += geist_expect(rel_err(n_out, y, yf) > 1e-5, what);
        } else
#endif
        {
            (void) neon;
            snprintf(what, sizeof what, "%s n_in=%zu: m1 == fp32", name, n_in);
            fails += geist_expect(rel_err(n_out, y, yf) < 1e-5, what);
        }

        /* m > 1: fp32-accumulating on every backend (trampoline / scalar). */
        w.linear_mN(m, x, &w, be, y);
        for (size_t t = 0; t < m; t++) {
            ref_fp32(n_out, n_in, W, x + t * n_in, yf);
            snprintf(what, sizeof what, "%s n_in=%zu: mN row %zu == fp32", name, n_in, t);
            fails += geist_expect(rel_err(n_out, y + t * n_out, yf) < 1e-4, what);
        }
        if ((w.flags & GEIST_W_AUX_HEAP_OWNED) != 0) {
            void *aux = (void *) w.aux_fp32;
            safe_free(&aux);
        }
        free(W);
        free(x);
        free(y);
        free(yf);
        free(yq);
    }
    geist_backend_destroy(be);
    printf("%s: done\n", name);
    return fails;
}

int main(void) {
    int fails = 0;
    fails += check_layout();
    fails += check_backend("cpu_scalar", false);
    fails += check_backend("cpu_neon", true);
    /* The policy is read at backend create: a fresh backend sees it. */
    setenv("GEIST_PQ2_0_X8_GEMV", "0", 1);
    fails += check_backend("cpu_neon", false);
    if (fails == 0) {
        printf("PASS test_pq2_0_unit\n");
    }
    return fails ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
}

/*
 * test_mxfp4_unit — GGUF MXFP4 decode and the resolved linear kernels
 * (GPT-OSS's experts, #243).
 *
 *   1. dequant_mxfp4_row on hand-built blocks: every 4-bit code at every
 *      one of the 32 positions, with the exponent byte e = 0, 1, 2, 127,
 *      128 and 254 (there only the codes whose value fits a float), against values written out
 * here: the E2M1 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6} with bit 3 the sign, times 2^(e - 127) (e =
 * 0 and 1 are the denormal scales 2^-127 / 2^-126 of ggml's e8m0 "half" table times two). Element j
 * is the low nibble of qs[j], element j + 16 the high one.
 *   2. 4096 random blocks against dequantize_row_mxfp4 of ggml-quants.c,
 *      restated below with ggml_e8m0_to_fp32_half; bit-exact.
 *   3. The storage geometry: 17 bytes per 32 elements (a 2880-wide row is
 *      90 unaligned blocks), a partial block refused, and a weight whose
 *      bytes fall short of its shape refused at resolve.
 *   4. Every CPU backend's resolved m = 1 and m > 1 kernels (m = 3, 8, 17)
 *      against dequantize + double dot, within the bound the W4A8 kernels'
 *      activation rounding allows (derived as in test_x86_iq4_unit: each
 *      int8 activation moves by at most amax / 127 / 2 of its 32-block),
 *      plus 2e-5 of sum |w x| for fp32 accumulation. Every output must be
 *      written. n_in 32, 2880 (GPT-OSS's width) and 4096; n_out 37.
 *      cpu_x86 and cpu_neon must bind kernels of their own, not keep
 *      cpu_scalar's reference.
 *
 * Deterministic, no model needed.
 */
#include "test_helpers.h"

#include <geist_backend.h>
#include <geist_types.h>
#include <geist_weight.h>

#include "heap.h"
#include "quant.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t g_seed = 0x6A09E667u;

static uint32_t urand(void) {
    g_seed ^= g_seed << 13;
    g_seed ^= g_seed >> 17;
    g_seed ^= g_seed << 5;
    return g_seed;
}

static float frand(void) {
    return ((float) (urand() & 0xffff) / 65536.0f) * 2.0f - 1.0f;
}

/* ggml_e8m0_to_fp32_half (ggml-impl.h), as ggml writes it. */
static float ggml_e8m0_half(uint8_t x) {
    uint32_t bits;
    if (x < 2) {
        bits = 0x00200000 << x;
    } else {
        bits = (uint32_t) (x - 1) << 23;
    }
    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

/* dequantize_row_mxfp4 (ggml-quants.c), one block, as ggml writes it. */
static void ggml_ref_block(const uint8_t blk[static MXFP4_BLOCK_BYTES], float y[static 32]) {
    static const int8_t kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    const float         d      = ggml_e8m0_half(blk[0]);
    const uint8_t      *qs     = blk + 1;
    for (int j = 0; j < 16; ++j) {
        const int8_t x0 = kv[qs[j] & 0x0F];
        const int8_t x1 = kv[qs[j] >> 4];
        y[j + 0]        = x0 * d;
        y[j + 16]       = x1 * d;
    }
}

/* The E2M1 value of a code, independent of the tables above. */
static double e2m1(unsigned code) {
    static const double mag[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    return (code & 8u) ? -mag[code & 7u] : mag[code & 7u];
}

static int check_layout(void) {
    static const uint8_t ES[] = {0, 1, 2, 127, 128, 254};
    int                  bad  = 0;
    for (size_t ei = 0; ei < sizeof ES; ei++) {
        const uint8_t e = ES[ei];
        /* 2^(e - 127), e >= 1 a normal float exponent, e = 0 the denormal 2^-127. */
        const double scale = ldexp(1.0, (int) e - 127);
        for (unsigned code = 0; code < 16; code++) {
            /* e = 254: |value| >= 2 is past FLT_MAX (inf in ggml too). */
            if (fabs(e2m1(code) * scale) > (double) FLT_MAX) {
                continue;
            }
            for (unsigned pos = 0; pos < 32; pos++) {
                uint8_t blk[MXFP4_BLOCK_BYTES] = {0};
                blk[0]                         = e;
                /* Every other element gets code 0, value +0. */
                blk[1 + (pos & 15u)] = (uint8_t) (pos < 16 ? code : code << 4);
                float y[32];
                dequant_mxfp4_row(32, blk, y);
                for (unsigned k = 0; k < 32; k++) {
                    const double want = k == pos ? e2m1(code) * scale : 0.0;
                    if ((double) y[k] != want) {
                        if (bad++ < 5) {
                            printf("  e=%u code=%u pos=%u: element %u is %g, want %g\n",
                                   e,
                                   code,
                                   pos,
                                   k,
                                   (double) y[k],
                                   want);
                        }
                    }
                }
            }
        }
    }
    return geist_expect(bad == 0, "every code at every position, six exponents");
}

/* Random tensor [n_out, n_in]; exponents 120..130 (scale 2^-8 .. 2^2). */
static uint8_t *make_tensor(size_t n_out, size_t n_in, size_t *out_bytes) {
    const size_t nb   = n_out * n_in / MXFP4_BLOCK_ELEMS;
    uint8_t     *blob = xmalloc(nb * MXFP4_BLOCK_BYTES);
    for (size_t b = 0; b < nb; b++) {
        uint8_t *blk = blob + b * MXFP4_BLOCK_BYTES;
        blk[0]       = (uint8_t) (120u + urand() % 11u);
        for (size_t i = 1; i < MXFP4_BLOCK_BYTES; i++) {
            blk[i] = (uint8_t) urand();
        }
    }
    *out_bytes = nb * MXFP4_BLOCK_BYTES;
    return blob;
}

static int check_random_vs_ggml(void) {
    constexpr size_t nb    = 4096;
    size_t           bytes = 0;
    uint8_t         *W     = make_tensor(1, nb * MXFP4_BLOCK_ELEMS, &bytes);
    /* Any exponent but 255 (2^127 * 12 overflows; ggml leaves it as NaN-ish). */
    for (size_t b = 0; b < nb; b += 3) {
        W[b * MXFP4_BLOCK_BYTES] = (uint8_t) (urand() % 255u);
    }
    float *out = xmalloc(nb * MXFP4_BLOCK_ELEMS * sizeof(float));
    dequant_mxfp4_row(nb * MXFP4_BLOCK_ELEMS, W, out);
    bool ok = true;
    for (size_t b = 0; b < nb && ok; b++) {
        float ref[32];
        ggml_ref_block(W + b * MXFP4_BLOCK_BYTES, ref);
        ok = memcmp(ref, out + b * MXFP4_BLOCK_ELEMS, sizeof ref) == 0;
        if (!ok) {
            printf("  block %zu differs from ggml\n", b);
        }
    }
    free(W);
    free(out);
    return geist_expect(ok, "4096 random blocks bit-identical to dequantize_row_mxfp4");
}

static int check_geometry(void) {
    int    fails = 0;
    size_t bytes = 0;
    fails += geist_expect(!quant_raw_bytes(GEIST_DTYPE_MXFP4, 4096, &bytes) && bytes == 128 * 17,
                          "quant_raw_bytes: 17 bytes per 32 elements");
    fails += geist_expect(!quant_raw_bytes(GEIST_DTYPE_MXFP4, 2880, &bytes) && bytes == 1530,
                          "quant_raw_bytes: a 2880-wide row is 1530 bytes");
    fails += geist_expect(quant_raw_bytes(GEIST_DTYPE_MXFP4, 4096 + 16, &bytes),
                          "quant_raw_bytes: a partial block is refused");
    struct geist_backend *be = nullptr;
    if (geist_backend_create("cpu_scalar", nullptr, nullptr, &be) == GEIST_OK) {
        size_t              n = 0;
        uint8_t            *W = make_tensor(4, 64, &n);
        struct geist_weight w = {
                .raw = W, .raw_nbytes = n - 1, .n_in = 64, .n_out = 4, .dtype = GEIST_DTYPE_MXFP4};
        fails += geist_expect(be->desc->vtbl->resolve_weight(be, &w) == GEIST_E_FORMAT,
                              "resolve refuses a source one byte short");
        w.raw_nbytes = n;
        fails += geist_expect(be->desc->vtbl->resolve_weight(be, &w) == GEIST_OK,
                              "resolve accepts the exact extent");
        free(W);
        geist_backend_destroy(be);
    }
    return fails;
}

/* Output (t, j) against dequant + double dot, within the W4A8 bound. */
static size_t count_over_bound(
        size_t n_out, size_t n_in, size_t m, const uint8_t *W, const float *x, const float *y) {
    float       *row       = xmalloc(n_in * sizeof(float));
    const size_t row_bytes = n_in / MXFP4_BLOCK_ELEMS * MXFP4_BLOCK_BYTES;
    size_t       over      = 0;
    for (size_t j = 0; j < n_out; j++) {
        dequant_mxfp4_row(n_in, W + j * row_bytes, row);
        for (size_t t = 0; t < m; t++) {
            const float *xt  = x + t * n_in;
            double       acc = 0.0, bound = 0.0, mag = 0.0;
            for (size_t b = 0; b < n_in / 32; b++) {
                float amax = 0.0f;
                for (size_t k = 0; k < 32; k++) {
                    amax = fmaxf(amax, fabsf(xt[b * 32 + k]));
                }
                for (size_t k = 0; k < 32; k++) {
                    const double wv = (double) row[b * 32 + k];
                    acc += wv * (double) xt[b * 32 + k];
                    bound += fabs(wv) * (double) amax / 127.0 / 2.0;
                    mag += fabs(wv * (double) xt[b * 32 + k]);
                }
            }
            const float v = y[t * n_out + j];
            /* Unwritten outputs keep the poison (compared by bits: -ffast-math). */
            uint32_t bits;
            memcpy(&bits, &v, sizeof bits);
            if (bits == 0xFF800001u || !(fabs((double) v - acc) <= bound + 2e-5 * mag + 1e-7)) {
                over++;
            }
        }
    }
    free(row);
    return over;
}

static void poison(size_t n, float *y) {
    const uint32_t bits = 0xFF800001u; /* a NaN no kernel produces */
    for (size_t i = 0; i < n; i++) {
        memcpy(&y[i], &bits, sizeof bits);
    }
}

/* cpu_scalar's kernels for an MXFP4 weight (the reference), to tell a
 * backend that kept them from one with a native kernel. */
static geist_kernel_linear_m1_fn g_ref_m1;
static geist_kernel_linear_mN_fn g_ref_mN;

static int check_backend(const char *name) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create(name, nullptr, nullptr, &be) != GEIST_OK) {
        printf("%s: not compiled in, skipped\n", name);
        return 0;
    }
    int          fails   = 0;
    const size_t n_ins[] = {32, 2880, 4096};
    const size_t n_out   = 37;
    const size_t ms[]    = {3, 8, 17};
    const size_t m_max   = 17;
    char         what[160];
    for (size_t k = 0; k < sizeof n_ins / sizeof n_ins[0]; k++) {
        const size_t n_in  = n_ins[k];
        size_t       bytes = 0;
        uint8_t     *W     = make_tensor(n_out, n_in, &bytes);
        float       *x     = xmalloc(m_max * n_in * sizeof(float));
        float       *y     = xmalloc(m_max * n_out * sizeof(float));
        for (size_t i = 0; i < m_max * n_in; i++) {
            x[i] = frand() * (float) (1 + (i / n_in) % 4);
        }
        struct geist_weight w = {.raw        = W,
                                 .raw_nbytes = bytes,
                                 .n_in       = (int32_t) n_in,
                                 .n_out      = (int32_t) n_out,
                                 .dtype      = GEIST_DTYPE_MXFP4};
        snprintf(what, sizeof what, "%s n_in=%zu: resolve", name, n_in);
        fails += geist_expect(be->desc->vtbl->resolve_weight(be, &w) == GEIST_OK, what);
        if (strcmp(name, "cpu_scalar") == 0) {
            g_ref_m1 = w.linear_m1;
            g_ref_mN = w.linear_mN;
        } else {
            snprintf(what, sizeof what, "%s binds native MXFP4 kernels", name);
            fails += geist_expect(w.linear_m1 != g_ref_m1 && w.linear_mN != g_ref_mN, what);
        }
        if (w.linear_m1 == nullptr || w.linear_mN == nullptr) {
            fails += geist_expect(false, "kernels installed");
        } else {
            poison(n_out, y);
            w.linear_m1(x, &w, be, y);
            snprintf(what, sizeof what, "%s n_in=%zu m=1: within the W4A8 bound", name, n_in);
            fails += geist_expect(count_over_bound(n_out, n_in, 1, W, x, y) == 0, what);
            for (size_t mi = 0; mi < sizeof ms / sizeof ms[0]; mi++) {
                const size_t m = ms[mi];
                poison(m * n_out, y);
                w.linear_mN(m, x, &w, be, y);
                snprintf(what,
                         sizeof what,
                         "%s n_in=%zu m=%zu: within the W4A8 bound",
                         name,
                         n_in,
                         m);
                fails += geist_expect(count_over_bound(n_out, n_in, m, W, x, y) == 0, what);
            }
        }
        if ((w.flags & GEIST_W_AUX_HEAP_OWNED) != 0) {
            void *aux = (void *) w.aux_fp32;
            safe_free(&aux);
        }
        free(W);
        free(x);
        free(y);
    }
    geist_backend_destroy(be);
    printf("%s: done\n", name);
    return fails;
}

int main(void) {
    int fails = 0;
    fails += check_layout();
    fails += check_random_vs_ggml();
    fails += check_geometry();
    fails += check_backend("cpu_scalar");
    fails += check_backend("cpu_x86");
    fails += check_backend("cpu_neon");
    if (fails == 0) {
        printf("PASS test_mxfp4_unit\n");
    }
    return fails ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
}

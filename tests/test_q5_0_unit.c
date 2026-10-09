/*
 * test_q5_0_unit — GGUF Q5_0 decode and the resolved linear kernels (#675).
 *
 *   1. dequant_q5_0_row on hand-built blocks against values written out by
 *      hand: fp16 d first, then qh (element j's fifth bit is bit j of the
 *      little-endian uint32), then qs (element j in the low nibble of
 *      qs[j], j + 16 in the high one); value d * (code - 16). The blocks
 *      set the fifth bit of elements 0, 15, 16 and 31 only — the edges of
 *      both nibble halves and of the qh bytes — with d = 0.5 and -2.0.
 *   2. 4096 random blocks against dequantize_row_q5_0 of ggml-quants.c,
 *      restated below in its own shift form; bit-exact.
 *   3. The storage geometry: 22 bytes per 32 elements, and a weight whose
 *      bytes fall short of its shape is refused at resolve.
 *   4. Every CPU backend's resolved m = 1 and m > 1 kernels against a
 *      dequantize + double-dot reference, at n_in 32 / 4096 / 5120 (one
 *      block; whole and partial cpu_scalar tiles) and an odd n_out.
 *
 * Deterministic, no model needed.
 */
#include "test_helpers.h"
#include "weight_aux.h"

#include <geist_backend.h>
#include <geist_types.h>
#include <geist_weight.h>

#include "heap.h"
#include "quant.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t g_seed = 0x2545F491u;

static uint32_t urand(void) {
    g_seed ^= g_seed << 13;
    g_seed ^= g_seed >> 17;
    g_seed ^= g_seed << 5;
    return g_seed;
}

static float frand(void) {
    return ((float) (urand() & 0xffff) / 65536.0f) * 2.0f - 1.0f;
}

/* dequantize_row_q5_0 (ggml-quants.c), one block, as ggml writes it. */
static void ggml_ref_block(const uint8_t blk[static Q5_0_BLOCK_BYTES], float y[static 32]) {
    const float d  = fp16_to_fp32((uint16_t) (blk[0] | (blk[1] << 8)));
    uint32_t    qh = 0;
    memcpy(&qh, blk + 2, sizeof qh); /* ggml memcpys; every geist host is little-endian */
    const uint8_t *qs = blk + 6;
    for (int j = 0; j < 16; ++j) {
        const uint8_t xh_0 = (uint8_t) (((qh >> (j + 0)) << 4) & 0x10);
        const uint8_t xh_1 = (uint8_t) (((qh >> (j + 12))) & 0x10);
        const int32_t x0   = ((qs[j] & 0x0F) | xh_0) - 16;
        const int32_t x1   = ((qs[j] >> 4) | xh_1) - 16;
        y[j + 0]           = (float) x0 * d;
        y[j + 16]          = (float) x1 * d;
    }
}

/* One block from 32 codes in 0..31 and an fp16 scale. */
static void pack_block(uint16_t d_bits, const uint8_t code[static 32], uint8_t blk[static 22]) {
    blk[0]      = (uint8_t) (d_bits & 0xff);
    blk[1]      = (uint8_t) (d_bits >> 8);
    uint32_t qh = 0;
    for (unsigned j = 0; j < 32; j++) {
        qh |= (uint32_t) ((code[j] >> 4) & 1u) << j;
    }
    for (unsigned b = 0; b < 4; b++) {
        blk[2 + b] = (uint8_t) (qh >> (8 * b));
    }
    for (unsigned j = 0; j < 16; j++) {
        blk[6 + j] = (uint8_t) ((code[j] & 0x0F) | ((code[j + 16] & 0x0F) << 4));
    }
}

static int check_layout(void) {
    int fails = 0;
    /* d = 0.5 (0x3800) and d = -2.0 (0xC000). */
    const uint16_t ds[2]    = {0x3800, 0xC000};
    const float    dvals[2] = {0.5f, -2.0f};
    for (size_t k = 0; k < 2; k++) {
        uint8_t blk[Q5_0_BLOCK_BYTES];
        blk[0] = (uint8_t) (ds[k] & 0xff);
        blk[1] = (uint8_t) (ds[k] >> 8);
        /* qh: bits 0, 15, 16, 31 — elements 0 and 15 (low nibbles, first
         * and last), 16 and 31 (high nibbles, first and last). */
        blk[2] = 0x01;
        blk[3] = 0x80;
        blk[4] = 0x01;
        blk[5] = 0x80;
        /* qs[j]: low nibble j, high nibble 15 - j. */
        for (unsigned j = 0; j < 16; j++) {
            blk[6 + j] = (uint8_t) (j | ((15u - j) << 4));
        }
        float out[Q5_0_BLOCK_ELEMS];
        dequant_q5_0_row(Q5_0_BLOCK_ELEMS, blk, out);
        bool ok = true;
        for (unsigned j = 0; j < 32; j++) {
            const unsigned nib  = j < 16 ? j : 15u - (j - 16u);
            const unsigned high = j == 0 || j == 15 || j == 16 || j == 31;
            const int      code = (int) (nib | (high << 4));
            const float    want = dvals[k] * (float) (code - 16);
            if (out[j] != want) {
                printf("  d=%g element %u: got %g want %g\n", (double) dvals[k], j, out[j], want);
                ok = false;
            }
        }
        /* The edges spelled out: element 0 is code 16 -> 0; element 15 is
         * code 31 -> +15 d; element 16 is code 31 -> +15 d; element 31 is
         * code 16 -> 0; element 1 has no fifth bit: code 1 -> -15 d. */
        ok = ok && out[0] == 0.0f && out[15] == 15.0f * dvals[k] && out[16] == 15.0f * dvals[k] &&
             out[31] == 0.0f && out[1] == -15.0f * dvals[k];
        float ref[32];
        ggml_ref_block(blk, ref);
        ok = ok && memcmp(out, ref, sizeof ref) == 0;
        char what[96];
        snprintf(what, sizeof what, "layout d=%g: hand-written values and ggml", (double) dvals[k]);
        fails += geist_expect(ok, what);
    }
    /* Every code in turn: element j gets code (j + s) % 32, so each of the
     * 32 codes visits every position over the 32 shifts. */
    bool all = true;
    for (unsigned s = 0; s < 32; s++) {
        uint8_t code[32];
        for (unsigned j = 0; j < 32; j++) {
            code[j] = (uint8_t) ((j + s) % 32u);
        }
        uint8_t blk[Q5_0_BLOCK_BYTES];
        pack_block(0x3800, code, blk);
        float out[32];
        dequant_q5_0_row(32, blk, out);
        for (unsigned j = 0; j < 32; j++) {
            all = all && out[j] == 0.5f * (float) ((int) code[j] - 16);
        }
    }
    fails += geist_expect(all, "layout: every code at every position, (code - 16) * d");
    return fails;
}

/* Random tensor [n_out, n_in]; scales ~2^-7..2^-5, both signs. */
static uint8_t *make_tensor(size_t n_out, size_t n_in, size_t *out_bytes) {
    const size_t nb   = n_out * n_in / Q5_0_BLOCK_ELEMS;
    uint8_t     *blob = xmalloc(nb * Q5_0_BLOCK_BYTES);
    for (size_t b = 0; b < nb; b++) {
        uint8_t *blk = blob + b * Q5_0_BLOCK_BYTES;
        for (size_t i = 2; i < Q5_0_BLOCK_BYTES; i++) {
            blk[i] = (uint8_t) urand();
        }
        const uint16_t bits =
                (uint16_t) (0x2000u + (urand() % 0x0800u)) | (uint16_t) ((urand() & 1u) << 15);
        blk[0] = (uint8_t) (bits & 0xff);
        blk[1] = (uint8_t) (bits >> 8);
    }
    *out_bytes = nb * Q5_0_BLOCK_BYTES;
    return blob;
}

static int check_random_vs_ggml(void) {
    constexpr size_t nb    = 4096;
    size_t           bytes = 0;
    uint8_t         *W     = make_tensor(1, nb * Q5_0_BLOCK_ELEMS, &bytes);
    /* Any finite scale, subnormals included (the build is -ffast-math, so
     * inf / NaN are out of contract). */
    for (size_t b = 0; b < nb; b += 7) {
        uint16_t bits = (uint16_t) urand();
        if ((bits & 0x7C00u) == 0x7C00u) {
            bits ^= 0x4000u;
        }
        W[b * Q5_0_BLOCK_BYTES]     = (uint8_t) (bits & 0xff);
        W[b * Q5_0_BLOCK_BYTES + 1] = (uint8_t) (bits >> 8);
    }
    float *out = xmalloc(nb * Q5_0_BLOCK_ELEMS * sizeof(float));
    dequant_q5_0_row(nb * Q5_0_BLOCK_ELEMS, W, out);
    bool ok = true;
    for (size_t b = 0; b < nb && ok; b++) {
        float ref[32];
        ggml_ref_block(W + b * Q5_0_BLOCK_BYTES, ref);
        ok = memcmp(ref, out + b * Q5_0_BLOCK_ELEMS, sizeof ref) == 0;
        if (!ok) {
            printf("  block %zu differs from ggml\n", b);
        }
    }
    free(W);
    free(out);
    return geist_expect(ok, "4096 random blocks bit-identical to dequantize_row_q5_0");
}

static int check_geometry(void) {
    int    fails = 0;
    size_t bytes = 0;
    fails += geist_expect(!quant_raw_bytes(GEIST_DTYPE_Q5_0, 4096, &bytes) && bytes == 128 * 22,
                          "quant_raw_bytes: 22 bytes per 32 elements");
    fails += geist_expect(quant_raw_bytes(GEIST_DTYPE_Q5_0, 4096 + 16, &bytes),
                          "quant_raw_bytes: a partial block is refused");
    struct geist_backend *be = nullptr;
    if (geist_backend_create("cpu_scalar", nullptr, nullptr, &be) == GEIST_OK) {
        size_t              n = 0;
        uint8_t            *W = make_tensor(4, 64, &n);
        struct geist_weight w = {
                .raw = W, .raw_nbytes = n - 1, .n_in = 64, .n_out = 4, .dtype = GEIST_DTYPE_Q5_0};
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

/* dequant row, dot in double. */
static void ref_fp32(size_t n_out, size_t n_in, const uint8_t *W, const float *x, float *y) {
    float       *row       = xmalloc(n_in * sizeof(float));
    const size_t row_bytes = n_in / Q5_0_BLOCK_ELEMS * Q5_0_BLOCK_BYTES;
    for (size_t r = 0; r < n_out; r++) {
        dequant_q5_0_row(n_in, W + r * row_bytes, row);
        double acc = 0.0;
        for (size_t i = 0; i < n_in; i++) {
            acc += (double) row[i] * (double) x[i];
        }
        y[r] = (float) acc;
    }
    free(row);
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

static int check_backend(const char *name) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create(name, nullptr, nullptr, &be) != GEIST_OK) {
        printf("%s: not compiled in, skipped\n", name);
        return 0;
    }
    int          fails   = 0;
    const size_t n_ins[] = {32, 4096, 5120};
    const size_t n_out   = 37;
    const size_t ms[]    = {3, 20};
    char         what[160];
    for (size_t k = 0; k < sizeof n_ins / sizeof n_ins[0]; k++) {
        const size_t n_in  = n_ins[k];
        size_t       bytes = 0;
        uint8_t     *W     = make_tensor(n_out, n_in, &bytes);
        float       *x     = xmalloc(ms[1] * n_in * sizeof(float));
        float       *y     = xmalloc(ms[1] * n_out * sizeof(float));
        float       *yf    = xmalloc(n_out * sizeof(float));
        for (size_t i = 0; i < ms[1] * n_in; i++) {
            x[i] = frand() * (float) (1 + (i / n_in) % 4);
        }
        struct geist_weight w = {.raw        = W,
                                 .raw_nbytes = bytes,
                                 .n_in       = (int32_t) n_in,
                                 .n_out      = (int32_t) n_out,
                                 .dtype      = GEIST_DTYPE_Q5_0};
        snprintf(what, sizeof what, "%s n_in=%zu: resolve", name, n_in);
        fails += geist_expect(be->desc->vtbl->resolve_weight(be, &w) == GEIST_OK, what);
        if (w.linear_m1 == nullptr || w.linear_mN == nullptr) {
            fails += geist_expect(false, "kernels installed");
        } else {
            ref_fp32(n_out, n_in, W, x, yf);
            w.linear_m1(x, &w, be, y);
            snprintf(what, sizeof what, "%s n_in=%zu: m1 == dequant + fp32", name, n_in);
            fails += geist_expect(rel_err(n_out, y, yf) < 1e-5, what);
            for (size_t mi = 0; mi < sizeof ms / sizeof ms[0]; mi++) {
                const size_t m = ms[mi];
                w.linear_mN(m, x, &w, be, y);
                bool ok = true;
                for (size_t t = 0; t < m; t++) {
                    ref_fp32(n_out, n_in, W, x + t * n_in, yf);
                    ok = ok && rel_err(n_out, y + t * n_out, yf) < 1e-5;
                }
                snprintf(what,
                         sizeof what,
                         "%s n_in=%zu m=%zu: mN == dequant + fp32",
                         name,
                         n_in,
                         m);
                fails += geist_expect(ok, what);
            }
        }
        weight_aux_free(&w);
        free(W);
        free(x);
        free(y);
        free(yf);
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
        printf("PASS test_q5_0_unit\n");
    }
    return fails ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
}

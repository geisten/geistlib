/* Vulkan elementwise / layout ops added for the qwen35 family, each compared
 * with the host reference formula:
 *
 *   - rope_apply with a rotated prefix narrower than head_dim (partial
 *     rotary, 64 of 256) and with full rotation (regression);
 *   - silu, fused silu_mul (SwiGLU epilogue, in place), sigmoid_mul (the
 *     attention output gate, in place) and attn_qgate_split.
 *
 * Every operand lives in a KV_CACHE-role buffer, which this backend keeps in
 * VRAM without a host mapping: an op that silently fell back to a host loop
 * would fail with "bad inputs" instead of passing by accident.
 *
 * SKIPs (exit 77) when no Vulkan runtime/device is present. */
#include "test_helpers.h"

#include "gemma4_kernels.h"

#include <geist.h>
#include <geist_backend.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
#define check(ok, what) (g_fail |= geist_expect((ok), (what)))

static struct geist_backend *g_be;

static struct geist_buffer *dev_buf(const float *src, size_t n) {
    const struct geist_backend_vtbl *v = g_be->desc->vtbl;
    struct geist_buffer             *b = nullptr;
    if (v->buffer_create(g_be, n * sizeof(float), GEIST_BUFFER_KV_CACHE, GEIST_MEMORY_AUTO, &b) !=
                GEIST_OK ||
        (src != nullptr &&
         v->buffer_upload(b, n * sizeof(float), (const uint8_t *) src) != GEIST_OK)) {
        return nullptr;
    }
    return b;
}

static struct geist_tensor
view(struct geist_buffer *b, int nd, int64_t d0, int64_t d1, int64_t d2) {
    struct geist_tensor t = {.buffer = b, .dtype = GEIST_DTYPE_F32, .layout = GEIST_LAYOUT_DENSE};
    t.ndim                = nd;
    t.shape[0]            = d0;
    t.shape[1]            = d1;
    t.shape[2]            = d2;
    if (nd == 1) {
        t.stride[0] = 1;
    } else if (nd == 2) {
        t.stride[0] = d1;
        t.stride[1] = 1;
    } else {
        t.stride[0] = d1 * d2;
        t.stride[1] = d2;
        t.stride[2] = 1;
    }
    return t;
}

static float *fill(size_t n, float freq, float amp) {
    float *p = malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        p[i] = sinf((float) i * freq + 0.3f) * amp;
    }
    return p;
}

static double max_abs(const float *a, const float *b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double d = fabs((double) a[i] - (double) b[i]);
        if (d > m) {
            m = d;
        }
    }
    return m;
}

static bool download(struct geist_buffer *b, float *dst, size_t n) {
    return g_be->desc->vtbl->buffer_download(n * sizeof(float), (uint8_t *) dst, b) == GEIST_OK;
}

static float silu_ref(float v) {
    const float e = expf(-fabsf(v));
    return v >= 0.0f ? v / (1.0f + e) : (v * e) / (1.0f + e);
}

static void test_rope(size_t seq, size_t heads, size_t hd, size_t rot, const char *name) {
    float *x   = fill(seq * heads * hd, 0.21f, 1.0f);
    float *cs  = fill(seq * rot, 0.13f, 1.0f);
    float *sn  = fill(seq * rot, 0.17f, 1.0f);
    float *ref = malloc(seq * heads * hd * sizeof(float));
    float *got = malloc(seq * heads * hd * sizeof(float));
    memcpy(ref, x, seq * heads * hd * sizeof(float));
    rope_apply(seq, heads, hd, rot, ref, cs, sn);

    struct geist_buffer *bx = dev_buf(x, seq * heads * hd), *bc = dev_buf(cs, seq * rot),
                        *bs = dev_buf(sn, seq * rot);
    check(bx && bc && bs, "rope buffers");
    struct geist_tensor     tx = view(bx, 3, (int64_t) seq, (int64_t) heads, (int64_t) hd);
    struct geist_tensor     tc = view(bc, 2, (int64_t) seq, (int64_t) rot, 0);
    struct geist_tensor     ts = view(bs, 2, (int64_t) seq, (int64_t) rot, 0);
    const enum geist_status s  = g_be->desc->prims->rope_apply(g_be, &tx, &tc, &ts);
    check(s == GEIST_OK, "rope_apply dispatch");
    check(download(bx, got, seq * heads * hd), "rope download");
    const double e = max_abs(got, ref, seq * heads * hd);
    printf("  rope %-22s max_abs %.2e\n", name, e);
    check(e < 2e-6, name);
    free(x);
    free(cs);
    free(sn);
    free(ref);
    free(got);
}

static void test_elementwise(size_t rows, size_t cols) {
    const size_t n = rows * cols;
    float *a = fill(n, 0.19f, 6.0f), *b = fill(n, 0.11f, 3.0f), *got = malloc(n * sizeof(float));
    float *ref = malloc(n * sizeof(float));

    struct geist_buffer *ba = dev_buf(a, n), *bb = dev_buf(b, n), *by = dev_buf(nullptr, n);
    check(ba && bb && by, "ew buffers");
    struct geist_tensor ta = view(ba, 2, (int64_t) rows, (int64_t) cols, 0),
                        tb = view(bb, 2, (int64_t) rows, (int64_t) cols, 0),
                        ty = view(by, 2, (int64_t) rows, (int64_t) cols, 0);

    /* silu (out of place) */
    for (size_t i = 0; i < n; i++) {
        ref[i] = silu_ref(a[i]);
    }
    check(g_be->desc->prims->silu(g_be, &ta, &ty) == GEIST_OK, "silu dispatch");
    check(download(by, got, n), "silu download");
    double e = max_abs(got, ref, n);
    printf("  silu        %zux%zu  max_abs %.2e\n", rows, cols, e);
    check(e < 2e-6, "silu");

    /* silu_mul, in place over a */
    const struct geist_backend_fused *f = geist_backend_fused_tbl(g_be);
    check(f->silu_mul != nullptr, "silu_mul present");
    for (size_t i = 0; i < n; i++) {
        ref[i] = silu_ref(a[i]) * b[i];
    }
    check(f->silu_mul(g_be, &ta, &tb, &ta) == GEIST_OK, "silu_mul dispatch");
    check(download(ba, got, n), "silu_mul download");
    e = max_abs(got, ref, n);
    printf("  silu_mul    %zux%zu  max_abs %.2e (in place)\n", rows, cols, e);
    check(e < 2e-5, "silu_mul");

    /* sigmoid_mul, in place over a (re-upload the original a) */
    check(g_be->desc->vtbl->buffer_upload(ba, n * sizeof(float), (const uint8_t *) a) == GEIST_OK,
          "re-upload");
    check(f->sigmoid_mul != nullptr, "sigmoid_mul present");
    for (size_t i = 0; i < n; i++) {
        ref[i] = a[i] * (1.0f / (1.0f + expf(-b[i])));
    }
    check(f->sigmoid_mul(g_be, &ta, &tb, &ta) == GEIST_OK, "sigmoid_mul dispatch");
    check(download(ba, got, n), "sigmoid_mul download");
    e = max_abs(got, ref, n);
    printf("  sigmoid_mul %zux%zu  max_abs %.2e (in place)\n", rows, cols, e);
    check(e < 2e-6, "sigmoid_mul");

    g_be->desc->vtbl->buffer_destroy(g_be, ba);
    g_be->desc->vtbl->buffer_destroy(g_be, bb);
    g_be->desc->vtbl->buffer_destroy(g_be, by);
    free(a);
    free(b);
    free(got);
    free(ref);
}

static void test_qgate(size_t rows, size_t heads, size_t hd) {
    const size_t q_out = heads * hd;
    float       *joint = fill(rows * 2 * q_out, 0.07f, 2.0f);
    float       *q_ref = malloc(rows * q_out * sizeof(float)),
          *g_ref       = malloc(rows * q_out * sizeof(float));
    float *q_got       = malloc(rows * q_out * sizeof(float)),
          *g_got       = malloc(rows * q_out * sizeof(float));
    for (size_t t = 0; t < rows; t++) {
        for (size_t h = 0; h < heads; h++) {
            const float *src = joint + t * 2 * q_out + h * 2 * hd;
            memcpy(q_ref + t * q_out + h * hd, src, hd * sizeof(float));
            memcpy(g_ref + t * q_out + h * hd, src + hd, hd * sizeof(float));
        }
    }
    struct geist_buffer *bj = dev_buf(joint, rows * 2 * q_out),
                        *bq = dev_buf(nullptr, rows * q_out), *bg = dev_buf(nullptr, rows * q_out);
    check(bj && bq && bg, "qgate buffers");
    struct geist_tensor               tj = view(bj, 2, (int64_t) rows, (int64_t) (2 * q_out), 0);
    struct geist_tensor               tq = view(bq, 2, (int64_t) rows, (int64_t) q_out, 0);
    struct geist_tensor               tg = view(bg, 2, (int64_t) rows, (int64_t) q_out, 0);
    const struct geist_backend_fused *f  = geist_backend_fused_tbl(g_be);
    check(f->attn_qgate_split != nullptr, "attn_qgate_split present");
    check(f->attn_qgate_split(g_be, &tj, heads, hd, &tq, &tg) == GEIST_OK, "qgate dispatch");
    check(download(bq, q_got, rows * q_out) && download(bg, g_got, rows * q_out), "qgate download");
    const double eq = max_abs(q_got, q_ref, rows * q_out), eg = max_abs(g_got, g_ref, rows * q_out);
    printf("  qgate_split %zu rows x %zu heads x %zu  q %.2e gate %.2e\n", rows, heads, hd, eq, eg);
    check(eq == 0.0 && eg == 0.0, "qgate_split is an exact copy");
    g_be->desc->vtbl->buffer_destroy(g_be, bj);
    g_be->desc->vtbl->buffer_destroy(g_be, bq);
    g_be->desc->vtbl->buffer_destroy(g_be, bg);
    free(joint);
    free(q_ref);
    free(g_ref);
    free(q_got);
    free(g_got);
}

/* prims->attention against the host reference attention_mqa_causal_kv, F32
 * KV. Shapes mirror the transformer families that use the generic (non
 * gemma) attention path. */
static void test_attention(size_t      n_q,
                           size_t      n_kv,
                           size_t      q_off,
                           size_t      qh,
                           size_t      kvh,
                           size_t      hd,
                           size_t      sliding,
                           const char *name) {
    float *q = fill(n_q * qh * hd, 0.031f, 0.6f), *k = fill(n_kv * kvh * hd, 0.023f, 0.6f);
    float *v   = fill(n_kv * kvh * hd, 0.017f, 1.0f);
    float *ref = malloc(n_q * qh * hd * sizeof(float)),
          *got = malloc(n_q * qh * hd * sizeof(float));
    attention_mqa_causal_kv(n_q, n_kv, q_off, qh, kvh, hd, sliding, q, k, v, ref);
    struct geist_buffer *bq = dev_buf(q, n_q * qh * hd), *bk = dev_buf(k, n_kv * kvh * hd),
                        *bv = dev_buf(v, n_kv * kvh * hd), *bo = dev_buf(nullptr, n_q * qh * hd);
    check(bq && bk && bv && bo, "attention buffers");
    struct geist_tensor tq = view(bq, 3, (int64_t) n_q, (int64_t) qh, (int64_t) hd);
    struct geist_tensor tk = view(bk, 3, (int64_t) n_kv, (int64_t) kvh, (int64_t) hd);
    struct geist_tensor tv = view(bv, 3, (int64_t) n_kv, (int64_t) kvh, (int64_t) hd);
    struct geist_tensor to = view(bo, 3, (int64_t) n_q, (int64_t) qh, (int64_t) hd);
    check(g_be->desc->prims->attention(g_be, &tq, &tk, &tv, q_off, sliding, &to) == GEIST_OK,
          "attention dispatch");
    check(download(bo, got, n_q * qh * hd), "attention download");
    const double e = max_abs(got, ref, n_q * qh * hd);
    printf("  attention %-28s max_abs %.2e\n", name, e);
    check(e < 1e-4, name);
    g_be->desc->vtbl->buffer_destroy(g_be, bq);
    g_be->desc->vtbl->buffer_destroy(g_be, bk);
    g_be->desc->vtbl->buffer_destroy(g_be, bv);
    g_be->desc->vtbl->buffer_destroy(g_be, bo);
    free(q);
    free(k);
    free(v);
    free(ref);
    free(got);
}

int main(void) {
    enum geist_status s = geist_backend_create("vulkan", nullptr, nullptr, &g_be);
    if (s == GEIST_E_NOT_FOUND || s == GEIST_E_UNSUPPORTED) {
        GEIST_SKIP("Vulkan backend unavailable");
    }
    if (s != GEIST_OK || g_be == nullptr) {
        fprintf(stderr, "FAIL: Vulkan create: %s\n", geist_last_create_error());
        return GEIST_TEST_FAIL;
    }
    test_rope(5, 3, 256, 64, "partial 64/256");
    test_rope(4, 2, 128, 128, "full 128/128");
    test_rope(3, 2, 96, 32, "partial 32/96");
    test_elementwise(7, 129);
    test_elementwise(1, 4096);
    test_qgate(5, 4, 64);
    test_qgate(1, 24, 256);
    test_attention(22, 22, 0, 8, 2, 256, 0, "qwen35 prefill 22 (8/2x256)");
    test_attention(1, 23, 22, 8, 2, 256, 0, "qwen35 decode kv=23");
    test_attention(9, 40, 31, 8, 2, 256, 0, "chunked prefill q_off=31");
    test_attention(16, 16, 0, 16, 16, 128, 0, "MHA 16x128");
    test_attention(1, 300, 299, 8, 2, 256, 0, "decode kv=300 (flash path)");
    geist_backend_destroy(g_be);
    if (g_fail == 0) {
        printf("PASS: Vulkan qwen35 ops (partial rope, silu, silu_mul, sigmoid_mul, "
               "qgate_split)\n");
    }
    return g_fail == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}

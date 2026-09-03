/*
 * test_backend_vs_direct_int — verifies backend->vtbl->linear on real
 * Q4_K weight data matches the direct gguf_quant.c kernel.
 *
 * Phase B-4e verification: proves that lm.c's per-layer Q4_K matmul calls
 * can be replaced one-for-one with backend->vtbl->linear without changing
 * output. The actual lm.c surgery to route hot-path calls through the
 * vtable is left for a follow-up commit; this test demonstrates that the
 * architectural foundation is sound.
 *
 * SKIPs cleanly if no GGUF model is available (set GEIST_GGUF_PATH).
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>

#include "quant.h"
#include "gguf_reader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int
check_close(const float *a, const float *b, size_t n, float rtol, float atol, const char *what) {
    ptrdiff_t bad = geist_fp32_close_array(a, b, n, rtol, atol);
    if (bad < 0) {
        return 0;
    }
    fprintf(stderr,
            "FAIL %s: idx %td: direct=%.6f vtable=%.6f diff=%g\n",
            what,
            bad,
            (double) a[bad],
            (double) b[bad],
            (double) fabsf(a[bad] - b[bad]));
    return 1;
}

int main(void) {
    GEIST_REQUIRE_GGUF(model_path);

    /* Open GGUF and find a Q4_K weight tensor. Gemma 4 Q4_K_M has many. */
    const char      *err = nullptr;
    struct gguf_ctx *ctx = gguf_open(model_path, &err);
    if (ctx == nullptr) {
        fprintf(stderr, "gguf_open failed: %s\n", err != nullptr ? err : "(no detail)");
        return GEIST_TEST_ERROR;
    }

    /* Pick a representative Q4_K tensor — Gemma 4's first-layer
     * attn_q.weight is 2048 x 1536, which is 256-aligned for Q4_K. */
    const struct gguf_tensor_t *t = gguf_get_tensor(ctx, "blk.0.attn_q.weight");
    if (t == nullptr || t->dtype != GGUF_TYPE_Q4_K) {
        fprintf(stderr,
                "blk.0.attn_q.weight not present or not Q4_K (got dtype=%d)\n",
                t == nullptr ? -1 : (int) t->dtype);
        gguf_close(ctx);
        return t == nullptr ? GEIST_TEST_SKIP : GEIST_TEST_FAIL;
    }
    if (t->n_dims != 2) {
        gguf_close(ctx);
        return GEIST_TEST_FAIL;
    }
    const size_t n_in  = t->dims[0];
    const size_t n_out = t->dims[1];
    if (n_in % Q4_K_BLOCK_ELEMS != 0) {
        gguf_close(ctx);
        return GEIST_TEST_FAIL;
    }
    printf("found %s — Q4_K (%zu x %zu)\n", "blk.0.attn_q.weight", n_in, n_out);

    /* Generate a deterministic input vector. */
    float *x_in = aligned_alloc(64, n_in * sizeof(float));
    if (x_in == nullptr) {
        gguf_close(ctx);
        return GEIST_TEST_ERROR;
    }
    uint32_t seed = 0xABCDEF01;
    for (size_t i = 0; i < n_in; i++) {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        x_in[i] = (float) ((int32_t) seed) * (1.0f / (float) INT32_MAX);
    }

    /* ---- Direct path: legacy gguf_quant.c kernel ---- */
    float *y_direct = aligned_alloc(64, n_out * sizeof(float));
    if (y_direct == nullptr) {
        free(x_in);
        gguf_close(ctx);
        return GEIST_TEST_ERROR;
    }
    linear_q4k_decode_w4a8(n_in, n_out, x_in, t->data, y_direct);

    /* ---- Vtable path: cpu_neon backend->vtbl->linear ---- */
    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create("cpu_neon", nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        fprintf(stderr, "cpu_neon create failed (build with BACKENDS=\"cpu_scalar cpu_neon\")\n");
        free(x_in);
        free(y_direct);
        gguf_close(ctx);
        return GEIST_TEST_FAIL;
    }

    struct geist_buffer *bx = nullptr, *bw = nullptr, *by = nullptr;
    s = be->desc->vtbl->buffer_create(
            be, n_in * sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &bx);
    if (s != GEIST_OK)
        goto fail;
    s = be->desc->vtbl->buffer_create(be, t->nbytes, GEIST_BUFFER_WEIGHT, GEIST_MEMORY_AUTO, &bw);
    if (s != GEIST_OK)
        goto fail;
    s = be->desc->vtbl->buffer_create(
            be, n_out * sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &by);
    if (s != GEIST_OK)
        goto fail;

    s = be->desc->vtbl->buffer_upload(bx, n_in * sizeof(float), (const uint8_t *) x_in);
    if (s != GEIST_OK)
        goto fail;
    s = be->desc->vtbl->buffer_upload(bw, t->nbytes, (const uint8_t *) t->data);
    if (s != GEIST_OK)
        goto fail;

    /* P2.e: route through the resolver, not v->linear (which has been
     * removed). Build a geist_weight that wraps the same raw bytes,
     * let cpu_neon's resolve_weight install the kernel function
     * pointer, then call it directly. */
    void *w_host = be->desc->vtbl->buffer_map(bw);
    if (w_host == nullptr)
        goto fail;
    struct geist_weight wkr = {
            .raw        = w_host,
            .raw_nbytes = t->nbytes,
            .n_in       = (int32_t) n_in,
            .n_out      = (int32_t) n_out,
            .dtype      = (uint16_t) GEIST_DTYPE_Q4_K,
    };
    s = be->desc->vtbl->resolve_weight(be, &wkr);
    if (s != GEIST_OK || wkr.linear_m1 == nullptr) {
        fprintf(stderr, "resolve_weight(Q4_K) failed: %s\n", geist_status_to_string(s));
        be->desc->vtbl->buffer_unmap(bw);
        goto fail;
    }
    void *x_host = be->desc->vtbl->buffer_map(bx);
    void *y_host = be->desc->vtbl->buffer_map(by);
    if (x_host == nullptr || y_host == nullptr) {
        be->desc->vtbl->buffer_unmap(bw);
        goto fail;
    }
    wkr.linear_m1((const float *) x_host, &wkr, be, (float *) y_host);
    be->desc->vtbl->buffer_unmap(bx);
    be->desc->vtbl->buffer_unmap(bw);
    be->desc->vtbl->buffer_unmap(by);

    float *y_vtable = aligned_alloc(64, n_out * sizeof(float));
    if (y_vtable == nullptr)
        goto fail;
    s = be->desc->vtbl->buffer_download(n_out * sizeof(float), (uint8_t *) y_vtable, by);
    if (s != GEIST_OK) {
        free(y_vtable);
        goto fail;
    }

    int fails = check_close(
            y_direct, y_vtable, n_out, 1e-6f, 1e-6f, "vtable-Q4_K vs direct-Q4_K (cpu_neon)");

    /* ---- Cleanup ---- */
    be->desc->vtbl->buffer_destroy(be, bx);
    be->desc->vtbl->buffer_destroy(be, bw);
    be->desc->vtbl->buffer_destroy(be, by);
    geist_backend_destroy(be);
    free(x_in);
    free(y_direct);
    free(y_vtable);
    gguf_close(ctx);

    if (fails == 0) {
        printf("PASS: backend->vtbl->linear == linear_q4k_decode_w4a8 on real GGUF "
               "tensor (%zu outputs)\n",
               n_out);
        return GEIST_TEST_PASS;
    }
    return GEIST_TEST_FAIL;

fail:
    if (bx)
        be->desc->vtbl->buffer_destroy(be, bx);
    if (bw)
        be->desc->vtbl->buffer_destroy(be, bw);
    if (by)
        be->desc->vtbl->buffer_destroy(be, by);
    geist_backend_destroy(be);
    free(x_in);
    free(y_direct);
    gguf_close(ctx);
    return GEIST_TEST_FAIL;
}

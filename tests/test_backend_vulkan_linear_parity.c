/*
 * test_backend_vulkan_linear_parity — numerical parity gate for the Vulkan
 * linear path (Phase 2): resolve_weight + linear_m1/linear_mN for Q4_K,
 * Q5_K, Q6_K, Q4_0, Q4_1, Q8_0 and F32 weights (plus fused->linear_t, the
 * device-resident path the engine actually runs), compared against the
 * cpu_scalar resolver on the SAME weight bytes. cpu_scalar dequantizes with an independent
 * implementation (src/formats/gguf), so agreement means the GLSL dequant
 * and the full dispatch chain (VRAM upload, registry, staging, shader) are
 * correct.
 *
 * Weight blobs are random bytes with the f16 super-block scales pinned to
 * finite values (random f16 can be NaN/Inf, which would poison the compare).
 *
 * Tolerance: GPU sums in f32, reference in double — 1e-3 relative on
 * n_in=512 dots is generous headroom; a dequant bug is orders of magnitude.
 *
 * SKIPs (exit 0) when no Vulkan runtime/device is present.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
#define check(ok, what) (g_fail |= geist_expect((ok), (what)))

/* Block geometry of every quantized dtype under test: elements per block,
 * bytes per block, and which leading bytes are f16 scale fields to pin. */
struct qfmt {
    int    dtype;
    size_t elems, bytes;
    int    scale_bytes; /* leading bytes: f16 fields pinned to finite values */
};

static const struct qfmt QF[] = {
        {GEIST_DTYPE_Q4_K, 256, 144, 4},
        {GEIST_DTYPE_Q5_K, 256, 176, 4},
        {GEIST_DTYPE_Q6_K, 256, 210, -2}, /* d is the TRAILING f16 in Q6_K */
        {GEIST_DTYPE_Q4_0, 32, 18, 2},
        {GEIST_DTYPE_Q4_1, 32, 20, 4},
        {GEIST_DTYPE_Q8_0, 32, 34, 2},
};

static const struct qfmt *qfmt_of(int dtype) {
    for (size_t i = 0; i < sizeof QF / sizeof QF[0]; i++) {
        if (QF[i].dtype == dtype) {
            return &QF[i];
        }
    }
    return nullptr;
}

static uint32_t rng_state = 0x12345678u;
static uint8_t  rng_u8(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (uint8_t) (rng_state >> 24);
}

static size_t weight_bytes(int dtype, size_t n_in, size_t n_out) {
    if (dtype == GEIST_DTYPE_F32) {
        return n_in * n_out * sizeof(float);
    }
    const struct qfmt *q = qfmt_of(dtype);
    return n_out * (n_in / q->elems) * q->bytes;
}

/* Random-but-finite quant blob: random bytes, f16 scale fields pinned. */
static void fill_blob(uint8_t *dst, size_t n_in, size_t n_out, int dtype) {
    const struct qfmt *q   = qfmt_of(dtype);
    const size_t       bpr = n_in / q->elems;
    for (size_t r = 0; r < n_out; r++) {
        for (size_t b = 0; b < bpr; b++) {
            uint8_t *blk = dst + (r * bpr + b) * q->bytes;
            for (size_t i = 0; i < q->bytes; i++) {
                blk[i] = rng_u8();
            }
            if (q->scale_bytes < 0) {
                blk[q->bytes - 2] = 0x00; /* d = fp16(1.0) */
                blk[q->bytes - 1] = 0x3C;
                continue;
            }
            blk[0] = 0x00; /* d    = fp16(1.0) */
            blk[1] = 0x3C;
            if (q->scale_bytes >= 4) {
                blk[2] = 0x00; /* dmin / m = fp16(0.5) */
                blk[3] = 0x38;
            }
        }
    }
}

static void run_case_tol(struct geist_backend *vk,
                         struct geist_backend *ref,
                         int                   dtype,
                         const char           *name,
                         size_t                n_in,
                         size_t                n_out,
                         size_t                m,
                         double                tol) {
    const size_t w_bytes = weight_bytes(dtype, n_in, n_out);
    uint8_t     *blob    = malloc(w_bytes);
    float       *x       = malloc(m * n_in * sizeof(float));
    float       *y_vk    = malloc(m * n_out * sizeof(float));
    float       *y_rf    = malloc(m * n_out * sizeof(float));
    if (blob == nullptr || x == nullptr || y_vk == nullptr || y_rf == nullptr) {
        check(false, "alloc");
        return;
    }
    if (dtype == GEIST_DTYPE_F32) {
        float *wf = (float *) blob;
        for (size_t i = 0; i < n_in * n_out; i++) {
            wf[i] = ((float) rng_u8() - 127.5f) / 64.0f;
        }
    } else {
        fill_blob(blob, n_in, n_out, dtype);
    }
    for (size_t i = 0; i < m * n_in; i++) {
        x[i] = ((float) rng_u8() - 127.5f) / 32.0f;
    }

    struct geist_weight w_vk = {.raw        = blob,
                                .raw_nbytes = w_bytes,
                                .n_in       = (int32_t) n_in,
                                .n_out      = (int32_t) n_out,
                                .dtype      = (uint16_t) dtype};
    struct geist_weight w_rf = w_vk;

    check(vk->desc->vtbl->resolve_weight(vk, &w_vk) == GEIST_OK, "vulkan resolve_weight");
    check(ref->desc->vtbl->resolve_weight(ref, &w_rf) == GEIST_OK, "cpu_scalar resolve_weight");
    if (w_vk.linear_mN == nullptr || w_rf.linear_mN == nullptr) {
        check(false, "resolver installed no kernel");
        return;
    }
    if (m == 1) {
        w_vk.linear_m1(x, &w_vk, vk, y_vk);
        w_rf.linear_m1(x, &w_rf, ref, y_rf);
    } else {
        w_vk.linear_mN(m, x, &w_vk, vk, y_vk);
        w_rf.linear_mN(m, x, &w_rf, ref, y_rf);
    }

    double max_rel = 0.0, ref_mag = 0.0;
    for (size_t i = 0; i < m * n_out; i++) {
        const double a   = y_vk[i];
        const double b   = y_rf[i];
        const double rel = fabs(a - b) / (fabs(b) > 1.0 ? fabs(b) : 1.0);
        if (rel > max_rel) {
            max_rel = rel;
        }
        if (fabs(b) > ref_mag) {
            ref_mag = fabs(b);
        }
    }
    /* A comparison of two all-zero outputs proves nothing (a failed dispatch
     * zeroes y): the reference itself must be non-trivial. */
    check(ref_mag > 1e-2, "reference output is non-trivial");
    char label[128];
    snprintf(label,
             sizeof label,
             "%s m=%zu (%zux%zu) parity, max_rel=%.2e",
             name,
             m,
             n_out,
             n_in,
             max_rel);
    check(max_rel < tol, label);
    printf("  %-10s m=%-3zu  max_rel %.2e  |ref| %.1f %s\n",
           name,
           m,
           max_rel,
           ref_mag,
           max_rel < tol ? "OK" : "FAIL");

    free(blob);
    free(x);
    free(y_vk);
    free(y_rf);
}

static struct geist_tensor mat_view(struct geist_buffer *b, size_t rows, size_t cols) {
    return (struct geist_tensor) {.buffer = b,
                                  .dtype  = GEIST_DTYPE_F32,
                                  .layout = GEIST_LAYOUT_DENSE,
                                  .ndim   = 2,
                                  .shape  = {(int64_t) rows, (int64_t) cols},
                                  .stride = {(int64_t) cols, 1}};
}

/* Same comparison through fused->linear_t: activations in backend buffers,
 * the weight resolved to its VRAM copy, no host-pointer round trip — the path
 * the transformer forward runs. */
static void run_case_t(struct geist_backend *vk,
                       struct geist_backend *ref,
                       int                   dtype,
                       const char           *name,
                       size_t                n_in,
                       size_t                n_out,
                       size_t                m,
                       double                tol) {
    const struct geist_backend_fused *f = geist_backend_fused_tbl(vk);
    if (f->linear_t == nullptr) {
        check(false, "vulkan linear_t missing");
        return;
    }
    const struct geist_backend_vtbl *v       = vk->desc->vtbl;
    const size_t                     w_bytes = weight_bytes(dtype, n_in, n_out);
    uint8_t                         *blob    = malloc(w_bytes);
    float                           *x       = malloc(m * n_in * sizeof(float));
    float                           *y_vk    = malloc(m * n_out * sizeof(float));
    float                           *y_rf    = malloc(m * n_out * sizeof(float));
    if (blob == nullptr || x == nullptr || y_vk == nullptr || y_rf == nullptr) {
        check(false, "alloc");
        return;
    }
    if (dtype == GEIST_DTYPE_F32) {
        float *wf = (float *) blob;
        for (size_t i = 0; i < n_in * n_out; i++) {
            wf[i] = ((float) rng_u8() - 127.5f) / 64.0f;
        }
    } else {
        fill_blob(blob, n_in, n_out, dtype);
    }
    for (size_t i = 0; i < m * n_in; i++) {
        x[i] = ((float) rng_u8() - 127.5f) / 32.0f;
    }
    struct geist_weight w_vk = {.raw        = blob,
                                .raw_nbytes = w_bytes,
                                .n_in       = (int32_t) n_in,
                                .n_out      = (int32_t) n_out,
                                .dtype      = (uint16_t) dtype};
    struct geist_weight w_rf = w_vk;
    check(v->resolve_weight(vk, &w_vk) == GEIST_OK, "vulkan resolve_weight (linear_t)");
    check(ref->desc->vtbl->resolve_weight(ref, &w_rf) == GEIST_OK, "cpu_scalar resolve_weight");

    struct geist_buffer *bx = nullptr, *by = nullptr;
    check(v->buffer_create(vk, m * n_in * 4, GEIST_BUFFER_SCRATCH, GEIST_MEMORY_AUTO, &bx) ==
                          GEIST_OK &&
                  v->buffer_create(
                          vk, m * n_out * 4, GEIST_BUFFER_SCRATCH, GEIST_MEMORY_AUTO, &by) ==
                          GEIST_OK &&
                  v->buffer_upload(bx, m * n_in * 4, (const uint8_t *) x) == GEIST_OK,
          "linear_t buffers");
    struct geist_tensor     tx = mat_view(bx, m, n_in), ty = mat_view(by, m, n_out);
    struct geist_tensor     tw = {.dtype = (uint16_t) dtype};
    const enum geist_status ls = f->linear_t(vk, &tx, &w_vk, &tw, m, &ty);
    check(ls == GEIST_OK, "vulkan linear_t dispatch");
    check(v->buffer_download(m * n_out * 4, (uint8_t *) y_vk, by) == GEIST_OK, "download");
    if (m == 1) {
        w_rf.linear_m1(x, &w_rf, ref, y_rf);
    } else {
        w_rf.linear_mN(m, x, &w_rf, ref, y_rf);
    }
    double max_rel = 0.0, ref_mag = 0.0;
    for (size_t i = 0; i < m * n_out; i++) {
        const double b   = y_rf[i];
        const double rel = fabs((double) y_vk[i] - b) / (fabs(b) > 1.0 ? fabs(b) : 1.0);
        if (rel > max_rel) {
            max_rel = rel;
        }
        if (fabs(b) > ref_mag) {
            ref_mag = fabs(b);
        }
    }
    check(ref_mag > 1e-2, "reference output is non-trivial");
    char label[128];
    snprintf(label,
             sizeof label,
             "%s linear_t m=%zu (%zux%zu), max_rel=%.2e",
             name,
             m,
             n_out,
             n_in,
             max_rel);
    check(max_rel < tol, label);
    printf("  %-10s linear_t m=%-3zu  max_rel %.2e %s\n",
           name,
           m,
           max_rel,
           max_rel < tol ? "OK" : "FAIL");
    v->buffer_destroy(vk, bx);
    v->buffer_destroy(vk, by);
    free(blob);
    free(x);
    free(y_vk);
    free(y_rf);
}

int main(void) {
    struct geist_backend *vk = nullptr;
    enum geist_status     vs = geist_backend_create("vulkan", nullptr, nullptr, &vk);
    if (vs == GEIST_E_UNSUPPORTED || vs == GEIST_E_NOT_FOUND) {
        /* Not built in (default CI BACKENDS) or no loader/device — skip. */
        fprintf(stderr, "SKIP: vulkan backend unavailable (not built or no device)\n");
        return GEIST_TEST_SKIP;
    }
    struct geist_backend *ref = nullptr;
    check(vk != nullptr, "vulkan backend");
    check(geist_backend_create("cpu_scalar", nullptr, nullptr, &ref) == GEIST_OK,
          "cpu_scalar backend");
    if (vk == nullptr || ref == nullptr) {
        return 1;
    }

#define run_case(vk, ref, dt, name, ni, no, m) run_case_tol(vk, ref, dt, name, ni, no, m, 1e-3)
    /* n_in must be a multiple of 256 for k-quants; n_out deliberately not a
     * multiple of the workgroup count to catch tail bugs. */
    run_case(vk, ref, GEIST_DTYPE_Q4_K, "Q4_K", 512, 383, 1);
    run_case(vk, ref, GEIST_DTYPE_Q4_K, "Q4_K", 512, 383, 8);
    run_case(vk, ref, GEIST_DTYPE_Q6_K, "Q6_K", 512, 383, 1);
    run_case(vk, ref, GEIST_DTYPE_Q6_K, "Q6_K", 512, 383, 8);
    run_case(vk, ref, GEIST_DTYPE_F32, "F32", 200, 130, 1);
    run_case(vk, ref, GEIST_DTYPE_F32, "F32", 200, 130, 8);
    /* The legacy 32-element formats and Q5_K. n_in values include block
     * counts that are not a multiple of the per-step block stride (35 and 11
     * blocks) so the tail lanes are exercised; n_out is not a multiple of 8;
     * m = 37 is not a multiple of the 32-row batch tile. */
    static const int   newq[] = {GEIST_DTYPE_Q4_0, GEIST_DTYPE_Q4_1, GEIST_DTYPE_Q8_0};
    static const char *newn[] = {"Q4_0", "Q4_1", "Q8_0"};
    for (size_t i = 0; i < 3; i++) {
        run_case(vk, ref, newq[i], newn[i], 512, 383, 1);
        run_case(vk, ref, newq[i], newn[i], 1120, 131, 1);
        run_case(vk, ref, newq[i], newn[i], 352, 45, 8);
        run_case(vk, ref, newq[i], newn[i], 1120, 131, 37);
        run_case_t(vk, ref, newq[i], newn[i], 1120, 131, 1, 1e-3);
        run_case_t(vk, ref, newq[i], newn[i], 512, 383, 37, 1e-3);
        run_case_t(vk, ref, newq[i], newn[i], 352, 45, 64, 1e-3);
    }
    run_case(vk, ref, GEIST_DTYPE_Q5_K, "Q5_K", 512, 383, 1);
    run_case(vk, ref, GEIST_DTYPE_Q5_K, "Q5_K", 768, 131, 8);
    run_case(vk, ref, GEIST_DTYPE_Q5_K, "Q5_K", 768, 131, 37);
    run_case_t(vk, ref, GEIST_DTYPE_Q5_K, "Q5_K", 768, 131, 1, 1e-3);
    run_case_t(vk, ref, GEIST_DTYPE_Q5_K, "Q5_K", 512, 383, 37, 1e-3);
    /* the existing dtypes through linear_t as well */
    run_case_t(vk, ref, GEIST_DTYPE_Q4_K, "Q4_K", 512, 383, 1, 1e-3);
    run_case_t(vk, ref, GEIST_DTYPE_Q6_K, "Q6_K", 512, 383, 1, 1e-3);
    run_case_t(vk, ref, GEIST_DTYPE_F32, "F32", 200, 130, 1, 1e-3);

    /* coopmat tensor-core path (m%16==0, n_out%64==0): f16 inputs, f32
     * accumulate — looser tolerance by design (prefill-only path; the
     * MMLU gate judges end-to-end). */
    run_case_tol(vk, ref, GEIST_DTYPE_Q4_K, "Q4_K-cm", 512, 256, 16, 2e-2);
    run_case_tol(vk, ref, GEIST_DTYPE_Q4_K, "Q4_K-cm", 768, 128, 64, 2e-2);
    /* wide n_out routes to the 128-row register-tiled kernel */
    run_case_tol(vk, ref, GEIST_DTYPE_Q4_K, "Q4_K-cm", 512, 4096, 16, 2e-2);
    run_case_tol(vk, ref, GEIST_DTYPE_Q4_K, "Q4_K-cm", 512, 4096, 64, 2e-2);

    geist_backend_destroy(vk);
    geist_backend_destroy(ref);
    if (g_fail == 0) {
        printf("test_backend_vulkan_linear_parity: all checks passed\n");
    }
    return g_fail == 0 ? 0 : 1;
}

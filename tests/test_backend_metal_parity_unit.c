/*
 * test_backend_metal_parity_unit — numerical parity gate for the Metal
 * linear path (#181): resolve_weight + linear_m1/linear_mN for Q4_K, Q6_K
 * and F32 weights, compared against the cpu_scalar resolver on the SAME
 * weight bytes. cpu_scalar dequantizes with an independent implementation
 * (src/formats/gguf), so agreement means the MSL dequant and the full
 * dispatch chain (pipeline creation, buffer registry, batching) are
 * correct.
 *
 * The Metal linear op falls back to a HOST compute path when a pointer is
 * not a registered backend buffer — a parity test on malloc memory would
 * silently gate the CPU, not the GPU. So x, w and y all come from
 * buffer_create + buffer_map: the registry lookup hits by construction and
 * the GPU path runs. buffer_download(y) drains the pending batch before
 * the compare.
 *
 * Tolerance: GPU sums in f32, reference in double — 1e-3 relative on
 * n_in=512 dots is generous headroom; a dequant bug is orders of magnitude.
 *
 * SKIPs (exit 0) when the metal backend is not built in or no device is
 * present (Linux legs, GPU-less macs).
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

static void check(bool cond, const char *what) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", what);
        g_fail++;
    }
}

enum { Q4K_BB = 144, Q6K_BB = 210, BLOCK = 256 };

static uint32_t rng_state = 0x12345678u;
static uint8_t  rng_u8(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (uint8_t) (rng_state >> 24);
}

/* Random-but-finite quant blob: random bytes, f16 scale fields pinned. */
static void fill_blob(uint8_t *dst, size_t n_in, size_t n_out, int dtype) {
    const size_t bpr = n_in / BLOCK;
    const size_t bb  = dtype == GEIST_DTYPE_Q4_K ? Q4K_BB : Q6K_BB;
    for (size_t r = 0; r < n_out; r++) {
        for (size_t b = 0; b < bpr; b++) {
            uint8_t *blk = dst + (r * bpr + b) * bb;
            for (size_t i = 0; i < bb; i++) {
                blk[i] = rng_u8();
            }
            if (dtype == GEIST_DTYPE_Q4_K) {
                blk[0] = 0x00; /* d    = fp16(1.0)    */
                blk[1] = 0x3C;
                blk[2] = 0x00; /* dmin = fp16(0.5)    */
                blk[3] = 0x38;
            } else {
                blk[208] = 0x00; /* d = fp16(1.0), trailing field in Q6_K */
                blk[209] = 0x3C;
                /* Pin the 16 int8 block scales to small values: the metal
                 * simdgroup GEMM stages dequantized weights in HALF, whose
                 * integer lattice ends at 2048 — random ±127 scales put
                 * weights at ~4000 where half rounds, which is staging
                 * precision, not a dequant bug. Small scales keep every
                 * staged weight exactly representable, so parity is exact
                 * and any addressing bug still explodes. */
                for (size_t s = 192; s < 208; s++) {
                    blk[s] = (uint8_t) ((rng_u8() & 7u) - 3u);
                }
            }
        }
    }
}

/* Backend-owned, registry-visible allocation; returns the mapped host view. */
static void *dev_alloc(struct geist_backend *be, size_t bytes, struct geist_buffer **out) {
    const struct geist_backend_vtbl *v = be->desc->vtbl;
    if (v->buffer_create(be, bytes, GEIST_BUFFER_SCRATCH, GEIST_MEMORY_AUTO, out) != GEIST_OK) {
        return nullptr;
    }
    return v->buffer_map(*out);
}

static void run_case(struct geist_backend *mt,
                     struct geist_backend *ref,
                     int                   dtype,
                     const char           *name,
                     size_t                n_in,
                     size_t                n_out,
                     size_t                m) {
    const double tol = 1e-3;
    size_t       w_bytes;
    if (dtype == GEIST_DTYPE_F32) {
        w_bytes = n_in * n_out * sizeof(float);
    } else {
        w_bytes = n_out * (n_in / BLOCK) * (dtype == GEIST_DTYPE_Q4_K ? Q4K_BB : Q6K_BB);
    }
    const struct geist_backend_vtbl *v = mt->desc->vtbl;

    struct geist_buffer *bw = nullptr, *bx = nullptr, *by = nullptr;
    uint8_t             *blob = dev_alloc(mt, w_bytes, &bw);
    float               *x    = dev_alloc(mt, m * n_in * sizeof(float), &bx);
    float               *y_mt = dev_alloc(mt, m * n_out * sizeof(float), &by);
    float               *y_dl = malloc(m * n_out * sizeof(float));
    float               *y_rf = malloc(m * n_out * sizeof(float));
    if (blob == nullptr || x == nullptr || y_mt == nullptr || y_dl == nullptr || y_rf == nullptr) {
        check(false, "backend buffer alloc + map");
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

    struct geist_weight w_mt = {.raw   = blob,
                                .n_in  = (int32_t) n_in,
                                .n_out = (int32_t) n_out,
                                .dtype = (uint16_t) dtype};
    struct geist_weight w_rf = w_mt;

    check(mt->desc->vtbl->resolve_weight(mt, &w_mt) == GEIST_OK, "metal resolve_weight");
    check(ref->desc->vtbl->resolve_weight(ref, &w_rf) == GEIST_OK, "cpu_scalar resolve_weight");
    if (w_mt.linear_mN == nullptr || w_rf.linear_mN == nullptr) {
        check(false, "resolver installed no kernel");
        return;
    }
    if (m == 1) {
        w_mt.linear_m1(x, &w_mt, mt, y_mt);
        w_rf.linear_m1(x, &w_rf, ref, y_rf);
    } else {
        w_mt.linear_mN(x, &w_mt, m, mt, y_mt);
        w_rf.linear_mN(x, &w_rf, m, ref, y_rf);
    }
    /* Drains the pending GPU batch before reading y. */
    check(v->buffer_download(m * n_out * sizeof(float), (uint8_t *) y_dl, by) == GEIST_OK,
          "y download");

    double max_rel = 0.0;
    for (size_t i = 0; i < m * n_out; i++) {
        const double a   = y_dl[i];
        const double b   = y_rf[i];
        const double rel = fabs(a - b) / (fabs(b) > 1.0 ? fabs(b) : 1.0);
        if (rel > max_rel) {
            max_rel = rel;
        }
    }
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
    printf("  %-6s m=%-3zu  max_rel %.2e %s\n", name, m, max_rel, max_rel < tol ? "OK" : "FAIL");

    v->buffer_destroy(mt, bw);
    v->buffer_destroy(mt, bx);
    v->buffer_destroy(mt, by);
    free(y_dl);
    free(y_rf);
}

int main(void) {
    struct geist_backend *mt = nullptr;
    enum geist_status     ms = geist_backend_create("metal", nullptr, nullptr, &mt);
    if (ms == GEIST_E_UNSUPPORTED || ms == GEIST_E_NOT_FOUND) {
        /* Not built in (Linux legs) or no Metal device — skip. */
        fprintf(stderr, "SKIP: metal backend unavailable (not built or no device)\n");
        return GEIST_TEST_SKIP;
    }
    struct geist_backend *ref = nullptr;
    check(mt != nullptr, "metal backend");
    check(geist_backend_create("cpu_scalar", nullptr, nullptr, &ref) == GEIST_OK,
          "cpu_scalar backend");
    if (mt == nullptr || ref == nullptr) {
        return 1;
    }

    /* n_in must be a multiple of 256 for k-quants; n_out deliberately not a
     * multiple of the threadgroup width to catch tail bugs. */
    run_case(mt, ref, GEIST_DTYPE_Q4_K, "Q4_K", 512, 383, 1);
    run_case(mt, ref, GEIST_DTYPE_Q4_K, "Q4_K", 512, 383, 8);
    run_case(mt, ref, GEIST_DTYPE_Q6_K, "Q6_K", 512, 383, 1);
    run_case(mt, ref, GEIST_DTYPE_Q6_K, "Q6_K", 512, 383, 8);
    run_case(mt, ref, GEIST_DTYPE_F32, "F32", 256, 130, 1);
    run_case(mt, ref, GEIST_DTYPE_F32, "F32", 256, 130, 8);

    geist_backend_destroy(mt);
    geist_backend_destroy(ref);
    if (g_fail == 0) {
        printf("test_backend_metal_parity_unit: all checks passed\n");
    }
    return g_fail == 0 ? 0 : 1;
}

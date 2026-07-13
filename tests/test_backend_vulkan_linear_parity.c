/*
 * test_backend_vulkan_linear_parity — numerical parity gate for the Vulkan
 * linear path (Phase 2): resolve_weight + linear_m1/linear_mN for Q4_K,
 * Q6_K and F32 weights, compared against the cpu_scalar resolver on the
 * SAME weight bytes. cpu_scalar dequantizes with an independent
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
#define _POSIX_C_SOURCE 200809L /* setenv */
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
            }
        }
    }
}

/* Mirror of the dp4a shader's x quantization: per-4-chunk int8 with an
 * f32 scale. nearbyintf under the default rounding mode matches GLSL
 * round() (round-to-nearest-even) on NVIDIA. */
static void quant4_ref(const float *x, float *xq, size_t n) {
    for (size_t c = 0; c < n; c += 4) {
        float amax = 0.0f;
        for (int i = 0; i < 4; i++) {
            amax = fmaxf(amax, fabsf(x[c + i]));
        }
        if (amax == 0.0f) {
            for (int i = 0; i < 4; i++) {
                xq[c + i] = 0.0f;
            }
            continue;
        }
        const float id = 127.0f / amax, d = amax * (1.0f / 127.0f);
        for (int i = 0; i < 4; i++) {
            xq[c + i] = nearbyintf(x[c + i] * id) * d;
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
                         double                tol,
                         bool                  dp4a_x) {
    size_t w_bytes;
    if (dtype == GEIST_DTYPE_F32) {
        w_bytes = n_in * n_out * sizeof(float);
    } else {
        w_bytes = n_out * (n_in / BLOCK) * (dtype == GEIST_DTYPE_Q4_K ? Q4K_BB : Q6K_BB);
    }
    uint8_t *blob = malloc(w_bytes);
    float   *x    = malloc(m * n_in * sizeof(float));
    float   *y_vk = malloc(m * n_out * sizeof(float));
    float   *y_rf = malloc(m * n_out * sizeof(float));
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

    struct geist_weight w_vk = {
            .raw = blob, .n_in = (int32_t) n_in, .n_out = (int32_t) n_out,
            .dtype = (uint16_t) dtype};
    struct geist_weight w_rf = w_vk;

    check(vk->desc->vtbl->resolve_weight(vk, &w_vk) == GEIST_OK, "vulkan resolve_weight");
    check(ref->desc->vtbl->resolve_weight(ref, &w_rf) == GEIST_OK, "cpu_scalar resolve_weight");
    if (w_vk.linear_mN == nullptr || w_rf.linear_mN == nullptr) {
        check(false, "resolver installed no kernel");
        return;
    }
    /* dp4a_x: reference runs on x pre-quantized exactly like the dp4a
     * shader, so the compare stays tight. A device without int8-dot serves
     * the classic f32 kernel instead — then the EXACT-x reference is the
     * matching one, so take the better of the two. */
    float *xr = x;
    if (dp4a_x) {
        xr = malloc(m * n_in * sizeof(float));
        if (xr == nullptr) {
            check(false, "alloc");
            return;
        }
        quant4_ref(x, xr, m * n_in);
    }
    if (m == 1) {
        w_vk.linear_m1(x, &w_vk, vk, y_vk);
        w_rf.linear_m1(xr, &w_rf, ref, y_rf);
    } else {
        w_vk.linear_mN(x, &w_vk, m, vk, y_vk);
        w_rf.linear_mN(xr, &w_rf, m, ref, y_rf);
    }

    double max_rel = 0.0;
    for (size_t i = 0; i < m * n_out; i++) {
        const double a   = y_vk[i];
        const double b   = y_rf[i];
        const double rel = fabs(a - b) / (fabs(b) > 1.0 ? fabs(b) : 1.0);
        if (rel > max_rel) {
            max_rel = rel;
        }
    }
    if (dp4a_x && max_rel >= tol) {
        w_rf.linear_m1(x, &w_rf, ref, y_rf); /* exact-x fallback compare */
        double max_rel2 = 0.0;
        for (size_t i = 0; i < m * n_out; i++) {
            const double rel = fabs((double) y_vk[i] - (double) y_rf[i]) /
                               (fabs((double) y_rf[i]) > 1.0 ? fabs((double) y_rf[i]) : 1.0);
            if (rel > max_rel2) {
                max_rel2 = rel;
            }
        }
        if (max_rel2 < max_rel) {
            max_rel = max_rel2;
        }
    }
    if (dp4a_x) {
        free(xr);
    }
    char label[128];
    snprintf(label, sizeof label, "%s m=%zu (%zux%zu) parity, max_rel=%.2e", name, m, n_out,
             n_in, max_rel);
    check(max_rel < tol, label);
    printf("  %-10s m=%-3zu  max_rel %.2e %s\n", name, m, max_rel,
           max_rel < tol ? "OK" : "FAIL");

    free(blob);
    free(x);
    free(y_vk);
    free(y_rf);
}

int main(void) {
    /* The exact-parity cases pin the classic f32 kernels; the dp4a case
     * below creates its own backend with the int8-dot matvec enabled. */
    setenv("GEIST_VK_DP4A", "0", 1);
    struct geist_backend *vk = nullptr;
    if (geist_backend_create("vulkan", nullptr, nullptr, &vk) == GEIST_E_UNSUPPORTED) {
        fprintf(stderr, "SKIP: no Vulkan runtime/device on this machine\n");
        return 0;
    }
    struct geist_backend *ref = nullptr;
    check(vk != nullptr, "vulkan backend");
    check(geist_backend_create("cpu_scalar", nullptr, nullptr, &ref) == GEIST_OK,
          "cpu_scalar backend");
    if (vk == nullptr || ref == nullptr) {
        return 1;
    }

#define run_case(vk, ref, dt, name, ni, no, m) \
    run_case_tol(vk, ref, dt, name, ni, no, m, 1e-3, false)
    /* n_in must be a multiple of 256 for k-quants; n_out deliberately not a
     * multiple of the workgroup count to catch tail bugs. */
    run_case(vk, ref, GEIST_DTYPE_Q4_K, "Q4_K", 512, 383, 1);
    run_case(vk, ref, GEIST_DTYPE_Q4_K, "Q4_K", 512, 383, 8);
    run_case(vk, ref, GEIST_DTYPE_Q6_K, "Q6_K", 512, 383, 1);
    run_case(vk, ref, GEIST_DTYPE_Q6_K, "Q6_K", 512, 383, 8);
    run_case(vk, ref, GEIST_DTYPE_F32, "F32", 200, 130, 1);
    run_case(vk, ref, GEIST_DTYPE_F32, "F32", 200, 130, 8);
    /* coopmat tensor-core path (m%16==0, n_out%64==0): f16 inputs, f32
     * accumulate — looser tolerance by design (prefill-only path; the
     * MMLU gate judges end-to-end). */
    run_case_tol(vk, ref, GEIST_DTYPE_Q4_K, "Q4_K-cm", 512, 256, 16, 2e-2, false);
    run_case_tol(vk, ref, GEIST_DTYPE_Q4_K, "Q4_K-cm", 768, 128, 64, 2e-2, false);
    /* wide n_out routes to the 128-row register-tiled kernel */
    run_case_tol(vk, ref, GEIST_DTYPE_Q4_K, "Q4_K-cm", 512, 4096, 16, 2e-2, false);
    run_case_tol(vk, ref, GEIST_DTYPE_Q4_K, "Q4_K-cm", 512, 4096, 64, 2e-2, false);

    /* dp4a decode matvec: reference on identically pre-quantized x keeps
     * the compare tight (raw-x parity is ~9e-2 by construction — per-4
     * int8 noise under random-weight cancellation, not a bug). */
    setenv("GEIST_VK_DP4A", "1", 1);
    struct geist_backend *vkd = nullptr;
    check(geist_backend_create("vulkan", nullptr, nullptr, &vkd) == GEIST_OK,
          "vulkan backend (dp4a)");
    if (vkd != nullptr) {
        run_case_tol(vkd, ref, GEIST_DTYPE_Q4_K, "Q4_K-dp4a", 512, 383, 1, 1e-3, true);
        geist_backend_destroy(vkd);
    }

    geist_backend_destroy(vk);
    geist_backend_destroy(ref);
    if (g_fail == 0) {
        printf("test_backend_vulkan_linear_parity: all checks passed\n");
    }
    return g_fail == 0 ? 0 : 1;
}

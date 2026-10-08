/*
 * test_backend_vulkan_linear_parity — numerical parity gate for the Vulkan
 * linear path: resolve_weight + linear_m1/linear_mN for Q4_K,
 * Q5_K, Q6_K, Q4_0, Q4_1, Q8_0, TQ2_0, PQ2_0 and F32 weights (plus fused->linear_t, the
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
        {GEIST_DTYPE_TQ2_0, 256, 66, -2}, /* d is the TRAILING f16 */
        {GEIST_DTYPE_PQ2_0, 128, 34, 2},
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

static struct geist_tensor mat_view(struct geist_buffer *b, size_t rows, size_t cols) {
    return (struct geist_tensor) {.buffer = b,
                                  .dtype  = GEIST_DTYPE_F32,
                                  .layout = GEIST_LAYOUT_DENSE,
                                  .ndim   = 2,
                                  .shape  = {(int64_t) rows, (int64_t) cols},
                                  .stride = {(int64_t) cols, 1}};
}

/* Which Vulkan entry point a case drives. The cpu_scalar reference always
 * runs through its resolved weight's linear_m1/linear_mN. */
enum parity_path {
    VIA_WEIGHT,   /* the resolved weight's linear_m1/linear_mN, host pointers */
    VIA_LINEAR_T, /* fused->linear_t: activations in backend buffers, the weight
                   * resolved to its VRAM copy, no host-pointer round trip —
                   * the path the transformer forward runs */
};

/* How a case's error is normalized. PARITY_ELEM: each element against its
 * own reference magnitude (floored at 1) — strict, right for the f32 paths.
 * PARITY_MAG: the largest absolute error against the largest reference
 * magnitude — for the tensor-core paths, whose f16 operands carry a relative
 * error of ~2^-11 per product; on outputs that cancel to near zero that
 * rounding is a large fraction of the element, while it stays ~1e-4 of the
 * output scale. */
enum parity_metric { PARITY_ELEM, PARITY_MAG };

/* One parity case: the same random weight and activations through the Vulkan
 * path `via` and through cpu_scalar, compared at tolerance tol under
 * `metric`. */
static void run_parity(enum parity_path      via,
                       struct geist_backend *vk,
                       struct geist_backend *ref,
                       int                   dtype,
                       const char           *name,
                       size_t                n_in,
                       size_t                n_out,
                       size_t                m,
                       enum parity_metric    metric,
                       double                tol) {
    const struct geist_backend_fused *f = geist_backend_fused_tbl(vk);
    if (via == VIA_LINEAR_T && f->linear_t == nullptr) {
        check(false, "vulkan linear_t missing");
        return;
    }
    const struct geist_backend_vtbl *v       = vk->desc->vtbl;
    const size_t                     w_bytes = weight_bytes(dtype, n_in, n_out);
    uint8_t                         *blob    = malloc(w_bytes);
    float                           *x       = malloc(m * n_in * sizeof(float));
    float                           *y_vk    = malloc(m * n_out * sizeof(float));
    float                           *y_rf    = malloc(m * n_out * sizeof(float));
    struct geist_buffer             *bx = nullptr, *by = nullptr;
    if (blob == nullptr || x == nullptr || y_vk == nullptr || y_rf == nullptr) {
        check(false, "alloc");
        goto done;
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

    check(v->resolve_weight(vk, &w_vk) == GEIST_OK,
          via == VIA_LINEAR_T ? "vulkan resolve_weight (linear_t)" : "vulkan resolve_weight");
    check(ref->desc->vtbl->resolve_weight(ref, &w_rf) == GEIST_OK, "cpu_scalar resolve_weight");
    if ((via == VIA_WEIGHT && w_vk.linear_mN == nullptr) || w_rf.linear_mN == nullptr) {
        check(false, "resolver installed no kernel");
        goto done;
    }
    if (via == VIA_LINEAR_T) {
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
    } else if (m == 1) {
        w_vk.linear_m1(x, &w_vk, vk, y_vk);
    } else {
        w_vk.linear_mN(m, x, &w_vk, vk, y_vk);
    }
    if (m == 1) {
        w_rf.linear_m1(x, &w_rf, ref, y_rf);
    } else {
        w_rf.linear_mN(m, x, &w_rf, ref, y_rf);
    }

    double max_rel = 0.0, ref_mag = 0.0, max_abs = 0.0;
    for (size_t i = 0; i < m * n_out; i++) {
        const double b   = y_rf[i];
        const double err = fabs((double) y_vk[i] - b);
        const double rel = err / (fabs(b) > 1.0 ? fabs(b) : 1.0);
        if (rel > max_rel) {
            max_rel = rel;
        }
        if (err > max_abs) {
            max_abs = err;
        }
        if (fabs(b) > ref_mag) {
            ref_mag = fabs(b);
        }
    }
    if (metric == PARITY_MAG) {
        max_rel = max_abs / (ref_mag > 1.0 ? ref_mag : 1.0);
    }
    /* A comparison of two all-zero outputs proves nothing (a failed dispatch
     * zeroes y): the reference itself must be non-trivial. */
    check(ref_mag > 1e-2, "reference output is non-trivial");
    const char *via_name = via == VIA_LINEAR_T ? " linear_t" : "";
    char        label[128];
    snprintf(label,
             sizeof label,
             "%s%s m=%zu (%zux%zu) parity, max_rel=%.2e",
             name,
             via_name,
             m,
             n_out,
             n_in,
             max_rel);
    check(max_rel < tol, label);
    printf("  %-10s%s m=%-3zu  max_rel %.2e  |ref| %.1f %s\n",
           name,
           via_name,
           m,
           max_rel,
           ref_mag,
           max_rel < tol ? "OK" : "FAIL");

done:
    if (bx != nullptr) {
        v->buffer_destroy(vk, bx);
    }
    if (by != nullptr) {
        v->buffer_destroy(vk, by);
    }
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

#define run_case(vk, ref, dt, name, ni, no, m) \
    run_parity(VIA_WEIGHT, vk, ref, dt, name, ni, no, m, PARITY_ELEM, 1e-3)
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
        run_parity(VIA_LINEAR_T, vk, ref, newq[i], newn[i], 1120, 131, 1, PARITY_ELEM, 1e-3);
        run_parity(VIA_LINEAR_T, vk, ref, newq[i], newn[i], 512, 383, 37, PARITY_ELEM, 1e-3);
        run_parity(VIA_LINEAR_T, vk, ref, newq[i], newn[i], 352, 45, 64, PARITY_ELEM, 1e-3);
    }
    run_case(vk, ref, GEIST_DTYPE_Q5_K, "Q5_K", 512, 383, 1);
    run_case(vk, ref, GEIST_DTYPE_Q5_K, "Q5_K", 768, 131, 8);
    run_case(vk, ref, GEIST_DTYPE_Q5_K, "Q5_K", 768, 131, 37);
    run_parity(VIA_LINEAR_T, vk, ref, GEIST_DTYPE_Q5_K, "Q5_K", 768, 131, 1, PARITY_ELEM, 1e-3);
    run_parity(VIA_LINEAR_T, vk, ref, GEIST_DTYPE_Q5_K, "Q5_K", 512, 383, 37, PARITY_ELEM, 1e-3);
    run_case(vk, ref, GEIST_DTYPE_TQ2_0, "TQ2_0", 512, 383, 1);
    run_case(vk, ref, GEIST_DTYPE_TQ2_0, "TQ2_0", 768, 131, 8);
    run_case(vk, ref, GEIST_DTYPE_TQ2_0, "TQ2_0", 768, 131, 37);
    run_parity(VIA_LINEAR_T, vk, ref, GEIST_DTYPE_TQ2_0, "TQ2_0", 768, 131, 1, PARITY_ELEM, 1e-3);
    run_parity(VIA_LINEAR_T, vk, ref, GEIST_DTYPE_TQ2_0, "TQ2_0", 512, 383, 37, PARITY_ELEM, 1e-3);
    /* PQ2_0: 128-element blocks, two blocks per warp step — 1408 = 11 blocks
     * exercises the odd tail, 384 a single-step row. */
    run_case(vk, ref, GEIST_DTYPE_PQ2_0, "PQ2_0", 512, 383, 1);
    run_case(vk, ref, GEIST_DTYPE_PQ2_0, "PQ2_0", 1408, 131, 1);
    /* rows longer than one 16-block step of the matvec warp (40 and 136 blocks) */
    run_case(vk, ref, GEIST_DTYPE_PQ2_0, "PQ2_0", 5120, 200, 1);
    run_case(vk, ref, GEIST_DTYPE_PQ2_0, "PQ2_0", 17408, 96, 1);
    run_case(vk, ref, GEIST_DTYPE_PQ2_0, "PQ2_0", 384, 45, 8);
    run_case(vk, ref, GEIST_DTYPE_PQ2_0, "PQ2_0", 1408, 131, 37);
    run_parity(VIA_LINEAR_T, vk, ref, GEIST_DTYPE_PQ2_0, "PQ2_0", 1408, 131, 1, PARITY_ELEM, 1e-3);
    run_parity(VIA_LINEAR_T, vk, ref, GEIST_DTYPE_PQ2_0, "PQ2_0", 512, 383, 37, PARITY_ELEM, 1e-3);
    /* the dtypes above through linear_t as well */
    run_parity(VIA_LINEAR_T, vk, ref, GEIST_DTYPE_Q4_K, "Q4_K", 512, 383, 1, PARITY_ELEM, 1e-3);
    run_parity(VIA_LINEAR_T, vk, ref, GEIST_DTYPE_Q6_K, "Q6_K", 512, 383, 1, PARITY_ELEM, 1e-3);
    run_parity(VIA_LINEAR_T, vk, ref, GEIST_DTYPE_F32, "F32", 200, 130, 1, PARITY_ELEM, 1e-3);

    /* coopmat tensor-core path (m%16==0, n_out%64==0): f16 operands, f32
     * accumulate, judged by PARITY_MAG — f16 rounding is ~2^-11 of a product,
     * observed <= 3.2e-4 of the output scale on this data; 2e-3 leaves 6x. */
    run_parity(VIA_WEIGHT, vk, ref, GEIST_DTYPE_Q4_K, "Q4_K-cm", 512, 256, 16, PARITY_MAG, 2e-3);
    run_parity(VIA_WEIGHT, vk, ref, GEIST_DTYPE_Q4_K, "Q4_K-cm", 768, 128, 64, PARITY_MAG, 2e-3);
    /* wide n_out routes to the 128-row register-tiled kernel */
    run_parity(VIA_WEIGHT, vk, ref, GEIST_DTYPE_Q4_K, "Q4_K-cm", 512, 4096, 16, PARITY_MAG, 2e-3);
    run_parity(VIA_WEIGHT, vk, ref, GEIST_DTYPE_Q4_K, "Q4_K-cm", 512, 4096, 64, PARITY_MAG, 2e-3);
    /* PQ2_0 on the tensor cores. The ternary values are exact in f16; the
     * activations are rounded to f16 and the default kernel accumulates in f16
     * (folded into f32 every 64 k), so its bound is wider than the f32-
     * accumulate kernels'. The model-level check (logits vs cpu_scalar, the
     * fork goldens) remains the real gate; the f32-accumulate variant is exact
     * on this data (second backend below). */
    run_parity(VIA_WEIGHT, vk, ref, GEIST_DTYPE_PQ2_0, "PQ2_0-cm", 512, 256, 16, PARITY_MAG, 1e-2);
    run_parity(VIA_WEIGHT, vk, ref, GEIST_DTYPE_PQ2_0, "PQ2_0-cm", 640, 128, 64, PARITY_MAG, 1e-2);
    run_parity(VIA_WEIGHT, vk, ref, GEIST_DTYPE_PQ2_0, "PQ2_0-cm", 512, 4096, 48, PARITY_MAG, 1e-2);
    run_parity(
            VIA_WEIGHT, vk, ref, GEIST_DTYPE_PQ2_0, "PQ2_0-cm", 5120, 256, 128, PARITY_MAG, 1e-2);
    /* Q8_0 on the tensor cores: 1120 = 35 blocks (odd k-step count), m = 112
     * leaves the second 64-token tile partly empty, and linear_t reaches the
     * same kernel through the staged-x path. */
    run_parity(VIA_WEIGHT, vk, ref, GEIST_DTYPE_Q8_0, "Q8_0-cm", 512, 256, 16, PARITY_MAG, 2e-3);
    run_parity(VIA_WEIGHT, vk, ref, GEIST_DTYPE_Q8_0, "Q8_0-cm", 1120, 128, 64, PARITY_MAG, 2e-3);
    run_parity(VIA_WEIGHT, vk, ref, GEIST_DTYPE_Q8_0, "Q8_0-cm", 512, 192, 112, PARITY_MAG, 2e-3);
    run_parity(VIA_LINEAR_T, vk, ref, GEIST_DTYPE_Q8_0, "Q8_0-cm", 1120, 128, 48, PARITY_MAG, 2e-3);
    /* m % 16 != 0 on conforming shapes: the leading m & ~15 rows run on the
     * tensor cores, the tail on the register-tiled GEMM (vk_gemm_dispatch);
     * m = 20 is a 16-row head and a 4-row tail, m = 101 a 96-row head */
    run_parity(VIA_WEIGHT, vk, ref, GEIST_DTYPE_Q8_0, "Q8_0-split", 512, 192, 20, PARITY_MAG, 2e-3);
    run_parity(
            VIA_WEIGHT, vk, ref, GEIST_DTYPE_Q8_0, "Q8_0-split", 1120, 128, 101, PARITY_MAG, 2e-3);
    run_parity(
            VIA_LINEAR_T, vk, ref, GEIST_DTYPE_Q8_0, "Q8_0-split", 512, 256, 37, PARITY_MAG, 2e-3);
    run_parity(VIA_WEIGHT, vk, ref, GEIST_DTYPE_Q4_K, "Q4_K-split", 512, 256, 37, PARITY_MAG, 2e-3);
    run_parity(
            VIA_LINEAR_T, vk, ref, GEIST_DTYPE_Q4_K, "Q4_K-split", 768, 4096, 53, PARITY_MAG, 2e-3);
    /* Q6_K on the tensor cores: its products reach +-4096 on this data, past
     * f16's exact-integer range, so it is the case PARITY_ELEM cannot judge */
    run_parity(VIA_WEIGHT, vk, ref, GEIST_DTYPE_Q6_K, "Q6_K-cm", 512, 128, 16, PARITY_MAG, 2e-3);
    run_parity(VIA_WEIGHT, vk, ref, GEIST_DTYPE_Q6_K, "Q6_K-cm", 768, 256, 64, PARITY_MAG, 2e-3);
    run_parity(VIA_WEIGHT, vk, ref, GEIST_DTYPE_Q6_K, "Q6_K-split", 512, 128, 23, PARITY_MAG, 2e-3);
    /* Q4_0 on the tensor cores (#467), the Q8_0 shapes again: odd k-step
     * count, a partly empty token tile, linear_t, and the m % 16 split. */
    run_parity(VIA_WEIGHT, vk, ref, GEIST_DTYPE_Q4_0, "Q4_0-cm", 512, 256, 16, PARITY_MAG, 2e-3);
    run_parity(VIA_WEIGHT, vk, ref, GEIST_DTYPE_Q4_0, "Q4_0-cm", 1120, 128, 64, PARITY_MAG, 2e-3);
    run_parity(VIA_WEIGHT, vk, ref, GEIST_DTYPE_Q4_0, "Q4_0-cm", 512, 192, 112, PARITY_MAG, 2e-3);
    run_parity(VIA_LINEAR_T, vk, ref, GEIST_DTYPE_Q4_0, "Q4_0-cm", 1120, 128, 48, PARITY_MAG, 2e-3);
    run_parity(
            VIA_WEIGHT, vk, ref, GEIST_DTYPE_Q4_0, "Q4_0-split", 1120, 128, 101, PARITY_MAG, 2e-3);
    geist_backend_destroy(vk);

    /* the exact f32-accumulate tensor-core GEMM (GEIST_VK_PQ2_F32_ACC) */
    setenv("GEIST_VK_PQ2_F32_ACC", "1", 1);
    struct geist_backend *vk32 = nullptr;
    check(geist_backend_create("vulkan", nullptr, nullptr, &vk32) == GEIST_OK, "vulkan f32-acc");
    if (vk32 != nullptr) {
        run_parity(VIA_WEIGHT,
                   vk32,
                   ref,
                   GEIST_DTYPE_PQ2_0,
                   "PQ2_0-cm32",
                   512,
                   256,
                   16,
                   PARITY_ELEM,
                   1e-3);
        run_parity(VIA_WEIGHT,
                   vk32,
                   ref,
                   GEIST_DTYPE_PQ2_0,
                   "PQ2_0-cm32",
                   640,
                   128,
                   64,
                   PARITY_ELEM,
                   1e-3);
        run_parity(VIA_WEIGHT,
                   vk32,
                   ref,
                   GEIST_DTYPE_PQ2_0,
                   "PQ2_0-cm32",
                   5120,
                   256,
                   128,
                   PARITY_ELEM,
                   1e-3);
        geist_backend_destroy(vk32);
    }
    unsetenv("GEIST_VK_PQ2_F32_ACC");
    geist_backend_destroy(ref);
    if (g_fail == 0) {
        printf("test_backend_vulkan_linear_parity: all checks passed\n");
    }
    return g_fail == 0 ? 0 : 1;
}

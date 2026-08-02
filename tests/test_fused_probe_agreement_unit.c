/*
 * test_fused_probe_agreement_unit — the probe-and-bind contract.
 *
 * geist_backend_fused.supported answers at plan-build time whether a
 * fused kernel will handle a geometry; bound call sites then invoke the
 * kernel with no per-call fallback. That contract only holds if probe
 * and kernel agree, so this test checks the pairing on every backend
 * available in the build:
 *
 *   - positive agreement (elementwise): probe says yes for a small F32
 *     geometry → the kernel must return GEIST_OK on real buffers.
 *   - negative agreement (neon GEGLU tile): probe says no for a
 *     misaligned/wrong-dtype geometry → the kernel's own entry checks
 *     must reject with GEIST_E_UNSUPPORTED for the same shape (the
 *     entry checks run before any weight bytes are read, so calling
 *     with a hollow weight struct is safe).
 *
 * The positive GEGLU case (real Q4_K/Q6_K weights) is covered end-to-end
 * by running the decode suite with GEIST_FFN_TILE_FUSION=1 — the plan
 * binds the tile kernel and the canonical-token tests prove the output.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>

#include <stdio.h>
#include <string.h>

static int check_backend(const char *name) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create(name, nullptr, nullptr, &be) != GEIST_OK) {
        printf("  %s: not available, skipped\n", name);
        return 0;
    }
    int                               fails = 0;
    const struct geist_backend_fused *fused = geist_backend_fused_tbl(be);

    /* ---- Elementwise positive agreement. */
    struct geist_fusion_query q = {
            .op = GEIST_FUSED_GELU_TANH_MUL, .m = 4, .d_model = 64, .inter = 64};
    if (fused->supported != nullptr && fused->supported(be, &q)) {
        const size_t                     n  = 4 * 64;
        struct geist_buffer             *bx = nullptr, *bz = nullptr, *by = nullptr;
        const struct geist_backend_vtbl *v = be->desc->vtbl;
        if (v->buffer_create(be, n * sizeof(float), GEIST_BUFFER_SCRATCH, 0, &bx) == GEIST_OK &&
            v->buffer_create(be, n * sizeof(float), GEIST_BUFFER_SCRATCH, 0, &bz) == GEIST_OK &&
            v->buffer_create(be, n * sizeof(float), GEIST_BUFFER_SCRATCH, 0, &by) == GEIST_OK) {
            struct geist_tensor tx = {.buffer = bx,
                                      .dtype  = GEIST_DTYPE_F32,
                                      .layout = GEIST_LAYOUT_DENSE,
                                      .ndim   = 2,
                                      .shape  = {4, 64},
                                      .stride = {64, 1}};
            struct geist_tensor tz = tx, ty = tx;
            tz.buffer           = bz;
            ty.buffer           = by;
            enum geist_status s = fused->gelu_tanh_mul(be, &tx, &tz, &ty);
            fails += geist_expect(s == GEIST_OK,
                                  "probe said yes: gelu_tanh_mul must return GEIST_OK");
        }
        if (bx)
            v->buffer_destroy(be, bx);
        if (bz)
            v->buffer_destroy(be, bz);
        if (by)
            v->buffer_destroy(be, by);
    } else {
        fails += geist_expect(fused->gelu_tanh_mul == nullptr || fused->supported == nullptr,
                              "kernel present but probe rejects the plain F32 case");
    }

    /* ---- GEGLU tile negative agreement: wrong dtype + misaligned dims.
     * Entry checks reject before touching weight bytes. */
    if (fused->ffn_geglu_q4q6_mN != nullptr) {
        struct geist_weight       gate = {.dtype = GEIST_DTYPE_Q8_0, .n_in = 100, .n_out = 300};
        struct geist_weight       up = gate, down = gate;
        struct geist_fusion_query gq = {.op      = GEIST_FUSED_FFN_GEGLU_Q4Q6_MN,
                                        .m       = 4,
                                        .d_model = 100,
                                        .inter   = 300,
                                        .gate_w  = &gate,
                                        .up_w    = &up,
                                        .down_w  = &down};
        const bool probe_yes         = fused->supported != nullptr && fused->supported(be, &gq);
        fails += geist_expect(!probe_yes, "probe rejects misaligned/wrong-dtype GEGLU");
        float             x[4], y[4];
        enum geist_status s =
                fused->ffn_geglu_q4q6_mN(be, x, 4, 100, 300, &gate, &up, &down, nullptr, y);
        fails += geist_expect(s == GEIST_E_UNSUPPORTED,
                              "kernel entry checks agree: GEIST_E_UNSUPPORTED");
    }

    printf("  %s: probe/kernel agreement ok\n", name);
    geist_backend_destroy(be);
    return fails;
}

int main(void) {
    int fails = 0;
    fails += check_backend("cpu_neon");
    fails += check_backend("cpu_scalar");
    fails += check_backend("metal");
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("fused probe agreement: pass\n");
    return GEIST_TEST_PASS;
}

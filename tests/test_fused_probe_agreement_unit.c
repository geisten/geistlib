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
    const struct geist_backend_vtbl  *vt    = be->desc->vtbl;
    (void) vt;

    /* ---- Elementwise positive agreement. */
    struct geist_fusion_query q = {
            .op = GEIST_FUSED_GELU_TANH_MUL, .m = 4, .d_model = 64, .inter = 64};
    if (fused->supported != nullptr && fused->supported(be, &q)) {
        const size_t         n  = 4 * 64;
        struct geist_buffer *bx = nullptr, *bz = nullptr, *by = nullptr;
        if (vt->buffer_create(be, n * sizeof(float), GEIST_BUFFER_SCRATCH, 0, &bx) == GEIST_OK &&
            vt->buffer_create(be, n * sizeof(float), GEIST_BUFFER_SCRATCH, 0, &bz) == GEIST_OK &&
            vt->buffer_create(be, n * sizeof(float), GEIST_BUFFER_SCRATCH, 0, &by) == GEIST_OK) {
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
            vt->buffer_destroy(be, bx);
        if (bz)
            vt->buffer_destroy(be, bz);
        if (by)
            vt->buffer_destroy(be, by);
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

    /* ---- Newly bound stages: positive agreement on F32 buffers where
     * the op is expressible without model weights. */
    const struct geist_backend_vtbl *v = be->desc->vtbl;
    if (fused->supported != nullptr) {
        /* rmsnorm_add: res + rmsnorm(x)*w over a [2, 64] block. */
        struct geist_fusion_query rq = {
                .op = GEIST_FUSED_RMSNORM_ADD, .m = 2, .d_model = 64, .inter = 64};
        if (fused->supported(be, &rq)) {
            struct geist_buffer *br = nullptr, *bx = nullptr, *bw = nullptr, *by = nullptr;
            const size_t         n = 2 * 64;
            if (vt->buffer_create(be, n * sizeof(float), GEIST_BUFFER_SCRATCH, 0, &br) ==
                        GEIST_OK &&
                v->buffer_create(be, n * sizeof(float), GEIST_BUFFER_SCRATCH, 0, &bx) == GEIST_OK &&
                v->buffer_create(be, 64 * sizeof(float), GEIST_BUFFER_SCRATCH, 0, &bw) ==
                        GEIST_OK &&
                vt->buffer_create(be, n * sizeof(float), GEIST_BUFFER_SCRATCH, 0, &by) ==
                        GEIST_OK) {
                struct geist_tensor tr = {.buffer = br,
                                          .dtype  = GEIST_DTYPE_F32,
                                          .layout = GEIST_LAYOUT_DENSE,
                                          .ndim   = 2,
                                          .shape  = {2, 64},
                                          .stride = {64, 1}};
                struct geist_tensor tx = tr, ty = tr;
                tx.buffer              = bx;
                ty.buffer              = by;
                struct geist_tensor tw = {.buffer = bw,
                                          .dtype  = GEIST_DTYPE_F32,
                                          .layout = GEIST_LAYOUT_DENSE,
                                          .ndim   = 1,
                                          .shape  = {64},
                                          .stride = {1}};
                fails += geist_expect(fused->rmsnorm_add(be, &tr, &tx, &tw, 1e-6f, &ty) == GEIST_OK,
                                      "probe said yes: rmsnorm_add must return GEIST_OK");
            }
            if (br)
                v->buffer_destroy(be, br);
            if (bx)
                v->buffer_destroy(be, bx);
            if (bw)
                v->buffer_destroy(be, bw);
            if (by)
                v->buffer_destroy(be, by);
        }
        /* argmax over a [1, 256] logits row. */
        struct geist_fusion_query aq = {.op = GEIST_FUSED_ARGMAX_F32, .m = 1, .d_model = 256};
        if (fused->supported(be, &aq)) {
            struct geist_buffer *bl = nullptr;
            if (v->buffer_create(be, 256 * sizeof(float), GEIST_BUFFER_SCRATCH, 0, &bl) ==
                GEIST_OK) {
                float *p = (float *) v->buffer_map(bl);
                for (int i = 0; i < 256; i++)
                    p[i] = (float) (i == 77 ? 100 : i % 7);
                v->buffer_unmap(bl);
                struct geist_tensor tl  = {.buffer = bl,
                                           .dtype  = GEIST_DTYPE_F32,
                                           .layout = GEIST_LAYOUT_DENSE,
                                           .ndim   = 2,
                                           .shape  = {1, 256},
                                           .stride = {256, 1}};
                int32_t             idx = -1;
                fails += geist_expect(fused->argmax_f32(be, &tl, &idx) == GEIST_OK && idx == 77,
                                      "probe said yes: argmax_f32 must return OK and index 77");
                v->buffer_destroy(be, bl);
            }
        }
        /* embedding_lookup_scaled over a tiny F32 [4, 8] table. */
        struct geist_fusion_query eq = {.op          = GEIST_FUSED_EMBEDDING_LOOKUP_SCALED,
                                        .m           = 1,
                                        .d_model     = 8,
                                        .table_dtype = GEIST_DTYPE_F32};
        if (fused->supported(be, &eq)) {
            struct geist_buffer *bt = nullptr, *bo = nullptr;
            if (v->buffer_create(be, 4 * 8 * sizeof(float), GEIST_BUFFER_WEIGHT, 0, &bt) ==
                        GEIST_OK &&
                v->buffer_create(be, 8 * sizeof(float), GEIST_BUFFER_SCRATCH, 0, &bo) == GEIST_OK) {
                float *p = (float *) v->buffer_map(bt);
                for (int i = 0; i < 32; i++)
                    p[i] = (float) i;
                v->buffer_unmap(bt);
                struct geist_tensor tt = {.buffer = bt,
                                          .dtype  = GEIST_DTYPE_F32,
                                          .layout = GEIST_LAYOUT_DENSE,
                                          .ndim   = 2,
                                          .shape  = {4, 8},
                                          .stride = {8, 1}};
                struct geist_tensor to = {.buffer = bo,
                                          .dtype  = GEIST_DTYPE_F32,
                                          .layout = GEIST_LAYOUT_DENSE,
                                          .ndim   = 1,
                                          .shape  = {8},
                                          .stride = {1}};
                fails += geist_expect(fused->embedding_lookup_scaled(be, &tt, 2, 2.0f, &to) ==
                                              GEIST_OK,
                                      "probe said yes: embedding_lookup_scaled must return OK");
            }
            if (bt)
                v->buffer_destroy(be, bt);
            if (bo)
                v->buffer_destroy(be, bo);
        }
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

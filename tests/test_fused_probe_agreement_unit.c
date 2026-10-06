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
 *   - INT8- and INT4-KV attention: yes for a GQA prefill chunk → GEIST_OK
 *     on zeroed buffers; no for a head_dim past 512, for query heads that
 *     are not a multiple of the KV heads and (INT4) for an odd head_dim →
 *     GEIST_E_UNSUPPORTED.
 *   - attn_qkv_prep and the decode gate/up front (#474): bound with no
 *     fallback, so the probe must not say yes to a head_dim or a weight
 *     geometry the kernel refuses.
 *   - hadamard_rotate (#495): the model loader refuses a rotated model
 *     unless the probe says yes, so the probe must answer exactly what the
 *     kernel runs. Each geometry runs the kernel and compares: yes →
 *     GEIST_OK, no → an error (the kernel's geometry checks answer
 *     GEIST_E_INVALID_ARG, not GEIST_E_UNSUPPORTED). Blocks of 2 and 8192
 *     sit outside metal's 4..4096 kernel range but inside the CPU's.
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

/* A zeroed DENSE view of `dtype` over a new buffer on `be`; t->buffer is
 * nullptr when the buffer cannot be made. */
static struct geist_tensor zeroed(struct geist_backend *be,
                                  enum geist_dtype      dtype,
                                  size_t                elem,
                                  int                   ndim,
                                  const int64_t         shape[static ndim]) {
    const struct geist_backend_vtbl *vt = be->desc->vtbl;
    struct geist_tensor t = {.dtype = dtype, .layout = GEIST_LAYOUT_DENSE, .ndim = ndim};
    size_t              n = 1;
    for (int d = ndim - 1; d >= 0; d--) {
        t.shape[d]  = shape[d];
        t.stride[d] = (int64_t) n;
        n *= (size_t) shape[d];
    }
    if (vt->buffer_create(be, n * elem, GEIST_BUFFER_SCRATCH, 0, &t.buffer) != GEIST_OK) {
        t.buffer = nullptr;
        return t;
    }
    memset(vt->buffer_map(t.buffer), 0, n * elem);
    vt->buffer_unmap(t.buffer);
    return t;
}

/* INT8-KV attention (INT4 when int4, K and V packed two values a byte) at
 * one geometry, n_q queries at the end of a cache of n_q + 1 positions: the
 * probe must answer `want`, and the kernel must return GEIST_OK when it
 * says yes and GEIST_E_UNSUPPORTED when it says no (its entry checks refuse
 * before reading anything). */
static int kv_quant_agreement(struct geist_backend *be,
                              bool                  int4,
                              size_t                n_q,
                              size_t                n_q_heads,
                              size_t                n_kv_heads,
                              size_t                head_dim,
                              bool                  want) {
    const struct geist_backend_fused *fused = geist_backend_fused_tbl(be);
    const struct geist_fusion_query   q     = {.op         = int4 ? GEIST_FUSED_ATTN_KV_INT4
                                                                  : GEIST_FUSED_ATTN_KV_INT8,
                                               .m          = n_q,
                                               .head_dim   = head_dim,
                                               .n_q_heads  = n_q_heads,
                                               .n_kv_heads = n_kv_heads};
    const char *const                 op    = int4 ? "attention_kv_int4" : "attention_kv_int8";
    const bool                        yes = fused->supported != nullptr && fused->supported(be, &q);
    char                              msg[96];
    snprintf(msg, sizeof msg, "%s: the probe's answer", op);
    int           fails = geist_expect(yes == want, msg);
    const int64_t n_kv = (int64_t) n_q + 1, qh = (int64_t) n_q_heads, kh = (int64_t) n_kv_heads,
                  hd = (int64_t) head_dim, row = int4 ? hd / 2 : hd;
    const enum geist_dtype kvt = int4 ? GEIST_DTYPE_U8 : GEIST_DTYPE_I8;
    struct geist_tensor    tq =
            zeroed(be, GEIST_DTYPE_F32, sizeof(float), 3, (int64_t[]) {(int64_t) n_q, qh, hd});
    struct geist_tensor tk  = zeroed(be, kvt, 1, 3, (int64_t[]) {n_kv, kh, row});
    struct geist_tensor tv  = zeroed(be, kvt, 1, 3, (int64_t[]) {n_kv, kh, row});
    struct geist_tensor tks = zeroed(be, GEIST_DTYPE_F32, sizeof(float), 2, (int64_t[]) {n_kv, kh});
    struct geist_tensor tvs = zeroed(be, GEIST_DTYPE_F32, sizeof(float), 2, (int64_t[]) {n_kv, kh});
    struct geist_tensor to =
            zeroed(be, GEIST_DTYPE_F32, sizeof(float), 3, (int64_t[]) {(int64_t) n_q, qh, hd});
    struct geist_tensor *all[] = {&tq, &tk, &tv, &tks, &tvs, &to};
    bool                 made  = true;
    for (size_t i = 0; i < 6; i++) {
        made = made && all[i]->buffer != nullptr;
    }
    if (!made) {
        snprintf(msg, sizeof msg, "%s: buffers", op);
        fails += geist_expect(false, msg);
    } else {
        enum geist_status s;
        if (int4) {
            const struct geist_attention_kv_args args = {.q        = &tq,
                                                         .k        = &tk,
                                                         .k_scale  = &tks,
                                                         .v        = &tv,
                                                         .v_scale  = &tvs,
                                                         .out      = &to,
                                                         .q_offset = 1};
            s                                         = fused->attention_kv_int4(be, &args);
        } else {
            const struct geist_attention_kv_args args = {.q        = &tq,
                                                         .k        = &tk,
                                                         .k_scale  = &tks,
                                                         .v        = &tv,
                                                         .v_scale  = &tvs,
                                                         .out      = &to,
                                                         .q_offset = 1};
            s                                         = fused->attention_kv_int8(be, &args);
        }
        snprintf(msg,
                 sizeof msg,
                 yes ? "probe said yes: %s must return GEIST_OK"
                     : "probe said no: %s must refuse with GEIST_E_UNSUPPORTED",
                 op);
        fails += geist_expect(s == (yes ? GEIST_OK : GEIST_E_UNSUPPORTED), msg);
    }
    for (size_t i = 0; i < 6; i++) {
        if (all[i]->buffer != nullptr) {
            be->desc->vtbl->buffer_destroy(be, all[i]->buffer);
        }
    }
    return fails;
}

/* attn_qkv_prep, q only, over [1, 2, head_dim]: the probe must answer
 * `want` (-1 = backend-dependent) and the kernel must succeed when it
 * says yes and refuse when it says no. The plan binds the kernel with no
 * fallback, so a yes the kernel refuses fails the layer (#474). */
static int qkv_prep_agreement(struct geist_backend *be, size_t head_dim, int want) {
    const struct geist_backend_fused *fused = geist_backend_fused_tbl(be);
    const struct geist_fusion_query   q     = {.op         = GEIST_FUSED_ATTN_QKV_PREP,
                                               .m          = 1,
                                               .head_dim   = head_dim,
                                               .n_q_heads  = 2,
                                               .n_kv_heads = 1};
    const bool                        yes = fused->supported != nullptr && fused->supported(be, &q);
    int                               fails = 0;
    char                              msg[96];
    if (want >= 0) {
        snprintf(msg, sizeof msg, "attn_qkv_prep head_dim %zu: the probe's answer", head_dim);
        fails += geist_expect(yes == (want == 1), msg);
    }
    const int64_t        hd = (int64_t) head_dim;
    struct geist_tensor  tq = zeroed(be, GEIST_DTYPE_F32, sizeof(float), 3, (int64_t[]) {1, 2, hd});
    struct geist_tensor  tw = zeroed(be, GEIST_DTYPE_F32, sizeof(float), 1, (int64_t[]) {hd});
    struct geist_tensor  tc = zeroed(be, GEIST_DTYPE_F32, sizeof(float), 2, (int64_t[]) {1, hd});
    struct geist_tensor  ts = zeroed(be, GEIST_DTYPE_F32, sizeof(float), 2, (int64_t[]) {1, hd});
    struct geist_tensor *all[] = {&tq, &tw, &tc, &ts};
    bool                 made  = true;
    for (size_t i = 0; i < 4; i++) {
        made = made && all[i]->buffer != nullptr;
    }
    if (!made) {
        fails += geist_expect(false, "attn_qkv_prep: buffers");
    } else {
        const enum geist_status s = fused->attn_qkv_prep(be,
                                                         &tq,
                                                         nullptr,
                                                         nullptr,
                                                         &tw,
                                                         nullptr,
                                                         nullptr,
                                                         &tc,
                                                         &ts,
                                                         1e-6f,
                                                         0,
                                                         nullptr,
                                                         nullptr);
        snprintf(msg,
                 sizeof msg,
                 yes ? "probe said yes: attn_qkv_prep head_dim %zu must return GEIST_OK"
                     : "probe said no: attn_qkv_prep head_dim %zu must refuse",
                 head_dim);
        fails += geist_expect(yes ? s == GEIST_OK : s != GEIST_OK, msg);
    }
    for (size_t i = 0; i < 4; i++) {
        if (all[i]->buffer != nullptr) {
            be->desc->vtbl->buffer_destroy(be, all[i]->buffer);
        }
    }
    return fails;
}

/* Decode gate/up front, plain and with the folded norm, at a geometry no
 * backend runs (`n_in` not a whole Q4_K block, or `n_out` not a multiple
 * of the norm kernel's 8-row tile): the probe must say no and the entry
 * must refuse before it reads a weight byte. */
static int
ffn_gate_up_refusal(struct geist_backend *be, bool with_norm, int64_t n_in, int64_t n_out) {
    const struct geist_backend_fused *fused = geist_backend_fused_tbl(be);
    if (with_norm ? fused->ffn_norm_gate_up == nullptr : fused->ffn_gate_up == nullptr) {
        return 0;
    }
    const struct geist_weight w = {
            .dtype = GEIST_DTYPE_Q4_K, .n_in = (int32_t) n_in, .n_out = (int32_t) n_out};
    const struct geist_fusion_query q  = {.op      = with_norm ? GEIST_FUSED_FFN_NORM_GATE_UP
                                                               : GEIST_FUSED_FFN_GATE_UP,
                                          .m       = 1,
                                          .d_model = (size_t) n_in,
                                          .inter   = (size_t) n_out,
                                          .gate_w  = &w,
                                          .up_w    = &w};
    const char *const               op = with_norm ? "ffn_norm_gate_up" : "ffn_gate_up";
    char                            msg[96];
    snprintf(msg,
             sizeof msg,
             "%s %lldx%lld: the probe says no",
             op,
             (long long) n_out,
             (long long) n_in);
    int fails = geist_expect(fused->supported == nullptr || !fused->supported(be, &q), msg);
    struct geist_tensor  tx = zeroed(be, GEIST_DTYPE_F32, sizeof(float), 2, (int64_t[]) {1, n_in});
    struct geist_tensor  tn = zeroed(be, GEIST_DTYPE_F32, sizeof(float), 1, (int64_t[]) {n_in});
    struct geist_tensor  tg = zeroed(be, GEIST_DTYPE_Q4_K, 1, 2, (int64_t[]) {n_out, n_in});
    struct geist_tensor  tu = zeroed(be, GEIST_DTYPE_Q4_K, 1, 2, (int64_t[]) {n_out, n_in});
    struct geist_tensor  ty = zeroed(be, GEIST_DTYPE_F32, sizeof(float), 2, (int64_t[]) {1, n_out});
    struct geist_tensor *all[] = {&tx, &tn, &tg, &tu, &ty};
    bool                 made  = true;
    for (size_t i = 0; i < 5; i++) {
        made = made && all[i]->buffer != nullptr;
    }
    if (!made) {
        snprintf(msg, sizeof msg, "%s: buffers", op);
        fails += geist_expect(false, msg);
    } else {
        const enum geist_status s =
                with_norm ? fused->ffn_norm_gate_up(be, &tx, &tn, 1e-6f, &tg, &tu, &ty)
                          : fused->ffn_gate_up(be, &tx, &tg, &tu, &ty);
        snprintf(msg, sizeof msg, "probe said no: %s must refuse", op);
        fails += geist_expect(s != GEIST_OK, msg);
    }
    for (size_t i = 0; i < 5; i++) {
        if (all[i]->buffer != nullptr) {
            be->desc->vtbl->buffer_destroy(be, all[i]->buffer);
        }
    }
    return fails;
}

/* hadamard_rotate over [2, width] rows: the probe's answer must match the
 * kernel's. `want` is the answer every backend gives; -1 = backend-
 * dependent (only the agreement is checked). */
static int hadamard_agreement(struct geist_backend *be,
                              size_t                width,
                              size_t                block,
                              size_t                perm_hd,
                              size_t                perm_nk,
                              size_t                perm_rep,
                              bool                  inverse,
                              int                   want) {
    const struct geist_backend_fused *fused = geist_backend_fused_tbl(be);
    const struct geist_fusion_query   q     = {.op       = GEIST_FUSED_HADAMARD_ROTATE,
                                               .m        = 2,
                                               .width    = width,
                                               .block    = block,
                                               .perm_hd  = perm_hd,
                                               .perm_nk  = perm_nk,
                                               .perm_rep = perm_rep,
                                               .inverse  = inverse};
    const bool                        yes = fused->supported != nullptr && fused->supported(be, &q);
    char                              msg[128];
    int                               fails = 0;
    if (want >= 0) {
        snprintf(msg,
                 sizeof msg,
                 "hadamard_rotate width %zu block %zu rep %zu%s: the probe's answer",
                 width,
                 block,
                 perm_rep,
                 inverse ? " inverse" : "");
        fails += geist_expect(yes == (want == 1), msg);
    }
    const int64_t       shape[2] = {2, (int64_t) width};
    struct geist_tensor tx       = zeroed(be, GEIST_DTYPE_F32, sizeof(float), 2, shape);
    struct geist_tensor ty       = zeroed(be, GEIST_DTYPE_F32, sizeof(float), 2, shape);
    if (tx.buffer == nullptr || ty.buffer == nullptr) {
        fails += geist_expect(false, "hadamard_rotate: buffers");
    } else {
        const struct geist_hadamard_args args = {.x        = &tx,
                                                 .y        = &ty,
                                                 .block    = block,
                                                 .perm_hd  = perm_hd,
                                                 .perm_nk  = perm_nk,
                                                 .perm_rep = perm_rep,
                                                 .inverse  = inverse};
        const enum geist_status          s    = fused->hadamard_rotate(be, &args);
        snprintf(msg,
                 sizeof msg,
                 yes ? "probe said yes: hadamard_rotate (width %zu block %zu) must return OK"
                     : "probe said no: hadamard_rotate (width %zu block %zu) must refuse",
                 width,
                 block);
        fails += geist_expect(yes ? s == GEIST_OK : s != GEIST_OK, msg);
    }
    if (tx.buffer != nullptr) {
        be->desc->vtbl->buffer_destroy(be, tx.buffer);
    }
    if (ty.buffer != nullptr) {
        be->desc->vtbl->buffer_destroy(be, ty.buffer);
    }
    return fails;
}

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

    /* ---- PLE block negative agreement: the metal kernels are decode
     * GEMVs, so an m>1 probe must answer no. The missing m check here
     * bound fuse_ple_block_mN and hard-failed every gemma4 Metal prefill
     * at layer 0 (found 2026-08-27). */
    {
        struct geist_weight       gate = {.dtype = GEIST_DTYPE_F32, .n_in = 64, .n_out = 32};
        struct geist_weight       up   = {.dtype = GEIST_DTYPE_F32, .n_in = 32, .n_out = 64};
        struct geist_fusion_query pq   = {.op      = GEIST_FUSED_PLE_BLOCK,
                                          .m       = 4,
                                          .d_model = 64,
                                          .inter   = 32,
                                          .gate_w  = &gate,
                                          .up_w    = &up};
        const bool probe_yes           = fused->supported != nullptr && fused->supported(be, &pq);
        if (fused->ple_block != nullptr && probe_yes) {
            fails += geist_expect(false, "PLE probe accepts m>1 but the backend has no mN kernel");
        }
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
                fused->ffn_geglu_q4q6_mN(be, 4, 100, 300, x, &gate, &up, &down, nullptr, y);
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

    /* ---- INT8- and INT4-KV attention: a prefill chunk of 4 heads on 2; a
     * head_dim past 512 and 3 query heads on 2 KV heads, which no kernel
     * runs, and an odd head_dim, which cannot be packed. */
    if (fused->attention_kv_int8 != nullptr) {
        fails += kv_quant_agreement(be, false, 2, 4, 2, 64, true);
        fails += kv_quant_agreement(be, false, 1, 2, 1, 520, false);
        fails += kv_quant_agreement(be, false, 1, 3, 2, 64, false);
    }
    if (fused->attention_kv_int4 != nullptr) {
        fails += kv_quant_agreement(be, true, 2, 4, 2, 64, true);
        fails += kv_quant_agreement(be, true, 1, 2, 1, 520, false);
        fails += kv_quant_agreement(be, true, 1, 3, 2, 64, false);
        fails += kv_quant_agreement(be, true, 1, 2, 1, 63, false);
    }

    /* ---- attn_qkv_prep: an even head_dim the kernels run, one past the
     * Vulkan kernel's 512-float shared row, and an odd one. */
    if (fused->attn_qkv_prep != nullptr) {
        fails += qkv_prep_agreement(be, 64, 1);
        fails += qkv_prep_agreement(be, 520, -1);
        fails += qkv_prep_agreement(be, 63, 0);
    }

    /* ---- Decode gate/up front: refused geometries. */
    fails += ffn_gate_up_refusal(be, false, 100, 64);
    fails += ffn_gate_up_refusal(be, true, 256, 60);

    /* ---- gelu_tanh_mul_scaled: yes -> the kernel runs [2, 64]. */
    {
        struct geist_fusion_query sq = {
                .op = GEIST_FUSED_GELU_TANH_MUL_SCALED, .m = 2, .d_model = 64, .inter = 64};
        if (fused->supported != nullptr && fused->supported(be, &sq)) {
            struct geist_tensor tx =
                    zeroed(be, GEIST_DTYPE_F32, sizeof(float), 2, (int64_t[]) {2, 64});
            struct geist_tensor tz =
                    zeroed(be, GEIST_DTYPE_F32, sizeof(float), 2, (int64_t[]) {2, 64});
            float scale[64];
            for (size_t i = 0; i < 64; i++) {
                scale[i] = 1.0f;
            }
            if (tx.buffer != nullptr && tz.buffer != nullptr) {
                fails += geist_expect(fused->gelu_tanh_mul_scaled(be, &tx, &tz, scale, &tx) ==
                                              GEIST_OK,
                                      "probe said yes: gelu_tanh_mul_scaled must return GEIST_OK");
            }
            if (tx.buffer != nullptr)
                v->buffer_destroy(be, tx.buffer);
            if (tz.buffer != nullptr)
                v->buffer_destroy(be, tz.buffer);
        }
    }

    /* ---- hadamard_rotate: plain and grouped-value forward, the inverse,
     * a permuted inverse and a permutation that does not multiply out to
     * the width (no backend runs those), and blocks at the edges of
     * metal's kernel range. */
    if (fused->hadamard_rotate != nullptr) {
        fails += hadamard_agreement(be, 64, 16, 0, 0, 0, false, 1);
        fails += hadamard_agreement(be, 64, 16, 8, 2, 4, false, 1);
        fails += hadamard_agreement(be, 64, 64, 0, 0, 0, true, 1);
        fails += hadamard_agreement(be, 64, 16, 8, 2, 4, true, 0);
        fails += hadamard_agreement(be, 64, 16, 8, 2, 3, false, 0);
        fails += hadamard_agreement(be, 64, 12, 0, 0, 0, false, 0);
        fails += hadamard_agreement(be, 64, 2, 0, 0, 0, false, -1);
        fails += hadamard_agreement(be, 8192, 8192, 0, 0, 0, false, -1);
    } else {
        struct geist_fusion_query hq = {
                .op = GEIST_FUSED_HADAMARD_ROTATE, .m = 1, .width = 64, .block = 16};
        fails += geist_expect(fused->supported == nullptr || !fused->supported(be, &hq),
                              "no hadamard_rotate slot, but the probe says yes");
    }

    printf("  %s: probe/kernel agreement ok\n", name);
    geist_backend_destroy(be);
    return fails;
}

int main(void) {
    int fails = 0;
    fails += check_backend("cpu_x86");
    fails += check_backend("cpu_neon");
    fails += check_backend("cpu_scalar");
    fails += check_backend("metal");
    fails += check_backend("vulkan");
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("fused probe agreement: pass\n");
    return GEIST_TEST_PASS;
}

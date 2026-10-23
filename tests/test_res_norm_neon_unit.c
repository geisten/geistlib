/*
 * test_res_norm_neon_unit — cpu_neon rmsnorm_add, rmsnorm_add_norm and
 * attn_qkv_prep (#195).
 *
 * The exec plan swaps the separate ops for these fusions wherever the probe
 * says yes, and cpu_neon's decode is W8A8: one flipped bit in a residual
 * can flip an int8 rounding downstream. So each must match the separate
 * ops byte for byte, on the same backend:
 *
 *   1. rmsnorm_add      == rmsnorm(x) * w, then add(res, .)
 *   2. rmsnorm_add_norm == the same (or add(res, x) with w nullptr), then
 *      the layer scale's multiply (scale_output) and rmsnorm(y) * z_w
 *   3. rows 1, 7 (the calling thread), 8, 64 and 65 (across threads), d
 *      1536 (Gemma 4 E2B) and 3072 (Llama 3.2 3B), scales 1 and 0.37,
 *      and y aliasing res, as the arch calls it
 *   4. the probe: yes up to 8192 columns, no past it
 *   5. attn_qkv_prep == per-head rmsnorm of q, RoPE of q; rmsnorm of k
 *      and v, RoPE of k, both appended at q_position (F32 and F16 caches),
 *      and q only for a KV-shared layer; Gemma 4 E2B's 8 q heads, 1 KV
 *      head, head_dim 256, for 1, 7, 8 and 64 tokens
 *   6. attn_rope_append == the same without v's norm, with blk_q_prep's
 *      q scale, without norms (Llama) and with them over a partial RoPE
 *      (Qwen3-like)
 *
 * SKIPs if cpu_neon is not built.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t g_seed = 0x9E3779B9u;

static float frand(void) {
    g_seed = g_seed * 1664525u + 1013904223u;
    return (float) (g_seed >> 8) / 16777216.0f; /* [0, 1) */
}

/* An F32 [rows, cols] tensor over a fresh buffer holding `data`. */
static struct geist_tensor
upload(struct geist_backend *be, size_t rows, size_t cols, const float *data) {
    struct geist_buffer *b = nullptr;
    be->desc->vtbl->buffer_create(
            be, rows * cols * sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &b);
    if (b != nullptr) {
        be->desc->vtbl->buffer_upload(b, rows * cols * sizeof(float), (const uint8_t *) data);
    }
    return (struct geist_tensor) {.buffer = b,
                                  .dtype  = GEIST_DTYPE_F32,
                                  .layout = GEIST_LAYOUT_DENSE,
                                  .ndim   = 2,
                                  .shape  = {(int64_t) rows, (int64_t) cols},
                                  .stride = {(int64_t) cols, 1}};
}

static struct geist_tensor vec(struct geist_backend *be, size_t cols, const float *data) {
    struct geist_tensor t = upload(be, 1, cols, data);
    t.ndim                = 1;
    t.shape[0]            = (int64_t) cols;
    t.stride[0]           = 1;
    return t;
}

static void download(struct geist_backend *be, size_t n, const struct geist_tensor *t, float *out) {
    be->desc->vtbl->buffer_download(n * sizeof(float), (uint8_t *) out, t->buffer);
}

static void destroy(struct geist_backend *be, struct geist_tensor *t) {
    be->desc->vtbl->buffer_destroy(be, t->buffer);
}

/* The separate ops as the arch issues them: rmsnorm into tmp, add, the
 * host scale loop of scale_output (skipped at 1), and the next pre-norm. */
static int reference(struct geist_backend *be,
                     size_t                rows,
                     size_t                cols,
                     float                 eps,
                     float                 scale,
                     const float          *res,
                     const float          *x,
                     const float          *w,
                     const float          *z_w,
                     float                *y_out,
                     float                *z_out) {
    const struct geist_backend_primitives *prims = be->desc->prims;
    const size_t                           n     = rows * cols;
    struct geist_tensor                    tr    = upload(be, rows, cols, res);
    struct geist_tensor                    tx    = upload(be, rows, cols, x);
    struct geist_tensor                    ty    = upload(be, rows, cols, res);
    struct geist_tensor                    tz    = upload(be, rows, cols, res);
    int                                    fails = 0;
    if (w != nullptr) {
        struct geist_tensor tw = vec(be, cols, w);
        fails += geist_expect(prims->rmsnorm(be, &tx, &tw, eps, &tx) == GEIST_OK, "rmsnorm ran");
        destroy(be, &tw);
    }
    fails += geist_expect(prims->add(be, &tr, &tx, &ty) == GEIST_OK, "add ran");
    if (scale != 1.0f) {
        float *yp = be->desc->vtbl->buffer_map(ty.buffer);
        for (size_t i = 0; i < n; i++) {
            yp[i] *= scale;
        }
        be->desc->vtbl->buffer_unmap(ty.buffer);
    }
    download(be, n, &ty, y_out);
    if (z_w != nullptr) {
        struct geist_tensor tzw = vec(be, cols, z_w);
        fails += geist_expect(prims->rmsnorm(be, &ty, &tzw, eps, &tz) == GEIST_OK, "rmsnorm z ran");
        download(be, n, &tz, z_out);
        destroy(be, &tzw);
    }
    destroy(be, &tr);
    destroy(be, &tx);
    destroy(be, &ty);
    destroy(be, &tz);
    return fails;
}

static int check(struct geist_backend *be, size_t rows, size_t cols, float scale, bool with_w) {
    const struct geist_backend_fused *fused = geist_backend_fused_tbl(be);
    const size_t                      n     = rows * cols;
    const float                       eps   = 1e-6f;
    float                            *res   = xmalloc(n * sizeof(float));
    float                            *x     = xmalloc(n * sizeof(float));
    float                            *w     = xmalloc(cols * sizeof(float));
    float                            *z_w   = xmalloc(cols * sizeof(float));
    float                            *ry    = xmalloc(n * sizeof(float));
    float                            *rz    = xmalloc(n * sizeof(float));
    float                            *y     = xmalloc(n * sizeof(float));
    float                            *z     = xmalloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        res[i] = (frand() * 2.0f - 1.0f) * 40.0f;
        x[i]   = (frand() * 2.0f - 1.0f) * 3.0f;
    }
    for (size_t i = 0; i < cols; i++) {
        w[i]   = frand() * 2.0f;
        z_w[i] = frand() * 1.5f + 0.25f;
    }
    int  fails = 0;
    char what[160];

    /* 1: rmsnorm_add, y aliasing res as the post-norm residual may */
    if (with_w && scale == 1.0f) {
        fails += reference(be, rows, cols, eps, 1.0f, res, x, w, nullptr, ry, nullptr);
        struct geist_tensor tr = upload(be, rows, cols, res);
        struct geist_tensor tx = upload(be, rows, cols, x);
        struct geist_tensor tw = vec(be, cols, w);
        fails += geist_expect(fused->rmsnorm_add(be, &tr, &tx, &tw, eps, &tr) == GEIST_OK,
                              "rmsnorm_add ran");
        download(be, n, &tr, y);
        snprintf(what, sizeof what, "rows=%zu cols=%zu: rmsnorm_add == rmsnorm, add", rows, cols);
        fails += geist_expect(memcmp(ry, y, n * sizeof(float)) == 0, what);
        destroy(be, &tr);
        destroy(be, &tx);
        destroy(be, &tw);
    }

    /* 2: rmsnorm_add_norm, y separate and y aliasing res */
    fails += reference(be, rows, cols, eps, scale, res, x, with_w ? w : nullptr, z_w, ry, rz);
    for (int alias = 0; alias < 2; alias++) {
        struct geist_tensor  tr  = upload(be, rows, cols, res);
        struct geist_tensor  tx  = upload(be, rows, cols, x);
        struct geist_tensor  tw  = vec(be, cols, w);
        struct geist_tensor  tzw = vec(be, cols, z_w);
        struct geist_tensor  ty  = upload(be, rows, cols, x);
        struct geist_tensor  tz  = upload(be, rows, cols, x);
        struct geist_tensor *yt  = alias ? &tr : &ty;
        fails += geist_expect(
                fused->rmsnorm_add_norm(
                        be, eps, scale, &tr, &tx, with_w ? &tw : nullptr, &tzw, yt, &tz) ==
                        GEIST_OK,
                "rmsnorm_add_norm ran");
        download(be, n, yt, y);
        download(be, n, &tz, z);
        snprintf(what,
                 sizeof what,
                 "rows=%zu cols=%zu scale=%g w=%d alias=%d: y bit-identical",
                 rows,
                 cols,
                 (double) scale,
                 (int) with_w,
                 alias);
        fails += geist_expect(memcmp(ry, y, n * sizeof(float)) == 0, what);
        snprintf(what,
                 sizeof what,
                 "rows=%zu cols=%zu scale=%g w=%d alias=%d: z bit-identical",
                 rows,
                 cols,
                 (double) scale,
                 (int) with_w,
                 alias);
        fails += geist_expect(memcmp(rz, z, n * sizeof(float)) == 0, what);
        destroy(be, &tr);
        destroy(be, &tx);
        destroy(be, &tw);
        destroy(be, &tzw);
        destroy(be, &ty);
        destroy(be, &tz);
    }

    free(res);
    free(x);
    free(w);
    free(z_w);
    free(ry);
    free(rz);
    free(y);
    free(z);
    return fails;
}

/* 5: attn_qkv_prep against rmsnorm, rope_apply and the cache append. */
static struct geist_tensor
upload3(struct geist_backend *be, size_t a, size_t b, size_t c, const float *data) {
    struct geist_tensor t = upload(be, a * b, c, data);
    t.ndim                = 3;
    t.shape[0]            = (int64_t) a;
    t.shape[1]            = (int64_t) b;
    t.shape[2]            = (int64_t) c;
    t.stride[0]           = (int64_t) (b * c);
    t.stride[1]           = (int64_t) c;
    t.stride[2]           = 1;
    return t;
}

enum prep_mode { PREP_GEMMA, PREP_GEMMA_Q_ONLY, PREP_LLAMA, PREP_QWEN3 };

static int check_qkv(struct geist_backend *be, size_t seq, bool f16, enum prep_mode mode) {
    const bool gemma   = mode == PREP_GEMMA || mode == PREP_GEMMA_Q_ONLY;
    const bool with_kv = mode != PREP_GEMMA_Q_ONLY;
    const bool norms   = mode != PREP_LLAMA;
    /* Gemma 4 E2B 8/1 x 256 and no q scale; Llama-like 4/2 x 128 at
     * 1/sqrt(128), RoPE over the first 64 dims for Qwen3's case. */
    const float                            q_scale = gemma ? 1.0f : 0.08838834764831845f;
    const struct geist_backend_primitives *prims   = be->desc->prims;
    const struct geist_backend_fused      *fused   = geist_backend_fused_tbl(be);
    const struct geist_backend_vtbl       *vt      = be->desc->vtbl;
    const size_t qh = gemma ? 8 : 4, kvh = gemma ? 1 : 2, hd = gemma ? 256 : 128;
    const size_t nr = mode == PREP_QWEN3 ? 64 : hd, pos = 5, rows = pos + seq;
    const size_t nq = seq * qh * hd, nk = seq * kvh * hd, nc = rows * kvh * hd;
    const float  eps = 1e-6f;
    float       *q   = xmalloc(nq * sizeof(float));
    float       *k   = xmalloc(nk * sizeof(float));
    float       *v   = xmalloc(nk * sizeof(float));
    float       *cs  = xmalloc(seq * nr * sizeof(float));
    float       *sn  = xmalloc(seq * nr * sizeof(float));
    float       *w3  = xmalloc(3 * hd * sizeof(float));
    uint8_t     *ref = xmalloc(4 * (nq + 2 * nk + 2 * nc));
    uint8_t     *got = xmalloc(4 * (nq + 2 * nk + 2 * nc));
    for (size_t i = 0; i < nq; i++)
        q[i] = (frand() * 2.0f - 1.0f) * 9.0f;
    for (size_t i = 0; i < nk; i++) {
        k[i] = (frand() * 2.0f - 1.0f) * 9.0f;
        v[i] = (frand() * 2.0f - 1.0f) * 2.0f;
    }
    for (size_t i = 0; i < seq * nr; i++) {
        cs[i] = frand() * 2.0f - 1.0f;
        sn[i] = frand() * 2.0f - 1.0f;
    }
    for (size_t i = 0; i < hd; i++) {
        w3[i]          = frand() * 2.0f;
        w3[hd + i]     = frand() * 2.0f;
        w3[2 * hd + i] = 1.0f;
    }
    const size_t esz   = f16 ? sizeof(uint16_t) : sizeof(float);
    int          fails = 0;
    for (int fused_run = 0; fused_run < 2; fused_run++) {
        struct geist_tensor tq  = upload3(be, seq, qh, hd, q);
        struct geist_tensor tk  = upload3(be, seq, kvh, hd, k);
        struct geist_tensor tv  = upload3(be, seq, kvh, hd, v);
        struct geist_tensor tc  = upload(be, seq, nr, cs);
        struct geist_tensor ts  = upload(be, seq, nr, sn);
        struct geist_tensor tqw = vec(be, hd, w3), tkw = vec(be, hd, w3 + hd);
        struct geist_tensor tvw = vec(be, hd, w3 + 2 * hd);
        struct geist_tensor tkc = {0}, tvc = {0};
        for (int c = 0; c < 2; c++) {
            struct geist_tensor *t = c ? &tvc : &tkc;
            vt->buffer_create(be, nc * esz, GEIST_BUFFER_SCRATCH, GEIST_MEMORY_AUTO, &t->buffer);
            memset(vt->buffer_map(t->buffer), 0x5a, nc * esz);
            vt->buffer_unmap(t->buffer);
            t->dtype     = f16 ? GEIST_DTYPE_F16 : GEIST_DTYPE_F32;
            t->layout    = GEIST_LAYOUT_DENSE;
            t->ndim      = 3;
            t->shape[0]  = (int64_t) rows;
            t->shape[1]  = (int64_t) kvh;
            t->shape[2]  = (int64_t) hd;
            t->stride[0] = (int64_t) (kvh * hd);
            t->stride[1] = (int64_t) hd;
            t->stride[2] = 1;
        }
        if (fused_run && !gemma) {
            fails += geist_expect(fused->attn_rope_append(be,
                                                          &tq,
                                                          &tk,
                                                          &tv,
                                                          norms ? &tqw : nullptr,
                                                          norms ? &tkw : nullptr,
                                                          &tc,
                                                          &ts,
                                                          eps,
                                                          q_scale,
                                                          pos,
                                                          &tkc,
                                                          &tvc) == GEIST_OK,
                                  "attn_rope_append ran");
        } else if (fused_run) {
            fails += geist_expect(fused->attn_qkv_prep(be,
                                                       &tq,
                                                       with_kv ? &tk : nullptr,
                                                       with_kv ? &tv : nullptr,
                                                       &tqw,
                                                       with_kv ? &tkw : nullptr,
                                                       with_kv ? &tvw : nullptr,
                                                       &tc,
                                                       &ts,
                                                       eps,
                                                       pos,
                                                       with_kv ? &tkc : nullptr,
                                                       with_kv ? &tvc : nullptr) == GEIST_OK,
                                  "attn_qkv_prep ran");
        } else {
            struct geist_tensor tq2 = tq;
            tq2.ndim                = 2;
            tq2.shape[0]            = (int64_t) (seq * qh);
            tq2.shape[1]            = (int64_t) hd;
            tq2.stride[0]           = (int64_t) hd;
            tq2.stride[1]           = 1;
            if (norms) {
                fails += geist_expect(prims->rmsnorm(be, &tq2, &tqw, eps, &tq2) == GEIST_OK,
                                      "q norm");
            }
            fails += geist_expect(prims->rope_apply(be, &tq, &tc, &ts) == GEIST_OK, "q rope");
            if (q_scale != 1.0f) { /* blk_q_prep's host loop */
                float *qp = vt->buffer_map(tq.buffer);
                for (size_t i = 0; i < nq; i++) {
                    qp[i] *= q_scale;
                }
                vt->buffer_unmap(tq.buffer);
            }
            if (with_kv) {
                struct geist_tensor tk2 = tk, tv2 = tv;
                tk2.ndim = tv2.ndim = 2;
                tk2.shape[0] = tv2.shape[0] = (int64_t) (seq * kvh);
                tk2.shape[1] = tv2.shape[1] = (int64_t) hd;
                tk2.stride[0] = tv2.stride[0] = (int64_t) hd;
                tk2.stride[1] = tv2.stride[1] = 1;
                if (norms) {
                    fails += geist_expect(prims->rmsnorm(be, &tk2, &tkw, eps, &tk2) == GEIST_OK,
                                          "k norm");
                }
                if (gemma) {
                    fails += geist_expect(prims->rmsnorm(be, &tv2, &tvw, eps, &tv2) == GEIST_OK,
                                          "v norm");
                }
                fails += geist_expect(prims->rope_apply(be, &tk, &tc, &ts) == GEIST_OK, "k rope");
                if (f16) {
                    fails += geist_expect(fused->kv_append_f16(be, &tk, &tv, pos, &tkc, &tvc) ==
                                                  GEIST_OK,
                                          "kv_append_f16");
                } else {
                    uint8_t *kc = vt->buffer_map(tkc.buffer), *vc = vt->buffer_map(tvc.buffer);
                    download(be, nk, &tk, (float *) (kc + pos * kvh * hd * esz));
                    download(be, nk, &tv, (float *) (vc + pos * kvh * hd * esz));
                    vt->buffer_unmap(tkc.buffer);
                    vt->buffer_unmap(tvc.buffer);
                }
            }
        }
        uint8_t *out = fused_run ? got : ref;
        download(be, nq, &tq, (float *) out);
        download(be, nk, &tk, (float *) (out + 4 * nq));
        download(be, nk, &tv, (float *) (out + 4 * (nq + nk)));
        vt->buffer_download(nc * esz, out + 4 * (nq + 2 * nk), tkc.buffer);
        vt->buffer_download(nc * esz, out + 4 * (nq + 2 * nk) + nc * esz, tvc.buffer);
        struct geist_tensor *all[] = {&tq, &tk, &tv, &tc, &ts, &tqw, &tkw, &tvw, &tkc, &tvc};
        for (size_t i = 0; i < sizeof all / sizeof all[0]; i++) {
            destroy(be, all[i]);
        }
    }
    char what[128];
    snprintf(what,
             sizeof what,
             "%s seq=%zu %s: bit-identical to the separate ops",
             (const char *[]) {"attn_qkv_prep",
                               "attn_qkv_prep q only",
                               "attn_rope_append llama",
                               "attn_rope_append qwen3"}[mode],
             seq,
             f16 ? "f16" : "f32");
    fails += geist_expect(memcmp(ref, got, 4 * (nq + 2 * nk) + 2 * nc * esz) == 0, what);
    free(q);
    free(k);
    free(v);
    free(cs);
    free(sn);
    free(w3);
    free(ref);
    free(got);
    return fails;
}

int main(void) {
    struct geist_backend *be = nullptr;
    GEIST_SKIP_IF(geist_backend_create("cpu_neon", nullptr, nullptr, &be) != GEIST_OK,
                  "cpu_neon backend not compiled in");
    const struct geist_backend_fused *fused = geist_backend_fused_tbl(be);
    int                               fails = 0;

    fails += geist_expect(fused->rmsnorm_add != nullptr && fused->rmsnorm_add_norm != nullptr &&
                                  fused->supported != nullptr,
                          "cpu_neon has both residual-norm fusions");
    if (fails != 0) {
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }
    const enum geist_fused_op ops[] = {GEIST_FUSED_RMSNORM_ADD, GEIST_FUSED_RMSNORM_ADD_NORM};
    for (size_t k = 0; k < 2; k++) {
        struct geist_fusion_query q = {.op = ops[k], .m = 512, .d_model = 8192};
        fails += geist_expect(fused->supported(be, &q), "probe: yes at 8192 columns");
        q.d_model = 8193;
        fails += geist_expect(!fused->supported(be, &q), "probe: no past 8192 columns");
    }

    const size_t rows[]   = {1, 7, 8, 64, 65};
    const size_t cols[]   = {1536, 3072};
    const float  scales[] = {1.0f, 0.37f};
    for (size_t r = 0; r < sizeof rows / sizeof rows[0]; r++) {
        for (size_t c = 0; c < sizeof cols / sizeof cols[0]; c++) {
            for (size_t s = 0; s < sizeof scales / sizeof scales[0]; s++) {
                fails += check(be, rows[r], cols[c], scales[s], true);
                fails += check(be, rows[r], cols[c], scales[s], false);
            }
        }
    }

    fails += geist_expect(fused->attn_qkv_prep != nullptr && fused->attn_rope_append != nullptr,
                          "cpu_neon has attn_qkv_prep and attn_rope_append");
    const size_t seqs[] = {1, 7, 8, 64};
    for (size_t i = 0; fails == 0 && i < sizeof seqs / sizeof seqs[0]; i++) {
        for (int mode = PREP_GEMMA; mode <= PREP_QWEN3; mode++) {
            fails += check_qkv(be, seqs[i], false, (enum prep_mode) mode);
            if (mode != PREP_GEMMA_Q_ONLY) {
                fails += check_qkv(be, seqs[i], true, (enum prep_mode) mode);
            }
        }
    }

    geist_backend_destroy(be);
    if (fails == 0) {
        printf("PASS: cpu_neon rmsnorm_add / rmsnorm_add_norm / attn_qkv_prep / "
               "attn_rope_append bit-identical to the separate ops\n");
    }
    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}

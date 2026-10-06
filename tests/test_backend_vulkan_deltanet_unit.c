/* Vulkan Gated-DeltaNet mixer parity. Drives fused->deltanet_mix directly
 * with non-zero recurrent state and compares the mixer output (z) plus both
 * advanced state tensors against a scalar oracle that mirrors the host path
 * in layer_deltanet.c. The two state tensors are KV_CACHE-role buffers, which
 * this backend keeps in VRAM (not host-mappable) — the placement the engine
 * uses for a real session.
 *
 * Each geometry runs a multi-row prefill and then a single decode row that
 * continues from the advanced state: that catches state reset, double
 * advance and the seq == 1 path. Covers the qwen35 shapes that matter: a
 * 1:1 head ratio, the 27B-style 1:3 k/v-head sharing (tiled hk = hv % n_kh),
 * 128-wide heads, odd sizes, and conv kernels other than 4.
 *
 * Load-time geometry (#470): the probe answers exactly the shader limits
 * (d_k <= 256, d_v <= 128, conv 2..8), and an in-memory qwen35 hybrid
 * whose head_k is past them is refused at load, naming the limit, because
 * the host fallback cannot map this backend's VRAM state. One within the
 * limits loads and prefills.
 *
 * SKIPs (exit 77) when no Vulkan runtime/device is present. */
#include "deltanet_ref.h"
#include "model_fixtures.h"
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct geom {
    const char *name;
    size_t      nkh, nvh, dk, dv, K, seq;
    double      tol;
};

static bool make_buf(struct geist_backend  *be,
                     struct geist_buffer  **out,
                     enum geist_buffer_role role,
                     const float           *src,
                     size_t                 n) {
    const struct geist_backend_vtbl *v = be->desc->vtbl;
    return v->buffer_create(be, n * sizeof(float), role, GEIST_MEMORY_AUTO, out) == GEIST_OK &&
           v->buffer_upload(*out, n * sizeof(float), (const uint8_t *) src) == GEIST_OK;
}

static bool run_geom(struct geist_backend *be, const struct geom *g) {
    const struct geist_backend_fused *f    = geist_backend_fused_tbl(be);
    const struct geist_backend_vtbl  *v    = be->desc->vtbl;
    const size_t                      keyd = g->nkh * g->dk, vd = g->nvh * g->dv;
    const size_t                      cd = 2 * keyd + vd, cn = (g->K - 1) * cd;
    const size_t                      sn = g->nvh * g->dk * g->dv;

    float *qkv   = geist_test_fill(g->seq * cd, 0.17f, 0.0f, 0.4f, 0.0f),
          *z     = geist_test_fill(g->seq * vd, 0.11f, 0.0f, 0.3f, 0.0f);
    float *beta  = geist_test_fill(g->seq * g->nvh, 0.13f, 0.0f, 0.6f, -0.1f);
    float *alpha = geist_test_fill(g->seq * g->nvh, 0.07f, 0.0f, 0.5f, -0.2f);
    float *cw    = geist_test_fill(cd * g->K, 0.09f, 0.0f, 0.2f, 0.0f),
          *aw    = malloc(g->nvh * sizeof(float));
    float *dt    = malloc(g->nvh * sizeof(float)),
          *nw    = geist_test_fill(g->dv, 0.31f, 0.0f, 0.1f, 0.9f);
    float *cs    = geist_test_fill(cn, 0.05f, 0.0f, 0.1f, 0.0f),
          *st    = geist_test_fill(sn, 0.03f, 0.0f, 0.05f, 0.0f);
    for (size_t i = 0; i < g->nvh; i++) {
        aw[i] = -0.5f - (float) i * 0.1f;
        dt[i] = 0.1f + (float) i * 0.03f;
    }
    float *q_ref  = malloc(g->seq * cd * sizeof(float)),
          *z_ref  = malloc(g->seq * vd * sizeof(float));
    float *cs_ref = malloc(cn * sizeof(float)), *s_ref = malloc(sn * sizeof(float));
    memcpy(q_ref, qkv, g->seq * cd * sizeof(float));
    memcpy(z_ref, z, g->seq * vd * sizeof(float));
    memcpy(cs_ref, cs, cn * sizeof(float));
    memcpy(s_ref, st, sn * sizeof(float));
    deltanet_mix_ref(g->seq,
                     g->nkh,
                     g->nvh,
                     g->dk,
                     g->dv,
                     g->K,
                     1e-6f,
                     beta,
                     alpha,
                     cw,
                     aw,
                     dt,
                     nw,
                     q_ref,
                     z_ref,
                     cs_ref,
                     s_ref);

    struct geist_buffer *bq = nullptr, *bz = nullptr, *bb = nullptr, *ba = nullptr, *bw = nullptr,
                        *bA = nullptr, *bd = nullptr, *bn = nullptr, *bc = nullptr, *bs = nullptr;
    bool ok = make_buf(be, &bq, GEIST_BUFFER_SCRATCH, qkv, g->seq * cd) &&
              make_buf(be, &bz, GEIST_BUFFER_SCRATCH, z, g->seq * vd) &&
              make_buf(be, &bb, GEIST_BUFFER_SCRATCH, beta, g->seq * g->nvh) &&
              make_buf(be, &ba, GEIST_BUFFER_SCRATCH, alpha, g->seq * g->nvh) &&
              make_buf(be, &bw, GEIST_BUFFER_WEIGHT, cw, cd * g->K) &&
              make_buf(be, &bA, GEIST_BUFFER_WEIGHT, aw, g->nvh) &&
              make_buf(be, &bd, GEIST_BUFFER_WEIGHT, dt, g->nvh) &&
              make_buf(be, &bn, GEIST_BUFFER_WEIGHT, nw, g->dv) &&
              make_buf(be, &bc, GEIST_BUFFER_KV_CACHE, cs, cn) &&
              make_buf(be, &bs, GEIST_BUFFER_KV_CACHE, st, sn);
    if (!ok) {
        fprintf(stderr, "FAIL [%s]: buffer setup\n", g->name);
        return false;
    }
    struct geist_tensor tq = geist_test_tensor_f32(bq, 2, (int64_t) g->seq, (int64_t) cd, 0);
    struct geist_tensor tz = geist_test_tensor_f32(bz, 2, (int64_t) g->seq, (int64_t) vd, 0);
    struct geist_tensor tb = geist_test_tensor_f32(bb, 2, (int64_t) g->seq, (int64_t) g->nvh, 0);
    struct geist_tensor ta = geist_test_tensor_f32(ba, 2, (int64_t) g->seq, (int64_t) g->nvh, 0);
    struct geist_tensor tw = geist_test_tensor_f32(bw, 2, (int64_t) cd, (int64_t) g->K, 0);
    struct geist_tensor tA = geist_test_tensor_f32(bA, 1, (int64_t) g->nvh, 0, 0);
    struct geist_tensor td = geist_test_tensor_f32(bd, 1, (int64_t) g->nvh, 0, 0);
    struct geist_tensor tn = geist_test_tensor_f32(bn, 1, (int64_t) g->dv, 0, 0);
    struct geist_tensor tc = geist_test_tensor_f32(bc, 2, (int64_t) (g->K - 1), (int64_t) cd, 0);
    struct geist_tensor ts =
            geist_test_tensor_f32(bs, 3, (int64_t) g->nvh, (int64_t) g->dk, (int64_t) g->dv);

    struct geist_deltanet_mix_args args = {.qkv         = &tq,
                                           .z           = &tz,
                                           .beta        = &tb,
                                           .alpha       = &ta,
                                           .conv_w      = &tw,
                                           .ssm_a       = &tA,
                                           .dt_bias     = &td,
                                           .norm_w      = &tn,
                                           .conv_state  = &tc,
                                           .delta_state = &ts,
                                           .seq         = g->seq,
                                           .n_k_heads   = g->nkh,
                                           .n_v_heads   = g->nvh,
                                           .head_k      = g->dk,
                                           .head_v      = g->dv,
                                           .conv_kernel = g->K,
                                           .eps         = 1e-6f};
    const enum geist_status        s    = f->deltanet_mix(be, &args);
    if (s != GEIST_OK) {
        fprintf(stderr,
                "FAIL [%s]: prefill dispatch (%d): %s\n",
                g->name,
                (int) s,
                geist_backend_errmsg(be));
        return false;
    }
    float *z_got = malloc(g->seq * vd * sizeof(float)), *cs_got = malloc(cn * sizeof(float));
    float *s_got = malloc(sn * sizeof(float));
    ok = v->buffer_download(g->seq * vd * sizeof(float), (uint8_t *) z_got, bz) == GEIST_OK &&
         v->buffer_download(cn * sizeof(float), (uint8_t *) cs_got, bc) == GEIST_OK &&
         v->buffer_download(sn * sizeof(float), (uint8_t *) s_got, bs) == GEIST_OK;
    const double ze = geist_test_max_abs(g->seq * vd, z_got, z_ref),
                 ce = geist_test_max_abs(cn, cs_got, cs_ref);
    const double se = geist_test_max_abs(sn, s_got, s_ref);
    printf("  %-22s prefill seq=%-3zu  z %.2e  conv-state %.2e  delta-state %.2e\n",
           g->name,
           g->seq,
           ze,
           ce,
           se);
    ok = ok && ze < g->tol && ce < 1e-6 && se < g->tol;

    /* One decode row continuing from the advanced state. */
    float *qd  = geist_test_fill(cd, 0.19f, 0.0f, 0.35f, 0.0f),
          *zd  = geist_test_fill(vd, 0.23f, 0.0f, 0.25f, 0.0f);
    float *bd1 = malloc(g->nvh * sizeof(float)), *ad1 = malloc(g->nvh * sizeof(float));
    for (size_t i = 0; i < g->nvh; i++) {
        bd1[i] = 0.2f - (float) i * 0.3f;
        ad1[i] = -0.1f + (float) i * 0.2f;
    }
    float *qd_ref = malloc(cd * sizeof(float)), *zd_ref = malloc(vd * sizeof(float));
    memcpy(qd_ref, qd, cd * sizeof(float));
    memcpy(zd_ref, zd, vd * sizeof(float));
    deltanet_mix_ref(1,
                     g->nkh,
                     g->nvh,
                     g->dk,
                     g->dv,
                     g->K,
                     1e-6f,
                     bd1,
                     ad1,
                     cw,
                     aw,
                     dt,
                     nw,
                     qd_ref,
                     zd_ref,
                     cs_ref,
                     s_ref);
    ok       = ok && v->buffer_upload(bq, cd * sizeof(float), (const uint8_t *) qd) == GEIST_OK &&
               v->buffer_upload(bz, vd * sizeof(float), (const uint8_t *) zd) == GEIST_OK &&
               v->buffer_upload(bb, g->nvh * sizeof(float), (const uint8_t *) bd1) == GEIST_OK &&
               v->buffer_upload(ba, g->nvh * sizeof(float), (const uint8_t *) ad1) == GEIST_OK;
    tq       = geist_test_tensor_f32(bq, 2, 1, (int64_t) cd, 0);
    tz       = geist_test_tensor_f32(bz, 2, 1, (int64_t) vd, 0);
    tb       = geist_test_tensor_f32(bb, 2, 1, (int64_t) g->nvh, 0);
    ta       = geist_test_tensor_f32(ba, 2, 1, (int64_t) g->nvh, 0);
    args.seq = 1;
    ok       = ok && f->deltanet_mix(be, &args) == GEIST_OK;
    float *zd_got = malloc(vd * sizeof(float));
    ok = ok && v->buffer_download(vd * sizeof(float), (uint8_t *) zd_got, bz) == GEIST_OK &&
         v->buffer_download(cn * sizeof(float), (uint8_t *) cs_got, bc) == GEIST_OK &&
         v->buffer_download(sn * sizeof(float), (uint8_t *) s_got, bs) == GEIST_OK;
    const double dze = geist_test_max_abs(vd, zd_got, zd_ref),
                 dce = geist_test_max_abs(cn, cs_got, cs_ref);
    const double dse = geist_test_max_abs(sn, s_got, s_ref);
    printf("  %-22s decode  seq=1    z %.2e  conv-state %.2e  delta-state %.2e\n",
           g->name,
           dze,
           dce,
           dse);
    ok = ok && dze < g->tol && dce < 1e-6 && dse < g->tol;

    struct geist_buffer *all[] = {bq, bz, bb, ba, bw, bA, bd, bn, bc, bs};
    for (size_t i = 0; i < sizeof all / sizeof all[0]; i++) {
        v->buffer_destroy(be, all[i]);
    }
    float *frees[] = {qkv,   z,  beta,  alpha, cw,     aw,     dt,     nw,
                      cs,    st, q_ref, z_ref, cs_ref, s_ref,  z_got,  cs_got,
                      s_got, qd, zd,    bd1,   ad1,    qd_ref, zd_ref, zd_got};
    for (size_t i = 0; i < sizeof frees / sizeof frees[0]; i++) {
        free(frees[i]);
    }
    if (!ok) {
        fprintf(stderr, "FAIL [%s]\n", g->name);
    }
    return ok;
}

static bool probe_says(struct geist_backend *be, size_t dk, size_t dv, size_t K) {
    const struct geist_fusion_query q = {.op             = GEIST_FUSED_DELTANET_MIX,
                                         .m              = 64,
                                         .dn_n_k_heads   = 2,
                                         .dn_n_v_heads   = 4,
                                         .dn_head_k      = dk,
                                         .dn_head_v      = dv,
                                         .dn_conv_kernel = K};
    return geist_backend_fused_tbl(be)->supported(be, &q);
}

/* An in-memory qwen35 hybrid (3 DeltaNet layers, 1 attention) with the
 * given DeltaNet head_k: loads and prefills, or is refused at load. */
static bool load_case(struct geist_backend *be, uint32_t head_k, bool want_load) {
    struct tf_vocab     v  = tf_make_vocab("\xc4\xa0", false);
    struct tf_buf       g  = mf_qwen35_gguf(&(struct mf_qwen35) {.layers     = 4,
                                                                 .interval   = 4,
                                                                 .d_model    = 64,
                                                                 .heads      = 4,
                                                                 .kv_heads   = 2,
                                                                 .head_dim   = 16,
                                                                 .rope_dims  = 8,
                                                                 .ffn        = 128,
                                                                 .dn_k_heads = 2,
                                                                 .dn_v_heads = 4,
                                                                 .dn_head_k  = head_k,
                                                                 .dn_head_v  = 16,
                                                                 .dn_conv    = 4,
                                                                 .seed       = 7,
                                                                 .tok        = &v});
    struct geist_model *m  = nullptr;
    const bool          ld = geist_model_load_from_memory(g.b, g.n, be, &m) == GEIST_OK;
    bool                ok = ld == want_load;
    bool                pf = false;
    if (ld) {
        struct geist_session           *s      = nullptr;
        const struct geist_session_opts o      = {.top_p = 1.0f, .m_max = 4};
        const int32_t                   ids[6] = {1, 2, 3, 4, 5, 6};
        size_t                          n      = 0;
        pf = geist_session_create(m, be, &o, &s) == GEIST_OK &&
             geist_session_prefill_tokens(s, 6, ids) == GEIST_OK &&
             geist_session_peek_logits(&n, s) != nullptr;
        ok = ok && pf;
        geist_session_destroy(s);
        geist_model_destroy(m);
    } else {
        const char *err = geist_last_create_error();
        ok              = ok && err != nullptr && strstr(err, "head_k=") != nullptr;
        if (!ok) {
            fprintf(stderr, "  refusal message: %s\n", err != nullptr ? err : "(none)");
        }
    }
    printf("  load qwen35 head_k=%-4u %s\n",
           head_k,
           !ld  ? "refused at load"
           : pf ? "loads, prefills"
                : "loads, prefill fails");
    if (!ok) {
        fprintf(stderr, "FAIL [load head_k=%u]\n", head_k);
    }
    free(g.b);
    tf_free_vocab(&v);
    return ok;
}

int main(void) {
    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create("vulkan", nullptr, nullptr, &be);
    if (s == GEIST_E_NOT_FOUND || s == GEIST_E_UNSUPPORTED) {
        GEIST_SKIP("Vulkan backend unavailable");
    }
    if (s != GEIST_OK || be == nullptr) {
        fprintf(stderr, "FAIL: Vulkan create: %s\n", geist_last_create_error());
        return GEIST_TEST_FAIL;
    }
    if (geist_backend_fused_tbl(be)->deltanet_mix == nullptr) {
        fprintf(stderr, "FAIL: Vulkan DeltaNet fusion missing\n");
        return GEIST_TEST_FAIL;
    }
    static const struct geom geoms[] = {
            {"tiny 1:2 odd dv", 1, 2, 7, 12, 4, 4, 2e-5},
            {"0.8B-like 1:1 128", 4, 4, 128, 128, 4, 13, 1e-4},
            {"27B-like 1:3 128", 2, 6, 128, 128, 4, 19, 1e-4},
            {"K=5 window", 1, 3, 16, 16, 5, 7, 2e-5},
            {"K=2 window", 2, 2, 32, 24, 2, 5, 2e-5},
    };
    bool ok = true;
    for (size_t i = 0; i < sizeof geoms / sizeof geoms[0]; i++) {
        ok = run_geom(be, &geoms[i]) && ok;
    }
    ok = geist_expect(probe_says(be, 128, 128, 4), "probe: d_k 128, d_v 128, conv 4") == 0 && ok;
    ok = geist_expect(!probe_says(be, 320, 128, 4), "probe: d_k 320 is past the shader") == 0 && ok;
    ok = geist_expect(!probe_says(be, 128, 160, 4), "probe: d_v 160 is past the shader") == 0 && ok;
    ok = geist_expect(!probe_says(be, 128, 128, 9), "probe: conv 9 is past the shader") == 0 && ok;
    ok = load_case(be, 16, true) && ok;
    ok = load_case(be, 320, false) && ok;
    geist_backend_destroy(be);
    if (ok) {
        printf("PASS: Vulkan DeltaNet prefill and stateful decode parity\n");
    }
    return ok ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}

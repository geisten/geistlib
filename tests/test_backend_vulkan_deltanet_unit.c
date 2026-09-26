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
 * SKIPs (exit 77) when no Vulkan runtime/device is present. */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct geom {
    const char *name;
    size_t      nkh, nvh, dk, dv, K, seq;
    double      tol;
};

static float silu_ref(float x) {
    const float e = expf(-fabsf(x));
    return x >= 0.0f ? x / (1.0f + e) : x * e / (1.0f + e);
}

/* Scalar oracle: advances qkv (conv + norm in place), z (output), cs, state. */
static void mix_ref(const struct geom *g,
                    size_t             seq,
                    float             *qkv,
                    float             *z,
                    const float       *beta,
                    const float       *alpha,
                    const float       *cw,
                    const float       *aw,
                    const float       *dt,
                    const float       *nw,
                    float             *cs,
                    float             *state) {
    const float  eps  = 1e-6f;
    const size_t keyd = g->nkh * g->dk, vd = g->nvh * g->dv, cd = 2 * keyd + vd;
    float       *y   = malloc(cd * sizeof(float));
    float       *out = malloc(g->dv * sizeof(float));
    for (size_t t = 0; t < seq; t++) {
        for (size_t c = 0; c < cd; c++) {
            float acc = 0.0f;
            for (size_t r = 0; r < g->K; r++) {
                const float x = r + 1 < g->K ? cs[r * cd + c] : qkv[t * cd + c];
                acc += cw[c * g->K + r] * x;
            }
            y[c] = silu_ref(acc);
        }
        memmove(cs, cs + cd, (g->K - 2) * cd * sizeof(float));
        memcpy(cs + (g->K - 2) * cd, qkv + t * cd, cd * sizeof(float));
        memcpy(qkv + t * cd, y, cd * sizeof(float));

        for (size_t h = 0; h < g->nkh; h++) {
            double qss = 0.0, kss = 0.0;
            for (size_t i = 0; i < g->dk; i++) {
                qss += (double) y[h * g->dk + i] * y[h * g->dk + i];
                kss += (double) y[keyd + h * g->dk + i] * y[keyd + h * g->dk + i];
            }
            const float qi = (float) (1.0 / sqrt(qss + eps)) / sqrtf((float) g->dk);
            const float ki = (float) (1.0 / sqrt(kss + eps));
            for (size_t i = 0; i < g->dk; i++) {
                qkv[t * cd + h * g->dk + i] *= qi;
                qkv[t * cd + keyd + h * g->dk + i] *= ki;
            }
        }
        for (size_t h = 0; h < g->nvh; h++) {
            const size_t hk    = h % g->nkh;
            const float  b     = 1.0f / (1.0f + expf(-beta[t * g->nvh + h]));
            const float  decay = expf(aw[h] * log1pf(expf(alpha[t * g->nvh + h] + dt[h])));
            for (size_t j = 0; j < g->dv; j++) {
                out[j] = 0.0f;
            }
            for (size_t j = 0; j < g->dv; j++) {
                float mem = 0.0f;
                for (size_t i = 0; i < g->dk; i++) {
                    float *s = state + (h * g->dk + i) * g->dv + j;
                    *s *= decay;
                    mem += *s * qkv[t * cd + keyd + hk * g->dk + i];
                }
                const float d = (qkv[t * cd + 2 * keyd + h * g->dv + j] - mem) * b;
                for (size_t i = 0; i < g->dk; i++) {
                    float *s = state + (h * g->dk + i) * g->dv + j;
                    *s += qkv[t * cd + keyd + hk * g->dk + i] * d;
                    out[j] += *s * qkv[t * cd + hk * g->dk + i];
                }
            }
            double ss = 0.0;
            for (size_t j = 0; j < g->dv; j++) {
                ss += (double) out[j] * out[j];
            }
            const float inv = (float) (1.0 / sqrt(ss / (double) g->dv + eps));
            for (size_t j = 0; j < g->dv; j++) {
                z[t * vd + h * g->dv + j] =
                        out[j] * inv * nw[j] * silu_ref(z[t * vd + h * g->dv + j]);
            }
        }
    }
    free(y);
    free(out);
}

static struct geist_tensor
tensor_nd(struct geist_buffer *b, int nd, int64_t d0, int64_t d1, int64_t d2) {
    struct geist_tensor t = {.buffer = b, .dtype = GEIST_DTYPE_F32, .layout = GEIST_LAYOUT_DENSE};
    t.ndim                = nd;
    t.shape[0]            = d0;
    t.shape[1]            = d1;
    t.shape[2]            = d2;
    if (nd == 1) {
        t.stride[0] = 1;
    } else if (nd == 2) {
        t.stride[0] = d1;
        t.stride[1] = 1;
    } else {
        t.stride[0] = d1 * d2;
        t.stride[1] = d2;
        t.stride[2] = 1;
    }
    return t;
}

static bool make_buf(struct geist_backend  *be,
                     struct geist_buffer  **out,
                     enum geist_buffer_role role,
                     const float           *src,
                     size_t                 n) {
    const struct geist_backend_vtbl *v = be->desc->vtbl;
    return v->buffer_create(be, n * sizeof(float), role, GEIST_MEMORY_AUTO, out) == GEIST_OK &&
           v->buffer_upload(*out, n * sizeof(float), (const uint8_t *) src) == GEIST_OK;
}

static double max_abs(const float *a, const float *b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double d = fabs((double) a[i] - (double) b[i]);
        if (d > m) {
            m = d;
        }
    }
    return m;
}

static float *fill(size_t n, float freq, float amp, float bias) {
    float *p = malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        p[i] = sinf((float) i * freq) * amp + bias;
    }
    return p;
}

static bool run_geom(struct geist_backend *be, const struct geom *g) {
    const struct geist_backend_fused *f    = geist_backend_fused_tbl(be);
    const struct geist_backend_vtbl  *v    = be->desc->vtbl;
    const size_t                      keyd = g->nkh * g->dk, vd = g->nvh * g->dv;
    const size_t                      cd = 2 * keyd + vd, cn = (g->K - 1) * cd;
    const size_t                      sn = g->nvh * g->dk * g->dv;

    float *qkv = fill(g->seq * cd, 0.17f, 0.4f, 0.0f), *z = fill(g->seq * vd, 0.11f, 0.3f, 0.0f);
    float *beta  = fill(g->seq * g->nvh, 0.13f, 0.6f, -0.1f);
    float *alpha = fill(g->seq * g->nvh, 0.07f, 0.5f, -0.2f);
    float *cw = fill(cd * g->K, 0.09f, 0.2f, 0.0f), *aw = malloc(g->nvh * sizeof(float));
    float *dt = malloc(g->nvh * sizeof(float)), *nw = fill(g->dv, 0.31f, 0.1f, 0.9f);
    float *cs = fill(cn, 0.05f, 0.1f, 0.0f), *st = fill(sn, 0.03f, 0.05f, 0.0f);
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
    mix_ref(g, g->seq, q_ref, z_ref, beta, alpha, cw, aw, dt, nw, cs_ref, s_ref);

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
    struct geist_tensor tq = tensor_nd(bq, 2, (int64_t) g->seq, (int64_t) cd, 0);
    struct geist_tensor tz = tensor_nd(bz, 2, (int64_t) g->seq, (int64_t) vd, 0);
    struct geist_tensor tb = tensor_nd(bb, 2, (int64_t) g->seq, (int64_t) g->nvh, 0);
    struct geist_tensor ta = tensor_nd(ba, 2, (int64_t) g->seq, (int64_t) g->nvh, 0);
    struct geist_tensor tw = tensor_nd(bw, 2, (int64_t) cd, (int64_t) g->K, 0);
    struct geist_tensor tA = tensor_nd(bA, 1, (int64_t) g->nvh, 0, 0);
    struct geist_tensor td = tensor_nd(bd, 1, (int64_t) g->nvh, 0, 0);
    struct geist_tensor tn = tensor_nd(bn, 1, (int64_t) g->dv, 0, 0);
    struct geist_tensor tc = tensor_nd(bc, 2, (int64_t) (g->K - 1), (int64_t) cd, 0);
    struct geist_tensor ts = tensor_nd(bs, 3, (int64_t) g->nvh, (int64_t) g->dk, (int64_t) g->dv);

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
    const double ze = max_abs(z_got, z_ref, g->seq * vd), ce = max_abs(cs_got, cs_ref, cn);
    const double se = max_abs(s_got, s_ref, sn);
    printf("  %-22s prefill seq=%-3zu  z %.2e  conv-state %.2e  delta-state %.2e\n",
           g->name,
           g->seq,
           ze,
           ce,
           se);
    ok = ok && ze < g->tol && ce < 1e-6 && se < g->tol;

    /* One decode row continuing from the advanced state. */
    float *qd = fill(cd, 0.19f, 0.35f, 0.0f), *zd = fill(vd, 0.23f, 0.25f, 0.0f);
    float *bd1 = malloc(g->nvh * sizeof(float)), *ad1 = malloc(g->nvh * sizeof(float));
    for (size_t i = 0; i < g->nvh; i++) {
        bd1[i] = 0.2f - (float) i * 0.3f;
        ad1[i] = -0.1f + (float) i * 0.2f;
    }
    float *qd_ref = malloc(cd * sizeof(float)), *zd_ref = malloc(vd * sizeof(float));
    memcpy(qd_ref, qd, cd * sizeof(float));
    memcpy(zd_ref, zd, vd * sizeof(float));
    mix_ref(g, 1, qd_ref, zd_ref, bd1, ad1, cw, aw, dt, nw, cs_ref, s_ref);
    ok       = ok && v->buffer_upload(bq, cd * sizeof(float), (const uint8_t *) qd) == GEIST_OK &&
               v->buffer_upload(bz, vd * sizeof(float), (const uint8_t *) zd) == GEIST_OK &&
               v->buffer_upload(bb, g->nvh * sizeof(float), (const uint8_t *) bd1) == GEIST_OK &&
               v->buffer_upload(ba, g->nvh * sizeof(float), (const uint8_t *) ad1) == GEIST_OK;
    tq       = tensor_nd(bq, 2, 1, (int64_t) cd, 0);
    tz       = tensor_nd(bz, 2, 1, (int64_t) vd, 0);
    tb       = tensor_nd(bb, 2, 1, (int64_t) g->nvh, 0);
    ta       = tensor_nd(ba, 2, 1, (int64_t) g->nvh, 0);
    args.seq = 1;
    ok       = ok && f->deltanet_mix(be, &args) == GEIST_OK;
    float *zd_got = malloc(vd * sizeof(float));
    ok = ok && v->buffer_download(vd * sizeof(float), (uint8_t *) zd_got, bz) == GEIST_OK &&
         v->buffer_download(cn * sizeof(float), (uint8_t *) cs_got, bc) == GEIST_OK &&
         v->buffer_download(sn * sizeof(float), (uint8_t *) s_got, bs) == GEIST_OK;
    const double dze = max_abs(zd_got, zd_ref, vd), dce = max_abs(cs_got, cs_ref, cn);
    const double dse = max_abs(s_got, s_ref, sn);
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
    geist_backend_destroy(be);
    if (ok) {
        printf("PASS: Vulkan DeltaNet prefill and stateful decode parity\n");
    }
    return ok ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}

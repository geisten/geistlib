/* Metal Gated-DeltaNet prefill/decode parity (#296). Exercises the fused
 * backend contract directly with non-zero recurrent state, then compares
 * mixer output plus both advanced states against a scalar reference. */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    SEQ = 4,
    NKH = 1,
    NH  = 2,
    DK  = 7,
    /* dv%4==0 keeps the chunked prefill path engaged (its wide kernels
     * emit 4-wide j blocks and the host falls back to the serial mixer
     * otherwise); 12 stays a non-power-of-two awkward size. */
    DV   = 12,
    K    = 4,
    KEYD = NKH * DK,
    VD   = NH * DV,
    CD   = 2 * KEYD + VD
};

static float silu_ref(float x) {
    const float e = expf(-fabsf(x));
    return x >= 0.0f ? x / (1.0f + e) : x * e / (1.0f + e);
}

static void mix_ref(size_t       seq,
                    float       *qkv,
                    float       *z,
                    const float *beta,
                    const float *alpha,
                    const float  cw[CD * K],
                    const float  aw[NH],
                    const float  dt[NH],
                    const float  nw[DV],
                    float        cs[(K - 1) * CD],
                    float        state[NH * DK * DV]) {
    const float eps = 1e-6f;
    for (size_t t = 0; t < seq; t++) {
        float y[CD];
        for (size_t c = 0; c < CD; c++) {
            float acc = 0.0f;
            for (size_t r = 0; r < K; r++) {
                const float x = r + 1 < K ? cs[r * CD + c] : qkv[t * CD + c];
                acc += cw[c * K + r] * x;
            }
            y[c] = silu_ref(acc);
        }
        memmove(cs, cs + CD, (K - 2) * CD * sizeof(float));
        memcpy(cs + (K - 2) * CD, qkv + t * CD, CD * sizeof(float));
        memcpy(qkv + t * CD, y, CD * sizeof(float));

        for (size_t h = 0; h < NKH; h++) {
            double qss = 0.0, kss = 0.0;
            for (size_t i = 0; i < DK; i++) {
                qss += (double) y[h * DK + i] * y[h * DK + i];
                kss += (double) y[KEYD + h * DK + i] * y[KEYD + h * DK + i];
            }
            const float qi = (float) (1.0 / sqrt(qss + eps)) / sqrtf((float) DK);
            const float ki = (float) (1.0 / sqrt(kss + eps));
            for (size_t i = 0; i < DK; i++) {
                qkv[t * CD + h * DK + i] *= qi;
                qkv[t * CD + KEYD + h * DK + i] *= ki;
            }
        }

        for (size_t h = 0; h < NH; h++) {
            const size_t hk      = h % NKH;
            const float  b       = 1.0f / (1.0f + expf(-beta[t * NH + h]));
            const float  decay   = expf(aw[h] * log1pf(expf(alpha[t * NH + h] + dt[h])));
            float        out[DV] = {0};
            for (size_t j = 0; j < DV; j++) {
                float mem = 0.0f;
                for (size_t i = 0; i < DK; i++) {
                    float *s = state + (h * DK + i) * DV + j;
                    *s *= decay;
                    mem += *s * qkv[t * CD + KEYD + hk * DK + i];
                }
                const float d = (qkv[t * CD + 2 * KEYD + h * DV + j] - mem) * b;
                for (size_t i = 0; i < DK; i++) {
                    float *s = state + (h * DK + i) * DV + j;
                    *s += qkv[t * CD + KEYD + hk * DK + i] * d;
                    out[j] += *s * qkv[t * CD + hk * DK + i];
                }
            }
            double ss = 0.0;
            for (size_t j = 0; j < DV; j++)
                ss += (double) out[j] * out[j];
            const float inv = (float) (1.0 / sqrt(ss / DV + eps));
            for (size_t j = 0; j < DV; j++)
                z[t * VD + h * DV + j] = out[j] * inv * nw[j] * silu_ref(z[t * VD + h * DV + j]);
        }
    }
}

static struct geist_tensor matrix(struct geist_buffer *b, size_t rows, size_t cols) {
    return (struct geist_tensor) {.buffer = b,
                                  .dtype  = GEIST_DTYPE_F32,
                                  .layout = GEIST_LAYOUT_DENSE,
                                  .ndim   = 2,
                                  .shape  = {(int64_t) rows, (int64_t) cols},
                                  .stride = {(int64_t) cols, 1}};
}

static struct geist_tensor vector(struct geist_buffer *b, size_t n) {
    return (struct geist_tensor) {.buffer = b,
                                  .dtype  = GEIST_DTYPE_F32,
                                  .layout = GEIST_LAYOUT_DENSE,
                                  .ndim   = 1,
                                  .shape  = {(int64_t) n},
                                  .stride = {1}};
}

static int upload(struct geist_backend *be, struct geist_buffer **out, const float *src, size_t n) {
    const struct geist_backend_vtbl *v = be->desc->vtbl;
    return v->buffer_create(be, n * sizeof(float), GEIST_BUFFER_SCRATCH, GEIST_MEMORY_AUTO, out) ==
                   GEIST_OK &&
           v->buffer_upload(*out, n * sizeof(float), (const uint8_t *) src) == GEIST_OK;
}

static double max_abs(const float *a, const float *b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double d = fabs((double) a[i] - b[i]);
        if (d > m)
            m = d;
    }
    return m;
}

int main(void) {
    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create("metal", nullptr, nullptr, &be);
    if (s == GEIST_E_NOT_FOUND || s == GEIST_E_UNSUPPORTED)
        GEIST_SKIP("Metal backend unavailable");
    if (s != GEIST_OK || be == nullptr) {
        fprintf(stderr, "FAIL: Metal create: %s\n", geist_last_create_error());
        return GEIST_TEST_FAIL;
    }
    const struct geist_backend_fused *f = geist_backend_fused_tbl(be);
    if (f->deltanet_mix == nullptr) {
        fprintf(stderr, "FAIL: Metal DeltaNet fusion missing\n");
        return GEIST_TEST_FAIL;
    }

    float qkv[SEQ * CD], z[SEQ * VD], beta[SEQ * NH], alpha[SEQ * NH];
    float cw[CD * K], aw[NH], dt[NH], nw[DV];
    float cs[(K - 1) * CD], state[NH * DK * DV];
    for (size_t i = 0; i < SEQ * CD; i++)
        qkv[i] = sinf((float) i * 0.17f) * 0.4f;
    for (size_t i = 0; i < SEQ * VD; i++)
        z[i] = cosf((float) i * 0.11f) * 0.3f;
    for (size_t i = 0; i < SEQ * NH; i++) {
        beta[i]  = (float) i * 0.13f - 0.4f;
        alpha[i] = (float) i * 0.07f - 0.2f;
    }
    for (size_t i = 0; i < CD * K; i++)
        cw[i] = sinf((float) i * 0.09f) * 0.2f;
    for (size_t i = 0; i < NH; i++) {
        aw[i] = -0.5f - (float) i * 0.2f;
        dt[i] = 0.1f + (float) i * 0.05f;
    }
    for (size_t i = 0; i < DV; i++)
        nw[i] = 0.8f + (float) i * 0.03f;
    for (size_t i = 0; i < (K - 1) * CD; i++)
        cs[i] = cosf((float) i * 0.05f) * 0.1f;
    for (size_t i = 0; i < NH * DK * DV; i++)
        state[i] = sinf((float) i * 0.03f) * 0.05f;

    float q_ref[SEQ * CD], z_ref[SEQ * VD], cs_ref[(K - 1) * CD], s_ref[NH * DK * DV];
    memcpy(q_ref, qkv, sizeof qkv);
    memcpy(z_ref, z, sizeof z);
    memcpy(cs_ref, cs, sizeof cs);
    memcpy(s_ref, state, sizeof state);
    mix_ref(SEQ, q_ref, z_ref, beta, alpha, cw, aw, dt, nw, cs_ref, s_ref);

    struct geist_buffer *buf[10] = {0};
    const float         *src[10] = {qkv, z, beta, alpha, cw, aw, dt, nw, cs, state};
    const size_t         n[10]   = {
            SEQ * CD, SEQ * VD, SEQ * NH, SEQ * NH, CD * K, NH, NH, DV, (K - 1) * CD, NH * DK * DV};
    int ok = 1;
    for (size_t i = 0; i < 10; i++)
        ok &= upload(be, &buf[i], src[i], n[i]);
    struct geist_tensor            tq = matrix(buf[0], SEQ, CD), tz = matrix(buf[1], SEQ, VD);
    struct geist_tensor            tb = matrix(buf[2], SEQ, NH), ta = matrix(buf[3], SEQ, NH);
    struct geist_tensor            tcw = matrix(buf[4], CD, K), taw = vector(buf[5], NH);
    struct geist_tensor            tdt = vector(buf[6], NH), tnw = vector(buf[7], DV);
    struct geist_tensor            tcs        = matrix(buf[8], K - 1, CD);
    struct geist_tensor            tst        = {.buffer = buf[9],
                                                 .dtype  = GEIST_DTYPE_F32,
                                                 .layout = GEIST_LAYOUT_DENSE,
                                                 .ndim   = 3,
                                                 .shape  = {NH, DK, DV},
                                                 .stride = {DK * DV, DV, 1}};
    struct geist_deltanet_mix_args args       = {.qkv         = &tq,
                                                 .z           = &tz,
                                                 .beta        = &tb,
                                                 .alpha       = &ta,
                                                 .conv_w      = &tcw,
                                                 .ssm_a       = &taw,
                                                 .dt_bias     = &tdt,
                                                 .norm_w      = &tnw,
                                                 .conv_state  = &tcs,
                                                 .delta_state = &tst,
                                                 .seq         = SEQ,
                                                 .n_k_heads   = NKH,
                                                 .n_v_heads   = NH,
                                                 .head_k      = DK,
                                                 .head_v      = DV,
                                                 .conv_kernel = K,
                                                 .eps         = 1e-6f};
    const enum geist_status        mix_status = ok ? f->deltanet_mix(be, &args) : GEIST_E_BACKEND;
    if (!ok || mix_status != GEIST_OK) {
        fprintf(stderr,
                "FAIL: Metal DeltaNet dispatch (%d): %s\n",
                (int) mix_status,
                geist_backend_errmsg(be));
        return GEIST_TEST_FAIL;
    }
    float q_got[SEQ * CD], z_got[SEQ * VD], cs_got[(K - 1) * CD], s_got[NH * DK * DV];
    const struct geist_backend_vtbl *v = be->desc->vtbl;
    ok &= v->buffer_download(sizeof q_got, (uint8_t *) q_got, buf[0]) == GEIST_OK;
    ok &= v->buffer_download(sizeof z_got, (uint8_t *) z_got, buf[1]) == GEIST_OK;
    ok &= v->buffer_download(sizeof cs_got, (uint8_t *) cs_got, buf[8]) == GEIST_OK;
    ok &= v->buffer_download(sizeof s_got, (uint8_t *) s_got, buf[9]) == GEIST_OK;
    const double qe = max_abs(q_got, q_ref, SEQ * CD);
    const double ze = max_abs(z_got, z_ref, SEQ * VD);
    const double ce = max_abs(cs_got, cs_ref, (K - 1) * CD);
    const double se = max_abs(s_got, s_ref, NH * DK * DV);
    printf("prefill max_abs: qkv %.2e (informational), z %.2e, conv-state %.2e, "
           "delta-state %.2e\n",
           qe,
           ze,
           ce,
           se);
    /* qkv is NOT asserted for prefill: the fusion contract only defines z
     * and the two state tensors as outputs. The serial kernel happens to
     * mutate qkv in place; the chunked path (like the CPU chunked fusion)
     * stages in scratch and leaves qkv pre-conv. */
    ok &= ze < 2e-5 && ce < 1e-7 && se < 2e-5;

    /* Continue from the prefill states with one decode row. This catches
     * accidental state reset, double-advance and the seq==1 dispatch path. */
    float qd[CD], zd[VD], bd[NH], ad[NH];
    for (size_t i = 0; i < CD; i++)
        qd[i] = cosf((float) i * 0.19f) * 0.35f;
    for (size_t i = 0; i < VD; i++)
        zd[i] = sinf((float) i * 0.23f) * 0.25f;
    for (size_t i = 0; i < NH; i++) {
        bd[i] = 0.2f - (float) i * 0.3f;
        ad[i] = -0.1f + (float) i * 0.2f;
    }
    float qd_ref[CD], zd_ref[VD];
    memcpy(qd_ref, qd, sizeof qd);
    memcpy(zd_ref, zd, sizeof zd);
    mix_ref(1, qd_ref, zd_ref, bd, ad, cw, aw, dt, nw, cs_ref, s_ref);
    ok &= v->buffer_upload(buf[0], sizeof qd, (const uint8_t *) qd) == GEIST_OK;
    ok &= v->buffer_upload(buf[1], sizeof zd, (const uint8_t *) zd) == GEIST_OK;
    ok &= v->buffer_upload(buf[2], sizeof bd, (const uint8_t *) bd) == GEIST_OK;
    ok &= v->buffer_upload(buf[3], sizeof ad, (const uint8_t *) ad) == GEIST_OK;
    tq         = matrix(buf[0], 1, CD);
    tz         = matrix(buf[1], 1, VD);
    tb         = matrix(buf[2], 1, NH);
    ta         = matrix(buf[3], 1, NH);
    args.qkv   = &tq;
    args.z     = &tz;
    args.beta  = &tb;
    args.alpha = &ta;
    args.seq   = 1;
    ok &= f->deltanet_mix(be, &args) == GEIST_OK;
    float qd_got[CD], zd_got[VD];
    ok &= v->buffer_download(sizeof qd_got, (uint8_t *) qd_got, buf[0]) == GEIST_OK;
    ok &= v->buffer_download(sizeof zd_got, (uint8_t *) zd_got, buf[1]) == GEIST_OK;
    ok &= v->buffer_download(sizeof cs_got, (uint8_t *) cs_got, buf[8]) == GEIST_OK;
    ok &= v->buffer_download(sizeof s_got, (uint8_t *) s_got, buf[9]) == GEIST_OK;
    const double dqe = max_abs(qd_got, qd_ref, CD);
    const double dze = max_abs(zd_got, zd_ref, VD);
    const double dce = max_abs(cs_got, cs_ref, (K - 1) * CD);
    const double dse = max_abs(s_got, s_ref, NH * DK * DV);
    printf("decode  max_abs: qkv %.2e, z %.2e, conv-state %.2e, delta-state %.2e\n",
           dqe,
           dze,
           dce,
           dse);
    ok &= dqe < 2e-5 && dze < 2e-5 && dce < 1e-7 && dse < 2e-5;
    for (size_t i = 0; i < 10; i++)
        if (buf[i] != nullptr)
            v->buffer_destroy(be, buf[i]);
    geist_backend_destroy(be);
    if (ok)
        printf("PASS: Metal DeltaNet prefill and stateful decode parity\n");
    return ok ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}

/* Metal Gated-DeltaNet prefill at real head sizes (#99): the simdgroup
 * matrix recipe (dnf_*) against a double-precision oracle and the float
 * per-token oracle, over a sequence of calls that continue one recurrent
 * state:
 *
 *   ragged last chunks (37, 65, 291, 130 tokens), an exact chunk (64),
 *   calls shorter than the conv window (2), decode steps in between (1),
 *   and a call longer than one dnf piece (600 > 512);
 *   a snapshot restore: the states saved before the 291-token call are
 *   uploaded again and the call re-runs (must reproduce z bit for bit);
 *   a truncate-and-continue: from the same snapshot the 291 tokens run as
 *   100 + 191, so every chunk boundary moves (must agree within tolerance).
 *
 * Shapes: 2 key / 6 value heads of 128 (Bonsai's 3:1 tiling at dk = dv =
 * 128) and 1 key / 2 value heads with dk 64, dv 96 (the generic tiling).
 * z is checked after every call, both states after the last; errors are
 * max |metal - f64| over |f64| + 1, next to the float oracle's own error
 * against f64 for scale. */
#include "deltanet_ref.h"
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { CONV_K = 4, N_CALLS = 9, SNAP_CALL = 4 };

static const size_t calls[N_CALLS] = {37, 64, 1, 65, 291, 2, 600, 1, 130};

struct cfg {
    const char *name;
    size_t      nkh, nvh, dk, dv;
};

struct dev {
    struct geist_buffer *b[10];
};

static struct geist_tensor mat(struct geist_buffer *b, size_t rows, size_t cols) {
    return (struct geist_tensor) {.buffer = b,
                                  .dtype  = GEIST_DTYPE_F32,
                                  .layout = GEIST_LAYOUT_DENSE,
                                  .ndim   = 2,
                                  .shape  = {(int64_t) rows, (int64_t) cols},
                                  .stride = {(int64_t) cols, 1}};
}

static struct geist_tensor vec(struct geist_buffer *b, size_t n) {
    return (struct geist_tensor) {.buffer = b,
                                  .dtype  = GEIST_DTYPE_F32,
                                  .layout = GEIST_LAYOUT_DENSE,
                                  .ndim   = 1,
                                  .shape  = {(int64_t) n},
                                  .stride = {1}};
}

static float urand(uint32_t *s, float lo, float hi) {
    *s = *s * 1664525u + 1013904223u;
    return lo + (hi - lo) * ((float) (*s >> 8) / 16777216.0f);
}

/* max |got - ref| / (|ref| + 1) */
static double err_rel1(size_t n, const float *got, const double *ref) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double e = fabs((double) got[i] - ref[i]) / (fabs(ref[i]) + 1.0);
        if (!(e <= m)) {
            m = e; /* also catches NaN */
        }
    }
    return m;
}

/* One deltanet_mix call on the device: seq rows from token offset `at` of
 * the full input arrays. z_got receives the seq output rows. */
static bool run_call(struct geist_backend *be,
                     const struct cfg     *c,
                     const struct dev     *d,
                     size_t                at,
                     size_t                seq,
                     const float          *qkv,
                     const float          *z,
                     const float          *beta,
                     const float          *alpha,
                     float                *z_got) {
    const struct geist_backend_vtbl  *v    = be->desc->vtbl;
    const struct geist_backend_fused *f    = geist_backend_fused_tbl(be);
    const size_t                      keyd = c->nkh * c->dk, vd = c->nvh * c->dv;
    const size_t                      cd = 2 * keyd + vd;
    bool ok = v->buffer_upload(d->b[0],
                               seq * cd * sizeof(float),
                               (const uint8_t *) (qkv + at * cd)) == GEIST_OK;
    ok &= v->buffer_upload(d->b[1], seq * vd * sizeof(float), (const uint8_t *) (z + at * vd)) ==
          GEIST_OK;
    ok &= v->buffer_upload(d->b[2],
                           seq * c->nvh * sizeof(float),
                           (const uint8_t *) (beta + at * c->nvh)) == GEIST_OK;
    ok &= v->buffer_upload(d->b[3],
                           seq * c->nvh * sizeof(float),
                           (const uint8_t *) (alpha + at * c->nvh)) == GEIST_OK;
    struct geist_tensor tq = mat(d->b[0], seq, cd), tz = mat(d->b[1], seq, vd);
    struct geist_tensor tb = mat(d->b[2], seq, c->nvh), ta = mat(d->b[3], seq, c->nvh);
    struct geist_tensor tcw = mat(d->b[4], cd, CONV_K), taw = vec(d->b[5], c->nvh);
    struct geist_tensor tdt = vec(d->b[6], c->nvh), tnw = vec(d->b[7], c->dv);
    struct geist_tensor tcs = mat(d->b[8], CONV_K - 1, cd);
    struct geist_tensor tst = {.buffer = d->b[9],
                               .dtype  = GEIST_DTYPE_F32,
                               .layout = GEIST_LAYOUT_DENSE,
                               .ndim   = 3,
                               .shape  = {(int64_t) c->nvh, (int64_t) c->dk, (int64_t) c->dv},
                               .stride = {(int64_t) (c->dk * c->dv), (int64_t) c->dv, 1}};
    const struct geist_deltanet_mix_args args = {.qkv         = &tq,
                                                 .z           = &tz,
                                                 .beta        = &tb,
                                                 .alpha       = &ta,
                                                 .conv_w      = &tcw,
                                                 .ssm_a       = &taw,
                                                 .dt_bias     = &tdt,
                                                 .norm_w      = &tnw,
                                                 .conv_state  = &tcs,
                                                 .delta_state = &tst,
                                                 .seq         = seq,
                                                 .n_k_heads   = c->nkh,
                                                 .n_v_heads   = c->nvh,
                                                 .head_k      = c->dk,
                                                 .head_v      = c->dv,
                                                 .conv_kernel = CONV_K,
                                                 .eps         = 1e-6f};
    if (ok && f->deltanet_mix(be, &args) != GEIST_OK) {
        fprintf(stderr, "deltanet_mix(seq %zu): %s\n", seq, geist_backend_errmsg(be));
        ok = false;
    }
    ok = ok && v->buffer_download(seq * vd * sizeof(float), (uint8_t *) z_got, d->b[1]) == GEIST_OK;
    return ok;
}

static bool load_states(struct geist_backend *be,
                        const struct dev     *d,
                        size_t                ncs,
                        const float          *cs,
                        size_t                ns,
                        const float          *s) {
    const struct geist_backend_vtbl *v = be->desc->vtbl;
    return v->buffer_upload(d->b[8], ncs * sizeof(float), (const uint8_t *) cs) == GEIST_OK &&
           v->buffer_upload(d->b[9], ns * sizeof(float), (const uint8_t *) s) == GEIST_OK;
}

static bool save_states(
        struct geist_backend *be, const struct dev *d, size_t ncs, float *cs, size_t ns, float *s) {
    const struct geist_backend_vtbl *v = be->desc->vtbl;
    return v->buffer_download(ncs * sizeof(float), (uint8_t *) cs, d->b[8]) == GEIST_OK &&
           v->buffer_download(ns * sizeof(float), (uint8_t *) s, d->b[9]) == GEIST_OK;
}

static int run_cfg(struct geist_backend *be, const struct cfg *c) {
    const struct geist_backend_vtbl *v    = be->desc->vtbl;
    const size_t                     keyd = c->nkh * c->dk, vd = c->nvh * c->dv;
    const size_t                     cd = 2 * keyd + vd, ncs = (CONV_K - 1) * cd;
    const size_t                     ns    = c->nvh * c->dk * c->dv;
    size_t                           n_tok = 0, max_seq = 0;
    for (size_t i = 0; i < N_CALLS; i++) {
        n_tok += calls[i];
        max_seq = calls[i] > max_seq ? calls[i] : max_seq;
    }
    uint32_t rng   = 0x2545f491u;
    float   *qkv   = xmalloc(n_tok * cd * sizeof(float));
    float   *z     = xmalloc(n_tok * vd * sizeof(float));
    float   *beta  = xmalloc(n_tok * c->nvh * sizeof(float));
    float   *alpha = xmalloc(n_tok * c->nvh * sizeof(float));
    float   *cw    = xmalloc(cd * CONV_K * sizeof(float));
    float   *aw    = xmalloc(c->nvh * sizeof(float));
    float   *dt    = xmalloc(c->nvh * sizeof(float));
    float   *nw    = xmalloc(c->dv * sizeof(float));
    float   *cs    = xmalloc(ncs * sizeof(float));
    float   *st    = xmalloc(ns * sizeof(float));
    for (size_t i = 0; i < n_tok * cd; i++)
        qkv[i] = urand(&rng, -1.0f, 1.0f);
    for (size_t i = 0; i < n_tok * vd; i++)
        z[i] = urand(&rng, -2.0f, 2.0f);
    for (size_t i = 0; i < n_tok * c->nvh; i++) {
        beta[i]  = urand(&rng, -2.0f, 2.0f);
        alpha[i] = urand(&rng, -3.0f, 1.0f);
    }
    for (size_t i = 0; i < cd * CONV_K; i++)
        cw[i] = urand(&rng, -0.6f, 0.6f);
    for (size_t i = 0; i < c->nvh; i++) {
        aw[i] = urand(&rng, -4.0f, -0.5f);
        dt[i] = urand(&rng, -1.0f, 1.0f);
    }
    for (size_t i = 0; i < c->dv; i++)
        nw[i] = urand(&rng, 0.5f, 1.5f);
    for (size_t i = 0; i < ncs; i++)
        cs[i] = urand(&rng, -1.0f, 1.0f);
    for (size_t i = 0; i < ns; i++)
        st[i] = urand(&rng, -0.3f, 0.3f);

    /* f64 oracle state, float oracle state (+ its scratch copies of qkv/z) */
    double *cs64 = xmalloc(ncs * sizeof(double)), *st64 = xmalloc(ns * sizeof(double));
    float  *csf = xmalloc(ncs * sizeof(float)), *stf = xmalloc(ns * sizeof(float));
    for (size_t i = 0; i < ncs; i++)
        cs64[i] = cs[i];
    for (size_t i = 0; i < ns; i++)
        st64[i] = st[i];
    memcpy(csf, cs, ncs * sizeof(float));
    memcpy(stf, st, ns * sizeof(float));
    double *z64   = xmalloc(max_seq * vd * sizeof(double));
    float  *zgot  = xmalloc(max_seq * vd * sizeof(float));
    float  *zsnap = xmalloc(max_seq * vd * sizeof(float));
    float  *qf    = xmalloc(max_seq * cd * sizeof(float));
    float  *zf    = xmalloc(max_seq * vd * sizeof(float));
    float  *cs_s = xmalloc(ncs * sizeof(float)), *st_s = xmalloc(ns * sizeof(float));
    float  *cs_g = xmalloc(ncs * sizeof(float)), *st_g = xmalloc(ns * sizeof(float));

    struct dev   d        = {0};
    const size_t n[10]    = {max_seq * cd,
                             max_seq * vd,
                             max_seq * c->nvh,
                             max_seq * c->nvh,
                             cd * CONV_K,
                             c->nvh,
                             c->nvh,
                             c->dv,
                             ncs,
                             ns};
    const float *init[10] = {nullptr, nullptr, nullptr, nullptr, cw, aw, dt, nw, cs, st};
    bool         ok       = true;
    for (size_t i = 0; i < 10 && ok; i++) {
        ok = v->buffer_create(
                     be, n[i] * sizeof(float), GEIST_BUFFER_SCRATCH, GEIST_MEMORY_AUTO, &d.b[i]) ==
             GEIST_OK;
        if (ok && init[i] != nullptr) {
            ok = v->buffer_upload(d.b[i], n[i] * sizeof(float), (const uint8_t *) init[i]) ==
                 GEIST_OK;
        }
    }

    /* Tolerance: the chunked recipe reorders the sums of the per-token
     * recurrence; both F32 paths stay within a few 1e-6 of f64 here (see
     * the printed float-oracle error), the bound leaves 10x margin. */
    const double tol   = 1e-4;
    double       worst = 0.0, worst_f = 0.0;
    size_t       at = 0;
    for (size_t i = 0; i < N_CALLS && ok; i++) {
        const size_t s = calls[i];
        if (i == SNAP_CALL) {
            ok &= save_states(be, &d, ncs, cs_s, ns, st_s);
        }
        ok &= run_call(be, c, &d, at, s, qkv, z, beta, alpha, zgot);
        deltanet_mix_ref_f64(s,
                             c->nkh,
                             c->nvh,
                             c->dk,
                             c->dv,
                             CONV_K,
                             1e-6,
                             qkv + at * cd,
                             z + at * vd,
                             beta + at * c->nvh,
                             alpha + at * c->nvh,
                             cw,
                             aw,
                             dt,
                             nw,
                             cs64,
                             st64,
                             z64);
        memcpy(qf, qkv + at * cd, s * cd * sizeof(float));
        memcpy(zf, z + at * vd, s * vd * sizeof(float));
        deltanet_mix_ref(s,
                         c->nkh,
                         c->nvh,
                         c->dk,
                         c->dv,
                         CONV_K,
                         1e-6f,
                         beta + at * c->nvh,
                         alpha + at * c->nvh,
                         cw,
                         aw,
                         dt,
                         nw,
                         qf,
                         zf,
                         csf,
                         stf);
        const double e = err_rel1(s * vd, zgot, z64), ef = err_rel1(s * vd, zf, z64);
        printf("%s call %zu seq %3zu: z err %.2e (float oracle %.2e)\n", c->name, i, s, e, ef);
        worst   = e > worst || e != e ? e : worst;
        worst_f = ef > worst_f ? ef : worst_f;
        ok &= e <= tol;
        if (i == SNAP_CALL && ok) {
            /* restore: same call again from the saved states */
            memcpy(zsnap, zgot, s * vd * sizeof(float));
            ok &= save_states(be, &d, ncs, cs_g, ns, st_g);
            ok &= load_states(be, &d, ncs, cs_s, ns, st_s);
            ok &= run_call(be, c, &d, at, s, qkv, z, beta, alpha, zgot);
            const bool same = ok && memcmp(zgot, zsnap, s * vd * sizeof(float)) == 0;
            printf("%s restore + rerun: z %s\n", c->name, same ? "bit-identical" : "DIFFERS");
            ok &= same;
            /* truncate-and-continue: same tokens as 100 + 191 */
            ok &= load_states(be, &d, ncs, cs_s, ns, st_s);
            ok &= run_call(be, c, &d, at, 100, qkv, z, beta, alpha, zgot);
            ok &= run_call(be, c, &d, at + 100, s - 100, qkv, z, beta, alpha, zgot + 100 * vd);
            const double es = err_rel1(s * vd, zgot, z64);
            printf("%s restore + 100/191 split: z err %.2e\n", c->name, es);
            ok &= es <= tol;
            /* both orders leave the state the single call left */
            float *cs_t = xmalloc(ncs * sizeof(float)), *st_t = xmalloc(ns * sizeof(float));
            ok &= save_states(be, &d, ncs, cs_t, ns, st_t);
            const double dcs = geist_test_max_abs(ncs, cs_t, cs_g);
            const double dst = geist_test_max_abs(ns, st_t, st_g);
            printf("%s split vs single call: conv-state %.2e, delta-state %.2e\n",
                   c->name,
                   dcs,
                   dst);
            ok &= dcs == 0.0 && dst <= tol;
            free(cs_t);
            free(st_t);
        }
        at += s;
    }
    if (ok) {
        ok &= save_states(be, &d, ncs, cs_g, ns, st_g);
        double ecs = 0.0, est = 0.0;
        for (size_t i = 0; i < ncs; i++) {
            const double e = fabs((double) cs_g[i] - cs64[i]);
            ecs            = e > ecs ? e : ecs;
        }
        for (size_t i = 0; i < ns; i++) {
            const double e = fabs((double) st_g[i] - st64[i]) / (fabs(st64[i]) + 1.0);
            est            = e > est || e != e ? e : est;
        }
        printf("%s final: conv-state err %.2e, delta-state err %.2e; worst z %.2e (float "
               "oracle %.2e)\n",
               c->name,
               ecs,
               est,
               worst,
               worst_f);
        ok &= ecs == 0.0 && est <= tol;
    }
    for (size_t i = 0; i < 10; i++) {
        if (d.b[i] != nullptr) {
            v->buffer_destroy(be, d.b[i]);
        }
    }
    void *bufs[] = {qkv, z,   beta, alpha, cw,    aw, dt, nw,   cs,   st,   cs64, st64,
                    csf, stf, z64,  zgot,  zsnap, qf, zf, cs_s, st_s, cs_g, st_g};
    for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; i++) {
        free(bufs[i]);
    }
    return ok ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
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
    static const struct cfg cfgs[] = {
            {"dk128/dv128 2:6", 2, 6, 128, 128},
            {"dk64/dv96 1:2", 1, 2, 64, 96},
    };
    int rc = GEIST_TEST_PASS;
    for (size_t i = 0; i < sizeof cfgs / sizeof cfgs[0]; i++) {
        if (run_cfg(be, &cfgs[i]) != GEIST_TEST_PASS) {
            fprintf(stderr, "FAIL: %s\n", cfgs[i].name);
            rc = GEIST_TEST_FAIL;
        }
    }
    geist_backend_destroy(be);
    if (rc == GEIST_TEST_PASS)
        printf("PASS: Metal chunked DeltaNet prefill parity at real head sizes\n");
    return rc;
}

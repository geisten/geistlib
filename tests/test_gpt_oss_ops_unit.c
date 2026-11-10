/*
 * test_gpt_oss_ops_unit - GPT-OSS's two new kernels against double
 * references (#243). No model.
 *
 * attention_sinks_items (backends/common): random q / k / v and sinks, F32
 * and F16 caches, GQA, prefill and decode shapes, an unbounded and a sliding
 * window, n_kv past one ATTN_F32_BLOCK so the running max is rescaled, and
 * rows split over several calls; within 2e-6 of the logits' scale of the
 * double softmax with the sink as one more logit. A sink of -1e30 gives
 * attention_mqa_causal_kv's bytes' worth (1e-6), a sink far above every
 * score an output near zero.
 *
 * Every backend's fused->attention_sinks (Metal) against the same
 * reference: F32 and F16 caches, prefill, a window, a decode at 1099 with a
 * 128-key window, head_dim 48; within 1e-5 of scale.
 *
 * rope_compute_yarn_at: the cos/sin table against ggml's rope_yarn in
 * double (corr dims, ramp, mscale) at positions 0..4096 for head_dim 64,
 * base 150000, factor 32, original context 4096 (gpt-oss-20b's), within
 * 2e-6 relative to mscale plus the float angle's own error at the position.
 */
#include "test_helpers.h"

#include "gemma4_kernels.h"

#include <geist.h>
#include <geist_backend.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;

static void expect(bool ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        fails++;
    }
}

static uint64_t g_seed = 0x9e3779b97f4a7c15ull;

static float rnd(void) {
    g_seed ^= g_seed << 13;
    g_seed ^= g_seed >> 7;
    g_seed ^= g_seed << 17;
    return (float) ((double) (g_seed >> 11) / (double) (1ull << 53) * 2.0 - 1.0);
}

static uint16_t to_half(float f) {
    const _Float16 h = (_Float16) f;
    uint16_t       u;
    memcpy(&u, &h, sizeof u);
    return u;
}

static float from_half(uint16_t u) {
    _Float16 h;
    memcpy(&h, &u, sizeof h);
    return (float) h;
}

struct shape {
    size_t n_q, n_kv, q_offset, n_q_heads, n_kv_heads, head_dim, window;
};

/* out[t, h] by the definition, in double. */
static void ref_sinks(const struct shape *sh,
                      const float        *q,
                      const float        *k,
                      const float        *v,
                      const float        *sinks,
                      double             *out) {
    const size_t group = sh->n_q_heads / sh->n_kv_heads, hd = sh->head_dim;
    for (size_t t = 0; t < sh->n_q; t++) {
        const size_t p    = sh->q_offset + t;
        const size_t s_lo = sh->window > 0 && p + 1 > sh->window ? p + 1 - sh->window : 0;
        for (size_t h = 0; h < sh->n_q_heads; h++) {
            const size_t kvh = h / group;
            const float *qv  = q + (t * sh->n_q_heads + h) * hd;
            double       mx  = sinks[h];
            double      *sc  = malloc((p + 1) * sizeof *sc);
            for (size_t s = s_lo; s <= p; s++) {
                double dot = 0.0;
                for (size_t i = 0; i < hd; i++)
                    dot += (double) qv[i] * k[(s * sh->n_kv_heads + kvh) * hd + i];
                sc[s] = dot;
                mx    = fmax(mx, dot);
            }
            double den = exp(sinks[h] - mx);
            for (size_t s = s_lo; s <= p; s++)
                den += exp(sc[s] - mx);
            for (size_t i = 0; i < hd; i++) {
                double acc = 0.0;
                for (size_t s = s_lo; s <= p; s++)
                    acc += exp(sc[s] - mx) * v[(s * sh->n_kv_heads + kvh) * hd + i];
                out[(t * sh->n_q_heads + h) * hd + i] = acc / den;
            }
            free(sc);
        }
    }
}

static void check_sinks(const char *what, struct shape sh, bool f16, float sink_scale) {
    const size_t nq = sh.n_q * sh.n_q_heads * sh.head_dim;
    const size_t nk = sh.n_kv * sh.n_kv_heads * sh.head_dim;
    float    *q = malloc(nq * sizeof *q), *k = malloc(nk * sizeof *k), *v = malloc(nk * sizeof *v);
    float    *out = malloc(nq * sizeof *out), sinks[64];
    uint16_t *kh = malloc(nk * sizeof *kh), *vh = malloc(nk * sizeof *vh);
    double   *ref = malloc(nq * sizeof *ref);
    for (size_t i = 0; i < nq; i++)
        q[i] = rnd() * 0.5f;
    for (size_t i = 0; i < nk; i++) {
        k[i] = rnd();
        v[i] = rnd();
        if (f16) { /* the reference sees the cache's own values */
            kh[i] = to_half(k[i]);
            vh[i] = to_half(v[i]);
            k[i]  = from_half(kh[i]);
            v[i]  = from_half(vh[i]);
        }
    }
    for (size_t h = 0; h < sh.n_q_heads; h++)
        sinks[h] = rnd() * sink_scale;
    ref_sinks(&sh, q, k, v, sinks, ref);
    /* The rows in three uneven calls, as threads would split them. */
    const size_t n      = sh.n_q * sh.n_q_heads;
    const size_t cut[4] = {0, n / 3, n / 3 + (n > 2 ? 1 : 0), n};
    for (size_t c = 0; c < 3; c++)
        attention_sinks_items(cut[c],
                              cut[c + 1],
                              sh.n_kv,
                              sh.q_offset,
                              sh.n_q_heads,
                              sh.n_kv_heads,
                              sh.head_dim,
                              sh.window,
                              f16,
                              q,
                              f16 ? (const void *) kh : k,
                              f16 ? (const void *) vh : v,
                              sinks,
                              out);
    double err = 0.0, scale = 1e-30;
    for (size_t i = 0; i < nq; i++) {
        err   = fmax(err, fabs(out[i] - ref[i]));
        scale = fmax(scale, fabs(ref[i]));
    }
    printf("%-44s max|d| %.3g of scale %.3g\n", what, err, scale);
    expect(err <= 2e-6 * fmax(scale, 1.0), what);
    free(q);
    free(k);
    free(v);
    free(out);
    free(kh);
    free(vh);
    free(ref);
}

/* A sink at -1e30 is no sink: attention_mqa_causal_kv's output. */
static void check_no_sink(void) {
    const struct shape sh = {.n_q        = 5,
                             .n_kv       = 40,
                             .q_offset   = 35,
                             .n_q_heads  = 4,
                             .n_kv_heads = 2,
                             .head_dim   = 32,
                             .window     = 0};
    const size_t       nq = sh.n_q * sh.n_q_heads * sh.head_dim, nk = sh.n_kv * sh.n_kv_heads * 32;
    float *q = malloc(nq * 4), *k = malloc(nk * 4), *v = malloc(nk * 4), *a = malloc(nq * 4),
          *b = malloc(nq * 4);
    for (size_t i = 0; i < nq; i++)
        q[i] = rnd();
    for (size_t i = 0; i < nk; i++) {
        k[i] = rnd();
        v[i] = rnd();
    }
    const float none[4] = {-1e30f, -1e30f, -1e30f, -1e30f};
    attention_sinks_items(
            0, sh.n_q * sh.n_q_heads, sh.n_kv, sh.q_offset, 4, 2, 32, 0, false, q, k, v, none, a);
    attention_mqa_causal_kv(sh.n_q, sh.n_kv, sh.q_offset, 4, 2, 32, 0, q, k, v, b);
    double err = 0.0;
    for (size_t i = 0; i < nq; i++)
        err = fmax(err, fabs((double) a[i] - b[i]));
    printf("%-44s max|d| %.3g\n", "sink -1e30 vs attention_mqa_causal_kv", err);
    expect(err <= 1e-6, "a sink of -1e30 is plain attention");
    /* A sink far above every score takes the whole softmax. */
    const float big[4] = {80.0f, 80.0f, 80.0f, 80.0f};
    attention_sinks_items(
            0, sh.n_q * sh.n_q_heads, sh.n_kv, sh.q_offset, 4, 2, 32, 0, false, q, k, v, big, a);
    double mx = 0.0;
    for (size_t i = 0; i < nq; i++)
        mx = fmax(mx, fabs(a[i]));
    expect(mx < 1e-20, "a dominant sink leaves the output near zero");
    free(q);
    free(k);
    free(v);
    free(a);
    free(b);
}

static void check_yarn(void) {
    enum { HD = 64, N = 4097 };
    const double base = 150000.0, factor = 32.0, orig = 4096.0, fast = 32.0, slow = 1.0;
    const float  mscale = (float) (1.0 + 0.1 * log(factor));
    float       *c = malloc(N * HD * sizeof *c), *s = malloc(N * HD * sizeof *s);
    rope_compute_yarn_at(0,
                         N,
                         HD,
                         (float) base,
                         (float) factor,
                         (size_t) orig,
                         (float) fast,
                         (float) slow,
                         mscale,
                         c,
                         s);
    const double pi    = 3.14159265358979323846;
    const double lo    = fmax(0.0, floor(HD * log(orig / (fast * 2 * pi)) / (2 * log(base))));
    const double hi    = fmin(HD - 1.0, ceil(HD * log(orig / (slow * 2 * pi)) / (2 * log(base))));
    double       worst = 0.0;
    for (size_t p = 0; p < N; p++) {
        for (size_t i = 0; i < HD / 2; i++) {
            const double extrap = (double) p * pow(base, -2.0 * (double) i / HD);
            const double ramp =
                    1.0 - fmin(1.0, fmax(0.0, ((double) i - lo) / fmax(0.001, hi - lo)));
            const double th = extrap / factor * (1.0 - ramp) + extrap * ramp;
            /* The float angle's own error: ~2^-23 of it, per multiply. */
            const double tol = 2e-6 + 1.2e-7 * (double) (i + 2) * th;
            const double ec  = fabs(c[p * HD + i] - cos(th) * mscale) / mscale;
            const double es  = fabs(s[p * HD + i] - sin(th) * mscale) / mscale;
            worst            = fmax(worst, fmax(ec, es) / tol);
            if (c[p * HD + i] != c[p * HD + i + HD / 2] || s[p * HD + i] != s[p * HD + i + HD / 2])
                worst = INFINITY; /* NEOX halves duplicated */
        }
    }
    printf("%-44s worst error / tolerance %.3g\n", "rope_compute_yarn_at vs ggml rope_yarn", worst);
    expect(worst <= 1.0, "the YaRN table matches ggml's rope_yarn in double");
    free(c);
    free(s);
}

/* A backend's fused->attention_sinks against the double reference: every
 * backend in the build that has the op (Metal). */
static struct geist_buffer *upload(struct geist_backend *be, const void *x, size_t bytes) {
    struct geist_buffer *b = nullptr;
    if (be->desc->vtbl->buffer_create(be, bytes, GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &b) !=
                GEIST_OK ||
        be->desc->vtbl->buffer_upload(b, bytes, (const uint8_t *) x) != GEIST_OK) {
        fprintf(stderr, "buffer create/upload failed\n");
        exit(GEIST_TEST_ERROR);
    }
    return b;
}

static struct geist_tensor
view3(struct geist_buffer *b, size_t d0, size_t d1, size_t d2, enum geist_dtype dt) {
    return (struct geist_tensor) {.buffer = b,
                                  .dtype  = dt,
                                  .layout = GEIST_LAYOUT_DENSE,
                                  .ndim   = 3,
                                  .shape  = {(int64_t) d0, (int64_t) d1, (int64_t) d2},
                                  .stride = {(int64_t) (d1 * d2), (int64_t) d2, 1}};
}

/* tol: of the reference's scale. The one-simdgroup kernel (caches under
 * 64 keys in a prefill, under 32 at decode) reads Q and the weights in
 * f32; the flash kernels (#487) read Q, K, V and P as half, as the plain
 * attention's do, and the split decode an F32 cache rounded to half. */
static void
check_backend_sinks(const char *name, struct shape sh, bool f16, float sink_scale, double tol) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create(name, nullptr, nullptr, &be) != GEIST_OK)
        return;
    if (be->desc->fused == nullptr || be->desc->fused->attention_sinks == nullptr) {
        geist_backend_destroy(be);
        return;
    }
    const size_t nq = sh.n_q * sh.n_q_heads * sh.head_dim,
                 nk = sh.n_kv * sh.n_kv_heads * sh.head_dim;
    float    *q = malloc(nq * 4), *k = malloc(nk * 4), *v = malloc(nk * 4), *out = malloc(nq * 4);
    uint16_t *kh = malloc(nk * 2), *vh = malloc(nk * 2);
    double   *ref = malloc(nq * sizeof *ref);
    float     sinks[64];
    for (size_t i = 0; i < nq; i++)
        q[i] = rnd() * 0.5f;
    for (size_t i = 0; i < nk; i++) {
        k[i]  = rnd();
        v[i]  = rnd();
        kh[i] = to_half(k[i]);
        vh[i] = to_half(v[i]);
        if (f16) {
            k[i] = from_half(kh[i]);
            v[i] = from_half(vh[i]);
        }
    }
    for (size_t h = 0; h < sh.n_q_heads; h++)
        sinks[h] = rnd() * sink_scale;
    ref_sinks(&sh, q, k, v, sinks, ref);
    const enum geist_dtype kt = f16 ? GEIST_DTYPE_F16 : GEIST_DTYPE_F32;
    struct geist_buffer   *bq = upload(be, q, nq * 4), *bo = upload(be, q, nq * 4);
    struct geist_buffer   *bk = f16 ? upload(be, kh, nk * 2) : upload(be, k, nk * 4);
    struct geist_buffer   *bv = f16 ? upload(be, vh, nk * 2) : upload(be, v, nk * 4);
    struct geist_buffer   *bs = upload(be, sinks, sh.n_q_heads * 4);
    struct geist_tensor    tq = view3(bq, sh.n_q, sh.n_q_heads, sh.head_dim, GEIST_DTYPE_F32);
    struct geist_tensor    to = view3(bo, sh.n_q, sh.n_q_heads, sh.head_dim, GEIST_DTYPE_F32);
    struct geist_tensor    tk = view3(bk, sh.n_kv, sh.n_kv_heads, sh.head_dim, kt);
    struct geist_tensor    tv = view3(bv, sh.n_kv, sh.n_kv_heads, sh.head_dim, kt);
    struct geist_tensor    ts = {.buffer = bs,
                                 .dtype  = GEIST_DTYPE_F32,
                                 .layout = GEIST_LAYOUT_DENSE,
                                 .ndim   = 1,
                                 .shape  = {(int64_t) sh.n_q_heads},
                                 .stride = {1}};
    const struct geist_attention_sinks_args args = {.q              = &tq,
                                                    .k              = &tk,
                                                    .v              = &tv,
                                                    .sinks          = &ts,
                                                    .out            = &to,
                                                    .q_offset       = sh.q_offset,
                                                    .sliding_window = sh.window};
    const enum geist_status                 st   = be->desc->fused->attention_sinks(be, &args);
    double                                  err = 1e30, scale = 1e-30;
    if (st == GEIST_OK &&
        be->desc->vtbl->buffer_download(nq * 4, (uint8_t *) out, bo) == GEIST_OK) {
        err = 0.0;
        for (size_t i = 0; i < nq; i++) {
            err   = fmax(err, fabs(out[i] - ref[i]));
            scale = fmax(scale, fabs(ref[i]));
        }
    }
    char what[160];
    snprintf(what,
             sizeof what,
             "%s attention_sinks %s n_q %zu at %zu n_kv %zu hd %zu window %zu sink*%g",
             name,
             f16 ? "F16" : "F32",
             sh.n_q,
             sh.q_offset,
             sh.n_kv,
             sh.head_dim,
             sh.window,
             (double) sink_scale);
    printf("%-44s max|d| %.3g of scale %.3g\n", what, err, scale);
    expect(err <= tol * fmax(scale, 1.0), what);
    struct geist_buffer *bufs[] = {bq, bo, bk, bv, bs};
    for (size_t i = 0; i < 5; i++)
        be->desc->vtbl->buffer_destroy(be, bufs[i]);
    free(q);
    free(k);
    free(v);
    free(out);
    free(kh);
    free(vh);
    free(ref);
    geist_backend_destroy(be);
}

int main(void) {
    const struct shape prefill = {.n_q        = 37,
                                  .n_kv       = 37,
                                  .q_offset   = 0,
                                  .n_q_heads  = 8,
                                  .n_kv_heads = 2,
                                  .head_dim   = 64,
                                  .window     = 0};
    struct shape       decode  = {.n_q        = 1,
                                  .n_kv       = 1100,
                                  .q_offset   = 1099,
                                  .n_q_heads  = 8,
                                  .n_kv_heads = 2,
                                  .head_dim   = 64,
                                  .window     = 0};
    check_sinks("prefill, F32, unbounded", prefill, false, 2.0f);
    check_sinks("prefill, F16, unbounded", prefill, true, 2.0f);
    struct shape sw = prefill;
    sw.window       = 9;
    check_sinks("prefill, F32, window 9", sw, false, 2.0f);
    check_sinks("decode at 1099 (3 blocks), F32", decode, false, 2.0f);
    check_sinks("decode at 1099 (3 blocks), F16", decode, true, 6.0f);
    decode.window = 128;
    check_sinks("decode at 1099, window 128, F32", decode, false, 6.0f);
    const struct shape chunk = {.n_q        = 7,
                                .n_kv       = 600,
                                .q_offset   = 593,
                                .n_q_heads  = 6,
                                .n_kv_heads = 3,
                                .head_dim   = 48,
                                .window     = 0};
    check_sinks("a chunk after 593 rows, head_dim 48", chunk, false, 3.0f);
    const char *const gpus[] = {"metal", "vulkan"};
    for (size_t b = 0; b < 2; b++) {
        const struct shape pf  = {.n_q        = 37,
                                  .n_kv       = 37,
                                  .q_offset   = 0,
                                  .n_q_heads  = 8,
                                  .n_kv_heads = 2,
                                  .head_dim   = 64,
                                  .window     = 0};
        struct shape       sw  = pf;
        sw.window              = 9;
        const struct shape dc  = {.n_q        = 1,
                                  .n_kv       = 1100,
                                  .q_offset   = 1099,
                                  .n_q_heads  = 8,
                                  .n_kv_heads = 2,
                                  .head_dim   = 64,
                                  .window     = 128};
        const struct shape h48 = {.n_q        = 7,
                                  .n_kv       = 600,
                                  .q_offset   = 593,
                                  .n_q_heads  = 6,
                                  .n_kv_heads = 3,
                                  .head_dim   = 48,
                                  .window     = 0};
        /* Under 64 keys and head_dim 48: the one-simdgroup kernel. */
        check_backend_sinks(gpus[b], pf, false, 3.0f, 1e-5);
        check_backend_sinks(gpus[b], sw, true, 3.0f, 1e-5);
        check_backend_sinks(gpus[b], h48, false, 3.0f, 1e-5);
        /* Decode on the split-KV kernel, the sink in its combine (#487). */
        check_backend_sinks(gpus[b], dc, false, 3.0f, 2e-3);
        check_backend_sinks(gpus[b], dc, true, 3.0f, 1e-5);
        struct shape dl = dc;
        dl.window       = 0;
        check_backend_sinks(gpus[b], dl, true, 3.0f, 1e-5);
        /* A sink far above every score: the output near zero, and its
         * error with it; far below: plain attention. */
        check_backend_sinks(gpus[b], dc, true, 40.0f, 1e-5);
        check_backend_sinks(gpus[b], dl, true, -1e30f, 1e-5);
        /* Prefill on the flash kernels (#487), gpt-oss-20b's head shape
         * (64 dims, 8 q heads per KV head here): a whole prompt with and
         * without its 128-key window, ragged last tiles, a chunk after
         * 400 rows whose window starts inside a key block, and sinks
         * above every score. */
        const struct shape fp = {.n_q        = 203,
                                 .n_kv       = 203,
                                 .q_offset   = 0,
                                 .n_q_heads  = 16,
                                 .n_kv_heads = 2,
                                 .head_dim   = 64,
                                 .window     = 0};
        struct shape       fw = fp;
        fw.window             = 128;
        struct shape fc       = fw;
        fc.n_q                = 77;
        fc.q_offset           = 413;
        fc.n_kv               = 490;
        struct shape fs       = fc;
        fs.window             = 37;
        check_backend_sinks(gpus[b], fp, true, 3.0f, 2e-3);
        check_backend_sinks(gpus[b], fp, false, 3.0f, 2e-3);
        check_backend_sinks(gpus[b], fw, true, 3.0f, 2e-3);
        check_backend_sinks(gpus[b], fc, true, 3.0f, 2e-3);
        check_backend_sinks(gpus[b], fc, false, 3.0f, 2e-3);
        check_backend_sinks(gpus[b], fs, true, 3.0f, 2e-3);
        check_backend_sinks(gpus[b], fw, true, 40.0f, 2e-3);
        check_backend_sinks(gpus[b], fc, true, -1e30f, 2e-3);
    }
    check_no_sink();
    check_yarn();
    if (fails) {
        fprintf(stderr, "FAIL: %d check(s)\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: attention sinks and the YaRN table match their double references\n");
    return GEIST_TEST_PASS;
}

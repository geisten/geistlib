/*
 * test_attn_gate_x86_unit — cpu_x86 scale_f32, sigmoid_mul and
 * attn_qgate_split, the three steps of Qwen3.5's gated attention that the
 * arch otherwise loops over on the calling thread (layer_attn.c).
 *
 *   1. scale_f32 byte for byte against x * s, the arch's loop, for lengths
 *      with and without a vector tail, on the calling thread and on the
 *      team (from 16384 floats), in place and not.
 *   2. sigmoid_mul against a double reference, x / (1 + e^-g), over gates
 *      in [-120, 120] and the edges (+-0, the -87 exp floor, +-17): the
 *      exp is 1.26 ulp at worst and gcc divides with RCPPS and a Newton
 *      step under -ffast-math, so 8 ulp of the result (or 1e-30 times |x|
 *      below the floor) is the gate. In place as the arch calls it, and a
 *      NaN gate gives NaN.
 *   3. attn_qgate_split byte for byte against the arch's copies, for 1, 3
 *      and 64 rows of 1, 4 and 24 heads of 8, 128 and 256 floats; a shape
 *      it cannot take (a joint row not 2 * heads * head_dim wide, rows
 *      that do not match) is GEIST_E_UNSUPPORTED, so that the arch's loop
 *      runs; no heads is GEIST_E_INVALID_ARG.
 *   4. cpu_x86 binds all three.
 *   5. End to end, the prefill logits of a Qwen3.5-style fixture whose
 *      second block is gated attention (model_fixtures.h) on cpu_x86,
 *      through these ops, against cpu_scalar's, through the arch's loops.
 *
 * SKIPs if cpu_x86 is not built.
 */
#include "test_helpers.h"
#include "model_fixtures.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* cpu_x86 quantizes the activations to int8 in its linears, so its logits
 * on the fixture below sit 3.4e-2 of the largest one from cpu_scalar's
 * (gcc 14). Swapping query and gate moved them by 0.23, a sigmoid of the
 * wrong sign by 0.17 and a q scaled twice by 0.22. */
constexpr double E2E_TOL = 0.08;

static uint32_t g_seed = 0x6C8E9CF5u;

static float frand(void) {
    g_seed = g_seed * 1664525u + 1013904223u;
    return (float) (g_seed >> 8) / 16777216.0f; /* [0, 1) */
}

/* An F32 [rows, cols] tensor over a fresh buffer holding `data`. */
static struct geist_tensor
upload(struct geist_backend *be, size_t rows, size_t cols, const float *data) {
    const size_t         n = rows * cols;
    struct geist_buffer *b = nullptr;
    be->desc->vtbl->buffer_create(
            be, n * sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &b);
    if (b != nullptr) {
        be->desc->vtbl->buffer_upload(b, n * sizeof(float), (const uint8_t *) data);
    }
    return (struct geist_tensor) {.buffer = b,
                                  .dtype  = GEIST_DTYPE_F32,
                                  .layout = GEIST_LAYOUT_DENSE,
                                  .ndim   = 2,
                                  .shape  = {(int64_t) rows, (int64_t) cols},
                                  .stride = {(int64_t) cols, 1}};
}

static void download(struct geist_backend *be, size_t n, const struct geist_tensor *t, float *out) {
    be->desc->vtbl->buffer_download(n * sizeof(float), (uint8_t *) out, t->buffer);
}

static void drop(struct geist_backend *be, struct geist_tensor *t) {
    be->desc->vtbl->buffer_destroy(be, t->buffer);
}

static int check_scale(struct geist_backend *be, size_t n) {
    const float s     = 0.3f;
    float      *x     = xmalloc(n * sizeof(float));
    float      *y     = xmalloc(n * sizeof(float));
    float      *ref   = xmalloc(n * sizeof(float));
    int         fails = 0;
    char        what[96];
    for (size_t i = 0; i < n; i++) {
        x[i]   = (frand() * 2.0f - 1.0f) * 50.0f;
        ref[i] = x[i] * s;
    }
    for (int in_place = 0; in_place <= 1; in_place++) {
        struct geist_tensor tx = upload(be, 1, n, x);
        struct geist_tensor ty = in_place ? tx : upload(be, 1, n, x);
        fails += geist_expect(be->desc->prims->scale_f32(be, &tx, s, &ty) == GEIST_OK,
                              "scale_f32 ran");
        download(be, n, &ty, y);
        snprintf(what,
                 sizeof what,
                 "scale_f32 n=%zu%s equals x * s",
                 n,
                 in_place ? " in place" : "");
        fails += geist_expect(memcmp(y, ref, n * sizeof(float)) == 0, what);
        drop(be, &tx);
        if (!in_place) {
            drop(be, &ty);
        }
    }
    free(x);
    free(y);
    free(ref);
    return fails;
}

static int check_sigmoid_mul(struct geist_backend *be, size_t n) {
    const struct geist_backend_fused *fused = geist_backend_fused_tbl(be);
    float                            *x     = xmalloc(n * sizeof(float));
    float                            *g     = xmalloc(n * sizeof(float));
    float                            *y     = xmalloc(n * sizeof(float));
    int                               fails = 0;
    for (size_t i = 0; i < n; i++) {
        x[i] = (frand() * 2.0f - 1.0f) * 8.0f;
        g[i] = (frand() * 2.0f - 1.0f) * 120.0f;
    }
    const float edges[] = {0.0f, -0.0f, -87.0f, -87.5f, 87.5f, 17.0f, -17.0f, 1e-30f, -1e-30f};
    for (size_t i = 0; i < sizeof edges / sizeof edges[0] && i < n; i++) {
        g[i] = edges[i];
    }
    struct geist_tensor tx = upload(be, 1, n, x);
    struct geist_tensor tg = upload(be, 1, n, g);
    fails += geist_expect(fused->sigmoid_mul(be, &tx, &tg, &tx) == GEIST_OK, "sigmoid_mul ran");
    download(be, n, &tx, y);
    double worst = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double ref = (double) x[i] / (1.0 + exp(-(double) g[i]));
        const double tol = fmax(8.0 * 5.96e-8 * fabs(ref), 1e-30 * fabs((double) x[i]));
        worst            = fmax(worst, fabs((double) y[i] - ref) / fmax(tol, 1e-45));
    }
    char what[96];
    snprintf(what, sizeof what, "sigmoid_mul n=%zu within 8 ulp of the double reference", n);
    fails += geist_expect(worst <= 1.0, what);
    printf("  n=%-6zu worst |sigmoid_mul - ref| / 8 ulp = %.3f\n", n, worst);
    drop(be, &tx);
    drop(be, &tg);
    free(x);
    free(g);
    free(y);
    return fails;
}

/* The arch's split (layer_attn.c): per row and head, query then gate. */
static void
split_ref(size_t rows, size_t heads, size_t hd, const float *joint, float *q, float *gate) {
    for (size_t t = 0; t < rows; t++) {
        for (size_t h = 0; h < heads; h++) {
            const float *src = joint + t * 2 * heads * hd + h * 2 * hd;
            memcpy(q + t * heads * hd + h * hd, src, hd * sizeof(float));
            memcpy(gate + t * heads * hd + h * hd, src + hd, hd * sizeof(float));
        }
    }
}

static int check_split(struct geist_backend *be, size_t rows, size_t heads, size_t hd) {
    const struct geist_backend_fused *fused = geist_backend_fused_tbl(be);
    const size_t                      w     = heads * hd;
    float                            *j     = xmalloc(rows * 2 * w * sizeof(float));
    float                            *q     = xmalloc(rows * w * sizeof(float));
    float                            *g     = xmalloc(rows * w * sizeof(float));
    float                            *qr    = xmalloc(rows * w * sizeof(float));
    float                            *gr    = xmalloc(rows * w * sizeof(float));
    for (size_t i = 0; i < rows * 2 * w; i++) {
        j[i] = frand() - 0.5f;
    }
    memset(q, 0, rows * w * sizeof(float));
    memset(g, 0, rows * w * sizeof(float));
    split_ref(rows, heads, hd, j, qr, gr);
    struct geist_tensor tj = upload(be, rows, 2 * w, j);
    struct geist_tensor tq = upload(be, rows, w, q);
    struct geist_tensor tg = upload(be, rows, w, g);
    int fails = geist_expect(fused->attn_qgate_split(be, &tj, heads, hd, &tq, &tg) == GEIST_OK,
                             "split ran");
    download(be, rows * w, &tq, q);
    download(be, rows * w, &tg, g);
    char what[96];
    snprintf(what,
             sizeof what,
             "split %zu rows x %zu heads x %zu equals the arch's",
             rows,
             heads,
             hd);
    fails += geist_expect(memcmp(q, qr, rows * w * sizeof(float)) == 0 &&
                                  memcmp(g, gr, rows * w * sizeof(float)) == 0,
                          what);
    drop(be, &tj);
    drop(be, &tq);
    drop(be, &tg);
    free(j);
    free(q);
    free(g);
    free(qr);
    free(gr);
    return fails;
}

/* Logits after a prompt on `backend` for a Qwen3.5-style hybrid whose
 * second block is gated attention, sized so that the joint q+gate rows
 * (40 x 8 x 2 x 64 floats) cross the team threshold. false if the backend
 * is not in this build. */
static bool attn_logits(const char *backend, size_t vocab, float *out) {
    struct tf_vocab       v  = tf_make_vocab("\xc4\xa0", false);
    struct tf_buf         g  = mf_qwen35_gguf(&(struct mf_qwen35) {.layers     = 2,
                                                                   .interval   = 2,
                                                                   .d_model    = 256,
                                                                   .heads      = 8,
                                                                   .kv_heads   = 2,
                                                                   .head_dim   = 64,
                                                                   .rope_dims  = 16,
                                                                   .ffn        = 512,
                                                                   .dn_k_heads = 2,
                                                                   .dn_v_heads = 4,
                                                                   .dn_head_k  = 16,
                                                                   .dn_head_v  = 16,
                                                                   .dn_conv    = 4,
                                                                   .seed       = 11,
                                                                   .tok        = &v});
    struct geist_backend *be = nullptr;
    struct geist_model   *m  = nullptr;
    struct geist_session *s  = nullptr;
    bool                  ok = false;
    if (v.n_tok == vocab && geist_backend_create(backend, nullptr, nullptr, &be) == GEIST_OK) {
        const struct geist_session_opts o = {.top_p = 1.0f, .m_max = 64};
        geist_token_t                   prompt[40];
        for (size_t i = 0; i < 40; i++) {
            prompt[i] = (geist_token_t) ((i * 7 + 3) % vocab);
        }
        size_t       n = 0;
        const float *p = nullptr;
        ok             = geist_model_load_from_memory(g.b, g.n, be, &m) == GEIST_OK &&
                         geist_session_create(m, be, &o, &s) == GEIST_OK &&
                         geist_session_prefill_tokens(s, 40, prompt) == GEIST_OK &&
                         (p = geist_session_peek_logits(&n, s)) != nullptr && n == vocab;
        if (ok) {
            memcpy(out, p, vocab * sizeof *out);
        }
    }
    geist_session_destroy(s);
    geist_model_destroy(m);
    geist_backend_destroy(be);
    free(g.b);
    tf_free_vocab(&v);
    return ok;
}

int main(void) {
    struct geist_backend *be = nullptr;
    GEIST_SKIP_IF(geist_backend_create("cpu_x86", nullptr, nullptr, &be) != GEIST_OK,
                  "cpu_x86 backend not compiled in");
    const struct geist_backend_fused *fused = geist_backend_fused_tbl(be);
    int                               fails =
            geist_expect(be->desc->prims->scale_f32 != nullptr && fused->sigmoid_mul != nullptr &&
                                 fused->attn_qgate_split != nullptr,
                         "cpu_x86 binds scale_f32, sigmoid_mul and attn_qgate_split");

    const size_t lengths[] = {1, 7, 8, 9, 1023, 16383, 16384, 100003};
    for (size_t k = 0; k < sizeof lengths / sizeof lengths[0]; k++) {
        fails += check_scale(be, lengths[k]);
        fails += check_sigmoid_mul(be, lengths[k]);
    }
    const size_t rows[] = {1, 3, 64}, heads[] = {1, 4, 24}, hds[] = {8, 128, 256};
    for (size_t a = 0; a < 3; a++) {
        for (size_t b = 0; b < 3; b++) {
            for (size_t c = 0; c < 3; c++) {
                fails += check_split(be, rows[a], heads[b], hds[c]);
            }
        }
    }

    /* NaN gate, NaN out, by bit pattern (-ffast-math may fold isnan) */
    const uint32_t qnan = 0x7FC00000u;
    float          nan_g, one = 1.0f, out = 0.0f;
    memcpy(&nan_g, &qnan, sizeof nan_g);
    struct geist_tensor tx = upload(be, 1, 1, &one);
    struct geist_tensor tg = upload(be, 1, 1, &nan_g);
    fails += geist_expect(fused->sigmoid_mul(be, &tx, &tg, &tx) == GEIST_OK, "sigmoid_mul on NaN");
    download(be, 1, &tx, &out);
    uint32_t bits = 0;
    memcpy(&bits, &out, sizeof bits);
    fails += geist_expect((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0,
                          "a NaN gate gives NaN");
    drop(be, &tx);
    drop(be, &tg);

    /* refusals: a joint row that is not 2 * heads * head_dim wide, no heads */
    float               buf[128] = {0};
    struct geist_tensor tj       = upload(be, 2, 48, buf);
    struct geist_tensor tq       = upload(be, 2, 24, buf);
    struct geist_tensor tg2      = upload(be, 2, 24, buf);
    fails += geist_expect(fused->attn_qgate_split(be, &tj, 2, 8, &tq, &tg2) == GEIST_E_UNSUPPORTED,
                          "a joint row of the wrong width is unsupported");
    fails += geist_expect(fused->attn_qgate_split(be, &tj, 0, 12, &tq, &tg2) == GEIST_E_INVALID_ARG,
                          "no heads is an invalid argument");
    /* the counts agree (128 = 2 * 64) but the joint has 2 rows of 64, q 4
     * of 16 */
    struct geist_tensor tj2 = upload(be, 2, 64, buf);
    struct geist_tensor tq2 = upload(be, 4, 16, buf);
    struct geist_tensor tg3 = upload(be, 4, 16, buf);
    fails +=
            geist_expect(fused->attn_qgate_split(be, &tj2, 2, 8, &tq2, &tg3) == GEIST_E_UNSUPPORTED,
                         "rows that do not match are unsupported");
    drop(be, &tj2);
    drop(be, &tq2);
    drop(be, &tg3);
    drop(be, &tj);
    drop(be, &tq);
    drop(be, &tg2);

    geist_backend_destroy(be);

    /* end to end: the gated attention through these ops on cpu_x86 against
     * cpu_scalar's loops */
    struct tf_vocab tv    = tf_make_vocab("\xc4\xa0", false);
    const size_t    vocab = tv.n_tok;
    tf_free_vocab(&tv);
    float *lx = xmalloc(vocab * sizeof *lx), *ls = xmalloc(vocab * sizeof *ls);
    if (attn_logits("cpu_x86", vocab, lx) && attn_logits("cpu_scalar", vocab, ls)) {
        double md = 0.0, scale = 1e-6;
        for (size_t i = 0; i < vocab; i++) {
            md    = fmax(md, fabs((double) lx[i] - ls[i]));
            scale = fmax(scale, fabs((double) ls[i]));
        }
        printf("  prefill logits cpu_x86 vs cpu_scalar: rel %.2e\n", md / scale);
        fails += geist_expect(md / scale <= E2E_TOL,
                              "cpu_x86's gated attention agrees with cpu_scalar's");
    } else {
        fails += geist_expect(false, "fixture prefill ran on cpu_x86 and cpu_scalar");
    }
    free(lx);
    free(ls);
    if (fails == 0) {
        printf("PASS: cpu_x86 scale_f32 and attn_qgate_split bit-identical, sigmoid_mul within "
               "8 ulp\n");
    }
    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}

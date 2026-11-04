/*
 * test_granitemoe_unit - the Granite-MoE family on an in-memory model (#241).
 *
 * mf_qwen3moe_gguf with .granite writes a two-layer Granite-MoE with F32
 * weights: 8 experts, 2 used per token, 8 query heads over 4 KV heads, and
 * all four scales. On every backend in the build:
 *   - a 21-token prompt prefilled in chunks and one token at a time give the
 *     same next-token logits, within 1e-4 of their scale;
 *   - a reset session reproduces its logits bit for bit;
 *   - the logits match cpu_scalar's: cpu_neon and Metal within relative L2
 *     1e-3, other backends by cosine >= 0.999 (cpu_x86 runs F32 as W8A8).
 * On cpu_scalar:
 *   - one token's logits match the literal graph, written out below with the
 *     residual scale on both sums, which the family moves into the embedding
 *     multiplier at load; a single token's attention returns its own v, so
 *     the reference needs no RoPE;
 *   - doubling logit_scale halves the logits; attention.scale changes them
 *     from the second token on, and not for the first;
 *   - a shared expert and a negative or missing scale are refused.
 */
#define GEIST_INTERNAL_ARCH_LAYER
#define GEIST_INTERNAL_ENGINE_LAYER

#include "../src/archs/transformer/arch_state.h"
#include "../src/engine/model.h"
#include "gguf_reader.h"
#include "model_fixtures.h"
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    VOCAB    = 96,
    N_PROMPT = 21,
    D        = 64,
    HEADS    = 8,
    KV_HEADS = 4,
    HEAD_DIM = 8,
    EXPERTS  = 8,
    USED     = 2,
    FF       = 32,
    LAYERS   = 2,
};

static const struct mf_qwen3moe GEOM = {.layers          = LAYERS,
                                        .d_model         = D,
                                        .heads           = HEADS,
                                        .kv_heads        = KV_HEADS,
                                        .head_dim        = HEAD_DIM,
                                        .experts         = EXPERTS,
                                        .used            = USED,
                                        .expert_ff       = FF,
                                        .vocab           = VOCAB,
                                        .seed            = 13,
                                        .granite         = true,
                                        .attention_scale = 0.05f,
                                        .embedding_scale = 12.0f,
                                        .residual_scale  = 0.22f,
                                        .logit_scale     = 6.0f};

static const struct geist_session_opts OPTS = {.top_p = 1.0f, .m_max = 8, .max_seq_len = 256};

static bool
logits_after(struct geist_session *s, size_t n_ids, size_t step, float out[static VOCAB]) {
    geist_token_t ids[N_PROMPT];
    for (size_t i = 0; i < N_PROMPT; i++)
        ids[i] = (geist_token_t) ((i * 37 + 5) % VOCAB);
    if (geist_session_reset(s) != GEIST_OK)
        return false;
    for (size_t i = 0; i < n_ids; i += step) {
        const size_t k = n_ids - i < step ? n_ids - i : step;
        if (geist_session_prefill_tokens(s, k, ids + i) != GEIST_OK)
            return false;
    }
    size_t       nl = 0;
    const float *lg = geist_session_peek_logits(&nl, s);
    if (lg == nullptr || nl != VOCAB)
        return false;
    memcpy(out, lg, VOCAB * sizeof(float));
    return true;
}

static double cosine(const float a[static VOCAB], const float b[static VOCAB]) {
    double ab = 0.0, aa = 0.0, bb = 0.0;
    for (size_t i = 0; i < VOCAB; i++) {
        ab += (double) a[i] * b[i];
        aa += (double) a[i] * a[i];
        bb += (double) b[i] * b[i];
    }
    return ab / sqrt(aa * bb > 0.0 ? aa * bb : 1.0);
}

static double rel_l2(const float a[static VOCAB], const float b[static VOCAB]) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < VOCAB; i++) {
        num += ((double) a[i] - b[i]) * ((double) a[i] - b[i]);
        den += (double) b[i] * b[i];
    }
    return sqrt(num / (den > 0.0 ? den : 1.0));
}

struct loaded {
    struct geist_backend *be;
    struct geist_model   *m;
    struct geist_session *s;
    struct tf_buf         g;
};

static bool load(const struct mf_qwen3moe *c, const char *backend, struct loaded *l) {
    *l = (struct loaded) {0};
    if (geist_backend_create(backend, nullptr, nullptr, &l->be) != GEIST_OK)
        return false;
    l->g = mf_qwen3moe_gguf(c);
    return geist_model_load_from_memory(l->g.b, l->g.n, l->be, &l->m) == GEIST_OK &&
           geist_session_create(l->m, l->be, &OPTS, &l->s) == GEIST_OK;
}

static void unload(struct loaded *l) {
    geist_session_destroy(l->s);
    geist_model_destroy(l->m);
    geist_backend_destroy(l->be);
    free(l->g.b);
}

/* One backend. Writes its chunked logits to `ref`. Returns failures, or -1
 * when not built. */
static int check_backend(const char *name, float ref[static VOCAB]) {
    struct geist_backend *probe = nullptr;
    if (geist_backend_create(name, nullptr, nullptr, &probe) != GEIST_OK) {
        printf("skip %s: not in this build\n", name);
        return -1;
    }
    geist_backend_destroy(probe);
    struct loaded l;
    int           fails = 0;
    float         chunked[VOCAB], single[VOCAB], again[VOCAB];
    if (!load(&GEOM, name, &l)) {
        fails += geist_expect(false, "the granite fixture loads");
    } else if (!logits_after(l.s, N_PROMPT, N_PROMPT, chunked) ||
               !logits_after(l.s, N_PROMPT, 1, single) ||
               !logits_after(l.s, N_PROMPT, N_PROMPT, again)) {
        fails += geist_expect(false, "prefill runs");
    } else {
        double scale = 0.0, diff = 0.0;
        for (size_t i = 0; i < VOCAB; i++) {
            scale = fmax(scale, fabs(chunked[i]));
            diff  = fmax(diff, fabs((double) chunked[i] - single[i]));
        }
        printf("%s: chunks vs one at a time: max|d| %.3g of scale %.3g\n", name, diff, scale);
        fails += geist_expect(diff <= 1e-4 * scale, "chunks and single tokens agree");
        fails += geist_expect(memcmp(chunked, again, sizeof chunked) == 0,
                              "a reset session reproduces its logits bit for bit");
        memcpy(ref, chunked, sizeof chunked);
    }
    unload(&l);
    return fails;
}

/* ---- The literal graph, for one token --------------------------------- */

/* y = W x, with W's `rows` rows of `cols` elements. */
static void matvec(size_t rows, size_t cols, const float *w, const double *x, double *y) {
    for (size_t i = 0; i < rows; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < cols; j++)
            sum += (double) w[i * cols + j] * x[j];
        y[i] = sum;
    }
}

static void rmsnorm(const double x[static D], const float w[static D], double y[static D]) {
    double mean = 0.0;
    for (size_t i = 0; i < D; i++)
        mean += x[i] * x[i];
    mean /= D;
    const double rms = sqrt(mean + 1e-6);
    for (size_t i = 0; i < D; i++)
        y[i] = (double) w[i] * x[i] / rms;
}

[[gnu::format(printf, 2, 3)]] static const float *weight(struct gguf_ctx *g, const char *fmt, ...) {
    char    name[64];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(name, sizeof name, fmt, ap);
    va_end(ap);
    const struct gguf_tensor_t *t = gguf_get_tensor(g, name);
    return t == nullptr ? nullptr : (const float *) t->data;
}

static void literal_logits(struct gguf_ctx          *g,
                           const struct mf_qwen3moe *c,
                           int                       token,
                           float                     out[static VOCAB]) {
    const double r = c->residual_scale;
    const float *e = weight(g, "token_embd.weight");
    double       h[D], x[D], y[D];
    for (size_t i = 0; i < D; i++)
        h[i] = (double) c->embedding_scale * e[(size_t) token * D + i];
    for (int l = 0; l < LAYERS; l++) {
        double v[KV_HEADS * HEAD_DIM], a[HEADS * HEAD_DIM];
        rmsnorm(h, weight(g, "blk.%d.attn_norm.weight", l), x);
        matvec(KV_HEADS * HEAD_DIM, D, weight(g, "blk.%d.attn_v.weight", l), x, v);
        for (size_t q = 0; q < HEADS; q++) /* attention over one token: its v */
            memcpy(a + q * HEAD_DIM,
                   v + q / (HEADS / KV_HEADS) * HEAD_DIM,
                   HEAD_DIM * sizeof(double));
        matvec(D, HEADS * HEAD_DIM, weight(g, "blk.%d.attn_output.weight", l), a, y);
        for (size_t i = 0; i < D; i++)
            h[i] += r * y[i];

        double router[EXPERTS], p[EXPERTS], acc[D] = {0.0};
        rmsnorm(h, weight(g, "blk.%d.ffn_norm.weight", l), x);
        matvec(EXPERTS, D, weight(g, "blk.%d.ffn_gate_inp.weight", l), x, router);
        double sum = 0.0;
        for (size_t i = 0; i < EXPERTS; i++)
            sum += exp(router[i]);
        for (size_t i = 0; i < EXPERTS; i++)
            p[i] = exp(router[i]) / sum;
        size_t top[USED];
        double kept = 0.0;
        for (size_t u = 0; u < USED; u++) {
            size_t best = EXPERTS;
            for (size_t i = 0; i < EXPERTS; i++)
                if (p[i] >= 0.0 && (best == EXPERTS || p[i] > p[best]))
                    best = i;
            top[u] = best;
            kept += p[best];
            p[best] = -1.0 - p[best]; /* taken; the weight stays recoverable */
        }
        const float *gate = weight(g, "blk.%d.ffn_gate_exps.weight", l);
        const float *up   = weight(g, "blk.%d.ffn_up_exps.weight", l);
        const float *down = weight(g, "blk.%d.ffn_down_exps.weight", l);
        for (size_t u = 0; u < USED; u++) {
            const size_t ex = top[u];
            const double w  = (-1.0 - p[ex]) / kept;
            double       gv[FF], uv[FF], mid[FF];
            matvec(FF, D, gate + ex * FF * D, x, gv);
            matvec(FF, D, up + ex * FF * D, x, uv);
            for (size_t i = 0; i < FF; i++)
                mid[i] = gv[i] / (1.0 + exp(-gv[i])) * uv[i];
            matvec(D, FF, down + ex * FF * D, mid, y);
            for (size_t i = 0; i < D; i++)
                acc[i] += w * y[i];
        }
        for (size_t i = 0; i < D; i++)
            h[i] += r * acc[i];
    }
    rmsnorm(h, weight(g, "output_norm.weight"), x);
    for (size_t t = 0; t < VOCAB; t++) {
        double sum = 0.0;
        for (size_t i = 0; i < D; i++)
            sum += (double) e[t * D + i] * x[i];
        out[t] = (float) (sum / (double) c->logit_scale);
    }
}

static int check_literal_graph(void) {
    struct loaded l;
    int           fails = 0;
    float         got[VOCAB], want[VOCAB];
    if (!load(&GEOM, "cpu_scalar", &l) || !logits_after(l.s, 1, 1, got)) {
        fails += geist_expect(false, "the granite fixture runs one token");
    } else {
        struct transformer_arch_session *a = tf_sess(geist_session_internal_arch_session(l.s));
        literal_logits(a->model->gguf, &GEOM, 5, want);
        const double d = rel_l2(got, want);
        printf("cpu_scalar vs the literal graph: relative L2 %.3g\n", d);
        fails += geist_expect(d <= 1e-4, "one token's logits match the literal graph");
    }
    unload(&l);
    return fails;
}

static int check_scales(void) {
    struct mf_qwen3moe doubled = GEOM, other_attn = GEOM;
    doubled.logit_scale        = 2.0f * GEOM.logit_scale;
    other_attn.attention_scale = 100.0f * GEOM.attention_scale;
    struct loaded base, dl, at;
    int           fails = 0;
    float         b[VOCAB], d[VOCAB], a[VOCAB], half[VOCAB], first[VOCAB], first_at[VOCAB];
    if (!load(&GEOM, "cpu_scalar", &base) || !load(&doubled, "cpu_scalar", &dl) ||
        !load(&other_attn, "cpu_scalar", &at) || !logits_after(base.s, N_PROMPT, N_PROMPT, b) ||
        !logits_after(dl.s, N_PROMPT, N_PROMPT, d) || !logits_after(at.s, N_PROMPT, N_PROMPT, a) ||
        !logits_after(base.s, 1, 1, first) || !logits_after(at.s, 1, 1, first_at)) {
        fails += geist_expect(false, "the scale variants run");
    } else {
        for (size_t i = 0; i < VOCAB; i++)
            half[i] = b[i] * 0.5f;
        fails += geist_expect(rel_l2(d, half) <= 1e-5, "doubling logit_scale halves the logits");
        printf("attention.scale x100: relative L2 %.3g\n", rel_l2(a, b));
        fails += geist_expect(rel_l2(a, b) > 1e-3, "attention.scale changes later tokens");
        fails += geist_expect(rel_l2(first_at, first) <= 1e-6,
                              "attention.scale leaves the first token alone");
    }
    unload(&base);
    unload(&dl);
    unload(&at);
    return fails;
}

static int refused(const struct mf_qwen3moe *c, const char *what) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create("cpu_scalar", nullptr, nullptr, &be) != GEIST_OK)
        return geist_expect(false, "cpu_scalar backend");
    struct tf_buf       g = mf_qwen3moe_gguf(c);
    struct geist_model *m = nullptr;
    const int           fails =
            geist_expect(geist_model_load_from_memory(g.b, g.n, be, &m) != GEIST_OK, what);
    geist_model_destroy(m);
    geist_backend_destroy(be);
    free(g.b);
    return fails;
}

static int check_refusals(void) {
    struct mf_qwen3moe shared = GEOM, negative = GEOM, no_logit = GEOM;
    shared.shared_ff        = FF;
    negative.residual_scale = -0.22f;
    no_logit.logit_scale    = 0.0f;
    int fails               = refused(&shared, "a shared expert is refused");
    fails += refused(&negative, "a negative scale is refused");
    fails += refused(&no_logit, "a missing logit_scale is refused");
    return fails;
}

int main(void) {
    setenv("GEIST_MOE_PROFILE_CACHE", "0", 1);
    int       fails = 0;
    float     scalar[VOCAB], other[VOCAB];
    const int sc = check_backend("cpu_scalar", scalar);
    fails += sc != 0 ? geist_expect(false, "cpu_scalar runs the fixture") : 0;
    const char *const more[] = {"cpu_neon", "cpu_x86", "metal", "vulkan"};
    for (size_t b = 0; b < sizeof more / sizeof more[0]; b++) {
        const int r = check_backend(more[b], other);
        if (r < 0)
            continue;
        fails += r;
        if (r == 0 && sc == 0) {
            const double d = rel_l2(other, scalar), c = cosine(other, scalar);
            printf("%s vs cpu_scalar: relative L2 %.3g, cosine %.6f\n", more[b], d, c);
            const bool exact = strcmp(more[b], "cpu_neon") == 0 || strcmp(more[b], "metal") == 0;
            fails += geist_expect(exact ? d <= 1e-3 : c >= 0.999,
                                  "a backend matches cpu_scalar within its tolerance");
        }
    }
    fails += check_literal_graph();
    fails += check_scales();
    fails += check_refusals();
    return fails ? 1 : 0;
}

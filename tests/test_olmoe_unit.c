/*
 * test_olmoe_unit - the OLMoE family on an in-memory model (#241).
 *
 * mf_qwen3moe_gguf with .olmoe writes a two-layer OLMoE with F32 weights: 8
 * experts, 2 used per token. On every backend in the build:
 *   - a 21-token prompt prefilled in chunks and one token at a time give the
 *     same next-token logits, within 1e-4 of their scale;
 *   - a reset session reproduces its logits bit for bit;
 *   - the logits match cpu_scalar's: cpu_neon and Metal within relative L2
 *     1e-3, other backends by cosine >= 0.999 (cpu_x86 runs F32 as W8A8).
 * What makes the family OLMoE, on cpu_scalar: the expert weights are not
 * renormalised (turning moe_norm_topk on changes the logits), and QK norms
 * of head_dim elements are refused.
 */
#define GEIST_INTERNAL_ARCH_LAYER
#define GEIST_INTERNAL_ENGINE_LAYER

#include "../src/archs/transformer/arch_state.h"
#include "../src/engine/model.h"
#include "model_fixtures.h"
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { VOCAB = 96, N_PROMPT = 21 };

static const struct mf_qwen3moe GEOM = {.layers    = 2,
                                        .d_model   = 64,
                                        .heads     = 4,
                                        .kv_heads  = 4,
                                        .head_dim  = 16,
                                        .experts   = 8,
                                        .used      = 2,
                                        .expert_ff = 32,
                                        .vocab     = VOCAB,
                                        .seed      = 13,
                                        .olmoe     = true};

static const struct geist_session_opts OPTS = {.top_p = 1.0f, .m_max = 8, .max_seq_len = 256};

static bool logits_after(struct geist_session *s, size_t step, float out[static VOCAB]) {
    geist_token_t ids[N_PROMPT];
    for (size_t i = 0; i < N_PROMPT; i++)
        ids[i] = (geist_token_t) ((i * 37 + 5) % VOCAB);
    if (geist_session_reset(s) != GEIST_OK)
        return false;
    for (size_t i = 0; i < N_PROMPT; i += step) {
        const size_t k = N_PROMPT - i < step ? N_PROMPT - i : step;
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

/* One backend. Writes its chunked logits to `ref`. Returns failures, or -1
 * when not built. */
static int check_backend(const char *name, float ref[static VOCAB]) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create(name, nullptr, nullptr, &be) != GEIST_OK) {
        printf("skip %s: not in this build\n", name);
        return -1;
    }
    struct tf_buf         g     = mf_qwen3moe_gguf(&GEOM);
    struct geist_model   *m     = nullptr;
    struct geist_session *s     = nullptr;
    int                   fails = 0;
    float                 chunked[VOCAB], single[VOCAB], again[VOCAB];
    if (geist_model_load_from_memory(g.b, g.n, be, &m) != GEIST_OK ||
        geist_session_create(m, be, &OPTS, &s) != GEIST_OK) {
        fails += geist_expect(false, "the olmoe fixture loads");
    } else if (!logits_after(s, N_PROMPT, chunked) || !logits_after(s, 1, single) ||
               !logits_after(s, N_PROMPT, again)) {
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
    geist_session_destroy(s);
    geist_model_destroy(m);
    geist_backend_destroy(be);
    free(g.b);
    return fails;
}

/* The weights stay the softmax over all experts. */
static int check_not_renormalised(void) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create("cpu_scalar", nullptr, nullptr, &be) != GEIST_OK)
        return geist_expect(false, "cpu_scalar backend");
    struct tf_buf         g     = mf_qwen3moe_gguf(&GEOM);
    struct geist_model   *m     = nullptr;
    struct geist_session *s     = nullptr;
    int                   fails = 0;
    float                 plain[VOCAB], norm[VOCAB];
    if (geist_model_load_from_memory(g.b, g.n, be, &m) != GEIST_OK ||
        geist_session_create(m, be, &OPTS, &s) != GEIST_OK || !logits_after(s, N_PROMPT, plain)) {
        fails += geist_expect(false, "the olmoe fixture runs");
    } else {
        struct transformer_arch_session *a = geist_session_internal_arch_session(s);
        fails += geist_expect(!a->model->config.moe_norm_topk, "olmoe keeps the raw weights");
        a->model->config.moe_norm_topk = true;
        fails += geist_expect(logits_after(s, N_PROMPT, norm) && rel_l2(norm, plain) > 1e-2,
                              "renormalising would change the logits");
        a->model->config.moe_norm_topk = false;
    }
    geist_session_destroy(s);
    geist_model_destroy(m);
    geist_backend_destroy(be);
    free(g.b);
    return fails;
}

static int check_refuses_per_head_norms(void) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create("cpu_scalar", nullptr, nullptr, &be) != GEIST_OK)
        return geist_expect(false, "cpu_scalar backend");
    struct mf_qwen3moe c  = GEOM;
    c.bad_qk_norm         = true;
    struct tf_buf       g = mf_qwen3moe_gguf(&c);
    struct geist_model *m = nullptr;
    const int fails       = geist_expect(geist_model_load_from_memory(g.b, g.n, be, &m) != GEIST_OK,
                                         "QK norms of head_dim elements are refused");
    geist_model_destroy(m);
    geist_backend_destroy(be);
    free(g.b);
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
    fails += check_not_renormalised();
    fails += check_refuses_per_head_norms();
    return fails ? 1 : 0;
}

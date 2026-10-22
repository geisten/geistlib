/*
 * test_qwen35moe_unit - the Qwen3.5/3.6 MoE hybrid on an in-memory model
 * (#169).
 *
 * mf_qwen35_gguf with experts builds a four-layer qwen35moe with F32
 * weights: DeltaNet and attention layers alternating, 8 routed experts, 2
 * used per token, and a shared expert.
 *   - Dense twin: the router, the routed experts and the shared gate zero,
 *     the shared expert drawing the dense FFN's values with down doubled:
 *     on every CPU backend the logits equal the dense qwen35's of the same
 *     seed and KV cache (relative L2 <= 1e-5), as sigmoid(0) = 1/2. This
 *     checks the shared expert's slot, gate and weight against an
 *     independent model.
 *   - The random model: a 21-token prompt prefilled in m_max = 8 chunks and
 *     token by token give the same logits on cpu_scalar (1e-4); cpu_neon
 *     matches it within 1e-3, cpu_x86 (W8A8) within cosine 0.999.
 *   - Hot experts on each GPU backend in the build (the model on
 *     cpu_scalar, the GPU as split backend): 4 copies (the shared experts
 *     come first) and all 36; the logits match cpu_scalar's (1e-3).
 *   - The model on each GPU backend, 0, 4 and 36 copies there, chunked
 *     and token by token: cosine >= 0.999 against cpu_scalar.
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
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { N_PROMPT = 21, MAX_VOCAB = 1024, SLOTS = 4 * 9 };

static struct tf_vocab g_vocab;

static struct mf_qwen35 geom(bool moe, bool twin) {
    return (struct mf_qwen35) {.layers     = 4,
                               .interval   = 2,
                               .d_model    = 64,
                               .heads      = 4,
                               .kv_heads   = 2,
                               .head_dim   = 16,
                               .rope_dims  = 8,
                               .ffn        = 32,
                               .dn_k_heads = 2,
                               .dn_v_heads = 4,
                               .dn_head_k  = 16,
                               .dn_head_v  = 16,
                               .dn_conv    = 4,
                               .seed       = 11,
                               .tok        = &g_vocab,
                               .experts    = moe ? 8 : 0,
                               .used       = moe ? 2 : 0,
                               .dense_twin = twin};
}

static const struct geist_session_opts OPTS = {.top_p = 1.0f, .m_max = 8, .max_seq_len = 256};

static size_t vocab(void) {
    return g_vocab.n_tok;
}

/* Next-token logits after the prompt fed `step` at a time. */
static bool logits_after(struct geist_session *s, size_t step, float out[static MAX_VOCAB]) {
    geist_token_t ids[N_PROMPT];
    for (size_t i = 0; i < N_PROMPT; i++)
        ids[i] = (geist_token_t) ((i * 37 + 5) % vocab());
    if (geist_session_reset(s) != GEIST_OK)
        return false;
    for (size_t i = 0; i < N_PROMPT; i += step) {
        const size_t k = N_PROMPT - i < step ? N_PROMPT - i : step;
        if (geist_session_prefill_tokens(s, k, ids + i) != GEIST_OK)
            return false;
    }
    size_t       nl = 0;
    const float *lg = geist_session_peek_logits(&nl, s);
    if (lg == nullptr || nl != vocab())
        return false;
    memcpy(out, lg, nl * sizeof(float));
    return true;
}

static double cosine(const float *a, const float *b) {
    double ab = 0.0, aa = 0.0, bb = 0.0;
    for (size_t i = 0; i < vocab(); i++) {
        ab += (double) a[i] * b[i];
        aa += (double) a[i] * a[i];
        bb += (double) b[i] * b[i];
    }
    return ab / sqrt(aa * bb > 0.0 ? aa * bb : 1.0);
}

static double rel_l2(const float *a, const float *b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < vocab(); i++) {
        num += ((double) a[i] - b[i]) * ((double) a[i] - b[i]);
        den += (double) b[i] * b[i];
    }
    return sqrt(num / (den > 0.0 ? den : 1.0));
}

/* Load `c` on `be` (with `split` as split backend when set) and run the
 * prompt `step` at a time. */
static bool run(const struct mf_qwen35 *c,
                struct geist_backend   *be,
                struct geist_backend   *split,
                size_t                  step,
                float                   out[static MAX_VOCAB],
                size_t                 *n_hot) {
    struct tf_buf                   g  = mf_qwen35_gguf(c);
    const struct geist_session_opts lo = {.split_backend = split, .split_layers = 1};
    struct geist_model             *m  = nullptr;
    struct geist_session           *s  = nullptr;
    const bool                      ok =
            (split != nullptr ? geist_model_load_from_memory_with_opts(g.b, g.n, be, &lo, &m)
                              : geist_model_load_from_memory(g.b, g.n, be, &m)) == GEIST_OK &&
            geist_session_create(m, be, &OPTS, &s) == GEIST_OK && logits_after(s, step, out);
    if (!ok)
        fprintf(stderr,
                "  %s: %s\n",
                geist_backend_name(be),
                geist_last_create_error() ? geist_last_create_error() : geist_backend_errmsg(be));
    if (ok && n_hot != nullptr) {
        const struct transformer_arch_state *st = geist_model_internal_arch_meta(m);
        *n_hot                                  = st->moe_gpu != nullptr ? st->moe_gpu->n_hot : 0;
    }
    geist_session_destroy(s);
    geist_model_destroy(m);
    free(g.b);
    return ok;
}

static void set_hot(size_t n) {
    char v[16];
    snprintf(v, sizeof v, "%zu", n);
    setenv("GEIST_MOE_HOT", v, 1);
}

int main(void) {
    g_vocab   = tf_make_vocab("\xc4\xa0", false);
    int fails = geist_expect(vocab() <= MAX_VOCAB, "the vocabulary fits");
    if (fails > 0)
        return 1;
    static float dense[MAX_VOCAB], twin[MAX_VOCAB], ref[MAX_VOCAB], a[MAX_VOCAB], b[MAX_VOCAB];
    const struct mf_qwen35 c_dense = geom(false, false), c_twin = geom(true, true),
                           c_moe = geom(true, false);

    const char *const cpus[]   = {"cpu_scalar", "cpu_neon", "cpu_x86"};
    bool              have_ref = false;
    for (size_t i = 0; i < 3; i++) {
        struct geist_backend *be = nullptr;
        if (geist_backend_create(cpus[i], nullptr, nullptr, &be) != GEIST_OK) {
            printf("skip %s: not in this build\n", cpus[i]);
            continue;
        }
        /* One KV cache for both: the MoE family's default is F16. */
        setenv("GEIST_KV_F16", "0", 1);
        const bool pair = run(&c_dense, be, nullptr, 8, dense, nullptr) &&
                          run(&c_twin, be, nullptr, 8, twin, nullptr);
        unsetenv("GEIST_KV_F16");
        if (pair) {
            printf("%s: dense twin relative L2 %.3g\n", cpus[i], rel_l2(twin, dense));
            fails += geist_expect(rel_l2(twin, dense) <= 1e-5, "the dense twin's logits");
        } else
            fails += geist_expect(false, "the dense pair runs");
        if (run(&c_moe, be, nullptr, 8, a, nullptr) && run(&c_moe, be, nullptr, 1, b, nullptr)) {
            printf("%s: chunked vs token by token relative L2 %.3g\n", cpus[i], rel_l2(a, b));
            fails += geist_expect(rel_l2(a, b) <= 1e-4, "chunked and token by token agree");
            if (i == 0) {
                memcpy(ref, a, sizeof ref);
                have_ref = true;
            } else if (have_ref) {
                const double d = rel_l2(a, ref), co = cosine(a, ref);
                printf("%s vs cpu_scalar: relative L2 %.3g, cosine %.6f\n", cpus[i], d, co);
                fails += geist_expect(i == 1 ? d <= 1e-3 : co >= 0.999,
                                      "a fast backend matches cpu_scalar");
            }
        } else
            fails += geist_expect(false, "the MoE model runs");
        geist_backend_destroy(be);
    }

    const char *const gpus[] = {"vulkan", "metal"};
    for (size_t gi = 0; gi < 2 && have_ref; gi++) {
        struct geist_backend *gbe = nullptr, *cpu = nullptr;
        if (geist_backend_create(gpus[gi], nullptr, nullptr, &gbe) != GEIST_OK) {
            printf("skip %s: not in this build\n", gpus[gi]);
            continue;
        }
        fails +=
                geist_expect(geist_backend_create("cpu_scalar", nullptr, nullptr, &cpu) == GEIST_OK,
                             "cpu_scalar");
        const size_t hots[] = {4, SLOTS};
        for (size_t h = 0; h < 2 && cpu != nullptr; h++) {
            size_t placed = 0;
            set_hot(hots[h]);
            const bool ok = run(&c_moe, cpu, gbe, 1, a, &placed);
            printf("%s split, %zu hot: %zu placed, relative L2 %.3g\n",
                   gpus[gi],
                   hots[h],
                   placed,
                   ok ? rel_l2(a, ref) : -1.0);
            fails += geist_expect(ok && placed == hots[h] && rel_l2(a, ref) <= 1e-3,
                                  "hot experts on the split GPU match cpu_scalar");
        }
        const size_t mains[] = {0, 4, SLOTS};
        for (size_t h = 0; h < 3; h++) {
            size_t placed = 0, placed1 = 0;
            set_hot(mains[h]);
            const bool ok = run(&c_moe, gbe, nullptr, N_PROMPT, a, &placed) &&
                            run(&c_moe, gbe, nullptr, 1, b, &placed1);
            printf("%s main, %zu hot: %zu placed, cosine %.6f (chunked) %.6f (token by token)\n",
                   gpus[gi],
                   mains[h],
                   placed,
                   ok ? cosine(a, ref) : -1.0,
                   ok ? cosine(b, ref) : -1.0);
            fails += geist_expect(ok && placed == mains[h] && cosine(a, ref) >= 0.999 &&
                                          cosine(b, ref) >= 0.999,
                                  "the model on the GPU matches cpu_scalar");
        }
        unsetenv("GEIST_MOE_HOT");
        geist_backend_destroy(cpu);
        geist_backend_destroy(gbe);
    }
    fails += geist_expect(have_ref, "cpu_scalar ran the MoE model");
    tf_free_vocab(&g_vocab);
    if (fails == 0)
        printf("PASS: qwen35moe dense twin, backends, hot experts, model on the GPU\n");
    return fails == 0 ? 0 : 1;
}

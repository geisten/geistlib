/*
 * test_moe_prefill_unit - a prefill chunk's experts routed and run on the
 * device (#482), end to end on an in-memory Qwen3-MoE.
 *
 * mf_qwen3moe_gguf with its routed experts stored as Q8_0 (16 experts, 4
 * used per token, d_model 64, expert width 64): a 45-token prompt
 * prefilled in m_max = 32 chunks (32, then 13 tokens). On each GPU backend
 * in the build that routes a chunk on the device (Metal):
 *   - the model on that backend places every expert there and takes the
 *     device route for prefill chunks; the chunks' MoE layers run there
 *     (moe_gpu's layer-step counter grows by layers x chunks);
 *   - its logits match cpu_scalar's (the reference, Q8_0 dequantized
 *     exactly) within a relative L2 of 1e-3: the grouped GEMMs stage x as
 *     half, as the per-expert prefill GEMMs and llama.cpp's mul_mm_id do;
 *   - they match the same backend with GEIST_MOE_DEVICE_PREFILL=0 (the host
 *     routes each layer, the experts run as per-expert GEMMs) within the
 *     same bound, with the same top token;
 *   - they match the prompt fed one token at a time (the decode route)
 *     within the same bound;
 *   - a reset session reproduces them bit for bit.
 * SKIPs when no backend in the build takes the route.
 */
#define GEIST_INTERNAL_ARCH_LAYER
#define GEIST_INTERNAL_ENGINE_LAYER

#include "../src/archs/transformer/arch_state.h"
#include "../src/archs/transformer/moe_gpu.h"
#include "../src/engine/model.h"
#include "model_fixtures.h"
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { VOCAB = 96, N_PROMPT = 45 };

static const struct mf_qwen3moe GEOM = {.layers     = 2,
                                        .d_model    = 64,
                                        .heads      = 4,
                                        .kv_heads   = 2,
                                        .head_dim   = 16,
                                        .experts    = 16,
                                        .used       = 4,
                                        .expert_ff  = 64,
                                        .vocab      = VOCAB,
                                        .seed       = 23,
                                        .q8_experts = true};

static const struct geist_session_opts OPTS = {.top_p = 1.0f, .m_max = 32, .max_seq_len = 256};

static bool logits_after(struct geist_session *s,
                         size_t                n,
                         const geist_token_t   ids[static n],
                         size_t                step,
                         float                 out[static VOCAB]) {
    if (geist_session_reset(s) != GEIST_OK)
        return false;
    for (size_t i = 0; i < n; i += step) {
        const size_t k = n - i < step ? n - i : step;
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

static double rel_l2(const float a[static VOCAB], const float b[static VOCAB]) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < VOCAB; i++) {
        num += ((double) a[i] - b[i]) * ((double) a[i] - b[i]);
        den += (double) b[i] * b[i];
    }
    return sqrt(num / (den > 0.0 ? den : 1.0));
}

static size_t argmax(const float a[static VOCAB]) {
    size_t best = 0;
    for (size_t i = 1; i < VOCAB; i++)
        best = a[i] > a[best] ? i : best;
    return best;
}

/* The prompt's chunked prefill on backend `name`: its logits into `out`,
 * whether the session's model took the device route for prefill chunks,
 * and the layer steps routed on the device during the prefill. With
 * `single` the prompt is fed one token at a time instead. Returns false
 * when the backend is missing or a step fails. */
static bool run(const char *name,
                bool        single,
                float       out[static VOCAB],
                bool       *device_prefill,
                uint64_t   *steps,
                bool       *bit_again) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create(name, nullptr, nullptr, &be) != GEIST_OK)
        return false;
    struct tf_buf         g  = mf_qwen3moe_gguf(&GEOM);
    struct geist_model   *m  = nullptr;
    struct geist_session *s  = nullptr;
    bool                  ok = geist_model_load_from_memory(g.b, g.n, be, &m) == GEIST_OK &&
                               geist_session_create(m, be, &OPTS, &s) == GEIST_OK;
    geist_token_t         ids[N_PROMPT];
    for (size_t i = 0; i < N_PROMPT; i++)
        ids[i] = (geist_token_t) ((i * 37 + 5) % VOCAB);
    *device_prefill = false;
    *steps          = 0;
    if (ok) {
        struct transformer_arch_session *a  = tf_sess(geist_session_internal_arch_session(s));
        struct transformer_moe_gpu      *mg = a->model->moe_gpu;
        *device_prefill                     = mg != nullptr && mg->device_prefill;
        const uint64_t before               = mg != nullptr ? atomic_load(&mg->n_device) : 0;
        ok     = logits_after(s, N_PROMPT, ids, single ? 1 : OPTS.m_max, out);
        *steps = mg != nullptr ? atomic_load(&mg->n_device) - before : 0;
        float again[VOCAB];
        *bit_again = ok && logits_after(s, N_PROMPT, ids, single ? 1 : OPTS.m_max, again) &&
                     memcmp(out, again, sizeof again) == 0;
    }
    geist_session_destroy(s);
    geist_model_destroy(m);
    geist_backend_destroy(be);
    free(g.b);
    return ok;
}

int main(void) {
    setenv("GEIST_MOE_PROFILE_CACHE", "0", 1);
    float    ref[VOCAB];
    bool     dev = false, again = false;
    uint64_t steps = 0;
    int      fails = 0;
    if (!run("cpu_scalar", false, ref, &dev, &steps, &again)) {
        fprintf(stderr, "FAIL: the fixture runs on cpu_scalar\n");
        return 1;
    }
    int ran = 0;
    for (const char *const *gpu = (const char *const[]) {"metal", "vulkan", nullptr};
         *gpu != nullptr;
         gpu++) {
        float chunked[VOCAB], host[VOCAB], single[VOCAB];
        if (!run(*gpu, false, chunked, &dev, &steps, &again) || !dev) {
            printf("skip %s: not built or no device route for a prefill chunk\n", *gpu);
            continue;
        }
        ran++;
        /* 2 layers x 2 chunks */
        fails += geist_expect(steps == 4, "every chunk's MoE layers run on the device");
        fails += geist_expect(again, "a reset session reproduces the logits bit for bit");
        setenv("GEIST_MOE_DEVICE_PREFILL", "0", 1);
        bool     dev0 = true, again0 = false;
        uint64_t steps0 = 0;
        fails += geist_expect(run(*gpu, false, host, &dev0, &steps0, &again0) && !dev0 &&
                                      steps0 == 0,
                              "GEIST_MOE_DEVICE_PREFILL=0: the host routes the chunks");
        unsetenv("GEIST_MOE_DEVICE_PREFILL");
        bool     dev1 = false, again1 = false;
        uint64_t steps1 = 0;
        fails += geist_expect(run(*gpu, true, single, &dev1, &steps1, &again1),
                              "token by token runs");
        const double d_ref = rel_l2(chunked, ref), d_host = rel_l2(chunked, host);
        const double d_one = rel_l2(chunked, single);
        printf("%s: device-routed prefill, relative L2 %.3g to cpu_scalar, %.3g to the host "
               "route, %.3g to token by token; top token %zu (cpu_scalar %zu, host route %zu)\n",
               *gpu,
               d_ref,
               d_host,
               d_one,
               argmax(chunked),
               argmax(ref),
               argmax(host));
        fails += geist_expect(d_ref <= 1e-3, "device-routed prefill matches cpu_scalar");
        fails += geist_expect(d_host <= 1e-3, "device-routed prefill matches the host route");
        fails += geist_expect(d_one <= 1e-3, "device-routed prefill matches token by token");
        fails += geist_expect(argmax(chunked) == argmax(host), "the same top token");
    }
    if (ran == 0) {
        printf("SKIP: no backend in this build routes a prefill chunk on the device\n");
        return 0;
    }
    if (fails == 0)
        printf("PASS: device-routed MoE prefill end to end\n");
    return fails == 0 ? 0 : 1;
}

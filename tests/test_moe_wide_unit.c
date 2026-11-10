/*
 * test_moe_wide_unit - a prefill chunk's CPU experts split over the team
 * (#374) give the same bits as one expert per worker.
 *
 * mf_qwen3moe_gguf with its routed experts stored as Q8_0 (16 experts, 4
 * used per token, d_model 64, expert width 64): a 45-token prompt
 * prefilled in m_max = 32 chunks. On every CPU backend in the build the
 * logits with GEIST_MOE_WIDE=0 (each expert on one worker), =1 (every
 * expert on the whole team, the kernels' own split) and unset (the wide
 * ones by their share of the chunk's rows) are bit-identical: each output
 * element is the same dot product, whichever thread computes it. The
 * same prompt fed one token at a time (decode steps) gives the same bits
 * with the split off and forced (=1: every step's experts on the team).
 */
#include "model_fixtures.h"
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

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

/* The prompt's chunked prefill on backend `name` with GEIST_MOE_WIDE set
 * to `wide` (nullptr: unset), `step` tokens at a time: its last logits
 * into `out`. False when the
 * backend is not in the build or a step fails. */
static bool run(const char *name, const char *wide, size_t step, float out[static VOCAB]) {
    if (wide != nullptr)
        setenv("GEIST_MOE_WIDE", wide, 1);
    else
        unsetenv("GEIST_MOE_WIDE");
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
    for (size_t i = 0; ok && i < N_PROMPT; i += step) {
        const size_t k = N_PROMPT - i < step ? N_PROMPT - i : step;
        ok             = geist_session_prefill_tokens(s, k, ids + i) == GEIST_OK;
    }
    size_t       nl = 0;
    const float *lg = ok ? geist_session_peek_logits(&nl, s) : nullptr;
    ok              = lg != nullptr && nl == VOCAB;
    if (ok)
        memcpy(out, lg, VOCAB * sizeof(float));
    geist_session_destroy(s);
    geist_model_destroy(m);
    geist_backend_destroy(be);
    free(g.b);
    return ok;
}

int main(void) {
    setenv("GEIST_MOE_PROFILE_CACHE", "0", 1);
    int fails = 0, ran = 0;
    for (const char *const *b =
                 (const char *const[]) {"cpu_scalar", "cpu_neon", "cpu_x86", nullptr};
         *b != nullptr;
         b++) {
        float off[VOCAB], all[VOCAB], autov[VOCAB], one0[VOCAB], one[VOCAB];
        if (!run(*b, "0", OPTS.m_max, off)) {
            printf("skip %s: not in this build\n", *b);
            continue;
        }
        ran++;
        fails += geist_expect(run(*b, "1", OPTS.m_max, all), "every expert wide runs");
        fails += geist_expect(run(*b, nullptr, OPTS.m_max, autov), "the default runs");
        fails += geist_expect(run(*b, "0", 1, one0) && run(*b, "1", 1, one), "token by token runs");
        fails += geist_expect(memcmp(one0, one, sizeof one) == 0,
                              "decode steps: the same bits with the split on and off");
        fails += geist_expect(memcmp(off, all, sizeof off) == 0,
                              "every expert on the team: the same bits as one per worker");
        fails += geist_expect(memcmp(off, autov, sizeof off) == 0,
                              "the default split: the same bits as one per worker");
        printf("%s: checked\n", *b);
    }
    unsetenv("GEIST_MOE_WIDE");
    if (ran == 0) {
        fprintf(stderr, "FAIL: no CPU backend in this build\n");
        return 1;
    }
    if (fails == 0)
        printf("PASS: wide MoE experts are bit-identical to one per worker\n");
    return fails == 0 ? 0 : 1;
}

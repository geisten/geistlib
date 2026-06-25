/*
 * test_known_answer_e2e — a model-agnostic "is the forward pass sane?" canary.
 *
 * Scores a handful of trivial cloze facts by LOG-PROBABILITY (argmax over the
 * first token of each candidate continuation), NOT by free generation — greedy
 * decoding loops or drifts on base models even when the logits are fine, so a
 * generation-based known-answer check would be flaky. Logprob cloze is exactly
 * what MMLU does (gemma scores 52.8% that way), and it is what caught the BitNet
 * 2B-4T relu2 bug: a broken forward scores at chance here, a sane one aces these.
 *
 * Gated on GEIST_GGUF_PATH (skips cleanly without it). Runs against whatever
 * model is pointed at — gemma in CI, a BitNet GGUF locally — so it guards every
 * decoder family's forward, not just one. No assert(); exit code carries the
 * verdict. The candidate answers are single-token first words in the gemma /
 * llama3 / bitnet vocabs.
 */
#define _POSIX_C_SOURCE 200809L

#include "test_helpers.h"

#include <geist.h>
#include <geist_util.h>

#include <stdio.h>
#include <string.h>

struct cloze {
    const char *stem;
    const char *opts[3]; /* opts[0] is the correct answer */
};

/* Deliberately trivial world-knowledge — any non-broken 2B model gets these. */
static const struct cloze ITEMS[] = {
        {"The capital of France is", {" Paris", " London", " Berlin"}},
        {"The largest planet in our solar system is", {" Jupiter", " Mercury", " Earth"}},
        {"Water is made of hydrogen and", {" oxygen", " nitrogen", " carbon"}},
        {"The opposite of hot is", {" cold", " warm", " fast"}},
        {"Two plus two equals", {" four", " five", " three"}},
};
enum { N_ITEMS = sizeof ITEMS / sizeof ITEMS[0] };

/* Index of the highest first-token-logit option, or -1 on error. */
static int pick(struct geist_session *s, const struct cloze *c) {
    if (geist_session_reset(s) != GEIST_OK || geist_session_set_prompt(s, c->stem) != GEIST_OK) {
        return -1;
    }
    size_t       n_logits = 0;
    const float *logits   = geist_session_peek_logits(s, &n_logits);
    if (logits == nullptr || n_logits == 0) {
        return -1;
    }
    int   best  = -1;
    float bestv = 0.0f;
    for (int o = 0; o < 3; o++) {
        geist_token_t ids[8];
        size_t        n = 0;
        if (geist_session_tokenize(s, c->opts[o], 8, ids, &n) != GEIST_OK || n == 0 ||
            ids[0] >= (geist_token_t) n_logits) {
            continue;
        }
        float v = logits[ids[0]];
        if (best < 0 || v > bestv) {
            best = o, bestv = v;
        }
    }
    return best;
}

int main(void) {
    GEIST_REQUIRE_GGUF(model_path);

    struct geist_backend *be = nullptr;
    if (geist_backend_create("auto", nullptr, nullptr, &be) != GEIST_OK) {
        GEIST_SKIP("backend_create failed");
    }
    struct geist_model *model = nullptr;
    if (geist_model_load(model_path, be, &model) != GEIST_OK) {
        geist_backend_destroy(be);
        GEIST_SKIP("model_load failed (set GEIST_GGUF_PATH)");
    }
    struct geist_session_opts opts = {0};
    struct geist_session     *sess = nullptr;
    if (geist_session_create(model, be, &opts, &sess) != GEIST_OK) {
        geist_model_destroy(model);
        geist_backend_destroy(be);
        GEIST_SKIP("session_create failed");
    }

    int correct = 0;
    for (int i = 0; i < N_ITEMS; i++) {
        int p = pick(sess, &ITEMS[i]);
        fprintf(stderr,
                "  %-46s pred=%s %s\n",
                ITEMS[i].stem,
                p >= 0 ? ITEMS[i].opts[p] : "?",
                p == 0 ? "OK" : "MISS");
        correct += (p == 0);
    }

    /* A sane forward aces these; a broken one (e.g. wrong FFN activation) scores
     * at chance ~= N/3. Floor at N-1: allow one quirk, reject anything chance-like. */
    const int floor = N_ITEMS - 1;
    fprintf(stderr, "known-answer cloze: %d/%d correct (floor %d)\n", correct, N_ITEMS, floor);
    if (correct < floor) {
        fprintf(stderr, "FAIL: forward pass looks broken (near-chance world knowledge)\n");
    }

    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return correct >= floor ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}

/* Real GGUF consumer, including Gemma's lazily materialized logit softcap. */
#include "test_helpers.h"
#include <geist_decision.h>
#include <geist_util.h>

int main(void) {
    if (!geist_decision_available()) {
        GEIST_SKIP("DECISION=0");
    }
    GEIST_REQUIRE_GGUF(path);
    const char                      *backend = getenv("GEIST_TEST_BACKEND");
    struct geist_backend            *be      = nullptr;
    struct geist_model              *m       = nullptr;
    struct geist_session            *s       = nullptr;
    struct geist_decision           *d       = nullptr;
    const struct geist_session_opts  so      = {.max_seq_len = 64, .kv_mode = GEIST_KV_FP32};
    const struct geist_decision_opts o       = {
            .max_prompt_tokens = 64, .max_candidates = 4, .kv_mode = GEIST_KV_FP32};
    int fails = 0;
    if (geist_backend_create(backend != nullptr ? backend : "auto", nullptr, nullptr, &be) !=
                GEIST_OK ||
        geist_model_load_with_opts(path, be, &so, &m) != GEIST_OK ||
        geist_session_create(m, be, &so, &s) != GEIST_OK ||
        geist_decision_create(m, be, &o, &d) != GEIST_OK) {
        fprintf(stderr, "decision setup: %s\n", geist_last_create_error());
        fails++;
        goto done;
    }
    geist_token_t prompt[64];
    size_t        np = 0;
    if (geist_session_tokenize(s, "Hello", 64, prompt, &np) != GEIST_OK || np == 0) {
        fails += geist_expect(false, "real model tokenizes prompt");
        goto done;
    }
    const geist_token_t          ids[] = {1, 2, 3, 4};
    struct geist_decision_result out;
    if (geist_decision_score(d, np, 4, prompt, ids, &out) != GEIST_OK ||
        geist_session_prefill_tokens(s, np, prompt) != GEIST_OK) {
        fails += geist_expect(false, "real model decision/reference prefill");
        goto done;
    }
    size_t       vocab = 0;
    const float *p     = geist_session_peek_logits(&vocab, s);
    if (p == nullptr || vocab != geist_decision_vocab_size(d) || vocab <= 4) {
        fails += geist_expect(false, "real model dense logits vocabulary");
        goto done;
    }
    double mx = -INFINITY, sum = 0;
    for (size_t i = 0; i < 4; i++) {
        mx = fmax(mx, p[ids[i]]);
    }
    for (size_t i = 0; i < 4; i++) {
        sum += exp((double) p[ids[i]] - mx);
    }
    for (size_t i = 0; i < 4; i++) {
        fails += geist_expect(out.logits[i] == p[ids[i]],
                              "real model score matches model-conformant peek_logits exactly");
        fails += geist_expect(fabs(out.probabilities[i] - exp((double) p[ids[i]] - mx) / sum) <
                                      1e-12,
                              "real model candidate probability matches independent normalization");
    }
    geist_token_t generated;
    fails += geist_expect(geist_session_decode_step(s, &generated) == GEIST_OK,
                          "ordinary greedy generation contract preserved");
done:
    geist_decision_destroy(d);
    geist_session_destroy(s);
    geist_model_destroy(m);
    geist_backend_destroy(be);
    printf("decision: real-model dense logits/softcap %s\n", fails ? "FAIL" : "PASS");
    return fails ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
}

/* Public consumer: model fixtures only build bytes; scoring uses no internals.
 * Interleave two handles and ordinary generation, including Qwen35 recurrence. */
#include "test_helpers.h"
#include "model_fixtures.h"
#include <geist_decision.h>
#include <geist_util.h>

static int empty_result(enum geist_status                   got,
                        enum geist_status                   want,
                        const struct geist_decision_result *out) {
    return geist_expect(got == want && out->n_candidates == 0 && out->logits == nullptr &&
                                out->probabilities == nullptr && out->best_index == SIZE_MAX,
                        "failure has expected status and clears the complete result");
}

static int run(const struct tf_buf *g, const char *backend, enum geist_decision_mode mode) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create(backend, nullptr, nullptr, &be) != GEIST_OK) {
        return 0; /* not linked / no device */
    }
    struct geist_model              *m   = nullptr;
    struct geist_session            *ref = nullptr;
    struct geist_decision           *a = nullptr, *b = nullptr;
    const struct geist_session_opts  so = {.max_seq_len = 64, .kv_mode = GEIST_KV_FP32, .m_max = 8};
    const struct geist_decision_opts o  = {.mode              = mode,
                                           .max_prompt_tokens = 64,
                                           .max_candidates    = 4,
                                           .kv_mode           = GEIST_KV_FP32,
                                           .m_max             = 8};
    int                              fails = 0;
    if (geist_model_load_from_memory(g->b, g->n, be, &m) != GEIST_OK ||
        geist_session_create(m, be, &so, &ref) != GEIST_OK) {
        fails = geist_expect(false, "fixture model and reference session load");
        goto done;
    }
    fails +=
            geist_expect(geist_decision_supported(m), "loaded generative model supports decisions");
    if (strcmp(backend, "cpu_x86") == 0) {
        fails += geist_expect(!geist_decision_mode_supported(m, GEIST_DECISION_SELECTED_ROWS),
                              "x86 wrapper cannot inherit Scalar row capability");
    }
    if (!geist_decision_mode_supported(m, mode)) {
        fails += geist_expect(geist_decision_create(m, be, &o, &a) == GEIST_E_UNSUPPORTED &&
                                      a == nullptr,
                              "unsupported mode fails explicitly");
        goto done;
    }
    if (geist_decision_create(m, be, &o, &a) != GEIST_OK ||
        geist_decision_create(m, be, &o, &b) != GEIST_OK) {
        fails += geist_expect(false, "two independent decision instances create");
        goto done;
    }
    const geist_token_t          prompt[] = {1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45};
    const geist_token_t          other[]  = {7, 8, 9};
    const geist_token_t          ids[]    = {1, 7, 21, 32};
    struct geist_decision_result out      = {0};
    if (geist_session_prefill_tokens(ref, 12, prompt) != GEIST_OK ||
        geist_decision_score(a, 12, 4, prompt, ids, &out) != GEIST_OK) {
        fails += geist_expect(false, "public prefill/scoring succeed");
        goto done;
    }
    size_t       n      = 0;
    const float *logits = geist_session_peek_logits(&n, ref);
    fails += geist_expect(logits != nullptr && n == geist_decision_vocab_size(a),
                          "declared vocabulary equals reference logits");
    if (logits == nullptr || n == 0) {
        goto done;
    }
    double mx = logits[ids[0]], sum = 0;
    for (size_t i = 0; i < 4; i++) {
        mx = fmax(mx, logits[ids[i]]);
    }
    for (size_t i = 0; i < 4; i++) {
        sum += exp((double) logits[ids[i]] - mx);
    }
    float saved[4];
    memcpy(saved, out.logits, sizeof saved);
    double p_sum = 0;
    for (size_t i = 0; i < 4; i++) {
        fails += geist_expect(out.logits[i] == logits[ids[i]],
                              "selected logit equals independent dense reference");
        const double p = exp((double) logits[ids[i]] - mx) / sum;
        fails += geist_expect(fabs(out.probabilities[i] - p) < 1e-12,
                              "conditional softmax equals reference");
        p_sum += out.probabilities[i];
    }
    fails += geist_expect(fabs(p_sum - 1) < 1e-12, "candidate probabilities sum to one");
    fails += geist_expect(out.logits[out.best_index] == (float) mx, "best index selects maximum");
    const float                 *borrowed = out.logits;
    struct geist_decision_result other_out;
    fails += geist_expect(geist_decision_score(b, 3, 4, other, ids, &other_out) == GEIST_OK,
                          "second handle can score another prompt");
    geist_token_t generated;
    fails += geist_expect(geist_session_decode_step(ref, &generated) == GEIST_OK,
                          "ordinary generation remains usable on shared model");
    fails += geist_expect(memcmp(borrowed, saved, sizeof saved) == 0,
                          "another handle and generation preserve borrowed scores");
    for (int i = 0; i < 3; i++) {
        fails += geist_expect(geist_decision_score(a, 3, 4, other, ids, &out) == GEIST_OK &&
                                      geist_decision_score(a, 12, 4, prompt, ids, &out) == GEIST_OK,
                              "each query resets attention and recurrence");
        fails += geist_expect(out.n_candidates == 4 && memcmp(out.logits, saved, sizeof saved) == 0,
                              "repeated independent query is byte-identical");
    }
    geist_token_t bad[] = {-1, 7, 21, 32};
    fails += empty_result(
            geist_decision_score(a, 12, 4, prompt, bad, &out), GEIST_E_INVALID_ARG, &out);
    bad[0] = (geist_token_t) n;
    fails += empty_result(
            geist_decision_score(a, 12, 4, prompt, bad, &out), GEIST_E_INVALID_ARG, &out);
    bad[0] = 7;
    fails += empty_result(
            geist_decision_score(a, 12, 4, prompt, bad, &out), GEIST_E_INVALID_ARG, &out);
    fails += empty_result(
            geist_decision_score(a, 12, 0, prompt, ids, &out), GEIST_E_INVALID_ARG, &out);
    fails += empty_result(
            geist_decision_score(a, 0, 4, prompt, ids, &out), GEIST_E_INVALID_ARG, &out);
    fails += empty_result(
            geist_decision_score(a, 65, 4, prompt, ids, &out), GEIST_E_TOO_MANY_TOKENS, &out);
    fails += empty_result(
            geist_decision_score(a, 12, 5, prompt, ids, &out), GEIST_E_INVALID_ARG, &out);
    bad[0] = -1;
    fails += empty_result(geist_decision_score(a, 1, 4, bad, ids, &out), GEIST_E_INVALID_ARG, &out);
    fails += empty_result(
            geist_decision_score(a, 12, 4, nullptr, ids, &out), GEIST_E_INVALID_ARG, &out);
    fails += empty_result(
            geist_decision_score(a, 12, 4, prompt, nullptr, &out), GEIST_E_INVALID_ARG, &out);
    fails += geist_expect(geist_decision_score(a, 12, 4, prompt, ids, nullptr) ==
                                  GEIST_E_INVALID_ARG,
                          "null output rejected");
    fails += geist_expect(geist_decision_reset(a) == GEIST_OK &&
                                  geist_decision_score(a, 12, 4, prompt, ids, &out) == GEIST_OK &&
                                  memcmp(out.logits, saved, sizeof saved) == 0,
                          "reset and recovery after errors");
    const geist_token_t reversed[] = {32, 21, 7, 1};
    fails += geist_expect(geist_decision_score(a, 12, 4, prompt, reversed, &out) == GEIST_OK,
                          "candidate order can change");
    for (size_t i = 0; i < 4; i++) {
        fails += geist_expect(out.logits[i] == saved[3 - i],
                              "results preserve caller candidate order");
    }
    fails += geist_expect(geist_decision_score(a, 12, 1, prompt, ids, &out) == GEIST_OK &&
                                  out.probabilities[0] == 1 && out.best_index == 0,
                          "one candidate has conditional probability one");
    printf("  %s: public dense reference, reset, isolation, validation PASS\n", backend);
done:
    geist_decision_destroy(b);
    geist_decision_destroy(a);
    geist_session_destroy(ref);
    geist_model_destroy(m);
    geist_backend_destroy(be);
    return fails;
}

int main(void) {
    const char *expected = getenv("GEIST_EXPECT_DECISION");
    if (expected != nullptr && geist_decision_available() != (strcmp(expected, "1") == 0)) {
        fprintf(stderr, "FAIL: linked decision capability differs from requested build flag\n");
        return GEIST_TEST_FAIL;
    }
    if (!geist_decision_available()) {
        struct geist_decision       *d   = (struct geist_decision *) (uintptr_t) 1;
        struct geist_decision_result out = {.n_candidates = 9, .best_index = 1};
        int fails = geist_expect(!geist_decision_supported(nullptr), "disabled capability false");
        fails += geist_expect(geist_decision_create(nullptr, nullptr, nullptr, &d) ==
                                              GEIST_E_UNSUPPORTED &&
                                      d == nullptr,
                              "disabled create is linkable and clears output");
        fails += empty_result(geist_decision_score(nullptr, 0, 0, nullptr, nullptr, &out),
                              GEIST_E_UNSUPPORTED,
                              &out);
        fails += geist_expect(geist_decision_reset(nullptr) == GEIST_E_UNSUPPORTED &&
                                      geist_decision_vocab_size(nullptr) == 0,
                              "disabled reset/vocabulary contract");
        geist_decision_destroy(nullptr);
        printf("decision: disabled stubs PASS\n");
        return fails ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
    }
    int fails         = geist_expect(!geist_decision_supported(nullptr), "null model unsupported");
    struct tf_vocab v = tf_make_vocab("\xc4\xa0", false);
    struct tf_buf   models[] = {
            mf_llama_gguf(&(struct mf_llama) {.layers   = 2,
                                              .d_model  = 64,
                                              .heads    = 4,
                                              .kv_heads = 2,
                                              .ffn      = 128,
                                              .vocab    = 512,
                                              .context  = 256,
                                              .seed     = 7}),
            mf_qwen35_gguf(&(struct mf_qwen35) {.layers     = 4,
                                                .interval   = 4,
                                                .d_model    = 64,
                                                .heads      = 4,
                                                .kv_heads   = 2,
                                                .head_dim   = 16,
                                                .rope_dims  = 8,
                                                .ffn        = 128,
                                                .dn_k_heads = 2,
                                                .dn_v_heads = 4,
                                                .dn_head_k  = 16,
                                                .dn_head_v  = 16,
                                                .dn_conv    = 4,
                                                .seed       = 1,
                                                .tok        = &v}),
    };
    const char *backends[] = {"cpu_scalar", "cpu_neon", "cpu_x86", "metal"};
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 4; j++) {
            fails += run(&models[i], backends[j], GEIST_DECISION_DENSE);
            fails += run(&models[i], backends[j], GEIST_DECISION_SELECTED_ROWS);
        }
        free(models[i].b);
    }
    tf_free_vocab(&v);
    printf("decision: public reference and hybrid state isolation %s\n", fails ? "FAIL" : "PASS");
    return fails ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
}

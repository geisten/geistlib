/*
 * test_spec_head_sampling_int — the speculative output head must not change
 * decoded output (#102 Phase 1).
 *
 * Contract under test:
 *   1. Greedy: spec head ON vs exact dense head → byte-identical tokens
 *      (the long-standing spec-head guarantee, previously only verified
 *      manually — this pins it as a regression test).
 *   2. Sampling (temperature > 0): the spec head must NOT engage — decode
 *      falls back to the dense head, so spec ON vs OFF is identical by
 *      construction. This pins the #102 Phase 1 finding: extending the
 *      sketch to top-k sampling FAILED its exactness gate (the stride-4
 *      sketch's rank noise beyond rank 1 is enormous — some true top-40
 *      rows ranked outside the top-8192 rough candidates; perfect recall
 *      required a full-width phase 1, i.e. the same bytes as the dense Q8
 *      head). If someone re-enables sampling in spec_head.c without solving
 *      recall, this test fails.
 *
 * Both runs force GEIST_Q8_LMHEAD=0 so the dense reference is the exact
 * F16/W-native head. On hosts/models where the spec head is ineligible
 * (e.g. Gemma Q6_K head on x86) both runs use the dense head and the test
 * degenerates to an RNG determinism check — still valid, never flaky.
 */
#define _POSIX_C_SOURCE 200809L /* setenv */

#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_DECODE 16

static enum geist_status run_decode(const char                      *model_path,
                                    struct geist_backend            *be,
                                    const struct geist_session_opts *opts,
                                    geist_token_t                    out_tokens[static N_DECODE]) {
    struct geist_model *model = nullptr;
    enum geist_status   s     = geist_model_load(model_path, be, &model);
    if (s != GEIST_OK) {
        fprintf(stderr, "model_load failed: %s\n", geist_status_to_string(s));
        return s;
    }
    struct geist_session *sess = nullptr;
    s                          = geist_session_create(model, be, opts, &sess);
    if (s != GEIST_OK) {
        fprintf(stderr, "session_create failed: %s\n", geist_status_to_string(s));
        geist_model_destroy(model);
        return s;
    }
    s = geist_session_set_prompt(sess, "The quick brown fox");
    if (s != GEIST_OK) {
        if (s != GEIST_E_NOT_FOUND) {
            fprintf(stderr,
                    "set_prompt failed: %s — %s\n",
                    geist_status_to_string(s),
                    geist_session_errmsg(sess));
        }
        geist_session_destroy(sess);
        geist_model_destroy(model);
        return s;
    }
    for (int i = 0; i < N_DECODE; i++) {
        s = geist_session_decode_step(sess, &out_tokens[i]);
        if (s != GEIST_OK) {
            fprintf(stderr, "decode_step[%d] failed: %s\n", i, geist_status_to_string(s));
            geist_session_destroy(sess);
            geist_model_destroy(model);
            return s;
        }
    }
    geist_session_destroy(sess);
    geist_model_destroy(model);
    return GEIST_OK;
}

/* Decode twice — spec head enabled vs forced-dense — and require identical
 * token sequences. Returns 0 pass, 1 fail, -1 skip (no tokenizer). */
static int compare_runs(const char                      *label,
                        const char                      *model_path,
                        struct geist_backend            *be,
                        const struct geist_session_opts *opts) {
    geist_token_t spec[N_DECODE], dense[N_DECODE];

    setenv("GEIST_SPEC_HEAD", "1", 1);
    enum geist_status s = run_decode(model_path, be, opts, spec);
    if (s == GEIST_E_NOT_FOUND) {
        printf("SKIP: no usable tokenizer for this GGUF\n");
        return -1;
    }
    if (s != GEIST_OK) {
        return 1;
    }
    setenv("GEIST_SPEC_HEAD", "0", 1);
    if (run_decode(model_path, be, opts, dense) != GEIST_OK) {
        return 1;
    }

    if (memcmp(spec, dense, sizeof spec) != 0) {
        fprintf(stderr, "%s: spec head changes decoded output:\n", label);
        for (int i = 0; i < N_DECODE; i++) {
            fprintf(stderr,
                    "  [%2d] spec=%d dense=%d%s\n",
                    i,
                    (int) spec[i],
                    (int) dense[i],
                    spec[i] != dense[i] ? "  <-- MISMATCH" : "");
        }
        return 1;
    }
    printf("  %s: %d tokens identical (spec vs exact dense)\n", label, N_DECODE);
    return 0;
}

int main(void) {
    GEIST_REQUIRE_GGUF(model_path);

    /* Exact dense reference: keep the F16 head un-quantized in BOTH runs. */
    setenv("GEIST_Q8_LMHEAD", "0", 1);

    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create("auto", nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        fprintf(stderr, "backend create failed: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }

    int fails = 0;

    /* Greedy: spec head engages; output must be byte-identical. */
    struct geist_session_opts greedy = {.temperature = 0.0f};
    int                       r      = compare_runs("greedy", model_path, be, &greedy);
    if (r < 0) {
        geist_backend_destroy(be);
        return GEIST_TEST_SKIP;
    }
    fails += r;

    /* Sampling: spec head must NOT engage (falls back to dense), so spec
     * ON vs OFF is identical. Guards against re-enabling sampling in
     * spec_head.c without solving the sketch's rank-40 recall problem. */
    struct geist_session_opts topk = {
            .temperature = 0.8f,
            .top_k       = 40,
            .random_seed = 0xC0FFEEu,
    };
    r = compare_runs("top-k(40) t=0.8 [dense fallback]", model_path, be, &topk);
    fails += (r > 0);

    geist_backend_destroy(be);
    if (fails != 0) {
        return GEIST_TEST_FAIL;
    }
    printf("PASS: spec head output identical to exact dense head\n");
    return GEIST_TEST_PASS;
}

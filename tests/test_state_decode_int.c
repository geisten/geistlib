/*
 * test_state_decode_int — Phase B-4e sub-step 4: end-to-end token gate.
 *
 * Runs transformer_decode_step (pure backend-vtable forward) for the
 * "Hello" prompt under whatever Gemma 4 GGUF is provided and asserts the
 * produced tokens match the canonical [9259, 9259, 9259] sequence. That
 * canonical was originally cross-validated against lm.c — the lm.c
 * reference oracle was dropped along with the legacy archive in B-6
 * cleanup; the [9259, 9259, 9259] expectation is independently verified
 * via test_session_lifecycle_int which drives the public session API.
 *
 * Prefill is via repeated decode_step calls (decode-only path; the
 * batched-m>1 prefill optimization is deferred). The KV cache built up
 * by the prefill calls is what the actual decode-3-tokens reads from.
 *
 * SKIPs cleanly if no GGUF is available.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "test_helpers.h"

#include "src/archs/transformer/arch_state.h"

#include <geist.h>
#include <geist_backend.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* sp_bpe_tokenizer_encode("Hello") = [9259]. No BOS prefix. We hard-code
 * to keep the test focused on the forward pass; tokenizer correctness has
 * its own unit tests. */
#define HELLO_TOKEN 9259

#define N_DECODE_STEPS 3

int main(void) {
    GEIST_REQUIRE_GGUF(model_path);

    /* cpu_neon now handles Q5_K via dequant + cblas_sgemm (added alongside
     * the production swap). Falls back to cpu_scalar if neon isn't built. */
    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create("cpu_neon", nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        s = geist_backend_create("cpu_scalar", nullptr, nullptr, &be);
    }
    if (s != GEIST_OK) {
        fprintf(stderr, "backend create: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }

    /* ---- v2 path ---- */
    struct transformer_arch_state *st = nullptr;
    s                                 = transformer_state_create(be, model_path, nullptr, &st);
    struct transformer_arch_session *sess [[maybe_unused]] = transformer_default_session(st);
    if (s != GEIST_OK) {
        fprintf(stderr,
                "state_create: %s — %s\n",
                geist_status_to_string(s),
                geist_backend_errmsg(be));
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    /* Prefill: feed BOS then "Hello" as two decode_step calls. We don't
     * actually need the predicted tokens for prefill — only the final
     * call (at q_position=1, with token=HELLO_TOKEN) gives the logits
     * that yield the first decoded token. But running decode_step on the
     * BOS first builds up the KV cache at position 0 the way prefill
     * would. */
    const geist_token_t prompt[] = {HELLO_TOKEN};
    geist_token_t       next     = -1;
    for (size_t i = 0; i < sizeof prompt / sizeof prompt[0]; i++) {
        s = transformer_decode_step(sess, prompt[i], &next);
        if (s != GEIST_OK) {
            fprintf(stderr,
                    "v2 decode_step prefill[%zu] failed: %s — %s\n",
                    i,
                    geist_status_to_string(s),
                    geist_backend_errmsg(be));
            transformer_state_destroy(st);
            geist_backend_destroy(be);
            return GEIST_TEST_FAIL;
        }
    }
    printf("v2: after prefill, next predicted token = %d\n", (int) next);

    /* Now decode N more tokens. */
    geist_token_t _decoded[N_DECODE_STEPS];
    _decoded[0] = next;
    for (int t = 1; t < N_DECODE_STEPS; t++) {
        s = transformer_decode_step(sess, _decoded[t - 1], &_decoded[t]);
        if (s != GEIST_OK) {
            fprintf(stderr,
                    "v2 decode_step[%d] failed: %s — %s\n",
                    t,
                    geist_status_to_string(s),
                    geist_backend_errmsg(be));
            transformer_state_destroy(st);
            geist_backend_destroy(be);
            return GEIST_TEST_FAIL;
        }
    }

    transformer_state_destroy(st);
    geist_backend_destroy(be);

    /* ---- Compare against canonical ---- */
    int fails = 0;
    printf("                v2       canonical\n");
    for (int t = 0; t < N_DECODE_STEPS; t++) {
        const int   canonical = HELLO_TOKEN;
        const char *vs_can    = (_decoded[t] == canonical) ? "==" : "!=";
        printf("  step[%d]  %6d  %s  %6d\n", t, (int) _decoded[t], vs_can, canonical);
        if (_decoded[t] != canonical) {
            fails++;
        }
    }

    if (fails == 0) {
        printf("PASS: v2 decode produces canonical tokens "
               "[%d, %d, %d] for \"Hello\" prompt\n",
               HELLO_TOKEN,
               HELLO_TOKEN,
               HELLO_TOKEN);
        return GEIST_TEST_PASS;
    }
    fprintf(stderr, "FAIL: %d of %d tokens didn't match canonical\n", fails, N_DECODE_STEPS);
    return GEIST_TEST_FAIL;
}

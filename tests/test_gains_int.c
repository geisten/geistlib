/*
 * test_gains_int — ZO-tuning gains over the public API (model-gated).
 *
 * The gains array is the whole contract behind forward-only fine-tuning: a
 * tuned model is the unmodified GGUF plus n floats. This test asserts the
 * three properties that makes true, on a real model:
 *
 *   1. IDENTITY   — after load every gain is 1.0f, and the logits equal
 *                   those of a plain untuned run. Wiring the array in must
 *                   not perturb anything by itself.
 *   2. EFFECT     — writing a gain changes the logits. The overlay is
 *                   actually reaching the forward pass, not silently
 *                   dropped (the failure mode the fused tensor path would
 *                   have, which op_gains refuses outright).
 *   3. REVERSIBLE — restoring 1.0f reproduces the identity logits BIT FOR
 *                   BIT. The overlay leaves no residue in the weights, so
 *                   swapping one tuning profile for another at runtime is
 *                   a memcpy and never a reload.
 *
 * SKIPs cleanly without a GGUF (GEIST_GGUF_PATH) and without GEIST_TUNE
 * (geist_model_gains then reports UNSUPPORTED, which is a build config, not
 * a failure).
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_util.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { PROBE_TOKENS = 8 };

/* Prefill a fixed token run and copy out the resulting next-token logits.
 * Uses prefill_tokens rather than set_prompt so the probe does not depend
 * on the model's tokenizer or chat template. */
static float *probe_logits(struct geist_session *s, size_t *n_out) {
    static const geist_token_t ids[PROBE_TOKENS] = {1, 2, 3, 4, 5, 6, 7, 8};

    if (geist_session_reset(s) != GEIST_OK) {
        return nullptr;
    }
    if (geist_session_prefill_tokens(s, PROBE_TOKENS, ids) != GEIST_OK) {
        return nullptr;
    }
    size_t       n  = 0;
    const float *lg = geist_session_peek_logits(&n, s);
    if (lg == nullptr || n == 0) {
        return nullptr;
    }
    float *copy = malloc(n * sizeof *copy);
    if (copy == nullptr) {
        return nullptr;
    }
    memcpy(copy, lg, n * sizeof *copy);
    *n_out = n;
    return copy;
}

static size_t count_differing(const float *a, const float *b, size_t n) {
    size_t d = 0;
    for (size_t i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            d++;
        }
    }
    return d;
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

    float            *gains   = nullptr;
    size_t            n_gains = 0;
    enum geist_status gs      = geist_model_gains(model, &gains, &n_gains);
    if (gs == GEIST_E_UNSUPPORTED) {
        geist_model_destroy(model);
        geist_backend_destroy(be);
        GEIST_SKIP("gains unsupported (build with EXTRA_CFLAGS=-DGEIST_TUNE)");
    }
    if (gs != GEIST_OK || gains == nullptr || n_gains == 0) {
        fprintf(stderr,
                "FAIL: geist_model_gains -> %s, n=%zu\n",
                geist_status_to_string(gs),
                n_gains);
        return GEIST_TEST_FAIL;
    }
    /* Slot count is 9 per layer + 2 model-level, so it is always ≡ 2 mod 9
     * — a cheap witness that the wiring covered whole layers. */
    if (n_gains < 11 || (n_gains - 2) % 9 != 0) {
        fprintf(stderr, "FAIL: n_gains=%zu is not 9*n_layers+2\n", n_gains);
        return GEIST_TEST_FAIL;
    }

    /* --- 1. IDENTITY ---------------------------------------------------- */
    for (size_t i = 0; i < n_gains; i++) {
        if (gains[i] != 1.0f) {
            fprintf(stderr, "FAIL: gain[%zu] = %g at load, expected 1.0\n", i, (double) gains[i]);
            return GEIST_TEST_FAIL;
        }
    }

    struct geist_session_opts opts = {0}; /* greedy, deterministic */
    struct geist_session     *sess = nullptr;
    if (geist_session_create(model, be, &opts, &sess) != GEIST_OK) {
        geist_model_destroy(model);
        geist_backend_destroy(be);
        GEIST_SKIP("session_create failed");
    }

    size_t n_base = 0;
    float *base   = probe_logits(sess, &n_base);
    if (base == nullptr) {
        fprintf(stderr, "FAIL: baseline probe produced no logits\n");
        return GEIST_TEST_FAIL;
    }

    /* --- 2. EFFECT ------------------------------------------------------ *
     * Perturb the first layer's q_proj (slot 0). One tensor is enough: if a
     * single gain moves the logits, the overlay reaches the forward pass. */
    gains[0]     = 1.25f;
    size_t n_mod = 0;
    float *tuned = probe_logits(sess, &n_mod);
    if (tuned == nullptr || n_mod != n_base) {
        fprintf(stderr, "FAIL: tuned probe produced no logits\n");
        return GEIST_TEST_FAIL;
    }
    const size_t moved = count_differing(base, tuned, n_base);
    if (moved == 0) {
        fprintf(stderr,
                "FAIL: gain[0]=1.25 left all %zu logits unchanged — the "
                "overlay never reached the forward pass\n",
                n_base);
        return GEIST_TEST_FAIL;
    }

    /* --- 3. REVERSIBLE -------------------------------------------------- */
    gains[0]      = 1.0f;
    size_t n_back = 0;
    float *back   = probe_logits(sess, &n_back);
    if (back == nullptr || n_back != n_base) {
        fprintf(stderr, "FAIL: restored probe produced no logits\n");
        return GEIST_TEST_FAIL;
    }
    const size_t residue = count_differing(base, back, n_base);
    if (residue != 0) {
        fprintf(stderr,
                "FAIL: restoring gain[0]=1.0 left %zu/%zu logits changed — "
                "the overlay is not residue-free\n",
                residue,
                n_base);
        return GEIST_TEST_FAIL;
    }

    printf("PASS: %zu gains, %zu/%zu logits moved on gain[0]=1.25, exact restore\n",
           n_gains,
           moved,
           n_base);

    free(base);
    free(tuned);
    free(back);
    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return GEIST_TEST_PASS;
}

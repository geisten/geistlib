/*
 * test_prefill_determinism_int — same prompt prefilled twice must produce
 * bit-identical logits.
 *
 * Regression guard for GPU kernel races: a missing threadgroup barrier in
 * the Metal scalar attention online-softmax (fixed 2026-07-04) corrupted
 * attention nondeterministically, but ONLY when the prefill tail chunk
 * missed the flash gate (seq % 8 != 0 → scalar fallback) AND kv_len
 * spanned more than one 256-key tile. The 24-token greedy parity gate
 * (single tile) and pp512 benches (%8-aligned → flash path) both sat
 * outside that window, so the race survived every existing gate while
 * silently costing ~8 MMLU points at 5-shot prompt lengths.
 *
 * This test prefills a 301-token prompt (chunks 128+128+45: tail 45%8!=0,
 * kv_len 301 > 256 — inside the historical race window) three times in
 * one process and memcmps the exposed logits. Any nondeterministic kernel
 * on the prefill path — attention, reductions, GEMM tiles — fails it.
 * Backend-agnostic: runs on whatever backend "auto" resolves to.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_PROMPT 301
#define N_REPEATS 3

int main(void) {
    GEIST_REQUIRE_GGUF(model_path);

    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create("auto", nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        fprintf(stderr, "backend create failed: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }
    struct geist_model *model = nullptr;
    s                         = geist_model_load(model_path, be, &model);
    if (s != GEIST_OK) {
        fprintf(stderr, "model_load failed: %s\n", geist_last_create_error());
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }
    struct geist_session_opts opts = {
            .max_seq_len = 2048,
            .temperature = 0.0f,
    };
    struct geist_session *sess = nullptr;
    s                          = geist_session_create(model, be, &opts, &sess);
    if (s != GEIST_OK) {
        fprintf(stderr, "session_create failed\n");
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    /* Fixed synthetic prompt: BOS + a deterministic id pattern kept well
     * below any real vocab size. Content is irrelevant — only that both
     * prefills see identical ids. */
    geist_token_t ids[N_PROMPT];
    ids[0] = 2;
    for (size_t i = 1; i < N_PROMPT; i++) {
        ids[i] = (geist_token_t) (3 + (i * 2654435761u) % 50000u);
    }

    float *ref    = nullptr;
    size_t ref_n  = 0;
    int    result = GEIST_TEST_PASS;

    for (int rep = 0; rep < N_REPEATS; rep++) {
        s = geist_session_reset(sess);
        if (s != GEIST_OK) {
            fprintf(stderr, "reset failed (rep %d)\n", rep);
            result = GEIST_TEST_FAIL;
            break;
        }
        s = geist_session_prefill_tokens(sess, N_PROMPT, ids);
        if (s != GEIST_OK) {
            fprintf(stderr, "prefill failed (rep %d): %s\n", rep, geist_session_errmsg(sess));
            result = GEIST_TEST_FAIL;
            break;
        }
        size_t       n_logits = 0;
        const float *logits   = geist_session_peek_logits(sess, &n_logits);
        if (logits == nullptr || n_logits == 0) {
            fprintf(stderr, "peek_logits returned nothing (rep %d)\n", rep);
            result = GEIST_TEST_FAIL;
            break;
        }
        if (rep == 0) {
            ref_n = n_logits;
            ref   = malloc(n_logits * sizeof(float));
            if (ref == nullptr) {
                result = GEIST_TEST_ERROR;
                break;
            }
            memcpy(ref, logits, n_logits * sizeof(float));
            continue;
        }
        if (n_logits != ref_n) {
            fprintf(stderr, "n_logits changed: %zu vs %zu (rep %d)\n", ref_n, n_logits, rep);
            result = GEIST_TEST_FAIL;
            break;
        }
        if (memcmp(ref, logits, n_logits * sizeof(float)) != 0) {
            size_t first = 0;
            while (first < n_logits && ref[first] == logits[first]) {
                first++;
            }
            fprintf(stderr,
                    "NONDETERMINISTIC PREFILL (rep %d): logits differ at index %zu "
                    "(%.9g vs %.9g) — kernel race on the prefill path?\n",
                    rep,
                    first,
                    (double) ref[first],
                    (double) logits[first]);
            result = GEIST_TEST_FAIL;
            break;
        }
        printf("rep %d: %zu logits bit-identical\n", rep, n_logits);
    }

    free(ref);
    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return result;
}

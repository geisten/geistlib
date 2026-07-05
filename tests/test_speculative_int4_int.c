/*
 * test_speculative_int4_int — speculative decode must produce the SAME token
 * stream as sequential decode_step when the KV cache is the packed 4-bit mode
 * (GEIST_KV_INT4, issue #61 / #68).
 *
 * verify_forward tentatively writes k tokens into the cache and kv_truncate
 * rewinds kv_len on a reject. For packed INT4 the writes go through the nibble
 * packer and the reads through the unpacking kernel, and rotation (default-on
 * for INT4) rotates Q/K/V per token. If the tentative-write / rewind is off by
 * a nibble or a rotation, the speculative stream diverges from the sequential
 * one. Equivalence is spec-decode's fundamental guarantee, so any divergence
 * here is a real bug in the INT4 truncate/rewind path.
 *
 * Covers INT4 with rotation (the shipped default) and INT4 rotation-off. SKIPs
 * cleanly if no GGUF model is reachable.
 */
#define GEIST_INTERNAL_ENGINE_LAYER

/* setenv/unsetenv need a POSIX feature macro on glibc; no-op on macOS. */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static struct geist_backend *be;
static struct geist_model   *model;

/* Decode N tokens sequentially (reference) then via decode_speculative, both
 * under kv_mode + the given GEIST_KV_ROT setting; assert identical streams.
 * Returns the number of mismatches (0 = pass). */
static int equiv_under(enum geist_kv_mode mode, const char *rot_env, const char *label) {
    if (rot_env != nullptr)
        setenv("GEIST_KV_ROT", rot_env, 1);
    else
        unsetenv("GEIST_KV_ROT");

    /* Repetitive prompt so the n-gram drafter actually proposes tokens →
     * the accept/reject (and thus kv_truncate) path is exercised. */
    const geist_token_t prompt_ids[] = {2, 9259, 1018, 9259, 1018, 9259, 1018};
    const size_t        prompt_n     = sizeof prompt_ids / sizeof prompt_ids[0];
    const size_t        N            = 30;
    const size_t        K_MAX        = 4;

    struct geist_session_opts opts = {.max_seq_len = 1024, .temperature = 0.0f, .kv_mode = mode};

    /* Reference: sequential decode_step. */
    geist_token_t         ref[N];
    struct geist_session *sref = nullptr;
    if (geist_session_create(model, be, &opts, &sref) != GEIST_OK ||
        geist_session_prefill_tokens(sref, prompt_n, prompt_ids) != GEIST_OK) {
        fprintf(stderr, "[%s] ref session setup failed\n", label);
        return 1;
    }
    for (size_t i = 0; i < N; i++) {
        if (geist_session_decode_step(sref, &ref[i]) != GEIST_OK) {
            fprintf(stderr, "[%s] decode_step[%zu] failed\n", label, i);
            geist_session_destroy(sref);
            return 1;
        }
    }
    geist_session_destroy(sref);

    /* Speculative: same prompt, same N tokens. */
    geist_token_t         spec[N];
    size_t                spec_n = 0;
    struct geist_session *sspec  = nullptr;
    if (geist_session_create(model, be, &opts, &sspec) != GEIST_OK ||
        geist_session_prefill_tokens(sspec, prompt_n, prompt_ids) != GEIST_OK) {
        fprintf(stderr, "[%s] spec session setup failed\n", label);
        return 1;
    }
    geist_token_t history[1024];
    size_t        history_n = prompt_n;
    memcpy(history, prompt_ids, prompt_n * sizeof(*prompt_ids));
    geist_token_t emitted[K_MAX + 1];
    while (spec_n < N) {
        size_t            emit_n = 0;
        enum geist_status s      = geist_session_decode_speculative(
                sspec, K_MAX, history_n, history, K_MAX + 1, emitted, &emit_n);
        if (s != GEIST_OK || emit_n == 0) {
            fprintf(stderr,
                    "[%s] decode_speculative failed: %s (emit_n=%zu)\n",
                    label,
                    geist_status_to_string(s),
                    emit_n);
            geist_session_destroy(sspec);
            return 1;
        }
        for (size_t i = 0; i < emit_n && spec_n < N; i++) {
            spec[spec_n++]       = emitted[i];
            history[history_n++] = emitted[i];
        }
    }
    geist_session_destroy(sspec);

    int fails = 0;
    for (size_t i = 0; i < N; i++) {
        if (spec[i] != ref[i]) {
            fprintf(stderr, "[%s] FAIL @ %zu: ref=%d spec=%d\n", label, i, ref[i], spec[i]);
            if (++fails >= 5)
                break;
        }
    }
    printf("[%s] %s: speculative == sequential over %zu tokens\n",
           label,
           fails == 0 ? "PASS" : "FAIL",
           N);
    return fails;
}

int main(void) {
    GEIST_REQUIRE_GGUF(model_path);
    enum geist_status s = geist_backend_create("cpu_neon", nullptr, nullptr, &be);
    if (s != GEIST_OK)
        s = geist_backend_create("cpu_scalar", nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        fprintf(stderr, "backend_create: %s\n", geist_last_create_error());
        return GEIST_TEST_FAIL;
    }
    if (geist_model_load(model_path, be, &model) != GEIST_OK) {
        fprintf(stderr, "model_load: %s\n", geist_last_create_error());
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    /* Force GEIST_KV_ROT explicitly (rather than relying on the mode's
     * default) so both the rotated and unrotated paths are exercised
     * regardless of what the packed-INT4 default happens to be. */
    int fails = 0;
    fails += equiv_under(GEIST_KV_INT4, "1", "INT4+ROT");
    fails += equiv_under(GEIST_KV_INT4, "0", "INT4(rot-off)");

    geist_model_destroy(model);
    geist_backend_destroy(be);
    if (fails == 0) {
        printf("PASS: packed-INT4 speculative decode matches sequential (truncate/rewind ok)\n");
        return GEIST_TEST_PASS;
    }
    return GEIST_TEST_FAIL;
}

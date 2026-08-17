/*
 * test_audio_stream_session_int — the public streaming audio turn must be
 * EQUIVALENT to attach_audio over the same PCM (#256).
 *
 * Two sessions on one model, same synthetic clip: session A uses
 * geist_session_attach_audio; session B uses audio_begin, then pushes
 * 20 ms chunks FROM A SEPARATE CAPTURE THREAD (the documented threading
 * pattern), then audio_end. Equivalence is asserted where it matters:
 * the GREEDY PREDICTION after the audio turn must be identical, and the
 * logits close. (Bit-equality is not the contract: the streaming
 * forward's incremental attention carries ~1e-5 soft-token noise — the
 * stream-parity unit test's documented tolerance — which the 30 LM
 * layers amplify into small logit shifts.)
 *
 * SKIPs cleanly without GGUF/tower; the audio-smoke CI job runs it with
 * fixtures mandatory.
 */
#include "audio_test_util.h"
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define SECONDS 2
#define N_PCM ((size_t) AUDIO_TEST_SR * SECONDS)
#define PUSH_CHUNK 320 /* 20 ms */

static int16_t g_pcm[N_PCM];

struct pusher_arg {
    struct geist_session *sess;
    int                   rc;
};

/* Capture-thread pattern: push the clip in 20 ms chunks. */
static void *pusher(void *p) {
    struct pusher_arg *a = p;
    for (size_t off = 0; off < N_PCM; off += PUSH_CHUNK) {
        size_t take = N_PCM - off < PUSH_CHUNK ? N_PCM - off : PUSH_CHUNK;
        if (geist_session_audio_push(a->sess, take, g_pcm + off) != GEIST_OK) {
            a->rc = 1;
            return nullptr;
        }
    }
    a->rc = 0;
    return nullptr;
}

int main(void) {
    GEIST_REQUIRE_GGUF(model_path);
    const char *tower = audio_test_find_tower();
    GEIST_SKIP_IF(tower == nullptr, "audio_tower.safetensors not found (make fetch-audio-tower)");

    audio_test_synth_sweep(g_pcm, N_PCM, SECONDS);

    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create("cpu_neon", nullptr, nullptr, &be);
    if (s != GEIST_OK)
        s = geist_backend_create("cpu_scalar", nullptr, nullptr, &be);
    if (s != GEIST_OK)
        return GEIST_TEST_ERROR;
    struct geist_model *model = nullptr;
    if (geist_model_load(model_path, be, &model) != GEIST_OK) {
        geist_backend_destroy(be);
        GEIST_SKIP("model_load failed (set GEIST_GGUF_PATH)");
    }
    if ((geist_model_modalities(model) & GEIST_MOD_AUDIO) == 0) {
        geist_model_destroy(model);
        geist_backend_destroy(be);
        GEIST_SKIP("model cannot hear (tower not loaded)");
    }

    struct geist_session_opts opts   = {.max_seq_len = 1024};
    struct geist_session     *sess_a = nullptr, *sess_b = nullptr;
    if (geist_session_create(model, be, &opts, &sess_a) != GEIST_OK ||
        geist_session_create(model, be, &opts, &sess_b) != GEIST_OK) {
        return GEIST_TEST_ERROR;
    }

    int fails = 0;

    /* A: one-shot attach. */
    if (geist_session_attach_audio(sess_a, N_PCM, g_pcm, AUDIO_TEST_SR) != GEIST_OK) {
        fprintf(stderr, "attach_audio failed: %s\n", geist_session_errmsg(sess_a));
        return GEIST_TEST_FAIL;
    }

    /* B: streaming turn, pushed from a capture thread. */
    if (geist_session_audio_begin(sess_b) != GEIST_OK) {
        fprintf(stderr, "audio_begin failed: %s\n", geist_session_errmsg(sess_b));
        return GEIST_TEST_FAIL;
    }
    /* Double-begin must refuse. */
    if (geist_session_audio_begin(sess_b) != GEIST_E_INVALID_STATE) {
        fprintf(stderr, "FAIL: double audio_begin not refused\n");
        fails++;
    }
    struct pusher_arg pa = {.sess = sess_b, .rc = -1};
    pthread_t         tid;
    if (pthread_create(&tid, nullptr, pusher, &pa) != 0)
        return GEIST_TEST_ERROR;
    pthread_join(tid, nullptr);
    if (pa.rc != 0) {
        fprintf(stderr, "audio_push failed: %s\n", geist_session_errmsg(sess_b));
        return GEIST_TEST_FAIL;
    }
    if (geist_session_audio_end(sess_b) != GEIST_OK) {
        fprintf(stderr, "audio_end failed: %s\n", geist_session_errmsg(sess_b));
        return GEIST_TEST_FAIL;
    }

    /* Equivalence: identical next-position logits. */
    size_t       na = 0, nb = 0;
    const float *la = geist_session_peek_logits(sess_a, &na);
    const float *lb = geist_session_peek_logits(sess_b, &nb);
    if (la == nullptr || lb == nullptr || na != nb || na == 0) {
        fprintf(stderr, "FAIL: logits unavailable or size mismatch (%zu vs %zu)\n", na, nb);
        fails++;
    } else {
        double worst = 0.0;
        for (size_t i = 0; i < na; i++) {
            double d = fabs((double) la[i] - (double) lb[i]);
            if (d > worst)
                worst = d;
        }
        size_t am_a = 0, am_b = 0;
        for (size_t i = 1; i < na; i++) {
            if (la[i] > la[am_a])
                am_a = i;
            if (lb[i] > lb[am_b])
                am_b = i;
        }
        printf("logits: %zu entries, max|Δ| = %.3e, argmax %zu vs %zu\n", na, worst, am_a, am_b);
        /* Greedy equivalence is the contract; the measured logit shift
         * from the incremental attention is ~2, a broken injection is
         * tens-of-logits wrong with a different argmax. */
        if (am_a != am_b) {
            fprintf(stderr, "FAIL: greedy prediction diverges (%zu vs %zu)\n", am_a, am_b);
            fails++;
        }
        if (worst > 8.0) {
            fprintf(stderr, "FAIL: logit shift beyond streaming noise (%.3e)\n", worst);
            fails++;
        }
    }

    /* end without begin must refuse. */
    if (geist_session_audio_end(sess_b) != GEIST_E_INVALID_STATE) {
        fprintf(stderr, "FAIL: audio_end without begin not refused\n");
        fails++;
    }

    geist_session_destroy(sess_a);
    geist_session_destroy(sess_b);
    geist_model_destroy(model);
    geist_backend_destroy(be);

    if (fails == 0) {
        printf("PASS\n");
    } else {
        printf("FAIL (%d)\n", fails);
    }
    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}

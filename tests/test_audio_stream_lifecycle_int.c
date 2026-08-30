/*
 * test_audio_stream_lifecycle_int — the streaming audio turn's begin /
 * push / poll / end / abort contract (issue #333).
 *
 * The turn owns two things that outlive a single call: a session-lifetime
 * scratch buffer, and a stream inside the ENCODER, which is shared across
 * sessions of the same model. Both used to be released only by a
 * successful audio_end. A session destroyed mid-turn leaked the scratch
 * and left the encoder stream open for whoever came next.
 *
 * Run this under ASan/LSan — the leak half of the contract is not visible
 * any other way:
 *
 *   make MODE=asan && bin/<target>/asan/tests/test_audio_stream_lifecycle_int
 *
 * Skips cleanly when the model has no audio tower, like every other audio
 * suite here.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_util.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;

#define CHECK(cond, what)                          \
    do {                                           \
        if (!(cond)) {                             \
            fprintf(stderr, "FAIL: %s\n", (what)); \
            g_fail = 1;                            \
        }                                          \
    } while (0)

/* A second of quiet 16 kHz PCM with a little signal in it, so the encoder
 * has something to buffer. */
enum { N_SAMPLES = 16000 };

static struct geist_session *new_session(struct geist_model *model, struct geist_backend *be) {
    struct geist_session_opts opts = {.max_seq_len = 2048};
    struct geist_session     *sess = nullptr;
    if (geist_session_create(model, be, &opts, &sess) != GEIST_OK) {
        return nullptr;
    }
    return sess;
}

int main(void) {
    GEIST_REQUIRE_GGUF(model_path);

    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create("auto", nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        fprintf(stderr, "backend create failed: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }
    struct geist_model *model = nullptr;
    if (geist_model_load(model_path, be, &model) != GEIST_OK) {
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    int16_t *pcm = malloc(N_SAMPLES * sizeof(int16_t));
    if (pcm == nullptr) {
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return GEIST_TEST_ERROR;
    }
    for (int i = 0; i < N_SAMPLES; i++) {
        pcm[i] = (int16_t) ((i % 251) * 64 - 8000);
    }

    /* ---- does this model hear at all? ---------------------------------- */
    struct geist_session *probe = new_session(model, be);
    if (probe == nullptr) {
        free(pcm);
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }
    s = geist_session_audio_begin(probe);
    if (s == GEIST_E_NOT_FOUND || s == GEIST_E_UNSUPPORTED) {
        printf("SKIP: no streaming audio encoder — %s\n", geist_session_errmsg(probe));
        geist_session_destroy(probe);
        free(pcm);
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return GEIST_TEST_SKIP;
    }
    CHECK(s == GEIST_OK, "audio_begin on a fresh session");

    /* Re-opening an already-open turn is a state error, not a second
     * stream_begin that silently resets the first. */
    CHECK(geist_session_audio_begin(probe) == GEIST_E_INVALID_STATE,
          "audio_begin twice must be refused");

    /* ---- destroy mid-turn ----------------------------------------------
     * The case the issue is about: push audio, then destroy without ever
     * calling end. The scratch must be freed (LSan) and the encoder
     * stream must be closed — which the next section proves by opening a
     * new one. */
    CHECK(geist_session_audio_push(probe, N_SAMPLES, pcm) == GEIST_OK, "audio_push mid-turn");
    geist_session_destroy(probe);

    /* ---- the encoder is usable again ------------------------------------
     * If destroy had abandoned the stream instead of aborting it, this
     * turn would inherit the previous session's buffered audio. It runs
     * to a normal end, which is the observable form of "the stream was
     * actually closed". */
    struct geist_session *sess = new_session(model, be);
    CHECK(sess != nullptr, "second session created");
    if (sess != nullptr) {
        CHECK(geist_session_audio_begin(sess) == GEIST_OK, "audio_begin after a destroyed turn");
        CHECK(geist_session_audio_push(sess, N_SAMPLES, pcm) == GEIST_OK, "audio_push");
        /* poll is best-effort: it may legitimately have nothing ready. */
        const enum geist_status ps = geist_session_audio_poll(sess);
        CHECK(ps == GEIST_OK, "audio_poll during a turn");
        CHECK(geist_session_audio_end(sess) == GEIST_OK, "audio_end completes the turn");

        /* After end the turn is closed: the streaming ops must say so
         * rather than touch a freed scratch. */
        CHECK(geist_session_audio_push(sess, N_SAMPLES, pcm) == GEIST_E_INVALID_STATE,
              "audio_push after end must be refused");
        CHECK(geist_session_audio_poll(sess) == GEIST_E_INVALID_STATE,
              "audio_poll after end must be refused");
        CHECK(geist_session_audio_end(sess) == GEIST_E_INVALID_STATE,
              "audio_end twice must be refused");

        /* Destroying a session whose turn already ended must not double
         * free the scratch that end() already released. */
        geist_session_destroy(sess);
    }

    /* ---- destroy with a turn open but nothing pushed --------------------
     * The degenerate unwind: begin, then straight to destroy. */
    struct geist_session *empty = new_session(model, be);
    CHECK(empty != nullptr, "third session created");
    if (empty != nullptr) {
        CHECK(geist_session_audio_begin(empty) == GEIST_OK, "audio_begin on the third session");
        geist_session_destroy(empty);
    }

    /* And the encoder still works after that one too. */
    struct geist_session *last = new_session(model, be);
    CHECK(last != nullptr, "fourth session created");
    if (last != nullptr) {
        CHECK(geist_session_audio_begin(last) == GEIST_OK, "audio_begin after an empty turn");
        CHECK(geist_session_audio_push(last, N_SAMPLES, pcm) == GEIST_OK, "audio_push");
        CHECK(geist_session_audio_end(last) == GEIST_OK, "audio_end");
        geist_session_destroy(last);
    }

    free(pcm);
    geist_model_destroy(model);
    geist_backend_destroy(be);

    if (g_fail) {
        return GEIST_TEST_FAIL;
    }
    printf("PASS: streaming audio turn — destroy mid-turn aborts the encoder stream and "
           "frees its scratch, and the encoder is reusable afterwards\n");
    return GEIST_TEST_PASS;
}

/*
 * test_session_cancel_unit — geist_session_cancel (#628): a prefill stops
 * at a sub-batch boundary when another thread asks it to, keeps what it
 * finished, leaves no logits pending, and goes on to the same logits as a
 * prefill that was never interrupted.
 *
 * On an in-memory llama (model_fixtures.h), for every CPU backend in the
 * build, with a 4-row prefill chunk:
 *   1. a cancel with no call running stops the next prefill before its
 *      first sub-batch (length 0, GEIST_E_CANCELLED) and is consumed: the
 *      prefill after it runs;
 *   2. a cancel from a second thread stops a long prefill: GEIST_E_CANCELLED,
 *      the length is a whole number of sub-batches short of the prompt,
 *      decode_step has nothing pending (GEIST_E_INVALID_STATE);
 *   3. prefilling the rest gives logits bit-identical to an uninterrupted
 *      session (the chunk boundaries stay where they were);
 *   4. a cancel stops a decode_step before it starts.
 * The threaded part is what the TSan leg runs this test for.
 */
#define _POSIX_C_SOURCE 200809L

#include "test_helpers.h"
#include "model_fixtures.h"

#include <geist.h>
#include <geist_util.h>

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

constexpr size_t VOCAB = 256;
constexpr size_t N     = 3000; /* prompt length: 750 sub-batches of 4 */
constexpr size_t CHUNK = 4;

static const char *const BACKENDS[] = {"cpu_x86", "cpu_neon", "cpu_scalar"};

struct canceller {
    struct geist_session *s;
    long                  delay_us;
};

static void *cancel_after(void *arg) {
    const struct canceller *c  = arg;
    struct timespec         ts = {.tv_sec = 0, .tv_nsec = c->delay_us * 1000};
    nanosleep(&ts, nullptr);
    geist_session_cancel(c->s);
    return nullptr;
}

static struct geist_session *open_session(struct geist_model *m, struct geist_backend *be) {
    struct geist_session_opts o = {.max_seq_len = N + 16, .m_max = CHUNK, .top_p = 1.0f};
    struct geist_session     *s = nullptr;
    return geist_session_create(m, be, &o, &s) == GEIST_OK ? s : nullptr;
}

static bool peek(struct geist_session *s, float *dst) {
    size_t       n = 0;
    const float *p = geist_session_peek_logits(&n, s);
    if (p == nullptr || n != VOCAB) {
        return false;
    }
    memcpy(dst, p, VOCAB * sizeof *dst);
    return true;
}

static int run(const char *name, struct geist_model *m, struct geist_backend *be) {
    int            fails = 0;
    geist_token_t *ids   = malloc(N * sizeof *ids);
    float          want[VOCAB], got[VOCAB];
    for (size_t i = 0; i < N; i++) {
        ids[i] = (geist_token_t) (1 + (i * 37) % (VOCAB - 1));
    }
    struct geist_session *ref = open_session(m, be), *s = open_session(m, be);
    if (ids == nullptr || ref == nullptr || s == nullptr) {
        fails += geist_expect(false, "setup");
        goto out;
    }
    fails += geist_expect(geist_session_prefill_tokens(ref, N, ids) == GEIST_OK && peek(ref, want),
                          "uninterrupted reference prefill");

    /* 1. A cancel before the call stops it before its first sub-batch. */
    geist_session_cancel(s);
    fails += geist_expect(geist_session_prefill_tokens(s, N, ids) == GEIST_E_CANCELLED &&
                                  geist_session_length(s) == 0,
                          "a pending cancel stops the next prefill before any sub-batch");
    fails += geist_expect(geist_session_prefill_tokens(s, CHUNK, ids) == GEIST_OK &&
                                  geist_session_length(s) == CHUNK,
                          "the cancel was consumed: the prefill after it runs");
    geist_session_reset(s);

    /* 2. From another thread, mid-prefill. Retry with a shorter delay if
     *    the prefill finished first (the cancel then waits for the next
     *    call, which consumes it here). */
    bool   stopped = false;
    size_t len     = 0;
    for (long delay = 2000; !stopped && delay >= 10; delay /= 4) {
        geist_session_reset(s);
        struct canceller c = {s, delay};
        pthread_t        th;
        if (pthread_create(&th, nullptr, cancel_after, &c) != 0) {
            fails += geist_expect(false, "pthread_create");
            goto out;
        }
        const enum geist_status ps = geist_session_prefill_tokens(s, N, ids);
        pthread_join(th, nullptr);
        if (ps == GEIST_E_CANCELLED) {
            stopped = true;
            len     = geist_session_length(s);
        } else if (ps == GEIST_OK) {
            geist_session_cancel(s); /* make sure no request is left over */
            geist_token_t t;
            (void) geist_session_decode_step(s, &t);
        } else {
            fails += geist_expect(false, "the prefill fails only by being cancelled");
            goto out;
        }
    }
    char msg[160];
    snprintf(msg,
             sizeof msg,
             "%s: a cancel from another thread stops the prefill (at %zu)",
             name,
             len);
    fails += geist_expect(stopped && len < N && len % CHUNK == 0, msg);
    printf("  %-10s cancelled at %zu of %zu positions\n", name, len, N);
    if (stopped) {
        geist_token_t t;
        size_t        nl = 1;
        fails += geist_expect(geist_session_peek_logits(&nl, s) == nullptr && nl == 0 &&
                                      geist_session_decode_step(s, &t) == GEIST_E_INVALID_STATE,
                              "after the cancel no logits are pending");
        /* 3. The rest of the prompt gives the uninterrupted logits. */
        fails += geist_expect(geist_session_prefill_tokens(s, N - len, ids + len) == GEIST_OK &&
                                      geist_session_length(s) == N && peek(s, got) &&
                                      memcmp(want, got, sizeof want) == 0,
                              "prefilling the rest gives the uninterrupted logits, bit for bit");
    }

    /* 4. A decode_step stops before it starts. */
    geist_token_t t1 = -1, t2 = -1;
    geist_session_cancel(ref);
    fails += geist_expect(geist_session_decode_step(ref, &t1) == GEIST_E_CANCELLED && t1 == -1 &&
                                  geist_session_decode_step(ref, &t2) == GEIST_OK,
                          "a cancel stops one decode_step before it starts");

out:
    geist_session_destroy(s);
    geist_session_destroy(ref);
    free(ids);
    return fails;
}

int main(void) {
    struct tf_buf g     = mf_llama_gguf(&(struct mf_llama) {.layers   = 2,
                                                            .d_model  = 64,
                                                            .heads    = 4,
                                                            .kv_heads = 2,
                                                            .ffn      = 128,
                                                            .vocab    = VOCAB,
                                                            .context  = 4096,
                                                            .seed     = 7});
    int           fails = 0, ran = 0;
    for (size_t b = 0; b < sizeof BACKENDS / sizeof BACKENDS[0]; b++) {
        struct geist_backend *be = nullptr;
        if (geist_backend_create(BACKENDS[b], nullptr, nullptr, &be) != GEIST_OK || be == nullptr) {
            continue; /* not in this build */
        }
        struct geist_model *m = nullptr;
        if (geist_model_load_from_memory(g.b, g.n, be, &m) != GEIST_OK) {
            fprintf(stderr, "FAIL: %s: model load: %s\n", BACKENDS[b], geist_last_create_error());
            fails++;
        } else {
            ran++;
            fails += run(BACKENDS[b], m, be);
            geist_model_destroy(m);
        }
        geist_backend_destroy(be);
    }
    free(g.b);
    if (ran == 0) {
        GEIST_SKIP("no CPU backend in this build");
    }
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: geist_session_cancel stops prefill and decode_step cleanly\n");
    return GEIST_TEST_PASS;
}

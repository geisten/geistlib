/*
 * test_session_penalties_int — the repetition penalties (#695) through a
 * real session. The arithmetic itself is pinned against llama.cpp in
 * test_sampler_penalties_unit; this checks the wiring around it:
 *
 *   1. Each greedy token is the argmax of the model's logits (peek_logits)
 *      penalized over the session's own history: the prompt plus every
 *      token decoded so far, last repeat_last_n of them.
 *   2. geist_session_decode_speculative: each verify row sees the history
 *      up to its own position (checked with a ban, see there).
 *   3. truncate and snapshot/restore bring the history back with the
 *      positions: the same tokens follow.
 *   4. Invalid values fail geist_session_create; DRY with the default
 *      breakers, and with none, decodes.
 *
 * SKIPs cleanly if no GGUF model is reachable.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_util.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_DEC 24
#define K_MAX 4

static struct geist_backend *be;
static struct geist_model   *model;
static int                   fails;

static void check(bool cond, const char *what) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", what);
        fails++;
    }
}

static const struct geist_session_opts PEN = {
        .max_seq_len       = 512,
        .repeat_penalty    = 1.5f,
        .repeat_last_n     = 8,
        .frequency_penalty = 0.2f,
        .presence_penalty  = 0.1f,
};

static struct geist_session *
open_session(const struct geist_session_opts *o, size_t n, const geist_token_t *ids) {
    struct geist_session *s = nullptr;
    if (geist_session_create(model, be, o, &s) != GEIST_OK) {
        fprintf(stderr, "session_create: %s\n", geist_last_create_error());
        return nullptr;
    }
    if (geist_session_prefill_tokens(s, n, ids) != GEIST_OK) {
        fprintf(stderr, "prefill: %s\n", geist_session_errmsg(s));
        geist_session_destroy(s);
        return nullptr;
    }
    return s;
}

static bool decode(struct geist_session *s, size_t n, geist_token_t *out) {
    for (size_t i = 0; i < n; i++) {
        if (geist_session_decode_step(s, &out[i]) != GEIST_OK) {
            return false;
        }
    }
    return true;
}

/* llama.cpp's penalties (PEN) on a copy of `l`, argmax of the result
 * (first on ties, as the sampler). */
static geist_token_t
penalized_argmax(size_t n_vocab, const float *l, size_t n_hist, const geist_token_t *hist) {
    const size_t  last_n = (size_t) PEN.repeat_last_n;
    const size_t  from   = n_hist > last_n ? n_hist - last_n : 0;
    geist_token_t best   = 0;
    float         best_l = -INFINITY;
    for (size_t t = 0; t < n_vocab; t++) {
        int count = 0;
        for (size_t i = from; i < n_hist; i++) {
            count += hist[i] == (geist_token_t) t;
        }
        float v = l[t];
        if (count > 0) {
            v = v <= 0 ? v * PEN.repeat_penalty : v / PEN.repeat_penalty;
            v -= (float) count * PEN.frequency_penalty + PEN.presence_penalty;
        }
        if (v > best_l) {
            best_l = v;
            best   = (geist_token_t) t;
        }
    }
    return best;
}

int main(void) {
    GEIST_REQUIRE_GGUF(model_path);
    if (geist_backend_create("auto", nullptr, nullptr, &be) != GEIST_OK ||
        geist_model_load(model_path, be, &model) != GEIST_OK) {
        fprintf(stderr, "setup: %s\n", geist_last_create_error());
        return GEIST_TEST_FAIL;
    }

    /* A prompt that repeats, so the penalties and the drafter both bite. */
    geist_token_t prompt[64];
    size_t        prompt_n = 0;
    {
        struct geist_session *s = nullptr;
        if (geist_session_create(model, be, nullptr, &s) != GEIST_OK) {
            return GEIST_TEST_FAIL;
        }
        const enum geist_status ts = geist_session_tokenize(
                s, "The sea is blue. The sea is blue. The sea is", 64, prompt, &prompt_n);
        geist_session_destroy(s);
        if (ts == GEIST_E_NOT_FOUND || ts == GEIST_E_UNSUPPORTED) {
            GEIST_SKIP_IF(true, "the model's tokenizer cannot encode");
        }
        if (ts != GEIST_OK || prompt_n == 0) {
            return GEIST_TEST_FAIL;
        }
    }

    /* ---- 1. Each token is the penalized argmax over the history. */
    geist_token_t ref[N_DEC];
    {
        struct geist_session *s = open_session(&PEN, prompt_n, prompt);
        if (s == nullptr) {
            return GEIST_TEST_FAIL;
        }
        geist_token_t hist[64 + N_DEC];
        size_t        n_hist = prompt_n;
        memcpy(hist, prompt, prompt_n * sizeof *prompt);
        for (size_t i = 0; i < N_DEC; i++) {
            size_t       nv = 0;
            const float *l  = geist_session_peek_logits(&nv, s);
            check(l != nullptr && nv > 0, "peek_logits");
            const geist_token_t want = l != nullptr ? penalized_argmax(nv, l, n_hist, hist) : -1;
            check(geist_session_decode_step(s, &ref[i]) == GEIST_OK, "decode_step");
            if (ref[i] != want) {
                fprintf(stderr,
                        "FAIL: step %zu: decoded %d, penalized argmax %d\n",
                        i,
                        ref[i],
                        want);
                fails++;
            }
            hist[n_hist++] = ref[i];
        }
        /* The penalties change what greedy decoding emits here at all. */
        struct geist_session *g =
                open_session(&(struct geist_session_opts) {.max_seq_len = 512}, prompt_n, prompt);
        geist_token_t plain[N_DEC];
        check(g != nullptr && decode(g, N_DEC, plain), "plain greedy decode");
        check(memcmp(plain, ref, sizeof ref) != 0, "repeat_penalty 1.5 changes greedy output");
        geist_session_destroy(g);

        /* ---- 3a. truncate back into the prompt: the same tokens follow
         * as in a session that never went further. truncate drops the
         * pending logits, so cut one token more and feed it again (a
         * one-token prefill, which the comparison session does too: its
         * logits need not match the batched prompt's bit for bit). */
        geist_token_t again[N_DEC];
        check(geist_session_truncate(s, prompt_n - 1) == GEIST_OK, "truncate to prompt - 1");
        check(geist_session_prefill_tokens(s, 1, &prompt[prompt_n - 1]) == GEIST_OK, "refeed");
        check(decode(s, N_DEC, again), "decode after truncate");
        geist_session_destroy(s);
        struct geist_session *f = open_session(&PEN, prompt_n - 1, prompt);
        geist_token_t         fresh[N_DEC];
        check(f != nullptr &&
                      geist_session_prefill_tokens(f, 1, &prompt[prompt_n - 1]) == GEIST_OK &&
                      decode(f, N_DEC, fresh),
              "fresh session, same feeding");
        check(memcmp(again, fresh, sizeof fresh) == 0, "truncate restores the history");
        geist_session_destroy(f);
    }

    /* ---- 2. Speculative decoding: each verify row sees the history up to
     * its own position. Checked structurally, not against decode_step's
     * tokens: the batched verify rows round differently from single-token
     * forwards (by up to 2 logits on Gemma 4 E2B Q4_K_M), and penalties
     * turn that into different picks. A presence penalty of 1e4 over a
     * window of 1 bans repeating the previous token, which no rounding
     * undoes, and a prompt of doubled letters makes the model want exactly
     * that while the drafter keeps proposing. A row that missed the drafts
     * before it (history ending at the verify base) repeats; on Qwen3 0.6B
     * and Gemma 4 E2B that mutation fails this check. */
    {
        const struct geist_session_opts ban = {
                .max_seq_len      = 512,
                .repeat_last_n    = 1,
                .presence_penalty = 1e4f,
        };
        geist_token_t pairs[64];
        size_t        pairs_n = 0;
        {
            struct geist_session *t = nullptr;
            check(geist_session_create(model, be, nullptr, &t) == GEIST_OK &&
                          geist_session_tokenize(t,
                                                 "a a b b c c d d e e f f g g h h i i j j k k l l",
                                                 64,
                                                 pairs,
                                                 &pairs_n) == GEIST_OK,
                  "tokenize the doubled letters");
            geist_session_destroy(t);
        }
        struct geist_session *s = open_session(&ban, pairs_n, pairs);
        geist_token_t         hist[64 + 48 + K_MAX + 1];
        size_t                n_hist = pairs_n;
        memcpy(hist, pairs, pairs_n * sizeof *pairs);
        size_t multi = 0;
        while (s != nullptr && n_hist < pairs_n + 48) {
            geist_token_t e[K_MAX + 1];
            size_t        n = 0;
            if (geist_session_decode_speculative(s, K_MAX, n_hist, hist, K_MAX + 1, e, &n) !=
                GEIST_OK) {
                check(false, "decode_speculative");
                break;
            }
            multi += n > 1;
            for (size_t i = 0; i < n; i++) {
                if (e[i] == hist[n_hist - 1]) {
                    fprintf(stderr,
                            "FAIL: speculative token at %zu = %d repeats the one before\n",
                            n_hist,
                            e[i]);
                    fails++;
                }
                hist[n_hist++] = e[i];
            }
        }
        geist_session_destroy(s);
        if (multi == 0) {
            printf("no draft was accepted on this fixture; the verify rows went unchecked\n");
        }
    }

    /* ---- 3b. snapshot / restore carries the history. */
    {
        struct geist_session *a = open_session(&PEN, prompt_n, prompt);
        if (a == nullptr) {
            return GEIST_TEST_FAIL;
        }
        geist_token_t head[N_DEC / 2];
        check(decode(a, N_DEC / 2, head), "decode before snapshot");
        size_t            need = 0;
        enum geist_status ss   = geist_session_snapshot_size(&need, a);
        if (ss == GEIST_OK) {
            void  *img = malloc(need);
            size_t got = 0;
            check(img != nullptr && geist_session_snapshot(&got, need, img, a) == GEIST_OK,
                  "snapshot");
            geist_token_t tail_a[N_DEC / 2];
            check(decode(a, N_DEC / 2, tail_a), "decode after snapshot");

            /* A fresh session that has seen nothing of the prompt. */
            struct geist_session *b = nullptr;
            check(geist_session_create(model, be, &PEN, &b) == GEIST_OK, "second session");
            check(geist_session_restore(got, img, b) == GEIST_OK, "restore");
            geist_token_t tail_b[N_DEC / 2];
            check(decode(b, N_DEC / 2, tail_b), "decode after restore");
            check(memcmp(tail_a, tail_b, sizeof tail_a) == 0, "restore brings the history back");
            check(memcmp(tail_a, ref + N_DEC / 2, sizeof tail_a) == 0, "and matches the run");
            geist_session_destroy(b);
            free(img);
        } else {
            printf("snapshot unsupported here (%s), skipped\n", geist_status_to_string(ss));
        }
        geist_session_destroy(a);
    }

    /* ---- 4. Validation, and DRY with and without breakers. */
    {
        const struct geist_session_opts bad[] = {
                {.repeat_penalty = -1.0f},
                {.repeat_last_n = -5},
                {.dry_multiplier = 0.8f, .dry_base = 0.5f},
                {.presence_penalty = NAN},
        };
        for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++) {
            struct geist_session *s = nullptr;
            check(geist_session_create(model, be, &bad[i], &s) == GEIST_E_INVALID_ARG &&
                          s == nullptr,
                  "invalid opts fail session_create");
        }
        static const char *const        none[1] = {nullptr};
        const struct geist_session_opts dry[]   = {
                {.max_seq_len = 512, .dry_multiplier = 0.8f},
                {.max_seq_len             = 512,
                 .dry_multiplier          = 0.8f,
                 .dry_sequence_breakers   = none,
                 .n_dry_sequence_breakers = 0},
        };
        for (size_t i = 0; i < 2; i++) {
            struct geist_session *s = open_session(&dry[i], prompt_n, prompt);
            geist_token_t         out[N_DEC];
            check(s != nullptr && decode(s, N_DEC, out), "DRY decodes");
            geist_session_destroy(s);
        }
    }

    geist_model_destroy(model);
    geist_backend_destroy(be);
    if (fails == 0) {
        printf("test_session_penalties_int: OK\n");
        return 0;
    }
    fprintf(stderr, "test_session_penalties_int: %d failure(s)\n", fails);
    return GEIST_TEST_FAIL;
}

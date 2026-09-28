/*
 * test_pin_prefix_unit — a pinned prefix is the same cache however the
 * session was used around it: the logits after the prefix and a prompt
 * are those of a fresh session, bit for bit,
 *
 *   - when the prefix is pinned on a session that has held 127 or 200
 *     tokens: one short of a KIVI residual group (R = 128) and past it,
 *     so that under KIVI a group has been packed into the 2-bit cache;
 *   - after turns of 150 decode steps and a reset back to the prefix, for
 *     prefixes of 20, 128 and 130 tokens: under KIVI such a turn packs
 *     the group that holds the prefix's last tokens and reuses their
 *     residual rows, unless the prefix fills whole groups.
 *
 * Every CPU backend in the build, every KV-cache mode, on the two-layer
 * llama from model_fixtures.h.
 */
#include "test_helpers.h"
#include "model_fixtures.h"

#include <geist.h>
#include <geist_util.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

constexpr size_t VOCAB   = 512;
constexpr size_t PROMPT  = 8;
constexpr size_t PIN_LEN = 20;
constexpr size_t PIN_MAX = 130;
constexpr size_t USED    = 200; /* most tokens held before a pin */
constexpr size_t TURN    = 150; /* decode steps in a turn */

static const geist_token_t P1[PROMPT] = {1, 5, 9, 13, 17, 21, 25, 29};

static const struct {
    enum geist_kv_mode kv;
    const char        *name;
} MODES[] = {
        {GEIST_KV_INT8, "INT8"},
        {GEIST_KV_INT4, "INT4"},
        {GEIST_KV_KIVI, "KIVI"},
        {GEIST_KV_FP32, "FP32"},
};

static const char *const BACKENDS[] = {"cpu_x86", "cpu_neon", "cpu_scalar"};

struct run {
    struct geist_model   *m;
    struct geist_backend *be;
    const char           *backend;
    size_t                mode;
};

static geist_token_t PIN[PIN_MAX];
static geist_token_t BEFORE[USED];

static struct geist_session *open_session(const struct run *r) {
    const struct geist_session_opts o = {.kv_mode = MODES[r->mode].kv, .top_p = 1.0f};
    struct geist_session           *s = nullptr;
    if (geist_session_create(r->m, r->be, &o, &s) != GEIST_OK) {
        return nullptr;
    }
    return s;
}

/* A copy of the pending logits into dst[VOCAB]; false if none are pending. */
static bool peek(struct geist_session *s, float dst[static VOCAB]) {
    size_t       n = 0;
    const float *p = geist_session_peek_logits(&n, s);
    if (p == nullptr || n != VOCAB) {
        return false;
    }
    memcpy(dst, p, VOCAB * sizeof *dst);
    return true;
}

static int expect(const struct run *r, bool cond, const char *what) {
    char msg[192];
    snprintf(msg, sizeof msg, "%s KV %s: %s", r->backend, MODES[r->mode].name, what);
    return geist_expect(cond, msg);
}

/* Pin and prefill on a session that held `used` tokens before. */
static bool pin_and_prefill(const struct run *r, size_t used, float out[static VOCAB]) {
    struct geist_session *s  = open_session(r);
    bool                  ok = s != nullptr;
    if (ok && used > 0) {
        ok = geist_session_prefill_tokens(s, used, BEFORE) == GEIST_OK;
    }
    ok = ok && geist_session_pin_prefix(s, PIN_LEN, PIN) == GEIST_OK &&
         geist_session_prefill_tokens(s, PROMPT, P1) == GEIST_OK && peek(s, out);
    geist_session_destroy(s);
    return ok;
}

static int check_pin_after_use(const struct run *r) {
    float fresh[VOCAB];
    float used[VOCAB];
    int   fails = expect(r, pin_and_prefill(r, 0, fresh), "pin and prefill on a fresh session");
    if (fails != 0) {
        return fails;
    }
    static const size_t HELD[] = {127, USED};
    for (size_t i = 0; i < sizeof HELD / sizeof HELD[0]; i++) {
        char what[128];
        snprintf(what,
                 sizeof what,
                 "a pin after %zu tokens gives the logits of a fresh session",
                 HELD[i]);
        const bool ok = pin_and_prefill(r, HELD[i], used);
        fails += expect(r, ok && memcmp(used, fresh, sizeof used) == 0, what);
    }
    return fails;
}

/* Turns after the pin, each decoding TURN tokens and resetting to the
 * prefix: the prompt after each reset sees the fresh session's cache. */
static int check_reset_after_turns(const struct run *r) {
    static const size_t LENS[] = {PIN_LEN, 128, PIN_MAX};
    int                 fails  = 0;
    float               fresh[VOCAB];
    float               again[VOCAB];
    for (size_t i = 0; i < sizeof LENS / sizeof LENS[0]; i++) {
        struct geist_session *s = open_session(r);
        bool ok   = s != nullptr && geist_session_pin_prefix(s, LENS[i], PIN) == GEIST_OK &&
                    geist_session_prefill_tokens(s, PROMPT, P1) == GEIST_OK && peek(s, fresh);
        bool same = true;
        for (int turn = 0; ok && turn < 2; turn++) {
            geist_token_t t = 0;
            for (size_t j = 0; ok && j < TURN; j++) {
                ok = geist_session_decode_step(s, &t) == GEIST_OK;
            }
            ok   = ok && geist_session_reset(s) == GEIST_OK &&
                   geist_session_prefill_tokens(s, PROMPT, P1) == GEIST_OK && peek(s, again);
            same = same && memcmp(again, fresh, sizeof again) == 0;
        }
        geist_session_destroy(s);
        char what[128];
        snprintf(what,
                 sizeof what,
                 "a %zu-token prefix after two turns and resets gives the fresh logits",
                 LENS[i]);
        fails += expect(r, ok && same, what);
    }
    return fails;
}

int main(void) {
    for (size_t i = 0; i < PIN_MAX; i++) {
        PIN[i] = (geist_token_t) (2 + (i * 37) % (VOCAB - 2));
    }
    for (size_t i = 0; i < USED; i++) {
        BEFORE[i] = (geist_token_t) (3 + (i * 53) % (VOCAB - 3));
    }
    struct tf_buf g     = mf_llama_gguf(&(struct mf_llama) {.layers   = 2,
                                                            .d_model  = 128,
                                                            .heads    = 4,
                                                            .kv_heads = 2,
                                                            .ffn      = 256,
                                                            .vocab    = VOCAB,
                                                            .context  = 512,
                                                            .seed     = 14});
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
            for (size_t k = 0; k < sizeof MODES / sizeof MODES[0]; k++) {
                const struct run r = {m, be, BACKENDS[b], k};
                fails += check_pin_after_use(&r);
                fails += check_reset_after_turns(&r);
            }
            geist_model_destroy(m);
        }
        geist_backend_destroy(be);
    }
    free(g.b);
    if (ran == 0 && fails == 0) {
        printf("SKIP: no CPU backend in this build\n");
        return GEIST_TEST_SKIP;
    }
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: a pinned prefix gives the cache of a fresh session, before and after turns\n");
    return GEIST_TEST_PASS;
}

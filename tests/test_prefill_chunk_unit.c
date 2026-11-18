/*
 * test_prefill_chunk_unit - a prefill never ends in a chunk of one row
 * (#554).
 *
 * A chunk of one row is a decode step to every stage of the forward
 * (seq == 1): the M=1 kernels and fusions, which sum in another order than
 * a chunk's M>1 GEMMs. A prompt of m_max + 1 tokens used to split m_max + 1
 * and so gave other logits than the same prompt in one chunk. The chunk
 * before the last now gives it rows (transformer_prefill_chunk): the last
 * has 8 from m_max 16 on, 4 from 8 on, else 2.
 *
 * On an in-memory llama, on every CPU backend in the build:
 *   - the plan, for m_max 1..9, 16, 64, 128 and every prompt length up to
 *     3 * m_max + 2: the chunks add up to the prompt, none exceeds m_max,
 *     there are ceil(n / m_max) of them, a chunk below m_max has at most
 *     8 rows after it, and none has one row unless the prompt has one
 *     token or m_max < 3 (2 + 1 has no other split);
 *   - the forward, for m_max 3, 8, 64 and prompts of m_max + 1 and
 *     2 * m_max + 1 tokens: one prefill call gives, bit for bit, the logits
 *     of the calls prefill(n - t), prefill(t), t the last chunk's rows (2,
 *     4, 8), whose chunks are the plan's by construction; a call of one
 *     token still runs (the decode step).
 *
 * That the M>1 kernels give a row the same bits for every row count, which
 * makes those chunks equal to one chunk, is test_moe_row_invariant_unit's.
 */
#define GEIST_INTERNAL_ARCH_LAYER
#define GEIST_INTERNAL_ENGINE_LAYER

#include "../src/archs/transformer/arch_state.h"
#include "../src/engine/model.h"
#include "model_fixtures.h"
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { VOCAB = 96, CTX = 512, N_MAX = 3 * 128 + 2 };

static const struct mf_llama GEOM = {.layers   = 2,
                                     .d_model  = 64,
                                     .heads    = 4,
                                     .kv_heads = 2,
                                     .ffn      = 128,
                                     .vocab    = VOCAB,
                                     .context  = CTX,
                                     .seed     = 7};

static struct geist_session *
session(struct geist_model *m, struct geist_backend *be, size_t m_max) {
    const struct geist_session_opts o = {.top_p = 1.0f, .m_max = m_max, .max_seq_len = CTX};
    struct geist_session           *s = nullptr;
    return geist_session_create(m, be, &o, &s) == GEIST_OK ? s : nullptr;
}

/* The chunks of an n-token prefill at this session's m_max. */
static int check_plan(const struct transformer_arch_session *a, size_t m_max, size_t n) {
    size_t off = 0, chunks = 0, ones = 0, short_early = 0;
    bool   over = false;
    while (off < n && chunks <= n) {
        const size_t c = transformer_prefill_chunk(a, off, n - off);
        if (c == 0 || c > n - off) {
            break;
        }
        over = over || c > m_max;
        ones += c == 1 ? 1 : 0;
        /* A chunk below m_max with more than a last chunk's rows after it. */
        short_early += c < m_max && n - off - c > 8 ? 1 : 0;
        off += c;
        chunks++;
    }
    const bool one_ok = n == 1 || m_max < 3;
    if (off == n && !over && chunks == (n + m_max - 1) / m_max && short_early == 0 &&
        (ones == 0 || one_ok)) {
        return 0;
    }
    char what[160];
    snprintf(what,
             sizeof what,
             "m_max %zu, %zu tokens: %zu chunks covering %zu, %zu of one row, %zu short early%s",
             m_max,
             n,
             chunks,
             off,
             ones,
             short_early,
             over ? ", one above m_max" : "");
    return geist_expect(false, what);
}

/* The logits after the prompt's first n tokens: one call, or the calls
 * prefill(n - tail), prefill(tail). */
static bool logits_after(struct geist_session *s, size_t n, size_t tail, float out[static VOCAB]) {
    geist_token_t ids[N_MAX];
    for (size_t i = 0; i < n; i++) {
        ids[i] = (geist_token_t) ((i * 37 + 5) % VOCAB);
    }
    if (geist_session_reset(s) != GEIST_OK ||
        geist_session_prefill_tokens(s, n - tail, ids) != GEIST_OK ||
        (tail != 0 && geist_session_prefill_tokens(s, tail, ids + (n - tail)) != GEIST_OK)) {
        return false;
    }
    size_t       nl = 0;
    const float *lg = geist_session_peek_logits(&nl, s);
    if (lg == nullptr || nl != VOCAB) {
        return false;
    }
    memcpy(out, lg, VOCAB * sizeof(float));
    return true;
}

static int check_backend(const char *name, struct geist_backend *be) {
    struct tf_buf       g     = mf_llama_gguf(&GEOM);
    struct geist_model *m     = nullptr;
    int                 fails = 0;
    char                what[160];
    if (geist_model_load_from_memory(g.b, g.n, be, &m) != GEIST_OK) {
        free(g.b);
        snprintf(what, sizeof what, "%s: the fixture loads", name);
        return geist_expect(false, what);
    }
    static const size_t PLAN[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 64, 128};
    for (size_t i = 0; i < sizeof PLAN / sizeof PLAN[0]; i++) {
        struct geist_session *s = session(m, be, PLAN[i]);
        snprintf(what, sizeof what, "%s: a session with m_max %zu", name, PLAN[i]);
        fails += geist_expect(s != nullptr, what);
        if (s == nullptr) {
            continue;
        }
        const struct transformer_arch_session *a = tf_sess(geist_session_internal_arch_session(s));
        for (size_t n = 1; n <= 3 * PLAN[i] + 2 && fails == 0; n++) {
            fails += check_plan(a, PLAN[i], n);
        }
        geist_session_destroy(s);
    }
    static const size_t FWD[] = {3, 8, 64};
    for (size_t i = 0; i < sizeof FWD / sizeof FWD[0]; i++) {
        struct geist_session *s = session(m, be, FWD[i]);
        if (s == nullptr) {
            continue; /* reported above for 3, 8 and 64 */
        }
        for (size_t k = 1; k <= 2; k++) {
            const size_t n    = k * FWD[i] + 1;
            const size_t tail = FWD[i] >= 16 ? 8 : FWD[i] >= 8 ? 4 : 2;
            float        one[VOCAB], split[VOCAB];
            const bool   ok = logits_after(s, n, 0, one) && logits_after(s, n, tail, split);
            snprintf(what,
                     sizeof what,
                     "%s: m_max %zu, %zu tokens in one call are the bits of prefill(%zu), "
                     "prefill(%zu)",
                     name,
                     FWD[i],
                     n,
                     n - tail,
                     tail);
            fails += geist_expect(ok && memcmp(one, split, sizeof one) == 0, what);
        }
        float one[VOCAB];
        snprintf(what, sizeof what, "%s: m_max %zu, a call of one token runs", name, FWD[i]);
        fails += geist_expect(logits_after(s, 1, 0, one), what);
        geist_session_destroy(s);
    }
    geist_model_destroy(m);
    free(g.b);
    return fails;
}

int main(void) {
    static const char *BACKENDS[] = {"cpu_neon", "cpu_x86", "cpu_scalar"};
    int                fails      = 0;
    size_t             ran        = 0;
    for (size_t b = 0; b < sizeof BACKENDS / sizeof BACKENDS[0]; b++) {
        struct geist_backend *be = nullptr;
        if (geist_backend_create(BACKENDS[b], nullptr, nullptr, &be) != GEIST_OK) {
            continue;
        }
        ran++;
        fails += check_backend(BACKENDS[b], be);
        geist_backend_destroy(be);
    }
    if (ran == 0) {
        GEIST_SKIP("no CPU backend compiled in");
    }
    if (fails != 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: no prefill ends in a chunk of one row (%zu backends)\n", ran);
    return GEIST_TEST_PASS;
}

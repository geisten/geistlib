/*
 * test_session_snapshot_unit — geist_session_snapshot / geist_session_restore
 * (#548): a restored session continues exactly as the session the image was
 * taken from, whether the image goes back into that session or into another
 * session of the same model, and a bad image is refused without touching
 * the session.
 *
 * Checked on two models built in memory (model_fixtures.h), a two-layer
 * llama and a Qwen3.5-style hybrid (three gated-DeltaNet blocks and an
 * attention block: a recurrent state beside the KV cache, which pin_prefix
 * cannot rewind), for every backend in the build (GPU ones keep the KV and
 * recurrent state in device memory, so the image goes through download and
 * upload there), every KV-cache mode the snapshot supports, greedy and
 * sampling:
 *
 *   - after a prefill: restored into the same session (after it moved on)
 *     and into a fresh one, the pending logits are bit-identical and the
 *     decode steps give the reference tokens and logits; so does a second
 *     turn (a prefill after the restore) and restoring the image twice;
 *   - mid-decode, with the forward of the last decode step still owed;
 *   - the prefix-reuse case of the issue: prefill a constant prefix once,
 *     snapshot, then restore + prefill a suffix per request — the same as
 *     prefilling prefix and suffix in a session that never snapshotted;
 *   - taking a snapshot leaves the session's own continuation unchanged;
 *   - an empty session's image empties a used session;
 *   - a pinned prefix travels with the image (llama: a reset after the
 *     restore returns to it);
 *   - refusals: a short buffer (INVALID_ARG, out size 0), a truncated or
 *     corrupted image, another model's image, another KV mode, too small a
 *     max_seq_len (all FORMAT, the session untouched), and the KIVI cache
 *     (UNSUPPORTED).
 *
 * Logits are compared bit for bit: a restore is a copy, not a recomputation.
 */
#include "test_helpers.h"
#include "model_fixtures.h"

#include <geist.h>
#include <geist_util.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

constexpr size_t VOCAB  = 512; /* most tokens a model here has */
constexpr size_t PROMPT = 8;
constexpr size_t STEPS  = 6;
constexpr size_t TURN2  = 5;
constexpr size_t CTX    = 64;

static const geist_token_t P1[PROMPT] = {1, 5, 9, 13, 17, 21, 25, 29};
static const geist_token_t P2[TURN2]  = {40, 41, 42, 43, 44};
static const geist_token_t PIN[4]     = {2, 3, 4, 6};

static const struct {
    enum geist_kv_mode kv;
    const char        *name;
} MODES[] = {
        {GEIST_KV_FP32, "FP32"},
        {GEIST_KV_F16, "F16"},
        {GEIST_KV_INT8, "INT8"},
        {GEIST_KV_INT4, "INT4"},
};

static const char *const BACKENDS[] = {"cpu_x86", "cpu_neon", "cpu_scalar", "vulkan", "metal"};

struct run {
    struct geist_model   *m;
    struct geist_model   *other; /* a second model, for the cross-model refusal */
    struct geist_backend *be;
    const char           *backend;
    const char           *model;
    size_t                vocab;
    bool                  recurrent;
    enum geist_kv_mode    kv;
    const char           *kv_name;
    bool                  sampling;
};

/* STEPS decode steps and the logits pending before each of them. */
struct trace {
    geist_token_t tok[STEPS];
    float         logits[STEPS][VOCAB];
};

static struct geist_session *
open_session(const struct run *r, struct geist_model *m, enum geist_kv_mode kv, size_t ctx) {
    struct geist_session_opts o = {.kv_mode = kv, .top_p = 1.0f, .max_seq_len = ctx};
    if (r->sampling) {
        o.temperature = 0.8f;
        o.top_k       = 40;
        o.top_p       = 0.9f;
        o.random_seed = 7;
    }
    struct geist_session *s = nullptr;
    if (geist_session_create(m, r->be, &o, &s) != GEIST_OK) {
        return nullptr;
    }
    return s;
}

static int expect(const struct run *r, bool cond, const char *what) {
    char msg[224];
    snprintf(msg,
             sizeof msg,
             "%s %s KV %s %s: %s",
             r->backend,
             r->model,
             r->kv_name,
             r->sampling ? "sampling" : "greedy",
             what);
    return geist_expect(cond, msg);
}

static bool peek(const struct run *r, struct geist_session *s, float *dst) {
    size_t       n = 0;
    const float *p = geist_session_peek_logits(&n, s);
    if (p == nullptr || n != r->vocab) {
        return false;
    }
    memcpy(dst, p, r->vocab * sizeof *dst);
    return true;
}

/* STEPS decode steps, peeking before each; false if one fails. */
static bool trace(const struct run *r, struct geist_session *s, struct trace *t) {
    for (size_t i = 0; i < STEPS; i++) {
        if (!peek(r, s, t->logits[i]) || geist_session_decode_step(s, &t->tok[i]) != GEIST_OK) {
            return false;
        }
    }
    return true;
}

static bool same_trace(const struct run *r, const struct trace *a, const struct trace *b) {
    for (size_t i = 0; i < STEPS; i++) {
        if (a->tok[i] != b->tok[i] ||
            memcmp(a->logits[i], b->logits[i], r->vocab * sizeof(float)) != 0) {
            return false;
        }
    }
    return true;
}

/* The session's image in a fresh heap buffer; nullptr if the calls fail or
 * disagree on its size. */
static uint8_t *take(struct geist_session *s, size_t *n) {
    size_t need = 0;
    *n          = 0;
    if (geist_session_snapshot_size(&need, s) != GEIST_OK || need == 0) {
        return nullptr;
    }
    uint8_t *buf = malloc(need);
    size_t   got = 0;
    if (buf == nullptr || geist_session_snapshot(&got, need, buf, s) != GEIST_OK || got != need) {
        free(buf);
        return nullptr;
    }
    *n = got;
    return buf;
}

/* Restore after a prefill, into the same session and a fresh one; a second
 * turn; restoring the same image twice. */
static int check_after_prefill(const struct run *r) {
    int                   fails = 0;
    struct geist_session *a     = open_session(r, r->m, r->kv, CTX);
    struct geist_session *ref   = open_session(r, r->m, r->kv, CTX);
    struct geist_session *b     = open_session(r, r->m, r->kv, CTX);
    struct trace         *want  = calloc(1, sizeof *want);
    struct trace         *got   = calloc(1, sizeof *got);
    float                *l1    = malloc(VOCAB * sizeof(float));
    float                *l2    = malloc(VOCAB * sizeof(float));
    uint8_t              *img   = nullptr;
    size_t                n     = 0;
    if (a == nullptr || ref == nullptr || b == nullptr || want == nullptr || got == nullptr ||
        l1 == nullptr || l2 == nullptr) {
        fails += expect(r, false, "setup");
        goto out;
    }
    bool ok = geist_session_prefill_tokens(ref, PROMPT, P1) == GEIST_OK && trace(r, ref, want) &&
              geist_session_prefill_tokens(ref, TURN2, P2) == GEIST_OK && peek(r, ref, l1);
    fails += expect(r, ok, "reference prefill, decode steps and second turn");

    ok  = geist_session_prefill_tokens(a, PROMPT, P1) == GEIST_OK;
    img = ok ? take(a, &n) : nullptr;
    fails += expect(r, img != nullptr, "snapshot after a prefill");
    if (img == nullptr) {
        goto out;
    }
    /* The session the image came from continues as if nothing happened. */
    ok = trace(r, a, got);
    fails += expect(r, ok && same_trace(r, want, got), "taking a snapshot changes nothing");

    /* Back into the same session, which has moved on. */
    ok = geist_session_restore(n, img, a) == GEIST_OK && trace(r, a, got);
    fails += expect(r, ok && same_trace(r, want, got), "restored into its own session");
    ok = geist_session_prefill_tokens(a, TURN2, P2) == GEIST_OK && peek(r, a, l2);
    fails += expect(r,
                    ok && memcmp(l1, l2, r->vocab * sizeof(float)) == 0,
                    "a second turn after the restore");

    /* Into another session of the model: a fork. */
    ok = geist_session_restore(n, img, b) == GEIST_OK && trace(r, b, got);
    fails += expect(r, ok && same_trace(r, want, got), "restored into another session");
    ok = geist_session_restore(n, img, b) == GEIST_OK && trace(r, b, got);
    fails += expect(r, ok && same_trace(r, want, got), "the same image restored twice");

out:
    free(img);
    free(l1);
    free(l2);
    free(want);
    free(got);
    geist_session_destroy(a);
    geist_session_destroy(b);
    geist_session_destroy(ref);
    return fails;
}

/* An image taken between decode steps, while the last step's forward is
 * still owed. */
static int check_mid_decode(const struct run *r) {
    int                   fails = 0;
    struct geist_session *a     = open_session(r, r->m, r->kv, CTX);
    struct geist_session *b     = open_session(r, r->m, r->kv, CTX);
    struct trace         *want  = calloc(1, sizeof *want);
    struct trace         *got   = calloc(1, sizeof *got);
    uint8_t              *img   = nullptr;
    size_t                n     = 0;
    geist_token_t         t     = 0;
    bool                  ok = a != nullptr && b != nullptr && want != nullptr && got != nullptr &&
                               geist_session_prefill_tokens(a, PROMPT, P1) == GEIST_OK;
    for (size_t i = 0; ok && i < 3; i++) {
        ok = geist_session_decode_step(a, &t) == GEIST_OK;
    }
    img = ok ? take(a, &n) : nullptr;
    fails += expect(r, img != nullptr, "snapshot between decode steps");
    if (img != nullptr) {
        ok = trace(r, a, want);
        fails += expect(r, ok, "decode steps after the snapshot");
        ok = geist_session_restore(n, img, b) == GEIST_OK && trace(r, b, got);
        fails += expect(r, ok && same_trace(r, want, got), "restored mid-decode");
    }
    free(img);
    free(want);
    free(got);
    geist_session_destroy(a);
    geist_session_destroy(b);
    return fails;
}

/* The issue's use case: a constant prefix prefilled once, then per request
 * restore + suffix, against prefix + suffix in a session without images.
 * Requests interleave across two sessions. */
static int check_prefix_reuse(const struct run *r) {
    int                   fails = 0;
    struct geist_session *ref   = open_session(r, r->m, r->kv, CTX);
    struct geist_session *a     = open_session(r, r->m, r->kv, CTX);
    struct geist_session *b     = open_session(r, r->m, r->kv, CTX);
    struct trace         *want  = calloc(1, sizeof *want);
    struct trace         *got   = calloc(1, sizeof *got);
    uint8_t              *img   = nullptr;
    size_t                n     = 0;
    bool ok = ref != nullptr && a != nullptr && b != nullptr && want != nullptr && got != nullptr &&
              geist_session_prefill_tokens(ref, PROMPT, P1) == GEIST_OK &&
              geist_session_prefill_tokens(ref, TURN2, P2) == GEIST_OK && trace(r, ref, want);
    fails += expect(r, ok, "reference prefix + suffix");
    ok  = ok && geist_session_prefill_tokens(a, PROMPT, P1) == GEIST_OK;
    img = ok ? take(a, &n) : nullptr;
    fails += expect(r, img != nullptr, "snapshot of the prefix");
    for (size_t req = 0; img != nullptr && req < 3; req++) {
        struct geist_session *s = req % 2 == 0 ? a : b;
        ok = geist_session_restore(n, img, s) == GEIST_OK &&
             geist_session_prefill_tokens(s, TURN2, P2) == GEIST_OK && trace(r, s, got);
        fails += expect(r, ok && same_trace(r, want, got), "restore + suffix per request");
    }
    free(img);
    free(want);
    free(got);
    geist_session_destroy(ref);
    geist_session_destroy(a);
    geist_session_destroy(b);
    return fails;
}

/* An empty session's image; a pinned prefix inside an image. */
static int check_empty_and_pin(const struct run *r) {
    int                   fails = 0;
    struct geist_session *a     = open_session(r, r->m, r->kv, CTX);
    struct geist_session *e     = open_session(r, r->m, r->kv, CTX);
    struct geist_session *b     = open_session(r, r->m, r->kv, CTX);
    float                *l1    = malloc(VOCAB * sizeof(float));
    float                *l2    = malloc(VOCAB * sizeof(float));
    uint8_t              *img   = nullptr;
    size_t                n     = 0;
    geist_token_t         t     = 0;
    if (a == nullptr || e == nullptr || b == nullptr || l1 == nullptr || l2 == nullptr) {
        fails += expect(r, false, "setup");
        goto out;
    }
    img = take(e, &n);
    fails += expect(r, img != nullptr, "snapshot of an empty session");
    bool ok = img != nullptr && geist_session_prefill_tokens(a, PROMPT, P1) == GEIST_OK &&
              geist_session_restore(n, img, a) == GEIST_OK;
    fails += expect(r, ok, "an empty image restores into a used session");
    size_t nl = 1;
    fails += expect(r,
                    geist_session_peek_logits(&nl, a) == nullptr && nl == 0 &&
                            geist_session_decode_step(a, &t) == GEIST_E_INVALID_STATE,
                    "after it nothing is pending");
    ok = geist_session_prefill_tokens(a, TURN2, P2) == GEIST_OK && peek(r, a, l1) &&
         geist_session_prefill_tokens(e, TURN2, P2) == GEIST_OK && peek(r, e, l2);
    fails += expect(r,
                    ok && memcmp(l1, l2, r->vocab * sizeof(float)) == 0,
                    "and a prefill gives a fresh session's logits");
    free(img);
    img = nullptr;

    /* A reset session's image is an empty one: the recurrent state the reset
     * only marked fresh stays out of it. */
    ok  = geist_session_reset(a) == GEIST_OK;
    img = ok ? take(a, &n) : nullptr;
    ok  = img != nullptr && geist_session_prefill_tokens(b, PROMPT, P1) == GEIST_OK &&
          geist_session_restore(n, img, b) == GEIST_OK &&
          geist_session_prefill_tokens(b, TURN2, P2) == GEIST_OK && peek(r, b, l1);
    fails += expect(r,
                    ok && memcmp(l1, l2, r->vocab * sizeof(float)) == 0,
                    "a reset session's image restores as an empty one");
    free(img);
    img = nullptr;

    if (r->recurrent) {
        goto out; /* pin_prefix refuses a prefix on the hybrid */
    }
    /* Pin, extend, image; restored elsewhere, a reset returns to the pin. */
    ok  = geist_session_pin_prefix(a, 4, PIN) == GEIST_OK &&
          geist_session_prefill_tokens(a, PROMPT, P1) == GEIST_OK;
    img = ok ? take(a, &n) : nullptr;
    ok  = img != nullptr && geist_session_restore(n, img, b) == GEIST_OK &&
          geist_session_reset(b) == GEIST_OK &&
          geist_session_prefill_tokens(b, TURN2, P2) == GEIST_OK && peek(r, b, l1) &&
          geist_session_reset(a) == GEIST_OK &&
          geist_session_prefill_tokens(a, TURN2, P2) == GEIST_OK && peek(r, a, l2);
    fails += expect(r,
                    ok && memcmp(l1, l2, r->vocab * sizeof(float)) == 0,
                    "a pinned prefix travels with the image");

out:
    free(img);
    free(l1);
    free(l2);
    geist_session_destroy(a);
    geist_session_destroy(e);
    geist_session_destroy(b);
    return fails;
}

/* Images that do not fit are refused and leave the session as it was. */
static int check_refusals(const struct run *r) {
    int                   fails = 0;
    struct geist_session *a     = open_session(r, r->m, r->kv, CTX);
    struct geist_session *b     = open_session(r, r->m, r->kv, CTX);
    struct geist_session *small = open_session(r, r->m, r->kv, PROMPT - 1);
    struct geist_session *other = open_session(r, r->other, r->kv, CTX);
    struct geist_session *mode =
            open_session(r, r->m, r->kv == GEIST_KV_INT8 ? GEIST_KV_INT4 : GEIST_KV_INT8, CTX);
    float   *l1  = malloc(VOCAB * sizeof(float));
    float   *l2  = malloc(VOCAB * sizeof(float));
    uint8_t *img = nullptr, *bad = nullptr;
    size_t   n = 0;
    if (a == nullptr || b == nullptr || small == nullptr || other == nullptr || mode == nullptr ||
        l1 == nullptr || l2 == nullptr) {
        fails += expect(r, false, "setup");
        goto out;
    }
    bool ok = geist_session_prefill_tokens(a, PROMPT, P1) == GEIST_OK &&
              geist_session_prefill_tokens(b, TURN2, P2) == GEIST_OK && peek(r, b, l1);
    img     = ok ? take(a, &n) : nullptr;
    bad     = img != nullptr ? malloc(n) : nullptr;
    if (bad == nullptr) {
        fails += expect(r, false, "setup: an image to break");
        goto out;
    }

    size_t got = 123;
    fails += expect(r,
                    geist_session_snapshot(&got, n - 1, bad, a) == GEIST_E_INVALID_ARG && got == 0,
                    "a buffer one byte short is refused, out size 0");

    /* Each broken image: FORMAT, and b still shows its own logits. */
    const struct {
        size_t      len;
        size_t      flip; /* SIZE_MAX: no byte flipped */
        const char *what;
    } broken[] = {
            {n - 1, SIZE_MAX, "a truncated image"},
            {7, SIZE_MAX, "a fragment shorter than the header"},
            {n, 0, "a corrupted magic"},
            {n, 8, "another model id"},
    };
    for (size_t i = 0; i < sizeof broken / sizeof broken[0]; i++) {
        memcpy(bad, img, n);
        if (broken[i].flip != SIZE_MAX) {
            bad[broken[i].flip] ^= 0x5a;
        }
        char what[96];
        snprintf(what, sizeof what, "%s is refused, the session untouched", broken[i].what);
        fails += expect(r,
                        geist_session_restore(broken[i].len, bad, b) == GEIST_E_FORMAT &&
                                peek(r, b, l2) && memcmp(l1, l2, r->vocab * sizeof(float)) == 0,
                        what);
    }
    fails += expect(r,
                    geist_session_restore(n, img, other) == GEIST_E_FORMAT,
                    "an image of another model is refused");
    fails += expect(r,
                    geist_session_restore(n, img, mode) == GEIST_E_FORMAT,
                    "an image of another KV mode is refused");
    fails += expect(r,
                    geist_session_restore(n, img, small) == GEIST_E_FORMAT,
                    "an image longer than max_seq_len is refused");
    fails += expect(r,
                    geist_session_restore(n, nullptr, b) == GEIST_E_INVALID_ARG &&
                            geist_session_snapshot_size(nullptr, b) == GEIST_E_INVALID_ARG,
                    "null arguments");

out:
    free(img);
    free(bad);
    free(l1);
    free(l2);
    geist_session_destroy(a);
    geist_session_destroy(b);
    geist_session_destroy(small);
    geist_session_destroy(other);
    geist_session_destroy(mode);
    return fails;
}

static int check_kivi(const struct run *r) {
    struct geist_session *s = open_session(r, r->m, GEIST_KV_KIVI, CTX);
    if (s == nullptr) {
        return expect(r, false, "KIVI session");
    }
    size_t  n       = 9;
    uint8_t buf[64] = {0};
    int     fails =
            expect(r,
                   geist_session_prefill_tokens(s, PROMPT, P1) == GEIST_OK &&
                           geist_session_snapshot_size(&n, s) == GEIST_E_UNSUPPORTED && n == 0 &&
                           geist_session_restore(sizeof buf, buf, s) == GEIST_E_UNSUPPORTED,
                   "the KIVI cache is unsupported");
    geist_session_destroy(s);
    return fails;
}

int main(void) {
    struct tf_vocab v = tf_make_vocab("\xc4\xa0", false); /* the hybrid's vocabulary */
    const struct {
        const char   *name;
        struct tf_buf g;
        size_t        vocab;
        bool          recurrent;
    } models[] = {
            {"llama",
             mf_llama_gguf(&(struct mf_llama) {.layers   = 2,
                                               .d_model  = 128,
                                               .heads    = 4,
                                               .kv_heads = 2,
                                               .ffn      = 256,
                                               .vocab    = VOCAB,
                                               .context  = 256,
                                               .seed     = 14}),
             VOCAB,
             false},
            {"qwen35",
             mf_qwen35_gguf(&(struct mf_qwen35) {.layers     = 4,
                                                 .interval   = 4,
                                                 .d_model    = 64,
                                                 .heads      = 4,
                                                 .kv_heads   = 2,
                                                 .head_dim   = 16,
                                                 .rope_dims  = 8,
                                                 .ffn        = 128,
                                                 .dn_k_heads = 2,
                                                 .dn_v_heads = 4,
                                                 .dn_head_k  = 16,
                                                 .dn_head_v  = 16,
                                                 .dn_conv    = 4,
                                                 .seed       = 1,
                                                 .tok        = &v}),
             v.n_tok,
             true},
    };
    constexpr size_t N_MODELS = sizeof models / sizeof models[0];
    int              fails = 0, ran = 0;
    for (size_t b = 0; b < sizeof BACKENDS / sizeof BACKENDS[0]; b++) {
        struct geist_backend *be = nullptr;
        if (geist_backend_create(BACKENDS[b], nullptr, nullptr, &be) != GEIST_OK || be == nullptr) {
            continue; /* not in this build */
        }
        ran++;
        struct geist_model *m[N_MODELS] = {nullptr};
        for (size_t i = 0; i < N_MODELS; i++) {
            if (geist_model_load_from_memory(models[i].g.b, models[i].g.n, be, &m[i]) != GEIST_OK) {
                fprintf(stderr,
                        "FAIL: %s %s: model load: %s\n",
                        BACKENDS[b],
                        models[i].name,
                        geist_last_create_error());
                fails++;
            }
        }
        for (size_t i = 0; i < N_MODELS; i++) {
            if (m[i] == nullptr || m[(i + 1) % N_MODELS] == nullptr) {
                continue;
            }
            for (size_t k = 0; k < sizeof MODES / sizeof MODES[0]; k++) {
                for (int sampling = 0; sampling < 2; sampling++) {
                    const struct run r = {m[i],
                                          m[(i + 1) % N_MODELS],
                                          be,
                                          BACKENDS[b],
                                          models[i].name,
                                          models[i].vocab,
                                          models[i].recurrent,
                                          MODES[k].kv,
                                          MODES[k].name,
                                          sampling != 0};
                    fails += check_after_prefill(&r);
                    fails += check_mid_decode(&r);
                    fails += check_prefix_reuse(&r);
                    fails += check_empty_and_pin(&r);
                    fails += check_refusals(&r);
                    if (k == 0 && sampling == 0) {
                        fails += check_kivi(&r);
                    }
                }
            }
        }
        for (size_t i = 0; i < N_MODELS; i++) {
            geist_model_destroy(m[i]);
        }
        geist_backend_destroy(be);
    }
    for (size_t i = 0; i < N_MODELS; i++) {
        free(models[i].g.b);
    }
    tf_free_vocab(&v);
    if (ran == 0 && fails == 0) {
        printf("SKIP: no backend in this build\n");
        return GEIST_TEST_SKIP;
    }
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: snapshots restore into the same and other sessions bit-exactly, bad images "
           "are refused\n");
    return GEIST_TEST_PASS;
}

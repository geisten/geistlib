/*
 * test_session_truncate_unit — the calls a chat runtime needs from the
 * engine (#622), on models built in memory (model_fixtures.h), for every
 * backend in the build:
 *
 *   - geist_model_metadata_str / geist_model_context_length: string entries
 *     kept from the GGUF (a chat template with a NUL byte keeps its full
 *     length), absent or non-string keys give nullptr, the trained length;
 *   - geist_session_length counts prefilled and decoded positions;
 *   - geist_session_truncate: prefill A + B, truncate to A, prefill C gives
 *     bit for bit the logits and tokens of prefilling A + C; truncating to
 *     the length keeps the pending logits; refusals past the length and
 *     below a pinned prefix (INVALID_ARG), on the recurrent hybrid for any
 *     n but 0 and the length, and inside the compressed KIVI region
 *     (UNSUPPORTED), each leaving the session as it was;
 *   - geist_session_kv_bytes_per_token against the fixtures' geometry, for
 *     each KV mode, and without the DeltaNet layers on the hybrid.
 */
#include "test_helpers.h"
#include "model_fixtures.h"

#include <geist.h>
#include <geist_util.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

constexpr size_t VOCAB = 512;
constexpr size_t STEPS = 4;

static const geist_token_t A[6] = {1, 5, 9, 13, 17, 21};
static const geist_token_t B[5] = {40, 41, 42, 43, 44};
static const geist_token_t C[4] = {60, 7, 61, 8};

static const char *const BACKENDS[] = {"cpu_x86", "cpu_neon", "cpu_scalar", "vulkan", "metal"};

static const char TEMPLATE[] = "{% for m in messages %}<|im_start|>\0tail";

/* Llama fixture geometry, as built in main. */
constexpr size_t L_LAYERS = 2;
constexpr size_t L_DMODEL = 128;
constexpr size_t L_HEADS  = 4;
constexpr size_t L_KV     = 2;
constexpr size_t L_CTX    = 256;

static struct geist_session *
open_session(struct geist_model *m, struct geist_backend *be, enum geist_kv_mode kv, size_t ctx) {
    struct geist_session_opts o = {.kv_mode = kv, .top_p = 1.0f, .max_seq_len = ctx};
    struct geist_session     *s = nullptr;
    return geist_session_create(m, be, &o, &s) == GEIST_OK ? s : nullptr;
}

/* STEPS greedy steps with the logits pending before each. */
static bool run(struct geist_session *s, size_t vocab, geist_token_t tok[STEPS], float *logits) {
    for (size_t i = 0; i < STEPS; i++) {
        size_t       n = 0;
        const float *p = geist_session_peek_logits(&n, s);
        if (p == nullptr || n != vocab) {
            return false;
        }
        memcpy(logits + i * vocab, p, vocab * sizeof(float));
        if (geist_session_decode_step(s, &tok[i]) != GEIST_OK) {
            return false;
        }
    }
    return true;
}

static int metadata(const char *backend, struct geist_model *m) {
    char        what[160];
    int         fails = 0;
    size_t      len   = 0;
    const char *t     = geist_model_metadata_str(m, "tokenizer.chat_template", &len);
    snprintf(what, sizeof what, "%s: chat template kept with its full length", backend);
    fails += geist_expect(t != nullptr && len == sizeof TEMPLATE - 1 &&
                                  memcmp(t, TEMPLATE, len) == 0 && t[len] == '\0',
                          what);
    snprintf(what, sizeof what, "%s: general.architecture and absent / non-string keys", backend);
    fails += geist_expect(
            !strcmp(geist_model_metadata_str(m, "general.architecture", nullptr), "llama") &&
                    geist_model_metadata_str(m, "no.such.key", &len) == nullptr && len == 0 &&
                    geist_model_metadata_str(m, "llama.block_count", nullptr) == nullptr &&
                    geist_model_metadata_str(m, nullptr, nullptr) == nullptr &&
                    geist_model_metadata_str(nullptr, "general.architecture", nullptr) == nullptr,
            what);
    snprintf(what, sizeof what, "%s: the trained context length", backend);
    fails += geist_expect(geist_model_context_length(m) == L_CTX &&
                                  geist_model_context_length(nullptr) == 0,
                          what);
    return fails;
}

static int truncation(const char *backend, struct geist_model *m, struct geist_backend *be) {
    char                  what[160];
    int                   fails = 0;
    struct geist_session *s     = open_session(m, be, GEIST_KV_FP32, 64);
    struct geist_session *ref   = open_session(m, be, GEIST_KV_FP32, 64);
    geist_token_t         got[STEPS], want[STEPS];
    float                *lg = malloc(STEPS * VOCAB * sizeof(float));
    float                *lw = malloc(STEPS * VOCAB * sizeof(float));
    if (s == nullptr || ref == nullptr || lg == nullptr || lw == nullptr) {
        snprintf(what, sizeof what, "%s: setup", backend);
        fails += geist_expect(false, what);
        goto out;
    }
    geist_token_t ac[10];
    memcpy(ac, A, sizeof A);
    memcpy(ac + 6, C, sizeof C);
    bool ok = geist_session_prefill_tokens(ref, 10, ac) == GEIST_OK && run(ref, VOCAB, want, lw);

    ok = ok && geist_session_prefill_tokens(s, 6, A) == GEIST_OK && geist_session_length(s) == 6 &&
         geist_session_prefill_tokens(s, 5, B) == GEIST_OK && geist_session_length(s) == 11;
    snprintf(what, sizeof what, "%s: length counts prefilled positions", backend);
    fails += geist_expect(ok, what);

    snprintf(what,
             sizeof what,
             "%s: past the length is INVALID_ARG, the session unchanged",
             backend);
    fails += geist_expect(geist_session_truncate(s, 12) == GEIST_E_INVALID_ARG &&
                                  geist_session_length(s) == 11,
                          what);

    ok              = geist_session_truncate(s, 6) == GEIST_OK && geist_session_length(s) == 6;
    geist_token_t t = -1;
    snprintf(what, sizeof what, "%s: after a truncate, decode_step needs a prefill first", backend);
    fails += geist_expect(ok && geist_session_decode_step(s, &t) != GEIST_OK, what);

    ok = geist_session_prefill_tokens(s, 4, C) == GEIST_OK && run(s, VOCAB, got, lg);
    snprintf(what, sizeof what, "%s: A + B, truncate to A, + C equals A + C bit for bit", backend);
    fails += geist_expect(ok && !memcmp(got, want, sizeof got) &&
                                  !memcmp(lg, lw, STEPS * VOCAB * sizeof(float)),
                          what);
    snprintf(what, sizeof what, "%s: decoded positions count", backend);
    fails += geist_expect(geist_session_length(s) == 10 + STEPS, what);

    /* Truncating to the length drops nothing: the pending logits stay. */
    ok = geist_session_truncate(s, geist_session_length(s)) == GEIST_OK &&
         geist_session_decode_step(s, &t) == GEIST_OK;
    snprintf(what, sizeof what, "%s: truncate to the length keeps the pending logits", backend);
    fails += geist_expect(ok, what);

    /* A pinned prefix is a floor. */
    ok = geist_session_pin_prefix(s, 6, A) == GEIST_OK &&
         geist_session_prefill_tokens(s, 5, B) == GEIST_OK;
    snprintf(what, sizeof what, "%s: below a pinned prefix is INVALID_ARG", backend);
    fails += geist_expect(ok && geist_session_truncate(s, 3) == GEIST_E_INVALID_ARG &&
                                  geist_session_length(s) == 11 &&
                                  geist_session_truncate(s, 6) == GEIST_OK &&
                                  geist_session_length(s) == 6,
                          what);
    snprintf(what, sizeof what, "%s: nullptr", backend);
    fails += geist_expect(geist_session_truncate(nullptr, 0) == GEIST_E_INVALID_ARG &&
                                  geist_session_length(nullptr) == 0,
                          what);
out:
    geist_session_destroy(s);
    geist_session_destroy(ref);
    free(lg);
    free(lw);
    return fails;
}

/* KIVI drains groups of 128 positions to 2 bits: those cannot come back. */
static int kivi(const char *backend, struct geist_model *m, struct geist_backend *be) {
    char                  what[160];
    struct geist_session *s = open_session(m, be, GEIST_KV_KIVI, 256);
    if (s == nullptr) {
        return 0; /* KIVI not available on this backend */
    }
    geist_token_t ids[200];
    for (size_t i = 0; i < 200; i++) {
        ids[i] = (geist_token_t) (1 + i % (VOCAB - 2));
    }
    bool ok =
            geist_session_prefill_tokens(s, 200, ids) == GEIST_OK && geist_session_length(s) == 200;
    snprintf(what,
             sizeof what,
             "%s: KIVI: into the compressed region UNSUPPORTED, above it OK",
             backend);
    int fails = geist_expect(ok && geist_session_truncate(s, 100) == GEIST_E_UNSUPPORTED &&
                                     geist_session_length(s) == 200 &&
                                     geist_session_truncate(s, 150) == GEIST_OK &&
                                     geist_session_length(s) == 150 &&
                                     geist_session_prefill_tokens(s, 4, C) == GEIST_OK,
                             what);
    geist_session_destroy(s);
    return fails;
}

static int recurrent(const char *backend, struct geist_model *m, struct geist_backend *be) {
    char                  what[160];
    int                   fails = 0;
    struct geist_session *s     = open_session(m, be, GEIST_KV_FP32, 64);
    bool                  ok = s != nullptr && geist_session_prefill_tokens(s, 6, A) == GEIST_OK &&
                               geist_session_length(s) == 6;
    snprintf(what, sizeof what, "%s: hybrid: a position inside is UNSUPPORTED, unchanged", backend);
    fails += geist_expect(ok && geist_session_truncate(s, 3) == GEIST_E_UNSUPPORTED &&
                                  geist_session_length(s) == 6 &&
                                  geist_session_truncate(s, 6) == GEIST_OK,
                          what);
    snprintf(what, sizeof what, "%s: hybrid: truncate to 0 empties it", backend);
    fails += geist_expect(ok && geist_session_truncate(s, 0) == GEIST_OK &&
                                  geist_session_length(s) == 0 &&
                                  geist_session_prefill_tokens(s, 4, C) == GEIST_OK,
                          what);
    geist_session_destroy(s);
    return fails;
}

static int bytes(const char           *backend,
                 struct geist_model   *llama,
                 struct geist_model   *hybrid,
                 struct geist_backend *be) {
    const size_t hd = L_DMODEL / L_HEADS, row = L_KV * hd;
    const struct {
        enum geist_kv_mode kv;
        const char        *name;
        size_t             want; /* per layer */
    } modes[] = {
            {GEIST_KV_FP32, "FP32", 2 * row * 4},
            {GEIST_KV_F16, "F16", 2 * row * 2},
            {GEIST_KV_INT8, "INT8", 2 * row + 2 * L_KV * 4},
            {GEIST_KV_INT4, "INT4", 2 * (row / 2) + 2 * L_KV * 4},
            {GEIST_KV_KIVI, "KIVI", 2 * (row / 4) + 2 * L_KV * 4 + (2 * row * 4 + 127) / 128},
    };
    char what[160];
    int  fails = 0;
    for (size_t i = 0; i < sizeof modes / sizeof modes[0]; i++) {
        struct geist_session *s = open_session(llama, be, modes[i].kv, 64);
        if (s == nullptr) {
            continue; /* this backend does not offer the mode */
        }
        size_t got = 0;
        /* F16 falls back to FP32 where the backend has no F16 KV append. */
        const bool ok = geist_session_kv_bytes_per_token(s, &got) == GEIST_OK &&
                        (got == L_LAYERS * modes[i].want ||
                         (modes[i].kv == GEIST_KV_F16 && got == L_LAYERS * 2 * row * 4));
        snprintf(what,
                 sizeof what,
                 "%s: KV bytes per position, %s (got %zu)",
                 backend,
                 modes[i].name,
                 got);
        fails += geist_expect(ok, what);
        geist_session_destroy(s);
    }
    /* The hybrid: one attention layer (head_dim 16, 2 KV heads) of four. */
    struct geist_session *s   = open_session(hybrid, be, GEIST_KV_FP32, 64);
    size_t                got = 0;
    snprintf(what, sizeof what, "%s: hybrid: only attention layers count", backend);
    fails += geist_expect(s != nullptr && geist_session_kv_bytes_per_token(s, &got) == GEIST_OK &&
                                  got == 2 * (2 * 16) * 4,
                          what);
    geist_session_destroy(s);
    size_t none = 7;
    fails += geist_expect(geist_session_kv_bytes_per_token(nullptr, &none) == GEIST_E_INVALID_ARG,
                          "nullptr session");
    return fails;
}

int main(void) {
    struct tf_vocab v = tf_make_vocab("\xc4\xa0", false);
    struct tf_buf   llama =
            mf_llama_gguf(&(struct mf_llama) {.layers            = L_LAYERS,
                                              .d_model           = L_DMODEL,
                                              .heads             = L_HEADS,
                                              .kv_heads          = L_KV,
                                              .ffn               = 256,
                                              .vocab             = VOCAB,
                                              .context           = L_CTX,
                                              .seed              = 14,
                                              .chat_template     = TEMPLATE,
                                              .chat_template_len = sizeof TEMPLATE - 1});
    struct tf_buf hybrid = mf_qwen35_gguf(&(struct mf_qwen35) {.layers     = 4,
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
                                                               .tok        = &v});
    int           fails = 0, ran = 0;
    for (size_t b = 0; b < sizeof BACKENDS / sizeof BACKENDS[0]; b++) {
        struct geist_backend *be = nullptr;
        if (geist_backend_create(BACKENDS[b], nullptr, nullptr, &be) != GEIST_OK || be == nullptr) {
            continue; /* not in this build */
        }
        ran++;
        struct geist_model *m = nullptr, *h = nullptr;
        if (geist_model_load_from_memory(llama.b, llama.n, be, &m) != GEIST_OK ||
            geist_model_load_from_memory(hybrid.b, hybrid.n, be, &h) != GEIST_OK) {
            fprintf(stderr, "FAIL: %s: model load: %s\n", BACKENDS[b], geist_last_create_error());
            fails++;
        } else {
            fails += metadata(BACKENDS[b], m);
            fails += truncation(BACKENDS[b], m, be);
            fails += kivi(BACKENDS[b], m, be);
            fails += recurrent(BACKENDS[b], h, be);
            fails += bytes(BACKENDS[b], m, h, be);
        }
        geist_model_destroy(m);
        geist_model_destroy(h);
        geist_backend_destroy(be);
    }
    free(llama.b);
    free(hybrid.b);
    tf_free_vocab(&v);
    if (ran == 0) {
        GEIST_SKIP("no backend in this build");
    }
    if (fails) {
        fprintf(stderr, "test_session_truncate_unit: %d failure(s)\n", fails);
        return 1;
    }
    printf("test_session_truncate_unit: metadata, length, truncate (bit-identical, refusals), KV "
           "bytes "
           "per position on %d backend(s) passed\n",
           ran);
    return 0;
}

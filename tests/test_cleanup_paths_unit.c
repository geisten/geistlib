/*
 * test_cleanup_paths_unit — the error paths #349 moved to GEIST_CLEANUP,
 * walked allocation by allocation (heap_fail_after) on cpu_scalar. ASan
 * reports a leak or a double free on any of them.
 *
 *   - Prompt encoding (encode_upto, encode_spm): set_prompt on a GPT-2 and
 *     a SentencePiece GGUF tokenizer, with the n-th allocation failing for
 *     every n until it succeeds; then the logits equal an undisturbed run.
 *   - AWQ at load (apply_awq_to_state): scales of 1 for all four inputs of
 *     every layer give bit-identical logits, scales of 2 do not; a wrong
 *     length and a missing file are refused; the load is walked the same
 *     way, every failure a clean refusal.
 *   - Audio prefill (transformer_prefill_audio_batch): 20 soft tokens in
 *     sub-batches of 8 and in one give the same logits and advance the KV
 *     by 20; walked the same way, every failure a clean refusal.
 */
#define GEIST_INTERNAL_ARCH_LAYER
#define GEIST_INTERNAL_ENGINE_LAYER

#include "../src/archs/transformer/arch_state.h"
#include "../src/engine/model.h"
#include "heap.h"
#include "model_fixtures.h"
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { D = 64, LAYERS = 2, N_AUDIO = 20, WALK_MAX = 100000 };

static const char PROMPT[] = "the cat sat on the mat, and the dog <|x|> ran far away";

static struct geist_backend *g_be;

/* Logits of the pending token, copied; false when there are none. */
static bool logits(struct geist_session *s, size_t vocab, float *out) {
    size_t       n  = 0;
    const float *lg = geist_session_peek_logits(&n, s);
    if (lg == nullptr || n != vocab)
        return false;
    memcpy(out, lg, vocab * sizeof(float));
    return true;
}

static struct tf_buf llama(const struct tf_vocab *v, const char *tok_model, uint32_t vocab) {
    return mf_llama_gguf(&(struct mf_llama) {.layers     = LAYERS,
                                             .d_model    = D,
                                             .heads      = 4,
                                             .kv_heads   = 4,
                                             .ffn        = 128,
                                             .vocab      = vocab,
                                             .context    = 256,
                                             .seed       = 7,
                                             .tok        = v,
                                             .tok_model  = tok_model,
                                             .tok_merges = v != nullptr});
}

static const struct geist_session_opts SESSION = {.top_p = 1.0f, .m_max = 8, .max_seq_len = 256};

/* ---- prompt encoding ---------------------------------------------------- */

static int check_prompt(const char *name, const char *marker, bool byte_tokens, const char *model) {
    struct tf_vocab v = tf_make_vocab(marker, byte_tokens);
    tf_add_tok(&v, "<|x|>", 5, 0.0f);
    struct tf_buf         g     = llama(&v, model, (uint32_t) v.n_tok);
    struct geist_model   *m     = nullptr;
    struct geist_session *s     = nullptr;
    float                *ref   = calloc(v.n_tok, sizeof(float));
    float                *got   = calloc(v.n_tok, sizeof(float));
    int                   fails = 0;
    if (ref == nullptr || got == nullptr ||
        geist_model_load_from_memory(g.b, g.n, g_be, &m) != GEIST_OK ||
        geist_session_create(m, g_be, &SESSION, &s) != GEIST_OK ||
        geist_session_set_prompt(s, PROMPT) != GEIST_OK || !logits(s, v.n_tok, ref)) {
        fails = geist_expect(false, "prompt fixture");
    } else {
        int64_t n = 0;
        for (; n < WALK_MAX; n++) {
            /* set_prompt appends; each attempt starts a new conversation. */
            if (geist_session_reset(s) != GEIST_OK)
                break;
            heap_fail_after(n);
            const enum geist_status st = geist_session_set_prompt(s, PROMPT);
            heap_fail_after(-1);
            if (st == GEIST_OK)
                break;
        }
        fails += geist_expect(n < WALK_MAX, "set_prompt succeeds once allocations do");
        /* A failed attempt must not leave the session unusable. */
        fails += geist_expect(geist_session_reset(s) == GEIST_OK &&
                                      geist_session_set_prompt(s, PROMPT) == GEIST_OK &&
                                      logits(s, v.n_tok, got) &&
                                      memcmp(ref, got, v.n_tok * sizeof(float)) == 0,
                              "after the failures, the same logits");
        printf("%s: set_prompt walked through %lld failing allocations\n", name, (long long) n);
    }
    geist_session_destroy(s);
    geist_model_destroy(m);
    free(ref);
    free(got);
    free(g.b);
    tf_free_vocab(&v);
    return fails;
}

/* ---- AWQ ---------------------------------------------------------------- */

enum { VOCAB = 96 };

static char awq_path[] = "/tmp/geist_awq_cleanup_XXXXXX";

static void put_u32(FILE *f, uint32_t x) {
    fwrite(&x, 4, 1, f);
}

/* An AWQS file with all four inputs of every layer at `scale`; `bad_len`
 * gives blk.0.attn.out one channel too many. */
static bool write_awq(float scale, bool bad_len) {
    FILE *f = fopen(awq_path, "wb");
    if (f == nullptr)
        return false;
    static const char *const in[4]  = {"attn_norm.out", "ffn_norm.out", "attn.out", "ffn.out"};
    const uint32_t           len[4] = {D, D, D, 128};
    fwrite("AWQS", 1, 4, f);
    put_u32(f, 1);
    put_u32(f, LAYERS * 4);
    for (int l = 0; l < LAYERS; l++)
        for (int k = 0; k < 4; k++) {
            char name[64];
            snprintf(name, sizeof name, "blk.%d.%s", l, in[k]);
            const uint32_t n = len[k] + (bad_len && l == 0 && k == 2 ? 1u : 0u);
            put_u32(f, (uint32_t) strlen(name));
            fwrite(name, 1, strlen(name), f);
            put_u32(f, n);
            for (uint32_t i = 0; i < n; i++)
                fwrite(&scale, 4, 1, f);
        }
    return fclose(f) == 0;
}

/* Loads the fixture with `awq` (nullptr: none) and returns the status; on
 * success `out` gets the logits after a fixed prompt. */
static enum geist_status run_awq(const struct tf_buf *g, const char *awq, float out[static VOCAB]) {
    const struct geist_session_opts lo = {.awq_scales_path = awq};
    struct geist_model             *m  = nullptr;
    struct geist_session           *s  = nullptr;
    enum geist_status st = geist_model_load_from_memory_with_opts(g->b, g->n, g_be, &lo, &m);
    if (st == GEIST_OK)
        st = geist_session_create(m, g_be, &SESSION, &s);
    const geist_token_t ids[5] = {3, 14, 15, 92, 65};
    if (st == GEIST_OK)
        st = geist_session_prefill_tokens(s, 5, ids);
    if (st == GEIST_OK && !logits(s, VOCAB, out))
        st = GEIST_E_BACKEND;
    geist_session_destroy(s);
    geist_model_destroy(m);
    return st;
}

static int check_awq(void) {
    const int fd = mkstemp(awq_path);
    if (fd < 0)
        return geist_expect(false, "mkstemp");
    close(fd);
    struct tf_buf g = llama(nullptr, nullptr, VOCAB);
    float         plain[VOCAB], one[VOCAB], two[VOCAB];
    int           fails = 0;
    fails += geist_expect(run_awq(&g, nullptr, plain) == GEIST_OK, "the model loads without AWQ");
    fails += geist_expect(write_awq(1.0f, false) && run_awq(&g, awq_path, one) == GEIST_OK &&
                                  memcmp(plain, one, sizeof plain) == 0,
                          "AWQ scales of 1 change nothing, bit for bit");
    fails += geist_expect(write_awq(2.0f, false) && run_awq(&g, awq_path, two) == GEIST_OK &&
                                  memcmp(plain, two, sizeof plain) != 0,
                          "AWQ scales of 2 change the logits");
    fails +=
            geist_expect(write_awq(1.0f, true) && run_awq(&g, awq_path, two) == GEIST_E_INVALID_ARG,
                         "a scale vector of the wrong length is refused");

    fails += geist_expect(write_awq(2.0f, false), "AWQ file");
    int64_t n = 0;
    for (; n < WALK_MAX; n++) {
        heap_fail_after(n);
        const enum geist_status st = run_awq(&g, awq_path, two);
        heap_fail_after(-1);
        if (st == GEIST_OK)
            break;
    }
    fails += geist_expect(n < WALK_MAX, "the AWQ load succeeds once allocations do");
    printf("awq: load walked through %lld failing allocations\n", (long long) n);

    unlink(awq_path);
    fails += geist_expect(run_awq(&g, awq_path, two) == GEIST_E_FILE_NOT_FOUND,
                          "a missing AWQ file is refused");
    free(g.b);
    return fails;
}

/* ---- audio prefill ------------------------------------------------------ */

/* Prefills the soft tokens in sub-batches of m_max; KV length after it in
 * *kv, logits in `out` when it succeeded. */
static enum geist_status run_audio(struct geist_model *m,
                                   size_t              m_max,
                                   const float        *soft,
                                   size_t             *kv,
                                   float               out[static VOCAB]) {
    const struct geist_session_opts  so = {.top_p = 1.0f, .m_max = m_max, .max_seq_len = 256};
    struct geist_session            *s  = nullptr;
    enum geist_status                st = geist_session_create(m, g_be, &so, &s);
    struct transformer_arch_session *a =
            st == GEIST_OK ? geist_session_internal_arch_session(s) : nullptr;
    if (a != nullptr)
        st = transformer_prefill_audio_batch(a, N_AUDIO, soft);
    *kv = a != nullptr ? a->kv_len : 0;
    if (st == GEIST_OK && !logits(s, VOCAB, out))
        st = GEIST_E_BACKEND;
    geist_session_destroy(s);
    return st;
}

static int check_audio(void) {
    struct tf_buf       g     = llama(nullptr, nullptr, VOCAB);
    struct geist_model *m     = nullptr;
    int                 fails = 0;
    float               soft[N_AUDIO * D], chunked[VOCAB], whole[VOCAB];
    for (size_t i = 0; i < N_AUDIO * D; i++)
        soft[i] = 0.02f * (float) ((i * 37u) % 51u) - 0.5f;
    size_t kv = 0;
    if (geist_model_load_from_memory(g.b, g.n, g_be, &m) != GEIST_OK) {
        fails = geist_expect(false, "audio fixture");
    } else {
        fails += geist_expect(run_audio(m, 8, soft, &kv, chunked) == GEIST_OK && kv == N_AUDIO,
                              "20 soft tokens in sub-batches of 8 advance the KV by 20");
        fails += geist_expect(run_audio(m, 32, soft, &kv, whole) == GEIST_OK && kv == N_AUDIO,
                              "20 soft tokens in one sub-batch advance the KV by 20");
        float worst = 0.0f;
        for (size_t i = 0; i < VOCAB; i++)
            worst = fmaxf(worst, fabsf(chunked[i] - whole[i]));
        fails += geist_expect(worst <= 1e-4f, "sub-batches and one batch agree");
        printf("audio: sub-batches vs one batch max |d| %.3g\n", (double) worst);

        int64_t n = 0;
        for (; n < WALK_MAX; n++) {
            heap_fail_after(n);
            const enum geist_status st = run_audio(m, 8, soft, &kv, chunked);
            heap_fail_after(-1);
            if (st == GEIST_OK)
                break;
        }
        fails += geist_expect(n < WALK_MAX && kv == N_AUDIO,
                              "audio prefill succeeds once allocations do");
        printf("audio: prefill walked through %lld failing allocations\n", (long long) n);
    }
    geist_model_destroy(m);
    free(g.b);
    return fails;
}

int main(void) {
    if (geist_backend_create("cpu_scalar", nullptr, nullptr, &g_be) != GEIST_OK)
        GEIST_SKIP("cpu_scalar not in this build");
    int fails = 0;
    fails += check_prompt("gpt2", "\xc4\xa0", false, "gpt2");
    fails += check_prompt("spm", "\xe2\x96\x81", true, "llama");
    fails += check_awq();
    fails += check_audio();
    geist_backend_destroy(g_be);
    if (fails == 0)
        printf("PASS: prompt encoding, AWQ load and audio prefill, every allocation failing "
               "once\n");
    return fails ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
}

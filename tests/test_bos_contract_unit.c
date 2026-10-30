/*
 * test_bos_contract_unit — who adds BOS (#263), the same on both tokenizer
 * paths:
 *
 *   - geist_session_tokenize returns the content tokens only: no BOS of its
 *     own, and a BOS the text spells ("<s>") as that one id;
 *   - geist_model_add_bos says the model expects BOS;
 *   - geist_session_set_prompt then starts the sequence with exactly one
 *     BOS: it prepends it unless the prompt's tokens already begin with it,
 *     as a rendered chat template's do.
 *
 * Before #263 the GGUF path prepended BOS in front of a template's own
 * (two BOS), and the tokenizer.bin path never prepended it although
 * add_bos was true.
 *
 * A one-layer llama with F32 weights carries the tokenizer_fixtures vocab,
 * "<s>" its BOS and a special token, either as the GGUF's SentencePiece
 * tokenizer (loaded from memory) or as a tokenizer.bin beside a GGUF
 * without one (loaded from a file). For each prompt the session length
 * after set_prompt must be the content tokens plus one, and the first
 * greedy token must be the one prefill_tokens gives for [BOS, content].
 */
#include "test_helpers.h"
#include "model_fixtures.h"
#include "tokenizer_fixtures.h"

#include <geist.h>
#include <geist_util.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { D = 64, FFN = 64, CAP = 64 };

static const char SPM_MARKER[] = "\xe2\x96\x81"; /* ▁ */

static struct tf_buf model_gguf(const struct tf_vocab *v, bool embedded) {
    return mf_llama_gguf(&(struct mf_llama) {.layers     = 1,
                                             .d_model    = D,
                                             .heads      = 1,
                                             .kv_heads   = 1,
                                             .ffn        = FFN,
                                             .vocab      = (uint32_t) v->n_tok,
                                             .context    = CAP,
                                             .seed       = 7,
                                             .tok        = embedded ? v : nullptr,
                                             .tok_model  = "llama",
                                             .tok_merges = true,
                                             .tok_bos    = true});
}

/* The greedy token after prefilling ids[0, n), on a fresh session. */
static bool first_token_of_ids(struct geist_model   *m,
                               struct geist_backend *be,
                               size_t                n,
                               const geist_token_t  *ids,
                               geist_token_t        *out) {
    struct geist_session *s  = nullptr;
    bool                  ok = geist_session_create(m, be, nullptr, &s) == GEIST_OK &&
                               geist_session_prefill_tokens(s, n, ids) == GEIST_OK &&
                               geist_session_decode_step(s, out) == GEIST_OK;
    geist_session_destroy(s);
    return ok;
}

static int check_model(struct geist_model *m, struct geist_backend *be, const char *kind) {
    int  fails = 0;
    char what[192];

    const geist_token_t bos = geist_model_bos_token(m);
    snprintf(what, sizeof what, "%s: BOS is \"<s>\" (1) and add_bos is true", kind);
    fails += geist_expect(bos == 1 && geist_model_add_bos(m), what);

    struct geist_session *s = nullptr;
    if (geist_session_create(m, be, nullptr, &s) != GEIST_OK) {
        fprintf(stderr, "FAIL: %s: session_create\n", kind);
        return fails + 1;
    }

    /* The content tokens, and tokenize leaves BOS to the caller. */
    static const char CONTENT[] = "the model text";
    geist_token_t     ids[CAP + 1];
    size_t            n = 0;
    if (geist_session_tokenize(s, CONTENT, CAP, ids + 1, &n) != GEIST_OK || n == 0) {
        fprintf(stderr, "FAIL: %s: tokenize\n", kind);
        geist_session_destroy(s);
        return fails + 1;
    }
    snprintf(what, sizeof what, "%s: tokenize adds no BOS", kind);
    fails += geist_expect(ids[1] != bos, what);

    geist_token_t spelled[CAP];
    size_t        n_spelled = 0;
    snprintf(what, sizeof what, "%s: tokenize(\"<s>\" + text) is BOS, then the content", kind);
    fails += geist_expect(
            geist_session_tokenize(s, "<s>the model text", CAP, spelled, &n_spelled) == GEIST_OK &&
                    n_spelled == n + 1 && spelled[0] == bos &&
                    memcmp(spelled + 1, ids + 1, n * sizeof *ids) == 0,
            what);

    ids[0]             = bos;
    geist_token_t want = -1;
    if (!first_token_of_ids(m, be, n + 1, ids, &want)) {
        fprintf(stderr, "FAIL: %s: prefill_tokens\n", kind);
        geist_session_destroy(s);
        return fails + 1;
    }

    static const char *const PROMPTS[] = {"the model text", "<s>the model text"};
    for (size_t p = 0; p < 2; p++) {
        geist_token_t got = -1;
        const bool    ok  = geist_session_reset(s) == GEIST_OK &&
                            geist_session_set_prompt(s, PROMPTS[p]) == GEIST_OK;
        const size_t  len = ok ? geist_session_length(s) : 0;
        snprintf(what,
                 sizeof what,
                 "%s: set_prompt(\"%s\") prefills one BOS and the %zu content tokens (got %zu)",
                 kind,
                 PROMPTS[p],
                 n,
                 len);
        fails += geist_expect(ok && len == n + 1, what);
        snprintf(what,
                 sizeof what,
                 "%s: set_prompt(\"%s\") decodes as prefill_tokens([BOS, content])",
                 kind,
                 PROMPTS[p]);
        fails += geist_expect(ok && geist_session_decode_step(s, &got) == GEIST_OK && got == want,
                              what);
    }
    geist_session_destroy(s);
    return fails;
}

static int run_embedded(struct geist_backend *be) {
    struct tf_vocab     v     = tf_make_vocab(SPM_MARKER, true);
    struct tf_buf       g     = model_gguf(&v, true);
    struct geist_model *m     = nullptr;
    int                 fails = 1;
    if (geist_model_load_from_memory(g.b, g.n, be, &m) != GEIST_OK) {
        fprintf(stderr,
                "FAIL: GGUF tokenizer: model did not load: %s\n",
                geist_last_create_error());
    } else {
        fails = check_model(m, be, "GGUF tokenizer");
        geist_model_destroy(m);
    }
    free(g.b);
    tf_free_vocab(&v);
    return fails;
}

static bool write_file(const char *path, const struct tf_buf *b) {
    FILE *f = fopen(path, "wb");
    if (f == nullptr) {
        return false;
    }
    const bool ok = fwrite(b->b, 1, b->n, f) == b->n;
    return fclose(f) == 0 && ok;
}

static int run_tokenizer_bin(struct geist_backend *be) {
    struct tf_vocab v     = tf_make_vocab(SPM_MARKER, true);
    struct tf_buf   g     = model_gguf(&v, false);
    struct tf_buf   bin   = tf_tokenizer_bin_ex(&v, true);
    char            dir[] = "/tmp/test_bos_contract_XXXXXX";
    char            gpath[64], tpath[64];
    int             fails = 1;
    if (mkdtemp(dir) == nullptr) {
        fprintf(stderr, "FAIL: mkdtemp\n");
    } else {
        snprintf(gpath, sizeof gpath, "%s/model.gguf", dir);
        snprintf(tpath, sizeof tpath, "%s/tokenizer.bin", dir);
        if (!write_file(gpath, &g) || !write_file(tpath, &bin) ||
            setenv("GEIST_TOKENIZER_PATH", tpath, 1) != 0) {
            fprintf(stderr, "FAIL: cannot write the fixture files in %s\n", dir);
        } else {
            struct geist_model *m = nullptr;
            if (geist_model_load(gpath, be, &m) != GEIST_OK) {
                fprintf(stderr,
                        "FAIL: tokenizer.bin: model did not load: %s\n",
                        geist_last_create_error());
            } else {
                fails = check_model(m, be, "tokenizer.bin");
                geist_model_destroy(m);
            }
        }
        unlink(gpath);
        unlink(tpath);
        rmdir(dir);
    }
    free(g.b);
    free(bin.b);
    tf_free_vocab(&v);
    return fails;
}

int main(void) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create("cpu_scalar", nullptr, nullptr, &be) != GEIST_OK || be == nullptr) {
        printf("SKIP: cpu_scalar backend did not register\n");
        return GEIST_TEST_SKIP;
    }
    int fails = run_embedded(be) + run_tokenizer_bin(be);
    geist_backend_destroy(be);
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: set_prompt starts with exactly one BOS on both tokenizer paths\n");
    return GEIST_TEST_PASS;
}

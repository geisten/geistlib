/*
 * test_pooling_select_unit — pooling-kind selection and the embedding
 * API's edges. No model, no backend.
 *
 * geist_pooling_select decides, from one GGUF metadata string, whether a
 * model is generative or an embedding model — and therefore whether the
 * forward pass ends in the LM head or in pooling. Two properties matter
 * and neither is visible at a call site:
 *
 *   1. An absent key means GENERATIVE. Every existing model in the tree
 *      relies on this; a regression that made the default anything else
 *      would silently turn Gemma, Llama, Qwen3 and BitNet b1.58 into
 *      models that emit no tokens.
 *   2. An unrecognised value is REFUSED, not guessed. Pooling the wrong
 *      way yields a vector that looks entirely plausible and means
 *      something else, which no downstream check would catch.
 *
 * No assert(): release builds define NDEBUG. Checks set a flag and the
 * exit code carries PASS/FAIL.
 */
#define GEIST_INTERNAL_ARCH_LAYER /* arch_config.h is layer-internal */

#include "../src/archs/transformer/arch_config.h"

#include <geist_util.h>

#include <stdio.h>

#define S(lit) (sizeof(lit) - 1), (lit)

static int fails = 0;

static void check(const char *what, bool ok) {
    printf("  %s  %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) {
        fails++;
    }
}

static bool selects(size_t len, const char *s, enum geist_pooling_kind want) {
    enum geist_pooling_kind got = (enum geist_pooling_kind) - 1;
    return geist_pooling_select(len, s, &got) && got == want;
}

int main(void) {
    puts("=== pooling selection ===");

    /* The default that every existing model depends on. */
    check("no pooling key -> generative", selects(0, nullptr, GEIST_POOLING_NONE));
    check("empty pooling value -> generative", selects(0, "", GEIST_POOLING_NONE));

    check("\"last_token\" -> last-token pooling",
          selects(S("last_token"), GEIST_POOLING_LAST_TOKEN));
    check("\"mean\" is recognised", selects(S("mean"), GEIST_POOLING_MEAN));

    /* Refusal, not a guess. */
    enum geist_pooling_kind sink = GEIST_POOLING_NONE;
    check("unknown pooling is refused", !geist_pooling_select(S("cls"), &sink));
    check("a near-miss is refused too", !geist_pooling_select(S("last-token"), &sink));

    /* What makes a model an embedding model. */
    check("NONE is not an embedding model", !geist_pooling_is_embedding(GEIST_POOLING_NONE));
    check("LAST_TOKEN is an embedding model", geist_pooling_is_embedding(GEIST_POOLING_LAST_TOKEN));
    check("MEAN is an embedding model", geist_pooling_is_embedding(GEIST_POOLING_MEAN));

    puts("=== embedding API edges ===");

    /* The public accessor must be safe to call before anything exists, and
     * must always write *n_dims so a caller can trust it after a null. */
    size_t n = 12345;
    check("peek_embedding(nullptr) is null and zeroes the dimension",
          geist_session_peek_embedding(nullptr, &n) == nullptr && n == 0);
    check("peek_embedding with no out-param is null",
          geist_session_peek_embedding(nullptr, nullptr) == nullptr);

    if (fails > 0) {
        printf("FAIL: %d check(s) failed\n", fails);
        return 1;
    }
    puts("PASS: pooling is selected, refused and classified correctly");
    return 0;
}

/*
 * test_bitlinear_subln_e2e — the per-projection SubLN layout against a real
 * BitNet-embedding checkpoint.
 *
 * has_bitlinear_subln is the one thing in this family that a synthetic GGUF
 * cannot pin: the flag is detected from the tensor table, and its effect —
 * seven RMSNorms per block instead of two — only becomes observable once
 * the loader reaches the per-layer stage, which needs a complete weight
 * set. So this test wants a real file.
 *
 * Point GEIST_EMBED_GGUF_PATH at
 *
 *   microsoft/bitnet-embedding-0.6b  bitnet-embeddings-0.6b-bf16-i2_s.gguf
 *
 * SKIPs cleanly without the env var — the file is not in CI (428 MB).
 *
 * The sibling 270M checkpoint carries the identical seven *_norm_in weights
 * per block, but on a Gemma 3 backbone, which geist does not implement — the
 * arch gate refuses it.
 *
 * What it asserts is loading and forward SANITY, not embedding quality:
 * every gamma present and correctly shaped, the I2_S projections resolving,
 * and a forward that produces finite activations. Quality needs pooled
 * embeddings compared against microsoft/BitNet's llama-embedding reference,
 * which lands with geist_session_embed.
 */
#define _POSIX_C_SOURCE 200809L

#include "test_helpers.h"

#include <geist.h>
#include <geist_util.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    const char *path = getenv("GEIST_EMBED_GGUF_PATH");
    if (path == nullptr || path[0] == '\0') {
        printf("SKIP: set GEIST_EMBED_GGUF_PATH to a microsoft/bitnet-embedding-* GGUF\n");
        return GEIST_TEST_SKIP;
    }

    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create("cpu_neon", nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        s = geist_backend_create("cpu_scalar", nullptr, nullptr, &be);
    }
    if (s != GEIST_OK) {
        fprintf(stderr, "backend create failed: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }

    int fails = 0;

    struct geist_model *m = nullptr;
    s                     = geist_model_load(path, be, &m);
    if (s != GEIST_OK) {
        fprintf(stderr, "load failed (%d): %s\n", (int) s, geist_backend_errmsg(be));
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }
    const char *arch = geist_model_arch(m);
    printf("  loaded %s (arch=%s)\n", path, arch != nullptr ? arch : "?");
    fails += geist_expect(arch != nullptr && strcmp(arch, "qwen3") == 0,
                          "the BitNet-embedding checkpoint declares its BACKBONE arch");

    struct geist_session           *sess = nullptr;
    const struct geist_session_opts o    = {.max_seq_len = 512};
    s                                    = geist_session_create(m, be, &o, &sess);
    if (s != GEIST_OK) {
        fprintf(stderr, "session create failed: %s\n", geist_model_errmsg(m));
        geist_model_destroy(m);
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    geist_token_t ids[64];
    size_t        n_ids = 0;
    s = geist_session_tokenize(sess, "The capital of France is Paris.", 64, ids, &n_ids);
    fails += geist_expect(s == GEIST_OK && n_ids > 0, "tokenizer rides inside the GGUF");

    if (s == GEIST_OK && n_ids > 0) {
        s = geist_session_prefill_tokens(sess, n_ids, ids);
        fails += geist_expect(s == GEIST_OK, "full forward through the per-projection SubLN stack");
    }

    /* Reaching the head at all means all seven gammas loaded at the right
     * extents — a missing or mis-shaped one fails the load, a skipped one
     * would still be caught here as a non-finite blow-up through 28 layers
     * of unnormalized BitLinear input. */
    size_t       n_logits = 0;
    const float *logits   = geist_session_peek_logits(sess, &n_logits);
    fails += geist_expect(logits != nullptr && n_logits > 0, "forward produced logits");
    if (logits != nullptr) {
        size_t nonfinite = 0;
        float  lo = logits[0], hi = logits[0];
        for (size_t i = 0; i < n_logits; i++) {
            const float x = logits[i];
            if (!isfinite(x)) {
                nonfinite++;
                continue;
            }
            lo = x < lo ? x : lo;
            hi = x > hi ? x : hi;
        }
        printf("  n_logits=%zu range=[%.3f, %.3f] nonfinite=%zu\n", n_logits, lo, hi, nonfinite);
        fails += geist_expect(nonfinite == 0, "every activation is finite");
        fails += geist_expect(hi > lo, "logits are not a constant — the stack actually ran");
    }

    geist_session_destroy(sess);
    geist_model_destroy(m);
    geist_backend_destroy(be);
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("bitlinear subln: load + forward sane\n");
    return GEIST_TEST_PASS;
}

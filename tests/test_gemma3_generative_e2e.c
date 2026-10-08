/*
 * test_gemma3_generative_e2e — a stock Gemma 3 predicts text (#674).
 *
 * google/gemma-3-270m-it scored on the repository's LICENSE (Apache 2.0,
 * its first 4 KiB, ~900 tokens: past the 512-token sliding window, so local and global
 * layers both matter). On the BF16 GGUF, same 843 tokens after BOS:
 *   llama.cpp at the cross-engine pin   17.09
 *   this engine                          17.28
 *   this engine before #674            2983   (every layer global with the
 *     global RoPE base, queries scaled by 1/sqrt(head_dim), V normalized)
 * The bound is llama.cpp + 10 %: room for quantized GGUFs and kernel
 * rounding, none for a wrong layer pattern.
 *
 * GEIST_GEMMA3_GGUF_PATH names the GGUF (convert it with
 * tools/convert_hf.py or llama.cpp's convert_hf_to_gguf.py); skips without.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_util.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

constexpr size_t CAP      = 2048;
constexpr double PPL_HIGH = 17.09 * 1.10;

int main(void) {
    const char *path = getenv("GEIST_GEMMA3_GGUF_PATH");
    if (path == nullptr || path[0] == '\0') {
        GEIST_SKIP("set GEIST_GEMMA3_GGUF_PATH to a gemma-3-270m-it GGUF");
    }
    FILE *f = fopen("LICENSE", "rb");
    if (f == nullptr) {
        GEIST_SKIP("run from the repository root (reads LICENSE)");
    }
    static char text[4096]; /* the first 4 KiB: ~900 tokens */
    text[fread(text, 1, sizeof text - 1, f)] = '\0';
    fclose(f);

    struct geist_backend *be = nullptr;
    if (geist_backend_create("cpu_x86", nullptr, nullptr, &be) != GEIST_OK &&
        geist_backend_create("cpu_neon", nullptr, nullptr, &be) != GEIST_OK &&
        geist_backend_create("cpu_scalar", nullptr, nullptr, &be) != GEIST_OK) {
        fprintf(stderr, "backend create failed: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }
    struct geist_model             *m    = nullptr;
    struct geist_session           *s    = nullptr;
    const struct geist_session_opts opts = {.max_seq_len = CAP + 1, .top_p = 1.0f};
    if (geist_model_load(path, be, &m) != GEIST_OK ||
        geist_session_create(m, be, &opts, &s) != GEIST_OK) {
        fprintf(stderr, "load failed: %s\n", geist_backend_errmsg(be));
        return GEIST_TEST_FAIL;
    }

    static geist_token_t ids[CAP];
    size_t               n = 0;
    if (geist_session_tokenize(s, text, CAP, ids, &n) != GEIST_OK || n < 600) {
        fprintf(stderr, "tokenize: %zu tokens\n", n);
        return GEIST_TEST_FAIL;
    }
    /* Gemma is trained with BOS; without it the perplexity is meaningless. */
    geist_token_t prev = geist_model_bos_token(m);
    double        nll  = 0.0;
    for (size_t i = 0; i < n; i++) {
        size_t       nv = 0;
        const float *lg = nullptr;
        if (geist_session_prefill_tokens(s, 1, &prev) != GEIST_OK ||
            (lg = geist_session_peek_logits(&nv, s)) == nullptr || (size_t) ids[i] >= nv) {
            fprintf(stderr, "scoring failed at token %zu\n", i);
            return GEIST_TEST_FAIL;
        }
        double mx = lg[0], z = 0.0;
        for (size_t v = 1; v < nv; v++) {
            mx = lg[v] > mx ? lg[v] : mx;
        }
        for (size_t v = 0; v < nv; v++) {
            z += exp(lg[v] - mx);
        }
        nll += mx + log(z) - lg[ids[i]];
        prev = ids[i];
    }
    const double ppl = exp(nll / (double) n);
    printf("  %zu tokens, perplexity %.3f (bound %.1f)\n", n, ppl, PPL_HIGH);

    geist_session_destroy(s);
    geist_model_destroy(m);
    geist_backend_destroy(be);
    if (!(ppl < PPL_HIGH)) {
        printf("FAIL: gemma3 perplexity %.1f\n", ppl);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: gemma3 predicts the text (perplexity %.2f)\n", ppl);
    return GEIST_TEST_PASS;
}

/*
 * test_load_from_memory_opts_unit — geist_model_load_from_memory_with_opts
 * carries max_seq_len to the model, like geist_model_load_with_opts (#428).
 *
 * A model's max_seq_len caps every session created on it. The in-memory
 * loader used to pass no options, so an embedded model stayed at the 4096
 * default whatever its consumer needed. On an in-memory llama:
 *   - without options a 6000-token session is refused (cap 4096);
 *   - with opts.max_seq_len = 8192 the same session is created;
 *   - with opts.max_seq_len = 1024 a 2048-token session is refused, so the
 *     option also lowers the cap.
 */
#include "model_fixtures.h"
#include "test_helpers.h"

#include <geist.h>

#include <stdio.h>
#include <stdlib.h>

static bool session_fits(struct geist_backend *be, struct tf_buf g, size_t model_cap, size_t want) {
    const struct geist_session_opts lo = {.max_seq_len = model_cap};
    struct geist_model             *m  = nullptr;
    if (geist_model_load_from_memory_with_opts(g.b, g.n, be, model_cap != 0 ? &lo : nullptr, &m) !=
        GEIST_OK) {
        fprintf(stderr, "  model load: %s\n", geist_last_create_error());
        return false;
    }
    const struct geist_session_opts so = {.max_seq_len = want, .top_p = 1.0f};
    struct geist_session           *s  = nullptr;
    const bool                      ok = geist_session_create(m, be, &so, &s) == GEIST_OK;
    geist_session_destroy(s);
    geist_model_destroy(m);
    return ok;
}

int main(void) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create("cpu_scalar", nullptr, nullptr, &be) != GEIST_OK || be == nullptr) {
        printf("SKIP: cpu_scalar backend not in this build\n");
        return GEIST_TEST_SKIP;
    }
    struct tf_buf g     = mf_llama_gguf(&(struct mf_llama) {.layers   = 1,
                                                            .d_model  = 64,
                                                            .heads    = 2,
                                                            .kv_heads = 1,
                                                            .ffn      = 64,
                                                            .vocab    = 32,
                                                            .context  = 64,
                                                            .seed     = 3});
    int           fails = 0;
    fails += geist_expect(!session_fits(be, g, 0, 6000),
                          "no options: a 6000-token session exceeds the 4096 default cap");
    fails += geist_expect(session_fits(be, g, 8192, 6000),
                          "max_seq_len 8192: a 6000-token session is created");
    fails += geist_expect(!session_fits(be, g, 1024, 2048),
                          "max_seq_len 1024: a 2048-token session is refused");
    fails += geist_expect(session_fits(be, g, 1024, 1024),
                          "max_seq_len 1024: a 1024-token session is created");
    free(g.b);
    geist_backend_destroy(be);
    return fails ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
}

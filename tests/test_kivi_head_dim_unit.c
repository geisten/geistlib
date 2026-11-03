/*
 * test_kivi_head_dim_unit - KIVI needs a head_dim that is a multiple of 4
 * (#391).
 *
 * KIVI packs four channels per byte. With head_dim 6, which RoPE accepts, the
 * packer would read past the row and attention would score uninitialized
 * floats, so a KIVI session is refused by name, explicitly or through the
 * environment. head_dim 8 runs past its first drained group, and the 6 runs
 * under FP32.
 */
#define _POSIX_C_SOURCE 200809L /* setenv, unsetenv */

#include "model_fixtures.h"
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { VOCAB = 64, N_PROMPT = 200 };

static struct tf_buf model_with(uint32_t d_model) {
    return mf_llama_gguf(&(struct mf_llama) {.layers   = 1,
                                             .d_model  = d_model,
                                             .heads    = 4,
                                             .kv_heads = 2,
                                             .ffn      = 32,
                                             .vocab    = VOCAB,
                                             .context  = 512,
                                             .seed     = 3});
}

/* GEIST_OK when a session of this mode takes the whole prompt. */
static enum geist_status run(struct geist_model   *m,
                             struct geist_backend *be,
                             enum geist_kv_mode    mode,
                             char                  err[static 256]) {
    struct geist_session           *s = nullptr;
    const struct geist_session_opts o = {
            .top_p = 1.0f, .m_max = 8, .max_seq_len = 256, .kv_mode = mode};
    enum geist_status st = geist_session_create(m, be, &o, &s);
    snprintf(err, 256, "%s", st == GEIST_OK ? "" : geist_backend_errmsg(be));
    geist_token_t ids[N_PROMPT];
    for (size_t i = 0; i < N_PROMPT; i++)
        ids[i] = (geist_token_t) ((i * 37 + 5) % VOCAB);
    for (size_t i = 0; st == GEIST_OK && i < N_PROMPT; i += 8)
        st = geist_session_prefill_tokens(s, 8, ids + i);
    geist_session_destroy(s);
    return st;
}

int main(void) {
    unsetenv("GEIST_KV_KIVI");
    struct geist_backend *be = nullptr;
    if (geist_backend_create("cpu_scalar", nullptr, nullptr, &be) != GEIST_OK)
        return geist_expect(false, "cpu_scalar backend");
    struct tf_buf       six = model_with(24), eight = model_with(32);
    struct geist_model *m6 = nullptr, *m8 = nullptr;
    int                 fails = 0;
    char                err[256];
    if (geist_model_load_from_memory(six.b, six.n, be, &m6) != GEIST_OK ||
        geist_model_load_from_memory(eight.b, eight.n, be, &m8) != GEIST_OK) {
        fails += geist_expect(false, "the fixtures load");
    } else {
        fails += geist_expect(run(m6, be, GEIST_KV_KIVI, err) != GEIST_OK,
                              "KIVI on head_dim 6 is refused");
        fails += geist_expect(strstr(err, "KIVI") != nullptr, "the refusal names KIVI");
        setenv("GEIST_KV_KIVI", "1", 1);
        fails += geist_expect(run(m6, be, GEIST_KV_AUTO, err) != GEIST_OK,
                              "so is KIVI from the environment");
        unsetenv("GEIST_KV_KIVI");
        fails += geist_expect(run(m6, be, GEIST_KV_FP32, err) == GEIST_OK, "FP32 runs head_dim 6");
        fails += geist_expect(run(m8, be, GEIST_KV_KIVI, err) == GEIST_OK, "KIVI runs head_dim 8");
    }
    geist_model_destroy(m6);
    geist_model_destroy(m8);
    geist_backend_destroy(be);
    free(six.b);
    free(eight.b);
    return fails ? 1 : 0;
}

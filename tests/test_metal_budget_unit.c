/*
 * test_metal_budget_unit — #531: Metal refuses an allocation past its GPU
 * working-set budget, so a model or a session that does not fit fails at
 * load or at session create, with a message that names the numbers and the
 * sysctl knob, instead of failing or paging in the middle of a generation.
 * GEIST_METAL_IGNORE_BUDGET=1 lifts the check.
 *
 * The budget is forced small by writing the backend state directly; there
 * is no test-only knob. SKIPs when the Metal backend is not built in or the
 * host has no Metal device.
 */
#include "test_helpers.h"
#include "model_fixtures.h"

#include <geist.h>
#include <geist_util.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef GEIST_BACKEND_METAL
#include "../src/backends/metal/metal_internal.h"

static int g_fail = 0;
#define check(ok, what) (g_fail |= geist_expect((ok), (what)))

static size_t allocated(struct geist_backend *be) {
    struct metal_state *st = be->state;
    return metal_msg_send_ulong0(st, st->device, "currentAllocatedSize");
}

/* The budget as headroom over what the device holds now. */
static void squeeze(struct geist_backend *be, size_t headroom) {
    ((struct metal_state *) be->state)->ws_budget = allocated(be) + headroom;
}
#endif

int main(void) {
#ifndef GEIST_BACKEND_METAL
    return GEIST_TEST_SKIP;
#else
    struct geist_backend   *be = nullptr;
    const enum geist_status bs = geist_backend_create("metal", nullptr, nullptr, &be);
    if (bs == GEIST_E_UNSUPPORTED || bs == GEIST_E_NOT_FOUND) {
        fprintf(stderr, "SKIP: metal backend unavailable (not built or no device)\n");
        return GEIST_TEST_SKIP;
    }
    if (bs != GEIST_OK) {
        return GEIST_TEST_ERROR;
    }
    char          path[] = "/tmp/geist_metal_budget_XXXXXX";
    const int     fd     = mkstemp(path);
    struct tf_buf g      = mf_llama_gguf(&(struct mf_llama) {.layers   = 2,
                                                             .d_model  = 128,
                                                             .heads    = 4,
                                                             .kv_heads = 2,
                                                             .ffn      = 256,
                                                             .vocab    = 64,
                                                             .context  = 256,
                                                             .seed     = 7});
    const bool    ok     = fd >= 0 && write(fd, g.b, g.n) == (ssize_t) g.n;
    if (fd >= 0) {
        close(fd);
    }
    free(g.b);
    if (!ok) {
        return GEIST_TEST_ERROR;
    }
    struct metal_state *st   = be->state;
    const size_t        real = st->ws_budget;
    check(real > 0, "budget read from the device");

    /* 1. A model past the budget fails to load, and says why. */
    struct geist_model *m = nullptr;
    squeeze(be, 64u << 10);
    check(geist_model_load(path, be, &m) == GEIST_E_OOM, "load over budget: GEIST_E_OOM");
    const char *msg = geist_backend_errmsg(be);
    check(msg != nullptr && strstr(msg, "MiB free") != nullptr &&
                  strstr(msg, "iogpu.wired_limit_mb") != nullptr,
          "load over budget: message names the numbers and the knob");
    if (m != nullptr) {
        geist_model_destroy(m);
        m = nullptr;
    }

    /* 2. GEIST_METAL_IGNORE_BUDGET=1 lifts the check. */
    setenv("GEIST_METAL_IGNORE_BUDGET", "1", 1);
    struct geist_backend *lax = nullptr;
    check(geist_backend_create("metal", nullptr, nullptr, &lax) == GEIST_OK, "second backend");
    if (lax != nullptr) {
        struct geist_model *lm = nullptr;
        squeeze(lax, 64u << 10);
        check(geist_model_load(path, lax, &lm) == GEIST_OK, "GEIST_METAL_IGNORE_BUDGET=1 loads");
        geist_model_destroy(lm);
        geist_backend_destroy(lax);
    }
    unsetenv("GEIST_METAL_IGNORE_BUDGET");

    /* 3. A session that does not fit fails at create, not mid-decode; with
     *    the real budget the same session runs. */
    st->ws_budget = real;
    check(geist_model_load(path, be, &m) == GEIST_OK, "load within the real budget");
    struct geist_session     *s  = nullptr;
    struct geist_session_opts so = {.max_seq_len = 256};
    if (m != nullptr) {
        squeeze(be, 4u << 10);
        check(geist_session_create(m, be, &so, &s) != GEIST_OK && s == nullptr,
              "session over budget fails at create");
        st->ws_budget = real;
        check(geist_session_create(m, be, &so, &s) == GEIST_OK, "session within the budget");
        const geist_token_t ids[4] = {1, 2, 3, 4};
        geist_token_t       t      = 0;
        check(s != nullptr && geist_session_prefill_tokens(s, 4, ids) == GEIST_OK &&
                      geist_session_decode_step(s, &t) == GEIST_OK,
              "the session decodes");
    }
    geist_session_destroy(s);
    geist_model_destroy(m);
    geist_backend_destroy(be);
    unlink(path);
    if (g_fail == 0) {
        printf("PASS: metal working-set budget refuses at load and session create\n");
    }
    return g_fail == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
#endif
}

/*
 * test_session_backend_mismatch_unit — geist_session_create's backend
 * parameter is a validated witness.
 *
 * The STABLE signature keeps the backend parameter, but a session on a
 * different backend instance than the model's would mix buffer
 * ownership and per-backend workspaces. Since the strict check landed,
 * passing any backend other than the one the model was loaded on
 * (pointer identity — two instances of the same backend type are
 * different backends) fails with GEIST_E_INVALID_ARG.
 *
 * SKIPs cleanly if no GGUF fixture is reachable.
 */
#include "test_helpers.h"

#include <geist.h>

#include <stdio.h>
#include <stdlib.h>

static const char *resolve_path(void) {
    const char *env = getenv("GEIST_GGUF_PATH");
    if (env != nullptr && env[0] != '\0')
        return env;
    static const char *candidates[] = {
            "gguf_artifacts/bitnet_b1_58-large-TQ2_0.gguf",
            "gguf_artifacts/gemma4-e2b-Q4_K_M.gguf",
            nullptr,
    };
    for (size_t i = 0; candidates[i] != nullptr; i++) {
        FILE *f = fopen(candidates[i], "rb");
        if (f != nullptr) {
            fclose(f);
            return candidates[i];
        }
    }
    return nullptr;
}

int main(void) {
    const char *path = resolve_path();
    if (path == nullptr) {
        printf("SKIP: no GGUF fixture found (set GEIST_GGUF_PATH)\n");
        return GEIST_TEST_SKIP;
    }

    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create("cpu_neon", nullptr, nullptr, &be);
    if (s != GEIST_OK)
        s = geist_backend_create("cpu_scalar", nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        fprintf(stderr, "backend create failed: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }
    /* A second instance of the SAME backend type — still a different
     * backend (own state, own workspaces). */
    struct geist_backend *other = nullptr;
    if (geist_backend_create(geist_backend_name(be), nullptr, nullptr, &other) != GEIST_OK) {
        fprintf(stderr, "second backend create failed\n");
        geist_backend_destroy(be);
        return GEIST_TEST_ERROR;
    }

    struct geist_model *model = nullptr;
    s                         = geist_model_load(path, be, &model);
    if (s != GEIST_OK) {
        fprintf(stderr, "model load failed: %s\n", geist_last_create_error());
        geist_backend_destroy(other);
        geist_backend_destroy(be);
        return GEIST_TEST_ERROR;
    }

    int fails = 0;

    struct geist_session *sess = nullptr;
    s                          = geist_session_create(model, other, nullptr, &sess);
    fails += geist_expect(s == GEIST_E_INVALID_ARG,
                          "mismatched backend instance -> GEIST_E_INVALID_ARG");
    fails += geist_expect(sess == nullptr, "mismatched backend leaves *out null");

    s = geist_session_create(model, be, nullptr, &sess);
    fails += geist_expect(s == GEIST_OK, "the model's own backend -> GEIST_OK");
    if (sess != nullptr) {
        geist_session_destroy(sess);
    }

    geist_model_destroy(model);
    geist_backend_destroy(other);
    geist_backend_destroy(be);

    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("session backend witness: mismatch rejected, match accepted\n");
    return GEIST_TEST_PASS;
}

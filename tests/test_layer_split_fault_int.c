/*
 * test_layer_split_fault_int — failures on either side of a layer split
 * (#49) come back as a status, nothing crashes and nothing leaks.
 *
 * A split model (SmolLM2-360M, layers 0..15 on a second CPU backend) runs a
 * prefill while one backend's buffer_map, buffer_upload and buffer_download
 * fail after K successful calls. Sweeping K moves the failure through the
 * whole step: the hidden-row transfer at the boundary in both directions,
 * the scratch and KV mappings inside a layer, the head, until K passes the
 * step's last call and the prefill completes. Every prefill must return
 * GEIST_OK or an error status, the session and the model must tear down
 * afterwards, and on each side some K must fail and the sweep must end in
 * a completed prefill. Run under ASan / UBSan (make MODE=asan) for the "no sanitizer
 * report" half.
 *
 * Model: GEIST_SPLIT_GGUF_PATH, else SmolLM2-360M (make fetch-llama-model).
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const struct geist_backend_vtbl *real;
static struct geist_backend_vtbl        faulty;
static int                              budget;

static void *map_or_fail(struct geist_buffer *b) {
    return budget-- > 0 ? real->buffer_map(b) : nullptr;
}

static enum geist_status
upload_or_fail(struct geist_buffer *dst, size_t n, const uint8_t src[GEIST_AT_LEAST(n)]) {
    return budget-- > 0 ? real->buffer_upload(dst, n, src) : GEIST_E_BACKEND;
}

static enum geist_status
download_or_fail(size_t n, uint8_t dst[GEIST_AT_LEAST(n)], const struct geist_buffer *src) {
    return budget-- > 0 ? real->buffer_download(n, dst, src) : GEIST_E_BACKEND;
}

static const char *resolve_path(void) {
    const char *env = getenv("GEIST_SPLIT_GGUF_PATH");
    if (env != nullptr && env[0] != '\0') {
        return env;
    }
    FILE *f = fopen("gguf_artifacts/smollm2-360m-instruct-q8_0.gguf", "rb");
    if (f == nullptr) {
        return nullptr;
    }
    fclose(f);
    return "gguf_artifacts/smollm2-360m-instruct-q8_0.gguf";
}

static struct geist_backend *cpu_backend(void) {
    static const char *const names[] = {"cpu_neon", "cpu_x86", "cpu_scalar"};
    for (size_t i = 0; i < sizeof names / sizeof names[0]; i++) {
        struct geist_backend *be = nullptr;
        if (geist_backend_create(names[i], nullptr, nullptr, &be) == GEIST_OK) {
            return be;
        }
    }
    return nullptr;
}

/* One prefill on a fresh split model with `victim`'s transfers failing after
 * k calls. Returns the prefill's status; GEIST_E_INTERNAL when the setup
 * itself fails. */
static enum geist_status prefill_with_fault(const char           *path,
                                            struct geist_backend *be,
                                            struct geist_backend *sbe,
                                            struct geist_backend *victim,
                                            int                   k) {
    struct geist_session_opts lo = {.split_backend = sbe, .split_layers = 16};
    struct geist_model       *m  = nullptr;
    if (geist_model_load_with_opts(path, be, &lo, &m) != GEIST_OK) {
        return GEIST_E_INTERNAL;
    }
    struct geist_session *sess = nullptr;
    if (geist_session_create(m, be, nullptr, &sess) != GEIST_OK) {
        geist_model_destroy(m);
        return GEIST_E_INTERNAL;
    }
    geist_token_t ids[100];
    for (size_t i = 0; i < 100; i++) {
        ids[i] = (geist_token_t) (100 + (i * 37) % 1000);
    }

    const struct geist_backend_descriptor *orig = victim->desc;
    struct geist_backend_descriptor        fd   = *orig;
    real                                        = orig->vtbl;
    faulty                                      = *real;
    faulty.buffer_map                           = map_or_fail;
    if (real->buffer_upload != nullptr) {
        faulty.buffer_upload = upload_or_fail;
    }
    if (real->buffer_download != nullptr) {
        faulty.buffer_download = download_or_fail;
    }
    fd.vtbl                   = &faulty;
    budget                    = k;
    victim->desc              = &fd;
    const enum geist_status s = geist_session_prefill_tokens(sess, 100, ids);
    victim->desc              = orig; /* teardown goes through the real table */

    geist_session_destroy(sess);
    geist_model_destroy(m);
    return s;
}

int main(void) {
    const char *path = resolve_path();
    if (path == nullptr) {
        GEIST_SKIP_FIXTURE("no GGUF for the layer split. Run `make fetch-llama-model`, or set "
                           "GEIST_SPLIT_GGUF_PATH");
    }
    struct geist_backend *be  = cpu_backend();
    struct geist_backend *sbe = cpu_backend();
    if (be == nullptr || sbe == nullptr) {
        fprintf(stderr, "backend create failed: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }

    int                   fails     = 0;
    struct geist_backend *victims[] = {sbe, be};
    const char           *names[]   = {"split stage", "load backend"};
    for (size_t v = 0; v < 2; v++) {
        int n_err = 0, n_ok = 0;
        for (int k = 0; k <= 1000000 && n_ok < 2; k = k < 16 ? k + 1 : k * 3 / 2) {
            const enum geist_status s = prefill_with_fault(path, be, sbe, victims[v], k);
            if (s == GEIST_E_INTERNAL) {
                fprintf(stderr, "FAIL: setup failed (%s, k=%d)\n", names[v], k);
                fails++;
                break;
            }
            s == GEIST_OK ? n_ok++ : n_err++;
        }
        printf("%s: %d faulted prefills returned an error status, %d completed\n",
               names[v],
               n_err,
               n_ok);
        fails += geist_expect(n_err > 0, "a fault on this side surfaces as an error status");
        fails += geist_expect(n_ok > 0, "the sweep reaches past the step's last transfer");
    }

    geist_backend_destroy(sbe);
    geist_backend_destroy(be);
    return fails ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
}

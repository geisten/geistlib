/*
 * test_layer_split_int — a model split by layer across two backends (#49).
 *
 * Loads one GGUF unsplit and split at several points (the first N layers
 * on a second backend, the rest and the head on the first), prefills the
 * same token ids across more than one prefill chunk, decodes greedy
 * tokens, resets and prefills again, and compares:
 *
 *   - both backends of the same kind (default: two instances of the host's
 *     CPU backend, cpu_neon / cpu_x86, else cpu_scalar): the logits after the prefill, the decoded
 * tokens and the logits after the reset are BYTE-identical to the unsplit run, at every split point
 * 1, L/2, L-1 and L, in the FP32, INT8 and INT4 KV modes. A split only moves where a layer runs;
 * the kernels are the same, so any difference is a bug in the split.
 *   - GEIST_SPLIT_BACKEND names another backend kind (metal, vulkan) for
 *     the first N layers: the outputs are reported (max |d| of the logits,
 *     matching greedy tokens), not required to match — GPU kernels are not
 *     bit-exact with the CPU by design. The baseline is the same model
 *     entirely on that backend ("gpu only"): a split should not stray
 *     further from the CPU than that.
 *
 * GEIST_SPLIT_POINTS ("1,20,28") replaces the split points, e.g. for a
 * model whose every layer does not fit the device.
 *
 * Offload (#78): with both backends of the same kind, the split at L/2
 * runs again with GEIST_OFFLOAD_M_MIN=8 (prompt chunks on the twins, decode
 * on the load backend): byte-identical with an FP32 cache, reported with
 * INT8. With another backend kind, GEIST_OFFLOAD_M_MIN set by the caller
 * applies to every split above and the outputs are reported.
 *
 * Refusals: split_layers past the layer count fails the load with
 * GEIST_E_INVALID_ARG; a KIVI session on a split model fails to create.
 *
 * Model: GEIST_SPLIT_GGUF_PATH, else SmolLM2-360M (make fetch-llama-model).
 * A model a split does not support (per-layer embeddings, shared KV,
 * DeltaNet, ...) must be refused with GEIST_E_UNSUPPORTED; the test then
 * checks that and skips the rest.
 */
#include "test_helpers.h"

#define GEIST_INTERNAL_ENGINE_LAYER
#include "src/io/gguf_reader.h"

#include <geist.h>
#include <geist_util.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { N_PROMPT = 100, N_DECODE = 16, N_AFTER_RESET = 10 };

struct run {
    float        *logits; /* after the prefill */
    float        *logits_reset;
    size_t        n_logits;
    geist_token_t tokens[N_DECODE];
};

static const char *resolve_path(void) {
    const char *env = getenv("GEIST_SPLIT_GGUF_PATH");
    if (env != nullptr && env[0] != '\0') {
        return env;
    }
    static const char *candidates[] = {
            "gguf_artifacts/smollm2-360m-instruct-q8_0.gguf",
            "./smollm2-360m-instruct-q8_0.gguf",
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

static size_t layer_count(const char *path) {
    const char      *err = nullptr;
    struct gguf_ctx *g   = gguf_open(path, &err);
    if (g == nullptr) {
        return 0;
    }
    size_t      n    = 0;
    size_t      len  = 0;
    const char *arch = gguf_get_meta_string(g, "general.architecture", &len);
    char        key[128];
    uint32_t    u = 0;
    if (arch != nullptr && len < 100) {
        snprintf(key, sizeof key, "%.*s.block_count", (int) len, arch);
        if (gguf_get_meta_u32(g, key, &u)) {
            n = u;
        }
    }
    gguf_close(g);
    return n;
}

static struct geist_backend *backend(const char *name) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create(name, nullptr, nullptr, &be) != GEIST_OK) {
        return nullptr;
    }
    return be;
}

static void run_free(struct run *r) {
    free(r->logits);
    free(r->logits_reset);
    *r = (struct run) {0};
}

static float *copy_logits(struct geist_session *s, size_t *n) {
    const float *l = geist_session_peek_logits(n, s);
    if (l == nullptr || *n == 0) {
        return nullptr;
    }
    float *c = malloc(*n * sizeof(float));
    if (c != nullptr) {
        memcpy(c, l, *n * sizeof(float));
    }
    return c;
}

/* Load (split when split_be != nullptr), prefill, decode, reset, prefill. */
static enum geist_status run_once(const char           *path,
                                  struct geist_backend *be,
                                  struct geist_backend *split_be,
                                  size_t                split_layers,
                                  enum geist_kv_mode    kv,
                                  struct run           *out) {
    *out                         = (struct run) {0};
    struct geist_session_opts lo = {0};
    lo.split_backend             = split_be;
    lo.split_layers              = split_layers;
    struct geist_model *m        = nullptr;
    enum geist_status   s        = geist_model_load_with_opts(path, be, &lo, &m);
    if (s != GEIST_OK) {
        return s;
    }
    struct geist_session_opts so = {0};
    so.kv_mode                   = kv;
    struct geist_session *sess   = nullptr;
    s                            = geist_session_create(m, be, &so, &sess);

    geist_token_t ids[N_PROMPT];
    for (size_t i = 0; i < N_PROMPT; i++) {
        ids[i] = (geist_token_t) (100 + (i * 37) % 1000);
    }
    if (s == GEIST_OK)
        s = geist_session_prefill_tokens(sess, N_PROMPT, ids);
    if (s == GEIST_OK) {
        out->logits = copy_logits(sess, &out->n_logits);
        s           = out->logits != nullptr ? GEIST_OK : GEIST_E_INTERNAL;
    }
    for (size_t i = 0; s == GEIST_OK && i < N_DECODE; i++) {
        s = geist_session_decode_step(sess, &out->tokens[i]);
    }
    if (s == GEIST_OK)
        s = geist_session_reset(sess);
    if (s == GEIST_OK)
        s = geist_session_prefill_tokens(sess, N_AFTER_RESET, ids);
    if (s == GEIST_OK) {
        size_t n          = 0;
        out->logits_reset = copy_logits(sess, &n);
        s = out->logits_reset != nullptr && n == out->n_logits ? GEIST_OK : GEIST_E_INTERNAL;
    }
    geist_session_destroy(sess);
    geist_model_destroy(m);
    if (s != GEIST_OK) {
        run_free(out);
    }
    return s;
}

static float max_abs_diff(const float *a, const float *b, size_t n) {
    float d = 0.0f;
    for (size_t i = 0; i < n; i++) {
        const float x = fabsf(a[i] - b[i]);
        d             = x > d ? x : d;
    }
    return d;
}

int main(void) {
    const char *path = resolve_path();
    if (path == nullptr) {
        GEIST_SKIP_FIXTURE("no GGUF for the layer split. Run `make fetch-llama-model`, or set "
                           "GEIST_SPLIT_GGUF_PATH");
    }
    const size_t n_layers = layer_count(path);
    if (n_layers < 2) {
        fprintf(stderr, "cannot read the layer count of %s\n", path);
        return GEIST_TEST_ERROR;
    }

    /* The host's SIMD CPU backend, else the portable one. */
    static const char *const cpus[]    = {"cpu_neon", "cpu_x86", "cpu_scalar"};
    const char              *base_name = nullptr;
    struct geist_backend    *be        = nullptr;
    for (size_t i = 0; be == nullptr && i < sizeof cpus / sizeof cpus[0]; i++) {
        base_name = cpus[i];
        be        = backend(base_name);
    }
    const char           *split_env = getenv("GEIST_SPLIT_BACKEND");
    const bool            exact     = split_env == nullptr || split_env[0] == '\0';
    struct geist_backend *sbe       = backend(exact ? base_name : split_env);
    if (be == nullptr || sbe == nullptr) {
        fprintf(stderr, "backend create failed: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }
    printf("model %s (%zu layers), %s + %s, %s\n",
           path,
           n_layers,
           base_name,
           geist_backend_name(sbe),
           exact ? "byte-identical" : "reported");

    int fails = 0;

    /* A model the split does not support is refused at load, cleanly. */
    {
        struct run        r = {0};
        enum geist_status s = run_once(path, be, sbe, 1, GEIST_KV_FP32, &r);
        if (s == GEIST_E_UNSUPPORTED) {
            run_free(&r);
            geist_backend_destroy(sbe);
            geist_backend_destroy(be);
            GEIST_SKIP("the model is refused by the layer split (GEIST_E_UNSUPPORTED), as "
                       "documented; use a Llama / Qwen3 / Gemma 3 / BitNet GGUF");
        }
        run_free(&r);
    }

    /* split_layers past the layer count. */
    {
        struct geist_session_opts lo = {.split_backend = sbe, .split_layers = n_layers + 1};
        struct geist_model       *m  = nullptr;
        const enum geist_status   s  = geist_model_load_with_opts(path, be, &lo, &m);
        fails += geist_expect(s == GEIST_E_INVALID_ARG && m == nullptr,
                              "split_layers > layer count fails with GEIST_E_INVALID_ARG");
        geist_model_destroy(m);
    }

    /* KIVI on a split model. */
    {
        struct geist_session_opts lo = {.split_backend = sbe, .split_layers = 1};
        struct geist_model       *m  = nullptr;
        enum geist_status         s  = geist_model_load_with_opts(path, be, &lo, &m);
        if (s == GEIST_OK) {
            struct geist_session_opts so   = {.kv_mode = GEIST_KV_KIVI};
            struct geist_session     *sess = nullptr;
            s                              = geist_session_create(m, be, &so, &sess);
            fails += geist_expect(s != GEIST_OK && sess == nullptr,
                                  "a KIVI session on a split model is refused");
            geist_session_destroy(sess);
        } else {
            fails += geist_expect(false, "split load for the KIVI check");
        }
        geist_model_destroy(m);
    }

    const enum geist_kv_mode kvs[]      = {GEIST_KV_FP32, GEIST_KV_INT8, GEIST_KV_INT4};
    const char              *kv_names[] = {"fp32", "int8", "int4"};
    size_t                   splits[16] = {1, n_layers / 2, n_layers - 1, n_layers};
    size_t                   n_splits   = 4;
    const char              *points     = getenv("GEIST_SPLIT_POINTS");
    if (points != nullptr && points[0] != '\0') {
        n_splits = 0;
        for (const char *p = points; *p != '\0' && n_splits < 16;) {
            char                    *end = nullptr;
            const unsigned long long v   = strtoull(p, &end, 10);
            if (end == p) {
                break;
            }
            splits[n_splits++] = (size_t) v;
            p                  = *end == ',' ? end + 1 : end;
        }
    }

    for (size_t k = 0; k < sizeof kvs / sizeof kvs[0]; k++) {
        struct run        ref = {0};
        enum geist_status s   = run_once(path, be, nullptr, 0, kvs[k], &ref);
        if (s != GEIST_OK) {
            fprintf(stderr,
                    "FAIL: unsplit run (kv %s): %s\n",
                    kv_names[k],
                    geist_status_to_string(s));
            fails++;
            continue;
        }
        if (!exact) {
            struct run g = {0};
            s            = run_once(path, sbe, nullptr, 0, kvs[k], &g);
            if (s == GEIST_OK) {
                size_t same = 0;
                while (same < N_DECODE && g.tokens[same] == ref.tokens[same]) {
                    same++;
                }
                printf("  kv %-4s gpu only : max|d| logits %.3g, after reset %.3g, "
                       "greedy tokens matching %zu/%d\n",
                       kv_names[k],
                       (double) max_abs_diff(g.logits, ref.logits, ref.n_logits),
                       (double) max_abs_diff(g.logits_reset, ref.logits_reset, ref.n_logits),
                       same,
                       N_DECODE);
            } else {
                printf("  kv %-4s gpu only : %s\n", kv_names[k], geist_status_to_string(s));
            }
            run_free(&g);
        }
        for (size_t i = 0; i < n_splits; i++) {
            struct run r = {0};
            s            = run_once(path, be, sbe, splits[i], kvs[k], &r);
            if (s != GEIST_OK) {
                fprintf(stderr,
                        "FAIL: split %zu (kv %s): %s\n",
                        splits[i],
                        kv_names[k],
                        geist_status_to_string(s));
                fails++;
                continue;
            }
            size_t same_tokens = 0;
            while (same_tokens < N_DECODE && r.tokens[same_tokens] == ref.tokens[same_tokens]) {
                same_tokens++;
            }
            const float d  = max_abs_diff(r.logits, ref.logits, ref.n_logits);
            const float dr = max_abs_diff(r.logits_reset, ref.logits_reset, ref.n_logits);
            printf("  kv %-4s split %3zu: max|d| logits %.3g, after reset %.3g, "
                   "greedy tokens matching %zu/%d\n",
                   kv_names[k],
                   splits[i],
                   (double) d,
                   (double) dr,
                   same_tokens,
                   N_DECODE);
            if (exact) {
                char what[96];
                snprintf(what,
                         sizeof what,
                         "kv %s split %zu byte-identical",
                         kv_names[k],
                         splits[i]);
                fails += geist_expect(memcmp(r.logits, ref.logits, ref.n_logits * sizeof(float)) ==
                                                      0 &&
                                              memcmp(r.logits_reset,
                                                     ref.logits_reset,
                                                     ref.n_logits * sizeof(float)) == 0 &&
                                              same_tokens == N_DECODE,
                                      what);
            }
            run_free(&r);
        }
        run_free(&ref);
    }

    /* Offload (#78) on two CPU instances: the twins run the same kernels.
     * Their cache is dense, so with an FP32 cache the run is byte-identical
     * to the unsplit one; with INT8 the prefill attends over FP32 rows the
     * load backend's cache does not hold, and is reported. m_min 8: the
     * prompt chunks run on the twins, decode on the load backend. */
    for (size_t k = 0; exact && k < 2; k++) {
        struct run        ref = {0};
        enum geist_status s   = run_once(path, be, nullptr, 0, kvs[k], &ref);
        struct run        r   = {0};
        setenv("GEIST_OFFLOAD_M_MIN", "8", 1);
        if (s == GEIST_OK) {
            s = run_once(path, be, sbe, n_layers / 2, kvs[k], &r);
        }
        unsetenv("GEIST_OFFLOAD_M_MIN");
        size_t same = 0;
        while (s == GEIST_OK && same < N_DECODE && r.tokens[same] == ref.tokens[same]) {
            same++;
        }
        printf("  kv %-4s split %3zu offload: %s, max|d| logits %.3g, after reset %.3g, "
               "greedy tokens matching %zu/%d\n",
               kv_names[k],
               n_layers / 2,
               geist_status_to_string(s),
               s == GEIST_OK ? (double) max_abs_diff(r.logits, ref.logits, ref.n_logits) : 0.0,
               s == GEIST_OK ? (double) max_abs_diff(r.logits_reset, ref.logits_reset, ref.n_logits)
                             : 0.0,
               same,
               N_DECODE);
        if (kvs[k] == GEIST_KV_FP32) {
            fails += geist_expect(
                    s == GEIST_OK &&
                            memcmp(r.logits, ref.logits, ref.n_logits * sizeof(float)) == 0 &&
                            memcmp(r.logits_reset,
                                   ref.logits_reset,
                                   ref.n_logits * sizeof(float)) == 0 &&
                            same == N_DECODE,
                    "kv fp32 offload byte-identical");
        } else {
            fails += geist_expect(s == GEIST_OK, "kv int8 offload runs");
        }
        run_free(&r);
        run_free(&ref);
    }

    geist_backend_destroy(sbe);
    geist_backend_destroy(be);
    return fails ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
}

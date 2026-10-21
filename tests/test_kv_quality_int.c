/*
 * test_kv_quality_int — the quality bar of the lossy KV caches (#53).
 *
 * A quantized KV cache may cost speed nothing and memory less, but it must
 * not change what the model says. The bar, against an FP32 KV cache on the
 * same backend and the same tokens:
 *
 *   mean Jensen-Shannon divergence of the next-token distributions
 *     <= KVQ_MAX_MEAN_JS nats, and
 *   top-1 agreement >= KVQ_MIN_TOP1.
 *
 * The corpus is benchmark/prompts/moe-long.txt (880 Qwen3 tokens): the
 * first n - KVQ_STEPS tokens are prefilled, then the last KVQ_STEPS are
 * teacher-forced one at a time through the decode path, each session
 * reading its distribution before every token. Measured on Qwen3-0.6B
 * Q8_0, cpu_neon (mean JS nats / top-1):
 *
 *   another backend's FP32 KV (the noise floor)     0.0004 / 95 %
 *   INT8, rotated (the default since #53)           0.0013 / 96 %
 *   INT4 = K INT8, V 4-bit, rotated                 0.0054 / 91 %
 *   INT8 unrotated (the default before #53)         0.016  / 82 %   fails
 *   INT4 with a 4-bit K (before #53)                0.31   / 35 %   fails
 *
 * Qwen3 is the subject because its QK norm leaves outlier channels in the
 * K rows, the case a low-bit K breaks. Checked: the session the model gets
 * with kv_mode AUTO (whatever this backend and the environment pick), INT8
 * and INT4. KIVI (2-bit) is an experiment that does not meet the bar
 * (0.10 / 62 %) and is not checked.
 *
 * GEIST_KV_QUALITY_BACKEND picks the backend (default: the build's CPU
 * backend), GEIST_QWEN3_GGUF_PATH the model.
 */
/* unsetenv needs a POSIX feature macro on glibc; no-op on macOS. */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

constexpr double KVQ_MAX_MEAN_JS = 0.01;
constexpr double KVQ_MIN_TOP1    = 0.85;
constexpr size_t KVQ_STEPS       = 256;
constexpr size_t KVQ_MAX_IDS     = 4096;

static const char *resolve_path(void) {
    const char *env = getenv("GEIST_QWEN3_GGUF_PATH");
    if (env != nullptr && env[0] != '\0') {
        return env;
    }
    static const char *candidates[] = {"gguf_artifacts/qwen3-0.6b-q8_0.gguf",
                                       "./qwen3-0.6b-q8_0.gguf"};
    for (size_t i = 0; i < sizeof candidates / sizeof candidates[0]; i++) {
        FILE *f = fopen(candidates[i], "rb");
        if (f != nullptr) {
            fclose(f);
            return candidates[i];
        }
    }
    return nullptr;
}

/* The whole file as a NUL-terminated string, or nullptr. */
static char *read_text(const char *path) {
    FILE *f = fopen(path, "rb");
    if (f == nullptr) {
        return nullptr;
    }
    char  *buf = nullptr;
    long   end = -1;
    size_t n   = 0;
    if (fseek(f, 0, SEEK_END) == 0 && (end = ftell(f)) > 0 && fseek(f, 0, SEEK_SET) == 0 &&
        (buf = malloc((size_t) end + 1)) != nullptr) {
        n      = fread(buf, 1, (size_t) end, f);
        buf[n] = '\0';
    }
    fclose(f);
    if (buf != nullptr && n != (size_t) end) {
        free(buf);
        buf = nullptr;
    }
    return buf;
}

/* p = softmax(x) in double; returns argmax(x). */
static size_t softmax(size_t n, const float x[static n], double p[static n]) {
    size_t top = 0;
    for (size_t i = 1; i < n; i++) {
        top = x[i] > x[top] ? i : top;
    }
    double z = 0.0;
    for (size_t i = 0; i < n; i++) {
        p[i] = exp((double) x[i] - (double) x[top]);
        z += p[i];
    }
    for (size_t i = 0; i < n; i++) {
        p[i] /= z;
    }
    return top;
}

/* Jensen-Shannon divergence of p and q, nats. */
static double js(size_t n, const double p[static n], const double q[static n]) {
    double d = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double m = 0.5 * (p[i] + q[i]);
        if (p[i] > 0.0) {
            d += 0.5 * p[i] * log(p[i] / m);
        }
        if (q[i] > 0.0) {
            d += 0.5 * q[i] * log(q[i] / m);
        }
    }
    return d;
}

struct kvq_mode {
    const char        *name;
    enum geist_kv_mode mode;
};

static const struct kvq_mode MODES[] = {
        {"AUTO", GEIST_KV_AUTO},
        {"INT8", GEIST_KV_INT8},
        {"INT4", GEIST_KV_INT4},
};
constexpr size_t N_MODES = sizeof MODES / sizeof MODES[0];

int main(void) {
    const char *path = resolve_path();
    if (path == nullptr) {
        GEIST_SKIP_FIXTURE("no qwen3 GGUF. Run `make fetch-qwen3-model`, or set "
                           "GEIST_QWEN3_GGUF_PATH");
    }
    char *text = read_text("benchmark/prompts/moe-long.txt");
    if (text == nullptr) {
        fprintf(stderr,
                "ERROR: benchmark/prompts/moe-long.txt not readable (run from the "
                "repository root)\n");
        return GEIST_TEST_ERROR;
    }
    const char *be_name = getenv("GEIST_KV_QUALITY_BACKEND");
    if (be_name == nullptr || be_name[0] == '\0') {
        be_name = GEIST_TEST_CPU_BACKEND;
    }
    /* The defaults are under test: no rotation override. */
    unsetenv("GEIST_KV_ROT");

    struct geist_backend *be                 = nullptr;
    struct geist_model   *model              = nullptr;
    struct geist_session *ref                = nullptr;
    struct geist_session *sess[N_MODES]      = {};
    geist_token_t        *ids                = nullptr;
    double               *p_ref              = nullptr;
    double               *p                  = nullptr;
    double                js_sum[N_MODES]    = {};
    size_t                top1_same[N_MODES] = {};
    int                   rc                 = GEIST_TEST_ERROR;
    size_t                n_ids              = 0;
    if (geist_backend_create(be_name, nullptr, nullptr, &be) != GEIST_OK ||
        geist_model_load(path, be, &model) != GEIST_OK) {
        fprintf(stderr, "ERROR: %s: %s\n", be_name, geist_last_create_error());
        goto done;
    }
    ids                             = malloc(KVQ_MAX_IDS * sizeof *ids);
    struct geist_session_opts o_ref = {
            .max_seq_len = KVQ_MAX_IDS, .temperature = 0.0f, .kv_mode = GEIST_KV_FP32};
    if (ids == nullptr || geist_session_create(model, be, &o_ref, &ref) != GEIST_OK ||
        geist_session_tokenize(ref, text, KVQ_MAX_IDS, ids, &n_ids) != GEIST_OK ||
        n_ids <= KVQ_STEPS + 64) {
        fprintf(stderr, "ERROR: reference session / tokenize (%zu ids)\n", n_ids);
        goto done;
    }
    for (size_t m = 0; m < N_MODES; m++) {
        struct geist_session_opts o = o_ref;
        o.kv_mode                   = MODES[m].mode;
        if (geist_session_create(model, be, &o, &sess[m]) != GEIST_OK) {
            fprintf(stderr, "ERROR: %s session: %s\n", MODES[m].name, geist_last_create_error());
            goto done;
        }
    }
    const size_t n_prompt = n_ids - KVQ_STEPS;
    if (geist_session_prefill_tokens(ref, n_prompt, ids) != GEIST_OK) {
        fprintf(stderr, "ERROR: FP32 prefill: %s\n", geist_session_errmsg(ref));
        goto done;
    }
    for (size_t m = 0; m < N_MODES; m++) {
        if (geist_session_prefill_tokens(sess[m], n_prompt, ids) != GEIST_OK) {
            fprintf(stderr,
                    "ERROR: %s prefill: %s\n",
                    MODES[m].name,
                    geist_session_errmsg(sess[m]));
            goto done;
        }
    }
    size_t n_vocab = 0;
    (void) geist_session_peek_logits(&n_vocab, ref);
    p_ref = n_vocab > 0 ? malloc(n_vocab * sizeof *p_ref) : nullptr;
    p     = n_vocab > 0 ? malloc(n_vocab * sizeof *p) : nullptr;
    if (p_ref == nullptr || p == nullptr) {
        fprintf(stderr, "ERROR: no logits or no memory\n");
        goto done;
    }
    for (size_t i = n_prompt; i < n_ids; i++) {
        size_t       n  = 0;
        const float *lr = geist_session_peek_logits(&n, ref);
        if (lr == nullptr || n != n_vocab) {
            fprintf(stderr, "ERROR: FP32 logits at %zu\n", i);
            goto done;
        }
        const size_t top_ref = softmax(n_vocab, lr, p_ref);
        for (size_t m = 0; m < N_MODES; m++) {
            const float *lm = geist_session_peek_logits(&n, sess[m]);
            if (lm == nullptr || n != n_vocab) {
                fprintf(stderr, "ERROR: %s logits at %zu\n", MODES[m].name, i);
                goto done;
            }
            top1_same[m] += softmax(n_vocab, lm, p) == top_ref;
            js_sum[m] += js(n_vocab, p_ref, p);
            if (geist_session_prefill_tokens(sess[m], 1, &ids[i]) != GEIST_OK) {
                fprintf(stderr, "ERROR: %s step %zu\n", MODES[m].name, i);
                goto done;
            }
        }
        if (geist_session_prefill_tokens(ref, 1, &ids[i]) != GEIST_OK) {
            fprintf(stderr, "ERROR: FP32 step %zu\n", i);
            goto done;
        }
    }
    int fails = 0;
    for (size_t m = 0; m < N_MODES; m++) {
        const double mean_js = js_sum[m] / (double) KVQ_STEPS;
        const double top1    = (double) top1_same[m] / (double) KVQ_STEPS;
        char         what[200];
        snprintf(what,
                 sizeof what,
                 "%s %s KV vs FP32: mean JS %.5f nats (bar %.3f), top-1 %.1f %% (bar %.0f %%) "
                 "over %zu positions after %zu",
                 be_name,
                 MODES[m].name,
                 mean_js,
                 KVQ_MAX_MEAN_JS,
                 100.0 * top1,
                 100.0 * KVQ_MIN_TOP1,
                 KVQ_STEPS,
                 n_prompt);
        printf("  %s\n", what);
        fails += geist_expect(mean_js <= KVQ_MAX_MEAN_JS && top1 >= KVQ_MIN_TOP1, what);
    }
    rc = fails ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
    if (rc == GEIST_TEST_PASS) {
        printf("test_kv_quality_int: every KV mode within the quality bar\n");
    }
done:
    free(p);
    free(p_ref);
    free(ids);
    for (size_t m = 0; m < N_MODES; m++) {
        geist_session_destroy(sess[m]);
    }
    geist_session_destroy(ref);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    free(text);
    return rc;
}

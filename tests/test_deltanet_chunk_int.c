/*
 * test_deltanet_chunk_int — #281 chunked prefill equivalence.
 *
 * The DeltaNet mixer has two mathematically equivalent prefill
 * paths: the sequential per-token recurrence and the chunked GEMM
 * formulation. This test pins their equivalence through the public
 * API: the same batch prefill runs in two sessions, one forced onto
 * the sequential path via GEIST_DN_SEQ_PREFILL=1 (flag read at model
 * load), then next-position logits and greedy continuations must
 * agree. Batch-vs-tokenwise is NOT a valid oracle here — the m1 and
 * mN quantized kernels legitimately differ by ~1.0 in logits.
 *
 * Tolerance: on Accelerate the two paths land bit-identical, but on
 * OpenBLAS (Linux CI) the GEMM rounding drifts activations across
 * int8 quantization boundaries, giving ~1.0 logit steps — the same
 * magnitude the engine's own kernel paths differ by among themselves.
 * So the hard gates are (a) max|dlogit| <= MAX_DIFF (a real math bug
 * produces deltas far above 2 — verified: garbage variants score
 * >20) and (b) no argmax flip where the reference top-2 gap exceeds
 * that jitter. Greedy continuations are informational: each step
 * re-rolls the jitter, so exact chain equality is platform luck
 * (holds on Accelerate, flipped once on the 4-core CI runner).
 *
 * Two prompt lengths: one inside a single engine batch (m_max = 64)
 * and one spanning multiple batches, which additionally exercises the
 * conv/delta state carry between consecutive chunks.
 *
 * SKIPs cleanly when the fixture is absent (make fetch-qwen35-model).
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_TOKENS 256
#define N_CONT 5
#define MAX_DIFF 2.0f

static const char *resolve_path(void) {
    const char *env = getenv("GEIST_QWEN35_GGUF_PATH");
    if (env != nullptr && env[0] != '\0')
        return env;
    static const char *candidates[] = {
            "gguf_artifacts/qwen3.5-0.8b-q8_0.gguf",
            "./qwen3.5-0.8b-q8_0.gguf",
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

static size_t argmax(const float *x, size_t n) {
    size_t best = 0;
    for (size_t i = 1; i < n; i++)
        if (x[i] > x[best])
            best = i;
    return best;
}

/* Batch-prefill `ids` (optionally split into a first call of `n1`
 * tokens — n1 = 2 pins the conv-state roll when seq < kernel-1), copy
 * the pending logits, greedily decode N_CONT continuation tokens. */
static int run_path(struct geist_session *sess,
                    const geist_token_t  *ids,
                    size_t                n,
                    size_t                n1,
                    float                *logits_out,
                    size_t               *n_logits_out,
                    geist_token_t         cont[N_CONT]) {
    geist_session_reset(sess);
    if (n1 > 0 && geist_session_prefill_tokens(sess, n1, ids) != GEIST_OK)
        return -1;
    if (geist_session_prefill_tokens(sess, n - n1, ids + n1) != GEIST_OK)
        return -1;
    size_t       nl = 0;
    const float *lg = geist_session_peek_logits(&nl, sess);
    if (lg == nullptr || nl == 0)
        return -1;
    memcpy(logits_out, lg, nl * sizeof(float));
    *n_logits_out = nl;
    for (int i = 0; i < N_CONT; i++)
        if (geist_session_decode_step(sess, &cont[i]) != GEIST_OK)
            return -1;
    return 0;
}

static int compare_case(struct geist_session *chunk_sess,
                        struct geist_session *seq_sess,
                        const char           *label,
                        size_t                n1,
                        const char           *text) {
    geist_token_t ids[MAX_TOKENS];
    size_t        n = 0;
    if (geist_session_tokenize(chunk_sess, text, MAX_TOKENS, ids, &n) != GEIST_OK || n < 8) {
        fprintf(stderr, "FAIL[%s]: tokenize (%zu tokens)\n", label, n);
        return 1;
    }

    static float  lg_batch[262144], lg_seq[262144];
    size_t        nl_b = 0, nl_s = 0;
    geist_token_t cont_b[N_CONT], cont_s[N_CONT];
    if (run_path(chunk_sess, ids, n, n1, lg_batch, &nl_b, cont_b) != 0 ||
        run_path(seq_sess, ids, n, n1, lg_seq, &nl_s, cont_s) != 0 || nl_b != nl_s) {
        fprintf(stderr, "FAIL[%s]: prefill/logits path\n", label);
        return 1;
    }

    float md = 0.0f;
    for (size_t i = 0; i < nl_b; i++) {
        const float d = fabsf(lg_batch[i] - lg_seq[i]);
        if (d > md)
            md = d;
    }
    const size_t am_b = argmax(lg_batch, nl_b);
    const size_t am_s = argmax(lg_seq, nl_s);
    /* Reference top-2 gap: an argmax flip is only evidence of a bug
     * when the sequential path's winner led by more than the known
     * kernel jitter. Greedy continuations are informational only —
     * each step re-rolls the jitter, so a chain can legitimately
     * diverge whenever some step's gap is small. */
    float second = -1e30f;
    for (size_t i = 0; i < nl_s; i++)
        if (i != am_s && lg_seq[i] > second)
            second = lg_seq[i];
    const float gap  = lg_seq[am_s] - second;
    int         cdiv = 0;
    for (int i = 0; i < N_CONT; i++)
        cdiv |= cont_b[i] != cont_s[i];
    printf("%s: %zu tokens, max|dlogit|=%.4f, argmax %zu/%zu (top2 gap %.2f), cont %s\n",
           label,
           n,
           (double) md,
           am_b,
           am_s,
           (double) gap,
           cdiv ? "diverged (informational)" : "identical");
    if (md > MAX_DIFF || (am_b != am_s && gap > MAX_DIFF)) {
        fprintf(stderr, "FAIL[%s]: chunked prefill != sequential\n", label);
        return 1;
    }
    return 0;
}

/* Load model + session; the DN prefill path is chosen by the
 * GEIST_DN_SEQ_PREFILL env at model-load time. */
static int load_one(const char            *path,
                    struct geist_backend **be,
                    struct geist_model   **model,
                    struct geist_session **sess) {
    enum geist_status s = geist_backend_create("cpu_neon", nullptr, nullptr, be);
    if (s != GEIST_OK)
        s = geist_backend_create("cpu_scalar", nullptr, nullptr, be);
    if (s != GEIST_OK) {
        fprintf(stderr, "backend create failed: %s\n", geist_last_create_error());
        return 1;
    }
    if (geist_model_load(path, *be, model) != GEIST_OK) {
        fprintf(stderr, "model load failed: %s\n", geist_last_create_error());
        return 1;
    }
    struct geist_session_opts opts = {.max_seq_len = 1024, .temperature = 0.0f};
    if (geist_session_create(*model, *be, &opts, sess) != GEIST_OK) {
        fprintf(stderr, "session create failed\n");
        return 1;
    }
    return 0;
}

int main(void) {
    const char *path = resolve_path();
    if (path == nullptr) {
        GEIST_SKIP_FIXTURE("no qwen35 GGUF. Run `make fetch-qwen35-model`, or set "
                           "GEIST_QWEN35_GGUF_PATH");
    }

    struct geist_backend *be_c = nullptr, *be_s = nullptr;
    struct geist_model   *m_c = nullptr, *m_s = nullptr;
    struct geist_session *sess_c = nullptr, *sess_s = nullptr;
    unsetenv("GEIST_DN_SEQ_PREFILL");
    if (load_one(path, &be_c, &m_c, &sess_c) != 0)
        return 1;
    setenv("GEIST_DN_SEQ_PREFILL", "1", 1);
    if (load_one(path, &be_s, &m_s, &sess_s) != 0)
        return 1;
    unsetenv("GEIST_DN_SEQ_PREFILL");

    int rc = 0;
    /* single engine batch (< 64 tokens) */
    rc |= compare_case(sess_c,
                       sess_s,
                       "single-chunk",
                       0,
                       "The quick brown fox jumps over the lazy dog while the "
                       "river runs quietly past the old stone bridge at dawn.");
    /* seq == 2 first call: conv-state roll with seq < kernel-1 */
    rc |= compare_case(sess_c,
                       sess_s,
                       "tiny-first-chunk",
                       2,
                       "The quick brown fox jumps over the lazy dog while the "
                       "river runs quietly past the old stone bridge at dawn.");
    /* multiple batches: chunk -> chunk state carry across m_max boundaries */
    rc |= compare_case(sess_c,
                       sess_s,
                       "multi-chunk",
                       0,
                       "In the early morning light, the research vessel left the "
                       "harbor and steered toward the open sea, its instruments "
                       "recording temperature, salinity, and current speed at "
                       "every depth. The crew had prepared for weeks, checking "
                       "each sensor twice, calibrating the sonar array, and "
                       "loading provisions for a month of continuous work far "
                       "from the coast. By noon the first samples were already "
                       "on deck, and the long tables in the wet lab filled with "
                       "carefully labeled bottles of seawater from different "
                       "depths and stations along the planned transect line.");

    geist_session_destroy(sess_c);
    geist_session_destroy(sess_s);
    geist_model_destroy(m_c);
    geist_model_destroy(m_s);
    geist_backend_destroy(be_c);
    geist_backend_destroy(be_s);
    if (rc == 0)
        printf("OK: chunked prefill == sequential recurrence\n");
    return rc;
}

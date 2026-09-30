/*
 * bench_speculative — wall-clock comparison of geist_session_decode_step
 * vs geist_session_decode_speculative on real chat prompts.
 *
 * For each prompt:
 *   1. Format as Gemma chat turn, prefill via geist_session_set_prompt.
 *   2. Sequential reference: N decode_step calls, time the loop.
 *   3. Reset + re-prefill, then speculative decode for N tokens via
 *      decode_speculative(k_max), feeding the growing history to the
 *      drafter. Time the loop. Track tokens-per-call.
 *
 * Reports wall-time, tokens/sec, speedup factor, and acceptance rate
 * per prompt + averages. Numerical correctness (token stream equiv)
 * is checked by test_speculative_loop_int; this bench is wall-clock
 * only.
 *
 *   ./bench_speculative                  — default model search path
 *   GEIST_GGUF_PATH=... ./bench_speculative
 *   GEIST_SPEC_KMAX=8 ./bench_speculative   — override k_max (default 4)
 *
 * Exits 77 (SKIP) if no GGUF / tokenizer reachable.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_util.h>
#include <geist_backend.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_HISTORY 2048
#define DECODE_N 50

static const char *const PROMPTS[] = {
        "What is the capital of France?",
        "Complete the poem: 'Roses are red, violets are'",
        "List five common smart-home commands.",
        "Write a Python function that returns the nth Fibonacci number.",
        "Translate 'Good morning, how are you?' to German, French, and Spanish.",
};
#define N_PROMPTS (sizeof PROMPTS / sizeof PROMPTS[0])

static double monotonic_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec * 1e3 + (double) ts.tv_nsec / 1e6;
}

static void format_chat(char *buf, size_t buf_size, const char *user_prompt) {
    snprintf(buf, buf_size, "<bos><|turn>user\n%s<turn|>\n<|turn>model\n", user_prompt);
}

struct stats {
    double t_seq_ms;
    double t_spec_ms;
    size_t spec_calls;
    size_t spec_emits;
};

static int run_prompt(struct geist_model              *model,
                      struct geist_backend            *be,
                      const struct geist_session_opts *opts,
                      const char                      *user_prompt,
                      size_t                           k_max,
                      struct stats                    *out) {
    char chat_buf[1024];
    format_chat(chat_buf, sizeof chat_buf, user_prompt);

    /* ---- Sequential reference. */
    struct geist_session *sess = nullptr;
    if (geist_session_create(model, be, opts, &sess) != GEIST_OK)
        return 1;
    if (geist_session_set_prompt(sess, chat_buf) != GEIST_OK) {
        fprintf(stderr, "  set_prompt: %s\n", geist_session_errmsg(sess));
        geist_session_destroy(sess);
        return 1;
    }

    double t0 = monotonic_ms();
    for (size_t i = 0; i < DECODE_N; i++) {
        geist_token_t tok;
        if (geist_session_decode_step(sess, &tok) != GEIST_OK) {
            geist_session_destroy(sess);
            return 1;
        }
    }
    out->t_seq_ms = monotonic_ms() - t0;
    geist_session_destroy(sess);

    /* ---- Speculative. */
    sess = nullptr;
    if (geist_session_create(model, be, opts, &sess) != GEIST_OK)
        return 1;

    /* Tokenize prompt explicitly so we can seed the drafter's history
     * with it — typical chat patterns (model echoing prompt content,
     * structural repeats) hit n-gram matches against the prompt much
     * more often than against just the previously-emitted tokens. */
    geist_token_t history[MAX_HISTORY];
    size_t        history_n = 0;
    if (geist_session_tokenize(sess, chat_buf, MAX_HISTORY, history, &history_n) != GEIST_OK) {
        fprintf(stderr, "  tokenize failed: %s\n", geist_session_errmsg(sess));
        geist_session_destroy(sess);
        return 1;
    }
    if (geist_session_prefill_tokens(sess, history_n, history) != GEIST_OK) {
        geist_session_destroy(sess);
        return 1;
    }
    size_t        spec_calls = 0;
    size_t        spec_emits = 0;
    geist_token_t buf[16 + 1];

    t0 = monotonic_ms();
    while (spec_emits < DECODE_N) {
        size_t            n;
        enum geist_status s = geist_session_decode_speculative(
                sess, k_max, history_n, history, k_max + 1, buf, &n);
        if (s != GEIST_OK || n == 0) {
            fprintf(stderr, "  spec_step failed: %s\n", geist_status_to_string(s));
            geist_session_destroy(sess);
            return 1;
        }
        spec_calls++;
        for (size_t i = 0; i < n && spec_emits < DECODE_N; i++) {
            spec_emits++;
            if (history_n < MAX_HISTORY)
                history[history_n++] = buf[i];
        }
    }
    out->t_spec_ms  = monotonic_ms() - t0;
    out->spec_calls = spec_calls;
    out->spec_emits = spec_emits;
    geist_session_destroy(sess);
    return 0;
}

int main(void) {
    GEIST_REQUIRE_GGUF(model_path);

    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create("auto", nullptr, nullptr, &be);
    if (s != GEIST_OK)
        s = geist_backend_create("cpu_scalar", nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        fprintf(stderr, "backend_create: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }
    struct geist_model *model = nullptr;
    s                         = geist_model_load(model_path, be, &model);
    if (s != GEIST_OK) {
        fprintf(stderr, "model_load: %s\n", geist_last_create_error());
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    const char *kmax_env = getenv("GEIST_SPEC_KMAX");
    size_t      k_max    = kmax_env != nullptr ? (size_t) atol(kmax_env) : 4;
    if (k_max == 0 || k_max > 16)
        k_max = 4;

    printf("Backend: %s  Model: %s  k_max=%zu  decode_n=%d\n\n",
           geist_backend_name(be),
           model_path,
           k_max,
           DECODE_N);
    printf("%-50s  %8s  %8s  %5s  %5s  %5s\n", "prompt", "seq ms", "spec ms", "x", "t/cl", "acc%");
    printf("%-50s  %8s  %8s  %5s  %5s  %5s\n", "----", "----", "----", "----", "----", "----");

    struct geist_session_opts opts      = {.max_seq_len = 2048, .temperature = 0.0f};
    double                    total_seq = 0.0, total_spec = 0.0;
    size_t                    total_calls = 0, total_emits = 0;

    for (size_t i = 0; i < N_PROMPTS; i++) {
        struct stats st = {0};
        const char  *p  = PROMPTS[i];
        if (run_prompt(model, be, &opts, p, k_max, &st) != 0) {
            fprintf(stderr, "prompt %zu failed\n", i);
            continue;
        }
        const double speedup    = st.t_seq_ms / st.t_spec_ms;
        const double t_per_call = (double) st.spec_emits / (double) st.spec_calls;
        const double accept_pct = ((t_per_call - 1.0) / (double) k_max) * 100.0;
        /* Trim prompt for display. */
        char disp[51];
        snprintf(disp, sizeof disp, "%.50s", p);
        printf("%-50s  %8.1f  %8.1f  %5.2f  %5.2f  %4.1f%%\n",
               disp,
               st.t_seq_ms,
               st.t_spec_ms,
               speedup,
               t_per_call,
               accept_pct);
        total_seq += st.t_seq_ms;
        total_spec += st.t_spec_ms;
        total_calls += st.spec_calls;
        total_emits += st.spec_emits;
    }

    printf("\n");
    printf("Aggregate (%zu prompts × %d tokens):\n", N_PROMPTS, DECODE_N);
    printf("  sequential : %.1f ms total  →  %.1f tok/s\n",
           total_seq,
           (N_PROMPTS * DECODE_N) / (total_seq / 1e3));
    printf("  speculative: %.1f ms total  →  %.1f tok/s\n",
           total_spec,
           total_emits / (total_spec / 1e3));
    printf("  speedup    : %.2fx\n", total_seq / total_spec);
    printf("  spec t/call: %.2f (1.00 = sequential)\n",
           (double) total_emits / (double) total_calls);

    geist_model_destroy(model);
    geist_backend_destroy(be);
    return GEIST_TEST_PASS;
}

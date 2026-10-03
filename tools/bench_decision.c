/* Public-API benchmark driver. tools/bench_decision.py owns dataset validation,
 * hashes, host metadata and summaries. This process owns inference timing.
 * Wire input per case: n_prompt n_candidates prompt_ids... candidate_ids...
 * No text templates or label truncation are performed here. */
#include <geist_decision.h>
#include <geist_util.h>

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

[[nodiscard]] static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t) t.tv_sec * 1000000000ULL + (uint64_t) t.tv_nsec;
}
[[nodiscard]] static bool number(size_t *out, const char *text) {
    *out = 0;
    if (*text == '\0' || *text == '-') {
        return false;
    }
    char *end;
    errno                          = 0;
    const unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || *end != '\0' || value > SIZE_MAX) {
        return false;
    }
    *out = (size_t) value;
    return true;
}
[[nodiscard]] static bool read_ids(size_t n, size_t vocab, geist_token_t ids[static n]) {
    for (size_t i = 0; i < n; i++) {
        int64_t id;
        if (scanf("%" SCNd64, &id) != 1 || id < 0 || (uint64_t) id >= vocab || id > INT32_MAX) {
            return false;
        }
        ids[i] = (geist_token_t) id;
    }
    return true;
}
/* Independent candidate reference, adapted from eval_geist's SCOREALT path.
 * Conditional normalization (not the full-vocabulary log_p emitted there). */
[[nodiscard]] static bool reference(struct geist_session *s,
                                    size_t                n,
                                    const geist_token_t   ids[static n],
                                    float                 logits[static n],
                                    double                probabilities[static n],
                                    size_t               *best) {
    size_t       vocab = 0;
    const float *p     = geist_session_peek_logits(&vocab, s);
    if (p == nullptr) {
        return false;
    }
    double maximum = -INFINITY;
    *best          = 0;
    for (size_t i = 0; i < n; i++) {
        if ((size_t) ids[i] >= vocab || !isfinite(p[ids[i]])) {
            return false;
        }
        logits[i] = p[ids[i]];
        if (logits[i] > maximum) {
            maximum = logits[i];
            *best   = i;
        }
    }
    double total = 0;
    for (size_t i = 0; i < n; i++) {
        probabilities[i] = exp((double) logits[i] - maximum);
        total += probabilities[i];
    }
    for (size_t i = 0; i < n; i++) {
        probabilities[i] /= total;
    }
    return true;
}

int main(int argc, char **argv) {
    size_t prompt_cap, candidate_cap, decode_n, warmup, repeats;
    if (argc != 8 || !number(&prompt_cap, argv[3]) || !number(&candidate_cap, argv[4]) ||
        !number(&decode_n, argv[5]) || !number(&warmup, argv[6]) || !number(&repeats, argv[7]) ||
        prompt_cap == 0 || candidate_cap == 0 || decode_n < 2 || repeats == 0 ||
        prompt_cap > 1048576 || candidate_cap > 1048576 || decode_n > 4096 || warmup > 1000 ||
        repeats > 10000) {
        fprintf(stderr,
                "usage: bench_decision model.gguf backend prompt_cap candidate_cap decode_n warmup "
                "repeats\n");
        return 2;
    }
    if (!geist_decision_available()) {
        fprintf(stderr, "rebuild with DECISION=1\n");
        return 2;
    }
    struct geist_backend           *be = nullptr;
    struct geist_model             *m  = nullptr;
    struct geist_session           *s  = nullptr;
    struct geist_decision          *d  = nullptr;
    const struct geist_backend_opts bo = {.log_level_max = GEIST_LOG_ERROR};
    /* Bounds above make this sum/product safe, even on 32-bit hosts. */
    const struct geist_session_opts so = {
            .max_seq_len = prompt_cap + decode_n, .top_p = 1.0f, .kv_mode = GEIST_KV_FP32};
    const struct geist_decision_opts o             = {.mode              = GEIST_DECISION_DENSE,
                                                      .max_prompt_tokens = prompt_cap,
                                                      .max_candidates    = candidate_cap,
                                                      .kv_mode           = GEIST_KV_FP32};
    int                              exit_code     = 1;
    geist_token_t                   *prompt        = calloc(prompt_cap, sizeof *prompt);
    geist_token_t                   *candidates    = calloc(candidate_cap, sizeof *candidates);
    geist_token_t                   *generated     = calloc(decode_n, sizeof *generated);
    float                           *logits        = calloc(candidate_cap, sizeof *logits);
    double                          *probabilities = calloc(candidate_cap, sizeof *probabilities);
    if (prompt == nullptr || candidates == nullptr || generated == nullptr || logits == nullptr ||
        probabilities == nullptr || geist_backend_create(argv[2], &bo, nullptr, &be) != GEIST_OK ||
        geist_model_load_with_opts(argv[1], be, &so, &m) != GEIST_OK ||
        geist_session_create(m, be, &so, &s) != GEIST_OK ||
        geist_decision_create(m, be, &o, &d) != GEIST_OK) {
        fprintf(stderr, "setup: %s\n", geist_last_create_error());
        goto done;
    }
    printf("{\"kind\":\"runtime\",\"version\":\"%s\",\"backend\":\"%s\",\"vocab\":%zu,\"kv_mode\":"
           "\"fp32\"}\n",
           geist_version_string(),
           geist_backend_name(be),
           geist_decision_vocab_size(d));
    const char *modes[]    = {"decision_dense", "scorealt_dense", "generate_1", "generate_long"};
    size_t      case_index = 0, np, nc;
    int         read;
    while ((read = scanf("%zu %zu", &np, &nc)) == 2) {
        if (np == 0 || np > prompt_cap || nc == 0 || nc > candidate_cap ||
            !read_ids(np, geist_decision_vocab_size(d), prompt) ||
            !read_ids(nc, geist_decision_vocab_size(d), candidates)) {
            fprintf(stderr, "invalid wire input at case %zu\n", case_index);
            goto done;
        }
        for (size_t trial = 0; trial < 1 + warmup + repeats; trial++) {
            const char *phase = trial == 0 ? "first" : trial <= warmup ? "warmup" : "warm";
            for (size_t position = 0; position < 4; position++) {
                const size_t                 mode   = (position + trial + case_index) % 4;
                struct geist_decision_result result = {0};
                size_t                       best = 0, n_generated = 0;
                enum geist_status            status = GEIST_OK;
                const uint64_t               start  = now_ns();
                if (mode == 0) {
                    status = geist_decision_score(d, np, nc, prompt, candidates, &result);
                    if (status == GEIST_OK) {
                        best = result.best_index;
                    }
                } else {
                    status = geist_session_reset(s);
                    if (status == GEIST_OK) {
                        status = geist_session_prefill_tokens(s, np, prompt);
                    }
                    if (mode == 1 && status == GEIST_OK) {
                        if (!reference(s, nc, candidates, logits, probabilities, &best)) {
                            status = GEIST_E_BACKEND;
                        }
                    } else if (status == GEIST_OK) {
                        const size_t limit = mode == 2 ? 1 : decode_n;
                        for (size_t i = 0; i < limit; i++) {
                            status = geist_session_decode_step(s, &generated[n_generated]);
                            if (status != GEIST_OK) {
                                break;
                            }
                            n_generated++;
                            if (generated[n_generated - 1] == geist_model_eos_token(m)) {
                                break;
                            }
                        }
                    }
                }
                const double elapsed = (double) (now_ns() - start) / 1e6;
                if (status != GEIST_OK) {
                    fprintf(stderr,
                            "case %zu %s: %s: %s\n",
                            case_index,
                            modes[mode],
                            geist_status_to_string(status),
                            mode == 0 ? geist_decision_errmsg(d) : geist_session_errmsg(s));
                    goto done;
                }
                printf("{\"kind\":\"sample\",\"case_index\":%zu,\"mode\":\"%s\",\"phase\":\"%s\","
                       "\"trial\":%zu,\"order\":%zu,\"elapsed_ms\":%.9f,\"prompt_tokens\":%zu,"
                       "\"candidate_count\":%zu,\"generated_ids\":[",
                       case_index,
                       modes[mode],
                       phase,
                       trial,
                       position,
                       elapsed,
                       np,
                       nc);
                for (size_t i = 0; i < n_generated; i++) {
                    printf("%s%d", i ? "," : "", generated[i]);
                }
                printf("],\"best_index\":");
                if (mode < 2) {
                    printf("%zu,\"logits\":[", best);
                    for (size_t i = 0; i < nc; i++) {
                        printf("%s%.9g", i ? "," : "", mode == 0 ? result.logits[i] : logits[i]);
                    }
                    printf("],\"conditional_probabilities\":[");
                    for (size_t i = 0; i < nc; i++) {
                        printf("%s%.17g",
                               i ? "," : "",
                               mode == 0 ? result.probabilities[i] : probabilities[i]);
                    }
                    printf("]}");
                } else {
                    printf("null}");
                }
                putchar('\n');
            }
        }
        case_index++;
    }
    if (read != EOF || case_index == 0) {
        fprintf(stderr, "missing/malformed input\n");
        goto done;
    }
    exit_code = 0;
done:
    free(prompt);
    free(candidates);
    free(generated);
    free(logits);
    free(probabilities);
    geist_decision_destroy(d);
    geist_session_destroy(s);
    geist_model_destroy(m);
    geist_backend_destroy(be);
    return exit_code;
}

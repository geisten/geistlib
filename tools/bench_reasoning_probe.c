/* Reasoning with answer probes (#236): greedy generation where, every
 * `interval` generated tokens, the session's complete state is saved
 * (geist_session_snapshot), a probe suffix is prefilled (the thinking-close
 * tokens, e.g. "\n</think>\n\n"), the candidates' logits are read, and the
 * state is restored. A restore puts back the pending logits and any owed
 * forward, so the generation continues exactly as without the probe.
 *
 *   bench_reasoning_probe model.gguf backend prompt_cap cap interval
 *
 * Wire input per case: n_prompt n_candidates n_probe prompt_ids...
 * candidate_ids... probe_ids... interval 0 generates without probes.
 *
 * Output, one JSON line per case (flushed): generated_ids, generation_stop
 * ("eos", or "limit" at cap), generated_utf8_hex, generated_text_complete,
 * surface_decode_ms, prefill_ms, decode_ms (decode steps only), probes
 * [{pos, logits[n_candidates], ms}] where pos is the number of generated
 * tokens the probe saw (0 = right after the prompt). */
#include <geist.h>
#include <geist_util.h>

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

[[nodiscard]] static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double) t.tv_sec * 1e3 + (double) t.tv_nsec / 1e6;
}

[[nodiscard]] static bool number(size_t *out, const char *text) {
    *out = 0;
    if (*text == '\0' || *text == '-')
        return false;
    char *end;
    errno                          = 0;
    const unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || *end != '\0' || value > SIZE_MAX)
        return false;
    *out = (size_t) value;
    return true;
}

[[nodiscard]] static int scan_number(size_t *out) {
    char      token[24];
    const int got = scanf("%23s", token);
    if (got != 1)
        return got;
    return number(out, token) ? 1 : 0;
}

[[nodiscard]] static bool read_ids(size_t n, geist_token_t ids[static n]) {
    for (size_t i = 0; i < n; i++) {
        size_t id;
        if (scan_number(&id) != 1 || id > INT32_MAX)
            return false;
        ids[i] = (geist_token_t) id;
    }
    return true;
}

struct probe {
    size_t pos;
    double ms;
    float  logits[32];
};

/* Save the state, prefill the probe, read the candidates, restore. */
[[nodiscard]] static enum geist_status probe_once(struct geist_session *s,
                                                  size_t                n_probe,
                                                  const geist_token_t   probe[static n_probe],
                                                  size_t                n_cand,
                                                  const geist_token_t   cand[static n_cand],
                                                  void                **img,
                                                  size_t               *img_cap,
                                                  struct probe         *out) {
    const double      t0   = now_ms();
    size_t            need = 0, got = 0;
    enum geist_status st = geist_session_snapshot_size(&need, s);
    if (st == GEIST_OK && need > *img_cap) {
        void *grown = realloc(*img, need);
        if (grown == nullptr)
            return GEIST_E_OOM;
        *img = grown, *img_cap = need;
    }
    if (st == GEIST_OK)
        st = geist_session_snapshot(&got, *img_cap, *img, s);
    if (st == GEIST_OK)
        st = geist_session_prefill_tokens(s, n_probe, probe);
    size_t       vocab  = 0;
    const float *logits = st == GEIST_OK ? geist_session_peek_logits(&vocab, s) : nullptr;
    if (st == GEIST_OK && logits == nullptr)
        st = GEIST_E_BACKEND;
    for (size_t i = 0; st == GEIST_OK && i < n_cand; i++) {
        if ((size_t) cand[i] >= vocab) {
            st = GEIST_E_INVALID_ARG;
            break;
        }
        out->logits[i] = logits[cand[i]];
    }
    const enum geist_status rs =
            got > 0 ? geist_session_restore(got, *img, s) : GEIST_E_INVALID_STATE;
    out->ms = now_ms() - t0;
    return st != GEIST_OK ? st : rs;
}

int main(int argc, char **argv) {
    size_t prompt_cap = 0, cap = 0, interval = 0;
    if (argc != 6 || !number(&prompt_cap, argv[3]) || !number(&cap, argv[4]) ||
        !number(&interval, argv[5]) || prompt_cap == 0 || prompt_cap > 1048576 || cap == 0 ||
        cap > 32768) {
        fprintf(stderr,
                "usage: bench_reasoning_probe model.gguf backend prompt_cap cap interval\n");
        return 2;
    }
    struct geist_backend           *be = nullptr;
    struct geist_model             *m  = nullptr;
    struct geist_session           *s  = nullptr;
    const struct geist_backend_opts bo = {.log_level_max = GEIST_LOG_ERROR};
    const struct geist_session_opts so = {
            .max_seq_len = prompt_cap + cap + 64, .top_p = 1.0f, .kv_mode = GEIST_KV_FP32};
    geist_token_t *prompt = calloc(prompt_cap, sizeof *prompt);
    geist_token_t *gen    = calloc(cap, sizeof *gen);
    struct probe  *probes = calloc(cap / (interval ? interval : cap) + 2, sizeof *probes);
    geist_token_t  cand[32], probe[64];
    void          *img     = nullptr;
    size_t         img_cap = 0, case_index = 0, np = 0, nc = 0, nq = 0;
    int            exit_code = 1, read;
    if (prompt == nullptr || gen == nullptr || probes == nullptr ||
        geist_backend_create(argv[2], &bo, nullptr, &be) != GEIST_OK ||
        geist_model_load_with_opts(argv[1], be, &so, &m) != GEIST_OK ||
        geist_session_create(m, be, &so, &s) != GEIST_OK) {
        fprintf(stderr, "setup: %s\n", geist_last_create_error());
        goto done;
    }
    const geist_token_t eos = geist_model_eos_token(m);
    printf("{\"kind\":\"runtime\",\"version\":\"%s\",\"backend\":\"%s\",\"kv_mode\":\"fp32\","
           "\"eos\":%d,\"interval\":%zu}\n",
           geist_version_string(),
           geist_backend_name(be),
           eos,
           interval);
    fflush(stdout);
    while ((read = scan_number(&np)) == 1) {
        if (scan_number(&nc) != 1 || scan_number(&nq) != 1 || np == 0 || np > prompt_cap ||
            nc == 0 || nc > 32 || nq == 0 || nq > 64 || !read_ids(np, prompt) ||
            !read_ids(nc, cand) || !read_ids(nq, probe)) {
            fprintf(stderr, "invalid wire input at case %zu\n", case_index);
            goto done;
        }
        const double      tp = now_ms();
        enum geist_status st = geist_session_reset(s);
        if (st == GEIST_OK)
            st = geist_session_prefill_tokens(s, np, prompt);
        const double prefill_ms = now_ms() - tp;
        size_t       n_gen = 0, n_probes = 0;
        double       decode_ms = 0;
        while (st == GEIST_OK) {
            if (interval > 0 && n_gen % interval == 0) {
                probes[n_probes].pos = n_gen;
                st = probe_once(s, nq, probe, nc, cand, &img, &img_cap, &probes[n_probes]);
                n_probes++;
                if (st != GEIST_OK)
                    break;
            }
            const double td = now_ms();
            st              = geist_session_decode_step(s, &gen[n_gen]);
            decode_ms += now_ms() - td;
            if (st != GEIST_OK)
                break;
            n_gen++;
            if (gen[n_gen - 1] == eos || n_gen == cap)
                break;
        }
        if (st != GEIST_OK) {
            fprintf(stderr,
                    "case %zu: %s: %s\n",
                    case_index,
                    geist_status_to_string(st),
                    geist_session_errmsg(s));
            goto done;
        }
        printf("{\"kind\":\"sample\",\"case_index\":%zu,\"mode\":\"generate_long_probed\","
               "\"prompt_tokens\":%zu,\"generated_ids\":[",
               case_index,
               np);
        for (size_t i = 0; i < n_gen; i++)
            printf("%s%d", i ? "," : "", gen[i]);
        printf("],\"generation_stop\":\"%s\",\"cap\":%zu",
               gen[n_gen - 1] == eos ? "eos" : "limit",
               cap);
        const double ts       = now_ms();
        bool         complete = true;
        printf(",\"generated_utf8_hex\":\"");
        for (size_t i = 0; i < n_gen; i++) {
            if (gen[i] == eos)
                continue;
            const char *piece = geist_session_token_to_str(s, gen[i]);
            if (piece == nullptr || *piece == '\0') {
                complete = false;
                continue;
            }
            for (const unsigned char *c = (const unsigned char *) piece; *c; c++)
                printf("%02x", (unsigned) *c);
        }
        printf("\",\"generated_text_complete\":%s,\"surface_decode_ms\":%.6f,\"prefill_ms\":%.3f,"
               "\"decode_ms\":%.3f,\"probes\":[",
               complete ? "true" : "false",
               now_ms() - ts,
               prefill_ms,
               decode_ms);
        for (size_t p = 0; p < n_probes; p++) {
            printf("%s{\"pos\":%zu,\"ms\":%.3f,\"logits\":[",
                   p ? "," : "",
                   probes[p].pos,
                   probes[p].ms);
            for (size_t i = 0; i < nc; i++)
                printf("%s%.9g", i ? "," : "", (double) probes[p].logits[i]);
            printf("]}");
        }
        printf("]}\n");
        fflush(stdout);
        case_index++;
    }
    if (read != EOF || case_index == 0) {
        fprintf(stderr, "missing/malformed input\n");
        goto done;
    }
    exit_code = 0;
done:
    free(prompt);
    free(gen);
    free(probes);
    free(img);
    geist_session_destroy(s);
    geist_model_destroy(m);
    geist_backend_destroy(be);
    return exit_code;
}

/* Batched generation driver (#26): greedy generation for many prompts with
 * up to `batch` sessions decoding together through geist_session_decode_batch,
 * refilled as questions finish (continuous batching).
 *
 *   bench_reasoning_batch model.gguf backend prompt_cap cap batch [interval tau]
 *
 * With interval > 0 (#236) every session's answer is probed right after its
 * prompt and every `interval` generated tokens, between batched steps: its
 * state is saved (geist_session_snapshot), the probe tokens are prefilled,
 * the candidates' logits are read, and the state is restored, so the
 * generation continues exactly as without the probe. Then each case on the
 * wire reads n_cached n_prompt n_candidates n_probe prompt_ids...
 * candidate_ids... probe_ids... With tau > 0 a session stops at the first
 * probe from position `interval` on whose top softmax probability over the
 * candidates is at least tau (generation_stop "early_exit", exit_answer its
 * argmax); tau 0 only records the probes.
 *
 * Wire input per case: n_cached n_prompt prompt_ids... Consecutive cases with
 * the same first n_cached tokens share one prefix snapshot
 * (geist_session_snapshot after prefilling them once); each case restores it
 * and prefills its remaining tokens. n_cached 0 prefills the whole prompt.
 *
 * Output, one JSON line each, flushed:
 *   runtime          version, backend, decode_batch_max, batch
 *   prefix_prepare   case_index, elapsed_ms, cached_tokens, snapshot_bytes
 *   sample           case_index, prompt_tokens, generated_ids, generation_stop
 *                    ("eos" or "limit" at cap), generated_utf8_hex,
 *                    generated_text_complete, surface_decode_ms, prefill_ms,
 *                    started_ms / finished_ms (since start), the token count
 *                    each batch size produced for it
 *   summary          steps, decode_ms, tokens, per-size step counts
 * A case is finished when its sample line is written; a stopped run loses the
 * cases in flight. Generation stops at the model's EOS token or at cap tokens,
 * as bench_decision's generate_long does. */
#include <geist.h>
#include <geist_util.h>

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { MAX_BATCH = 8 };

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

struct item {
    size_t         n_cut, n_prompt, n_cand, n_probe;
    geist_token_t *ids;
    geist_token_t  cand[32], probe[64];
};

struct probe {
    size_t pos;
    double ms;
    float  logits[32];
};

struct slot {
    struct geist_session *s;
    bool                  active;
    size_t                item;
    size_t                n_gen;
    geist_token_t        *gen;
    double                started, prefill_ms;
    size_t                by_size[MAX_BATCH + 1];
    struct probe         *probes;
    size_t                n_probes;
    int                   exit_answer; /* -1: no early exit */
};

static size_t vocab; /* logits width, read after the first prefill */

static void emit_sample(
        const struct slot *sl, const struct item *it, size_t cap, geist_token_t eos, double t0) {
    const bool  stop_eos = sl->n_gen > 0 && sl->gen[sl->n_gen - 1] == eos;
    const char *stop     = sl->exit_answer >= 0 ? "early_exit" : stop_eos ? "eos" : "limit";
    printf("{\"kind\":\"sample\",\"case_index\":%zu,\"mode\":\"generate_long_batched\","
           "\"prompt_tokens\":%zu,\"generated_ids\":[",
           sl->item,
           it->n_prompt);
    for (size_t i = 0; i < sl->n_gen; i++)
        printf("%s%d", i ? "," : "", sl->gen[i]);
    printf("],\"generation_stop\":\"%s\",\"cap\":%zu,\"vocab\":%zu,\"exit_answer\":%d",
           stop,
           cap,
           vocab,
           sl->exit_answer);
    printf(",\"probes\":[");
    for (size_t p = 0; p < sl->n_probes; p++) {
        printf("%s{\"pos\":%zu,\"ms\":%.3f,\"logits\":[",
               p ? "," : "",
               sl->probes[p].pos,
               sl->probes[p].ms);
        for (size_t i = 0; i < it->n_cand; i++)
            printf("%s%.9g", i ? "," : "", (double) sl->probes[p].logits[i]);
        printf("]}");
    }
    printf("]");
    const double surface  = now_ms();
    bool         complete = true;
    printf(",\"generated_utf8_hex\":\"");
    for (size_t i = 0; i < sl->n_gen; i++) {
        if (sl->gen[i] == eos)
            continue;
        const char *piece = geist_session_token_to_str(sl->s, sl->gen[i]);
        if (piece == nullptr || *piece == '\0') {
            complete = false;
            continue;
        }
        for (const unsigned char *c = (const unsigned char *) piece; *c; c++)
            printf("%02x", (unsigned) *c);
    }
    printf("\",\"generated_text_complete\":%s,\"surface_decode_ms\":%.6f,\"prefill_ms\":%.3f,"
           "\"started_ms\":%.3f,\"finished_ms\":%.3f,\"tokens_by_batch_size\":[",
           complete ? "true" : "false",
           now_ms() - surface,
           sl->prefill_ms,
           sl->started - t0,
           now_ms() - t0);
    for (size_t k = 0; k <= MAX_BATCH; k++)
        printf("%s%zu", k ? "," : "", sl->by_size[k]);
    printf("]}\n");
    fflush(stdout);
}

/* Save the state, prefill the probe, read the candidates, restore (#236).
 * The image buffer is the probe's own: the prefix image stays intact. */
[[nodiscard]] static enum geist_status probe_once(struct geist_session *s,
                                                  const struct item    *it,
                                                  size_t                pos,
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
        st = geist_session_prefill_tokens(s, it->n_probe, it->probe);
    size_t       nv     = 0;
    const float *logits = st == GEIST_OK ? geist_session_peek_logits(&nv, s) : nullptr;
    if (st == GEIST_OK && logits == nullptr)
        st = GEIST_E_BACKEND;
    for (size_t i = 0; st == GEIST_OK && i < it->n_cand; i++) {
        if ((size_t) it->cand[i] >= nv) {
            st = GEIST_E_INVALID_ARG;
            break;
        }
        out->logits[i] = logits[it->cand[i]];
    }
    const enum geist_status rs =
            got > 0 ? geist_session_restore(got, *img, s) : GEIST_E_INVALID_STATE;
    out->pos = pos;
    out->ms  = now_ms() - t0;
    return st != GEIST_OK ? st : rs;
}

/* The candidates' top softmax probability and its index. */
static double top_probability(size_t n, const float logits[static n], int *arg) {
    size_t best = 0;
    for (size_t i = 1; i < n; i++)
        best = logits[i] > logits[best] ? i : best;
    double total = 0;
    for (size_t i = 0; i < n; i++)
        total += exp((double) logits[i] - (double) logits[best]);
    *arg = (int) best;
    return 1.0 / total;
}

int main(int argc, char **argv) {
    size_t prompt_cap = 0, cap = 0, batch = 0, interval = 0;
    double tau = 0;
    char  *end = nullptr;
    if ((argc != 6 && argc != 8) || !number(&prompt_cap, argv[3]) || !number(&cap, argv[4]) ||
        !number(&batch, argv[5]) || prompt_cap == 0 || prompt_cap > 1048576 || cap == 0 ||
        cap > 32768 || batch == 0 || batch > MAX_BATCH ||
        (argc == 8 && (!number(&interval, argv[6]) || interval == 0 ||
                       (tau = strtod(argv[7], &end), *end != '\0') || !(tau >= 0 && tau < 1)))) {
        fprintf(stderr,
                "usage: bench_reasoning_batch model.gguf backend prompt_cap cap batch(1..8) "
                "[interval tau]\n");
        return 2;
    }
    /* Read every case first: the driver is fed one chunk at a time. */
    struct item *items   = nullptr;
    size_t       n_items = 0, n_alloc = 0, ncut;
    int          read;
    while ((read = scan_number(&ncut)) == 1) {
        size_t np = 0;
        if (scan_number(&np) != 1 || np == 0 || np > prompt_cap || ncut >= np) {
            fprintf(stderr, "invalid wire input at case %zu\n", n_items);
            return 2;
        }
        if (n_items == n_alloc) {
            n_alloc            = n_alloc ? 2 * n_alloc : 64;
            struct item *grown = realloc(items, n_alloc * sizeof *items);
            if (grown == nullptr)
                return 2;
            items = grown;
        }
        struct item *it = &items[n_items++];
        *it             = (struct item) {.n_cut = ncut, .n_prompt = np};
        if (interval > 0 &&
            (scan_number(&it->n_cand) != 1 || scan_number(&it->n_probe) != 1 || it->n_cand == 0 ||
             it->n_cand > 32 || it->n_probe == 0 || it->n_probe > 64)) {
            fprintf(stderr, "invalid probe header at case %zu\n", n_items - 1);
            return 2;
        }
        it->ids = calloc(np, sizeof *it->ids);
        if (it->ids == nullptr)
            return 2;
        for (size_t i = 0; i < np; i++) {
            size_t id;
            if (scan_number(&id) != 1 || id > INT32_MAX) {
                fprintf(stderr, "invalid token at case %zu\n", n_items - 1);
                return 2;
            }
            it->ids[i] = (geist_token_t) id;
        }
        for (size_t i = 0; i < it->n_cand + it->n_probe; i++) {
            size_t id;
            if (scan_number(&id) != 1 || id > INT32_MAX) {
                fprintf(stderr, "invalid candidate/probe token at case %zu\n", n_items - 1);
                return 2;
            }
            if (i < it->n_cand)
                it->cand[i] = (geist_token_t) id;
            else
                it->probe[i - it->n_cand] = (geist_token_t) id;
        }
    }
    if (read != EOF || n_items == 0) {
        fprintf(stderr, "missing/malformed input\n");
        return 2;
    }
    struct geist_backend           *be = nullptr;
    struct geist_model             *m  = nullptr;
    const struct geist_backend_opts bo = {.log_level_max = GEIST_LOG_ERROR};
    const struct geist_session_opts so = {
            .max_seq_len = prompt_cap + cap + 2, .top_p = 1.0f, .kv_mode = GEIST_KV_FP32};
    struct geist_session *prep             = nullptr;
    struct slot           slots[MAX_BATCH] = {};
    int                   exit_code        = 1;
    void                 *img = nullptr, *pimg = nullptr;
    size_t                img_cap = 0, img_n = 0, pimg_cap = 0, cached_case = SIZE_MAX;
    double                probe_ms       = 0;
    size_t                n_probes_total = 0;
    if (geist_backend_create(argv[2], &bo, nullptr, &be) != GEIST_OK ||
        geist_model_load_with_opts(argv[1], be, &so, &m) != GEIST_OK ||
        geist_session_create(m, be, &so, &prep) != GEIST_OK) {
        fprintf(stderr, "setup: %s\n", geist_last_create_error());
        goto done;
    }
    const size_t max = geist_model_decode_batch_max(m);
    if (batch > 1 && batch > max) {
        fprintf(stderr, "decode_batch_max %zu < batch %zu\n", max, batch);
        goto done;
    }
    for (size_t b = 0; b < batch; b++) {
        slots[b].gen    = calloc(cap, sizeof *slots[b].gen);
        slots[b].probes = calloc(interval ? cap / interval + 2 : 1, sizeof *slots[b].probes);
        if (slots[b].gen == nullptr || slots[b].probes == nullptr ||
            geist_session_create(m, be, &so, &slots[b].s) != GEIST_OK) {
            fprintf(stderr, "session: %s\n", geist_last_create_error());
            goto done;
        }
    }
    const geist_token_t eos = geist_model_eos_token(m);
    printf("{\"kind\":\"runtime\",\"version\":\"%s\",\"backend\":\"%s\",\"kv_mode\":\"fp32\","
           "\"decode_batch_max\":%zu,\"batch\":%zu,\"eos\":%d,\"interval\":%zu,\"tau\":%.6g}\n",
           geist_version_string(),
           geist_backend_name(be),
           max,
           batch,
           eos,
           interval,
           tau);
    fflush(stdout);
    const double t0   = now_ms();
    size_t       next = 0, steps = 0, tokens = 0, steps_by_size[MAX_BATCH + 1] = {};
    double       decode_ms = 0;
    for (;;) {
        /* Fill free slots with the next cases. */
        for (size_t b = 0; b < batch && next < n_items; b++) {
            if (slots[b].active)
                continue;
            const struct item *it = &items[next];
            const double       tp = now_ms();
            enum geist_status  s  = GEIST_OK;
            if (it->n_cut > 0 &&
                (cached_case == SIZE_MAX || items[cached_case].n_cut != it->n_cut ||
                 memcmp(items[cached_case].ids, it->ids, it->n_cut * sizeof *it->ids) != 0)) {
                const double tq   = now_ms();
                size_t       need = 0;
                s                 = geist_session_reset(prep);
                if (s == GEIST_OK)
                    s = geist_session_prefill_tokens(prep, it->n_cut, it->ids);
                if (s == GEIST_OK)
                    s = geist_session_snapshot_size(&need, prep);
                if (s == GEIST_OK && need > img_cap) {
                    void *grown = realloc(img, need);
                    s           = grown != nullptr ? GEIST_OK : GEIST_E_OOM;
                    if (grown != nullptr)
                        img = grown, img_cap = need;
                }
                if (s == GEIST_OK)
                    s = geist_session_snapshot(&img_n, img_cap, img, prep);
                if (s != GEIST_OK) {
                    fprintf(stderr, "case %zu prefix: %s\n", next, geist_status_to_string(s));
                    goto done;
                }
                cached_case = next;
                printf("{\"kind\":\"prefix_prepare\",\"case_index\":%zu,\"elapsed_ms\":%.3f,"
                       "\"cached_tokens\":%zu,\"snapshot_bytes\":%zu}\n",
                       next,
                       now_ms() - tq,
                       it->n_cut,
                       img_n);
                fflush(stdout);
            }
            s = it->n_cut > 0 ? geist_session_restore(img_n, img, slots[b].s)
                              : geist_session_reset(slots[b].s);
            if (s == GEIST_OK)
                s = geist_session_prefill_tokens(
                        slots[b].s, it->n_prompt - it->n_cut, it->ids + it->n_cut);
            if (s == GEIST_OK && vocab == 0) {
                size_t nl = 0;
                (void) geist_session_peek_logits(&nl, slots[b].s);
                vocab = nl;
            }
            if (s != GEIST_OK) {
                fprintf(stderr,
                        "case %zu prefill: %s: %s\n",
                        next,
                        geist_status_to_string(s),
                        geist_session_errmsg(slots[b].s));
                goto done;
            }
            slots[b] = (struct slot) {.s           = slots[b].s,
                                      .gen         = slots[b].gen,
                                      .probes      = slots[b].probes,
                                      .active      = true,
                                      .item        = next,
                                      .started     = tp,
                                      .prefill_ms  = now_ms() - tp,
                                      .exit_answer = -1};
            if (interval > 0) {
                s = probe_once(slots[b].s, it, 0, &pimg, &pimg_cap, &slots[b].probes[0]);
                if (s != GEIST_OK) {
                    fprintf(stderr, "case %zu probe: %s\n", next, geist_status_to_string(s));
                    goto done;
                }
                slots[b].n_probes = 1;
                probe_ms += slots[b].probes[0].ms, n_probes_total++;
            }
            next++;
        }
        struct geist_session *act[MAX_BATCH];
        size_t                idx[MAX_BATCH], n = 0;
        for (size_t b = 0; b < batch; b++)
            if (slots[b].active)
                act[n] = slots[b].s, idx[n++] = b;
        if (n == 0)
            break;
        geist_token_t     out[MAX_BATCH];
        const double      ts = now_ms();
        enum geist_status s  = geist_session_decode_batch(n, act, out);
        decode_ms += now_ms() - ts;
        if (s != GEIST_OK) {
            fprintf(stderr, "decode_batch(%zu): %s\n", n, geist_status_to_string(s));
            goto done;
        }
        steps++, steps_by_size[n]++, tokens += n;
        for (size_t k = 0; k < n; k++) {
            struct slot *sl      = &slots[idx[k]];
            sl->gen[sl->n_gen++] = out[k];
            sl->by_size[n]++;
            if (out[k] == eos || sl->n_gen == cap) {
                emit_sample(sl, &items[sl->item], cap, eos, t0);
                sl->active = false;
                continue;
            }
            if (interval == 0 || sl->n_gen % interval != 0)
                continue;
            struct probe *pr = &sl->probes[sl->n_probes];
            s                = probe_once(sl->s, &items[sl->item], sl->n_gen, &pimg, &pimg_cap, pr);
            if (s != GEIST_OK) {
                fprintf(stderr, "case %zu probe: %s\n", sl->item, geist_status_to_string(s));
                goto done;
            }
            sl->n_probes++;
            probe_ms += pr->ms, n_probes_total++;
            int arg = -1;
            if (tau > 0 && top_probability(items[sl->item].n_cand, pr->logits, &arg) >= tau) {
                sl->exit_answer = arg;
                emit_sample(sl, &items[sl->item], cap, eos, t0);
                sl->active = false;
            }
        }
    }
    printf("{\"kind\":\"summary\",\"steps\":%zu,\"decode_ms\":%.3f,\"tokens\":%zu,\"wall_ms\":%.3f,"
           "\"probes\":%zu,\"probe_ms\":%.3f,\"steps_by_batch_size\":[",
           steps,
           decode_ms,
           tokens,
           now_ms() - t0,
           n_probes_total,
           probe_ms);
    for (size_t k = 0; k <= MAX_BATCH; k++)
        printf("%s%zu", k ? "," : "", steps_by_size[k]);
    printf("]}\n");
    exit_code = 0;
done:
    for (size_t b = 0; b < MAX_BATCH; b++) {
        geist_session_destroy(slots[b].s);
        free(slots[b].gen);
        free(slots[b].probes);
    }
    for (size_t i = 0; i < n_items; i++)
        free(items[i].ids);
    free(items);
    free(img);
    free(pimg);
    geist_session_destroy(prep);
    geist_model_destroy(m);
    geist_backend_destroy(be);
    return exit_code;
}

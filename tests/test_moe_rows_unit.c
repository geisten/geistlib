/*
 * test_moe_rows_unit - a decode step's experts split over threads by
 * output rows (moe_split) give the same bits as one expert per thread,
 * also where the backend repacks the weights (#374).
 *
 * 1. The row-range entry (geist_weight::linear_m1_rows) at the expert
 *    shapes of DeepSeek-V2-Lite (gate / up 2048 -> 1408) and Qwen3-30B-A3B
 *    (gate / up 2048 -> 768, down 768 -> 2048): on every CPU backend that
 *    installs one (cpu_x86: Q4_Kx8 and Q6_K), each 32-row slice moe_split
 *    would cut is bit-identical to the same rows of linear_m1, and the
 *    whole is within Q8_K activation rounding of cpu_scalar's exact
 *    dequant.
 * 2. mf_qwen3moe_gguf with Q4_K experts (d_model 256, expert width 256, 2
 *    of 8 experts per token), fed one token at a time: the logits with the
 *    split off (GEIST_MOE_SPLIT=0) and on are bit-identical on every CPU
 *    backend in the build.
 */
#include "heap.h"
#include "model_fixtures.h"
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>
#include <geist_weight.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { VOCAB = 96, N_PROMPT = 12 };

/* ---- 1. the row-range entry ------------------------------------------- */

static uint32_t rng(uint32_t *s) {
    *s ^= *s << 13;
    *s ^= *s >> 17;
    *s ^= *s << 5;
    return *s;
}

/* Random Q4_K (dtype 12, 144-byte blocks) or Q6_K (14, 210-byte blocks)
 * with small positive scales. */
static uint8_t *random_kquant(enum geist_dtype dt, size_t n_in, size_t n_out, size_t *bytes) {
    const size_t bb = dt == GEIST_DTYPE_Q4_K ? 144 : 210, nb = n_in / 256 * n_out;
    uint8_t     *p = malloc(nb * bb);
    uint32_t     s = 0x9e3779b9u ^ (uint32_t) (n_in * 31 + n_out);
    if (p == nullptr)
        return nullptr;
    for (size_t b = 0; b < nb; b++) {
        uint8_t *blk = p + b * bb;
        for (size_t k = 0; k < bb; k++)
            blk[k] = (uint8_t) (rng(&s) >> 24);
        const _Float16 d = (_Float16) ((1.0f + (float) (rng(&s) % 1024) / 1024.0f) / 2048.0f);
        if (dt == GEIST_DTYPE_Q4_K) {
            memcpy(blk, &d, 2);     /* d */
            memcpy(blk + 2, &d, 2); /* dmin */
        } else {
            memcpy(blk + 208, &d, 2); /* d, after ql / qh / scales */
        }
    }
    *bytes = nb * bb;
    return p;
}

static int check_rows(const char *name, enum geist_dtype dt, size_t n_in, size_t n_out) {
    struct geist_backend *be = nullptr, *ref = nullptr;
    if (geist_backend_create(name, nullptr, nullptr, &be) != GEIST_OK)
        return 0;
    int      fails = 0;
    size_t   bytes = 0;
    uint8_t *raw   = random_kquant(dt, n_in, n_out, &bytes);
    float   *x = malloc(n_in * sizeof *x), *y = malloc(n_out * sizeof *y);
    float   *ys = malloc(n_out * sizeof *ys), *yr = malloc(n_out * sizeof *yr);
    if (raw == nullptr || x == nullptr || y == nullptr || ys == nullptr || yr == nullptr ||
        geist_backend_create("cpu_scalar", nullptr, nullptr, &ref) != GEIST_OK) {
        fails++;
        goto out;
    }
    uint32_t s = 7;
    for (size_t i = 0; i < n_in; i++)
        x[i] = (float) ((int32_t) rng(&s)) / 2147483648.0f;
    const struct geist_weight src = {.raw        = raw,
                                     .raw_nbytes = bytes,
                                     .n_in       = (int32_t) n_in,
                                     .n_out      = (int32_t) n_out,
                                     .dtype      = (uint16_t) dt};
    struct geist_weight       w = src, wr = src;
    if (be->desc->vtbl->resolve_weight(be, &w) != GEIST_OK ||
        ref->desc->vtbl->resolve_weight(ref, &wr) != GEIST_OK) {
        fails += geist_expect(false, "the weight resolves");
        goto out;
    }
    if (w.linear_m1_rows == nullptr) {
        printf("%s %s %zux%zu: no row-range entry\n",
               name,
               dt == GEIST_DTYPE_Q4_K ? "Q4_K" : "Q6_K",
               n_in,
               n_out);
        goto free_aux;
    }
    fails += geist_expect(w.linear_m1_rows_align >= 1 && 32 % w.linear_m1_rows_align == 0,
                          "the granularity divides moe_split's 32-row slices");
    w.linear_m1(x, &w, be, y);
    wr.linear_m1(x, &wr, ref, yr);
    for (size_t parts = 2; parts <= 16; parts *= 2) {
        const size_t chunk = ((n_out + parts - 1) / parts + 31) / 32 * 32;
        memset(ys, 0, n_out * sizeof *ys);
        for (size_t lo = 0; lo < n_out; lo += chunk) {
            const size_t hi = lo + chunk < n_out ? lo + chunk : n_out;
            w.linear_m1_rows(lo, hi, x, &w, be, ys + lo);
        }
        fails += geist_expect(memcmp(y, ys, n_out * sizeof *y) == 0,
                              "row slices: the same bits as linear_m1");
    }
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n_out; i++) {
        num += ((double) y[i] - yr[i]) * ((double) y[i] - yr[i]);
        den += (double) yr[i] * yr[i];
    }
    const double rel = den > 0.0 ? sqrt(num / den) : 0.0;
    fails += geist_expect(rel < 2e-2, "within activation rounding of cpu_scalar");
    printf("%s %s %zux%zu: slices bit-identical, rel L2 to cpu_scalar %.2e\n",
           name,
           dt == GEIST_DTYPE_Q4_K ? "Q4_K" : "Q6_K",
           n_in,
           n_out,
           rel);
free_aux:
    for (struct geist_weight *p = &w; p != nullptr; p = p == &w ? &wr : nullptr) {
        if ((p->flags & GEIST_W_AUX_HEAP_OWNED) != 0) {
            void *aux = (void *) p->aux_fp32;
            safe_free(&aux);
        }
    }
out:
    free(raw);
    free(x);
    free(y);
    free(ys);
    free(yr);
    geist_backend_destroy(ref);
    geist_backend_destroy(be);
    return fails;
}

/* ---- 2. a model's decode steps ---------------------------------------- */

static const struct mf_qwen3moe GEOM = {.layers      = 2,
                                        .d_model     = 256,
                                        .heads       = 4,
                                        .kv_heads    = 2,
                                        .head_dim    = 64,
                                        .experts     = 8,
                                        .used        = 2,
                                        .expert_ff   = 256,
                                        .vocab       = VOCAB,
                                        .seed        = 31,
                                        .q4k_experts = true};

static const struct geist_session_opts OPTS = {.top_p = 1.0f, .m_max = 32, .max_seq_len = 64};

/* The prompt one token at a time on backend `name` with GEIST_MOE_SPLIT
 * set to `split` (nullptr: unset): its last logits into `out`. */
static bool run(const char *name, const char *split, float out[static VOCAB]) {
    if (split != nullptr)
        setenv("GEIST_MOE_SPLIT", split, 1);
    else
        unsetenv("GEIST_MOE_SPLIT");
    struct geist_backend *be = nullptr;
    if (geist_backend_create(name, nullptr, nullptr, &be) != GEIST_OK)
        return false;
    struct tf_buf         g  = mf_qwen3moe_gguf(&GEOM);
    struct geist_model   *m  = nullptr;
    struct geist_session *s  = nullptr;
    bool                  ok = geist_model_load_from_memory(g.b, g.n, be, &m) == GEIST_OK &&
                               geist_session_create(m, be, &OPTS, &s) == GEIST_OK;
    for (size_t i = 0; ok && i < N_PROMPT; i++) {
        const geist_token_t id = (geist_token_t) ((i * 37 + 5) % VOCAB);
        ok                     = geist_session_prefill_tokens(s, 1, &id) == GEIST_OK;
    }
    if (!ok && be != nullptr)
        fprintf(stderr,
                "%s: %s\n",
                name,
                s != nullptr ? geist_session_errmsg(s) : geist_last_create_error());
    size_t       nl = 0;
    const float *lg = ok ? geist_session_peek_logits(&nl, s) : nullptr;
    ok              = lg != nullptr && nl == VOCAB;
    if (ok)
        memcpy(out, lg, VOCAB * sizeof(float));
    geist_session_destroy(s);
    geist_model_destroy(m);
    geist_backend_destroy(be);
    free(g.b);
    return ok;
}

int main(void) {
    setenv("GEIST_MOE_PROFILE_CACHE", "0", 1);
    int fails = 0, ran = 0;
    for (const char *const *b =
                 (const char *const[]) {"cpu_scalar", "cpu_neon", "cpu_x86", nullptr};
         *b != nullptr;
         b++) {
        float off[VOCAB], on[VOCAB];
        if (!run(*b, "0", off)) {
            printf("skip %s: not in this build\n", *b);
            continue;
        }
        ran++;
        fails += check_rows(*b, GEIST_DTYPE_Q4_K, 2048, 1408);
        fails += check_rows(*b, GEIST_DTYPE_Q4_K, 2048, 768);
        fails += check_rows(*b, GEIST_DTYPE_Q6_K, 768, 2048);
        fails += geist_expect(run(*b, nullptr, on), "the split runs");
        fails += geist_expect(memcmp(off, on, sizeof off) == 0,
                              "decode steps: the same bits with the split on and off");
        printf("%s: checked\n", *b);
    }
    unsetenv("GEIST_MOE_SPLIT");
    if (ran == 0) {
        fprintf(stderr, "FAIL: no CPU backend in this build\n");
        return 1;
    }
    if (fails == 0)
        printf("PASS: row-split MoE experts are bit-identical to unsplit ones\n");
    return fails == 0 ? 0 : 1;
}

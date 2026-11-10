/*
 * moe_hot_check.h - GPU mode with hot experts for a family's in-memory
 * fixture (#241), the checks test_qwen3moe_unit runs for Qwen3-MoE (#78):
 * the model loaded on a GPU backend, GEIST_MOE_HOT experts with a copy
 * there, the others on the state's own CPU backend. A prompt prefilled in
 * chunks and one token at a time:
 *   - the experts have a CPU home, `hot` of them are placed on the GPU and
 *     rows run there (hot > 0), the cold ones' rows on the CPU (hot below
 *     the total);
 *   - the logits match cpu_scalar's `ref`: Metal within relative L2 1e-3
 *     (its CPU experts are cpu_neon's, exact on F32 weights), other GPUs by
 *     cosine 0.999 (cpu_x86 runs F32 experts as W8A8).
 *
 * Header-only, for tests that include arch_state.h, moe_gpu.h and model.h.
 */
#ifndef GEIST_TESTS_MOE_HOT_CHECK_H
#define GEIST_TESTS_MOE_HOT_CHECK_H

#include "../src/archs/transformer/arch_state.h"
#include "../src/archs/transformer/moe_gpu.h"
#include "../src/engine/model.h"
#include "model_fixtures.h"
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The next-token logits after `ids` prefilled `step` tokens at a time. */
static inline bool moe_hot_logits(struct geist_session *s,
                                  size_t                n_vocab,
                                  size_t                n_ids,
                                  size_t                step,
                                  const geist_token_t   ids[static n_ids],
                                  float                 out[static n_vocab]) {
    if (geist_session_reset(s) != GEIST_OK)
        return false;
    for (size_t i = 0; i < n_ids; i += step) {
        const size_t k = n_ids - i < step ? n_ids - i : step;
        if (geist_session_prefill_tokens(s, k, ids + i) != GEIST_OK)
            return false;
    }
    size_t       nl = 0;
    const float *lg = geist_session_peek_logits(&nl, s);
    if (lg == nullptr || nl != n_vocab)
        return false;
    memcpy(out, lg, n_vocab * sizeof(float));
    return true;
}

/* rel_l2 >= 0 and cosine of a against b over n. */
static inline void moe_hot_compare(
        size_t n, const float a[static n], const float b[static n], double *rel, double *cos) {
    double num = 0.0, den = 0.0, ab = 0.0, aa = 0.0;
    for (size_t i = 0; i < n; i++) {
        num += ((double) a[i] - b[i]) * ((double) a[i] - b[i]);
        den += (double) b[i] * b[i];
        ab += (double) a[i] * b[i];
        aa += (double) a[i] * a[i];
    }
    *rel = sqrt(num / (den > 0.0 ? den : 1.0));
    *cos = ab / sqrt(aa * den > 0.0 ? aa * den : 1.0);
}

/* The fixture `blob` on GPU backend `gpu` with `hot` of its `total` experts
 * (layers times experts per layer) on the GPU. Returns failures, or -1 when
 * `gpu` is not in this build. */
static inline int moe_hot_check(const char                      *gpu,
                                const struct tf_buf             *blob,
                                const struct geist_session_opts *opts,
                                size_t                           hot,
                                size_t                           total,
                                size_t                           n_vocab,
                                size_t                           n_ids,
                                const geist_token_t              ids[static n_ids],
                                const float                      ref[static n_vocab]) {
    struct geist_backend *gbe = nullptr;
    if (geist_backend_create(gpu, nullptr, nullptr, &gbe) != GEIST_OK) {
        printf("skip hot experts on %s: not in this build\n", gpu);
        return -1;
    }
    char n[16];
    snprintf(n, sizeof n, "%zu", hot);
    setenv("GEIST_MOE_HOT", n, 1);
    struct geist_model   *m       = nullptr;
    struct geist_session *s       = nullptr;
    int                   fails   = 0;
    float                *chunked = malloc(n_vocab * sizeof *chunked);
    float                *single  = malloc(n_vocab * sizeof *single);
    if (chunked == nullptr || single == nullptr ||
        geist_model_load_from_memory(blob->b, blob->n, gbe, &m) != GEIST_OK ||
        geist_session_create(m, gbe, opts, &s) != GEIST_OK ||
        !moe_hot_logits(s, n_vocab, n_ids, n_ids, ids, chunked) ||
        !moe_hot_logits(s, n_vocab, n_ids, 1, ids, single)) {
        fprintf(stderr, "  %s: %s\n", gpu, geist_backend_errmsg(gbe));
        fails += geist_expect(false, "the fixture runs on the GPU with hot experts");
    } else {
        const struct transformer_arch_state *st = tf_state(geist_model_internal_arch_meta(m));
        const struct transformer_moe_gpu    *mg = st->moe_gpu;
        double                               r1, c1, r2, c2;
        moe_hot_compare(n_vocab, chunked, ref, &r1, &c1);
        moe_hot_compare(n_vocab, single, ref, &r2, &c2);
        printf("%s, %zu of %zu experts hot: %zu placed, %llu rows on the GPU, %llu on the CPU; "
               "vs cpu_scalar relative L2 %.3g / %.3g, cosine %.6f / %.6f (chunked / token by "
               "token)\n",
               gpu,
               hot,
               total,
               mg != nullptr ? mg->n_hot : 0,
               mg != nullptr ? (unsigned long long) mg->rows_gpu : 0ULL,
               mg != nullptr ? (unsigned long long) mg->rows_cpu : 0ULL,
               r1,
               r2,
               c1,
               c2);
        fails += geist_expect(st->moe_cpu != nullptr, "the experts have a CPU home");
        fails += geist_expect(hot == 0 || (mg != nullptr && mg->n_hot == hot && mg->rows_gpu > 0),
                              "the hot experts are placed and run on the GPU");
        fails += geist_expect(hot == total || (mg != nullptr && mg->rows_cpu > 0) ||
                                      (mg == nullptr && hot == 0),
                              "the cold experts' rows run on the CPU");
        const bool exact = strcmp(gpu, "metal") == 0;
        fails += geist_expect(exact ? r1 <= 1e-3 && r2 <= 1e-3 : c1 >= 0.999 && c2 >= 0.999,
                              "the logits match cpu_scalar's");
    }
    unsetenv("GEIST_MOE_HOT");
    free(chunked);
    free(single);
    geist_session_destroy(s);
    geist_model_destroy(m);
    geist_backend_destroy(gbe);
    return fails;
}

/* moe_hot_check on Metal and Vulkan with none, half and all of the experts
 * hot. Returns failures. */
static inline int moe_hot_checks(const struct tf_buf             *blob,
                                 const struct geist_session_opts *opts,
                                 size_t                           total,
                                 size_t                           n_vocab,
                                 size_t                           n_ids,
                                 const geist_token_t              ids[static n_ids],
                                 const float                      ref[static n_vocab]) {
    static const char *const gpus[] = {"metal", "vulkan"};
    int                      fails  = 0;
    for (size_t g = 0; g < sizeof gpus / sizeof gpus[0]; g++) {
        const size_t hots[] = {0, total / 2, total};
        for (size_t h = 0; h < sizeof hots / sizeof hots[0]; h++) {
            const int r =
                    moe_hot_check(gpus[g], blob, opts, hots[h], total, n_vocab, n_ids, ids, ref);
            if (r < 0)
                break;
            fails += r;
        }
    }
    return fails;
}

#endif /* GEIST_TESTS_MOE_HOT_CHECK_H */

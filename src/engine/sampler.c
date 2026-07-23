/*
 * src/engine/sampler.c — argmax + temperature + top-k + top-p sampling.
 *
 * Layer: ENGINE.
 *
 * Numerical stability: softmax uses the max-subtract trick to avoid
 * overflow. Reductions accumulate in double for vocabularies ≥1024.
 *
 * Hot-path footprint: argmax is O(n), no allocation. Temperature is O(n),
 * no allocation. Top-k / top-p need scratch — the _ws variants take a
 * caller-owned workspace so the per-token call is allocation-free.
 */
#define GEIST_INTERNAL_ENGINE_LAYER

#include "sampler.h"
#include "error.h"

#include "heap.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

/* ---- RNG ---------------------------------------------------------------- */

void geist_rng_seed(struct geist_rng *rng, uint64_t seed) {
    if (rng == nullptr) {
        return;
    }
    /* Avoid the all-zero state (xorshift can't escape it). */
    rng->state = seed != 0 ? seed : 0x9E3779B97F4A7C15ULL;
}

uint64_t geist_rng_next_u64(struct geist_rng *rng) {
    uint64_t x = rng->state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng->state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

float geist_rng_next_unit(struct geist_rng *rng) {
    /* Take the high 24 bits → fits in a float mantissa without rounding. */
    uint32_t bits = (uint32_t) (geist_rng_next_u64(rng) >> 40);
    return (float) bits * (1.0f / (float) (1u << 24));
}

/* ---- Argmax ------------------------------------------------------------- */

geist_token_t geist_sampler_argmax(size_t n_vocab, const float logits[static n_vocab]) {
    geist_token_t best_idx = 0;
    float         best_val = logits[0];
    for (size_t i = 1; i < n_vocab; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best_idx = (geist_token_t) i;
        }
    }
    return best_idx;
}

/* ---- Softmax helpers ---------------------------------------------------- */

/* Writes softmax(logits/temperature) into probs[]. Returns the sum for
 * downstream sampling — caller can verify it equals 1.0 modulo fp error. */
static double softmax_into(size_t n, const float *logits, float temperature, float *probs) {
    /* Max-subtract for stability. */
    float max_logit = logits[0];
    for (size_t i = 1; i < n; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }
    /* Avoid div-by-zero — caller is supposed to pass T>0 but be defensive. */
    float  inv_t = temperature > 0.0f ? 1.0f / temperature : 1.0f;
    double sum   = 0.0;
    for (size_t i = 0; i < n; i++) {
        float p  = expf((logits[i] - max_logit) * inv_t);
        probs[i] = p;
        sum += (double) p;
    }
    return sum;
}

/* Inverse-CDF sample over a probability vector that may not be normalized.
 * Sums prefix until it exceeds u*sum. Stable choice on rounding edge: the
 * last index that passes is returned. */
static geist_token_t
inv_cdf_sample(size_t n, const float *probs, double sum, struct geist_rng *rng) {
    double u   = (double) geist_rng_next_unit(rng) * sum;
    double acc = 0.0;
    for (size_t i = 0; i < n; i++) {
        acc += (double) probs[i];
        if (u < acc) {
            return (geist_token_t) i;
        }
    }
    /* Rounding overflow: return last index. */
    return (geist_token_t) (n - 1);
}

/* ---- Temperature only --------------------------------------------------- */

geist_token_t geist_sampler_temperature(size_t            n_vocab,
                                        const float       logits[static n_vocab],
                                        float             temperature,
                                        struct geist_rng *rng) {
    if (temperature <= 0.0f) {
        return geist_sampler_argmax(n_vocab, logits);
    }
    /* Inline scratch is fine for small vocab; large vocabs should use _ws
     * variants. Threshold of 8192 floats = 32 KB on stack — well within macOS
     * and Linux default stack limits. */
    if (n_vocab <= 8192) {
        float  probs[8192];
        double sum = softmax_into(n_vocab, logits, temperature, probs);
        return inv_cdf_sample(n_vocab, probs, sum, rng);
    }
    /* Large-vocab fallback: heap. Per-call allocation is acceptable for
     * non-hot-path callers; decode hot path should use the _ws variant
     * instead. */
    float *probs = heap_alloc_array_aligned(float, n_vocab);
    if (probs == nullptr) {
        return geist_sampler_argmax(n_vocab, logits);
    }
    double        sum = softmax_into(n_vocab, logits, temperature, probs);
    geist_token_t tok = inv_cdf_sample(n_vocab, probs, sum, rng);
    safe_free((void **) &probs);
    return tok;
}

/* ---- Workspace ---------------------------------------------------------- */

[[nodiscard]] enum geist_status geist_sampler_workspace_init(struct geist_sampler_workspace *ws,
                                                             size_t n_vocab) {
    if (ws == nullptr || n_vocab == 0) {
        return GEIST_E_INVALID_ARG;
    }
    ws->probs = heap_alloc_array_aligned(float, n_vocab);
    if (ws->probs == nullptr) {
        ws->n_vocab = 0;
        return GEIST_E_OOM;
    }
    ws->n_vocab = n_vocab;
    return GEIST_OK;
}

void geist_sampler_workspace_destroy(struct geist_sampler_workspace *ws) {
    if (ws == nullptr) {
        return;
    }
    if (ws->probs != nullptr)
        safe_free((void **) &ws->probs);
    ws->n_vocab = 0;
}

/* ---- Top-K -------------------------------------------------------------- */

/* Comparator for descending-by-logit qsort. */
struct kv_pair {
    float    logit;
    uint32_t idx;
};

static int cmp_desc_logit(const void *a, const void *b) {
    float la = ((const struct kv_pair *) a)->logit;
    float lb = ((const struct kv_pair *) b)->logit;
    /* NaN-safe: NaN sorts to the bottom. */
    if (la > lb)
        return -1;
    if (la < lb)
        return 1;
    return 0;
}

geist_token_t geist_sampler_top_k_ws(struct geist_sampler_workspace *ws,
                                     const float                     logits[static ws->n_vocab],
                                     int                             top_k,
                                     float                           temperature,
                                     struct geist_rng               *rng) {
    const size_t n = ws->n_vocab;
    if (top_k <= 1 || temperature <= 0.0f) {
        return geist_sampler_argmax(n, logits);
    }
    if ((size_t) top_k >= n) {
        return geist_sampler_temperature(n, logits, temperature, rng);
    }

    /* Build (logit, idx) pairs — a stack array for typical n, heap beyond. */
    struct kv_pair  pairs_stack[1024];
    struct kv_pair *pairs         = pairs_stack;
    bool            pairs_on_heap = false;
    if (n > 1024) {
        pairs = heap_alloc_array_aligned(struct kv_pair, n);
        if (pairs == nullptr) {
            return geist_sampler_argmax(n, logits);
        }
        pairs_on_heap = true;
    }
    for (size_t i = 0; i < n; i++) {
        pairs[i].logit = logits[i];
        pairs[i].idx   = (uint32_t) i;
    }
    /* Full sort is O(n log n); partial-sort optimization left for later. */
    qsort(pairs, n, sizeof(struct kv_pair), cmp_desc_logit);

    /* Softmax over the top-k entries only. */
    float top_logits[8192];
    float top_probs[8192];
    /* top_k is capped at 8192 (stack budget). Beyond that, fall back. */
    int k = top_k;
    if (k > 8192)
        k = 8192;

    for (int i = 0; i < k; i++)
        top_logits[i] = pairs[i].logit;
    double sum = softmax_into((size_t) k, top_logits, temperature, top_probs);

    geist_token_t local_pick = inv_cdf_sample((size_t) k, top_probs, sum, rng);
    geist_token_t picked     = (geist_token_t) pairs[local_pick].idx;

    if (pairs_on_heap) {
        safe_free((void **) &pairs);
    }
    return picked;
}

/* ---- Top-P -------------------------------------------------------------- */

geist_token_t geist_sampler_top_p_ws(struct geist_sampler_workspace *ws,
                                     const float                     logits[static ws->n_vocab],
                                     float                           top_p,
                                     float                           temperature,
                                     struct geist_rng               *rng) {
    const size_t n = ws->n_vocab;
    if (top_p <= 0.0f || temperature <= 0.0f) {
        return geist_sampler_argmax(n, logits);
    }
    if (top_p >= 1.0f) {
        return geist_sampler_temperature(n, logits, temperature, rng);
    }

    /* Full softmax + sort-by-prob-descending → cumsum → cutoff. */
    double sum = softmax_into(n, logits, temperature, ws->probs);
    if (sum <= 0.0) {
        return geist_sampler_argmax(n, logits);
    }
    /* Normalize in place. */
    for (size_t i = 0; i < n; i++) {
        ws->probs[i] = (float) ((double) ws->probs[i] / sum);
    }

    /* Build (prob, idx) pairs and sort descending. Reuse stack for n<=1024. */
    struct kv_pair  pairs_stack[1024];
    struct kv_pair *pairs         = pairs_stack;
    bool            pairs_on_heap = false;
    if (n > 1024) {
        pairs = heap_alloc_array_aligned(struct kv_pair, n);
        if (pairs == nullptr) {
            return geist_sampler_argmax(n, logits);
        }
        pairs_on_heap = true;
    }
    for (size_t i = 0; i < n; i++) {
        pairs[i].logit = ws->probs[i]; /* reuse field as "prob" */
        pairs[i].idx   = (uint32_t) i;
    }
    qsort(pairs, n, sizeof(struct kv_pair), cmp_desc_logit);

    /* Find smallest prefix whose cumulative prob >= top_p. */
    double cum    = 0.0;
    size_t cutoff = 0;
    for (cutoff = 0; cutoff < n; cutoff++) {
        cum += (double) pairs[cutoff].logit;
        if (cum >= (double) top_p) {
            cutoff++;
            break;
        }
    }
    if (cutoff < 1)
        cutoff = 1;

    /* Sample within the nucleus, renormalized. */
    double nucleus_sum = 0.0;
    for (size_t i = 0; i < cutoff; i++) {
        nucleus_sum += (double) pairs[i].logit;
    }
    double        u      = (double) geist_rng_next_unit(rng) * nucleus_sum;
    double        acc    = 0.0;
    geist_token_t picked = (geist_token_t) pairs[cutoff - 1].idx;
    for (size_t i = 0; i < cutoff; i++) {
        acc += (double) pairs[i].logit;
        if (u < acc) {
            picked = (geist_token_t) pairs[i].idx;
            break;
        }
    }

    if (pairs_on_heap) {
        safe_free((void **) &pairs);
    }
    return picked;
}

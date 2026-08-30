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
 * caller-owned workspace so the per-token call is allocation-free (#331).
 * Selection is a bounded min-heap, O(n log k) with k = the requested top-k
 * or the nucleus size, not a full O(n log n) sort of the vocabulary.
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
    ws->pairs = heap_alloc_array_aligned(struct geist_sampler_pair, n_vocab);
    if (ws->probs == nullptr || ws->pairs == nullptr) {
        geist_sampler_workspace_destroy(ws);
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
    if (ws->pairs != nullptr)
        safe_free((void **) &ws->pairs);
    ws->n_vocab = 0;
}

/* ---- Selection --------------------------------------------------------- */

/* Total order over pairs: higher score first, lower index on ties. NaN
 * scores are mapped to -inf on the way in, so the order is a real one
 * (qsort's NaN-"equal" comparator was not). */
static inline bool pair_worse(struct geist_sampler_pair a, struct geist_sampler_pair b) {
    if (a.score != b.score) {
        return a.score < b.score;
    }
    return a.idx > b.idx;
}

static void sift_down(struct geist_sampler_pair *heap, size_t n, size_t root) {
    for (;;) {
        size_t worst = root;
        size_t left  = 2 * root + 1;
        if (left < n && pair_worse(heap[left], heap[worst])) {
            worst = left;
        }
        if (left + 1 < n && pair_worse(heap[left + 1], heap[worst])) {
            worst = left + 1;
        }
        if (worst == root) {
            return;
        }
        struct geist_sampler_pair tmp = heap[root];
        heap[root]                    = heap[worst];
        heap[worst]                   = tmp;
        root                          = worst;
    }
}

/* Writes the k highest-scoring entries of scores[0..n) into out[0..k),
 * sorted descending. O(n log k): a min-heap of size k, then heapsort of
 * that heap. Requires 1 <= k <= n. */
static void
select_top_desc(size_t n, const float scores[static n], size_t k, struct geist_sampler_pair *out) {
    for (size_t i = 0; i < k; i++) {
        float s = scores[i];
        out[i]  = (struct geist_sampler_pair) {isnan(s) ? -INFINITY : s, (uint32_t) i};
    }
    for (size_t i = k / 2; i-- > 0;) {
        sift_down(out, k, i);
    }
    for (size_t i = k; i < n; i++) {
        float s = scores[i];
        if (isnan(s)) {
            s = -INFINITY;
        }
        /* Cheap reject first: the tail of a logit vector loses here almost
         * always, so this branch is what the loop actually costs. */
        if (s < out[0].score) {
            continue;
        }
        struct geist_sampler_pair cand = {s, (uint32_t) i};
        if (pair_worse(cand, out[0])) {
            continue;
        }
        out[0] = cand;
        sift_down(out, k, 0);
    }
    /* Heapsort the min-heap in place — extracting the minimum to the back
     * repeatedly leaves the array in descending order. */
    for (size_t m = k; m > 1; m--) {
        struct geist_sampler_pair tmp = out[0];
        out[0]                        = out[m - 1];
        out[m - 1]                    = tmp;
        sift_down(out, m - 1, 0);
    }
}

/* Temperature sampling over the workspace buffer — the _ws entry points must
 * not fall back to the allocating geist_sampler_temperature(). */
static geist_token_t sample_temperature_ws(struct geist_sampler_workspace *ws,
                                           const float                     logits[static 1],
                                           float                           temperature,
                                           struct geist_rng               *rng) {
    const size_t n   = ws->n_vocab;
    double       sum = softmax_into(n, logits, temperature, ws->probs);
    return inv_cdf_sample(n, ws->probs, sum, rng);
}

/* ---- Top-K -------------------------------------------------------------- */

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
        return sample_temperature_ws(ws, logits, temperature, rng);
    }
    const size_t k = (size_t) top_k; /* 1 < k < n */

    select_top_desc(n, logits, k, ws->pairs);

    /* Softmax over the k selected logits, in place in the probs scratch. */
    for (size_t i = 0; i < k; i++) {
        ws->probs[i] = ws->pairs[i].score;
    }
    double sum = softmax_into(k, ws->probs, temperature, ws->probs);

    geist_token_t local = inv_cdf_sample(k, ws->probs, sum, rng);
    return (geist_token_t) ws->pairs[local].idx;
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
        return sample_temperature_ws(ws, logits, temperature, rng);
    }

    /* Unnormalized softmax; the nucleus test scales the target by the sum
     * instead of normalizing all n probabilities (same cutoff, one pass
     * less, one rounding less). */
    const double sum = softmax_into(n, logits, temperature, ws->probs);
    if (!(sum > 0.0)) {
        return geist_sampler_argmax(n, logits);
    }
    const double target = (double) top_p * sum;

    /* The nucleus is a few dozen tokens for any realistic distribution, so
     * select a small prefix and grow only if it misses the target. Each
     * attempt is O(n log k); the doubling makes the total ~2x the last one. */
    size_t k      = 32 < n ? 32 : n;
    size_t cutoff = 0;
    double cum    = 0.0;
    for (;;) {
        select_top_desc(n, ws->probs, k, ws->pairs);
        cum    = 0.0;
        cutoff = 0;
        while (cutoff < k) {
            cum += (double) ws->pairs[cutoff].score;
            cutoff++;
            if (cum >= target) {
                break;
            }
        }
        if (cum >= target || k == n) {
            break;
        }
        k = k > n / 2 ? n : k * 2;
    }

    /* Sample within the nucleus, renormalized (u is drawn against `cum`). */
    double        u      = (double) geist_rng_next_unit(rng) * cum;
    double        acc    = 0.0;
    geist_token_t picked = (geist_token_t) ws->pairs[cutoff - 1].idx;
    for (size_t i = 0; i < cutoff; i++) {
        acc += (double) ws->pairs[i].score;
        if (u < acc) {
            picked = (geist_token_t) ws->pairs[i].idx;
            break;
        }
    }
    return picked;
}

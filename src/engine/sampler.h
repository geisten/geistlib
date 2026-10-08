/*
 * src/engine/sampler.h — token sampling on top of LM logits.
 *
 * Layer: ENGINE. Pure FP32 operations on host arrays; no backend
 * involvement. Caller passes the logit vector; sampler returns the
 * chosen token ID.
 *
 * All entry points are reentrant — RNG state is caller-owned. Use the
 * argmax variant when temperature=0 / top_k=1 (greedy decode).
 */
#ifndef GEIST_INTERNAL_SAMPLER_H
#define GEIST_INTERNAL_SAMPLER_H

/* No backend or arch-state coupling, so the engine and arch layers may
 * both include it. */
#if !defined(GEIST_INTERNAL_ENGINE_LAYER) && !defined(GEIST_INTERNAL_ARCH_LAYER)
#error "sampler.h is internal — define GEIST_INTERNAL_ENGINE_LAYER or _ARCH_LAYER."
#endif

#include <geist.h>

#include <stddef.h>
#include <stdint.h>

/* xorshift64* RNG state. Seed with any non-zero u64. */
struct geist_rng {
    uint64_t state;
};

void                   geist_rng_seed(struct geist_rng *rng, uint64_t seed);
[[nodiscard]] uint64_t geist_rng_next_u64(struct geist_rng *rng);
[[nodiscard]] float    geist_rng_next_unit(struct geist_rng *rng); /* [0, 1) */

/* Greedy: returns argmax(logits). Stable on duplicates (lowest index wins). */
[[nodiscard]] geist_token_t geist_sampler_argmax(size_t      n_vocab,
                                                 const float logits[static n_vocab]);

/* Temperature-scaled softmax sample (temperature > 0). When temperature
 * is exactly 0 returns argmax. */
[[nodiscard]] geist_token_t geist_sampler_temperature(size_t            n_vocab,
                                                      const float       logits[static n_vocab],
                                                      float             temperature,
                                                      struct geist_rng *rng);

/* (score, token) pair — the sampler's selection scratch. `score` carries a
 * logit or a probability depending on the caller. */
struct geist_sampler_pair {
    float    score;
    uint32_t idx;
};

/* Pre-allocated workspace for the sampler functions that need scratch
 * (top-k, top-p). Caller-owned, reusable across calls — the per-token call
 * is allocation-free (verified by bench_sampler via heap_alloc_count()). */
struct geist_sampler_workspace {
    float                     *probs; /* size n_vocab */
    struct geist_sampler_pair *pairs; /* size n_vocab */
    size_t                     n_vocab;
};

[[nodiscard]] enum geist_status geist_sampler_workspace_init(struct geist_sampler_workspace *ws,
                                                             size_t n_vocab);
void                            geist_sampler_workspace_destroy(struct geist_sampler_workspace *ws);

/* top_k is clamped to [1, n_vocab]; every value in that range keeps its
 * requested semantics (no silent cap). */
[[nodiscard]] geist_token_t geist_sampler_top_k_ws(struct geist_sampler_workspace *ws,
                                                   const float       logits[static ws->n_vocab],
                                                   int               top_k,
                                                   float             temperature,
                                                   struct geist_rng *rng);

/* Temperature-only sampling over the workspace — same result as
 * geist_sampler_temperature, without its per-call allocation for
 * vocabularies above 8192. */
[[nodiscard]] geist_token_t geist_sampler_temperature_ws(struct geist_sampler_workspace *ws,
                                                         const float logits[static ws->n_vocab],
                                                         float       temperature,
                                                         struct geist_rng *rng);

[[nodiscard]] geist_token_t geist_sampler_top_p_ws(struct geist_sampler_workspace *ws,
                                                   const float       logits[static ws->n_vocab],
                                                   float             top_p,
                                                   float             temperature,
                                                   struct geist_rng *rng);

/* ---- Repetition control (#695) -----------------------------------------
 *
 * llama.cpp's `penalties` and `dry` samplers (src/llama-sampler.cpp), ported
 * operation for operation so a fixed history and fixed logits give the same
 * floats (tests/test_sampler_penalties_unit.c pins them against llama.cpp).
 * They run on a copy of the logits, before the filters above. */

/* Resolved parameters: defaults filled in, validated. */
struct geist_sampler_penalty_params {
    size_t last_n; /* window of the three penalties */
    float  repeat;
    float  freq;
    float  present;
    float  dry_multiplier;
    float  dry_base;
    size_t dry_allowed_length;
    size_t dry_last_n;
};

/* The opts' repetition fields with their zero defaults resolved (see
 * geist.h). GEIST_E_INVALID_ARG for a value geist.h rules out; `out` is
 * all-off then. nullptr opts → all off. */
[[nodiscard]] enum geist_status
geist_sampler_penalty_params_from_opts(struct geist_sampler_penalty_params *out,
                                       const struct geist_session_opts     *opts);

/* llama.cpp's own enable tests: the penalties sampler is a no-op when
 * last_n is 0 or all three are neutral, DRY when the multiplier is 0, the
 * base below 1 or the window 0. */
[[nodiscard]] bool geist_sampler_penalties_on(const struct geist_sampler_penalty_params *p);
[[nodiscard]] bool geist_sampler_dry_on(const struct geist_sampler_penalty_params *p);

/* One DRY sequence breaker: its head token and its tail, tail_len ids at
 * tails[tail_off] (llama.cpp's dry_processed_breakers multimap, flat). */
struct geist_dry_breaker {
    geist_token_t head;
    uint32_t      tail_off;
    uint32_t      tail_len;
};

/* Parameters, breakers and the scratch the per-token call needs, all
 * allocated up front: geist_sampler_penalties_apply allocates nothing. */
struct geist_sampler_penalties {
    struct geist_sampler_penalty_params p;
    size_t                              n_vocab;
    size_t                              cap; /* history tokens the window holds */

    struct geist_dry_breaker *breakers; /* sorted by head */
    size_t                    n_breakers;
    geist_token_t            *tails;

    float         *logits;       /* [n_vocab] the penalized copy */
    uint32_t      *counts;       /* [n_vocab] zero between calls */
    uint32_t      *dry_max;      /* [n_vocab] longest repeat + 1, zero between calls */
    geist_token_t *window;       /* [cap] */
    uint32_t      *repeat_count; /* [cap] */
    geist_token_t *touched;      /* [cap] */
};

/* Size the scratch for `n_vocab` logits and a history of up to `max_ctx`
 * positions. Nothing is allocated when neither sampler is on. */
[[nodiscard]] enum geist_status
geist_sampler_penalties_init(struct geist_sampler_penalties            *pen,
                             const struct geist_sampler_penalty_params *p,
                             size_t                                     n_vocab,
                             size_t                                     max_ctx);
void geist_sampler_penalties_destroy(struct geist_sampler_penalties *pen);

[[nodiscard]] bool geist_sampler_penalties_active(const struct geist_sampler_penalties *pen);

/* Replace the DRY breakers with the sequences in `packed` (geist_arch.h,
 * set_dry_breakers: [len, head, tail...] back to back). GEIST_E_INVALID_ARG
 * on a malformed packing, GEIST_E_OOM; the old breakers stay then. */
[[nodiscard]] enum geist_status geist_sampler_penalties_set_breakers(
        struct geist_sampler_penalties *pen, size_t n_words, const geist_token_t *packed);

/* Apply the penalties, then DRY, for a context whose tokens are hist[0..n)
 * (oldest first; a negative id — a soft token — is skipped, as are ids
 * outside the vocabulary). Returns the penalized copy (pen->logits), or
 * `logits` itself when neither sampler is on. */
[[nodiscard]] const float *geist_sampler_penalties_apply(struct geist_sampler_penalties *pen,
                                                         size_t                          n_hist,
                                                         const geist_token_t            *hist,
                                                         const float                    *logits);

#endif /* GEIST_INTERNAL_SAMPLER_H */

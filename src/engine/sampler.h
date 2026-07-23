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

/* Sampler is a pure-FP32 utility — given a logit vector + opts, it
 * returns a token. No backend or arch-state coupling, so any internal
 * layer (engine for the high-level decode loop; architecture when it
 * already holds the logits buffer) may include it. */
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

/* Pre-allocated workspace for the sampler functions that need scratch
 * (top-k, top-p). Caller-owned, reusable across calls — avoids heap churn
 * in the decode hot path. */
struct geist_sampler_workspace {
    float *probs; /* size n_vocab */
    size_t n_vocab;
};

[[nodiscard]] enum geist_status geist_sampler_workspace_init(struct geist_sampler_workspace *ws,
                                                             size_t n_vocab);
void                            geist_sampler_workspace_destroy(struct geist_sampler_workspace *ws);

[[nodiscard]] geist_token_t geist_sampler_top_k_ws(struct geist_sampler_workspace *ws,
                                                   const float       logits[static ws->n_vocab],
                                                   int               top_k,
                                                   float             temperature,
                                                   struct geist_rng *rng);

[[nodiscard]] geist_token_t geist_sampler_top_p_ws(struct geist_sampler_workspace *ws,
                                                   const float       logits[static ws->n_vocab],
                                                   float             top_p,
                                                   float             temperature,
                                                   struct geist_rng *rng);

#endif /* GEIST_INTERNAL_SAMPLER_H */

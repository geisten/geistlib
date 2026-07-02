/*
 * src/archs/transformer/scratch_plan.h - per-session scratch sizing.
 */
#ifndef GEIST_INTERNAL_ARCH_TRANSFORMER_SCRATCH_PLAN_H
#define GEIST_INTERNAL_ARCH_TRANSFORMER_SCRATCH_PLAN_H

#ifndef GEIST_INTERNAL_ARCH_LAYER
#error "transformer/scratch_plan.h is internal to the architecture layer."
#endif

#include <stdbool.h>
#include <stddef.h>

/* Hard caps the scratch pool is sized against. Model geometry exceeding
 * either would overflow the pool at prefill; the loader validates against
 * these and rejects the model rather than corrupting the heap. */
#define GEIST_SCRATCH_HEAD_DIM_MAX ((size_t) 512)
#define GEIST_SCRATCH_INTER_MAX ((size_t) 12288)

struct transformer_arch_state;

/* Returns false if the loaded model's per-layer head_dim or intermediate
 * size exceeds the scratch caps above (the pool cannot hold it). */
bool transformer_scratch_caps_ok(const struct transformer_arch_state *st);

struct transformer_scratch_plan {
    size_t hidden;
    size_t q_out;
    size_t kv_out;
    size_t inter;
    size_t ple_out;
    size_t hidden_per;
    size_t vocab;
    size_t pool_align_slack;
    size_t pool_bytes;
};

void transformer_scratch_plan_build(const struct transformer_arch_state *st,
                                    struct transformer_scratch_plan     *out);

#endif /* GEIST_INTERNAL_ARCH_TRANSFORMER_SCRATCH_PLAN_H */

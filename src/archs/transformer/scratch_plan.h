/*
 * src/archs/transformer/scratch_plan.h - per-session scratch sizing.
 */
#ifndef GEIST_INTERNAL_ARCH_TRANSFORMER_SCRATCH_PLAN_H
#define GEIST_INTERNAL_ARCH_TRANSFORMER_SCRATCH_PLAN_H

#ifndef GEIST_INTERNAL_ARCH_LAYER
#error "transformer/scratch_plan.h is internal to the architecture layer."
#endif

#include <stddef.h>

struct transformer_arch_state;

struct transformer_scratch_plan {
    size_t hidden;
    size_t q_out;
    size_t kv_out;
    size_t inter;
    size_t ple_out;
    size_t hidden_per;
    size_t vocab;
    size_t ones;
    size_t proj_in; /* 0 unless the family has per-projection input norms */
    size_t pool_align_slack;
    size_t pool_bytes;
};

/* Sizes scratch for one session; m_max is the session's prefill chunk
 * size (0 = use the model default). */
void transformer_scratch_plan_build(const struct transformer_arch_state *st,
                                    size_t                               m_max,
                                    struct transformer_scratch_plan     *out);

#endif /* GEIST_INTERNAL_ARCH_TRANSFORMER_SCRATCH_PLAN_H */

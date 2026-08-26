/* Regression gate: scratch must follow the model's largest FFN width. */
#define GEIST_INTERNAL_ARCH_LAYER

#include "src/archs/transformer/arch_state.h"
#include "src/archs/transformer/scratch_plan.h"

#include <stdio.h>

int main(void) {
    struct transformer_layer_weights layers[3] = {
            {.intermediate = 12288},
            {.intermediate = 17408},
            {.intermediate = 8192},
    };
    struct transformer_arch_state st = {
            .layers     = layers,
            .n_layers   = 3,
            .d_model    = 5120,
            .n_q_heads  = 24,
            .n_kv_heads = 4,
            .vocab_size = 248320,
    };
    struct transformer_scratch_plan plan;
    transformer_scratch_plan_build(&st, 128, &plan);
    const size_t expected = 128u * 17408u * sizeof(float);
    if (plan.inter != expected) {
        fprintf(stderr, "FAIL: inter scratch=%zu expected=%zu\n", plan.inter, expected);
        return 1;
    }
    puts("PASS: scratch plan uses largest per-layer intermediate width");
    return 0;
}

/*
 * test_projection_input_norms_unit — scratch sizing for the BitNet
 * embedding models' per-projection input norms. No model, no backend.
 *
 * Those norms give q/k/v/gate/up each their own normalised input, which
 * needs one extra [m_max, d_model] scratch slice (scratch_proj_in). Two
 * things must hold and neither is visible at a call site:
 *
 *   1. Every OTHER model must not pay for it. The slice is sized 0 unless
 *      config.has_projection_input_norms is set, and a regression here is
 *      silent — it just quietly grows resident memory for Gemma, Llama,
 *      Qwen3 and every BitNet b1.58 session.
 *   2. pool_bytes must account for the slice exactly. The pool is one
 *      allocation carved into aliased slices, so an undercount does not
 *      fail loudly at plan time — it hands out a slice that runs past the
 *      pool.
 *
 * No assert(): release builds define NDEBUG. Checks set a flag and the
 * exit code carries PASS/FAIL.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "src/archs/transformer/arch_state.h"
#include "src/archs/transformer/scratch_plan.h"

#include <stdio.h>

static int fails = 0;

static void check(const char *what, bool ok) {
    printf("  %s  %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) {
        fails++;
    }
}

/* A BitNet-embedding-0.6B-shaped state: 28 layers, d_model 1024,
 * 16 q heads / 8 kv, intermediate 3072. */
static struct transformer_arch_state
make_state(struct transformer_layer_weights *layers, size_t n_layers, bool proj_norms) {
    for (size_t i = 0; i < n_layers; i++) {
        layers[i] = (struct transformer_layer_weights) {.intermediate = 3072};
    }
    struct transformer_arch_state st = {
            .layers     = layers,
            .n_layers   = n_layers,
            .d_model    = 1024,
            .n_q_heads  = 16,
            .n_kv_heads = 8,
            .vocab_size = 151936,
    };
    st.config.has_projection_input_norms = proj_norms;
    return st;
}

int main(void) {
    puts("=== per-projection input norms: scratch sizing ===");

    constexpr size_t                 N_LAYERS = 28;
    constexpr size_t                 M        = 64;
    struct transformer_layer_weights layers_off[N_LAYERS];
    struct transformer_layer_weights layers_on[N_LAYERS];

    struct transformer_arch_state st_off = make_state(layers_off, N_LAYERS, false);
    struct transformer_arch_state st_on  = make_state(layers_on, N_LAYERS, true);

    struct transformer_scratch_plan off;
    struct transformer_scratch_plan on;
    transformer_scratch_plan_build(&st_off, M, &off);
    transformer_scratch_plan_build(&st_on, M, &on);

    check("families without the norms get no proj_in slice", off.proj_in == 0);

    const size_t expected = M * 1024u * sizeof(float);
    check("with the norms the slice is [m_max, d_model] floats", on.proj_in == expected);

    /* The only difference between the two plans is that one slice, so the
     * pool has to grow by exactly its size — no more (nothing else may
     * have changed) and no less (the slice must be accounted for). */
    check("pool_bytes grows by exactly the slice",
          on.pool_bytes >= off.pool_bytes && on.pool_bytes - off.pool_bytes == expected);

    /* Everything else must be untouched: a change here would mean the
     * flag leaked into sizing it has no business affecting. */
    check("no other scratch dimension changes",
          on.hidden == off.hidden && on.q_out == off.q_out && on.kv_out == off.kv_out &&
                  on.inter == off.inter && on.vocab == off.vocab && on.ones == off.ones);

    /* The per-layer owning-buffer list must hold every tensor one layer
     * loads, and a BitNet embedding layer is the widest case: 2 block norms
     * + 2 QK norms + 4 attention projections + 3 FFN projections = 11, plus
     * the 7 per-projection input norms = 18. When the list was sized 16 the
     * seventh norm overflowed it, and layer_track_buf turns that into a
     * load failure on the last tensor it happens to reach --
     * 'layer buffer list overflow on blk.0.ffn_down_norm_in.weight' -- which
     * names a symptom, not the cause. Pin the capacity so growing a family
     * by one tensor fails here, at a line that says what is wrong. */
    constexpr size_t BUFS_CAP = sizeof layers_on[0].bufs / sizeof layers_on[0].bufs[0];
    constexpr size_t BITNET_EMBEDDING_LAYER_TENSORS = 18;
    check("the per-layer buffer list holds a whole BitNet embedding layer",
          BUFS_CAP >= BITNET_EMBEDDING_LAYER_TENSORS);

    if (fails > 0) {
        printf("FAIL: %d check(s) failed\n", fails);
        return 1;
    }
    puts("PASS: proj_in scratch is conditional and fully accounted for");
    return 0;
}

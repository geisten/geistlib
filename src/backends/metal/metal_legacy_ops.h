/* metal_legacy_ops.h — GENERATED: backend-private copies of the old
 * fine-grained GPU-op structs/enum that main's geist_backend.h no longer
 * declares. The metal backend keeps its old op implementations compilable
 * as internal/dead code during the port; main's engine never calls them
 * (they are not in the main-contract vtbl). Removed in the Stage-6 cleanup.
 */
#ifndef GEIST_METAL_LEGACY_OPS_H
#define GEIST_METAL_LEGACY_OPS_H
#include <geist.h>
#include <geist_types.h>
#include <geist_weight.h>

enum geist_command_sequence_kind {
    GEIST_COMMAND_SEQUENCE_DECODE_LAYER_LOOP,
    GEIST_COMMAND_SEQUENCE_DECODE_GREEDY_STEP,
    GEIST_COMMAND_SEQUENCE_PREFILL_TEXT,
    GEIST_COMMAND_SEQUENCE_VERIFY_GREEDY,
};

struct geist_backend_accel_caps {
    size_t struct_size;

    bool device_resident_buffers;
    bool compute_queue;
    bool pipeline_cache;
    bool subgroup_basic;
    bool shader_integer_dot_product;
    bool descriptor_indexing;
    bool timeline_semaphore;

    uint64_t device_local_bytes;
    char     device_name[128];

    /* ple_block accepts a per_layer_input whose row stride exceeds its
     * column count (a strided view into the [seq, n_layers*hpl] slab),
     * letting the arch skip the per-token gather copies. */
    bool ple_strided_input;
};

struct geist_backend_ffn_geglu_block {
    size_t struct_size;

    size_t seq;
    size_t d_model;
    size_t inter;
    float  eps;

    const struct geist_tensor *residual;
    const struct geist_tensor *ffn_norm_weight;
    const struct geist_tensor *gate_weight;
    const struct geist_tensor *up_weight;
    const struct geist_tensor *down_weight;
    const struct geist_tensor *post_ffw_norm_weight; /* nullable */

    struct geist_tensor *pre_ff_scratch;
    struct geist_tensor *gate_scratch;
    struct geist_tensor *up_scratch;
    struct geist_tensor *ffn_out_scratch;
    struct geist_tensor *post_ff_scratch; /* required when post_ffw_norm_weight != NULL */
    struct geist_tensor *out;
};

struct geist_backend_attention_block {
    size_t struct_size;

    size_t q_position;
    size_t kv_len;
    size_t d_model;
    size_t q_heads;
    size_t kv_heads;
    size_t head_dim;
    size_t sliding_window;
    float  eps;

    const struct geist_tensor *residual;
    const struct geist_tensor *attn_norm_weight;
    const struct geist_tensor *q_proj_weight;
    const struct geist_tensor *k_proj_weight;
    const struct geist_tensor *v_proj_weight;
    const struct geist_tensor *q_norm_weight;
    const struct geist_tensor *k_norm_weight;
    const struct geist_tensor *v_norm_weight;
    const struct geist_tensor *cos;
    const struct geist_tensor *sin;
    const struct geist_tensor *k_cache;
    const struct geist_tensor *v_cache;
    const struct geist_tensor *o_proj_weight;
    const struct geist_tensor *post_attn_norm_weight; /* nullable */

    struct geist_tensor *normed_scratch;
    struct geist_tensor *q_scratch;
    struct geist_tensor *k_scratch;
    struct geist_tensor *v_scratch;
    struct geist_tensor *attn_scratch;
    struct geist_tensor *o_scratch;
    struct geist_tensor *post_attn_scratch; /* required when post_attn_norm_weight != NULL */
    struct geist_tensor *out;
};

struct geist_backend_attention_query_block {
    size_t struct_size;

    size_t q_position;
    size_t kv_len;
    size_t d_model;
    size_t q_heads;
    size_t kv_heads;
    size_t head_dim;
    size_t sliding_window;
    float  eps;

    const struct geist_tensor *residual;
    const struct geist_tensor *attn_norm_weight;
    const struct geist_tensor *q_proj_weight;
    const struct geist_tensor *q_norm_weight;
    const struct geist_tensor *cos;
    const struct geist_tensor *sin;
    const struct geist_tensor *k_cache;
    const struct geist_tensor *v_cache;
    const struct geist_tensor *o_proj_weight;
    const struct geist_tensor *post_attn_norm_weight; /* nullable */

    struct geist_tensor *normed_scratch;
    struct geist_tensor *q_scratch;
    struct geist_tensor *attn_scratch;
    struct geist_tensor *o_scratch;
    struct geist_tensor *post_attn_scratch; /* required when post_attn_norm_weight != NULL */
    struct geist_tensor *out;
};

struct geist_backend_ple_block {
    size_t struct_size;

    size_t seq;
    size_t d_model;
    size_t hidden_per_layer;
    float  eps;

    const struct geist_tensor *hidden;
    const struct geist_tensor *per_layer_input;
    const struct geist_tensor *per_layer_gate_weight;
    const struct geist_tensor *per_layer_proj_weight;
    const struct geist_tensor *post_per_layer_norm_weight;

    struct geist_tensor *gate_scratch;
    struct geist_tensor *proj_scratch;
    struct geist_tensor *out;
};

struct geist_backend_greedy_head {
    size_t struct_size;

    size_t d_model;
    size_t vocab_size;
    size_t token_output_offset;
    float  eps;

    const struct geist_tensor *hidden;
    const struct geist_tensor *norm_weight;
    const struct geist_tensor *lm_head_weight;

    struct geist_tensor *normed_scratch;
    struct geist_tensor *logits;
};

struct geist_backend_greedy_head_batch {
    size_t struct_size;

    size_t d_model;
    size_t vocab_size;
    size_t row_count;
    size_t token_output_offset;
    float  eps;

    const struct geist_tensor *hidden;
    const struct geist_tensor *norm_weight;
    const struct geist_tensor *lm_head_weight;

    struct geist_tensor *normed_scratch;
    struct geist_tensor *logits;
};

#endif

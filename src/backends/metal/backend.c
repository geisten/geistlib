/*
 * src/backends/metal/backend.c - optional native Metal backend skeleton.
 *
 * Layer: BACKEND.
 *
 * This first increment intentionally exposes lifecycle and accelerator caps
 * only. Transformer kernels land behind this backend after the native Metal
 * object model is proven build-gated and runtime-gated.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include <geist.h>
#include <geist_backend.h>
#include "quant.h"
#include "heap.h"
#include "metal_legacy_ops.h"
#include <math.h>

#include <dlfcn.h>
#include <errno.h>
#include <stdalign.h>
#include <stdbool.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

enum metal_profile_stage {
    METAL_PROFILE_WAIT_DECODE_LAYER_LOOP = 0,
    METAL_PROFILE_WAIT_DECODE_GREEDY_STEP,
    METAL_PROFILE_WAIT_VERIFY_GREEDY,
    METAL_PROFILE_WAIT_PREFILL_TEXT,
    METAL_PROFILE_WAIT_FFN_STANDALONE,
    METAL_PROFILE_DISPATCH_RMSNORM_ROWS,
    METAL_PROFILE_DISPATCH_Q4K_GATE_UP_BASE,
    METAL_PROFILE_DISPATCH_Q4K_GATE_UP_N4,
    METAL_PROFILE_DISPATCH_Q4K_GATE_UP_NT4,
    METAL_PROFILE_DISPATCH_Q4K_GATE_UP_NT8,
    METAL_PROFILE_DISPATCH_Q4K_GATE_UP_W4A8,
    METAL_PROFILE_DISPATCH_Q4K_LINEAR_BASE,
    METAL_PROFILE_DISPATCH_Q4K_LINEAR_N4,
    METAL_PROFILE_DISPATCH_Q4K_LINEAR_NT4,
    METAL_PROFILE_DISPATCH_Q4K_LINEAR_NT8,
    METAL_PROFILE_DISPATCH_Q4K_LINEAR_W4A8,
    METAL_PROFILE_DISPATCH_Q4K_PLE_GATE_NT8,
    METAL_PROFILE_DISPATCH_F32_PLE_GATE,
    METAL_PROFILE_DISPATCH_Q4K_QUANT_X,
    METAL_PROFILE_DISPATCH_Q6K_LINEAR_BASE,
    METAL_PROFILE_DISPATCH_Q6K_LINEAR_N4,
    METAL_PROFILE_DISPATCH_Q6K_LINEAR_NT4,
    METAL_PROFILE_DISPATCH_Q6K_LINEAR_NT8,
    METAL_PROFILE_DISPATCH_Q4K_QK_BASE,
    METAL_PROFILE_DISPATCH_Q4K_QK_NT4,
    METAL_PROFILE_DISPATCH_F32_PLE_PROJ_NORM,
    METAL_PROFILE_DISPATCH_RMSNORM_ADD_ROWS,
    METAL_PROFILE_DISPATCH_Q_NORM_ROPE,
    METAL_PROFILE_DISPATCH_K_NORM_ROPE_APPEND,
    METAL_PROFILE_DISPATCH_V_NORM_APPEND,
    METAL_PROFILE_DISPATCH_KV_NORM_APPEND,
    METAL_PROFILE_DISPATCH_ROPE_ROWS,
    METAL_PROFILE_DISPATCH_KV_APPEND_ROWS,
    METAL_PROFILE_DISPATCH_ATTENTION_ROWS,
    METAL_PROFILE_DISPATCH_ATTENTION_QNORM_ROWS,
    METAL_PROFILE_DISPATCH_GELU_MUL_ROWS,
    METAL_PROFILE_DISPATCH_F32_MATMUL,
    METAL_PROFILE_DISPATCH_EMBED,
    METAL_PROFILE_STAGE_COUNT,
};

static const char * const metal_profile_stage_names[METAL_PROFILE_STAGE_COUNT] = {
    [METAL_PROFILE_WAIT_DECODE_LAYER_LOOP] = "wait.decode_layer_loop",
    [METAL_PROFILE_WAIT_DECODE_GREEDY_STEP] = "wait.decode_greedy_step",
    [METAL_PROFILE_WAIT_VERIFY_GREEDY] = "wait.verify_greedy",
    [METAL_PROFILE_WAIT_PREFILL_TEXT] = "wait.prefill_text",
    [METAL_PROFILE_WAIT_FFN_STANDALONE] = "wait.ffn_standalone",
    [METAL_PROFILE_DISPATCH_RMSNORM_ROWS] = "dispatch.rmsnorm_rows",
    [METAL_PROFILE_DISPATCH_Q4K_GATE_UP_BASE] =
        "dispatch.q4k_gate_up.base",
    [METAL_PROFILE_DISPATCH_Q4K_GATE_UP_N4] = "dispatch.q4k_gate_up.n4",
    [METAL_PROFILE_DISPATCH_Q4K_GATE_UP_NT4] = "dispatch.q4k_gate_up.nt4",
    [METAL_PROFILE_DISPATCH_Q4K_GATE_UP_NT8] = "dispatch.q4k_gate_up.nt8",
    [METAL_PROFILE_DISPATCH_Q4K_GATE_UP_W4A8] =
        "dispatch.q4k_gate_up.w4a8",
    [METAL_PROFILE_DISPATCH_Q4K_LINEAR_BASE] =
        "dispatch.q4k_linear.base",
    [METAL_PROFILE_DISPATCH_Q4K_LINEAR_N4] = "dispatch.q4k_linear.n4",
    [METAL_PROFILE_DISPATCH_Q4K_LINEAR_NT4] = "dispatch.q4k_linear.nt4",
    [METAL_PROFILE_DISPATCH_Q4K_LINEAR_NT8] = "dispatch.q4k_linear.nt8",
    [METAL_PROFILE_DISPATCH_Q4K_LINEAR_W4A8] =
        "dispatch.q4k_linear.w4a8",
    [METAL_PROFILE_DISPATCH_Q4K_PLE_GATE_NT8] =
        "dispatch.q4k_ple_gate.nt8",
    [METAL_PROFILE_DISPATCH_F32_PLE_GATE] =
        "dispatch.f32_ple_gate",
    [METAL_PROFILE_DISPATCH_Q4K_QUANT_X] = "dispatch.q4k_quant_x",
    [METAL_PROFILE_DISPATCH_Q6K_LINEAR_BASE] =
        "dispatch.q6k_linear.base",
    [METAL_PROFILE_DISPATCH_Q6K_LINEAR_N4] = "dispatch.q6k_linear.n4",
    [METAL_PROFILE_DISPATCH_Q6K_LINEAR_NT4] = "dispatch.q6k_linear.nt4",
    [METAL_PROFILE_DISPATCH_Q6K_LINEAR_NT8] = "dispatch.q6k_linear.nt8",
    [METAL_PROFILE_DISPATCH_Q4K_QK_BASE] = "dispatch.q4k_qk.base",
    [METAL_PROFILE_DISPATCH_Q4K_QK_NT4] = "dispatch.q4k_qk.nt4",
    [METAL_PROFILE_DISPATCH_F32_PLE_PROJ_NORM] =
        "dispatch.f32_ple_proj_norm",
    [METAL_PROFILE_DISPATCH_RMSNORM_ADD_ROWS] = "dispatch.rmsnorm_add_rows",
    [METAL_PROFILE_DISPATCH_Q_NORM_ROPE] = "dispatch.q_norm_rope",
    [METAL_PROFILE_DISPATCH_K_NORM_ROPE_APPEND] = "dispatch.k_norm_rope_append",
    [METAL_PROFILE_DISPATCH_V_NORM_APPEND] = "dispatch.v_norm_append",
    [METAL_PROFILE_DISPATCH_KV_NORM_APPEND] = "dispatch.kv_norm_append",
    [METAL_PROFILE_DISPATCH_ROPE_ROWS] = "dispatch.rope_rows",
    [METAL_PROFILE_DISPATCH_KV_APPEND_ROWS] = "dispatch.kv_append_rows",
    [METAL_PROFILE_DISPATCH_ATTENTION_ROWS] = "dispatch.attention_rows",
    [METAL_PROFILE_DISPATCH_ATTENTION_QNORM_ROWS] =
        "dispatch.attention_qnorm_rows",
    [METAL_PROFILE_DISPATCH_GELU_MUL_ROWS] = "dispatch.gelu_mul_rows",
    [METAL_PROFILE_DISPATCH_F32_MATMUL] = "dispatch.f32_matmul",
    [METAL_PROFILE_DISPATCH_EMBED] = "dispatch.embed",
};

struct metal_profile_stat {
    uint64_t ns;
    uint64_t calls;
    uint64_t workgroups;
};

struct metal_q4k_nt4_cache_entry {
    const struct geist_buffer *src;
    size_t src_offset;
    size_t n_in;
    size_t n_out;
    size_t raw_bytes;
    size_t packed_bytes;
    struct geist_buffer *packed;
    struct metal_q4k_nt4_cache_entry *next;
};

struct metal_q6k_nt4_cache_entry {
    const struct geist_buffer *src;
    size_t src_offset;
    size_t n_in;
    size_t n_out;
    size_t raw_bytes;
    size_t packed_bytes;
    struct geist_buffer *packed;
    struct metal_q6k_nt4_cache_entry *next;
};

struct metal_pack_cache_header {
    uint8_t magic[8];
    uint64_t version;
    uint64_t n_in;
    uint64_t n_out;
    uint64_t raw_bytes;
    uint64_t packed_bytes;
    uint64_t raw_hash;
    uint64_t packed_hash;
};

enum {
    METAL_DECODE_REPLAY_MAX_OPS = 192,
};

enum metal_decode_replay_op_kind {
    METAL_DECODE_REPLAY_EMBED_LOOKUP_SCALED,
    METAL_DECODE_REPLAY_F32_LINEAR,
    METAL_DECODE_REPLAY_RMSNORM,
    METAL_DECODE_REPLAY_ADD,
    METAL_DECODE_REPLAY_SCALE_F32,
    METAL_DECODE_REPLAY_ATTENTION,
    METAL_DECODE_REPLAY_ATTENTION_QUERY,
    METAL_DECODE_REPLAY_FFN_GEGLU,
    METAL_DECODE_REPLAY_PLE,
    METAL_DECODE_REPLAY_GREEDY_HEAD,
};

struct metal_decode_replay_embed_params {
    uint32_t n;
    uint32_t dtype;
    uint32_t blocks_per_row;
    uint32_t w_byte_offset;
    uint32_t y_offset;
    float scale;
};

struct metal_decode_replay_f32_params {
    uint32_t n_in;
    uint32_t n_out;
    uint32_t rows;
    uint32_t x_offset;
    uint32_t w_offset;
    uint32_t y_offset;
    uint32_t x_row_stride;
    uint32_t y_row_stride;
};

struct metal_decode_replay_rows_params {
    uint32_t rows;
    uint32_t cols;
    uint32_t x_offset;
    uint32_t w_offset;
    uint32_t y_offset;
    uint32_t x_row_stride;
    uint32_t y_row_stride;
    float eps;
};

struct metal_decode_replay_binary_rows_params {
    uint32_t rows;
    uint32_t cols;
    uint32_t a_offset;
    uint32_t b_offset;
    uint32_t y_offset;
    uint32_t a_row_stride;
    uint32_t b_row_stride;
    uint32_t y_row_stride;
};

struct metal_decode_replay_scale_rows_params {
    uint32_t rows;
    uint32_t cols;
    uint32_t x_offset;
    uint32_t y_offset;
    uint32_t x_row_stride;
    uint32_t y_row_stride;
    float scale;
};

struct metal_decode_replay_q4k_params {
    uint32_t n_in;
    uint32_t n_out;
    uint32_t rows;
    uint32_t blocks_per_row;
    uint32_t x_offset;
    uint32_t w_byte_offset;
    uint32_t y_offset;
    uint32_t x_row_stride;
    uint32_t y_row_stride;
};

struct metal_decode_replay_q4k_gate_up_params {
    uint32_t n_in;
    uint32_t n_out;
    uint32_t rows;
    uint32_t blocks_per_row;
    uint32_t x_offset;
    uint32_t gate_w_byte_offset;
    uint32_t up_w_byte_offset;
    uint32_t gate_y_offset;
    uint32_t up_y_offset;
    uint32_t x_row_stride;
    uint32_t y_row_stride;
};

struct metal_decode_replay_q4k_qk_params {
    uint32_t n_in;
    uint32_t q_out;
    uint32_t k_out;
    uint32_t rows;
    uint32_t blocks_per_row;
    uint32_t x_offset;
    uint32_t q_w_byte_offset;
    uint32_t k_w_byte_offset;
    uint32_t q_y_offset;
    uint32_t k_y_offset;
    uint32_t x_row_stride;
    uint32_t q_y_row_stride;
    uint32_t k_y_row_stride;
};

struct metal_decode_replay_post_norm_params {
    uint32_t rows;
    uint32_t cols;
    uint32_t residual_offset;
    uint32_t x_offset;
    uint32_t w_offset;
    uint32_t y_offset;
    uint32_t residual_row_stride;
    uint32_t x_row_stride;
    uint32_t y_row_stride;
    float eps;
};

struct metal_decode_replay_norm_rope_params {
    uint32_t rows;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t x_offset;
    uint32_t w_offset;
    uint32_t cos_offset;
    uint32_t sin_offset;
    uint32_t x_row_stride;
    uint32_t rope_row_stride;
    uint32_t rope_row_offset;
    float eps;
};

struct metal_decode_replay_k_norm_rope_append_params {
    uint32_t rows;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t x_offset;
    uint32_t w_offset;
    uint32_t cos_offset;
    uint32_t sin_offset;
    uint32_t cache_offset;
    uint32_t x_row_stride;
    uint32_t rope_row_stride;
    uint32_t rope_row_offset;
    uint32_t q_position;
    float eps;
};

struct metal_decode_replay_v_norm_append_params {
    uint32_t rows;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t x_offset;
    uint32_t w_offset;
    uint32_t cache_offset;
    uint32_t x_row_stride;
    uint32_t q_position;
    float eps;
};

struct metal_decode_replay_attention_params {
    uint32_t rows;
    uint32_t kv_len;
    uint32_t q_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t q_position;
    uint32_t sliding_window;
    uint32_t q_offset;
    uint32_t k_cache_offset;
    uint32_t v_cache_offset;
    uint32_t y_offset;
};

struct metal_decode_replay_argmax_params {
    uint32_t n;
    uint32_t x_offset;
};

struct metal_decode_replay_embed {
    struct geist_tensor table;
    struct geist_tensor out;
    struct metal_decode_replay_embed_params params;
};

struct metal_decode_replay_f32_linear {
    struct geist_tensor x;
    struct geist_tensor w;
    struct geist_tensor y;
    bool matrix;
    struct metal_decode_replay_f32_params params;
};

struct metal_decode_replay_rmsnorm {
    struct geist_tensor x;
    struct geist_tensor w;
    struct geist_tensor y;
    struct metal_decode_replay_rows_params params;
};

struct metal_decode_replay_binary {
    struct geist_tensor a;
    struct geist_tensor b;
    struct geist_tensor y;
    struct metal_decode_replay_binary_rows_params params;
};

struct metal_decode_replay_scale {
    struct geist_tensor x;
    struct geist_tensor y;
    struct metal_decode_replay_scale_rows_params params;
};

struct metal_decode_replay_attention {
    struct geist_backend_attention_block block;
    struct geist_tensor residual;
    struct geist_tensor attn_norm_weight;
    struct geist_tensor q_proj_weight;
    struct geist_tensor k_proj_weight;
    struct geist_tensor v_proj_weight;
    struct geist_tensor q_norm_weight;
    struct geist_tensor k_norm_weight;
    struct geist_tensor v_norm_weight;
    struct geist_tensor cos;
    struct geist_tensor sin;
    struct geist_tensor k_cache;
    struct geist_tensor v_cache;
    struct geist_tensor o_proj_weight;
    struct geist_tensor post_attn_norm_weight;
    struct geist_tensor normed_scratch;
    struct geist_tensor q_scratch;
    struct geist_tensor k_scratch;
    struct geist_tensor v_scratch;
    struct geist_tensor attn_scratch;
    struct geist_tensor o_scratch;
    struct geist_tensor post_attn_scratch;
    struct geist_tensor out;
    bool v_w_q6;
    bool use_fused_qk;
    bool use_fused_qk_nt4;
    bool rope_uses_q_position;
    struct geist_buffer *q_qk_packed;
    struct geist_buffer *k_qk_packed;
    struct metal_decode_replay_rows_params norm_params;
    struct metal_decode_replay_q4k_params q_params;
    struct metal_decode_replay_q4k_params k_params;
    struct metal_decode_replay_q4k_qk_params qk_params;
    struct metal_decode_replay_q4k_params v_params;
    struct metal_decode_replay_norm_rope_params q_norm_rope_params;
    struct metal_decode_replay_k_norm_rope_append_params k_norm_rope_append_params;
    struct metal_decode_replay_v_norm_append_params v_norm_append_params;
    struct metal_decode_replay_attention_params attention_params;
    struct metal_decode_replay_q4k_params o_params;
    struct metal_decode_replay_post_norm_params post_params;
};

struct metal_decode_replay_attention_query {
    struct geist_backend_attention_query_block block;
    struct geist_tensor residual;
    struct geist_tensor attn_norm_weight;
    struct geist_tensor q_proj_weight;
    struct geist_tensor q_norm_weight;
    struct geist_tensor cos;
    struct geist_tensor sin;
    struct geist_tensor k_cache;
    struct geist_tensor v_cache;
    struct geist_tensor o_proj_weight;
    struct geist_tensor post_attn_norm_weight;
    struct geist_tensor normed_scratch;
    struct geist_tensor q_scratch;
    struct geist_tensor attn_scratch;
    struct geist_tensor o_scratch;
    struct geist_tensor post_attn_scratch;
    struct geist_tensor out;
};

struct metal_decode_replay_ffn {
    struct geist_backend_ffn_geglu_block block;
    struct geist_tensor residual;
    struct geist_tensor ffn_norm_weight;
    struct geist_tensor gate_weight;
    struct geist_tensor up_weight;
    struct geist_tensor down_weight;
    struct geist_tensor post_ffw_norm_weight;
    struct geist_tensor pre_ff_scratch;
    struct geist_tensor gate_scratch;
    struct geist_tensor up_scratch;
    struct geist_tensor ffn_out_scratch;
    struct geist_tensor post_ff_scratch;
    struct geist_tensor out;
    bool down_w_q6;
    struct metal_decode_replay_rows_params pre_norm_params;
    struct metal_decode_replay_q4k_gate_up_params gate_up_params;
    struct metal_decode_replay_q4k_params down_params;
    struct metal_decode_replay_post_norm_params post_params;
};

struct metal_decode_replay_ple {
    struct geist_backend_ple_block block;
    struct geist_tensor hidden;
    struct geist_tensor per_layer_input;
    struct geist_tensor per_layer_gate_weight;
    struct geist_tensor per_layer_proj_weight;
    struct geist_tensor post_per_layer_norm_weight;
    struct geist_tensor gate_scratch;
    struct geist_tensor proj_scratch;
    struct geist_tensor out;
    bool gate_w_q4;
    bool proj_w_q4;
    struct metal_decode_replay_q4k_params gate_q4_params;
    struct metal_decode_replay_q4k_params proj_q4_params;
    struct metal_decode_replay_f32_params gate_f32_params;
    struct metal_decode_replay_f32_params proj_f32_params;
    struct metal_decode_replay_binary_rows_params act_params;
    struct metal_decode_replay_post_norm_params post_params;
};

struct metal_decode_replay_greedy_head {
    struct geist_backend_greedy_head head;
    struct geist_tensor hidden;
    struct geist_tensor norm_weight;
    struct geist_tensor lm_head_weight;
    struct geist_tensor normed_scratch;
    struct geist_tensor logits;
    bool lm_head_f32;
    bool lm_head_q4;
    bool lm_head_q6;
    struct metal_decode_replay_rows_params norm_params;
    struct metal_decode_replay_f32_params f32_params;
    struct metal_decode_replay_q4k_params q_params;
    struct metal_decode_replay_argmax_params argmax_params;
};

struct metal_decode_replay_op {
    enum metal_decode_replay_op_kind kind;
    union {
        struct metal_decode_replay_embed embed;
        struct metal_decode_replay_f32_linear f32_linear;
        struct metal_decode_replay_rmsnorm rmsnorm;
        struct metal_decode_replay_binary binary;
        struct metal_decode_replay_scale scale;
        struct metal_decode_replay_attention attention;
        struct metal_decode_replay_attention_query attention_query;
        struct metal_decode_replay_ffn ffn;
        struct metal_decode_replay_ple ple;
        struct metal_decode_replay_greedy_head greedy_head;
    } u;
};

/* Registry entry mapping a live buffer's host contents range back to its
 * geist_buffer, so resolver-installed linear kernels can translate the raw
 * host pointers main's engine passes (buffer_map aliases, w->raw) into
 * (MTLBuffer, offset) pairs for GPU dispatch. */
struct metal_buf_reg_entry {
    const uint8_t *base;
    size_t bytes;
    struct geist_buffer *buf;
};

struct metal_state {
    struct geist_backend *backend;
    void *metal_handle;
    void *objc_handle;
    void *device;
    void *command_queue;
    struct metal_buf_reg_entry *buf_reg;
    size_t buf_reg_count;
    size_t buf_reg_cap;
    /* MTLBuffers referenced by ops encoded on the open (unflushed) batch;
     * a host map/upload/download of a referenced buffer forces a flush.
     * Open-addressed pointer set; overflow degrades to always-flush. */
    void *seq_ref[4096];
    size_t seq_ref_count;
    bool seq_ref_overflow;
    void *q4k_library;
    void *q4k_n4_library;
    void *q4k_nt4_library;
    void *q4k_ple_gate_nt8_library;
    void *q4k_w4a8_library;
    void *q4k_gate_up_w4a8_library;
    void *q6k_library;
    void *q6k_n4_library;
    void *q6k_nt4_library;
    void *q6k_nt8_library;
    void *elem_library;
    void *elem_simd_library;
    void *attn_library;
    void *attn_f16_library;
    void *attn_qnorm_library;
    void *attn_qnorm_f16_library;
    void *q_norm_rope_library;
    void *k_norm_rope_append_library;
    void *v_norm_append_library;
    void *kv_norm_append_library;
    void *kv_norm_append_f16_library;
    void *q4k_function;
    void *q4k_pipeline;
    void *q4k_n4_function;
    void *q4k_n4_pipeline;
    void *q4k_nt4_function;
    void *q4k_nt4_pipeline;
    void *q4k_nt8_function;
    void *q4k_nt8_pipeline;
    void *q4k_ple_gate_nt8_function;
    void *q4k_ple_gate_nt8_pipeline;
    void *q4k_quant_x_function;
    void *q4k_quant_x_pipeline;
    void *q4k_w4a8_function;
    void *q4k_w4a8_pipeline;
    void *q4k_gate_up_w4a8_function;
    void *q4k_gate_up_w4a8_pipeline;
    void *q4k_matmul_m8_function;
    void *q4k_matmul_m8_pipeline;
    void *q4k_m16_library;
    void *q4k_matmul_m16_function;
    void *q4k_matmul_m16_pipeline;
    void *q4k_m16_n2_library;
    void *q4k_matmul_m16_n2_function;
    void *q4k_matmul_m16_n2_pipeline;
    void *q4k_mm_sg_library;
    void *q4k_mm_sg_function;
    void *q4k_mm_sg_pipeline;
    void *q4k_mm_sg_fast_library;
    void *q4k_mm_sg_fast_function;
    void *q4k_mm_sg_fast_pipeline;
    void *q4k_qk_library;
    void *q4k_qk_function;
    void *q4k_qk_pipeline;
    void *q4k_qk_nt4_function;
    void *q4k_qk_nt4_pipeline;
    void *q4k_gate_up_library;
    void *q4k_gate_up_n4_library;
    void *q4k_gate_up_nt4_library;
    void *q4k_gate_up_nt8_library;
    void *embed_library;
    void *argmax_library;
    void *q4k_gate_up_function;
    void *q4k_gate_up_pipeline;
    void *q4k_gate_up_n4_function;
    void *q4k_gate_up_n4_pipeline;
    void *q4k_gate_up_nt4_function;
    void *q4k_gate_up_nt4_pipeline;
    void *q4k_gate_up_nt8_function;
    void *q4k_gate_up_nt8_pipeline;
    void *q6k_function;
    void *q6k_pipeline;
    void *q6k_n4_function;
    void *q6k_n4_pipeline;
    void *q6k_nt4_function;
    void *q6k_nt4_pipeline;
    void *q6k_nt8_function;
    void *q6k_nt8_pipeline;
    void *q6k_matmul_m8_function;
    void *q6k_matmul_m8_pipeline;
    void *q6k_mm_sg_library;
    void *q6k_matmul_sg_function;
    void *q6k_matmul_sg_pipeline;
    void *q6k_mm_sg_fast_library;
    void *q6k_matmul_sg_fast_function;
    void *q6k_matmul_sg_fast_pipeline;
    void *q6k_m16_library;
    void *q6k_matmul_m16_function;
    void *q6k_matmul_m16_pipeline;
    void *rmsnorm_rows_function;
    void *rmsnorm_rows_pipeline;
    void *rmsnorm_rows_simd_function;
    void *rmsnorm_rows_simd_pipeline;
    void *gelu_rows_function;
    void *gelu_rows_pipeline;
    void *mul_rows_function;
    void *mul_rows_pipeline;
    void *gelu_mul_rows_function;
    void *gelu_mul_rows_pipeline;
    void *add_rows_function;
    void *add_rows_pipeline;
    void *scale_rows_function;
    void *scale_rows_pipeline;
    void *rmsnorm_add_rows_function;
    void *rmsnorm_add_rows_pipeline;
    void *rmsnorm_add_rows_simd_function;
    void *rmsnorm_add_rows_simd_pipeline;
    void *embed_lookup_scaled_function;
    void *embed_lookup_scaled_pipeline;
    void *f32_library;
    void *f32_matmul_function;
    void *f32_matmul_pipeline;
    void *f32_matmul_sg_function;
    void *f32_matmul_sg_pipeline;
    void *f32_matmul_mm_function;
    void *f32_matmul_mm_pipeline;
    void *f32_ple_gate_function;
    void *f32_ple_gate_pipeline;
    void *f32_ple_proj_norm_function;
    void *f32_ple_proj_norm_pipeline;
    void *argmax_function;
    void *argmax_pipeline;
    void *argmax_batch_function;
    void *argmax_batch_pipeline;
    void *argmax_result_buffer;
    void *argmax_result_mapped;
    uint32_t argmax_result_capacity;
    void *rope_rows_function;
    void *rope_rows_pipeline;
    void *kv_append_rows_function;
    void *kv_append_rows_pipeline;
    void *copy_u32_function;
    void *copy_u32_pipeline;
    void *kv_append_rows_f16_function;
    void *kv_append_rows_f16_pipeline;
    void *q_norm_rope_rows_function;
    void *q_norm_rope_rows_pipeline;
    void *k_norm_rope_append_rows_function;
    void *k_norm_rope_append_rows_pipeline;
    void *k_norm_rope_append_rows_f16_function;
    void *k_norm_rope_append_rows_f16_pipeline;
    void *v_norm_append_rows_function;
    void *v_norm_append_rows_pipeline;
    void *v_norm_append_rows_f16_function;
    void *v_norm_append_rows_f16_pipeline;
    void *kv_norm_append_rows_function;
    void *kv_norm_append_rows_pipeline;
    void *kv_norm_append_rows_f16_function;
    void *kv_norm_append_rows_f16_pipeline;
    void *attention_rows_function;
    void *attention_rows_pipeline;
    void *attention_rows_f16_function;
    void *attention_rows_f16_pipeline;
    void *attention_qnorm_rows_function;
    void *attention_qnorm_rows_pipeline;
    void *attention_qnorm_rows_f16_function;
    void *attention_qnorm_rows_f16_pipeline;
    void *attn_qnorm_dec_f16_library;
    void *attention_qnorm_dec_f16_function;
    void *attention_qnorm_dec_f16_pipeline;
    void *attn_flash_sg_f16_library;
    void *attention_qnorm_flash_sg_f16_function;
    void *attention_qnorm_flash_sg_f16_pipeline;
    void *attn_dec_combine_library;
    void *attention_dec_combine_function;
    void *attention_dec_combine_pipeline;
    struct geist_buffer *attn_dec_partials_buffer;
    size_t attn_dec_partials_capacity;
    void *sequence_command_buffer;
    void *sequence_compute_encoder;
    void *capture_manager;
    bool capture_done;
    int capture_skipped;
    /* diag: GEIST_METAL_SEQ_TRACE=1 — per-sequence encode/GPU timing. */
    uint64_t seq_dispatch_count;
    uint64_t seq_begin_ns;
    int sequence_token;
    enum geist_command_sequence_kind sequence_kind;
    bool sequence_active;
    bool sequence_has_work;
    bool captured_greedy_token_pending;
    uint32_t captured_greedy_vocab_size;
    uint32_t captured_greedy_token_count;
    bool decode_replay_valid;
    bool decode_replay_failed;
    bool decode_replay_replaying;
    bool decode_replay_enabled;
    uint32_t decode_replay_vocab_size;
    uint32_t decode_replay_token_count;
    size_t decode_replay_op_count;
    struct metal_decode_replay_op
        decode_replay_ops[METAL_DECODE_REPLAY_MAX_OPS];
    bool use_fused_attention_norm;
    bool use_fused_kv_norm_append;
    bool use_fused_attention_qk;
    bool use_fused_attention_qk_nt4;
    bool use_ple_block;
    bool use_q4k_n4;
    bool use_q4k_nt4;
    bool use_q4k_nt8;
    bool use_q4k_w4a8;
    bool use_q4k_linear_raw;
    bool use_q4k_gate_up_raw;
    bool use_q4k_m16_n2;
    bool use_q4k_mm_sg;
    bool use_ple_proj_norm_fused;
    bool use_rmsnorm_simd;
    bool use_q6k_n4;
    bool use_q6k_nt4;
    bool use_q6k_nt8;
    bool use_q6k_linear_raw;
    size_t q4k_nt4_max_n_out;
    bool pack_cache_enabled;
    char pack_cache_dir[PATH_MAX];
    bool profile_enabled;
    bool skip_next_dispatch; /* subtractive profiler: drop the next dispatch */
    bool double_next_dispatch; /* additive profiler: re-issue next dispatch */
    struct metal_profile_stat profile[METAL_PROFILE_STAGE_COUNT];
    char device_name[128];
    struct metal_q4k_nt4_cache_entry *q4k_nt4_cache;
    struct metal_q6k_nt4_cache_entry *q6k_nt4_cache;
    struct geist_buffer *q4k_w4a8_xq_buffer;
    struct geist_buffer *q4k_w4a8_scale_buffer;
    size_t q4k_w4a8_xq_capacity;

    void *MTLCreateSystemDefaultDevice;
    void *objc_msgSend;
    void *sel_registerName;
    void *objc_getClass;
};

struct geist_buffer {
    struct metal_state *owner;
    void *buffer;
    void *mapped;
    size_t bytes;
    enum geist_buffer_role role;
    unsigned int memory_flags;
    bool host_visible;
};

enum {
    METAL_RESOURCE_STORAGE_MODE_SHARED = 0u,
    METAL_RESOURCE_STORAGE_MODE_PRIVATE = 2u << 4,
    METAL_Q4K_THREADS_PER_ROW = 256u,
    METAL_Q4K_N4_THREADS = 64u,
    METAL_Q4K_BLOCK_ELEMS = 256u,
    METAL_Q4K_BLOCK_BYTES = 144u,
    METAL_Q4K_NT4_DEFAULT_MAX_N_OUT = 8192u,
    METAL_Q4K_NT4_LARGE_MAX_N_OUT = 262144u,
    METAL_Q5K_BLOCK_ELEMS = 256u,
    METAL_Q5K_BLOCK_BYTES = 176u,
    METAL_Q6K_BLOCK_ELEMS = 256u,
    METAL_Q6K_BLOCK_BYTES = 210u,
    METAL_Q6K_NT4_MIN_N_OUT = 1024u,
    METAL_Q6K_NT4_MAX_N_OUT = 8192u,
    METAL_Q4K_M_TILE = 8u,
    METAL_Q4K_M16_TILE = 16u,
    METAL_ELEM_THREADS = 256u,
    METAL_QNORM_ATTENTION_MAX_HEAD_DIM = 512u,
};

struct metal_size {
    size_t width;
    size_t height;
    size_t depth;
};

struct metal_q4k_params {
    uint32_t n_in;
    uint32_t n_out;
    uint32_t rows;
    uint32_t blocks_per_row;
    uint32_t x_offset;
    uint32_t w_byte_offset;
    uint32_t y_offset;
    uint32_t x_row_stride;
    uint32_t y_row_stride;
};

struct metal_q4k_quant_x_params {
    uint32_t n_in;
    uint32_t x_offset;
};

struct metal_q4k_w4a8_params {
    uint32_t n_in;
    uint32_t n_out;
    uint32_t blocks_per_row;
    uint32_t w_byte_offset;
    uint32_t y_offset;
};

struct metal_q4k_ple_gate_params {
    uint32_t n_in;
    uint32_t n_out;
    uint32_t rows;
    uint32_t blocks_per_row;
    uint32_t x_offset;
    uint32_t w_byte_offset;
    uint32_t ple_offset;
    uint32_t y_offset;
    uint32_t x_row_stride;
    uint32_t ple_row_stride;
    uint32_t y_row_stride;
};

struct metal_q4k_gate_up_params {
    uint32_t n_in;
    uint32_t n_out;
    uint32_t rows;
    uint32_t blocks_per_row;
    uint32_t x_offset;
    uint32_t gate_w_byte_offset;
    uint32_t up_w_byte_offset;
    uint32_t gate_y_offset;
    uint32_t up_y_offset;
    uint32_t x_row_stride;
    uint32_t y_row_stride;
};

struct metal_q4k_qk_params {
    uint32_t n_in;
    uint32_t q_out;
    uint32_t k_out;
    uint32_t rows;
    uint32_t blocks_per_row;
    uint32_t x_offset;
    uint32_t q_w_byte_offset;
    uint32_t k_w_byte_offset;
    uint32_t q_y_offset;
    uint32_t k_y_offset;
    uint32_t x_row_stride;
    uint32_t q_y_row_stride;
    uint32_t k_y_row_stride;
};

struct metal_rows_params {
    uint32_t rows;
    uint32_t cols;
    uint32_t x_offset;
    uint32_t w_offset;
    uint32_t y_offset;
    uint32_t x_row_stride;
    uint32_t y_row_stride;
    float eps;
};

struct metal_binary_rows_params {
    uint32_t rows;
    uint32_t cols;
    uint32_t a_offset;
    uint32_t b_offset;
    uint32_t y_offset;
    uint32_t a_row_stride;
    uint32_t b_row_stride;
    uint32_t y_row_stride;
};

struct metal_scale_rows_params {
    uint32_t rows;
    uint32_t cols;
    uint32_t x_offset;
    uint32_t y_offset;
    uint32_t x_row_stride;
    uint32_t y_row_stride;
    float scale;
};

struct metal_post_norm_params {
    uint32_t rows;
    uint32_t cols;
    uint32_t residual_offset;
    uint32_t x_offset;
    uint32_t w_offset;
    uint32_t y_offset;
    uint32_t residual_row_stride;
    uint32_t x_row_stride;
    uint32_t y_row_stride;
    float eps;
};

struct metal_rope_params {
    uint32_t rows;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t x_offset;
    uint32_t cos_offset;
    uint32_t sin_offset;
    uint32_t x_row_stride;
    uint32_t rope_row_stride;
    uint32_t rope_row_offset;
};

struct metal_kv_append_params {
    uint32_t elems;
    uint32_t kv_out;
    uint32_t k_offset;
    uint32_t v_offset;
    uint32_t k_cache_offset;
    uint32_t v_cache_offset;
    uint32_t q_position;
};

struct metal_norm_rope_params {
    uint32_t rows;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t x_offset;
    uint32_t w_offset;
    uint32_t cos_offset;
    uint32_t sin_offset;
    uint32_t x_row_stride;
    uint32_t rope_row_stride;
    uint32_t rope_row_offset;
    float eps;
};

struct metal_k_norm_rope_append_params {
    uint32_t rows;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t x_offset;
    uint32_t w_offset;
    uint32_t cos_offset;
    uint32_t sin_offset;
    uint32_t cache_offset;
    uint32_t x_row_stride;
    uint32_t rope_row_stride;
    uint32_t rope_row_offset;
    uint32_t q_position;
    float eps;
};

struct metal_v_norm_append_params {
    uint32_t rows;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t x_offset;
    uint32_t w_offset;
    uint32_t cache_offset;
    uint32_t x_row_stride;
    uint32_t q_position;
    float eps;
};

struct metal_attention_params {
    uint32_t rows;
    uint32_t kv_len;
    uint32_t q_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t q_position;
    uint32_t sliding_window;
    uint32_t q_offset;
    uint32_t k_cache_offset;
    uint32_t v_cache_offset;
    uint32_t y_offset;
};

struct metal_embed_params {
    uint32_t n;
    uint32_t dtype;
    uint32_t blocks_per_row;
    uint32_t w_byte_offset;
    uint32_t y_offset;
    uint32_t token_id;
    float scale;
};

struct metal_f32_params {
    uint32_t n_in;
    uint32_t n_out;
    uint32_t rows;
    uint32_t x_offset;
    uint32_t w_offset;
    uint32_t y_offset;
    uint32_t x_row_stride;
    uint32_t y_row_stride;
};

struct metal_f32_ple_gate_params {
    uint32_t n_in;
    uint32_t n_out;
    uint32_t rows;
    uint32_t x_offset;
    uint32_t w_offset;
    uint32_t ple_offset;
    uint32_t y_offset;
    uint32_t x_row_stride;
    uint32_t ple_row_stride;
    uint32_t y_row_stride;
};

struct metal_f32_ple_proj_norm_params {
    uint32_t n_in;
    uint32_t n_out;
    uint32_t rows;
    uint32_t x_offset;
    uint32_t w_offset;
    uint32_t residual_offset;
    uint32_t norm_weight_offset;
    uint32_t y_offset;
    uint32_t x_row_stride;
    uint32_t residual_row_stride;
    uint32_t y_row_stride;
    float eps;
};

struct metal_argmax_params {
    uint32_t n;
    uint32_t x_offset;
};

struct metal_argmax_batch_params {
    uint32_t rows;
    uint32_t n;
    uint32_t x_offset;
    uint32_t x_row_stride;
    uint32_t out_offset;
};

static const char metal_q4k_source[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct Params{uint n_in,n_out,rows,blocks_per_row,x_offset,w_byte_offset,y_offset,x_row_stride,y_row_stride;};\n"
    "static inline float load_f16(device const uchar*w,uint o){ushort b=ushort(w[o])|(ushort(w[o+1])<<8);return float(as_type<half>(b));}\n"
    "static inline void get_scale_min(uint j,device const uchar*q,thread uint&s,thread uint&m){if(j<4){s=uint(q[j])&63u;m=uint(q[j+4])&63u;}else{uint a=uint(q[j+4]),b=uint(q[j-4]),c=uint(q[j]);s=(a&15u)|((b>>6u)<<4u);m=(a>>4u)|((c>>6u)<<4u);}}\n"
    "static inline float deq(device const uchar*w,constant Params&p,uint row,uint k){uint br=k/256u,ib=k-br*256u,sub=ib/32u,idx=ib-sub*32u;uint bo=p.w_byte_offset+(row*p.blocks_per_row+br)*144u;float d=load_f16(w,bo),dm=load_f16(w,bo+2u);uint s,m;get_scale_min(sub,w+bo+4u,s,m);uint qb=uint(w[bo+16u+(sub/2u)*32u+idx]);uint q=(sub&1u)==0u?(qb&15u):(qb>>4u);return d*float(s)*float(q)-dm*float(m);}\n"
    "kernel void matvec_q4k(device const float *x [[buffer(0)]],\n"
    "                       device const uchar *w [[buffer(1)]],\n"
    "                       device float *y [[buffer(2)]],\n"
    "                       constant Params &p [[buffer(3)]],\n"
    "                       uint3 tg [[threadgroup_position_in_grid]],\n"
    "                       uint lid [[thread_index_in_threadgroup]]) {\n"
    "    threadgroup float partial[256];\n"
    "    uint row = tg.x;\n"
    "    uint batch = tg.y;\n"
    "    if (row >= p.n_out || batch >= p.rows) { return; }\n"
    "    float sum = 0.0f;\n"
    "    for (uint k = lid; k < p.n_in; k += 256u) {\n"
    "        sum += x[p.x_offset + batch * p.x_row_stride + k] * deq(w,p,row,k);\n"
    "    }\n"
    "    partial[lid] = sum;\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    for (uint stride = 128u; stride > 0u; stride >>= 1u) {\n"
    "        if (lid < stride) { partial[lid] += partial[lid + stride]; }\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    }\n"
    "    if (lid == 0u) { y[p.y_offset + batch * p.y_row_stride + row] = partial[0]; }\n"
    "}\n"
    "kernel void matmul_q4k_m8(device const float *x [[buffer(0)]],\n"
    "                         device const uchar *w [[buffer(1)]],\n"
    "                         device float *y [[buffer(2)]],\n"
    "                         constant Params &p [[buffer(3)]],\n"
    "                         uint3 tg [[threadgroup_position_in_grid]],\n"
    "                         uint lid [[thread_index_in_threadgroup]]) {\n"
    "    threadgroup float partial[8][256];\n"
    "    uint row = tg.x;\n"
    "    uint batch_base = tg.y * 8u;\n"
    "    if (row >= p.n_out || batch_base >= p.rows) { return; }\n"
    "    float sum[8];\n"
    "    for (uint m = 0u; m < 8u; m++) { sum[m] = 0.0f; }\n"
    "    for (uint k = lid; k < p.n_in; k += 256u) {\n"
    "        float wv = deq(w,p,row,k);\n"
    "        for (uint m = 0u; m < 8u; m++) {\n"
    "            uint batch = batch_base + m;\n"
    "            if (batch < p.rows) {\n"
    "                sum[m] += x[p.x_offset + batch * p.x_row_stride + k] * wv;\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    for (uint m = 0u; m < 8u; m++) { partial[m][lid] = sum[m]; }\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    for (uint stride = 128u; stride > 0u; stride >>= 1u) {\n"
    "        if (lid < stride) {\n"
    "            for (uint m = 0u; m < 8u; m++) {\n"
    "                partial[m][lid] += partial[m][lid + stride];\n"
    "            }\n"
    "        }\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    }\n"
    "    if (lid == 0u) {\n"
    "        for (uint m = 0u; m < 8u; m++) {\n"
    "            uint batch = batch_base + m;\n"
    "            if (batch < p.rows) {\n"
    "                y[p.y_offset + batch * p.y_row_stride + row] = partial[m][0];\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "}\n";

static const char metal_q4k_n4_source[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct P{uint ni,no,rows,bpr,xo,wo,yo,xs,ys;};\n"
    "kernel void matvec_q4k_n4(device const float*x[[buffer(0)]],device const uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],uint3 tg[[threadgroup_position_in_grid]],ushort ti[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]]){constexpr ushort km1=0x3f3f,km2=0x0f0f,km3=0xc0c0;uint ix=uint(ti)/8u,it=uint(ti)&7u,iq=it>>2u,ir=it&3u,b=tg.y,fr=(tg.x*2u+uint(sg))*2u;if(fr>=p.no||b>=p.rows)return;uint nb=p.ni>>8u;float sum0=0.0f,sum1=0.0f;device const float*y4=x+p.xo+b*p.xs+ix*256u+64u*iq+8u*ir;ushort sc16[4];thread const uchar*sc8=(thread const uchar*)sc16;for(uint ib=ix;ib<nb;ib+=4u){float yl[16],yh[16];float4 sumy=float4(0.0f);for(uint i=0u;i<8u;i++){yl[i]=y4[i];sumy[0]+=yl[i];yl[i+8u]=y4[i+32u];sumy[1]+=yl[i+8u];yh[i]=y4[i+128u];sumy[2]+=yh[i];yh[i+8u]=y4[i+160u];sumy[3]+=yh[i+8u];}for(uint rr=0u;rr<2u;rr++){uint row=fr+rr;if(row>=p.no)continue;device const uchar*wb=w+p.wo+(row*p.bpr+ib)*144u;device const ushort*sc=(device const ushort*)(wb+4u)+iq;device const ushort*q1=(device const ushort*)(wb+16u)+16u*iq+4u*ir;device const half*dh=(device const half*)wb;sc16[0]=sc[0]&km1;sc16[1]=sc[2]&km1;sc16[2]=((sc[4]>>0)&km2)|((sc[0]&km3)>>2);sc16[3]=((sc[4]>>4)&km2)|((sc[2]&km3)>>2);device const ushort*q2=q1+32u;float4 a1=float4(0.0f),a2=float4(0.0f);for(uint i=0u;i<4u;i++){ushort q=q1[i];a1[0]+=yl[2u*i+0u]*float(q&0x000f);a1[1]+=yl[2u*i+1u]*float(q&0x0f00);a1[2]+=yl[2u*i+8u]*float(q&0x00f0);a1[3]+=yl[2u*i+9u]*float(q&0xf000);q=q2[i];a2[0]+=yh[2u*i+0u]*float(q&0x000f);a2[1]+=yh[2u*i+1u]*float(q&0x0f00);a2[2]+=yh[2u*i+8u]*float(q&0x00f0);a2[3]+=yh[2u*i+9u]*float(q&0xf000);}float acc=float(dh[0])*((a1[0]+(1.0f/256.0f)*a1[1])*float(sc8[0])+(a1[2]+(1.0f/256.0f)*a1[3])*float(sc8[1])*(1.0f/16.0f)+(a2[0]+(1.0f/256.0f)*a2[1])*float(sc8[4])+(a2[2]+(1.0f/256.0f)*a2[3])*float(sc8[5])*(1.0f/16.0f))-float(dh[1])*(sumy[0]*float(sc8[2])+sumy[1]*float(sc8[3])+sumy[2]*float(sc8[6])+sumy[3]*float(sc8[7]));if(rr==0u)sum0+=acc;else sum1+=acc;}y4+=1024u;}float all0=simd_sum(sum0),all1=simd_sum(sum1);if(ti==0){uint o=p.yo+b*p.ys+fr;y[o]=all0;if(fr+1u<p.no)y[o+1u]=all1;}}\n";

static const char metal_q4k_m16_source[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct P{uint ni,no,rows,bpr,xo,wo,yo,xs,ys;};\n"
    "static inline float h(device const uchar*w,uint o){ushort b=ushort(w[o])|(ushort(w[o+1])<<8);return float(as_type<half>(b));}\n"
    "static inline void sm(uint j,device const uchar*q,thread uint&s,thread uint&m){if(j<4){s=uint(q[j])&63u;m=uint(q[j+4])&63u;}else{uint a=uint(q[j+4]),b=uint(q[j-4]),c=uint(q[j]);s=(a&15u)|((b>>6u)<<4u);m=(a>>4u)|((c>>6u)<<4u);}}\n"
    "static inline float dq(device const uchar*w,constant P&p,uint r,uint k){uint br=k>>8u,ib=k&255u,sub=ib>>5u,idx=ib&31u,bo=p.wo+(r*p.bpr+br)*144u;float d=h(w,bo),dm=h(w,bo+2u);uint s,m;sm(sub,w+bo+4u,s,m);uint qb=uint(w[bo+16u+(sub>>1u)*32u+idx]);uint q=(sub&1u)==0u?(qb&15u):(qb>>4u);return d*float(s)*float(q)-dm*float(m);}\n"
    "kernel void matmul_q4k_m16(device const float*x[[buffer(0)]],device const uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],uint3 tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){threadgroup float part[16][256];uint row=tg.x,bb=tg.y*16u;if(row>=p.no||bb>=p.rows)return;float s[16];for(uint m=0u;m<16u;m++)s[m]=0.0f;for(uint k=lid;k<p.ni;k+=256u){float wv=dq(w,p,row,k);for(uint m=0u;m<16u;m++){uint b=bb+m;if(b<p.rows)s[m]+=x[p.xo+b*p.xs+k]*wv;}}for(uint m=0u;m<16u;m++)part[m][lid]=s[m];threadgroup_barrier(mem_flags::mem_threadgroup);for(uint st=128u;st>0u;st>>=1u){if(lid<st){for(uint m=0u;m<16u;m++)part[m][lid]+=part[m][lid+st];}threadgroup_barrier(mem_flags::mem_threadgroup);}if(lid==0u){for(uint m=0u;m<16u;m++){uint b=bb+m;if(b<p.rows)y[p.yo+b*p.ys+row]=part[m][0];}}}\n";

static const char metal_q4k_m16_n2_source[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct P{uint ni,no,rows,bpr,xo,wo,yo,xs,ys;};\n"
    "static inline float h(device const uchar*w,uint o){ushort b=ushort(w[o])|(ushort(w[o+1])<<8);return float(as_type<half>(b));}\n"
    "static inline void sm(uint j,device const uchar*q,thread uint&s,thread uint&m){if(j<4){s=uint(q[j])&63u;m=uint(q[j+4])&63u;}else{uint a=uint(q[j+4]),b=uint(q[j-4]),c=uint(q[j]);s=(a&15u)|((b>>6u)<<4u);m=(a>>4u)|((c>>6u)<<4u);}}\n"
    "static inline float dq(device const uchar*w,constant P&p,uint r,uint k){uint br=k>>8u,ib=k&255u,sub=ib>>5u,idx=ib&31u,bo=p.wo+(r*p.bpr+br)*144u;float d=h(w,bo),dm=h(w,bo+2u);uint s,m;sm(sub,w+bo+4u,s,m);uint qb=uint(w[bo+16u+(sub>>1u)*32u+idx]);uint q=(sub&1u)==0u?(qb&15u):(qb>>4u);return d*float(s)*float(q)-dm*float(m);}\n"
    "kernel void matmul_q4k_m16_n2(device const float*x[[buffer(0)]],device const uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],uint3 tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){threadgroup float p0[16][256];threadgroup float p1[16][256];uint r0=tg.x*2u,bb=tg.y*16u;if(r0>=p.no||bb>=p.rows)return;bool hr=r0+1u<p.no;float s0[16],s1[16];for(uint m=0u;m<16u;m++){s0[m]=0.0f;s1[m]=0.0f;}for(uint k=lid;k<p.ni;k+=256u){float w0=dq(w,p,r0,k),w1=hr?dq(w,p,r0+1u,k):0.0f;for(uint m=0u;m<16u;m++){uint b=bb+m;if(b<p.rows){float xv=x[p.xo+b*p.xs+k];s0[m]+=xv*w0;s1[m]+=xv*w1;}}}for(uint m=0u;m<16u;m++){p0[m][lid]=s0[m];p1[m][lid]=s1[m];}threadgroup_barrier(mem_flags::mem_threadgroup);for(uint st=128u;st>0u;st>>=1u){if(lid<st){for(uint m=0u;m<16u;m++){p0[m][lid]+=p0[m][lid+st];p1[m][lid]+=p1[m][lid+st];}}threadgroup_barrier(mem_flags::mem_threadgroup);}if(lid==0u){for(uint m=0u;m<16u;m++){uint b=bb+m;if(b<p.rows){uint o=p.yo+b*p.ys+r0;y[o]=p0[m][0];if(hr)y[o+1u]=p1[m][0];}}}}\n";

static const char metal_q4k_mm_sg_source[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "#define FOR_UNROLL(x) _Pragma(\"clang loop unroll(full)\") for(x)\n"
    "struct P{uint ni,no,rows,bpr,xo,wo,yo,xs,ys;};\n"
    "struct block_q4_K{half d;half dmin;uchar scales[12];uchar qs[128];};\n"
    "static inline uchar2 sm2(short j,short k,device const uchar*q){return j<4?uchar2(uchar(q[j+k]&63),uchar(q[j+4+k]&63)):uchar2(uchar((q[j+4+k]&15)|((q[j-4+k]&192)>>2)),uchar((q[j+4+k]>>4)|((q[j+k]&192)>>2)));}\n"
    "static inline void dq4(device const block_q4_K*xb,short il,thread half4x4&r){device const uchar*q=xb->qs;short is=(il/4)*2;q=q+(il/4)*32+16*(il&1);il=il&3;uchar2 sc=sm2(is,il/2,xb->scales);float d=il<2?float(xb->d):float(xb->d)/16.0f;float mn=float(xb->dmin);float dl=d*float(sc[0]);float ml=mn*float(sc[1]);ushort mask=il<2?0x0f:0xf0;FOR_UNROLL(short i=0;i<16;i++){r[i/4][i%4]=half(dl*float(q[i]&mask)-ml);}}\n"
    "kernel void matmul_q4k_mm_sg(device const float*x[[buffer(0)]],device const uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],threadgroup char*shmem[[threadgroup(0)]],uint3 tg[[threadgroup_position_in_grid]],ushort ti[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]]){\n"
    "threadgroup half*sa=(threadgroup half*)shmem;threadgroup half*sb=(threadgroup half*)(shmem+4096);constexpr short NR0=64;constexpr short NR1=32;constexpr short NK=32;constexpr short NL0=2;constexpr short NL1=4;constexpr short QK_NL=16;uint r0=tg.y*uint(NR0);uint r1=tg.x*uint(NR1);if(r0>=p.no||r1>=p.rows)return;short nr0=short(min(p.no-r0,uint(NR0)));short nr1=short(min(p.rows-r1,uint(NR1)));short lr0=min(short(ti/NL0),short(nr0-1));short lr1=min(short(ti/NL1),short(nr1-1));short il0=short(ti%NL0);short il=il0;device const block_q4_K*wp=(device const block_q4_K*)(w+p.wo)+(r0+uint(lr0))*p.bpr;short iy=short(8*(ti%NL1));device const float*xp=x+p.xo+(r1+uint(lr1))*p.xs+uint(iy);simdgroup_half8x8 ma[4];simdgroup_half8x8 mb[2];simdgroup_float8x8 mc[8];FOR_UNROLL(short i=0;i<8;i++){mc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);}for(uint loop_k=0u;loop_k<p.ni;loop_k+=uint(NK)){half4x4 ta;dq4(wp,il,ta);threadgroup_barrier(mem_flags::mem_threadgroup);FOR_UNROLL(short i=0;i<16;i++){short sx=short(2*il0+i/8);short sy=short((ti/NL0)/8);short lx=short((ti/NL0)%8);short ly=short(i%8);short ib=short(8*sx+sy);*(sa+64*ib+8*ly+lx)=(loop_k+uint(16*il+i)<p.ni)?ta[i/4][i%4]:half(0.0);}short sx=short(ti%NL1);short sy=short((ti/NL1)/8);short ly=short((ti/NL1)%8);short ib=short(4*sx+sy);FOR_UNROLL(short i=0;i<8;i++){uint kk=loop_k+uint(iy+i);*(sb+64*ib+8*ly+i)=kk<p.ni?half(xp[i]):half(0.0);}il=(il+2<QK_NL)?il+2:il%2;wp=(il<2)?wp+1:wp;xp+=NK;threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup const half*lsma=sa+4*64*(sg%2);threadgroup const half*lsmb=sb+2*64*(sg/2);FOR_UNROLL(short ik=0;ik<NK/8;ik++){simdgroup_barrier(mem_flags::mem_none);FOR_UNROLL(short i=0;i<4;i++){simdgroup_load(ma[i],lsma+64*i,8,0,false);}simdgroup_barrier(mem_flags::mem_none);FOR_UNROLL(short i=0;i<2;i++){simdgroup_load(mb[i],lsmb+64*i,8,0,false);}simdgroup_barrier(mem_flags::mem_none);FOR_UNROLL(short i=0;i<8;i++){simdgroup_multiply_accumulate(mc[i],mb[i/4],ma[i%4],mc[i]);}lsma+=8*64;lsmb+=4*64;}}\n"
    "if(r0+uint(NR0)<=p.no&&r1+uint(NR1)<=p.rows){device float*C=y+p.yo+(r1+16u*uint(sg>>1))*p.ys+r0+32u*uint(sg&1);FOR_UNROLL(short i=0;i<8;i++){simdgroup_store(mc[i],C+8*(i%4)+8*p.ys*(i/4),p.ys,0,false);}}else{threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup float*tmp=((threadgroup float*)shmem)+32*(sg&1)+16*(sg>>1)*NR0;FOR_UNROLL(short i=0;i<8;i++){simdgroup_store(mc[i],tmp+8*(i%4)+8*NR0*(i/4),NR0,0,false);}threadgroup_barrier(mem_flags::mem_threadgroup);if(sg==0){for(uint j=ti;j<uint(nr1);j+=uint(NR1)){device float*D=y+p.yo+(r1+j)*p.ys+r0;threadgroup float*C=((threadgroup float*)shmem)+j*NR0;uint i=0u;for(;i+3u<uint(nr0);i+=4u){*((device float4*)(D+i))=*((threadgroup float4*)(C+i));}for(;i<uint(nr0);i++){D[i]=C[i];}}}}\n"
    "}\n";

/* Interior fast path of matmul_q4k_mm_sg (llama's FC_mul_mm bc_inp/bc_out
 * = false variant): no per-element bounds checks, vectorized 8-wide
 * activation staging, direct device-memory output. Dispatched only when
 * rows%32==0, n_out%64==0, n_in%32==0 and x is 8-float aligned. */
static const char metal_q4k_mm_sg_fast_source[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "#define FOR_UNROLL(x) _Pragma(\"clang loop unroll(full)\") for(x)\n"
    "struct P{uint ni,no,rows,bpr,xo,wo,yo,xs,ys;};\n"
    "struct block_q4_K{half d;half dmin;uchar scales[12];uchar qs[128];};\n"
    "static inline uchar2 sm2(short j,short k,device const uchar*q){return j<4?uchar2(uchar(q[j+k]&63),uchar(q[j+4+k]&63)):uchar2(uchar((q[j+4+k]&15)|((q[j-4+k]&192)>>2)),uchar((q[j+4+k]>>4)|((q[j+k]&192)>>2)));}\n"
    "static inline void dq4(device const block_q4_K*xb,short il,thread half4x4&r){device const uchar*q=xb->qs;short is=(il/4)*2;q=q+(il/4)*32+16*(il&1);il=il&3;uchar2 sc=sm2(is,il/2,xb->scales);float d=il<2?float(xb->d):float(xb->d)/16.0f;float mn=float(xb->dmin);float dl=d*float(sc[0]);float ml=mn*float(sc[1]);ushort mask=il<2?0x0f:0xf0;FOR_UNROLL(short i=0;i<16;i++){r[i/4][i%4]=half(dl*float(q[i]&mask)-ml);}}\n"
    "kernel void matmul_q4k_mm_sg_fast(device const float*x[[buffer(0)]],device const uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],threadgroup char*shmem[[threadgroup(0)]],uint3 tg[[threadgroup_position_in_grid]],ushort ti[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]]){\n"
    "threadgroup half*sa=(threadgroup half*)shmem;threadgroup half*sb=(threadgroup half*)(shmem+4096);"
    "constexpr short NK=32;constexpr short NL0=2;constexpr short NL1=4;constexpr short QK_NL=16;"
    "uint r0=tg.y*64u;uint r1=tg.x*32u;"
    "short lr0=short(ti/NL0);short lr1=short(ti/NL1);short il0=short(ti%NL0);short il=il0;"
    "device const block_q4_K*wp=(device const block_q4_K*)(w+p.wo)+(r0+uint(lr0))*p.bpr;"
    "short iy=short(8*(ti%NL1));"
    "device const float*xp=x+p.xo+(r1+uint(lr1))*p.xs+uint(iy);"
    "simdgroup_half8x8 ma[4];simdgroup_half8x8 mb[2];simdgroup_float8x8 mc[8];"
    "FOR_UNROLL(short i=0;i<8;i++){mc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);}"
    "short sx=short(ti%NL1);short sy=short((ti/NL1)/8);short lyb=short((ti/NL1)%8);short ibb=short(4*sx+sy);"
    "for(uint loop_k=0u;loop_k<p.ni;loop_k+=uint(NK)){"
    "half4x4 ta;dq4(wp,il,ta);"
    "threadgroup_barrier(mem_flags::mem_threadgroup);"
    "FOR_UNROLL(short i=0;i<16;i++){short ax=short(2*il0+i/8);short ay=short((ti/NL0)/8);short lx=short((ti/NL0)%8);short ly=short(i%8);short ib=short(8*ax+ay);*(sa+64*ib+8*ly+lx)=ta[i/4][i%4];}"
    "*(threadgroup half2x4*)(sb+64*ibb+8*lyb)=half2x4(*((device const float2x4*)xp));"
    "il=(il+2<QK_NL)?il+2:il%2;wp=(il<2)?wp+1:wp;xp+=NK;"
    "threadgroup_barrier(mem_flags::mem_threadgroup);"
    "threadgroup const half*lsma=sa+4*64*(sg%2);threadgroup const half*lsmb=sb+2*64*(sg/2);"
    "FOR_UNROLL(short ik=0;ik<NK/8;ik++){"
    "simdgroup_barrier(mem_flags::mem_none);"
    "FOR_UNROLL(short i=0;i<4;i++){simdgroup_load(ma[i],lsma+64*i,8,0,false);}"
    "simdgroup_barrier(mem_flags::mem_none);"
    "FOR_UNROLL(short i=0;i<2;i++){simdgroup_load(mb[i],lsmb+64*i,8,0,false);}"
    "simdgroup_barrier(mem_flags::mem_none);"
    "FOR_UNROLL(short i=0;i<8;i++){simdgroup_multiply_accumulate(mc[i],mb[i/4],ma[i%4],mc[i]);}"
    "lsma+=8*64;lsmb+=4*64;}}"
    "device float*C=y+p.yo+(r1+16u*uint(sg>>1))*p.ys+r0+32u*uint(sg&1);"
    "FOR_UNROLL(short i=0;i<8;i++){simdgroup_store(mc[i],C+8*(i%4)+8*p.ys*(i/4),p.ys,0,false);}"
    "}\n";

static const char metal_q4k_nt4_source[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct P{uint ni,no,rows,bpr,xo,wo,yo,xs,ys;};\n"
    "static inline uint u32(device const uchar*w,uint o){return uint(w[o])|(uint(w[o+1])<<8)|(uint(w[o+2])<<16)|(uint(w[o+3])<<24);}\n"
    "static inline float f32(device const uchar*w,uint o){return as_type<float>(u32(w,o));}\n"
    "static inline float dq(device const uchar*w,constant P&p,uint r,uint k){uint nt=r>>2u,nr=r&3u,br=k>>8u,ib=k&255u,sub=ib>>5u,bo=p.wo+24u+((nt*p.bpr+br)*4u+nr)*280u;float d=f32(w,bo),dm=f32(w,bo+4u),sc=float(w[bo+8u+sub]),mn=float(w[bo+16u+sub]),q=float(w[bo+24u+ib]);return d*sc*q-dm*mn;}\n"
    "kernel void matvec_q4k_nt4(device const float*x[[buffer(0)]],device const uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],uint3 tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){threadgroup float p0[64],p1[64],p2[64],p3[64];uint r0=tg.x*4u,b=tg.y;if(r0>=p.no||b>=p.rows)return;bool h1=r0+1u<p.no,h2=r0+2u<p.no,h3=r0+3u<p.no;float s0=0.0f,s1=0.0f,s2=0.0f,s3=0.0f;for(uint k=lid;k<p.ni;k+=64u){float xv=x[p.xo+b*p.xs+k];s0+=xv*dq(w,p,r0,k);if(h1)s1+=xv*dq(w,p,r0+1u,k);if(h2)s2+=xv*dq(w,p,r0+2u,k);if(h3)s3+=xv*dq(w,p,r0+3u,k);}p0[lid]=s0;p1[lid]=s1;p2[lid]=s2;p3[lid]=s3;threadgroup_barrier(mem_flags::mem_threadgroup);for(uint st=32u;st>0u;st>>=1u){if(lid<st){p0[lid]+=p0[lid+st];p1[lid]+=p1[lid+st];p2[lid]+=p2[lid+st];p3[lid]+=p3[lid+st];}threadgroup_barrier(mem_flags::mem_threadgroup);}if(lid==0u){uint o=p.yo+b*p.ys+r0;y[o]=p0[0];if(h1)y[o+1u]=p1[0];if(h2)y[o+2u]=p2[0];if(h3)y[o+3u]=p3[0];}}\n"
    "kernel void matvec_q4k_nt8(device const float*x[[buffer(0)]],device const uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],uint3 tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]],uint sg[[simdgroup_index_in_threadgroup]],uint sl[[thread_index_in_simdgroup]]){threadgroup float p0[4],p1[4],p2[4],p3[4],p4[4],p5[4],p6[4],p7[4];uint r0=tg.x*8u,b=tg.y;if(r0>=p.no||b>=p.rows)return;if(lid<4u){p0[lid]=0.0f;p1[lid]=0.0f;p2[lid]=0.0f;p3[lid]=0.0f;p4[lid]=0.0f;p5[lid]=0.0f;p6[lid]=0.0f;p7[lid]=0.0f;}threadgroup_barrier(mem_flags::mem_threadgroup);bool h1=r0+1u<p.no,h2=r0+2u<p.no,h3=r0+3u<p.no,h4=r0+4u<p.no,h5=r0+5u<p.no,h6=r0+6u<p.no,h7=r0+7u<p.no;float s0=0.0f,s1=0.0f,s2=0.0f,s3=0.0f,s4=0.0f,s5=0.0f,s6=0.0f,s7=0.0f;for(uint k=lid;k<p.ni;k+=64u){float xv=x[p.xo+b*p.xs+k];s0+=xv*dq(w,p,r0,k);if(h1)s1+=xv*dq(w,p,r0+1u,k);if(h2)s2+=xv*dq(w,p,r0+2u,k);if(h3)s3+=xv*dq(w,p,r0+3u,k);if(h4)s4+=xv*dq(w,p,r0+4u,k);if(h5)s5+=xv*dq(w,p,r0+5u,k);if(h6)s6+=xv*dq(w,p,r0+6u,k);if(h7)s7+=xv*dq(w,p,r0+7u,k);}float t0=simd_sum(s0),t1=simd_sum(s1),t2=simd_sum(s2),t3=simd_sum(s3),t4=simd_sum(s4),t5=simd_sum(s5),t6=simd_sum(s6),t7=simd_sum(s7);if(sl==0u&&sg<4u){p0[sg]=t0;p1[sg]=t1;p2[sg]=t2;p3[sg]=t3;p4[sg]=t4;p5[sg]=t5;p6[sg]=t6;p7[sg]=t7;}threadgroup_barrier(mem_flags::mem_threadgroup);if(lid==0u){uint o=p.yo+b*p.ys+r0;y[o]=p0[0]+p0[1]+p0[2]+p0[3];if(h1)y[o+1u]=p1[0]+p1[1]+p1[2]+p1[3];if(h2)y[o+2u]=p2[0]+p2[1]+p2[2]+p2[3];if(h3)y[o+3u]=p3[0]+p3[1]+p3[2]+p3[3];if(h4)y[o+4u]=p4[0]+p4[1]+p4[2]+p4[3];if(h5)y[o+5u]=p5[0]+p5[1]+p5[2]+p5[3];if(h6)y[o+6u]=p6[0]+p6[1]+p6[2]+p6[3];if(h7)y[o+7u]=p7[0]+p7[1]+p7[2]+p7[3];}}\n";

static const char metal_q4k_ple_gate_nt8_source[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct P{uint ni,no,rows,bpr,xo,wo,po,yo,xs,ps,ys;};\n"
    "static inline uint u32(device const uchar*w,uint o){return uint(w[o])|(uint(w[o+1])<<8)|(uint(w[o+2])<<16)|(uint(w[o+3])<<24);}\n"
    "static inline float f32(device const uchar*w,uint o){return as_type<float>(u32(w,o));}\n"
    "static inline float dq(device const uchar*w,constant P&p,uint r,uint k){uint nt=r>>2u,nr=r&3u,br=k>>8u,ib=k&255u,sub=ib>>5u,bo=p.wo+24u+((nt*p.bpr+br)*4u+nr)*280u;float d=f32(w,bo),dm=f32(w,bo+4u),sc=float(w[bo+8u+sub]),mn=float(w[bo+16u+sub]),q=float(w[bo+24u+ib]);return d*sc*q-dm*mn;}\n"
    "static inline float gelu(float x){return 0.5f*x*(1.0f+tanh(clamp(0.7978845608028654f*(x+0.044715f*x*x*x),-10.0f,10.0f)));}\n"
    "kernel void ple_gate_q4k_nt8(device const float*x[[buffer(0)]],device const uchar*w[[buffer(1)]],device const float*ple[[buffer(2)]],device float*y[[buffer(3)]],constant P&p[[buffer(4)]],uint3 tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]],uint sg[[simdgroup_index_in_threadgroup]],uint sl[[thread_index_in_simdgroup]]){threadgroup float p0[4],p1[4],p2[4],p3[4],p4[4],p5[4],p6[4],p7[4];uint r0=tg.x*8u,b=tg.y;if(r0>=p.no||b>=p.rows)return;if(lid<4u){p0[lid]=0.0f;p1[lid]=0.0f;p2[lid]=0.0f;p3[lid]=0.0f;p4[lid]=0.0f;p5[lid]=0.0f;p6[lid]=0.0f;p7[lid]=0.0f;}threadgroup_barrier(mem_flags::mem_threadgroup);bool h1=r0+1u<p.no,h2=r0+2u<p.no,h3=r0+3u<p.no,h4=r0+4u<p.no,h5=r0+5u<p.no,h6=r0+6u<p.no,h7=r0+7u<p.no;float s0=0.0f,s1=0.0f,s2=0.0f,s3=0.0f,s4=0.0f,s5=0.0f,s6=0.0f,s7=0.0f;for(uint k=lid;k<p.ni;k+=64u){float xv=x[p.xo+b*p.xs+k];s0+=xv*dq(w,p,r0,k);if(h1)s1+=xv*dq(w,p,r0+1u,k);if(h2)s2+=xv*dq(w,p,r0+2u,k);if(h3)s3+=xv*dq(w,p,r0+3u,k);if(h4)s4+=xv*dq(w,p,r0+4u,k);if(h5)s5+=xv*dq(w,p,r0+5u,k);if(h6)s6+=xv*dq(w,p,r0+6u,k);if(h7)s7+=xv*dq(w,p,r0+7u,k);}float t0=simd_sum(s0),t1=simd_sum(s1),t2=simd_sum(s2),t3=simd_sum(s3),t4=simd_sum(s4),t5=simd_sum(s5),t6=simd_sum(s6),t7=simd_sum(s7);if(sl==0u&&sg<4u){p0[sg]=t0;p1[sg]=t1;p2[sg]=t2;p3[sg]=t3;p4[sg]=t4;p5[sg]=t5;p6[sg]=t6;p7[sg]=t7;}threadgroup_barrier(mem_flags::mem_threadgroup);if(lid==0u){uint po=p.po+b*p.ps+r0,o=p.yo+b*p.ys+r0;y[o]=gelu(p0[0]+p0[1]+p0[2]+p0[3])*ple[po];if(h1)y[o+1u]=gelu(p1[0]+p1[1]+p1[2]+p1[3])*ple[po+1u];if(h2)y[o+2u]=gelu(p2[0]+p2[1]+p2[2]+p2[3])*ple[po+2u];if(h3)y[o+3u]=gelu(p3[0]+p3[1]+p3[2]+p3[3])*ple[po+3u];if(h4)y[o+4u]=gelu(p4[0]+p4[1]+p4[2]+p4[3])*ple[po+4u];if(h5)y[o+5u]=gelu(p5[0]+p5[1]+p5[2]+p5[3])*ple[po+5u];if(h6)y[o+6u]=gelu(p6[0]+p6[1]+p6[2]+p6[3])*ple[po+6u];if(h7)y[o+7u]=gelu(p7[0]+p7[1]+p7[2]+p7[3])*ple[po+7u];}}\n";

static const char metal_q6k_source[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct Params{uint n_in,n_out,rows,blocks_per_row,x_offset,w_byte_offset,y_offset,x_row_stride,y_row_stride;};\n"
    "static inline float load_f16(device const uchar*w,uint o){ushort b=ushort(w[o])|(ushort(w[o+1])<<8);return float(as_type<half>(b));}\n"
    "static inline int load_i8(device const uchar*w,uint o){uint u=uint(w[o]);return u<128u?int(u):int(u)-256;}\n"
    "static inline float deq6(device const uchar*w,constant Params&p,uint row,uint k){uint br=k/256u,ib=k-br*256u,hi=ib/128u,ih=ib-hi*128u,stream=ih/32u,l=ih-stream*32u,is=l/16u;uint bo=p.w_byte_offset+(row*p.blocks_per_row+br)*210u;uint qlb=bo+hi*64u,qhb=bo+128u+hi*32u,scb=bo+192u+hi*8u;uint ql0=uint(w[qlb+l]),ql1=uint(w[qlb+l+32u]),qh=uint(w[qhb+l]);uint qu,si;if(stream==0u){qu=(ql0&15u)|(((qh>>0u)&3u)<<4u);si=is+0u;}else if(stream==1u){qu=(ql1&15u)|(((qh>>2u)&3u)<<4u);si=is+2u;}else if(stream==2u){qu=(ql0>>4u)|(((qh>>4u)&3u)<<4u);si=is+4u;}else{qu=(ql1>>4u)|(((qh>>6u)&3u)<<4u);si=is+6u;}float d=load_f16(w,bo+208u);return d*float(load_i8(w,scb+si))*float(int(qu)-32);}\n"
    "kernel void matvec_q6k(device const float*x[[buffer(0)]],device const uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant Params&p[[buffer(3)]],uint3 tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){threadgroup float partial[256];uint row=tg.x,batch=tg.y;if(row>=p.n_out||batch>=p.rows){return;}float sum=0.0f;for(uint k=lid;k<p.n_in;k+=256u){sum+=x[p.x_offset+batch*p.x_row_stride+k]*deq6(w,p,row,k);}partial[lid]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);for(uint stride=128u;stride>0u;stride>>=1u){if(lid<stride){partial[lid]+=partial[lid+stride];}threadgroup_barrier(mem_flags::mem_threadgroup);}if(lid==0u){y[p.y_offset+batch*p.y_row_stride+row]=partial[0];}}\n"
    "kernel void matmul_q6k_m8(device const float*x[[buffer(0)]],device const uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant Params&p[[buffer(3)]],uint3 tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){threadgroup float partial[8][256];uint row=tg.x,batch_base=tg.y*8u;if(row>=p.n_out||batch_base>=p.rows){return;}float sum[8];for(uint m=0u;m<8u;m++){sum[m]=0.0f;}for(uint k=lid;k<p.n_in;k+=256u){float wv=deq6(w,p,row,k);for(uint m=0u;m<8u;m++){uint batch=batch_base+m;if(batch<p.rows){sum[m]+=x[p.x_offset+batch*p.x_row_stride+k]*wv;}}}for(uint m=0u;m<8u;m++){partial[m][lid]=sum[m];}threadgroup_barrier(mem_flags::mem_threadgroup);for(uint stride=128u;stride>0u;stride>>=1u){if(lid<stride){for(uint m=0u;m<8u;m++){partial[m][lid]+=partial[m][lid+stride];}}threadgroup_barrier(mem_flags::mem_threadgroup);}if(lid==0u){for(uint m=0u;m<8u;m++){uint batch=batch_base+m;if(batch<p.rows){y[p.y_offset+batch*p.y_row_stride+row]=partial[m][0];}}}}\n";

/* 64x32 q6k GEMM (NK=8, proven tile structure). Weight staging dequants one
 * 8-elem k-run per thread: d and the int8 scale load once per run instead of
 * per element (deq6 was ~46% of the kernel's time). */
static const char metal_q6k_mm_sg_source[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct Params{uint n_in,n_out,rows,blocks_per_row,x_offset,w_byte_offset,y_offset,x_row_stride,y_row_stride;};\n"
    "static inline float load_f16(device const uchar*w,uint o){ushort b=ushort(w[o])|(ushort(w[o+1])<<8);return float(as_type<half>(b));}\n"
    "static inline int load_i8(device const uchar*w,uint o){uint u=uint(w[o]);return u<128u?int(u):int(u)-256;}\n"
    "kernel void matmul_q6k_sg(device const float*x[[buffer(0)]],device const uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant Params&p[[buffer(3)]],uint3 tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]],ushort sgitg[[simdgroup_index_in_threadgroup]],ushort tiisg[[thread_index_in_simdgroup]]){"
    "threadgroup half as[512];threadgroup half bs[256];threadgroup float cs[256];"
    "uint b0=tg.y*64u,o0=tg.x*32u;"
    "simdgroup_float8x8 acc[8];for(uint i=0u;i<8u;i++)acc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);"
    "for(uint k0=0u;k0<p.n_in;k0+=8u){"
    "for(uint i=lid;i<512u;i+=128u){uint r=i/8u,c=i%8u;as[i]=(b0+r<p.rows&&k0+c<p.n_in)?half(x[p.x_offset+(b0+r)*p.x_row_stride+k0+c]):half(0.0f);}"
    "if(lid<32u){uint o=o0+lid;"
    "if(o<p.n_out&&k0<p.n_in){"
    "uint br=k0>>8u,ib=k0&255u,hi=ib>>7u,ih=ib&127u,stream=ih>>5u,l=ih&31u,is=l>>4u;"
    "uint bo=p.w_byte_offset+(o*p.blocks_per_row+br)*210u;"
    "uint qlb=bo+hi*64u+l+((stream&1u)?32u:0u),qhb=bo+128u+hi*32u+l,scb=bo+192u+hi*8u+is+2u*stream;"
    "uint qsh=(stream>=2u)?4u:0u,hsh=2u*stream;"
    "float ds=load_f16(w,bo+208u)*float(load_i8(w,scb));"
    "for(uint j=0u;j<8u;j++){uint qu=((uint(w[qlb+j])>>qsh)&15u)|(((uint(w[qhb+j])>>hsh)&3u)<<4u);bs[j*32u+lid]=half(ds*float(int(qu)-32));}"
    "}else{for(uint j=0u;j<8u;j++)bs[j*32u+lid]=half(0.0f);}}"
    "threadgroup_barrier(mem_flags::mem_threadgroup);"
    "simdgroup_half8x8 ma,mb;simdgroup_load(mb,bs+8u*uint(sgitg),32);"
    "for(uint rj=0u;rj<8u;rj++){simdgroup_load(ma,as+64u*rj,8);simdgroup_multiply_accumulate(acc[rj],ma,mb,acc[rj]);}"
    "threadgroup_barrier(mem_flags::mem_threadgroup);}"
    "threadgroup float*lcs=cs+64u*uint(sgitg);"
    "for(uint rj=0u;rj<8u;rj++){simdgroup_store(acc[rj],lcs,8);simdgroup_barrier(mem_flags::mem_threadgroup);"
    "for(uint i=uint(tiisg);i<64u;i+=32u){uint r=i/8u,c=i%8u;if(b0+8u*rj+r<p.rows&&o0+8u*uint(sgitg)+c<p.n_out)y[p.y_offset+(b0+8u*rj+r)*p.y_row_stride+o0+8u*uint(sgitg)+c]=lcs[i];}simdgroup_barrier(mem_flags::mem_threadgroup);}}\n";

/* Interior fast path of matmul_q6k_sg: no per-element bounds checks,
 * vectorized 8-wide activation staging, direct simdgroup_store output.
 * Dispatched only when rows%64==0, n_out%32==0 and x is 8-float aligned. */
static const char metal_q6k_mm_sg_fast_source[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct Params{uint n_in,n_out,rows,blocks_per_row,x_offset,w_byte_offset,y_offset,x_row_stride,y_row_stride;};\n"
    "static inline float load_f16(device const uchar*w,uint o){ushort b=ushort(w[o])|(ushort(w[o+1])<<8);return float(as_type<half>(b));}\n"
    "static inline int load_i8(device const uchar*w,uint o){uint u=uint(w[o]);return u<128u?int(u):int(u)-256;}\n"
    "kernel void matmul_q6k_sg_fast(device const float*x[[buffer(0)]],device const uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant Params&p[[buffer(3)]],uint3 tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]],ushort sgitg[[simdgroup_index_in_threadgroup]],ushort tiisg[[thread_index_in_simdgroup]]){"
    "threadgroup half as[512];threadgroup half bs[256];"
    "uint b0=tg.y*64u,o0=tg.x*32u;"
    "simdgroup_float8x8 acc[8];for(uint i=0u;i<8u;i++)acc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);"
    "for(uint k0=0u;k0<p.n_in;k0+=8u){"
    "if(lid<64u){*(threadgroup half2x4*)(as+lid*8u)=half2x4(*((device const float2x4*)(x+p.x_offset+(b0+lid)*p.x_row_stride+k0)));}"
    "else if(lid<96u){uint o=o0+(lid-64u);"
    "uint br=k0>>8u,ib=k0&255u,hi=ib>>7u,ih=ib&127u,stream=ih>>5u,l=ih&31u,is=l>>4u;"
    "uint bo=p.w_byte_offset+(o*p.blocks_per_row+br)*210u;"
    "uint qlb=bo+hi*64u+l+((stream&1u)?32u:0u),qhb=bo+128u+hi*32u+l,scb=bo+192u+hi*8u+is+2u*stream;"
    "uint qsh=(stream>=2u)?4u:0u,hsh=2u*stream;"
    "float ds=load_f16(w,bo+208u)*float(load_i8(w,scb));"
    "for(uint j=0u;j<8u;j++){uint qu=((uint(w[qlb+j])>>qsh)&15u)|(((uint(w[qhb+j])>>hsh)&3u)<<4u);bs[j*32u+(lid-64u)]=half(ds*float(int(qu)-32));}}"
    "threadgroup_barrier(mem_flags::mem_threadgroup);"
    "simdgroup_half8x8 ma,mb;simdgroup_load(mb,bs+8u*uint(sgitg),32);"
    "for(uint rj=0u;rj<8u;rj++){simdgroup_load(ma,as+64u*rj,8);simdgroup_multiply_accumulate(acc[rj],ma,mb,acc[rj]);}"
    "threadgroup_barrier(mem_flags::mem_threadgroup);}"
    "uint co=p.y_offset+o0+8u*uint(sgitg);"
    "for(uint rj=0u;rj<8u;rj++){simdgroup_store(acc[rj],y+co+(b0+8u*rj)*p.y_row_stride,p.y_row_stride,0,false);}"
    "}\n";

/* llama.cpp kernel_mul_mv_q6_K_f32 structure: 2 simdgroups/threadgroup, 2 rows
 * per simdgroup (grid.x*4 rows/tg, unchanged), block-coherent loads — each
 * thread handles 16 elems of a 256-block via 4 contiguous ql/qh bytes and
 * int8 scales, one f16 d per (row,block). */
static const char metal_q6k_n4_source[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct P{uint ni,no,rows,bpr,xo,wo,yo,xs,ys;};\n"
    "static inline float h(device const uchar*w,uint o){ushort b=ushort(w[o])|(ushort(w[o+1])<<8);return float(as_type<half>(b));}\n"
    "static inline int i8(device const uchar*w,uint o){uint u=uint(w[o]);return u<128u?int(u):int(u)-256;}\n"
    "kernel void matvec_q6k_n4(device const float*x[[buffer(0)]],device const uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],uint3 tg[[threadgroup_position_in_grid]],ushort ti[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]]){"
    "uint b=tg.y,fr=(tg.x*2u+uint(sg))*2u;if(fr>=p.no||b>=p.rows)return;"
    "uint nb=p.ni>>8u;"
    "ushort tid=ti/2u,ix=ti%2u,ip=tid/8u,il=tid%8u,l0=4u*il,is=8u*ip+l0/16u;"
    "uint yoff=128u*uint(ip)+uint(l0),qlo=64u*uint(ip)+uint(l0),qho=32u*uint(ip)+uint(l0);"
    "device const float*yy=x+p.xo+b*p.xs+yoff;"
    "float yl[16];float sf0=0.0f,sf1=0.0f;"
    "for(uint i=uint(ix);i<nb;i+=2u){"
    "device const float*yv=yy+i*256u;"
    "for(ushort l=0;l<4;l++){yl[4*l+0]=yv[l];yl[4*l+1]=yv[l+32];yl[4*l+2]=yv[l+64];yl[4*l+3]=yv[l+96];}"
    "for(uint rr=0u;rr<2u;rr++){uint row=fr+rr;if(row>=p.no)break;"
    "uint bo=p.wo+(row*p.bpr+i)*210u;"
    "device const uchar*q1=w+bo+qlo;device const uchar*q2=q1+32u;device const uchar*qh=w+bo+128u+qho;"
    "uint sco=bo+192u+uint(is);"
    "float4 s=float4(0.0f);"
    "for(ushort l=0;l<4;l++){uint hb=uint(qh[l]);"
    "s[0]+=yl[4*l+0]*float(int((uint(q1[l])&15u)|((hb&3u)<<4u))-32);"
    "s[1]+=yl[4*l+1]*float(int((uint(q2[l])&15u)|(((hb>>2u)&3u)<<4u))-32);"
    "s[2]+=yl[4*l+2]*float(int((uint(q1[l])>>4u)|(((hb>>4u)&3u)<<4u))-32);"
    "s[3]+=yl[4*l+3]*float(int((uint(q2[l])>>4u)|(((hb>>6u)&3u)<<4u))-32);}"
    "float acc=h(w,bo+208u)*(s[0]*float(i8(w,sco))+s[1]*float(i8(w,sco+2u))+s[2]*float(i8(w,sco+4u))+s[3]*float(i8(w,sco+6u)));"
    "if(rr==0u)sf0+=acc;else sf1+=acc;}}"
    "float a0=simd_sum(sf0),a1=simd_sum(sf1);"
    "if(ti==0){uint o=p.yo+b*p.ys+fr;y[o]=a0;if(fr+1u<p.no)y[o+1u]=a1;}}\n";

static const char metal_q6k_m16_source[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct P{uint ni,no,rows,bpr,xo,wo,yo,xs,ys;};\n"
    "static inline float h(device const uchar*w,uint o){ushort b=ushort(w[o])|(ushort(w[o+1])<<8);return float(as_type<half>(b));}\n"
    "static inline int i8(device const uchar*w,uint o){uint u=uint(w[o]);return u<128u?int(u):int(u)-256;}\n"
    "static inline float dq6(device const uchar*w,constant P&p,uint r,uint k){uint br=k>>8u,ib=k&255u,hi=ib>>7u,ih=ib&127u,stream=ih>>5u,l=ih&31u,is=l>>4u,bo=p.wo+(r*p.bpr+br)*210u;uint qlb=bo+hi*64u,qhb=bo+128u+hi*32u,scb=bo+192u+hi*8u;uint ql0=uint(w[qlb+l]),ql1=uint(w[qlb+l+32u]),qh=uint(w[qhb+l]),qu,si;if(stream==0u){qu=(ql0&15u)|(((qh>>0u)&3u)<<4u);si=is+0u;}else if(stream==1u){qu=(ql1&15u)|(((qh>>2u)&3u)<<4u);si=is+2u;}else if(stream==2u){qu=(ql0>>4u)|(((qh>>4u)&3u)<<4u);si=is+4u;}else{qu=(ql1>>4u)|(((qh>>6u)&3u)<<4u);si=is+6u;}return h(w,bo+208u)*float(i8(w,scb+si))*float(int(qu)-32);}\n"
    "kernel void matmul_q6k_m16(device const float*x[[buffer(0)]],device const uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],uint3 tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){threadgroup float part[16][256];uint row=tg.x,bb=tg.y*16u;if(row>=p.no||bb>=p.rows)return;float s[16];for(uint m=0u;m<16u;m++)s[m]=0.0f;for(uint k=lid;k<p.ni;k+=256u){float wv=dq6(w,p,row,k);for(uint m=0u;m<16u;m++){uint b=bb+m;if(b<p.rows)s[m]+=x[p.xo+b*p.xs+k]*wv;}}for(uint m=0u;m<16u;m++)part[m][lid]=s[m];threadgroup_barrier(mem_flags::mem_threadgroup);for(uint st=128u;st>0u;st>>=1u){if(lid<st){for(uint m=0u;m<16u;m++)part[m][lid]+=part[m][lid+st];}threadgroup_barrier(mem_flags::mem_threadgroup);}if(lid==0u){for(uint m=0u;m<16u;m++){uint b=bb+m;if(b<p.rows)y[p.yo+b*p.ys+row]=part[m][0];}}}\n";

static const char metal_q6k_nt4_source[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct P{uint ni,no,rows,bpr,xo,wo,yo,xs,ys;};\n"
    "static inline float h(device const uchar*w,uint o){ushort b=ushort(w[o])|(ushort(w[o+1])<<8);return float(as_type<half>(b));}\n"
    "static inline int i8(device const uchar*w,uint o){uint u=uint(w[o]);return u<128u?int(u):int(u)-256;}\n"
    "static inline float dq6(device const uchar*w,constant P&p,uint r,uint k){uint nt=r>>2u,nr=r&3u,br=k>>8u,ib=k&255u,hi=ib/128u,ih=ib-hi*128u,stream=ih/32u,l=ih-stream*32u,is=l/16u,bo=p.wo+((nt*p.bpr+br)*4u+nr)*210u;uint qlb=bo+hi*64u,qhb=bo+128u+hi*32u,scb=bo+192u+hi*8u;uint ql0=uint(w[qlb+l]),ql1=uint(w[qlb+l+32u]),qh=uint(w[qhb+l]),qu,si;if(stream==0u){qu=(ql0&15u)|(((qh>>0u)&3u)<<4u);si=is+0u;}else if(stream==1u){qu=(ql1&15u)|(((qh>>2u)&3u)<<4u);si=is+2u;}else if(stream==2u){qu=(ql0>>4u)|(((qh>>4u)&3u)<<4u);si=is+4u;}else{qu=(ql1>>4u)|(((qh>>6u)&3u)<<4u);si=is+6u;}return h(w,bo+208u)*float(i8(w,scb+si))*float(int(qu)-32);}\n"
    "kernel void matvec_q6k_nt4(device const float*x[[buffer(0)]],device const uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],uint3 tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]],uint sg[[simdgroup_index_in_threadgroup]],uint sl[[thread_index_in_simdgroup]]){threadgroup float p0[4],p1[4],p2[4],p3[4];uint r0=tg.x*4u,b=tg.y;if(r0>=p.no||b>=p.rows)return;if(lid<4u){p0[lid]=0.0f;p1[lid]=0.0f;p2[lid]=0.0f;p3[lid]=0.0f;}threadgroup_barrier(mem_flags::mem_threadgroup);bool h1=r0+1u<p.no,h2=r0+2u<p.no,h3=r0+3u<p.no;float s0=0.0f,s1=0.0f,s2=0.0f,s3=0.0f;for(uint k=lid;k<p.ni;k+=64u){float xv=x[p.xo+b*p.xs+k];s0+=xv*dq6(w,p,r0,k);if(h1)s1+=xv*dq6(w,p,r0+1u,k);if(h2)s2+=xv*dq6(w,p,r0+2u,k);if(h3)s3+=xv*dq6(w,p,r0+3u,k);}float t0=simd_sum(s0),t1=simd_sum(s1),t2=simd_sum(s2),t3=simd_sum(s3);if(sl==0u&&sg<4u){p0[sg]=t0;p1[sg]=t1;p2[sg]=t2;p3[sg]=t3;}threadgroup_barrier(mem_flags::mem_threadgroup);if(lid==0u){uint o=p.yo+b*p.ys+r0;y[o]=p0[0]+p0[1]+p0[2]+p0[3];if(h1)y[o+1u]=p1[0]+p1[1]+p1[2]+p1[3];if(h2)y[o+2u]=p2[0]+p2[1]+p2[2]+p2[3];if(h3)y[o+3u]=p3[0]+p3[1]+p3[2]+p3[3];}}\n";

static const char metal_q6k_nt8_source[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct P{uint ni,no,rows,bpr,xo,wo,yo,xs,ys;};\n"
    "static inline float h(device const uchar*w,uint o){ushort b=ushort(w[o])|(ushort(w[o+1])<<8);return float(as_type<half>(b));}\n"
    "static inline int i8(device const uchar*w,uint o){uint u=uint(w[o]);return u<128u?int(u):int(u)-256;}\n"
    "static inline float dq6(device const uchar*w,constant P&p,uint r,uint k){uint nt=r>>2u,nr=r&3u,br=k>>8u,ib=k&255u,hi=ib/128u,ih=ib-hi*128u,stream=ih/32u,l=ih-stream*32u,is=l/16u,bo=p.wo+((nt*p.bpr+br)*4u+nr)*210u;uint qlb=bo+hi*64u,qhb=bo+128u+hi*32u,scb=bo+192u+hi*8u;uint ql0=uint(w[qlb+l]),ql1=uint(w[qlb+l+32u]),qh=uint(w[qhb+l]),qu,si;if(stream==0u){qu=(ql0&15u)|(((qh>>0u)&3u)<<4u);si=is+0u;}else if(stream==1u){qu=(ql1&15u)|(((qh>>2u)&3u)<<4u);si=is+2u;}else if(stream==2u){qu=(ql0>>4u)|(((qh>>4u)&3u)<<4u);si=is+4u;}else{qu=(ql1>>4u)|(((qh>>6u)&3u)<<4u);si=is+6u;}return h(w,bo+208u)*float(i8(w,scb+si))*float(int(qu)-32);}\n"
    "kernel void matvec_q6k_nt8(device const float*x[[buffer(0)]],device const uchar*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],uint3 tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]],uint sg[[simdgroup_index_in_threadgroup]],uint sl[[thread_index_in_simdgroup]]){threadgroup float p0[4],p1[4],p2[4],p3[4],p4[4],p5[4],p6[4],p7[4];uint r0=tg.x*8u,b=tg.y;if(r0>=p.no||b>=p.rows)return;if(lid<4u){p0[lid]=0.0f;p1[lid]=0.0f;p2[lid]=0.0f;p3[lid]=0.0f;p4[lid]=0.0f;p5[lid]=0.0f;p6[lid]=0.0f;p7[lid]=0.0f;}threadgroup_barrier(mem_flags::mem_threadgroup);bool h1=r0+1u<p.no,h2=r0+2u<p.no,h3=r0+3u<p.no,h4=r0+4u<p.no,h5=r0+5u<p.no,h6=r0+6u<p.no,h7=r0+7u<p.no;float s0=0.0f,s1=0.0f,s2=0.0f,s3=0.0f,s4=0.0f,s5=0.0f,s6=0.0f,s7=0.0f;for(uint k=lid;k<p.ni;k+=64u){float xv=x[p.xo+b*p.xs+k];s0+=xv*dq6(w,p,r0,k);if(h1)s1+=xv*dq6(w,p,r0+1u,k);if(h2)s2+=xv*dq6(w,p,r0+2u,k);if(h3)s3+=xv*dq6(w,p,r0+3u,k);if(h4)s4+=xv*dq6(w,p,r0+4u,k);if(h5)s5+=xv*dq6(w,p,r0+5u,k);if(h6)s6+=xv*dq6(w,p,r0+6u,k);if(h7)s7+=xv*dq6(w,p,r0+7u,k);}float t0=simd_sum(s0),t1=simd_sum(s1),t2=simd_sum(s2),t3=simd_sum(s3),t4=simd_sum(s4),t5=simd_sum(s5),t6=simd_sum(s6),t7=simd_sum(s7);if(sl==0u&&sg<4u){p0[sg]=t0;p1[sg]=t1;p2[sg]=t2;p3[sg]=t3;p4[sg]=t4;p5[sg]=t5;p6[sg]=t6;p7[sg]=t7;}threadgroup_barrier(mem_flags::mem_threadgroup);if(lid==0u){uint o=p.yo+b*p.ys+r0;y[o]=p0[0]+p0[1]+p0[2]+p0[3];if(h1)y[o+1u]=p1[0]+p1[1]+p1[2]+p1[3];if(h2)y[o+2u]=p2[0]+p2[1]+p2[2]+p2[3];if(h3)y[o+3u]=p3[0]+p3[1]+p3[2]+p3[3];if(h4)y[o+4u]=p4[0]+p4[1]+p4[2]+p4[3];if(h5)y[o+5u]=p5[0]+p5[1]+p5[2]+p5[3];if(h6)y[o+6u]=p6[0]+p6[1]+p6[2]+p6[3];if(h7)y[o+7u]=p7[0]+p7[1]+p7[2]+p7[3];}}\n";

static const char metal_q4k_w4a8_source[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct QP{uint ni,xo;};\n"
    "struct MP{uint ni,no,bpr,wo,yo;};\n"
    "static inline float ab(float v){return v<0.0f?-v:v;}\n"
    "static inline int ci8(float v){float r=round(v);if(r>127.0f)return 127;if(r<-128.0f)return -128;return int(r);}\n"
    "static inline uint pi8(int v,uint sh){return (uint(v)&255u)<<sh;}\n"
    "kernel void q4k_quant_x(device const float*x[[buffer(0)]],device uint*xq[[buffer(1)]],device float*scale[[buffer(2)]],constant QP&p[[buffer(3)]],uint lid[[thread_index_in_threadgroup]]){threadgroup float pm[256];float m=0.0f;for(uint k=lid;k<p.ni;k+=256u)m=max(m,ab(x[p.xo+k]));pm[lid]=m;threadgroup_barrier(mem_flags::mem_threadgroup);for(uint st=128u;st>0u;st>>=1u){if(lid<st)pm[lid]=max(pm[lid],pm[lid+st]);threadgroup_barrier(mem_flags::mem_threadgroup);}float sx=pm[0]/127.0f;if(sx==0.0f)sx=1.0f;if(lid==0u)scale[0]=sx;threadgroup_barrier(mem_flags::mem_threadgroup);float inv=1.0f/sx;uint nw=(p.ni+3u)>>2u;for(uint word=lid;word<nw;word+=256u){uint base=word<<2u,packed=0u;if(base+0u<p.ni)packed|=pi8(ci8(x[p.xo+base+0u]*inv),0u);if(base+1u<p.ni)packed|=pi8(ci8(x[p.xo+base+1u]*inv),8u);if(base+2u<p.ni)packed|=pi8(ci8(x[p.xo+base+2u]*inv),16u);if(base+3u<p.ni)packed|=pi8(ci8(x[p.xo+base+3u]*inv),24u);xq[word]=packed;}}\n"
    "static inline uint u32(device const uchar*w,uint o){return uint(w[o])|(uint(w[o+1])<<8)|(uint(w[o+2])<<16)|(uint(w[o+3])<<24);}\n"
    "static inline float f32(device const uchar*w,uint o){return as_type<float>(u32(w,o));}\n"
    "static inline int lx(device const uint*xq,uint i){uint word=xq[i>>2u],u=(word>>((i&3u)*8u))&255u;return u<128u?int(u):int(u)-256;}\n"
    "static inline float qw(device const uchar*w,constant MP&p,uint r,uint k){uint nt=r>>2u,nr=r&3u,br=k>>8u,ib=k&255u,sub=ib>>5u,bo=p.wo+24u+((nt*p.bpr+br)*4u+nr)*280u;float d=f32(w,bo),dm=f32(w,bo+4u),sc=float(w[bo+8u+sub]),mn=float(w[bo+16u+sub]),q=float(w[bo+24u+ib]);return d*sc*q-dm*mn;}\n"
    "kernel void matvec_q4k_nt4_w4a8(device const uint*xq[[buffer(0)]],device const uchar*w[[buffer(1)]],device float*y[[buffer(2)]],device const float*scale[[buffer(3)]],constant MP&p[[buffer(4)]],uint3 tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){threadgroup float p0[64],p1[64],p2[64],p3[64];uint r0=tg.x*4u;if(r0>=p.no)return;bool h1=r0+1u<p.no,h2=r0+2u<p.no,h3=r0+3u<p.no;float s0=0.0f,s1=0.0f,s2=0.0f,s3=0.0f;for(uint k=lid;k<p.ni;k+=64u){float xv=float(lx(xq,k));s0+=xv*qw(w,p,r0,k);if(h1)s1+=xv*qw(w,p,r0+1u,k);if(h2)s2+=xv*qw(w,p,r0+2u,k);if(h3)s3+=xv*qw(w,p,r0+3u,k);}p0[lid]=s0;p1[lid]=s1;p2[lid]=s2;p3[lid]=s3;threadgroup_barrier(mem_flags::mem_threadgroup);for(uint st=32u;st>0u;st>>=1u){if(lid<st){p0[lid]+=p0[lid+st];p1[lid]+=p1[lid+st];p2[lid]+=p2[lid+st];p3[lid]+=p3[lid+st];}threadgroup_barrier(mem_flags::mem_threadgroup);}float sx=scale[0];if(lid==0u){uint o=p.yo+r0;y[o]=sx*p0[0];if(h1)y[o+1u]=sx*p1[0];if(h2)y[o+2u]=sx*p2[0];if(h3)y[o+3u]=sx*p3[0];}}\n";

static const char metal_q4k_gate_up_w4a8_source[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct GP{uint ni,no,rows,bpr,xo,gw,uw,gy,uy,xs,ys;};\n"
    "static inline uint u32(device const uchar*w,uint o){return uint(w[o])|(uint(w[o+1])<<8)|(uint(w[o+2])<<16)|(uint(w[o+3])<<24);}\n"
    "static inline float f32(device const uchar*w,uint o){return as_type<float>(u32(w,o));}\n"
    "static inline int lx(device const uint*xq,uint i){uint word=xq[i>>2u],u=(word>>((i&3u)*8u))&255u;return u<128u?int(u):int(u)-256;}\n"
    "static inline float qw(device const uchar*w,uint wo,uint bpr,uint r,uint k){uint nt=r>>2u,nr=r&3u,br=k>>8u,ib=k&255u,sub=ib>>5u,bo=wo+24u+((nt*bpr+br)*4u+nr)*280u;float d=f32(w,bo),dm=f32(w,bo+4u),sc=float(w[bo+8u+sub]),mn=float(w[bo+16u+sub]),q=float(w[bo+24u+ib]);return d*sc*q-dm*mn;}\n"
    "static inline float gelu(float x){return 0.5f*x*(1.0f+tanh(clamp(0.7978845608f*(x+0.044715f*x*x*x),-10.0f,10.0f)));}\n"
    "kernel void gate_up_q4k_nt4_w4a8(device const uint*xq[[buffer(0)]],device const uchar*wg[[buffer(1)]],device const uchar*wu[[buffer(2)]],device float*gy[[buffer(3)]],device const float*scale[[buffer(4)]],constant GP&p[[buffer(5)]],uint3 tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){threadgroup float g0[64],g1[64],g2[64],g3[64],u0[64],u1[64],u2[64],u3[64];uint r0=tg.x*4u,b=tg.y;if(r0>=p.no||b>=p.rows)return;bool h1=r0+1u<p.no,h2=r0+2u<p.no,h3=r0+3u<p.no;float gs0=0.0f,gs1=0.0f,gs2=0.0f,gs3=0.0f,us0=0.0f,us1=0.0f,us2=0.0f,us3=0.0f;for(uint k=lid;k<p.ni;k+=64u){float xv=float(lx(xq,k));gs0+=xv*qw(wg,p.gw,p.bpr,r0,k);us0+=xv*qw(wu,p.uw,p.bpr,r0,k);if(h1){gs1+=xv*qw(wg,p.gw,p.bpr,r0+1u,k);us1+=xv*qw(wu,p.uw,p.bpr,r0+1u,k);}if(h2){gs2+=xv*qw(wg,p.gw,p.bpr,r0+2u,k);us2+=xv*qw(wu,p.uw,p.bpr,r0+2u,k);}if(h3){gs3+=xv*qw(wg,p.gw,p.bpr,r0+3u,k);us3+=xv*qw(wu,p.uw,p.bpr,r0+3u,k);}}g0[lid]=gs0;g1[lid]=gs1;g2[lid]=gs2;g3[lid]=gs3;u0[lid]=us0;u1[lid]=us1;u2[lid]=us2;u3[lid]=us3;threadgroup_barrier(mem_flags::mem_threadgroup);for(uint st=32u;st>0u;st>>=1u){if(lid<st){g0[lid]+=g0[lid+st];g1[lid]+=g1[lid+st];g2[lid]+=g2[lid+st];g3[lid]+=g3[lid+st];u0[lid]+=u0[lid+st];u1[lid]+=u1[lid+st];u2[lid]+=u2[lid+st];u3[lid]+=u3[lid+st];}threadgroup_barrier(mem_flags::mem_threadgroup);}float sx=scale[0];if(lid==0u){uint o=p.gy+b*p.ys+r0;gy[o]=gelu(sx*g0[0])*(sx*u0[0]);if(h1)gy[o+1u]=gelu(sx*g1[0])*(sx*u1[0]);if(h2)gy[o+2u]=gelu(sx*g2[0])*(sx*u2[0]);if(h3)gy[o+3u]=gelu(sx*g3[0])*(sx*u3[0]);}}\n";

static const char metal_elem_source[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct Rows{uint rows,cols,x_offset,w_offset,y_offset,x_row_stride,y_row_stride;float eps;};\n"
    "struct Bin{uint rows,cols,a_offset,b_offset,y_offset,a_row_stride,b_row_stride,y_row_stride;};\n"
    "struct Sc{uint rows,cols,x_offset,y_offset,x_row_stride,y_row_stride;float scale;};\n"
    "struct Post{uint rows,cols,residual_offset,x_offset,w_offset,y_offset,residual_row_stride,x_row_stride,y_row_stride;float eps;};\n"
    "static inline float gelu_tanh_f(float x){return 0.5f*x*(1.0f+tanh(clamp(0.7978845608028654f*(x+0.044715f*x*x*x),-10.0f,10.0f)));}\n"
    "kernel void rmsnorm_rows(device const float*x[[buffer(0)]],device const float*w[[buffer(1)]],device float*y[[buffer(2)]],constant Rows&p[[buffer(3)]],uint row[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){\n"
    "    threadgroup float partial[256];if(row>=p.rows){return;}float ss=0.0f;for(uint c=lid;c<p.cols;c+=256u){float v=x[p.x_offset+row*p.x_row_stride+c];ss+=v*v;}partial[lid]=ss;threadgroup_barrier(mem_flags::mem_threadgroup);for(uint s=128u;s>0u;s>>=1u){if(lid<s){partial[lid]+=partial[lid+s];}threadgroup_barrier(mem_flags::mem_threadgroup);}float inv=rsqrt(partial[0]/float(p.cols)+p.eps);for(uint c=lid;c<p.cols;c+=256u){y[p.y_offset+row*p.y_row_stride+c]=x[p.x_offset+row*p.x_row_stride+c]*inv*w[p.w_offset+c];}\n"
    "}\n"
    "kernel void gelu_rows(device const float*x[[buffer(0)]],device float*y[[buffer(1)]],constant Sc&p[[buffer(2)]],uint gid[[thread_position_in_grid]]){\n"
    "    uint total=p.rows*p.cols;if(gid>=total){return;}uint r=gid/p.cols,c=gid-r*p.cols;y[p.y_offset+r*p.y_row_stride+c]=gelu_tanh_f(x[p.x_offset+r*p.x_row_stride+c]);\n"
    "}\n"
    "kernel void mul_rows(device const float*a[[buffer(0)]],device const float*b[[buffer(1)]],device float*y[[buffer(2)]],constant Bin&p[[buffer(3)]],uint gid[[thread_position_in_grid]]){\n"
    "    uint total=p.rows*p.cols;if(gid>=total){return;}uint r=gid/p.cols,c=gid-r*p.cols;y[p.y_offset+r*p.y_row_stride+c]=a[p.a_offset+r*p.a_row_stride+c]*b[p.b_offset+r*p.b_row_stride+c];\n"
    "}\n"
    "kernel void gelu_mul_rows(device const float*a[[buffer(0)]],device const float*b[[buffer(1)]],device float*y[[buffer(2)]],constant Bin&p[[buffer(3)]],uint gid[[thread_position_in_grid]]){\n"
    "    uint total=p.rows*p.cols;if(gid>=total){return;}uint r=gid/p.cols,c=gid-r*p.cols;float av=a[p.a_offset+r*p.a_row_stride+c];float bv=b[p.b_offset+r*p.b_row_stride+c];y[p.y_offset+r*p.y_row_stride+c]=gelu_tanh_f(av)*bv;\n"
    "}\n"
    "kernel void add_rows(device const float*a[[buffer(0)]],device const float*b[[buffer(1)]],device float*y[[buffer(2)]],constant Bin&p[[buffer(3)]],uint gid[[thread_position_in_grid]]){\n"
    "    uint total=p.rows*p.cols;if(gid>=total){return;}uint r=gid/p.cols,c=gid-r*p.cols;y[p.y_offset+r*p.y_row_stride+c]=a[p.a_offset+r*p.a_row_stride+c]+b[p.b_offset+r*p.b_row_stride+c];\n"
    "}\n"
    "kernel void scale_rows(device const float*x[[buffer(0)]],device float*y[[buffer(1)]],constant Sc&p[[buffer(2)]],uint gid[[thread_position_in_grid]]){\n"
    "    uint total=p.rows*p.cols;if(gid>=total){return;}uint r=gid/p.cols,c=gid-r*p.cols;y[p.y_offset+r*p.y_row_stride+c]=x[p.x_offset+r*p.x_row_stride+c]*p.scale;\n"
    "}\n"
    "kernel void rmsnorm_add_rows(device const float*res[[buffer(0)]],device const float*x[[buffer(1)]],device const float*w[[buffer(2)]],device float*y[[buffer(3)]],constant Post&p[[buffer(4)]],uint row[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){\n"
    "    threadgroup float partial[256];if(row>=p.rows){return;}float ss=0.0f;for(uint c=lid;c<p.cols;c+=256u){float v=x[p.x_offset+row*p.x_row_stride+c];ss+=v*v;}partial[lid]=ss;threadgroup_barrier(mem_flags::mem_threadgroup);for(uint s=128u;s>0u;s>>=1u){if(lid<s){partial[lid]+=partial[lid+s];}threadgroup_barrier(mem_flags::mem_threadgroup);}float inv=rsqrt(partial[0]/float(p.cols)+p.eps);for(uint c=lid;c<p.cols;c+=256u){float n=x[p.x_offset+row*p.x_row_stride+c]*inv*w[p.w_offset+c];y[p.y_offset+row*p.y_row_stride+c]=res[p.residual_offset+row*p.residual_row_stride+c]+n;}\n"
    "}\n";

static const char metal_elem_simd_source[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct Rows{uint rows,cols,x_offset,w_offset,y_offset,x_row_stride,y_row_stride;float eps;};\n"
    "struct Post{uint rows,cols,residual_offset,x_offset,w_offset,y_offset,residual_row_stride,x_row_stride,y_row_stride;float eps;};\n"
    "kernel void rmsnorm_rows_simd(device const float*x[[buffer(0)]],device const float*w[[buffer(1)]],device float*y[[buffer(2)]],constant Rows&p[[buffer(3)]],uint row[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]],uint sg[[simdgroup_index_in_threadgroup]],uint sl[[thread_index_in_simdgroup]]){\n"
    "threadgroup float partial[8];if(row>=p.rows)return;if(lid<8u)partial[lid]=0.0f;threadgroup_barrier(mem_flags::mem_threadgroup);float ss=0.0f;for(uint c=lid;c<p.cols;c+=256u){float v=x[p.x_offset+row*p.x_row_stride+c];ss+=v*v;}float sum=simd_sum(ss);if(sl==0u&&sg<8u)partial[sg]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);float total=partial[0]+partial[1]+partial[2]+partial[3]+partial[4]+partial[5]+partial[6]+partial[7];float inv=rsqrt(total/float(p.cols)+p.eps);for(uint c=lid;c<p.cols;c+=256u)y[p.y_offset+row*p.y_row_stride+c]=x[p.x_offset+row*p.x_row_stride+c]*inv*w[p.w_offset+c];}\n"
    "kernel void rmsnorm_add_rows_simd(device const float*res[[buffer(0)]],device const float*x[[buffer(1)]],device const float*w[[buffer(2)]],device float*y[[buffer(3)]],constant Post&p[[buffer(4)]],uint row[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]],uint sg[[simdgroup_index_in_threadgroup]],uint sl[[thread_index_in_simdgroup]]){\n"
    "threadgroup float partial[8];if(row>=p.rows)return;if(lid<8u)partial[lid]=0.0f;threadgroup_barrier(mem_flags::mem_threadgroup);float ss=0.0f;for(uint c=lid;c<p.cols;c+=256u){float v=x[p.x_offset+row*p.x_row_stride+c];ss+=v*v;}float sum=simd_sum(ss);if(sl==0u&&sg<8u)partial[sg]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);float total=partial[0]+partial[1]+partial[2]+partial[3]+partial[4]+partial[5]+partial[6]+partial[7];float inv=rsqrt(total/float(p.cols)+p.eps);for(uint c=lid;c<p.cols;c+=256u){float n=x[p.x_offset+row*p.x_row_stride+c]*inv*w[p.w_offset+c];y[p.y_offset+row*p.y_row_stride+c]=res[p.residual_offset+row*p.residual_row_stride+c]+n;}}\n";

static const char metal_embed_source[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct E{uint n,dtype,bpr,wbo,yo,token;float scale;};\n"
    "static inline float h(device const uchar*w,uint o){ushort b=ushort(w[o])|(ushort(w[o+1])<<8);return float(as_type<half>(b));}\n"
    "static inline float bf(device const uchar*w,uint o){return as_type<float>((uint(w[o])|(uint(w[o+1])<<8))<<16);}\n"
    "static inline float f32(device const uchar*w,uint o){return as_type<float>(uint(w[o])|(uint(w[o+1])<<8)|(uint(w[o+2])<<16)|(uint(w[o+3])<<24));}\n"
    "static inline int i8(device const uchar*w,uint o){uint u=uint(w[o]);return u<128u?int(u):int(u)-256;}\n"
    "static inline void sm(uint j,device const uchar*q,thread uint&s,thread uint&m){if(j<4){s=uint(q[j])&63u;m=uint(q[j+4])&63u;}else{uint a=uint(q[j+4]),b=uint(q[j-4]),c=uint(q[j]);s=(a&15u)|((b>>6u)<<4u);m=(a>>4u)|((c>>6u)<<4u);}}\n"
    "static inline float q4(device const uchar*w,constant E&p,uint row,uint k){uint br=k/256u,ib=k-br*256u,sub=ib/32u,idx=ib-sub*32u;uint bo=p.wbo+(row*p.bpr+br)*144u;float d=h(w,bo),dm=h(w,bo+2u);uint s,m;sm(sub,w+bo+4u,s,m);uint qb=uint(w[bo+16u+(sub/2u)*32u+idx]);uint q=(sub&1u)==0u?(qb&15u):(qb>>4u);return d*float(s)*float(q)-dm*float(m);}\n"
    "static inline float q5(device const uchar*w,constant E&p,uint row,uint k){uint br=k/256u,ib=k-br*256u,g=ib/64u,ih=ib-g*64u,ha=ih/32u,l=ih-ha*32u;uint bo=p.wbo+(row*p.bpr+br)*176u;float d=h(w,bo),dm=h(w,bo+2u);uint s,m;sm(g*2u+ha,w+bo+4u,s,m);uint qb=uint(w[bo+48u+g*32u+l]);uint q=(ha==0u?(qb&15u):(qb>>4u));uint bit=1u<<(g*2u+ha);q+=(uint(w[bo+16u+l])&bit)?16u:0u;return d*float(s)*float(q)-dm*float(m);}\n"
    "static inline float q6(device const uchar*w,constant E&p,uint row,uint k){uint br=k/256u,ib=k-br*256u,hi=ib/128u,ih=ib-hi*128u,st=ih/32u,l=ih-st*32u,is=l/16u;uint bo=p.wbo+(row*p.bpr+br)*210u;uint qlb=bo+hi*64u,qhb=bo+128u+hi*32u,scb=bo+192u+hi*8u;uint ql0=uint(w[qlb+l]),ql1=uint(w[qlb+l+32u]),qh=uint(w[qhb+l]);uint qu,si;if(st==0u){qu=(ql0&15u)|(((qh>>0u)&3u)<<4u);si=is;}else if(st==1u){qu=(ql1&15u)|(((qh>>2u)&3u)<<4u);si=is+2u;}else if(st==2u){qu=(ql0>>4u)|(((qh>>4u)&3u)<<4u);si=is+4u;}else{qu=(ql1>>4u)|(((qh>>6u)&3u)<<4u);si=is+6u;}return h(w,bo+208u)*float(i8(w,scb+si))*float(int(qu)-32);}\n"
    "kernel void embed_lookup_scaled(device const uchar*w[[buffer(0)]],device float*y[[buffer(1)]],constant E&p[[buffer(2)]],uint gid[[thread_position_in_grid]]){if(gid>=p.n)return;float v=p.dtype==0u?f32(w,p.wbo+(p.token*p.n+gid)*4u):(p.dtype==1u?h(w,p.wbo+(p.token*p.n+gid)*2u):(p.dtype==2u?bf(w,p.wbo+(p.token*p.n+gid)*2u):(p.dtype==8u?q4(w,p,p.token,gid):(p.dtype==9u?q5(w,p,p.token,gid):q6(w,p,p.token,gid)))));y[p.yo+gid]=v*p.scale;}\n";

static const char metal_f32_source[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "#define FOR_UNROLL(x) _Pragma(\"clang loop unroll(full)\") for(x)\n"
    "struct P{uint ni,no,rows,xo,wo,yo,xs,ys;};\n"
    "struct GP{uint ni,no,rows,xo,wo,po,yo,xs,ps,ys;};\n"
    "struct PN{uint ni,no,rows,xo,wo,ro,nwo,yo,xs,rs,ys;float eps;};\n"
    "static inline float gelu(float x){return 0.5f*x*(1.0f+tanh(clamp(0.7978845608028654f*(x+0.044715f*x*x*x),-10.0f,10.0f)));}\n"
    "kernel void matmul_f32(device const float*x[[buffer(0)]],device const float*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],uint3 tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){threadgroup float part[256];uint row=tg.x,b=tg.y;if(row>=p.no||b>=p.rows)return;float s=0.0f;for(uint k=lid;k<p.ni;k+=256u){s+=x[p.xo+b*p.xs+k]*w[p.wo+row*p.ni+k];}part[lid]=s;threadgroup_barrier(mem_flags::mem_threadgroup);for(uint st=128u;st>0u;st>>=1u){if(lid<st){part[lid]+=part[lid+st];}threadgroup_barrier(mem_flags::mem_threadgroup);}if(lid==0u){y[p.yo+b*p.ys+row]=part[0];}}\n"
    "kernel void matmul_f32_sg(device const float*x[[buffer(0)]],device const float*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],uint3 tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){threadgroup float as[64];threadgroup float bs[64];threadgroup float cs[64];uint b0=tg.y*8u,o0=tg.x*8u;simdgroup_float8x8 acc=make_filled_simdgroup_matrix<float,8>(0.0f);for(uint k0=0u;k0<p.ni;k0+=8u){for(uint i=lid;i<64u;i+=32u){uint r=i/8u,c=i%8u;as[i]=(b0+r<p.rows&&k0+c<p.ni)?x[p.xo+(b0+r)*p.xs+k0+c]:0.0f;}for(uint i=lid;i<64u;i+=32u){uint kk=i/8u,c=i%8u;bs[i]=(o0+c<p.no&&k0+kk<p.ni)?w[p.wo+(o0+c)*p.ni+k0+kk]:0.0f;}threadgroup_barrier(mem_flags::mem_threadgroup);simdgroup_float8x8 ma,mb;simdgroup_load(ma,as,8);simdgroup_load(mb,bs,8);simdgroup_multiply_accumulate(acc,ma,mb,acc);threadgroup_barrier(mem_flags::mem_threadgroup);}simdgroup_store(acc,cs,8);threadgroup_barrier(mem_flags::mem_threadgroup);for(uint i=lid;i<64u;i+=32u){uint r=i/8u,c=i%8u;if(b0+r<p.rows&&o0+c<p.no)y[p.yo+(b0+r)*p.ys+o0+c]=cs[i];}}\n"
    "kernel void ple_gate_f32(device const float*x[[buffer(0)]],device const float*w[[buffer(1)]],device const float*ple[[buffer(2)]],device float*y[[buffer(3)]],constant GP&p[[buffer(4)]],uint3 tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){threadgroup float part[256];uint row=tg.x,b=tg.y;if(row>=p.no||b>=p.rows)return;float s=0.0f;for(uint k=lid;k<p.ni;k+=256u){s+=x[p.xo+b*p.xs+k]*w[p.wo+row*p.ni+k];}part[lid]=s;threadgroup_barrier(mem_flags::mem_threadgroup);for(uint st=128u;st>0u;st>>=1u){if(lid<st){part[lid]+=part[lid+st];}threadgroup_barrier(mem_flags::mem_threadgroup);}if(lid==0u){uint po=p.po+b*p.ps+row;y[p.yo+b*p.ys+row]=gelu(part[0])*ple[po];}}\n"
    "kernel void ple_proj_norm_f32(device const float*x[[buffer(0)]],device const float*w[[buffer(1)]],device const float*res[[buffer(2)]],device const float*nw[[buffer(3)]],device float*y[[buffer(4)]],constant PN&p[[buffer(5)]],uint b[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){threadgroup float part[256];if(b>=p.rows)return;float ss=0.0f;for(uint c=lid;c<p.no;c+=256u){float s=0.0f;for(uint k=0u;k<p.ni;k++){s+=x[p.xo+b*p.xs+k]*w[p.wo+c*p.ni+k];}y[p.yo+b*p.ys+c]=s;ss+=s*s;}part[lid]=ss;threadgroup_barrier(mem_flags::mem_threadgroup);for(uint st=128u;st>0u;st>>=1u){if(lid<st){part[lid]+=part[lid+st];}threadgroup_barrier(mem_flags::mem_threadgroup);}float inv=rsqrt(part[0]/float(p.no)+p.eps);for(uint c=lid;c<p.no;c+=256u){float v=y[p.yo+b*p.ys+c]*inv*nw[p.nwo+c];y[p.yo+b*p.ys+c]=res[p.ro+b*p.rs+c]+v;}}\n";

/* llama-mm_sg-structured f32 GEMM (64-out x 32-row tile, 4 simdgroups,
 * f32 threadgroup staging so the per-element summation order matches
 * matmul_f32_sg = bit-identical results). Interior fast path only:
 * dispatched when rows%32==0, n_out%64==0, n_in%32==0 and x/w aligned.
 * Own array for the C99 4095-char literal limit; concatenated with
 * metal_f32_source at library init (FOR_UNROLL/struct P live there). */
static const char metal_f32_mm_source[] =
    "kernel void matmul_f32_mm_sg(device const float*x[[buffer(0)]],device const float*w[[buffer(1)]],device float*y[[buffer(2)]],constant P&p[[buffer(3)]],threadgroup char*shmem[[threadgroup(0)]],uint3 tg[[threadgroup_position_in_grid]],ushort ti[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]]){\n"
    "threadgroup float*sa=(threadgroup float*)shmem;threadgroup float*sb=(threadgroup float*)(shmem+8192);"
    "constexpr short NK=32;constexpr short NL0=2;constexpr short NL1=4;"
    "uint r0=tg.y*64u;uint r1=tg.x*32u;"
    "short lr0=short(ti/NL0);short lr1=short(ti/NL1);short il0=short(ti%NL0);"
    "device const float*wpf=w+p.wo+(r0+uint(lr0))*p.ni+uint(16*il0);"
    "short iy=short(8*(ti%NL1));"
    "device const float*xp=x+p.xo+(r1+uint(lr1))*p.xs+uint(iy);"
    "simdgroup_float8x8 ma[4];simdgroup_float8x8 mb[2];simdgroup_float8x8 mc[8];"
    "FOR_UNROLL(short i=0;i<8;i++){mc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);}"
    "short ay=short((ti/NL0)/8);short lx=short((ti/NL0)%8);"
    "short sx=short(ti%NL1);short sy=short((ti/NL1)/8);short lyb=short((ti/NL1)%8);short ibb=short(4*sx+sy);"
    "for(uint loop_k=0u;loop_k<p.ni;loop_k+=uint(NK)){"
    "float4 f0=*(device const float4*)(wpf);float4 f1=*(device const float4*)(wpf+4);float4 f2=*(device const float4*)(wpf+8);float4 f3=*(device const float4*)(wpf+12);"
    "threadgroup_barrier(mem_flags::mem_threadgroup);"
    "FOR_UNROLL(short i=0;i<16;i++){short ax=short(2*il0+i/8);short ly=short(i%8);float v=i<4?f0[i]:(i<8?f1[i-4]:(i<12?f2[i-8]:f3[i-12]));*(sa+64*(8*ax+ay)+8*ly+lx)=v;}"
    "*(threadgroup float2x4*)(sb+64*ibb+8*lyb)=*((device const float2x4*)xp);"
    "wpf+=NK;xp+=NK;"
    "threadgroup_barrier(mem_flags::mem_threadgroup);"
    "threadgroup const float*lsma=sa+4*64*(sg%2);threadgroup const float*lsmb=sb+2*64*(sg/2);"
    "FOR_UNROLL(short ik=0;ik<NK/8;ik++){"
    "simdgroup_barrier(mem_flags::mem_none);"
    "FOR_UNROLL(short i=0;i<4;i++){simdgroup_load(ma[i],lsma+64*i,8,0,false);}"
    "simdgroup_barrier(mem_flags::mem_none);"
    "FOR_UNROLL(short i=0;i<2;i++){simdgroup_load(mb[i],lsmb+64*i,8,0,false);}"
    "simdgroup_barrier(mem_flags::mem_none);"
    "FOR_UNROLL(short i=0;i<8;i++){simdgroup_multiply_accumulate(mc[i],mb[i/4],ma[i%4],mc[i]);}"
    "lsma+=8*64;lsmb+=4*64;}}"
    "device float*C=y+p.yo+(r1+16u*uint(sg>>1))*p.ys+r0+32u*uint(sg&1);"
    "FOR_UNROLL(short i=0;i<8;i++){simdgroup_store(mc[i],C+8*(i%4)+8*p.ys*(i/4),p.ys,0,false);}"
    "}\n";

static const char metal_q_norm_rope_source[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct NR{uint rows,heads,head_dim,x_offset,w_offset,cos_offset,sin_offset,x_row_stride,rope_row_stride,rope_row_offset;float eps;};\n"
    "kernel void q_norm_rope_rows(device float*x[[buffer(0)]],device const float*w[[buffer(1)]],device const float*c[[buffer(2)]],device const float*s[[buffer(3)]],constant NR&p[[buffer(4)]],uint2 tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){\n"
    " threadgroup float part[256];uint r=tg.x,h=tg.y;if(r>=p.rows||h>=p.heads){return;}uint base=p.x_offset+r*p.x_row_stride+h*p.head_dim;float ss=0.0f;for(uint i=lid;i<p.head_dim;i+=256u){float v=x[base+i];ss+=v*v;}part[lid]=ss;threadgroup_barrier(mem_flags::mem_threadgroup);for(uint st=128u;st>0u;st>>=1u){if(lid<st){part[lid]+=part[lid+st];}threadgroup_barrier(mem_flags::mem_threadgroup);}float inv=rsqrt(part[0]/float(p.head_dim)+p.eps);uint hd2=p.head_dim/2u,ro=p.rope_row_offset+r;for(uint i=lid;i<hd2;i+=256u){float x0=x[base+i]*inv*w[p.w_offset+i],x1=x[base+i+hd2]*inv*w[p.w_offset+i+hd2];float co=c[p.cos_offset+ro*p.rope_row_stride+i],si=s[p.sin_offset+ro*p.rope_row_stride+i];x[base+i]=x0*co-x1*si;x[base+i+hd2]=x0*si+x1*co;}\n"
    "}\n"
;

static const char metal_k_norm_rope_append_source[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct KRA{uint rows,heads,head_dim,x_offset,w_offset,cos_offset,sin_offset,cache_offset,x_row_stride,rope_row_stride,rope_row_offset,q_position;float eps;};\n"
    "kernel void k_norm_rope_append_rows(device float*x[[buffer(0)]],device const float*w[[buffer(1)]],device const float*c[[buffer(2)]],device const float*s[[buffer(3)]],device float*cache[[buffer(4)]],constant KRA&p[[buffer(5)]],uint2 tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){\n"
    " threadgroup float part[256];uint r=tg.x,h=tg.y;if(r>=p.rows||h>=p.heads){return;}uint base=p.x_offset+r*p.x_row_stride+h*p.head_dim;float ss=0.0f;for(uint i=lid;i<p.head_dim;i+=256u){float v=x[base+i];ss+=v*v;}part[lid]=ss;threadgroup_barrier(mem_flags::mem_threadgroup);for(uint st=128u;st>0u;st>>=1u){if(lid<st){part[lid]+=part[lid+st];}threadgroup_barrier(mem_flags::mem_threadgroup);}float inv=rsqrt(part[0]/float(p.head_dim)+p.eps);uint hd2=p.head_dim/2u,ro=p.rope_row_offset+r,cb=p.cache_offset+(p.q_position+r)*p.heads*p.head_dim+h*p.head_dim;for(uint i=lid;i<hd2;i+=256u){float x0=x[base+i]*inv*w[p.w_offset+i],x1=x[base+i+hd2]*inv*w[p.w_offset+i+hd2];float co=c[p.cos_offset+ro*p.rope_row_stride+i],si=s[p.sin_offset+ro*p.rope_row_stride+i];float y0=x0*co-x1*si,y1=x0*si+x1*co;x[base+i]=y0;x[base+i+hd2]=y1;cache[cb+i]=y0;cache[cb+i+hd2]=y1;}\n"
    "}\n"
    "kernel void k_norm_rope_append_rows_f16(device float*x[[buffer(0)]],device const float*w[[buffer(1)]],device const float*c[[buffer(2)]],device const float*s[[buffer(3)]],device half*cache[[buffer(4)]],constant KRA&p[[buffer(5)]],uint2 tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){\n"
    " threadgroup float part[256];uint r=tg.x,h=tg.y;if(r>=p.rows||h>=p.heads){return;}uint base=p.x_offset+r*p.x_row_stride+h*p.head_dim;float ss=0.0f;for(uint i=lid;i<p.head_dim;i+=256u){float v=x[base+i];ss+=v*v;}part[lid]=ss;threadgroup_barrier(mem_flags::mem_threadgroup);for(uint st=128u;st>0u;st>>=1u){if(lid<st){part[lid]+=part[lid+st];}threadgroup_barrier(mem_flags::mem_threadgroup);}float inv=rsqrt(part[0]/float(p.head_dim)+p.eps);uint hd2=p.head_dim/2u,ro=p.rope_row_offset+r,cb=p.cache_offset+(p.q_position+r)*p.heads*p.head_dim+h*p.head_dim;for(uint i=lid;i<hd2;i+=256u){float x0=x[base+i]*inv*w[p.w_offset+i],x1=x[base+i+hd2]*inv*w[p.w_offset+i+hd2];float co=c[p.cos_offset+ro*p.rope_row_stride+i],si=s[p.sin_offset+ro*p.rope_row_stride+i];float y0=x0*co-x1*si,y1=x0*si+x1*co;x[base+i]=y0;x[base+i+hd2]=y1;cache[cb+i]=half(y0);cache[cb+i+hd2]=half(y1);}\n"
    "}\n"
;

static const char metal_v_norm_append_source[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct VA{uint rows,heads,head_dim,x_offset,w_offset,cache_offset,x_row_stride,q_position;float eps;};\n"
    "kernel void v_norm_append_rows(device float*x[[buffer(0)]],device const float*w[[buffer(1)]],device float*cache[[buffer(2)]],constant VA&p[[buffer(3)]],uint2 tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){\n"
    " threadgroup float part[256];uint r=tg.x,h=tg.y;if(r>=p.rows||h>=p.heads){return;}uint base=p.x_offset+r*p.x_row_stride+h*p.head_dim;float ss=0.0f;for(uint i=lid;i<p.head_dim;i+=256u){float v=x[base+i];ss+=v*v;}part[lid]=ss;threadgroup_barrier(mem_flags::mem_threadgroup);for(uint st=128u;st>0u;st>>=1u){if(lid<st){part[lid]+=part[lid+st];}threadgroup_barrier(mem_flags::mem_threadgroup);}float inv=rsqrt(part[0]/float(p.head_dim)+p.eps);uint cb=p.cache_offset+(p.q_position+r)*p.heads*p.head_dim+h*p.head_dim;for(uint i=lid;i<p.head_dim;i+=256u){float y=x[base+i]*inv*w[p.w_offset+i];x[base+i]=y;cache[cb+i]=y;}\n"
    "}\n"
    "kernel void v_norm_append_rows_f16(device float*x[[buffer(0)]],device const float*w[[buffer(1)]],device half*cache[[buffer(2)]],constant VA&p[[buffer(3)]],uint2 tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){\n"
    " threadgroup float part[256];uint r=tg.x,h=tg.y;if(r>=p.rows||h>=p.heads){return;}uint base=p.x_offset+r*p.x_row_stride+h*p.head_dim;float ss=0.0f;for(uint i=lid;i<p.head_dim;i+=256u){float v=x[base+i];ss+=v*v;}part[lid]=ss;threadgroup_barrier(mem_flags::mem_threadgroup);for(uint st=128u;st>0u;st>>=1u){if(lid<st){part[lid]+=part[lid+st];}threadgroup_barrier(mem_flags::mem_threadgroup);}float inv=rsqrt(part[0]/float(p.head_dim)+p.eps);uint cb=p.cache_offset+(p.q_position+r)*p.heads*p.head_dim+h*p.head_dim;for(uint i=lid;i<p.head_dim;i+=256u){float y=x[base+i]*inv*w[p.w_offset+i];x[base+i]=y;cache[cb+i]=half(y);}\n"
    "}\n"
;

static const char metal_kv_norm_append_source[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct K{uint rows,heads,hd,xo,wo,co,so,cao,xs,rs,ro,qp;float eps;};\n"
    "struct V{uint rows,heads,hd,xo,wo,cao,xs,qp;float eps;};\n"
    "kernel void kv_norm_append_rows(device float*k[[buffer(0)]],device float*v[[buffer(1)]],device const float*kw[[buffer(2)]],device const float*vw[[buffer(3)]],device const float*c[[buffer(4)]],device const float*s[[buffer(5)]],device float*kc[[buffer(6)]],device float*vc[[buffer(7)]],constant K&kp[[buffer(8)]],constant V&vp[[buffer(9)]],uint2 tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){\n"
    " threadgroup float kr[256];threadgroup float vr[256];uint r=tg.x,h=tg.y;if(r>=kp.rows||h>=kp.heads){return;}uint kb=kp.xo+r*kp.xs+h*kp.hd,vb=vp.xo+r*vp.xs+h*vp.hd;float ks=0.0f,vs=0.0f;for(uint i=lid;i<kp.hd;i+=256u){float x=k[kb+i];ks+=x*x;}for(uint i=lid;i<vp.hd;i+=256u){float x=v[vb+i];vs+=x*x;}kr[lid]=ks;vr[lid]=vs;threadgroup_barrier(mem_flags::mem_threadgroup);for(uint st=128u;st>0u;st>>=1u){if(lid<st){kr[lid]+=kr[lid+st];vr[lid]+=vr[lid+st];}threadgroup_barrier(mem_flags::mem_threadgroup);}float kinv=rsqrt(kr[0]/float(kp.hd)+kp.eps),vinv=rsqrt(vr[0]/float(vp.hd)+vp.eps);uint hd2=kp.hd/2u,ro=kp.ro+r,kdst=kp.cao+(kp.qp+r)*kp.heads*kp.hd+h*kp.hd,vdst=vp.cao+(vp.qp+r)*vp.heads*vp.hd+h*vp.hd;for(uint i=lid;i<hd2;i+=256u){float x0=k[kb+i]*kinv*kw[kp.wo+i],x1=k[kb+i+hd2]*kinv*kw[kp.wo+i+hd2];float co=c[kp.co+ro*kp.rs+i],si=s[kp.so+ro*kp.rs+i];float y0=x0*co-x1*si,y1=x0*si+x1*co;k[kb+i]=y0;k[kb+i+hd2]=y1;kc[kdst+i]=y0;kc[kdst+i+hd2]=y1;}for(uint i=lid;i<vp.hd;i+=256u){float y=v[vb+i]*vinv*vw[vp.wo+i];v[vb+i]=y;vc[vdst+i]=y;}\n"
    "}\n";

static const char metal_kv_norm_append_f16_source[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct K{uint rows,heads,hd,xo,wo,co,so,cao,xs,rs,ro,qp;float eps;};\n"
    "struct V{uint rows,heads,hd,xo,wo,cao,xs,qp;float eps;};\n"
    "kernel void kv_norm_append_rows_f16(device float*k[[buffer(0)]],device float*v[[buffer(1)]],device const float*kw[[buffer(2)]],device const float*vw[[buffer(3)]],device const float*c[[buffer(4)]],device const float*s[[buffer(5)]],device half*kc[[buffer(6)]],device half*vc[[buffer(7)]],constant K&kp[[buffer(8)]],constant V&vp[[buffer(9)]],uint2 tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){\n"
    " threadgroup float kr[256];threadgroup float vr[256];uint r=tg.x,h=tg.y;if(r>=kp.rows||h>=kp.heads){return;}uint kb=kp.xo+r*kp.xs+h*kp.hd,vb=vp.xo+r*vp.xs+h*vp.hd;float ks=0.0f,vs=0.0f;for(uint i=lid;i<kp.hd;i+=256u){float x=k[kb+i];ks+=x*x;}for(uint i=lid;i<vp.hd;i+=256u){float x=v[vb+i];vs+=x*x;}kr[lid]=ks;vr[lid]=vs;threadgroup_barrier(mem_flags::mem_threadgroup);for(uint st=128u;st>0u;st>>=1u){if(lid<st){kr[lid]+=kr[lid+st];vr[lid]+=vr[lid+st];}threadgroup_barrier(mem_flags::mem_threadgroup);}float kinv=rsqrt(kr[0]/float(kp.hd)+kp.eps),vinv=rsqrt(vr[0]/float(vp.hd)+vp.eps);uint hd2=kp.hd/2u,ro=kp.ro+r,kdst=kp.cao+(kp.qp+r)*kp.heads*kp.hd+h*kp.hd,vdst=vp.cao+(vp.qp+r)*vp.heads*vp.hd+h*vp.hd;for(uint i=lid;i<hd2;i+=256u){float x0=k[kb+i]*kinv*kw[kp.wo+i],x1=k[kb+i+hd2]*kinv*kw[kp.wo+i+hd2];float co=c[kp.co+ro*kp.rs+i],si=s[kp.so+ro*kp.rs+i];float y0=x0*co-x1*si,y1=x0*si+x1*co;k[kb+i]=y0;k[kb+i+hd2]=y1;kc[kdst+i]=half(y0);kc[kdst+i+hd2]=half(y1);}for(uint i=lid;i<vp.hd;i+=256u){float y=v[vb+i]*vinv*vw[vp.wo+i];v[vb+i]=y;vc[vdst+i]=half(y);}\n"
    "}\n";

static const char metal_attn_source[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct Rope{uint rows,heads,head_dim,x_offset,cos_offset,sin_offset,x_row_stride,rope_row_stride,rope_row_offset;};\n"
    "struct Append{uint elems,kv_out,k_offset,v_offset,k_cache_offset,v_cache_offset,q_position;};\n"
    "struct Attn{uint rows,kv_len,q_heads,kv_heads,head_dim,q_position,sliding_window,q_offset,k_cache_offset,v_cache_offset,y_offset;};\n"
    "kernel void rope_rows(device float*x[[buffer(0)]],device const float*c[[buffer(1)]],device const float*s[[buffer(2)]],constant Rope&p[[buffer(3)]],uint gid[[thread_position_in_grid]]){\n"
    " uint hd2=p.head_dim/2u,total=p.rows*p.heads*hd2;if(gid>=total){return;}uint i=gid%hd2,h=(gid/hd2)%p.heads,r=gid/(hd2*p.heads);uint base=p.x_offset+r*p.x_row_stride+h*p.head_dim;uint ro=p.rope_row_offset+r;float x0=x[base+i],x1=x[base+i+hd2];float co=c[p.cos_offset+ro*p.rope_row_stride+i],si=s[p.sin_offset+ro*p.rope_row_stride+i];x[base+i]=x0*co-x1*si;x[base+i+hd2]=x0*si+x1*co;\n"
    "}\n"
    "kernel void kv_append_rows(device const float*k[[buffer(0)]],device const float*v[[buffer(1)]],device float*kc[[buffer(2)]],device float*vc[[buffer(3)]],constant Append&p[[buffer(4)]],uint gid[[thread_position_in_grid]]){\n"
    " if(gid>=p.elems){return;}uint r=gid/p.kv_out,c=gid-r*p.kv_out;uint dst=(p.q_position+r)*p.kv_out+c;kc[p.k_cache_offset+dst]=k[p.k_offset+gid];vc[p.v_cache_offset+dst]=v[p.v_offset+gid];\n"
    "}\n"
    "struct CP{uint so,dof,n;};\n"
    "kernel void copy_u32(device const uint*src[[buffer(0)]],device uint*dst[[buffer(1)]],constant CP&p[[buffer(2)]],uint gid[[thread_position_in_grid]]){if(gid>=p.n){return;}dst[p.dof+gid]=src[p.so+gid];}\n"
    "kernel void attention_rows(device const float*q[[buffer(0)]],device const float*kc[[buffer(1)]],device const float*vc[[buffer(2)]],device float*y[[buffer(3)]],constant Attn&p[[buffer(4)]],uint2 tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){\n"
    " threadgroup float qv[512];threadgroup float pt[256];threadgroup float red[256];threadgroup float mls[2];uint r=tg.x,h=tg.y,hd=p.head_dim;if(r>=p.rows||h>=p.q_heads||hd>512u){return;}uint kvh=h/(p.q_heads/p.kv_heads),qpos=p.q_position+r,qb=p.q_offset+r*p.q_heads*hd+h*hd;for(uint i=lid;i<hd;i+=256u)qv[i]=q[qb+i];uint slo=(p.sliding_window>0u&&qpos+1u>p.sliding_window)?qpos+1u-p.sliding_window:0u;uint shi=qpos<p.kv_len?qpos:p.kv_len-1u;float a0=0.0f,a1=0.0f;if(lid==0u){mls[0]=-3.402823466e+38f;mls[1]=0.0f;}threadgroup_barrier(mem_flags::mem_threadgroup);for(uint tb=slo;tb<=shi;tb+=256u){uint t=tb+lid;float sc=-3.402823466e+38f;if(t<=shi){float d=0.0f;for(uint i=0u;i<hd;i++)d+=qv[i]*kc[p.k_cache_offset+t*p.kv_heads*hd+kvh*hd+i];sc=d;}red[lid]=sc;threadgroup_barrier(mem_flags::mem_threadgroup);for(uint st=128u;st>0u;st>>=1u){if(lid<st)red[lid]=red[lid]>red[lid+st]?red[lid]:red[lid+st];threadgroup_barrier(mem_flags::mem_threadgroup);}float mold=mls[0],mnew=mold>red[0]?mold:red[0],corr=exp(mold-mnew);float e=t<=shi?exp(sc-mnew):0.0f;pt[lid]=e;red[lid]=e;threadgroup_barrier(mem_flags::mem_threadgroup);for(uint st=128u;st>0u;st>>=1u){if(lid<st)red[lid]+=red[lid+st];threadgroup_barrier(mem_flags::mem_threadgroup);}float tsum=red[0];uint tw=shi-tb+1u<256u?shi-tb+1u:256u;a0*=corr;a1*=corr;for(uint j=0u;j<tw;j++){float pv=pt[j];uint vb=p.v_cache_offset+(tb+j)*p.kv_heads*hd+kvh*hd;if(lid<hd)a0+=pv*vc[vb+lid];if(lid+256u<hd)a1+=pv*vc[vb+lid+256u];}threadgroup_barrier(mem_flags::mem_threadgroup);if(lid==0u){mls[0]=mnew;mls[1]=mls[1]*corr+tsum;}threadgroup_barrier(mem_flags::mem_threadgroup);}float inv=1.0f/mls[1];uint yb=p.y_offset+r*p.q_heads*hd+h*hd;if(lid<hd)y[yb+lid]=a0*inv;if(lid+256u<hd)y[yb+lid+256u]=a1*inv;\n"
    "}\n";

static const char metal_attn_f16_source[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct Append{uint elems,kv_out,k_offset,v_offset,k_cache_offset,v_cache_offset,q_position;};\n"
    "struct Attn{uint rows,kv_len,q_heads,kv_heads,head_dim,q_position,sliding_window,q_offset,k_cache_offset,v_cache_offset,y_offset;};\n"
    "kernel void kv_append_rows_f16(device const float*k[[buffer(0)]],device const float*v[[buffer(1)]],device half*kc[[buffer(2)]],device half*vc[[buffer(3)]],constant Append&p[[buffer(4)]],uint gid[[thread_position_in_grid]]){\n"
    " if(gid>=p.elems){return;}uint r=gid/p.kv_out,c=gid-r*p.kv_out;uint dst=(p.q_position+r)*p.kv_out+c;kc[p.k_cache_offset+dst]=half(k[p.k_offset+gid]);vc[p.v_cache_offset+dst]=half(v[p.v_offset+gid]);\n"
    "}\n"
    "kernel void attention_rows_f16(device const float*q[[buffer(0)]],device const half*kc[[buffer(1)]],device const half*vc[[buffer(2)]],device float*y[[buffer(3)]],constant Attn&p[[buffer(4)]],uint2 tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){\n"
    " threadgroup float qv[512];threadgroup float pt[256];threadgroup float red[256];threadgroup float mls[2];uint r=tg.x,h=tg.y,hd=p.head_dim;if(r>=p.rows||h>=p.q_heads||hd>512u){return;}uint kvh=h/(p.q_heads/p.kv_heads),qpos=p.q_position+r,qb=p.q_offset+r*p.q_heads*hd+h*hd;for(uint i=lid;i<hd;i+=256u)qv[i]=q[qb+i];uint slo=(p.sliding_window>0u&&qpos+1u>p.sliding_window)?qpos+1u-p.sliding_window:0u;uint shi=qpos<p.kv_len?qpos:p.kv_len-1u;float a0=0.0f,a1=0.0f;if(lid==0u){mls[0]=-3.402823466e+38f;mls[1]=0.0f;}threadgroup_barrier(mem_flags::mem_threadgroup);for(uint tb=slo;tb<=shi;tb+=256u){uint t=tb+lid;float sc=-3.402823466e+38f;if(t<=shi){float d=0.0f;for(uint i=0u;i<hd;i++)d+=qv[i]*float(kc[p.k_cache_offset+t*p.kv_heads*hd+kvh*hd+i]);sc=d;}red[lid]=sc;threadgroup_barrier(mem_flags::mem_threadgroup);for(uint st=128u;st>0u;st>>=1u){if(lid<st)red[lid]=red[lid]>red[lid+st]?red[lid]:red[lid+st];threadgroup_barrier(mem_flags::mem_threadgroup);}float mold=mls[0],mnew=mold>red[0]?mold:red[0],corr=exp(mold-mnew);float e=t<=shi?exp(sc-mnew):0.0f;pt[lid]=e;red[lid]=e;threadgroup_barrier(mem_flags::mem_threadgroup);for(uint st=128u;st>0u;st>>=1u){if(lid<st)red[lid]+=red[lid+st];threadgroup_barrier(mem_flags::mem_threadgroup);}float tsum=red[0];uint tw=shi-tb+1u<256u?shi-tb+1u:256u;a0*=corr;a1*=corr;for(uint j=0u;j<tw;j++){float pv=pt[j];uint vb=p.v_cache_offset+(tb+j)*p.kv_heads*hd+kvh*hd;if(lid<hd)a0+=pv*float(vc[vb+lid]);if(lid+256u<hd)a1+=pv*float(vc[vb+lid+256u]);}threadgroup_barrier(mem_flags::mem_threadgroup);if(lid==0u){mls[0]=mnew;mls[1]=mls[1]*corr+tsum;}threadgroup_barrier(mem_flags::mem_threadgroup);}float inv=1.0f/mls[1];uint yb=p.y_offset+r*p.q_heads*hd+h*hd;if(lid<hd)y[yb+lid]=a0*inv;if(lid+256u<hd)y[yb+lid+256u]=a1*inv;\n"
    "}\n";

static const char metal_attn_qnorm_source[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct NR{uint rows,heads,hd,xo,wo,co,so,xs,rs,ro;float eps;};\n"
    "struct A{uint rows,kv_len,qh,kvh,hd,qpos,sw,qo,kco,vco,yo;};\n"
    "kernel void attention_qnorm_rows(device const float*q[[buffer(0)]],device const float*qw[[buffer(1)]],device const float*c[[buffer(2)]],device const float*s[[buffer(3)]],device const float*kc[[buffer(4)]],device const float*vc[[buffer(5)]],device float*y[[buffer(6)]],constant NR&nr[[buffer(7)]],constant A&p[[buffer(8)]],uint2 tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){\n"
    " threadgroup float red[256];threadgroup float qv[512];threadgroup float pt[256];threadgroup float mls[2];uint r=tg.x,h=tg.y;if(r>=p.rows||h>=p.qh||p.hd>512u){return;}uint base=nr.xo+r*nr.xs+h*nr.hd;float ssq=0.0f;for(uint i=lid;i<nr.hd;i+=256u){float v=q[base+i];ssq+=v*v;}red[lid]=ssq;threadgroup_barrier(mem_flags::mem_threadgroup);for(uint st=128u;st>0u;st>>=1u){if(lid<st){red[lid]+=red[lid+st];}threadgroup_barrier(mem_flags::mem_threadgroup);}float invn=rsqrt(red[0]/float(nr.hd)+nr.eps);uint hd2=nr.hd/2u,rr=nr.ro+r;for(uint i=lid;i<hd2;i+=256u){float x0=q[base+i]*invn*qw[nr.wo+i],x1=q[base+i+hd2]*invn*qw[nr.wo+i+hd2];float co=c[nr.co+rr*nr.rs+i],si=s[nr.so+rr*nr.rs+i];qv[i]=x0*co-x1*si;qv[i+hd2]=x0*si+x1*co;}threadgroup_barrier(mem_flags::mem_threadgroup);uint kvh=h/(p.qh/p.kvh),qp=p.qpos+r,hd=p.hd;uint slo=(p.sw>0u&&qp+1u>p.sw)?qp+1u-p.sw:0u;uint shi=qp<p.kv_len?qp:p.kv_len-1u;float a0=0.0f,a1=0.0f;if(lid==0u){mls[0]=-3.402823466e+38f;mls[1]=0.0f;}threadgroup_barrier(mem_flags::mem_threadgroup);for(uint tb=slo;tb<=shi;tb+=256u){uint t=tb+lid;float sc=-3.402823466e+38f;if(t<=shi){float d=0.0f;uint kb=p.kco+t*p.kvh*hd+kvh*hd;uint i=0u;for(;i+4u<=hd;i+=4u){d+=dot(*(threadgroup float4*)(qv+i),*(device const float4*)(kc+kb+i));}for(;i<hd;i++)d+=qv[i]*kc[kb+i];sc=d;}red[lid]=sc;threadgroup_barrier(mem_flags::mem_threadgroup);for(uint st=128u;st>0u;st>>=1u){if(lid<st)red[lid]=red[lid]>red[lid+st]?red[lid]:red[lid+st];threadgroup_barrier(mem_flags::mem_threadgroup);}float mold=mls[0],mnew=mold>red[0]?mold:red[0],corr=exp(mold-mnew);float e=t<=shi?exp(sc-mnew):0.0f;pt[lid]=e;red[lid]=e;threadgroup_barrier(mem_flags::mem_threadgroup);for(uint st=128u;st>0u;st>>=1u){if(lid<st)red[lid]+=red[lid+st];threadgroup_barrier(mem_flags::mem_threadgroup);}float tsum=red[0];uint tw=shi-tb+1u<256u?shi-tb+1u:256u;a0*=corr;a1*=corr;for(uint j=0u;j<tw;j++){float pv=pt[j];uint vb=p.vco+(tb+j)*p.kvh*hd+kvh*hd;if(lid<hd)a0+=pv*vc[vb+lid];if(lid+256u<hd)a1+=pv*vc[vb+lid+256u];}threadgroup_barrier(mem_flags::mem_threadgroup);if(lid==0u){mls[0]=mnew;mls[1]=mls[1]*corr+tsum;}threadgroup_barrier(mem_flags::mem_threadgroup);}float invs=1.0f/mls[1];uint yb=p.yo+r*p.qh*hd+h*hd;if(lid<hd)y[yb+lid]=a0*invs;if(lid+256u<hd)y[yb+lid+256u]=a1*invs;\n"
    "}\n";

static const char metal_attn_qnorm_f16_source[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct NR{uint rows,heads,hd,xo,wo,co,so,xs,rs,ro;float eps;};\n"
    "struct A{uint rows,kv_len,qh,kvh,hd,qpos,sw,qo,kco,vco,yo;};\n"
    "kernel void attention_qnorm_rows_f16(device const float*q[[buffer(0)]],device const float*qw[[buffer(1)]],device const float*c[[buffer(2)]],device const float*s[[buffer(3)]],device const half*kc[[buffer(4)]],device const half*vc[[buffer(5)]],device float*y[[buffer(6)]],constant NR&nr[[buffer(7)]],constant A&p[[buffer(8)]],uint2 tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){\n"
    " threadgroup float red[256];threadgroup float qv[512];threadgroup float pt[256];threadgroup float mls[2];uint r=tg.x,h=tg.y;if(r>=p.rows||h>=p.qh||p.hd>512u){return;}uint base=nr.xo+r*nr.xs+h*nr.hd;float ssq=0.0f;for(uint i=lid;i<nr.hd;i+=256u){float v=q[base+i];ssq+=v*v;}red[lid]=ssq;threadgroup_barrier(mem_flags::mem_threadgroup);for(uint st=128u;st>0u;st>>=1u){if(lid<st){red[lid]+=red[lid+st];}threadgroup_barrier(mem_flags::mem_threadgroup);}float invn=rsqrt(red[0]/float(nr.hd)+nr.eps);uint hd2=nr.hd/2u,rr=nr.ro+r;for(uint i=lid;i<hd2;i+=256u){float x0=q[base+i]*invn*qw[nr.wo+i],x1=q[base+i+hd2]*invn*qw[nr.wo+i+hd2];float co=c[nr.co+rr*nr.rs+i],si=s[nr.so+rr*nr.rs+i];qv[i]=x0*co-x1*si;qv[i+hd2]=x0*si+x1*co;}threadgroup_barrier(mem_flags::mem_threadgroup);uint kvh=h/(p.qh/p.kvh),qp=p.qpos+r,hd=p.hd;uint slo=(p.sw>0u&&qp+1u>p.sw)?qp+1u-p.sw:0u;uint shi=qp<p.kv_len?qp:p.kv_len-1u;float a0=0.0f,a1=0.0f;if(lid==0u){mls[0]=-3.402823466e+38f;mls[1]=0.0f;}threadgroup_barrier(mem_flags::mem_threadgroup);for(uint tb=slo;tb<=shi;tb+=256u){uint t=tb+lid;float sc=-3.402823466e+38f;if(t<=shi){float d=0.0f;uint kb=p.kco+t*p.kvh*hd+kvh*hd;uint i=0u;for(;i+4u<=hd;i+=4u){half4 k4=*(device const half4*)(kc+kb+i);d+=dot(*(threadgroup float4*)(qv+i),float4(k4));}for(;i<hd;i++)d+=qv[i]*float(kc[kb+i]);sc=d;}red[lid]=sc;threadgroup_barrier(mem_flags::mem_threadgroup);for(uint st=128u;st>0u;st>>=1u){if(lid<st)red[lid]=red[lid]>red[lid+st]?red[lid]:red[lid+st];threadgroup_barrier(mem_flags::mem_threadgroup);}float mold=mls[0],mnew=mold>red[0]?mold:red[0],corr=exp(mold-mnew);float e=t<=shi?exp(sc-mnew):0.0f;pt[lid]=e;red[lid]=e;threadgroup_barrier(mem_flags::mem_threadgroup);for(uint st=128u;st>0u;st>>=1u){if(lid<st)red[lid]+=red[lid+st];threadgroup_barrier(mem_flags::mem_threadgroup);}float tsum=red[0];uint tw=shi-tb+1u<256u?shi-tb+1u:256u;a0*=corr;a1*=corr;for(uint j=0u;j<tw;j++){float pv=pt[j];uint vb=p.vco+(tb+j)*p.kvh*hd+kvh*hd;if(lid<hd)a0+=pv*float(vc[vb+lid]);if(lid+256u<hd)a1+=pv*float(vc[vb+lid+256u]);}threadgroup_barrier(mem_flags::mem_threadgroup);if(lid==0u){mls[0]=mnew;mls[1]=mls[1]*corr+tsum;}threadgroup_barrier(mem_flags::mem_threadgroup);}float invs=1.0f/mls[1];uint yb=p.yo+r*p.qh*hd+h*hd;if(lid<hd)y[yb+lid]=a0*invs;if(lid+256u<hd)y[yb+lid+256u]=a1*invs;\n"
    "}\n";

/* Decode-specialized fused qnorm+rope+attention (rows==1, f16 KV): tokens are
 * split across the 8 simdgroups with per-simdgroup online softmax kept in
 * registers (simd_sum reductions, no tree-reduce barriers, no serial PV
 * loop); one threadgroup-memory merge at the end. Replaces the two-pass
 * 256-thread kernel that left the GPU idle at decode. */
static const char metal_attn_qnorm_dec_f16_source[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct NR{uint rows,heads,hd,xo,wo,co,so,xs,rs,ro;float eps;};\n"
    "struct A{uint rows,kv_len,qh,kvh,hd,qpos,sw,qo,kco,vco,yo;};\n"
    "struct SP{uint ns;};\n"
    "kernel void attention_qnorm_dec_f16(device const float*q[[buffer(0)]],device const float*qw[[buffer(1)]],device const float*c[[buffer(2)]],device const float*s[[buffer(3)]],device const half*kc[[buffer(4)]],device const half*vc[[buffer(5)]],device float*pb[[buffer(6)]],constant NR&nr[[buffer(7)]],constant A&p[[buffer(8)]],constant SP&sp[[buffer(9)]],uint2 tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]],uint sg[[simdgroup_index_in_threadgroup]],uint ln[[thread_index_in_simdgroup]]){"
    "threadgroup float qv[256];threadgroup float red[8];threadgroup float tm[8];threadgroup float tl[8];threadgroup float ta[2048];"
    "uint h=tg.y,ss=tg.x,hd=p.hd;if(h>=p.qh||hd>256u)return;"
    "uint base=nr.xo+h*nr.hd;"
    "float ssq=0.0f;for(uint i=lid;i<hd;i+=256u){float v=q[base+i];ssq+=v*v;}"
    "ssq=simd_sum(ssq);if(ln==0u)red[sg]=ssq;threadgroup_barrier(mem_flags::mem_threadgroup);"
    "float tot=red[0]+red[1]+red[2]+red[3]+red[4]+red[5]+red[6]+red[7];"
    "float invn=rsqrt(tot/float(hd)+nr.eps);"
    "uint hd2=hd/2u,rr=nr.ro;"
    "for(uint i=lid;i<hd2;i+=256u){float x0=q[base+i]*invn*qw[nr.wo+i],x1=q[base+i+hd2]*invn*qw[nr.wo+i+hd2];float co=c[nr.co+rr*nr.rs+i],si=s[nr.so+rr*nr.rs+i];qv[i]=x0*co-x1*si;qv[i+hd2]=x0*si+x1*co;}"
    "threadgroup_barrier(mem_flags::mem_threadgroup);"
    "uint kvh=h/(p.qh/p.kvh),qp=p.qpos;"
    "uint slo=(p.sw>0u&&qp+1u>p.sw)?qp+1u-p.sw:0u;uint shi=qp<p.kv_len?qp:p.kv_len-1u;"
    /* fixed 8-chunk unroll (hd<=256) keeps a[] in registers — a runtime
     * bound would spill the accumulators to thread-private device memory */
    "float m=-3.402823466e+38f,l=0.0f,a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;"
    "bool g1=ln+32u<hd,g2=ln+64u<hd,g3=ln+96u<hd,g4=ln+128u<hd,g5=ln+160u<hd,g6=ln+192u<hd,g7=ln+224u<hd;"
    "for(uint t=slo+ss*8u+sg;t<=shi;t+=sp.ns*8u){"
    "uint kb=p.kco+t*p.kvh*hd+kvh*hd;"
    "float d=qv[ln]*float(kc[kb+ln]);"
    "if(g1)d+=qv[ln+32u]*float(kc[kb+ln+32u]);"
    "if(g2)d+=qv[ln+64u]*float(kc[kb+ln+64u]);"
    "if(g3)d+=qv[ln+96u]*float(kc[kb+ln+96u]);"
    "if(g4)d+=qv[ln+128u]*float(kc[kb+ln+128u]);"
    "if(g5)d+=qv[ln+160u]*float(kc[kb+ln+160u]);"
    "if(g6)d+=qv[ln+192u]*float(kc[kb+ln+192u]);"
    "if(g7)d+=qv[ln+224u]*float(kc[kb+ln+224u]);"
    "d=simd_sum(d);"
    "float mn=max(m,d),corr=exp(m-mn),e=exp(d-mn);"
    "l=l*corr+e;"
    "uint vb=p.vco+t*p.kvh*hd+kvh*hd;"
    "a0=a0*corr+e*float(vc[vb+ln]);"
    "if(g1)a1=a1*corr+e*float(vc[vb+ln+32u]);"
    "if(g2)a2=a2*corr+e*float(vc[vb+ln+64u]);"
    "if(g3)a3=a3*corr+e*float(vc[vb+ln+96u]);"
    "if(g4)a4=a4*corr+e*float(vc[vb+ln+128u]);"
    "if(g5)a5=a5*corr+e*float(vc[vb+ln+160u]);"
    "if(g6)a6=a6*corr+e*float(vc[vb+ln+192u]);"
    "if(g7)a7=a7*corr+e*float(vc[vb+ln+224u]);"
    "m=mn;}"
    "if(ln==0u){tm[sg]=m;tl[sg]=l;}"
    "uint tb=sg*hd+ln;"
    "ta[tb]=a0;"
    "if(g1)ta[tb+32u]=a1;if(g2)ta[tb+64u]=a2;if(g3)ta[tb+96u]=a3;"
    "if(g4)ta[tb+128u]=a4;if(g5)ta[tb+160u]=a5;if(g6)ta[tb+192u]=a6;if(g7)ta[tb+224u]=a7;"
    "threadgroup_barrier(mem_flags::mem_threadgroup);"
    "float M=tm[0];for(uint j=1u;j<8u;j++)M=max(M,tm[j]);"
    "float L=0.0f;for(uint j=0u;j<8u;j++)L+=tl[j]*exp(tm[j]-M);"
    "uint pbb=(h*16u+ss)*(hd+2u);"
    "if(lid==0u){pb[pbb]=M;pb[pbb+1u]=L;}"
    "for(uint i=lid;i<hd;i+=256u){float acc=0.0f;for(uint j=0u;j<8u;j++)acc+=ta[j*hd+i]*exp(tm[j]-M);pb[pbb+2u+i]=acc;}}\n";

/* Prefill simdgroup flash attention (rows%8==0, hd%32==0, hd<=256, f16 KV,
 * kv_len>=32): one threadgroup per (8-query-row tile, head), 4 simdgroups
 * split head_dim. QK^T and P.V run as simdgroup_multiply_accumulate with K
 * loaded transposed straight from the cache; online softmax is scalar on
 * 8x32 score tiles; accumulator rescale/normalize via diagonal-matrix
 * multiply. The tail kv tile re-covers [thi-31, thi] with already-processed
 * positions masked out, so no out-of-bounds cache reads. Source exceeds the
 * ISO 4095-char literal limit -> two parts, concatenated at init. */
static const char metal_attn_flash_sg_f16_source_a[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct NR{uint rows,heads,hd,xo,wo,co,so,xs,rs,ro;float eps;};\n"
    "struct A{uint rows,kv_len,qh,kvh,hd,qpos,sw,qo,kco,vco,yo;};\n"
    "kernel void attention_qnorm_flash_sg_f16(device const float*q[[buffer(0)]],device const float*qw[[buffer(1)]],device const float*c[[buffer(2)]],device const float*s[[buffer(3)]],device const half*kc[[buffer(4)]],device const half*vc[[buffer(5)]],device float*y[[buffer(6)]],constant NR&nr[[buffer(7)]],constant A&p[[buffer(8)]],uint2 tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]],uint sg[[simdgroup_index_in_threadgroup]],uint ln[[thread_index_in_simdgroup]]){"
    "threadgroup half Qt[2048];threadgroup float Sred[1024];threadgroup float Sf[256];threadgroup half Pt[256];threadgroup float Mt[8];threadgroup float Lt[8];threadgroup float Dg[64];threadgroup float Mn[8];"
    "uint hd=p.hd,h=tg.y,q0=tg.x*8u;"
    "uint kvh=h/(p.qh/p.kvh),kstr=p.kvh*hd;"
    "for(uint rr=0u;rr<2u;rr++){"
    "uint r=sg*2u+rr,row=q0+r,base=nr.xo+row*nr.xs+h*nr.hd;"
    "float ss=0.0f;for(uint i=ln;i<hd;i+=32u){float v=q[base+i];ss+=v*v;}"
    "ss=simd_sum(ss);float invn=rsqrt(ss/float(hd)+nr.eps);"
    "uint hd2=hd/2u,ro=nr.ro+row;"
    "for(uint i=ln;i<hd2;i+=32u){float x0=q[base+i]*invn*qw[nr.wo+i],x1=q[base+i+hd2]*invn*qw[nr.wo+i+hd2];float co=c[nr.co+ro*nr.rs+i],si=s[nr.so+ro*nr.rs+i];Qt[r*hd+i]=half(x0*co-x1*si);Qt[r*hd+i+hd2]=half(x0*si+x1*co);}}"
    "if(lid<8u){Mt[lid]=-3.402823466e+38f;Lt[lid]=0.0f;}"
    "if(lid<64u)Dg[lid]=0.0f;"
    "threadgroup_barrier(mem_flags::mem_threadgroup);"
    "uint slc=hd/4u,c0=sg*slc,nO=slc/8u;"
    "simdgroup_float8x8 O[8];for(uint i=0u;i<8u;i++)O[i]=make_filled_simdgroup_matrix<float,8>(0.0f);"
    "uint qmax=p.qpos+q0+7u;"
    "uint thi=qmax<p.kv_len?qmax:p.kv_len-1u;"
    "uint qp0=p.qpos+q0;"
    "uint tlo=(p.sw>0u&&qp0+1u>p.sw)?qp0+1u-p.sw:0u;"
    "uint span=thi+1u-tlo,nfull=span/32u,rem=span%32u;"
    "uint nit=nfull+(rem>0u?1u:0u);";

static const char metal_attn_flash_sg_f16_source_b[] =
    "for(uint it=0u;it<nit;it++){"
    "uint t0=(it<nfull)?tlo+it*32u:(thi+1u>=32u?thi+1u-32u:0u);"
    "uint skipb=(it<nfull)?0u:tlo+nfull*32u;"
    "simdgroup_float8x8 S4[4];for(uint i=0u;i<4u;i++)S4[i]=make_filled_simdgroup_matrix<float,8>(0.0f);"
    "simdgroup_half8x8 mq,mk;"
    "for(uint kk=0u;kk<4u;kk++){for(uint cc=0u;cc<nO;cc++){"
    "simdgroup_load(mq,Qt+c0+cc*8u,hd);"
    "simdgroup_load(mk,kc+p.kco+(t0+kk*8u)*kstr+kvh*hd+c0+cc*8u,kstr,0,true);"
    "simdgroup_multiply_accumulate(S4[kk],mq,mk,S4[kk]);}}"
    "for(uint kk=0u;kk<4u;kk++)simdgroup_store(S4[kk],Sred+sg*256u+kk*64u,8);"
    "threadgroup_barrier(mem_flags::mem_threadgroup);"
    "for(uint e=lid;e<256u;e+=128u){"
    "uint r=e/32u,j=e%32u,t=t0+j;"
    "float v=Sred[(j/8u)*64u+r*8u+(j%8u)]+Sred[256u+(j/8u)*64u+r*8u+(j%8u)]+Sred[512u+(j/8u)*64u+r*8u+(j%8u)]+Sred[768u+(j/8u)*64u+r*8u+(j%8u)];"
    "uint qp=p.qpos+q0+r;"
    "uint slr=(p.sw>0u&&qp+1u>p.sw)?qp+1u-p.sw:0u;"
    "bool ok=t>=slr&&t<=qp&&t>=skipb;"
    "Sf[e]=ok?v:-3.402823466e+38f;}"
    "threadgroup_barrier(mem_flags::mem_threadgroup);"
    /* softmax split so the expensive parts run on all lanes: row max via
     * simd_max (order-independent = exact) and the 256 exps in parallel
     * (elementwise = exact); only the l-sum stays serial ascending per row
     * to keep the summation order — results stay bit-identical. */
    "for(uint rr=0u;rr<2u;rr++){uint r=sg*2u+rr;float tm=simd_max(Sf[r*32u+ln]);"
    "if(ln==0u)Mn[r]=max(Mt[r],tm);}"
    "threadgroup_barrier(mem_flags::mem_threadgroup);"
    "for(uint e=lid;e<256u;e+=128u){uint r=e/32u;float e2=exp(Sf[e]-Mn[r]);Sf[e]=e2;Pt[e]=half(e2);}"
    "threadgroup_barrier(mem_flags::mem_threadgroup);"
    "if(lid<8u){uint r=lid;float mn=Mn[r],corr=exp(Mt[r]-mn),ls=0.0f;"
    "for(uint j=0u;j<32u;j++){ls+=Sf[r*32u+j];}"
    "Mt[r]=mn;Lt[r]=Lt[r]*corr+ls;Dg[r*9u]=corr;}"
    "threadgroup_barrier(mem_flags::mem_threadgroup);"
    "simdgroup_float8x8 mco;simdgroup_load(mco,Dg,8);"
    "for(uint cc=0u;cc<nO;cc++)simdgroup_multiply(O[cc],mco,O[cc]);"
    "simdgroup_half8x8 mp,mv;"
    "for(uint kk=0u;kk<4u;kk++){"
    "simdgroup_load(mp,Pt+kk*8u,32);"
    "for(uint cc=0u;cc<nO;cc++){"
    "simdgroup_load(mv,vc+p.vco+(t0+kk*8u)*kstr+kvh*hd+c0+cc*8u,kstr);"
    "simdgroup_multiply_accumulate(O[cc],mp,mv,O[cc]);}}"
    "threadgroup_barrier(mem_flags::mem_threadgroup);}"
    "if(lid<8u)Dg[lid*9u]=1.0f/Lt[lid];"
    "threadgroup_barrier(mem_flags::mem_threadgroup);"
    "simdgroup_float8x8 mfin;simdgroup_load(mfin,Dg,8);"
    "for(uint cc=0u;cc<nO;cc++){simdgroup_float8x8 t;simdgroup_multiply(t,mfin,O[cc]);"
    "simdgroup_store(t,y+p.yo+q0*p.qh*hd+h*hd+c0+cc*8u,p.qh*hd);}"
    "}\n";

/* Pass 2: merge the per-split partials (m, l, unnormalized acc) into y. */
static const char metal_attn_dec_combine_source[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct CB{uint qh,hd,ns,yo;};\n"
    "kernel void attention_dec_combine(device const float*pb[[buffer(0)]],device float*y[[buffer(1)]],constant CB&p[[buffer(2)]],uint2 tg[[threadgroup_position_in_grid]],uint lid[[thread_index_in_threadgroup]]){"
    "uint h=tg.y,hd=p.hd;if(h>=p.qh)return;"
    "uint b0=h*16u*(hd+2u);"
    "float M=pb[b0];for(uint s=1u;s<p.ns;s++)M=max(M,pb[b0+s*(hd+2u)]);"
    "float L=0.0f;for(uint s=0u;s<p.ns;s++)L+=pb[b0+s*(hd+2u)+1u]*exp(pb[b0+s*(hd+2u)]-M);"
    "float inv=1.0f/L;"
    "for(uint i=lid;i<hd;i+=256u){float acc=0.0f;for(uint s=0u;s<p.ns;s++)acc+=pb[b0+s*(hd+2u)+2u+i]*exp(pb[b0+s*(hd+2u)]-M);y[p.yo+h*hd+i]=acc*inv;}}\n";

static void *metal_dlsym(void *handle, const char *name) {
    return handle != nullptr ? dlsym(handle, name) : nullptr;
}

static void *metal_objc_get_class(struct metal_state *st, const char *name) {
    union {
        void *raw;
        void *(*fn)(const char *name);
    } get_class = {.raw = st->objc_getClass};
    return get_class.fn(name);
}

static void *metal_create_default_device(struct metal_state *st) {
    union {
        void *raw;
        void *(*fn)(void);
    } create_device = {.raw = st->MTLCreateSystemDefaultDevice};
    return create_device.fn();
}

static void *metal_sel_register_name(struct metal_state *st,
                                     const char *selector) {
    union {
        void *raw;
        void *(*fn)(const char *name);
    } sel_register = {.raw = st->sel_registerName};
    return sel_register.fn(selector);
}

static void *metal_msg_send_id0(struct metal_state *st,
                                void *receiver,
                                const char *selector) {
    void *sel = metal_sel_register_name(st, selector);
    union {
        void *raw;
        void *(*fn)(void *, void *);
    } send = {.raw = st->objc_msgSend};
    return send.fn(receiver, sel);
}

static void *metal_msg_send_id_size_uint(struct metal_state *st,
                                         void *receiver,
                                         const char *selector,
                                         size_t a,
                                         unsigned long b) {
    void *sel = metal_sel_register_name(st, selector);
    union {
        void *raw;
        void *(*fn)(void *, void *, size_t, unsigned long);
    } send = {.raw = st->objc_msgSend};
    return send.fn(receiver, sel, a, b);
}

/* newBufferWithBytes:length:options: — copies host bytes into a new MTLBuffer. */
static void *metal_msg_send_id_ptr_size_uint(struct metal_state *st,
                                             void *receiver,
                                             const char *selector,
                                             const void *ptr,
                                             size_t len,
                                             unsigned long opts) {
    void *sel = metal_sel_register_name(st, selector);
    union {
        void *raw;
        void *(*fn)(void *, void *, const void *, size_t, unsigned long);
    } send = {.raw = st->objc_msgSend};
    return send.fn(receiver, sel, ptr, len, opts);
}

static void *metal_msg_send_id_cstr(struct metal_state *st,
                                    void *receiver,
                                    const char *selector,
                                    const char *arg) {
    void *sel = metal_sel_register_name(st, selector);
    union {
        void *raw;
        void *(*fn)(void *, void *, const char *);
    } send = {.raw = st->objc_msgSend};
    return send.fn(receiver, sel, arg);
}

static void *metal_msg_send_id_id(struct metal_state *st,
                                  void *receiver,
                                  const char *selector,
                                  void *arg) {
    void *sel = metal_sel_register_name(st, selector);
    union {
        void *raw;
        void *(*fn)(void *, void *, void *);
    } send = {.raw = st->objc_msgSend};
    return send.fn(receiver, sel, arg);
}

static void *metal_msg_send_id_id_id_err(struct metal_state *st,
                                         void *receiver,
                                         const char *selector,
                                         void *a,
                                         void *b,
                                         void **err) {
    void *sel = metal_sel_register_name(st, selector);
    union {
        void *raw;
        void *(*fn)(void *, void *, void *, void *, void **);
    } send = {.raw = st->objc_msgSend};
    return send.fn(receiver, sel, a, b, err);
}

static void *metal_msg_send_id_id_err(struct metal_state *st,
                                      void *receiver,
                                      const char *selector,
                                      void *a,
                                      void **err) {
    void *sel = metal_sel_register_name(st, selector);
    union {
        void *raw;
        void *(*fn)(void *, void *, void *, void **);
    } send = {.raw = st->objc_msgSend};
    return send.fn(receiver, sel, a, err);
}

static void metal_msg_send_void_ulong(struct metal_state *st,
                                      void *receiver,
                                      const char *selector,
                                      unsigned long a) {
    void *sel = metal_sel_register_name(st, selector);
    union {
        void *raw;
        void (*fn)(void *, void *, unsigned long);
    } send = {.raw = st->objc_msgSend};
    send.fn(receiver, sel, a);
}

static bool metal_msg_send_bool_id_err(struct metal_state *st,
                                       void *receiver,
                                       const char *selector,
                                       void *a,
                                       void **err) {
    void *sel = metal_sel_register_name(st, selector);
    union {
        void *raw;
        unsigned char (*fn)(void *, void *, void *, void **);
    } send = {.raw = st->objc_msgSend};
    return send.fn(receiver, sel, a, err) != 0;
}

static const char *metal_msg_send_cstr0(struct metal_state *st,
                                        void *receiver,
                                        const char *selector) {
    void *sel = metal_sel_register_name(st, selector);
    union {
        void *raw;
        const char *(*fn)(void *, void *);
    } send = {.raw = st->objc_msgSend};
    return send.fn(receiver, sel);
}

static const char *metal_nserror_message(struct metal_state *st, void *err) {
    if (st == nullptr || err == nullptr) {
        return nullptr;
    }
    void *desc = metal_msg_send_id0(st, err, "localizedDescription");
    if (desc == nullptr) {
        return nullptr;
    }
    return metal_msg_send_cstr0(st, desc, "UTF8String");
}

static void *metal_msg_send_ptr0(struct metal_state *st,
                                 void *receiver,
                                 const char *selector) {
    void *sel = metal_sel_register_name(st, selector);
    union {
        void *raw;
        void *(*fn)(void *, void *);
    } send = {.raw = st->objc_msgSend};
    return send.fn(receiver, sel);
}

static void metal_msg_send_void0(struct metal_state *st,
                                 void *receiver,
                                 const char *selector) {
    if (receiver == nullptr) {
        return;
    }
    void *sel = metal_sel_register_name(st, selector);
    union {
        void *raw;
        void (*fn)(void *, void *);
    } send = {.raw = st->objc_msgSend};
    send.fn(receiver, sel);
}

static void metal_msg_send_copy_buffer(struct metal_state *st,
                                       void *receiver,
                                       const char *selector,
                                       void *src,
                                       size_t src_offset,
                                       void *dst,
                                       size_t dst_offset,
                                       size_t bytes) {
    void *sel = metal_sel_register_name(st, selector);
    union {
        void *raw;
        void (*fn)(void *, void *, void *, size_t, void *, size_t, size_t);
    } send = {.raw = st->objc_msgSend};
    send.fn(receiver, sel, src, src_offset, dst, dst_offset, bytes);
}

static void metal_seq_mark_buffer(struct metal_state *st, void *mtl_buf);

static void metal_msg_send_set_buffer(struct metal_state *st,
                                      void *receiver,
                                      void *buffer,
                                      size_t offset,
                                      size_t index) {
    void *sel = metal_sel_register_name(st, "setBuffer:offset:atIndex:");
    metal_seq_mark_buffer(st, buffer);
    union {
        void *raw;
        void (*fn)(void *, void *, void *, size_t, size_t);
    } send = {.raw = st->objc_msgSend};
    send.fn(receiver, sel, buffer, offset, index);
}

static void metal_msg_send_set_bytes(struct metal_state *st,
                                     void *receiver,
                                     const void *bytes,
                                     size_t length,
                                     size_t index) {
    void *sel = metal_sel_register_name(st, "setBytes:length:atIndex:");
    union {
        void *raw;
        void (*fn)(void *, void *, const void *, size_t, size_t);
    } send = {.raw = st->objc_msgSend};
    send.fn(receiver, sel, bytes, length, index);
}

static void metal_msg_send_set_threadgroup_memory(struct metal_state *st,
                                                  void *receiver,
                                                  size_t length,
                                                  size_t index) {
    void *sel = metal_sel_register_name(st,
                                        "setThreadgroupMemoryLength:atIndex:");
    union {
        void *raw;
        void (*fn)(void *, void *, size_t, size_t);
    } send = {.raw = st->objc_msgSend};
    send.fn(receiver, sel, length, index);
}

static void metal_msg_send_dispatch(struct metal_state *st,
                                    void *receiver,
                                    struct metal_size groups,
                                    struct metal_size threads) {
    void *sel = metal_sel_register_name(
        st, "dispatchThreadgroups:threadsPerThreadgroup:");
    if (st->skip_next_dispatch) {
        st->skip_next_dispatch = false;
        return;
    }
    union {
        void *raw;
        void (*fn)(void *, void *, struct metal_size, struct metal_size);
    } send = {.raw = st->objc_msgSend};
    send.fn(receiver, sel, groups, threads);
    st->seq_dispatch_count++;
    /* diag: re-dispatch (idempotent ops) to measure a shape's pure kernel
     * duration as the runtime delta, without skip-style data corruption. */
    if (st->double_next_dispatch) {
        st->double_next_dispatch = false;
        send.fn(receiver, sel, groups, threads);
    }
}

static bool metal_env_enabled(const char *name) {
    const char *value = getenv(name);
    return value != nullptr && value[0] != '\0' && strcmp(value, "0") != 0;
}

/* Default-on switches: true only when the env var is explicitly "0". */
static bool metal_env_disabled(const char *name) {
    const char *value = getenv(name);
    return value != nullptr && strcmp(value, "0") == 0;
}

static uint64_t metal_now_ns(void) {
    struct timespec ts = {0};
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0;
    }
    return (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
}

static uint64_t metal_saturating_add_u64(uint64_t a, uint64_t b) {
    return UINT64_MAX - a < b ? UINT64_MAX : a + b;
}

static uint64_t metal_profile_workgroups(struct metal_size groups) {
    if (groups.width == 0 || groups.height == 0 || groups.depth == 0) {
        return 0;
    }
    if (groups.width > UINT64_MAX / groups.height) {
        return UINT64_MAX;
    }
    uint64_t xy = (uint64_t) groups.width * (uint64_t) groups.height;
    if (xy > UINT64_MAX / groups.depth) {
        return UINT64_MAX;
    }
    return xy * (uint64_t) groups.depth;
}

static void metal_profile_add_wait(struct metal_state *st,
                                   enum metal_profile_stage stage,
                                   uint64_t start_ns) {
    if (st == nullptr || !st->profile_enabled ||
        stage >= METAL_PROFILE_STAGE_COUNT || start_ns == 0) {
        return;
    }
    const uint64_t end_ns = metal_now_ns();
    if (end_ns < start_ns) {
        return;
    }
    struct metal_profile_stat *stat = &st->profile[stage];
    stat->ns = metal_saturating_add_u64(stat->ns, end_ns - start_ns);
    stat->calls = metal_saturating_add_u64(stat->calls, 1);
}

/* Subtractive profiler: env GEIST_SKIP_<CAT> drops a whole op category's
 * dispatches so the wait.prefill_text delta reveals that category's GPU time.
 * GEIST_SKIP_H additionally restricts a category skip to dispatches with a
 * matching threadgroup-grid height (mm_sg height = ceil(n_out/64), so this
 * isolates a single GEMM shape). Output is garbage during a skip run —
 * timing only. */
static bool metal_skip_grid_match(struct metal_size groups) {
    const char *h = getenv("GEIST_SKIP_H");
    return h == nullptr || (uint32_t) atoi(h) == groups.height;
}

static bool metal_skip_stage(enum metal_profile_stage s) {
    switch (s) {
    case METAL_PROFILE_DISPATCH_Q4K_GATE_UP_BASE:
    case METAL_PROFILE_DISPATCH_Q4K_GATE_UP_N4:
    case METAL_PROFILE_DISPATCH_Q4K_GATE_UP_NT4:
    case METAL_PROFILE_DISPATCH_Q4K_GATE_UP_NT8:
    case METAL_PROFILE_DISPATCH_Q4K_GATE_UP_W4A8:
        return metal_env_enabled("GEIST_SKIP_GEMM") ||
               metal_env_enabled("GEIST_SKIP_GATE_UP");
    case METAL_PROFILE_DISPATCH_Q4K_LINEAR_BASE:
    case METAL_PROFILE_DISPATCH_Q4K_LINEAR_N4:
    case METAL_PROFILE_DISPATCH_Q4K_LINEAR_NT4:
    case METAL_PROFILE_DISPATCH_Q4K_LINEAR_NT8:
    case METAL_PROFILE_DISPATCH_Q4K_LINEAR_W4A8:
    case METAL_PROFILE_DISPATCH_Q4K_QK_BASE:
    case METAL_PROFILE_DISPATCH_Q4K_QK_NT4:
        return metal_env_enabled("GEIST_SKIP_GEMM") ||
               metal_env_enabled("GEIST_SKIP_Q4K_LINEAR");
    case METAL_PROFILE_DISPATCH_Q4K_PLE_GATE_NT8:
    case METAL_PROFILE_DISPATCH_F32_PLE_GATE:
    case METAL_PROFILE_DISPATCH_F32_PLE_PROJ_NORM:
        return metal_env_enabled("GEIST_SKIP_GEMM") ||
               metal_env_enabled("GEIST_SKIP_PLE");
    case METAL_PROFILE_DISPATCH_Q6K_LINEAR_BASE:
    case METAL_PROFILE_DISPATCH_Q6K_LINEAR_N4:
    case METAL_PROFILE_DISPATCH_Q6K_LINEAR_NT4:
    case METAL_PROFILE_DISPATCH_Q6K_LINEAR_NT8:
        return metal_env_enabled("GEIST_SKIP_GEMM") ||
               metal_env_enabled("GEIST_SKIP_Q6K");
    case METAL_PROFILE_DISPATCH_RMSNORM_ROWS:
    case METAL_PROFILE_DISPATCH_RMSNORM_ADD_ROWS:
        return metal_env_enabled("GEIST_SKIP_NORM");
    case METAL_PROFILE_DISPATCH_ATTENTION_ROWS:
    case METAL_PROFILE_DISPATCH_ATTENTION_QNORM_ROWS:
        return metal_env_enabled("GEIST_SKIP_ATTN");
    case METAL_PROFILE_DISPATCH_Q_NORM_ROPE:
    case METAL_PROFILE_DISPATCH_K_NORM_ROPE_APPEND:
    case METAL_PROFILE_DISPATCH_V_NORM_APPEND:
    case METAL_PROFILE_DISPATCH_KV_NORM_APPEND:
    case METAL_PROFILE_DISPATCH_ROPE_ROWS:
        return metal_env_enabled("GEIST_SKIP_ROPE");
    case METAL_PROFILE_DISPATCH_KV_APPEND_ROWS:
        return metal_env_enabled("GEIST_SKIP_KV");
    case METAL_PROFILE_DISPATCH_GELU_MUL_ROWS:
        return metal_env_enabled("GEIST_SKIP_GELU");
    case METAL_PROFILE_DISPATCH_F32_MATMUL:
        return metal_env_enabled("GEIST_SKIP_F32");
    case METAL_PROFILE_DISPATCH_EMBED:
        return metal_env_enabled("GEIST_SKIP_EMBED");
    default:
        return false;
    }
}

static void metal_profile_add_dispatch(struct metal_state *st,
                                       enum metal_profile_stage stage,
                                       struct metal_size groups) {
    if (st == nullptr || !st->profile_enabled ||
        stage >= METAL_PROFILE_STAGE_COUNT) {
        return;
    }
    if (metal_skip_stage(stage) && metal_skip_grid_match(groups)) {
        st->skip_next_dispatch = true;
    }
    {
        const char *dh = getenv("GEIST_DOUBLE_H");
        if (dh != nullptr && (uint32_t) atoi(dh) == groups.height &&
            (stage == METAL_PROFILE_DISPATCH_Q4K_LINEAR_BASE ||
             stage == METAL_PROFILE_DISPATCH_Q6K_LINEAR_BASE)) {
            st->double_next_dispatch = true;
        }
    }
    struct metal_profile_stat *stat = &st->profile[stage];
    stat->calls = metal_saturating_add_u64(stat->calls, 1);
    stat->workgroups = metal_saturating_add_u64(
        stat->workgroups, metal_profile_workgroups(groups));
}

static enum metal_profile_stage metal_profile_wait_stage_for_sequence(
    enum geist_command_sequence_kind kind) {
    switch (kind) {
    case GEIST_COMMAND_SEQUENCE_DECODE_LAYER_LOOP:
        return METAL_PROFILE_WAIT_DECODE_LAYER_LOOP;
    case GEIST_COMMAND_SEQUENCE_DECODE_GREEDY_STEP:
        return METAL_PROFILE_WAIT_DECODE_GREEDY_STEP;
    case GEIST_COMMAND_SEQUENCE_VERIFY_GREEDY:
        return METAL_PROFILE_WAIT_VERIFY_GREEDY;
    case GEIST_COMMAND_SEQUENCE_PREFILL_TEXT:
        return METAL_PROFILE_WAIT_PREFILL_TEXT;
    default:
        return METAL_PROFILE_STAGE_COUNT;
    }
}

static void metal_profile_print_summary(const struct metal_state *st) {
    if (st == nullptr || !st->profile_enabled) {
        return;
    }
    bool any = false;
    for (size_t i = 0; i < METAL_PROFILE_STAGE_COUNT; i++) {
        if (st->profile[i].calls != 0) {
            any = true;
            break;
        }
    }
    if (!any) {
        return;
    }
    fprintf(stderr, "metal backend profile:\n");
    fprintf(stderr,
            "  note: wait.* is CPU wall time spent committing/waiting for "
            "submitted Metal command buffers; dispatch.* counts encoded "
            "kernel dispatches and threadgroups.\n");
    for (size_t i = 0; i < METAL_PROFILE_STAGE_COUNT; i++) {
        const struct metal_profile_stat *stat = &st->profile[i];
        if (stat->calls == 0) {
            continue;
        }
        if (stat->ns != 0) {
            fprintf(stderr, "  %-32s %9.2f ms  (%llu calls)\n",
                    metal_profile_stage_names[i],
                    (double) stat->ns / 1000000.0,
                    (unsigned long long) stat->calls);
        } else {
            fprintf(stderr,
                    "  %-32s %9llu dispatches  %llu threadgroups\n",
                    metal_profile_stage_names[i],
                    (unsigned long long) stat->calls,
                    (unsigned long long) stat->workgroups);
        }
    }
}

static bool metal_decode_replay_can_record(const struct metal_state *st) {
    return st != nullptr &&
           st->sequence_active &&
           st->sequence_kind == GEIST_COMMAND_SEQUENCE_DECODE_GREEDY_STEP &&
           !st->decode_replay_replaying &&
           !st->decode_replay_failed;
}

static struct metal_decode_replay_op *metal_decode_replay_append(
    struct metal_state *st,
    enum metal_decode_replay_op_kind kind) {

    if (!metal_decode_replay_can_record(st)) {
        return nullptr;
    }
    if (st->decode_replay_op_count >= METAL_DECODE_REPLAY_MAX_OPS) {
        st->decode_replay_failed = true;
        st->decode_replay_valid = false;
        return nullptr;
    }
    struct metal_decode_replay_op *op =
        &st->decode_replay_ops[st->decode_replay_op_count++];
    *op = (struct metal_decode_replay_op){.kind = kind};
    return op;
}

static bool metal_ranges_overlap(size_t a_offset,
                                 size_t b_offset,
                                 size_t n_bytes) {
    if (n_bytes == 0) {
        return false;
    }
    return a_offset < b_offset + n_bytes && b_offset < a_offset + n_bytes;
}

static void metal_release_sequence_objects(struct metal_state *st) {
    if (st == nullptr) {
        return;
    }
    metal_msg_send_void0(st, st->sequence_compute_encoder, "release");
    metal_msg_send_void0(st, st->sequence_command_buffer, "release");
    st->sequence_compute_encoder = nullptr;
    st->sequence_command_buffer = nullptr;
    st->sequence_active = false;
    st->sequence_has_work = false;
}

static void metal_destroy_q4k_nt4_cache(struct geist_backend *be);
static void metal_destroy_q6k_nt4_cache(struct geist_backend *be);
static void metal_buffer_destroy_internal(struct geist_backend *be,
                                          struct geist_buffer *buf);

static void metal_destroy_state(struct geist_backend *be,
                                struct metal_state *st) {
    if (be == nullptr || st == nullptr) {
        return;
    }
    metal_profile_print_summary(st);
    free(st->buf_reg);
    st->buf_reg = nullptr;
    st->buf_reg_count = 0;
    st->buf_reg_cap = 0;
    metal_destroy_q4k_nt4_cache(be);
    metal_destroy_q6k_nt4_cache(be);
    metal_buffer_destroy_internal(be, st->q4k_w4a8_xq_buffer);
    metal_buffer_destroy_internal(be, st->q4k_w4a8_scale_buffer);
    metal_buffer_destroy_internal(be, st->attn_dec_partials_buffer);
    st->attn_dec_partials_buffer = nullptr;
    st->attn_dec_partials_capacity = 0;
    st->q4k_w4a8_xq_buffer = nullptr;
    st->q4k_w4a8_scale_buffer = nullptr;
    st->q4k_w4a8_xq_capacity = 0;
    if (st->objc_msgSend != nullptr && st->sel_registerName != nullptr) {
        if (st->sequence_active) {
            metal_msg_send_void0(st, st->sequence_compute_encoder,
                                 "endEncoding");
        }
        metal_release_sequence_objects(st);
        metal_msg_send_void0(st, st->argmax_result_buffer, "release");
        metal_msg_send_void0(st, st->argmax_batch_pipeline, "release");
        metal_msg_send_void0(st, st->argmax_batch_function, "release");
        metal_msg_send_void0(st, st->argmax_pipeline, "release");
        metal_msg_send_void0(st, st->argmax_function, "release");
        metal_msg_send_void0(st, st->rmsnorm_add_rows_simd_pipeline,
                             "release");
        metal_msg_send_void0(st, st->rmsnorm_add_rows_simd_function,
                             "release");
        metal_msg_send_void0(st, st->rmsnorm_add_rows_pipeline, "release");
        metal_msg_send_void0(st, st->rmsnorm_add_rows_function, "release");
        metal_msg_send_void0(st, st->embed_lookup_scaled_pipeline, "release");
        metal_msg_send_void0(st, st->embed_lookup_scaled_function, "release");
        metal_msg_send_void0(st, st->f32_matmul_pipeline, "release");
        metal_msg_send_void0(st, st->f32_matmul_function, "release");
    }
    if (st->f32_matmul_sg_pipeline != nullptr) {
        metal_msg_send_void0(st, st->f32_matmul_sg_pipeline, "release");
        metal_msg_send_void0(st, st->f32_matmul_sg_function, "release");
        metal_msg_send_void0(st, st->f32_matmul_mm_pipeline, "release");
        metal_msg_send_void0(st, st->f32_matmul_mm_function, "release");
        metal_msg_send_void0(st, st->f32_ple_gate_pipeline, "release");
        metal_msg_send_void0(st, st->f32_ple_gate_function, "release");
        metal_msg_send_void0(st, st->f32_ple_proj_norm_pipeline, "release");
        metal_msg_send_void0(st, st->f32_ple_proj_norm_function, "release");
        metal_msg_send_void0(st, st->attention_rows_pipeline, "release");
        metal_msg_send_void0(st, st->attention_rows_function, "release");
        metal_msg_send_void0(st, st->attention_rows_f16_pipeline, "release");
        metal_msg_send_void0(st, st->attention_rows_f16_function, "release");
        metal_msg_send_void0(st, st->attention_qnorm_rows_pipeline, "release");
        metal_msg_send_void0(st, st->attention_qnorm_rows_function, "release");
        metal_msg_send_void0(st, st->attention_qnorm_rows_f16_pipeline, "release");
        metal_msg_send_void0(st, st->attention_qnorm_rows_f16_function, "release");
        metal_msg_send_void0(st, st->attention_qnorm_dec_f16_pipeline, "release");
        metal_msg_send_void0(st, st->attention_qnorm_dec_f16_function, "release");
        metal_msg_send_void0(st, st->attention_dec_combine_pipeline, "release");
        metal_msg_send_void0(st, st->attention_dec_combine_function, "release");
        metal_msg_send_void0(st, st->attention_qnorm_flash_sg_f16_pipeline,
                             "release");
        metal_msg_send_void0(st, st->attention_qnorm_flash_sg_f16_function,
                             "release");
        metal_msg_send_void0(st, st->kv_append_rows_pipeline, "release");
        metal_msg_send_void0(st, st->kv_append_rows_function, "release");
        metal_msg_send_void0(st, st->copy_u32_pipeline, "release");
        metal_msg_send_void0(st, st->copy_u32_function, "release");
        metal_msg_send_void0(st, st->kv_append_rows_f16_pipeline, "release");
        metal_msg_send_void0(st, st->kv_append_rows_f16_function, "release");
        metal_msg_send_void0(st, st->rope_rows_pipeline, "release");
        metal_msg_send_void0(st, st->rope_rows_function, "release");
        metal_msg_send_void0(st, st->v_norm_append_rows_pipeline, "release");
        metal_msg_send_void0(st, st->v_norm_append_rows_function, "release");
        metal_msg_send_void0(st, st->v_norm_append_rows_f16_pipeline,
                             "release");
        metal_msg_send_void0(st, st->v_norm_append_rows_f16_function,
                             "release");
        metal_msg_send_void0(st, st->kv_norm_append_rows_pipeline, "release");
        metal_msg_send_void0(st, st->kv_norm_append_rows_function, "release");
        metal_msg_send_void0(st, st->kv_norm_append_rows_f16_pipeline,
                             "release");
        metal_msg_send_void0(st, st->kv_norm_append_rows_f16_function,
                             "release");
        metal_msg_send_void0(st, st->k_norm_rope_append_rows_pipeline,
                             "release");
        metal_msg_send_void0(st, st->k_norm_rope_append_rows_function,
                             "release");
        metal_msg_send_void0(st, st->k_norm_rope_append_rows_f16_pipeline,
                             "release");
        metal_msg_send_void0(st, st->k_norm_rope_append_rows_f16_function,
                             "release");
        metal_msg_send_void0(st, st->q_norm_rope_rows_pipeline, "release");
        metal_msg_send_void0(st, st->q_norm_rope_rows_function, "release");
        metal_msg_send_void0(st, st->add_rows_pipeline, "release");
        metal_msg_send_void0(st, st->add_rows_function, "release");
        metal_msg_send_void0(st, st->scale_rows_pipeline, "release");
        metal_msg_send_void0(st, st->scale_rows_function, "release");
        metal_msg_send_void0(st, st->gelu_mul_rows_pipeline, "release");
        metal_msg_send_void0(st, st->gelu_mul_rows_function, "release");
        metal_msg_send_void0(st, st->mul_rows_pipeline, "release");
        metal_msg_send_void0(st, st->mul_rows_function, "release");
        metal_msg_send_void0(st, st->gelu_rows_pipeline, "release");
        metal_msg_send_void0(st, st->gelu_rows_function, "release");
        metal_msg_send_void0(st, st->rmsnorm_rows_simd_pipeline,
                             "release");
        metal_msg_send_void0(st, st->rmsnorm_rows_simd_function,
                             "release");
        metal_msg_send_void0(st, st->rmsnorm_rows_pipeline, "release");
        metal_msg_send_void0(st, st->rmsnorm_rows_function, "release");
        metal_msg_send_void0(st, st->q6k_matmul_m8_pipeline, "release");
        metal_msg_send_void0(st, st->q6k_matmul_sg_pipeline, "release");
        metal_msg_send_void0(st, st->q6k_matmul_sg_fast_pipeline, "release");
        metal_msg_send_void0(st, st->q6k_matmul_sg_fast_function, "release");
        metal_msg_send_void0(st, st->q6k_matmul_sg_function, "release");
        metal_msg_send_void0(st, st->q6k_matmul_m8_function, "release");
        metal_msg_send_void0(st, st->q6k_matmul_m16_pipeline, "release");
        metal_msg_send_void0(st, st->q6k_matmul_m16_function, "release");
        metal_msg_send_void0(st, st->q6k_nt8_pipeline, "release");
        metal_msg_send_void0(st, st->q6k_nt8_function, "release");
        metal_msg_send_void0(st, st->q6k_nt4_pipeline, "release");
        metal_msg_send_void0(st, st->q6k_nt4_function, "release");
        metal_msg_send_void0(st, st->q6k_n4_pipeline, "release");
        metal_msg_send_void0(st, st->q6k_n4_function, "release");
        metal_msg_send_void0(st, st->q4k_qk_nt4_pipeline, "release");
        metal_msg_send_void0(st, st->q4k_qk_nt4_function, "release");
        metal_msg_send_void0(st, st->q4k_qk_pipeline, "release");
        metal_msg_send_void0(st, st->q4k_qk_function, "release");
        metal_msg_send_void0(st, st->q4k_gate_up_pipeline, "release");
        metal_msg_send_void0(st, st->q4k_gate_up_function, "release");
        metal_msg_send_void0(st, st->q4k_gate_up_nt8_pipeline, "release");
        metal_msg_send_void0(st, st->q4k_gate_up_nt8_function, "release");
        metal_msg_send_void0(st, st->q4k_gate_up_nt4_pipeline, "release");
        metal_msg_send_void0(st, st->q4k_gate_up_nt4_function, "release");
        metal_msg_send_void0(st, st->q6k_pipeline, "release");
        metal_msg_send_void0(st, st->q6k_function, "release");
        metal_msg_send_void0(st, st->q4k_matmul_m8_pipeline, "release");
        metal_msg_send_void0(st, st->q4k_matmul_m8_function, "release");
        metal_msg_send_void0(st, st->q4k_matmul_m16_pipeline, "release");
        metal_msg_send_void0(st, st->q4k_matmul_m16_function, "release");
        metal_msg_send_void0(st, st->q4k_matmul_m16_n2_pipeline, "release");
        metal_msg_send_void0(st, st->q4k_matmul_m16_n2_function, "release");
        metal_msg_send_void0(st, st->q4k_mm_sg_pipeline, "release");
        metal_msg_send_void0(st, st->q4k_mm_sg_function, "release");
        metal_msg_send_void0(st, st->q4k_mm_sg_fast_pipeline, "release");
        metal_msg_send_void0(st, st->q4k_mm_sg_fast_function, "release");
        metal_msg_send_void0(st, st->q4k_gate_up_w4a8_pipeline, "release");
        metal_msg_send_void0(st, st->q4k_gate_up_w4a8_function, "release");
        metal_msg_send_void0(st, st->q4k_w4a8_pipeline, "release");
        metal_msg_send_void0(st, st->q4k_w4a8_function, "release");
        metal_msg_send_void0(st, st->q4k_quant_x_pipeline, "release");
        metal_msg_send_void0(st, st->q4k_quant_x_function, "release");
        metal_msg_send_void0(st, st->q4k_nt8_pipeline, "release");
        metal_msg_send_void0(st, st->q4k_nt8_function, "release");
        metal_msg_send_void0(st, st->q4k_ple_gate_nt8_pipeline, "release");
        metal_msg_send_void0(st, st->q4k_ple_gate_nt8_function, "release");
        metal_msg_send_void0(st, st->q4k_nt4_pipeline, "release");
        metal_msg_send_void0(st, st->q4k_nt4_function, "release");
        metal_msg_send_void0(st, st->q4k_n4_pipeline, "release");
        metal_msg_send_void0(st, st->q4k_n4_function, "release");
        metal_msg_send_void0(st, st->q4k_pipeline, "release");
        metal_msg_send_void0(st, st->q4k_function, "release");
        metal_msg_send_void0(st, st->attn_library, "release");
        metal_msg_send_void0(st, st->attn_qnorm_library, "release");
        metal_msg_send_void0(st, st->attn_qnorm_f16_library, "release");
        metal_msg_send_void0(st, st->attn_qnorm_dec_f16_library, "release");
        metal_msg_send_void0(st, st->attn_dec_combine_library, "release");
        metal_msg_send_void0(st, st->attn_flash_sg_f16_library, "release");
        metal_msg_send_void0(st, st->kv_norm_append_library, "release");
        metal_msg_send_void0(st, st->kv_norm_append_f16_library, "release");
        metal_msg_send_void0(st, st->v_norm_append_library, "release");
        metal_msg_send_void0(st, st->k_norm_rope_append_library, "release");
        metal_msg_send_void0(st, st->q_norm_rope_library, "release");
        metal_msg_send_void0(st, st->elem_simd_library, "release");
        metal_msg_send_void0(st, st->elem_library, "release");
        metal_msg_send_void0(st, st->embed_library, "release");
        metal_msg_send_void0(st, st->argmax_library, "release");
        metal_msg_send_void0(st, st->f32_library, "release");
        metal_msg_send_void0(st, st->q4k_qk_library, "release");
        metal_msg_send_void0(st, st->q4k_gate_up_nt8_library, "release");
        metal_msg_send_void0(st, st->q4k_gate_up_nt4_library, "release");
        metal_msg_send_void0(st, st->q4k_gate_up_n4_library, "release");
        metal_msg_send_void0(st, st->q4k_gate_up_library, "release");
        metal_msg_send_void0(st, st->q4k_gate_up_w4a8_library, "release");
        metal_msg_send_void0(st, st->q4k_w4a8_library, "release");
        metal_msg_send_void0(st, st->q4k_ple_gate_nt8_library, "release");
        metal_msg_send_void0(st, st->q4k_nt4_library, "release");
        metal_msg_send_void0(st, st->q4k_n4_library, "release");
        metal_msg_send_void0(st, st->q6k_mm_sg_library, "release");
        metal_msg_send_void0(st, st->q6k_mm_sg_fast_library, "release");
        metal_msg_send_void0(st, st->q6k_m16_library, "release");
        metal_msg_send_void0(st, st->q4k_m16_library, "release");
        metal_msg_send_void0(st, st->q4k_m16_n2_library, "release");
        metal_msg_send_void0(st, st->q4k_mm_sg_library, "release");
        metal_msg_send_void0(st, st->q4k_mm_sg_fast_library, "release");
        metal_msg_send_void0(st, st->q6k_nt4_library, "release");
        metal_msg_send_void0(st, st->q6k_nt8_library, "release");
        metal_msg_send_void0(st, st->q6k_n4_library, "release");
        metal_msg_send_void0(st, st->q6k_library, "release");
        metal_msg_send_void0(st, st->q4k_library, "release");
        metal_msg_send_void0(st, st->attn_f16_library, "release");
        metal_msg_send_void0(st, st->command_queue, "release");
        metal_msg_send_void0(st, st->device, "release");
    }
    st->q4k_pipeline = nullptr;
    st->q4k_function = nullptr;
    st->q4k_n4_pipeline = nullptr;
    st->q4k_n4_function = nullptr;
    st->q4k_nt4_pipeline = nullptr;
    st->q4k_nt4_function = nullptr;
    st->q4k_nt8_pipeline = nullptr;
    st->q4k_nt8_function = nullptr;
    st->q4k_ple_gate_nt8_pipeline = nullptr;
    st->q4k_ple_gate_nt8_function = nullptr;
    st->q4k_matmul_m8_pipeline = nullptr;
    st->q4k_matmul_m8_function = nullptr;
    st->q4k_matmul_m16_pipeline = nullptr;
    st->q4k_matmul_m16_function = nullptr;
    st->q4k_matmul_m16_n2_pipeline = nullptr;
    st->q4k_matmul_m16_n2_function = nullptr;
    st->q4k_mm_sg_pipeline = nullptr;
    st->q4k_mm_sg_function = nullptr;
    st->q4k_mm_sg_fast_pipeline = nullptr;
    st->q4k_mm_sg_fast_function = nullptr;
    st->q4k_gate_up_w4a8_pipeline = nullptr;
    st->q4k_gate_up_w4a8_function = nullptr;
    st->q4k_w4a8_pipeline = nullptr;
    st->q4k_w4a8_function = nullptr;
    st->q4k_quant_x_pipeline = nullptr;
    st->q4k_quant_x_function = nullptr;
    st->q4k_m16_library = nullptr;
    st->q4k_mm_sg_library = nullptr;
    st->q4k_mm_sg_fast_library = nullptr;
    st->q6k_pipeline = nullptr;
    st->q6k_function = nullptr;
    st->q6k_n4_pipeline = nullptr;
    st->q6k_n4_function = nullptr;
    st->q6k_nt4_pipeline = nullptr;
    st->q6k_nt4_function = nullptr;
    st->q6k_nt8_pipeline = nullptr;
    st->q6k_nt8_function = nullptr;
    st->q6k_matmul_m8_pipeline = nullptr;
    st->q6k_matmul_sg_function = nullptr;
    st->q6k_matmul_sg_pipeline = nullptr;
    st->q6k_matmul_sg_fast_pipeline = nullptr;
    st->q6k_matmul_sg_fast_function = nullptr;
    st->q6k_matmul_m8_function = nullptr;
    st->q6k_matmul_m16_pipeline = nullptr;
    st->q6k_matmul_m16_function = nullptr;
    st->q6k_mm_sg_library = nullptr;
    st->q6k_mm_sg_fast_library = nullptr;
    st->q6k_m16_library = nullptr;
    st->q4k_qk_nt4_pipeline = nullptr;
    st->q4k_qk_nt4_function = nullptr;
    st->q4k_qk_pipeline = nullptr;
    st->q4k_qk_function = nullptr;
    st->q4k_gate_up_pipeline = nullptr;
    st->q4k_gate_up_function = nullptr;
    st->q4k_gate_up_n4_pipeline = nullptr;
    st->q4k_gate_up_n4_function = nullptr;
    st->q4k_gate_up_nt4_pipeline = nullptr;
    st->q4k_gate_up_nt4_function = nullptr;
    st->q4k_gate_up_nt8_pipeline = nullptr;
    st->q4k_gate_up_nt8_function = nullptr;
    st->rmsnorm_rows_pipeline = nullptr;
    st->rmsnorm_rows_function = nullptr;
    st->rmsnorm_rows_simd_pipeline = nullptr;
    st->rmsnorm_rows_simd_function = nullptr;
    st->gelu_rows_pipeline = nullptr;
    st->gelu_rows_function = nullptr;
    st->mul_rows_pipeline = nullptr;
    st->mul_rows_function = nullptr;
    st->gelu_mul_rows_pipeline = nullptr;
    st->gelu_mul_rows_function = nullptr;
    st->add_rows_pipeline = nullptr;
    st->add_rows_function = nullptr;
    st->scale_rows_pipeline = nullptr;
    st->scale_rows_function = nullptr;
    st->rmsnorm_add_rows_pipeline = nullptr;
    st->rmsnorm_add_rows_function = nullptr;
    st->rmsnorm_add_rows_simd_pipeline = nullptr;
    st->rmsnorm_add_rows_simd_function = nullptr;
    st->embed_lookup_scaled_pipeline = nullptr;
    st->embed_lookup_scaled_function = nullptr;
    st->f32_matmul_pipeline = nullptr;
    st->f32_matmul_function = nullptr;
    st->f32_matmul_sg_pipeline = nullptr;
    st->f32_matmul_sg_function = nullptr;
    st->f32_matmul_mm_pipeline = nullptr;
    st->f32_matmul_mm_function = nullptr;
    st->f32_ple_gate_pipeline = nullptr;
    st->f32_ple_gate_function = nullptr;
    st->f32_ple_proj_norm_pipeline = nullptr;
    st->f32_ple_proj_norm_function = nullptr;
    st->argmax_pipeline = nullptr;
    st->argmax_function = nullptr;
    st->argmax_batch_pipeline = nullptr;
    st->argmax_batch_function = nullptr;
    st->argmax_result_buffer = nullptr;
    st->argmax_result_mapped = nullptr;
    st->argmax_result_capacity = 0;
    st->rope_rows_pipeline = nullptr;
    st->rope_rows_function = nullptr;
    st->kv_append_rows_pipeline = nullptr;
    st->kv_append_rows_function = nullptr;
    st->copy_u32_pipeline = nullptr;
    st->copy_u32_function = nullptr;
    st->kv_append_rows_f16_pipeline = nullptr;
    st->kv_append_rows_f16_function = nullptr;
    st->q_norm_rope_rows_pipeline = nullptr;
    st->q_norm_rope_rows_function = nullptr;
    st->k_norm_rope_append_rows_pipeline = nullptr;
    st->k_norm_rope_append_rows_function = nullptr;
    st->k_norm_rope_append_rows_f16_pipeline = nullptr;
    st->k_norm_rope_append_rows_f16_function = nullptr;
    st->v_norm_append_rows_pipeline = nullptr;
    st->v_norm_append_rows_function = nullptr;
    st->v_norm_append_rows_f16_pipeline = nullptr;
    st->v_norm_append_rows_f16_function = nullptr;
    st->kv_norm_append_rows_pipeline = nullptr;
    st->kv_norm_append_rows_function = nullptr;
    st->kv_norm_append_rows_f16_pipeline = nullptr;
    st->kv_norm_append_rows_f16_function = nullptr;
    st->attention_rows_pipeline = nullptr;
    st->attention_rows_function = nullptr;
    st->attention_rows_f16_pipeline = nullptr;
    st->attention_rows_f16_function = nullptr;
    st->attention_qnorm_rows_pipeline = nullptr;
    st->attention_qnorm_rows_function = nullptr;
    st->attention_qnorm_rows_f16_pipeline = nullptr;
    st->attention_qnorm_rows_f16_function = nullptr;
    st->attention_qnorm_dec_f16_pipeline = nullptr;
    st->attention_qnorm_dec_f16_function = nullptr;
    st->attention_dec_combine_pipeline = nullptr;
    st->attention_dec_combine_function = nullptr;
    st->attention_qnorm_flash_sg_f16_pipeline = nullptr;
    st->attention_qnorm_flash_sg_f16_function = nullptr;
    st->attn_library = nullptr;
    st->attn_f16_library = nullptr;
    st->attn_qnorm_library = nullptr;
    st->attn_qnorm_f16_library = nullptr;
    st->attn_qnorm_dec_f16_library = nullptr;
    st->attn_dec_combine_library = nullptr;
    st->attn_flash_sg_f16_library = nullptr;
    st->kv_norm_append_library = nullptr;
    st->kv_norm_append_f16_library = nullptr;
    st->q_norm_rope_library = nullptr;
    st->k_norm_rope_append_library = nullptr;
    st->v_norm_append_library = nullptr;
    st->elem_library = nullptr;
    st->elem_simd_library = nullptr;
    st->embed_library = nullptr;
    st->f32_library = nullptr;
    st->q4k_gate_up_n4_library = nullptr;
    st->q4k_gate_up_nt4_library = nullptr;
    st->q4k_gate_up_nt8_library = nullptr;
    st->q4k_gate_up_library = nullptr;
    st->q4k_gate_up_w4a8_library = nullptr;
    st->q4k_w4a8_library = nullptr;
    st->q4k_n4_library = nullptr;
    st->q4k_nt4_library = nullptr;
    st->q4k_ple_gate_nt8_library = nullptr;
    st->q4k_m16_n2_library = nullptr;
    st->q6k_nt4_library = nullptr;
    st->q6k_nt8_library = nullptr;
    st->q6k_n4_library = nullptr;
    st->q6k_library = nullptr;
    st->q4k_library = nullptr;
    st->q4k_qk_library = nullptr;
    st->command_queue = nullptr;
    st->device = nullptr;
    if (st->metal_handle != nullptr) {
        dlclose(st->metal_handle);
        st->metal_handle = nullptr;
    }
    if (st->objc_handle != nullptr) {
        dlclose(st->objc_handle);
        st->objc_handle = nullptr;
    }
    geist_backend_free(be, st);
}

static void metal_destroy(struct geist_backend *be) {
    if (be == nullptr || be->state == nullptr) {
        return;
    }
    metal_destroy_state(be, be->state);
    be->state = nullptr;
}

static bool metal_buffer_wants_host_visible(enum geist_buffer_role role,
                                            unsigned int memory_flags) {
    (void) role;
    (void) memory_flags;
    /* main's contract maps weights and scratch to host pointers (resolve_
     * weight reads w->raw; linear_w_or_legacy buffer_maps x/y). On Apple
     * unified memory SHARED storage is the same physical memory as PRIVATE
     * with zero perf cost, so make every buffer host-visible — otherwise
     * buffer_map returns null and state_create fails. */
    return true;
}

/* ---- Buffer registry (host contents pointer -> geist_buffer) ---------- */

static void metal_buf_reg_add(struct metal_state *st,
                              struct geist_buffer *buf) {
    if (st == nullptr || buf == nullptr || buf->mapped == nullptr) {
        return;
    }
    if (st->buf_reg_count == st->buf_reg_cap) {
        const size_t ncap = st->buf_reg_cap != 0 ? st->buf_reg_cap * 2 : 64;
        void *grown = realloc(st->buf_reg, ncap * sizeof(*st->buf_reg));
        if (grown == nullptr) {
            return; /* lookup will miss and report loudly */
        }
        st->buf_reg = grown;
        st->buf_reg_cap = ncap;
    }
    st->buf_reg[st->buf_reg_count++] = (struct metal_buf_reg_entry){
        .base = buf->mapped,
        .bytes = buf->bytes,
        .buf = buf,
    };
}

static void metal_buf_reg_remove(struct metal_state *st,
                                 const struct geist_buffer *buf) {
    if (st == nullptr || st->buf_reg == nullptr) {
        return;
    }
    for (size_t i = 0; i < st->buf_reg_count; i++) {
        if (st->buf_reg[i].buf == buf) {
            st->buf_reg[i] = st->buf_reg[--st->buf_reg_count];
            return;
        }
    }
}

static struct geist_buffer *metal_buf_reg_find(struct metal_state *st,
                                               const void *p,
                                               size_t *out_off) {
    if (st == nullptr || p == nullptr) {
        return nullptr;
    }
    const uint8_t *q = p;
    for (size_t i = 0; i < st->buf_reg_count; i++) {
        const struct metal_buf_reg_entry *e = &st->buf_reg[i];
        if (q >= e->base && q < e->base + e->bytes) {
            *out_off = (size_t) (q - e->base);
            return e->buf;
        }
    }
    return nullptr;
}

/* ---- Batched-submit reference tracking + flush ------------------------ */

[[nodiscard]] static enum geist_status metal_command_sequence_begin(
    struct geist_backend *be, enum geist_command_sequence_kind kind,
    int *out_token);
[[nodiscard]] static enum geist_status metal_command_sequence_end(
    struct geist_backend *be, int token, bool submit);

static void metal_seq_ref_clear(struct metal_state *st) {
    memset(st->seq_ref, 0, sizeof(st->seq_ref));
    st->seq_ref_count = 0;
    st->seq_ref_overflow = false;
}

static void metal_seq_mark_buffer(struct metal_state *st, void *mtl_buf) {
    if (st == nullptr || !st->sequence_active || mtl_buf == nullptr) {
        return;
    }
    const size_t mask = (sizeof(st->seq_ref) / sizeof(st->seq_ref[0])) - 1u;
    size_t h = ((uintptr_t) mtl_buf >> 4) & mask;
    for (size_t i = 0; i <= mask; i++) {
        const size_t slot = (h + i) & mask;
        if (st->seq_ref[slot] == mtl_buf) {
            return;
        }
        if (st->seq_ref[slot] == nullptr) {
            st->seq_ref[slot] = mtl_buf;
            if (++st->seq_ref_count > mask - 256u) {
                st->seq_ref_overflow = true;
            }
            return;
        }
    }
    st->seq_ref_overflow = true;
}

static bool metal_seq_references(struct metal_state *st, const void *mtl_buf) {
    if (st == nullptr || !st->sequence_active || mtl_buf == nullptr) {
        return false;
    }
    if (st->seq_ref_overflow) {
        return true;
    }
    const size_t mask = (sizeof(st->seq_ref) / sizeof(st->seq_ref[0])) - 1u;
    size_t h = ((uintptr_t) mtl_buf >> 4) & mask;
    for (size_t i = 0; i <= mask; i++) {
        const size_t slot = (h + i) & mask;
        if (st->seq_ref[slot] == mtl_buf) {
            return true;
        }
        if (st->seq_ref[slot] == nullptr) {
            return false;
        }
    }
    return true;
}

/* Submit the open batch and start a fresh one of the same kind. Called
 * whenever the host is about to read or overwrite GPU-referenced memory. */
static void metal_batch_flush(struct metal_state *st) {
    if (st == nullptr || !st->sequence_active || !st->sequence_has_work) {
        return;
    }
    struct geist_backend *be = st->backend;
    const enum geist_command_sequence_kind kind = st->sequence_kind;
    (void) metal_command_sequence_end(be, st->sequence_token, true);
    metal_seq_ref_clear(st);
    int tok = 0;
    (void) metal_command_sequence_begin(be, kind, &tok);
}

static void metal_flush_if_referenced(struct metal_state *st,
                                      const void *mtl_buf) {
    if (metal_seq_references(st, mtl_buf)) {
        static int dbg = -1;
        if (dbg < 0) { const char *e = getenv("GEIST_SEQ_COUNT"); dbg = (e && e[0]) ? 1 : 0; }
        if (dbg) { fprintf(stderr, "[flush] buf=%p\n", mtl_buf); }
        metal_batch_flush(st);
    }
}

[[nodiscard]] static enum geist_status metal_new_buffer(
    struct geist_backend *be,
    size_t bytes,
    enum geist_buffer_role role,
    unsigned int memory_flags,
    bool host_visible,
    struct geist_buffer **out) {

    if (out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out = nullptr;
    if (be == nullptr || be->state == nullptr || bytes == 0) {
        if (be != nullptr) {
            geist_backend_set_error(be, GEIST_E_INVALID_ARG,
                                    "metal: invalid buffer create request");
        }
        return GEIST_E_INVALID_ARG;
    }

    struct metal_state *st = be->state;
    struct geist_buffer *buf =
        geist_backend_alloc(be, sizeof(*buf), alignof(struct geist_buffer));
    if (buf == nullptr) {
        geist_backend_set_error(be, GEIST_E_OOM,
                                "metal: failed to allocate buffer handle");
        return GEIST_E_OOM;
    }

    const unsigned long options =
        host_visible ? METAL_RESOURCE_STORAGE_MODE_SHARED
                     : METAL_RESOURCE_STORAGE_MODE_PRIVATE;
    void *mtl_buffer = metal_msg_send_id_size_uint(
        st, st->device, "newBufferWithLength:options:", bytes, options);
    if (mtl_buffer == nullptr) {
        geist_backend_free(be, buf);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal: failed to allocate %zu-byte buffer",
                                bytes);
        return GEIST_E_BACKEND;
    }

    void *mapped = host_visible
                       ? metal_msg_send_ptr0(st, mtl_buffer, "contents")
                       : nullptr;
    if (host_visible && mapped == nullptr) {
        metal_msg_send_void0(st, mtl_buffer, "release");
        geist_backend_free(be, buf);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal: host-visible buffer has no contents");
        return GEIST_E_BACKEND;
    }

    const unsigned int actual_flags =
        memory_flags |
        (host_visible ? (GEIST_MEMORY_HOST_VISIBLE | GEIST_MEMORY_MAPPED)
                      : GEIST_MEMORY_DEVICE);
    *buf = (struct geist_buffer){
        .owner = st,
        .buffer = mtl_buffer,
        .mapped = mapped,
        .bytes = bytes,
        .role = role,
        .memory_flags = actual_flags,
        .host_visible = host_visible,
    };
    metal_buf_reg_add(st, buf);
    *out = buf;
    return GEIST_OK;
}

static void metal_buffer_destroy_internal(struct geist_backend *be,
                                          struct geist_buffer *buf) {
    if (buf == nullptr) {
        return;
    }
    struct metal_state *st = buf->owner;
    metal_buf_reg_remove(st, buf);
    if (be == nullptr && st != nullptr) {
        be = st->backend;
    }
    if (be != nullptr) {
        struct metal_state *bst = be->state;
        if (bst != nullptr) {
            struct metal_q4k_nt4_cache_entry **link =
                &bst->q4k_nt4_cache;
            while (*link != nullptr) {
                struct metal_q4k_nt4_cache_entry *entry = *link;
                if (entry->src != buf) {
                    link = &entry->next;
                    continue;
                }
                *link = entry->next;
                if (entry->packed != nullptr) {
                    metal_buffer_destroy_internal(be, entry->packed);
                }
                geist_backend_free(be, entry);
            }
            struct metal_q6k_nt4_cache_entry **q6_link =
                &bst->q6k_nt4_cache;
            while (*q6_link != nullptr) {
                struct metal_q6k_nt4_cache_entry *entry = *q6_link;
                if (entry->src != buf) {
                    q6_link = &entry->next;
                    continue;
                }
                *q6_link = entry->next;
                if (entry->packed != nullptr) {
                    metal_buffer_destroy_internal(be, entry->packed);
                }
                geist_backend_free(be, entry);
            }
        }
    }
    if (st != nullptr && st->objc_msgSend != nullptr &&
        st->sel_registerName != nullptr) {
        metal_msg_send_void0(st, buf->buffer, "release");
    }
    if (be != nullptr) {
        geist_backend_free(be, buf);
    }
}

[[nodiscard]] static enum geist_status metal_submit_copy(
    struct metal_state *st,
    void *src,
    size_t src_offset,
    void *dst,
    size_t dst_offset,
    size_t n_bytes) {

    if (st == nullptr || src == nullptr || dst == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    if (n_bytes == 0) {
        return GEIST_OK;
    }

    /* In-sequence fast path: when a command sequence is recording, encode the
     * copy as a compute dispatch (copy_u32) on the SEQUENCE's existing compute
     * encoder instead of committing a standalone buffer + waitUntilCompleted.
     * The per-copy GPU round-trip was ~53% of prefill wall (per-layer PLE/KV
     * copies). Dispatches run in order on a serial compute encoder, so the copy
     * correctly sees prior writes; all commits once at sequence_end. Using a
     * compute dispatch (not a blit encoder) avoids exhausting the per-command-
     * buffer encoder limit at long context. Requires 4-byte alignment; other
     * copies fall through to the standalone blit below. */
    if (st->sequence_active && st->sequence_compute_encoder != nullptr &&
        st->copy_u32_pipeline != nullptr &&
        (src_offset % 4u) == 0 && (dst_offset % 4u) == 0 &&
        (n_bytes % 4u) == 0) {
        void *enc = st->sequence_compute_encoder;
        struct {
            uint32_t so, dof, n;
        } cp = {(uint32_t) (src_offset / 4u), (uint32_t) (dst_offset / 4u),
                (uint32_t) (n_bytes / 4u)};
        (void) metal_msg_send_id_id(st, enc, "setComputePipelineState:",
                                    st->copy_u32_pipeline);
        metal_msg_send_set_buffer(st, enc, src, 0, 0);
        metal_msg_send_set_buffer(st, enc, dst, 0, 1);
        metal_msg_send_set_bytes(st, enc, &cp, sizeof(cp), 2);
        const struct metal_size groups = {(cp.n + 255u) / 256u, 1, 1};
        const struct metal_size threads = {256, 1, 1};
        metal_msg_send_dispatch(st, enc, groups, threads);
        st->sequence_has_work = true;
        return GEIST_OK;
    }

    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    if (cmd == nullptr) {
        geist_backend_set_error(st->backend, GEIST_E_BACKEND,
                                "metal: failed to create command buffer");
        return GEIST_E_BACKEND;
    }
    void *blit = metal_msg_send_id0(st, cmd, "blitCommandEncoder");
    if (blit == nullptr) {
        geist_backend_set_error(st->backend, GEIST_E_BACKEND,
                                "metal: failed to create blit encoder");
        return GEIST_E_BACKEND;
    }

    metal_msg_send_copy_buffer(
        st, blit,
        "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:",
        src, src_offset, dst, dst_offset, n_bytes);
    metal_msg_send_void0(st, blit, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");

    void *err = metal_msg_send_id0(st, cmd, "error");
    if (err != nullptr) {
        geist_backend_set_error(st->backend, GEIST_E_BACKEND,
                                "metal: blit command failed");
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status metal_buffer_create(
    struct geist_backend *be,
    size_t bytes,
    enum geist_buffer_role role,
    unsigned int memory_flags,
    struct geist_buffer **out) {

    const bool host_visible =
        metal_buffer_wants_host_visible(role, memory_flags);
    return metal_new_buffer(be, bytes, role, memory_flags, host_visible, out);
}

/* Alias a host-resident region (mmap'd weight or an arena sub-range) as a
 * device buffer. main's memory model hands the backend arbitrary host
 * sub-pointers (64-byte aligned), which Metal's newBufferWithBytesNoCopy
 * cannot wrap (it needs page alignment). Correctness-first: copy the bytes
 * into a SHARED MTLBuffer (unified memory, host+GPU coherent). Weights are
 * read-only; arena scratch is always accessed via this handle, so a per-
 * buffer copy stays coherent. ponytail: zero-copy via a per-arena MTLBuffer
 * + offset is the Stage-6 memory optimization (saves the weight duplication). */
[[nodiscard]] static enum geist_status metal_buffer_create_aliased(
    struct geist_backend *be,
    void *host_ptr,
    size_t n_bytes,
    enum geist_buffer_role role,
    struct geist_buffer **out) {

    if (out == nullptr) { return GEIST_E_INVALID_ARG; }
    *out = nullptr;
    if (be == nullptr || be->state == nullptr || host_ptr == nullptr ||
        n_bytes == 0) {
        if (be != nullptr) {
            geist_backend_set_error(be, GEIST_E_INVALID_ARG,
                                    "metal: invalid aliased buffer request");
        }
        return GEIST_E_INVALID_ARG;
    }
    struct metal_state *st = be->state;
    struct geist_buffer *buf =
        geist_backend_alloc(be, sizeof(*buf), alignof(struct geist_buffer));
    if (buf == nullptr) {
        geist_backend_set_error(be, GEIST_E_OOM,
                                "metal: failed to allocate buffer handle");
        return GEIST_E_OOM;
    }
    void *mtl_buffer = metal_msg_send_id_ptr_size_uint(
        st, st->device, "newBufferWithBytes:length:options:",
        host_ptr, n_bytes, METAL_RESOURCE_STORAGE_MODE_SHARED);
    if (mtl_buffer == nullptr) {
        geist_backend_free(be, buf);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal: failed to alias %zu bytes", n_bytes);
        return GEIST_E_BACKEND;
    }
    void *mapped = metal_msg_send_ptr0(st, mtl_buffer, "contents");
    if (mapped == nullptr) {
        metal_msg_send_void0(st, mtl_buffer, "release");
        geist_backend_free(be, buf);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal: aliased buffer has no contents");
        return GEIST_E_BACKEND;
    }
    *buf = (struct geist_buffer){
        .owner = st,
        .buffer = mtl_buffer,
        .mapped = mapped,
        .bytes = n_bytes,
        .role = role,
        .memory_flags = GEIST_MEMORY_HOST_VISIBLE | GEIST_MEMORY_MAPPED |
                        GEIST_MEMORY_ALIASED,
        .host_visible = true,
    };
    metal_buf_reg_add(st, buf);
    *out = buf;
    return GEIST_OK;
}

static void metal_buffer_destroy(struct geist_backend *be,
                                 struct geist_buffer *buf) {
    metal_buffer_destroy_internal(be, buf);
}

[[nodiscard]] static enum geist_status metal_buffer_copy(
    struct geist_buffer *dst,
    size_t dst_offset,
    const struct geist_buffer *src,
    size_t src_offset,
    size_t n_bytes) {

    if (dst == nullptr || src == nullptr || dst->owner == nullptr ||
        src->owner == nullptr || dst->owner != src->owner) {
        return GEIST_E_INVALID_ARG;
    }
    if (dst_offset > dst->bytes || src_offset > src->bytes ||
        n_bytes > dst->bytes - dst_offset ||
        n_bytes > src->bytes - src_offset) {
        return GEIST_E_INVALID_ARG;
    }
    if (n_bytes == 0) {
        return GEIST_OK;
    }

    struct metal_state *st = dst->owner;
    if (dst == src && metal_ranges_overlap(dst_offset, src_offset, n_bytes)) {
        struct geist_buffer *tmp = nullptr;
        enum geist_status s = metal_new_buffer(
            st->backend, n_bytes, GEIST_BUFFER_SCRATCH,
            GEIST_MEMORY_DEVICE, false, &tmp);
        if (s != GEIST_OK) {
            return s;
        }
        s = metal_submit_copy(st, src->buffer, src_offset, tmp->buffer, 0,
                              n_bytes);
        if (s == GEIST_OK) {
            s = metal_submit_copy(st, tmp->buffer, 0, dst->buffer, dst_offset,
                                  n_bytes);
        }
        metal_buffer_destroy_internal(st->backend, tmp);
        return s;
    }

    return metal_submit_copy(st, src->buffer, src_offset, dst->buffer,
                             dst_offset, n_bytes);
}

[[nodiscard]] static enum geist_status metal_buffer_upload(
    struct geist_buffer *buf,
    size_t n_bytes,
    const uint8_t *src) {

    if (buf == nullptr || n_bytes > buf->bytes ||
        (n_bytes > 0 && src == nullptr)) {
        return GEIST_E_INVALID_ARG;
    }
    if (n_bytes == 0) {
        return GEIST_OK;
    }
    metal_flush_if_referenced(buf->owner, buf->buffer);

    if (buf->host_visible) {
        memcpy(buf->mapped, src, n_bytes);
        return GEIST_OK;
    }

    struct metal_state *st = buf->owner;
    struct geist_buffer *staging = nullptr;
    enum geist_status s = metal_new_buffer(
        st->backend, n_bytes, GEIST_BUFFER_STAGING,
        GEIST_MEMORY_HOST_VISIBLE, true, &staging);
    if (s != GEIST_OK) {
        return s;
    }
    memcpy(staging->mapped, src, n_bytes);
    s = metal_submit_copy(st, staging->buffer, 0, buf->buffer, 0, n_bytes);
    metal_buffer_destroy_internal(st->backend, staging);
    return s;
}

[[nodiscard]] static enum geist_status metal_buffer_download(
    size_t n_bytes,
    uint8_t *dst,
    const struct geist_buffer *buf) {

    if (buf == nullptr || n_bytes > buf->bytes ||
        (n_bytes > 0 && dst == nullptr)) {
        return GEIST_E_INVALID_ARG;
    }
    if (n_bytes == 0) {
        return GEIST_OK;
    }
    metal_flush_if_referenced(buf->owner, buf->buffer);

    if (buf->host_visible) {
        memcpy(dst, buf->mapped, n_bytes);
        return GEIST_OK;
    }

    struct metal_state *st = buf->owner;
    struct geist_buffer *staging = nullptr;
    enum geist_status s = metal_new_buffer(
        st->backend, n_bytes, GEIST_BUFFER_STAGING,
        GEIST_MEMORY_HOST_VISIBLE, true, &staging);
    if (s != GEIST_OK) {
        return s;
    }
    s = metal_submit_copy(st, buf->buffer, 0, staging->buffer, 0, n_bytes);
    if (s == GEIST_OK) {
        memcpy(dst, staging->mapped, n_bytes);
    }
    metal_buffer_destroy_internal(st->backend, staging);
    return s;
}

static struct geist_buffer *metal_q4k_nt4_cache_find(
    const struct metal_state *st,
    const struct geist_tensor *w,
    size_t n_in,
    size_t n_out) {

    if (st == nullptr || w == nullptr || w->buffer == nullptr ||
        w->dtype != GEIST_DTYPE_Q4_K ||
        w->layout != GEIST_LAYOUT_BLOCK_QUANTIZED) {
        return nullptr;
    }
    for (struct metal_q4k_nt4_cache_entry *entry = st->q4k_nt4_cache;
         entry != nullptr; entry = entry->next) {
        if (entry->src == w->buffer &&
            entry->src_offset == w->offset &&
            entry->n_in == n_in &&
            entry->n_out == n_out) {
            return entry->packed;
        }
    }
    return nullptr;
}

static struct geist_buffer *metal_q4k_nt4_cache_find_tensor(
    const struct metal_state *st,
    const struct geist_tensor *w) {

    if (w == nullptr || w->ndim != 2 ||
        w->shape[0] <= 0 || w->shape[1] <= 0) {
        return nullptr;
    }
    return metal_q4k_nt4_cache_find(st, w, (size_t) w->shape[1],
                                    (size_t) w->shape[0]);
}

static void metal_destroy_q4k_nt4_cache(struct geist_backend *be) {
    if (be == nullptr || be->state == nullptr) {
        return;
    }
    struct metal_state *st = be->state;
    struct metal_q4k_nt4_cache_entry *entry = st->q4k_nt4_cache;
    st->q4k_nt4_cache = nullptr;
    while (entry != nullptr) {
        struct metal_q4k_nt4_cache_entry *next = entry->next;
        if (entry->packed != nullptr) {
            metal_buffer_destroy_internal(be, entry->packed);
        }
        geist_backend_free(be, entry);
        entry = next;
    }
}

static struct geist_buffer *metal_q6k_nt4_cache_find(
    const struct metal_state *st,
    const struct geist_tensor *w,
    size_t n_in,
    size_t n_out) {

    if (st == nullptr || w == nullptr || w->buffer == nullptr ||
        w->dtype != GEIST_DTYPE_Q6_K ||
        w->layout != GEIST_LAYOUT_BLOCK_QUANTIZED) {
        return nullptr;
    }
    for (struct metal_q6k_nt4_cache_entry *entry = st->q6k_nt4_cache;
         entry != nullptr; entry = entry->next) {
        if (entry->src == w->buffer &&
            entry->src_offset == w->offset &&
            entry->n_in == n_in &&
            entry->n_out == n_out) {
            return entry->packed;
        }
    }
    return nullptr;
}

static struct geist_buffer *metal_q6k_nt4_cache_find_tensor(
    const struct metal_state *st,
    const struct geist_tensor *w) {

    if (w == nullptr || w->ndim != 2 ||
        w->shape[0] <= 0 || w->shape[1] <= 0) {
        return nullptr;
    }
    return metal_q6k_nt4_cache_find(st, w, (size_t) w->shape[1],
                                    (size_t) w->shape[0]);
}

static void metal_destroy_q6k_nt4_cache(struct geist_backend *be) {
    if (be == nullptr || be->state == nullptr) {
        return;
    }
    struct metal_state *st = be->state;
    struct metal_q6k_nt4_cache_entry *entry = st->q6k_nt4_cache;
    st->q6k_nt4_cache = nullptr;
    while (entry != nullptr) {
        struct metal_q6k_nt4_cache_entry *next = entry->next;
        if (entry->packed != nullptr) {
            metal_buffer_destroy_internal(be, entry->packed);
        }
        geist_backend_free(be, entry);
        entry = next;
    }
}

static void *metal_buffer_map(struct geist_buffer *buf) {
    if (buf == nullptr || !buf->host_visible) {
        return nullptr;
    }
    /* The engine reads/writes through this pointer; if pending batched GPU
     * work references the buffer, submit it first (read: results must be
     * visible; write: encoded ops must not observe the new contents). */
    {
        struct metal_state *st = buf->owner;
        if (metal_seq_references(st, buf->buffer)) {
            static int dbg = -1;
            if (dbg < 0) { const char *e = getenv("GEIST_SEQ_COUNT"); dbg = (e && e[0]) ? 1 : 0; }
            if (dbg) { fprintf(stderr, "[flushmap] bytes=%zu role=%d\n", buf->bytes, (int) buf->role); }
            metal_batch_flush(st);
        }
    }
    return buf->mapped;
}

static void metal_buffer_unmap(struct geist_buffer *buf) {
    (void) buf;
}

static bool metal_tensor_is_f32_vector(const struct geist_tensor *t,
                                       size_t *out_n,
                                       size_t *out_offset_floats) {
    if (t == nullptr || t->buffer == nullptr ||
        t->dtype != GEIST_DTYPE_F32 ||
        t->layout != GEIST_LAYOUT_DENSE ||
        (t->ndim != 1 && t->ndim != 2) ||
        t->offset % sizeof(float) != 0) {
        return false;
    }
    size_t n = 0;
    if (t->ndim == 1) {
        if (t->shape[0] <= 0 || t->stride[0] != 1) {
            return false;
        }
        n = (size_t) t->shape[0];
    } else {
        if (t->shape[0] != 1 || t->shape[1] <= 0 ||
            t->stride[0] != t->shape[1] || t->stride[1] != 1) {
            return false;
        }
        n = (size_t) t->shape[1];
    }
    if (t->offset > t->buffer->bytes ||
        n > (t->buffer->bytes - t->offset) / sizeof(float)) {
        return false;
    }
    *out_n = n;
    *out_offset_floats = t->offset / sizeof(float);
    return true;
}

static bool metal_tensor_is_f32_matrix(const struct geist_tensor *t,
                                       size_t *out_rows,
                                       size_t *out_cols,
                                       size_t *out_offset_floats,
                                       size_t *out_row_stride) {
    if (t == nullptr || t->buffer == nullptr ||
        t->dtype != GEIST_DTYPE_F32 ||
        t->layout != GEIST_LAYOUT_DENSE ||
        t->ndim != 2 ||
        t->shape[0] <= 0 || t->shape[1] <= 0 ||
        t->stride[0] != t->shape[1] || t->stride[1] != 1 ||
        t->offset % sizeof(float) != 0) {
        return false;
    }
    const size_t rows = (size_t) t->shape[0];
    const size_t cols = (size_t) t->shape[1];
    if (rows > SIZE_MAX / cols) {
        return false;
    }
    const size_t elems = rows * cols;
    if (t->offset > t->buffer->bytes ||
        elems > (t->buffer->bytes - t->offset) / sizeof(float)) {
        return false;
    }
    *out_rows = rows;
    *out_cols = cols;
    *out_offset_floats = t->offset / sizeof(float);
    *out_row_stride = cols;
    return true;
}

/* Like metal_tensor_is_f32_matrix but accepts a row stride wider than the
 * column count (a strided view into a larger slab, e.g. the per-layer PLE
 * slice). Kernels take the row stride from their params. */
static bool metal_tensor_is_f32_matrix_strided(const struct geist_tensor *t,
                                               size_t *out_rows,
                                               size_t *out_cols,
                                               size_t *out_offset_floats,
                                               size_t *out_row_stride) {
    if (t == nullptr || t->buffer == nullptr ||
        t->dtype != GEIST_DTYPE_F32 ||
        t->layout != GEIST_LAYOUT_DENSE ||
        t->ndim != 2 ||
        t->shape[0] <= 0 || t->shape[1] <= 0 ||
        t->stride[0] < t->shape[1] || t->stride[1] != 1 ||
        t->offset % sizeof(float) != 0) {
        return false;
    }
    const size_t rows = (size_t) t->shape[0];
    const size_t cols = (size_t) t->shape[1];
    const size_t stride = (size_t) t->stride[0];
    if (rows > 1 && stride > (SIZE_MAX - cols) / (rows - 1)) {
        return false;
    }
    const size_t elems = (rows - 1) * stride + cols;
    if (t->offset > t->buffer->bytes ||
        elems > (t->buffer->bytes - t->offset) / sizeof(float)) {
        return false;
    }
    *out_rows = rows;
    *out_cols = cols;
    *out_offset_floats = t->offset / sizeof(float);
    *out_row_stride = stride;
    return true;
}

static bool metal_tensor_is_f32_rows(const struct geist_tensor *t,
                                     size_t *out_rows,
                                     size_t *out_cols,
                                     size_t *out_offset_floats,
                                     size_t *out_row_stride) {
    if (t == nullptr || out_rows == nullptr || out_cols == nullptr ||
        out_offset_floats == nullptr || out_row_stride == nullptr) {
        return false;
    }
    if (t->ndim == 1) {
        size_t n = 0;
        size_t off = 0;
        if (!metal_tensor_is_f32_vector(t, &n, &off)) {
            return false;
        }
        *out_rows = 1;
        *out_cols = n;
        *out_offset_floats = off;
        *out_row_stride = n;
        return true;
    }
    /* The elementwise kernels take per-tensor row strides from their
     * params, so a strided 2D view (e.g. the per-layer PLE slice of the
     * [seq, n_layers*hpl] slab) is fine here. */
    return metal_tensor_is_f32_matrix_strided(
        t, out_rows, out_cols, out_offset_floats, out_row_stride);
}

static bool metal_tensor_is_q4k_matrix(const struct geist_tensor *t,
                                       size_t *out_rows,
                                       size_t *out_cols,
                                       size_t *out_offset_bytes) {
    if (t == nullptr || t->buffer == nullptr ||
        t->dtype != GEIST_DTYPE_Q4_K ||
        t->layout != GEIST_LAYOUT_BLOCK_QUANTIZED ||
        t->ndim != 2 || t->shape[0] <= 0 || t->shape[1] <= 0 ||
        ((size_t) t->shape[1] % METAL_Q4K_BLOCK_ELEMS) != 0) {
        return false;
    }
    const size_t rows = (size_t) t->shape[0];
    const size_t cols = (size_t) t->shape[1];
    const size_t blocks_per_row = cols / METAL_Q4K_BLOCK_ELEMS;
    if (rows > SIZE_MAX / blocks_per_row ||
        rows * blocks_per_row > SIZE_MAX / METAL_Q4K_BLOCK_BYTES) {
        return false;
    }
    const size_t bytes = rows * blocks_per_row * METAL_Q4K_BLOCK_BYTES;
    if (t->offset > t->buffer->bytes ||
        bytes > t->buffer->bytes - t->offset) {
        return false;
    }
    *out_rows = rows;
    *out_cols = cols;
    *out_offset_bytes = t->offset;
    return true;
}

static bool metal_tensor_is_q6k_matrix(const struct geist_tensor *t,
                                       size_t *out_rows,
                                       size_t *out_cols,
                                       size_t *out_offset_bytes) {
    if (t == nullptr || t->buffer == nullptr ||
        t->dtype != GEIST_DTYPE_Q6_K ||
        t->layout != GEIST_LAYOUT_BLOCK_QUANTIZED ||
        t->ndim != 2 || t->shape[0] <= 0 || t->shape[1] <= 0 ||
        ((size_t) t->shape[1] % METAL_Q6K_BLOCK_ELEMS) != 0) {
        return false;
    }
    const size_t rows = (size_t) t->shape[0];
    const size_t cols = (size_t) t->shape[1];
    const size_t blocks_per_row = cols / METAL_Q6K_BLOCK_ELEMS;
    if (rows > SIZE_MAX / blocks_per_row ||
        rows * blocks_per_row > SIZE_MAX / METAL_Q6K_BLOCK_BYTES) {
        return false;
    }
    const size_t bytes = rows * blocks_per_row * METAL_Q6K_BLOCK_BYTES;
    if (t->offset > t->buffer->bytes ||
        bytes > t->buffer->bytes - t->offset) {
        return false;
    }
    *out_rows = rows;
    *out_cols = cols;
    *out_offset_bytes = t->offset;
    return true;
}

[[nodiscard]] static enum geist_status metal_create_named_pipeline(
    struct geist_backend *be,
    void *library,
    void *ns_string,
    const char *name,
    void **out_function,
    void **out_pipeline) {

    if (be == nullptr || be->state == nullptr || library == nullptr ||
        ns_string == nullptr || name == nullptr || out_function == nullptr ||
        out_pipeline == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct metal_state *st = be->state;
    void *fn_name = metal_msg_send_id_cstr(
        st, ns_string, "stringWithUTF8String:", name);
    if (fn_name == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal: failed to create %s shader name",
                                name);
        return GEIST_E_BACKEND;
    }
    *out_function = metal_msg_send_id_id(
        st, library, "newFunctionWithName:", fn_name);
    if (*out_function == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal: %s shader function missing", name);
        return GEIST_E_BACKEND;
    }

    void *err = nullptr;
    *out_pipeline = metal_msg_send_id_id_err(
        st, st->device, "newComputePipelineStateWithFunction:error:",
        *out_function, &err);
    if (*out_pipeline == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal: %s pipeline creation failed%s%s",
                                name, msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status metal_ensure_q4k_pipeline(
    struct geist_backend *be) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct metal_state *st = be->state;
    if (st->q4k_pipeline != nullptr &&
        st->q4k_n4_pipeline != nullptr &&
        st->q4k_nt4_pipeline != nullptr &&
        st->q4k_nt8_pipeline != nullptr &&
        st->q4k_ple_gate_nt8_pipeline != nullptr &&
        st->q4k_quant_x_pipeline != nullptr &&
        st->q4k_w4a8_pipeline != nullptr &&
        st->q4k_gate_up_w4a8_pipeline != nullptr &&
        st->q4k_matmul_m8_pipeline != nullptr &&
        st->q4k_matmul_m16_pipeline != nullptr &&
        st->q4k_matmul_m16_n2_pipeline != nullptr &&
        (!st->use_q4k_mm_sg || st->q4k_mm_sg_pipeline != nullptr) &&
        st->q6k_pipeline != nullptr &&
        st->q6k_n4_pipeline != nullptr &&
        st->q6k_nt4_pipeline != nullptr &&
        st->q6k_nt8_pipeline != nullptr &&
        st->q6k_matmul_m8_pipeline != nullptr &&
        st->q6k_matmul_m16_pipeline != nullptr &&
        st->rmsnorm_rows_pipeline != nullptr &&
        st->rmsnorm_rows_simd_pipeline != nullptr &&
        st->gelu_rows_pipeline != nullptr &&
        st->mul_rows_pipeline != nullptr &&
        st->gelu_mul_rows_pipeline != nullptr &&
        st->add_rows_pipeline != nullptr &&
        st->scale_rows_pipeline != nullptr &&
        st->rmsnorm_add_rows_pipeline != nullptr &&
        st->rmsnorm_add_rows_simd_pipeline != nullptr &&
        st->embed_lookup_scaled_pipeline != nullptr &&
        st->f32_matmul_pipeline != nullptr &&
        st->f32_ple_gate_pipeline != nullptr &&
        st->f32_ple_proj_norm_pipeline != nullptr) {
        return GEIST_OK;
    }

    void *ns_string = metal_objc_get_class(st, "NSString");
    if (ns_string == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal: NSString class unavailable");
        return GEIST_E_BACKEND;
    }
    void *source = metal_msg_send_id_cstr(
        st, ns_string, "stringWithUTF8String:", metal_q4k_source);
    void *q4k_n4_source = metal_msg_send_id_cstr(
        st, ns_string, "stringWithUTF8String:", metal_q4k_n4_source);
    void *q4k_m16_source = metal_msg_send_id_cstr(
        st, ns_string, "stringWithUTF8String:", metal_q4k_m16_source);
    void *q4k_m16_n2_source = metal_msg_send_id_cstr(
        st, ns_string, "stringWithUTF8String:", metal_q4k_m16_n2_source);
    void *q4k_mm_sg_ns_source =
        st->use_q4k_mm_sg
            ? metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:",
                                     metal_q4k_mm_sg_source)
            : nullptr;
    void *q4k_mm_sg_fast_ns_source =
        st->use_q4k_mm_sg
            ? metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:",
                                     metal_q4k_mm_sg_fast_source)
            : nullptr;
    void *q4k_nt4_source = metal_msg_send_id_cstr(
        st, ns_string, "stringWithUTF8String:", metal_q4k_nt4_source);
    void *q4k_ple_gate_nt8_source = metal_msg_send_id_cstr(
        st, ns_string, "stringWithUTF8String:",
        metal_q4k_ple_gate_nt8_source);
    void *q4k_w4a8_source = metal_msg_send_id_cstr(
        st, ns_string, "stringWithUTF8String:", metal_q4k_w4a8_source);
    void *q4k_gate_up_w4a8_source = metal_msg_send_id_cstr(
        st, ns_string, "stringWithUTF8String:",
        metal_q4k_gate_up_w4a8_source);
    void *q6_source = metal_msg_send_id_cstr(
        st, ns_string, "stringWithUTF8String:", metal_q6k_source);
    void *q6_mm_sg_source = metal_msg_send_id_cstr(
        st, ns_string, "stringWithUTF8String:", metal_q6k_mm_sg_source);
    void *q6_mm_sg_fast_source = metal_msg_send_id_cstr(
        st, ns_string, "stringWithUTF8String:", metal_q6k_mm_sg_fast_source);
    void *q6_n4_source = metal_msg_send_id_cstr(
        st, ns_string, "stringWithUTF8String:", metal_q6k_n4_source);
    void *q6_m16_source = metal_msg_send_id_cstr(
        st, ns_string, "stringWithUTF8String:", metal_q6k_m16_source);
    void *q6_nt4_source = metal_msg_send_id_cstr(
        st, ns_string, "stringWithUTF8String:", metal_q6k_nt4_source);
    void *q6_nt8_source = metal_msg_send_id_cstr(
        st, ns_string, "stringWithUTF8String:", metal_q6k_nt8_source);
    void *elem_source = metal_msg_send_id_cstr(
        st, ns_string, "stringWithUTF8String:", metal_elem_source);
    void *elem_simd_source = metal_msg_send_id_cstr(
        st, ns_string, "stringWithUTF8String:", metal_elem_simd_source);
    void *embed_source = metal_msg_send_id_cstr(
        st, ns_string, "stringWithUTF8String:", metal_embed_source);
    void *f32_source = nullptr;
    {
        /* two literals concatenated at init (C99 4095-char literal limit) */
        const size_t f32_len_a = strlen(metal_f32_source);
        const size_t f32_len_b = strlen(metal_f32_mm_source);
        char *f32_src = malloc(f32_len_a + f32_len_b + 1u);
        if (f32_src == nullptr) {
            geist_backend_set_error(be, GEIST_E_OOM,
                                    "metal: f32 shader source alloc failed");
            return GEIST_E_OOM;
        }
        memcpy(f32_src, metal_f32_source, f32_len_a);
        memcpy(f32_src + f32_len_a, metal_f32_mm_source, f32_len_b + 1u);
        f32_source = metal_msg_send_id_cstr(
            st, ns_string, "stringWithUTF8String:", f32_src);
        free(f32_src);
    }
    if (source == nullptr || q4k_n4_source == nullptr ||
        q4k_m16_source == nullptr ||
        q4k_m16_n2_source == nullptr ||
        (st->use_q4k_mm_sg && q4k_mm_sg_ns_source == nullptr) ||
        q4k_nt4_source == nullptr ||
        q4k_ple_gate_nt8_source == nullptr ||
        q4k_w4a8_source == nullptr ||
        q4k_gate_up_w4a8_source == nullptr ||
        q6_source == nullptr || q6_n4_source == nullptr ||
        q6_m16_source == nullptr ||
        q6_nt4_source == nullptr ||
        q6_nt8_source == nullptr ||
        elem_source == nullptr ||
        elem_simd_source == nullptr ||
        embed_source == nullptr || f32_source == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal: failed to create shader source");
        return GEIST_E_BACKEND;
    }

    void *err = nullptr;
    st->q4k_library = metal_msg_send_id_id_id_err(
        st, st->device, "newLibraryWithSource:options:error:",
        source, nullptr, &err);
    if (st->q4k_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal: Q4_K shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err = nullptr;
    st->q4k_n4_library = metal_msg_send_id_id_id_err(
        st, st->device, "newLibraryWithSource:options:error:",
        q4k_n4_source, nullptr, &err);
    if (st->q4k_n4_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal: Q4_K n4 shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err = nullptr;
    st->q4k_nt4_library = metal_msg_send_id_id_id_err(
        st, st->device, "newLibraryWithSource:options:error:",
        q4k_nt4_source, nullptr, &err);
    if (st->q4k_nt4_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal: Q4_K nt4 shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err = nullptr;
    st->q4k_ple_gate_nt8_library = metal_msg_send_id_id_id_err(
        st, st->device, "newLibraryWithSource:options:error:",
        q4k_ple_gate_nt8_source, nullptr, &err);
    if (st->q4k_ple_gate_nt8_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "metal: Q4_K PLE gate nt8 shader compile failed%s%s",
            msg != nullptr ? ": " : "",
            msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err = nullptr;
    st->q4k_w4a8_library = metal_msg_send_id_id_id_err(
        st, st->device, "newLibraryWithSource:options:error:",
        q4k_w4a8_source, nullptr, &err);
    if (st->q4k_w4a8_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal: Q4_K W4A8 shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err = nullptr;
    st->q4k_gate_up_w4a8_library = metal_msg_send_id_id_id_err(
        st, st->device, "newLibraryWithSource:options:error:",
        q4k_gate_up_w4a8_source, nullptr, &err);
    if (st->q4k_gate_up_w4a8_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "metal: Q4_K gate/up W4A8 shader compile failed%s%s",
            msg != nullptr ? ": " : "",
            msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err = nullptr;
    st->q4k_m16_library = metal_msg_send_id_id_id_err(
        st, st->device, "newLibraryWithSource:options:error:",
        q4k_m16_source, nullptr, &err);
    if (st->q4k_m16_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal: Q4_K m16 shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err = nullptr;
    st->q4k_m16_n2_library = metal_msg_send_id_id_id_err(
        st, st->device, "newLibraryWithSource:options:error:",
        q4k_m16_n2_source, nullptr, &err);
    if (st->q4k_m16_n2_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal: Q4_K m16 n2 shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    if (st->use_q4k_mm_sg) {
        err = nullptr;
        st->q4k_mm_sg_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:",
            q4k_mm_sg_ns_source, nullptr, &err);
        if (st->q4k_mm_sg_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(
                be, GEIST_E_BACKEND,
                "metal: Q4_K simdgroup mm shader compile failed%s%s",
                msg != nullptr ? ": " : "",
                msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
        err = nullptr;
        st->q4k_mm_sg_fast_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:",
            q4k_mm_sg_fast_ns_source, nullptr, &err);
        if (st->q4k_mm_sg_fast_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(
                be, GEIST_E_BACKEND,
                "metal: Q4_K simdgroup mm fast shader compile failed%s%s",
                msg != nullptr ? ": " : "",
                msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
    }
    err = nullptr;
    st->q6k_library = metal_msg_send_id_id_id_err(
        st, st->device, "newLibraryWithSource:options:error:",
        q6_source, nullptr, &err);
    if (st->q6k_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal: Q6_K shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err = nullptr;
    st->q6k_mm_sg_library = metal_msg_send_id_id_id_err(
        st, st->device, "newLibraryWithSource:options:error:",
        q6_mm_sg_source, nullptr, &err);
    if (st->q6k_mm_sg_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal: Q6_K mm_sg shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err = nullptr;
    st->q6k_mm_sg_fast_library = metal_msg_send_id_id_id_err(
        st, st->device, "newLibraryWithSource:options:error:",
        q6_mm_sg_fast_source, nullptr, &err);
    if (st->q6k_mm_sg_fast_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal: Q6_K mm_sg fast shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err = nullptr;
    st->q6k_m16_library = metal_msg_send_id_id_id_err(
        st, st->device, "newLibraryWithSource:options:error:",
        q6_m16_source, nullptr, &err);
    if (st->q6k_m16_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal: Q6_K m16 shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err = nullptr;
    st->q6k_n4_library = metal_msg_send_id_id_id_err(
        st, st->device, "newLibraryWithSource:options:error:",
        q6_n4_source, nullptr, &err);
    if (st->q6k_n4_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal: Q6_K n4 shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err = nullptr;
    st->q6k_nt4_library = metal_msg_send_id_id_id_err(
        st, st->device, "newLibraryWithSource:options:error:",
        q6_nt4_source, nullptr, &err);
    if (st->q6k_nt4_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal: Q6_K nt4 shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err = nullptr;
    st->q6k_nt8_library = metal_msg_send_id_id_id_err(
        st, st->device, "newLibraryWithSource:options:error:",
        q6_nt8_source, nullptr, &err);
    if (st->q6k_nt8_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal: Q6_K nt8 shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err = nullptr;
    st->elem_library = metal_msg_send_id_id_id_err(
        st, st->device, "newLibraryWithSource:options:error:",
        elem_source, nullptr, &err);
    if (st->elem_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal: elementwise shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err = nullptr;
    st->elem_simd_library = metal_msg_send_id_id_id_err(
        st, st->device, "newLibraryWithSource:options:error:",
        elem_simd_source, nullptr, &err);
    if (st->elem_simd_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal: SIMD elementwise shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err = nullptr;
    st->embed_library = metal_msg_send_id_id_id_err(
        st, st->device, "newLibraryWithSource:options:error:",
        embed_source, nullptr, &err);
    if (st->embed_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal: embedding shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err = nullptr;
    st->f32_library = metal_msg_send_id_id_id_err(
        st, st->device, "newLibraryWithSource:options:error:",
        f32_source, nullptr, &err);
    if (st->f32_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal: F32 shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    enum geist_status s = metal_create_named_pipeline(
        be, st->q4k_library, ns_string, "matvec_q4k", &st->q4k_function,
        &st->q4k_pipeline);
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->q4k_n4_library, ns_string, "matvec_q4k_n4",
            &st->q4k_n4_function, &st->q4k_n4_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->q4k_nt4_library, ns_string, "matvec_q4k_nt4",
            &st->q4k_nt4_function, &st->q4k_nt4_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->q4k_nt4_library, ns_string, "matvec_q4k_nt8",
            &st->q4k_nt8_function, &st->q4k_nt8_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->q4k_ple_gate_nt8_library, ns_string,
            "ple_gate_q4k_nt8",
            &st->q4k_ple_gate_nt8_function,
            &st->q4k_ple_gate_nt8_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->q4k_w4a8_library, ns_string, "q4k_quant_x",
            &st->q4k_quant_x_function, &st->q4k_quant_x_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->q4k_w4a8_library, ns_string,
            "matvec_q4k_nt4_w4a8",
            &st->q4k_w4a8_function, &st->q4k_w4a8_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->q4k_gate_up_w4a8_library, ns_string,
            "gate_up_q4k_nt4_w4a8",
            &st->q4k_gate_up_w4a8_function,
            &st->q4k_gate_up_w4a8_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->q4k_library, ns_string, "matmul_q4k_m8",
            &st->q4k_matmul_m8_function, &st->q4k_matmul_m8_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->q4k_m16_library, ns_string, "matmul_q4k_m16",
            &st->q4k_matmul_m16_function, &st->q4k_matmul_m16_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->q4k_m16_n2_library, ns_string, "matmul_q4k_m16_n2",
            &st->q4k_matmul_m16_n2_function,
            &st->q4k_matmul_m16_n2_pipeline);
    }
    if (s == GEIST_OK && st->use_q4k_mm_sg) {
        s = metal_create_named_pipeline(
            be, st->q4k_mm_sg_library, ns_string, "matmul_q4k_mm_sg",
            &st->q4k_mm_sg_function, &st->q4k_mm_sg_pipeline);
    }
    if (s == GEIST_OK && st->use_q4k_mm_sg) {
        s = metal_create_named_pipeline(
            be, st->q4k_mm_sg_fast_library, ns_string, "matmul_q4k_mm_sg_fast",
            &st->q4k_mm_sg_fast_function, &st->q4k_mm_sg_fast_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->q6k_library, ns_string, "matvec_q6k",
            &st->q6k_function, &st->q6k_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->q6k_mm_sg_library, ns_string, "matmul_q6k_sg",
            &st->q6k_matmul_sg_function, &st->q6k_matmul_sg_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->q6k_mm_sg_fast_library, ns_string, "matmul_q6k_sg_fast",
            &st->q6k_matmul_sg_fast_function,
            &st->q6k_matmul_sg_fast_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->q6k_n4_library, ns_string, "matvec_q6k_n4",
            &st->q6k_n4_function, &st->q6k_n4_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->q6k_nt4_library, ns_string, "matvec_q6k_nt4",
            &st->q6k_nt4_function, &st->q6k_nt4_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->q6k_nt8_library, ns_string, "matvec_q6k_nt8",
            &st->q6k_nt8_function, &st->q6k_nt8_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->q6k_library, ns_string, "matmul_q6k_m8",
            &st->q6k_matmul_m8_function, &st->q6k_matmul_m8_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->q6k_m16_library, ns_string, "matmul_q6k_m16",
            &st->q6k_matmul_m16_function, &st->q6k_matmul_m16_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->elem_library, ns_string, "rmsnorm_rows",
            &st->rmsnorm_rows_function, &st->rmsnorm_rows_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->elem_simd_library, ns_string, "rmsnorm_rows_simd",
            &st->rmsnorm_rows_simd_function,
            &st->rmsnorm_rows_simd_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->elem_library, ns_string, "gelu_rows",
            &st->gelu_rows_function, &st->gelu_rows_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->elem_library, ns_string, "mul_rows",
            &st->mul_rows_function, &st->mul_rows_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->elem_library, ns_string, "gelu_mul_rows",
            &st->gelu_mul_rows_function, &st->gelu_mul_rows_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->elem_library, ns_string, "add_rows",
            &st->add_rows_function, &st->add_rows_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->elem_library, ns_string, "scale_rows",
            &st->scale_rows_function, &st->scale_rows_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->elem_library, ns_string, "rmsnorm_add_rows",
            &st->rmsnorm_add_rows_function,
            &st->rmsnorm_add_rows_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->elem_simd_library, ns_string,
            "rmsnorm_add_rows_simd",
            &st->rmsnorm_add_rows_simd_function,
            &st->rmsnorm_add_rows_simd_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->embed_library, ns_string, "embed_lookup_scaled",
            &st->embed_lookup_scaled_function,
            &st->embed_lookup_scaled_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->f32_library, ns_string, "matmul_f32",
            &st->f32_matmul_function, &st->f32_matmul_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->f32_library, ns_string, "matmul_f32_sg",
            &st->f32_matmul_sg_function, &st->f32_matmul_sg_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->f32_library, ns_string, "matmul_f32_mm_sg",
            &st->f32_matmul_mm_function, &st->f32_matmul_mm_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->f32_library, ns_string, "ple_gate_f32",
            &st->f32_ple_gate_function, &st->f32_ple_gate_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->f32_library, ns_string, "ple_proj_norm_f32",
            &st->f32_ple_proj_norm_function,
            &st->f32_ple_proj_norm_pipeline);
    }
    return s;
}

[[nodiscard]] static enum geist_status metal_ensure_attention_pipeline(
    struct geist_backend *be) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct metal_state *st = be->state;
    if (st->q_norm_rope_rows_pipeline != nullptr &&
        st->k_norm_rope_append_rows_pipeline != nullptr &&
        st->k_norm_rope_append_rows_f16_pipeline != nullptr &&
        st->v_norm_append_rows_pipeline != nullptr &&
        st->v_norm_append_rows_f16_pipeline != nullptr &&
        st->kv_norm_append_rows_pipeline != nullptr &&
        st->kv_norm_append_rows_f16_pipeline != nullptr &&
        st->rope_rows_pipeline != nullptr &&
        st->kv_append_rows_pipeline != nullptr &&
        st->kv_append_rows_f16_pipeline != nullptr &&
        st->attention_rows_pipeline != nullptr &&
        st->attention_rows_f16_pipeline != nullptr &&
        st->attention_qnorm_rows_pipeline != nullptr &&
        st->attention_qnorm_rows_f16_pipeline != nullptr) {
        return GEIST_OK;
    }

    void *ns_string = metal_objc_get_class(st, "NSString");
    if (ns_string == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal: NSString class unavailable");
        return GEIST_E_BACKEND;
    }
    void *source = metal_msg_send_id_cstr(
        st, ns_string, "stringWithUTF8String:", metal_q_norm_rope_source);
    if (source == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal: failed to create Q norm/RoPE shader source");
        return GEIST_E_BACKEND;
    }
    void *err = nullptr;
    st->q_norm_rope_library = metal_msg_send_id_id_id_err(
        st, st->device, "newLibraryWithSource:options:error:",
        source, nullptr, &err);
    if (st->q_norm_rope_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal: Q norm/RoPE shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    enum geist_status s = metal_create_named_pipeline(
        be, st->q_norm_rope_library, ns_string, "q_norm_rope_rows",
        &st->q_norm_rope_rows_function,
        &st->q_norm_rope_rows_pipeline);

    if (s == GEIST_OK) {
        source = metal_msg_send_id_cstr(
            st, ns_string, "stringWithUTF8String:",
            metal_k_norm_rope_append_source);
        if (source == nullptr) {
            geist_backend_set_error(
                be, GEIST_E_BACKEND,
                "metal: failed to create K norm/RoPE append shader source");
            return GEIST_E_BACKEND;
        }
        err = nullptr;
        st->k_norm_rope_append_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:",
            source, nullptr, &err);
        if (st->k_norm_rope_append_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(
                be, GEIST_E_BACKEND,
                "metal: K norm/RoPE append shader compile failed%s%s",
                msg != nullptr ? ": " : "",
                msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
        s = metal_create_named_pipeline(
            be, st->k_norm_rope_append_library, ns_string,
            "k_norm_rope_append_rows",
            &st->k_norm_rope_append_rows_function,
            &st->k_norm_rope_append_rows_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->k_norm_rope_append_library, ns_string,
            "k_norm_rope_append_rows_f16",
            &st->k_norm_rope_append_rows_f16_function,
            &st->k_norm_rope_append_rows_f16_pipeline);
    }
    if (s == GEIST_OK) {
        source = metal_msg_send_id_cstr(
            st, ns_string, "stringWithUTF8String:",
            metal_v_norm_append_source);
        if (source == nullptr) {
            geist_backend_set_error(
                be, GEIST_E_BACKEND,
                "metal: failed to create V norm append shader source");
            return GEIST_E_BACKEND;
        }
        err = nullptr;
        st->v_norm_append_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:",
            source, nullptr, &err);
        if (st->v_norm_append_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(
                be, GEIST_E_BACKEND,
                "metal: V norm append shader compile failed%s%s",
                msg != nullptr ? ": " : "",
                msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
        s = metal_create_named_pipeline(
            be, st->v_norm_append_library, ns_string, "v_norm_append_rows",
            &st->v_norm_append_rows_function,
            &st->v_norm_append_rows_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->v_norm_append_library, ns_string,
            "v_norm_append_rows_f16",
            &st->v_norm_append_rows_f16_function,
            &st->v_norm_append_rows_f16_pipeline);
    }
    if (s == GEIST_OK) {
        source = metal_msg_send_id_cstr(
            st, ns_string, "stringWithUTF8String:",
            metal_kv_norm_append_source);
        if (source == nullptr) {
            geist_backend_set_error(
                be, GEIST_E_BACKEND,
                "metal: failed to create K/V norm append shader source");
            return GEIST_E_BACKEND;
        }
        err = nullptr;
        st->kv_norm_append_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:",
            source, nullptr, &err);
        if (st->kv_norm_append_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(
                be, GEIST_E_BACKEND,
                "metal: K/V norm append shader compile failed%s%s",
                msg != nullptr ? ": " : "",
                msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
        s = metal_create_named_pipeline(
            be, st->kv_norm_append_library, ns_string,
            "kv_norm_append_rows",
            &st->kv_norm_append_rows_function,
            &st->kv_norm_append_rows_pipeline);
    }
    if (s == GEIST_OK) {
        source = metal_msg_send_id_cstr(
            st, ns_string, "stringWithUTF8String:",
            metal_kv_norm_append_f16_source);
        if (source == nullptr) {
            geist_backend_set_error(
                be, GEIST_E_BACKEND,
                "metal: failed to create F16 K/V norm append shader source");
            return GEIST_E_BACKEND;
        }
        err = nullptr;
        st->kv_norm_append_f16_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:",
            source, nullptr, &err);
        if (st->kv_norm_append_f16_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(
                be, GEIST_E_BACKEND,
                "metal: F16 K/V norm append shader compile failed%s%s",
                msg != nullptr ? ": " : "",
                msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
        s = metal_create_named_pipeline(
            be, st->kv_norm_append_f16_library, ns_string,
            "kv_norm_append_rows_f16",
            &st->kv_norm_append_rows_f16_function,
            &st->kv_norm_append_rows_f16_pipeline);
    }
    if (s == GEIST_OK) {
        source = metal_msg_send_id_cstr(
            st, ns_string, "stringWithUTF8String:", metal_attn_source);
        if (source == nullptr) {
            geist_backend_set_error(
                be, GEIST_E_BACKEND,
                "metal: failed to create attention shader source");
            return GEIST_E_BACKEND;
        }
        err = nullptr;
        st->attn_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:",
            source, nullptr, &err);
        if (st->attn_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(
                be, GEIST_E_BACKEND,
                "metal: attention shader compile failed%s%s",
                msg != nullptr ? ": " : "",
                msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->attn_library, ns_string, "rope_rows",
            &st->rope_rows_function, &st->rope_rows_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->attn_library, ns_string, "kv_append_rows",
            &st->kv_append_rows_function, &st->kv_append_rows_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->attn_library, ns_string, "copy_u32",
            &st->copy_u32_function, &st->copy_u32_pipeline);
    }
    if (s == GEIST_OK) {
        source = metal_msg_send_id_cstr(
            st, ns_string, "stringWithUTF8String:", metal_attn_f16_source);
        if (source == nullptr) {
            geist_backend_set_error(
                be, GEIST_E_BACKEND,
                "metal: failed to create F16 attention shader source");
            return GEIST_E_BACKEND;
        }
        err = nullptr;
        st->attn_f16_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:",
            source, nullptr, &err);
        if (st->attn_f16_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(
                be, GEIST_E_BACKEND,
                "metal: F16 attention shader compile failed%s%s",
                msg != nullptr ? ": " : "",
                msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->attn_f16_library, ns_string, "kv_append_rows_f16",
            &st->kv_append_rows_f16_function,
            &st->kv_append_rows_f16_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->attn_library, ns_string, "attention_rows",
            &st->attention_rows_function, &st->attention_rows_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
            be, st->attn_f16_library, ns_string, "attention_rows_f16",
            &st->attention_rows_f16_function,
            &st->attention_rows_f16_pipeline);
    }
    if (s == GEIST_OK) {
        source = metal_msg_send_id_cstr(
            st, ns_string, "stringWithUTF8String:", metal_attn_qnorm_source);
        if (source == nullptr) {
            geist_backend_set_error(
                be, GEIST_E_BACKEND,
                "metal: failed to create Q-norm attention shader source");
            return GEIST_E_BACKEND;
        }
        err = nullptr;
        st->attn_qnorm_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:",
            source, nullptr, &err);
        if (st->attn_qnorm_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(
                be, GEIST_E_BACKEND,
                "metal: Q-norm attention shader compile failed%s%s",
                msg != nullptr ? ": " : "",
                msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
        s = metal_create_named_pipeline(
            be, st->attn_qnorm_library, ns_string, "attention_qnorm_rows",
            &st->attention_qnorm_rows_function,
            &st->attention_qnorm_rows_pipeline);
    }
    if (s == GEIST_OK) {
        source = metal_msg_send_id_cstr(
            st, ns_string, "stringWithUTF8String:",
            metal_attn_qnorm_f16_source);
        if (source == nullptr) {
            geist_backend_set_error(
                be, GEIST_E_BACKEND,
                "metal: failed to create F16 Q-norm attention shader source");
            return GEIST_E_BACKEND;
        }
        err = nullptr;
        st->attn_qnorm_f16_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:",
            source, nullptr, &err);
        if (st->attn_qnorm_f16_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(
                be, GEIST_E_BACKEND,
                "metal: F16 Q-norm attention shader compile failed%s%s",
                msg != nullptr ? ": " : "",
                msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
        s = metal_create_named_pipeline(
            be, st->attn_qnorm_f16_library, ns_string,
            "attention_qnorm_rows_f16",
            &st->attention_qnorm_rows_f16_function,
            &st->attention_qnorm_rows_f16_pipeline);
    }
    if (s == GEIST_OK) {
        source = metal_msg_send_id_cstr(
            st, ns_string, "stringWithUTF8String:",
            metal_attn_qnorm_dec_f16_source);
        if (source == nullptr) {
            geist_backend_set_error(
                be, GEIST_E_BACKEND,
                "metal: failed to create decode attention shader source");
            return GEIST_E_BACKEND;
        }
        err = nullptr;
        st->attn_qnorm_dec_f16_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:",
            source, nullptr, &err);
        if (st->attn_qnorm_dec_f16_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(
                be, GEIST_E_BACKEND,
                "metal: decode attention shader compile failed%s%s",
                msg != nullptr ? ": " : "",
                msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
        s = metal_create_named_pipeline(
            be, st->attn_qnorm_dec_f16_library, ns_string,
            "attention_qnorm_dec_f16",
            &st->attention_qnorm_dec_f16_function,
            &st->attention_qnorm_dec_f16_pipeline);
    }
    if (s == GEIST_OK) {
        source = metal_msg_send_id_cstr(
            st, ns_string, "stringWithUTF8String:",
            metal_attn_dec_combine_source);
        if (source == nullptr) {
            geist_backend_set_error(
                be, GEIST_E_BACKEND,
                "metal: failed to create attention combine shader source");
            return GEIST_E_BACKEND;
        }
        err = nullptr;
        st->attn_dec_combine_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:",
            source, nullptr, &err);
        if (st->attn_dec_combine_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(
                be, GEIST_E_BACKEND,
                "metal: attention combine shader compile failed%s%s",
                msg != nullptr ? ": " : "",
                msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
        s = metal_create_named_pipeline(
            be, st->attn_dec_combine_library, ns_string,
            "attention_dec_combine",
            &st->attention_dec_combine_function,
            &st->attention_dec_combine_pipeline);
    }
    if (s == GEIST_OK) {
        const size_t len_a = strlen(metal_attn_flash_sg_f16_source_a);
        const size_t len_b = strlen(metal_attn_flash_sg_f16_source_b);
        char *flash_src = malloc(len_a + len_b + 1u);
        if (flash_src == nullptr) {
            geist_backend_set_error(
                be, GEIST_E_OOM,
                "metal: flash attention shader source alloc failed");
            return GEIST_E_OOM;
        }
        memcpy(flash_src, metal_attn_flash_sg_f16_source_a, len_a);
        memcpy(flash_src + len_a, metal_attn_flash_sg_f16_source_b,
               len_b + 1u);
        source = metal_msg_send_id_cstr(
            st, ns_string, "stringWithUTF8String:", flash_src);
        free(flash_src);
        if (source == nullptr) {
            geist_backend_set_error(
                be, GEIST_E_BACKEND,
                "metal: failed to create flash attention shader source");
            return GEIST_E_BACKEND;
        }
        err = nullptr;
        st->attn_flash_sg_f16_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:",
            source, nullptr, &err);
        if (st->attn_flash_sg_f16_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(
                be, GEIST_E_BACKEND,
                "metal: flash attention shader compile failed%s%s",
                msg != nullptr ? ": " : "",
                msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
        s = metal_create_named_pipeline(
            be, st->attn_flash_sg_f16_library, ns_string,
            "attention_qnorm_flash_sg_f16",
            &st->attention_qnorm_flash_sg_f16_function,
            &st->attention_qnorm_flash_sg_f16_pipeline);
    }
    return s;
}

[[nodiscard]] static enum geist_status metal_ensure_q4k_w4a8_scratch(
    struct geist_backend *be,
    size_t n_in) {

    if (be == nullptr || be->state == nullptr || n_in == 0 ||
        n_in > SIZE_MAX - 3u) {
        return GEIST_E_INVALID_ARG;
    }
    struct metal_state *st = be->state;
    const size_t xq_bytes = ((n_in + 3u) / 4u) * sizeof(uint32_t);
    if (st->q4k_w4a8_xq_buffer != nullptr &&
        st->q4k_w4a8_scale_buffer != nullptr &&
        st->q4k_w4a8_xq_capacity >= xq_bytes) {
        return GEIST_OK;
    }

    if (st->q4k_w4a8_xq_buffer != nullptr &&
        st->q4k_w4a8_xq_capacity < xq_bytes) {
        metal_buffer_destroy_internal(be, st->q4k_w4a8_xq_buffer);
        st->q4k_w4a8_xq_buffer = nullptr;
        st->q4k_w4a8_xq_capacity = 0;
    }
    enum geist_status s = GEIST_OK;
    if (st->q4k_w4a8_xq_buffer == nullptr) {
        s = metal_new_buffer(be, xq_bytes, GEIST_BUFFER_SCRATCH,
                             GEIST_MEMORY_DEVICE, false,
                             &st->q4k_w4a8_xq_buffer);
        if (s != GEIST_OK) {
            return s;
        }
        st->q4k_w4a8_xq_capacity = xq_bytes;
    }
    if (st->q4k_w4a8_scale_buffer == nullptr) {
        s = metal_new_buffer(be, sizeof(float), GEIST_BUFFER_SCRATCH,
                             GEIST_MEMORY_DEVICE, false,
                             &st->q4k_w4a8_scale_buffer);
        if (s != GEIST_OK) {
            return s;
        }
    }
    return GEIST_OK;
}

static bool metal_q4k_w4a8_ready(const struct metal_state *st,
                                 const struct metal_q4k_params *params,
                                 const struct geist_buffer *packed) {
    if (st == nullptr || params == nullptr || packed == nullptr ||
        !st->use_q4k_w4a8 || params->rows != 1u ||
        st->q4k_quant_x_pipeline == nullptr ||
        st->q4k_w4a8_pipeline == nullptr ||
        st->q4k_w4a8_xq_buffer == nullptr ||
        st->q4k_w4a8_scale_buffer == nullptr) {
        return false;
    }
    const size_t xq_bytes = ((size_t) params->n_in + 3u) / 4u *
                            sizeof(uint32_t);
    return st->q4k_w4a8_xq_capacity >= xq_bytes;
}

static void metal_encode_q4k_w4a8_linear(
    struct metal_state *st,
    void *enc,
    const struct geist_tensor *x,
    const struct geist_buffer *packed,
    const struct geist_tensor *y,
    const struct metal_q4k_params *params) {

    const struct metal_q4k_quant_x_params quant_params = {
        .n_in = params->n_in,
        .x_offset = params->x_offset,
    };
    const struct metal_q4k_w4a8_params matvec_params = {
        .n_in = params->n_in,
        .n_out = params->n_out,
        .blocks_per_row = params->blocks_per_row,
        .w_byte_offset = 0u,
        .y_offset = params->y_offset,
    };

    (void) metal_msg_send_id_id(st, enc, "setComputePipelineState:",
                                st->q4k_quant_x_pipeline);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, 0, 0);
    metal_msg_send_set_buffer(st, enc, st->q4k_w4a8_xq_buffer->buffer, 0, 1);
    metal_msg_send_set_buffer(st, enc, st->q4k_w4a8_scale_buffer->buffer,
                              0, 2);
    metal_msg_send_set_bytes(st, enc, &quant_params, sizeof(quant_params),
                             3);
    const struct metal_size quant_groups = {.width = 1, .height = 1,
                                            .depth = 1};
    const struct metal_size quant_threads = {
        .width = METAL_Q4K_THREADS_PER_ROW,
        .height = 1,
        .depth = 1,
    };
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_Q4K_QUANT_X,
                               quant_groups);
    metal_msg_send_dispatch(st, enc, quant_groups, quant_threads);

    (void) metal_msg_send_id_id(st, enc, "setComputePipelineState:",
                                st->q4k_w4a8_pipeline);
    metal_msg_send_set_buffer(st, enc, st->q4k_w4a8_xq_buffer->buffer, 0, 0);
    metal_msg_send_set_buffer(st, enc, packed->buffer, 0, 1);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, 0, 2);
    metal_msg_send_set_buffer(st, enc, st->q4k_w4a8_scale_buffer->buffer,
                              0, 3);
    metal_msg_send_set_bytes(st, enc, &matvec_params, sizeof(matvec_params),
                             4);
    const struct metal_size groups = {
        .width = (params->n_out + 3u) / 4u,
        .height = 1,
        .depth = 1,
    };
    const struct metal_size threads = {
        .width = METAL_Q4K_N4_THREADS,
        .height = 1,
        .depth = 1,
    };
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_Q4K_LINEAR_W4A8,
                               groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

static void metal_encode_q4k_linear(struct metal_state *st,
                                    void *enc,
                                    const struct geist_tensor *x,
                                    const struct geist_tensor *w,
                                    const struct geist_tensor *y,
                                    const struct metal_q4k_params *params,
                                    bool m_tile8) {
    const bool m_tile16 = m_tile8 && params->rows >= METAL_Q4K_M16_TILE;
    const bool m_tile16_n2 =
        m_tile16 && st->use_q4k_m16_n2 &&
        st->q4k_matmul_m16_n2_pipeline != nullptr &&
        params->n_out >= 2u;
    const bool m_tile_sg =
        m_tile8 && st->use_q4k_mm_sg &&
        st->q4k_mm_sg_pipeline != nullptr &&
        params->rows >= 32u && params->n_out >= 64u &&
        (params->rows % 32u) == 0u && (params->n_out % 64u) == 0u;
    /* interior fast variant: no bounds checks, vectorized activation
     * staging (needs n_in%32 and 8-float-aligned x rows). */
    const bool m_tile_sg_fast =
        m_tile_sg && st->q4k_mm_sg_fast_pipeline != nullptr &&
        (params->n_in % 32u) == 0u && (params->x_offset % 8u) == 0u &&
        (params->x_row_stride % 8u) == 0u;
    const bool m_tile8_active = m_tile8 && !m_tile16;
    const bool n_tile4 = st->use_q4k_n4 && !m_tile8 && params->rows == 1u;
    struct geist_buffer *packed =
        st->use_q4k_nt4 && !st->use_q4k_linear_raw && n_tile4
            ? metal_q4k_nt4_cache_find_tensor(st, w)
            : nullptr;
    struct metal_q4k_params packed_params = *params;
    if (packed != nullptr) {
        packed_params.w_byte_offset = 0u;
    }
    if (metal_q4k_w4a8_ready(st, params, packed)) {
        metal_encode_q4k_w4a8_linear(st, enc, x, packed, y, params);
        return;
    }
    const bool n_tile8 =
        packed != nullptr && st->use_q4k_nt8 && params->n_out >= 8u;
    (void) metal_msg_send_id_id(st, enc, "setComputePipelineState:",
                                n_tile8 ? st->q4k_nt8_pipeline
                                : packed != nullptr ? st->q4k_nt4_pipeline
                                : m_tile_sg_fast ? st->q4k_mm_sg_fast_pipeline
                                : m_tile_sg ? st->q4k_mm_sg_pipeline
                                : m_tile16_n2 ? st->q4k_matmul_m16_n2_pipeline
                                : m_tile16 ? st->q4k_matmul_m16_pipeline
                                : m_tile8_active ? st->q4k_matmul_m8_pipeline
                                : n_tile4 ? st->q4k_n4_pipeline
                                          : st->q4k_pipeline);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, 0, 0);
    metal_msg_send_set_buffer(st, enc,
                              packed != nullptr ? packed->buffer
                                                : w->buffer->buffer,
                              0, 1);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, 0, 2);
    metal_msg_send_set_bytes(st, enc,
                             packed != nullptr ? &packed_params : params,
                             sizeof(*params), 3);
    if (m_tile_sg) {
        metal_msg_send_set_threadgroup_memory(
            st, enc, m_tile_sg_fast ? 6144u : 8192u, 0u);
    }

    const struct metal_size groups = {
        .width = n_tile8 ? (params->n_out + 7u) / 8u
                 : n_tile4 ? (params->n_out + 3u) / 4u
                 : m_tile_sg ? (params->rows + 31u) / 32u
                 : m_tile16_n2 ? (params->n_out + 1u) / 2u
                         : params->n_out,
        .height = m_tile_sg
                      ? (params->n_out + 63u) / 64u
                  : m_tile16
                      ? (params->rows + METAL_Q4K_M16_TILE - 1u) /
                            METAL_Q4K_M16_TILE
                  : m_tile8_active
                      ? (params->rows + METAL_Q4K_M_TILE - 1u) /
                            METAL_Q4K_M_TILE
                      : params->rows,
        .depth = 1,
    };
    const struct metal_size threads = {
        .width = m_tile_sg ? 32u
                 : (n_tile8 || n_tile4) ? METAL_Q4K_N4_THREADS
                         : METAL_Q4K_THREADS_PER_ROW,
        .height = m_tile_sg ? 4u : 1u,
        .depth = 1,
    };
    const enum metal_profile_stage profile_stage =
        n_tile8 ? METAL_PROFILE_DISPATCH_Q4K_LINEAR_NT8
        : packed != nullptr ? METAL_PROFILE_DISPATCH_Q4K_LINEAR_NT4
        : n_tile4 ? METAL_PROFILE_DISPATCH_Q4K_LINEAR_N4
                  : METAL_PROFILE_DISPATCH_Q4K_LINEAR_BASE;
    metal_profile_add_dispatch(st, profile_stage, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

static void metal_encode_q6k_linear(struct metal_state *st,
                                    void *enc,
                                    const struct geist_tensor *x,
                                    const struct geist_tensor *w,
                                    const struct geist_tensor *y,
                                    const struct metal_q4k_params *params,
                                    bool m_tile8) {
    /* rows==1 always takes the matvec kernel — the 64-row GEMM tile is a
     * waste for a single row, and n4 (llama mul_mv structure) wins at any
     * n_out that fills at least one threadgroup. */
    const bool single_row = params->rows == 1u;
    const bool n_tile4 = st->use_q6k_n4 && single_row && params->n_out >= 4u;
    const bool m_tile_sg = m_tile8 && !n_tile4 &&
                           st->q6k_matmul_sg_pipeline != nullptr;
    /* interior fast variant: no bounds checks, vectorized activation
     * staging, direct simdgroup_store output. */
    const bool m_tile_sg_fast =
        m_tile_sg && st->q6k_matmul_sg_fast_pipeline != nullptr &&
        (params->rows % 64u) == 0u && (params->n_out % 32u) == 0u &&
        (params->x_offset % 8u) == 0u && (params->x_row_stride % 8u) == 0u;
    const bool m_tile16 = m_tile8 && !n_tile4 && !m_tile_sg &&
                          params->rows >= METAL_Q4K_M16_TILE;
    const bool m_tile8_active =
        m_tile8 && !n_tile4 && !m_tile_sg && !m_tile16;
    struct geist_buffer *packed =
        st->use_q6k_nt4 && !st->use_q6k_linear_raw && n_tile4
            ? metal_q6k_nt4_cache_find_tensor(st, w)
            : nullptr;
    struct metal_q4k_params packed_params = *params;
    if (packed != nullptr) {
        packed_params.w_byte_offset = 0u;
    }
    const bool n_tile8 =
        packed != nullptr && st->use_q6k_nt8 && params->n_out >= 8u;
    (void) metal_msg_send_id_id(st, enc, "setComputePipelineState:",
                                n_tile8 ? st->q6k_nt8_pipeline
                                : packed != nullptr ? st->q6k_nt4_pipeline
                                : m_tile_sg_fast ? st->q6k_matmul_sg_fast_pipeline
                                : m_tile_sg ? st->q6k_matmul_sg_pipeline
                                : m_tile16 ? st->q6k_matmul_m16_pipeline
                                : m_tile8_active ? st->q6k_matmul_m8_pipeline
                                : n_tile4 ? st->q6k_n4_pipeline
                                        : st->q6k_pipeline);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, 0, 0);
    metal_msg_send_set_buffer(st, enc,
                              packed != nullptr ? packed->buffer
                                                : w->buffer->buffer,
                              0, 1);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, 0, 2);
    metal_msg_send_set_bytes(st, enc,
                             packed != nullptr ? &packed_params : params,
                             sizeof(*params), 3);

    /* The q6k sg kernels are 128-thread / 4-simdgroup with a 32-output x
     * 64-batch-row tile (b0=tg.y*64, o0=tg.x*32). The old (n_out+7)/8 x
     * 32-thread dispatch here only ever ran simdgroup 0 — a latent bug the
     * old engine never hit because its prefill routed q6k through the
     * fused blocks, not this vtbl op. */
    const struct metal_size groups = {
        .width = m_tile_sg ? (params->n_out + 31u) / 32u
                 : n_tile8 ? (params->n_out + 7u) / 8u
                 : n_tile4 ? (params->n_out + 3u) / 4u
                         : params->n_out,
        .height = m_tile_sg
                      ? (params->rows + 63u) / 64u
                  : m_tile16
                      ? (params->rows + METAL_Q4K_M16_TILE - 1u) /
                            METAL_Q4K_M16_TILE
                  : m_tile8_active
                      ? (params->rows + METAL_Q4K_M_TILE - 1u) /
                            METAL_Q4K_M_TILE
                      : params->rows,
        .depth = 1,
    };
    const struct metal_size threads = {
        .width = m_tile_sg ? 32u
                 : (n_tile8 || n_tile4) ? METAL_Q4K_N4_THREADS
                         : METAL_Q4K_THREADS_PER_ROW,
        .height = m_tile_sg ? 4u : 1,
        .depth = 1,
    };
    const enum metal_profile_stage profile_stage =
        n_tile8 ? METAL_PROFILE_DISPATCH_Q6K_LINEAR_NT8
        : packed != nullptr ? METAL_PROFILE_DISPATCH_Q6K_LINEAR_NT4
        : n_tile4 ? METAL_PROFILE_DISPATCH_Q6K_LINEAR_N4
                  : METAL_PROFILE_DISPATCH_Q6K_LINEAR_BASE;
    metal_profile_add_dispatch(st, profile_stage, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

static void metal_encode_rmsnorm_rows(
    struct metal_state *st,
    void *enc,
    const struct geist_tensor *x,
    const struct geist_tensor *w,
    const struct geist_tensor *y,
    const struct metal_rows_params *params) {

    (void) metal_msg_send_id_id(st, enc, "setComputePipelineState:",
                                st->use_rmsnorm_simd
                                    ? st->rmsnorm_rows_simd_pipeline
                                    : st->rmsnorm_rows_pipeline);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, 0, 0);
    metal_msg_send_set_buffer(st, enc, w->buffer->buffer, 0, 1);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, 0, 2);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 3);
    const struct metal_size groups = {
        .width = params->rows,
        .height = 1,
        .depth = 1,
    };
    const struct metal_size threads = {
        .width = METAL_ELEM_THREADS,
        .height = 1,
        .depth = 1,
    };
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_RMSNORM_ROWS,
                               groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

static void metal_encode_add_rows(struct metal_state *st,
                                  void *enc,
                                  const struct geist_tensor *a,
                                  const struct geist_tensor *b,
                                  const struct geist_tensor *y,
                                  const struct metal_binary_rows_params *params) {
    (void) metal_msg_send_id_id(st, enc, "setComputePipelineState:",
                                st->add_rows_pipeline);
    metal_msg_send_set_buffer(st, enc, a->buffer->buffer, 0, 0);
    metal_msg_send_set_buffer(st, enc, b->buffer->buffer, 0, 1);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, 0, 2);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 3);
    const struct metal_size groups = {
        .width = (params->rows * params->cols + METAL_ELEM_THREADS - 1u) /
                 METAL_ELEM_THREADS,
        .height = 1,
        .depth = 1,
    };
    const struct metal_size threads = {
        .width = METAL_ELEM_THREADS,
        .height = 1,
        .depth = 1,
    };
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_ATTENTION_ROWS,
                               groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

static void metal_encode_mul_rows(struct metal_state *st,
                                  void *enc,
                                  const struct geist_tensor *a,
                                  const struct geist_tensor *b,
                                  const struct geist_tensor *y,
                                  const struct metal_binary_rows_params *params) {
    (void) metal_msg_send_id_id(st, enc, "setComputePipelineState:",
                                st->mul_rows_pipeline);
    metal_msg_send_set_buffer(st, enc, a->buffer->buffer, 0, 0);
    metal_msg_send_set_buffer(st, enc, b->buffer->buffer, 0, 1);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, 0, 2);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 3);
    const struct metal_size groups = {
        .width = (params->rows * params->cols + METAL_ELEM_THREADS - 1u) /
                 METAL_ELEM_THREADS,
        .height = 1,
        .depth = 1,
    };
    const struct metal_size threads = {
        .width = METAL_ELEM_THREADS,
        .height = 1,
        .depth = 1,
    };
    metal_msg_send_dispatch(st, enc, groups, threads);
}

static void metal_encode_gelu_mul_rows(
    struct metal_state *st,
    void *enc,
    const struct geist_tensor *a,
    const struct geist_tensor *b,
    const struct geist_tensor *y,
    const struct metal_binary_rows_params *params) {

    (void) metal_msg_send_id_id(st, enc, "setComputePipelineState:",
                                st->gelu_mul_rows_pipeline);
    metal_msg_send_set_buffer(st, enc, a->buffer->buffer, 0, 0);
    metal_msg_send_set_buffer(st, enc, b->buffer->buffer, 0, 1);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, 0, 2);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 3);
    const struct metal_size groups = {
        .width = (params->rows * params->cols + METAL_ELEM_THREADS - 1u) /
                 METAL_ELEM_THREADS,
        .height = 1,
        .depth = 1,
    };
    const struct metal_size threads = {
        .width = METAL_ELEM_THREADS,
        .height = 1,
        .depth = 1,
    };
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_GELU_MUL_ROWS,
                               groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

static void metal_encode_scale_rows(struct metal_state *st,
                                    void *enc,
                                    const struct geist_tensor *x,
                                    const struct geist_tensor *y,
                                    const struct metal_scale_rows_params *params) {
    (void) metal_msg_send_id_id(st, enc, "setComputePipelineState:",
                                st->scale_rows_pipeline);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, 0, 0);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, 0, 1);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 2);
    const struct metal_size groups = {
        .width = (params->rows * params->cols + METAL_ELEM_THREADS - 1u) /
                 METAL_ELEM_THREADS,
        .height = 1,
        .depth = 1,
    };
    const struct metal_size threads = {
        .width = METAL_ELEM_THREADS,
        .height = 1,
        .depth = 1,
    };
    metal_msg_send_dispatch(st, enc, groups, threads);
}

static void metal_encode_gelu_rows(struct metal_state *st,
                                   void *enc,
                                   const struct geist_tensor *x,
                                   const struct geist_tensor *y,
                                   const struct metal_scale_rows_params *params) {
    (void) metal_msg_send_id_id(st, enc, "setComputePipelineState:",
                                st->gelu_rows_pipeline);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, 0, 0);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, 0, 1);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 2);
    const struct metal_size groups = {
        .width = (params->rows * params->cols + METAL_ELEM_THREADS - 1u) /
                 METAL_ELEM_THREADS,
        .height = 1,
        .depth = 1,
    };
    const struct metal_size threads = {
        .width = METAL_ELEM_THREADS,
        .height = 1,
        .depth = 1,
    };
    metal_msg_send_dispatch(st, enc, groups, threads);
}

static void metal_encode_embed_lookup_scaled(
    struct metal_state *st,
    void *enc,
    const struct geist_tensor *embed_table,
    const struct geist_tensor *out,
    const struct metal_embed_params *params) {

    (void) metal_msg_send_id_id(st, enc, "setComputePipelineState:",
                                st->embed_lookup_scaled_pipeline);
    metal_msg_send_set_buffer(st, enc, embed_table->buffer->buffer, 0, 0);
    metal_msg_send_set_buffer(st, enc, out->buffer->buffer, 0, 1);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 2);
    const struct metal_size groups = {
        .width = (params->n + METAL_ELEM_THREADS - 1u) / METAL_ELEM_THREADS,
        .height = 1,
        .depth = 1,
    };
    const struct metal_size threads = {
        .width = METAL_ELEM_THREADS,
        .height = 1,
        .depth = 1,
    };
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_EMBED, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

static void metal_encode_rope_rows(
    struct metal_state *st,
    void *enc,
    struct geist_tensor *x,
    const struct geist_tensor *cos,
    const struct geist_tensor *sin,
    const struct metal_rope_params *params) {

    (void) metal_msg_send_id_id(st, enc, "setComputePipelineState:",
                                st->rope_rows_pipeline);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, 0, 0);
    metal_msg_send_set_buffer(st, enc, cos->buffer->buffer, 0, 1);
    metal_msg_send_set_buffer(st, enc, sin->buffer->buffer, 0, 2);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 3);
    const size_t half = (size_t) params->head_dim / 2u;
    const size_t total =
        (size_t) params->rows * (size_t) params->heads * half;
    const struct metal_size groups = {
        .width = (total + METAL_ELEM_THREADS - 1u) / METAL_ELEM_THREADS,
        .height = 1,
        .depth = 1,
    };
    const struct metal_size threads = {
        .width = METAL_ELEM_THREADS,
        .height = 1,
        .depth = 1,
    };
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_ROPE_ROWS,
                               groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

static void metal_encode_attention_rows(
    struct metal_state *st,
    void *enc,
    const struct geist_tensor *q,
    const struct geist_tensor *k_cache,
    const struct geist_tensor *v_cache,
    const struct geist_tensor *y,
    const struct metal_attention_params *params) {

    void *pipeline = k_cache->dtype == GEIST_DTYPE_F16
                         ? st->attention_rows_f16_pipeline
                         : st->attention_rows_pipeline;
    (void) metal_msg_send_id_id(st, enc, "setComputePipelineState:",
                                pipeline);
    metal_msg_send_set_buffer(st, enc, q->buffer->buffer, 0, 0);
    metal_msg_send_set_buffer(st, enc, k_cache->buffer->buffer, 0, 1);
    metal_msg_send_set_buffer(st, enc, v_cache->buffer->buffer, 0, 2);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, 0, 3);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 4);
    const struct metal_size groups = {
        .width = params->rows,
        .height = params->q_heads,
        .depth = 1,
    };
    const struct metal_size threads = {
        .width = METAL_ELEM_THREADS,
        .height = 1,
        .depth = 1,
    };
    metal_msg_send_dispatch(st, enc, groups, threads);
}

static void metal_encode_f32_matmul(struct metal_state *st,
                                    void *enc,
                                    const struct geist_tensor *x,
                                    const struct geist_tensor *w,
                                    const struct geist_tensor *y,
                                    const struct metal_f32_params *params) {
    /* Multi-row (prefill): full-tile shapes take the 64x32 4-simdgroup GEMM
     * (mm_sg structure, f32 staging = bit-identical to the 8x8 kernel);
     * others the 8x8 simdgroup GEMM. Single-row keeps the reduction kernel. */
    const bool use_sg = params->rows > 1u && st->f32_matmul_sg_pipeline != nullptr;
    const bool use_mm =
        use_sg && st->f32_matmul_mm_pipeline != nullptr &&
        !metal_env_disabled("GEIST_METAL_F32_MM") &&
        (params->rows % 32u) == 0u && (params->n_out % 64u) == 0u &&
        (params->n_in % 32u) == 0u &&
        (params->x_offset % 8u) == 0u && (params->x_row_stride % 8u) == 0u &&
        (params->w_offset % 4u) == 0u;
    (void) metal_msg_send_id_id(st, enc, "setComputePipelineState:",
                                use_mm ? st->f32_matmul_mm_pipeline
                                : use_sg ? st->f32_matmul_sg_pipeline
                                         : st->f32_matmul_pipeline);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, 0, 0);
    metal_msg_send_set_buffer(st, enc, w->buffer->buffer, 0, 1);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, 0, 2);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 3);
    if (use_mm) {
        /* sa 64x32 f32 + sb 32x32 f32 */
        metal_msg_send_set_threadgroup_memory(st, enc, 12288u, 0u);
    }
    const struct metal_size groups = {
        .width = use_mm ? params->rows / 32u
                 : use_sg ? (params->n_out + 7u) / 8u
                          : params->n_out,
        .height = use_mm ? params->n_out / 64u
                  : use_sg ? (params->rows + 7u) / 8u
                           : params->rows,
        .depth = 1,
    };
    const struct metal_size threads = {
        .width = use_sg ? 32u : METAL_ELEM_THREADS,
        .height = use_mm ? 4u : 1,
        .depth = 1,
    };
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_F32_MATMUL, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

[[nodiscard]] static enum geist_status metal_f32_linear(
    struct geist_backend *be,
    const struct geist_tensor *x,
    const struct geist_tensor *w,
    struct geist_tensor *y,
    bool matrix) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }

    size_t rows = 1;
    size_t n_in = 0;
    size_t x_offset = 0;
    size_t x_row_stride = 0;
    size_t y_rows = 1;
    size_t y_cols = 0;
    size_t y_offset = 0;
    size_t y_row_stride = 0;
    size_t w_rows = 0;
    size_t w_cols = 0;
    size_t w_offset = 0;
    size_t w_row_stride = 0;

    bool ok = false;
    if (matrix) {
        ok = metal_tensor_is_f32_matrix(x, &rows, &n_in, &x_offset,
                                        &x_row_stride) &&
             metal_tensor_is_f32_matrix(y, &y_rows, &y_cols, &y_offset,
                                        &y_row_stride);
    } else {
        ok = metal_tensor_is_f32_vector(x, &n_in, &x_offset) &&
             metal_tensor_is_f32_vector(y, &y_cols, &y_offset);
        x_row_stride = n_in;
        y_row_stride = y_cols;
    }
    ok = ok && metal_tensor_is_f32_matrix(w, &w_rows, &w_cols, &w_offset,
                                          &w_row_stride) &&
         w_row_stride == w_cols &&
         w_cols == n_in &&
         y_cols == w_rows &&
         y_rows == rows;
    if (!ok) {
        geist_backend_set_error(
            be, GEIST_E_UNSUPPORTED,
            matrix
                ? "metal F32 matmul: expected x F32 [rows,n], w F32 [out,n], y F32 [rows,out]"
                : "metal F32 matvec: expected x F32 [n], w F32 [out,n], y F32 [out]");
        return GEIST_E_UNSUPPORTED;
    }
    if (rows > UINT32_MAX || n_in > UINT32_MAX || w_rows > UINT32_MAX ||
        x_offset > UINT32_MAX || w_offset > UINT32_MAX ||
        y_offset > UINT32_MAX || x_row_stride > UINT32_MAX ||
        y_row_stride > UINT32_MAX) {
        return GEIST_E_INVALID_ARG;
    }
    if (x->buffer->owner != be->state || w->buffer->owner != be->state ||
        y->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }

    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }

    struct metal_state *st = be->state;
    const struct metal_f32_params params = {
        .n_in = (uint32_t) n_in,
        .n_out = (uint32_t) w_rows,
        .rows = (uint32_t) rows,
        .x_offset = (uint32_t) x_offset,
        .w_offset = (uint32_t) w_offset,
        .y_offset = (uint32_t) y_offset,
        .x_row_stride = (uint32_t) x_row_stride,
        .y_row_stride = (uint32_t) y_row_stride,
    };

    if (st->sequence_active) {
        if (st->sequence_compute_encoder == nullptr) {
            geist_backend_set_error(
                be, GEIST_E_BACKEND,
                "metal F32 linear: command sequence has no encoder");
            return GEIST_E_BACKEND;
        }
        struct metal_decode_replay_op *op = metal_decode_replay_append(
            st, METAL_DECODE_REPLAY_F32_LINEAR);
        if (op != nullptr) {
            op->u.f32_linear = (struct metal_decode_replay_f32_linear){
                .x = *x,
                .w = *w,
                .y = *y,
                .matrix = matrix,
                .params = {
                    .n_in = params.n_in,
                    .n_out = params.n_out,
                    .rows = params.rows,
                    .x_offset = params.x_offset,
                    .w_offset = params.w_offset,
                    .y_offset = params.y_offset,
                    .x_row_stride = params.x_row_stride,
                    .y_row_stride = params.y_row_stride,
                },
            };
        }
        metal_encode_f32_matmul(st, st->sequence_compute_encoder,
                                x, w, y, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }

    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    if (cmd == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal F32 linear: command buffer failed");
        return GEIST_E_BACKEND;
    }
    void *enc = metal_msg_send_id0(st, cmd, "computeCommandEncoder");
    if (enc == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal F32 linear: encoder failed");
        return GEIST_E_BACKEND;
    }
    metal_encode_f32_matmul(st, enc, x, w, y, &params);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");

    void *err = metal_msg_send_id0(st, cmd, "error");
    if (err != nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal F32 linear: command failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status metal_matvec_f32_dense(
    struct geist_backend *be,
    const struct geist_tensor *x,
    const struct geist_tensor *w,
    struct geist_tensor *y) {

    return metal_f32_linear(be, x, w, y, false);
}

[[nodiscard]] static enum geist_status metal_matmul_f32_dense(
    struct geist_backend *be,
    const struct geist_tensor *x,
    const struct geist_tensor *w,
    struct geist_tensor *y) {

    return metal_f32_linear(be, x, w, y, true);
}

[[nodiscard]] static enum geist_status metal_rmsnorm(
    struct geist_backend *be,
    const struct geist_tensor *x,
    const struct geist_tensor *w,
    float eps,
    struct geist_tensor *y) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows = 0, cols = 0, x_off = 0, x_stride = 0;
    size_t y_rows = 0, y_cols = 0, y_off = 0, y_stride = 0;
    size_t w_n = 0, w_off = 0;
    if (!metal_tensor_is_f32_rows(x, &rows, &cols, &x_off, &x_stride) ||
        !metal_tensor_is_f32_rows(y, &y_rows, &y_cols, &y_off, &y_stride) ||
        !metal_tensor_is_f32_vector(w, &w_n, &w_off) ||
        y_rows != rows || y_cols != cols || w_n != cols) {
        return GEIST_E_UNSUPPORTED;
    }
    if (rows > UINT32_MAX || cols > UINT32_MAX || x_off > UINT32_MAX ||
        y_off > UINT32_MAX || w_off > UINT32_MAX ||
        x_stride > UINT32_MAX || y_stride > UINT32_MAX ||
        x->buffer->owner != be->state || w->buffer->owner != be->state ||
        y->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) { return s; }
    struct metal_state *st = be->state;
    const struct metal_rows_params params = {
        .rows = (uint32_t) rows,
        .cols = (uint32_t) cols,
        .x_offset = (uint32_t) x_off,
        .w_offset = (uint32_t) w_off,
        .y_offset = (uint32_t) y_off,
        .x_row_stride = (uint32_t) x_stride,
        .y_row_stride = (uint32_t) y_stride,
        .eps = eps,
    };
    if (st->sequence_active) {
        struct metal_decode_replay_op *op = metal_decode_replay_append(
            st, METAL_DECODE_REPLAY_RMSNORM);
        if (op != nullptr) {
            op->u.rmsnorm = (struct metal_decode_replay_rmsnorm){
                .x = *x,
                .w = *w,
                .y = *y,
                .params = {
                    .rows = params.rows,
                    .cols = params.cols,
                    .x_offset = params.x_offset,
                    .w_offset = params.w_offset,
                    .y_offset = params.y_offset,
                    .x_row_stride = params.x_row_stride,
                    .y_row_stride = params.y_row_stride,
                    .eps = params.eps,
                },
            };
        }
        metal_encode_rmsnorm_rows(st, st->sequence_compute_encoder, x, w, y,
                                  &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    void *enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder")
                               : nullptr;
    if (cmd == nullptr || enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_rmsnorm_rows(st, enc, x, w, y, &params);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    return metal_msg_send_id0(st, cmd, "error") == nullptr ? GEIST_OK
                                                           : GEIST_E_BACKEND;
}

[[nodiscard]] static enum geist_status metal_add(
    struct geist_backend *be,
    const struct geist_tensor *a,
    const struct geist_tensor *b,
    struct geist_tensor *y) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows = 0, cols = 0, a_off = 0, a_stride = 0;
    size_t b_rows = 0, b_cols = 0, b_off = 0, b_stride = 0;
    size_t y_rows = 0, y_cols = 0, y_off = 0, y_stride = 0;
    if (!metal_tensor_is_f32_rows(a, &rows, &cols, &a_off, &a_stride) ||
        !metal_tensor_is_f32_rows(b, &b_rows, &b_cols, &b_off, &b_stride) ||
        !metal_tensor_is_f32_rows(y, &y_rows, &y_cols, &y_off, &y_stride) ||
        b_rows != rows || y_rows != rows || b_cols != cols ||
        y_cols != cols) {
        return GEIST_E_UNSUPPORTED;
    }
    if (rows > UINT32_MAX || cols > UINT32_MAX || a_off > UINT32_MAX ||
        b_off > UINT32_MAX || y_off > UINT32_MAX ||
        a_stride > UINT32_MAX || b_stride > UINT32_MAX ||
        y_stride > UINT32_MAX ||
        a->buffer->owner != be->state || b->buffer->owner != be->state ||
        y->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) { return s; }
    struct metal_state *st = be->state;
    const struct metal_binary_rows_params params = {
        .rows = (uint32_t) rows,
        .cols = (uint32_t) cols,
        .a_offset = (uint32_t) a_off,
        .b_offset = (uint32_t) b_off,
        .y_offset = (uint32_t) y_off,
        .a_row_stride = (uint32_t) a_stride,
        .b_row_stride = (uint32_t) b_stride,
        .y_row_stride = (uint32_t) y_stride,
    };
    if (st->sequence_active) {
        struct metal_decode_replay_op *op = metal_decode_replay_append(
            st, METAL_DECODE_REPLAY_ADD);
        if (op != nullptr) {
            op->u.binary = (struct metal_decode_replay_binary){
                .a = *a,
                .b = *b,
                .y = *y,
                .params = {
                    .rows = params.rows,
                    .cols = params.cols,
                    .a_offset = params.a_offset,
                    .b_offset = params.b_offset,
                    .y_offset = params.y_offset,
                    .a_row_stride = params.a_row_stride,
                    .b_row_stride = params.b_row_stride,
                    .y_row_stride = params.y_row_stride,
                },
            };
        }
        metal_encode_add_rows(st, st->sequence_compute_encoder, a, b, y,
                              &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    void *enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder")
                               : nullptr;
    if (cmd == nullptr || enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_add_rows(st, enc, a, b, y, &params);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    return metal_msg_send_id0(st, cmd, "error") == nullptr ? GEIST_OK
                                                           : GEIST_E_BACKEND;
}

[[nodiscard]] static enum geist_status metal_mul(
    struct geist_backend *be,
    const struct geist_tensor *a,
    const struct geist_tensor *b,
    struct geist_tensor *y) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows = 0, cols = 0, a_off = 0, a_stride = 0;
    size_t b_rows = 0, b_cols = 0, b_off = 0, b_stride = 0;
    size_t y_rows = 0, y_cols = 0, y_off = 0, y_stride = 0;
    if (!metal_tensor_is_f32_rows(a, &rows, &cols, &a_off, &a_stride) ||
        !metal_tensor_is_f32_rows(b, &b_rows, &b_cols, &b_off, &b_stride) ||
        !metal_tensor_is_f32_rows(y, &y_rows, &y_cols, &y_off, &y_stride) ||
        b_rows != rows || y_rows != rows || b_cols != cols ||
        y_cols != cols) {
        return GEIST_E_UNSUPPORTED;
    }
    if (rows > UINT32_MAX || cols > UINT32_MAX || a_off > UINT32_MAX ||
        b_off > UINT32_MAX || y_off > UINT32_MAX ||
        a_stride > UINT32_MAX || b_stride > UINT32_MAX ||
        y_stride > UINT32_MAX ||
        a->buffer->owner != be->state || b->buffer->owner != be->state ||
        y->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) { return s; }
    struct metal_state *st = be->state;
    const struct metal_binary_rows_params params = {
        .rows = (uint32_t) rows,
        .cols = (uint32_t) cols,
        .a_offset = (uint32_t) a_off,
        .b_offset = (uint32_t) b_off,
        .y_offset = (uint32_t) y_off,
        .a_row_stride = (uint32_t) a_stride,
        .b_row_stride = (uint32_t) b_stride,
        .y_row_stride = (uint32_t) y_stride,
    };
    if (st->sequence_active) {
        metal_encode_mul_rows(st, st->sequence_compute_encoder, a, b, y,
                              &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    void *enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder")
                               : nullptr;
    if (cmd == nullptr || enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_mul_rows(st, enc, a, b, y, &params);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    return metal_msg_send_id0(st, cmd, "error") == nullptr ? GEIST_OK
                                                           : GEIST_E_BACKEND;
}

[[nodiscard]] static enum geist_status metal_scale_f32(
    struct geist_backend *be,
    const struct geist_tensor *x,
    float scale,
    struct geist_tensor *y) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows = 0, cols = 0, x_off = 0, x_stride = 0;
    size_t y_rows = 0, y_cols = 0, y_off = 0, y_stride = 0;
    if (!metal_tensor_is_f32_rows(x, &rows, &cols, &x_off, &x_stride) ||
        !metal_tensor_is_f32_rows(y, &y_rows, &y_cols, &y_off, &y_stride) ||
        y_rows != rows || y_cols != cols) {
        return GEIST_E_UNSUPPORTED;
    }
    if (rows > UINT32_MAX || cols > UINT32_MAX || x_off > UINT32_MAX ||
        y_off > UINT32_MAX || x_stride > UINT32_MAX ||
        y_stride > UINT32_MAX ||
        x->buffer->owner != be->state || y->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) { return s; }
    struct metal_state *st = be->state;
    const struct metal_scale_rows_params params = {
        .rows = (uint32_t) rows,
        .cols = (uint32_t) cols,
        .x_offset = (uint32_t) x_off,
        .y_offset = (uint32_t) y_off,
        .x_row_stride = (uint32_t) x_stride,
        .y_row_stride = (uint32_t) y_stride,
        .scale = scale,
    };
    if (st->sequence_active) {
        struct metal_decode_replay_op *op = metal_decode_replay_append(
            st, METAL_DECODE_REPLAY_SCALE_F32);
        if (op != nullptr) {
            op->u.scale = (struct metal_decode_replay_scale){
                .x = *x,
                .y = *y,
                .params = {
                    .rows = params.rows,
                    .cols = params.cols,
                    .x_offset = params.x_offset,
                    .y_offset = params.y_offset,
                    .x_row_stride = params.x_row_stride,
                    .y_row_stride = params.y_row_stride,
                    .scale = params.scale,
                },
            };
        }
        metal_encode_scale_rows(st, st->sequence_compute_encoder, x, y,
                                &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    void *enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder")
                               : nullptr;
    if (cmd == nullptr || enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_scale_rows(st, enc, x, y, &params);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    return metal_msg_send_id0(st, cmd, "error") == nullptr ? GEIST_OK
                                                           : GEIST_E_BACKEND;
}

[[nodiscard]] static enum geist_status metal_gelu_tanh(
    struct geist_backend *be,
    const struct geist_tensor *x,
    struct geist_tensor *y) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows = 0, cols = 0, x_off = 0, x_stride = 0;
    size_t y_rows = 0, y_cols = 0, y_off = 0, y_stride = 0;
    if (!metal_tensor_is_f32_rows(x, &rows, &cols, &x_off, &x_stride) ||
        !metal_tensor_is_f32_rows(y, &y_rows, &y_cols, &y_off, &y_stride) ||
        y_rows != rows || y_cols != cols) {
        return GEIST_E_UNSUPPORTED;
    }
    if (rows > UINT32_MAX || cols > UINT32_MAX || x_off > UINT32_MAX ||
        y_off > UINT32_MAX || x_stride > UINT32_MAX ||
        y_stride > UINT32_MAX ||
        x->buffer->owner != be->state || y->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) { return s; }
    struct metal_state *st = be->state;
    const struct metal_scale_rows_params params = {
        .rows = (uint32_t) rows,
        .cols = (uint32_t) cols,
        .x_offset = (uint32_t) x_off,
        .y_offset = (uint32_t) y_off,
        .x_row_stride = (uint32_t) x_stride,
        .y_row_stride = (uint32_t) y_stride,
        .scale = 0.0f,
    };
    if (st->sequence_active) {
        metal_encode_gelu_rows(st, st->sequence_compute_encoder, x, y,
                               &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    void *enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder")
                               : nullptr;
    if (cmd == nullptr || enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_gelu_rows(st, enc, x, y, &params);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    return metal_msg_send_id0(st, cmd, "error") == nullptr ? GEIST_OK
                                                           : GEIST_E_BACKEND;
}

[[nodiscard]] static enum geist_status metal_gelu_tanh_mul(
    struct geist_backend *be,
    const struct geist_tensor *x,
    const struct geist_tensor *z,
    struct geist_tensor *y) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows = 0, cols = 0, x_off = 0, x_stride = 0;
    size_t z_rows = 0, z_cols = 0, z_off = 0, z_stride = 0;
    size_t y_rows = 0, y_cols = 0, y_off = 0, y_stride = 0;
    if (!metal_tensor_is_f32_rows(x, &rows, &cols, &x_off, &x_stride) ||
        !metal_tensor_is_f32_rows(z, &z_rows, &z_cols, &z_off, &z_stride) ||
        !metal_tensor_is_f32_rows(y, &y_rows, &y_cols, &y_off, &y_stride) ||
        z_rows != rows || y_rows != rows || z_cols != cols ||
        y_cols != cols) {
        return GEIST_E_UNSUPPORTED;
    }
    if (rows > UINT32_MAX || cols > UINT32_MAX || x_off > UINT32_MAX ||
        z_off > UINT32_MAX || y_off > UINT32_MAX ||
        x_stride > UINT32_MAX || z_stride > UINT32_MAX ||
        y_stride > UINT32_MAX ||
        x->buffer->owner != be->state || z->buffer->owner != be->state ||
        y->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) { return s; }
    struct metal_state *st = be->state;
    const struct metal_binary_rows_params params = {
        .rows = (uint32_t) rows,
        .cols = (uint32_t) cols,
        .a_offset = (uint32_t) x_off,
        .b_offset = (uint32_t) z_off,
        .y_offset = (uint32_t) y_off,
        .a_row_stride = (uint32_t) x_stride,
        .b_row_stride = (uint32_t) z_stride,
        .y_row_stride = (uint32_t) y_stride,
    };
    if (st->sequence_active) {
        metal_encode_gelu_mul_rows(st, st->sequence_compute_encoder, x, z, y,
                                   &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    void *enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder")
                               : nullptr;
    if (cmd == nullptr || enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_gelu_mul_rows(st, enc, x, z, y, &params);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    return metal_msg_send_id0(st, cmd, "error") == nullptr ? GEIST_OK
                                                           : GEIST_E_BACKEND;
}

[[nodiscard]] static enum geist_status metal_embedding_lookup_scaled(
    struct geist_backend *be,
    const struct geist_tensor *embed_table,
    geist_token_t token_id,
    float scale,
    struct geist_tensor *out) {

    if (be == nullptr || be->state == nullptr ||
        embed_table == nullptr || out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }

    size_t out_n = 0;
    size_t out_offset = 0;
    if (!metal_tensor_is_f32_vector(out, &out_n, &out_offset) ||
        embed_table->buffer == nullptr ||
        embed_table->ndim != 2 ||
        embed_table->shape[0] <= 0 ||
        embed_table->shape[1] <= 0) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t vocab = (size_t) embed_table->shape[0];
    const size_t d_model = (size_t) embed_table->shape[1];
    if (token_id < 0 || (size_t) token_id >= vocab || out_n != d_model) {
        return GEIST_E_INVALID_ARG;
    }

    size_t row_bytes = 0;
    size_t blocks_per_row = 0;
    if (embed_table->layout == GEIST_LAYOUT_DENSE &&
        embed_table->dtype == GEIST_DTYPE_F32) {
        if (d_model > SIZE_MAX / sizeof(float)) {
            return GEIST_E_INVALID_ARG;
        }
        row_bytes = d_model * sizeof(float);
    } else if (embed_table->layout == GEIST_LAYOUT_DENSE &&
               (embed_table->dtype == GEIST_DTYPE_F16 ||
                embed_table->dtype == GEIST_DTYPE_BF16)) {
        if (d_model > SIZE_MAX / sizeof(uint16_t)) {
            return GEIST_E_INVALID_ARG;
        }
        row_bytes = d_model * sizeof(uint16_t);
    } else if (embed_table->layout == GEIST_LAYOUT_BLOCK_QUANTIZED &&
               embed_table->dtype == GEIST_DTYPE_Q4_K) {
        if ((d_model % METAL_Q4K_BLOCK_ELEMS) != 0) {
            return GEIST_E_INVALID_ARG;
        }
        blocks_per_row = d_model / METAL_Q4K_BLOCK_ELEMS;
        if (blocks_per_row > SIZE_MAX / METAL_Q4K_BLOCK_BYTES) {
            return GEIST_E_INVALID_ARG;
        }
        row_bytes = blocks_per_row * METAL_Q4K_BLOCK_BYTES;
    } else if (embed_table->layout == GEIST_LAYOUT_BLOCK_QUANTIZED &&
               embed_table->dtype == GEIST_DTYPE_Q5_K) {
        if ((d_model % METAL_Q5K_BLOCK_ELEMS) != 0) {
            return GEIST_E_INVALID_ARG;
        }
        blocks_per_row = d_model / METAL_Q5K_BLOCK_ELEMS;
        if (blocks_per_row > SIZE_MAX / METAL_Q5K_BLOCK_BYTES) {
            return GEIST_E_INVALID_ARG;
        }
        row_bytes = blocks_per_row * METAL_Q5K_BLOCK_BYTES;
    } else if (embed_table->layout == GEIST_LAYOUT_BLOCK_QUANTIZED &&
               embed_table->dtype == GEIST_DTYPE_Q6_K) {
        if ((d_model % METAL_Q6K_BLOCK_ELEMS) != 0) {
            return GEIST_E_INVALID_ARG;
        }
        blocks_per_row = d_model / METAL_Q6K_BLOCK_ELEMS;
        if (blocks_per_row > SIZE_MAX / METAL_Q6K_BLOCK_BYTES) {
            return GEIST_E_INVALID_ARG;
        }
        row_bytes = blocks_per_row * METAL_Q6K_BLOCK_BYTES;
    } else {
        geist_backend_set_error(
            be, GEIST_E_UNSUPPORTED,
            "metal embedding_lookup_scaled: unsupported table dtype/layout");
        return GEIST_E_UNSUPPORTED;
    }

    if (row_bytes == 0 ||
        vocab > SIZE_MAX / row_bytes ||
        embed_table->offset > embed_table->buffer->bytes ||
        vocab * row_bytes > embed_table->buffer->bytes - embed_table->offset ||
        out->offset > out->buffer->bytes ||
        d_model > (out->buffer->bytes - out->offset) / sizeof(float) ||
        d_model > UINT32_MAX ||
        blocks_per_row > UINT32_MAX ||
        embed_table->offset > UINT32_MAX ||
        out_offset > UINT32_MAX ||
        (size_t) token_id > UINT32_MAX) {
        return GEIST_E_INVALID_ARG;
    }
    if (embed_table->buffer->owner != be->state ||
        out->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }

    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }

    struct metal_state *st = be->state;
    const struct metal_embed_params params = {
        .n = (uint32_t) d_model,
        .dtype = (uint32_t) embed_table->dtype,
        .blocks_per_row = (uint32_t) blocks_per_row,
        .w_byte_offset = (uint32_t) embed_table->offset,
        .y_offset = (uint32_t) out_offset,
        .token_id = (uint32_t) token_id,
        .scale = scale,
    };

    if (st->sequence_active) {
        if (st->sequence_compute_encoder == nullptr) {
            geist_backend_set_error(
                be, GEIST_E_BACKEND,
                "metal embedding_lookup_scaled: command sequence has no encoder");
            return GEIST_E_BACKEND;
        }
        struct metal_decode_replay_op *op = metal_decode_replay_append(
            st, METAL_DECODE_REPLAY_EMBED_LOOKUP_SCALED);
        if (op != nullptr) {
            op->u.embed = (struct metal_decode_replay_embed){
                .table = *embed_table,
                .out = *out,
                .params = {
                    .n = params.n,
                    .dtype = params.dtype,
                    .blocks_per_row = params.blocks_per_row,
                    .w_byte_offset = params.w_byte_offset,
                    .y_offset = params.y_offset,
                    .scale = params.scale,
                },
            };
        }
        metal_encode_embed_lookup_scaled(st, st->sequence_compute_encoder,
                                         embed_table, out, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }

    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    if (cmd == nullptr) {
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "metal embedding_lookup_scaled: command buffer failed");
        return GEIST_E_BACKEND;
    }
    void *enc = metal_msg_send_id0(st, cmd, "computeCommandEncoder");
    if (enc == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal embedding_lookup_scaled: encoder failed");
        return GEIST_E_BACKEND;
    }

    metal_encode_embed_lookup_scaled(st, enc, embed_table, out, &params);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");

    void *err = metal_msg_send_id0(st, cmd, "error");
    if (err != nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "metal embedding_lookup_scaled: command failed%s%s",
            msg != nullptr ? ": " : "",
            msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status metal_embedding_lookup(
    struct geist_backend *be,
    const struct geist_tensor *embed_table,
    geist_token_t token_id,
    struct geist_tensor *out) {

    return metal_embedding_lookup_scaled(be, embed_table, token_id, 1.0f, out);
}

[[nodiscard]] static enum geist_status metal_matvec_q4k(
    struct geist_backend *be,
    const struct geist_tensor *x,
    const struct geist_tensor *w,
    struct geist_tensor *y) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t n_in = 0;
    size_t x_offset = 0;
    size_t y_n = 0;
    size_t y_offset = 0;
    size_t n_out = 0;
    size_t w_cols = 0;
    size_t w_offset = 0;
    if (!metal_tensor_is_f32_vector(x, &n_in, &x_offset) ||
        !metal_tensor_is_f32_vector(y, &y_n, &y_offset) ||
        !metal_tensor_is_q4k_matrix(w, &n_out, &w_cols, &w_offset) ||
        w_cols != n_in || y_n != n_out) {
        geist_backend_set_error(
            be, GEIST_E_UNSUPPORTED,
            "metal Q4_K matvec: expected x F32 [n], w Q4_K [out,n], y F32 [out]");
        return GEIST_E_UNSUPPORTED;
    }
    if (n_in > UINT32_MAX || n_out > UINT32_MAX ||
        x_offset > UINT32_MAX || y_offset > UINT32_MAX ||
        w_offset > UINT32_MAX) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG,
                                "metal Q4_K matvec: tensor too large");
        return GEIST_E_INVALID_ARG;
    }
    if (x->buffer->owner != be->state || w->buffer->owner != be->state ||
        y->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }

    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }

    struct metal_state *st = be->state;
    if (st->use_q4k_w4a8) {
        s = metal_ensure_q4k_w4a8_scratch(be, n_in);
        if (s != GEIST_OK) {
            return s;
        }
    }
    const struct metal_q4k_params params = {
        .n_in = (uint32_t) n_in,
        .n_out = (uint32_t) n_out,
        .rows = 1,
        .blocks_per_row = (uint32_t) (n_in / METAL_Q4K_BLOCK_ELEMS),
        .x_offset = (uint32_t) x_offset,
        .w_byte_offset = (uint32_t) w_offset,
        .y_offset = (uint32_t) y_offset,
        .x_row_stride = (uint32_t) n_in,
        .y_row_stride = (uint32_t) n_out,
    };

    if (st->sequence_active) {
        if (st->sequence_compute_encoder == nullptr) {
            geist_backend_set_error(
                be, GEIST_E_BACKEND,
                "metal Q4_K matvec: command sequence has no encoder");
            return GEIST_E_BACKEND;
        }
        metal_encode_q4k_linear(st, st->sequence_compute_encoder,
                                x, w, y, &params, false);
        st->sequence_has_work = true;
        return GEIST_OK;
    }

    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    if (cmd == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal Q4_K matvec: command buffer failed");
        return GEIST_E_BACKEND;
    }
    void *enc = metal_msg_send_id0(st, cmd, "computeCommandEncoder");
    if (enc == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal Q4_K matvec: encoder failed");
        return GEIST_E_BACKEND;
    }

    metal_encode_q4k_linear(st, enc, x, w, y, &params, false);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");

    void *err = metal_msg_send_id0(st, cmd, "error");
    if (err != nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal Q4_K matvec: command failed");
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status metal_matmul_q4k(
    struct geist_backend *be,
    const struct geist_tensor *x,
    const struct geist_tensor *w,
    struct geist_tensor *y) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows = 0;
    size_t n_in = 0;
    size_t x_offset = 0;
    size_t x_row_stride = 0;
    size_t y_rows = 0;
    size_t y_cols = 0;
    size_t y_offset = 0;
    size_t y_row_stride = 0;
    size_t n_out = 0;
    size_t w_cols = 0;
    size_t w_offset = 0;
    if (!metal_tensor_is_f32_matrix(x, &rows, &n_in, &x_offset,
                                    &x_row_stride) ||
        !metal_tensor_is_f32_matrix(y, &y_rows, &y_cols, &y_offset,
                                    &y_row_stride) ||
        !metal_tensor_is_q4k_matrix(w, &n_out, &w_cols, &w_offset) ||
        y_rows != rows || w_cols != n_in || y_cols != n_out) {
        geist_backend_set_error(
            be, GEIST_E_UNSUPPORTED,
            "metal Q4_K matmul: expected x F32 [rows,n], w Q4_K [out,n], y F32 [rows,out]");
        return GEIST_E_UNSUPPORTED;
    }
    if (rows > UINT32_MAX || n_in > UINT32_MAX || n_out > UINT32_MAX ||
        x_offset > UINT32_MAX || y_offset > UINT32_MAX ||
        w_offset > UINT32_MAX || x_row_stride > UINT32_MAX ||
        y_row_stride > UINT32_MAX) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG,
                                "metal Q4_K matmul: tensor too large");
        return GEIST_E_INVALID_ARG;
    }
    if (x->buffer->owner != be->state || w->buffer->owner != be->state ||
        y->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }

    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }

    struct metal_state *st = be->state;
    const struct metal_q4k_params params = {
        .n_in = (uint32_t) n_in,
        .n_out = (uint32_t) n_out,
        .rows = (uint32_t) rows,
        .blocks_per_row = (uint32_t) (n_in / METAL_Q4K_BLOCK_ELEMS),
        .x_offset = (uint32_t) x_offset,
        .w_byte_offset = (uint32_t) w_offset,
        .y_offset = (uint32_t) y_offset,
        .x_row_stride = (uint32_t) x_row_stride,
        .y_row_stride = (uint32_t) y_row_stride,
    };

    if (st->sequence_active) {
        if (st->sequence_compute_encoder == nullptr) {
            geist_backend_set_error(
                be, GEIST_E_BACKEND,
                "metal Q4_K matmul: command sequence has no encoder");
            return GEIST_E_BACKEND;
        }
        metal_encode_q4k_linear(st, st->sequence_compute_encoder,
                                x, w, y, &params, true);
        st->sequence_has_work = true;
        return GEIST_OK;
    }

    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    if (cmd == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal Q4_K matmul: command buffer failed");
        return GEIST_E_BACKEND;
    }
    void *enc = metal_msg_send_id0(st, cmd, "computeCommandEncoder");
    if (enc == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal Q4_K matmul: encoder failed");
        return GEIST_E_BACKEND;
    }

    metal_encode_q4k_linear(st, enc, x, w, y, &params, true);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");

    void *err = metal_msg_send_id0(st, cmd, "error");
    if (err != nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal Q4_K matmul: command failed");
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status metal_q6k_linear(
    struct geist_backend *be,
    const struct geist_tensor *x,
    const struct geist_tensor *w,
    struct geist_tensor *y,
    bool matrix) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows = 1;
    size_t n_in = 0;
    size_t x_offset = 0;
    size_t x_row_stride = 0;
    size_t y_rows = 1;
    size_t y_cols = 0;
    size_t y_offset = 0;
    size_t y_row_stride = 0;
    size_t n_out = 0;
    size_t w_cols = 0;
    size_t w_offset = 0;
    bool ok = false;
    if (matrix) {
        ok = metal_tensor_is_f32_matrix(x, &rows, &n_in, &x_offset,
                                        &x_row_stride) &&
             metal_tensor_is_f32_matrix(y, &y_rows, &y_cols, &y_offset,
                                        &y_row_stride) &&
             y_rows == rows;
    } else {
        ok = metal_tensor_is_f32_vector(x, &n_in, &x_offset) &&
             metal_tensor_is_f32_vector(y, &y_cols, &y_offset);
        x_row_stride = n_in;
        y_row_stride = y_cols;
    }
    if (!ok ||
        !metal_tensor_is_q6k_matrix(w, &n_out, &w_cols, &w_offset) ||
        w_cols != n_in || y_cols != n_out) {
        geist_backend_set_error(
            be, GEIST_E_UNSUPPORTED,
            matrix
                ? "metal Q6_K matmul: expected x F32 [rows,n], w Q6_K [out,n], y F32 [rows,out]"
                : "metal Q6_K matvec: expected x F32 [n], w Q6_K [out,n], y F32 [out]");
        return GEIST_E_UNSUPPORTED;
    }
    if (rows > UINT32_MAX || n_in > UINT32_MAX || n_out > UINT32_MAX ||
        x_offset > UINT32_MAX || y_offset > UINT32_MAX ||
        w_offset > UINT32_MAX || x_row_stride > UINT32_MAX ||
        y_row_stride > UINT32_MAX) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG,
                                "metal Q6_K linear: tensor too large");
        return GEIST_E_INVALID_ARG;
    }
    if (x->buffer->owner != be->state || w->buffer->owner != be->state ||
        y->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }

    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }

    struct metal_state *st = be->state;
    const struct metal_q4k_params params = {
        .n_in = (uint32_t) n_in,
        .n_out = (uint32_t) n_out,
        .rows = (uint32_t) rows,
        .blocks_per_row = (uint32_t) (n_in / METAL_Q6K_BLOCK_ELEMS),
        .x_offset = (uint32_t) x_offset,
        .w_byte_offset = (uint32_t) w_offset,
        .y_offset = (uint32_t) y_offset,
        .x_row_stride = (uint32_t) x_row_stride,
        .y_row_stride = (uint32_t) y_row_stride,
    };

    if (st->sequence_active) {
        if (st->sequence_compute_encoder == nullptr) {
            geist_backend_set_error(
                be, GEIST_E_BACKEND,
                "metal Q6_K linear: command sequence has no encoder");
            return GEIST_E_BACKEND;
        }
        metal_encode_q6k_linear(st, st->sequence_compute_encoder,
                                x, w, y, &params, matrix);
        st->sequence_has_work = true;
        return GEIST_OK;
    }

    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    if (cmd == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal Q6_K linear: command buffer failed");
        return GEIST_E_BACKEND;
    }
    void *enc = metal_msg_send_id0(st, cmd, "computeCommandEncoder");
    if (enc == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal Q6_K linear: encoder failed");
        return GEIST_E_BACKEND;
    }

    metal_encode_q6k_linear(st, enc, x, w, y, &params, matrix);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");

    void *err = metal_msg_send_id0(st, cmd, "error");
    if (err != nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal Q6_K linear: command failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status metal_matvec_q6k(
    struct geist_backend *be,
    const struct geist_tensor *x,
    const struct geist_tensor *w,
    struct geist_tensor *y) {
    return metal_q6k_linear(be, x, w, y, false);
}

[[nodiscard]] static enum geist_status metal_matmul_q6k(
    struct geist_backend *be,
    const struct geist_tensor *x,
    const struct geist_tensor *w,
    struct geist_tensor *y) {
    return metal_q6k_linear(be, x, w, y, true);
}

static bool metal_tensor_is_dense_3d_dtype(const struct geist_tensor *t,
                                           enum geist_dtype dtype,
                                           size_t elem_size,
                                           size_t *out_d0,
                                           size_t *out_d1,
                                           size_t *out_d2,
                                           size_t *out_offset_elems);

static bool metal_tensor_is_f32_3d(const struct geist_tensor *t,
                                   size_t *out_d0,
                                   size_t *out_d1,
                                   size_t *out_d2,
                                   size_t *out_offset_floats) {
    return metal_tensor_is_dense_3d_dtype(t, GEIST_DTYPE_F32, sizeof(float),
                                          out_d0, out_d1, out_d2,
                                          out_offset_floats);
}

static bool metal_tensor_is_f16_3d(const struct geist_tensor *t,
                                   size_t *out_d0,
                                   size_t *out_d1,
                                   size_t *out_d2,
                                   size_t *out_offset_halfs) {
    return metal_tensor_is_dense_3d_dtype(t, GEIST_DTYPE_F16, sizeof(uint16_t),
                                          out_d0, out_d1, out_d2,
                                          out_offset_halfs);
}

static bool metal_tensor_is_dense_3d_dtype(const struct geist_tensor *t,
                                           enum geist_dtype dtype,
                                           size_t elem_size,
                                           size_t *out_d0,
                                           size_t *out_d1,
                                           size_t *out_d2,
                                           size_t *out_offset_elems) {
    if (t == nullptr || t->buffer == nullptr ||
        t->dtype != dtype ||
        t->layout != GEIST_LAYOUT_DENSE ||
        t->ndim != 3 ||
        t->shape[0] <= 0 || t->shape[1] <= 0 || t->shape[2] <= 0 ||
        t->stride[2] != 1 ||
        t->stride[1] != t->shape[2] ||
        t->stride[0] != t->shape[1] * t->shape[2] ||
        elem_size == 0 ||
        t->offset % elem_size != 0) {
        return false;
    }
    const size_t d0 = (size_t) t->shape[0];
    const size_t d1 = (size_t) t->shape[1];
    const size_t d2 = (size_t) t->shape[2];
    if (d0 > SIZE_MAX / d1 || d0 * d1 > SIZE_MAX / d2) {
        return false;
    }
    const size_t elems = d0 * d1 * d2;
    if (t->offset > t->buffer->bytes ||
        elems > (t->buffer->bytes - t->offset) / elem_size) {
        return false;
    }
    *out_d0 = d0;
    *out_d1 = d1;
    *out_d2 = d2;
    *out_offset_elems = t->offset / elem_size;
    return true;
}

[[nodiscard]] static enum geist_status metal_rope_apply(
    struct geist_backend *be,
    struct geist_tensor *x,
    const struct geist_tensor *cos,
    const struct geist_tensor *sin) {

    if (be == nullptr || be->state == nullptr ||
        x == nullptr || cos == nullptr || sin == nullptr) {
        return GEIST_E_INVALID_ARG;
    }

    {
        static int dbg = -1;
        if (dbg < 0) {
            const char *e = getenv("GEIST_METAL_DEBUG_LINEAR");
            dbg = (e != nullptr && e[0] != '\0' && strcmp(e, "0") != 0) ? 1 : 0;
        }
        if (dbg && x->buffer != nullptr && x->buffer->mapped != nullptr &&
            cos->buffer != nullptr && cos->buffer->mapped != nullptr) {
            const float *xp = (const float *) ((const uint8_t *) x->buffer->mapped + x->offset);
            const float *cp = (const float *) ((const uint8_t *) cos->buffer->mapped + cos->offset);
            size_t nx = 1, nc = 1;
            for (int i = 0; i < x->ndim; i++) nx *= (size_t) x->shape[i];
            for (int i = 0; i < cos->ndim; i++) nc *= (size_t) cos->shape[i];
            size_t nanx = 0, nanc = 0;
            for (size_t i = 0; i < nx; i++) if (isnan(xp[i])) nanx++;
            for (size_t i = 0; i < nc; i++) if (isnan(cp[i])) nanc++;
            fprintf(stderr, "rope_in x[%lld,%lld,%lld] nanx=%zu cos[%lld,%lld] nanc=%zu xoff=%zu coff=%zu\n",
                    (long long) x->shape[0], (long long) x->shape[1], (long long) x->shape[2],
                    nanx, (long long) cos->shape[0], (long long) cos->shape[1], nanc,
                    x->offset, cos->offset);
        }
    }
    size_t rows = 0, heads = 0, head_dim = 0, x_offset = 0;
    size_t cos_rows = 0, cos_cols = 0, cos_offset = 0, cos_stride = 0;
    size_t sin_rows = 0, sin_cols = 0, sin_offset = 0, sin_stride = 0;
    if (!metal_tensor_is_f32_3d(x, &rows, &heads, &head_dim, &x_offset) ||
        !metal_tensor_is_f32_matrix(cos, &cos_rows, &cos_cols, &cos_offset,
                                    &cos_stride) ||
        !metal_tensor_is_f32_matrix(sin, &sin_rows, &sin_cols, &sin_offset,
                                    &sin_stride)) {
        geist_backend_set_error(
            be, GEIST_E_UNSUPPORTED,
            "metal rope_apply: tensors must be F32 DENSE x[seq,heads,dim]");
        return GEIST_E_UNSUPPORTED;
    }
    if (head_dim == 0 || (head_dim % 2u) != 0 ||
        cos_rows != rows || sin_rows != rows ||
        cos_cols != head_dim || sin_cols != head_dim ||
        cos_stride != head_dim || sin_stride != head_dim) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG,
                                "metal rope_apply: shape mismatch");
        return GEIST_E_INVALID_ARG;
    }
    if (rows > UINT32_MAX || heads > UINT32_MAX ||
        head_dim > UINT32_MAX || x_offset > UINT32_MAX ||
        cos_offset > UINT32_MAX || sin_offset > UINT32_MAX ||
        rows > UINT32_MAX / heads ||
        rows * heads > UINT32_MAX / (head_dim / 2u) ||
        x->buffer->owner != be->state ||
        cos->buffer->owner != be->state ||
        sin->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }

    enum geist_status s = metal_ensure_attention_pipeline(be);
    if (s != GEIST_OK) { return s; }
    struct metal_state *st = be->state;
    const struct metal_rope_params params = {
        .rows = (uint32_t) rows,
        .heads = (uint32_t) heads,
        .head_dim = (uint32_t) head_dim,
        .x_offset = (uint32_t) x_offset,
        .cos_offset = (uint32_t) cos_offset,
        .sin_offset = (uint32_t) sin_offset,
        .x_row_stride = (uint32_t) (heads * head_dim),
        .rope_row_stride = (uint32_t) cos_stride,
        .rope_row_offset = 0,
    };
    if (st->sequence_active) {
        if (st->sequence_compute_encoder == nullptr) {
            geist_backend_set_error(be, GEIST_E_BACKEND,
                                    "metal rope_apply: sequence has no encoder");
            return GEIST_E_BACKEND;
        }
        metal_encode_rope_rows(st, st->sequence_compute_encoder, x, cos, sin,
                               &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    void *enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder")
                               : nullptr;
    if (cmd == nullptr || enc == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal rope_apply: command encoder failed");
        return GEIST_E_BACKEND;
    }
    metal_encode_rope_rows(st, enc, x, cos, sin, &params);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    void *err = metal_msg_send_id0(st, cmd, "error");
    if (err != nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal rope_apply: command failed");
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status metal_attention(
    struct geist_backend *be,
    const struct geist_tensor *q,
    const struct geist_tensor *k,
    const struct geist_tensor *value,
    size_t q_offset,
    size_t sliding_window,
    struct geist_tensor *out) {

    if (be == nullptr || be->state == nullptr ||
        q == nullptr || k == nullptr || value == nullptr || out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }

    {
        static int dbg = -1;
        if (dbg < 0) {
            const char *e = getenv("GEIST_METAL_DEBUG_LINEAR");
            dbg = (e != nullptr && e[0] != '\0' && strcmp(e, "0") != 0) ? 1 : 0;
        }
        if (dbg && q->buffer != nullptr && q->buffer->mapped != nullptr &&
            k->buffer != nullptr && k->buffer->mapped != nullptr) {
            const float *qp = (const float *) ((const uint8_t *) q->buffer->mapped + q->offset);
            const float *kp = (const float *) ((const uint8_t *) k->buffer->mapped + k->offset);
            size_t nq = 1, nk = 1;
            for (int i = 0; i < q->ndim; i++) nq *= (size_t) q->shape[i];
            for (int i = 0; i < k->ndim; i++) nk *= (size_t) k->shape[i];
            float aq = 0, ak = 0; size_t nanq = 0, nank = 0;
            for (size_t i = 0; i < nq; i++) { if (isnan(qp[i])) nanq++; else if (fabsf(qp[i]) > aq) aq = fabsf(qp[i]); }
            for (size_t i = 0; i < nk; i++) { if (isnan(kp[i])) nank++; else if (fabsf(kp[i]) > ak) ak = fabsf(kp[i]); }
            size_t firstnan = (size_t) -1;
            for (size_t i = 0; i < nq && firstnan == (size_t) -1; i++)
                if (isnan(qp[i])) firstnan = i;
            fprintf(stderr, "attn firstnan_q=%zd (row %zd) ", (ssize_t) firstnan,
                    firstnan == (size_t) -1 ? (ssize_t) -1 : (ssize_t) (firstnan / ((size_t) q->shape[1] * (size_t) q->shape[2])));
            fprintf(stderr, "attn q[%lld,%lld,%lld] |q|=%g nanq=%zu k[%lld,..] |k|=%g nank=%zu qoff=%zu sw=%zu kdt=%d\n",
                    (long long) q->shape[0], (long long) q->shape[1], (long long) q->shape[2],
                    (double) aq, nanq, (long long) k->shape[0], (double) ak, nank,
                    q_offset, sliding_window, (int) k->dtype);
        }
    }
    size_t q_rows = 0, q_heads = 0, head_dim = 0, q_off = 0;
    size_t k_rows = 0, k_heads = 0, k_head_dim = 0, k_off = 0;
    size_t v_rows = 0, v_heads = 0, v_head_dim = 0, v_off = 0;
    size_t out_rows = 0, out_heads = 0, out_head_dim = 0, out_off = 0;
    if (!metal_tensor_is_f32_3d(q, &q_rows, &q_heads, &head_dim, &q_off) ||
        !metal_tensor_is_f32_3d(out, &out_rows, &out_heads, &out_head_dim,
                                &out_off)) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED,
                                "metal attention: q/out tensors must be F32 DENSE 3D");
        return GEIST_E_UNSUPPORTED;
    }
    const bool kv_is_f32 =
        metal_tensor_is_f32_3d(k, &k_rows, &k_heads, &k_head_dim, &k_off) &&
        metal_tensor_is_f32_3d(value, &v_rows, &v_heads, &v_head_dim, &v_off);
    const bool kv_is_f16 =
        !kv_is_f32 &&
        metal_tensor_is_f16_3d(k, &k_rows, &k_heads, &k_head_dim, &k_off) &&
        metal_tensor_is_f16_3d(value, &v_rows, &v_heads, &v_head_dim, &v_off);
    if (!kv_is_f32 && !kv_is_f16) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED,
                                "metal attention: K/V cache must be matching F32 or F16 DENSE 3D");
        return GEIST_E_UNSUPPORTED;
    }
    if (q_rows != out_rows || q_heads != out_heads ||
        head_dim != out_head_dim || k_rows != v_rows ||
        k_heads != v_heads || head_dim != k_head_dim ||
        head_dim != v_head_dim || k_heads == 0 ||
        q_heads % k_heads != 0) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG,
                                "metal attention: shape mismatch");
        return GEIST_E_INVALID_ARG;
    }
    if (head_dim > 512u) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED,
                                "metal attention: head_dim exceeds shader limit");
        return GEIST_E_UNSUPPORTED;
    }
    if (q_rows > UINT32_MAX || k_rows > UINT32_MAX ||
        q_heads > UINT32_MAX || k_heads > UINT32_MAX ||
        head_dim > UINT32_MAX || q_offset > UINT32_MAX ||
        sliding_window > UINT32_MAX || q_off > UINT32_MAX ||
        k_off > UINT32_MAX || v_off > UINT32_MAX ||
        out_off > UINT32_MAX || q_offset > SIZE_MAX - q_rows ||
        q_offset + q_rows > UINT32_MAX ||
        q->buffer->owner != be->state ||
        k->buffer->owner != be->state ||
        value->buffer->owner != be->state ||
        out->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }

    enum geist_status s = metal_ensure_attention_pipeline(be);
    if (s != GEIST_OK) { return s; }
    struct metal_state *st = be->state;
    const struct metal_attention_params params = {
        .rows = (uint32_t) q_rows,
        .kv_len = (uint32_t) k_rows,
        .q_heads = (uint32_t) q_heads,
        .kv_heads = (uint32_t) k_heads,
        .head_dim = (uint32_t) head_dim,
        .q_position = (uint32_t) q_offset,
        .sliding_window = (uint32_t) sliding_window,
        .q_offset = (uint32_t) q_off,
        .k_cache_offset = (uint32_t) k_off,
        .v_cache_offset = (uint32_t) v_off,
        .y_offset = (uint32_t) out_off,
    };
    if (st->sequence_active) {
        if (st->sequence_compute_encoder == nullptr) {
            geist_backend_set_error(be, GEIST_E_BACKEND,
                                    "metal attention: sequence has no encoder");
            return GEIST_E_BACKEND;
        }
        metal_encode_attention_rows(st, st->sequence_compute_encoder, q, k,
                                    value, out, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    void *enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder")
                               : nullptr;
    if (cmd == nullptr || enc == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal attention: command encoder failed");
        return GEIST_E_BACKEND;
    }
    metal_encode_attention_rows(st, enc, q, k, value, out, &params);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    void *err = metal_msg_send_id0(st, cmd, "error");
    if (err != nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal attention: command failed");
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

/* One-shot Xcode GPU capture of the first prefill command sequence, for
 * per-dispatch kernel timing (works on M1; open the .gputrace in Xcode).
 * Usage: METAL_CAPTURE_ENABLED=1 GEIST_METAL_CAPTURE=/path/out.gputrace
 * METAL_CAPTURE_ENABLED must be set at process launch or startCapture fails.
 * GEIST_METAL_CAPTURE_SKIP=N skips the first N prefill sequences (e.g. a
 * bench warmup) before capturing. */
static void metal_capture_begin(struct metal_state *st,
                                enum geist_command_sequence_kind kind) {
    if (st->capture_done || kind != GEIST_COMMAND_SEQUENCE_PREFILL_TEXT) {
        return;
    }
    const char *skip_env = getenv("GEIST_METAL_CAPTURE_SKIP");
    if (skip_env != nullptr && st->capture_skipped < atoi(skip_env)) {
        st->capture_skipped++;
        return;
    }
    const char *path = getenv("GEIST_METAL_CAPTURE");
    if (path == nullptr || path[0] == '\0') {
        st->capture_done = true;
        return;
    }
    st->capture_done = true;
    void *mgr_class = metal_objc_get_class(st, "MTLCaptureManager");
    void *desc_class = metal_objc_get_class(st, "MTLCaptureDescriptor");
    void *str_class = metal_objc_get_class(st, "NSString");
    void *url_class = metal_objc_get_class(st, "NSURL");
    if (mgr_class == nullptr || desc_class == nullptr ||
        str_class == nullptr || url_class == nullptr) {
        return;
    }
    void *mgr = metal_msg_send_id0(st, mgr_class, "sharedCaptureManager");
    void *desc = metal_msg_send_id0(st, desc_class, "new");
    void *ns_path =
        metal_msg_send_id_cstr(st, str_class, "stringWithUTF8String:", path);
    if (mgr == nullptr || desc == nullptr || ns_path == nullptr) {
        return;
    }
    void *url = metal_msg_send_id_id(st, url_class, "fileURLWithPath:", ns_path);
    metal_msg_send_id_id(st, desc, "setCaptureObject:", st->device);
    /* 2 = MTLCaptureDestinationGPUTraceDocument */
    metal_msg_send_void_ulong(st, desc, "setDestination:", 2);
    metal_msg_send_id_id(st, desc, "setOutputURL:", url);
    void *err = nullptr;
    if (!metal_msg_send_bool_id_err(
            st, mgr, "startCaptureWithDescriptor:error:", desc, &err)) {
        const char *msg = metal_nserror_message(st, err);
        fprintf(stderr,
                "geist metal: GPU capture start failed: %s "
                "(launch with METAL_CAPTURE_ENABLED=1)\n",
                msg != nullptr ? msg : "unknown error");
        return;
    }
    st->capture_manager = mgr;
    fprintf(stderr, "geist metal: GPU capture started -> %s\n", path);
}

static void metal_capture_end(struct metal_state *st) {
    if (st->capture_manager == nullptr) {
        return;
    }
    metal_msg_send_void0(st, st->capture_manager, "stopCapture");
    st->capture_manager = nullptr;
    fprintf(stderr, "geist metal: GPU capture written\n");
}

[[nodiscard]] static enum geist_status metal_command_sequence_begin(
    struct geist_backend *be,
    enum geist_command_sequence_kind kind,
    int *out_token) {

    if (out_token == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out_token = 0;
    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct metal_state *st = be->state;
    if (st->sequence_active) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG,
                                "metal command sequence: nested begin");
        return GEIST_E_INVALID_ARG;
    }
    switch (kind) {
    case GEIST_COMMAND_SEQUENCE_VERIFY_GREEDY:
    case GEIST_COMMAND_SEQUENCE_DECODE_LAYER_LOOP:
    case GEIST_COMMAND_SEQUENCE_DECODE_GREEDY_STEP:
    case GEIST_COMMAND_SEQUENCE_PREFILL_TEXT:
        break;
    default:
        geist_backend_set_error(be, GEIST_E_INVALID_ARG,
                                "metal command sequence: invalid kind");
        return GEIST_E_INVALID_ARG;
    }

    metal_capture_begin(st, kind);

    static int g_seq_created; g_seq_created++;
    if (getenv("GEIST_SEQ_COUNT") && (g_seq_created % 16) == 0) fprintf(stderr, "[seqdbg] created=%d\n", g_seq_created);
    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    if (cmd == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal command sequence: command buffer failed");
        return GEIST_E_BACKEND;
    }
    void *enc = metal_msg_send_id0(st, cmd, "computeCommandEncoder");
    if (enc == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal command sequence: encoder failed");
        return GEIST_E_BACKEND;
    }
    metal_msg_send_void0(st, cmd, "retain");
    metal_msg_send_void0(st, enc, "retain");

    if (st->sequence_token == INT_MAX) {
        st->sequence_token = 0;
    }
    st->sequence_token++;
    st->sequence_kind = kind;
    st->sequence_command_buffer = cmd;
    st->sequence_compute_encoder = enc;
    st->sequence_active = true;
    st->sequence_has_work = false;
    st->seq_dispatch_count = 0;
    st->seq_begin_ns = metal_now_ns();
    st->captured_greedy_token_pending = false;
    st->captured_greedy_vocab_size = 0;
    st->captured_greedy_token_count = 0;
    if (kind == GEIST_COMMAND_SEQUENCE_DECODE_GREEDY_STEP &&
        !st->decode_replay_replaying) {
        st->decode_replay_valid = false;
        st->decode_replay_failed = false;
        st->decode_replay_vocab_size = 0;
        st->decode_replay_token_count = 0;
        st->decode_replay_op_count = 0;
    }
    *out_token = st->sequence_token;
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status metal_command_sequence_end(
    struct geist_backend *be,
    int token,
    bool submit) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct metal_state *st = be->state;
    if (!st->sequence_active || token == 0 || token != st->sequence_token) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG,
                                "metal command sequence: invalid token");
        return GEIST_E_INVALID_ARG;
    }

    void *cmd = st->sequence_command_buffer;
    void *enc = st->sequence_compute_encoder;
    const bool has_work = st->sequence_has_work;
    st->sequence_compute_encoder = nullptr;
    st->sequence_command_buffer = nullptr;
    st->sequence_active = false;
    st->sequence_has_work = false;
    if (!submit) {
        st->captured_greedy_token_pending = false;
        st->captured_greedy_vocab_size = 0;
        st->captured_greedy_token_count = 0;
        if (st->sequence_kind == GEIST_COMMAND_SEQUENCE_DECODE_GREEDY_STEP &&
            !st->decode_replay_replaying) {
            st->decode_replay_valid = false;
            st->decode_replay_failed = false;
            st->decode_replay_vocab_size = 0;
            st->decode_replay_token_count = 0;
            st->decode_replay_op_count = 0;
        }
    }

    metal_msg_send_void0(st, enc, "endEncoding");
    enum geist_status out = GEIST_OK;
    /* Commit even when no work was encoded: an uncommitted command buffer
     * permanently occupies one of the queue's (default 64) slots, and the
     * batched-submit region hooks legitimately close empty sequences. */
    { static int g_seq_ended; g_seq_ended++;
      if (getenv("GEIST_SEQ_COUNT") && (g_seq_ended % 16) == 0) fprintf(stderr, "[seqdbg] ended=%d submit=%d\n", g_seq_ended, (int) submit); }
    if (submit) {
        const enum metal_profile_stage wait_stage =
            metal_profile_wait_stage_for_sequence(st->sequence_kind);
        const uint64_t wait_start_ns =
            st->profile_enabled ? metal_now_ns() : 0;
        const bool seq_trace = metal_env_enabled("GEIST_METAL_SEQ_TRACE");
        const uint64_t commit_ns = seq_trace ? metal_now_ns() : 0;
        metal_msg_send_void0(st, cmd, "commit");
        metal_msg_send_void0(st, cmd, "waitUntilCompleted");
        metal_profile_add_wait(st, wait_stage, wait_start_ns);
        if (seq_trace) {
            const uint64_t done_ns = metal_now_ns();
            union {
                void *raw;
                double (*fn)(void *, void *);
            } getd = {.raw = st->objc_msgSend};
            const double gpu_start = getd.fn(
                cmd, metal_sel_register_name(st, "GPUStartTime"));
            const double gpu_end = getd.fn(
                cmd, metal_sel_register_name(st, "GPUEndTime"));
            fprintf(stderr,
                    "seq_trace kind=%d dispatches=%llu encode=%.1fms "
                    "gpu=%.1fms wall=%.1fms wait=%.1fms\n",
                    (int) st->sequence_kind,
                    (unsigned long long) st->seq_dispatch_count,
                    (double) (commit_ns - st->seq_begin_ns) / 1e6,
                    (gpu_end - gpu_start) * 1e3,
                    (double) (done_ns - st->seq_begin_ns) / 1e6,
                    (double) (done_ns - commit_ns) / 1e6);
        }
        void *err = metal_msg_send_id0(st, cmd, "error");
        if (err != nullptr) {
            geist_backend_set_error(be, GEIST_E_BACKEND,
                                    "metal command sequence: command failed");
            out = GEIST_E_BACKEND;
            st->captured_greedy_token_pending = false;
            st->captured_greedy_vocab_size = 0;
            st->captured_greedy_token_count = 0;
            if (st->sequence_kind == GEIST_COMMAND_SEQUENCE_DECODE_GREEDY_STEP &&
                !st->decode_replay_replaying) {
                st->decode_replay_valid = false;
                st->decode_replay_failed = false;
                st->decode_replay_vocab_size = 0;
                st->decode_replay_token_count = 0;
                st->decode_replay_op_count = 0;
            }
        }
    } else if (!has_work) {
        st->captured_greedy_token_pending = false;
        st->captured_greedy_vocab_size = 0;
        st->captured_greedy_token_count = 0;
        if (st->sequence_kind == GEIST_COMMAND_SEQUENCE_DECODE_GREEDY_STEP &&
            !st->decode_replay_replaying) {
            st->decode_replay_valid = false;
            st->decode_replay_failed = false;
            st->decode_replay_vocab_size = 0;
            st->decode_replay_token_count = 0;
            st->decode_replay_op_count = 0;
        }
    }
    if (out == GEIST_OK &&
        submit &&
        has_work &&
        st->sequence_kind == GEIST_COMMAND_SEQUENCE_DECODE_GREEDY_STEP &&
        !st->decode_replay_replaying) {
        st->decode_replay_valid =
            !st->decode_replay_failed &&
            st->decode_replay_op_count != 0 &&
            st->captured_greedy_token_pending &&
            st->captured_greedy_vocab_size != 0 &&
            st->captured_greedy_token_count != 0;
        st->decode_replay_vocab_size = st->captured_greedy_vocab_size;
        st->decode_replay_token_count = st->captured_greedy_token_count;
    }

    metal_capture_end(st);

    metal_msg_send_void0(st, enc, "release");
    metal_msg_send_void0(st, cmd, "release");
    return out;
}

[[nodiscard]] static enum geist_status metal_load_runtime(
    struct geist_backend *be,
    struct metal_state *st) {

#if defined(__APPLE__)
    static const char *const metal_paths[] = {
        "/System/Library/Frameworks/Metal.framework/Metal",
        "Metal.framework/Metal",
        nullptr,
    };
    static const char *const objc_paths[] = {
        "/usr/lib/libobjc.A.dylib",
        "libobjc.A.dylib",
        nullptr,
    };

    for (size_t i = 0; metal_paths[i] != nullptr; i++) {
        st->metal_handle = dlopen(metal_paths[i], RTLD_NOW | RTLD_LOCAL);
        if (st->metal_handle != nullptr) {
            break;
        }
    }
    if (st->metal_handle == nullptr) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED,
                                "metal: Metal framework is unavailable");
        return GEIST_E_UNSUPPORTED;
    }

    for (size_t i = 0; objc_paths[i] != nullptr; i++) {
        st->objc_handle = dlopen(objc_paths[i], RTLD_NOW | RTLD_LOCAL);
        if (st->objc_handle != nullptr) {
            break;
        }
    }
    if (st->objc_handle == nullptr) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED,
                                "metal: Objective-C runtime is unavailable");
        return GEIST_E_UNSUPPORTED;
    }

    st->MTLCreateSystemDefaultDevice =
        metal_dlsym(st->metal_handle, "MTLCreateSystemDefaultDevice");
    st->objc_msgSend = metal_dlsym(st->objc_handle, "objc_msgSend");
    st->sel_registerName = metal_dlsym(st->objc_handle, "sel_registerName");
    st->objc_getClass = metal_dlsym(st->objc_handle, "objc_getClass");
    if (st->MTLCreateSystemDefaultDevice == nullptr ||
        st->objc_msgSend == nullptr ||
        st->sel_registerName == nullptr ||
        st->objc_getClass == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal: runtime symbols are incomplete");
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
#else
    (void) st;
    geist_backend_set_error(be, GEIST_E_UNSUPPORTED,
                            "metal: backend is only available on Apple platforms");
    return GEIST_E_UNSUPPORTED;
#endif
}

[[nodiscard]] static enum geist_status metal_create(
    struct geist_backend *be,
    const struct geist_backend_opts *opts) {

    (void) opts;
    struct metal_state *st =
        geist_backend_alloc(be, sizeof(*st), alignof(struct metal_state));
    if (st == nullptr) {
        geist_backend_set_error(be, GEIST_E_OOM,
                                "metal: failed to allocate %zu-byte state",
                                sizeof(*st));
        return GEIST_E_OOM;
    }
    *st = (struct metal_state){0};
    st->backend = be;
    st->use_fused_attention_norm =
        getenv("GEIST_METAL_ATTENTION_FUSED_NORM") != nullptr;
    const char *fused_kv_norm =
        getenv("GEIST_METAL_ATTENTION_FUSED_KV_NORM");
    st->use_fused_kv_norm_append =
        fused_kv_norm != nullptr && strcmp(fused_kv_norm, "1") == 0;
    /* Prefill q/k projections default to the tiled mm_sg GEMM (the separate
     * q/k matmul_q4k path, parity-gated). The fused reduction QK kernel is
     * compute-bound for batched prefill; set GEIST_METAL_ATTENTION_FUSED_QK=1
     * to restore it. (This flag only affects rows>1; decode is unaffected.) */
    const char *fused_qk = getenv("GEIST_METAL_ATTENTION_FUSED_QK");
    st->use_fused_attention_qk =
        fused_qk != nullptr && strcmp(fused_qk, "1") == 0;
    const char *fused_qk_nt4 =
        getenv("GEIST_METAL_ATTENTION_FUSED_QK_NT4");
    st->use_fused_attention_qk_nt4 =
        fused_qk_nt4 != nullptr && strcmp(fused_qk_nt4, "1") == 0;
    const char *ple_block = getenv("GEIST_METAL_PLE_BLOCK");
    st->use_ple_block = ple_block != nullptr && strcmp(ple_block, "1") == 0;
    const char *q4k_n4 = getenv("GEIST_METAL_Q4K_N4");
    st->use_q4k_n4 = q4k_n4 == nullptr || strcmp(q4k_n4, "0") != 0;
    const char *q4k_nt4 = getenv("GEIST_METAL_Q4K_NT4");
    st->use_q4k_nt4 = q4k_nt4 != nullptr && strcmp(q4k_nt4, "1") == 0;
    const char *q4k_nt8 = getenv("GEIST_METAL_Q4K_NT8");
    st->use_q4k_nt8 = q4k_nt8 != nullptr && strcmp(q4k_nt8, "1") == 0;
    const char *q4k_w4a8 = getenv("GEIST_METAL_Q4K_W4A8");
    st->use_q4k_w4a8 =
        q4k_w4a8 != nullptr && strcmp(q4k_w4a8, "1") == 0;
    const char *q4k_linear_raw = getenv("GEIST_METAL_Q4K_LINEAR_RAW");
    st->use_q4k_linear_raw =
        q4k_linear_raw == nullptr || strcmp(q4k_linear_raw, "0") != 0;
    const char *q4k_gate_up_raw = getenv("GEIST_METAL_Q4K_GATE_UP_RAW");
    st->use_q4k_gate_up_raw =
        q4k_gate_up_raw != nullptr && strcmp(q4k_gate_up_raw, "1") == 0;
    const char *q4k_m16_n2 = getenv("GEIST_METAL_Q4K_M16_N2");
    st->use_q4k_m16_n2 =
        q4k_m16_n2 != nullptr && strcmp(q4k_m16_n2, "1") == 0;
    /* Simdgroup-matmul Q4_K GEMM (llama.cpp mul_mm-derived). Default ON: only
     * runs for full tiles (dispatch guard requires rows%32==0 && n_out%64==0),
     * so the partial-tile path never executes and non-conforming shapes fall
     * back to the m16 kernel. Numerical parity for both paths is covered by
     * tests/test_backend_metal_q4k_matmul_parity.c. Set
     * GEIST_METAL_Q4K_MM_SG=0 to disable. */
    const char *q4k_mm_sg = getenv("GEIST_METAL_Q4K_MM_SG");
    st->use_q4k_mm_sg = q4k_mm_sg == nullptr || strcmp(q4k_mm_sg, "0") != 0;
    const char *ple_proj_norm_fused =
        getenv("GEIST_METAL_PLE_PROJ_NORM_FUSED");
    st->use_ple_proj_norm_fused =
        ple_proj_norm_fused != nullptr &&
        strcmp(ple_proj_norm_fused, "1") == 0;
    const char *rmsnorm_simd = getenv("GEIST_METAL_RMSNORM_SIMD");
    st->use_rmsnorm_simd =
        rmsnorm_simd == nullptr || strcmp(rmsnorm_simd, "0") != 0;
    const char *q4k_nt4_large = getenv("GEIST_METAL_Q4K_NT4_LARGE");
    st->q4k_nt4_max_n_out =
        q4k_nt4_large != nullptr && strcmp(q4k_nt4_large, "1") == 0
            ? METAL_Q4K_NT4_LARGE_MAX_N_OUT
            : METAL_Q4K_NT4_DEFAULT_MAX_N_OUT;
    const char *pack_cache_dir = getenv("GEIST_METAL_PACK_CACHE_DIR");
    if (pack_cache_dir != nullptr && pack_cache_dir[0] != '\0' &&
        strlen(pack_cache_dir) < sizeof(st->pack_cache_dir)) {
        struct stat cache_stat = {0};
        bool cache_dir_ok = false;
        if (stat(pack_cache_dir, &cache_stat) == 0) {
            cache_dir_ok = S_ISDIR(cache_stat.st_mode);
        } else if (errno == ENOENT && mkdir(pack_cache_dir, 0700) == 0) {
            cache_dir_ok = true;
        }
        errno = 0;
        if (cache_dir_ok) {
            memcpy(st->pack_cache_dir, pack_cache_dir,
                   strlen(pack_cache_dir) + 1u);
            st->pack_cache_enabled = true;
        }
    }
    const char *q6k_n4 = getenv("GEIST_METAL_Q6K_N4");
    st->use_q6k_n4 = q6k_n4 == nullptr || strcmp(q6k_n4, "0") != 0;
    /* Off by default: the plain-layout n4 kernel (llama mul_mv structure)
     * outruns the packed nt4 path and needs no load-time repack. */
    const char *q6k_nt4 = getenv("GEIST_METAL_Q6K_NT4");
    st->use_q6k_nt4 = q6k_nt4 != nullptr && strcmp(q6k_nt4, "1") == 0;
    const char *q6k_nt8 = getenv("GEIST_METAL_Q6K_NT8");
    st->use_q6k_nt8 = q6k_nt8 != nullptr && strcmp(q6k_nt8, "1") == 0;
    const char *q6k_linear_raw = getenv("GEIST_METAL_Q6K_LINEAR_RAW");
    st->use_q6k_linear_raw =
        q6k_linear_raw != nullptr && strcmp(q6k_linear_raw, "1") == 0;
    const char *decode_replay = getenv("GEIST_METAL_DECODE_REPLAY");
    st->decode_replay_enabled =
        decode_replay == nullptr || strcmp(decode_replay, "0") != 0;
    st->profile_enabled = metal_env_enabled("GEIST_METAL_PROFILE");

    enum geist_status s = metal_load_runtime(be, st);
    if (s != GEIST_OK) {
        metal_destroy_state(be, st);
        return s;
    }

    st->device = metal_create_default_device(st);
    if (st->device == nullptr) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED,
                                "metal: no default Metal device");
        metal_destroy_state(be, st);
        return GEIST_E_UNSUPPORTED;
    }
    metal_msg_send_void0(st, st->device, "retain");

    st->command_queue = metal_msg_send_id0(st, st->device, "newCommandQueue");
    if (st->command_queue == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "metal: failed to create command queue");
        metal_destroy_state(be, st);
        return GEIST_E_BACKEND;
    }

    void *name = metal_msg_send_id0(st, st->device, "name");
    const char *utf8 = name != nullptr
                           ? metal_msg_send_cstr0(st, name, "UTF8String")
                           : nullptr;
    snprintf(st->device_name, sizeof(st->device_name), "%s",
             utf8 != nullptr && utf8[0] != '\0' ? utf8 : "Apple Metal GPU");

    be->state = st;
    return GEIST_OK;
}

static enum geist_support metal_supports_op(
    struct geist_backend *be,
    const struct geist_op_support_query *query) {

    (void) be;
    if (query == nullptr || query->op != GEIST_OP_LINEAR ||
        query->input_count < 2) {
        return GEIST_SUPPORT_NONE;
    }
    const struct geist_tensor_format *x_fmt = &query->inputs[0];
    const struct geist_tensor_format *w_fmt = &query->inputs[1];
    if (x_fmt->dtype == GEIST_DTYPE_F32 &&
        x_fmt->layout == GEIST_LAYOUT_DENSE &&
        (w_fmt->dtype == GEIST_DTYPE_Q4_K ||
         w_fmt->dtype == GEIST_DTYPE_Q6_K) &&
        w_fmt->layout == GEIST_LAYOUT_BLOCK_QUANTIZED) {
        return GEIST_SUPPORT_NATIVE;
    }
    return GEIST_SUPPORT_NONE;
}

/* Resolver-installed GPU linear (main contract). main's engine passes raw
 * host pointers (buffer_map aliases; w->raw = buffer_map(weight buf) +
 * view offset). On Apple unified memory every such pointer lives inside a
 * metal-created SHARED MTLBuffer, so translate them back to (buffer,
 * offset) via the registry and dispatch the existing GEMM ops (mm_sg /
 * q6k / f32 simdgroup kernels; standalone command buffer, commit + wait).
 * rows==1 routes to the matvec kernels inside metal_matmul_*. The kernel
 * signature has no error path: failures report to stderr and zero y so a
 * defect is loud in the token-parity gate rather than silent garbage. */
static void metal_linear_debug_stats(const float *x, size_t nx,
                                     const float *y, size_t ny,
                                     const struct geist_weight *w, size_t m);

static void metal_linear_mN(const float *x, const struct geist_weight *w,
                            size_t m, struct geist_backend *be, float *y) {
    struct metal_state *st = be->state;
    const size_t n_in = (size_t) w->n_in;
    const size_t n_out = (size_t) w->n_out;
    size_t xo = 0;
    size_t wo = 0;
    size_t yo = 0;
    struct geist_buffer *bx = metal_buf_reg_find(st, x, &xo);
    struct geist_buffer *bw = metal_buf_reg_find(st, w->raw, &wo);
    struct geist_buffer *by = metal_buf_reg_find(st, y, &yo);
    if (bx == nullptr || bw == nullptr || by == nullptr) {
        /* One of the pointers is plain host memory (the engine passes heap
         * scratch for small helper projections, e.g. single-row views).
         * Compute on host — same math as cpu_scalar (dequant row, double
         * accumulator), so token parity with the CPU backends holds. The
         * inputs may be GPU-pending, so drain the batch first. */
        metal_batch_flush(st);
        float *row = malloc(n_in * sizeof(float));
        if (row == nullptr) {
            memset(y, 0, m * n_out * sizeof(float));
            return;
        }
        const uint8_t *base = (const uint8_t *) w->raw;
        for (size_t j = 0; j < n_out; j++) {
            switch ((enum geist_dtype) w->dtype) {
            case GEIST_DTYPE_F32:
                memcpy(row, base + j * n_in * sizeof(float),
                       n_in * sizeof(float));
                break;
            case GEIST_DTYPE_Q4_K:
                dequant_q4_K_row(
                    base + j * n_in / Q4_K_BLOCK_ELEMS * Q4_K_BLOCK_BYTES,
                    row, n_in);
                break;
            case GEIST_DTYPE_Q6_K:
                dequant_q6_K_row(
                    base + j * n_in / Q6_K_BLOCK_ELEMS * Q6_K_BLOCK_BYTES,
                    row, n_in);
                break;
            default:
                memset(row, 0, n_in * sizeof(float));
                break;
            }
            for (size_t i = 0; i < m; i++) {
                double acc = 0.0;
                for (size_t k = 0; k < n_in; k++) {
                    acc += (double) x[i * n_in + k] * (double) row[k];
                }
                y[i * n_out + j] = (float) acc;
            }
        }
        free(row);
        metal_linear_debug_stats(x, m * n_in, y, m * n_out, w, m);
        return;
    }
    struct geist_tensor tx = {
        .buffer = bx,
        .offset = xo,
        .dtype = GEIST_DTYPE_F32,
        .layout = GEIST_LAYOUT_DENSE,
        .ndim = 2,
        .shape = {(int64_t) m, (int64_t) n_in},
        .stride = {(int64_t) n_in, 1},
    };
    struct geist_tensor ty = {
        .buffer = by,
        .offset = yo,
        .dtype = GEIST_DTYPE_F32,
        .layout = GEIST_LAYOUT_DENSE,
        .ndim = 2,
        .shape = {(int64_t) m, (int64_t) n_out},
        .stride = {(int64_t) n_out, 1},
    };
    struct geist_tensor tw = {
        .buffer = bw,
        .offset = wo,
        .dtype = (enum geist_dtype) w->dtype,
        .layout = GEIST_LAYOUT_BLOCK_QUANTIZED,
        .ndim = 2,
        .shape = {(int64_t) n_out, (int64_t) n_in},
        .stride = {0, 0},
    };
    enum geist_status s;
    switch ((enum geist_dtype) w->dtype) {
    case GEIST_DTYPE_Q4_K:
        s = metal_matmul_q4k(be, &tx, &tw, &ty);
        break;
    case GEIST_DTYPE_Q6_K:
        s = metal_matmul_q6k(be, &tx, &tw, &ty);
        break;
    case GEIST_DTYPE_F32:
        tw.layout = GEIST_LAYOUT_DENSE;
        tw.stride[0] = (int64_t) n_in;
        tw.stride[1] = 1;
        s = metal_matmul_f32_dense(be, &tx, &tw, &ty);
        break;
    default:
        s = GEIST_E_UNSUPPORTED;
        break;
    }
    if (s != GEIST_OK) {
        fprintf(stderr,
                "geist metal: linear dispatch failed "
                "(dtype=%u m=%zu %zux%zu xo=%zu wo=%zu yo=%zu): %s\n",
                (unsigned) w->dtype, m, n_out, n_in, xo, wo, yo,
                geist_backend_errmsg(be));
        memset(y, 0, m * n_out * sizeof(float));
    }
    metal_linear_debug_stats(x, m * n_in, y, m * n_out, w, m);
}

static void metal_linear_m1(const float *x, const struct geist_weight *w,
                            struct geist_backend *be, float *y) {
    metal_linear_mN(x, w, 1, be, y);
}

/* GEIST_METAL_DEBUG_LINEAR=1: per-linear x/y absmax trace. Every layer
 * stage flows through linear, so the first all-zero input pinpoints which
 * in-between op (rmsnorm/rope/attention/add/mul) lost the data. */
static void metal_linear_debug_stats(const float *x, size_t nx,
                                     const float *y, size_t ny,
                                     const struct geist_weight *w, size_t m) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *e = getenv("GEIST_METAL_DEBUG_LINEAR");
        enabled = (e != nullptr && e[0] != '\0' && strcmp(e, "0") != 0) ? 1 : 0;
    }
    if (!enabled) {
        return;
    }
    float ax = 0.0f;
    float ay = 0.0f;
    size_t nanx = 0;
    size_t nany = 0;
    for (size_t i = 0; i < nx; i++) {
        if (isnan(x[i])) { nanx++; continue; }
        const float a = fabsf(x[i]);
        if (a > ax) { ax = a; }
    }
    for (size_t i = 0; i < ny; i++) {
        if (isnan(y[i])) { nany++; continue; }
        const float a = fabsf(y[i]);
        if (a > ay) { ay = a; }
    }
    fprintf(stderr, "linear dtype=%u m=%zu %dx%d |x|=%g |y|=%g nanx=%zu nany=%zu\n",
            (unsigned) w->dtype, m, w->n_out, w->n_in, (double) ax,
            (double) ay, nanx, nany);
}

/* Tensor-based linear (main's optional vtbl slot): dispatch the GEMM from
 * the engine's existing tensor views — no host pointers, so the op encodes
 * onto the open batch when one is active. UNSUPPORTED falls back to the
 * resolved host-pointer kernels. */
[[nodiscard]] static enum geist_status metal_linear_t(
    struct geist_backend *be,
    const struct geist_tensor *x,
    const struct geist_weight *w,
    const struct geist_tensor *t_w,
    size_t m,
    struct geist_tensor *y) {

    if (w == nullptr || t_w == nullptr || x == nullptr || y == nullptr) {
        return GEIST_E_UNSUPPORTED;
    }
    /* m == 1 (decode) routes to the matvec ops — the GEMM tile kernels are
     * an order of magnitude slower for a single row than the llama-style
     * mul_mv kernels. The engine passes [1, n] 2D views; rebuild them 1D. */
    if (m == 1) {
        struct geist_tensor x1 = *x;
        struct geist_tensor y1 = *y;
        if (x1.ndim == 2 && x1.shape[0] == 1) {
            x1.ndim = 1;
            x1.shape[0] = x1.shape[1];
            x1.stride[0] = x1.stride[1];
            x1.shape[1] = 0;
            x1.stride[1] = 0;
        }
        if (y1.ndim == 2 && y1.shape[0] == 1) {
            y1.ndim = 1;
            y1.shape[0] = y1.shape[1];
            y1.stride[0] = y1.stride[1];
            y1.shape[1] = 0;
            y1.stride[1] = 0;
        }
        switch ((enum geist_dtype) w->dtype) {
        case GEIST_DTYPE_Q4_K:
            return metal_matvec_q4k(be, &x1, t_w, &y1);
        case GEIST_DTYPE_Q6_K:
            return metal_matvec_q6k(be, &x1, t_w, &y1);
        case GEIST_DTYPE_F32:
            return metal_matvec_f32_dense(be, &x1, t_w, &y1);
        default:
            return GEIST_E_UNSUPPORTED;
        }
    }
    switch ((enum geist_dtype) w->dtype) {
    case GEIST_DTYPE_Q4_K:
        return metal_matmul_q4k(be, x, t_w, y);
    case GEIST_DTYPE_Q6_K:
        return metal_matmul_q6k(be, x, t_w, y);
    case GEIST_DTYPE_F32:
        return metal_matmul_f32_dense(be, x, t_w, y);
    default:
        return GEIST_E_UNSUPPORTED;
    }
}

/* Batched-submit region hooks. main brackets each prefill batch and each
 * decode step with these; we open one command buffer per region and encode
 * every op onto it. Host access to GPU-referenced buffers flushes early
 * (see metal_flush_if_referenced). The engine treats the token as opaque;
 * flushes rotate st->sequence_token, so region_end closes the CURRENT
 * sequence, not the original token. */
static int metal_parallel_region_begin(struct geist_backend *be,
                                       enum geist_parallel_region region) {
    if (be == nullptr || be->state == nullptr) {
        return 0;
    }
    struct metal_state *st = be->state;
    if (st->sequence_active) {
        return 0; /* nested region: leave the outer batch in charge */
    }
    const enum geist_command_sequence_kind kind =
        region == GEIST_REGION_PREFILL_BATCH
            ? GEIST_COMMAND_SEQUENCE_PREFILL_TEXT
            : GEIST_COMMAND_SEQUENCE_DECODE_LAYER_LOOP;
    int tok = 0;
    if (metal_command_sequence_begin(be, kind, &tok) != GEIST_OK) {
        return 0;
    }
    metal_seq_ref_clear(st);
    return tok;
}

static void metal_parallel_region_end(struct geist_backend *be, int token) {
    if (be == nullptr || be->state == nullptr || token == 0) {
        return;
    }
    struct metal_state *st = be->state;
    if (!st->sequence_active) {
        return;
    }
    (void) metal_command_sequence_end(be, st->sequence_token, true);
    metal_seq_ref_clear(st);
}

[[nodiscard]] static enum geist_status metal_resolve_weight(
    struct geist_backend *be, struct geist_weight *w) {
    (void) be;
    if (w == nullptr || w->raw == nullptr || w->n_in <= 0 || w->n_out <= 0) {
        return GEIST_E_INVALID_ARG;
    }
    switch ((enum geist_dtype) w->dtype) {
    case GEIST_DTYPE_Q4_K:
    case GEIST_DTYPE_Q6_K:
    case GEIST_DTYPE_F32:
        w->linear_m1 = metal_linear_m1;
        w->linear_mN = metal_linear_mN;
        return GEIST_OK;
    default:
        /* Callers fall back per weight (linear_m1 stays null). */
        return GEIST_E_UNSUPPORTED;
    }
}

/* main-contract vtbl. The old fine-grained GPU ops (command_sequence_*,
 * ffn_geglu_block, ple_block, attention_block, greedy_head, matmul_q4k)
 * are not part of main's contract; their impls remain in this file as
 * internal/dead code (behind metal_legacy_ops.h) pending the Stage-6
 * cleanup, but are not exposed here. */
static const struct geist_backend_vtbl metal_vtbl = {
    .create = metal_create,
    .destroy = metal_destroy,
    .supports_op = metal_supports_op,
    .buffer_create = metal_buffer_create,
    .buffer_destroy = metal_buffer_destroy,
    .buffer_create_aliased = metal_buffer_create_aliased,
    .buffer_upload = metal_buffer_upload,
    .buffer_download = metal_buffer_download,
    .buffer_map = metal_buffer_map,
    .buffer_unmap = metal_buffer_unmap,
    .resolve_weight = metal_resolve_weight,
    .rmsnorm = metal_rmsnorm,
    .add = metal_add,
    .mul = metal_mul,
    .gelu_tanh = metal_gelu_tanh,
    .gelu_tanh_mul = metal_gelu_tanh_mul,
    .gelu_tanh_mul_scaled = nullptr,
    .relu_squared = nullptr,
    .silu = nullptr,
    .rope_apply = metal_rope_apply,
    .embedding_lookup = metal_embedding_lookup,
    .attention = metal_attention,
    .ffn_geglu_q4q6_mN = nullptr,
    .transformer_block = nullptr,
    .parallel_region_begin = metal_parallel_region_begin,
    .parallel_region_end = metal_parallel_region_end,
    .linear_t = metal_linear_t,
    .buffer_copy = metal_buffer_copy,
    .scale_f32 = metal_scale_f32,
};

const struct geist_backend_descriptor geist_backend_metal = {
    .name = "metal",
    .vtbl = &metal_vtbl,
    .caps = nullptr,
    .n_caps = 0,
};

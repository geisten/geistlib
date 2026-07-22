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
    METAL_PROFILE_DISPATCH_ADD_ROWS,
    METAL_PROFILE_DISPATCH_MUL_ROWS,
    METAL_PROFILE_DISPATCH_SCALE_ROWS,
    METAL_PROFILE_DISPATCH_GELU_ROWS,
    METAL_PROFILE_DISPATCH_COPY_U32,
    METAL_PROFILE_DISPATCH_ARGMAX,
    METAL_PROFILE_STAGE_COUNT,
};

static const char *const metal_profile_stage_names[METAL_PROFILE_STAGE_COUNT] = {
        [METAL_PROFILE_WAIT_DECODE_LAYER_LOOP]        = "wait.decode_layer_loop",
        [METAL_PROFILE_WAIT_DECODE_GREEDY_STEP]       = "wait.decode_greedy_step",
        [METAL_PROFILE_WAIT_VERIFY_GREEDY]            = "wait.verify_greedy",
        [METAL_PROFILE_WAIT_PREFILL_TEXT]             = "wait.prefill_text",
        [METAL_PROFILE_WAIT_FFN_STANDALONE]           = "wait.ffn_standalone",
        [METAL_PROFILE_DISPATCH_RMSNORM_ROWS]         = "dispatch.rmsnorm_rows",
        [METAL_PROFILE_DISPATCH_Q4K_GATE_UP_BASE]     = "dispatch.q4k_gate_up.base",
        [METAL_PROFILE_DISPATCH_Q4K_GATE_UP_N4]       = "dispatch.q4k_gate_up.n4",
        [METAL_PROFILE_DISPATCH_Q4K_GATE_UP_NT4]      = "dispatch.q4k_gate_up.nt4",
        [METAL_PROFILE_DISPATCH_Q4K_GATE_UP_NT8]      = "dispatch.q4k_gate_up.nt8",
        [METAL_PROFILE_DISPATCH_Q4K_GATE_UP_W4A8]     = "dispatch.q4k_gate_up.w4a8",
        [METAL_PROFILE_DISPATCH_Q4K_LINEAR_BASE]      = "dispatch.q4k_linear.base",
        [METAL_PROFILE_DISPATCH_Q4K_LINEAR_N4]        = "dispatch.q4k_linear.n4",
        [METAL_PROFILE_DISPATCH_Q4K_LINEAR_NT4]       = "dispatch.q4k_linear.nt4",
        [METAL_PROFILE_DISPATCH_Q4K_LINEAR_NT8]       = "dispatch.q4k_linear.nt8",
        [METAL_PROFILE_DISPATCH_Q4K_LINEAR_W4A8]      = "dispatch.q4k_linear.w4a8",
        [METAL_PROFILE_DISPATCH_Q4K_PLE_GATE_NT8]     = "dispatch.q4k_ple_gate.nt8",
        [METAL_PROFILE_DISPATCH_F32_PLE_GATE]         = "dispatch.f32_ple_gate",
        [METAL_PROFILE_DISPATCH_Q4K_QUANT_X]          = "dispatch.q4k_quant_x",
        [METAL_PROFILE_DISPATCH_Q6K_LINEAR_BASE]      = "dispatch.q6k_linear.base",
        [METAL_PROFILE_DISPATCH_Q6K_LINEAR_N4]        = "dispatch.q6k_linear.n4",
        [METAL_PROFILE_DISPATCH_Q6K_LINEAR_NT4]       = "dispatch.q6k_linear.nt4",
        [METAL_PROFILE_DISPATCH_Q6K_LINEAR_NT8]       = "dispatch.q6k_linear.nt8",
        [METAL_PROFILE_DISPATCH_Q4K_QK_BASE]          = "dispatch.q4k_qk.base",
        [METAL_PROFILE_DISPATCH_Q4K_QK_NT4]           = "dispatch.q4k_qk.nt4",
        [METAL_PROFILE_DISPATCH_F32_PLE_PROJ_NORM]    = "dispatch.f32_ple_proj_norm",
        [METAL_PROFILE_DISPATCH_RMSNORM_ADD_ROWS]     = "dispatch.rmsnorm_add_rows",
        [METAL_PROFILE_DISPATCH_Q_NORM_ROPE]          = "dispatch.q_norm_rope",
        [METAL_PROFILE_DISPATCH_K_NORM_ROPE_APPEND]   = "dispatch.k_norm_rope_append",
        [METAL_PROFILE_DISPATCH_V_NORM_APPEND]        = "dispatch.v_norm_append",
        [METAL_PROFILE_DISPATCH_KV_NORM_APPEND]       = "dispatch.kv_norm_append",
        [METAL_PROFILE_DISPATCH_ROPE_ROWS]            = "dispatch.rope_rows",
        [METAL_PROFILE_DISPATCH_KV_APPEND_ROWS]       = "dispatch.kv_append_rows",
        [METAL_PROFILE_DISPATCH_ATTENTION_ROWS]       = "dispatch.attention_rows",
        [METAL_PROFILE_DISPATCH_ATTENTION_QNORM_ROWS] = "dispatch.attention_qnorm_rows",
        [METAL_PROFILE_DISPATCH_GELU_MUL_ROWS]        = "dispatch.gelu_mul_rows",
        [METAL_PROFILE_DISPATCH_F32_MATMUL]           = "dispatch.f32_matmul",
        [METAL_PROFILE_DISPATCH_EMBED]                = "dispatch.embed",
        [METAL_PROFILE_DISPATCH_ADD_ROWS]             = "dispatch.add_rows",
        [METAL_PROFILE_DISPATCH_MUL_ROWS]             = "dispatch.mul_rows",
        [METAL_PROFILE_DISPATCH_SCALE_ROWS]           = "dispatch.scale_rows",
        [METAL_PROFILE_DISPATCH_GELU_ROWS]            = "dispatch.gelu_rows",
        [METAL_PROFILE_DISPATCH_COPY_U32]             = "dispatch.copy_u32",
        [METAL_PROFILE_DISPATCH_ARGMAX]               = "dispatch.argmax",
};

struct metal_profile_stat {
    uint64_t ns;
    uint64_t calls;
    uint64_t workgroups;
};

/* Registry entry mapping a live buffer's host contents range back to its
 * geist_buffer, so resolver-installed linear kernels can translate the raw
 * host pointers main's engine passes (buffer_map aliases, w->raw) into
 * (MTLBuffer, offset) pairs for GPU dispatch. */
struct metal_buf_reg_entry {
    const uint8_t       *base;
    size_t               bytes;
    struct geist_buffer *buf;
};

struct metal_state {
    struct geist_backend       *backend;
    void                       *metal_handle;
    void                       *objc_handle;
    void                       *device;
    void                       *command_queue;
    struct metal_buf_reg_entry *buf_reg;
    size_t                      buf_reg_count;
    size_t                      buf_reg_cap;
    /* MTLBuffers referenced by ops encoded on the open (unflushed) batch;
     * a host map/upload/download of a referenced buffer forces a flush.
     * Open-addressed pointer set; overflow degrades to always-flush. */
    void    *seq_ref[4096];
    size_t   seq_ref_count;
    bool     seq_ref_overflow;
    void    *q4k_library;
    void    *q4k_n4_library;
    void    *q6k_library;
    void    *q6k_n4_library;
    void    *elem_library;
    void    *elem_simd_library;
    void    *attn_library;
    void    *attn_f16_library;
    void    *q_norm_rope_library;
    void    *k_norm_rope_append_library;
    void    *v_norm_append_library;
    void    *kv_norm_append_library;
    void    *kv_norm_append_f16_library;
    void    *q4k_function;
    void    *q4k_pipeline;
    void    *q4k_n4_function;
    void    *q4k_n4_pipeline;
    void    *q4k_matmul_m8_function;
    void    *q4k_matmul_m8_pipeline;
    void    *q4k_m16_library;
    void    *q4k_matmul_m16_function;
    void    *q4k_matmul_m16_pipeline;
    void    *q4k_m16_n2_library;
    void    *q4k_matmul_m16_n2_function;
    void    *q4k_matmul_m16_n2_pipeline;
    void    *q4k_mm_sg_library;
    void    *q4k_mm_sg_function;
    void    *q4k_mm_sg_pipeline;
    void    *q4k_mm_sg_fast_library;
    void    *q4k_mm_sg_fast_function;
    void    *q4k_mm_sg_fast_pipeline;
    void    *q4k_qk_library;
    void    *q4k_qk_function;
    void    *q4k_qk_pipeline;
    void    *q4k_gate_up_library;
    void    *q4k_gate_up_n4_library;
    void    *q4k_pair_n4_library;
    void    *q4k_pair_n4_function;
    void    *q4k_pair_n4_pipeline;
    void    *embed_library;
    void    *argmax_library;
    void    *q4k_gate_up_function;
    void    *q4k_gate_up_pipeline;
    void    *q4k_gate_up_n4_function;
    void    *q4k_gate_up_n4_pipeline;
    void    *q6k_function;
    void    *q6k_pipeline;
    void    *q6k_n4_function;
    void    *q6k_n4_pipeline;
    void    *q6k_matmul_m8_function;
    void    *q6k_matmul_m8_pipeline;
    void    *q6k_mm_sg_library;
    void    *q6k_matmul_sg_function;
    void    *q6k_matmul_sg_pipeline;
    void    *q6k_mm_sg_fast_library;
    void    *q6k_matmul_sg_fast_function;
    void    *q6k_matmul_sg_fast_pipeline;
    void    *q6k_m16_library;
    void    *q6k_matmul_m16_function;
    void    *q6k_matmul_m16_pipeline;
    void    *rmsnorm_rows_function;
    void    *rmsnorm_rows_pipeline;
    void    *rmsnorm_rows_simd_function;
    void    *rmsnorm_rows_simd_pipeline;
    void    *gelu_rows_function;
    void    *gelu_rows_pipeline;
    void    *mul_rows_function;
    void    *mul_rows_pipeline;
    void    *gelu_mul_rows_function;
    void    *gelu_mul_rows_pipeline;
    void    *add_rows_function;
    void    *add_rows_pipeline;
    void    *scale_rows_function;
    void    *scale_rows_pipeline;
    void    *rmsnorm_add_rows_function;
    void    *rmsnorm_add_rows_pipeline;
    void    *rmsnorm_add_rows_simd_function;
    void    *rmsnorm_add_rows_simd_pipeline;
    void    *embed_lookup_scaled_function;
    void    *embed_lookup_scaled_pipeline;
    void    *f32_library;
    void    *f32_matmul_function;
    void    *f32_matmul_pipeline;
    void    *f32_matmul_sg_function;
    void    *f32_matmul_sg_pipeline;
    void    *f32_matmul_mm_function;
    void    *f32_matmul_mm_pipeline;
    void    *f32_ple_gate_function;
    void    *f32_ple_gate_pipeline;
    void    *f32_ple_proj_norm_function;
    void    *f32_ple_proj_norm_pipeline;
    void    *argmax_function;
    void    *argmax_pipeline;
    void    *argmax_batch_function;
    void    *argmax_batch_pipeline;
    void    *argmax_result_buffer;
    void    *argmax_result_mapped;
    uint32_t argmax_result_capacity;
    void    *rope_rows_function;
    void    *rope_rows_pipeline;
    void    *kv_append_rows_function;
    void    *kv_append_rows_pipeline;
    void    *copy_u32_function;
    void    *copy_u32_pipeline;
    void    *kv_append_rows_f16_function;
    void    *kv_append_rows_f16_pipeline;
    void    *q_norm_rope_rows_function;
    void    *q_norm_rope_rows_pipeline;
    void    *k_norm_rope_append_rows_function;
    void    *k_norm_rope_append_rows_pipeline;
    void    *k_norm_rope_append_rows_f16_function;
    void    *k_norm_rope_append_rows_f16_pipeline;
    void    *v_norm_append_rows_function;
    void    *v_norm_append_rows_pipeline;
    void    *v_norm_append_rows_f16_function;
    void    *v_norm_append_rows_f16_pipeline;
    void    *kv_norm_append_rows_function;
    void    *kv_norm_append_rows_pipeline;
    void    *kv_norm_append_rows_f16_function;
    void    *kv_norm_append_rows_f16_pipeline;
    void    *attention_rows_function;
    void    *attention_rows_pipeline;
    void    *attention_rows_f16_function;
    void    *attention_rows_f16_pipeline;
    void    *attn_qnorm_dec_f16_library;
    void    *attention_qnorm_dec_f16_function;
    void    *attention_qnorm_dec_f16_pipeline;
    void    *attention_dec_f16_function;
    void    *attention_dec_f16_pipeline;
    void    *attn_flash_sg_f16_library;
    void    *attention_qnorm_flash_sg_f16_function;
    void    *attention_qnorm_flash_sg_f16_pipeline;
    void    *attention_flash_sg_f16_function;
    void    *attention_flash_sg_f16_pipeline;
    void    *attention_dec512_f16_function;
    void    *attention_dec512_f16_pipeline;
    void    *attn_dec512_f16_library;
    void    *attention_flash_sg8_f16_function;
    void    *attention_flash_sg8_f16_pipeline;
    void    *attn_flash_sg8_f16_library;
    /* persistent f32->f16 K/V staging for the plain flash path (main's
     * engine keeps the KV cache f32). */
    struct geist_buffer *attn_kf16_buffer;
    struct geist_buffer *attn_vf16_buffer;
    size_t               attn_kvf16_capacity;
    void                *attn_dec_combine_library;
    void                *attention_dec_combine_function;
    void                *attention_dec_combine_pipeline;
    struct geist_buffer *attn_dec_partials_buffer;
    size_t               attn_dec_partials_capacity;
    void                *sequence_command_buffer;
    void                *sequence_compute_encoder;
    void                *capture_manager;
    bool                 capture_done;
    int                  capture_skipped;
    /* diag: GEIST_METAL_SEQ_TRACE=1 — per-sequence encode/GPU timing. */
    uint64_t seq_dispatch_count;
    /* Command-buffer pipelining (llama.cpp n_cb-style): the sequence
     * rotates to a fresh command buffer every seq_rotate_every dispatches,
     * committing the old one WITHOUT waiting — the GPU starts executing
     * (and its front-end starts parsing) buffer k while the CPU still
     * encodes buffer k+1. Buffers on one queue execute in commit order,
     * so cross-buffer data deps hold. 0 = pipelining off. */
    uint32_t                         seq_rotate_every;
    uint32_t                         seq_disp_at_rotate;
    uint32_t                         seq_pending_count;
    void                            *seq_pending_cmds[16];
    uint64_t                         seq_begin_ns;
    int                              sequence_token;
    enum geist_command_sequence_kind sequence_kind;
    bool                             sequence_active;
    bool                             sequence_has_work;
    bool                             use_ple_block;
    bool                             use_q4k_n4;
    bool                             use_q4k_m16_n2;
    bool                             use_q4k_mm_sg;
    bool                             use_rmsnorm_simd;
    bool                             use_q6k_n4;
    bool                             profile_enabled;
    bool                      skip_next_dispatch; /* subtractive profiler: drop the next dispatch */
    struct metal_profile_stat profile[METAL_PROFILE_STAGE_COUNT];
    char                      device_name[128];

    void *MTLCreateSystemDefaultDevice;
    void *objc_msgSend;
    void *sel_registerName;
    void *objc_getClass;
};

struct geist_buffer {
    struct metal_state    *owner;
    void                  *buffer;
    void                  *mapped;
    size_t                 bytes;
    enum geist_buffer_role role;
    unsigned int           memory_flags;
    bool                   host_visible;
};

enum {
    METAL_RESOURCE_STORAGE_MODE_SHARED  = 0u,
    METAL_RESOURCE_STORAGE_MODE_PRIVATE = 2u << 4,
    METAL_Q4K_THREADS_PER_ROW           = 256u,
    METAL_Q4K_N4_THREADS                = 64u,
    METAL_Q4K_BLOCK_ELEMS               = 256u,
    METAL_Q4K_BLOCK_BYTES               = 144u,
    METAL_Q4K_NT4_DEFAULT_MAX_N_OUT     = 8192u,
    METAL_Q4K_NT4_LARGE_MAX_N_OUT       = 262144u,
    METAL_Q5K_BLOCK_ELEMS               = 256u,
    METAL_Q5K_BLOCK_BYTES               = 176u,
    METAL_Q6K_BLOCK_ELEMS               = 256u,
    METAL_Q6K_BLOCK_BYTES               = 210u,
    METAL_Q6K_NT4_MIN_N_OUT             = 1024u,
    METAL_Q6K_NT4_MAX_N_OUT             = 8192u,
    METAL_Q4K_M_TILE                    = 8u,
    METAL_Q4K_M16_TILE                  = 16u,
    METAL_ELEM_THREADS                  = 256u,
    METAL_QNORM_ATTENTION_MAX_HEAD_DIM  = 512u,
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
    float    eps;
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
    float    scale;
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
    float    eps;
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
    float    eps;
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
    float    eps;
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
    float    eps;
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
    float    scale;
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
    float    eps;
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

#include "metal_shaders.h"
#include "metal_objc.h"

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

static void
metal_profile_add_wait(struct metal_state *st, enum metal_profile_stage stage, uint64_t start_ns) {
    if (st == nullptr || !st->profile_enabled || stage >= METAL_PROFILE_STAGE_COUNT ||
        start_ns == 0) {
        return;
    }
    const uint64_t end_ns = metal_now_ns();
    if (end_ns < start_ns) {
        return;
    }
    struct metal_profile_stat *stat = &st->profile[stage];
    stat->ns                        = metal_saturating_add_u64(stat->ns, end_ns - start_ns);
    stat->calls                     = metal_saturating_add_u64(stat->calls, 1);
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
        return metal_env_enabled("GEIST_SKIP_GEMM") || metal_env_enabled("GEIST_SKIP_GATE_UP");
    case METAL_PROFILE_DISPATCH_Q4K_LINEAR_BASE:
    case METAL_PROFILE_DISPATCH_Q4K_LINEAR_N4:
    case METAL_PROFILE_DISPATCH_Q4K_LINEAR_NT4:
    case METAL_PROFILE_DISPATCH_Q4K_LINEAR_NT8:
    case METAL_PROFILE_DISPATCH_Q4K_LINEAR_W4A8:
    case METAL_PROFILE_DISPATCH_Q4K_QK_BASE:
    case METAL_PROFILE_DISPATCH_Q4K_QK_NT4:
        return metal_env_enabled("GEIST_SKIP_GEMM") || metal_env_enabled("GEIST_SKIP_Q4K_LINEAR");
    case METAL_PROFILE_DISPATCH_Q4K_PLE_GATE_NT8:
    case METAL_PROFILE_DISPATCH_F32_PLE_GATE:
    case METAL_PROFILE_DISPATCH_F32_PLE_PROJ_NORM:
        return metal_env_enabled("GEIST_SKIP_GEMM") || metal_env_enabled("GEIST_SKIP_PLE");
    case METAL_PROFILE_DISPATCH_Q6K_LINEAR_BASE:
    case METAL_PROFILE_DISPATCH_Q6K_LINEAR_N4:
    case METAL_PROFILE_DISPATCH_Q6K_LINEAR_NT4:
    case METAL_PROFILE_DISPATCH_Q6K_LINEAR_NT8:
        return metal_env_enabled("GEIST_SKIP_GEMM") || metal_env_enabled("GEIST_SKIP_Q6K");
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
    case METAL_PROFILE_DISPATCH_ADD_ROWS:
    case METAL_PROFILE_DISPATCH_MUL_ROWS:
    case METAL_PROFILE_DISPATCH_SCALE_ROWS:
    case METAL_PROFILE_DISPATCH_GELU_ROWS:
        return metal_env_enabled("GEIST_SKIP_ELEM");
    case METAL_PROFILE_DISPATCH_COPY_U32:
        return metal_env_enabled("GEIST_SKIP_COPY");
    default:
        return false;
    }
}

static void metal_profile_add_dispatch(struct metal_state      *st,
                                       enum metal_profile_stage stage,
                                       struct metal_size        groups) {
    if (st == nullptr || !st->profile_enabled || stage >= METAL_PROFILE_STAGE_COUNT) {
        return;
    }
    if (metal_skip_stage(stage) && metal_skip_grid_match(groups)) {
        st->skip_next_dispatch = true;
    }
    struct metal_profile_stat *stat = &st->profile[stage];
    stat->calls                     = metal_saturating_add_u64(stat->calls, 1);
    stat->workgroups = metal_saturating_add_u64(stat->workgroups, metal_profile_workgroups(groups));
}

static enum metal_profile_stage
metal_profile_wait_stage_for_sequence(enum geist_command_sequence_kind kind) {
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
            fprintf(stderr,
                    "  %-32s %9.2f ms  (%llu calls)\n",
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

static bool metal_ranges_overlap(size_t a_offset, size_t b_offset, size_t n_bytes) {
    if (n_bytes == 0) {
        return false;
    }
    return a_offset < b_offset + n_bytes && b_offset < a_offset + n_bytes;
}

/* Drain the pipelined (committed, unwaited) command buffers: optionally
 * surface their errors, always release. Buffers execute in commit order,
 * so when the caller has waited on the FINAL buffer these are complete. */
static void metal_sequence_drain_pending(struct metal_state *st, bool *out_failed) {
    for (uint32_t i = 0; i < st->seq_pending_count; i++) {
        void *cmd = st->seq_pending_cmds[i];
        if (out_failed != nullptr && metal_msg_send_id0(st, cmd, "error") != nullptr) {
            *out_failed = true;
        }
        metal_msg_send_void0(st, cmd, "release");
        st->seq_pending_cmds[i] = nullptr;
    }
    st->seq_pending_count = 0;
}

/* Op-start encoder accessor with pipelined rotation. Called ONLY at op
 * boundaries — never mid-op, an op's local `enc` must stay valid across
 * its own dispatches. Returns the current encoder (possibly fresh). */
static void *metal_sequence_encoder(struct metal_state *st) {
    if (st == nullptr || !st->sequence_active) {
        return nullptr;
    }
    if (st->seq_rotate_every == 0u || st->sequence_compute_encoder == nullptr ||
        (uint32_t) st->seq_dispatch_count - st->seq_disp_at_rotate < st->seq_rotate_every) {
        return st->sequence_compute_encoder;
    }
    if (st->seq_pending_count >=
        (uint32_t) (sizeof(st->seq_pending_cmds) / sizeof(st->seq_pending_cmds[0]))) {
        /* Backpressure: encode has outrun the GPU by 16 buffers — wait
         * for the oldest before opening another. */
        metal_msg_send_void0(st, st->seq_pending_cmds[0], "waitUntilCompleted");
        metal_msg_send_void0(st, st->seq_pending_cmds[0], "release");
        st->seq_pending_count--;
        memmove(&st->seq_pending_cmds[0],
                &st->seq_pending_cmds[1],
                st->seq_pending_count * sizeof(st->seq_pending_cmds[0]));
    }
    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    void *enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
    if (cmd == nullptr || enc == nullptr) {
        return st->sequence_compute_encoder; /* keep the old buffer */
    }
    metal_msg_send_void0(st, cmd, "retain");
    metal_msg_send_void0(st, enc, "retain");
    metal_msg_send_void0(st, st->sequence_compute_encoder, "endEncoding");
    metal_msg_send_void0(st, st->sequence_command_buffer, "commit");
    metal_msg_send_void0(st, st->sequence_compute_encoder, "release");
    st->seq_pending_cmds[st->seq_pending_count++] =
            st->sequence_command_buffer; /* stays retained until drain */
    st->sequence_command_buffer  = cmd;
    st->sequence_compute_encoder = enc;
    st->seq_disp_at_rotate       = (uint32_t) st->seq_dispatch_count;
    return enc;
}

static void metal_release_sequence_objects(struct metal_state *st) {
    if (st == nullptr) {
        return;
    }
    metal_sequence_drain_pending(st, nullptr);
    metal_msg_send_void0(st, st->sequence_compute_encoder, "release");
    metal_msg_send_void0(st, st->sequence_command_buffer, "release");
    st->sequence_compute_encoder = nullptr;
    st->sequence_command_buffer  = nullptr;
    st->sequence_active          = false;
    st->sequence_has_work        = false;
}

static void metal_buffer_destroy_internal(struct geist_backend *be, struct geist_buffer *buf);

static void metal_destroy_state(struct geist_backend *be, struct metal_state *st) {
    if (be == nullptr || st == nullptr) {
        return;
    }
    metal_profile_print_summary(st);
    free(st->buf_reg);
    st->buf_reg       = nullptr;
    st->buf_reg_count = 0;
    st->buf_reg_cap   = 0;
    metal_buffer_destroy_internal(be, st->attn_dec_partials_buffer);
    st->attn_dec_partials_buffer = nullptr;
    metal_buffer_destroy_internal(be, st->attn_kf16_buffer);
    metal_buffer_destroy_internal(be, st->attn_vf16_buffer);
    st->attn_kf16_buffer           = nullptr;
    st->attn_vf16_buffer           = nullptr;
    st->attn_kvf16_capacity        = 0;
    st->attn_dec_partials_capacity = 0;
    if (st->objc_msgSend != nullptr && st->sel_registerName != nullptr) {
        if (st->sequence_active) {
            metal_msg_send_void0(st, st->sequence_compute_encoder, "endEncoding");
        }
        metal_release_sequence_objects(st);
        metal_msg_send_void0(st, st->argmax_result_buffer, "release");
        metal_msg_send_void0(st, st->argmax_batch_pipeline, "release");
        metal_msg_send_void0(st, st->argmax_batch_function, "release");
        metal_msg_send_void0(st, st->argmax_pipeline, "release");
        metal_msg_send_void0(st, st->argmax_function, "release");
        metal_msg_send_void0(st, st->rmsnorm_add_rows_simd_pipeline, "release");
        metal_msg_send_void0(st, st->rmsnorm_add_rows_simd_function, "release");
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
        metal_msg_send_void0(st, st->attention_qnorm_dec_f16_pipeline, "release");
        metal_msg_send_void0(st, st->attention_qnorm_dec_f16_function, "release");
        metal_msg_send_void0(st, st->attention_dec_f16_pipeline, "release");
        metal_msg_send_void0(st, st->attention_dec_f16_function, "release");
        metal_msg_send_void0(st, st->attention_dec_combine_pipeline, "release");
        metal_msg_send_void0(st, st->attention_dec_combine_function, "release");
        metal_msg_send_void0(st, st->attention_dec512_f16_pipeline, "release");
        metal_msg_send_void0(st, st->attention_dec512_f16_function, "release");
        metal_msg_send_void0(st, st->attn_dec512_f16_library, "release");
        metal_msg_send_void0(st, st->attention_flash_sg8_f16_pipeline, "release");
        metal_msg_send_void0(st, st->attention_flash_sg8_f16_function, "release");
        metal_msg_send_void0(st, st->attn_flash_sg8_f16_library, "release");
        metal_msg_send_void0(st, st->attention_qnorm_flash_sg_f16_pipeline, "release");
        metal_msg_send_void0(st, st->attention_qnorm_flash_sg_f16_function, "release");
        metal_msg_send_void0(st, st->attention_flash_sg_f16_pipeline, "release");
        metal_msg_send_void0(st, st->attention_flash_sg_f16_function, "release");
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
        metal_msg_send_void0(st, st->v_norm_append_rows_f16_pipeline, "release");
        metal_msg_send_void0(st, st->v_norm_append_rows_f16_function, "release");
        metal_msg_send_void0(st, st->kv_norm_append_rows_pipeline, "release");
        metal_msg_send_void0(st, st->kv_norm_append_rows_function, "release");
        metal_msg_send_void0(st, st->kv_norm_append_rows_f16_pipeline, "release");
        metal_msg_send_void0(st, st->kv_norm_append_rows_f16_function, "release");
        metal_msg_send_void0(st, st->k_norm_rope_append_rows_pipeline, "release");
        metal_msg_send_void0(st, st->k_norm_rope_append_rows_function, "release");
        metal_msg_send_void0(st, st->k_norm_rope_append_rows_f16_pipeline, "release");
        metal_msg_send_void0(st, st->k_norm_rope_append_rows_f16_function, "release");
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
        metal_msg_send_void0(st, st->rmsnorm_rows_simd_pipeline, "release");
        metal_msg_send_void0(st, st->rmsnorm_rows_simd_function, "release");
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
        metal_msg_send_void0(st, st->q6k_n4_pipeline, "release");
        metal_msg_send_void0(st, st->q6k_n4_function, "release");
        metal_msg_send_void0(st, st->q4k_qk_pipeline, "release");
        metal_msg_send_void0(st, st->q4k_qk_function, "release");
        metal_msg_send_void0(st, st->q4k_gate_up_pipeline, "release");
        metal_msg_send_void0(st, st->q4k_gate_up_function, "release");
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
        metal_msg_send_void0(st, st->q4k_gate_up_n4_pipeline, "release");
        metal_msg_send_void0(st, st->q4k_gate_up_n4_function, "release");
        metal_msg_send_void0(st, st->q4k_pair_n4_pipeline, "release");
        metal_msg_send_void0(st, st->q4k_pair_n4_function, "release");
        metal_msg_send_void0(st, st->q4k_n4_pipeline, "release");
        metal_msg_send_void0(st, st->q4k_n4_function, "release");
        metal_msg_send_void0(st, st->q4k_pipeline, "release");
        metal_msg_send_void0(st, st->q4k_function, "release");
        metal_msg_send_void0(st, st->attn_library, "release");
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
        metal_msg_send_void0(st, st->q4k_gate_up_n4_library, "release");
        metal_msg_send_void0(st, st->q4k_pair_n4_library, "release");
        metal_msg_send_void0(st, st->q4k_gate_up_library, "release");
        metal_msg_send_void0(st, st->q4k_n4_library, "release");
        metal_msg_send_void0(st, st->q6k_mm_sg_library, "release");
        metal_msg_send_void0(st, st->q6k_mm_sg_fast_library, "release");
        metal_msg_send_void0(st, st->q6k_m16_library, "release");
        metal_msg_send_void0(st, st->q4k_m16_library, "release");
        metal_msg_send_void0(st, st->q4k_m16_n2_library, "release");
        metal_msg_send_void0(st, st->q4k_mm_sg_library, "release");
        metal_msg_send_void0(st, st->q4k_mm_sg_fast_library, "release");
        metal_msg_send_void0(st, st->q6k_n4_library, "release");
        metal_msg_send_void0(st, st->q6k_library, "release");
        metal_msg_send_void0(st, st->q4k_library, "release");
        metal_msg_send_void0(st, st->attn_f16_library, "release");
        metal_msg_send_void0(st, st->command_queue, "release");
        metal_msg_send_void0(st, st->device, "release");
    }
    st->q4k_pipeline                          = nullptr;
    st->q4k_function                          = nullptr;
    st->q4k_n4_pipeline                       = nullptr;
    st->q4k_n4_function                       = nullptr;
    st->q4k_matmul_m8_pipeline                = nullptr;
    st->q4k_matmul_m8_function                = nullptr;
    st->q4k_matmul_m16_pipeline               = nullptr;
    st->q4k_matmul_m16_function               = nullptr;
    st->q4k_matmul_m16_n2_pipeline            = nullptr;
    st->q4k_matmul_m16_n2_function            = nullptr;
    st->q4k_mm_sg_pipeline                    = nullptr;
    st->q4k_mm_sg_function                    = nullptr;
    st->q4k_mm_sg_fast_pipeline               = nullptr;
    st->q4k_mm_sg_fast_function               = nullptr;
    st->q4k_gate_up_n4_pipeline               = nullptr;
    st->q4k_gate_up_n4_function               = nullptr;
    st->q4k_pair_n4_pipeline                  = nullptr;
    st->q4k_pair_n4_function                  = nullptr;
    st->q4k_m16_library                       = nullptr;
    st->q4k_mm_sg_library                     = nullptr;
    st->q4k_mm_sg_fast_library                = nullptr;
    st->q6k_pipeline                          = nullptr;
    st->q6k_function                          = nullptr;
    st->q6k_n4_pipeline                       = nullptr;
    st->q6k_n4_function                       = nullptr;
    st->q6k_matmul_m8_pipeline                = nullptr;
    st->q6k_matmul_sg_function                = nullptr;
    st->q6k_matmul_sg_pipeline                = nullptr;
    st->q6k_matmul_sg_fast_pipeline           = nullptr;
    st->q6k_matmul_sg_fast_function           = nullptr;
    st->q6k_matmul_m8_function                = nullptr;
    st->q6k_matmul_m16_pipeline               = nullptr;
    st->q6k_matmul_m16_function               = nullptr;
    st->q6k_mm_sg_library                     = nullptr;
    st->q6k_mm_sg_fast_library                = nullptr;
    st->q6k_m16_library                       = nullptr;
    st->q4k_qk_pipeline                       = nullptr;
    st->q4k_qk_function                       = nullptr;
    st->q4k_gate_up_pipeline                  = nullptr;
    st->q4k_gate_up_function                  = nullptr;
    st->rmsnorm_rows_pipeline                 = nullptr;
    st->rmsnorm_rows_function                 = nullptr;
    st->rmsnorm_rows_simd_pipeline            = nullptr;
    st->rmsnorm_rows_simd_function            = nullptr;
    st->gelu_rows_pipeline                    = nullptr;
    st->gelu_rows_function                    = nullptr;
    st->mul_rows_pipeline                     = nullptr;
    st->mul_rows_function                     = nullptr;
    st->gelu_mul_rows_pipeline                = nullptr;
    st->gelu_mul_rows_function                = nullptr;
    st->add_rows_pipeline                     = nullptr;
    st->add_rows_function                     = nullptr;
    st->scale_rows_pipeline                   = nullptr;
    st->scale_rows_function                   = nullptr;
    st->rmsnorm_add_rows_pipeline             = nullptr;
    st->rmsnorm_add_rows_function             = nullptr;
    st->rmsnorm_add_rows_simd_pipeline        = nullptr;
    st->rmsnorm_add_rows_simd_function        = nullptr;
    st->embed_lookup_scaled_pipeline          = nullptr;
    st->embed_lookup_scaled_function          = nullptr;
    st->f32_matmul_pipeline                   = nullptr;
    st->f32_matmul_function                   = nullptr;
    st->f32_matmul_sg_pipeline                = nullptr;
    st->f32_matmul_sg_function                = nullptr;
    st->f32_matmul_mm_pipeline                = nullptr;
    st->f32_matmul_mm_function                = nullptr;
    st->f32_ple_gate_pipeline                 = nullptr;
    st->f32_ple_gate_function                 = nullptr;
    st->f32_ple_proj_norm_pipeline            = nullptr;
    st->f32_ple_proj_norm_function            = nullptr;
    st->argmax_pipeline                       = nullptr;
    st->argmax_function                       = nullptr;
    st->argmax_batch_pipeline                 = nullptr;
    st->argmax_batch_function                 = nullptr;
    st->argmax_result_buffer                  = nullptr;
    st->argmax_result_mapped                  = nullptr;
    st->argmax_result_capacity                = 0;
    st->rope_rows_pipeline                    = nullptr;
    st->rope_rows_function                    = nullptr;
    st->kv_append_rows_pipeline               = nullptr;
    st->kv_append_rows_function               = nullptr;
    st->copy_u32_pipeline                     = nullptr;
    st->copy_u32_function                     = nullptr;
    st->kv_append_rows_f16_pipeline           = nullptr;
    st->kv_append_rows_f16_function           = nullptr;
    st->q_norm_rope_rows_pipeline             = nullptr;
    st->q_norm_rope_rows_function             = nullptr;
    st->k_norm_rope_append_rows_pipeline      = nullptr;
    st->k_norm_rope_append_rows_function      = nullptr;
    st->k_norm_rope_append_rows_f16_pipeline  = nullptr;
    st->k_norm_rope_append_rows_f16_function  = nullptr;
    st->v_norm_append_rows_pipeline           = nullptr;
    st->v_norm_append_rows_function           = nullptr;
    st->v_norm_append_rows_f16_pipeline       = nullptr;
    st->v_norm_append_rows_f16_function       = nullptr;
    st->kv_norm_append_rows_pipeline          = nullptr;
    st->kv_norm_append_rows_function          = nullptr;
    st->kv_norm_append_rows_f16_pipeline      = nullptr;
    st->kv_norm_append_rows_f16_function      = nullptr;
    st->attention_rows_pipeline               = nullptr;
    st->attention_rows_function               = nullptr;
    st->attention_rows_f16_pipeline           = nullptr;
    st->attention_rows_f16_function           = nullptr;
    st->attention_qnorm_dec_f16_pipeline      = nullptr;
    st->attention_qnorm_dec_f16_function      = nullptr;
    st->attention_dec_combine_pipeline        = nullptr;
    st->attention_dec_combine_function        = nullptr;
    st->attention_dec512_f16_pipeline         = nullptr;
    st->attention_dec512_f16_function         = nullptr;
    st->attn_dec512_f16_library               = nullptr;
    st->attention_flash_sg8_f16_pipeline      = nullptr;
    st->attention_flash_sg8_f16_function      = nullptr;
    st->attn_flash_sg8_f16_library            = nullptr;
    st->attention_qnorm_flash_sg_f16_pipeline = nullptr;
    st->attention_qnorm_flash_sg_f16_function = nullptr;
    st->attn_library                          = nullptr;
    st->attn_f16_library                      = nullptr;
    st->attn_qnorm_dec_f16_library            = nullptr;
    st->attn_dec_combine_library              = nullptr;
    st->attn_flash_sg_f16_library             = nullptr;
    st->kv_norm_append_library                = nullptr;
    st->kv_norm_append_f16_library            = nullptr;
    st->q_norm_rope_library                   = nullptr;
    st->k_norm_rope_append_library            = nullptr;
    st->v_norm_append_library                 = nullptr;
    st->elem_library                          = nullptr;
    st->elem_simd_library                     = nullptr;
    st->embed_library                         = nullptr;
    st->f32_library                           = nullptr;
    st->q4k_gate_up_n4_library                = nullptr;
    st->q4k_pair_n4_library                   = nullptr;
    st->q4k_gate_up_library                   = nullptr;
    st->q4k_n4_library                        = nullptr;
    st->q4k_m16_n2_library                    = nullptr;
    st->q6k_n4_library                        = nullptr;
    st->q6k_library                           = nullptr;
    st->q4k_library                           = nullptr;
    st->q4k_qk_library                        = nullptr;
    st->command_queue                         = nullptr;
    st->device                                = nullptr;
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
                                            unsigned int           memory_flags) {
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

static void metal_buf_reg_add(struct metal_state *st, struct geist_buffer *buf) {
    if (st == nullptr || buf == nullptr || buf->mapped == nullptr) {
        return;
    }
    if (st->buf_reg_count == st->buf_reg_cap) {
        const size_t ncap  = st->buf_reg_cap != 0 ? st->buf_reg_cap * 2 : 64;
        void        *grown = realloc(st->buf_reg, ncap * sizeof(*st->buf_reg));
        if (grown == nullptr) {
            return; /* lookup will miss and report loudly */
        }
        st->buf_reg     = grown;
        st->buf_reg_cap = ncap;
    }
    st->buf_reg[st->buf_reg_count++] = (struct metal_buf_reg_entry) {
            .base  = buf->mapped,
            .bytes = buf->bytes,
            .buf   = buf,
    };
}

static void metal_buf_reg_remove(struct metal_state *st, const struct geist_buffer *buf) {
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

static struct geist_buffer *
metal_buf_reg_find(struct metal_state *st, const void *p, size_t *out_off) {
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
        struct geist_backend *be, enum geist_command_sequence_kind kind, int *out_token);
[[nodiscard]] static enum geist_status
metal_command_sequence_end(struct geist_backend *be, int token, bool submit);

static void metal_seq_ref_clear(struct metal_state *st) {
    memset(st->seq_ref, 0, sizeof(st->seq_ref));
    st->seq_ref_count    = 0;
    st->seq_ref_overflow = false;
}

static void metal_seq_mark_buffer(struct metal_state *st, void *mtl_buf) {
    if (st == nullptr || !st->sequence_active || mtl_buf == nullptr) {
        return;
    }
    const size_t mask = (sizeof(st->seq_ref) / sizeof(st->seq_ref[0])) - 1u;
    size_t       h    = ((uintptr_t) mtl_buf >> 4) & mask;
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
    size_t       h    = ((uintptr_t) mtl_buf >> 4) & mask;
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
    struct geist_backend                  *be   = st->backend;
    const enum geist_command_sequence_kind kind = st->sequence_kind;
    (void) metal_command_sequence_end(be, st->sequence_token, true);
    metal_seq_ref_clear(st);
    int tok = 0;
    (void) metal_command_sequence_begin(be, kind, &tok);
}

static void metal_flush_if_referenced(struct metal_state *st, const void *mtl_buf) {
    if (metal_seq_references(st, mtl_buf)) {
        static int dbg = -1;
        if (dbg < 0) {
            const char *e = getenv("GEIST_SEQ_COUNT");
            dbg           = (e && e[0]) ? 1 : 0;
        }
        if (dbg) {
            fprintf(stderr, "[flush] buf=%p\n", mtl_buf);
        }
        metal_batch_flush(st);
    }
}

[[nodiscard]] static enum geist_status metal_new_buffer(struct geist_backend  *be,
                                                        size_t                 bytes,
                                                        enum geist_buffer_role role,
                                                        unsigned int           memory_flags,
                                                        bool                   host_visible,
                                                        struct geist_buffer  **out) {

    if (out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out = nullptr;
    if (be == nullptr || be->state == nullptr || bytes == 0) {
        if (be != nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_INVALID_ARG, "metal: invalid buffer create request");
        }
        return GEIST_E_INVALID_ARG;
    }

    struct metal_state  *st  = be->state;
    struct geist_buffer *buf = geist_backend_alloc(be, sizeof(*buf), alignof(struct geist_buffer));
    if (buf == nullptr) {
        geist_backend_set_error(be, GEIST_E_OOM, "metal: failed to allocate buffer handle");
        return GEIST_E_OOM;
    }

    const unsigned long options =
            host_visible ? METAL_RESOURCE_STORAGE_MODE_SHARED : METAL_RESOURCE_STORAGE_MODE_PRIVATE;
    void *mtl_buffer = metal_msg_send_id_size_uint(
            st, st->device, "newBufferWithLength:options:", bytes, options);
    if (mtl_buffer == nullptr) {
        geist_backend_free(be, buf);
        geist_backend_set_error(
                be, GEIST_E_BACKEND, "metal: failed to allocate %zu-byte buffer", bytes);
        return GEIST_E_BACKEND;
    }

    void *mapped = host_visible ? metal_msg_send_ptr0(st, mtl_buffer, "contents") : nullptr;
    if (host_visible && mapped == nullptr) {
        metal_msg_send_void0(st, mtl_buffer, "release");
        geist_backend_free(be, buf);
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal: host-visible buffer has no contents");
        return GEIST_E_BACKEND;
    }

    const unsigned int actual_flags =
            memory_flags | (host_visible ? (GEIST_MEMORY_HOST_VISIBLE | GEIST_MEMORY_MAPPED)
                                         : GEIST_MEMORY_DEVICE);
    *buf = (struct geist_buffer) {
            .owner        = st,
            .buffer       = mtl_buffer,
            .mapped       = mapped,
            .bytes        = bytes,
            .role         = role,
            .memory_flags = actual_flags,
            .host_visible = host_visible,
    };
    metal_buf_reg_add(st, buf);
    *out = buf;
    return GEIST_OK;
}

static void metal_buffer_destroy_internal(struct geist_backend *be, struct geist_buffer *buf) {
    if (buf == nullptr) {
        return;
    }
    struct metal_state *st = buf->owner;
    metal_buf_reg_remove(st, buf);
    if (be == nullptr && st != nullptr) {
        be = st->backend;
    }
    if (st != nullptr && st->objc_msgSend != nullptr && st->sel_registerName != nullptr) {
        metal_msg_send_void0(st, buf->buffer, "release");
    }
    if (be != nullptr) {
        geist_backend_free(be, buf);
    }
}

[[nodiscard]] static enum geist_status metal_submit_copy(struct metal_state *st,
                                                         void               *src,
                                                         size_t              src_offset,
                                                         void               *dst,
                                                         size_t              dst_offset,
                                                         size_t              n_bytes) {

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
        st->copy_u32_pipeline != nullptr && (src_offset % 4u) == 0 && (dst_offset % 4u) == 0 &&
        (n_bytes % 4u) == 0) {
        void *enc = metal_sequence_encoder(st);
        struct {
            uint32_t so, dof, n;
        } cp = {(uint32_t) (src_offset / 4u),
                (uint32_t) (dst_offset / 4u),
                (uint32_t) (n_bytes / 4u)};
        metal_msg_send_set_pipeline(st, enc, st->copy_u32_pipeline);
        metal_msg_send_set_buffer(st, enc, src, 0, 0);
        metal_msg_send_set_buffer(st, enc, dst, 0, 1);
        metal_msg_send_set_bytes(st, enc, &cp, sizeof(cp), 2);
        const struct metal_size groups  = {(cp.n + 255u) / 256u, 1, 1};
        const struct metal_size threads = {256, 1, 1};
        metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_COPY_U32, groups);
        metal_msg_send_dispatch(st, enc, groups, threads);
        st->sequence_has_work = true;
        return GEIST_OK;
    }

    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    if (cmd == nullptr) {
        geist_backend_set_error(
                st->backend, GEIST_E_BACKEND, "metal: failed to create command buffer");
        return GEIST_E_BACKEND;
    }
    void *blit = metal_msg_send_id0(st, cmd, "blitCommandEncoder");
    if (blit == nullptr) {
        geist_backend_set_error(
                st->backend, GEIST_E_BACKEND, "metal: failed to create blit encoder");
        return GEIST_E_BACKEND;
    }

    metal_msg_send_copy_buffer(st,
                               blit,
                               "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:",
                               src,
                               src_offset,
                               dst,
                               dst_offset,
                               n_bytes);
    metal_msg_send_void0(st, blit, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");

    void *err = metal_msg_send_id0(st, cmd, "error");
    if (err != nullptr) {
        geist_backend_set_error(st->backend, GEIST_E_BACKEND, "metal: blit command failed");
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status metal_buffer_create(struct geist_backend  *be,
                                                           size_t                 bytes,
                                                           enum geist_buffer_role role,
                                                           unsigned int           memory_flags,
                                                           struct geist_buffer  **out) {

    const bool host_visible = metal_buffer_wants_host_visible(role, memory_flags);
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
[[nodiscard]] static enum geist_status metal_buffer_create_aliased(struct geist_backend  *be,
                                                                   void                  *host_ptr,
                                                                   size_t                 n_bytes,
                                                                   enum geist_buffer_role role,
                                                                   struct geist_buffer  **out) {

    if (out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out = nullptr;
    if (be == nullptr || be->state == nullptr || host_ptr == nullptr || n_bytes == 0) {
        if (be != nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_INVALID_ARG, "metal: invalid aliased buffer request");
        }
        return GEIST_E_INVALID_ARG;
    }
    struct metal_state  *st  = be->state;
    struct geist_buffer *buf = geist_backend_alloc(be, sizeof(*buf), alignof(struct geist_buffer));
    if (buf == nullptr) {
        geist_backend_set_error(be, GEIST_E_OOM, "metal: failed to allocate buffer handle");
        return GEIST_E_OOM;
    }
    void *mtl_buffer = metal_msg_send_id_ptr_size_uint(st,
                                                       st->device,
                                                       "newBufferWithBytes:length:options:",
                                                       host_ptr,
                                                       n_bytes,
                                                       METAL_RESOURCE_STORAGE_MODE_SHARED);
    if (mtl_buffer == nullptr) {
        geist_backend_free(be, buf);
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal: failed to alias %zu bytes", n_bytes);
        return GEIST_E_BACKEND;
    }
    void *mapped = metal_msg_send_ptr0(st, mtl_buffer, "contents");
    if (mapped == nullptr) {
        metal_msg_send_void0(st, mtl_buffer, "release");
        geist_backend_free(be, buf);
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal: aliased buffer has no contents");
        return GEIST_E_BACKEND;
    }
    *buf = (struct geist_buffer) {
            .owner        = st,
            .buffer       = mtl_buffer,
            .mapped       = mapped,
            .bytes        = n_bytes,
            .role         = role,
            .memory_flags = GEIST_MEMORY_HOST_VISIBLE | GEIST_MEMORY_MAPPED | GEIST_MEMORY_ALIASED,
            .host_visible = true,
    };
    metal_buf_reg_add(st, buf);
    *out = buf;
    return GEIST_OK;
}

static void metal_buffer_destroy(struct geist_backend *be, struct geist_buffer *buf) {
    metal_buffer_destroy_internal(be, buf);
}

[[nodiscard]] static enum geist_status metal_buffer_copy(struct geist_buffer       *dst,
                                                         size_t                     dst_offset,
                                                         const struct geist_buffer *src,
                                                         size_t                     src_offset,
                                                         size_t                     n_bytes) {

    if (dst == nullptr || src == nullptr || dst->owner == nullptr || src->owner == nullptr ||
        dst->owner != src->owner) {
        return GEIST_E_INVALID_ARG;
    }
    if (dst_offset > dst->bytes || src_offset > src->bytes || n_bytes > dst->bytes - dst_offset ||
        n_bytes > src->bytes - src_offset) {
        return GEIST_E_INVALID_ARG;
    }
    if (n_bytes == 0) {
        return GEIST_OK;
    }

    struct metal_state *st = dst->owner;
    if (dst == src && metal_ranges_overlap(dst_offset, src_offset, n_bytes)) {
        struct geist_buffer *tmp = nullptr;
        enum geist_status    s   = metal_new_buffer(
                st->backend, n_bytes, GEIST_BUFFER_SCRATCH, GEIST_MEMORY_DEVICE, false, &tmp);
        if (s != GEIST_OK) {
            return s;
        }
        s = metal_submit_copy(st, src->buffer, src_offset, tmp->buffer, 0, n_bytes);
        if (s == GEIST_OK) {
            s = metal_submit_copy(st, tmp->buffer, 0, dst->buffer, dst_offset, n_bytes);
        }
        metal_buffer_destroy_internal(st->backend, tmp);
        return s;
    }

    return metal_submit_copy(st, src->buffer, src_offset, dst->buffer, dst_offset, n_bytes);
}

[[nodiscard]] static enum geist_status
metal_buffer_upload(struct geist_buffer *buf, size_t n_bytes, const uint8_t *src) {

    if (buf == nullptr || n_bytes > buf->bytes || (n_bytes > 0 && src == nullptr)) {
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

    struct metal_state  *st      = buf->owner;
    struct geist_buffer *staging = nullptr;
    enum geist_status    s       = metal_new_buffer(
            st->backend, n_bytes, GEIST_BUFFER_STAGING, GEIST_MEMORY_HOST_VISIBLE, true, &staging);
    if (s != GEIST_OK) {
        return s;
    }
    memcpy(staging->mapped, src, n_bytes);
    s = metal_submit_copy(st, staging->buffer, 0, buf->buffer, 0, n_bytes);
    metal_buffer_destroy_internal(st->backend, staging);
    return s;
}

[[nodiscard]] static enum geist_status
metal_buffer_download(size_t n_bytes, uint8_t *dst, const struct geist_buffer *buf) {

    if (buf == nullptr || n_bytes > buf->bytes || (n_bytes > 0 && dst == nullptr)) {
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

    struct metal_state  *st      = buf->owner;
    struct geist_buffer *staging = nullptr;
    enum geist_status    s       = metal_new_buffer(
            st->backend, n_bytes, GEIST_BUFFER_STAGING, GEIST_MEMORY_HOST_VISIBLE, true, &staging);
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
            if (dbg < 0) {
                const char *e = getenv("GEIST_SEQ_COUNT");
                dbg           = (e && e[0]) ? 1 : 0;
            }
            if (dbg) {
                fprintf(stderr, "[flushmap] bytes=%zu role=%d\n", buf->bytes, (int) buf->role);
            }
            metal_batch_flush(st);
        }
    }
    return buf->mapped;
}

static void metal_buffer_unmap(struct geist_buffer *buf) {
    (void) buf;
}

static bool
metal_tensor_is_f32_vector(const struct geist_tensor *t, size_t *out_n, size_t *out_offset_floats) {
    if (t == nullptr || t->buffer == nullptr || t->dtype != GEIST_DTYPE_F32 ||
        t->layout != GEIST_LAYOUT_DENSE || (t->ndim != 1 && t->ndim != 2) ||
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
        if (t->shape[0] != 1 || t->shape[1] <= 0 || t->stride[0] != t->shape[1] ||
            t->stride[1] != 1) {
            return false;
        }
        n = (size_t) t->shape[1];
    }
    if (t->offset > t->buffer->bytes || n > (t->buffer->bytes - t->offset) / sizeof(float)) {
        return false;
    }
    *out_n             = n;
    *out_offset_floats = t->offset / sizeof(float);
    return true;
}

static bool metal_tensor_is_f32_matrix(const struct geist_tensor *t,
                                       size_t                    *out_rows,
                                       size_t                    *out_cols,
                                       size_t                    *out_offset_floats,
                                       size_t                    *out_row_stride) {
    if (t == nullptr || t->buffer == nullptr || t->dtype != GEIST_DTYPE_F32 ||
        t->layout != GEIST_LAYOUT_DENSE || t->ndim != 2 || t->shape[0] <= 0 || t->shape[1] <= 0 ||
        t->stride[0] != t->shape[1] || t->stride[1] != 1 || t->offset % sizeof(float) != 0) {
        return false;
    }
    const size_t rows = (size_t) t->shape[0];
    const size_t cols = (size_t) t->shape[1];
    if (rows > SIZE_MAX / cols) {
        return false;
    }
    const size_t elems = rows * cols;
    if (t->offset > t->buffer->bytes || elems > (t->buffer->bytes - t->offset) / sizeof(float)) {
        return false;
    }
    *out_rows          = rows;
    *out_cols          = cols;
    *out_offset_floats = t->offset / sizeof(float);
    *out_row_stride    = cols;
    return true;
}

/* Like metal_tensor_is_f32_matrix but accepts a row stride wider than the
 * column count (a strided view into a larger slab, e.g. the per-layer PLE
 * slice). Kernels take the row stride from their params. */
static bool metal_tensor_is_f32_matrix_strided(const struct geist_tensor *t,
                                               size_t                    *out_rows,
                                               size_t                    *out_cols,
                                               size_t                    *out_offset_floats,
                                               size_t                    *out_row_stride) {
    if (t == nullptr || t->buffer == nullptr || t->dtype != GEIST_DTYPE_F32 ||
        t->layout != GEIST_LAYOUT_DENSE || t->ndim != 2 || t->shape[0] <= 0 || t->shape[1] <= 0 ||
        t->stride[0] < t->shape[1] || t->stride[1] != 1 || t->offset % sizeof(float) != 0) {
        return false;
    }
    const size_t rows   = (size_t) t->shape[0];
    const size_t cols   = (size_t) t->shape[1];
    const size_t stride = (size_t) t->stride[0];
    if (rows > 1 && stride > (SIZE_MAX - cols) / (rows - 1)) {
        return false;
    }
    const size_t elems = (rows - 1) * stride + cols;
    if (t->offset > t->buffer->bytes || elems > (t->buffer->bytes - t->offset) / sizeof(float)) {
        return false;
    }
    *out_rows          = rows;
    *out_cols          = cols;
    *out_offset_floats = t->offset / sizeof(float);
    *out_row_stride    = stride;
    return true;
}

static bool metal_tensor_is_f32_rows(const struct geist_tensor *t,
                                     size_t                    *out_rows,
                                     size_t                    *out_cols,
                                     size_t                    *out_offset_floats,
                                     size_t                    *out_row_stride) {
    if (t == nullptr || out_rows == nullptr || out_cols == nullptr ||
        out_offset_floats == nullptr || out_row_stride == nullptr) {
        return false;
    }
    if (t->ndim == 1) {
        size_t n   = 0;
        size_t off = 0;
        if (!metal_tensor_is_f32_vector(t, &n, &off)) {
            return false;
        }
        *out_rows          = 1;
        *out_cols          = n;
        *out_offset_floats = off;
        *out_row_stride    = n;
        return true;
    }
    /* The elementwise kernels take per-tensor row strides from their
     * params, so a strided 2D view (e.g. the per-layer PLE slice of the
     * [seq, n_layers*hpl] slab) is fine here. */
    return metal_tensor_is_f32_matrix_strided(
            t, out_rows, out_cols, out_offset_floats, out_row_stride);
}

static bool metal_tensor_is_q4k_matrix(const struct geist_tensor *t,
                                       size_t                    *out_rows,
                                       size_t                    *out_cols,
                                       size_t                    *out_offset_bytes) {
    if (t == nullptr || t->buffer == nullptr || t->dtype != GEIST_DTYPE_Q4_K ||
        t->layout != GEIST_LAYOUT_BLOCK_QUANTIZED || t->ndim != 2 || t->shape[0] <= 0 ||
        t->shape[1] <= 0 || ((size_t) t->shape[1] % METAL_Q4K_BLOCK_ELEMS) != 0) {
        return false;
    }
    const size_t rows           = (size_t) t->shape[0];
    const size_t cols           = (size_t) t->shape[1];
    const size_t blocks_per_row = cols / METAL_Q4K_BLOCK_ELEMS;
    if (rows > SIZE_MAX / blocks_per_row ||
        rows * blocks_per_row > SIZE_MAX / METAL_Q4K_BLOCK_BYTES) {
        return false;
    }
    const size_t bytes = rows * blocks_per_row * METAL_Q4K_BLOCK_BYTES;
    if (t->offset > t->buffer->bytes || bytes > t->buffer->bytes - t->offset) {
        return false;
    }
    *out_rows         = rows;
    *out_cols         = cols;
    *out_offset_bytes = t->offset;
    return true;
}

static bool metal_tensor_is_q6k_matrix(const struct geist_tensor *t,
                                       size_t                    *out_rows,
                                       size_t                    *out_cols,
                                       size_t                    *out_offset_bytes) {
    if (t == nullptr || t->buffer == nullptr || t->dtype != GEIST_DTYPE_Q6_K ||
        t->layout != GEIST_LAYOUT_BLOCK_QUANTIZED || t->ndim != 2 || t->shape[0] <= 0 ||
        t->shape[1] <= 0 || ((size_t) t->shape[1] % METAL_Q6K_BLOCK_ELEMS) != 0) {
        return false;
    }
    const size_t rows           = (size_t) t->shape[0];
    const size_t cols           = (size_t) t->shape[1];
    const size_t blocks_per_row = cols / METAL_Q6K_BLOCK_ELEMS;
    if (rows > SIZE_MAX / blocks_per_row ||
        rows * blocks_per_row > SIZE_MAX / METAL_Q6K_BLOCK_BYTES) {
        return false;
    }
    const size_t bytes = rows * blocks_per_row * METAL_Q6K_BLOCK_BYTES;
    if (t->offset > t->buffer->bytes || bytes > t->buffer->bytes - t->offset) {
        return false;
    }
    *out_rows         = rows;
    *out_cols         = cols;
    *out_offset_bytes = t->offset;
    return true;
}

[[nodiscard]] static enum geist_status metal_create_named_pipeline(struct geist_backend *be,
                                                                   void                 *library,
                                                                   void                 *ns_string,
                                                                   const char           *name,
                                                                   void **out_function,
                                                                   void **out_pipeline) {

    if (be == nullptr || be->state == nullptr || library == nullptr || ns_string == nullptr ||
        name == nullptr || out_function == nullptr || out_pipeline == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct metal_state *st = be->state;
    void *fn_name          = metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", name);
    if (fn_name == nullptr) {
        geist_backend_set_error(
                be, GEIST_E_BACKEND, "metal: failed to create %s shader name", name);
        return GEIST_E_BACKEND;
    }
    *out_function = metal_msg_send_id_id(st, library, "newFunctionWithName:", fn_name);
    if (*out_function == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal: %s shader function missing", name);
        return GEIST_E_BACKEND;
    }

    void *err     = nullptr;
    *out_pipeline = metal_msg_send_id_id_err(
            st, st->device, "newComputePipelineStateWithFunction:error:", *out_function, &err);
    if (*out_pipeline == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: %s pipeline creation failed%s%s",
                                name,
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status metal_ensure_q4k_pipeline(struct geist_backend *be) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct metal_state *st = be->state;
    if (st->q4k_pipeline != nullptr && st->q4k_n4_pipeline != nullptr &&
        st->q4k_matmul_m8_pipeline != nullptr && st->q4k_matmul_m16_pipeline != nullptr &&
        st->q4k_matmul_m16_n2_pipeline != nullptr &&
        (!st->use_q4k_mm_sg || st->q4k_mm_sg_pipeline != nullptr) && st->q6k_pipeline != nullptr &&
        st->q6k_n4_pipeline != nullptr && st->q6k_matmul_m8_pipeline != nullptr &&
        st->q6k_matmul_m16_pipeline != nullptr && st->rmsnorm_rows_pipeline != nullptr &&
        st->rmsnorm_rows_simd_pipeline != nullptr && st->gelu_rows_pipeline != nullptr &&
        st->mul_rows_pipeline != nullptr && st->gelu_mul_rows_pipeline != nullptr &&
        st->add_rows_pipeline != nullptr && st->scale_rows_pipeline != nullptr &&
        st->rmsnorm_add_rows_pipeline != nullptr && st->rmsnorm_add_rows_simd_pipeline != nullptr &&
        st->embed_lookup_scaled_pipeline != nullptr && st->f32_matmul_pipeline != nullptr &&
        st->f32_ple_gate_pipeline != nullptr && st->f32_ple_proj_norm_pipeline != nullptr) {
        return GEIST_OK;
    }

    void *ns_string = metal_objc_get_class(st, "NSString");
    if (ns_string == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal: NSString class unavailable");
        return GEIST_E_BACKEND;
    }
    void *source = metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", metal_q4k_source);
    void *q4k_n4_source =
            metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", metal_q4k_n4_source);
    void *q4k_m16_source =
            metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", metal_q4k_m16_source);
    void *q4k_m16_n2_source =
            metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", metal_q4k_m16_n2_source);
    void *q4k_mm_sg_ns_source =
            st->use_q4k_mm_sg
                    ? metal_msg_send_id_cstr(
                              st, ns_string, "stringWithUTF8String:", metal_q4k_mm_sg_source)
                    : nullptr;
    void *q4k_mm_sg_fast_ns_source =
            st->use_q4k_mm_sg
                    ? metal_msg_send_id_cstr(
                              st, ns_string, "stringWithUTF8String:", metal_q4k_mm_sg_fast_source)
                    : nullptr;
    void *q4k_gate_up_n4_src = metal_msg_send_id_cstr(
            st, ns_string, "stringWithUTF8String:", metal_q4k_gate_up_n4_source);
    void *q4k_pair_n4_src = metal_msg_send_id_cstr(
            st, ns_string, "stringWithUTF8String:", metal_q4k_pair_n4_source);
    void *q6_source =
            metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", metal_q6k_source);
    void *q6_mm_sg_source =
            metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", metal_q6k_mm_sg_source);
    void *q6_mm_sg_fast_source = metal_msg_send_id_cstr(
            st, ns_string, "stringWithUTF8String:", metal_q6k_mm_sg_fast_source);
    void *q6_n4_source =
            metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", metal_q6k_n4_source);
    void *q6_m16_source =
            metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", metal_q6k_m16_source);
    void *elem_source =
            metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", metal_elem_source);
    void *elem_simd_source =
            metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", metal_elem_simd_source);
    void *embed_source =
            metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", metal_embed_source);
    void *f32_source = nullptr;
    {
        /* two literals concatenated at init (C99 4095-char literal limit) */
        const size_t f32_len_a = strlen(metal_f32_source);
        const size_t f32_len_b = strlen(metal_f32_mm_source);
        char        *f32_src   = malloc(f32_len_a + f32_len_b + 1u);
        if (f32_src == nullptr) {
            geist_backend_set_error(be, GEIST_E_OOM, "metal: f32 shader source alloc failed");
            return GEIST_E_OOM;
        }
        memcpy(f32_src, metal_f32_source, f32_len_a);
        memcpy(f32_src + f32_len_a, metal_f32_mm_source, f32_len_b + 1u);
        f32_source = metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", f32_src);
        free(f32_src);
    }
    if (source == nullptr || q4k_n4_source == nullptr || q4k_m16_source == nullptr ||
        q4k_m16_n2_source == nullptr || (st->use_q4k_mm_sg && q4k_mm_sg_ns_source == nullptr) ||
        q4k_gate_up_n4_src == nullptr || q4k_pair_n4_src == nullptr || q6_source == nullptr ||
        q6_n4_source == nullptr || q6_m16_source == nullptr || elem_source == nullptr ||
        elem_simd_source == nullptr || embed_source == nullptr || f32_source == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal: failed to create shader source");
        return GEIST_E_BACKEND;
    }

    void *err       = nullptr;
    st->q4k_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:", source, nullptr, &err);
    if (st->q4k_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: Q4_K shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err                = nullptr;
    st->q4k_n4_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:", q4k_n4_source, nullptr, &err);
    if (st->q4k_n4_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: Q4_K n4 shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err                        = nullptr;
    st->q4k_gate_up_n4_library = metal_msg_send_id_id_id_err(st,
                                                             st->device,
                                                             "newLibraryWithSource:options:error:",
                                                             q4k_gate_up_n4_src,
                                                             nullptr,
                                                             &err);
    if (st->q4k_gate_up_n4_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: Q4_K gate/up n4 shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err                     = nullptr;
    st->q4k_pair_n4_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:", q4k_pair_n4_src, nullptr, &err);
    if (st->q4k_pair_n4_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: Q4_K pair n4 shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err                 = nullptr;
    st->q4k_m16_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:", q4k_m16_source, nullptr, &err);
    if (st->q4k_m16_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: Q4_K m16 shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err                    = nullptr;
    st->q4k_m16_n2_library = metal_msg_send_id_id_id_err(st,
                                                         st->device,
                                                         "newLibraryWithSource:options:error:",
                                                         q4k_m16_n2_source,
                                                         nullptr,
                                                         &err);
    if (st->q4k_m16_n2_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: Q4_K m16 n2 shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    if (st->use_q4k_mm_sg) {
        err                   = nullptr;
        st->q4k_mm_sg_library = metal_msg_send_id_id_id_err(st,
                                                            st->device,
                                                            "newLibraryWithSource:options:error:",
                                                            q4k_mm_sg_ns_source,
                                                            nullptr,
                                                            &err);
        if (st->q4k_mm_sg_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(be,
                                    GEIST_E_BACKEND,
                                    "metal: Q4_K simdgroup mm shader compile failed%s%s",
                                    msg != nullptr ? ": " : "",
                                    msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
        err = nullptr;
        st->q4k_mm_sg_fast_library =
                metal_msg_send_id_id_id_err(st,
                                            st->device,
                                            "newLibraryWithSource:options:error:",
                                            q4k_mm_sg_fast_ns_source,
                                            nullptr,
                                            &err);
        if (st->q4k_mm_sg_fast_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(be,
                                    GEIST_E_BACKEND,
                                    "metal: Q4_K simdgroup mm fast shader compile failed%s%s",
                                    msg != nullptr ? ": " : "",
                                    msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
    }
    err             = nullptr;
    st->q6k_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:", q6_source, nullptr, &err);
    if (st->q6k_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: Q6_K shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err                   = nullptr;
    st->q6k_mm_sg_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:", q6_mm_sg_source, nullptr, &err);
    if (st->q6k_mm_sg_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: Q6_K mm_sg shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err                        = nullptr;
    st->q6k_mm_sg_fast_library = metal_msg_send_id_id_id_err(st,
                                                             st->device,
                                                             "newLibraryWithSource:options:error:",
                                                             q6_mm_sg_fast_source,
                                                             nullptr,
                                                             &err);
    if (st->q6k_mm_sg_fast_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: Q6_K mm_sg fast shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err                 = nullptr;
    st->q6k_m16_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:", q6_m16_source, nullptr, &err);
    if (st->q6k_m16_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: Q6_K m16 shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err                = nullptr;
    st->q6k_n4_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:", q6_n4_source, nullptr, &err);
    if (st->q6k_n4_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: Q6_K n4 shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err              = nullptr;
    st->elem_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:", elem_source, nullptr, &err);
    if (st->elem_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: elementwise shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err                   = nullptr;
    st->elem_simd_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:", elem_simd_source, nullptr, &err);
    if (st->elem_simd_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: SIMD elementwise shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err               = nullptr;
    st->embed_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:", embed_source, nullptr, &err);
    if (st->embed_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: embedding shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    err             = nullptr;
    st->f32_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:", f32_source, nullptr, &err);
    if (st->f32_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: F32 shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    enum geist_status s = metal_create_named_pipeline(
            be, st->q4k_library, ns_string, "matvec_q4k", &st->q4k_function, &st->q4k_pipeline);
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->q4k_n4_library,
                                        ns_string,
                                        "matvec_q4k_n4",
                                        &st->q4k_n4_function,
                                        &st->q4k_n4_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->q4k_gate_up_n4_library,
                                        ns_string,
                                        "gate_up_q4k_n4",
                                        &st->q4k_gate_up_n4_function,
                                        &st->q4k_gate_up_n4_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->q4k_pair_n4_library,
                                        ns_string,
                                        "pair_q4k_n4",
                                        &st->q4k_pair_n4_function,
                                        &st->q4k_pair_n4_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->q4k_library,
                                        ns_string,
                                        "matmul_q4k_m8",
                                        &st->q4k_matmul_m8_function,
                                        &st->q4k_matmul_m8_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->q4k_m16_library,
                                        ns_string,
                                        "matmul_q4k_m16",
                                        &st->q4k_matmul_m16_function,
                                        &st->q4k_matmul_m16_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->q4k_m16_n2_library,
                                        ns_string,
                                        "matmul_q4k_m16_n2",
                                        &st->q4k_matmul_m16_n2_function,
                                        &st->q4k_matmul_m16_n2_pipeline);
    }
    if (s == GEIST_OK && st->use_q4k_mm_sg) {
        s = metal_create_named_pipeline(be,
                                        st->q4k_mm_sg_library,
                                        ns_string,
                                        "matmul_q4k_mm_sg",
                                        &st->q4k_mm_sg_function,
                                        &st->q4k_mm_sg_pipeline);
    }
    if (s == GEIST_OK && st->use_q4k_mm_sg) {
        s = metal_create_named_pipeline(be,
                                        st->q4k_mm_sg_fast_library,
                                        ns_string,
                                        "matmul_q4k_mm_sg_fast",
                                        &st->q4k_mm_sg_fast_function,
                                        &st->q4k_mm_sg_fast_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(
                be, st->q6k_library, ns_string, "matvec_q6k", &st->q6k_function, &st->q6k_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->q6k_mm_sg_library,
                                        ns_string,
                                        "matmul_q6k_sg",
                                        &st->q6k_matmul_sg_function,
                                        &st->q6k_matmul_sg_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->q6k_mm_sg_fast_library,
                                        ns_string,
                                        "matmul_q6k_sg_fast",
                                        &st->q6k_matmul_sg_fast_function,
                                        &st->q6k_matmul_sg_fast_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->q6k_n4_library,
                                        ns_string,
                                        "matvec_q6k_n4",
                                        &st->q6k_n4_function,
                                        &st->q6k_n4_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->q6k_library,
                                        ns_string,
                                        "matmul_q6k_m8",
                                        &st->q6k_matmul_m8_function,
                                        &st->q6k_matmul_m8_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->q6k_m16_library,
                                        ns_string,
                                        "matmul_q6k_m16",
                                        &st->q6k_matmul_m16_function,
                                        &st->q6k_matmul_m16_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->elem_library,
                                        ns_string,
                                        "rmsnorm_rows",
                                        &st->rmsnorm_rows_function,
                                        &st->rmsnorm_rows_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->elem_simd_library,
                                        ns_string,
                                        "rmsnorm_rows_simd",
                                        &st->rmsnorm_rows_simd_function,
                                        &st->rmsnorm_rows_simd_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->elem_library,
                                        ns_string,
                                        "gelu_rows",
                                        &st->gelu_rows_function,
                                        &st->gelu_rows_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->elem_library,
                                        ns_string,
                                        "mul_rows",
                                        &st->mul_rows_function,
                                        &st->mul_rows_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->elem_library,
                                        ns_string,
                                        "gelu_mul_rows",
                                        &st->gelu_mul_rows_function,
                                        &st->gelu_mul_rows_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->elem_library,
                                        ns_string,
                                        "add_rows",
                                        &st->add_rows_function,
                                        &st->add_rows_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->elem_library,
                                        ns_string,
                                        "scale_rows",
                                        &st->scale_rows_function,
                                        &st->scale_rows_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->elem_library,
                                        ns_string,
                                        "rmsnorm_add_rows",
                                        &st->rmsnorm_add_rows_function,
                                        &st->rmsnorm_add_rows_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->elem_simd_library,
                                        ns_string,
                                        "rmsnorm_add_rows_simd",
                                        &st->rmsnorm_add_rows_simd_function,
                                        &st->rmsnorm_add_rows_simd_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->embed_library,
                                        ns_string,
                                        "embed_lookup_scaled",
                                        &st->embed_lookup_scaled_function,
                                        &st->embed_lookup_scaled_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->f32_library,
                                        ns_string,
                                        "matmul_f32",
                                        &st->f32_matmul_function,
                                        &st->f32_matmul_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->f32_library,
                                        ns_string,
                                        "matmul_f32_sg",
                                        &st->f32_matmul_sg_function,
                                        &st->f32_matmul_sg_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->f32_library,
                                        ns_string,
                                        "matmul_f32_mm_sg",
                                        &st->f32_matmul_mm_function,
                                        &st->f32_matmul_mm_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->f32_library,
                                        ns_string,
                                        "ple_gate_f32",
                                        &st->f32_ple_gate_function,
                                        &st->f32_ple_gate_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->f32_library,
                                        ns_string,
                                        "ple_proj_norm_f32",
                                        &st->f32_ple_proj_norm_function,
                                        &st->f32_ple_proj_norm_pipeline);
    }
    return s;
}

[[nodiscard]] static enum geist_status metal_ensure_attention_pipeline(struct geist_backend *be) {

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
        st->kv_norm_append_rows_f16_pipeline != nullptr && st->rope_rows_pipeline != nullptr &&
        st->kv_append_rows_pipeline != nullptr && st->kv_append_rows_f16_pipeline != nullptr &&
        st->attention_rows_pipeline != nullptr && st->attention_rows_f16_pipeline != nullptr) {
        return GEIST_OK;
    }

    void *ns_string = metal_objc_get_class(st, "NSString");
    if (ns_string == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal: NSString class unavailable");
        return GEIST_E_BACKEND;
    }
    void *source = metal_msg_send_id_cstr(
            st, ns_string, "stringWithUTF8String:", metal_q_norm_rope_source);
    if (source == nullptr) {
        geist_backend_set_error(
                be, GEIST_E_BACKEND, "metal: failed to create Q norm/RoPE shader source");
        return GEIST_E_BACKEND;
    }
    void *err               = nullptr;
    st->q_norm_rope_library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:", source, nullptr, &err);
    if (st->q_norm_rope_library == nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal: Q norm/RoPE shader compile failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    enum geist_status s = metal_create_named_pipeline(be,
                                                      st->q_norm_rope_library,
                                                      ns_string,
                                                      "q_norm_rope_rows",
                                                      &st->q_norm_rope_rows_function,
                                                      &st->q_norm_rope_rows_pipeline);

    if (s == GEIST_OK) {
        source = metal_msg_send_id_cstr(
                st, ns_string, "stringWithUTF8String:", metal_k_norm_rope_append_source);
        if (source == nullptr) {
            geist_backend_set_error(be,
                                    GEIST_E_BACKEND,
                                    "metal: failed to create K norm/RoPE append shader source");
            return GEIST_E_BACKEND;
        }
        err                            = nullptr;
        st->k_norm_rope_append_library = metal_msg_send_id_id_id_err(
                st, st->device, "newLibraryWithSource:options:error:", source, nullptr, &err);
        if (st->k_norm_rope_append_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(be,
                                    GEIST_E_BACKEND,
                                    "metal: K norm/RoPE append shader compile failed%s%s",
                                    msg != nullptr ? ": " : "",
                                    msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
        s = metal_create_named_pipeline(be,
                                        st->k_norm_rope_append_library,
                                        ns_string,
                                        "k_norm_rope_append_rows",
                                        &st->k_norm_rope_append_rows_function,
                                        &st->k_norm_rope_append_rows_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->k_norm_rope_append_library,
                                        ns_string,
                                        "k_norm_rope_append_rows_f16",
                                        &st->k_norm_rope_append_rows_f16_function,
                                        &st->k_norm_rope_append_rows_f16_pipeline);
    }
    if (s == GEIST_OK) {
        source = metal_msg_send_id_cstr(
                st, ns_string, "stringWithUTF8String:", metal_v_norm_append_source);
        if (source == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal: failed to create V norm append shader source");
            return GEIST_E_BACKEND;
        }
        err                       = nullptr;
        st->v_norm_append_library = metal_msg_send_id_id_id_err(
                st, st->device, "newLibraryWithSource:options:error:", source, nullptr, &err);
        if (st->v_norm_append_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(be,
                                    GEIST_E_BACKEND,
                                    "metal: V norm append shader compile failed%s%s",
                                    msg != nullptr ? ": " : "",
                                    msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
        s = metal_create_named_pipeline(be,
                                        st->v_norm_append_library,
                                        ns_string,
                                        "v_norm_append_rows",
                                        &st->v_norm_append_rows_function,
                                        &st->v_norm_append_rows_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->v_norm_append_library,
                                        ns_string,
                                        "v_norm_append_rows_f16",
                                        &st->v_norm_append_rows_f16_function,
                                        &st->v_norm_append_rows_f16_pipeline);
    }
    if (s == GEIST_OK) {
        source = metal_msg_send_id_cstr(
                st, ns_string, "stringWithUTF8String:", metal_kv_norm_append_source);
        if (source == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal: failed to create K/V norm append shader source");
            return GEIST_E_BACKEND;
        }
        err                        = nullptr;
        st->kv_norm_append_library = metal_msg_send_id_id_id_err(
                st, st->device, "newLibraryWithSource:options:error:", source, nullptr, &err);
        if (st->kv_norm_append_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(be,
                                    GEIST_E_BACKEND,
                                    "metal: K/V norm append shader compile failed%s%s",
                                    msg != nullptr ? ": " : "",
                                    msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
        s = metal_create_named_pipeline(be,
                                        st->kv_norm_append_library,
                                        ns_string,
                                        "kv_norm_append_rows",
                                        &st->kv_norm_append_rows_function,
                                        &st->kv_norm_append_rows_pipeline);
    }
    if (s == GEIST_OK) {
        source = metal_msg_send_id_cstr(
                st, ns_string, "stringWithUTF8String:", metal_kv_norm_append_f16_source);
        if (source == nullptr) {
            geist_backend_set_error(be,
                                    GEIST_E_BACKEND,
                                    "metal: failed to create F16 K/V norm append shader source");
            return GEIST_E_BACKEND;
        }
        err                            = nullptr;
        st->kv_norm_append_f16_library = metal_msg_send_id_id_id_err(
                st, st->device, "newLibraryWithSource:options:error:", source, nullptr, &err);
        if (st->kv_norm_append_f16_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(be,
                                    GEIST_E_BACKEND,
                                    "metal: F16 K/V norm append shader compile failed%s%s",
                                    msg != nullptr ? ": " : "",
                                    msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
        s = metal_create_named_pipeline(be,
                                        st->kv_norm_append_f16_library,
                                        ns_string,
                                        "kv_norm_append_rows_f16",
                                        &st->kv_norm_append_rows_f16_function,
                                        &st->kv_norm_append_rows_f16_pipeline);
    }
    if (s == GEIST_OK) {
        source = metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", metal_attn_source);
        if (source == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal: failed to create attention shader source");
            return GEIST_E_BACKEND;
        }
        err              = nullptr;
        st->attn_library = metal_msg_send_id_id_id_err(
                st, st->device, "newLibraryWithSource:options:error:", source, nullptr, &err);
        if (st->attn_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(be,
                                    GEIST_E_BACKEND,
                                    "metal: attention shader compile failed%s%s",
                                    msg != nullptr ? ": " : "",
                                    msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->attn_library,
                                        ns_string,
                                        "rope_rows",
                                        &st->rope_rows_function,
                                        &st->rope_rows_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->attn_library,
                                        ns_string,
                                        "kv_append_rows",
                                        &st->kv_append_rows_function,
                                        &st->kv_append_rows_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->attn_library,
                                        ns_string,
                                        "copy_u32",
                                        &st->copy_u32_function,
                                        &st->copy_u32_pipeline);
    }
    if (s == GEIST_OK) {
        source = metal_msg_send_id_cstr(
                st, ns_string, "stringWithUTF8String:", metal_attn_f16_source);
        if (source == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal: failed to create F16 attention shader source");
            return GEIST_E_BACKEND;
        }
        err                  = nullptr;
        st->attn_f16_library = metal_msg_send_id_id_id_err(
                st, st->device, "newLibraryWithSource:options:error:", source, nullptr, &err);
        if (st->attn_f16_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(be,
                                    GEIST_E_BACKEND,
                                    "metal: F16 attention shader compile failed%s%s",
                                    msg != nullptr ? ": " : "",
                                    msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->attn_f16_library,
                                        ns_string,
                                        "kv_append_rows_f16",
                                        &st->kv_append_rows_f16_function,
                                        &st->kv_append_rows_f16_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->attn_library,
                                        ns_string,
                                        "attention_rows",
                                        &st->attention_rows_function,
                                        &st->attention_rows_pipeline);
    }
    if (s == GEIST_OK) {
        s = metal_create_named_pipeline(be,
                                        st->attn_f16_library,
                                        ns_string,
                                        "attention_rows_f16",
                                        &st->attention_rows_f16_function,
                                        &st->attention_rows_f16_pipeline);
    }
    if (s == GEIST_OK) {
        const size_t dl_h    = strlen(metal_attn_qnorm_dec_f16_source);
        const size_t dl_b    = strlen(metal_attn_dec_f16_body);
        const size_t dl_p    = strlen(metal_attn_dec_f16_plain_head);
        char        *dec_src = malloc(dl_h + 2u * dl_b + dl_p + 1u);
        if (dec_src == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_OOM, "metal: decode attention shader source alloc failed");
            return GEIST_E_OOM;
        }
        memcpy(dec_src, metal_attn_qnorm_dec_f16_source, dl_h);
        memcpy(dec_src + dl_h, metal_attn_dec_f16_body, dl_b);
        memcpy(dec_src + dl_h + dl_b, metal_attn_dec_f16_plain_head, dl_p);
        memcpy(dec_src + dl_h + dl_b + dl_p, metal_attn_dec_f16_body, dl_b + 1u);
        source = metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", dec_src);
        free(dec_src);
        if (source == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal: failed to create decode attention shader source");
            return GEIST_E_BACKEND;
        }
        err                            = nullptr;
        st->attn_qnorm_dec_f16_library = metal_msg_send_id_id_id_err(
                st, st->device, "newLibraryWithSource:options:error:", source, nullptr, &err);
        if (st->attn_qnorm_dec_f16_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(be,
                                    GEIST_E_BACKEND,
                                    "metal: decode attention shader compile failed%s%s",
                                    msg != nullptr ? ": " : "",
                                    msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
        s = metal_create_named_pipeline(be,
                                        st->attn_qnorm_dec_f16_library,
                                        ns_string,
                                        "attention_qnorm_dec_f16",
                                        &st->attention_qnorm_dec_f16_function,
                                        &st->attention_qnorm_dec_f16_pipeline);
        if (s == GEIST_OK) {
            s = metal_create_named_pipeline(be,
                                            st->attn_qnorm_dec_f16_library,
                                            ns_string,
                                            "attention_dec_f16",
                                            &st->attention_dec_f16_function,
                                            &st->attention_dec_f16_pipeline);
        }
    }
    if (s == GEIST_OK) {
        source = metal_msg_send_id_cstr(
                st, ns_string, "stringWithUTF8String:", metal_attn_dec_combine_source);
        if (source == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal: failed to create attention combine shader source");
            return GEIST_E_BACKEND;
        }
        err                          = nullptr;
        st->attn_dec_combine_library = metal_msg_send_id_id_id_err(
                st, st->device, "newLibraryWithSource:options:error:", source, nullptr, &err);
        if (st->attn_dec_combine_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(be,
                                    GEIST_E_BACKEND,
                                    "metal: attention combine shader compile failed%s%s",
                                    msg != nullptr ? ": " : "",
                                    msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
        s = metal_create_named_pipeline(be,
                                        st->attn_dec_combine_library,
                                        ns_string,
                                        "attention_dec_combine",
                                        &st->attention_dec_combine_function,
                                        &st->attention_dec_combine_pipeline);
    }
    if (s == GEIST_OK) {
        const size_t len_a     = strlen(metal_attn_flash_sg_f16_source_a);
        const size_t len_b     = strlen(metal_attn_flash_sg_f16_source_b);
        const size_t len_c     = strlen(metal_attn_flash_sg_f16_plain_head);
        char        *flash_src = malloc(len_a + 2u * len_b + len_c + 1u);
        if (flash_src == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_OOM, "metal: flash attention shader source alloc failed");
            return GEIST_E_OOM;
        }
        memcpy(flash_src, metal_attn_flash_sg_f16_source_a, len_a);
        memcpy(flash_src + len_a, metal_attn_flash_sg_f16_source_b, len_b);
        memcpy(flash_src + len_a + len_b, metal_attn_flash_sg_f16_plain_head, len_c);
        memcpy(flash_src + len_a + len_b + len_c, metal_attn_flash_sg_f16_source_b, len_b + 1u);
        source = metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", flash_src);
        free(flash_src);
        if (source == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal: failed to create flash attention shader source");
            return GEIST_E_BACKEND;
        }
        err                           = nullptr;
        st->attn_flash_sg_f16_library = metal_msg_send_id_id_id_err(
                st, st->device, "newLibraryWithSource:options:error:", source, nullptr, &err);
        if (st->attn_flash_sg_f16_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(be,
                                    GEIST_E_BACKEND,
                                    "metal: flash attention shader compile failed%s%s",
                                    msg != nullptr ? ": " : "",
                                    msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
        s = metal_create_named_pipeline(be,
                                        st->attn_flash_sg_f16_library,
                                        ns_string,
                                        "attention_qnorm_flash_sg_f16",
                                        &st->attention_qnorm_flash_sg_f16_function,
                                        &st->attention_qnorm_flash_sg_f16_pipeline);
        if (s == GEIST_OK) {
            s = metal_create_named_pipeline(be,
                                            st->attn_flash_sg_f16_library,
                                            ns_string,
                                            "attention_flash_sg_f16",
                                            &st->attention_flash_sg_f16_function,
                                            &st->attention_flash_sg_f16_pipeline);
        }
    }
    if (s == GEIST_OK) {
        const size_t l_a     = strlen(metal_attn_flash_sg8_f16_source_a);
        const size_t l_b     = strlen(metal_attn_flash_sg8_f16_source_b);
        char        *sg8_src = malloc(l_a + l_b + 1u);
        if (sg8_src == nullptr) {
            geist_backend_set_error(be, GEIST_E_OOM, "metal: sg8 flash shader source alloc failed");
            return GEIST_E_OOM;
        }
        memcpy(sg8_src, metal_attn_flash_sg8_f16_source_a, l_a);
        memcpy(sg8_src + l_a, metal_attn_flash_sg8_f16_source_b, l_b + 1u);
        source = metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", sg8_src);
        free(sg8_src);
        if (source == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal: failed to create sg8 flash shader source");
            return GEIST_E_BACKEND;
        }
        err                            = nullptr;
        st->attn_flash_sg8_f16_library = metal_msg_send_id_id_id_err(
                st, st->device, "newLibraryWithSource:options:error:", source, nullptr, &err);
        if (st->attn_flash_sg8_f16_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(be,
                                    GEIST_E_BACKEND,
                                    "metal: sg8 flash shader compile failed%s%s",
                                    msg != nullptr ? ": " : "",
                                    msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
        s = metal_create_named_pipeline(be,
                                        st->attn_flash_sg8_f16_library,
                                        ns_string,
                                        "attention_flash_sg8_f16",
                                        &st->attention_flash_sg8_f16_function,
                                        &st->attention_flash_sg8_f16_pipeline);
    }
    if (s == GEIST_OK) {
        const size_t d_a        = strlen(metal_attn_dec512_f16_source_a);
        const size_t d_b        = strlen(metal_attn_dec512_f16_source_b);
        char        *dec512_src = malloc(d_a + d_b + 1u);
        if (dec512_src == nullptr) {
            geist_backend_set_error(be, GEIST_E_OOM, "metal: dec512 shader source alloc failed");
            return GEIST_E_OOM;
        }
        memcpy(dec512_src, metal_attn_dec512_f16_source_a, d_a);
        memcpy(dec512_src + d_a, metal_attn_dec512_f16_source_b, d_b + 1u);
        source = metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", dec512_src);
        free(dec512_src);
        if (source == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal: failed to create dec512 shader source");
            return GEIST_E_BACKEND;
        }
        err                         = nullptr;
        st->attn_dec512_f16_library = metal_msg_send_id_id_id_err(
                st, st->device, "newLibraryWithSource:options:error:", source, nullptr, &err);
        if (st->attn_dec512_f16_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(be,
                                    GEIST_E_BACKEND,
                                    "metal: dec512 shader compile failed%s%s",
                                    msg != nullptr ? ": " : "",
                                    msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
        s = metal_create_named_pipeline(be,
                                        st->attn_dec512_f16_library,
                                        ns_string,
                                        "attention_dec512_f16",
                                        &st->attention_dec512_f16_function,
                                        &st->attention_dec512_f16_pipeline);
    }
    return s;
}

static void metal_encode_q4k_linear(struct metal_state            *st,
                                    void                          *enc,
                                    const struct geist_tensor     *x,
                                    const struct geist_tensor     *w,
                                    const struct geist_tensor     *y,
                                    const struct metal_q4k_params *params,
                                    bool                           m_tile8) {
    const bool m_tile16    = m_tile8 && params->rows >= METAL_Q4K_M16_TILE;
    const bool m_tile16_n2 = m_tile16 && st->use_q4k_m16_n2 &&
                             st->q4k_matmul_m16_n2_pipeline != nullptr && params->n_out >= 2u;
    const bool m_tile_sg   = m_tile8 && st->use_q4k_mm_sg && st->q4k_mm_sg_pipeline != nullptr &&
                             params->rows >= 32u && params->n_out >= 64u &&
                             (params->rows % 32u) == 0u && (params->n_out % 64u) == 0u;
    /* interior fast variant: no bounds checks, vectorized activation
     * staging (needs n_in%32 and 8-float-aligned x rows). */
    const bool m_tile_sg_fast = m_tile_sg && st->q4k_mm_sg_fast_pipeline != nullptr &&
                                (params->n_in % 32u) == 0u && (params->x_offset % 8u) == 0u &&
                                (params->x_row_stride % 8u) == 0u;
    const bool m_tile8_active = m_tile8 && !m_tile16;
    const bool n_tile4        = st->use_q4k_n4 && !m_tile8 && params->rows == 1u;
    metal_msg_send_set_pipeline(st,
                                enc,
                                m_tile_sg_fast   ? st->q4k_mm_sg_fast_pipeline
                                : m_tile_sg      ? st->q4k_mm_sg_pipeline
                                : m_tile16_n2    ? st->q4k_matmul_m16_n2_pipeline
                                : m_tile16       ? st->q4k_matmul_m16_pipeline
                                : m_tile8_active ? st->q4k_matmul_m8_pipeline
                                : n_tile4        ? st->q4k_n4_pipeline
                                                 : st->q4k_pipeline);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, 0, 0);
    metal_msg_send_set_buffer(st, enc, w->buffer->buffer, 0, 1);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, 0, 2);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 3);
    if (m_tile_sg) {
        metal_msg_send_set_threadgroup_memory(st, enc, m_tile_sg_fast ? 6144u : 8192u, 0u);
    }

    const struct metal_size groups = {
            .width  = n_tile4       ? (params->n_out + 3u) / 4u
                      : m_tile_sg   ? (params->rows + 31u) / 32u
                      : m_tile16_n2 ? (params->n_out + 1u) / 2u
                                    : params->n_out,
            .height = m_tile_sg  ? (params->n_out + 63u) / 64u
                      : m_tile16 ? (params->rows + METAL_Q4K_M16_TILE - 1u) / METAL_Q4K_M16_TILE
                      : m_tile8_active ? (params->rows + METAL_Q4K_M_TILE - 1u) / METAL_Q4K_M_TILE
                                       : params->rows,
            .depth  = 1,
    };
    const struct metal_size threads = {
            .width  = m_tile_sg ? 32u
                      : n_tile4 ? METAL_Q4K_N4_THREADS
                                : METAL_Q4K_THREADS_PER_ROW,
            .height = m_tile_sg ? 4u : 1u,
            .depth  = 1,
    };
    const enum metal_profile_stage profile_stage =
            n_tile4 ? METAL_PROFILE_DISPATCH_Q4K_LINEAR_N4 : METAL_PROFILE_DISPATCH_Q4K_LINEAR_BASE;
    metal_profile_add_dispatch(st, profile_stage, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

static void metal_encode_q6k_linear(struct metal_state            *st,
                                    void                          *enc,
                                    const struct geist_tensor     *x,
                                    const struct geist_tensor     *w,
                                    const struct geist_tensor     *y,
                                    const struct metal_q4k_params *params,
                                    bool                           m_tile8) {
    /* rows==1 always takes the matvec kernel — the 64-row GEMM tile is a
     * waste for a single row, and n4 (llama mul_mv structure) wins at any
     * n_out that fills at least one threadgroup. */
    const bool single_row = params->rows == 1u;
    const bool n_tile4    = st->use_q6k_n4 && single_row && params->n_out >= 4u;
    const bool m_tile_sg  = m_tile8 && !n_tile4 && st->q6k_matmul_sg_pipeline != nullptr;
    /* interior fast variant: no bounds checks, vectorized activation
     * staging, direct simdgroup_store output. */
    const bool m_tile_sg_fast = m_tile_sg && st->q6k_matmul_sg_fast_pipeline != nullptr &&
                                (params->rows % 64u) == 0u && (params->n_out % 32u) == 0u &&
                                (params->x_offset % 8u) == 0u && (params->x_row_stride % 8u) == 0u;
    const bool m_tile16 = m_tile8 && !n_tile4 && !m_tile_sg && params->rows >= METAL_Q4K_M16_TILE;
    const bool m_tile8_active = m_tile8 && !n_tile4 && !m_tile_sg && !m_tile16;
    metal_msg_send_set_pipeline(st,
                                enc,
                                m_tile_sg_fast   ? st->q6k_matmul_sg_fast_pipeline
                                : m_tile_sg      ? st->q6k_matmul_sg_pipeline
                                : m_tile16       ? st->q6k_matmul_m16_pipeline
                                : m_tile8_active ? st->q6k_matmul_m8_pipeline
                                : n_tile4        ? st->q6k_n4_pipeline
                                                 : st->q6k_pipeline);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, 0, 0);
    metal_msg_send_set_buffer(st, enc, w->buffer->buffer, 0, 1);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, 0, 2);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 3);

    /* The q6k sg kernels are 128-thread / 4-simdgroup with a 32-output x
     * 64-batch-row tile (b0=tg.y*64, o0=tg.x*32). The old (n_out+7)/8 x
     * 32-thread dispatch here only ever ran simdgroup 0 — a latent bug the
     * old engine never hit because its prefill routed q6k through the
     * fused blocks, not this vtbl op. */
    const struct metal_size groups = {
            .width  = m_tile_sg ? (params->n_out + 31u) / 32u
                      : n_tile4 ? (params->n_out + 3u) / 4u
                                : params->n_out,
            .height = m_tile_sg  ? (params->rows + 63u) / 64u
                      : m_tile16 ? (params->rows + METAL_Q4K_M16_TILE - 1u) / METAL_Q4K_M16_TILE
                      : m_tile8_active ? (params->rows + METAL_Q4K_M_TILE - 1u) / METAL_Q4K_M_TILE
                                       : params->rows,
            .depth  = 1,
    };
    const struct metal_size threads = {
            .width  = m_tile_sg ? 32u
                      : n_tile4 ? METAL_Q4K_N4_THREADS
                                : METAL_Q4K_THREADS_PER_ROW,
            .height = m_tile_sg ? 4u : 1,
            .depth  = 1,
    };
    const enum metal_profile_stage profile_stage =
            n_tile4 ? METAL_PROFILE_DISPATCH_Q6K_LINEAR_N4 : METAL_PROFILE_DISPATCH_Q6K_LINEAR_BASE;
    metal_profile_add_dispatch(st, profile_stage, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

static void metal_encode_rmsnorm_rows(struct metal_state             *st,
                                      void                           *enc,
                                      const struct geist_tensor      *x,
                                      const struct geist_tensor      *w,
                                      const struct geist_tensor      *y,
                                      const struct metal_rows_params *params) {

    metal_msg_send_set_pipeline(st,
                                enc,
                                st->use_rmsnorm_simd ? st->rmsnorm_rows_simd_pipeline
                                                     : st->rmsnorm_rows_pipeline);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, 0, 0);
    metal_msg_send_set_buffer(st, enc, w->buffer->buffer, 0, 1);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, 0, 2);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 3);
    const struct metal_size groups = {
            .width  = params->rows,
            .height = 1,
            .depth  = 1,
    };
    const struct metal_size threads = {
            .width  = METAL_ELEM_THREADS,
            .height = 1,
            .depth  = 1,
    };
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_RMSNORM_ROWS, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

static void metal_encode_rmsnorm_add_rows(struct metal_state                  *st,
                                          void                                *enc,
                                          const struct geist_tensor           *res,
                                          const struct geist_tensor           *x,
                                          const struct geist_tensor           *w,
                                          const struct geist_tensor           *y,
                                          const struct metal_post_norm_params *params) {

    metal_msg_send_set_pipeline(st,
                                enc,
                                st->use_rmsnorm_simd ? st->rmsnorm_add_rows_simd_pipeline
                                                     : st->rmsnorm_add_rows_pipeline);
    metal_msg_send_set_buffer(st, enc, res->buffer->buffer, 0, 0);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, 0, 1);
    metal_msg_send_set_buffer(st, enc, w->buffer->buffer, 0, 2);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, 0, 3);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 4);
    const struct metal_size groups = {
            .width  = params->rows,
            .height = 1,
            .depth  = 1,
    };
    const struct metal_size threads = {
            .width  = METAL_ELEM_THREADS,
            .height = 1,
            .depth  = 1,
    };
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_RMSNORM_ADD_ROWS, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

static void metal_encode_add_rows(struct metal_state                    *st,
                                  void                                  *enc,
                                  const struct geist_tensor             *a,
                                  const struct geist_tensor             *b,
                                  const struct geist_tensor             *y,
                                  const struct metal_binary_rows_params *params) {
    metal_msg_send_set_pipeline(st, enc, st->add_rows_pipeline);
    metal_msg_send_set_buffer(st, enc, a->buffer->buffer, 0, 0);
    metal_msg_send_set_buffer(st, enc, b->buffer->buffer, 0, 1);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, 0, 2);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 3);
    const struct metal_size groups = {
            .width  = (params->rows * params->cols + METAL_ELEM_THREADS - 1u) / METAL_ELEM_THREADS,
            .height = 1,
            .depth  = 1,
    };
    const struct metal_size threads = {
            .width  = METAL_ELEM_THREADS,
            .height = 1,
            .depth  = 1,
    };
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_ADD_ROWS, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

static void metal_encode_mul_rows(struct metal_state                    *st,
                                  void                                  *enc,
                                  const struct geist_tensor             *a,
                                  const struct geist_tensor             *b,
                                  const struct geist_tensor             *y,
                                  const struct metal_binary_rows_params *params) {
    metal_msg_send_set_pipeline(st, enc, st->mul_rows_pipeline);
    metal_msg_send_set_buffer(st, enc, a->buffer->buffer, 0, 0);
    metal_msg_send_set_buffer(st, enc, b->buffer->buffer, 0, 1);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, 0, 2);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 3);
    const struct metal_size groups = {
            .width  = (params->rows * params->cols + METAL_ELEM_THREADS - 1u) / METAL_ELEM_THREADS,
            .height = 1,
            .depth  = 1,
    };
    const struct metal_size threads = {
            .width  = METAL_ELEM_THREADS,
            .height = 1,
            .depth  = 1,
    };
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_MUL_ROWS, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

static void metal_encode_gelu_mul_rows(struct metal_state                    *st,
                                       void                                  *enc,
                                       const struct geist_tensor             *a,
                                       const struct geist_tensor             *b,
                                       const struct geist_tensor             *y,
                                       const struct metal_binary_rows_params *params) {

    metal_msg_send_set_pipeline(st, enc, st->gelu_mul_rows_pipeline);
    metal_msg_send_set_buffer(st, enc, a->buffer->buffer, 0, 0);
    metal_msg_send_set_buffer(st, enc, b->buffer->buffer, 0, 1);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, 0, 2);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 3);
    const struct metal_size groups = {
            .width  = (params->rows * params->cols + METAL_ELEM_THREADS - 1u) / METAL_ELEM_THREADS,
            .height = 1,
            .depth  = 1,
    };
    const struct metal_size threads = {
            .width  = METAL_ELEM_THREADS,
            .height = 1,
            .depth  = 1,
    };
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_GELU_MUL_ROWS, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

static void metal_encode_scale_rows(struct metal_state                   *st,
                                    void                                 *enc,
                                    const struct geist_tensor            *x,
                                    const struct geist_tensor            *y,
                                    const struct metal_scale_rows_params *params) {
    metal_msg_send_set_pipeline(st, enc, st->scale_rows_pipeline);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, 0, 0);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, 0, 1);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 2);
    const struct metal_size groups = {
            .width  = (params->rows * params->cols + METAL_ELEM_THREADS - 1u) / METAL_ELEM_THREADS,
            .height = 1,
            .depth  = 1,
    };
    const struct metal_size threads = {
            .width  = METAL_ELEM_THREADS,
            .height = 1,
            .depth  = 1,
    };
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_SCALE_ROWS, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

static void metal_encode_gelu_rows(struct metal_state                   *st,
                                   void                                 *enc,
                                   const struct geist_tensor            *x,
                                   const struct geist_tensor            *y,
                                   const struct metal_scale_rows_params *params) {
    metal_msg_send_set_pipeline(st, enc, st->gelu_rows_pipeline);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, 0, 0);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, 0, 1);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 2);
    const struct metal_size groups = {
            .width  = (params->rows * params->cols + METAL_ELEM_THREADS - 1u) / METAL_ELEM_THREADS,
            .height = 1,
            .depth  = 1,
    };
    const struct metal_size threads = {
            .width  = METAL_ELEM_THREADS,
            .height = 1,
            .depth  = 1,
    };
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_GELU_ROWS, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

static void metal_encode_embed_lookup_scaled(struct metal_state              *st,
                                             void                            *enc,
                                             const struct geist_tensor       *embed_table,
                                             const struct geist_tensor       *out,
                                             const struct metal_embed_params *params) {

    metal_msg_send_set_pipeline(st, enc, st->embed_lookup_scaled_pipeline);
    metal_msg_send_set_buffer(st, enc, embed_table->buffer->buffer, 0, 0);
    metal_msg_send_set_buffer(st, enc, out->buffer->buffer, 0, 1);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 2);
    const struct metal_size groups = {
            .width  = (params->n + METAL_ELEM_THREADS - 1u) / METAL_ELEM_THREADS,
            .height = 1,
            .depth  = 1,
    };
    const struct metal_size threads = {
            .width  = METAL_ELEM_THREADS,
            .height = 1,
            .depth  = 1,
    };
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_EMBED, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

static void metal_encode_rope_rows(struct metal_state             *st,
                                   void                           *enc,
                                   struct geist_tensor            *x,
                                   const struct geist_tensor      *cos,
                                   const struct geist_tensor      *sin,
                                   const struct metal_rope_params *params) {

    metal_msg_send_set_pipeline(st, enc, st->rope_rows_pipeline);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, 0, 0);
    metal_msg_send_set_buffer(st, enc, cos->buffer->buffer, 0, 1);
    metal_msg_send_set_buffer(st, enc, sin->buffer->buffer, 0, 2);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 3);
    const size_t            half   = (size_t) params->head_dim / 2u;
    const size_t            total  = (size_t) params->rows * (size_t) params->heads * half;
    const struct metal_size groups = {
            .width  = (total + METAL_ELEM_THREADS - 1u) / METAL_ELEM_THREADS,
            .height = 1,
            .depth  = 1,
    };
    const struct metal_size threads = {
            .width  = METAL_ELEM_THREADS,
            .height = 1,
            .depth  = 1,
    };
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_ROPE_ROWS, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

static void metal_encode_attention_rows(struct metal_state                  *st,
                                        void                                *enc,
                                        const struct geist_tensor           *q,
                                        const struct geist_tensor           *k_cache,
                                        const struct geist_tensor           *v_cache,
                                        const struct geist_tensor           *y,
                                        const struct metal_attention_params *params) {

    void *pipeline = k_cache->dtype == GEIST_DTYPE_F16 ? st->attention_rows_f16_pipeline
                                                       : st->attention_rows_pipeline;
    metal_msg_send_set_pipeline(st, enc, pipeline);
    metal_msg_send_set_buffer(st, enc, q->buffer->buffer, 0, 0);
    metal_msg_send_set_buffer(st, enc, k_cache->buffer->buffer, 0, 1);
    metal_msg_send_set_buffer(st, enc, v_cache->buffer->buffer, 0, 2);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, 0, 3);
    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 4);
    const struct metal_size groups = {
            .width  = params->rows,
            .height = params->q_heads,
            .depth  = 1,
    };
    const struct metal_size threads = {
            .width  = METAL_ELEM_THREADS,
            .height = 1,
            .depth  = 1,
    };
    /* Books the SCALAR stage: this two-pass kernel is the O(kv) fallback
     * the head_dim-512 full-attention layers take (flash gate is <=256) —
     * it was invisible to the profiler/skips until 2026-07-04. */
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_ATTENTION_QNORM_ROWS, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

static void metal_encode_f32_matmul(struct metal_state            *st,
                                    void                          *enc,
                                    const struct geist_tensor     *x,
                                    const struct geist_tensor     *w,
                                    const struct geist_tensor     *y,
                                    const struct metal_f32_params *params) {
    /* Multi-row (prefill): full-tile shapes take the 64x32 4-simdgroup GEMM
     * (mm_sg structure, f32 staging = bit-identical to the 8x8 kernel);
     * others the 8x8 simdgroup GEMM. Single-row keeps the reduction kernel. */
    const bool use_sg = params->rows > 1u && st->f32_matmul_sg_pipeline != nullptr;
    const bool use_mm = use_sg && st->f32_matmul_mm_pipeline != nullptr &&
                        !metal_env_disabled("GEIST_METAL_F32_MM") && (params->rows % 32u) == 0u &&
                        (params->n_out % 64u) == 0u && (params->n_in % 32u) == 0u &&
                        (params->x_offset % 8u) == 0u && (params->x_row_stride % 8u) == 0u &&
                        (params->w_offset % 4u) == 0u;
    metal_msg_send_set_pipeline(st,
                                enc,
                                use_mm   ? st->f32_matmul_mm_pipeline
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
            .width  = use_mm   ? params->rows / 32u
                      : use_sg ? (params->n_out + 7u) / 8u
                               : params->n_out,
            .height = use_mm   ? params->n_out / 64u
                      : use_sg ? (params->rows + 7u) / 8u
                               : params->rows,
            .depth  = 1,
    };
    const struct metal_size threads = {
            .width  = use_sg ? 32u : METAL_ELEM_THREADS,
            .height = use_mm ? 4u : 1,
            .depth  = 1,
    };
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_F32_MATMUL, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
}

[[nodiscard]] static enum geist_status metal_f32_linear(struct geist_backend      *be,
                                                        const struct geist_tensor *x,
                                                        const struct geist_tensor *w,
                                                        struct geist_tensor       *y,
                                                        bool                       matrix) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }

    size_t rows         = 1;
    size_t n_in         = 0;
    size_t x_offset     = 0;
    size_t x_row_stride = 0;
    size_t y_rows       = 1;
    size_t y_cols       = 0;
    size_t y_offset     = 0;
    size_t y_row_stride = 0;
    size_t w_rows       = 0;
    size_t w_cols       = 0;
    size_t w_offset     = 0;
    size_t w_row_stride = 0;

    bool ok = false;
    if (matrix) {
        ok = metal_tensor_is_f32_matrix(x, &rows, &n_in, &x_offset, &x_row_stride) &&
             metal_tensor_is_f32_matrix(y, &y_rows, &y_cols, &y_offset, &y_row_stride);
    } else {
        ok           = metal_tensor_is_f32_vector(x, &n_in, &x_offset) &&
                       metal_tensor_is_f32_vector(y, &y_cols, &y_offset);
        x_row_stride = n_in;
        y_row_stride = y_cols;
    }
    ok = ok && metal_tensor_is_f32_matrix(w, &w_rows, &w_cols, &w_offset, &w_row_stride) &&
         w_row_stride == w_cols && w_cols == n_in && y_cols == w_rows && y_rows == rows;
    if (!ok) {
        geist_backend_set_error(
                be,
                GEIST_E_UNSUPPORTED,
                matrix ? "metal F32 matmul: expected x F32 [rows,n], w F32 [out,n], y F32 "
                         "[rows,out]"
                       : "metal F32 matvec: expected x F32 [n], w F32 [out,n], y F32 [out]");
        return GEIST_E_UNSUPPORTED;
    }
    if (rows > UINT32_MAX || n_in > UINT32_MAX || w_rows > UINT32_MAX || x_offset > UINT32_MAX ||
        w_offset > UINT32_MAX || y_offset > UINT32_MAX || x_row_stride > UINT32_MAX ||
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

    struct metal_state           *st     = be->state;
    const struct metal_f32_params params = {
            .n_in         = (uint32_t) n_in,
            .n_out        = (uint32_t) w_rows,
            .rows         = (uint32_t) rows,
            .x_offset     = (uint32_t) x_offset,
            .w_offset     = (uint32_t) w_offset,
            .y_offset     = (uint32_t) y_offset,
            .x_row_stride = (uint32_t) x_row_stride,
            .y_row_stride = (uint32_t) y_row_stride,
    };

    if (st->sequence_active) {
        if (st->sequence_compute_encoder == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal F32 linear: command sequence has no encoder");
            return GEIST_E_BACKEND;
        }
        metal_encode_f32_matmul(st, metal_sequence_encoder(st), x, w, y, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }

    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    if (cmd == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal F32 linear: command buffer failed");
        return GEIST_E_BACKEND;
    }
    void *enc = metal_msg_send_id0(st, cmd, "computeCommandEncoder");
    if (enc == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal F32 linear: encoder failed");
        return GEIST_E_BACKEND;
    }
    metal_encode_f32_matmul(st, enc, x, w, y, &params);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");

    void *err = metal_msg_send_id0(st, cmd, "error");
    if (err != nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal F32 linear: command failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status metal_matvec_f32_dense(struct geist_backend      *be,
                                                              const struct geist_tensor *x,
                                                              const struct geist_tensor *w,
                                                              struct geist_tensor       *y) {

    return metal_f32_linear(be, x, w, y, false);
}

[[nodiscard]] static enum geist_status metal_matmul_f32_dense(struct geist_backend      *be,
                                                              const struct geist_tensor *x,
                                                              const struct geist_tensor *w,
                                                              struct geist_tensor       *y) {

    return metal_f32_linear(be, x, w, y, true);
}

[[nodiscard]] static enum geist_status metal_rmsnorm(struct geist_backend      *be,
                                                     const struct geist_tensor *x,
                                                     const struct geist_tensor *w,
                                                     float                      eps,
                                                     struct geist_tensor       *y) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows = 0, cols = 0, x_off = 0, x_stride = 0;
    size_t y_rows = 0, y_cols = 0, y_off = 0, y_stride = 0;
    size_t w_n = 0, w_off = 0;
    if (!metal_tensor_is_f32_rows(x, &rows, &cols, &x_off, &x_stride) ||
        !metal_tensor_is_f32_rows(y, &y_rows, &y_cols, &y_off, &y_stride) ||
        !metal_tensor_is_f32_vector(w, &w_n, &w_off) || y_rows != rows || y_cols != cols ||
        w_n != cols) {
        return GEIST_E_UNSUPPORTED;
    }
    if (rows > UINT32_MAX || cols > UINT32_MAX || x_off > UINT32_MAX || y_off > UINT32_MAX ||
        w_off > UINT32_MAX || x_stride > UINT32_MAX || y_stride > UINT32_MAX ||
        x->buffer->owner != be->state || w->buffer->owner != be->state ||
        y->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    struct metal_state            *st     = be->state;
    const struct metal_rows_params params = {
            .rows         = (uint32_t) rows,
            .cols         = (uint32_t) cols,
            .x_offset     = (uint32_t) x_off,
            .w_offset     = (uint32_t) w_off,
            .y_offset     = (uint32_t) y_off,
            .x_row_stride = (uint32_t) x_stride,
            .y_row_stride = (uint32_t) y_stride,
            .eps          = eps,
    };
    if (st->sequence_active) {
        metal_encode_rmsnorm_rows(st, metal_sequence_encoder(st), x, w, y, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    void *enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
    if (cmd == nullptr || enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_rmsnorm_rows(st, enc, x, w, y, &params);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    return metal_msg_send_id0(st, cmd, "error") == nullptr ? GEIST_OK : GEIST_E_BACKEND;
}

[[nodiscard]] static enum geist_status metal_rmsnorm_add(struct geist_backend      *be,
                                                         const struct geist_tensor *res,
                                                         const struct geist_tensor *x,
                                                         const struct geist_tensor *w,
                                                         float                      eps,
                                                         struct geist_tensor       *y) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows = 0, cols = 0, x_off = 0, x_stride = 0;
    size_t r_rows = 0, r_cols = 0, r_off = 0, r_stride = 0;
    size_t y_rows = 0, y_cols = 0, y_off = 0, y_stride = 0;
    size_t w_n = 0, w_off = 0;
    if (!metal_tensor_is_f32_rows(x, &rows, &cols, &x_off, &x_stride) ||
        !metal_tensor_is_f32_rows(res, &r_rows, &r_cols, &r_off, &r_stride) ||
        !metal_tensor_is_f32_rows(y, &y_rows, &y_cols, &y_off, &y_stride) ||
        !metal_tensor_is_f32_vector(w, &w_n, &w_off) || r_rows != rows || r_cols != cols ||
        y_rows != rows || y_cols != cols || w_n != cols) {
        return GEIST_E_UNSUPPORTED;
    }
    if (rows > UINT32_MAX || cols > UINT32_MAX || x_off > UINT32_MAX || r_off > UINT32_MAX ||
        y_off > UINT32_MAX || w_off > UINT32_MAX || x_stride > UINT32_MAX ||
        r_stride > UINT32_MAX || y_stride > UINT32_MAX || x->buffer->owner != be->state ||
        res->buffer->owner != be->state || w->buffer->owner != be->state ||
        y->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    struct metal_state                 *st     = be->state;
    const struct metal_post_norm_params params = {
            .rows                = (uint32_t) rows,
            .cols                = (uint32_t) cols,
            .residual_offset     = (uint32_t) r_off,
            .x_offset            = (uint32_t) x_off,
            .w_offset            = (uint32_t) w_off,
            .y_offset            = (uint32_t) y_off,
            .residual_row_stride = (uint32_t) r_stride,
            .x_row_stride        = (uint32_t) x_stride,
            .y_row_stride        = (uint32_t) y_stride,
            .eps                 = eps,
    };
    if (st->sequence_active) {
        metal_encode_rmsnorm_add_rows(st, metal_sequence_encoder(st), res, x, w, y, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    void *enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
    if (cmd == nullptr || enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_rmsnorm_add_rows(st, enc, res, x, w, y, &params);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    return metal_msg_send_id0(st, cmd, "error") == nullptr ? GEIST_OK : GEIST_E_BACKEND;
}

[[nodiscard]] static enum geist_status metal_add(struct geist_backend      *be,
                                                 const struct geist_tensor *a,
                                                 const struct geist_tensor *b,
                                                 struct geist_tensor       *y) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows = 0, cols = 0, a_off = 0, a_stride = 0;
    size_t b_rows = 0, b_cols = 0, b_off = 0, b_stride = 0;
    size_t y_rows = 0, y_cols = 0, y_off = 0, y_stride = 0;
    if (!metal_tensor_is_f32_rows(a, &rows, &cols, &a_off, &a_stride) ||
        !metal_tensor_is_f32_rows(b, &b_rows, &b_cols, &b_off, &b_stride) ||
        !metal_tensor_is_f32_rows(y, &y_rows, &y_cols, &y_off, &y_stride) || b_rows != rows ||
        y_rows != rows || b_cols != cols || y_cols != cols) {
        return GEIST_E_UNSUPPORTED;
    }
    if (rows > UINT32_MAX || cols > UINT32_MAX || a_off > UINT32_MAX || b_off > UINT32_MAX ||
        y_off > UINT32_MAX || a_stride > UINT32_MAX || b_stride > UINT32_MAX ||
        y_stride > UINT32_MAX || a->buffer->owner != be->state || b->buffer->owner != be->state ||
        y->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    struct metal_state                   *st     = be->state;
    const struct metal_binary_rows_params params = {
            .rows         = (uint32_t) rows,
            .cols         = (uint32_t) cols,
            .a_offset     = (uint32_t) a_off,
            .b_offset     = (uint32_t) b_off,
            .y_offset     = (uint32_t) y_off,
            .a_row_stride = (uint32_t) a_stride,
            .b_row_stride = (uint32_t) b_stride,
            .y_row_stride = (uint32_t) y_stride,
    };
    if (st->sequence_active) {
        metal_encode_add_rows(st, metal_sequence_encoder(st), a, b, y, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    void *enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
    if (cmd == nullptr || enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_add_rows(st, enc, a, b, y, &params);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    return metal_msg_send_id0(st, cmd, "error") == nullptr ? GEIST_OK : GEIST_E_BACKEND;
}

[[nodiscard]] static enum geist_status metal_mul(struct geist_backend      *be,
                                                 const struct geist_tensor *a,
                                                 const struct geist_tensor *b,
                                                 struct geist_tensor       *y) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows = 0, cols = 0, a_off = 0, a_stride = 0;
    size_t b_rows = 0, b_cols = 0, b_off = 0, b_stride = 0;
    size_t y_rows = 0, y_cols = 0, y_off = 0, y_stride = 0;
    if (!metal_tensor_is_f32_rows(a, &rows, &cols, &a_off, &a_stride) ||
        !metal_tensor_is_f32_rows(b, &b_rows, &b_cols, &b_off, &b_stride) ||
        !metal_tensor_is_f32_rows(y, &y_rows, &y_cols, &y_off, &y_stride) || b_rows != rows ||
        y_rows != rows || b_cols != cols || y_cols != cols) {
        return GEIST_E_UNSUPPORTED;
    }
    if (rows > UINT32_MAX || cols > UINT32_MAX || a_off > UINT32_MAX || b_off > UINT32_MAX ||
        y_off > UINT32_MAX || a_stride > UINT32_MAX || b_stride > UINT32_MAX ||
        y_stride > UINT32_MAX || a->buffer->owner != be->state || b->buffer->owner != be->state ||
        y->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    struct metal_state                   *st     = be->state;
    const struct metal_binary_rows_params params = {
            .rows         = (uint32_t) rows,
            .cols         = (uint32_t) cols,
            .a_offset     = (uint32_t) a_off,
            .b_offset     = (uint32_t) b_off,
            .y_offset     = (uint32_t) y_off,
            .a_row_stride = (uint32_t) a_stride,
            .b_row_stride = (uint32_t) b_stride,
            .y_row_stride = (uint32_t) y_stride,
    };
    if (st->sequence_active) {
        metal_encode_mul_rows(st, metal_sequence_encoder(st), a, b, y, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    void *enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
    if (cmd == nullptr || enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_mul_rows(st, enc, a, b, y, &params);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    return metal_msg_send_id0(st, cmd, "error") == nullptr ? GEIST_OK : GEIST_E_BACKEND;
}

[[nodiscard]] static enum geist_status metal_scale_f32(struct geist_backend      *be,
                                                       const struct geist_tensor *x,
                                                       float                      scale,
                                                       struct geist_tensor       *y) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows = 0, cols = 0, x_off = 0, x_stride = 0;
    size_t y_rows = 0, y_cols = 0, y_off = 0, y_stride = 0;
    if (!metal_tensor_is_f32_rows(x, &rows, &cols, &x_off, &x_stride) ||
        !metal_tensor_is_f32_rows(y, &y_rows, &y_cols, &y_off, &y_stride) || y_rows != rows ||
        y_cols != cols) {
        return GEIST_E_UNSUPPORTED;
    }
    if (rows > UINT32_MAX || cols > UINT32_MAX || x_off > UINT32_MAX || y_off > UINT32_MAX ||
        x_stride > UINT32_MAX || y_stride > UINT32_MAX || x->buffer->owner != be->state ||
        y->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    struct metal_state                  *st     = be->state;
    const struct metal_scale_rows_params params = {
            .rows         = (uint32_t) rows,
            .cols         = (uint32_t) cols,
            .x_offset     = (uint32_t) x_off,
            .y_offset     = (uint32_t) y_off,
            .x_row_stride = (uint32_t) x_stride,
            .y_row_stride = (uint32_t) y_stride,
            .scale        = scale,
    };
    if (st->sequence_active) {
        metal_encode_scale_rows(st, metal_sequence_encoder(st), x, y, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    void *enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
    if (cmd == nullptr || enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_scale_rows(st, enc, x, y, &params);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    return metal_msg_send_id0(st, cmd, "error") == nullptr ? GEIST_OK : GEIST_E_BACKEND;
}

[[nodiscard]] static enum geist_status
metal_gelu_tanh(struct geist_backend *be, const struct geist_tensor *x, struct geist_tensor *y) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows = 0, cols = 0, x_off = 0, x_stride = 0;
    size_t y_rows = 0, y_cols = 0, y_off = 0, y_stride = 0;
    if (!metal_tensor_is_f32_rows(x, &rows, &cols, &x_off, &x_stride) ||
        !metal_tensor_is_f32_rows(y, &y_rows, &y_cols, &y_off, &y_stride) || y_rows != rows ||
        y_cols != cols) {
        return GEIST_E_UNSUPPORTED;
    }
    if (rows > UINT32_MAX || cols > UINT32_MAX || x_off > UINT32_MAX || y_off > UINT32_MAX ||
        x_stride > UINT32_MAX || y_stride > UINT32_MAX || x->buffer->owner != be->state ||
        y->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    struct metal_state                  *st     = be->state;
    const struct metal_scale_rows_params params = {
            .rows         = (uint32_t) rows,
            .cols         = (uint32_t) cols,
            .x_offset     = (uint32_t) x_off,
            .y_offset     = (uint32_t) y_off,
            .x_row_stride = (uint32_t) x_stride,
            .y_row_stride = (uint32_t) y_stride,
            .scale        = 0.0f,
    };
    if (st->sequence_active) {
        metal_encode_gelu_rows(st, metal_sequence_encoder(st), x, y, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    void *enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
    if (cmd == nullptr || enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_gelu_rows(st, enc, x, y, &params);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    return metal_msg_send_id0(st, cmd, "error") == nullptr ? GEIST_OK : GEIST_E_BACKEND;
}

[[nodiscard]] static enum geist_status metal_gelu_tanh_mul(struct geist_backend      *be,
                                                           const struct geist_tensor *x,
                                                           const struct geist_tensor *z,
                                                           struct geist_tensor       *y) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows = 0, cols = 0, x_off = 0, x_stride = 0;
    size_t z_rows = 0, z_cols = 0, z_off = 0, z_stride = 0;
    size_t y_rows = 0, y_cols = 0, y_off = 0, y_stride = 0;
    if (!metal_tensor_is_f32_rows(x, &rows, &cols, &x_off, &x_stride) ||
        !metal_tensor_is_f32_rows(z, &z_rows, &z_cols, &z_off, &z_stride) ||
        !metal_tensor_is_f32_rows(y, &y_rows, &y_cols, &y_off, &y_stride) || z_rows != rows ||
        y_rows != rows || z_cols != cols || y_cols != cols) {
        return GEIST_E_UNSUPPORTED;
    }
    if (rows > UINT32_MAX || cols > UINT32_MAX || x_off > UINT32_MAX || z_off > UINT32_MAX ||
        y_off > UINT32_MAX || x_stride > UINT32_MAX || z_stride > UINT32_MAX ||
        y_stride > UINT32_MAX || x->buffer->owner != be->state || z->buffer->owner != be->state ||
        y->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    struct metal_state                   *st     = be->state;
    const struct metal_binary_rows_params params = {
            .rows         = (uint32_t) rows,
            .cols         = (uint32_t) cols,
            .a_offset     = (uint32_t) x_off,
            .b_offset     = (uint32_t) z_off,
            .y_offset     = (uint32_t) y_off,
            .a_row_stride = (uint32_t) x_stride,
            .b_row_stride = (uint32_t) z_stride,
            .y_row_stride = (uint32_t) y_stride,
    };
    if (st->sequence_active) {
        metal_encode_gelu_mul_rows(st, metal_sequence_encoder(st), x, z, y, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    void *enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
    if (cmd == nullptr || enc == nullptr) {
        return GEIST_E_BACKEND;
    }
    metal_encode_gelu_mul_rows(st, enc, x, z, y, &params);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    return metal_msg_send_id0(st, cmd, "error") == nullptr ? GEIST_OK : GEIST_E_BACKEND;
}

[[nodiscard]] static enum geist_status
metal_embedding_lookup_scaled(struct geist_backend      *be,
                              const struct geist_tensor *embed_table,
                              geist_token_t              token_id,
                              float                      scale,
                              struct geist_tensor       *out) {

    if (be == nullptr || be->state == nullptr || embed_table == nullptr || out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }

    size_t out_n      = 0;
    size_t out_offset = 0;
    if (!metal_tensor_is_f32_vector(out, &out_n, &out_offset) || embed_table->buffer == nullptr ||
        embed_table->ndim != 2 || embed_table->shape[0] <= 0 || embed_table->shape[1] <= 0) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t vocab   = (size_t) embed_table->shape[0];
    const size_t d_model = (size_t) embed_table->shape[1];
    if (token_id < 0 || (size_t) token_id >= vocab || out_n != d_model) {
        return GEIST_E_INVALID_ARG;
    }

    size_t row_bytes      = 0;
    size_t blocks_per_row = 0;
    if (embed_table->layout == GEIST_LAYOUT_DENSE && embed_table->dtype == GEIST_DTYPE_F32) {
        if (d_model > SIZE_MAX / sizeof(float)) {
            return GEIST_E_INVALID_ARG;
        }
        row_bytes = d_model * sizeof(float);
    } else if (embed_table->layout == GEIST_LAYOUT_DENSE &&
               (embed_table->dtype == GEIST_DTYPE_F16 || embed_table->dtype == GEIST_DTYPE_BF16)) {
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
        geist_backend_set_error(be,
                                GEIST_E_UNSUPPORTED,
                                "metal embedding_lookup_scaled: unsupported table dtype/layout");
        return GEIST_E_UNSUPPORTED;
    }

    if (row_bytes == 0 || vocab > SIZE_MAX / row_bytes ||
        embed_table->offset > embed_table->buffer->bytes ||
        vocab * row_bytes > embed_table->buffer->bytes - embed_table->offset ||
        out->offset > out->buffer->bytes ||
        d_model > (out->buffer->bytes - out->offset) / sizeof(float) || d_model > UINT32_MAX ||
        blocks_per_row > UINT32_MAX || embed_table->offset > UINT32_MAX ||
        out_offset > UINT32_MAX || (size_t) token_id > UINT32_MAX) {
        return GEIST_E_INVALID_ARG;
    }
    if (embed_table->buffer->owner != be->state || out->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }

    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }

    struct metal_state             *st     = be->state;
    const struct metal_embed_params params = {
            .n              = (uint32_t) d_model,
            .dtype          = (uint32_t) embed_table->dtype,
            .blocks_per_row = (uint32_t) blocks_per_row,
            .w_byte_offset  = (uint32_t) embed_table->offset,
            .y_offset       = (uint32_t) out_offset,
            .token_id       = (uint32_t) token_id,
            .scale          = scale,
    };

    if (st->sequence_active) {
        if (st->sequence_compute_encoder == nullptr) {
            geist_backend_set_error(
                    be,
                    GEIST_E_BACKEND,
                    "metal embedding_lookup_scaled: command sequence has no encoder");
            return GEIST_E_BACKEND;
        }
        metal_encode_embed_lookup_scaled(st, metal_sequence_encoder(st), embed_table, out, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }

    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    if (cmd == nullptr) {
        geist_backend_set_error(
                be, GEIST_E_BACKEND, "metal embedding_lookup_scaled: command buffer failed");
        return GEIST_E_BACKEND;
    }
    void *enc = metal_msg_send_id0(st, cmd, "computeCommandEncoder");
    if (enc == nullptr) {
        geist_backend_set_error(
                be, GEIST_E_BACKEND, "metal embedding_lookup_scaled: encoder failed");
        return GEIST_E_BACKEND;
    }

    metal_encode_embed_lookup_scaled(st, enc, embed_table, out, &params);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");

    void *err = metal_msg_send_id0(st, cmd, "error");
    if (err != nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal embedding_lookup_scaled: command failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status
metal_embedding_lookup(struct geist_backend      *be,
                       const struct geist_tensor *embed_table,
                       geist_token_t              token_id,
                       struct geist_tensor       *out) {

    return metal_embedding_lookup_scaled(be, embed_table, token_id, 1.0f, out);
}

[[nodiscard]] static enum geist_status metal_matvec_q4k(struct geist_backend      *be,
                                                        const struct geist_tensor *x,
                                                        const struct geist_tensor *w,
                                                        struct geist_tensor       *y) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t n_in     = 0;
    size_t x_offset = 0;
    size_t y_n      = 0;
    size_t y_offset = 0;
    size_t n_out    = 0;
    size_t w_cols   = 0;
    size_t w_offset = 0;
    if (!metal_tensor_is_f32_vector(x, &n_in, &x_offset) ||
        !metal_tensor_is_f32_vector(y, &y_n, &y_offset) ||
        !metal_tensor_is_q4k_matrix(w, &n_out, &w_cols, &w_offset) || w_cols != n_in ||
        y_n != n_out) {
        geist_backend_set_error(
                be,
                GEIST_E_UNSUPPORTED,
                "metal Q4_K matvec: expected x F32 [n], w Q4_K [out,n], y F32 [out]");
        return GEIST_E_UNSUPPORTED;
    }
    if (n_in > UINT32_MAX || n_out > UINT32_MAX || x_offset > UINT32_MAX || y_offset > UINT32_MAX ||
        w_offset > UINT32_MAX) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "metal Q4_K matvec: tensor too large");
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

    struct metal_state           *st     = be->state;
    const struct metal_q4k_params params = {
            .n_in           = (uint32_t) n_in,
            .n_out          = (uint32_t) n_out,
            .rows           = 1,
            .blocks_per_row = (uint32_t) (n_in / METAL_Q4K_BLOCK_ELEMS),
            .x_offset       = (uint32_t) x_offset,
            .w_byte_offset  = (uint32_t) w_offset,
            .y_offset       = (uint32_t) y_offset,
            .x_row_stride   = (uint32_t) n_in,
            .y_row_stride   = (uint32_t) n_out,
    };

    if (st->sequence_active) {
        if (st->sequence_compute_encoder == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal Q4_K matvec: command sequence has no encoder");
            return GEIST_E_BACKEND;
        }
        metal_encode_q4k_linear(st, metal_sequence_encoder(st), x, w, y, &params, false);
        st->sequence_has_work = true;
        return GEIST_OK;
    }

    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    if (cmd == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal Q4_K matvec: command buffer failed");
        return GEIST_E_BACKEND;
    }
    void *enc = metal_msg_send_id0(st, cmd, "computeCommandEncoder");
    if (enc == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal Q4_K matvec: encoder failed");
        return GEIST_E_BACKEND;
    }

    metal_encode_q4k_linear(st, enc, x, w, y, &params, false);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");

    void *err = metal_msg_send_id0(st, cmd, "error");
    if (err != nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal Q4_K matvec: command failed");
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status metal_matmul_q4k(struct geist_backend      *be,
                                                        const struct geist_tensor *x,
                                                        const struct geist_tensor *w,
                                                        struct geist_tensor       *y) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows         = 0;
    size_t n_in         = 0;
    size_t x_offset     = 0;
    size_t x_row_stride = 0;
    size_t y_rows       = 0;
    size_t y_cols       = 0;
    size_t y_offset     = 0;
    size_t y_row_stride = 0;
    size_t n_out        = 0;
    size_t w_cols       = 0;
    size_t w_offset     = 0;
    if (!metal_tensor_is_f32_matrix(x, &rows, &n_in, &x_offset, &x_row_stride) ||
        !metal_tensor_is_f32_matrix(y, &y_rows, &y_cols, &y_offset, &y_row_stride) ||
        !metal_tensor_is_q4k_matrix(w, &n_out, &w_cols, &w_offset) || y_rows != rows ||
        w_cols != n_in || y_cols != n_out) {
        geist_backend_set_error(
                be,
                GEIST_E_UNSUPPORTED,
                "metal Q4_K matmul: expected x F32 [rows,n], w Q4_K [out,n], y F32 [rows,out]");
        return GEIST_E_UNSUPPORTED;
    }
    if (rows > UINT32_MAX || n_in > UINT32_MAX || n_out > UINT32_MAX || x_offset > UINT32_MAX ||
        y_offset > UINT32_MAX || w_offset > UINT32_MAX || x_row_stride > UINT32_MAX ||
        y_row_stride > UINT32_MAX) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "metal Q4_K matmul: tensor too large");
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

    struct metal_state           *st     = be->state;
    const struct metal_q4k_params params = {
            .n_in           = (uint32_t) n_in,
            .n_out          = (uint32_t) n_out,
            .rows           = (uint32_t) rows,
            .blocks_per_row = (uint32_t) (n_in / METAL_Q4K_BLOCK_ELEMS),
            .x_offset       = (uint32_t) x_offset,
            .w_byte_offset  = (uint32_t) w_offset,
            .y_offset       = (uint32_t) y_offset,
            .x_row_stride   = (uint32_t) x_row_stride,
            .y_row_stride   = (uint32_t) y_row_stride,
    };

    if (st->sequence_active) {
        if (st->sequence_compute_encoder == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal Q4_K matmul: command sequence has no encoder");
            return GEIST_E_BACKEND;
        }
        metal_encode_q4k_linear(st, metal_sequence_encoder(st), x, w, y, &params, true);
        st->sequence_has_work = true;
        return GEIST_OK;
    }

    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    if (cmd == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal Q4_K matmul: command buffer failed");
        return GEIST_E_BACKEND;
    }
    void *enc = metal_msg_send_id0(st, cmd, "computeCommandEncoder");
    if (enc == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal Q4_K matmul: encoder failed");
        return GEIST_E_BACKEND;
    }

    metal_encode_q4k_linear(st, enc, x, w, y, &params, true);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");

    void *err = metal_msg_send_id0(st, cmd, "error");
    if (err != nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal Q4_K matmul: command failed");
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status metal_q6k_linear(struct geist_backend      *be,
                                                        const struct geist_tensor *x,
                                                        const struct geist_tensor *w,
                                                        struct geist_tensor       *y,
                                                        bool                       matrix) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows         = 1;
    size_t n_in         = 0;
    size_t x_offset     = 0;
    size_t x_row_stride = 0;
    size_t y_rows       = 1;
    size_t y_cols       = 0;
    size_t y_offset     = 0;
    size_t y_row_stride = 0;
    size_t n_out        = 0;
    size_t w_cols       = 0;
    size_t w_offset     = 0;
    bool   ok           = false;
    if (matrix) {
        ok = metal_tensor_is_f32_matrix(x, &rows, &n_in, &x_offset, &x_row_stride) &&
             metal_tensor_is_f32_matrix(y, &y_rows, &y_cols, &y_offset, &y_row_stride) &&
             y_rows == rows;
    } else {
        ok           = metal_tensor_is_f32_vector(x, &n_in, &x_offset) &&
                       metal_tensor_is_f32_vector(y, &y_cols, &y_offset);
        x_row_stride = n_in;
        y_row_stride = y_cols;
    }
    if (!ok || !metal_tensor_is_q6k_matrix(w, &n_out, &w_cols, &w_offset) || w_cols != n_in ||
        y_cols != n_out) {
        geist_backend_set_error(
                be,
                GEIST_E_UNSUPPORTED,
                matrix ? "metal Q6_K matmul: expected x F32 [rows,n], w Q6_K [out,n], y F32 "
                         "[rows,out]"
                       : "metal Q6_K matvec: expected x F32 [n], w Q6_K [out,n], y F32 [out]");
        return GEIST_E_UNSUPPORTED;
    }
    if (rows > UINT32_MAX || n_in > UINT32_MAX || n_out > UINT32_MAX || x_offset > UINT32_MAX ||
        y_offset > UINT32_MAX || w_offset > UINT32_MAX || x_row_stride > UINT32_MAX ||
        y_row_stride > UINT32_MAX) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "metal Q6_K linear: tensor too large");
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

    struct metal_state           *st     = be->state;
    const struct metal_q4k_params params = {
            .n_in           = (uint32_t) n_in,
            .n_out          = (uint32_t) n_out,
            .rows           = (uint32_t) rows,
            .blocks_per_row = (uint32_t) (n_in / METAL_Q6K_BLOCK_ELEMS),
            .x_offset       = (uint32_t) x_offset,
            .w_byte_offset  = (uint32_t) w_offset,
            .y_offset       = (uint32_t) y_offset,
            .x_row_stride   = (uint32_t) x_row_stride,
            .y_row_stride   = (uint32_t) y_row_stride,
    };

    if (st->sequence_active) {
        if (st->sequence_compute_encoder == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal Q6_K linear: command sequence has no encoder");
            return GEIST_E_BACKEND;
        }
        metal_encode_q6k_linear(st, metal_sequence_encoder(st), x, w, y, &params, matrix);
        st->sequence_has_work = true;
        return GEIST_OK;
    }

    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    if (cmd == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal Q6_K linear: command buffer failed");
        return GEIST_E_BACKEND;
    }
    void *enc = metal_msg_send_id0(st, cmd, "computeCommandEncoder");
    if (enc == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal Q6_K linear: encoder failed");
        return GEIST_E_BACKEND;
    }

    metal_encode_q6k_linear(st, enc, x, w, y, &params, matrix);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");

    void *err = metal_msg_send_id0(st, cmd, "error");
    if (err != nullptr) {
        const char *msg = metal_nserror_message(st, err);
        geist_backend_set_error(be,
                                GEIST_E_BACKEND,
                                "metal Q6_K linear: command failed%s%s",
                                msg != nullptr ? ": " : "",
                                msg != nullptr ? msg : "");
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status metal_matvec_q6k(struct geist_backend      *be,
                                                        const struct geist_tensor *x,
                                                        const struct geist_tensor *w,
                                                        struct geist_tensor       *y) {
    return metal_q6k_linear(be, x, w, y, false);
}

[[nodiscard]] static enum geist_status metal_matmul_q6k(struct geist_backend      *be,
                                                        const struct geist_tensor *x,
                                                        const struct geist_tensor *w,
                                                        struct geist_tensor       *y) {
    return metal_q6k_linear(be, x, w, y, true);
}

static bool metal_tensor_is_dense_3d_dtype(const struct geist_tensor *t,
                                           enum geist_dtype           dtype,
                                           size_t                     elem_size,
                                           size_t                    *out_d0,
                                           size_t                    *out_d1,
                                           size_t                    *out_d2,
                                           size_t                    *out_offset_elems);

static bool metal_tensor_is_f32_3d(const struct geist_tensor *t,
                                   size_t                    *out_d0,
                                   size_t                    *out_d1,
                                   size_t                    *out_d2,
                                   size_t                    *out_offset_floats) {
    return metal_tensor_is_dense_3d_dtype(
            t, GEIST_DTYPE_F32, sizeof(float), out_d0, out_d1, out_d2, out_offset_floats);
}

static bool metal_tensor_is_f16_3d(const struct geist_tensor *t,
                                   size_t                    *out_d0,
                                   size_t                    *out_d1,
                                   size_t                    *out_d2,
                                   size_t                    *out_offset_halfs) {
    return metal_tensor_is_dense_3d_dtype(
            t, GEIST_DTYPE_F16, sizeof(uint16_t), out_d0, out_d1, out_d2, out_offset_halfs);
}

static bool metal_tensor_is_dense_3d_dtype(const struct geist_tensor *t,
                                           enum geist_dtype           dtype,
                                           size_t                     elem_size,
                                           size_t                    *out_d0,
                                           size_t                    *out_d1,
                                           size_t                    *out_d2,
                                           size_t                    *out_offset_elems) {
    if (t == nullptr || t->buffer == nullptr || t->dtype != dtype ||
        t->layout != GEIST_LAYOUT_DENSE || t->ndim != 3 || t->shape[0] <= 0 || t->shape[1] <= 0 ||
        t->shape[2] <= 0 || t->stride[2] != 1 || t->stride[1] != t->shape[2] ||
        t->stride[0] != t->shape[1] * t->shape[2] || elem_size == 0 || t->offset % elem_size != 0) {
        return false;
    }
    const size_t d0 = (size_t) t->shape[0];
    const size_t d1 = (size_t) t->shape[1];
    const size_t d2 = (size_t) t->shape[2];
    if (d0 > SIZE_MAX / d1 || d0 * d1 > SIZE_MAX / d2) {
        return false;
    }
    const size_t elems = d0 * d1 * d2;
    if (t->offset > t->buffer->bytes || elems > (t->buffer->bytes - t->offset) / elem_size) {
        return false;
    }
    *out_d0           = d0;
    *out_d1           = d1;
    *out_d2           = d2;
    *out_offset_elems = t->offset / elem_size;
    return true;
}

[[nodiscard]] static enum geist_status metal_rope_apply(struct geist_backend      *be,
                                                        struct geist_tensor       *x,
                                                        const struct geist_tensor *cos,
                                                        const struct geist_tensor *sin) {

    if (be == nullptr || be->state == nullptr || x == nullptr || cos == nullptr || sin == nullptr) {
        return GEIST_E_INVALID_ARG;
    }

    {
        static int dbg = -1;
        if (dbg < 0) {
            const char *e = getenv("GEIST_METAL_DEBUG_LINEAR");
            dbg           = (e != nullptr && e[0] != '\0' && strcmp(e, "0") != 0) ? 1 : 0;
        }
        if (dbg && x->buffer != nullptr && x->buffer->mapped != nullptr && cos->buffer != nullptr &&
            cos->buffer->mapped != nullptr) {
            const float *xp = (const float *) ((const uint8_t *) x->buffer->mapped + x->offset);
            const float *cp = (const float *) ((const uint8_t *) cos->buffer->mapped + cos->offset);
            size_t       nx = 1, nc = 1;
            for (int i = 0; i < x->ndim; i++)
                nx *= (size_t) x->shape[i];
            for (int i = 0; i < cos->ndim; i++)
                nc *= (size_t) cos->shape[i];
            size_t nanx = 0, nanc = 0;
            for (size_t i = 0; i < nx; i++)
                if (isnan(xp[i]))
                    nanx++;
            for (size_t i = 0; i < nc; i++)
                if (isnan(cp[i]))
                    nanc++;
            fprintf(stderr,
                    "rope_in x[%lld,%lld,%lld] nanx=%zu cos[%lld,%lld] nanc=%zu xoff=%zu "
                    "coff=%zu\n",
                    (long long) x->shape[0],
                    (long long) x->shape[1],
                    (long long) x->shape[2],
                    nanx,
                    (long long) cos->shape[0],
                    (long long) cos->shape[1],
                    nanc,
                    x->offset,
                    cos->offset);
        }
    }
    size_t rows = 0, heads = 0, head_dim = 0, x_offset = 0;
    size_t cos_rows = 0, cos_cols = 0, cos_offset = 0, cos_stride = 0;
    size_t sin_rows = 0, sin_cols = 0, sin_offset = 0, sin_stride = 0;
    if (!metal_tensor_is_f32_3d(x, &rows, &heads, &head_dim, &x_offset) ||
        !metal_tensor_is_f32_matrix(cos, &cos_rows, &cos_cols, &cos_offset, &cos_stride) ||
        !metal_tensor_is_f32_matrix(sin, &sin_rows, &sin_cols, &sin_offset, &sin_stride)) {
        geist_backend_set_error(be,
                                GEIST_E_UNSUPPORTED,
                                "metal rope_apply: tensors must be F32 DENSE x[seq,heads,dim]");
        return GEIST_E_UNSUPPORTED;
    }
    if (head_dim == 0 || (head_dim % 2u) != 0 || cos_rows != rows || sin_rows != rows ||
        cos_cols != head_dim || sin_cols != head_dim || cos_stride != head_dim ||
        sin_stride != head_dim) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "metal rope_apply: shape mismatch");
        return GEIST_E_INVALID_ARG;
    }
    if (rows > UINT32_MAX || heads > UINT32_MAX || head_dim > UINT32_MAX || x_offset > UINT32_MAX ||
        cos_offset > UINT32_MAX || sin_offset > UINT32_MAX || rows > UINT32_MAX / heads ||
        rows * heads > UINT32_MAX / (head_dim / 2u) || x->buffer->owner != be->state ||
        cos->buffer->owner != be->state || sin->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }

    enum geist_status s = metal_ensure_attention_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    struct metal_state            *st     = be->state;
    const struct metal_rope_params params = {
            .rows            = (uint32_t) rows,
            .heads           = (uint32_t) heads,
            .head_dim        = (uint32_t) head_dim,
            .x_offset        = (uint32_t) x_offset,
            .cos_offset      = (uint32_t) cos_offset,
            .sin_offset      = (uint32_t) sin_offset,
            .x_row_stride    = (uint32_t) (heads * head_dim),
            .rope_row_stride = (uint32_t) cos_stride,
            .rope_row_offset = 0,
    };
    if (st->sequence_active) {
        if (st->sequence_compute_encoder == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal rope_apply: sequence has no encoder");
            return GEIST_E_BACKEND;
        }
        metal_encode_rope_rows(st, metal_sequence_encoder(st), x, cos, sin, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    void *enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
    if (cmd == nullptr || enc == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal rope_apply: command encoder failed");
        return GEIST_E_BACKEND;
    }
    metal_encode_rope_rows(st, enc, x, cos, sin, &params);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    void *err = metal_msg_send_id0(st, cmd, "error");
    if (err != nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal rope_apply: command failed");
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

/* rows>1 f32-KV fast path: convert K/V to persistent f16 staging (one
 * kv_append_rows_f16 dispatch) and run the no-norm simdgroup flash kernel.
 * The scalar f32 kernel this replaces is the dominant prefill cost; the
 * conversion is ~1%% of the savings. Serial-encoder ordering makes the
 * staging reuse across layers safe. */
[[nodiscard]] static enum geist_status metal_attention_flash_kv(struct geist_backend      *be,
                                                                struct metal_state        *st,
                                                                const struct geist_tensor *q,
                                                                const struct geist_tensor *k,
                                                                const struct geist_tensor *value,
                                                                struct geist_tensor       *out,
                                                                size_t                     q_rows,
                                                                size_t                     k_rows,
                                                                size_t                     q_heads,
                                                                size_t                     k_heads,
                                                                size_t                     head_dim,
                                                                size_t                     q_offset,
                                                                size_t sliding_window,
                                                                size_t q_off,
                                                                size_t k_off,
                                                                size_t v_off,
                                                                size_t out_off,
                                                                bool   kv_native_f16) {

    const size_t kv_out    = k_heads * head_dim;
    const size_t elems     = k_rows * kv_out;
    const size_t f16_bytes = elems * 2u;
    if (!kv_native_f16 && st->attn_kvf16_capacity < f16_bytes) {
        metal_buffer_destroy_internal(be, st->attn_kf16_buffer);
        metal_buffer_destroy_internal(be, st->attn_vf16_buffer);
        st->attn_kf16_buffer    = nullptr;
        st->attn_vf16_buffer    = nullptr;
        st->attn_kvf16_capacity = 0;
        const size_t      cap   = f16_bytes * 2u; /* headroom: no regrow per chunk */
        enum geist_status bs =
                metal_new_buffer(be, cap, GEIST_BUFFER_SCRATCH, 0, true, &st->attn_kf16_buffer);
        if (bs == GEIST_OK) {
            bs = metal_new_buffer(be, cap, GEIST_BUFFER_SCRATCH, 0, true, &st->attn_vf16_buffer);
        }
        if (bs != GEIST_OK) {
            metal_buffer_destroy_internal(be, st->attn_kf16_buffer);
            st->attn_kf16_buffer = nullptr;
            return bs;
        }
        st->attn_kvf16_capacity = cap;
    }
    struct {
        uint32_t elems, kv_out, k_offset, v_offset, k_cache_offset, v_cache_offset, q_position;
    } ap = {(uint32_t) elems, (uint32_t) kv_out, (uint32_t) k_off, (uint32_t) v_off, 0u, 0u, 0u};
    struct {
        uint32_t rows, kv_len, qh, kvh, hd, qpos, sw, qo, kco, vco, yo;
    } fp       = {(uint32_t) q_rows,
                  (uint32_t) k_rows,
                  (uint32_t) q_heads,
                  (uint32_t) k_heads,
                  (uint32_t) head_dim,
                  (uint32_t) q_offset,
                  (uint32_t) sliding_window,
                  (uint32_t) q_off,
                  kv_native_f16 ? (uint32_t) k_off : 0u,
                  kv_native_f16 ? (uint32_t) v_off : 0u,
                  (uint32_t) out_off};
    void *kf16 = kv_native_f16 ? k->buffer->buffer : st->attn_kf16_buffer->buffer;
    void *vf16 = kv_native_f16 ? value->buffer->buffer : st->attn_vf16_buffer->buffer;

    void *cmd = nullptr;
    void *enc = nullptr;
    if (st->sequence_active) {
        if (st->sequence_compute_encoder == nullptr) {
            return GEIST_E_BACKEND;
        }
        enc = metal_sequence_encoder(st);
    } else {
        cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
        enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
        if (cmd == nullptr || enc == nullptr) {
            return GEIST_E_BACKEND;
        }
    }
    if (!kv_native_f16) {
        metal_msg_send_set_pipeline(st, enc, st->kv_append_rows_f16_pipeline);
        metal_msg_send_set_buffer(st, enc, k->buffer->buffer, 0, 0);
        metal_msg_send_set_buffer(st, enc, value->buffer->buffer, 0, 1);
        metal_msg_send_set_buffer(st, enc, st->attn_kf16_buffer->buffer, 0, 2);
        metal_msg_send_set_buffer(st, enc, st->attn_vf16_buffer->buffer, 0, 3);
        metal_msg_send_set_bytes(st, enc, &ap, sizeof(ap), 4);
        const struct metal_size cgroups  = {(elems + 255u) / 256u, 1, 1};
        const struct metal_size cthreads = {256, 1, 1};
        metal_msg_send_dispatch(st, enc, cgroups, cthreads);
    }

    /* head_dim <= 256: 4-simdgroup kernel (128 threads). 256 < hd <= 512
     * (gemma-3n full-attention layers): 8-simdgroup variant (256 threads),
     * one query row + one 64-column output slice per simdgroup. */
    const bool sg8 = head_dim > 256u;
    metal_msg_send_set_pipeline(st,
                                enc,
                                sg8 ? st->attention_flash_sg8_f16_pipeline
                                    : st->attention_flash_sg_f16_pipeline);
    metal_msg_send_set_buffer(st, enc, q->buffer->buffer, 0, 0);
    metal_msg_send_set_buffer(st, enc, kf16, 0, 1);
    metal_msg_send_set_buffer(st, enc, vf16, 0, 2);
    metal_msg_send_set_buffer(st, enc, out->buffer->buffer, 0, 3);
    metal_msg_send_set_bytes(st, enc, &fp, sizeof(fp), 4);
    const struct metal_size fgroups  = {q_rows / 8u, q_heads, 1};
    const struct metal_size fthreads = {sg8 ? 256u : 128u, 1, 1};
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_ATTENTION_ROWS, fgroups);
    metal_msg_send_dispatch(st, enc, fgroups, fthreads);

    if (st->sequence_active) {
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    void *err = metal_msg_send_id0(st, cmd, "error");
    if (err != nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal flash attention: command failed");
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

/* rows==1 (decode) f32-KV fast path: convert K/V to the f16 staging and run
 * the split-KV decode kernel — the scalar kernel uses one threadgroup per
 * head (8 tgs on a 32-core GPU) and its cost grows linearly with kv_len. */
[[nodiscard]] static enum geist_status metal_attention_dec_kv(struct geist_backend      *be,
                                                              struct metal_state        *st,
                                                              const struct geist_tensor *q,
                                                              const struct geist_tensor *k,
                                                              const struct geist_tensor *value,
                                                              struct geist_tensor       *out,
                                                              size_t                     k_rows,
                                                              size_t                     q_heads,
                                                              size_t                     k_heads,
                                                              size_t                     head_dim,
                                                              size_t                     q_offset,
                                                              size_t sliding_window,
                                                              size_t q_off,
                                                              size_t k_off,
                                                              size_t v_off,
                                                              size_t out_off,
                                                              bool   kv_native_f16) {

    const size_t kv_out    = k_heads * head_dim;
    const size_t elems     = k_rows * kv_out;
    const size_t f16_bytes = elems * 2u;
    if (!kv_native_f16 && st->attn_kvf16_capacity < f16_bytes) {
        metal_buffer_destroy_internal(be, st->attn_kf16_buffer);
        metal_buffer_destroy_internal(be, st->attn_vf16_buffer);
        st->attn_kf16_buffer    = nullptr;
        st->attn_vf16_buffer    = nullptr;
        st->attn_kvf16_capacity = 0;
        const size_t      cap   = f16_bytes * 2u;
        enum geist_status bs =
                metal_new_buffer(be, cap, GEIST_BUFFER_SCRATCH, 0, true, &st->attn_kf16_buffer);
        if (bs == GEIST_OK) {
            bs = metal_new_buffer(be, cap, GEIST_BUFFER_SCRATCH, 0, true, &st->attn_vf16_buffer);
        }
        if (bs != GEIST_OK) {
            metal_buffer_destroy_internal(be, st->attn_kf16_buffer);
            st->attn_kf16_buffer = nullptr;
            return bs;
        }
        st->attn_kvf16_capacity = cap;
    }
    uint32_t window = (uint32_t) q_offset + 1u;
    if ((uint32_t) k_rows < window) {
        window = (uint32_t) k_rows;
    }
    if (sliding_window > 0u && (uint32_t) sliding_window < window) {
        window = (uint32_t) sliding_window;
    }
    uint32_t nsplit = window / 32u;
    if (nsplit < 1u) {
        nsplit = 1u;
    }
    if (nsplit > 16u) {
        nsplit = 16u;
    }
    const size_t partial_bytes = q_heads * 16u * (head_dim + 2u) * sizeof(float);
    if (st->attn_dec_partials_buffer != nullptr && st->attn_dec_partials_capacity < partial_bytes) {
        metal_buffer_destroy_internal(be, st->attn_dec_partials_buffer);
        st->attn_dec_partials_buffer   = nullptr;
        st->attn_dec_partials_capacity = 0;
    }
    if (st->attn_dec_partials_buffer == nullptr) {
        if (metal_new_buffer(be,
                             partial_bytes,
                             GEIST_BUFFER_SCRATCH,
                             GEIST_MEMORY_DEVICE,
                             false,
                             &st->attn_dec_partials_buffer) != GEIST_OK) {
            return GEIST_E_BACKEND;
        }
        st->attn_dec_partials_capacity = partial_bytes;
    }
    struct {
        uint32_t elems, kv_out, k_offset, v_offset, k_cache_offset, v_cache_offset, q_position;
    } ap = {(uint32_t) elems, (uint32_t) kv_out, (uint32_t) k_off, (uint32_t) v_off, 0u, 0u, 0u};
    struct {
        uint32_t rows, kv_len, qh, kvh, hd, qpos, sw, qo, kco, vco, yo;
    } fp       = {1u,
                  (uint32_t) k_rows,
                  (uint32_t) q_heads,
                  (uint32_t) k_heads,
                  (uint32_t) head_dim,
                  (uint32_t) q_offset,
                  (uint32_t) sliding_window,
                  (uint32_t) q_off,
                  kv_native_f16 ? (uint32_t) k_off : 0u,
                  kv_native_f16 ? (uint32_t) v_off : 0u,
                  (uint32_t) out_off};
    void *kf16 = kv_native_f16 ? k->buffer->buffer : st->attn_kf16_buffer->buffer;
    void *vf16 = kv_native_f16 ? value->buffer->buffer : st->attn_vf16_buffer->buffer;

    void *cmd = nullptr;
    void *enc = nullptr;
    if (st->sequence_active) {
        if (st->sequence_compute_encoder == nullptr) {
            return GEIST_E_BACKEND;
        }
        enc = metal_sequence_encoder(st);
    } else {
        cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
        enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
        if (cmd == nullptr || enc == nullptr) {
            return GEIST_E_BACKEND;
        }
    }
    const struct metal_size threads256 = {256, 1, 1};
    if (!kv_native_f16) {
        metal_msg_send_set_pipeline(st, enc, st->kv_append_rows_f16_pipeline);
        metal_msg_send_set_buffer(st, enc, k->buffer->buffer, 0, 0);
        metal_msg_send_set_buffer(st, enc, value->buffer->buffer, 0, 1);
        metal_msg_send_set_buffer(st, enc, st->attn_kf16_buffer->buffer, 0, 2);
        metal_msg_send_set_buffer(st, enc, st->attn_vf16_buffer->buffer, 0, 3);
        metal_msg_send_set_bytes(st, enc, &ap, sizeof(ap), 4);
        const struct metal_size cgroups = {(elems + 255u) / 256u, 1, 1};
        metal_msg_send_dispatch(st, enc, cgroups, threads256);
    }

    /* head_dim <= 256: 8-chunk lane unroll; 256 < hd <= 512 (gemma-3n
     * full-attention layers): the 16-chunk dec512 variant. */
    metal_msg_send_set_pipeline(st,
                                enc,
                                head_dim > 256u ? st->attention_dec512_f16_pipeline
                                                : st->attention_dec_f16_pipeline);
    metal_msg_send_set_buffer(st, enc, q->buffer->buffer, 0, 0);
    metal_msg_send_set_buffer(st, enc, kf16, 0, 1);
    metal_msg_send_set_buffer(st, enc, vf16, 0, 2);
    metal_msg_send_set_buffer(st, enc, st->attn_dec_partials_buffer->buffer, 0, 3);
    metal_msg_send_set_bytes(st, enc, &fp, sizeof(fp), 4);
    metal_msg_send_set_bytes(st, enc, &nsplit, sizeof(nsplit), 5);
    const struct metal_size dgroups = {nsplit, q_heads, 1};
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_ATTENTION_ROWS, dgroups);
    metal_msg_send_dispatch(st, enc, dgroups, threads256);

    const uint32_t cb[4] = {(uint32_t) q_heads, (uint32_t) head_dim, nsplit, (uint32_t) out_off};
    metal_msg_send_set_pipeline(st, enc, st->attention_dec_combine_pipeline);
    metal_msg_send_set_buffer(st, enc, st->attn_dec_partials_buffer->buffer, 0, 0);
    metal_msg_send_set_buffer(st, enc, out->buffer->buffer, 0, 1);
    metal_msg_send_set_bytes(st, enc, cb, sizeof(cb), 2);
    const struct metal_size ggroups = {1, q_heads, 1};
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_ATTENTION_ROWS, ggroups);
    metal_msg_send_dispatch(st, enc, ggroups, threads256);

    if (st->sequence_active) {
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    void *err = metal_msg_send_id0(st, cmd, "error");
    if (err != nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal decode attention: command failed");
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

/* Fused two-weight matvec (vtbl slot linear_t_pair): same structure as
 * ffn_gate_up but with a raw two-output epilogue — one activation pass
 * feeds both weights (used for the k/v projections, which share shape).
 * rows==1 + Q4_K + equal shapes only. */
[[nodiscard]] static enum geist_status metal_linear_t_pair(struct geist_backend      *be,
                                                           const struct geist_tensor *x,
                                                           const struct geist_weight *w0,
                                                           const struct geist_tensor *t_w0,
                                                           const struct geist_weight *w1,
                                                           const struct geist_tensor *t_w1,
                                                           size_t                     m,
                                                           struct geist_tensor       *y0,
                                                           struct geist_tensor       *y1) {

    if (be == nullptr || be->state == nullptr || x == nullptr || w0 == nullptr || t_w0 == nullptr ||
        w1 == nullptr || t_w1 == nullptr || y0 == nullptr || y1 == nullptr) {
        return GEIST_E_UNSUPPORTED;
    }
    if (m != 1u || w0->dtype != GEIST_DTYPE_Q4_K || w1->dtype != GEIST_DTYPE_Q4_K) {
        return GEIST_E_UNSUPPORTED;
    }
    size_t rows = 0, d_in = 0, x_off = 0, x_stride = 0;
    size_t a_out = 0, a_in = 0, a_woff = 0;
    size_t b_out = 0, b_in = 0, b_woff = 0;
    size_t y0_rows = 0, y0_cols = 0, y0_off = 0, y0_stride = 0;
    size_t y1_rows = 0, y1_cols = 0, y1_off = 0, y1_stride = 0;
    if (!metal_tensor_is_f32_rows(x, &rows, &d_in, &x_off, &x_stride) ||
        !metal_tensor_is_q4k_matrix(t_w0, &a_out, &a_in, &a_woff) ||
        !metal_tensor_is_q4k_matrix(t_w1, &b_out, &b_in, &b_woff) ||
        !metal_tensor_is_f32_rows(y0, &y0_rows, &y0_cols, &y0_off, &y0_stride) ||
        !metal_tensor_is_f32_rows(y1, &y1_rows, &y1_cols, &y1_off, &y1_stride) || rows != 1u ||
        a_in != d_in || b_in != d_in || b_out != a_out || (d_in % 256u) != 0u || y0_rows != rows ||
        y1_rows != rows || y0_cols != a_out || y1_cols != a_out || y0_stride != y1_stride) {
        return GEIST_E_UNSUPPORTED;
    }
    if (d_in > UINT32_MAX || a_out > UINT32_MAX || x_off > UINT32_MAX || a_woff > UINT32_MAX ||
        b_woff > UINT32_MAX || y0_off > UINT32_MAX || y1_off > UINT32_MAX ||
        x->buffer->owner != be->state || t_w0->buffer->owner != be->state ||
        t_w1->buffer->owner != be->state || y0->buffer->owner != be->state ||
        y1->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    struct metal_state *st = be->state;
    if (st->q4k_pair_n4_pipeline == nullptr) {
        return GEIST_E_UNSUPPORTED;
    }
    const struct metal_q4k_gate_up_params params = {
            .n_in               = (uint32_t) d_in,
            .n_out              = (uint32_t) a_out,
            .rows               = (uint32_t) rows,
            .blocks_per_row     = (uint32_t) (d_in / 256u),
            .x_offset           = (uint32_t) x_off,
            .gate_w_byte_offset = (uint32_t) a_woff,
            .up_w_byte_offset   = (uint32_t) b_woff,
            .gate_y_offset      = (uint32_t) y0_off,
            .up_y_offset        = (uint32_t) y1_off,
            .x_row_stride       = (uint32_t) x_stride,
            .y_row_stride       = (uint32_t) y0_stride,
    };
    void *cmd = nullptr;
    void *enc = nullptr;
    if (st->sequence_active) {
        if (st->sequence_compute_encoder == nullptr) {
            return GEIST_E_BACKEND;
        }
        enc = metal_sequence_encoder(st);
    } else {
        cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
        enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
        if (cmd == nullptr || enc == nullptr) {
            return GEIST_E_BACKEND;
        }
    }
    metal_msg_send_set_pipeline(st, enc, st->q4k_pair_n4_pipeline);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, 0, 0);
    metal_msg_send_set_buffer(st, enc, t_w0->buffer->buffer, 0, 1);
    metal_msg_send_set_buffer(st, enc, t_w1->buffer->buffer, 0, 2);
    metal_msg_send_set_buffer(st, enc, y0->buffer->buffer, 0, 3);
    metal_msg_send_set_buffer(st, enc, y1->buffer->buffer, 0, 4);
    metal_msg_send_set_bytes(st, enc, &params, sizeof(params), 5);
    const struct metal_size groups  = {(uint32_t) ((a_out + 3u) / 4u), (uint32_t) rows, 1};
    const struct metal_size threads = {METAL_Q4K_N4_THREADS, 1, 1};
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_Q4K_QK_BASE, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
    if (st->sequence_active) {
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    return metal_msg_send_id0(st, cmd, "error") == nullptr ? GEIST_OK : GEIST_E_BACKEND;
}

/* Fused FFN gate+up matvec with GeGLU epilogue (vtbl slot): the restored
 * wip gate_up_q4k_n4 kernel — llama mul_mv structure, 2 rows/simdgroup,
 * both weights against one activation pass, gelu(g)*u written directly.
 * rows==1 (decode) and Q4_K only; prefill keeps the mm_sg GEMMs. */
[[nodiscard]] static enum geist_status metal_ffn_gate_up(struct geist_backend      *be,
                                                         const struct geist_tensor *x,
                                                         const struct geist_tensor *gate_w,
                                                         const struct geist_tensor *up_w,
                                                         struct geist_tensor       *y) {

    if (be == nullptr || be->state == nullptr || x == nullptr || gate_w == nullptr ||
        up_w == nullptr || y == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows = 0, d_in = 0, x_off = 0, x_stride = 0;
    size_t y_rows = 0, y_cols = 0, y_off = 0, y_stride = 0;
    size_t g_out = 0, g_in = 0, g_woff = 0;
    size_t u_out = 0, u_in = 0, u_woff = 0;
    if (!metal_tensor_is_f32_rows(x, &rows, &d_in, &x_off, &x_stride) ||
        !metal_tensor_is_q4k_matrix(gate_w, &g_out, &g_in, &g_woff) ||
        !metal_tensor_is_q4k_matrix(up_w, &u_out, &u_in, &u_woff) ||
        !metal_tensor_is_f32_rows(y, &y_rows, &y_cols, &y_off, &y_stride) ||
        rows != 1u || /* decode only: matvec-structured kernel */
        g_in != d_in || u_in != d_in || u_out != g_out || (d_in % 256u) != 0u || y_rows != rows ||
        y_cols != g_out) {
        return GEIST_E_UNSUPPORTED;
    }
    if (d_in > UINT32_MAX || g_out > UINT32_MAX || x_off > UINT32_MAX || g_woff > UINT32_MAX ||
        u_woff > UINT32_MAX || y_off > UINT32_MAX || x->buffer->owner != be->state ||
        gate_w->buffer->owner != be->state || up_w->buffer->owner != be->state ||
        y->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    struct metal_state *st = be->state;
    if (st->q4k_gate_up_n4_pipeline == nullptr) {
        return GEIST_E_UNSUPPORTED;
    }
    const struct metal_q4k_gate_up_params params = {
            .n_in               = (uint32_t) d_in,
            .n_out              = (uint32_t) g_out,
            .rows               = (uint32_t) rows,
            .blocks_per_row     = (uint32_t) (d_in / 256u),
            .x_offset           = (uint32_t) x_off,
            .gate_w_byte_offset = (uint32_t) g_woff,
            .up_w_byte_offset   = (uint32_t) u_woff,
            .gate_y_offset      = (uint32_t) y_off,
            .up_y_offset        = (uint32_t) y_off, /* unused by the epilogue kernel */
            .x_row_stride       = (uint32_t) x_stride,
            .y_row_stride       = (uint32_t) y_stride,
    };
    void *cmd = nullptr;
    void *enc = nullptr;
    if (st->sequence_active) {
        if (st->sequence_compute_encoder == nullptr) {
            return GEIST_E_BACKEND;
        }
        enc = metal_sequence_encoder(st);
    } else {
        cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
        enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
        if (cmd == nullptr || enc == nullptr) {
            return GEIST_E_BACKEND;
        }
    }
    metal_msg_send_set_pipeline(st, enc, st->q4k_gate_up_n4_pipeline);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, 0, 0);
    metal_msg_send_set_buffer(st, enc, gate_w->buffer->buffer, 0, 1);
    metal_msg_send_set_buffer(st, enc, up_w->buffer->buffer, 0, 2);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, 0, 3);
    metal_msg_send_set_buffer(st, enc, y->buffer->buffer, 0, 4);
    metal_msg_send_set_bytes(st, enc, &params, sizeof(params), 5);
    const struct metal_size groups  = {(uint32_t) ((g_out + 3u) / 4u), (uint32_t) rows, 1};
    const struct metal_size threads = {METAL_Q4K_N4_THREADS, 1, 1};
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_Q4K_GATE_UP_N4, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
    if (st->sequence_active) {
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    return metal_msg_send_id0(st, cmd, "error") == nullptr ? GEIST_OK : GEIST_E_BACKEND;
}

/* Fused gemma attention q/k/v prep (vtbl slot): q_norm_rope_rows for q,
 * kv_norm_append_rows{,_f16} for k+v (norm + RoPE + cache append in one
 * dispatch). Replaces up to six decomposed ops with two dispatches. */
[[nodiscard]] static enum geist_status metal_attn_qkv_prep(struct geist_backend      *be,
                                                           struct geist_tensor       *q,
                                                           struct geist_tensor       *k,
                                                           struct geist_tensor       *v,
                                                           const struct geist_tensor *q_norm_w,
                                                           const struct geist_tensor *k_norm_w,
                                                           const struct geist_tensor *v_norm_w,
                                                           const struct geist_tensor *cos,
                                                           const struct geist_tensor *sin,
                                                           float                      eps,
                                                           size_t                     q_position,
                                                           struct geist_tensor       *k_cache,
                                                           struct geist_tensor       *v_cache) {

    if (be == nullptr || be->state == nullptr || q == nullptr || q_norm_w == nullptr ||
        cos == nullptr || sin == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    const bool has_kv = k != nullptr;
    if (has_kv && (v == nullptr || k_norm_w == nullptr || v_norm_w == nullptr ||
                   k_cache == nullptr || v_cache == nullptr)) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows = 0, q_heads = 0, hd = 0, q_off = 0;
    size_t c_rows = 0, c_cols = 0, c_off = 0, c_stride = 0;
    size_t s_rows = 0, s_cols = 0, s_off = 0, s_stride = 0;
    size_t qw_n = 0, qw_off = 0;
    if (!metal_tensor_is_f32_3d(q, &rows, &q_heads, &hd, &q_off) ||
        !metal_tensor_is_f32_matrix(cos, &c_rows, &c_cols, &c_off, &c_stride) ||
        !metal_tensor_is_f32_matrix(sin, &s_rows, &s_cols, &s_off, &s_stride) ||
        !metal_tensor_is_f32_vector(q_norm_w, &qw_n, &qw_off) || hd == 0 || (hd % 2u) != 0 ||
        qw_n != hd || c_rows != rows || s_rows != rows || c_cols != hd || s_cols != hd ||
        c_stride != hd || s_stride != hd) {
        return GEIST_E_UNSUPPORTED;
    }
    size_t k_rows = 0, kv_heads = 0, k_hd = 0, k_off = 0;
    size_t v_rows = 0, v_heads = 0, v_hd = 0, v_off = 0;
    size_t kc_rows = 0, kc_heads = 0, kc_hd = 0, kc_off = 0;
    size_t vc_rows = 0, vc_heads = 0, vc_hd = 0, vc_off = 0;
    size_t kw_n = 0, kw_off = 0, vw_n = 0, vw_off = 0;
    bool   cache_f16 = false;
    if (has_kv) {
        if (!metal_tensor_is_f32_3d(k, &k_rows, &kv_heads, &k_hd, &k_off) ||
            !metal_tensor_is_f32_3d(v, &v_rows, &v_heads, &v_hd, &v_off) ||
            !metal_tensor_is_f32_vector(k_norm_w, &kw_n, &kw_off) ||
            !metal_tensor_is_f32_vector(v_norm_w, &vw_n, &vw_off) || k_rows != rows ||
            v_rows != rows || v_heads != kv_heads || k_hd != hd || v_hd != hd || kw_n != hd ||
            vw_n != hd) {
            return GEIST_E_UNSUPPORTED;
        }
        const bool cache_f32 =
                metal_tensor_is_f32_3d(k_cache, &kc_rows, &kc_heads, &kc_hd, &kc_off) &&
                metal_tensor_is_f32_3d(v_cache, &vc_rows, &vc_heads, &vc_hd, &vc_off);
        cache_f16 = !cache_f32 &&
                    metal_tensor_is_f16_3d(k_cache, &kc_rows, &kc_heads, &kc_hd, &kc_off) &&
                    metal_tensor_is_f16_3d(v_cache, &vc_rows, &vc_heads, &vc_hd, &vc_off);
        if ((!cache_f32 && !cache_f16) || kc_heads != kv_heads || vc_heads != kv_heads ||
            kc_hd != hd || vc_hd != hd || kc_rows < q_position + rows ||
            vc_rows < q_position + rows) {
            return GEIST_E_UNSUPPORTED;
        }
    }
    if (rows > UINT32_MAX || q_heads > UINT32_MAX || hd > UINT32_MAX || q_off > UINT32_MAX ||
        c_off > UINT32_MAX || s_off > UINT32_MAX || qw_off > UINT32_MAX ||
        q_position > UINT32_MAX || q->buffer->owner != be->state ||
        cos->buffer->owner != be->state || sin->buffer->owner != be->state ||
        q_norm_w->buffer->owner != be->state ||
        (has_kv && (k_off > UINT32_MAX || v_off > UINT32_MAX || kc_off > UINT32_MAX ||
                    vc_off > UINT32_MAX || kw_off > UINT32_MAX || vw_off > UINT32_MAX ||
                    k->buffer->owner != be->state || v->buffer->owner != be->state ||
                    k_norm_w->buffer->owner != be->state || v_norm_w->buffer->owner != be->state ||
                    k_cache->buffer->owner != be->state || v_cache->buffer->owner != be->state))) {
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status s = metal_ensure_attention_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    struct metal_state *st = be->state;
    void               *kv_pipeline =
            cache_f16 ? st->kv_norm_append_rows_f16_pipeline : st->kv_norm_append_rows_pipeline;
    if (st->q_norm_rope_rows_pipeline == nullptr || (has_kv && kv_pipeline == nullptr)) {
        return GEIST_E_UNSUPPORTED;
    }
    struct {
        uint32_t rows, heads, hd, xo, wo, co, so, xs, rs, ro;
        float    eps;
    } nr = {(uint32_t) rows,
            (uint32_t) q_heads,
            (uint32_t) hd,
            (uint32_t) q_off,
            (uint32_t) qw_off,
            (uint32_t) c_off,
            (uint32_t) s_off,
            (uint32_t) (q_heads * hd),
            (uint32_t) hd,
            0u,
            eps};
    struct {
        uint32_t rows, heads, hd, xo, wo, co, so, cao, xs, rs, ro, qp;
        float    eps;
    } kp = {(uint32_t) rows,
            (uint32_t) kv_heads,
            (uint32_t) hd,
            (uint32_t) k_off,
            (uint32_t) kw_off,
            (uint32_t) c_off,
            (uint32_t) s_off,
            (uint32_t) kc_off,
            (uint32_t) (kv_heads * hd),
            (uint32_t) hd,
            0u,
            (uint32_t) q_position,
            eps};
    struct {
        uint32_t rows, heads, hd, xo, wo, cao, xs, qp;
        float    eps;
    } vp = {(uint32_t) rows,
            (uint32_t) kv_heads,
            (uint32_t) hd,
            (uint32_t) v_off,
            (uint32_t) vw_off,
            (uint32_t) vc_off,
            (uint32_t) (kv_heads * hd),
            (uint32_t) q_position,
            eps};

    void *cmd = nullptr;
    void *enc = nullptr;
    if (st->sequence_active) {
        if (st->sequence_compute_encoder == nullptr) {
            return GEIST_E_BACKEND;
        }
        enc = metal_sequence_encoder(st);
    } else {
        cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
        enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
        if (cmd == nullptr || enc == nullptr) {
            return GEIST_E_BACKEND;
        }
    }
    const struct metal_size threads256 = {256, 1, 1};
    metal_msg_send_set_pipeline(st, enc, st->q_norm_rope_rows_pipeline);
    metal_msg_send_set_buffer(st, enc, q->buffer->buffer, 0, 0);
    metal_msg_send_set_buffer(st, enc, q_norm_w->buffer->buffer, 0, 1);
    metal_msg_send_set_buffer(st, enc, cos->buffer->buffer, 0, 2);
    metal_msg_send_set_buffer(st, enc, sin->buffer->buffer, 0, 3);
    metal_msg_send_set_bytes(st, enc, &nr, sizeof(nr), 4);
    const struct metal_size qgroups = {(uint32_t) rows, (uint32_t) q_heads, 1};
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_Q_NORM_ROPE, qgroups);
    metal_msg_send_dispatch(st, enc, qgroups, threads256);

    if (has_kv) {
        metal_msg_send_set_pipeline(st, enc, kv_pipeline);
        metal_msg_send_set_buffer(st, enc, k->buffer->buffer, 0, 0);
        metal_msg_send_set_buffer(st, enc, v->buffer->buffer, 0, 1);
        metal_msg_send_set_buffer(st, enc, k_norm_w->buffer->buffer, 0, 2);
        metal_msg_send_set_buffer(st, enc, v_norm_w->buffer->buffer, 0, 3);
        metal_msg_send_set_buffer(st, enc, cos->buffer->buffer, 0, 4);
        metal_msg_send_set_buffer(st, enc, sin->buffer->buffer, 0, 5);
        metal_msg_send_set_buffer(st, enc, k_cache->buffer->buffer, 0, 6);
        metal_msg_send_set_buffer(st, enc, v_cache->buffer->buffer, 0, 7);
        metal_msg_send_set_bytes(st, enc, &kp, sizeof(kp), 8);
        metal_msg_send_set_bytes(st, enc, &vp, sizeof(vp), 9);
        const struct metal_size kvgroups = {(uint32_t) rows, (uint32_t) kv_heads, 1};
        metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_KV_NORM_APPEND, kvgroups);
        metal_msg_send_dispatch(st, enc, kvgroups, threads256);
    }

    if (st->sequence_active) {
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    return metal_msg_send_id0(st, cmd, "error") == nullptr ? GEIST_OK : GEIST_E_BACKEND;
}

/* Fused gemma-3n PLE block (vtbl slot), decode fast path: two dispatches
 * (ple_gate_f32: gate GEMV + gelu*ple; ple_proj_norm_f32: proj GEMV +
 * rmsnorm + residual add) replace four decomposed ops. rows==1 and F32
 * weights only — the kernels are naive GEMVs; prefill keeps the mm_sg
 * GEMM path via the decomposed fallback. */
[[nodiscard]] static enum geist_status metal_ple_block(struct geist_backend      *be,
                                                       const struct geist_tensor *x,
                                                       const struct geist_tensor *gate_w,
                                                       const struct geist_tensor *ple_in,
                                                       const struct geist_tensor *proj_w,
                                                       const struct geist_tensor *res,
                                                       const struct geist_tensor *norm_w,
                                                       float                      eps,
                                                       struct geist_tensor       *gate_scratch,
                                                       struct geist_tensor       *proj_scratch,
                                                       struct geist_tensor       *y) {

    if (be == nullptr || be->state == nullptr || x == nullptr || gate_w == nullptr ||
        ple_in == nullptr || proj_w == nullptr || res == nullptr || norm_w == nullptr ||
        gate_scratch == nullptr || proj_scratch == nullptr || y == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows = 0, d_in = 0, x_off = 0, x_stride = 0;
    size_t g_rows = 0, g_cols = 0, g_off = 0, g_stride = 0;
    size_t p_rows = 0, p_cols = 0, p_off = 0, p_stride = 0;
    size_t r_rows = 0, r_cols = 0, r_off = 0, r_stride = 0;
    size_t y_rows = 0, y_cols = 0, y_off = 0, y_stride = 0;
    size_t ps_rows = 0, ps_cols = 0, ps_off = 0, ps_stride = 0;
    size_t hpl = 0, gw_in = 0, gw_off = 0, gw_stride = 0;
    size_t d_model = 0, pw_in = 0, pw_off = 0, pw_stride = 0;
    size_t nw_n = 0, nw_off = 0;
    if (!metal_tensor_is_f32_rows(x, &rows, &d_in, &x_off, &x_stride) ||
        !metal_tensor_is_f32_matrix(gate_w, &hpl, &gw_in, &gw_off, &gw_stride) ||
        !metal_tensor_is_f32_rows(ple_in, &p_rows, &p_cols, &p_off, &p_stride) ||
        !metal_tensor_is_f32_matrix(proj_w, &d_model, &pw_in, &pw_off, &pw_stride) ||
        gw_stride != gw_in || pw_stride != pw_in || /* kernels assume dense */
        !metal_tensor_is_f32_rows(res, &r_rows, &r_cols, &r_off, &r_stride) ||
        !metal_tensor_is_f32_vector(norm_w, &nw_n, &nw_off) ||
        !metal_tensor_is_f32_rows(gate_scratch, &g_rows, &g_cols, &g_off, &g_stride) ||
        !metal_tensor_is_f32_rows(proj_scratch, &ps_rows, &ps_cols, &ps_off, &ps_stride) ||
        !metal_tensor_is_f32_rows(y, &y_rows, &y_cols, &y_off, &y_stride) ||
        rows != 1u || /* decode only: naive GEMV kernels */
        gw_in != d_in || p_rows != rows || p_cols != hpl || g_rows != rows || g_cols != hpl ||
        pw_in != hpl || r_rows != rows || r_cols != d_model || ps_rows != rows ||
        ps_cols != d_model || y_rows != rows || y_cols != d_model || nw_n != d_model) {
        return GEIST_E_UNSUPPORTED;
    }
    if (d_in > UINT32_MAX || hpl > UINT32_MAX || d_model > UINT32_MAX || x_off > UINT32_MAX ||
        gw_off > UINT32_MAX || p_off > UINT32_MAX || g_off > UINT32_MAX || pw_off > UINT32_MAX ||
        r_off > UINT32_MAX || nw_off > UINT32_MAX || y_off > UINT32_MAX ||
        x->buffer->owner != be->state || gate_w->buffer->owner != be->state ||
        ple_in->buffer->owner != be->state || proj_w->buffer->owner != be->state ||
        res->buffer->owner != be->state || norm_w->buffer->owner != be->state ||
        gate_scratch->buffer->owner != be->state || y->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status s = metal_ensure_q4k_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    struct metal_state *st = be->state;
    if (!st->use_ple_block || st->f32_ple_gate_pipeline == nullptr ||
        st->f32_ple_proj_norm_pipeline == nullptr) {
        return GEIST_E_UNSUPPORTED;
    }
    struct {
        uint32_t ni, no, rows, xo, wo, po, yo, xs, ps, ys;
    } gp      = {(uint32_t) d_in,
                 (uint32_t) hpl,
                 (uint32_t) rows,
                 (uint32_t) x_off,
                 (uint32_t) gw_off,
                 (uint32_t) p_off,
                 (uint32_t) g_off,
                 (uint32_t) x_stride,
                 (uint32_t) p_stride,
                 (uint32_t) g_stride};
    void *cmd = nullptr;
    void *enc = nullptr;
    if (st->sequence_active) {
        if (st->sequence_compute_encoder == nullptr) {
            return GEIST_E_BACKEND;
        }
        enc = metal_sequence_encoder(st);
    } else {
        cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
        enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
        if (cmd == nullptr || enc == nullptr) {
            return GEIST_E_BACKEND;
        }
    }
    const struct metal_size threads256 = {256, 1, 1};
    metal_msg_send_set_pipeline(st, enc, st->f32_ple_gate_pipeline);
    metal_msg_send_set_buffer(st, enc, x->buffer->buffer, 0, 0);
    metal_msg_send_set_buffer(st, enc, gate_w->buffer->buffer, 0, 1);
    metal_msg_send_set_buffer(st, enc, ple_in->buffer->buffer, 0, 2);
    metal_msg_send_set_buffer(st, enc, gate_scratch->buffer->buffer, 0, 3);
    metal_msg_send_set_bytes(st, enc, &gp, sizeof(gp), 4);
    const struct metal_size ggroups = {(uint32_t) hpl, (uint32_t) rows, 1};
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_F32_PLE_GATE, ggroups);
    metal_msg_send_dispatch(st, enc, ggroups, threads256);

    if (st->sequence_active) {
        st->sequence_has_work = true;
    } else {
        metal_msg_send_void0(st, enc, "endEncoding");
        metal_msg_send_void0(st, cmd, "commit");
        metal_msg_send_void0(st, cmd, "waitUntilCompleted");
        if (metal_msg_send_id0(st, cmd, "error") != nullptr) {
            return GEIST_E_BACKEND;
        }
    }

    /* Proj side: the fused ple_proj_norm_f32 kernel is a single-threadgroup
     * GEMV — 400us/layer at d_model 2048 (measured 2026-07-04, decode
     * 18.7->32.9 ms/tok). Route through the fast f32 matvec + fused
     * rmsnorm_add instead. */
    struct geist_tensor gate_1d = *gate_scratch;
    if (gate_1d.ndim == 2 && gate_1d.shape[0] == 1) {
        gate_1d.ndim      = 1;
        gate_1d.shape[0]  = gate_1d.shape[1];
        gate_1d.stride[0] = 1;
        gate_1d.shape[1]  = 0;
        gate_1d.stride[1] = 0;
    }
    struct geist_tensor proj_1d = *proj_scratch;
    if (proj_1d.ndim == 2 && proj_1d.shape[0] == 1) {
        proj_1d.ndim      = 1;
        proj_1d.shape[0]  = proj_1d.shape[1];
        proj_1d.stride[0] = 1;
        proj_1d.shape[1]  = 0;
        proj_1d.stride[1] = 0;
    }
    s = metal_matvec_f32_dense(be, &gate_1d, proj_w, &proj_1d);
    if (s != GEIST_OK) {
        return s;
    }
    return metal_rmsnorm_add(be, res, proj_scratch, norm_w, eps, y);
}

[[nodiscard]] static enum geist_status metal_ensure_argmax_pipeline(struct geist_backend *be) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct metal_state *st = be->state;
    if (st->argmax_pipeline != nullptr && st->argmax_batch_pipeline != nullptr &&
        st->argmax_result_buffer != nullptr && st->argmax_result_mapped != nullptr &&
        st->argmax_result_capacity >= 1u) {
        return GEIST_OK;
    }

    void *ns_string = metal_objc_get_class(st, "NSString");
    if (ns_string == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal: NSString class unavailable");
        return GEIST_E_BACKEND;
    }
    if (st->argmax_pipeline == nullptr || st->argmax_batch_pipeline == nullptr) {
        void *source =
                metal_msg_send_id_cstr(st, ns_string, "stringWithUTF8String:", metal_argmax_source);
        if (source == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal: failed to create argmax shader source");
            return GEIST_E_BACKEND;
        }
        void *err          = nullptr;
        st->argmax_library = metal_msg_send_id_id_id_err(
                st, st->device, "newLibraryWithSource:options:error:", source, nullptr, &err);
        if (st->argmax_library == nullptr) {
            const char *msg = metal_nserror_message(st, err);
            geist_backend_set_error(be,
                                    GEIST_E_BACKEND,
                                    "metal: argmax shader compile failed%s%s",
                                    msg != nullptr ? ": " : "",
                                    msg != nullptr ? msg : "");
            return GEIST_E_BACKEND;
        }
        enum geist_status s = metal_create_named_pipeline(be,
                                                          st->argmax_library,
                                                          ns_string,
                                                          "argmax_f32",
                                                          &st->argmax_function,
                                                          &st->argmax_pipeline);
        if (s != GEIST_OK) {
            return s;
        }
        s = metal_create_named_pipeline(be,
                                        st->argmax_library,
                                        ns_string,
                                        "argmax_f32_batch",
                                        &st->argmax_batch_function,
                                        &st->argmax_batch_pipeline);
        if (s != GEIST_OK) {
            return s;
        }
    }
    if (st->argmax_result_buffer == nullptr) {
        st->argmax_result_buffer = metal_msg_send_id_size_uint(st,
                                                               st->device,
                                                               "newBufferWithLength:options:",
                                                               sizeof(uint32_t),
                                                               METAL_RESOURCE_STORAGE_MODE_SHARED);
        if (st->argmax_result_buffer == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal argmax: result buffer allocation failed");
            return GEIST_E_BACKEND;
        }
        st->argmax_result_mapped = metal_msg_send_ptr0(st, st->argmax_result_buffer, "contents");
        if (st->argmax_result_mapped == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal argmax: result buffer is not mappable");
            return GEIST_E_BACKEND;
        }
        st->argmax_result_capacity = 1u;
    }
    return GEIST_OK;
}

/* Device greedy argmax (vtbl slot): one 256-thread threadgroup scans the
 * [1, n] logits row on the GPU; the host flush then reads a 4-byte token
 * instead of mapping the whole 1 MB logits row (plan Phase A1). Tie-break
 * = lowest index, matching geist_sampler_argmax. */
[[nodiscard]] static enum geist_status
metal_argmax_f32(struct geist_backend *be, const struct geist_tensor *logits, int32_t *out_index) {

    if (be == nullptr || be->state == nullptr || logits == nullptr || out_index == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t rows = 0, n = 0, x_off = 0, x_stride = 0;
    if (!metal_tensor_is_f32_rows(logits, &rows, &n, &x_off, &x_stride) || rows != 1u || n == 0 ||
        n > UINT32_MAX || x_off > UINT32_MAX || logits->buffer->owner != be->state) {
        return GEIST_E_UNSUPPORTED;
    }
    enum geist_status s = metal_ensure_argmax_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    struct metal_state *st = be->state;

    struct {
        uint32_t n, xo;
    } ap      = {(uint32_t) n, (uint32_t) x_off};
    void *cmd = nullptr;
    void *enc = nullptr;
    if (st->sequence_active) {
        if (st->sequence_compute_encoder == nullptr) {
            return GEIST_E_BACKEND;
        }
        enc = metal_sequence_encoder(st);
    } else {
        cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
        enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
        if (cmd == nullptr || enc == nullptr) {
            return GEIST_E_BACKEND;
        }
    }
    metal_msg_send_set_pipeline(st, enc, st->argmax_pipeline);
    metal_msg_send_set_buffer(st, enc, logits->buffer->buffer, 0, 0);
    metal_msg_send_set_buffer(st, enc, st->argmax_result_buffer, 0, 1);
    metal_msg_send_set_bytes(st, enc, &ap, sizeof(ap), 2);
    const struct metal_size groups  = {1, 1, 1};
    const struct metal_size threads = {256, 1, 1};
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_ARGMAX, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
    if (st->sequence_active) {
        st->sequence_has_work = true;
        /* The 4-byte result read is the token's only host sync point. */
        metal_flush_if_referenced(st, st->argmax_result_buffer);
    } else {
        metal_msg_send_void0(st, enc, "endEncoding");
        metal_msg_send_void0(st, cmd, "commit");
        metal_msg_send_void0(st, cmd, "waitUntilCompleted");
        if (metal_msg_send_id0(st, cmd, "error") != nullptr) {
            return GEIST_E_BACKEND;
        }
    }
    *out_index = (int32_t) ((const uint32_t *) st->argmax_result_mapped)[0];
    return GEIST_OK;
}

/* Fused f32→f16 KV append (vtbl slot): convert seq rows of scratch K/V and
 * store them at row q_position of the f16 caches — the kernel the f32-KV
 * flash paths use for their staging, aimed at the cache instead. */
[[nodiscard]] static enum geist_status metal_kv_append_f16(struct geist_backend      *be,
                                                           const struct geist_tensor *k_src,
                                                           const struct geist_tensor *v_src,
                                                           size_t                     q_position,
                                                           struct geist_tensor       *k_cache,
                                                           struct geist_tensor       *v_cache) {

    if (be == nullptr || be->state == nullptr || k_src == nullptr || v_src == nullptr ||
        k_cache == nullptr || v_cache == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t s_rows = 0, s_heads = 0, s_hd = 0, k_off = 0;
    size_t vs_rows = 0, vs_heads = 0, vs_hd = 0, v_off = 0;
    size_t kc_rows = 0, kc_heads = 0, kc_hd = 0, kc_off = 0;
    size_t vc_rows = 0, vc_heads = 0, vc_hd = 0, vc_off = 0;
    if (!metal_tensor_is_f32_3d(k_src, &s_rows, &s_heads, &s_hd, &k_off) ||
        !metal_tensor_is_f32_3d(v_src, &vs_rows, &vs_heads, &vs_hd, &v_off) ||
        !metal_tensor_is_f16_3d(k_cache, &kc_rows, &kc_heads, &kc_hd, &kc_off) ||
        !metal_tensor_is_f16_3d(v_cache, &vc_rows, &vc_heads, &vc_hd, &vc_off) ||
        vs_rows != s_rows || vs_heads != s_heads || vs_hd != s_hd || kc_heads != s_heads ||
        kc_hd != s_hd || vc_heads != s_heads || vc_hd != s_hd || kc_rows < q_position + s_rows ||
        vc_rows < q_position + s_rows) {
        return GEIST_E_UNSUPPORTED;
    }
    const size_t kv_out = s_heads * s_hd;
    const size_t elems  = s_rows * kv_out;
    if (elems == 0 || elems > UINT32_MAX || k_off > UINT32_MAX || v_off > UINT32_MAX ||
        kc_off > UINT32_MAX || vc_off > UINT32_MAX || q_position > UINT32_MAX ||
        k_src->buffer->owner != be->state || v_src->buffer->owner != be->state ||
        k_cache->buffer->owner != be->state || v_cache->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }
    enum geist_status s = metal_ensure_attention_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    struct metal_state *st = be->state;
    if (st->kv_append_rows_f16_pipeline == nullptr) {
        return GEIST_E_UNSUPPORTED;
    }
    struct {
        uint32_t elems, kv_out, k_offset, v_offset, k_cache_offset, v_cache_offset, q_position;
    } ap = {(uint32_t) elems,
            (uint32_t) kv_out,
            (uint32_t) k_off,
            (uint32_t) v_off,
            (uint32_t) kc_off,
            (uint32_t) vc_off,
            (uint32_t) q_position};

    void *cmd = nullptr;
    void *enc = nullptr;
    if (st->sequence_active) {
        if (st->sequence_compute_encoder == nullptr) {
            return GEIST_E_BACKEND;
        }
        enc = metal_sequence_encoder(st);
    } else {
        cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
        enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
        if (cmd == nullptr || enc == nullptr) {
            return GEIST_E_BACKEND;
        }
    }
    metal_msg_send_set_pipeline(st, enc, st->kv_append_rows_f16_pipeline);
    metal_msg_send_set_buffer(st, enc, k_src->buffer->buffer, 0, 0);
    metal_msg_send_set_buffer(st, enc, v_src->buffer->buffer, 0, 1);
    metal_msg_send_set_buffer(st, enc, k_cache->buffer->buffer, 0, 2);
    metal_msg_send_set_buffer(st, enc, v_cache->buffer->buffer, 0, 3);
    metal_msg_send_set_bytes(st, enc, &ap, sizeof(ap), 4);
    const struct metal_size groups  = {(elems + 255u) / 256u, 1, 1};
    const struct metal_size threads = {256, 1, 1};
    metal_profile_add_dispatch(st, METAL_PROFILE_DISPATCH_KV_APPEND_ROWS, groups);
    metal_msg_send_dispatch(st, enc, groups, threads);
    if (st->sequence_active) {
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    return metal_msg_send_id0(st, cmd, "error") == nullptr ? GEIST_OK : GEIST_E_BACKEND;
}

[[nodiscard]] static enum geist_status metal_attention(struct geist_backend      *be,
                                                       const struct geist_tensor *q,
                                                       const struct geist_tensor *k,
                                                       const struct geist_tensor *value,
                                                       size_t                     q_offset,
                                                       size_t                     sliding_window,
                                                       struct geist_tensor       *out) {

    if (be == nullptr || be->state == nullptr || q == nullptr || k == nullptr || value == nullptr ||
        out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }

    {
        static int dbg = -1;
        if (dbg < 0) {
            const char *e = getenv("GEIST_METAL_DEBUG_LINEAR");
            dbg           = (e != nullptr && e[0] != '\0' && strcmp(e, "0") != 0) ? 1 : 0;
        }
        if (dbg && q->buffer != nullptr && q->buffer->mapped != nullptr && k->buffer != nullptr &&
            k->buffer->mapped != nullptr) {
            const float *qp = (const float *) ((const uint8_t *) q->buffer->mapped + q->offset);
            const float *kp = (const float *) ((const uint8_t *) k->buffer->mapped + k->offset);
            size_t       nq = 1, nk = 1;
            for (int i = 0; i < q->ndim; i++)
                nq *= (size_t) q->shape[i];
            for (int i = 0; i < k->ndim; i++)
                nk *= (size_t) k->shape[i];
            float  aq = 0, ak = 0;
            size_t nanq = 0, nank = 0;
            for (size_t i = 0; i < nq; i++) {
                if (isnan(qp[i]))
                    nanq++;
                else if (fabsf(qp[i]) > aq)
                    aq = fabsf(qp[i]);
            }
            for (size_t i = 0; i < nk; i++) {
                if (isnan(kp[i]))
                    nank++;
                else if (fabsf(kp[i]) > ak)
                    ak = fabsf(kp[i]);
            }
            size_t firstnan = (size_t) -1;
            for (size_t i = 0; i < nq && firstnan == (size_t) -1; i++)
                if (isnan(qp[i]))
                    firstnan = i;
            fprintf(stderr,
                    "attn firstnan_q=%zd (row %zd) ",
                    (ssize_t) firstnan,
                    firstnan == (size_t) -1
                            ? (ssize_t) -1
                            : (ssize_t) (firstnan / ((size_t) q->shape[1] * (size_t) q->shape[2])));
            fprintf(stderr,
                    "attn q[%lld,%lld,%lld] |q|=%g nanq=%zu k[%lld,..] |k|=%g nank=%zu qoff=%zu "
                    "sw=%zu kdt=%d\n",
                    (long long) q->shape[0],
                    (long long) q->shape[1],
                    (long long) q->shape[2],
                    (double) aq,
                    nanq,
                    (long long) k->shape[0],
                    (double) ak,
                    nank,
                    q_offset,
                    sliding_window,
                    (int) k->dtype);
        }
    }
    size_t q_rows = 0, q_heads = 0, head_dim = 0, q_off = 0;
    size_t k_rows = 0, k_heads = 0, k_head_dim = 0, k_off = 0;
    size_t v_rows = 0, v_heads = 0, v_head_dim = 0, v_off = 0;
    size_t out_rows = 0, out_heads = 0, out_head_dim = 0, out_off = 0;
    if (!metal_tensor_is_f32_3d(q, &q_rows, &q_heads, &head_dim, &q_off) ||
        !metal_tensor_is_f32_3d(out, &out_rows, &out_heads, &out_head_dim, &out_off)) {
        geist_backend_set_error(
                be, GEIST_E_UNSUPPORTED, "metal attention: q/out tensors must be F32 DENSE 3D");
        return GEIST_E_UNSUPPORTED;
    }
    const bool kv_is_f32 = metal_tensor_is_f32_3d(k, &k_rows, &k_heads, &k_head_dim, &k_off) &&
                           metal_tensor_is_f32_3d(value, &v_rows, &v_heads, &v_head_dim, &v_off);
    const bool kv_is_f16 = !kv_is_f32 &&
                           metal_tensor_is_f16_3d(k, &k_rows, &k_heads, &k_head_dim, &k_off) &&
                           metal_tensor_is_f16_3d(value, &v_rows, &v_heads, &v_head_dim, &v_off);
    if (!kv_is_f32 && !kv_is_f16) {
        geist_backend_set_error(be,
                                GEIST_E_UNSUPPORTED,
                                "metal attention: K/V cache must be matching F32 or F16 DENSE 3D");
        return GEIST_E_UNSUPPORTED;
    }
    if (q_rows != out_rows || q_heads != out_heads || head_dim != out_head_dim ||
        k_rows != v_rows || k_heads != v_heads || head_dim != k_head_dim ||
        head_dim != v_head_dim || k_heads == 0 || q_heads % k_heads != 0) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "metal attention: shape mismatch");
        return GEIST_E_INVALID_ARG;
    }
    if (head_dim > 512u) {
        geist_backend_set_error(
                be, GEIST_E_UNSUPPORTED, "metal attention: head_dim exceeds shader limit");
        return GEIST_E_UNSUPPORTED;
    }
    if (q_rows > UINT32_MAX || k_rows > UINT32_MAX || q_heads > UINT32_MAX ||
        k_heads > UINT32_MAX || head_dim > UINT32_MAX || q_offset > UINT32_MAX ||
        sliding_window > UINT32_MAX || q_off > UINT32_MAX || k_off > UINT32_MAX ||
        v_off > UINT32_MAX || out_off > UINT32_MAX || q_offset > SIZE_MAX - q_rows ||
        q_offset + q_rows > UINT32_MAX || q->buffer->owner != be->state ||
        k->buffer->owner != be->state || value->buffer->owner != be->state ||
        out->buffer->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }

    enum geist_status s = metal_ensure_attention_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    struct metal_state *st = be->state;
    if (q_rows == 1u &&
        (head_dim <= 256u || (head_dim <= 512u && st->attention_dec512_f16_pipeline != nullptr)) &&
        k_rows >= 32u && st->attention_dec_f16_pipeline != nullptr &&
        st->attention_dec_combine_pipeline != nullptr &&
        st->kv_append_rows_f16_pipeline != nullptr && !metal_env_disabled("GEIST_METAL_FLASH")) {
        s = metal_attention_dec_kv(be,
                                   st,
                                   q,
                                   k,
                                   value,
                                   out,
                                   k_rows,
                                   q_heads,
                                   k_heads,
                                   head_dim,
                                   q_offset,
                                   sliding_window,
                                   q_off,
                                   k_off,
                                   v_off,
                                   out_off,
                                   kv_is_f16);
        if (s == GEIST_OK) {
            return GEIST_OK;
        }
        /* fall through to the scalar kernel on failure */
    }
    if (q_rows > 1u && (q_rows % 8u) == 0u &&
        ((head_dim <= 256u && (head_dim % 32u) == 0u) ||
         (head_dim <= 512u && (head_dim % 64u) == 0u &&
          st->attention_flash_sg8_f16_pipeline != nullptr)) &&
        k_rows >= 32u && st->attention_flash_sg_f16_pipeline != nullptr &&
        st->kv_append_rows_f16_pipeline != nullptr && !metal_env_disabled("GEIST_METAL_FLASH")) {
        s = metal_attention_flash_kv(be,
                                     st,
                                     q,
                                     k,
                                     value,
                                     out,
                                     q_rows,
                                     k_rows,
                                     q_heads,
                                     k_heads,
                                     head_dim,
                                     q_offset,
                                     sliding_window,
                                     q_off,
                                     k_off,
                                     v_off,
                                     out_off,
                                     kv_is_f16);
        if (s == GEIST_OK) {
            return GEIST_OK;
        }
        /* fall through to the scalar kernel on failure */
    }
    const struct metal_attention_params params = {
            .rows           = (uint32_t) q_rows,
            .kv_len         = (uint32_t) k_rows,
            .q_heads        = (uint32_t) q_heads,
            .kv_heads       = (uint32_t) k_heads,
            .head_dim       = (uint32_t) head_dim,
            .q_position     = (uint32_t) q_offset,
            .sliding_window = (uint32_t) sliding_window,
            .q_offset       = (uint32_t) q_off,
            .k_cache_offset = (uint32_t) k_off,
            .v_cache_offset = (uint32_t) v_off,
            .y_offset       = (uint32_t) out_off,
    };
    if (st->sequence_active) {
        if (st->sequence_compute_encoder == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal attention: sequence has no encoder");
            return GEIST_E_BACKEND;
        }
        metal_encode_attention_rows(st, metal_sequence_encoder(st), q, k, value, out, &params);
        st->sequence_has_work = true;
        return GEIST_OK;
    }
    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    void *enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
    if (cmd == nullptr || enc == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal attention: command encoder failed");
        return GEIST_E_BACKEND;
    }
    metal_encode_attention_rows(st, enc, q, k, value, out, &params);
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    void *err = metal_msg_send_id0(st, cmd, "error");
    if (err != nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal attention: command failed");
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
static void metal_capture_begin(struct metal_state *st, enum geist_command_sequence_kind kind) {
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
    void *mgr_class  = metal_objc_get_class(st, "MTLCaptureManager");
    void *desc_class = metal_objc_get_class(st, "MTLCaptureDescriptor");
    void *str_class  = metal_objc_get_class(st, "NSString");
    void *url_class  = metal_objc_get_class(st, "NSURL");
    if (mgr_class == nullptr || desc_class == nullptr || str_class == nullptr ||
        url_class == nullptr) {
        return;
    }
    void *mgr     = metal_msg_send_id0(st, mgr_class, "sharedCaptureManager");
    void *desc    = metal_msg_send_id0(st, desc_class, "new");
    void *ns_path = metal_msg_send_id_cstr(st, str_class, "stringWithUTF8String:", path);
    if (mgr == nullptr || desc == nullptr || ns_path == nullptr) {
        return;
    }
    void *url = metal_msg_send_id_id(st, url_class, "fileURLWithPath:", ns_path);
    metal_msg_send_id_id(st, desc, "setCaptureObject:", st->device);
    /* 2 = MTLCaptureDestinationGPUTraceDocument */
    metal_msg_send_void_ulong(st, desc, "setDestination:", 2);
    metal_msg_send_id_id(st, desc, "setOutputURL:", url);
    void *err = nullptr;
    if (!metal_msg_send_bool_id_err(st, mgr, "startCaptureWithDescriptor:error:", desc, &err)) {
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
        struct geist_backend *be, enum geist_command_sequence_kind kind, int *out_token) {

    if (out_token == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out_token = 0;
    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct metal_state *st = be->state;
    if (st->sequence_active) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "metal command sequence: nested begin");
        return GEIST_E_INVALID_ARG;
    }
    switch (kind) {
    case GEIST_COMMAND_SEQUENCE_VERIFY_GREEDY:
    case GEIST_COMMAND_SEQUENCE_DECODE_LAYER_LOOP:
    case GEIST_COMMAND_SEQUENCE_DECODE_GREEDY_STEP:
    case GEIST_COMMAND_SEQUENCE_PREFILL_TEXT:
        break;
    default:
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "metal command sequence: invalid kind");
        return GEIST_E_INVALID_ARG;
    }

    metal_capture_begin(st, kind);

    static int g_seq_created;
    g_seq_created++;
    if (getenv("GEIST_SEQ_COUNT") && (g_seq_created % 16) == 0)
        fprintf(stderr, "[seqdbg] created=%d\n", g_seq_created);
    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    if (cmd == nullptr) {
        geist_backend_set_error(
                be, GEIST_E_BACKEND, "metal command sequence: command buffer failed");
        return GEIST_E_BACKEND;
    }
    void *enc = metal_msg_send_id0(st, cmd, "computeCommandEncoder");
    if (enc == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal command sequence: encoder failed");
        return GEIST_E_BACKEND;
    }
    metal_msg_send_void0(st, cmd, "retain");
    metal_msg_send_void0(st, enc, "retain");

    if (st->sequence_token == INT_MAX) {
        st->sequence_token = 0;
    }
    st->sequence_token++;
    st->sequence_kind            = kind;
    st->sequence_command_buffer  = cmd;
    st->sequence_compute_encoder = enc;
    st->sequence_active          = true;
    st->sequence_has_work        = false;
    st->seq_dispatch_count       = 0;
    st->seq_disp_at_rotate       = 0;
    st->seq_begin_ns             = metal_now_ns();
    *out_token                   = st->sequence_token;
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status
metal_command_sequence_end(struct geist_backend *be, int token, bool submit) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct metal_state *st = be->state;
    if (!st->sequence_active || token == 0 || token != st->sequence_token) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "metal command sequence: invalid token");
        return GEIST_E_INVALID_ARG;
    }

    void      *cmd               = st->sequence_command_buffer;
    void      *enc               = st->sequence_compute_encoder;
    const bool has_work          = st->sequence_has_work;
    st->sequence_compute_encoder = nullptr;
    st->sequence_command_buffer  = nullptr;
    st->sequence_active          = false;
    st->sequence_has_work        = false;

    metal_msg_send_void0(st, enc, "endEncoding");
    enum geist_status out = GEIST_OK;
    /* Commit even when no work was encoded: an uncommitted command buffer
     * permanently occupies one of the queue's (default 64) slots, and the
     * batched-submit region hooks legitimately close empty sequences. */
    if (submit && !has_work) {
        /* Empty sequence (e.g. the rotation right after a flush): commit
         * to free the queue slot, but skip the ~0.4 ms wait handshake —
         * nothing observes its completion. */
        metal_msg_send_void0(st, cmd, "commit");
    } else if (submit) {
        const enum metal_profile_stage wait_stage =
                metal_profile_wait_stage_for_sequence(st->sequence_kind);
        const uint64_t wait_start_ns = st->profile_enabled ? metal_now_ns() : 0;
        const bool     seq_trace     = metal_env_enabled("GEIST_METAL_SEQ_TRACE");
        const uint64_t commit_ns     = seq_trace ? metal_now_ns() : 0;
        metal_msg_send_void0(st, cmd, "commit");
        metal_msg_send_void0(st, cmd, "waitUntilCompleted");
        /* Pipelined buffers committed before this one are complete now
         * (same queue, commit order); surface their errors and release. */
        bool pending_failed = false;
        metal_sequence_drain_pending(st, &pending_failed);
        if (pending_failed && out == GEIST_OK) {
            geist_backend_set_error(
                    be, GEIST_E_BACKEND, "metal command sequence: pipelined buffer failed");
            out = GEIST_E_BACKEND;
        }
        metal_profile_add_wait(st, wait_stage, wait_start_ns);
        if (seq_trace) {
            const uint64_t done_ns = metal_now_ns();
            union {
                void *raw;
                double (*fn)(void *, void *);
            } getd                 = {.raw = st->objc_msgSend};
            const double gpu_start = getd.fn(cmd, metal_sel_register_name(st, "GPUStartTime"));
            const double gpu_end   = getd.fn(cmd, metal_sel_register_name(st, "GPUEndTime"));
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
            geist_backend_set_error(be, GEIST_E_BACKEND, "metal command sequence: command failed");
            out = GEIST_E_BACKEND;
        }
    }

    /* No-op on the normal submit path (already drained after the wait);
     * covers the empty/!submit paths where pendings were committed but
     * never waited on — release only, Metal keeps its internal refs. */
    metal_sequence_drain_pending(st, nullptr);
    metal_capture_end(st);

    metal_msg_send_void0(st, enc, "release");
    metal_msg_send_void0(st, cmd, "release");
    return out;
}

[[nodiscard]] static enum geist_status metal_load_runtime(struct geist_backend *be,
                                                          struct metal_state   *st) {

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
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED, "metal: Metal framework is unavailable");
        return GEIST_E_UNSUPPORTED;
    }

    for (size_t i = 0; objc_paths[i] != nullptr; i++) {
        st->objc_handle = dlopen(objc_paths[i], RTLD_NOW | RTLD_LOCAL);
        if (st->objc_handle != nullptr) {
            break;
        }
    }
    if (st->objc_handle == nullptr) {
        geist_backend_set_error(
                be, GEIST_E_UNSUPPORTED, "metal: Objective-C runtime is unavailable");
        return GEIST_E_UNSUPPORTED;
    }

    st->MTLCreateSystemDefaultDevice =
            metal_dlsym(st->metal_handle, "MTLCreateSystemDefaultDevice");
    st->objc_msgSend     = metal_dlsym(st->objc_handle, "objc_msgSend");
    st->sel_registerName = metal_dlsym(st->objc_handle, "sel_registerName");
    st->objc_getClass    = metal_dlsym(st->objc_handle, "objc_getClass");
    if (st->MTLCreateSystemDefaultDevice == nullptr || st->objc_msgSend == nullptr ||
        st->sel_registerName == nullptr || st->objc_getClass == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal: runtime symbols are incomplete");
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
#else
    (void) st;
    geist_backend_set_error(
            be, GEIST_E_UNSUPPORTED, "metal: backend is only available on Apple platforms");
    return GEIST_E_UNSUPPORTED;
#endif
}

[[nodiscard]] static enum geist_status metal_create(struct geist_backend            *be,
                                                    const struct geist_backend_opts *opts) {

    (void) opts;
    struct metal_state *st = geist_backend_alloc(be, sizeof(*st), alignof(struct metal_state));
    if (st == nullptr) {
        geist_backend_set_error(
                be, GEIST_E_OOM, "metal: failed to allocate %zu-byte state", sizeof(*st));
        return GEIST_E_OOM;
    }
    *st                    = (struct metal_state) {0};
    st->backend            = be;
    const char *ple_block  = getenv("GEIST_METAL_PLE_BLOCK");
    st->use_ple_block      = ple_block == nullptr || strcmp(ple_block, "0") != 0;
    const char *q4k_n4     = getenv("GEIST_METAL_Q4K_N4");
    st->use_q4k_n4         = q4k_n4 == nullptr || strcmp(q4k_n4, "0") != 0;
    const char *q4k_m16_n2 = getenv("GEIST_METAL_Q4K_M16_N2");
    st->use_q4k_m16_n2     = q4k_m16_n2 != nullptr && strcmp(q4k_m16_n2, "1") == 0;
    /* Simdgroup-matmul Q4_K GEMM (llama.cpp mul_mm-derived). Default ON: only
     * runs for full tiles (dispatch guard requires rows%32==0 && n_out%64==0),
     * so the partial-tile path never executes and non-conforming shapes fall
     * back to the m16 kernel. Numerical parity for both paths is covered by
     * tests/test_backend_metal_q4k_matmul_parity.c. Set
     * GEIST_METAL_Q4K_MM_SG=0 to disable. */
    const char *q4k_mm_sg    = getenv("GEIST_METAL_Q4K_MM_SG");
    st->use_q4k_mm_sg        = q4k_mm_sg == nullptr || strcmp(q4k_mm_sg, "0") != 0;
    const char *rmsnorm_simd = getenv("GEIST_METAL_RMSNORM_SIMD");
    st->use_rmsnorm_simd     = rmsnorm_simd == nullptr || strcmp(rmsnorm_simd, "0") != 0;
    const char *q6k_n4       = getenv("GEIST_METAL_Q6K_N4");
    st->use_q6k_n4           = q6k_n4 == nullptr || strcmp(q6k_n4, "0") != 0;
    /* Off by default: the plain-layout n4 kernel (llama mul_mv structure)
     * outruns the packed nt4 path and needs no load-time repack. */
    /* Command-buffer pipelining (llama n_cb-style): rotate every N
     * dispatches, default 192 (~3 buffers per decode token — llama's
     * measured optimum on M-series is 2-3 buffers per graph).
     * GEIST_METAL_PIPELINE=0 disables, =N sets the rotation period. */
    const char *pipeline_env = getenv("GEIST_METAL_PIPELINE");
    st->seq_rotate_every     = pipeline_env != nullptr ? (uint32_t) atoi(pipeline_env) : 192u;
    st->profile_enabled      = metal_env_enabled("GEIST_METAL_PROFILE");

    enum geist_status s = metal_load_runtime(be, st);
    if (s != GEIST_OK) {
        metal_destroy_state(be, st);
        return s;
    }

    st->device = metal_create_default_device(st);
    if (st->device == nullptr) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED, "metal: no default Metal device");
        metal_destroy_state(be, st);
        return GEIST_E_UNSUPPORTED;
    }
    metal_msg_send_void0(st, st->device, "retain");

    st->command_queue = metal_msg_send_id0(st, st->device, "newCommandQueue");
    if (st->command_queue == nullptr) {
        geist_backend_set_error(be, GEIST_E_BACKEND, "metal: failed to create command queue");
        metal_destroy_state(be, st);
        return GEIST_E_BACKEND;
    }

    void       *name = metal_msg_send_id0(st, st->device, "name");
    const char *utf8 = name != nullptr ? metal_msg_send_cstr0(st, name, "UTF8String") : nullptr;
    snprintf(st->device_name,
             sizeof(st->device_name),
             "%s",
             utf8 != nullptr && utf8[0] != '\0' ? utf8 : "Apple Metal GPU");

    be->state = st;
    return GEIST_OK;
}

static enum geist_support metal_supports_op(struct geist_backend                *be,
                                            const struct geist_op_support_query *query) {

    (void) be;
    if (query == nullptr || query->op != GEIST_OP_LINEAR || query->input_count < 2) {
        return GEIST_SUPPORT_NONE;
    }
    const struct geist_tensor_format *x_fmt = &query->inputs[0];
    const struct geist_tensor_format *w_fmt = &query->inputs[1];
    if (x_fmt->dtype == GEIST_DTYPE_F32 && x_fmt->layout == GEIST_LAYOUT_DENSE &&
        (w_fmt->dtype == GEIST_DTYPE_Q4_K || w_fmt->dtype == GEIST_DTYPE_Q6_K) &&
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
static void metal_linear_debug_stats(const float               *x,
                                     size_t                     nx,
                                     const float               *y,
                                     size_t                     ny,
                                     const struct geist_weight *w,
                                     size_t                     m);

static void metal_linear_mN(const float               *x,
                            const struct geist_weight *w,
                            size_t                     m,
                            struct geist_backend      *be,
                            float                     *y) {
    struct metal_state  *st    = be->state;
    const size_t         n_in  = (size_t) w->n_in;
    const size_t         n_out = (size_t) w->n_out;
    size_t               xo    = 0;
    size_t               wo    = 0;
    size_t               yo    = 0;
    struct geist_buffer *bx    = metal_buf_reg_find(st, x, &xo);
    struct geist_buffer *bw    = metal_buf_reg_find(st, w->raw, &wo);
    struct geist_buffer *by    = metal_buf_reg_find(st, y, &yo);
    if (bx == nullptr || bw == nullptr || by == nullptr) {
        /* One of the pointers is plain host memory (the engine passes heap
         * scratch for small helper projections, e.g. single-row views).
         * Compute on host — same math as cpu_scalar (dequant row, double
         * accumulator), so token parity with the CPU backends holds. The
         * inputs may be GPU-pending, so drain the batch first. */
        metal_batch_flush(st);
        float *row = heap_alloc_array_aligned(float, n_in);
        if (row == nullptr) {
            memset(y, 0, m * n_out * sizeof(float));
            return;
        }
        const uint8_t *base = (const uint8_t *) w->raw;
        for (size_t j = 0; j < n_out; j++) {
            switch ((enum geist_dtype) w->dtype) {
            case GEIST_DTYPE_F32:
                memcpy(row, base + j * n_in * sizeof(float), n_in * sizeof(float));
                break;
            case GEIST_DTYPE_Q4_K:
                dequant_q4_K_row(base + j * n_in / Q4_K_BLOCK_ELEMS * Q4_K_BLOCK_BYTES, row, n_in);
                break;
            case GEIST_DTYPE_Q6_K:
                dequant_q6_K_row(base + j * n_in / Q6_K_BLOCK_ELEMS * Q6_K_BLOCK_BYTES, row, n_in);
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
        safe_free((void **) &row);
        metal_linear_debug_stats(x, m * n_in, y, m * n_out, w, m);
        return;
    }
    struct geist_tensor tx = {
            .buffer = bx,
            .offset = xo,
            .dtype  = GEIST_DTYPE_F32,
            .layout = GEIST_LAYOUT_DENSE,
            .ndim   = 2,
            .shape  = {(int64_t) m, (int64_t) n_in},
            .stride = {(int64_t) n_in, 1},
    };
    struct geist_tensor ty = {
            .buffer = by,
            .offset = yo,
            .dtype  = GEIST_DTYPE_F32,
            .layout = GEIST_LAYOUT_DENSE,
            .ndim   = 2,
            .shape  = {(int64_t) m, (int64_t) n_out},
            .stride = {(int64_t) n_out, 1},
    };
    struct geist_tensor tw = {
            .buffer = bw,
            .offset = wo,
            .dtype  = (enum geist_dtype) w->dtype,
            .layout = GEIST_LAYOUT_BLOCK_QUANTIZED,
            .ndim   = 2,
            .shape  = {(int64_t) n_out, (int64_t) n_in},
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
        tw.layout    = GEIST_LAYOUT_DENSE;
        tw.stride[0] = (int64_t) n_in;
        tw.stride[1] = 1;
        s            = metal_matmul_f32_dense(be, &tx, &tw, &ty);
        break;
    default:
        s = GEIST_E_UNSUPPORTED;
        break;
    }
    if (s != GEIST_OK) {
        fprintf(stderr,
                "geist metal: linear dispatch failed "
                "(dtype=%u m=%zu %zux%zu xo=%zu wo=%zu yo=%zu): %s\n",
                (unsigned) w->dtype,
                m,
                n_out,
                n_in,
                xo,
                wo,
                yo,
                geist_backend_errmsg(be));
        memset(y, 0, m * n_out * sizeof(float));
    }
    metal_linear_debug_stats(x, m * n_in, y, m * n_out, w, m);
}

static void
metal_linear_m1(const float *x, const struct geist_weight *w, struct geist_backend *be, float *y) {
    metal_linear_mN(x, w, 1, be, y);
}

/* GEIST_METAL_DEBUG_LINEAR=1: per-linear x/y absmax trace. Every layer
 * stage flows through linear, so the first all-zero input pinpoints which
 * in-between op (rmsnorm/rope/attention/add/mul) lost the data. */
static void metal_linear_debug_stats(const float               *x,
                                     size_t                     nx,
                                     const float               *y,
                                     size_t                     ny,
                                     const struct geist_weight *w,
                                     size_t                     m) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *e = getenv("GEIST_METAL_DEBUG_LINEAR");
        enabled       = (e != nullptr && e[0] != '\0' && strcmp(e, "0") != 0) ? 1 : 0;
    }
    if (!enabled) {
        return;
    }
    float  ax   = 0.0f;
    float  ay   = 0.0f;
    size_t nanx = 0;
    size_t nany = 0;
    for (size_t i = 0; i < nx; i++) {
        if (isnan(x[i])) {
            nanx++;
            continue;
        }
        const float a = fabsf(x[i]);
        if (a > ax) {
            ax = a;
        }
    }
    for (size_t i = 0; i < ny; i++) {
        if (isnan(y[i])) {
            nany++;
            continue;
        }
        const float a = fabsf(y[i]);
        if (a > ay) {
            ay = a;
        }
    }
    fprintf(stderr,
            "linear dtype=%u m=%zu %dx%d |x|=%g |y|=%g nanx=%zu nany=%zu\n",
            (unsigned) w->dtype,
            m,
            w->n_out,
            w->n_in,
            (double) ax,
            (double) ay,
            nanx,
            nany);
}

/* Tensor-based linear (main's optional vtbl slot): dispatch the GEMM from
 * the engine's existing tensor views — no host pointers, so the op encodes
 * onto the open batch when one is active. UNSUPPORTED falls back to the
 * resolved host-pointer kernels. */
[[nodiscard]] static enum geist_status metal_linear_t(struct geist_backend      *be,
                                                      const struct geist_tensor *x,
                                                      const struct geist_weight *w,
                                                      const struct geist_tensor *t_w,
                                                      size_t                     m,
                                                      struct geist_tensor       *y) {

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
            x1.ndim      = 1;
            x1.shape[0]  = x1.shape[1];
            x1.stride[0] = x1.stride[1];
            x1.shape[1]  = 0;
            x1.stride[1] = 0;
        }
        if (y1.ndim == 2 && y1.shape[0] == 1) {
            y1.ndim      = 1;
            y1.shape[0]  = y1.shape[1];
            y1.stride[0] = y1.stride[1];
            y1.shape[1]  = 0;
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
static int metal_parallel_region_begin(struct geist_backend      *be,
                                       enum geist_parallel_region region) {
    if (be == nullptr || be->state == nullptr) {
        return 0;
    }
    struct metal_state *st = be->state;
    if (st->sequence_active) {
        return 0; /* nested region: leave the outer batch in charge */
    }
    const enum geist_command_sequence_kind kind =
            region == GEIST_REGION_PREFILL_BATCH ? GEIST_COMMAND_SEQUENCE_PREFILL_TEXT
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

[[nodiscard]] static enum geist_status metal_resolve_weight(struct geist_backend *be,
                                                            struct geist_weight  *w) {
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
        .create                  = metal_create,
        .destroy                 = metal_destroy,
        .supports_op             = metal_supports_op,
        .buffer_create           = metal_buffer_create,
        .buffer_destroy          = metal_buffer_destroy,
        .buffer_create_aliased   = metal_buffer_create_aliased,
        .buffer_upload           = metal_buffer_upload,
        .buffer_download         = metal_buffer_download,
        .buffer_map              = metal_buffer_map,
        .buffer_unmap            = metal_buffer_unmap,
        .resolve_weight          = metal_resolve_weight,
        .rmsnorm                 = metal_rmsnorm,
        .add                     = metal_add,
        .mul                     = metal_mul,
        .gelu_tanh               = metal_gelu_tanh,
        .gelu_tanh_mul           = metal_gelu_tanh_mul,
        .gelu_tanh_mul_scaled    = nullptr,
        .relu_squared            = nullptr,
        .silu                    = nullptr,
        .rope_apply              = metal_rope_apply,
        .embedding_lookup        = metal_embedding_lookup,
        .attention               = metal_attention,
        .ffn_geglu_q4q6_mN       = nullptr,
        .transformer_block       = nullptr,
        .parallel_region_begin   = metal_parallel_region_begin,
        .parallel_region_end     = metal_parallel_region_end,
        .linear_t                = metal_linear_t,
        .buffer_copy             = metal_buffer_copy,
        .scale_f32               = metal_scale_f32,
        .embedding_lookup_scaled = metal_embedding_lookup_scaled,
        .rmsnorm_add             = metal_rmsnorm_add,
        .kv_append_f16           = metal_kv_append_f16,
        .ple_block               = metal_ple_block,
        .attn_qkv_prep           = metal_attn_qkv_prep,
        .ffn_gate_up             = metal_ffn_gate_up,
        .linear_t_pair           = metal_linear_t_pair,
        .argmax_f32              = metal_argmax_f32,
};

const struct geist_backend_descriptor geist_backend_metal = {
        .name   = "metal",
        .vtbl   = &metal_vtbl,
        .caps   = nullptr,
        .n_caps = 0,
};

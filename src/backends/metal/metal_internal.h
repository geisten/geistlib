/*
 * src/backends/metal/metal_internal.h — shared state, types, and
 * cross-module prototypes of the Metal backend.
 *
 * Layer: BACKEND (metal, internal). The backend is split by
 * responsibility: lifecycle.c, resources.c, pipelines.c, sequence.c,
 * ops.c, profiling.c. Everything here is internal to those six
 * translation units.
 */
#ifndef GEIST_INTERNAL_METAL_INTERNAL_H
#define GEIST_INTERNAL_METAL_INTERNAL_H
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
#include <stdatomic.h>
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
    METAL_PROFILE_DISPATCH_DELTANET_PREFILL,
    METAL_PROFILE_DISPATCH_DELTANET_DECODE,
    /* chunked-prefill sub-stages, individually skippable for the
     * subtractive profiler (all still count as DELTANET for the
     * category-level GEIST_SKIP_DELTANET). */
    METAL_PROFILE_DISPATCH_DN_PREP,  /* cst_copy + conv_prep + state_roll */
    METAL_PROFILE_DISPATCH_DN_NORM,  /* qk_norm */
    METAL_PROFILE_DISPATCH_DN_STAGE, /* dn_chunk_stage */
    METAL_PROFILE_DISPATCH_DN_SUBST, /* dn_chunk_subst */
    METAL_PROFILE_DISPATCH_DN_WIDE,  /* amat/vnew/out/supd/gate */
    METAL_PROFILE_DISPATCH_QGATE_SPLIT,
    METAL_PROFILE_DISPATCH_SIGMOID_MUL,
    METAL_PROFILE_STAGE_COUNT,
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
    void  *seq_ref[4096];
    size_t seq_ref_count;
    bool   seq_ref_overflow;
    void  *q4k_library;
    void  *q40_q80_library;
    void  *q5k_library;
    void  *q41_library;
    void  *quant_sg_library;
    void  *q4k_n4_library;
    void  *q6k_library;
    void  *q6k_n4_library;
    void  *elem_library;
    void  *silu_library;
    void  *deltanet_library;
    void  *qgate_library;
    void  *elem_simd_library;
    void  *attn_library;
    void  *attn_f16_library;
    void  *q_norm_rope_library;
    void  *k_norm_rope_append_library;
    void  *v_norm_append_library;
    void  *kv_norm_append_library;
    void  *kv_norm_append_f16_library;
    void  *q4k_function;
    void  *q4k_pipeline;
    void  *q40_function;
    void  *q40_pipeline;
    void  *q40_m8_function;
    void  *q40_m8_pipeline;
    void  *q80_function;
    void  *q80_pipeline;
    void  *q80_m8_function;
    void  *q80_m8_pipeline;
    void  *q5k_function;
    void  *q41_function;
    void  *q41_pipeline;
    void  *q41_m8_function;
    void  *q41_m8_pipeline;
    void  *q5k_pipeline;
    void  *q5k_m8_function;
    void  *q5k_m8_pipeline;
    void  *q40_n4_function;
    void  *q40_n4_pipeline;
    void  *q40_mm_function;
    void  *q40_mm_pipeline;
    void  *q80_n4_function;
    void  *q80_n4_pipeline;
    void  *q80_mm_function;
    void  *q80_mm_pipeline;
    void  *q41_n4_function;
    void  *q41_n4_pipeline;
    void  *q41_mm_function;
    void  *q41_mm_pipeline;
    void  *q5k_n4_function;
    void  *q5k_n4_pipeline;
    void  *q5k_mm_function;
    void  *q5k_mm_pipeline;
    void  *q40_mm_fast_function;
    void  *q40_mm_fast_pipeline;
    void  *q80_mm_fast_function;
    void  *q80_mm_fast_pipeline;
    void  *q41_mm_fast_function;
    void  *q41_mm_fast_pipeline;
    void  *q5k_mm_fast_function;
    void  *q5k_mm_fast_pipeline;
    void  *iq4nl_n4_function;
    void  *iq4nl_n4_pipeline;
    void  *iq4nl_mm_function;
    void  *iq4nl_mm_pipeline;
    void  *iq4xs_n4_function;
    void  *iq4xs_n4_pipeline;
    void  *iq4xs_mm_function;
    void  *iq4xs_mm_pipeline;
    void  *q3k_n4_function;
    void  *q3k_n4_pipeline;
    void  *q3k_mm_function;
    void  *q3k_mm_pipeline;
    void  *iq3s_n4_function;
    void  *iq3s_n4_pipeline;
    void  *iq3s_mm_function;
    void  *iq3s_mm_pipeline;
    void  *iq4xs_mm_fast_function;
    void  *iq4xs_mm_fast_pipeline;
    void  *q4k_n4_function;
    void  *q4k_n4_pipeline;
    void  *q4k_matmul_m8_function;
    void  *q4k_matmul_m8_pipeline;
    void  *q4k_m16_library;
    void  *q4k_matmul_m16_function;
    void  *q4k_matmul_m16_pipeline;
    void  *q4k_m16_n2_library;
    void  *q4k_matmul_m16_n2_function;
    void  *q4k_matmul_m16_n2_pipeline;
    void  *q4k_mm_sg_library;
    void  *q4k_mm_sg_function;
    void  *q4k_mm_sg_pipeline;
    void  *q4k_mm_sg_fast_library;
    void  *q4k_mm_sg_fast_function;
    void  *q4k_mm_sg_fast_pipeline;
    void  *q4k_qk_library;
    void  *q4k_qk_function;
    void  *q4k_qk_pipeline;
    void  *q4k_gate_up_library;
    void  *q4k_gate_up_n4_library;
    void  *q4k_pair_n4_library;
    void  *q4k_pair_n4_function;
    void  *q4k_pair_n4_pipeline;
    void  *embed_library;
    void  *argmax_library;
    void  *q4k_gate_up_function;
    void  *q4k_gate_up_pipeline;
    void  *q4k_gate_up_n4_function;
    void  *q4k_gate_up_n4_pipeline;
    void  *q6k_function;
    void  *q6k_pipeline;
    void  *q6k_n4_function;
    void  *q6k_n4_pipeline;
    void  *q6k_matmul_m8_function;
    void  *q6k_matmul_m8_pipeline;
    void  *q6k_mm_sg_library;
    void  *q6k_matmul_sg_function;
    void  *q6k_matmul_sg_pipeline;
    void  *q6k_mm_sg_fast_library;
    void  *q6k_matmul_sg_fast_function;
    void  *q6k_matmul_sg_fast_pipeline;
    void  *q6k_m16_library;
    void  *q6k_matmul_m16_function;
    void  *q6k_matmul_m16_pipeline;
    void  *rmsnorm_rows_function;
    void  *rmsnorm_rows_pipeline;
    void  *rmsnorm_rows_simd_function;
    void  *rmsnorm_rows_simd_pipeline;
    void  *gelu_rows_function;
    void  *gelu_rows_pipeline;
    void  *silu_rows_function;
    void  *silu_rows_pipeline;
    void  *deltanet_mix_function;
    void  *deltanet_mix_pipeline;
    void  *dn_cst_copy_function;
    void  *dn_cst_copy_pipeline;
    void  *dn_conv_prep_function;
    void  *dn_conv_prep_pipeline;
    void  *dn_qk_norm_function;
    void  *dn_qk_norm_pipeline;
    void  *dn_state_roll_function;
    void  *dn_state_roll_pipeline;
    void  *dn_chunk_stage_function;
    void  *dn_chunk_stage_pipeline;
    void  *dn_chunk_subst_function;
    void  *dn_chunk_subst_pipeline;
    void  *dn_chunk_amat_function;
    void  *dn_chunk_amat_pipeline;
    void  *dn_chunk_vnew1_function;
    void  *dn_chunk_vnew1_pipeline;
    void  *dn_chunk_vnew2_function;
    void  *dn_chunk_vnew2_pipeline;
    void  *dn_chunk_out_function;
    void  *dn_chunk_out_pipeline;
    void  *dn_chunk_supd_function;
    void  *dn_chunk_supd_pipeline;
    void  *dn_chunk_gate_function;
    void  *dn_chunk_gate_pipeline;
    /* Grow-only private MTLBuffer scratch for the chunked DeltaNet
     * prefill (raw handle, argmax_result_buffer pattern). */
    void    *dn_scratch;
    size_t   dn_scratch_bytes;
    bool     use_dn_chunk;
    void    *qgate_split_function;
    void    *qgate_split_pipeline;
    void    *sigmoid_mul_function;
    void    *sigmoid_mul_pipeline;
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
    METAL_Q40_Q80_BLOCK_ELEMS           = 32u,
    METAL_Q40_BLOCK_BYTES               = 18u,
    METAL_Q41_BLOCK_BYTES               = 20u,
    METAL_Q80_BLOCK_BYTES               = 34u,
    METAL_IQ4NL_BLOCK_BYTES             = 18u,
    METAL_IQ4XS_BLOCK_ELEMS             = 256u,
    METAL_IQ4XS_BLOCK_BYTES             = 136u,
    METAL_Q3K_BLOCK_BYTES               = 110u,
    METAL_IQ3S_BLOCK_BYTES              = 110u,
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

struct metal_deltanet_params {
    uint32_t seq;
    uint32_t n_k_heads;
    uint32_t n_v_heads;
    uint32_t head_k;
    uint32_t head_v;
    uint32_t conv_kernel;
    uint32_t qkv_offset;
    uint32_t z_offset;
    uint32_t beta_offset;
    uint32_t alpha_offset;
    uint32_t conv_w_offset;
    uint32_t ssm_a_offset;
    uint32_t dt_bias_offset;
    uint32_t norm_w_offset;
    uint32_t conv_state_offset;
    uint32_t delta_state_offset;
    float    eps;
};

struct metal_qgate_params {
    uint32_t rows;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t joint_offset;
    uint32_t q_offset;
    uint32_t gate_offset;
    uint32_t joint_row_stride;
    uint32_t q_row_stride;
    uint32_t gate_row_stride;
};
/* ---- Shared data (defined in profiling.c) ----------------------------- */
extern const char *const metal_profile_stage_names[METAL_PROFILE_STAGE_COUNT];

/* ---- Cross-module prototypes ------------------------------------------ */
void metal_destroy(struct geist_backend *be);

[[nodiscard]] enum geist_status metal_create(struct geist_backend            *be,
                                             const struct geist_backend_opts *opts);

int metal_parallel_region_begin(struct geist_backend *be, enum geist_parallel_region region);

void metal_parallel_region_end(struct geist_backend *be, int token);

struct geist_buffer *metal_buf_reg_find(struct metal_state *st, const void *p, size_t *out_off);

[[nodiscard]] enum geist_status metal_new_buffer(struct geist_backend  *be,
                                                 size_t                 bytes,
                                                 enum geist_buffer_role role,
                                                 unsigned int           memory_flags,
                                                 bool                   host_visible,
                                                 struct geist_buffer  **out);

void metal_buffer_destroy_internal(struct geist_backend *be, struct geist_buffer *buf);

[[nodiscard]] enum geist_status metal_buffer_create(struct geist_backend  *be,
                                                    size_t                 bytes,
                                                    enum geist_buffer_role role,
                                                    unsigned int           memory_flags,
                                                    struct geist_buffer  **out);

[[nodiscard]] enum geist_status metal_buffer_create_aliased(struct geist_backend  *be,
                                                            void                  *host_ptr,
                                                            size_t                 n_bytes,
                                                            enum geist_buffer_role role,
                                                            struct geist_buffer  **out);

void metal_buffer_destroy(struct geist_backend *be, struct geist_buffer *buf);

[[nodiscard]] enum geist_status metal_buffer_copy(struct geist_buffer       *dst,
                                                  size_t                     dst_offset,
                                                  const struct geist_buffer *src,
                                                  size_t                     src_offset,
                                                  size_t                     n_bytes);

[[nodiscard]] enum geist_status
metal_buffer_upload(struct geist_buffer *buf, size_t n_bytes, const uint8_t *src);

[[nodiscard]] enum geist_status
metal_buffer_download(size_t n_bytes, uint8_t *dst, const struct geist_buffer *buf);

void *metal_buffer_map(struct geist_buffer *buf);

void metal_buffer_unmap(struct geist_buffer *buf);

bool metal_tensor_is_f32_vector(const struct geist_tensor *t,
                                size_t                    *out_n,
                                size_t                    *out_offset_floats);

bool metal_tensor_is_f32_matrix(const struct geist_tensor *t,
                                size_t                    *out_rows,
                                size_t                    *out_cols,
                                size_t                    *out_offset_floats,
                                size_t                    *out_row_stride);

bool metal_tensor_is_f32_rows(const struct geist_tensor *t,
                              size_t                    *out_rows,
                              size_t                    *out_cols,
                              size_t                    *out_offset_floats,
                              size_t                    *out_row_stride);

bool metal_tensor_is_q4k_matrix(const struct geist_tensor *t,
                                size_t                    *out_rows,
                                size_t                    *out_cols,
                                size_t                    *out_offset_bytes);

bool metal_tensor_is_q6k_matrix(const struct geist_tensor *t,
                                size_t                    *out_rows,
                                size_t                    *out_cols,
                                size_t                    *out_offset_bytes);
bool metal_tensor_is_q5k_matrix(const struct geist_tensor *t,
                                size_t                    *out_rows,
                                size_t                    *out_cols,
                                size_t                    *out_offset_bytes);

bool metal_tensor_is_q40_q80_matrix(const struct geist_tensor *t,
                                    enum geist_dtype           dtype,
                                    size_t                    *out_rows,
                                    size_t                    *out_cols,
                                    size_t                    *out_offset_bytes);

bool metal_tensor_is_f32_3d(const struct geist_tensor *t,
                            size_t                    *out_d0,
                            size_t                    *out_d1,
                            size_t                    *out_d2,
                            size_t                    *out_offset_floats);

bool metal_tensor_is_f16_3d(const struct geist_tensor *t,
                            size_t                    *out_d0,
                            size_t                    *out_d1,
                            size_t                    *out_d2,
                            size_t                    *out_offset_halfs);

[[nodiscard]] enum geist_status metal_ensure_q4k_pipeline(struct geist_backend *be);

[[nodiscard]] enum geist_status metal_ensure_attention_pipeline(struct geist_backend *be);

[[nodiscard]] enum geist_status metal_ensure_argmax_pipeline(struct geist_backend *be);

[[nodiscard]] enum geist_status metal_ensure_deltanet_pipeline(struct geist_backend *be);

bool metal_ranges_overlap(size_t a_offset, size_t b_offset, size_t n_bytes);

void *metal_sequence_encoder(struct metal_state *st);

void metal_release_sequence_objects(struct metal_state *st);

void metal_seq_ref_clear(struct metal_state *st);

bool metal_seq_references(struct metal_state *st, const void *mtl_buf);

void metal_batch_flush(struct metal_state *st);

void metal_flush_if_referenced(struct metal_state *st, const void *mtl_buf);

[[nodiscard]] enum geist_status metal_command_sequence_begin(struct geist_backend            *be,
                                                             enum geist_command_sequence_kind kind,
                                                             int *out_token);

[[nodiscard]] enum geist_status
metal_command_sequence_end(struct geist_backend *be, int token, bool submit);

#include "metal_shaders.h"
#include "metal_objc.h"

bool metal_env_enabled(const char *name);

bool metal_env_disabled(const char *name);

uint64_t metal_now_ns(void);

void metal_profile_add_wait(struct metal_state      *st,
                            enum metal_profile_stage stage,
                            uint64_t                 start_ns);

void metal_profile_add_dispatch(struct metal_state      *st,
                                enum metal_profile_stage stage,
                                struct metal_size        groups);

enum metal_profile_stage
metal_profile_wait_stage_for_sequence(enum geist_command_sequence_kind kind);

void metal_profile_print_summary(const struct metal_state *st);

void metal_capture_begin(struct metal_state *st, enum geist_command_sequence_kind kind);

void metal_capture_end(struct metal_state *st);

void metal_linear_debug_stats(const float               *x,
                              size_t                     nx,
                              const float               *y,
                              size_t                     ny,
                              const struct geist_weight *w,
                              size_t                     m);

#endif /* GEIST_INTERNAL_METAL_INTERNAL_H */

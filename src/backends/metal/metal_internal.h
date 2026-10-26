/*
 * src/backends/metal/metal_internal.h — shared state, types, and
 * cross-module prototypes of the Metal backend.
 *
 * Layer: BACKEND (metal, internal). Shared by the backend's translation
 * units (lifecycle, resources, pipelines, sequence, ops, profiling, tuning).
 */
#ifndef GEIST_INTERNAL_METAL_INTERNAL_H
#define GEIST_INTERNAL_METAL_INTERNAL_H
#define GEIST_INTERNAL_BACKEND_LAYER

#include <geist.h>
#include <geist_backend.h>
#include "quant.h"
#include "heap.h"
#include <math.h>

/* Which GPU command sequence a capture/profile wait belongs to. */
enum geist_command_sequence_kind {
    GEIST_COMMAND_SEQUENCE_DECODE_LAYER_LOOP,
    GEIST_COMMAND_SEQUENCE_DECODE_GREEDY_STEP,
    GEIST_COMMAND_SEQUENCE_PREFILL_TEXT,
    GEIST_COMMAND_SEQUENCE_VERIFY_GREEDY,
};

#include <dlfcn.h>
#include <errno.h>
#include <stdalign.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <limits.h>
#include <pthread.h>
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
    METAL_PROFILE_DISPATCH_Q4K_LINEAR_MM_FAST,
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
    METAL_PROFILE_DISPATCH_QKV3,
    METAL_PROFILE_DISPATCH_Q80_GATE_UP,
    METAL_PROFILE_DISPATCH_F32_PLE_PROJ_NORM,
    METAL_PROFILE_DISPATCH_RMSNORM_ADD_ROWS,
    METAL_PROFILE_DISPATCH_ADD_NORM_ROWS,
    METAL_PROFILE_DISPATCH_Q_NORM_ROPE,
    METAL_PROFILE_DISPATCH_K_NORM_ROPE_APPEND,
    METAL_PROFILE_DISPATCH_V_NORM_APPEND,
    METAL_PROFILE_DISPATCH_KV_NORM_APPEND,
    METAL_PROFILE_DISPATCH_ROPE_APPEND,
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
    METAL_PROFILE_DISPATCH_DN_PREP,  /* cst_copy + conv + norm + roll */
    METAL_PROFILE_DISPATCH_DN_CHUNK, /* dnf_chunk */
    METAL_PROFILE_DISPATCH_DN_STATE, /* dnf_state */
    METAL_PROFILE_DISPATCH_DN_OUT,   /* dnf_out */
    METAL_PROFILE_DISPATCH_QGATE_SPLIT,
    METAL_PROFILE_DISPATCH_SIGMOID_MUL,
    METAL_PROFILE_DISPATCH_HADAMARD,
    METAL_PROFILE_DISPATCH_PQ2_X_F16,
    METAL_PROFILE_DISPATCH_PQ2_M8,
    METAL_PROFILE_STAGE_COUNT,
};

struct metal_profile_stat {
    uint64_t ns;
    uint64_t calls;
    uint64_t workgroups;
};

/* GPU time of the command buffers a profiled run committed, keyed by the
 * set of dispatch stages each buffer carried (a bit per metal_profile_stage).
 * With GEIST_METAL_PIPELINE=1 every op gets its own buffer, so the keys are
 * single stages (or one op's sub-stages) and the times a per-stage split. */
struct metal_profile_gpu_slot {
    uint64_t mask;
    uint64_t buffers;
    double   seconds;
};

/* Registry entry mapping a live buffer's host contents range back to its
 * geist_buffer, so resolver-installed linear kernels can translate the raw
 * host pointers the engine passes (buffer_map aliases, w->raw) into
 * (MTLBuffer, offset) pairs for GPU dispatch. */
struct metal_buf_reg_entry {
    const uint8_t       *base;
    size_t               bytes;
    struct geist_buffer *buf;
};

/* Head-dim slots of the prefill flash kernels: hd/32 - 1 for hd up to 512. */
enum { METAL_FLASH_SLOTS = 16 };

struct metal_state {
    struct geist_backend       *backend;
    void                       *metal_handle;
    void                       *objc_handle;
    void                       *device;
    void                       *command_queue;
    struct metal_buf_reg_entry *buf_reg;
    size_t                      buf_reg_count;
    size_t                      buf_reg_cap;
    /* Buffers bound by ops encoded on the open (unflushed) batch, as
     * (MTLBuffer, bind offset) pairs; a host map/upload/download of a
     * range holding a bind offset forces a flush. Every handle binds at
     * its base_off, so the offset tells views of one MTLBuffer apart
     * (#528). */
    struct metal_seq_ref {
        const void *buf;
        size_t      off;
    } seq_ref[4096];
    uint16_t seq_ref_used[4096];
    /* seq_ref_used[0 .. seq_ref_count) are the occupied slots, in insertion
     * order. Open-addressed set; overflow degrades to always-flush. */
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
    void  *rope_append_library; /* Llama/Qwen3 q/k prep + KV append (#17) */
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
    void  *pq2_n4_function;
    void  *pq2_n4_pipeline;
    void  *pq2_mm_function;
    void  *pq2_mm_pipeline;
    void  *pq2_mm_fast_function;
    void  *pq2_mm_fast_pipeline;
    void  *pq2_m8_function;
    void  *pq2_m8_pipeline;
    /* x -> f16 for pq2_mm_fast (cvt_f32_f16_rows), into pq2_xh_buffer:
     * grow-only, rows x n_in halves of the widest call so far. */
    void                *cvt_f16_function;
    void                *cvt_f16_pipeline;
    void                *cvt_kmaj_function;
    void                *cvt_kmaj_pipeline;
    struct geist_buffer *pq2_xh_buffer;
    size_t               pq2_xh_capacity;
    /* Device-dependent crossovers; see tuning.c. Seeded from M1 Max
     * measurements, then a calibration blob, then the env. */
    struct metal_tuning {
        uint32_t pq2_n8_min_n_out;
        /* Set once the calibration blob has been folded in; see
         * metal_tuning_resolve. */
        bool resolved;
    } tuning;
    void *pq2_n8_function;
    void *pq2_n8_pipeline;
    void *iq4nl_n4_function;
    void *iq4nl_n4_pipeline;
    void *iq4nl_mm_function;
    void *iq4nl_mm_pipeline;
    void *iq4xs_n4_function;
    void *iq4xs_n4_pipeline;
    void *iq4xs_mm_function;
    void *iq4xs_mm_pipeline;
    void *q3k_n4_function;
    void *q3k_n4_pipeline;
    void *q3k_mm_function;
    void *q3k_mm_pipeline;
    void *tq2_n4_function;
    void *tq2_n4_pipeline;
    void *tq2_mm_function;
    void *tq2_mm_pipeline;
    void *i2s_n4_function;
    void *i2s_n4_pipeline;
    void *i2s_mm_function;
    void *i2s_mm_pipeline;
    void *iq3s_n4_function;
    void *iq3s_n4_pipeline;
    void *iq3s_mm_function;
    void *iq3s_mm_pipeline;
    void *iq4xs_mm_fast_function;
    void *iq4xs_mm_fast_pipeline;
    void *q4k_n4_function;
    void *q4k_n4_pipeline;
    void *q4k_h_function;
    void *q4k_h_pipeline;
    void *q4k_matmul_m8_function;
    void *q4k_matmul_m8_pipeline;
    void *q4k_m16_library;
    void *q4k_matmul_m16_function;
    void *q4k_matmul_m16_pipeline;
    void *q4k_mm_sg_library;
    void *q4k_mm_sg_function;
    void *q4k_mm_sg_pipeline;
    void *q4k_mm_sg_fast_library;
    void *q4k_mm_sg_fast_function;
    void *q4k_mm_sg_fast_pipeline;
    void *q6k_mm32_library; /* matmul_q6k_mm_sg_fast (#17) */
    void *q6k_mm32_function;
    void *q6k_mm32_pipeline;
    void *q4k_gate_up_n4_library;
    void *q4k_pair_n4_library;
    void *q4k_pair_n4_function;
    void *q4k_pair_n4_pipeline;
    /* gate_up_silu_q4k_n4 and the Q8_0 gate/up kernels live in
     * q4k_gate_up_n4_library (#17). */
    void *q4k_gate_up_silu_n4_function;
    void *q4k_gate_up_silu_n4_pipeline;
    void *q80_gate_up_n4_function;
    void *q80_gate_up_n4_pipeline;
    void *q80_gate_up_silu_n4_function;
    void *q80_gate_up_silu_n4_pipeline;
    /* q/k/v decode matvecs in one dispatch, one kernel per dtype triple (#17) */
    void *mv3_library;
    void *mv3_q4k_q4k_q4k_function;
    void *mv3_q4k_q4k_q4k_pipeline;
    void *mv3_q4k_q4k_q6k_function;
    void *mv3_q4k_q4k_q6k_pipeline;
    void *mv3_q80_q80_q80_function;
    void *mv3_q80_q80_q80_pipeline;
    void *mv3_q4k_h_q4k_q4k_function;
    void *mv3_q4k_h_q4k_q4k_pipeline;
    void *mv3_q4k_h_q4k_q6k_function;
    void *mv3_q4k_h_q4k_q6k_pipeline;
    void *embed_library;
    void *argmax_library;
    void *moe_library; /* #201: routing and id-indexed expert matvecs */
    void *hadamard_library;
    void *hadamard_function;
    void *hadamard_pipeline;
    void *hadamard4_function; /* hadamard_rows4: blocks of 128..4096 */
    void *hadamard4_pipeline;
    void *q4k_gate_up_n4_function;
    void *q4k_gate_up_n4_pipeline;
    void *q4k_gate_up_h_function;
    void *q4k_gate_up_h_pipeline;
    void *q4k_gate_up_silu_h_function;
    void *q4k_gate_up_silu_h_pipeline;
    void *q6k_function;
    void *q6k_pipeline;
    void *q6k_n4_function;
    void *q6k_n4_pipeline;
    void *q6k_mm_sg_library;
    void *q6k_matmul_sg_function;
    void *q6k_matmul_sg_pipeline;
    void *q6k_mm_sg_fast_library;
    void *q6k_matmul_sg_fast_function;
    void *q6k_matmul_sg_fast_pipeline;
    void *rmsnorm_rows_simd_function;
    void *rmsnorm_rows_simd_pipeline;
    void *gelu_rows_function;
    void *gelu_rows_pipeline;
    void *silu_rows_function;
    void *silu_rows_pipeline;
    void *relu2_rows_function;
    void *relu2_rows_pipeline;
    void *silu_mul_rows_function;
    void *silu_mul_rows_pipeline;
    void *deltanet_mix_function;
    void *deltanet_mix_pipeline;
    void *dn_dec_qk_function;
    void *dn_dec_qk_pipeline;
    void *dn_dec_v_function;
    void *dn_dec_v_pipeline;
    void *dn_dec_v_rows_function;
    void *dn_dec_v_rows_pipeline;
    void *dnf_cst_copy_function;
    void *dnf_cst_copy_pipeline;
    void *dnf_conv_function;
    void *dnf_conv_pipeline;
    void *dnf_norm_function;
    void *dnf_norm_pipeline;
    void *dnf_roll_function;
    void *dnf_roll_pipeline;
    void *dnf_chunk_function;
    void *dnf_chunk_pipeline;
    void *dnf_state_function;
    void *dnf_state_pipeline;
    void *dnf_out_function;
    void *dnf_out_pipeline;
    /* Grow-only private MTLBuffer scratch for the chunked DeltaNet
     * prefill (raw handle, argmax_result_buffer pattern). */
    void    *dn_scratch;
    size_t   dn_scratch_bytes;
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
    void    *rmsnorm_add_rows_simd_function;
    void    *rmsnorm_add_rows_simd_pipeline;
    void    *rmsnorm_add_norm_rows_simd_function;
    void    *rmsnorm_add_norm_rows_simd_pipeline;
    void    *embed_lookup_scaled_function;
    void    *embed_lookup_scaled_pipeline;
    void    *embed_lookup_scaled_rows_function;
    void    *embed_lookup_scaled_rows_pipeline;
    void    *f32_library;
    void    *f16w_matmul_function;
    void    *f16w_matmul_pipeline;
    void    *f16w_matmul_sg_function;
    void    *f16w_matmul_sg_pipeline;
    void    *bf16w_matmul_function;
    void    *bf16w_matmul_pipeline;
    void    *bf16w_matmul_sg_function;
    void    *bf16w_matmul_sg_pipeline;
    void    *f32_matmul_function;
    void    *f32_matmul_pipeline;
    void    *f32_matmul_sg_function;
    void    *f32_matmul_sg_pipeline;
    void    *f32_matmul_mm_function;
    void    *f32_matmul_mm_pipeline;
    void    *f32_ple_gate_function;
    void    *f32_ple_gate_pipeline;
    void    *f32_ple_gate_sg_function; /* decode, a simdgroup per row (#196) */
    void    *f32_ple_gate_sg_pipeline;
    void    *f32_matvec_rows_function;
    void    *f32_matvec_rows_pipeline;
    void    *f32_ple_proj_norm_function;
    void    *f32_ple_proj_norm_pipeline;
    void    *argmax_batch_function;
    void    *argmax_batch_pipeline;
    void    *argmax_part_function;
    void    *argmax_part_pipeline;
    void    *argmax_merge_function;
    void    *argmax_merge_pipeline;
    void    *moe_topk_function;
    void    *moe_topk_pipeline;
    void    *moe_mv_q4k_function;
    void    *moe_mv_q4k_pipeline;
    void    *moe_mv_q6k_function;
    void    *moe_mv_q6k_pipeline;
    void    *moe_mv_q80_function;
    void    *moe_mv_q80_pipeline;
    void    *moe_mv_f32_function;
    void    *moe_mv_f32_pipeline;
    void    *moe_wsum_function;
    void    *moe_wsum_pipeline;
    void    *argmax_result_buffer;
    void    *argmax_result_mapped;
    uint32_t argmax_result_capacity;
    void    *argmax_part_buffer; /* METAL_ARGMAX_PARTS (value, index) pairs, GPU only */
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
    void    *rope_append_rows_function;
    void    *rope_append_rows_pipeline;
    void    *rope_append_rows_f16_function;
    void    *rope_append_rows_f16_pipeline;
    void    *attention_rows_function;
    void    *attention_rows_pipeline;
    void    *attention_rows_f16_function;
    void    *attention_rows_f16_pipeline;
    void    *attn_qnorm_dec_f16_library;
    void    *attention_qnorm_dec_f16_function;
    void    *attention_qnorm_dec_f16_pipeline;
    void    *attention_dec_f16_function;
    void    *attention_dec_f16_pipeline;
    void    *attention_dec512_f16_function;
    void    *attention_dec512_f16_pipeline;
    void    *attn_dec512_f16_library;
    /* The prefill flash kernels (#91), one per head_dim: slot hd/32 - 1
     * holds head_dim hd (multiples of 32 up to 256, of 64 up to 512; the
     * other slots stay nullptr). See metal_flash_pipeline. */
    void *attn_flash_library;
    void *attention_flash_function[METAL_FLASH_SLOTS];
    void *attention_flash_pipeline[METAL_FLASH_SLOTS];
    /* The split-KV decode kernels on an F32 KV cache (#64). */
    void *attn_dec_f32kv_library;
    void *attention_dec_f32kv_function;
    void *attention_dec_f32kv_pipeline;
    void *attention_dec512_f32kv_function;
    void *attention_dec512_f32kv_pipeline;
    /* persistent f32->f16 K/V staging for the plain flash (prefill)
     * path on an F32 KV cache; decode reads the F32 cache directly. */
    struct geist_buffer *attn_kf16_buffer;
    struct geist_buffer *attn_vf16_buffer;
    size_t               attn_kvf16_capacity;
    /* The 32-key-block split-KV decode kernels (#17), F16 and F32 KV, in
     * the prefill flash kernels' head_dim slots (64, 128, 256 and 512
     * built; see metal_decv_pipeline). */
    void *attn_decv_library;
    void *attention_decv_f16_function[METAL_FLASH_SLOTS];
    void *attention_decv_f16_pipeline[METAL_FLASH_SLOTS];
    void *attention_decv_f32kv_function[METAL_FLASH_SLOTS];
    void *attention_decv_f32kv_pipeline[METAL_FLASH_SLOTS];
    /* attention_decq (#196): 128-thread blocks, head_dim 128/256/512. */
    void                *attention_decq_f16_function[METAL_FLASH_SLOTS];
    void                *attention_decq_f16_pipeline[METAL_FLASH_SLOTS];
    void                *attention_decq_f32kv_function[METAL_FLASH_SLOTS];
    void                *attention_decq_f32kv_pipeline[METAL_FLASH_SLOTS];
    void                *attn_dec_combine_library;
    void                *attention_dec_combine_function;
    void                *attention_dec_combine_pipeline;
    struct geist_buffer *attn_dec_partials_buffer;
    size_t               attn_dec_partials_capacity;
    void                *sequence_command_buffer;
    void                *sequence_compute_encoder;
    void                *sequence_pool; /* autorelease pool, sequence begin..end */
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
     * so cross-buffer data deps hold. 0 = pipelining off. Prefill
     * sequences rotate every seq_rotate_prefill dispatches instead. */
    uint32_t                         seq_rotate_every;
    uint32_t                         seq_rotate_prefill;
    uint32_t                         seq_disp_at_rotate;
    uint32_t                         seq_pending_count;
    void                            *seq_pending_cmds[16];
    uint64_t                         seq_pending_mask[16]; /* profile: stages per buffer */
    uint64_t                         seq_cur_mask;         /* profile: stages so far */
    uint64_t                         seq_begin_ns;
    int                              sequence_token;
    enum geist_command_sequence_kind sequence_kind;
    bool                             sequence_active;
    /* One command sequence per backend, shared by every session of the
     * model (#544). seq_lock (recursive) is held by the thread that opened
     * the sequence from region begin to region end, and by each standalone
     * submission, so sessions on other threads wait instead of encoding
     * into a sequence they do not own. seq_owner is that thread while the
     * sequence is open, else 0; only the owner writes its own id, so a
     * thread reading its own id back is never wrong (metal_seq_mine). */
    pthread_mutex_t           seq_lock;
    _Atomic(uintptr_t)        seq_owner;
    bool                      sequence_has_work;
    bool                      profile_enabled;
    bool                      skip_next_dispatch; /* subtractive profiler: drop the next dispatch */
    struct metal_profile_stat profile[METAL_PROFILE_STAGE_COUNT];
    /* GPU time of committed command buffers (profile only): the sum of the
     * buffers' own times, and their union on the GPU timeline (busy). */
    double                        gpu_sum_s;
    double                        gpu_busy_s;
    double                        gpu_last_end_s;
    uint64_t                      gpu_buffers;
    struct metal_profile_gpu_slot gpu_slots[64];
    char                          device_name[128];

    /* #530: an MTLResidencySet on the command queue (macOS 15+; null there
     * otherwise, or with GEIST_METAL_KEEP_ALIVE_S=0). macOS unwires what a
     * command buffer made resident about 2 s after the GPU goes idle, and
     * the next submission wires it again, stalling the first token after a
     * pause. res_thread requests the set's residency every
     * 500 ms until res_keep_s pass without a dispatch (res_used), which
     * keeps the members wired across such pauses; then it sleeps
     * (res_parked) until the next dispatch.
     * A buffer joins the set the first time a dispatch binds it, so what
     * stays wired is what the GPU uses and never a table the host gathers
     * (#529). res_bufs holds the members: open addressing, linear probing,
     * backward-shift deletion; at half full it stops adding. A join marks
     * res_dirty and res_thread commits it with its next request; a leave
     * commits at once, because the set keeps its members alive until then.
     * res_lock guards the set's calls and res_dirty against res_thread. */
    void           *residency_set;
    void           *res_bufs[8192];
    size_t          res_count;
    bool            res_dirty;
    bool            res_stop;
    atomic_bool     res_used;
    atomic_bool     res_parked;
    uint64_t        res_keep_s;
    pthread_t       res_thread;
    pthread_mutex_t res_lock;
    pthread_cond_t  res_wake;

    /* #531: the device's recommendedMaxWorkingSetSize, read once at create.
     * metal_budget_admit refuses an allocation past it unless
     * GEIST_METAL_IGNORE_BUDGET=1 (ws_ignore). ws_warned: the 90 % notice is
     * printed once. */
    size_t ws_budget;
    bool   ws_ignore;
    bool   ws_warned;

    void *MTLCreateSystemDefaultDevice;
    void *objc_msgSend;
    void *sel_registerName;
    void *objc_getClass;
    void *objc_autoreleasePoolPush;
    void *objc_autoreleasePoolPop;
};

struct geist_buffer {
    struct metal_state *owner;
    void               *buffer;
    void               *mapped;
    size_t              bytes;
    /* Where this buffer's bytes start inside `buffer`. Non-zero only for a
     * zero-copy alias: of the loader's mmap (#357), where the MTLBuffer
     * wraps the enclosing page range, or of another Metal buffer (#528),
     * whose MTLBuffer the view shares. Every bind and blit adds it;
     * `mapped` already has it applied. */
    size_t                 base_off;
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

/* Elements per raw block of `dtype` (quant_block_layout); 1 for a dtype
 * without a fixed layout. */
static inline size_t metal_quant_block_elems(enum geist_dtype dtype) {
    size_t elems = 1, bytes = 0, tail = 0;
    return quant_block_layout(dtype, &elems, &bytes, &tail) ? elems : 1u;
}

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

/* Field order matches Post2 in metal_elem_simd_source. */
struct metal_post_norm2_params {
    uint32_t rows;
    uint32_t cols;
    uint32_t residual_offset;
    uint32_t x_offset;
    uint32_t w_offset;
    uint32_t zw_offset;
    uint32_t y_offset;
    uint32_t z_offset;
    uint32_t residual_row_stride;
    uint32_t x_row_stride;
    uint32_t y_row_stride;
    uint32_t z_row_stride;
    uint32_t has_w;
    float    eps;
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

/* Threadgroups of the decode argmax's first pass (argmax_f32_part). */
enum { METAL_ARGMAX_PARTS = 128 };

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
    uint32_t row_cs_offset; /* per-row state slots (#192), row_slots 0 = none */
    uint32_t row_s_offset;
    uint32_t row_slots;
    uint32_t row; /* dn_dec_qk: the row this dispatch runs */
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

void metal_residency_create(struct metal_state *st, uint64_t keep_alive_s);

void metal_residency_destroy(struct metal_state *st);

void metal_residency_forget(struct metal_state *st, void *mtl_buf);

/* tuning.c: fold an applied calibration blob into the crossovers, once,
 * at the first weight resolve -- the engine closes the apply window just
 * before calling us, so this is the earliest point a blob is visible.
 * A no-op on every call after the first. */
void metal_tuning_resolve(struct geist_backend *be, struct metal_state *st);

/* tuning.c: resolve the device crossovers (seed -> calibration -> env)
 * and the tunable table the calibration driver measures. */
void                        metal_tuning_init(struct geist_backend *be, struct metal_state *st);
const struct geist_tunable *metal_tunables(size_t *out_count);

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
metal_buffer_upload(struct geist_buffer *buf, size_t n_bytes, const uint8_t src[static n_bytes]);

[[nodiscard]] enum geist_status
metal_buffer_download(size_t n_bytes, uint8_t dst[static n_bytes], const struct geist_buffer *buf);

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
[[nodiscard]] enum geist_status metal_ensure_moe_pipeline(struct geist_backend *be);
[[nodiscard]] enum geist_status metal_ensure_hadamard_pipeline(struct geist_backend *be);

[[nodiscard]] enum geist_status metal_ensure_deltanet_pipeline(struct geist_backend *be);

/* Releases every library, pipeline and function the ensure_* tables in
 * pipelines.c build. */
void metal_release_pipeline_tables(struct metal_state *st);

bool metal_ranges_overlap(size_t a_offset, size_t b_offset, size_t n_bytes);

void *metal_sequence_encoder(struct metal_state *st);

void metal_release_sequence_objects(struct metal_state *st);

void metal_seq_ref_clear(struct metal_state *st);

/* Could the open batch touch bytes [off, off + n) of mtl_buf? n = SIZE_MAX
 * asks about the whole buffer. */
bool metal_seq_references(struct metal_state *st, const void *mtl_buf, size_t off, size_t n);

void metal_batch_flush(struct metal_state *st);

void metal_flush_if_referenced(struct metal_state *st, const void *mtl_buf, size_t off, size_t n);

[[nodiscard]] enum geist_status metal_command_sequence_begin(struct geist_backend            *be,
                                                             enum geist_command_sequence_kind kind,
                                                             int *out_token);

[[nodiscard]] enum geist_status
metal_command_sequence_end(struct geist_backend *be, int token, bool submit);

#include "metal_shaders.h"
#include "metal_objc.h"

bool metal_env_enabled(const char *name);

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
/* Profile: account a completed command buffer's GPU time to the stages it
 * carried. No-op unless GEIST_METAL_PROFILE is set. */
void metal_profile_add_cmd_buffer(struct metal_state *st, void *cmd, uint64_t stage_mask);

void metal_capture_begin(struct metal_state *st, enum geist_command_sequence_kind kind);

void metal_capture_end(struct metal_state *st);

#endif /* GEIST_INTERNAL_METAL_INTERNAL_H */

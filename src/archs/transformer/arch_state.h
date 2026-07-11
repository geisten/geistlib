/*
 * src/archs/transformer/arch_state.h — Gemma 4 transformer state built
 * entirely on backend buffers (no `LM*` legacy delegation).
 *
 * Layer: ARCHITECTURE.
 *
 * Phase B-4e sub-step 1 (this file): weight loading. Reads every per-layer
 * + global tensor from the GGUF and stages the raw bytes into backend-owned
 * geist_buffer*'s. No forward pass yet — that lands in sub-steps 2 and 3.
 *
 * Why a v2 path: the legacy transformer_arch_state in arch.c delegates to
 * lm.c's `LM*`. Sub-step 1 stands up the replacement plumbing alongside it
 * (live arch_ops vtable still points at the legacy path); once sub-steps
 * 2..4 land and verify byte-identical token output, arch.c will be flipped
 * over and `LM*` retired.
 *
 * Gemma 4 specifics encoded here:
 *   - 35 layers, hidden 1536, vocab 262144, n_q_heads 8, n_kv_heads 1.
 *   - Alternating sliding (256 head_dim, window 512) / full (512 head_dim,
 *     unbounded) attention in a 4+1 pattern.
 *   - Per-Layer Embeddings (PLE): every layer adds a gated contribution
 *     from a per-token 256-dim vector projected from a Q-quantized table.
 *   - KV-cache sharing: layers 15..34 reuse the K/V from layers 13/14
 *     (sliding) and 14 (full) — they hold no own K/V projections.
 *   - Per-layer output scalar (layer_output_scale) and softcap on logits.
 */
#ifndef GEIST_INTERNAL_ARCH_TRANSFORMER_STATE_V2_H
#define GEIST_INTERNAL_ARCH_TRANSFORMER_STATE_V2_H

#ifndef GEIST_INTERNAL_ARCH_LAYER
#error "transformer/arch_state.h is internal to the architecture layer."
#endif

#define GEIST_INTERNAL_ENGINE_LAYER

#include <geist.h>
#include <geist_types.h>
#include <geist_weight.h>

#include "arch_config.h"
#include "arena.h"
#include "exec_plan.h"
#include "sampler.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* ---- Gemma 4 structural dimensions (E2B variant) ---------------------- *
 *
 * These remain compile-time macros because they size fixed-length arrays
 * in struct transformer_arch_state (k_cache[NUM_LAYERS], scratch
 * pools sized by HIDDEN, VOCAB-wide logits buffer, …). Migrating them
 * to runtime fields is P1.4.b — first the member arrays have to lose
 * their compile-time-sized declarations.
 *
 * Gemma-specific numeric knobs (RMS eps, logit softcap, PLE scales,
 * KV-shared layer indices) have already moved to struct geist_arch_config
 * — st->config.X is the right reader. P1.4.a migrated those. */
#define GEIST_GEMMA4_HIDDEN 1536
#define GEIST_GEMMA4_NUM_LAYERS 35
#define GEIST_GEMMA4_HIDDEN_PER_LAYER 256
#define GEIST_GEMMA4_PLE_OUT (GEIST_GEMMA4_NUM_LAYERS * GEIST_GEMMA4_HIDDEN_PER_LAYER)
#define GEIST_GEMMA4_N_Q_HEADS 8
#define GEIST_GEMMA4_N_KV_HEADS 1
#define GEIST_GEMMA4_VOCAB 262144

/* ---- Per-layer weight bundle ------------------------------------------- */

/* Holds every weight tensor needed to run one transformer layer. Layouts
 * that may be either F32 DENSE or BLOCK_QUANTIZED (k-quants) keep their
 * native dtype/layout — the linear() vtable dispatches on these. KV-shared
 * layers (idx >= 15) leave k_proj / v_proj / k_norm as NULL since they
 * reuse the source layer's projections at runtime. */
struct transformer_layer_weights {
    /* ---- Geometry derived from layer_idx (set by loader). ------------- */
    int    layer_idx;
    bool   is_full;        /* full-attn (vs sliding-window) */
    bool   is_kv_shared;   /* layers 15..34: no own K/V projections */
    size_t head_dim;       /* 512 if is_full else 256 */
    size_t q_out;          /* N_Q_HEADS * head_dim */
    size_t kv_out;         /* N_KV_HEADS * head_dim */
    size_t intermediate;   /* FFN inner dim — 12288 if kv_shared else 6144 */
    size_t sliding_window; /* 0 (unbounded) for full; 512 for sliding */
    float  rope_theta;     /* 1000000.0 for full; 10000.0 for sliding */
    int    n_rotated_dims; /* 128 for full; head_dim for sliding */
    float  layer_scalar;   /* per-layer output scale (scalar tensor) */

    /* ---- Norm weights (always F32 DENSE). ----------------------------- */
    struct geist_tensor attn_norm;           /* [HIDDEN] */
    struct geist_tensor q_norm;              /* [head_dim] */
    struct geist_tensor k_norm;              /* [head_dim] — unused if kv_shared */
    struct geist_tensor post_attn_norm;      /* [HIDDEN] */
    struct geist_tensor ffn_norm;            /* [HIDDEN] */
    struct geist_tensor post_ffw_norm;       /* [HIDDEN] */
    struct geist_tensor post_per_layer_norm; /* [HIDDEN] */
    /* SubLN (BitNet b1.58): RMSNorm before o_proj input ([q_out]) and
     * before down_proj input ([intermediate]). buffer == nullptr when
     * the family doesn't use SubLN (Gemma / Llama); the forward path
     * skips the rmsnorm call in that case. */
    struct geist_tensor attn_sub_norm; /* [q_out] — SubLN before o_proj */
    struct geist_tensor ffn_sub_norm;  /* [intermediate] — SubLN before down_proj */

    /* ---- Projection weights (Q-format or F32, BLOCK_QUANTIZED/DENSE). - */
    struct geist_tensor q_proj;         /* [q_out, HIDDEN] */
    struct geist_tensor k_proj;         /* [kv_out, HIDDEN] — null/unused if kv_shared */
    struct geist_tensor v_proj;         /* [kv_out, HIDDEN] — null/unused if kv_shared */
    struct geist_tensor o_proj;         /* [HIDDEN, q_out] */
    struct geist_tensor gate_proj;      /* [intermediate, HIDDEN] */
    struct geist_tensor up_proj;        /* [intermediate, HIDDEN] */
    struct geist_tensor down_proj;      /* [HIDDEN, intermediate] */
    struct geist_tensor per_layer_gate; /* [HIDDEN_PER_LAYER, HIDDEN] */
    struct geist_tensor per_layer_proj; /* [HIDDEN, HIDDEN_PER_LAYER] */

    /* ---- Pre-resolved kernel-pointer table (P1.1.c..d refactor v2). ---- *
     * Populated at load time from each projection's raw bytes via the
     * backend's resolve_weight. Forward hot path calls these directly
     * (no vtable, no dtype switch). If linear_m1 is nullptr, the
     * backend doesn't yet support that dtype/shape; the legacy
     * v->linear() path runs as fallback.                              */
    struct geist_weight q_proj_w;         /* P1.1.c */
    struct geist_weight k_proj_w;         /* P1.1.d */
    struct geist_weight v_proj_w;         /* P1.1.d */
    struct geist_weight o_proj_w;         /* P1.1.d */
    struct geist_weight gate_proj_w;      /* P1.1.d */
    struct geist_weight up_proj_w;        /* P1.1.d */
    struct geist_weight down_proj_w;      /* P1.1.d */
    struct geist_weight per_layer_gate_w; /* P1.1.d */
    struct geist_weight per_layer_proj_w; /* P1.1.d */

    /* ---- AWQ inverse scales (host arrays; nullptr if no AWQ). --------- *
     * Applied per-token to attn_out (before o_proj) and to post-GeGLU
     * gate (before down_proj). attn_norm/ffn_norm gammas are pre-divided
     * at load time so no runtime AWQ work is needed for those. */
    float *o_awq_inv_scale;    /* [q_out] or nullptr */
    float *down_awq_inv_scale; /* [intermediate] or nullptr */

    /* ---- Owning buffer handles (so destroy can free them). ------------ *
     * The structs above carry .buffer pointers but the LIST OF buffers we
     * created is stored here so cleanup is unambiguous and order-stable.
     * NULL slots denote "not allocated" (e.g. k_proj_buf for kv_shared). */
    struct geist_buffer *bufs[16];
    size_t               n_bufs;
};

/* ---- Per-session mutable state (P1.2.d refactor v2) -------------------- *
 *
 * All mutable state for ONE inference stream lives here. Separated from
 * the immutable transformer_arch_model fields so future sub-phase
 * P1.2.e can detach session lifetime from model lifetime (N sessions
 * per model). For now this struct is nested inside transformer_arch_
 * state with a 1:1 lifecycle, but the field grouping already
 * mirrors the eventual ownership model.
 *
 *   KV cache       : per-session inference state (kv_len, the buffers,
 *                    drain counters).
 *   scratch pool   : per-session reserved memory backing the long-
 *                    lived scratch buffers (P1.2.c).
 *   scratch arena  : per-session frame-arena for hot-path transients
 *                    (P1.2.b).
 *   sampler        : per-session RNG + workspace, temperature, top-k,
 *                    top-p.
 *   pending logits : last decode_step's argmax, lazily consumed by
 *                    the next call.
 */
struct transformer_arch_session {
    /* ---- KV-cache state. Layers 0..14 own buffers; 15..34 alias them
     * via GEMMA4_KV_*_SRC. One of three representations is active:
     *   FP32 (default on Apple): k_cache[] / v_cache[].
     *   INT8 (GEIST_KV_INT8=1; default non-Apple): k_cache_q8[] /
     *        v_cache_q8[] + per-(token, kv_head) scales.
     *   KIVI (GEIST_KV_KIVI=1): 2-bit channel-grouped K + 2-bit per-
     *        token V + R-token FP32 residual ring (k_residual /
     *        v_residual); kivi_residual_count + kivi_drained_count
     *        shared across layers (lock-step drain). */
    bool kv_int8_enabled;
    bool kv_kivi_enabled;
    /* Experiment (issue #61): FWHT-rotate Q/K/V before INT8 quant to
     * suppress activation outliers. Honored only in the INT8 path and
     * only when head_dim is a power of two. Env: GEIST_KV_ROT=1. */
    bool kv_rot_enabled;
    /* Experiment (issue #61): quantize the INT8 K/V cache on an N-bit grid
     * (scale = amax / (2^(N-1)-1)) — quality-only simulation of a symmetric
     * low-bit cache that reuses the INT8 storage + kernel (no packing, no
     * memory win yet). Measures whether rotation rescues low-bit quality.
     * 0 = native 8-bit; 2..7 forces the INT8 storage path on. Env:
     * GEIST_KV_QBITS=N. */
    int kv_sim_qbits;
    /* Packed symmetric 4-bit KV cache (issue #61). Rides the INT8 storage
     * path (kv_int8_enabled is also set for buffer alloc + ctx wiring) but
     * the k/v data buffers are allocated half-size and hold two 4-bit values
     * per byte; append packs, attention unpacks. Half the INT8 KV footprint.
     * Env: GEIST_KV_INT4=1. */
    bool kv_int4_packed_enabled;
    /* F16 KV cache: k_cache[]/v_cache[] hold half floats (2 bytes/elem);
     * appends convert through the backend's kv_append_f16 slot and
     * attention reads F16 views. Only set when that slot is non-null. */
    bool                                 kv_f16_enabled;
    struct transformer_session_exec_plan exec_plan;
    /* P1.4.c: per-layer KV slot arrays are heap-allocated at
     * session_alloc, sized to state->n_layers. Exactly one of the
     * three representations (FP32 / INT8 / KIVI) holds non-null slots
     * per non-KV-shared layer per the kv_*_enabled flags. */
    struct geist_buffer **k_cache;
    struct geist_buffer **v_cache;
    struct geist_buffer **k_cache_q8;
    struct geist_buffer **v_cache_q8;
    struct geist_buffer **k_cache_scale;
    struct geist_buffer **v_cache_scale;
    struct geist_buffer **k_kivi_q;
    struct geist_buffer **v_kivi_q;
    struct geist_buffer **k_kivi_scales;
    struct geist_buffer **k_kivi_zeros;
    struct geist_buffer **v_kivi_scales;
    struct geist_buffer **v_kivi_zeros;
    struct geist_buffer **k_residual;
    struct geist_buffer **v_residual;
    size_t                kivi_residual_count;
    size_t                kivi_drained_count;
    size_t                kv_len;        /* valid prefix across all caches */
    size_t                prefix_length; /* pinned prefix; reset truncates here */
    size_t                m_max;         /* prefill chunk size for this session */
    size_t                max_seq_len;   /* KV-cache capacity in rows — the state
                                          * max_seq_len at alloc time; forward paths
                                          * reject writes past this */

    /* ---- Scratch buffers (per-forward-pass workspace).
     * 21 buffers backed by the consolidated scratch pool (P1.2.c). */
    struct geist_buffer *scratch_normed;
    struct geist_buffer *scratch_q;
    struct geist_buffer *scratch_k;
    struct geist_buffer *scratch_v;
    struct geist_buffer *scratch_attn;
    struct geist_buffer *scratch_o;
    struct geist_buffer *scratch_post_attn;
    struct geist_buffer *scratch_h_post_attn;
    struct geist_buffer *scratch_pre_ff;
    struct geist_buffer *scratch_gate;
    struct geist_buffer *scratch_up;
    struct geist_buffer *scratch_ffn_out;
    struct geist_buffer *scratch_post_ff;
    struct geist_buffer *scratch_h_post_ff;
    struct geist_buffer *scratch_gate_ple;
    struct geist_buffer *scratch_proj_ple;
    struct geist_buffer *scratch_ones_headdim_max;
    struct geist_buffer *scratch_ple_lookup;
    struct geist_buffer *scratch_h_a;
    struct geist_buffer *scratch_h_b;
    struct geist_buffer *scratch_per_layer_input;
    struct geist_buffer *scratch_logits;

    /* ---- Memory backing for the buffers above.
     * scratch_arena: per-forward frame-arena (P1.2.b), reset per layer.
     * scratch_pool: never reset, holds slices for 21 scratch buffers. */
    void              *scratch_arena_base;
    size_t             scratch_arena_bytes;
    struct frame_arena scratch_arena;
    void              *scratch_pool_base;
    size_t             scratch_pool_bytes;
    size_t             scratch_pool_used;

    /* ---- Last-decode prediction (consumed by next decode_step). */
    bool logits_valid;
    /* Whether scratch_logits already carries the Gemma final-logit softcap.
     * The greedy argmax path skips the softcap (monotonic → argmax
     * invariant), so peek_logits applies it lazily for value consumers
     * (scoring/perplexity). Reset per forward in finalize_logits_one_row. */
    bool logits_softcapped;
    /* Whether scratch_logits is SPARSE — the spec-head fast path computed
     * exact logits only for its top-K candidates and set every other entry to
     * -inf (fine for the greedy argmax it serves). Value consumers must not
     * see that: peek_logits lazily recomputes the dense head from the still-
     * valid normalized hidden in scratch_h_a when this is set. */
    bool          logits_sparse;
    geist_token_t next_token_pending;

    /* ---- Sampler state.
     * temperature == 0.0 → greedy argmax; top_k>1 / top_p<1 narrow the
     * candidate set before the temperature-scaled softmax sample. */
    float                          temperature;
    float                          top_p;
    int                            top_k;
    struct geist_rng               rng;
    struct geist_sampler_workspace sampler_ws;
};

struct transformer_runtime_flags {
    bool bitnet_sub_ln_enabled;
    bool dump_act_sparsity;
};

/* ---- Top-level state --------------------------------------------------- */

struct transformer_arch_state {
    struct geist_backend *backend; /* not owned */

    /* GGUF source (kept open while the state lives: weight buffers reuploaded
     * could be re-fetched but for simplicity we just retain the handle for
     * diagnostics; mmap stays valid). */
    struct gguf_ctx *gguf; /* opaque; from gguf_reader.h */

    /* ---- Arch family config (P1.4.a). Populated at load time; holds
     * all Gemma-specific numeric knobs (RMS eps, logit softcap, PLE
     * scales, KV-shared layer mapping). Future arch families plug in
     * by swapping the populator at state_create. */
    struct geist_arch_config         config;
    struct transformer_runtime_flags runtime_flags;

    /* ---- Geometry (P1.4.b: structural dims as runtime fields). The
     * GEIST_GEMMA4_* macros remain in this header as Gemma-4 default
     * values + as compile-time caps on the fixed-length member arrays
     * below — runtime code reads these `st->*` fields instead. */
    size_t n_layers;         /* Gemma 4: 35 */
    size_t d_model;          /* Gemma 4: 1536 */
    size_t vocab_size;       /* Gemma 4: 262144 */
    size_t n_q_heads;        /* Gemma 4: 8 */
    size_t n_kv_heads;       /* Gemma 4: 1 (MQA) */
    size_t hidden_per_layer; /* Gemma 4: 256 (PLE slice per layer) */
    size_t ple_out;          /* Gemma 4: n_layers * hidden_per_layer = 8960 */
    size_t max_seq_len;      /* from opts, default 4096 */
    size_t m_max;            /* prefill chunk size — scratch sized for this many
                              * tokens; longer prompts are chunked. Default 64. */

    /* ---- Weight arena (P1.1.g refactor v2). -------------------------- *
     * ONE heap_alloc_aligned(arena_capacity) at load time; every weight
     * tensor bump-allocates into it. Released once in
     * transformer_state_destroy. */
    void  *weight_arena;
    size_t weight_arena_used;
    size_t weight_arena_capacity;

    /* ---- Per-layer weight blocks. P1.4.c heap-sizes this array to
     * st->n_layers (was GEIST_GEMMA4_NUM_LAYERS-sized in P1.4.b). */
    struct transformer_layer_weights   *layers;
    struct transformer_layer_exec_plan *layer_plans;

    struct geist_tensor embed_table;     /* [VOCAB, HIDDEN] — Q-format */
    struct geist_tensor ple_table;       /* [VOCAB, PLE_OUT] — Q-format */
    struct geist_tensor model_proj;      /* [PLE_OUT, HIDDEN] — Q-format */
    struct geist_tensor model_proj_norm; /* [HIDDEN_PER_LAYER] — F32 */
    struct geist_tensor output_norm;     /* [HIDDEN] — F32 */
    struct geist_weight embed_table_w;   /* P1.1.d lm_head kernels */
    struct geist_weight model_proj_w;    /* P1.1.e F32 dense kernels */

    /* Speculative output head (GEIST_SPEC_HEAD=1): for a large tied F16
     * lm_head, a stride-subsampled i8 "sketch" of the embedding table lets
     * decode rough-rank the whole vocab cheaply, pick top-K, then compute
     * EXACT f16 logits for only those K. Cuts the lm_head from ~656 MB to
     * ~82 MB read per token (the decode bottleneck on the BitNet-2B-4T
     * i2_s model). Lazily built on first use; nullptr/0 until then.
     * spec_state: 0 = unbuilt, 1 = built+active, -1 = ineligible/disabled. */
    int8_t *spec_sketch;     /* [VOCAB * spec_sketch_dim] i8 */
    float  *spec_row_scale;  /* [VOCAB] sketch dequant scale per row */
    int8_t *spec_x_i8;       /* [HIDDEN] quantized activation scratch */
    int8_t *spec_act_sketch; /* [spec_sketch_dim] subsampled activation */
    float  *spec_rough;      /* [VOCAB] rough scores scratch */
    float  *spec_row_f32;    /* [HIDDEN] dequant scratch for the exact pass
                              * (quantized embeddings, e.g. Gemma's Q6_K) */
    void  *spec_heap;        /* [spec_topk] (score,idx) top-K min-heap */
    size_t spec_sketch_dim;
    size_t spec_stride; /* sketch subsample stride (GEIST_SPEC_STRIDE) */
    size_t spec_topk;   /* exact-finalist count (GEIST_SPEC_TOPK) */
    int    spec_state;

    /* Owning buffer handles for the globals (mirrors layer .bufs[] pattern). */
    struct geist_buffer *global_bufs[8];
    size_t               n_global_bufs;

    /* ---- Precomputed RoPE cos/sin tables. Gemma 4 has two layer types
     * with different (rope_theta, n_rotated_dims) → two tables. Shape
     * each: [max_seq_len, head_dim]. Shared across sessions; rebuilt
     * (grow-only) when session_alloc sees a larger opts.max_seq_len. */
    struct geist_buffer *rope_cos_sliding;
    struct geist_buffer *rope_sin_sliding;
    struct geist_buffer *rope_cos_full;
    struct geist_buffer *rope_sin_full;

    /* ---- Per-session mutable state (P1.2.d/e/f).
     *
     * `sess` points at the currently-active session. `default_sess` is
     * the session that state_create allocates so direct-state callers
     * (no engine session) keep working unchanged.
     *
     * Engine-owned sessions allocated via transformer_session_alloc
     * are heap-separate; transformer_session_attach swaps the active
     * pointer before each dispatch. Single-active by design — true
     * concurrent multi-session is future work (would require passing
     * session_meta through every vtable call). */
    struct transformer_arch_session *sess;
    struct transformer_arch_session *default_sess;
};

/* ---- Forward-pass helpers (architecture-internal) --------------------- *
 * Single-token, single-layer forward. Used by sub-step 2 cross-reference
 * tests; the full prefill/decode wrappers land in sub-steps 2-loop and 3.
 *
 *   h_in_buf            [HIDDEN]  — residual stream input (F32).
 *   per_layer_input_buf [HIDDEN_PER_LAYER] — PLE input (or NULL to skip
 *                                            PLE block, useful for tests).
 *   h_out_buf           [HIDDEN]  — residual stream output.
 *   q_position          — absolute position of this token (KV cache index).
 *
 * Side effects: appends one (K, V) pair into the layer's KV cache (or the
 * source layer's cache if kv_shared); advances state->kv_len IFF
 * advance_kv == true (caller controls this so per-layer cross-refs can run
 * without polluting state).
 */
[[nodiscard]] enum geist_status
transformer_forward_one_layer(struct transformer_arch_state *state,
                              int                            layer_idx,
                              size_t                         q_position,
                              size_t                         seq,
                              bool                           advance_kv,
                              struct geist_buffer           *h_in_buf,
                              struct geist_buffer           *per_layer_input_buf,
                              struct geist_buffer           *h_out_buf);

/* Compute the PLE per-layer-input for ONE token (decode m=1) starting
 * from the residual stream `h` at this point in the pipeline.
 *
 *   token_id            — vocab id used to look up the PLE row.
 *   h_buf               [HIDDEN] — residual stream for the model_proj path.
 *   per_layer_input_buf [NUM_LAYERS * HIDDEN_PER_LAYER] — output, ready
 *                          to be sliced per-layer at runtime.
 *
 * PLE table is Q-quantized in the GGUF; this function dequantizes the
 * single needed row on the fly (lm.c's approach to keep the Pi 5 RAM
 * budget — full FP32 expansion of the table would be ~9 GB). */
[[nodiscard]] enum geist_status
transformer_compute_per_layer_input(struct transformer_arch_state *state,
                                    geist_token_t                  token_id,
                                    struct geist_buffer           *h_buf,
                                    struct geist_buffer           *per_layer_input_buf);

/* End-to-end single-token forward + sample. Consumes the input token at
 * position state->kv_len, advances state->kv_len by 1, and returns the
 * predicted next token via greedy argmax.
 *
 * This is the v2 equivalent of lm.c::lm_decode_step + the embedding scale
 * + the PLE precompute + the softcap'd lm_head all rolled into one call.
 * Used by the prefill+decode loop in the test gate; the full decoder
 * arch_ops wiring lands once this is verified end-to-end. */
[[nodiscard]] enum geist_status transformer_decode_step(struct transformer_arch_state *state,
                                                        geist_token_t                  input_token,
                                                        geist_token_t                 *out_token);

/* Append one audio soft-token (HIDDEN floats, already produced by the
 * audio encoder) to the KV cache.
 *
 * Differences vs the text path (from lm.c::lm_prefill_audio):
 *   - h_in_host is the soft-token bytes verbatim — no embedding lookup
 *     and no sqrt(HIDDEN) scale (soft-tokens enter post-lookup unscaled).
 *   - PLE token-identity is the pad_token_id (0), not the audio token's
 *     "identity" — matches HF's masked-scatter where audio positions are
 *     remapped to pad before PLE lookup.
 *
 * After return, kv_len is advanced by 1 and next_token_pending is the
 * greedy prediction for the position following this soft-token.
 * advance_audio_token is intended to be called n times in a row for an
 * n-soft-token audio clip; the engine's session_attach_audio takes care
 * of the encoder pipeline upstream. */
[[nodiscard]] enum geist_status
transformer_advance_audio_token(struct transformer_arch_state *state, const float *h_in_host);

/* Batched text prefill: append `n` tokens to the KV cache via the
 * seq>1 path. Processes the input in chunks of at most state->m_max
 * tokens. After the last chunk, the next-token-pending field holds the
 * greedy prediction for the position after the prefill, ready for
 * ops->decode_step to return.
 *
 * Returns GEIST_OK on success. n==0 is a no-op. */
[[nodiscard]] enum geist_status transformer_prefill_text_batch(struct transformer_arch_state *state,
                                                               size_t                         n,
                                                               const geist_token_t           *ids);

/* Batched audio prefill: append `n` soft-tokens to the KV cache via the
 * seq>1 path. Each soft-token is HIDDEN F32 values (no embed lookup,
 * no sqrt scale); PLE uses pad_token_id (0) for all positions. Same
 * chunking and side-effect semantics as the text batched prefill. */
[[nodiscard]] enum geist_status transformer_prefill_audio_batch(
        struct transformer_arch_state *state, size_t n, const float *soft_tokens);

/* Pin a prefix into the KV cache. Mirrors lm.c::lm_pin_prefix semantics:
 * truncates the cache to empty, runs a batched prefill of `ids` (no-op if
 * n==0), then snapshots the resulting kv_len into state->prefix_length.
 * Subsequent calls to transformer_state_reset truncate kv_len back to
 * this length (rather than 0), preserving the prefix's KV state across
 * conversation turns.
 *
 * Returns GEIST_OK on success; GEIST_E_INVALID_ARG if state is null. */
[[nodiscard]] enum geist_status
transformer_pin_prefix(struct transformer_arch_state *state, size_t n, const geist_token_t *ids);

/* Speculative-decode primitives.
 *
 * transformer_verify_forward: process `k` candidate tokens through
 * the full layer stack and return the model's sampled token for EACH
 * position. Like prefill_text_batch but produces k outputs (one per
 * position) instead of only the last. Advances kv_len by k. The
 * caller decides accept/reject and may truncate kv_len afterwards.
 *
 * For greedy mode the produced tokens are argmaxes; for sampled mode
 * they're samples from the configured distribution at each position.
 *
 * k must be ≥ 1 and small enough that one batched chunk handles it
 * (current m_max default 64 covers any realistic speculative K).
 *
 * transformer_kv_truncate: shrink kv_len to new_len. KV state at
 * positions ≥ new_len is implicitly invalid (future writes will
 * overwrite). Invalidates logits_valid. Used to undo speculative-pass
 * KV writes after a draft mismatch. */
[[nodiscard]] enum geist_status transformer_verify_forward(struct transformer_arch_state *state,
                                                           size_t                         k,
                                                           const geist_token_t           *ids,
                                                           geist_token_t *out_tokens);

void transformer_kv_truncate(struct transformer_arch_state *state, size_t new_len);

/* ---- Public functions (architecture-internal) -------------------------- */

/* Build a transformer arch state from a GGUF file. All weight bytes land in
 * backend-owned buffers via be->vtbl->buffer_create+upload. KV cache and
 * scratch buffer allocation are deferred to sub-step 2.
 *
 * Returns GEIST_OK on success; on failure *out is left as nullptr and the
 * backend's error slot is populated. Caller owns *out and frees it via
 * transformer_state_destroy. */
[[nodiscard]] enum geist_status transformer_state_create(struct geist_backend            *be,
                                                         const char                      *gguf_path,
                                                         const struct geist_session_opts *opts,
                                                         struct transformer_arch_state  **out);

/* Same, but the GGUF is already in memory (embedded blob); the buffer is
 * aliased read-only and must outlive the state. No aux files searched. */
[[nodiscard]] enum geist_status
transformer_state_create_from_memory(struct geist_backend            *be,
                                     const void                      *data,
                                     size_t                           size,
                                     const struct geist_session_opts *opts,
                                     struct transformer_arch_state  **out);

/* Shared body for both entry points; takes ownership of an open gguf_ctx. */
[[nodiscard]] enum geist_status
transformer_state_create_from_gguf(struct geist_backend            *be,
                                   struct gguf_ctx                 *gguf,
                                   const struct geist_session_opts *opts,
                                   struct transformer_arch_state  **out);

void transformer_state_destroy(struct transformer_arch_state *state);

/* Reset conversational state — drops the KV cache and stored logits;
 * keeps the loaded weights. Used by ops->state_reset. */
void transformer_state_reset(struct transformer_arch_state *state);

/* Apply per-session opts to the arch state. Called by the engine from
 * geist_session_create. Re-seeds the RNG; (re)allocates the sampler
 * workspace if the opts request a non-greedy mode. */
void transformer_state_apply_opts(struct transformer_arch_state   *state,
                                  const struct geist_session_opts *opts);

/* ---- Multi-session API (P1.2.f) --------------------------------------- *
 *
 * Each engine-level geist_session may own its own session (KV cache,
 * scratch pool, sampler RNG, ...). The model state's `sess` slot points
 * at the *active* session — re-bound by transformer_session_attach
 * before each dispatch.
 *
 * state_create still installs a default session for backward-compat with
 * direct-state callers (e.g. test_state_decode_int). Engine sessions
 * allocate their own via transformer_session_alloc; on destroy the
 * engine re-binds the model's default session via session_attach(nullptr).
 *
 * Concurrency: single active session at a time. True concurrent multi-
 * session would require passing session_meta through every vtable call;
 * out of scope for P1.2.f. */
[[nodiscard]] struct transformer_arch_session *
transformer_session_alloc(struct transformer_arch_state   *state,
                          const struct geist_session_opts *opts);

void transformer_session_free(struct transformer_arch_state   *state,
                              struct transformer_arch_session *sess);

void transformer_session_attach(struct transformer_arch_state   *state,
                                struct transformer_arch_session *sess);

/* Internal accessor for the model's default session (the one installed
 * by transformer_state_create). Engine uses this to restore the
 * default after detaching an engine-owned session. */
struct transformer_arch_session *transformer_default_session(struct transformer_arch_state *state);

#endif /* GEIST_INTERNAL_ARCH_TRANSFORMER_STATE_V2_H */

/*
 * src/archs/transformer/arch_state.c — Phase B-4e sub-step 1.
 *
 * Loads Gemma 4 weights from GGUF into backend-owned buffers. Each per-layer
 * and global tensor is staged via be->vtbl->buffer_create + buffer_upload;
 * the resulting geist_tensor views (dtype + layout + shape) live in the v2
 * state struct ready for sub-step 2 to feed into the linear/rmsnorm/etc.
 * vtable ops.
 *
 * Layer: ARCHITECTURE.
 *
 * On any failure mid-load the partial state is torn down before returning —
 * no leaked buffers. The backend's err_msg is populated with the failing
 * tensor name and reason.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "arch_state.h"
#include "arch_family.h"
#include "exec_plan.h"
#include "arch_ops.h"
#include "forward.h"
#include "scratch_plan.h"
#include "weight_load.h"

#include "quant.h"
#include "gguf_reader.h"
#include "gemma4_kernels.h"
#include "heap.h"
#include "kivi.h"
#include "ptqtp_awq.h"

#include <geist.h>
#include <geist_backend.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static bool env_flag_enabled(const char *name, bool fallback) {
    const char *env = getenv(name);
    if (env == nullptr || env[0] == '\0') {
        return fallback;
    }
    return env[0] == '1' || env[0] == 'y' || env[0] == 'Y' || env[0] == 't' || env[0] == 'T';
}

static struct transformer_runtime_flags transformer_runtime_flags_from_env(void) {
    return (struct transformer_runtime_flags) {
            .bitnet_sub_ln_enabled = !env_flag_enabled("GEIST_BITNET_NO_SUBLN", false),
            .dump_act_sparsity     = env_flag_enabled("GEIST_DUMP_ACT_SPARSITY", false),
    };
}

static void release_weight_aux(struct geist_weight *w) {
    if (w == nullptr || w->aux_fp32 == nullptr) {
        return;
    }
    if ((w->flags & GEIST_W_AUX_HEAP_OWNED) != 0) {
        void *p = (void *) w->aux_fp32;
        safe_free(&p);
        w->aux_fp32 = nullptr;
        w->aux_n    = 0;
        w->flags &= (uint16_t) ~(GEIST_W_AUX_HEAP_OWNED | GEIST_W_AUX_BACKEND_REPACK);
        w->backend_layout    = GEIST_W_LAYOUT_SOURCE;
        w->backend_alignment = 0;
    }
}

static void release_layer_weight_aux(struct transformer_layer_weights *L) {
    if (L == nullptr) {
        return;
    }
    release_weight_aux(&L->q_proj_w);
    release_weight_aux(&L->k_proj_w);
    release_weight_aux(&L->v_proj_w);
    release_weight_aux(&L->o_proj_w);
    release_weight_aux(&L->gate_proj_w);
    release_weight_aux(&L->up_proj_w);
    release_weight_aux(&L->down_proj_w);
    release_weight_aux(&L->per_layer_gate_w);
    release_weight_aux(&L->per_layer_proj_w);
}

/* PLE scaling constants moved to forward.c (P1.3.a) — used only by the
 * per-layer-input precompute path which now lives there. */

/* 4 sliding + 1 full, 7 cycles — same as lm.c::LAYER_IS_FULL. */
/* ---- Runtime infrastructure: KV caches, RoPE tables, scratch ---------- */

/* Allocate a single scratch buffer of `bytes` and clear it. */
[[nodiscard]] static enum geist_status
alloc_scratch(struct geist_backend *be, size_t bytes, struct geist_buffer **out) {
    enum geist_status s =
            be->desc->vtbl->buffer_create(be, bytes, GEIST_BUFFER_SCRATCH, GEIST_MEMORY_AUTO, out);
    if (s != GEIST_OK) {
        return s;
    }
    void *p = be->desc->vtbl->buffer_map(*out);
    if (p != nullptr) {
        memset(p, 0, bytes);
    }
    be->desc->vtbl->buffer_unmap(*out);
    return GEIST_OK;
}

/* P1.2.c: bump-allocate from the per-state scratch pool and wrap the
 * slice in an aliased buffer. Pool is allocated once at create-time
 * with capacity = sum of all scratch slot sizes; the helper aligns
 * each slot to 64 B (NEON-friendly), zeros the slice (matches the
 * pre-P1.2.c alloc_scratch behavior), and constructs a
 * GEIST_MEMORY_ALIASED geist_buffer wrapping the pool offset.
 *
 * Buffer ownership: the buffer header (struct geist_buffer) is owned
 * by the backend allocator and destroyed normally via
 * buffer_destroy; the underlying bytes belong to the pool and are
 * released exactly once when transformer_state_destroy frees
 * scratch_pool_base. */
[[nodiscard]] static enum geist_status
alloc_pool_buffer(struct transformer_arch_state *st, size_t bytes, struct geist_buffer **out_buf) {

    struct geist_backend *be      = st->backend;
    const size_t          align   = 64;
    const size_t          mask    = align - 1;
    const size_t          aligned = (st->sess->scratch_pool_used + mask) & ~mask;
    if (aligned + bytes > st->sess->scratch_pool_bytes) {
        geist_backend_set_error(be,
                                GEIST_E_OOM,
                                "transformer: scratch pool exhausted "
                                "(used %zu, need %zu, capacity %zu)",
                                st->sess->scratch_pool_used,
                                bytes,
                                st->sess->scratch_pool_bytes);
        return GEIST_E_OOM;
    }
    void *p = (uint8_t *) st->sess->scratch_pool_base + aligned;
    memset(p, 0, bytes);
    st->sess->scratch_pool_used = aligned + bytes;
    return be->desc->vtbl->buffer_create_aliased(be, p, bytes, GEIST_BUFFER_SCRATCH, out_buf);
}

/* Upload a host-side cos or sin table into a backend buffer. */
[[nodiscard]] static enum geist_status upload_table(struct geist_backend *be,
                                                    const float          *src,
                                                    size_t                n_floats,
                                                    struct geist_buffer **out) {
    enum geist_status s = be->desc->vtbl->buffer_create(
            be, n_floats * sizeof(float), GEIST_BUFFER_WEIGHT, GEIST_MEMORY_AUTO, out);
    if (s != GEIST_OK) {
        return s;
    }
    return be->desc->vtbl->buffer_upload(*out, n_floats * sizeof(float), (const uint8_t *) src);
}

/* P1.2.f: model-owned runtime alloc — RoPE cos/sin tables. Per-layer
 * fields (head_dim, n_rotated_dims, rope_theta) come from the family
 * populator — Gemma 4 uses two sets (sliding 256-dim theta=1e4 full-
 * rotation, full 512-dim theta=1e6 25%-rotation); Llama / BitNet share
 * one set across all layers. We pick the largest-stride layer of each
 * (is_full=true / false) so a single table covers every layer that
 * indexes into it. */
[[nodiscard]] static enum geist_status allocate_runtime_rope(struct transformer_arch_state *st) {

    struct geist_backend *be = st->backend;

    /* Find a representative sliding-attn and full-attn layer. Families
     * without sliding (BitNet / Llama / Mistral) fall back to layer 0's
     * params for the sliding table too — the table won't be indexed at
     * runtime since is_full is always true. */
    int sliding_idx = -1, full_idx = -1;
    for (size_t i = 0; i < st->n_layers; i++) {
        if (full_idx == -1 && st->layers[i].is_full)
            full_idx = (int) i;
        if (sliding_idx == -1 && !st->layers[i].is_full)
            sliding_idx = (int) i;
    }
    if (full_idx == -1)
        full_idx = 0;
    if (sliding_idx == -1)
        sliding_idx = full_idx;

    const size_t sliding_hd  = st->layers[sliding_idx].head_dim;
    const size_t full_hd     = st->layers[full_idx].head_dim;
    const size_t sliding_rot = (size_t) st->layers[sliding_idx].n_rotated_dims;
    const size_t full_rot    = (size_t) st->layers[full_idx].n_rotated_dims;
    const float  sliding_th  = st->layers[sliding_idx].rope_theta;
    const float  full_th     = st->layers[full_idx].rope_theta;
    const size_t n_sl_floats = st->max_seq_len * sliding_hd;
    const size_t n_fl_floats = st->max_seq_len * full_hd;

    float *cos_sl = heap_alloc_aligned(n_sl_floats * sizeof(float), 64);
    float *sin_sl = heap_alloc_aligned(n_sl_floats * sizeof(float), 64);
    float *cos_fl = heap_alloc_aligned(n_fl_floats * sizeof(float), 64);
    float *sin_fl = heap_alloc_aligned(n_fl_floats * sizeof(float), 64);
    if (cos_sl == nullptr || sin_sl == nullptr || cos_fl == nullptr || sin_fl == nullptr) {
        void *p[4] = {cos_sl, sin_sl, cos_fl, sin_fl};
        for (size_t i = 0; i < 4; i++) {
            safe_free(&p[i]);
        }
        geist_backend_set_error(be, GEIST_E_OOM, "transformer: RoPE table host alloc failed");
        return GEIST_E_OOM;
    }
    rope_compute_at(0, st->max_seq_len, sliding_hd, sliding_rot, sliding_th, cos_sl, sin_sl);
    rope_compute_at(0, st->max_seq_len, full_hd, full_rot, full_th, cos_fl, sin_fl);

    enum geist_status s = upload_table(be, cos_sl, n_sl_floats, &st->rope_cos_sliding);
    if (s == GEIST_OK) {
        s = upload_table(be, sin_sl, n_sl_floats, &st->rope_sin_sliding);
    }
    if (s == GEIST_OK) {
        s = upload_table(be, cos_fl, n_fl_floats, &st->rope_cos_full);
    }
    if (s == GEIST_OK) {
        s = upload_table(be, sin_fl, n_fl_floats, &st->rope_sin_full);
    }

    void *to_free[4] = {cos_sl, sin_sl, cos_fl, sin_fl};
    for (size_t i = 0; i < 4; i++) {
        safe_free(&to_free[i]);
    }
    return s;
}

/* P1.2.f: session-owned runtime allocs — KV caches + scratch pool +
 * per-forward arena + ones-row scratch. Operates on the session currently
 * installed at st->sess; caller must install the target session before
 * calling and restore the previous one afterwards. */
[[nodiscard]] static enum geist_status allocate_runtime_session(struct transformer_arch_state *st) {

    struct geist_backend *be = st->backend;
    enum geist_status     s;

    /* ---- KV caches: per-layer for 0..14, NULL for 15..34 (those alias
     * the source layer's cache at runtime). Three branches, exactly one
     * fires per layer based on the kv_*_enabled flags. */
    for (size_t li = 0; li < (size_t) st->n_layers; li++) {
        if (st->layers[li].is_kv_shared) {
            continue; /* all KV pointer slots stay nullptr */
        }
        const size_t hd       = st->layers[li].head_dim;
        const size_t n_elems  = st->max_seq_len * st->n_kv_heads * hd;
        const size_t n_scales = st->max_seq_len * st->n_kv_heads;
        if (st->sess->kv_kivi_enabled) {
            /* Drained region packs at 2 bits per element (4 vals/byte).
             * Per-channel K scales/zeros: one fp32 per (group, channel).
             * Per-token V scales/zeros: one fp32 per (token, kv_head).
             * Residual is (R + m_max) fp32 K/V rows so a verify_forward
             * burst of m_max tokens never overflows. */
            const size_t R              = KIVI_K_GROUP_SIZE;
            const size_t n_drain_groups = (st->max_seq_len + R - 1) / R;
            const size_t k_scales_elems = n_drain_groups * st->n_kv_heads * hd;
            const size_t v_scales_elems = st->max_seq_len * st->n_kv_heads;
            const size_t residual_slots = R + st->sess->m_max;
            const size_t residual_elems = residual_slots * st->n_kv_heads * hd;
            s                           = alloc_scratch(be, n_elems / 4, &st->sess->k_kivi_q[li]);
            if (s != GEIST_OK) {
                return s;
            }
            s = alloc_scratch(be, n_elems / 4, &st->sess->v_kivi_q[li]);
            if (s != GEIST_OK) {
                return s;
            }
            s = alloc_scratch(be, k_scales_elems * sizeof(float), &st->sess->k_kivi_scales[li]);
            if (s != GEIST_OK) {
                return s;
            }
            s = alloc_scratch(be, k_scales_elems * sizeof(float), &st->sess->k_kivi_zeros[li]);
            if (s != GEIST_OK) {
                return s;
            }
            s = alloc_scratch(be, v_scales_elems * sizeof(float), &st->sess->v_kivi_scales[li]);
            if (s != GEIST_OK) {
                return s;
            }
            s = alloc_scratch(be, v_scales_elems * sizeof(float), &st->sess->v_kivi_zeros[li]);
            if (s != GEIST_OK) {
                return s;
            }
            s = alloc_scratch(be, residual_elems * sizeof(float), &st->sess->k_residual[li]);
            if (s != GEIST_OK) {
                return s;
            }
            s = alloc_scratch(be, residual_elems * sizeof(float), &st->sess->v_residual[li]);
            if (s != GEIST_OK) {
                return s;
            }
        } else if (st->sess->kv_int8_enabled) {
            s = alloc_scratch(be, n_elems * sizeof(int8_t), &st->sess->k_cache_q8[li]);
            if (s != GEIST_OK) {
                return s;
            }
            s = alloc_scratch(be, n_elems * sizeof(int8_t), &st->sess->v_cache_q8[li]);
            if (s != GEIST_OK) {
                return s;
            }
            s = alloc_scratch(be, n_scales * sizeof(float), &st->sess->k_cache_scale[li]);
            if (s != GEIST_OK) {
                return s;
            }
            s = alloc_scratch(be, n_scales * sizeof(float), &st->sess->v_cache_scale[li]);
            if (s != GEIST_OK) {
                return s;
            }
        } else {
            const size_t kv_elem_bytes =
                    st->sess->kv_f16_enabled ? 2u : sizeof(float);
            s = alloc_scratch(be, n_elems * kv_elem_bytes, &st->sess->k_cache[li]);
            if (s != GEIST_OK) {
                return s;
            }
            s = alloc_scratch(be, n_elems * kv_elem_bytes, &st->sess->v_cache[li]);
            if (s != GEIST_OK) {
                return s;
            }
        }
    }
    st->sess->kv_len = 0;

    /* ---- Scratch buffers — sized for m_max tokens to support batched
     * prefill. Decode (seq=1) only touches the first row of each buffer,
     * extra capacity is just resident memory cost (~30 MB at m_max=64).
     * max head_dim = 512, max intermediate = 12288, max q_out = 4096.
     *
     * P1.2.c (refactor v2): all 21 scratch buffers backed by a single
     * heap_alloc_aligned'd pool. One allocation per state instead of
     * 21 separate ones; each buffer is a GEIST_MEMORY_ALIASED slice. */
    struct transformer_scratch_plan scratch_plan;
    transformer_scratch_plan_build(st, &scratch_plan);
    const size_t F               = sizeof(float);
    const size_t head_dim_max    = 512;
    st->sess->scratch_pool_bytes = scratch_plan.pool_bytes;
    st->sess->scratch_pool_base  = heap_alloc_aligned(st->sess->scratch_pool_bytes, 64);
    if (st->sess->scratch_pool_base == nullptr) {
        geist_backend_set_error(be,
                                GEIST_E_OOM,
                                "transformer: scratch pool alloc failed (%zu bytes)",
                                st->sess->scratch_pool_bytes);
        return GEIST_E_OOM;
    }
    st->sess->scratch_pool_used = 0;

    s = alloc_pool_buffer(st, scratch_plan.hidden, &st->sess->scratch_normed);
    if (s != GEIST_OK) {
        return s;
    }
    s = alloc_pool_buffer(st, scratch_plan.q_out, &st->sess->scratch_q);
    if (s != GEIST_OK) {
        return s;
    }
    s = alloc_pool_buffer(st, scratch_plan.kv_out, &st->sess->scratch_k);
    if (s != GEIST_OK) {
        return s;
    }
    s = alloc_pool_buffer(st, scratch_plan.kv_out, &st->sess->scratch_v);
    if (s != GEIST_OK) {
        return s;
    }
    s = alloc_pool_buffer(st, scratch_plan.q_out, &st->sess->scratch_attn);
    if (s != GEIST_OK) {
        return s;
    }
    s = alloc_pool_buffer(st, scratch_plan.hidden, &st->sess->scratch_o);
    if (s != GEIST_OK) {
        return s;
    }
    s = alloc_pool_buffer(st, scratch_plan.hidden, &st->sess->scratch_post_attn);
    if (s != GEIST_OK) {
        return s;
    }
    s = alloc_pool_buffer(st, scratch_plan.hidden, &st->sess->scratch_h_post_attn);
    if (s != GEIST_OK) {
        return s;
    }
    s = alloc_pool_buffer(st, scratch_plan.hidden, &st->sess->scratch_pre_ff);
    if (s != GEIST_OK) {
        return s;
    }
    s = alloc_pool_buffer(st, scratch_plan.inter, &st->sess->scratch_gate);
    if (s != GEIST_OK) {
        return s;
    }
    s = alloc_pool_buffer(st, scratch_plan.inter, &st->sess->scratch_up);
    if (s != GEIST_OK) {
        return s;
    }
    s = alloc_pool_buffer(st, scratch_plan.hidden, &st->sess->scratch_ffn_out);
    if (s != GEIST_OK) {
        return s;
    }
    s = alloc_pool_buffer(st, scratch_plan.hidden, &st->sess->scratch_post_ff);
    if (s != GEIST_OK) {
        return s;
    }
    s = alloc_pool_buffer(st, scratch_plan.hidden, &st->sess->scratch_h_post_ff);
    if (s != GEIST_OK) {
        return s;
    }
    /* P1.5.d: PLE-only scratch buffers — sized to 0 for non-PLE
     * families (sz_hidden_per / sz_ple_out depend on hidden_per_layer
     * + ple_out which are 0 when has_ple is false). The backend
     * rejects 0-byte aliased buffers, so skip these allocs entirely
     * when the family doesn't need PLE. The forward path's PLE-skip
     * guards (P1.5.b) leave these pointers null and the PLE block
     * never executes. */
    if (st->config.has_ple) {
        s = alloc_pool_buffer(st, scratch_plan.hidden_per, &st->sess->scratch_gate_ple);
        if (s != GEIST_OK) {
            return s;
        }
        s = alloc_pool_buffer(st, scratch_plan.hidden, &st->sess->scratch_proj_ple);
        if (s != GEIST_OK) {
            return s;
        }
        /* scratch_ple_lookup hosts (a) m_max dequant'd PLE rows during
         * batched compute_per_layer_inputs, and (b) the [seq, HIDDEN_PER_LAYER]
         * per-layer-input slice during the layer loop. The larger of the
         * two is (a) at m_max * PLE_OUT. */
        s = alloc_pool_buffer(st, scratch_plan.ple_out, &st->sess->scratch_ple_lookup);
        if (s != GEIST_OK) {
            return s;
        }
        s = alloc_pool_buffer(st, scratch_plan.ple_out, &st->sess->scratch_per_layer_input);
        if (s != GEIST_OK) {
            return s;
        }
    } else {
        st->sess->scratch_gate_ple        = nullptr;
        st->sess->scratch_proj_ple        = nullptr;
        st->sess->scratch_ple_lookup      = nullptr;
        st->sess->scratch_per_layer_input = nullptr;
    }
    s = alloc_pool_buffer(st, scratch_plan.hidden, &st->sess->scratch_h_a);
    if (s != GEIST_OK) {
        return s;
    }
    s = alloc_pool_buffer(st, scratch_plan.hidden, &st->sess->scratch_h_b);
    if (s != GEIST_OK) {
        return s;
    }
    /* scratch_logits holds m_max × VOCAB floats so verify_forward can do
     * ONE batched lm_head per call (the lm_head is the dominant cost on
     * Pi 5; M>1 IQ kernels amortize the 262K-wide weight stream over
     * k columns). At m_max=64 and VOCAB=262144 this is ~64 MB. Lives
     * in the scratch pool. */
    s = alloc_pool_buffer(st, scratch_plan.vocab, &st->sess->scratch_logits);
    if (s != GEIST_OK) {
        return s;
    }

    /* All-ones buffer of length head_dim_max — used as the weight for the V
     * rmsnorm step (lm.c passes NULL meaning all-ones; the vtable rmsnorm
     * needs a real weight tensor). We slice this down to the layer's actual
     * head_dim per call by setting the tensor view's shape[0]. */
    /* P1.2.b (refactor v2): per-forward scratch arena. Sized for the
     * current set of arena consumers (attention scores buffer); will
     * grow as P1.2.c migrates more scratch sites into the arena.
     *
     *   scores  : max_seq_len floats = 16 KB on default config
     *   slack   : a few KB for alignment + future small allocs
     *
     * Round to 64 KB so the cap is comfortable for P1.2.c growth. */
    st->sess->scratch_arena_bytes = 64u * 1024u;
    st->sess->scratch_arena_base  = heap_alloc_aligned(st->sess->scratch_arena_bytes, 64);
    if (st->sess->scratch_arena_base == nullptr) {
        geist_backend_set_error(be,
                                GEIST_E_OOM,
                                "transformer: scratch arena alloc failed (%zu bytes)",
                                st->sess->scratch_arena_bytes);
        return GEIST_E_OOM;
    }
    frame_arena_init(
            &st->sess->scratch_arena, st->sess->scratch_arena_base, st->sess->scratch_arena_bytes);

    s = alloc_scratch(be, head_dim_max * F, &st->sess->scratch_ones_headdim_max);
    if (s != GEIST_OK) {
        return s;
    }
    {
        float *p = (float *) be->desc->vtbl->buffer_map(st->sess->scratch_ones_headdim_max);
        for (size_t i = 0; i < head_dim_max; i++) {
            p[i] = 1.0f;
        }
        be->desc->vtbl->buffer_unmap(st->sess->scratch_ones_headdim_max);
    }

    return GEIST_OK;
}

/* ---- Public entry points ---------------------------------------------- */

/* Shared body: takes ownership of an already-open `gguf` (closes it on error,
 * and on success either keeps it open for zero-copy weight aliasing or closes
 * it after copying, depending on mmap_alias_mode). The path/from-memory entry
 * points below just open the gguf and delegate here. */
enum geist_status transformer_state_create_from_gguf(struct geist_backend            *be,
                                                     struct gguf_ctx                 *gguf,
                                                     const struct geist_session_opts *opts,
                                                     struct transformer_arch_state  **out) {
    if (be == nullptr || gguf == nullptr || out == nullptr) {
        if (gguf != nullptr) {
            gguf_close(gguf);
        }
        return GEIST_E_INVALID_ARG;
    }
    *out = nullptr;

    struct transformer_arch_state *st =
            heap_alloc_aligned(sizeof(*st), alignof(struct transformer_arch_state));
    if (st == nullptr) {
        gguf_close(gguf);
        geist_backend_set_error(
                be, GEIST_E_OOM, "transformer: state alloc failed (%zu bytes)", sizeof(*st));
        return GEIST_E_OOM;
    }
    memset(st, 0, sizeof(*st));
    st->backend       = be;
    st->gguf          = (struct gguf_ctx *) gguf;
    st->runtime_flags = transformer_runtime_flags_from_env();
    /* P1.4.b: structural dims as runtime fields. Currently hardcoded
     * to Gemma-4 E2B defaults; future GGUF-metadata reader extension
     * (P1.4.c) will derive these from `general.architecture`-specific
     * keys. Existing GEIST_GEMMA4_* macros are bit-equal — kept in
     * arch_state.h as compile-time caps for the fixed-length
     * member arrays. */
    st->n_layers         = GEIST_GEMMA4_NUM_LAYERS;
    st->d_model          = GEIST_GEMMA4_HIDDEN;
    st->vocab_size       = GEIST_GEMMA4_VOCAB;
    st->n_q_heads        = GEIST_GEMMA4_N_Q_HEADS;
    st->n_kv_heads       = GEIST_GEMMA4_N_KV_HEADS;
    st->hidden_per_layer = GEIST_GEMMA4_HIDDEN_PER_LAYER;
    st->ple_out          = st->n_layers * st->hidden_per_layer;
    st->max_seq_len      = (opts != nullptr && opts->max_seq_len > 0) ? opts->max_seq_len : 4096;
#if defined(GEIST_TARGET_PI5)
    st->m_max = 64; /* Pi 5: m=32 was chosen pre-packing to keep the
                     * activation tile L1-resident (else it re-fetched from
                     * L2 per output row). Activation packing (q4_K.c §10.7)
                     * now makes that access sequential/prefetchable (L1-miss
                     * ~0.95% even at larger m), so the L1-fit constraint is
                     * gone and the bigger batch wins by amortizing the
                     * per-block-row weight setup (recon + scales) over more
                     * tokens. Measured 2026-06-07 (seq256, 4t): m=32 27.8 →
                     * m=64 29.3 (+5.4%); m=128 regresses to 27.0 (working
                     * set too large). GEIST_M_MAX overrides. */
#else
    st->m_max = 64; /* Mac/Accelerate prefers larger batches: the
                     * predecoded SGEMM path scales up to m=128 (m=32
                     * regresses Accelerate). 64 keeps scratch under ~30 MB
                     * while feeding Accelerate a decent batch. */
    if (be->desc != nullptr && be->desc->name != nullptr &&
        strcmp(be->desc->name, "metal") == 0) {
        st->m_max = 128; /* GPU sweet spot (measured, M1 Max): the mm_sg
                          * GEMM fast paths want rows%64==0 and fewer,
                          * larger chunks; 128 was the bench default via
                          * GEIST_M_MAX all along. */
    }
#endif
    { /* GEIST_M_MAX override — for tuning the prefill activation tile vs L1
       * fit (m×n_in int8 should fit the 64 KB L1: m=32→48 KB, m=64→96 KB).
       * GEIST_QUANT_M_CAP guards CPU quant-kernel stack arrays; the metal
       * path never runs those in prefill, so the GPU may batch larger. */
        const bool is_metal = be->desc != nullptr &&
                              be->desc->name != nullptr &&
                              strcmp(be->desc->name, "metal") == 0;
        const int cap = is_metal ? 512 : (int) GEIST_QUANT_M_CAP;
        const char *mm = getenv("GEIST_M_MAX");
        if (mm != nullptr && mm[0] != '\0') {
            const int v = atoi(mm);
            if (v > 0 && v <= cap)
                st->m_max = (size_t) v;
        }
    }

    /* P1.4.a: populate the arch-family config from Gemma-4 defaults.
     * PLE scales + KV-shared layer pattern are Gemma-4 architectural
     * constants (not in GGUF metadata); rms_eps + logit_softcap are
     * overwritten from gguf meta below when present. */
    st->config = (struct geist_arch_config) {
            .family               = "gemma4",
            .rms_eps              = 1e-6f,
            .logit_softcap        = 30.0f,
            .has_ple              = true,
            .ple_input_scale      = 0.7071067811865476f,
            .ple_model_proj_scale = 0.02551551815399144f,
            .ple_table_scale      = 16.0f,
            .kv_sliding_src       = 13,
            .kv_full_src          = 14,
            .has_gemma_attn_norms = true,
            .has_sub_ln           = false,
            .ffn_activation       = GEIST_FFN_GEGLU,
    };

    /* P1.5: dispatch to the per-family populator selected by
     * `general.architecture`. Unknown / missing arch falls back to
     * Gemma-4 (see arch_family.c::transformer_family_select). For the
     * Gemma-4 IQ2_M test file the meta values equal the defaults so
     * the populator's overrides are a no-op — bit-identical load. */
    const struct transformer_family *fam = transformer_family_select(gguf);
    st->config.family                    = fam->name;
    fam->populate(gguf, st);

    /* P1.4.c: heap-allocate the per-layer weight array sized to the
     * model's actual layer count. (Was a compile-time `[NUM_LAYERS]`
     * field; freed in state_destroy below.) */
    st->layers = heap_alloc_aligned(st->n_layers * sizeof(*st->layers),
                                    alignof(struct transformer_layer_weights));
    if (st->layers == nullptr) {
        void *p = st;
        safe_free(&p);
        gguf_close(gguf);
        geist_backend_set_error(be,
                                GEIST_E_OOM,
                                "transformer: layer array alloc failed "
                                "(%zu layers × %zu bytes)",
                                st->n_layers,
                                sizeof(*st->layers));
        return GEIST_E_OOM;
    }
    memset(st->layers, 0, st->n_layers * sizeof(*st->layers));

    /* P1.5.c: fill per-layer geometry. The family populator decides
     * the attention pattern (Gemma: 4 sliding + 1 full × 7 with KV
     * sharing at idx >= 15; Llama: uniform full-attn, no sharing).
     * weight_load.c::load_one_layer reads these pre-filled fields
     * instead of deriving them. */
    fam->populate_layers(st);

    /* Storage mode (mmap-alias default vs β-mode override). mmap-alias
     * (the P0.3 behavior) keeps weight bytes aliased to the GGUF mmap and
     * demand-pages disk reads. This is the default because copying the
     * whole file resident blows RSS on memory-constrained targets: e.g.
     * Gemma 4 E2B Q4_K_M is 3.41 GB on disk but only ~1.49 GB is streamed
     * per decode token — the 1.93 GB `per_layer_token_embd` PLE table is
     * lookup-only (one row/token). β-mode copies that 1.93 GB resident,
     * pushing RSS to ~3.7 GB on a 4 GB Pi 5 → thrash. mmap-alias leaves it
     * disk-backed, paging only touched rows.
     *
     * GEIST_WEIGHT_MMAP=0 forces legacy β-mode (single backend arena, full
     * copy, mmap dropped post-load — full backend ownership, no retained
     * fd). Any other value (or unset) keeps the mmap-alias default. */
    bool mmap_alias_mode = true;
    {
        const char *env = getenv("GEIST_WEIGHT_MMAP");
        if (env != nullptr && env[0] != '\0') {
            mmap_alias_mode = (env[0] != '0');
        }
    }

    if (!mmap_alias_mode) {
        /* P1.1.g: compute total weight arena capacity from GGUF metadata,
         * then ONE heap_alloc_aligned. All weight tensors will bump-
         * allocate from here. */
        size_t            cap = 0;
        enum geist_status cs  = compute_weight_arena_capacity(gguf, &cap);
        if (cs != GEIST_OK) {
            transformer_state_destroy(st);
            return cs;
        }
        st->weight_arena = heap_alloc_aligned(cap, 64);
        if (st->weight_arena == nullptr) {
            geist_backend_set_error(
                    be, GEIST_E_OOM, "transformer: weight arena alloc failed (%zu bytes)", cap);
            transformer_state_destroy(st);
            return GEIST_E_OOM;
        }
        st->weight_arena_capacity = cap;
        st->weight_arena_used     = 0;
    }
    /* mmap-alias mode: leave weight_arena == nullptr. load_tensor_to_buffer
     * branches on this to pick its storage path. */

    enum geist_status s = load_globals(be, gguf, st);
    if (s != GEIST_OK) {
        transformer_state_destroy(st);
        return s;
    }

    for (size_t i = 0; i < (size_t) st->n_layers; i++) {
        st->layers[i].layer_idx = (int) i;
        s                       = load_one_layer(st, gguf, &st->layers[i]);
        if (s != GEIST_OK) {
            transformer_state_destroy(st);
            return s;
        }
    }

    s = allocate_runtime_rope(st);
    if (s != GEIST_OK) {
        transformer_state_destroy(st);
        return s;
    }

    /* P1.2.f: allocate the default session (KV + scratch pool + arena +
     * sampler). transformer_session_alloc reads kv_mode + m_max from
     * opts; AUTO falls back to env / platform default for full backward
     * compat. The default sess is the one installed on direct-state
     * callers (state-only int-tests, internal helpers). Engine-level
     * geist_session_create allocates additional sessions on top. */
    st->default_sess = transformer_session_alloc(st, opts);
    if (st->default_sess == nullptr) {
        /* session_alloc populated the backend error. */
        s = geist_backend_errcode(be);
        if (s == GEIST_OK)
            s = GEIST_E_OOM;
        transformer_state_destroy(st);
        return s;
    }
    st->sess = st->default_sess;

    s = transformer_exec_plan_build(st);
    if (s != GEIST_OK) {
        transformer_state_destroy(st);
        return s;
    }

    /* AWQ (optional): fold attn_norm/ffn_norm gammas and stash per-layer
     * inv-scales for o/down inputs. Runs after weights are mapped but
     * before any forward pass. */
    if (opts != nullptr) {
        s = apply_awq_to_state(st, opts->awq_scales_path);
        if (s != GEIST_OK) {
            transformer_state_destroy(st);
            return s;
        }
    }

    /* β mode: GGUF is fully transpiled into the backend arena by now;
     * drop the mmap. mmap-alias mode keeps it — kernels read weight
     * bytes directly from the mmap pages, so the GGUF must stay open
     * for state lifetime. (Closed in transformer_state_destroy.) */
    if (!mmap_alias_mode && st->gguf != nullptr) {
        gguf_close((struct gguf_ctx *) st->gguf);
        st->gguf = nullptr;
    }

    *out = st;
    return GEIST_OK;
}

enum geist_status transformer_state_create(struct geist_backend            *be,
                                           const char                      *gguf_path,
                                           const struct geist_session_opts *opts,
                                           struct transformer_arch_state  **out) {
    if (be == nullptr || gguf_path == nullptr || out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out                  = nullptr;
    const char      *err  = nullptr;
    struct gguf_ctx *gguf = gguf_open(gguf_path, &err);
    if (gguf == nullptr) {
        geist_backend_set_error(be,
                                GEIST_E_FILE_NOT_FOUND,
                                "transformer: gguf_open(%s): %s",
                                gguf_path,
                                err != nullptr ? err : "(no detail)");
        return GEIST_E_FILE_NOT_FOUND;
    }
    return transformer_state_create_from_gguf(be, gguf, opts, out);
}

enum geist_status transformer_state_create_from_memory(struct geist_backend            *be,
                                                       const void                      *data,
                                                       size_t                           size,
                                                       const struct geist_session_opts *opts,
                                                       struct transformer_arch_state  **out) {
    if (be == nullptr || data == nullptr || out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out                  = nullptr;
    const char      *err  = nullptr;
    struct gguf_ctx *gguf = gguf_open_memory(data, size, &err);
    if (gguf == nullptr) {
        geist_backend_set_error(be,
                                GEIST_E_FORMAT,
                                "transformer: gguf_open_memory: %s",
                                err != nullptr ? err : "(no detail)");
        return GEIST_E_FORMAT;
    }
    return transformer_state_create_from_gguf(be, gguf, opts, out);
}

void transformer_state_destroy(struct transformer_arch_state *st) {
    if (st == nullptr) {
        return;
    }
    struct geist_backend *be = st->backend;

    /* P1.2.f: tear down the default session first — releases its KV
     * buffers + scratch pool + per-forward arena + sampler workspace +
     * the session struct itself. Engine-owned sessions must have been
     * destroyed via geist_session_destroy by now. */
    if (st->default_sess != nullptr) {
        transformer_session_free(st, st->default_sess);
        st->default_sess = nullptr;
        st->sess         = nullptr;
    }

    if (be != nullptr) {
        for (size_t l = 0; l < (size_t) st->n_layers; l++) {
            struct transformer_layer_weights *L = &st->layers[l];
            release_layer_weight_aux(L);
            for (size_t b = 0; b < L->n_bufs; b++) {
                if (L->bufs[b] != nullptr) {
                    be->desc->vtbl->buffer_destroy(be, L->bufs[b]);
                }
            }
            {
                void *p1 = L->o_awq_inv_scale;
                void *p2 = L->down_awq_inv_scale;
                safe_free(&p1);
                safe_free(&p2);
                L->o_awq_inv_scale    = nullptr;
                L->down_awq_inv_scale = nullptr;
            }
        }
        release_weight_aux(&st->embed_table_w);
        release_weight_aux(&st->model_proj_w);
        safe_free((void **) &st->spec_sketch);
        safe_free((void **) &st->spec_row_scale);
        safe_free((void **) &st->spec_x_i8);
        safe_free((void **) &st->spec_act_sketch);
        safe_free((void **) &st->spec_rough);
        safe_free((void **) &st->spec_row_f32);
        safe_free((void **) &st->spec_heap);
        for (size_t b = 0; b < st->n_global_bufs; b++) {
            if (st->global_bufs[b] != nullptr) {
                be->desc->vtbl->buffer_destroy(be, st->global_bufs[b]);
            }
        }
        /* Model-owned RoPE tables. */
        struct geist_buffer *rope[] = {
                st->rope_cos_sliding,
                st->rope_sin_sliding,
                st->rope_cos_full,
                st->rope_sin_full,
        };
        for (size_t i = 0; i < sizeof rope / sizeof rope[0]; i++) {
            if (rope[i] != nullptr) {
                be->desc->vtbl->buffer_destroy(be, rope[i]);
            }
        }
    }
    if (st->gguf != nullptr) {
        gguf_close((struct gguf_ctx *) st->gguf);
    }
    /* P1.1.g: release the weight arena ONCE. All per-weight buffers
     * were GEIST_MEMORY_ALIASED wrappers around slices of this arena
     * — their buffer_destroy already ran above and skipped the free,
     * so the underlying bytes are reachable here exactly. */
    if (st->weight_arena != nullptr) {
        safe_free(&st->weight_arena);
        st->weight_arena_capacity = 0;
        st->weight_arena_used     = 0;
    }
    transformer_exec_plan_destroy(st);
    /* P1.4.c: release the heap-allocated per-layer weight array. */
    if (st->layers != nullptr) {
        void *p_layers = st->layers;
        safe_free(&p_layers);
        st->layers = nullptr;
    }
    void *p = st;
    safe_free(&p);
}

/* ---- Multi-session API (P1.2.f) --------------------------------------- */

[[nodiscard]] static enum geist_kv_mode resolve_kv_mode(const struct geist_session_opts *opts) {
    enum geist_kv_mode m = (opts != nullptr) ? opts->kv_mode : GEIST_KV_AUTO;
    if (m != GEIST_KV_AUTO) {
        return m;
    }
    const char *env_kivi = getenv("GEIST_KV_KIVI");
    const char *env_int8 = getenv("GEIST_KV_INT8");
    if (env_kivi != nullptr && env_kivi[0] == '1') {
        return GEIST_KV_KIVI;
    }
    if (env_int8 != nullptr) {
        return (env_int8[0] == '1') ? GEIST_KV_INT8 : GEIST_KV_FP32;
    }
#if defined(__APPLE__)
    return GEIST_KV_FP32;
#else
    return GEIST_KV_INT8;
#endif
}

struct transformer_arch_session *transformer_session_alloc(struct transformer_arch_state   *state,
                                                           const struct geist_session_opts *opts) {
    if (state == nullptr) {
        return nullptr;
    }
    struct geist_backend *be = state->backend;

    struct transformer_arch_session *sess =
            heap_alloc_aligned(sizeof(*sess), alignof(struct transformer_arch_session));
    if (sess == nullptr) {
        geist_backend_set_error(
                be, GEIST_E_OOM, "transformer_session_alloc: %zu bytes failed", sizeof(*sess));
        return nullptr;
    }
    memset(sess, 0, sizeof(*sess));
    sess->m_max = (opts != nullptr && opts->m_max > 0) ? opts->m_max : state->m_max;
    /* GEIST_QUANT_M_CAP guards CPU quant-kernel stack arrays; metal's
     * prefill never runs those, so its sessions may batch up to 512. */
    const size_t m_cap =
            (be->desc != nullptr && be->desc->name != nullptr &&
             strcmp(be->desc->name, "metal") == 0)
                    ? 512u
                    : (size_t) GEIST_QUANT_M_CAP;
    if (sess->m_max == 0 || sess->m_max > m_cap) {
        geist_backend_set_error(
                be,
                GEIST_E_INVALID_ARG,
                "transformer_session_alloc: m_max=%zu outside supported range 1..%zu",
                sess->m_max,
                m_cap);
        void *p_sess = sess;
        safe_free(&p_sess);
        return nullptr;
    }

    /* P1.4.c: heap-allocate the 14 per-layer KV slot arrays. One
     * combined allocation, partitioned across the 14 pointer-array
     * slots; freed in one safe_free at session_free. Sized to the
     * model's actual layer count, not a compile-time cap. */
    const size_t          n_layers = state->n_layers;
    const size_t          kv_slots = 14;
    const size_t          kv_bytes = kv_slots * n_layers * sizeof(struct geist_buffer *);
    struct geist_buffer **kv_block = heap_alloc_aligned(kv_bytes, alignof(struct geist_buffer *));
    if (kv_block == nullptr) {
        geist_backend_set_error(be,
                                GEIST_E_OOM,
                                "transformer_session_alloc: KV slot array "
                                "alloc failed (%zu × %zu bytes)",
                                kv_slots,
                                n_layers * sizeof(struct geist_buffer *));
        void *p_sess = sess;
        safe_free(&p_sess);
        return nullptr;
    }
    memset(kv_block, 0, kv_bytes);
    sess->k_cache       = kv_block + 0 * n_layers;
    sess->v_cache       = kv_block + 1 * n_layers;
    sess->k_cache_q8    = kv_block + 2 * n_layers;
    sess->v_cache_q8    = kv_block + 3 * n_layers;
    sess->k_cache_scale = kv_block + 4 * n_layers;
    sess->v_cache_scale = kv_block + 5 * n_layers;
    sess->k_kivi_q      = kv_block + 6 * n_layers;
    sess->v_kivi_q      = kv_block + 7 * n_layers;
    sess->k_kivi_scales = kv_block + 8 * n_layers;
    sess->k_kivi_zeros  = kv_block + 9 * n_layers;
    sess->v_kivi_scales = kv_block + 10 * n_layers;
    sess->v_kivi_zeros  = kv_block + 11 * n_layers;
    sess->k_residual    = kv_block + 12 * n_layers;
    sess->v_residual    = kv_block + 13 * n_layers;

    /* KV-mode resolution: opts override > env > platform default. */
    const enum geist_kv_mode mode = resolve_kv_mode(opts);
    sess->kv_kivi_enabled         = (mode == GEIST_KV_KIVI);
    sess->kv_int8_enabled         = (mode == GEIST_KV_INT8);
    /* F16 cache: explicit request, or AUTO-resolved FP32 upgraded when the
     * backend has the fused converting append (env GEIST_KV_F16=0 forces
     * FP32, =1 requests it under AUTO). Without the slot F16 silently
     * degrades to FP32 — no host-side half-float path exists. */
    {
        const bool slot_ok = state->backend != nullptr &&
                             state->backend->desc->vtbl->kv_append_f16 != nullptr;
        const char *env_f16 = getenv("GEIST_KV_F16");
        bool want = mode == GEIST_KV_F16;
        if (mode == GEIST_KV_FP32 &&
            (opts == nullptr || opts->kv_mode == GEIST_KV_AUTO)) {
            want = env_f16 == nullptr || env_f16[0] != '0';
        }
        sess->kv_f16_enabled = want && slot_ok &&
                               !(env_f16 != nullptr && env_f16[0] == '0');
    }
    transformer_session_exec_plan_build(sess);
    sess->kivi_residual_count = 0;
    sess->kivi_drained_count  = 0;

    /* Sampler defaults (greedy); apply_opts may override below. */
    sess->temperature = 0.0f;
    sess->top_p       = 1.0f;
    sess->top_k       = 0;
    sess->sampler_ws  = (struct geist_sampler_workspace) {0};
    geist_rng_seed(&sess->rng, 0xCAFEBABE1234ULL);

    /* Temporarily install so the alloc helpers (alloc_pool_buffer /
     * alloc_scratch) mutate this session's scratch_pool_* slots, not
     * any previously-attached session's. Restore the previous active
     * session before returning — engine binds the new one explicitly
     * via session_attach. */
    struct transformer_arch_session *prev = state->sess;
    state->sess                           = sess;

    enum geist_status s = allocate_runtime_session(state);
    if (s == GEIST_OK && opts != nullptr) {
        transformer_state_apply_opts(state, opts);
    }
    state->sess = prev;

    if (s != GEIST_OK) {
        transformer_session_free(state, sess);
        return nullptr;
    }
    return sess;
}

void transformer_session_free(struct transformer_arch_state   *state,
                              struct transformer_arch_session *sess) {
    if (sess == nullptr) {
        return;
    }
    struct geist_backend *be = (state != nullptr) ? state->backend : nullptr;

    /* Per-layer KV cache buffers. Each slot may be NULL — exactly one
     * representation (FP32 / INT8 / KIVI) was allocated per non-shared
     * layer at session_alloc time. */
    if (be != nullptr) {
        for (size_t li = 0; li < (size_t) state->n_layers; li++) {
            struct geist_buffer *kv_bufs[] = {
                    sess->k_cache[li],
                    sess->v_cache[li],
                    sess->k_cache_q8[li],
                    sess->v_cache_q8[li],
                    sess->k_cache_scale[li],
                    sess->v_cache_scale[li],
                    sess->k_kivi_q[li],
                    sess->v_kivi_q[li],
                    sess->k_kivi_scales[li],
                    sess->k_kivi_zeros[li],
                    sess->v_kivi_scales[li],
                    sess->v_kivi_zeros[li],
                    sess->k_residual[li],
                    sess->v_residual[li],
            };
            for (size_t i = 0; i < sizeof kv_bufs / sizeof kv_bufs[0]; i++) {
                if (kv_bufs[i] != nullptr) {
                    be->desc->vtbl->buffer_destroy(be, kv_bufs[i]);
                }
            }
        }
        /* Scratch pool aliased buffers — buffer_destroy releases the
         * metadata; the underlying scratch_pool_base bytes are freed
         * once below. */
        struct geist_buffer *infra[] = {
                sess->scratch_normed,
                sess->scratch_q,
                sess->scratch_k,
                sess->scratch_v,
                sess->scratch_attn,
                sess->scratch_o,
                sess->scratch_post_attn,
                sess->scratch_h_post_attn,
                sess->scratch_pre_ff,
                sess->scratch_gate,
                sess->scratch_up,
                sess->scratch_ffn_out,
                sess->scratch_post_ff,
                sess->scratch_h_post_ff,
                sess->scratch_gate_ple,
                sess->scratch_proj_ple,
                sess->scratch_ones_headdim_max,
                sess->scratch_ple_lookup,
                sess->scratch_h_a,
                sess->scratch_h_b,
                sess->scratch_per_layer_input,
                sess->scratch_logits,
        };
        for (size_t i = 0; i < sizeof infra / sizeof infra[0]; i++) {
            if (infra[i] != nullptr) {
                be->desc->vtbl->buffer_destroy(be, infra[i]);
            }
        }
    }

    /* Per-forward arena backing store. */
    if (sess->scratch_arena_base != nullptr) {
        safe_free(&sess->scratch_arena_base);
        sess->scratch_arena_bytes = 0;
        sess->scratch_arena       = (struct frame_arena) {0};
    }
    /* Scratch pool backing store. */
    if (sess->scratch_pool_base != nullptr) {
        safe_free(&sess->scratch_pool_base);
        sess->scratch_pool_bytes = 0;
        sess->scratch_pool_used  = 0;
    }
    geist_sampler_workspace_destroy(&sess->sampler_ws);

    /* P1.4.c: release the combined 14-slot KV pointer block. The
     * underlying geist_buffer headers were destroyed above; this just
     * reclaims the pointer-array slab. k_cache happens to be the base
     * pointer (slots 0..n_layers-1 of the block); the other 13 are
     * slices that point further into the same allocation. */
    if (sess->k_cache != nullptr) {
        void *p_kv = sess->k_cache;
        safe_free(&p_kv);
        sess->k_cache       = nullptr;
        sess->v_cache       = nullptr;
        sess->k_cache_q8    = nullptr;
        sess->v_cache_q8    = nullptr;
        sess->k_cache_scale = nullptr;
        sess->v_cache_scale = nullptr;
        sess->k_kivi_q      = nullptr;
        sess->v_kivi_q      = nullptr;
        sess->k_kivi_scales = nullptr;
        sess->k_kivi_zeros  = nullptr;
        sess->v_kivi_scales = nullptr;
        sess->v_kivi_zeros  = nullptr;
        sess->k_residual    = nullptr;
        sess->v_residual    = nullptr;
    }

    /* If this session was the currently-active one, detach. The caller
     * (typically engine session_destroy) usually re-attaches the model
     * default first, but defensively clear the slot if we're freeing it
     * directly. */
    if (state != nullptr && state->sess == sess) {
        state->sess = state->default_sess;
    }

    void *sp = sess;
    safe_free(&sp);
}

void transformer_session_attach(struct transformer_arch_state   *state,
                                struct transformer_arch_session *sess) {
    if (state == nullptr) {
        return;
    }
    state->sess = (sess != nullptr) ? sess : state->default_sess;
}

struct transformer_arch_session *transformer_default_session(struct transformer_arch_state *state) {
    return (state != nullptr) ? state->default_sess : nullptr;
}

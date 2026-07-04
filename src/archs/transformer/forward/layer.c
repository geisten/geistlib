/*
 * src/archs/transformer/forward/layer.c — per-layer forward pass.
 *
 * Layer: ARCHITECTURE.
 *
 * Per-layer forward orchestration and per-layer-input precompute.
 * Extracted from forward.c during R4 of the C23/AGENT.md cleanup.
 * Contains:
 *
 *   transformer_forward_one_layer       — one full transformer block
 *   transformer_compute_per_layer_input — single-token PLE precompute
 *   compute_per_layer_inputs_batch      — batched PLE precompute (M>1)
 *   dequant_one_row                     — single-row PLE table dequant
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "internal.h"
#include "../arch_state.h"
#include "../forward.h"

#include "quant.h"
#include <geist.h>
#include <geist_backend.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum transformer_profile_stage {
    TRANSFORMER_PROFILE_ATTENTION = 0,
    TRANSFORMER_PROFILE_FFN,
    TRANSFORMER_PROFILE_PLE,
    TRANSFORMER_PROFILE_SCALE,
    TRANSFORMER_PROFILE_COUNT,
};

static uint64_t g_transformer_profile_ns[TRANSFORMER_PROFILE_COUNT];
static uint64_t g_transformer_profile_calls[TRANSFORMER_PROFILE_COUNT];

static uint64_t transformer_profile_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ULL + (uint64_t) ts.tv_nsec;
}

static void transformer_profile_print(void) {
    static const char *names[TRANSFORMER_PROFILE_COUNT] = {
            "attention",
            "ffn",
            "ple",
            "scale",
    };
    uint64_t total = 0;
    for (size_t i = 0; i < TRANSFORMER_PROFILE_COUNT; i++) {
        total += g_transformer_profile_ns[i];
    }
    if (total == 0) {
        return;
    }

    fprintf(stderr, "transformer profile:\n");
    for (size_t i = 0; i < TRANSFORMER_PROFILE_COUNT; i++) {
        const double ms  = (double) g_transformer_profile_ns[i] / 1000000.0;
        const double pct = 100.0 * (double) g_transformer_profile_ns[i] / (double) total;
        fprintf(stderr,
                "  %-10s %10.2f ms  %5.1f%%  (%llu calls)\n",
                names[i],
                ms,
                pct,
                (unsigned long long) g_transformer_profile_calls[i]);
    }
}

static bool transformer_profile_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *env = getenv("GEIST_PROFILE_PREFILL");
        if (env == nullptr || env[0] == '\0') {
            env = getenv("GEIST_PROFILE_FORWARD");
        }
        enabled = (env != nullptr && env[0] == '1') ? 1 : 0;
        if (enabled) {
            atexit(transformer_profile_print);
        }
    }
    return enabled != 0;
}

static void transformer_profile_add(enum transformer_profile_stage stage, uint64_t t0) {
    if (t0 == 0) {
        return;
    }
    g_transformer_profile_ns[stage] += transformer_profile_now_ns() - t0;
    g_transformer_profile_calls[stage]++;
}

static void transformer_layer_bind_kv_buffers(struct transformer_layer_forward_ctx *ctx) {

    struct transformer_arch_session *sess   = ctx->st->sess;
    const int                        kv_src = ctx->kv_src;
    ctx->k_cache_buf                        = sess->k_cache[kv_src];
    ctx->v_cache_buf                        = sess->v_cache[kv_src];
    ctx->k_cache_q8_buf                     = sess->k_cache_q8[kv_src];
    ctx->v_cache_q8_buf                     = sess->v_cache_q8[kv_src];
    ctx->k_cache_scale_buf                  = sess->k_cache_scale[kv_src];
    ctx->v_cache_scale_buf                  = sess->v_cache_scale[kv_src];
    ctx->k_kivi_q_buf                       = sess->k_kivi_q[kv_src];
    ctx->v_kivi_q_buf                       = sess->v_kivi_q[kv_src];
    ctx->k_kivi_scales_buf                  = sess->k_kivi_scales[kv_src];
    ctx->k_kivi_zeros_buf                   = sess->k_kivi_zeros[kv_src];
    ctx->v_kivi_scales_buf                  = sess->v_kivi_scales[kv_src];
    ctx->v_kivi_zeros_buf                   = sess->v_kivi_zeros[kv_src];
    ctx->k_residual_buf                     = sess->k_residual[kv_src];
    ctx->v_residual_buf                     = sess->v_residual[kv_src];
}

static void transformer_layer_ctx_init(struct transformer_layer_forward_ctx *ctx,
                                       struct transformer_arch_state        *st,
                                       int                                   layer_idx,
                                       size_t                                q_position,
                                       size_t                                seq,
                                       bool                                  advance_kv,
                                       struct geist_buffer                  *h_in_buf,
                                       struct geist_buffer                  *per_layer_input_buf,
                                       struct geist_buffer                  *h_out_buf) {

    memset(ctx, 0, sizeof(*ctx));
    struct transformer_layer_weights         *L = &st->layers[layer_idx];
    const struct transformer_layer_exec_plan *P =
            st->layer_plans != nullptr ? &st->layer_plans[layer_idx] : nullptr;
    const bool plan_apply_sub_ln = P != nullptr ? P->apply_sub_ln : st->config.has_sub_ln;

    ctx->st                  = st;
    ctx->be                  = st->backend;
    ctx->v                   = st->backend->desc->vtbl;
    ctx->L                   = L;
    ctx->P                   = P;
    ctx->SP                  = &st->sess->exec_plan;
    ctx->layer_idx           = layer_idx;
    ctx->q_position          = q_position;
    ctx->seq                 = seq;
    ctx->advance_kv          = advance_kv;
    ctx->h_in_buf            = h_in_buf;
    ctx->per_layer_input_buf = per_layer_input_buf;
    ctx->h_out_buf           = h_out_buf;
    ctx->kv_src     = P != nullptr ? P->kv_src
                                   : (L->is_kv_shared ? (L->is_full ? st->config.kv_full_src
                                                                    : st->config.kv_sliding_src)
                                                      : layer_idx);
    ctx->compute_kv = P != nullptr ? P->compute_kv : !L->is_kv_shared;
    ctx->apply_bitnet_input_quant = plan_apply_sub_ln;
    ctx->apply_sub_ln             = plan_apply_sub_ln && st->runtime_flags.bitnet_sub_ln_enabled;
    ctx->apply_gemma_attn_norms =
            P != nullptr ? P->apply_gemma_attn_norms : st->config.has_gemma_attn_norms;
    ctx->rope_interleaved = P != nullptr ? P->rope_interleaved : st->config.rope_interleaved;
    ctx->apply_ple        = P != nullptr ? P->apply_ple : st->config.has_ple;
    ctx->kv_int8_enabled  = ctx->SP->kv_int8_enabled;
    ctx->kv_kivi_enabled  = ctx->SP->kv_kivi_enabled;
    ctx->kv_f16_enabled   = ctx->SP->kv_f16_enabled;
    ctx->ffn_activation   = P != nullptr ? P->ffn_activation : st->config.ffn_activation;
    ctx->eps              = st->config.rms_eps;
    ctx->hd               = L->head_dim;
    ctx->q_out            = L->q_out;
    ctx->kv_out           = L->kv_out;
    ctx->inter            = L->intermediate;
    ctx->SEQ              = (int64_t) seq;
    ctx->kv_len_now       = q_position + seq;
    transformer_layer_bind_kv_buffers(ctx);
}

enum geist_status transformer_forward_one_layer(struct transformer_arch_state *st,
                                                int                            layer_idx,
                                                size_t                         q_position,
                                                size_t                         seq,
                                                bool                           advance_kv,
                                                struct geist_buffer           *h_in_buf,
                                                struct geist_buffer           *per_layer_input_buf,
                                                struct geist_buffer           *h_out_buf) {
    if (st == nullptr || layer_idx < 0 || (size_t) layer_idx >= st->n_layers ||
        h_in_buf == nullptr || h_out_buf == nullptr || seq == 0 || seq > st->sess->m_max) {
        return GEIST_E_INVALID_ARG;
    }

    struct transformer_layer_forward_ctx ctx;
    transformer_layer_ctx_init(&ctx,
                               st,
                               layer_idx,
                               q_position,
                               seq,
                               advance_kv,
                               h_in_buf,
                               per_layer_input_buf,
                               h_out_buf);

    frame_arena_reset(&st->sess->scratch_arena);

    const transformer_layer_stage_fn run_attention =
            ctx.P != nullptr && ctx.P->run_attention_block != nullptr
                    ? ctx.P->run_attention_block
                    : transformer_layer_run_attention_block;
    const transformer_layer_stage_fn run_ffn = ctx.P != nullptr && ctx.P->run_ffn_block != nullptr
                                                       ? ctx.P->run_ffn_block
                                                       : transformer_layer_run_ffn_block;
    const transformer_layer_stage_fn run_ple = ctx.P != nullptr && ctx.P->run_ple_or_copy != nullptr
                                                       ? ctx.P->run_ple_or_copy
                                                       : transformer_layer_run_ple_or_copy;

    const bool        profile = transformer_profile_enabled();
    uint64_t          t0      = profile ? transformer_profile_now_ns() : 0;
    enum geist_status s       = run_attention(&ctx);
    transformer_profile_add(TRANSFORMER_PROFILE_ATTENTION, t0);
    if (s != GEIST_OK) {
        return s;
    }
    t0 = profile ? transformer_profile_now_ns() : 0;
    s  = run_ffn(&ctx);
    transformer_profile_add(TRANSFORMER_PROFILE_FFN, t0);
    if (s != GEIST_OK) {
        return s;
    }
    t0 = profile ? transformer_profile_now_ns() : 0;
    s  = run_ple(&ctx);
    transformer_profile_add(TRANSFORMER_PROFILE_PLE, t0);
    if (s != GEIST_OK) {
        return s;
    }

    t0 = profile ? transformer_profile_now_ns() : 0;
    transformer_layer_scale_output(&ctx);
    transformer_profile_add(TRANSFORMER_PROFILE_SCALE, t0);
    if (ctx.advance_kv) {
        st->sess->kv_len = ctx.kv_len_now;
    }
    return GEIST_OK;
}

/* ---- PLE per-layer-input precompute ----------------------------------- */

/* Dequantize one row of a 2D weight tensor (by row index) into a host
 * float buffer. Used for the PLE table whose full FP32 expansion (262144
 * rows × 8960 = 9.4 GB) doesn't fit in memory on small targets. */
[[nodiscard]] enum geist_status dequant_one_row(struct geist_backend      *be,
                                                const struct geist_tensor *t,
                                                size_t                     row_idx,
                                                float                     *dst) {

    const uint8_t *raw = (const uint8_t *) be->desc->vtbl->buffer_map(t->buffer);
    if (raw == nullptr) {
        geist_backend_set_error(
                be, GEIST_E_BACKEND, "transformer: buffer_map failed for PLE table");
        return GEIST_E_BACKEND;
    }
    const size_t      n_in = (size_t) t->shape[1];
    enum geist_status rc   = GEIST_OK;
    switch (t->dtype) {
    case GEIST_DTYPE_F32:
        memcpy(dst, raw + row_idx * n_in * sizeof(float), n_in * sizeof(float));
        break;
    case GEIST_DTYPE_F16: {
        const uint8_t *r = raw + row_idx * n_in * 2;
        for (size_t i = 0; i < n_in; i++) {
            uint16_t h = (uint16_t) r[2 * i] | ((uint16_t) r[2 * i + 1] << 8);
            dst[i]     = fp16_to_fp32(h);
        }
        break;
    }
    case GEIST_DTYPE_BF16: {
        const uint8_t *r = raw + row_idx * n_in * 2;
        for (size_t i = 0; i < n_in; i++) {
            uint16_t b = (uint16_t) r[2 * i] | ((uint16_t) r[2 * i + 1] << 8);
            uint32_t f = (uint32_t) b << 16;
            memcpy(&dst[i], &f, sizeof f);
        }
        break;
    }
    case GEIST_DTYPE_Q3_K:
        dequant_q3_K_row(raw + row_idx * n_in / Q3_K_BLOCK_ELEMS * Q3_K_BLOCK_BYTES, dst, n_in);
        break;
    case GEIST_DTYPE_Q4_K:
        dequant_q4_K_row(raw + row_idx * n_in / Q4_K_BLOCK_ELEMS * Q4_K_BLOCK_BYTES, dst, n_in);
        break;
    case GEIST_DTYPE_Q5_K:
        dequant_q5_K_row(raw + row_idx * n_in / Q5_K_BLOCK_ELEMS * Q5_K_BLOCK_BYTES, dst, n_in);
        break;
    case GEIST_DTYPE_Q6_K:
        dequant_q6_K_row(raw + row_idx * n_in / Q6_K_BLOCK_ELEMS * Q6_K_BLOCK_BYTES, dst, n_in);
        break;
    case GEIST_DTYPE_Q8_0:
        dequant_q8_0_row(raw + row_idx * n_in / Q8_0_BLOCK_ELEMS * Q8_0_BLOCK_BYTES, dst, n_in);
        break;
    case GEIST_DTYPE_IQ2_S:
        dequant_iq2_s_row(raw + row_idx * n_in / IQ2_S_BLOCK_ELEMS * IQ2_S_BLOCK_BYTES, dst, n_in);
        break;
    case GEIST_DTYPE_IQ3_S:
        dequant_iq3_s_row(raw + row_idx * n_in / IQ3_S_BLOCK_ELEMS * IQ3_S_BLOCK_BYTES, dst, n_in);
        break;
    case GEIST_DTYPE_I2_S: {
        /* BitNet i2_s: 256-elem/64-byte ternary blocks, reversed in-byte field
         * order vs TQ2_0, ONE f32 per-TENSOR scale at the tail (offset
         * total_elems/4). Used for the token-embedding table on BitNet-2B-4T. */
        const size_t total = (size_t) t->shape[0] * (size_t) t->shape[1];
        float        scale;
        memcpy(&scale, raw + total / 4, sizeof scale);
        const uint8_t *row = raw + row_idx * (n_in / 4);
        for (size_t b = 0; b < n_in / 256; b++) {
            const uint8_t *qs = row + b * 64;
            for (size_t h = 0; h < 2; h++) {
                for (size_t bb = 0; bb < 32; bb++) {
                    const uint8_t byte = qs[h * 32 + bb];
                    for (size_t g = 0; g < 4; g++) {
                        const int trit = (int) ((byte >> (6 - 2 * g)) & 3) - 1;
                        dst[b * 256 + h * 128 + g * 32 + bb] = (float) trit * scale;
                    }
                }
            }
        }
        break;
    }
    default:
        geist_backend_set_error(be,
                                GEIST_E_UNSUPPORTED,
                                "transformer: unsupported dtype %d for row dequant",
                                (int) t->dtype);
        rc = GEIST_E_UNSUPPORTED;
    }
    be->desc->vtbl->buffer_unmap(t->buffer);
    return rc;
}

enum geist_status transformer_compute_per_layer_input(struct transformer_arch_state *st,
                                                      geist_token_t                  token_id,
                                                      struct geist_buffer           *h_buf,
                                                      struct geist_buffer *per_layer_input_buf) {
    if (st == nullptr || h_buf == nullptr || per_layer_input_buf == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    if (token_id < 0 || (size_t) token_id >= (size_t) st->vocab_size) {
        return GEIST_E_INVALID_ARG;
    }
    struct geist_backend            *be = st->backend;
    const struct geist_backend_vtbl *v  = be->desc->vtbl;
    enum geist_status                s;

    /* 1. Dequant one row of the PLE table into scratch_ple_lookup, then
     *    multiply by PLE_TABLE_SCALE (16). */
    {
        bool on_device = false;
        if (v->embedding_lookup_scaled != nullptr) {
            struct geist_tensor t_row =
                    view_1d(st->sess->scratch_ple_lookup, st->ple_out);
            on_device = v->embedding_lookup_scaled(
                                be, &st->ple_table, token_id,
                                st->config.ple_table_scale, &t_row) == GEIST_OK;
        }
        if (!on_device) {
            float *dst = (float *) v->buffer_map(st->sess->scratch_ple_lookup);
            s          = dequant_one_row(be, &st->ple_table, (size_t) token_id, dst);
            if (s != GEIST_OK) {
                v->buffer_unmap(st->sess->scratch_ple_lookup);
                return s;
            }
            for (size_t i = 0; i < (size_t) st->ple_out; i++) {
                dst[i] *= st->config.ple_table_scale;
            }
            v->buffer_unmap(st->sess->scratch_ple_lookup);
        }
    }

    /* 2. linear(h, model_proj) → per_layer_input (reused as scratch).
     *    Shape: [1, HIDDEN] × [PLE_OUT, HIDDEN]^T → [1, PLE_OUT].
     * P1.1.e: model_proj is F32 dense → cblas trampoline via the
     * pre-resolved kernel pointer. */
    struct geist_tensor t_h_2d        = view_2d(h_buf, 1, st->d_model);
    struct geist_tensor t_ple_proj_2d = view_2d(per_layer_input_buf, 1, st->ple_out);
    s                                 = linear_w_or_legacy(be,
                                                           v,
                                                           h_buf,
                                                           per_layer_input_buf,
                                                           &st->model_proj_w,
                                                           /* seq = */ 1,
                                                           &t_h_2d,
                                                           &st->model_proj,
                                                           &t_ple_proj_2d);
    if (s != GEIST_OK) {
        return s;
    }

    /* 3. *= PLE_MODEL_PROJ_SCALE (in-place; device op keeps batched GPU
     *    backends from flushing for a host loop). */
    {
        struct geist_tensor t_all = view_1d(per_layer_input_buf, st->ple_out);
        if (v->scale_f32 == nullptr ||
            v->scale_f32(be, &t_all, st->config.ple_model_proj_scale,
                         &t_all) != GEIST_OK) {
            float *p = (float *) v->buffer_map(per_layer_input_buf);
            for (size_t i = 0; i < (size_t) st->ple_out; i++) {
                p[i] *= st->config.ple_model_proj_scale;
            }
            v->buffer_unmap(per_layer_input_buf);
        }
    }

    /* 4. rmsnorm as [NUM_LAYERS, HIDDEN_PER_LAYER] with model_proj_norm. */
    struct geist_tensor t_ple_proj_2dN =
            view_2d(per_layer_input_buf, st->n_layers, st->hidden_per_layer);
    struct geist_tensor t_norm_w = view_1d(st->model_proj_norm.buffer, st->hidden_per_layer);
    s = v->rmsnorm(be, &t_ple_proj_2dN, &t_norm_w, st->config.rms_eps, &t_ple_proj_2dN);
    if (s != GEIST_OK) {
        return s;
    }

    /* 5. per_layer_input = (ple_proj + ple_lookup) * PLE_INPUT_SCALE. */
    struct geist_tensor t_ple_proj_1d   = view_1d(per_layer_input_buf, st->ple_out);
    struct geist_tensor t_ple_lookup_1d = view_1d(st->sess->scratch_ple_lookup, st->ple_out);
    s = v->add(be, &t_ple_proj_1d, &t_ple_lookup_1d, &t_ple_proj_1d);
    if (s != GEIST_OK) {
        return s;
    }
    {
        struct geist_tensor t_all = view_1d(per_layer_input_buf, st->ple_out);
        if (v->scale_f32 == nullptr ||
            v->scale_f32(be, &t_all, st->config.ple_input_scale, &t_all) !=
                GEIST_OK) {
            float *p = (float *) v->buffer_map(per_layer_input_buf);
            for (size_t i = 0; i < (size_t) st->ple_out; i++) {
                p[i] *= st->config.ple_input_scale;
            }
            v->buffer_unmap(per_layer_input_buf);
        }
    }
    return GEIST_OK;
}

/* ---- KIVI drain across all non-shared layers -------------------------- *
 *
 * Called after committed writes when residual_count ≥ R. Drains as many
 * groups as fit; updates shared counters once at the end. Each layer's
 * buffers are mapped/packed/unmapped independently — at most one drain
 * group per outer iteration. No-op if KIVI is off or residual not yet
 * full. */

/* ---- Batched PLE precompute ------------------------------------------- *
 *
 * Batched version of transformer_compute_per_layer_input. Processes n
 * tokens at once:
 *
 *   1. Dequant n PLE table rows into scratch_ple_lookup [n, PLE_OUT].
 *   2. Multiply by PLE_TABLE_SCALE (16) in-place.
 *   3. linear(h [n, HIDDEN], model_proj) → out_buf [n, PLE_OUT].
 *   4. Multiply by PLE_MODEL_PROJ_SCALE in-place.
 *   5. rmsnorm shape [n * NUM_LAYERS, HIDDEN_PER_LAYER].
 *   6. out_buf += ple_lookup then *= PLE_INPUT_SCALE.
 *
 * Caller is responsible for n <= st->m_max. */
/* Once-per-chunk PLE precompute sub-stage profiler (GEIST_PROFILE_PREFILL=1).
 * Self-contained (the main layer.c profiler has fixed buckets); splits the
 * otherwise-hidden compute_per_layer_inputs_batch cost into its steps. */
enum plepre_stage {
    PLEPRE_GATHER = 0, /* Q5_K dequant of the per-layer-token-embd rows */
    PLEPRE_MODEL_PROJ, /* the BF16/F32 model_proj matmul */
    PLEPRE_SCALE,
    PLEPRE_RMSNORM,
    PLEPRE_COMBINE,
    PLEPRE_COUNT,
};
static uint64_t          g_plepre_ns[PLEPRE_COUNT];
static uint64_t          g_plepre_calls[PLEPRE_COUNT];
static const char *const g_plepre_names[PLEPRE_COUNT] = {
        "gather_q5k",
        "model_proj",
        "scale",
        "rmsnorm",
        "combine",
};

static void plepre_print(void) {
    uint64_t total = 0;
    for (size_t i = 0; i < PLEPRE_COUNT; i++)
        total += g_plepre_ns[i];
    if (total == 0)
        return;
    fprintf(stderr, "transformer ple precompute (per-chunk):\n");
    for (size_t i = 0; i < PLEPRE_COUNT; i++) {
        fprintf(stderr,
                "  %-12s %10.2f ms  %5.1f%%  (%llu calls)\n",
                g_plepre_names[i],
                (double) g_plepre_ns[i] / 1e6,
                100.0 * (double) g_plepre_ns[i] / (double) total,
                (unsigned long long) g_plepre_calls[i]);
    }
}

static bool plepre_enabled(void) {
    static int en = -1;
    if (en < 0) {
        const char *e = getenv("GEIST_PROFILE_PREFILL");
        if (e == nullptr || e[0] == '\0')
            e = getenv("GEIST_PROFILE_FORWARD");
        en = (e != nullptr && e[0] == '1') ? 1 : 0;
        if (en)
            atexit(plepre_print);
    }
    return en != 0;
}

static void plepre_add(enum plepre_stage stage, uint64_t t0) {
    if (t0 == 0)
        return;
    g_plepre_ns[stage] += transformer_profile_now_ns() - t0;
    g_plepre_calls[stage]++;
}

[[nodiscard]] enum geist_status compute_per_layer_inputs_batch(struct transformer_arch_state *st,
                                                               size_t                         n,
                                                               const geist_token_t *ple_ids,
                                                               struct geist_buffer *h_buf,
                                                               struct geist_buffer *out_buf) {

    if (n == 0 || n > st->sess->m_max) {
        return GEIST_E_INVALID_ARG;
    }
    struct geist_backend            *be      = st->backend;
    const struct geist_backend_vtbl *v       = be->desc->vtbl;
    const size_t                     PLE_OUT = (size_t) st->ple_out;
    const bool                       prof    = plepre_enabled();
    uint64_t                         t0;

    /* 1+2. Dequant n PLE rows + scale by 16. Device path: per-row fused
     * lookup+scale dispatches — no host dequant, no pipeline flush from
     * mapping the scratch mid-batch. */
    t0 = prof ? transformer_profile_now_ns() : 0;
    bool gather_on_device = v->embedding_lookup_scaled != nullptr;
    if (gather_on_device) {
        for (size_t t = 0; t < n; t++) {
            struct geist_tensor t_row = {
                    .buffer = st->sess->scratch_ple_lookup,
                    .offset = t * PLE_OUT * sizeof(float),
                    .dtype  = GEIST_DTYPE_F32,
                    .layout = GEIST_LAYOUT_DENSE,
                    .ndim   = 1,
                    .shape  = {(int64_t) PLE_OUT, 0, 0, 0, 0, 0, 0, 0},
                    .stride = {1, 0, 0, 0, 0, 0, 0, 0},
            };
            if (v->embedding_lookup_scaled(be, &st->ple_table, ple_ids[t],
                                           st->config.ple_table_scale,
                                           &t_row) != GEIST_OK) {
                gather_on_device = false;
                break;
            }
        }
    }
    if (!gather_on_device) {
        float *dst = (float *) v->buffer_map(st->sess->scratch_ple_lookup);
        for (size_t t = 0; t < n; t++) {
            enum geist_status s =
                    dequant_one_row(be, &st->ple_table, (size_t) ple_ids[t], dst + t * PLE_OUT);
            if (s != GEIST_OK) {
                v->buffer_unmap(st->sess->scratch_ple_lookup);
                return s;
            }
        }
        for (size_t i = 0; i < n * PLE_OUT; i++) {
            dst[i] *= st->config.ple_table_scale;
        }
        v->buffer_unmap(st->sess->scratch_ple_lookup);
    }
    plepre_add(PLEPRE_GATHER, t0);

    /* 3. linear(h, model_proj) → out_buf.
     * P1.1.e: model_proj F32 dense → cblas trampoline. */
    struct geist_tensor t_h_2d   = view_2d(h_buf, (int64_t) n, st->d_model);
    struct geist_tensor t_out_2d = view_2d(out_buf, (int64_t) n, (int64_t) PLE_OUT);
    t0                           = prof ? transformer_profile_now_ns() : 0;
    enum geist_status s          = linear_w_or_legacy(
            be, v, h_buf, out_buf, &st->model_proj_w, n, &t_h_2d, &st->model_proj, &t_out_2d);
    plepre_add(PLEPRE_MODEL_PROJ, t0);
    if (s != GEIST_OK) {
        return s;
    }

    /* 4. *= PLE_MODEL_PROJ_SCALE. */
    t0 = prof ? transformer_profile_now_ns() : 0;
    if (v->scale_f32 == nullptr ||
        v->scale_f32(be, &t_out_2d, st->config.ple_model_proj_scale,
                     &t_out_2d) != GEIST_OK) {
        float *p = (float *) v->buffer_map(out_buf);
        for (size_t i = 0; i < n * PLE_OUT; i++) {
            p[i] *= st->config.ple_model_proj_scale;
        }
        v->buffer_unmap(out_buf);
    }
    plepre_add(PLEPRE_SCALE, t0);

    /* 5. rmsnorm [n * NUM_LAYERS, HIDDEN_PER_LAYER]. */
    struct geist_tensor t_out_norm =
            view_2d(out_buf, (int64_t) (n * st->n_layers), st->hidden_per_layer);
    struct geist_tensor t_w = view_1d(st->model_proj_norm.buffer, st->hidden_per_layer);
    t0                      = prof ? transformer_profile_now_ns() : 0;
    s                       = v->rmsnorm(be, &t_out_norm, &t_w, st->config.rms_eps, &t_out_norm);
    plepre_add(PLEPRE_RMSNORM, t0);
    if (s != GEIST_OK) {
        return s;
    }

    /* 6. out_buf = (out_buf + ple_lookup) * PLE_INPUT_SCALE. */
    t0 = prof ? transformer_profile_now_ns() : 0;
    {
        bool combined_on_device = false;
        if (v->add != nullptr && v->scale_f32 != nullptr) {
            struct geist_tensor t_plu_2d = view_2d(
                    st->sess->scratch_ple_lookup, (int64_t) n, (int64_t) PLE_OUT);
            /* add validates before encoding; once it succeeded out_buf is
             * mutated and the host fallback must NOT rerun the combine. */
            if (v->add(be, &t_out_2d, &t_plu_2d, &t_out_2d) == GEIST_OK) {
                s = v->scale_f32(be, &t_out_2d, st->config.ple_input_scale,
                                 &t_out_2d);
                if (s != GEIST_OK) {
                    return s;
                }
                combined_on_device = true;
            }
        }
        if (!combined_on_device) {
            float *p   = (float *) v->buffer_map(out_buf);
            float *plu = (float *) v->buffer_map(st->sess->scratch_ple_lookup);
            for (size_t i = 0; i < n * PLE_OUT; i++) {
                p[i] = (p[i] + plu[i]) * st->config.ple_input_scale;
            }
            v->buffer_unmap(st->sess->scratch_ple_lookup);
            v->buffer_unmap(out_buf);
        }
    }
    plepre_add(PLEPRE_COMBINE, t0);
    return GEIST_OK;
}

/* ---- Batched output head (logits for the last row only) -------------- *
 *
 * After run_all_layers populates scratch_h_b [seq, HIDDEN], extract the
 * row for the final token (seq-1), run output_norm + lm_head + softcap +
 * argmax, and store the prediction in next_token_pending. */
/* Drive lm_head + softcap + sampler on a single row of scratch_h_b
 * (at index row_idx, 0..seq-1). Returns the sampled token via
 * *out_token. Greedy when state->sess->temperature == 0; otherwise uses the
 * session's configured top_k/top_p/temperature. */

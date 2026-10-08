/*
 * src/archs/transformer/forward/layer.c — per-layer forward pass.
 *
 * Layer: ARCHITECTURE.
 *
 * Per-layer forward orchestration and per-layer-input precompute.
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

#include "checked.h"
#include "heap.h"
#include "quant.h"
#include <geist.h>
#include <geist_backend.h>

#include <stdio.h>
#include <stdatomic.h>
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

/* Diagnostics only (env-gated); atomic so concurrent sessions with
 * profiling on accumulate instead of racing (TSan). */
static _Atomic uint64_t g_transformer_profile_ns[TRANSFORMER_PROFILE_COUNT];
static _Atomic uint64_t g_transformer_profile_calls[TRANSFORMER_PROFILE_COUNT];

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
    static _Atomic int enabled = -1;
    int                cur     = atomic_load(&enabled);
    if (cur < 0) {
        const char *env = getenv("GEIST_PROFILE_PREFILL");
        if (env == nullptr || env[0] == '\0') {
            env = getenv("GEIST_PROFILE_FORWARD");
        }
        const int on  = (env != nullptr && env[0] == '1') ? 1 : 0;
        int       exp = -1;
        /* First updater registers the atexit sink; losers just reload. */
        if (atomic_compare_exchange_strong(&enabled, &exp, on)) {
            if (on) {
                atexit(transformer_profile_print);
            }
        }
        cur = atomic_load(&enabled);
    }
    return cur != 0;
}

static void transformer_profile_add(enum transformer_profile_stage stage, uint64_t t0) {
    if (t0 == 0) {
        return;
    }
    atomic_fetch_add(&g_transformer_profile_ns[stage], transformer_profile_now_ns() - t0);
    atomic_fetch_add(&g_transformer_profile_calls[stage], 1);
}

static void transformer_layer_bind_kv_buffers(struct transformer_layer_forward_ctx *ctx) {

    struct transformer_arch_session *sess   = ctx->sess;
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
                                       struct transformer_arch_session      *sess,
                                       int                                   layer_idx,
                                       size_t                                q_position,
                                       size_t                                seq,
                                       bool                                  advance_kv,
                                       struct geist_buffer                  *h_in_buf,
                                       struct geist_buffer                  *per_layer_input_buf,
                                       struct geist_buffer                  *h_out_buf) {

    memset(ctx, 0, sizeof(*ctx));
    struct transformer_arch_state            *st = sess->model;
    struct transformer_layer_weights         *L  = &st->layers[layer_idx];
    const struct transformer_layer_exec_plan *P =
            st->layer_plans != nullptr ? &st->layer_plans[layer_idx] : nullptr;
    const bool plan_apply_sub_ln = P != nullptr ? P->apply_sub_ln : st->config.has_sub_ln;

    ctx->st                  = st;
    ctx->sess                = sess;
    ctx->be                  = st->backend;
    ctx->v                   = st->backend->desc->vtbl;
    ctx->prims               = st->backend->desc->prims;
    ctx->fused               = geist_backend_fused_tbl(st->backend);
    ctx->L                   = L;
    ctx->P                   = P;
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
    /* Needs the scratch slice as well as the flag: a session built before
     * the family was known would have no proj_in buffer to normalise into. */
    ctx->apply_projection_input_norms =
            st->config.has_projection_input_norms && sess->scratch_proj_in != nullptr;
    ctx->apply_gemma_attn_norms =
            P != nullptr ? P->apply_gemma_attn_norms : st->config.has_gemma_attn_norms;
    ctx->apply_qk_norms  = P != nullptr ? P->apply_qk_norms : st->config.has_qk_norms;
    ctx->apply_ple       = P != nullptr ? P->apply_ple : st->config.has_ple;
    ctx->run_ple         = ctx->apply_ple && per_layer_input_buf != nullptr;
    ctx->kv_int8_enabled = sess->kv_int8_enabled;
    ctx->kv_kivi_enabled = sess->kv_kivi_enabled;
    ctx->kv_f16_enabled  = sess->kv_f16_enabled;
    ctx->ffn_activation  = P != nullptr ? P->ffn_activation : st->config.ffn_activation;
    ctx->eps             = st->config.rms_eps;
    ctx->hd              = L->head_dim;
    ctx->q_out           = L->q_out;
    ctx->kv_out          = L->kv_out;
    ctx->inter           = L->intermediate;
    ctx->SEQ             = (int64_t) seq;
    ctx->kv_len_now      = q_position + seq;
    transformer_layer_bind_kv_buffers(ctx);
}

enum geist_status transformer_forward_one_layer(struct transformer_arch_session *sess,
                                                int                              layer_idx,
                                                size_t                           q_position,
                                                size_t                           seq,
                                                bool                             advance_kv,
                                                struct geist_buffer             *h_in_buf,
                                                struct geist_buffer *per_layer_input_buf,
                                                struct geist_buffer *h_out_buf) {
    struct transformer_arch_state *st = sess->model;
    if (st == nullptr || layer_idx < 0 || (size_t) layer_idx >= st->n_layers ||
        h_in_buf == nullptr || h_out_buf == nullptr || seq == 0 || seq > sess->m_max) {
        return GEIST_E_INVALID_ARG;
    }

    struct transformer_layer_forward_ctx ctx;
    transformer_layer_ctx_init(&ctx,
                               sess,
                               layer_idx,
                               q_position,
                               seq,
                               advance_kv,
                               h_in_buf,
                               per_layer_input_buf,
                               h_out_buf);

    frame_arena_reset(&sess->scratch_arena);

    const bool profile = transformer_profile_enabled();
    uint64_t   t0      = profile ? transformer_profile_now_ns() : 0;
    /* Per-layer token mixer: DeltaNet layers replace the whole
     * attention block (incl. its residual add); the FFN/PLE stages are
     * mixer-agnostic. */
    enum geist_status s = st->layers[layer_idx].mixer == GEIST_MIXER_DELTANET
                                  ? transformer_layer_run_deltanet_block(&ctx)
                                  : transformer_layer_run_attention_block(&ctx);
    transformer_profile_add(TRANSFORMER_PROFILE_ATTENTION, t0);
    if (s != GEIST_OK) {
        return s;
    }
    t0 = profile ? transformer_profile_now_ns() : 0;
    s  = transformer_layer_run_ffn_block(&ctx);
    transformer_profile_add(TRANSFORMER_PROFILE_FFN, t0);
    if (s != GEIST_OK) {
        return s;
    }
    t0 = profile ? transformer_profile_now_ns() : 0;
    s  = transformer_layer_run_ple_or_copy(&ctx);
    transformer_profile_add(TRANSFORMER_PROFILE_PLE, t0);
    if (s != GEIST_OK) {
        return s;
    }

    t0 = profile ? transformer_profile_now_ns() : 0;
    s  = transformer_layer_scale_output(&ctx);
    transformer_profile_add(TRANSFORMER_PROFILE_SCALE, t0);
    if (s != GEIST_OK) {
        return s;
    }
    if (ctx.advance_kv) {
        sess->kv_len = ctx.kv_len_now;
    }
    return GEIST_OK;
}

enum geist_status transformer_forward_mtp_layer(struct transformer_arch_session  *sess,
                                                struct transformer_layer_weights *layer,
                                                size_t                            q_position,
                                                size_t                            seq,
                                                struct geist_buffer              *h_in_buf,
                                                struct geist_buffer              *h_out_buf) {
    if (sess == nullptr || sess->model == nullptr || layer == nullptr || h_in_buf == nullptr ||
        h_out_buf == nullptr || sess->mtp_k_cache == nullptr || sess->mtp_v_cache == nullptr ||
        seq == 0 || seq > sess->m_max || q_position + seq > sess->max_seq_len) {
        return GEIST_E_INVALID_ARG;
    }

    /* Start from the family defaults established by the normal initializer,
     * then replace all layer- and cache-specific bindings.  The MTP block is
     * intentionally planless: target-layer fusion decisions and KV aliases
     * are invalid for the separately loaded trailing block. */
    struct transformer_layer_forward_ctx ctx;
    transformer_layer_ctx_init(&ctx, sess, 0, q_position, seq, false, h_in_buf, nullptr, h_out_buf);
    ctx.L                            = layer;
    ctx.P                            = nullptr;
    ctx.layer_idx                    = layer->layer_idx;
    ctx.kv_src                       = -1;
    ctx.compute_kv                   = true;
    ctx.apply_bitnet_input_quant     = false;
    ctx.apply_sub_ln                 = false;
    ctx.apply_projection_input_norms = false;
    ctx.apply_gemma_attn_norms       = false;
    ctx.apply_qk_norms               = true;
    ctx.apply_ple                    = false;
    ctx.run_ple                      = false;
    ctx.kv_int8_enabled              = false;
    ctx.kv_kivi_enabled              = false;
    ctx.kv_f16_enabled               = false;
    ctx.ffn_activation               = GEIST_FFN_SWIGLU;
    ctx.hd                           = layer->head_dim;
    ctx.q_out                        = layer->q_out;
    ctx.kv_out                       = layer->kv_out;
    ctx.inter                        = layer->intermediate;
    ctx.k_cache_buf                  = sess->mtp_k_cache;
    ctx.v_cache_buf                  = sess->mtp_v_cache;
    ctx.k_cache_q8_buf               = nullptr;
    ctx.v_cache_q8_buf               = nullptr;
    ctx.k_cache_scale_buf            = nullptr;
    ctx.v_cache_scale_buf            = nullptr;
    ctx.k_kivi_q_buf                 = nullptr;
    ctx.v_kivi_q_buf                 = nullptr;
    ctx.k_kivi_scales_buf            = nullptr;
    ctx.k_kivi_zeros_buf             = nullptr;
    ctx.v_kivi_scales_buf            = nullptr;
    ctx.v_kivi_zeros_buf             = nullptr;
    ctx.k_residual_buf               = nullptr;
    ctx.v_residual_buf               = nullptr;

    frame_arena_reset(&sess->scratch_arena);
    enum geist_status s = transformer_layer_run_attention_block(&ctx);
    if (s == GEIST_OK) {
        s = transformer_layer_run_ffn_block(&ctx);
    }
    if (s == GEIST_OK) {
        s = transformer_layer_run_ple_or_copy(&ctx);
    }
    if (s == GEIST_OK) {
        s = transformer_layer_scale_output(&ctx);
    }
    return s;
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
        dequant_q3_K_row(n_in, raw + row_idx * n_in / Q3_K_BLOCK_ELEMS * Q3_K_BLOCK_BYTES, dst);
        break;
    case GEIST_DTYPE_Q4_0:
        dequant_q4_0_row(n_in, raw + row_idx * n_in / Q4_0_BLOCK_ELEMS * Q4_0_BLOCK_BYTES, dst);
        break;
    case GEIST_DTYPE_Q4_1:
        dequant_q4_1_row(n_in, raw + row_idx * n_in / Q4_1_BLOCK_ELEMS * Q4_1_BLOCK_BYTES, dst);
        break;
    case GEIST_DTYPE_Q5_0:
        dequant_q5_0_row(n_in, raw + row_idx * n_in / Q5_0_BLOCK_ELEMS * Q5_0_BLOCK_BYTES, dst);
        break;
    case GEIST_DTYPE_Q4_K:
        dequant_q4_K_row(n_in, raw + row_idx * n_in / Q4_K_BLOCK_ELEMS * Q4_K_BLOCK_BYTES, dst);
        break;
    case GEIST_DTYPE_Q5_K:
        dequant_q5_K_row(n_in, raw + row_idx * n_in / Q5_K_BLOCK_ELEMS * Q5_K_BLOCK_BYTES, dst);
        break;
    case GEIST_DTYPE_Q6_K:
        dequant_q6_K_row(n_in, raw + row_idx * n_in / Q6_K_BLOCK_ELEMS * Q6_K_BLOCK_BYTES, dst);
        break;
    case GEIST_DTYPE_Q8_0:
        dequant_q8_0_row(n_in, raw + row_idx * n_in / Q8_0_BLOCK_ELEMS * Q8_0_BLOCK_BYTES, dst);
        break;
    case GEIST_DTYPE_IQ2_S:
        dequant_iq2_s_row(n_in, raw + row_idx * n_in / IQ2_S_BLOCK_ELEMS * IQ2_S_BLOCK_BYTES, dst);
        break;
    case GEIST_DTYPE_IQ3_S:
        dequant_iq3_s_row(n_in, raw + row_idx * n_in / IQ3_S_BLOCK_ELEMS * IQ3_S_BLOCK_BYTES, dst);
        break;
    case GEIST_DTYPE_IQ4_NL:
        dequant_iq4_nl_row(
                n_in, raw + row_idx * n_in / IQ4_NL_BLOCK_ELEMS * IQ4_NL_BLOCK_BYTES, dst);
        break;
    case GEIST_DTYPE_IQ4_XS:
        dequant_iq4_xs_row(
                n_in, raw + row_idx * n_in / IQ4_XS_BLOCK_ELEMS * IQ4_XS_BLOCK_BYTES, dst);
        break;
    case GEIST_DTYPE_PQ2_0:
        dequant_pq2_0_row(n_in, raw + row_idx * n_in / PQ2_0_BLOCK_ELEMS * PQ2_0_BLOCK_BYTES, dst);
        break;
    case GEIST_DTYPE_I2_S: {
        /* BitNet i2_s: 256-elem/64-byte ternary blocks, reversed in-byte field
         * order vs TQ2_0, ONE f32 per-TENSOR scale at the tail (offset
         * total_elems/4). Used for the token-embedding table on BitNet-2B-4T. */
        const size_t total = (size_t) t->shape[0] * (size_t) t->shape[1];
        float        scale;
        memcpy(&scale, raw + i2_s_scale_offset(total), sizeof scale);
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

/* The session's lookup_rows, grown to `floats`. nullptr if that allocation
 * fails; the old staging is then kept. */
static float *lookup_rows(struct transformer_arch_session *sess, size_t floats) {
    if (floats <= sess->lookup_rows_floats) {
        return sess->lookup_rows;
    }
    float *rows = heap_alloc_array_aligned(float, floats);
    if (rows == nullptr) {
        return nullptr;
    }
    void *old = sess->lookup_rows;
    safe_free(&old);
    sess->lookup_rows        = rows;
    sess->lookup_rows_floats = floats;
    return rows;
}

enum geist_status transformer_gather_rows(struct transformer_arch_session *sess,
                                          const struct geist_tensor       *table,
                                          size_t                           n,
                                          const geist_token_t              ids[static n],
                                          size_t                           row,
                                          float                            scale,
                                          struct geist_buffer             *dst) {
    struct geist_backend            *be = sess->model->backend;
    const struct geist_backend_vtbl *v  = be->desc->vtbl;
    size_t                           floats;
    if (ckd_mul(&floats, n, row)) {
        return GEIST_E_INVALID_ARG;
    }
    const bool staged = be->desc->caps.lookup_tables_on_host;
    float     *rows   = staged ? lookup_rows(sess, floats) : (float *) v->buffer_map(dst);
    if (rows == nullptr) {
        return staged ? GEIST_E_OOM : GEIST_E_BACKEND;
    }
    enum geist_status s = GEIST_OK;
    for (size_t t = 0; t < n && s == GEIST_OK; t++) {
        s = dequant_one_row(be, table, (size_t) ids[t], rows + t * row);
    }
    if (s == GEIST_OK && scale != 1.0f) {
        for (size_t i = 0; i < floats; i++) {
            rows[i] *= scale;
        }
    }
    if (!staged) {
        v->buffer_unmap(dst);
        return s;
    }
    return s != GEIST_OK ? s
                         : v->buffer_upload(dst, floats * sizeof(float), (const uint8_t *) rows);
}

enum geist_status transformer_compute_per_layer_input(struct transformer_arch_session *sess,
                                                      geist_token_t                    token_id,
                                                      struct geist_buffer             *h_buf,
                                                      struct geist_buffer *per_layer_input_buf) {
    struct transformer_arch_state *st = sess->model;
    if (st == nullptr || h_buf == nullptr || per_layer_input_buf == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    if (token_id < 0 || (size_t) token_id >= (size_t) st->vocab_size) {
        return GEIST_E_INVALID_ARG;
    }
    struct geist_backend                  *be    = st->backend;
    const struct geist_backend_vtbl       *v     = be->desc->vtbl;
    const struct geist_backend_primitives *prims = be->desc->prims;
    const struct geist_backend_fused      *fused = geist_backend_fused_tbl(be);
    enum geist_status                      s;

    /* 1. Dequant one row of the PLE table into scratch_ple_lookup, then
     *    multiply by PLE_TABLE_SCALE (16). */
    {
        bool on_device = false;
        if (st->model_fusions.ple_lookup_scaled) {
            struct geist_tensor t_row = view_1d(sess->scratch_ple_lookup, st->ple_out);
            s                         = fused->embedding_lookup_scaled(
                    be, &st->ple_table, token_id, st->config.ple_table_scale, &t_row);
            if (s != GEIST_OK) {
                return s;
            }
            on_device = true;
        }
        if (!on_device) {
            float *dst = (float *) v->buffer_map(sess->scratch_ple_lookup);
            if (dst == nullptr) {
                return GEIST_E_BACKEND; /* the backend said why */
            }
            s = dequant_one_row(be, &st->ple_table, (size_t) token_id, dst);
            if (s != GEIST_OK) {
                v->buffer_unmap(sess->scratch_ple_lookup);
                return s;
            }
            for (size_t i = 0; i < (size_t) st->ple_out; i++) {
                dst[i] *= st->config.ple_table_scale;
            }
            v->buffer_unmap(sess->scratch_ple_lookup);
        }
    }

    /* 2. linear(h, model_proj) → per_layer_input (reused as scratch).
     *    Shape: [1, HIDDEN] × [PLE_OUT, HIDDEN]^T → [1, PLE_OUT]. */
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
        if (st->model_fusions.prim_scale_f32) {
            s = prims->scale_f32(be, &t_all, st->config.ple_model_proj_scale, &t_all);
            if (s != GEIST_OK) {
                return s;
            }
        } else {
            float *p = (float *) v->buffer_map(per_layer_input_buf);
            if (p == nullptr) {
                return GEIST_E_BACKEND; /* the backend said why */
            }
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
    s = prims->rmsnorm(be, &t_ple_proj_2dN, &t_norm_w, st->config.rms_eps, &t_ple_proj_2dN);
    if (s != GEIST_OK) {
        return s;
    }

    /* 5. per_layer_input = (ple_proj + ple_lookup) * PLE_INPUT_SCALE. */
    struct geist_tensor t_ple_proj_1d   = view_1d(per_layer_input_buf, st->ple_out);
    struct geist_tensor t_ple_lookup_1d = view_1d(sess->scratch_ple_lookup, st->ple_out);
    s = prims->add(be, &t_ple_proj_1d, &t_ple_lookup_1d, &t_ple_proj_1d);
    if (s != GEIST_OK) {
        return s;
    }
    {
        struct geist_tensor t_all = view_1d(per_layer_input_buf, st->ple_out);
        if (st->model_fusions.prim_scale_f32) {
            s = prims->scale_f32(be, &t_all, st->config.ple_input_scale, &t_all);
            if (s != GEIST_OK) {
                return s;
            }
        } else {
            float *p = (float *) v->buffer_map(per_layer_input_buf);
            if (p == nullptr) {
                return GEIST_E_BACKEND; /* the backend said why */
            }
            for (size_t i = 0; i < (size_t) st->ple_out; i++) {
                p[i] *= st->config.ple_input_scale;
            }
            v->buffer_unmap(per_layer_input_buf);
        }
    }
    return GEIST_OK;
}

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
static _Atomic uint64_t  g_plepre_ns[PLEPRE_COUNT];
static _Atomic uint64_t  g_plepre_calls[PLEPRE_COUNT];
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
    /* Same idiom as transformer_profile_enabled above (atomic, so concurrent
     * sessions do not race). First updater registers the atexit sink;
     * losers just reload. */
    static _Atomic int en  = -1;
    int                cur = atomic_load(&en);
    if (cur < 0) {
        const char *e = getenv("GEIST_PROFILE_PREFILL");
        if (e == nullptr || e[0] == '\0')
            e = getenv("GEIST_PROFILE_FORWARD");
        const int on  = (e != nullptr && e[0] == '1') ? 1 : 0;
        int       exp = -1;
        if (atomic_compare_exchange_strong(&en, &exp, on)) {
            if (on)
                atexit(plepre_print);
        }
        cur = atomic_load(&en);
    }
    return cur != 0;
}

static void plepre_add(enum plepre_stage stage, uint64_t t0) {
    if (t0 == 0)
        return;
    g_plepre_ns[stage] += transformer_profile_now_ns() - t0;
    g_plepre_calls[stage]++;
}

[[nodiscard]] enum geist_status
compute_per_layer_inputs_batch(struct transformer_arch_session *sess,
                               size_t                           n,
                               const geist_token_t             *ple_ids,
                               struct geist_buffer             *h_buf,
                               struct geist_buffer             *out_buf) {
    struct transformer_arch_state *st = sess->model;

    if (n == 0 || n > sess->m_max) {
        return GEIST_E_INVALID_ARG;
    }
    struct geist_backend                  *be      = st->backend;
    const struct geist_backend_vtbl       *v       = be->desc->vtbl;
    const struct geist_backend_primitives *prims   = be->desc->prims;
    const struct geist_backend_fused      *fused   = geist_backend_fused_tbl(be);
    const size_t                           PLE_OUT = (size_t) st->ple_out;
    const bool                             prof    = plepre_enabled();
    uint64_t                               t0;

    /* 1+2. Dequant n PLE rows + scale by 16. Device path: per-row fused
     * lookup+scale dispatches — no host dequant, no pipeline flush from
     * mapping the scratch mid-batch. */
    t0                    = prof ? transformer_profile_now_ns() : 0;
    bool gather_on_device = st->model_fusions.ple_lookup_scaled;
    if (gather_on_device) {
        for (size_t t = 0; t < n; t++) {
            struct geist_tensor t_row = {
                    .buffer = sess->scratch_ple_lookup,
                    .offset = t * PLE_OUT * sizeof(float),
                    .dtype  = GEIST_DTYPE_F32,
                    .layout = GEIST_LAYOUT_DENSE,
                    .ndim   = 1,
                    .shape  = {(int64_t) PLE_OUT, 0, 0, 0, 0, 0, 0, 0},
                    .stride = {1, 0, 0, 0, 0, 0, 0, 0},
            };
            const enum geist_status es = fused->embedding_lookup_scaled(
                    be, &st->ple_table, ple_ids[t], st->config.ple_table_scale, &t_row);
            if (es != GEIST_OK) {
                return es; /* bound at plan build — failure is a real error */
            }
        }
    }
    if (!gather_on_device) {
        const enum geist_status s = transformer_gather_rows(sess,
                                                            &st->ple_table,
                                                            n,
                                                            ple_ids,
                                                            PLE_OUT,
                                                            st->config.ple_table_scale,
                                                            sess->scratch_ple_lookup);
        if (s != GEIST_OK) {
            return s;
        }
    }
    plepre_add(PLEPRE_GATHER, t0);

    /* 3. linear(h, model_proj) → out_buf. */
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
    if (st->model_fusions.prim_scale_f32) {
        s = prims->scale_f32(be, &t_out_2d, st->config.ple_model_proj_scale, &t_out_2d);
        if (s != GEIST_OK) {
            return s;
        }
    } else {
        float *p = (float *) v->buffer_map(out_buf);
        if (p == nullptr) {
            return GEIST_E_BACKEND; /* the backend said why */
        }
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
    s = prims->rmsnorm(be, &t_out_norm, &t_w, st->config.rms_eps, &t_out_norm);
    plepre_add(PLEPRE_RMSNORM, t0);
    if (s != GEIST_OK) {
        return s;
    }

    /* 6. out_buf = (out_buf + ple_lookup) * PLE_INPUT_SCALE. */
    t0 = prof ? transformer_profile_now_ns() : 0;
    /* Path chosen once, before anything mutates out_buf: `add` is a required
     * member, so only scale_f32 decides (#352). */
    if (st->model_fusions.prim_scale_f32) {
        struct geist_tensor t_plu_2d =
                view_2d(sess->scratch_ple_lookup, (int64_t) n, (int64_t) PLE_OUT);
        s = prims->add(be, &t_out_2d, &t_plu_2d, &t_out_2d);
        if (s == GEIST_OK) {
            s = prims->scale_f32(be, &t_out_2d, st->config.ple_input_scale, &t_out_2d);
        }
        if (s != GEIST_OK) {
            return s;
        }
    } else {
        float *p   = (float *) v->buffer_map(out_buf);
        float *plu = (float *) v->buffer_map(sess->scratch_ple_lookup);
        if (p == nullptr || plu == nullptr) {
            return GEIST_E_BACKEND;
        }
        for (size_t i = 0; i < n * PLE_OUT; i++) {
            p[i] = (p[i] + plu[i]) * st->config.ple_input_scale;
        }
        v->buffer_unmap(sess->scratch_ple_lookup);
        v->buffer_unmap(out_buf);
    }
    plepre_add(PLEPRE_COMBINE, t0);
    return GEIST_OK;
}

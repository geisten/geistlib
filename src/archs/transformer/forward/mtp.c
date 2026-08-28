/* Qwen3.5 multi-token-prediction head, isolated from engine scheduling. */
#define GEIST_INTERNAL_ARCH_LAYER

#include "internal.h"
#include "profile.h"
#include "../arch_state.h"
#include "../forward.h"

#include <geist.h>
#include <geist_backend.h>

#include <stdlib.h>
#include <string.h>

enum mtp_profile_stage {
    MTP_PROFILE_INPUT = 0,
    MTP_PROFILE_NORMS,
    MTP_PROFILE_CONCAT,
    MTP_PROFILE_EH_PROJ,
    MTP_PROFILE_BLOCK,
    MTP_PROFILE_HIDDEN_COPY,
    MTP_PROFILE_HEAD,
    MTP_PROFILE_COUNT,
};

static uint64_t          g_mtp_profile_ns[MTP_PROFILE_COUNT];
static uint64_t          g_mtp_profile_calls[MTP_PROFILE_COUNT];
static const char *const g_mtp_profile_names[MTP_PROFILE_COUNT] = {
        "input",
        "norms",
        "concat",
        "eh_proj",
        "block",
        "hidden_copy",
        "head",
};
static struct transformer_forward_profile g_mtp_profile = {
        .title       = "transformer MTP",
        .stage_names = g_mtp_profile_names,
        .stage_count = MTP_PROFILE_COUNT,
        .ns          = g_mtp_profile_ns,
        .calls       = g_mtp_profile_calls,
};

static bool mtp_spec_head_enabled(void) {
    const char *env = getenv("GEIST_MTP_SPEC_HEAD");
    return env != nullptr && env[0] == '1';
}

static enum geist_status
mtp_embed_tokens(struct transformer_arch_session *sess, size_t n, const geist_token_t *ids) {
    struct transformer_arch_state    *st    = sess->model;
    struct geist_backend             *be    = st->backend;
    const struct geist_backend_vtbl  *v     = be->desc->vtbl;
    const struct geist_backend_fused *fused = geist_backend_fused_tbl(be);

    if (st->model_fusions.embed_lookup_scaled) {
        for (size_t row = 0; row < n; row++) {
            struct geist_tensor out = view_1d(sess->mtp_embed, st->d_model);
            out.offset              = row * st->d_model * sizeof(float);
            enum geist_status s =
                    fused->embedding_lookup_scaled(be, &st->embed_table, ids[row], 1.0f, &out);
            if (s != GEIST_OK)
                return s;
        }
        return GEIST_OK;
    }

    float *dst = (float *) v->buffer_map(sess->mtp_embed);
    if (dst == nullptr)
        return GEIST_E_BACKEND;
    for (size_t row = 0; row < n; row++) {
        enum geist_status s =
                dequant_one_row(be, &st->embed_table, (size_t) ids[row], dst + row * st->d_model);
        if (s != GEIST_OK) {
            v->buffer_unmap(sess->mtp_embed);
            return s;
        }
    }
    v->buffer_unmap(sess->mtp_embed);
    return GEIST_OK;
}

void transformer_mtp_reset(struct transformer_arch_session *sess) {
    if (sess != nullptr) {
        sess->mtp_kv_len       = 0;
        sess->mtp_txn_base_len = 0;
        if (sess->mtp_pending_h != nullptr && sess->model != nullptr)
            memset(sess->mtp_pending_h, 0, sess->model->d_model * sizeof(float));
    }
}

enum geist_status transformer_mtp_forward(struct transformer_arch_session *sess,
                                          size_t                           n,
                                          const geist_token_t             *ids,
                                          const float                     *target_hidden,
                                          geist_token_t                   *out_tokens,
                                          float                           *out_hidden) {
    if (sess == nullptr || sess->model == nullptr || n == 0 || ids == nullptr ||
        target_hidden == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct transformer_arch_state *st = sess->model;
    if (st->n_mtp_layers != 1 || st->mtp_layers == nullptr) {
        return GEIST_E_UNSUPPORTED;
    }
    if (n > sess->m_max || sess->mtp_kv_len + n > sess->max_seq_len) {
        return GEIST_E_INVALID_ARG;
    }
    for (size_t row = 0; row < n; row++) {
        if (ids[row] < 0 || (size_t) ids[row] >= st->vocab_size)
            return GEIST_E_INVALID_ARG;
    }

    struct transformer_mtp_layer_weights  *M     = &st->mtp_layers[0];
    struct geist_backend                  *be    = st->backend;
    const struct geist_backend_vtbl       *v     = be->desc->vtbl;
    const struct geist_backend_primitives *prims = be->desc->prims;
    const size_t                           H     = st->d_model;
    const int64_t                          N     = (int64_t) n;
    enum geist_status                      s;
    const bool                             profile = transformer_profile_enabled(&g_mtp_profile);
    uint64_t                               t0      = profile ? transformer_profile_now_ns() : 0;

    /* Upload target residual rows before executing any stateful attention. */
    float *hp = (float *) v->buffer_map(sess->mtp_hidden_norm);
    if (hp == nullptr)
        return GEIST_E_BACKEND;
    memcpy(hp, target_hidden, n * H * sizeof(float));
    v->buffer_unmap(sess->mtp_hidden_norm);

    s = mtp_embed_tokens(sess, n, ids);
    transformer_profile_add(&g_mtp_profile, MTP_PROFILE_INPUT, t0);
    if (s != GEIST_OK)
        return s;

    struct geist_tensor t_e  = view_2d(sess->mtp_embed, N, (int64_t) H);
    struct geist_tensor t_h  = view_2d(sess->mtp_hidden_norm, N, (int64_t) H);
    struct geist_tensor t_en = view_1d(M->enorm.buffer, (int64_t) H);
    struct geist_tensor t_hn = view_1d(M->hnorm.buffer, (int64_t) H);
    t0                       = profile ? transformer_profile_now_ns() : 0;
    s                        = prims->rmsnorm(be, &t_e, &t_en, st->config.rms_eps, &t_e);
    if (s == GEIST_OK)
        s = prims->rmsnorm(be, &t_h, &t_hn, st->config.rms_eps, &t_h);
    transformer_profile_add(&g_mtp_profile, MTP_PROFILE_NORMS, t0);
    if (s != GEIST_OK)
        return s;

    /* llama.cpp/Qwen order is [normalized token embedding, normalized
     * target hidden].  Keep this explicit; swapping the halves produces
     * plausible but incorrect draft logits. */
    const float *ep = (const float *) v->buffer_map(sess->mtp_embed);
    t0              = profile ? transformer_profile_now_ns() : 0;
    hp              = (float *) v->buffer_map(sess->mtp_hidden_norm);
    float *cp       = (float *) v->buffer_map(sess->mtp_concat);
    if (ep == nullptr || hp == nullptr || cp == nullptr)
        return GEIST_E_BACKEND;
    for (size_t row = 0; row < n; row++) {
        memcpy(cp + row * 2 * H, ep + row * H, H * sizeof(float));
        memcpy(cp + row * 2 * H + H, hp + row * H, H * sizeof(float));
    }
    v->buffer_unmap(sess->mtp_embed);
    v->buffer_unmap(sess->mtp_hidden_norm);
    v->buffer_unmap(sess->mtp_concat);
    transformer_profile_add(&g_mtp_profile, MTP_PROFILE_CONCAT, t0);

    struct geist_tensor t_cat  = view_2d(sess->mtp_concat, N, (int64_t) (2 * H));
    struct geist_tensor t_seed = view_2d(sess->scratch_h_a, N, (int64_t) H);
    t0                         = profile ? transformer_profile_now_ns() : 0;
    s                          = linear_w_or_legacy(be,
                                                    v,
                                                    sess->mtp_concat,
                                                    sess->scratch_h_a,
                                                    &M->eh_proj_w,
                                                    n,
                                                    &t_cat,
                                                    &M->eh_proj,
                                                    &t_seed);
    transformer_profile_add(&g_mtp_profile, MTP_PROFILE_EH_PROJ, t0);
    if (s != GEIST_OK)
        return s;

    t0 = profile ? transformer_profile_now_ns() : 0;
    s  = transformer_forward_mtp_layer(
            sess, &M->block, sess->mtp_kv_len, n, sess->scratch_h_a, sess->scratch_h_b);
    transformer_profile_add(&g_mtp_profile, MTP_PROFILE_BLOCK, t0);
    if (s != GEIST_OK)
        return s;

    if (out_hidden != nullptr) {
        t0               = profile ? transformer_profile_now_ns() : 0;
        const float *raw = (const float *) v->buffer_map(sess->scratch_h_b);
        if (raw == nullptr)
            return GEIST_E_BACKEND;
        memcpy(out_hidden, raw, n * H * sizeof(float));
        v->buffer_unmap(sess->scratch_h_b);
        transformer_profile_add(&g_mtp_profile, MTP_PROFILE_HIDDEN_COPY, t0);
    }

    if (out_tokens == nullptr) {
        sess->mtp_kv_len += n;
        return GEIST_OK;
    }

    t0                         = profile ? transformer_profile_now_ns() : 0;
    struct geist_tensor t_raw  = view_2d(sess->scratch_h_b, N, (int64_t) H);
    struct geist_tensor t_norm = view_2d(sess->scratch_h_a, N, (int64_t) H);
    struct geist_tensor t_wn   = view_1d(M->shared_head_norm.buffer, (int64_t) H);
    s                          = prims->rmsnorm(be, &t_raw, &t_wn, st->config.rms_eps, &t_norm);
    if (s != GEIST_OK)
        return s;

    /* Draft logits are greedy and never exposed to the caller. Reuse the
     * host i8-sketch head for the recursive single-row case instead of
     * streaming the complete tied vocabulary matrix for every MTP token.
     * Keep the target head's metadata intact: the sketch shares scratch
     * buffers with it, but an MTP draft must not change whether the last
     * authoritative target logits were dense or sparse. */
    if (n == 1 && mtp_spec_head_enabled()) {
        const bool logits_sparse     = sess->logits_sparse;
        const bool logits_softcapped = sess->logits_softcapped;
        const bool handled           = transformer_spec_head_try(sess, out_tokens);
        sess->logits_sparse          = logits_sparse;
        sess->logits_softcapped      = logits_softcapped;
        if (handled) {
            transformer_profile_add(&g_mtp_profile, MTP_PROFILE_HEAD, t0);
            sess->mtp_kv_len++;
            return GEIST_OK;
        }
    }

    struct geist_tensor t_logits = view_2d(sess->scratch_logits, N, st->vocab_size);
    s                            = linear_w_or_legacy(be,
                                                      v,
                                                      sess->scratch_h_a,
                                                      sess->scratch_logits,
                                                      &st->embed_table_w,
                                                      n,
                                                      &t_norm,
                                                      &st->output_table,
                                                      &t_logits);
    if (s != GEIST_OK)
        return s;

    const float *logits = (const float *) v->buffer_map(sess->scratch_logits);
    if (logits == nullptr)
        return GEIST_E_BACKEND;
    for (size_t row = 0; row < n; row++) {
        out_tokens[row] = geist_sampler_argmax(st->vocab_size, logits + row * st->vocab_size);
    }
    v->buffer_unmap(sess->scratch_logits);
    transformer_profile_add(&g_mtp_profile, MTP_PROFILE_HEAD, t0);
    sess->mtp_kv_len += n;
    return GEIST_OK;
}

enum geist_status transformer_mtp_sync_target(struct transformer_arch_session *sess,
                                              size_t                           n,
                                              const geist_token_t             *ids,
                                              struct geist_buffer             *target_hidden_buf) {
    if (sess == nullptr || sess->model == nullptr || n == 0 || n > sess->m_max || ids == nullptr ||
        target_hidden_buf == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    if (sess->model->n_mtp_layers == 0 || !sess->mtp_enabled)
        return GEIST_OK;
    if (sess->mtp_pending_h == nullptr || sess->mtp_target_raw == nullptr ||
        sess->mtp_target_shifted == nullptr)
        return GEIST_E_INVALID_STATE;

    const size_t                     H     = sess->model->d_model;
    const size_t                     bytes = n * H * sizeof(float);
    const struct geist_backend_vtbl *v     = sess->model->backend->desc->vtbl;
    const float                     *raw   = (const float *) v->buffer_map(target_hidden_buf);
    if (raw == nullptr)
        return GEIST_E_BACKEND;
    memcpy(sess->mtp_target_raw, raw, bytes);
    v->buffer_unmap(target_hidden_buf);

    memcpy(sess->mtp_target_shifted, sess->mtp_pending_h, H * sizeof(float));
    if (n > 1) {
        memcpy(sess->mtp_target_shifted + H, sess->mtp_target_raw, (n - 1) * H * sizeof(float));
    }

    enum geist_status s =
            transformer_mtp_forward(sess, n, ids, sess->mtp_target_shifted, nullptr, nullptr);

    /* MTP shares the generic layer scratch with the target trunk. Restore
     * authoritative raw rows so the target output head sees its own state. */
    float *restore = (float *) v->buffer_map(target_hidden_buf);
    if (restore == nullptr)
        return GEIST_E_BACKEND;
    memcpy(restore, sess->mtp_target_raw, bytes);
    v->buffer_unmap(target_hidden_buf);
    if (s != GEIST_OK)
        return s;

    memcpy(sess->mtp_pending_h, sess->mtp_target_raw + (n - 1) * H, H * sizeof(float));
    return GEIST_OK;
}

enum geist_status transformer_mtp_draft(struct transformer_arch_session *sess,
                                        size_t                           k_max,
                                        geist_token_t                    seed,
                                        geist_token_t                   *out_tokens,
                                        size_t                          *n_out) {
    if (sess == nullptr || sess->model == nullptr || k_max == 0 || out_tokens == nullptr ||
        n_out == nullptr || seed < 0 || (size_t) seed >= sess->model->vocab_size) {
        return GEIST_E_INVALID_ARG;
    }
    *n_out = 0;
    if (!sess->mtp_enabled || sess->model->n_mtp_layers != 1 || sess->mtp_pending_h == nullptr ||
        sess->temperature != 0.0f) {
        return GEIST_E_UNSUPPORTED;
    }
    if (sess->mtp_kv_len != sess->kv_len || sess->mtp_kv_len + k_max > sess->max_seq_len)
        return GEIST_E_INVALID_STATE;

    const size_t base     = sess->mtp_kv_len;
    out_tokens[0]         = seed;
    *n_out                = 1;
    geist_token_t current = seed;
    const float  *hidden  = sess->mtp_pending_h;
    while (*n_out < k_max) {
        geist_token_t     next = -1;
        enum geist_status s =
                transformer_mtp_forward(sess, 1, &current, hidden, &next, sess->mtp_target_raw);
        if (s != GEIST_OK) {
            sess->mtp_kv_len = base;
            *n_out           = 0;
            return s;
        }
        out_tokens[(*n_out)++] = next;
        current                = next;
        hidden                 = sess->mtp_target_raw;
    }

    /* Draft rows are provisional. Target verification will overwrite them
     * at the same positions using authoritative target hidden states. */
    sess->mtp_kv_len = base;
    return GEIST_OK;
}

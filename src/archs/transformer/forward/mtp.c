/* Qwen3.5 multi-token-prediction head, isolated from engine scheduling. */
#define GEIST_INTERNAL_ARCH_LAYER

#include "internal.h"
#include "../arch_state.h"
#include "../forward.h"

#include <geist.h>
#include <geist_backend.h>

#include <string.h>

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
    if (sess != nullptr)
        sess->mtp_kv_len = 0;
}

enum geist_status transformer_mtp_forward(struct transformer_arch_session *sess,
                                          size_t                           n,
                                          const geist_token_t             *ids,
                                          const float                     *target_hidden,
                                          geist_token_t                   *out_tokens,
                                          float                           *out_hidden) {
    if (sess == nullptr || sess->model == nullptr || n == 0 || ids == nullptr ||
        target_hidden == nullptr || out_tokens == nullptr) {
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

    /* Upload target residual rows before executing any stateful attention. */
    float *hp = (float *) v->buffer_map(sess->mtp_hidden_norm);
    if (hp == nullptr)
        return GEIST_E_BACKEND;
    memcpy(hp, target_hidden, n * H * sizeof(float));
    v->buffer_unmap(sess->mtp_hidden_norm);

    s = mtp_embed_tokens(sess, n, ids);
    if (s != GEIST_OK)
        return s;

    struct geist_tensor t_e  = view_2d(sess->mtp_embed, N, (int64_t) H);
    struct geist_tensor t_h  = view_2d(sess->mtp_hidden_norm, N, (int64_t) H);
    struct geist_tensor t_en = view_1d(M->enorm.buffer, (int64_t) H);
    struct geist_tensor t_hn = view_1d(M->hnorm.buffer, (int64_t) H);
    s                        = prims->rmsnorm(be, &t_e, &t_en, st->config.rms_eps, &t_e);
    if (s == GEIST_OK)
        s = prims->rmsnorm(be, &t_h, &t_hn, st->config.rms_eps, &t_h);
    if (s != GEIST_OK)
        return s;

    /* llama.cpp/Qwen order is [normalized token embedding, normalized
     * target hidden].  Keep this explicit; swapping the halves produces
     * plausible but incorrect draft logits. */
    const float *ep = (const float *) v->buffer_map(sess->mtp_embed);
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

    struct geist_tensor t_cat  = view_2d(sess->mtp_concat, N, (int64_t) (2 * H));
    struct geist_tensor t_seed = view_2d(sess->scratch_h_a, N, (int64_t) H);
    s                          = linear_w_or_legacy(be,
                                                    v,
                                                    sess->mtp_concat,
                                                    sess->scratch_h_a,
                                                    &M->eh_proj_w,
                                                    n,
                                                    &t_cat,
                                                    &M->eh_proj,
                                                    &t_seed);
    if (s != GEIST_OK)
        return s;

    s = transformer_forward_mtp_layer(
            sess, &M->block, sess->mtp_kv_len, n, sess->scratch_h_a, sess->scratch_h_b);
    if (s != GEIST_OK)
        return s;

    if (out_hidden != nullptr) {
        const float *raw = (const float *) v->buffer_map(sess->scratch_h_b);
        if (raw == nullptr)
            return GEIST_E_BACKEND;
        memcpy(out_hidden, raw, n * H * sizeof(float));
        v->buffer_unmap(sess->scratch_h_b);
    }

    struct geist_tensor t_raw  = view_2d(sess->scratch_h_b, N, (int64_t) H);
    struct geist_tensor t_norm = view_2d(sess->scratch_h_a, N, (int64_t) H);
    struct geist_tensor t_wn   = view_1d(M->shared_head_norm.buffer, (int64_t) H);
    s                          = prims->rmsnorm(be, &t_raw, &t_wn, st->config.rms_eps, &t_norm);
    if (s != GEIST_OK)
        return s;

    struct geist_tensor t_logits = view_2d(sess->scratch_logits, N, st->vocab_size);
    s                            = linear_w_or_legacy(be,
                                                      v,
                                                      sess->scratch_h_a,
                                                      sess->scratch_logits,
                                                      &st->embed_table_w,
                                                      n,
                                                      &t_norm,
                                                      &st->embed_table,
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
    sess->mtp_kv_len += n;
    return GEIST_OK;
}

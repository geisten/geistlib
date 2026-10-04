/* Independent selected-row readout at the final transformer hidden state. */
#define GEIST_INTERNAL_ARCH_LAYER
#include "../arch_state.h"
#include "../forward.h"
#include "internal.h"
#include "profile.h"
#include "checked.h"
#include "heap.h"

#include <math.h>
#include <string.h>

struct transformer_decision_readout {
    struct transformer_arch_session *session;
    struct geist_buffer             *results;
    size_t         cap, tile, table_cap, table_bytes, n_tiles, n_candidates, projected;
    geist_token_t *keys, *tiles;
    size_t        *values, *slots;
    float         *out;
    uint64_t       head_ns;
};

bool transformer_decision_supported(const void *state) {
    const struct transformer_arch_state *st = state;
    return st != nullptr && !geist_pooling_is_embedding(st->config.pooling) &&
           st->embed_table_w.linear_rows != nullptr && st->embed_table_w.linear_rows_tile > 0 &&
           st->embed_table_w.linear_rows_tile <= 32;
}

void transformer_decision_destroy(void *readout) {
    struct transformer_decision_readout *r = readout;
    if (r == nullptr)
        return;
    if (r->results != nullptr) {
        struct geist_backend *be = r->session->model->backend;
        be->desc->vtbl->buffer_destroy(be, r->results);
    }
    safe_free((void **) &r->keys);
    safe_free((void **) &r->tiles);
    safe_free((void **) &r->values);
    safe_free((void **) &r->slots);
    safe_free((void **) &r);
}

enum geist_status transformer_decision_create(void *session, size_t cap, void **out) {
    if (out == nullptr)
        return GEIST_E_INVALID_ARG;
    *out                                  = nullptr;
    struct transformer_arch_session *sess = session;
    if (sess == nullptr || cap == 0)
        return GEIST_E_INVALID_ARG;
    if (!transformer_decision_supported(sess->model))
        return GEIST_E_UNSUPPORTED;
    const struct geist_weight *w = &sess->model->embed_table_w;
    if (w->linear_rows_prepare != nullptr) {
        const enum geist_status prepared = w->linear_rows_prepare(w, sess->model->backend);
        if (prepared != GEIST_OK)
            return prepared;
    }
    const size_t tile = sess->model->embed_table_w.linear_rows_tile;
    size_t       n, bytes, twice, table_bytes;
    if (cap > (size_t) sess->model->vocab_size || ckd_mul(&n, cap, tile) || n > INT32_MAX ||
        ckd_mul(&bytes, n, sizeof(float)) || ckd_mul(&twice, cap, 2)) {
        return GEIST_E_INVALID_ARG;
    }
    size_t table_cap = 1;
    while (table_cap < twice) {
        if (ckd_mul(&table_cap, table_cap, 2))
            return GEIST_E_INVALID_ARG;
    }
    if (ckd_mul(&table_bytes, table_cap, sizeof(geist_token_t)))
        return GEIST_E_INVALID_ARG;
    struct transformer_decision_readout *r =
            heap_calloc_array_aligned(struct transformer_decision_readout, 1);
    if (r == nullptr)
        return GEIST_E_OOM;
    r->session     = sess;
    r->cap         = cap;
    r->tile        = tile;
    r->table_cap   = table_cap;
    r->table_bytes = table_bytes;
    r->keys        = heap_alloc_array_aligned(geist_token_t, table_cap);
    r->values      = heap_alloc_array_aligned(size_t, table_cap);
    r->tiles       = heap_alloc_array_aligned(geist_token_t, cap);
    r->slots       = heap_alloc_array_aligned(size_t, cap);
    if (r->keys == nullptr || r->values == nullptr || r->tiles == nullptr || r->slots == nullptr) {
        transformer_decision_destroy(r);
        return GEIST_E_OOM;
    }
    struct geist_backend   *be = sess->model->backend;
    const enum geist_status s  = be->desc->vtbl->buffer_create(
            be, bytes, GEIST_BUFFER_SCRATCH, GEIST_MEMORY_HOST_VISIBLE, &r->results);
    if (s != GEIST_OK) {
        transformer_decision_destroy(r);
        return s;
    }
    *out = r;
    return GEIST_OK;
}

enum geist_status transformer_decision_finish(struct transformer_decision_readout *r,
                                              size_t                               row_idx) {
    struct transformer_arch_session *sess = r->session;
    struct transformer_arch_state   *st   = sess->model;
    struct geist_backend            *be   = st->backend;
    const struct geist_backend_vtbl *v    = be->desc->vtbl;
    const uint64_t                   t0   = transformer_profile_now_ns();
    enum geist_status                s    = transformer_head_prepare(sess, row_idx);
    if (s != GEIST_OK)
        return s;
    struct geist_tensor x = view_1d(sess->scratch_h_a, st->d_model);
    struct geist_tensor y = view_2d(r->results, r->n_tiles, r->tile);
    s                     = st->embed_table_w.linear_rows(
            r->n_tiles, r->tiles, &x, &st->embed_table_w, &st->output_table, &y, be);
    if (s != GEIST_OK)
        return s;
    const float *p = v->buffer_map(r->results);
    if (p == nullptr)
        return GEIST_E_BACKEND;
    const float softcap = st->config.logit_softcap;
    for (size_t i = 0; i < r->n_candidates; i++) {
        float value = p[r->slots[i]];
#ifdef GEIST_TUNE
        if (st->embed_table_w.gain_slot != nullptr && *st->embed_table_w.gain_slot != 1.0f)
            value *= *st->embed_table_w.gain_slot;
#endif
        r->out[i] = value;
    }
    /* Match peek_logits' dedicated softcap loop, keeping its invariant
     * scalar outside the loop rather than mixing it with row gathering. */
    if (softcap > 0.0f) {
        float       *logits = r->out;
        const float  c      = softcap;
        const size_t n      = r->n_candidates;
        for (size_t i = 0; i < n; i++)
            logits[i] = tanhf(logits[i] / c) * c;
    }
    v->buffer_unmap(r->results);
    r->head_ns = transformer_profile_now_ns() - t0;
    /* This session has no vocabulary logits or pending generation token.
     * In particular peek_logits must never interpret old scratch as fresh. */
    sess->logits_valid       = false;
    sess->next_token_pending = -1;
    return GEIST_OK;
}

enum geist_status transformer_decision_prefill(size_t              *rows,
                                               size_t              *readback_bytes,
                                               uint64_t            *head_ns,
                                               void                *readout,
                                               size_t               n_prompt,
                                               size_t               n_candidates,
                                               const geist_token_t *prompt,
                                               const geist_token_t *candidates,
                                               float               *out) {
    if (rows == nullptr || readback_bytes == nullptr || head_ns == nullptr)
        return GEIST_E_INVALID_ARG;
    *rows                                  = 0;
    *readback_bytes                        = 0;
    *head_ns                               = 0;
    struct transformer_decision_readout *r = readout;
    if (r == nullptr || n_prompt == 0 || n_candidates == 0 || n_candidates > r->cap ||
        prompt == nullptr || candidates == nullptr || out == nullptr)
        return GEIST_E_INVALID_ARG;
    memset(out, 0, n_candidates * sizeof(float)); /* cap was checked at creation */
    r->n_tiles      = 0;
    r->n_candidates = n_candidates;
    r->head_ns      = 0;
    r->projected    = 0;
    memset(r->keys, 0xff, r->table_bytes);
    const size_t vocab = (size_t) r->session->model->vocab_size;
    for (size_t i = 0; i < n_candidates; i++) {
        if (candidates[i] < 0 || (size_t) candidates[i] >= vocab)
            return GEIST_E_INVALID_ARG;
        const size_t        id   = (size_t) candidates[i];
        const geist_token_t base = (geist_token_t) (id - id % r->tile);
        size_t              slot = ((uint32_t) base * 0x9e3779b1u) & (r->table_cap - 1);
        while (r->keys[slot] != -1 && r->keys[slot] != base)
            slot = (slot + 1) & (r->table_cap - 1);
        if (r->keys[slot] == -1) {
            r->keys[slot]          = base;
            r->values[slot]        = r->n_tiles;
            r->tiles[r->n_tiles++] = base;
            r->projected += vocab - (size_t) base < r->tile ? vocab - (size_t) base : r->tile;
        }
        r->slots[i] = r->values[slot] * r->tile + id % r->tile;
    }
    r->out                    = out;
    const enum geist_status s = transformer_prefill_rows(r->session, n_prompt, prompt, r);
    r->out                    = nullptr;
    if (s != GEIST_OK) {
        memset(out, 0, n_candidates * sizeof(float));
        return s;
    }
    *rows           = r->projected;
    *readback_bytes = r->n_tiles * r->tile * sizeof(float); /* checked at creation */
    *head_ns        = r->head_ns;
    return GEIST_OK;
}

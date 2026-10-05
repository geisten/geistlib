/*
 * src/archs/transformer/snapshot.c — session snapshot / restore (#548).
 *
 * Layer: ARCHITECTURE.
 *
 * A snapshot is the complete decoding state of one session as a flat byte
 * image: the attention layers' KV rows [0, kv_len), every Gated-DeltaNet
 * layer's conv and recurrent state, the pending next-token logits and the
 * bookkeeping around them (kv_len, the pinned prefix, a decode_step whose
 * forward is still owed, the sampler RNG). Restoring it into any session
 * of the SAME loaded model, the one it came from or another, puts that
 * session exactly where the source was: the next decode_step, peek_logits
 * or prefill gives the same result.
 *
 * It is the reusable prefix pin_prefix cannot give a hybrid model: a reset
 * returns DeltaNet state to the empty sequence, a restore returns it to
 * wherever the snapshot was taken.
 *
 * Only the used KV rows are copied: every KV representation stores
 * position-major from offset 0 (kv_store.c), so the prefix of each buffer
 * is the state. The format is in-process only — host endianness, no
 * cross-version promise — and bound to one loaded model by its
 * snapshot_id.
 *
 * Not covered, refused with GEIST_E_UNSUPPORTED: the KIVI cache (a ring
 * and drain counters, not a position-major prefix), an enabled MTP drafter
 * (its own cache) and embedding models (no decoding state to resume).
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "arch_state.h"
#include "forward.h"

#include "checked.h"

#include <geist.h>
#include <geist_backend.h>

#include <string.h>

constexpr uint32_t SNAP_MAGIC   = 0x504E5347u; /* "GSNP" */
constexpr uint32_t SNAP_VERSION = 1;

/* Flags in snap_header.flags. */
constexpr uint32_t SNAP_LOGITS_VALID      = 1u << 0;
constexpr uint32_t SNAP_LOGITS_SOFTCAPPED = 1u << 1;
constexpr uint32_t SNAP_ADVANCE_DEFERRED  = 1u << 2;

struct snap_header {
    uint32_t magic;
    uint32_t version;
    uint64_t model_id;    /* transformer_arch_state.snapshot_id */
    uint64_t total_bytes; /* header + payload */
    uint64_t kv_len;
    uint64_t prefix_length;
    uint64_t rng_state;
    uint32_t kv_repr; /* snap_kv_repr() of the source session */
    uint32_t flags;
    int32_t  next_token_pending;
    uint32_t reserved;
};

/* What a session stores per KV row, packed: restore must land in a
 * session whose cache holds the same bytes per position. */
static uint32_t snap_kv_repr(const struct transformer_arch_session *sess) {
    return (sess->kv_int8_enabled ? 1u : 0u) | (sess->kv_int4_packed_enabled ? 2u : 0u) |
           (sess->kv_f16_enabled ? 4u : 0u) | (sess->kv_rot_enabled ? 8u : 0u) |
           ((uint32_t) (sess->kv_sim_qbits & 0xff) << 8);
}

static bool owns_kv(const struct transformer_layer_weights *L) {
    return !L->is_kv_shared && L->mixer != GEIST_MIXER_DELTANET;
}

/* Bytes one position takes in layer `L`'s K (and, the same, V) data
 * buffer, and in each of its K/V scale buffers (0 without scales). */
static void kv_row_bytes(const struct transformer_arch_session  *sess,
                         const struct transformer_layer_weights *L,
                         size_t                                 *data,
                         size_t                                 *scales) {
    const size_t elems = sess->model->n_kv_heads * L->head_dim;
    if (sess->kv_int8_enabled) {
        *data   = sess->kv_int4_packed_enabled ? elems / 2 : elems;
        *scales = sess->model->n_kv_heads * sizeof(float);
    } else {
        *data   = elems * (sess->kv_f16_enabled ? 2u : sizeof(float));
        *scales = 0;
    }
}

static void dn_state_floats(const struct transformer_arch_state *st, size_t *conv_n, size_t *s_n) {
    const size_t key_dim   = st->config.dn_n_k_heads * st->config.dn_head_k;
    const size_t value_dim = st->config.dn_n_v_heads * st->config.dn_head_v;
    *conv_n                = (st->config.dn_conv_kernel - 1) * (2 * key_dim + value_dim);
    *s_n                   = st->config.dn_n_v_heads * st->config.dn_head_k * st->config.dn_head_v;
}

/* GEIST_OK if `sess` can be snapshotted or restored at all. */
[[nodiscard]] static enum geist_status snap_supported(const struct transformer_arch_session *sess) {
    const struct transformer_arch_state *st = sess->model;
    const char *why = geist_pooling_is_embedding(st->config.pooling)
                              ? "an embedding model has no decoding state"
                      : sess->kv_kivi_enabled ? "the KIVI KV cache is not supported"
                      : sess->mtp_enabled     ? "the MTP drafter (GEIST_MTP=1) is not supported"
                                              : nullptr;
    if (why != nullptr) {
        geist_backend_set_error(st->backend, GEIST_E_UNSUPPORTED, "session snapshot: %s", why);
        return GEIST_E_UNSUPPORTED;
    }
    return GEIST_OK;
}

/* Total image size for `kv_len` cached positions, with or without logits.
 * True on overflow. */
[[nodiscard]] static bool
snap_bytes(const struct transformer_arch_session *sess, size_t kv_len, bool logits, size_t *out) {
    const struct transformer_arch_state *st    = sess->model;
    size_t                               total = sizeof(struct snap_header);
    size_t                               conv_n, s_n;
    dn_state_floats(st, &conv_n, &s_n);
    for (size_t li = 0; li < st->n_layers; li++) {
        const struct transformer_layer_weights *L = &st->layers[li];
        size_t                                  part;
        if (L->mixer == GEIST_MIXER_DELTANET) {
            if (ckd_add(&part, conv_n, s_n) || ckd_mul(&part, part, sizeof(float)) ||
                ckd_add(&total, total, part)) {
                return true;
            }
        } else if (owns_kv(L)) {
            size_t data, scales;
            kv_row_bytes(sess, L, &data, &scales);
            if (ckd_add(&part, data, scales) || ckd_mul(&part, part, 2) ||
                ckd_mul(&part, part, kv_len) || ckd_add(&total, total, part)) {
                return true;
            }
        }
    }
    if (logits) {
        size_t part;
        if (ckd_mul(&part, st->vocab_size, sizeof(float)) || ckd_add(&total, total, part)) {
            return true;
        }
    }
    *out = total;
    return false;
}

/* Walk the payload in its fixed order (per layer: K, V, K scales, V
 * scales, or conv, S; then the logits), moving each part between the
 * session's buffers and `p`, which has room for all of it. */
[[nodiscard]] static enum geist_status snap_payload(struct transformer_arch_session *sess,
                                                    size_t                           kv_len,
                                                    bool                             logits,
                                                    uint8_t                         *p,
                                                    bool                             to_device) {
    struct transformer_arch_state *st = sess->model;
    struct geist_backend          *be = st->backend;
    size_t                         conv_n, s_n;
    dn_state_floats(st, &conv_n, &s_n);
    enum geist_status s = GEIST_OK;
    for (size_t li = 0; li < st->n_layers && s == GEIST_OK; li++) {
        const struct transformer_layer_weights *L = &st->layers[li];
        if (L->mixer == GEIST_MIXER_DELTANET) {
            /* A fresh layer's state is zeros whatever its buffers hold
             * (dn_fresh): save writes the zeros, a restore unmarks it. */
            bool *fresh = sess->dn_fresh != nullptr ? &sess->dn_fresh[li] : nullptr;
            if (!to_device && fresh != nullptr && *fresh) {
                memset(p, 0, (conv_n + s_n) * sizeof(float));
                p += (conv_n + s_n) * sizeof(float);
                continue;
            }
            s = transformer_buffer_xfer(
                    be, sess->dn_conv_state[li], conv_n * sizeof(float), p, to_device);
            p += conv_n * sizeof(float);
            if (s == GEIST_OK) {
                s = transformer_buffer_xfer(be, sess->dn_S[li], s_n * sizeof(float), p, to_device);
                p += s_n * sizeof(float);
            }
            if (s == GEIST_OK && to_device && fresh != nullptr) {
                *fresh = false;
            }
            continue;
        }
        if (!owns_kv(L)) {
            continue;
        }
        size_t data, scales;
        kv_row_bytes(sess, L, &data, &scales);
        struct geist_buffer *bufs[4] = {
                sess->kv_int8_enabled ? sess->k_cache_q8[li] : sess->k_cache[li],
                sess->kv_int8_enabled ? sess->v_cache_q8[li] : sess->v_cache[li],
                sess->kv_int8_enabled ? sess->k_cache_scale[li] : nullptr,
                sess->kv_int8_enabled ? sess->v_cache_scale[li] : nullptr};
        const size_t rows[4] = {data, data, scales, scales};
        for (size_t b = 0; b < 4 && s == GEIST_OK; b++) {
            if (rows[b] == 0) {
                continue;
            }
            s = transformer_buffer_xfer(be, bufs[b], kv_len * rows[b], p, to_device);
            p += kv_len * rows[b];
        }
    }
    if (s == GEIST_OK && logits) {
        s = transformer_buffer_xfer(
                be, sess->scratch_logits, st->vocab_size * sizeof(float), p, to_device);
    }
    return s;
}

enum geist_status transformer_snapshot_size(size_t                                *out_bytes,
                                            const struct transformer_arch_session *sess) {
    *out_bytes                = 0;
    const enum geist_status s = snap_supported(sess);
    if (s != GEIST_OK) {
        return s;
    }
    if (snap_bytes(sess, sess->kv_len, sess->logits_valid, out_bytes)) {
        return GEIST_E_OOM;
    }
    return GEIST_OK;
}

enum geist_status transformer_snapshot_save(size_t                          *out_bytes,
                                            size_t                           capacity,
                                            void                            *buf,
                                            struct transformer_arch_session *sess) {
    *out_bytes          = 0;
    enum geist_status s = snap_supported(sess);
    if (s != GEIST_OK) {
        return s;
    }
    struct transformer_arch_state *st = sess->model;
    /* The spec-head fast path leaves the logits sparse; store them dense,
     * as peek_logits would show them, so the image does not depend on a
     * hidden state it does not carry. */
    if (sess->logits_valid && sess->logits_sparse) {
        s = transformer_head_dense_recompute(sess);
        if (s != GEIST_OK) {
            return s;
        }
    }
    size_t total;
    if (snap_bytes(sess, sess->kv_len, sess->logits_valid, &total)) {
        return GEIST_E_OOM;
    }
    if (buf == nullptr || capacity < total) {
        geist_backend_set_error(st->backend,
                                GEIST_E_INVALID_ARG,
                                "session snapshot: needs %zu bytes, buffer has %zu",
                                total,
                                buf == nullptr ? (size_t) 0 : capacity);
        return GEIST_E_INVALID_ARG;
    }
    const struct snap_header h = {
            .magic              = SNAP_MAGIC,
            .version            = SNAP_VERSION,
            .model_id           = st->snapshot_id,
            .total_bytes        = total,
            .kv_len             = sess->kv_len,
            .prefix_length      = sess->prefix_length,
            .rng_state          = sess->rng.state,
            .kv_repr            = snap_kv_repr(sess),
            .flags              = (sess->logits_valid ? SNAP_LOGITS_VALID : 0u) |
                                  (sess->logits_softcapped ? SNAP_LOGITS_SOFTCAPPED : 0u) |
                                  (sess->advance_deferred ? SNAP_ADVANCE_DEFERRED : 0u),
            .next_token_pending = sess->next_token_pending,
    };
    memcpy(buf, &h, sizeof h);
    s = snap_payload(sess, sess->kv_len, sess->logits_valid, (uint8_t *) buf + sizeof h, false);
    if (s != GEIST_OK) {
        return s;
    }
    *out_bytes = total;
    return GEIST_OK;
}

enum geist_status transformer_snapshot_restore(size_t                           n_bytes,
                                               const void                      *buf,
                                               struct transformer_arch_session *sess) {
    enum geist_status s = snap_supported(sess);
    if (s != GEIST_OK) {
        return s;
    }
    struct transformer_arch_state *st = sess->model;
    struct snap_header             h;
    if (buf == nullptr || n_bytes < sizeof h) {
        geist_backend_set_error(
                st->backend, GEIST_E_FORMAT, "session restore: %zu bytes is no snapshot", n_bytes);
        return GEIST_E_FORMAT;
    }
    memcpy(&h, buf, sizeof h);
    const bool  logits = (h.flags & SNAP_LOGITS_VALID) != 0;
    size_t      total;
    const char *why =
            h.magic != SNAP_MAGIC || h.version != SNAP_VERSION ? "not a snapshot of this version"
            : h.model_id != st->snapshot_id                    ? "taken on another model"
            : h.kv_repr != snap_kv_repr(sess) ? "the KV cache mode differs from this session's"
            : h.kv_len > sess->max_seq_len    ? "more positions than this session's max_seq_len"
            : h.prefix_length > h.kv_len ||
                            (h.flags & ~(SNAP_LOGITS_VALID | SNAP_LOGITS_SOFTCAPPED |
                                         SNAP_ADVANCE_DEFERRED)) != 0 ||
                            (!logits && (h.flags & SNAP_ADVANCE_DEFERRED) != 0) ||
                            (logits && (h.next_token_pending < 0 ||
                                        (uint64_t) h.next_token_pending >= st->vocab_size))
                    ? "inconsistent header"
            : snap_bytes(sess, (size_t) h.kv_len, logits, &total) || total != h.total_bytes ||
                            total != n_bytes
                    ? "size does not match its header"
                    : nullptr;
    if (why != nullptr) {
        geist_backend_set_error(st->backend, GEIST_E_FORMAT, "session restore: %s", why);
        return GEIST_E_FORMAT;
    }
    /* Everything is validated; from here on the session is overwritten. A
     * transfer that fails leaves it reset rather than half restored. */
    transformer_session_reset(sess);
    sess->prefix_length = 0;
    sess->kv_len        = 0;
    s = snap_payload(sess,
                     (size_t) h.kv_len,
                     logits,
                     (uint8_t *) buf + sizeof h, /* read only: to_device copies from it */
                     true);
    if (s != GEIST_OK) {
        transformer_session_reset(sess);
        return s;
    }
    sess->kv_len             = (size_t) h.kv_len;
    sess->prefix_length      = (size_t) h.prefix_length;
    sess->rng.state          = h.rng_state;
    sess->logits_valid       = logits;
    sess->logits_sparse      = false;
    sess->logits_softcapped  = (h.flags & SNAP_LOGITS_SOFTCAPPED) != 0;
    sess->embedding_valid    = false;
    sess->next_token_pending = h.next_token_pending;
    sess->advance_deferred   = (h.flags & SNAP_ADVANCE_DEFERRED) != 0;
    return GEIST_OK;
}

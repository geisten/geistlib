/*
 * src/archs/transformer/forward/step.c — decode step, layer loop,
 * embedding lookup, KIVI drain, and session reset / truncate / apply_opts.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "internal.h"
#include "../arch_family.h"
#include "../arch_state.h"
#include "../rotation.h"
#include "../forward.h"
#include "../scratch_plan.h"

#include "checked.h"
#include "heap.h"
#include "quant.h"
#include "gemma4_kernels.h"
#include "kivi.h"

#include <geist.h>
#include <geist_backend.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum geist_status transformer_kivi_drain_full(struct transformer_arch_session *sess) {
    struct transformer_arch_state *st = sess->model;
    if (!sess->kv_kivi_enabled)
        return GEIST_OK;
    if (sess->kivi_residual_count < KIVI_K_GROUP_SIZE)
        return GEIST_OK;
    struct geist_backend            *be = st->backend;
    const struct geist_backend_vtbl *v  = be->desc->vtbl;
    const size_t                     R  = KIVI_K_GROUP_SIZE;

    while (sess->kivi_residual_count >= R) {
        for (size_t li = 0; li < st->n_layers; li++) {
            if (st->layers[li].is_kv_shared)
                continue;
            const size_t hd       = st->layers[li].head_dim;
            float       *k_res    = (float *) v->buffer_map(sess->k_residual[li]);
            float       *v_res    = (float *) v->buffer_map(sess->v_residual[li]);
            uint8_t     *k_q4     = (uint8_t *) v->buffer_map(sess->k_kivi_q[li]);
            uint8_t     *v_q4     = (uint8_t *) v->buffer_map(sess->v_kivi_q[li]);
            float       *k_scales = (float *) v->buffer_map(sess->k_kivi_scales[li]);
            float       *k_zeros  = (float *) v->buffer_map(sess->k_kivi_zeros[li]);
            float       *v_scales = (float *) v->buffer_map(sess->v_kivi_scales[li]);
            float       *v_zeros  = (float *) v->buffer_map(sess->v_kivi_zeros[li]);
            if (k_res == nullptr || v_res == nullptr || k_q4 == nullptr || v_q4 == nullptr ||
                k_scales == nullptr || k_zeros == nullptr || v_scales == nullptr ||
                v_zeros == nullptr) {
                return GEIST_E_BACKEND; /* the backend said why */
            }
            kivi_drain_one_layer(sess->kivi_drained_count,
                                 sess->kivi_residual_count,
                                 R,
                                 hd,
                                 st->n_kv_heads,
                                 k_res,
                                 v_res,
                                 k_q4,
                                 v_q4,
                                 k_scales,
                                 k_zeros,
                                 v_scales,
                                 v_zeros);
            v->buffer_unmap(sess->k_residual[li]);
            v->buffer_unmap(sess->v_residual[li]);
            v->buffer_unmap(sess->k_kivi_q[li]);
            v->buffer_unmap(sess->v_kivi_q[li]);
            v->buffer_unmap(sess->k_kivi_scales[li]);
            v->buffer_unmap(sess->k_kivi_zeros[li]);
            v->buffer_unmap(sess->v_kivi_scales[li]);
            v->buffer_unmap(sess->v_kivi_zeros[li]);
        }
        sess->kivi_drained_count += R;
        sess->kivi_residual_count -= R;
    }
    return GEIST_OK;
}

/* A pinned prefix that is no whole number of groups keeps its last
 * (prefix_length mod R) rows in the residual ring. The first drain of a
 * later turn packs them into a 2-bit group with that turn's tokens and
 * moves later tokens over their rows, but a reset to the prefix reads them
 * from those rows again. So pinning copies them out, and reset writes them
 * back: the prefix is then the ring and groups it was at pin time. */
[[nodiscard]] static enum geist_status kivi_pin_copy(struct transformer_arch_session *sess,
                                                     bool                             save) {
    const struct transformer_arch_state *st   = sess->model;
    const struct geist_backend_vtbl     *v    = st->backend->desc->vtbl;
    float                               *tail = sess->kivi_pin_tail;
    for (size_t li = 0; li < st->n_layers; li++) {
        if (sess->k_residual[li] == nullptr) {
            continue; /* KV-shared or DeltaNet: no ring */
        }
        const size_t         len = sess->kivi_pin_rows * st->n_kv_heads * st->layers[li].head_dim;
        struct geist_buffer *ring[2] = {sess->k_residual[li], sess->v_residual[li]};
        for (size_t i = 0; i < 2; i++) {
            float *rows = (float *) v->buffer_map(ring[i]);
            if (rows == nullptr) {
                return GEIST_E_BACKEND; /* the backend said why */
            }
            memcpy(save ? tail : rows, save ? rows : tail, len * sizeof *tail);
            v->buffer_unmap(ring[i]);
            tail += len;
        }
    }
    return GEIST_OK;
}

enum geist_status transformer_kivi_pin_save(struct transformer_arch_session *sess) {
    void *old = sess->kivi_pin_tail;
    safe_free(&old);
    sess->kivi_pin_tail = nullptr;
    sess->kivi_pin_rows = 0;
    const size_t rows   = sess->kivi_residual_count; /* < R after the prefill's drains */
    if (!sess->kv_kivi_enabled || rows == 0) {
        return GEIST_OK;
    }
    const struct transformer_arch_state *st = sess->model;
    size_t                               n  = 0;
    for (size_t li = 0; li < st->n_layers; li++) {
        size_t layer = 0;
        if (sess->k_residual[li] != nullptr &&
            (ckd_mul(&layer, rows, 2 * st->n_kv_heads * st->layers[li].head_dim) ||
             ckd_add(&n, n, layer))) {
            return GEIST_E_OOM;
        }
    }
    if (n == 0) {
        return GEIST_OK; /* no layer has a ring */
    }
    sess->kivi_pin_tail = heap_alloc_array_aligned(float, n);
    if (sess->kivi_pin_tail == nullptr) {
        return GEIST_E_OOM;
    }
    sess->kivi_pin_rows = rows;
    return kivi_pin_copy(sess, true);
}

void transformer_kivi_pin_restore(struct transformer_arch_session *sess) {
    /* Reset has put the counters back to the prefix: exactly the rows the
     * pin copied are residual again. Anything else is not that prefix. */
    if (sess->kivi_pin_rows != 0 && sess->kivi_pin_rows == sess->kivi_residual_count) {
        /* Reset has no status: an unmappable ring leaves the rows as they
         * are, and the next drain or attention map reports the backend. */
        (void) kivi_pin_copy(sess, false);
    }
}

/* Run all layers for seq tokens starting at q_position.
 *
 *   initial_h_buf:           [seq, HIDDEN]
 *   per_layer_input_buf:     [seq, NUM_LAYERS * HIDDEN_PER_LAYER]
 *                             (concatenated per_layer_inputs for the seq
 *                              tokens; for token t, layer li, the slice
 *                              is at offset t*PLE_OUT + li*HIDDEN_PER_LAYER).
 *   out_h_buf:               [seq, HIDDEN]
 *
 * Internally ping-pongs between scratch_h_a and scratch_h_b. Each layer's
 * PLE input is gathered into scratch_ple_lookup as a contiguous
 * [seq, HIDDEN_PER_LAYER] slab before the forward call. */
/* Layer split (#49): run layers [0, N) of seq rows on the split stage,
 * from `in` (this session's backend) to `out` (same). The rows cross
 * through host memory both ways; a forward on one backend finishes before
 * the other reads its result, because the download waits for it. The
 * stage's layers run inside the stage backend's own region, as the
 * caller's run inside its backend's: a batched-submit GPU then records
 * them into one command buffer instead of submitting and waiting per op,
 * a CPU backend takes its decode or prefill thread regime. The stage's
 * session follows this one's position (it caches the same positions for
 * its own layers). */
[[nodiscard]] static enum geist_status run_split_stage(struct transformer_arch_session *sess,
                                                       size_t                           q_position,
                                                       size_t                           seq,
                                                       struct geist_buffer             *in,
                                                       struct geist_buffer             *out) {
    struct transformer_arch_session *sub   = sess->split_sess;
    struct transformer_arch_state   *sst   = sub->model;
    const size_t                     bytes = seq * sst->d_model * sizeof(float);
    struct geist_buffer             *h_in  = sub->scratch_h_a;
    struct geist_buffer             *h_out = sub->scratch_h_b;

    sub->kv_len         = sess->kv_len;
    enum geist_status s = transformer_buffer_xfer(
            sess->model->backend, in, bytes, sess->split_rows, /* to_device = */ false);

    const struct geist_backend_vtbl *sv = sst->backend->desc->vtbl;
    const enum geist_parallel_region region =
            seq > 1 ? GEIST_REGION_PREFILL_BATCH : GEIST_REGION_DECODE_STEP;
    const int region_tok = sv->parallel_region_begin != nullptr
                                   ? sv->parallel_region_begin(sst->backend, region)
                                   : 0;
    if (s == GEIST_OK)
        s = transformer_buffer_xfer(sst->backend, h_in, bytes, sess->split_rows, true);
    for (size_t li = sst->layer_lo; s == GEIST_OK && li < sst->layer_hi; li++) {
        s = transformer_forward_one_layer(
                sub, (int) li, q_position, seq, /* advance_kv = */ false, h_in, nullptr, h_out);
        struct geist_buffer *tmp = h_in;
        h_in                     = h_out;
        h_out                    = tmp;
    }
    if (sv->parallel_region_end != nullptr) {
        sv->parallel_region_end(sst->backend, region_tok);
    }
    /* Through split_h_io where it exists: a device copy into cached host
     * memory, not a host read of the scratch (#84). */
    if (s == GEIST_OK && sess->split_h_io != nullptr) {
        s    = sv->buffer_copy(sess->split_h_io, 0, h_in, 0, bytes);
        h_in = sess->split_h_io;
    }
    if (s == GEIST_OK)
        s = transformer_buffer_xfer(sst->backend, h_in, bytes, sess->split_rows, false);
    if (s == GEIST_OK)
        s = transformer_buffer_xfer(sess->model->backend, out, bytes, sess->split_rows, true);
    return s;
}

/* A copy between two buffers of one backend: recorded on the device where
 * the backend can (a GPU keeps its batch), else through host maps (the CPU
 * backends have no buffer_copy). */
[[nodiscard]] static enum geist_status stage_copy(struct geist_backend *be,
                                                  struct geist_buffer  *dst,
                                                  size_t                dst_off,
                                                  struct geist_buffer  *src,
                                                  size_t                src_off,
                                                  size_t                n) {
    const struct geist_backend_vtbl *v = be->desc->vtbl;
    if (v->buffer_copy != nullptr) {
        return v->buffer_copy(dst, dst_off, src, src_off, n);
    }
    uint8_t       *d = (uint8_t *) v->buffer_map(dst);
    const uint8_t *s = (const uint8_t *) v->buffer_map(src);
    if (d != nullptr && s != nullptr) {
        memcpy(d + dst_off, s + src_off, n);
    }
    if (d != nullptr) {
        v->buffer_unmap(dst);
    }
    if (s != nullptr) {
        v->buffer_unmap(src);
    }
    return d != nullptr && s != nullptr ? GEIST_OK : GEIST_E_BACKEND;
}

/* Offload (#78): a prefill chunk of the load-backend layers on their
 * twins. The split stage's layers and then the twins' run on the split
 * backend in one batch, the hidden rows passing on the device; after each
 * twin layer the chunk's rows of its K/V cache are copied into
 * twin_kv_io, the last layer's output into twin_h_io (IO buffers: cached
 * host memory, #79), and once the batch has run, the K/V rows go into this
 * session's caches through each layer's own append and the output rows to
 * `out`. The twins' cache is dense; F16 rows are widened in place, each
 * layer's K and V slot being sized for FP32. */
[[nodiscard]] static enum geist_status run_offload_chunk(struct transformer_arch_session *sess,
                                                         size_t               q_position,
                                                         size_t               seq,
                                                         struct geist_buffer *in,
                                                         struct geist_buffer *out) {
    struct transformer_arch_state   *st    = sess->model;
    struct transformer_arch_session *sub   = sess->split_sess;
    struct transformer_arch_session *tw    = sess->twin_sess;
    struct transformer_arch_state   *tst   = tw->model;
    struct geist_backend            *gbe   = tst->backend;
    const struct geist_backend_vtbl *gv    = gbe->desc->vtbl;
    const size_t                     bytes = seq * st->d_model * sizeof(float);
    const size_t                     esz   = tw->kv_f16_enabled ? sizeof(uint16_t) : sizeof(float);

    sub->kv_len         = sess->kv_len;
    tw->kv_len          = sess->kv_len;
    enum geist_status s = transformer_buffer_xfer(st->backend, in, bytes, sess->split_rows, false);
    const enum geist_parallel_region region = GEIST_REGION_PREFILL_BATCH;
    const int                        region_tok =
            gv->parallel_region_begin != nullptr ? gv->parallel_region_begin(gbe, region) : 0;
    if (s == GEIST_OK) {
        s = transformer_buffer_xfer(gbe, sub->scratch_h_a, bytes, sess->split_rows, true);
    }
    struct geist_buffer *h_in = sub->scratch_h_a, *h_out = sub->scratch_h_b;
    for (size_t li = sub->model->layer_lo; s == GEIST_OK && li < sub->model->layer_hi; li++) {
        s = transformer_forward_one_layer(
                sub, (int) li, q_position, seq, /* advance_kv = */ false, h_in, nullptr, h_out);
        struct geist_buffer *tmp = h_in;
        h_in                     = h_out;
        h_out                    = tmp;
    }
    if (s == GEIST_OK) {
        s = stage_copy(gbe, tw->scratch_h_a, 0, h_in, 0, bytes);
    }
    h_in        = tw->scratch_h_a;
    h_out       = tw->scratch_h_b;
    size_t slot = 0; /* byte offset of the layer's K slot in twin_kv_io */
    for (size_t li = tst->layer_lo; s == GEIST_OK && li < st->n_layers; li++) {
        s = transformer_forward_one_layer(
                tw, (int) li, q_position, seq, /* advance_kv = */ false, h_in, nullptr, h_out);
        struct geist_buffer *tmp = h_in;
        h_in                     = h_out;
        h_out                    = tmp;
        const size_t kv_out      = st->layers[li].kv_out;
        const size_t row         = kv_out * esz;
        const size_t span        = seq * kv_out * sizeof(float);
        if (s == GEIST_OK) {
            s = stage_copy(
                    gbe, sess->twin_kv_io, slot, tw->k_cache[li], q_position * row, seq * row);
        }
        if (s == GEIST_OK) {
            s = stage_copy(gbe,
                           sess->twin_kv_io,
                           slot + span,
                           tw->v_cache[li],
                           q_position * row,
                           seq * row);
        }
        slot += 2 * span;
    }
    if (s == GEIST_OK) {
        s = stage_copy(gbe, sess->twin_h_io, 0, h_in, 0, bytes);
    }
    if (gv->parallel_region_end != nullptr) {
        gv->parallel_region_end(gbe, region_tok);
    }
    uint8_t *kv = s == GEIST_OK ? (uint8_t *) gv->buffer_map(sess->twin_kv_io) : nullptr;
    void    *h  = kv != nullptr ? gv->buffer_map(sess->twin_h_io) : nullptr;
    if (s == GEIST_OK && (kv == nullptr || h == nullptr)) {
        s = GEIST_E_BACKEND; /* the backend said why */
    }
    if (s == GEIST_OK) {
        s = transformer_buffer_xfer(st->backend, out, bytes, h, true);
    }
    slot = 0;
    for (size_t li = tst->layer_lo; s == GEIST_OK && li < st->n_layers; li++) {
        const size_t n    = seq * st->layers[li].kv_out;
        float       *k    = (float *) (kv + slot);
        float       *vrow = (float *) (kv + slot + n * sizeof(float));
        if (esz == sizeof(uint16_t)) {
            /* back to front: element i's FP32 bytes start at or after its F16 bytes */
            const uint16_t *k16 = (const uint16_t *) k, *v16 = (const uint16_t *) vrow;
            for (size_t i = n; i-- > 0;) {
                k[i]    = fp16_to_fp32(k16[i]);
                vrow[i] = fp16_to_fp32(v16[i]);
            }
        }
        s = transformer_kv_store_rows(sess, (int) li, q_position, seq, k, vrow);
        slot += 2 * n * sizeof(float);
    }
    if (h != nullptr) {
        gv->buffer_unmap(sess->twin_h_io);
    }
    if (kv != nullptr) {
        gv->buffer_unmap(sess->twin_kv_io);
    }
    return s;
}

/* Offload (#78): layer li's K/V rows of this step, which the load backend
 * just appended from its scratch_k/scratch_v, into the twins' cache too. */
[[nodiscard]] static enum geist_status
mirror_kv_rows(struct transformer_arch_session *sess, size_t li, size_t q_position, size_t seq) {
    const struct geist_backend_vtbl *v  = sess->model->backend->desc->vtbl;
    const float                     *k  = (const float *) v->buffer_map(sess->scratch_k);
    const float                     *vr = (const float *) v->buffer_map(sess->scratch_v);
    enum geist_status                s =
            k != nullptr && vr != nullptr
                    ? transformer_kv_store_rows(sess->twin_sess, (int) li, q_position, seq, k, vr)
                    : GEIST_E_BACKEND;
    if (k != nullptr) {
        v->buffer_unmap(sess->scratch_k);
    }
    if (vr != nullptr) {
        v->buffer_unmap(sess->scratch_v);
    }
    return s;
}

/* Rows the load backend's kernels take per call (caps.max_m; 0 = any). */
static size_t load_m_cap(const struct transformer_arch_state *st) {
    const size_t c = st->backend->desc->caps.max_m;
    return c != 0 ? c : SIZE_MAX;
}

static bool
offload_takes(const struct transformer_arch_session *sess, size_t q_position, size_t seq) {
    return sess->twin_sess != nullptr && seq >= sess->model->offload->m_min &&
           q_position <= sess->twin_valid;
}

size_t transformer_prefill_chunk(const struct transformer_arch_session *sess,
                                 size_t                                 q_position,
                                 size_t                                 n) {
    const size_t m = n < sess->m_max ? n : sess->m_max;
    if (offload_takes(sess, q_position, m)) {
        return m;
    }
    const size_t cap = load_m_cap(sess->model);
    return m < cap ? m : cap;
}

[[nodiscard]] enum geist_status transformer_run_all_layers(struct transformer_arch_session *sess,
                                                           size_t               q_position,
                                                           size_t               seq,
                                                           struct geist_buffer *initial_h_buf,
                                                           struct geist_buffer *per_layer_input_buf,
                                                           struct geist_buffer *out_h_buf) {
    struct transformer_arch_state *st = sess->model;

    struct geist_backend             *be                    = st->backend;
    const struct geist_backend_vtbl  *v                     = be->desc->vtbl;
    const struct geist_backend_fused *fused                 = geist_backend_fused_tbl(be);
    const size_t                      row_bytes_h           = st->d_model * sizeof(float);
    const size_t                      row_bytes_ple         = st->hidden_per_layer * sizeof(float);
    const size_t                      row_bytes_per_tok_ple = st->ple_out * sizeof(float);

    bool mirror = false;
    /* Offload (#78): a large enough chunk from a position the twins' cache
     * covers runs every layer on the split backend. Anything else runs the
     * load backend's layers here, and the twins' cache is good only up to
     * where that starts. */
    if (sess->twin_sess != nullptr) {
        if (offload_takes(sess, q_position, seq)) {
            const enum geist_status os =
                    run_offload_chunk(sess, q_position, seq, initial_h_buf, out_h_buf);
            sess->twin_valid = os == GEIST_OK ? q_position + seq : 0;
            if (os == GEIST_OK) {
                atomic_fetch_add_explicit(&st->offload->chunks, 1, memory_order_relaxed);
            }
            return os;
        }
        /* Steps on the load backend (decode, small chunks) from where the
         * twins' cache is current keep it current: each layer's new K/V
         * rows go there too (mirror_kv_rows), so the next prompt (a chat's
         * next turn) can run on the twins again. */
        mirror = q_position <= sess->twin_valid;
        if (mirror) {
            sess->twin_valid = q_position; /* raised again once every layer is mirrored */
        }
        /* An offloaded session's chunk may exceed what the load backend's
         * kernels take; transformer_prefill_chunk keeps its own chunks below
         * that, any other caller is refused here, not overrun. */
        if (seq > load_m_cap(st)) {
            geist_backend_set_error(st->backend,
                                    GEIST_E_INVALID_ARG,
                                    "transformer: %zu rows on '%s', whose kernels take %zu "
                                    "(offload chunk off the twins)",
                                    seq,
                                    geist_backend_name(st->backend),
                                    load_m_cap(st));
            return GEIST_E_INVALID_ARG;
        }
    }

    /* Seed scratch_h_a with seq rows of HIDDEN: through the split stage's
     * layers when the model is split (#49). */
    if (sess->split_sess != nullptr) {
        const enum geist_status ss =
                run_split_stage(sess, q_position, seq, initial_h_buf, sess->scratch_h_a);
        if (ss != GEIST_OK) {
            return ss;
        }
    } else {
        /* Bound at plan build (#352): a device copy that FAILS is a device error, not
         * a missing capability, and must not become a quiet host memcpy. */
        if (st->model_fusions.backend_buffer_copy) {
            const enum geist_status cs =
                    v->buffer_copy(sess->scratch_h_a, 0, initial_h_buf, 0, seq * row_bytes_h);
            if (cs != GEIST_OK) {
                return cs;
            }
        } else {
            const uint8_t *src = (const uint8_t *) v->buffer_map(initial_h_buf);
            uint8_t       *dst = (uint8_t *) v->buffer_map(sess->scratch_h_a);
            if (src == nullptr || dst == nullptr) {
                return GEIST_E_BACKEND; /* the backend said why */
            }
            memcpy(dst, src, seq * row_bytes_h);
            v->buffer_unmap(initial_h_buf);
            v->buffer_unmap(sess->scratch_h_a);
        }
    }

    struct geist_buffer *h_in  = sess->scratch_h_a;
    struct geist_buffer *h_out = sess->scratch_h_b;

    for (size_t li = st->layer_lo; li < st->n_layers; li++) {
        /* Gather this layer's per_layer_input slices into
         * scratch_ple_lookup when the family has PLE. Non-PLE families
         * pass nullptr and forward_one_layer skips the PLE injection. */
        struct geist_buffer *layer_ple_buf = nullptr;
        if (per_layer_input_buf != nullptr && fused->linear_t != nullptr &&
            per_layer_input_buf == sess->scratch_per_layer_input) {
            /* Batched GPU backends read the layer's PLE slice directly from
             * the slab as a strided view (see layer_ple.c) — the per-layer
             * gather would be seq*n_layers copy dispatches per chunk. */
            layer_ple_buf = per_layer_input_buf;
        } else if (per_layer_input_buf != nullptr) {
            if (st->model_fusions.backend_buffer_copy) {
                /* Device-side gather: keeps batched GPU backends from
                 * flushing their pipeline for a host memcpy each layer. */
                enum geist_status cs = GEIST_OK;
                for (size_t t = 0; cs == GEIST_OK && t < seq; t++) {
                    cs = v->buffer_copy(sess->scratch_ple_lookup,
                                        t * row_bytes_ple,
                                        per_layer_input_buf,
                                        t * row_bytes_per_tok_ple + li * row_bytes_ple,
                                        row_bytes_ple);
                }
                if (cs != GEIST_OK) {
                    return cs;
                }
            } else {
                const uint8_t *src = (const uint8_t *) v->buffer_map(per_layer_input_buf);
                uint8_t       *dst = (uint8_t *) v->buffer_map(sess->scratch_ple_lookup);
                if (src == nullptr || dst == nullptr) {
                    return GEIST_E_BACKEND; /* the backend said why */
                }
                for (size_t t = 0; t < seq; t++) {
                    memcpy(dst + t * row_bytes_ple,
                           src + t * row_bytes_per_tok_ple + li * row_bytes_ple,
                           row_bytes_ple);
                }
                v->buffer_unmap(per_layer_input_buf);
                v->buffer_unmap(sess->scratch_ple_lookup);
            }
            layer_ple_buf = sess->scratch_ple_lookup;
        }

        enum geist_status s = transformer_forward_one_layer(sess,
                                                            (int) li,
                                                            q_position,
                                                            seq,
                                                            /* advance_kv = */ false,
                                                            h_in,
                                                            layer_ple_buf,
                                                            h_out);
        if (s == GEIST_OK && mirror) {
            s = mirror_kv_rows(sess, li, q_position, seq);
        }
        if (s != GEIST_OK) {
            return s;
        }

        struct geist_buffer *tmp = h_in;
        h_in                     = h_out;
        h_out                    = tmp;
    }

    if (mirror) {
        sess->twin_valid = q_position + seq;
        atomic_fetch_add_explicit(&st->offload->mirrored, 1, memory_order_relaxed);
    }
    /* h_in is the last layer's output; copy seq rows to out_h_buf. */
    {
        if (st->model_fusions.backend_buffer_copy) {
            const enum geist_status cs = v->buffer_copy(out_h_buf, 0, h_in, 0, seq * row_bytes_h);
            if (cs != GEIST_OK) {
                return cs;
            }
        } else {
            const uint8_t *src = (const uint8_t *) v->buffer_map(h_in);
            uint8_t       *dst = (uint8_t *) v->buffer_map(out_h_buf);
            if (src == nullptr || dst == nullptr) {
                return GEIST_E_BACKEND; /* the backend said why */
            }
            memcpy(dst, src, seq * row_bytes_h);
            v->buffer_unmap(h_in);
            v->buffer_unmap(out_h_buf);
        }
    }
    return GEIST_OK;
}

/* Dequantize one row of the embed_table into a HIDDEN-sized scratch
 * buffer. Gemma 3/4 multiply the embedding by sqrt(d_model); Llama /
 * BitNet do not. The scale is gated on has_ple because PLE is
 * Gemma-family-exclusive. */
/* Inlined into transformer_decode_step as before batched decode (#26) called
 * it too: the single-token path keeps its code. */
[[nodiscard, gnu::always_inline]] static inline enum geist_status
embed_lookup_and_scale(struct transformer_arch_session *sess,
                       geist_token_t                    token_id,
                       struct geist_buffer             *out_h_buf) {
    struct transformer_arch_state *st = sess->model;

    if (token_id < 0 || (size_t) token_id >= (size_t) st->vocab_size) {
        return GEIST_E_INVALID_ARG;
    }
    struct geist_backend *be    = st->backend;
    const float           scale = st->config.has_ple ? sqrtf((float) st->d_model) : 1.0f;

    const struct geist_backend_fused *fused = geist_backend_fused_tbl(be);
    /* Device path: fused lookup+scale keeps batched GPU backends from
     * dequantizing through a mapped host pointer every token. Bound at
     * plan build. */
    if (st->model_fusions.embed_lookup_scaled) {
        struct geist_tensor     t_out = view_1d(out_h_buf, st->d_model);
        const enum geist_status es =
                fused->embedding_lookup_scaled(be, &st->embed_table, token_id, scale, &t_out);
        if (es != GEIST_OK) {
            return es;
        }
        /* prism.hadamard: token_embd stores rotated rows. */
        return st->rotation.embed_inverse
                       ? transformer_rotate(st, 1, st->d_model, false, true, out_h_buf, out_h_buf)
                       : GEIST_OK;
    }

    /* Host path: the row is gathered on the host (an untied table on a
     * device-copy or unified-memory backend, or a CPU backend). */
    enum geist_status s = transformer_gather_rows(
            sess, &st->embed_table, 1, &token_id, st->d_model, scale, out_h_buf);
    if (s != GEIST_OK) {
        return s;
    }
    return st->rotation.embed_inverse
                   ? transformer_rotate(st, 1, st->d_model, false, true, out_h_buf, out_h_buf)
                   : GEIST_OK;
}

/* Post-seed step: with scratch_h_a already populated with the residual-
 * stream input (either embedded text + sqrt scale OR raw audio soft-token),
 * run PLE → layer loop → output_norm → lm_head → softcap → argmax,
 * stash the result in next_token_pending, advance kv_len by 1.
 *
 * ple_token_id selects which row of the PLE table is looked up:
 *   - Text  : the input token id (PLE row matches the actual token)
 *   - Audio : 0 (pad_token_id) per HF's masked-scatter semantics
 *
 * out_token receives the greedy argmax. */
[[nodiscard]] static enum geist_status
transformer_run_one_step(struct transformer_arch_session *sess,
                         geist_token_t                    ple_token_id,
                         geist_token_t                   *out_token) {
    struct transformer_arch_state *st = sess->model;

    enum geist_status s = transformer_check_kv_room(sess, 1);
    if (s != GEIST_OK) {
        return s;
    }

    /* 1. PLE precompute for this token using the seeded h. Non-PLE
     *    families pass nullptr to run_all_layers, which then skips the
     *    per-layer gather and the PLE injection. */
    struct geist_buffer *ple_buf = nullptr;
    if (st->config.has_ple) {
        s = transformer_compute_per_layer_input(
                sess, ple_token_id, sess->scratch_h_a, sess->scratch_per_layer_input);
        if (s != GEIST_OK) {
            return s;
        }
        ple_buf = sess->scratch_per_layer_input;
    }
    (void) ple_token_id; /* unused when !has_ple */

    /* 2. Layer loop (seq=1 for the single-token path). q_position = current
     *    kv_len; advance after. */
    const size_t q_position = sess->kv_len;
    s                       = transformer_run_all_layers(
            sess, q_position, /* seq = */ 1, sess->scratch_h_a, ple_buf, sess->scratch_h_b);
    if (s != GEIST_OK) {
        return s;
    }
    s = transformer_mtp_sync_target(sess, 1, &ple_token_id, sess->scratch_h_b);
    if (s != GEIST_OK) {
        return s;
    }

    geist_token_t best_id;
    s = finalize_logits_one_row(sess, 0, q_position + 1, &best_id);
    if (s != GEIST_OK) {
        return s;
    }

    /* 3. Advance KV, stash prediction. */
    sess->kv_len = q_position + 1;
    if (sess->kv_kivi_enabled) {
        sess->kivi_residual_count += 1;
        s = transformer_kivi_drain_full(sess);
        if (s != GEIST_OK) {
            return s;
        }
    }
    sess->next_token_pending = best_id;
    sess->logits_valid       = true;

    *out_token = best_id;
    return GEIST_OK;
}

struct transformer_region transformer_region_begin(const struct transformer_arch_state *st,
                                                   enum geist_parallel_region           region) {
    struct transformer_region tok = {0};
    struct geist_backend     *be  = st->backend;
    struct geist_backend     *cpu = st->moe_cpu;
    if (be->desc->vtbl->parallel_region_begin != nullptr)
        tok.be = be->desc->vtbl->parallel_region_begin(be, region);
    if (cpu != nullptr && cpu != be && cpu->desc->vtbl->parallel_region_begin != nullptr)
        tok.moe_cpu = cpu->desc->vtbl->parallel_region_begin(cpu, region);
    return tok;
}

void transformer_region_end(const struct transformer_arch_state *st,
                            struct transformer_region            tok) {
    struct geist_backend *be  = st->backend;
    struct geist_backend *cpu = st->moe_cpu;
    if (cpu != nullptr && cpu != be && cpu->desc->vtbl->parallel_region_end != nullptr)
        cpu->desc->vtbl->parallel_region_end(cpu, tok.moe_cpu);
    if (be->desc->vtbl->parallel_region_end != nullptr)
        be->desc->vtbl->parallel_region_end(be, tok.be);
}

enum geist_status transformer_decode_step(struct transformer_arch_session *sess,
                                          geist_token_t                    input_token,
                                          geist_token_t                   *out_token) {
    struct transformer_arch_state *st = sess->model;
    if (st == nullptr || out_token == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    transformer_recurrent_txn_commit(sess);
    /* Decode is memory-bound; let the backend enter its decode thread regime
     * (cpu_neon caps OMP threads). Restored after the step. */
    const struct transformer_region region_tok =
            transformer_region_begin(st, GEIST_REGION_DECODE_STEP);
    transformer_hist_put(sess, sess->kv_len, 1, &input_token);
    /* Embed the input token into scratch_h_a, scale by sqrt(HIDDEN). */
    enum geist_status s = embed_lookup_and_scale(sess, input_token, sess->scratch_h_a);
    if (s == GEIST_OK) {
        /* PLE uses the token's actual id (text path). */
        s = transformer_run_one_step(sess, input_token, out_token);
    }
    transformer_region_end(st, region_tok);
    return s;
}

enum geist_status transformer_advance_audio_token(struct transformer_arch_session *sess,
                                                  const float                     *h_in_host) {
    struct transformer_arch_state *st = sess->model;
    if (st == nullptr || h_in_host == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    transformer_recurrent_txn_commit(sess);
    struct geist_backend            *be = st->backend;
    const struct geist_backend_vtbl *v  = be->desc->vtbl;

    /* Audio soft-tokens enter the residual stream directly (no embed
     * lookup, no sqrt scale). Copy host bytes into scratch_h_a. */
    {
        const size_t bytes = (size_t) st->d_model * sizeof(float);
        uint8_t     *dst   = (uint8_t *) v->buffer_map(sess->scratch_h_a);
        if (dst == nullptr) {
            return GEIST_E_BACKEND; /* the backend said why */
        }
        memcpy(dst, h_in_host, bytes);
        v->buffer_unmap(sess->scratch_h_a);
    }
    transformer_hist_put(sess, sess->kv_len, 1, nullptr);
    /* PLE token-identity is the pad token (0) per HF masked-scatter. */
    geist_token_t out_unused;
    return transformer_run_one_step(sess, 0, &out_unused);
}

void transformer_session_reset(struct transformer_arch_session *sess) {
    if (sess == nullptr) {
        return;
    }
    transformer_recurrent_txn_commit(sess);
    /* Truncate to pinned prefix length (0 if no prefix has been pinned).
     * The KV state up to prefix_length stays valid in the cache buffers;
     * future prefill/decode appends start at kv_len. */
    sess->kv_len = sess->prefix_length;
    if (sess->kv_kivi_enabled) {
        /* Sync drain + residual counters to the new kv_len. The standard
         * pin_prefix flow pre-prefills with KIVI active, so the counters
         * are already aligned (drained = floor(kv_len/R)*R, residual =
         * remainder). */
        const size_t drained      = sess->kivi_drained_count;
        sess->kivi_drained_count  = (sess->kv_len / KIVI_K_GROUP_SIZE) * KIVI_K_GROUP_SIZE;
        sess->kivi_residual_count = sess->kv_len - sess->kivi_drained_count;
        /* The counters, not the rows: a drain since the pin has moved later
         * tokens over the prefix's residual rows. Nothing else writes
         * there, so without one they are still in place. */
        if (drained != sess->kivi_drained_count) {
            transformer_kivi_pin_restore(sess);
        }
    }
    sess->logits_valid       = false;
    sess->next_token_pending = 0;
    sess->advance_deferred   = false; /* the returned token is dropped, not appended */
    sess->n_ckpt             = 0;
    sess->ckpt_base          = 0;
    transformer_mtp_reset(sess);
    /* Gated-DeltaNet layers carry recurrent state with no rewind — a
     * reset clears it to the empty sequence (#281). pin_prefix refuses a
     * prefix for this family, so prefix_length stays 0. On the host mixer
     * it only marks the state fresh, avoiding a serial memset of the whole
     * state (17 ms on a 27B) ahead of every new prefill. */
    if (sess->dn_fresh != nullptr) {
        for (size_t li = 0; li < sess->model->n_layers; li++)
            sess->dn_fresh[li] = sess->dn_S[li] != nullptr;
    } else if (sess->dn_conv_state != nullptr || sess->dn_S != nullptr) {
        const struct transformer_arch_state *st = sess->model;
        const size_t key_dim                    = st->config.dn_n_k_heads * st->config.dn_head_k;
        const size_t value_dim                  = st->config.dn_n_v_heads * st->config.dn_head_v;
        const size_t conv_n = (st->config.dn_conv_kernel - 1) * (2 * key_dim + value_dim);
        const size_t s_n    = st->config.dn_n_v_heads * st->config.dn_head_k * st->config.dn_head_v;
        for (size_t li = 0; li < st->n_layers; li++) {
            if (sess->dn_conv_state != nullptr && sess->dn_conv_state[li] != nullptr) {
                (void) transformer_dn_state_zero(
                        st->backend, sess->dn_conv_state[li], conv_n * sizeof(float));
            }
            if (sess->dn_S != nullptr && sess->dn_S[li] != nullptr) {
                (void) transformer_dn_state_zero(st->backend, sess->dn_S[li], s_n * sizeof(float));
            }
        }
    }
}

enum geist_status transformer_session_truncate(struct transformer_arch_session *sess, size_t n) {
    if (sess == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct geist_backend *be = sess->model->backend;
    if (n > sess->kv_len || n < sess->prefix_length) {
        geist_backend_set_error(be,
                                GEIST_E_INVALID_ARG,
                                "truncate: %zu is outside [%zu, %zu] (the pinned prefix and "
                                "the session's length)",
                                n,
                                sess->prefix_length,
                                sess->kv_len);
        return GEIST_E_INVALID_ARG;
    }
    if (n == sess->kv_len) {
        return GEIST_OK; /* nothing to drop; pending logits stay valid */
    }
    if (n == 0) {
        transformer_session_reset(sess); /* prefix_length is 0 here */
        return GEIST_OK;
    }
    /* Gated-DeltaNet state has no way back to a position (#281): only to
     * empty, through a snapshot, or to a checkpoint (#192). */
    for (size_t j = 0; j < sess->n_ckpt; j++) {
        if (sess->ckpt_lens[j] == n && !sess->mtp_enabled) {
            const enum geist_status cs = transformer_checkpoint_restore(sess, j);
            if (cs != GEIST_OK)
                return cs;
            sess->logits_valid       = false;
            sess->next_token_pending = 0;
            sess->advance_deferred   = false;
            return GEIST_OK;
        }
    }
    if (sess->dn_conv_state != nullptr || sess->dn_S != nullptr) {
        geist_backend_set_error(be,
                                GEIST_E_UNSUPPORTED,
                                "truncate: the model has recurrent (DeltaNet) layers; use "
                                "geist_session_snapshot / geist_session_restore");
        return GEIST_E_UNSUPPORTED;
    }
    /* The MTP draft head keeps its own cache in step with the target. */
    if (sess->mtp_enabled) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED, "truncate: MTP drafting is enabled");
        return GEIST_E_UNSUPPORTED;
    }
    /* KIVI groups below kivi_drained_count are 2-bit committed and cannot be
     * un-quantized; the speculative rewind clamps there, this refuses. */
    if (sess->kv_kivi_enabled && n < sess->kivi_drained_count) {
        geist_backend_set_error(be,
                                GEIST_E_UNSUPPORTED,
                                "truncate: positions below %zu are in the compressed KIVI region",
                                sess->kivi_drained_count);
        return GEIST_E_UNSUPPORTED;
    }
    transformer_recurrent_txn_commit(sess);
    while (sess->n_ckpt > 0 && sess->ckpt_lens[sess->n_ckpt - 1] > n)
        sess->n_ckpt--; /* checkpoints past the new end are gone */
    sess->kv_len = n;
    if (sess->kv_kivi_enabled) {
        sess->kivi_residual_count = n - sess->kivi_drained_count;
    }
    sess->logits_valid       = false;
    sess->next_token_pending = 0;
    sess->advance_deferred   = false;
    return GEIST_OK;
}

enum geist_status transformer_session_apply_opts(struct transformer_arch_session *sess,
                                                 const struct geist_session_opts *opts) {
    if (sess == nullptr || opts == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    sess->temperature = opts->temperature;
    sess->top_p       = opts->top_p > 0.0f ? opts->top_p : 1.0f;
    sess->top_k       = opts->top_k;
    if (opts->random_seed != 0) {
        geist_rng_seed(&sess->rng, opts->random_seed);
    }

    /* (Re)allocate the sampler workspace if a non-greedy mode is now in
     * play and the workspace isn't already sized for the vocab. ~4 MB
     * for VOCAB=262144; greedy mode skips this. Every non-greedy path —
     * plain temperature included (#331) — samples out of the workspace. */
    const bool needs_ws = sess->temperature > 0.0f;
    if (needs_ws && sess->sampler_ws.n_vocab != (size_t) sess->model->vocab_size) {
        geist_sampler_workspace_destroy(&sess->sampler_ws);
        const enum geist_status ws =
                geist_sampler_workspace_init(&sess->sampler_ws, (size_t) sess->model->vocab_size);
        if (ws != GEIST_OK) {
            return ws;
        }
    }

    /* Repetition control (#695). geist_session_create validated the opts
     * already; the model's default session gets the load-time opts, which
     * did not pass through it. */
    struct geist_sampler_penalty_params pp;
    enum geist_status                   ps = geist_sampler_penalty_params_from_opts(&pp, opts);
    if (ps != GEIST_OK) {
        return ps;
    }
    return geist_sampler_penalties_init(
            &sess->pen, &pp, (size_t) sess->model->vocab_size, sess->max_seq_len);
}

/* Batched decode (#26, declared in arch_state.h). Qwen3.5 family (dense
 * FFN, gated attention) on a backend with the fused DeltaNet mixer, a device
 * buffer copy and a device argmax; 8 rows at most, the range the small-row
 * kernels cover (#186, #221). */
size_t transformer_decode_batch_max(const struct transformer_arch_state *st) {
    if (st == nullptr || st->family != &transformer_family_qwen35 || st->config.has_ple ||
        !st->config.has_attn_output_gate || geist_pooling_is_embedding(st->config.pooling) ||
        st->offload != nullptr) {
        return 0;
    }
    const struct geist_backend_fused *fused = geist_backend_fused_tbl(st->backend);
    if (fused == nullptr || fused->deltanet_mix == nullptr || fused->argmax_f32 == nullptr ||
        st->backend->desc->vtbl->buffer_copy == nullptr || !st->model_fusions.backend_buffer_copy) {
        return 0;
    }
    return st->m_max < TRANSFORMER_DECODE_BATCH_CAP ? st->m_max : TRANSFORMER_DECODE_BATCH_CAP;
}

/* Sample a session's token from logits already in its scratch_logits row 0:
 * the tail of finalize_logits_row (head.c) without the lm_head. */
[[nodiscard]] static enum geist_status batch_sample(struct transformer_arch_session *sess,
                                                    geist_token_t                   *out) {
    struct transformer_arch_state    *st    = sess->model;
    struct geist_backend             *be    = st->backend;
    const struct geist_backend_fused *fused = geist_backend_fused_tbl(be);
    sess->logits_softcapped                 = false;
    sess->logits_sparse                     = false;
    const bool penalties                    = geist_sampler_penalties_active(&sess->pen);
    if (sess->temperature == 0.0f && !penalties) {
        struct geist_tensor t_row = view_2d(sess->scratch_logits, 1, st->vocab_size);
        int32_t             idx   = -1;
        enum geist_status   s     = fused->argmax_f32(be, &t_row, &idx);
        if (s == GEIST_OK && (idx < 0 || (size_t) idx >= (size_t) st->vocab_size))
            s = GEIST_E_BACKEND;
        *out = (geist_token_t) idx;
        return s;
    }
    return transformer_head_sample_row0(sess, sess->kv_len + 1, out);
}

enum geist_status
transformer_decode_batch_forward(size_t n, struct transformer_arch_session *const seqs[static n]) {
    struct transformer_arch_session *carrier = seqs[0];
    struct transformer_arch_state   *st      = carrier->model;
    struct geist_backend            *be      = st->backend;
    const struct geist_backend_vtbl *v       = be->desc->vtbl;
    const size_t                     d       = st->d_model;
    const size_t                     row_h   = d * sizeof(float);
    const size_t                     row_v   = (size_t) st->vocab_size * sizeof(float);
    if (n < 2 || n > transformer_decode_batch_max(st) || n > carrier->m_max) {
        return GEIST_E_INVALID_ARG;
    }
    const struct transformer_region region_tok =
            transformer_region_begin(st, GEIST_REGION_DECODE_STEP);
    enum geist_status s = GEIST_OK;
    /* Embed each session's token; seqs[0]'s lands in its row 0 directly. */
    for (size_t b = 0; s == GEIST_OK && b < n; b++) {
        struct transformer_arch_session *q = seqs[b];
        transformer_recurrent_txn_commit(q);
        transformer_hist_put(q, q->kv_len, 1, &q->next_token_pending);
        s = embed_lookup_and_scale(q, q->next_token_pending, q->scratch_h_a);
        if (s == GEIST_OK && b > 0)
            s = v->buffer_copy(carrier->scratch_h_a, b * row_h, q->scratch_h_a, 0, row_h);
    }
    struct geist_buffer *h_in = carrier->scratch_h_a, *h_out = carrier->scratch_h_b;
    for (size_t li = 0; s == GEIST_OK && li < st->n_layers; li++) {
        s                      = transformer_multi_layer(carrier, n, seqs, (int) li, h_in, h_out);
        struct geist_buffer *t = h_in;
        h_in                   = h_out;
        h_out                  = t;
    }
    /* Head: output norm, rotation and the lm_head once over the n rows. */
    if (s == GEIST_OK && h_in != carrier->scratch_h_a)
        s = v->buffer_copy(carrier->scratch_h_a, 0, h_in, 0, n * row_h);
    struct geist_tensor t_h = view_2d(carrier->scratch_h_a, (int64_t) n, (int64_t) d);
    struct geist_tensor t_w = view_1d(st->output_norm.buffer, (int64_t) d);
    if (s == GEIST_OK)
        s = be->desc->prims->rmsnorm(be, &t_h, &t_w, st->config.rms_eps, &t_h);
    if (s == GEIST_OK)
        s = transformer_rotate_rows(st, n, d, carrier->scratch_h_a);
    struct geist_tensor t_logits = view_2d(carrier->scratch_logits, (int64_t) n, st->vocab_size);
    if (s == GEIST_OK)
        s = linear_w_or_legacy(be,
                               v,
                               carrier->scratch_h_a,
                               carrier->scratch_logits,
                               &st->embed_table_w,
                               n,
                               &t_h,
                               &st->output_table,
                               &t_logits);
    /* Each row to its session (row 0 already is seqs[0]'s): the logits for
     * sampling and peek, the normed hidden for a dense recompute. */
    for (size_t b = 1; s == GEIST_OK && b < n; b++) {
        s = v->buffer_copy(seqs[b]->scratch_logits, 0, carrier->scratch_logits, b * row_v, row_v);
        if (s == GEIST_OK)
            s = v->buffer_copy(seqs[b]->scratch_h_a, 0, carrier->scratch_h_a, b * row_h, row_h);
    }
    for (size_t b = 0; s == GEIST_OK && b < n; b++) {
        struct transformer_arch_session *q    = seqs[b];
        geist_token_t                    next = -1;
        s                                     = batch_sample(q, &next);
        if (s == GEIST_OK) {
            q->kv_len += 1;
            q->next_token_pending = next;
            q->logits_valid       = true;
        }
    }
    transformer_region_end(st, region_tok);
    return s;
}

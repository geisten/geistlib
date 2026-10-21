/*
 * src/archs/transformer/arch_state.c — model and session state lifecycle.
 *
 * Creates the model state from a GGUF (file or memory): the family and its
 * hyperparameters (arch_family.c), the weights staged into backend-owned
 * buffers (weight_load/), the rotation and the execution plan. Allocates
 * and frees sessions: KV cache, recurrent state, scratch.
 *
 * Layer: ARCHITECTURE.
 *
 * On any failure mid-load the partial state is torn down before returning —
 * no leaked buffers. The backend's err_msg is populated with the failing
 * tensor name and reason.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "arch_state.h"
#include "prefill_tuning.h"
#include "arch_family.h"
#include "exec_plan.h"
#include "arch_ops.h"
#include "forward.h"
#include "scratch_plan.h"
#include "weight_load.h"
#include "rotation.h"

#include "gguf_reader.h"
#include "gemma4_kernels.h"
#include "checked.h"
#include "heap.h"
#include "fwht.h"
#include "kivi.h"
#include "ptqtp_awq.h"
#include "parse.h"
#include "hw_probe.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>

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
            .dn_seq_prefill        = env_flag_enabled("GEIST_DN_SEQ_PREFILL", false),
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
        w->flags &= (uint16_t) ~(GEIST_W_AUX_HEAP_OWNED | GEIST_W_AUX_BACKEND_REPACK |
                                 GEIST_W_RAW_COLD);
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
    release_weight_aux(&L->dn_qkv_w);
    release_weight_aux(&L->dn_z_w);
    release_weight_aux(&L->dn_beta_w);
    release_weight_aux(&L->dn_alpha_w);
    release_weight_aux(&L->dn_out_w);
    release_weight_aux(&L->moe_router_w);
    /* The expert arrays: n_experts entries each, all set or all null. */
    struct geist_weight **experts[] = {&L->moe_gate_w, &L->moe_up_w, &L->moe_down_w};
    for (size_t p = 0; p < 3; p++) {
        struct geist_weight *w = *experts[p];
        for (size_t e = 0; w != nullptr && e < L->moe_n_experts; e++)
            release_weight_aux(&w[e]);
        void *pw = w;
        safe_free(&pw);
        *experts[p] = nullptr;
    }
    struct geist_tensor **views[] = {&L->moe_gate, &L->moe_up, &L->moe_down};
    for (size_t p = 0; p < 3; p++) {
        void *pv = *views[p];
        safe_free(&pv);
        *views[p] = nullptr;
    }
}

/* ---- Runtime infrastructure: KV caches, RoPE tables, scratch ---------- */

enum geist_status
transformer_scratch_alloc(struct geist_backend *be, size_t bytes, struct geist_buffer **out) {
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

void transformer_buffer_release(struct geist_backend *be, struct geist_buffer **buf) {
    if (be != nullptr && *buf != nullptr) {
        be->desc->vtbl->buffer_destroy(be, *buf);
    }
    *buf = nullptr;
}

/* A slot of the device-local pool (#488) — a buffer_create_view slice, no
 * host pointer. The pool was zeroed when it was created. */
[[nodiscard]] static enum geist_status alloc_device_slice(struct transformer_arch_session *sess,
                                                          size_t                           bytes,
                                                          struct geist_buffer **out_buf) {
    struct geist_backend *be      = sess->model->backend;
    const size_t          cap     = sess->scratch_dev_pool_bytes;
    size_t                aligned = 0;
    if (geist_ckd_round_up_pow2(sess->scratch_dev_pool_used, 64, &aligned) || aligned > cap ||
        bytes > cap - aligned) {
        geist_backend_set_error(be,
                                GEIST_E_OOM,
                                "transformer: device scratch pool exhausted "
                                "(used %zu, need %zu, capacity %zu)",
                                sess->scratch_dev_pool_used,
                                bytes,
                                cap);
        return GEIST_E_OOM;
    }
    sess->scratch_dev_pool_used = aligned + bytes;
    return be->desc->vtbl->buffer_create_view(
            be, sess->scratch_dev_pool_buf, aligned, bytes, GEIST_BUFFER_SCRATCH, out_buf);
}

/* Bump-allocate a 64-byte-aligned, zeroed slice from the session's scratch
 * pool and wrap it in an aliased buffer. The buffer header is destroyed via
 * buffer_destroy; the bytes belong to the pool, released once in
 * transformer_session_free.
 *
 * `host_mapped`: the host maps this slot (h_a, h_b, logits), so it stays in
 * the host-visible pool even when the session has a device-local one (#488);
 * every other slot then comes from alloc_device_slice. */
[[nodiscard]] static enum geist_status alloc_pool_buffer(struct transformer_arch_session *sess,
                                                         size_t                           bytes,
                                                         bool                  host_mapped,
                                                         struct geist_buffer **out_buf) {

    if (sess->scratch_dev_pool_buf != nullptr && !host_mapped) {
        return alloc_device_slice(sess, bytes, out_buf);
    }
    struct geist_backend *be      = sess->model->backend;
    const size_t          align   = 64;
    const size_t          mask    = align - 1;
    const size_t          aligned = (sess->scratch_pool_used + mask) & ~mask;
    if (aligned + bytes > sess->scratch_pool_bytes) {
        geist_backend_set_error(be,
                                GEIST_E_OOM,
                                "transformer: scratch pool exhausted "
                                "(used %zu, need %zu, capacity %zu)",
                                sess->scratch_pool_used,
                                bytes,
                                sess->scratch_pool_bytes);
        return GEIST_E_OOM;
    }
    void *p = (uint8_t *) sess->scratch_pool_base + aligned;
    memset(p, 0, bytes);
    sess->scratch_pool_used = aligned + bytes;
    return be->desc->vtbl->buffer_create_aliased(be, p, bytes, GEIST_BUFFER_SCRATCH, out_buf);
}

/* Zero `bytes` of a buffer the host cannot map, through views of it: a
 * staged upload per chunk keeps the host block small however large the
 * pool. Session setup only. */
[[nodiscard]] static enum geist_status
zero_unmapped(struct geist_backend *be, size_t bytes, struct geist_buffer *buf) {
    const struct geist_backend_vtbl *v     = be->desc->vtbl;
    constexpr size_t                 CHUNK = 4u << 20;
    uint8_t                         *zeros = heap_calloc_aligned(CHUNK, 1, 64);
    if (zeros == nullptr) {
        geist_backend_set_error(be, GEIST_E_OOM, "transformer: scratch zero block failed");
        return GEIST_E_OOM;
    }
    enum geist_status s = GEIST_OK;
    for (size_t off = 0; off < bytes && s == GEIST_OK; off += CHUNK) {
        const size_t         n    = bytes - off < CHUNK ? bytes - off : CHUNK;
        struct geist_buffer *view = nullptr;
        s = v->buffer_create_view(be, buf, off, n, GEIST_BUFFER_SCRATCH, &view);
        if (s == GEIST_OK) {
            s = v->buffer_upload(view, n, zeros);
            v->buffer_destroy(be, view);
        }
    }
    safe_free((void **) &zeros);
    return s;
}

/* The device-local part of the scratch pool (#488). The backend may still
 * hand back mappable memory (Vulkan under GEIST_VK_SCRATCH_DEVICE=0, a UMA device, or no
 * device-local type left): the buffer is then released and the session keeps
 * one host-visible pool — scratch_dev_pool_buf stays nullptr. */
[[nodiscard]] static enum geist_status alloc_device_pool(struct transformer_arch_session *sess,
                                                         size_t                           bytes) {
    struct geist_backend            *be  = sess->model->backend;
    const struct geist_backend_vtbl *v   = be->desc->vtbl;
    struct geist_buffer             *buf = nullptr;
    enum geist_status                s =
            v->buffer_create(be, bytes, GEIST_BUFFER_SCRATCH, GEIST_MEMORY_DEVICE, &buf);
    if (s != GEIST_OK) {
        return s;
    }
    if (v->buffer_map(buf) != nullptr) {
        v->buffer_unmap(buf);
        v->buffer_destroy(be, buf);
        sess->scratch_device = false;
        return GEIST_OK;
    }
    s = zero_unmapped(be, bytes, buf);
    if (s != GEIST_OK) {
        v->buffer_destroy(be, buf);
        return s;
    }
    sess->scratch_dev_pool_buf   = buf;
    sess->scratch_dev_pool_bytes = bytes;
    sess->scratch_dev_pool_used  = 0;
    return GEIST_OK;
}

/* Layer li's K and V data, k_bytes and v_bytes, in one zeroed buffer, kv_data[li],
 * with *k and *v aliased slices of it (caps.kv_q8_block, caps.kv_dense_block):
 * K at the start, V after K's pages and `shift` bytes into its page, a
 * multiple of 64 (V as aligned as K). A KV head's rows lie a cache row
 * apart, so at a power-of-two row they share their low address bits and take
 * one n_kv_heads-th of the cache sets. K and V as two buffers of their own
 * start at the same offset in a page, as large allocations do: a KV head's K
 * and V rows took the same sets. With V `shift` bytes on, its V rows take
 * another KV head's. */
[[nodiscard]] static enum geist_status alloc_kv_block(struct transformer_arch_session *sess,
                                                      size_t                           li,
                                                      size_t                           k_bytes,
                                                      size_t                           v_bytes,
                                                      size_t                           shift,
                                                      enum geist_buffer_role           role,
                                                      struct geist_buffer            **k,
                                                      struct geist_buffer            **v) {
    struct geist_backend *be     = sess->model->backend;
    size_t                k_span = 0;
    size_t                total  = 0;
    if (geist_ckd_round_up_pow2(k_bytes, 4096, &k_span) || ckd_add(&total, k_span, shift) ||
        ckd_add(&total, total, v_bytes)) {
        geist_backend_set_error(
                be, GEIST_E_OOM, "transformer: a %zu-byte KV cache overflows", k_bytes);
        return GEIST_E_OOM;
    }
    enum geist_status s =
            be->desc->vtbl->buffer_create(be, total, role, GEIST_MEMORY_MAPPED, &sess->kv_data[li]);
    if (s != GEIST_OK) {
        return s;
    }
    uint8_t *base = be->desc->vtbl->buffer_map(sess->kv_data[li]);
    if (base == nullptr) {
        geist_backend_set_error(
                be, GEIST_E_BACKEND, "transformer: the %zu-byte KV cache did not map", total);
        return GEIST_E_BACKEND;
    }
    memset(base, 0, total);
    s = be->desc->vtbl->buffer_create_aliased(be, base, k_bytes, role, k);
    if (s != GEIST_OK) {
        return s;
    }
    return be->desc->vtbl->buffer_create_aliased(be, base + k_span + shift, v_bytes, role, v);
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

/* A representative sliding-attn and full-attn layer for the RoPE tables
 * (the first of each kind; Gemma 4 has two parameter sets, Llama / BitNet
 * one).
 * Families without sliding (BitNet / Llama / Mistral) fall back to layer 0's
 * params for the sliding table too — the table won't be indexed at runtime
 * since is_full is always true. */
static void rope_layers(const struct transformer_arch_state *st, int *sliding_idx, int *full_idx) {
    *sliding_idx = -1;
    *full_idx    = -1;
    for (size_t i = 0; i < st->n_layers; i++) {
        if (*full_idx == -1 && st->layers[i].is_full)
            *full_idx = (int) i;
        if (*sliding_idx == -1 && !st->layers[i].is_full)
            *sliding_idx = (int) i;
    }
    if (*full_idx == -1)
        *full_idx = 0;
    if (*sliding_idx == -1)
        *sliding_idx = *full_idx;
}

/* Floats per position of the sliding and the full RoPE table rows:
 * allocate_runtime_rope sizes the tables by them, the plan reports them. */
static void
rope_row_widths(const struct transformer_arch_state *st, size_t *sl_width, size_t *fl_width) {
    int sl = 0, fl = 0;
    rope_layers(st, &sl, &fl);
    const bool pblock = st->config.rope_partial_block;
    *sl_width         = rope_table_width(
            st->layers[sl].head_dim, (size_t) st->layers[sl].n_rotated_dims, pblock);
    *fl_width = rope_table_width(
            st->layers[fl].head_dim, (size_t) st->layers[fl].n_rotated_dims, pblock);
}

[[nodiscard]] static enum geist_status allocate_runtime_rope(struct transformer_arch_state *st) {

    struct geist_backend *be = st->backend;

    /* Called exactly once, from state_create — the tables are frozen for
     * the model's lifetime so concurrent sessions can read them without
     * coordination (session_alloc rejects larger max_seq_len requests).
     * The destroy loop is a defensive no-op. */
    struct geist_buffer **old[] = {
            &st->rope_cos_sliding,
            &st->rope_sin_sliding,
            &st->rope_cos_full,
            &st->rope_sin_full,
    };
    for (size_t i = 0; i < sizeof old / sizeof old[0]; i++) {
        if (*old[i] != nullptr) {
            be->desc->vtbl->buffer_destroy(be, *old[i]);
            *old[i] = nullptr;
        }
    }

    int sliding_idx = 0, full_idx = 0;
    rope_layers(st, &sliding_idx, &full_idx);

    /* Rotary is pairwise, so an odd head_dim has a channel with no partner
     * (see rope_head_dim_supported). head_dim comes from model metadata —
     * d_model / n_q_heads for the Llama and BitNet families — so a
     * malformed file reaches here with one. Reject at plan construction,
     * once, rather than let every rotary call skip a channel and read the
     * table entry nobody wrote. */
    for (size_t i = 0; i < st->n_layers; i++) {
        const size_t hd = st->layers[i].head_dim;
        if (!rope_head_dim_supported(hd)) {
            geist_backend_set_error(be,
                                    GEIST_E_FORMAT,
                                    "transformer: layer %zu has head_dim %zu; rotary "
                                    "embedding requires an even, non-zero head_dim",
                                    i,
                                    hd);
            return GEIST_E_FORMAT;
        }
    }

    const size_t sliding_hd  = st->layers[sliding_idx].head_dim;
    const size_t full_hd     = st->layers[full_idx].head_dim;
    const size_t sliding_rot = (size_t) st->layers[sliding_idx].n_rotated_dims;
    const size_t full_rot    = (size_t) st->layers[full_idx].n_rotated_dims;
    const float  sliding_th  = st->layers[sliding_idx].rope_theta;
    const float  full_th     = st->layers[full_idx].rope_theta;
    /* The row width is the width the kernels rotate over; which one that
     * is depends on the family's partial-rotary layout (arch_config.h). */
    const bool pblock   = st->config.rope_partial_block;
    size_t     sl_width = 0, fl_width = 0;
    rope_row_widths(st, &sl_width, &fl_width);
    const size_t n_sl_floats = st->max_seq_len * sl_width;
    const size_t n_fl_floats = st->max_seq_len * fl_width;

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
    rope_compute_at(
            0, st->max_seq_len, sliding_hd, sliding_rot, pblock, sliding_th, cos_sl, sin_sl);
    rope_compute_at(0, st->max_seq_len, full_hd, full_rot, pblock, full_th, cos_fl, sin_fl);

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

/* The per-position share of the caches alloc_kv_caches (below) and the
 * qwen35 MTP hook create, branch for branch: keep them in step. Fixed parts (the
 * KIVI residual ring, block alignment, scratch, DeltaNet state) do not grow
 * with the length and are left out; the KIVI per-channel K scales and zeros,
 * one row per KIVI_K_GROUP_SIZE positions, are counted per position,
 * rounded up. */
enum geist_status transformer_kv_bytes_for(size_t                              *out_bytes,
                                           const struct transformer_arch_state *st,
                                           const struct transformer_kv_layout  *kl) {
    if (out_bytes == nullptr || st == nullptr || kl == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out_bytes         = 0;
    const size_t kv    = st->n_kv_heads;
    size_t       total = 0;
    for (size_t li = 0; li < (size_t) st->n_layers; li++) {
        if (st->layers[li].is_kv_shared || st->layers[li].mixer == GEIST_MIXER_DELTANET) {
            continue;
        }
        const size_t row = kv * st->layers[li].head_dim; /* K (or V) elements per position */
        size_t       per = 0;
        if (kl->kivi) {
            const size_t R      = KIVI_K_GROUP_SIZE;
            const size_t k_rows = (2 * row * sizeof(float) + R - 1) / R; /* K scales + zeros */
            per                 = 2 * (row / 4) + 2 * kv * sizeof(float) + k_rows;
        } else if (kl->int8) {
            const size_t v_data = kl->int4_packed ? row / 2 : row * sizeof(int8_t);
            per = row * sizeof(int8_t) + v_data + 2 * kv * sizeof(float); /* + K, V scales */
        } else {
            per = 2 * row * (kl->f16 ? 2u : sizeof(float));
        }
        if (ckd_add(&total, total, per)) {
            return GEIST_E_INVALID_ARG;
        }
    }
    if (st->n_mtp_layers > 0) { /* the draft head's own dense FP32 cache */
        const size_t row = kv * st->mtp_layers[0].block.head_dim;
        if (ckd_add(&total, total, 2 * row * sizeof(float))) {
            return GEIST_E_INVALID_ARG;
        }
    }
    *out_bytes = total;
    return GEIST_OK;
}

enum geist_status transformer_kv_bytes_per_token(size_t                                *out_bytes,
                                                 const struct transformer_arch_session *sess) {
    if (out_bytes == nullptr || sess == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    const struct transformer_kv_layout kl = {.kivi        = sess->kv_kivi_enabled,
                                             .int8        = sess->kv_int8_enabled,
                                             .int4_packed = sess->kv_int4_packed_enabled,
                                             .f16         = sess->kv_f16_enabled};
    return transformer_kv_bytes_for(out_bytes, sess->model, &kl);
}

/* ---- Session state: one alloc_* / free_* pair per part ---------------- *
 *
 * An alloc_* stops at its first failure, with the backend error set, and
 * leaves what it got in the session: transformer_session_free releases it
 * through the matching free_*, which takes every slot as possibly null. The
 * state only one family has is that family's session_alloc / session_free
 * (arch_family.h): DeltaNet, the q gate and MTP (qwen35), MoE routing
 * (qwen3moe). */

/* The 15 per-layer KV slot arrays, in the order they partition the one
 * allocation session_alloc makes; k_cache is its base. */
enum { KV_SLOT_ARRAYS = 15 };

static void kv_slot_arrays(struct transformer_arch_session *sess,
                           struct geist_buffer           ***out[static KV_SLOT_ARRAYS]) {
    struct geist_buffer ***slots[KV_SLOT_ARRAYS] = {
            &sess->k_cache,
            &sess->v_cache,
            &sess->k_cache_q8,
            &sess->v_cache_q8,
            &sess->k_cache_scale,
            &sess->v_cache_scale,
            &sess->k_kivi_q,
            &sess->v_kivi_q,
            &sess->k_kivi_scales,
            &sess->k_kivi_zeros,
            &sess->v_kivi_scales,
            &sess->v_kivi_zeros,
            &sess->k_residual,
            &sess->v_residual,
            &sess->kv_data, /* last: freed after the k/v caches, its slices */
    };
    memcpy(out, slots, sizeof slots);
}

/* Layer li's KIVI cache. The drained region packs at 2 bits per element
 * (4 values/byte). Per-channel K scales/zeros: one fp32 per (group,
 * channel). Per-token V scales/zeros: one fp32 per (token, kv_head). The
 * residual is (R + m_max) fp32 K/V rows so a verify_forward burst of m_max
 * tokens never overflows. */
[[nodiscard]] static enum geist_status alloc_layer_kv_kivi(struct transformer_arch_session *sess,
                                                           size_t                           li) {
    const struct transformer_arch_state *st             = sess->model;
    const size_t                         hd             = st->layers[li].head_dim;
    const size_t                         n_elems        = sess->max_seq_len * st->n_kv_heads * hd;
    const size_t                         R              = KIVI_K_GROUP_SIZE;
    const size_t                         n_drain_groups = (sess->max_seq_len + R - 1) / R;
    const size_t k_scales_bytes = n_drain_groups * st->n_kv_heads * hd * sizeof(float);
    const size_t v_scales_bytes = sess->max_seq_len * st->n_kv_heads * sizeof(float);
    const size_t residual_bytes = (R + sess->m_max) * st->n_kv_heads * hd * sizeof(float);
    const struct {
        size_t                bytes;
        struct geist_buffer **out;
    } bufs[] = {
            {n_elems / 4, &sess->k_kivi_q[li]},
            {n_elems / 4, &sess->v_kivi_q[li]},
            {k_scales_bytes, &sess->k_kivi_scales[li]},
            {k_scales_bytes, &sess->k_kivi_zeros[li]},
            {v_scales_bytes, &sess->v_kivi_scales[li]},
            {v_scales_bytes, &sess->v_kivi_zeros[li]},
            {residual_bytes, &sess->k_residual[li]},
            {residual_bytes, &sess->v_residual[li]},
    };
    for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; i++) {
        enum geist_status s = transformer_scratch_alloc(st->backend, bufs[i].bytes, bufs[i].out);
        if (s != GEIST_OK) {
            return s;
        }
    }
    return GEIST_OK;
}

/* Layer li's INT8 (or INT4) cache and its per-(token, kv_head) scales.
 * INT4 keeps K at INT8 and packs V two values a byte (#53), so its V
 * buffer is half the K buffer; scales are unchanged. hd is a power of two
 * (128/256/512) so n_elems is even. */
[[nodiscard]] static enum geist_status alloc_layer_kv_int8(struct transformer_arch_session *sess,
                                                           size_t                           li) {
    const struct transformer_arch_state *st       = sess->model;
    struct geist_backend                *be       = st->backend;
    const size_t                         hd       = st->layers[li].head_dim;
    const size_t                         n_elems  = sess->max_seq_len * st->n_kv_heads * hd;
    const size_t                         n_scales = sess->max_seq_len * st->n_kv_heads;
    const size_t                         k_bytes  = n_elems * sizeof(int8_t);
    const size_t      v_bytes = sess->kv_int4_packed_enabled ? n_elems / 2 : k_bytes;
    enum geist_status s;
    if (be->desc->caps.kv_q8_block) {
        /* V half a row on (half a page at most, a multiple of 64): the
         * cpu_x86 kernels gained 3-39 % over V at K's offset, more than
         * with V one KV head's slice on. */
        const size_t row   = st->n_kv_heads * hd;
        const size_t shift = ((row < 4096 ? row : 4096) / 2) & ~(size_t) 63;
        s                  = alloc_kv_block(sess,
                                            li,
                                            k_bytes,
                                            v_bytes,
                                            shift,
                                            GEIST_BUFFER_SCRATCH,
                                            &sess->k_cache_q8[li],
                                            &sess->v_cache_q8[li]);
    } else {
        s = transformer_scratch_alloc(be, k_bytes, &sess->k_cache_q8[li]);
        if (s == GEIST_OK) {
            s = transformer_scratch_alloc(be, v_bytes, &sess->v_cache_q8[li]);
        }
    }
    if (s == GEIST_OK) {
        s = transformer_scratch_alloc(be, n_scales * sizeof(float), &sess->k_cache_scale[li]);
    }
    if (s == GEIST_OK) {
        s = transformer_scratch_alloc(be, n_scales * sizeof(float), &sess->v_cache_scale[li]);
    }
    return s;
}

/* Create a KV_CACHE-role buffer of `bytes` and zero it where the host maps
 * it. */
[[nodiscard]] static enum geist_status
alloc_kv_buffer(struct geist_backend *be, size_t bytes, struct geist_buffer **out) {
    enum geist_status s =
            be->desc->vtbl->buffer_create(be, bytes, GEIST_BUFFER_KV_CACHE, GEIST_MEMORY_AUTO, out);
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

/* Layer li's dense (FP32 or F16) cache. Dense KV caches carry the KV_CACHE
 * role: GPU backends keep them in VRAM — the arch only touches them through
 * buffer_copy and prims->attention, never through buffer_map. A buffer the
 * host cannot map stays unzeroed, which is safe: causal attention reads
 * only rows that a prior append wrote. */
[[nodiscard]] static enum geist_status alloc_layer_kv_dense(struct transformer_arch_session *sess,
                                                            size_t                           li) {
    const struct transformer_arch_state *st            = sess->model;
    struct geist_backend                *be            = st->backend;
    const size_t                         hd            = st->layers[li].head_dim;
    const size_t                         n_elems       = sess->max_seq_len * st->n_kv_heads * hd;
    const size_t                         kv_elem_bytes = sess->kv_f16_enabled ? 2u : sizeof(float);
    if (be->desc->caps.kv_dense_block) {
        /* V one KV head's slice on (rounded up to 64 bytes, as aligned as
         * K); with one KV head, whose rows lie next to each other and take
         * every set, not at all. Half a row on, cpu_x86's attention gained
         * more up to 1024 positions but was 0-5 % slower from 4096 on. */
        const size_t shift = st->n_kv_heads > 1 ? (hd * kv_elem_bytes + 63) & ~(size_t) 63 : 0;
        return alloc_kv_block(sess,
                              li,
                              n_elems * kv_elem_bytes,
                              n_elems * kv_elem_bytes,
                              shift,
                              GEIST_BUFFER_KV_CACHE,
                              &sess->k_cache[li],
                              &sess->v_cache[li]);
    }
    enum geist_status s = alloc_kv_buffer(be, n_elems * kv_elem_bytes, &sess->k_cache[li]);
    if (s == GEIST_OK) {
        s = alloc_kv_buffer(be, n_elems * kv_elem_bytes, &sess->v_cache[li]);
    }
    return s;
}

/* The KV caches of every layer that owns one, in the representation the
 * kv_*_enabled flags pick. KV-shared layers alias an owner's and DeltaNet
 * layers carry recurrent state instead (dn_conv_state / dn_S): their slots
 * stay nullptr. */
[[nodiscard]] static enum geist_status alloc_kv_caches(struct transformer_arch_session *sess) {
    const struct transformer_arch_state *st = sess->model;
    for (size_t li = 0; li < (size_t) st->n_layers; li++) {
        if (st->layers[li].is_kv_shared || st->layers[li].mixer == GEIST_MIXER_DELTANET ||
            !transformer_layer_owned(st, li)) {
            continue; /* all KV pointer slots stay nullptr — shared layers
                       * alias an owner; DeltaNet layers carry recurrent
                       * state instead (dn_conv_state / dn_S); the other
                       * part of a layer split (#49) caches its own */
        }
        enum geist_status s = sess->kv_kivi_enabled   ? alloc_layer_kv_kivi(sess, li)
                              : sess->kv_int8_enabled ? alloc_layer_kv_int8(sess, li)
                                                      : alloc_layer_kv_dense(sess, li);
        if (s != GEIST_OK) {
            return s;
        }
    }
    sess->kv_len = 0;
    return GEIST_OK;
}

/* Every per-layer KV buffer, then the slot arrays themselves (one block,
 * k_cache its base). */
static void free_kv_caches(const struct transformer_arch_state *st,
                           struct geist_backend                *be,
                           struct transformer_arch_session     *sess) {
    struct geist_buffer ***slots[KV_SLOT_ARRAYS];
    kv_slot_arrays(sess, slots);
    if (be != nullptr && sess->k_cache != nullptr) {
        for (size_t li = 0; li < (size_t) st->n_layers; li++) {
            for (size_t i = 0; i < KV_SLOT_ARRAYS; i++) {
                transformer_buffer_release(be, &(*slots[i])[li]);
            }
        }
    }
    void *block = sess->k_cache;
    safe_free(&block);
    for (size_t i = 0; i < KV_SLOT_ARRAYS; i++) {
        *slots[i] = nullptr;
    }
}

/* The scratch pool's backing: the device-local part where the session has
 * one (#488), and the host-visible pool, routed through the backend so GPU
 * backends hand out memory they can bind (host-visible VkBuffer / shared
 * MTLBuffer); CPU backends malloc. Slices wrap it via
 * buffer_create_aliased. */
[[nodiscard]] static enum geist_status
alloc_scratch_pool(struct transformer_arch_session       *sess,
                   const struct transformer_scratch_plan *plan) {
    struct geist_backend *be = sess->model->backend;
    sess->scratch_pool_bytes = plan->pool_bytes;
    if (sess->scratch_device) {
        enum geist_status s = alloc_device_pool(sess, plan->pool_bytes - plan->host_bytes);
        if (s != GEIST_OK) {
            return s;
        }
        if (sess->scratch_dev_pool_buf != nullptr) {
            sess->scratch_pool_bytes = plan->host_bytes;
        }
    }
    enum geist_status s = be->desc->vtbl->buffer_create(be,
                                                        sess->scratch_pool_bytes,
                                                        GEIST_BUFFER_SCRATCH,
                                                        GEIST_MEMORY_MAPPED,
                                                        &sess->scratch_pool_buf);
    if (s != GEIST_OK) {
        return s; /* the backend set the error (e.g. a GPU budget, #531) */
    }
    sess->scratch_pool_base = be->desc->vtbl->buffer_map(sess->scratch_pool_buf);
    if (sess->scratch_pool_base == nullptr) {
        geist_backend_set_error(be,
                                GEIST_E_OOM,
                                "transformer: scratch pool alloc failed (%zu bytes)",
                                sess->scratch_pool_bytes);
        return GEIST_E_OOM;
    }
    memset(sess->scratch_pool_base, 0, sess->scratch_pool_bytes);
    sess->scratch_pool_used = 0;
    return GEIST_OK;
}

/* The scratch buffers, sized for m_max tokens (batched prefill; decode
 * touches only the first row), slices of the pool in this order
 * (scratch_plan.c). A slot whose `want` is false stays nullptr: the
 * per-projection input (only with per-projection input norms) and the PLE
 * slots (only with PLE; the backend rejects 0-byte aliased buffers, and the
 * PLE block never runs then). h_a, h_b and the logits are host-mapped. */
[[nodiscard]] static enum geist_status
alloc_scratch_slots(struct transformer_arch_session       *sess,
                    const struct transformer_scratch_plan *p) {
    const bool ple = sess->model->config.has_ple;
    const struct {
        bool                  want;
        size_t                bytes;
        bool                  host_mapped;
        struct geist_buffer **out;
    } slots[] = {
            {true, p->hidden, false, &sess->scratch_normed},
            {p->proj_in > 0, p->proj_in, false, &sess->scratch_proj_in},
            {true, p->q_out, false, &sess->scratch_q},
            {true, p->kv_out, false, &sess->scratch_k},
            {true, p->kv_out, false, &sess->scratch_v},
            {true, p->q_out, false, &sess->scratch_attn},
            {true, p->hidden, false, &sess->scratch_o},
            {true, p->hidden, false, &sess->scratch_post_attn},
            {true, p->hidden, false, &sess->scratch_h_post_attn},
            {true, p->hidden, false, &sess->scratch_pre_ff},
            {true, p->inter, false, &sess->scratch_gate},
            {true, p->inter, false, &sess->scratch_up},
            {true, p->hidden, false, &sess->scratch_ffn_out},
            {true, p->hidden, false, &sess->scratch_post_ff},
            {true, p->hidden, false, &sess->scratch_h_post_ff},
            {ple, p->hidden_per, false, &sess->scratch_gate_ple},
            {ple, p->hidden, false, &sess->scratch_proj_ple},
            /* (a) m_max dequant'd PLE rows during the batched
             * compute_per_layer_inputs, (b) the [seq, HIDDEN_PER_LAYER]
             * per-layer-input slice during the layer loop; (a) is the
             * larger, m_max * PLE_OUT. */
            {ple, p->ple_out, false, &sess->scratch_ple_lookup},
            {ple, p->ple_out, false, &sess->scratch_per_layer_input},
            {true, p->hidden, true, &sess->scratch_h_a},
            {true, p->hidden, true, &sess->scratch_h_b},
            /* min(m_max, TRANSFORMER_LOGITS_ROWS) x VOCAB floats, so
             * verify_forward can do ONE batched lm_head per call (the
             * lm_head is the dominant cost on Pi 5; M>1 IQ kernels amortize
             * the 262K-wide weight stream over k columns). */
            {true, p->vocab, true, &sess->scratch_logits},
    };
    for (size_t i = 0; i < sizeof slots / sizeof slots[0]; i++) {
        if (!slots[i].want) {
            *slots[i].out = nullptr;
            continue;
        }
        enum geist_status s =
                alloc_pool_buffer(sess, slots[i].bytes, slots[i].host_mapped, slots[i].out);
        if (s != GEIST_OK) {
            return s;
        }
    }
    return GEIST_OK;
}

/* Per-forward scratch arena: the attention scores (max_seq_len floats) plus
 * slack, 64 KB. This caps the KIVI scores path at max_seq_len 16384;
 * frame_arena_alloc fails cleanly for longer windows; size this from
 * max_seq_len if KIVI needs them.
 *
 * The INT8 KV attention also takes the partial results of a split decode
 * from here: attention_int8_scratch_floats() for the widest head, 33 KB for
 * Llama-3.2-1B. */
[[nodiscard]] static enum geist_status alloc_scratch_arena(struct transformer_arch_session *sess) {
    const struct transformer_arch_state *st = sess->model;
    sess->scratch_arena_bytes               = 64u * 1024u;
    if (sess->kv_int8_enabled && !sess->kv_int4_packed_enabled) {
        size_t hd_max = 0;
        for (size_t li = 0; li < st->n_layers + st->n_mtp_layers; li++) {
            const size_t hd = li < st->n_layers ? st->layers[li].head_dim
                                                : st->mtp_layers[li - st->n_layers].block.head_dim;
            hd_max          = hd > hd_max ? hd : hd_max;
        }
        sess->scratch_arena_bytes +=
                attention_int8_scratch_floats(st->n_q_heads, hd_max) * sizeof(float) + 64;
    }
    sess->scratch_arena_base = heap_alloc_aligned(sess->scratch_arena_bytes, 64);
    if (sess->scratch_arena_base == nullptr) {
        geist_backend_set_error(st->backend,
                                GEIST_E_OOM,
                                "transformer: scratch arena alloc failed (%zu bytes)",
                                sess->scratch_arena_bytes);
        return GEIST_E_OOM;
    }
    frame_arena_init(&sess->scratch_arena, sess->scratch_arena_base, sess->scratch_arena_bytes);
    return GEIST_OK;
}

/* All-ones row of TRANSFORMER_HEAD_DIM_MAX floats: the weight of the V
 * rmsnorm (the vtable rmsnorm needs a real weight tensor), viewed down to
 * the layer's head_dim per call. Pool-resident: Vulkan's fused
 * attn_qkv_prep needs q/k/v and the norm weights in ONE buffer and returns
 * UNSUPPORTED otherwise, which the plan-time probe does not catch. */
[[nodiscard]] static enum geist_status alloc_ones_row(struct transformer_arch_session       *sess,
                                                      const struct transformer_scratch_plan *plan) {
    struct geist_backend *be = sess->model->backend;
    enum geist_status     s =
            alloc_pool_buffer(sess, plan->ones, false, &sess->scratch_ones_headdim_max);
    if (s != GEIST_OK) {
        return s;
    }
    float *p = (float *) be->desc->vtbl->buffer_map(sess->scratch_ones_headdim_max);
    if (p != nullptr) {
        for (size_t i = 0; i < TRANSFORMER_HEAD_DIM_MAX; i++) {
            p[i] = 1.0f;
        }
        be->desc->vtbl->buffer_unmap(sess->scratch_ones_headdim_max);
        return GEIST_OK;
    }
    /* A device-local slot (#488): stage the ones in. A host slot whose map
     * failed after a lost submit fails here or at the next host access,
     * which reports the sticky error (#474). */
    float ones[TRANSFORMER_HEAD_DIM_MAX];
    for (size_t i = 0; i < TRANSFORMER_HEAD_DIM_MAX; i++) {
        ones[i] = 1.0f;
    }
    return be->desc->vtbl->buffer_upload(
            sess->scratch_ones_headdim_max, sizeof ones, (const uint8_t *) ones);
}

/* The scratch slots (aliased buffers: buffer_destroy releases the header,
 * the bytes are the pool's), the arena and the pool. */
static void free_scratch(struct geist_backend *be, struct transformer_arch_session *sess) {
    if (be != nullptr) {
        struct geist_buffer **slots[] = {
                &sess->scratch_normed,      &sess->scratch_proj_in,
                &sess->scratch_q,           &sess->scratch_k,
                &sess->scratch_v,           &sess->scratch_attn,
                &sess->scratch_o,           &sess->scratch_post_attn,
                &sess->scratch_h_post_attn, &sess->scratch_pre_ff,
                &sess->scratch_gate,        &sess->scratch_up,
                &sess->scratch_ffn_out,     &sess->scratch_post_ff,
                &sess->scratch_h_post_ff,   &sess->scratch_gate_ple,
                &sess->scratch_proj_ple,    &sess->scratch_ones_headdim_max,
                &sess->scratch_ple_lookup,  &sess->scratch_h_a,
                &sess->scratch_h_b,         &sess->scratch_per_layer_input,
                &sess->scratch_logits,
        };
        for (size_t i = 0; i < sizeof slots / sizeof slots[0]; i++) {
            transformer_buffer_release(be, slots[i]);
        }
    }
    if (sess->scratch_arena_base != nullptr) {
        safe_free(&sess->scratch_arena_base);
        sess->scratch_arena_bytes = 0;
        sess->scratch_arena       = (struct frame_arena) {0};
    }
    if (sess->scratch_dev_pool_buf != nullptr) {
        be->desc->vtbl->buffer_destroy(be, sess->scratch_dev_pool_buf);
        sess->scratch_dev_pool_buf   = nullptr;
        sess->scratch_dev_pool_bytes = 0;
        sess->scratch_dev_pool_used  = 0;
    }
    if (sess->scratch_pool_buf != nullptr) {
        be->desc->vtbl->buffer_destroy(be, sess->scratch_pool_buf);
        sess->scratch_pool_buf   = nullptr;
        sess->scratch_pool_base  = nullptr;
        sess->scratch_pool_bytes = 0;
        sess->scratch_pool_used  = 0;
    } else if (sess->scratch_pool_base != nullptr) {
        safe_free(&sess->scratch_pool_base);
        sess->scratch_pool_bytes = 0;
        sess->scratch_pool_used  = 0;
    }
}

/* Session-owned runtime allocations: KV caches, scratch pool and slots,
 * per-forward arena, ones row, then the family's own state. */
[[nodiscard]] static enum geist_status
allocate_runtime_session(struct transformer_arch_session *sess) {
    const struct transformer_arch_state *st = sess->model;
    struct transformer_scratch_plan      plan;
    transformer_scratch_plan_build(st, sess->m_max, &plan);

    enum geist_status s = alloc_kv_caches(sess);
    if (s == GEIST_OK)
        s = alloc_scratch_pool(sess, &plan);
    if (s == GEIST_OK)
        s = alloc_scratch_slots(sess, &plan);
    if (s == GEIST_OK)
        s = alloc_scratch_arena(sess);
    if (s == GEIST_OK)
        s = alloc_ones_row(sess, &plan);
    if (s == GEIST_OK && st->family != nullptr && st->family->session_alloc != nullptr)
        s = st->family->session_alloc(sess);
    return s;
}

/* ---- ZO-tuning gains --------------------------------------------------- *
 *
 * Allocate one f32 per linear weight and point each weight's gain_slot at
 * it. Runs once, after every weight is resolved, so the slot order is a
 * pure function of n_layers — see the contract on transformer_arch_state
 * .gains. Without GEIST_TUNE this is a no-op and st->gains stays null, so
 * the dispatcher's gain path is not merely skipped but absent.
 *
 * Gains are per-tensor, not per-block. The TQ2_0 block scales sit f16 at a
 * 66-byte stride inside the weight bytes and the q8a repack copies them out
 * again, so reaching them needs a backend-layout-aware accessor. */
[[nodiscard]] static enum geist_status gains_wire(struct transformer_arch_state *st) {
#ifndef GEIST_TUNE
    (void) st;
    return GEIST_OK;
#else
    const size_t n = 9u * st->n_layers + 2u;

    st->gains = heap_alloc_array_aligned(float, n);
    if (st->gains == nullptr) {
        return GEIST_E_OOM;
    }
    for (size_t i = 0; i < n; i++) {
        st->gains[i] = 1.0f;
    }
    st->n_gains = n;

    for (size_t l = 0; l < st->n_layers; l++) {
        struct transformer_layer_weights *L     = &st->layers[l];
        struct geist_weight *const        ws[9] = {
                &L->q_proj_w,
                &L->k_proj_w,
                &L->v_proj_w,
                &L->o_proj_w,
                &L->gate_proj_w,
                &L->up_proj_w,
                &L->down_proj_w,
                &L->per_layer_gate_w,
                &L->per_layer_proj_w,
        };
        for (size_t i = 0; i < 9; i++) {
            ws[i]->gain_slot = &st->gains[9u * l + i];
        }
    }
    st->embed_table_w.gain_slot = &st->gains[9u * st->n_layers + 0u];
    st->model_proj_w.gain_slot  = &st->gains[9u * st->n_layers + 1u];

    return GEIST_OK;
#endif
}

/* ---- Public entry points ---------------------------------------------- */

/* Source of transformer_arch_state.snapshot_id. */
static _Atomic uint64_t next_snapshot_id;

/* The model's geometry from its GGUF metadata alone: the family's
 * config populator and the checks on it. No weight is read, nothing is
 * allocated on the backend; on failure the error is set on be and the
 * caller cleans up. Shared by state_create and transformer_plan. */
[[nodiscard]] static enum geist_status
geometry_from_metadata(struct geist_backend             *be,
                       struct gguf_ctx                  *gguf,
                       struct transformer_arch_state    *st,
                       const struct transformer_family **out_fam) {
    *out_fam = nullptr;
    /* NEUTRAL family config: every feature off, every scale zero. A
     * family populator declares only what it HAS (Gemma: PLE + norms +
     * softcap; BitNet: SubLN; qwen3: QK-norms) instead of every other
     * family stripping Gemma defaults. rms_eps carries the one value
     * all families share as a fallback when the meta key is absent. */
    st->config = (struct geist_arch_config) {
            .rms_eps        = 1e-6f,
            .kv_sliding_src = -1,
            .kv_full_src    = -1,
            .ffn_activation = GEIST_FFN_SWIGLU,
    };

    /* Dispatch to the per-family populator selected by
     * `general.architecture`. Unknown / missing arch fails closed —
     * the engine gate (model.c) rejects such GGUFs before we get here,
     * so this branch only fires for callers bypassing the engine. */
    const struct transformer_family *fam = transformer_family_select(gguf);
    *out_fam                             = fam;
    if (fam == nullptr) {
        geist_backend_set_error(be,
                                GEIST_E_UNSUPPORTED,
                                "transformer: unsupported or missing "
                                "general.architecture in GGUF");
        return GEIST_E_UNSUPPORTED;
    }
    if (fam->attention_block == nullptr || fam->ffn_block == nullptr) {
        geist_backend_set_error(
                be, GEIST_E_INTERNAL, "transformer: family %s has no layer forward", fam->name);
        return GEIST_E_INTERNAL;
    }
    st->family = fam;
    fam->populate(gguf, st);
    if (st->n_layers == 0) {
        geist_backend_set_error(be,
                                GEIST_E_FORMAT,
                                "transformer: invalid layer metadata for %s "
                                "(autoregressive=%zu, mtp=%zu)",
                                fam->name,
                                st->n_layers,
                                st->n_mtp_layers);
        return GEIST_E_FORMAT;
    }
    /* The embedding table, the LM head and every token-id check are sized
     * by it; family_vocab_size leaves 0 when the GGUF says nothing. */
    if (st->vocab_size == 0) {
        geist_backend_set_error(be,
                                GEIST_E_FORMAT,
                                "transformer: %s GGUF gives no vocabulary size (no %s.vocab_size "
                                "key, no tokenizer.ggml.tokens)",
                                fam->name,
                                fam->name);
        return GEIST_E_FORMAT;
    }
    /* Head counts are metadata too. Every attention path maps query head h
     * to KV head h / (n_q_heads / n_kv_heads): more KV heads than query
     * heads divides by zero there, and a KV count that does not divide the
     * query count sends the last query heads to a KV head that does not
     * exist. */
    if (st->n_q_heads == 0 || st->n_kv_heads == 0 || st->n_q_heads % st->n_kv_heads != 0) {
        geist_backend_set_error(be,
                                GEIST_E_FORMAT,
                                "transformer: %s head counts %zu (query) and %zu (KV): the KV "
                                "count must divide the query count",
                                fam->name,
                                st->n_q_heads,
                                st->n_kv_heads);
        return GEIST_E_FORMAT;
    }
    return GEIST_OK;
}

/* The per-layer geometry: the layer arrays (host memory) and the
 * family's layer populator, with the head_dim bound. On failure the arrays
 * stay in st for the caller to free. */
[[nodiscard]] static enum geist_status geometry_layers(struct geist_backend            *be,
                                                       struct transformer_arch_state   *st,
                                                       const struct transformer_family *fam) {
    /* Per-layer weight array, freed in state_destroy. */
    st->layers = heap_alloc_aligned(st->n_layers * sizeof(*st->layers),
                                    alignof(struct transformer_layer_weights));
    if (st->layers == nullptr) {
        geist_backend_set_error(be,
                                GEIST_E_OOM,
                                "transformer: layer array alloc failed "
                                "(%zu layers × %zu bytes)",
                                st->n_layers,
                                sizeof(*st->layers));
        return GEIST_E_OOM;
    }
    memset(st->layers, 0, st->n_layers * sizeof(*st->layers));
    if (st->n_mtp_layers > 0) {
        st->mtp_layers = heap_alloc_aligned(st->n_mtp_layers * sizeof(*st->mtp_layers),
                                            alignof(struct transformer_mtp_layer_weights));
        if (st->mtp_layers == nullptr) {
            geist_backend_set_error(be,
                                    GEIST_E_OOM,
                                    "transformer: MTP layer array alloc failed (%zu layers)",
                                    st->n_mtp_layers);
            return GEIST_E_OOM;
        }
        memset(st->mtp_layers, 0, st->n_mtp_layers * sizeof(*st->mtp_layers));
    }

    /* Per-layer geometry from the family populator (Gemma: sliding/full mix
     * + KV sharing from metadata; Llama: uniform full attention);
     * load_one_layer reads these fields. Fails when the metadata doesn't pin
     * the geometry — a named error beats a downstream wiring one (#258). */
    if (!fam->populate_layers(st)) {
        geist_backend_set_error(be,
                                GEIST_E_UNSUPPORTED,
                                "transformer: %s geometry not derivable from GGUF metadata "
                                "(n_layers=%zu, d_model=%zu — missing/inconsistent "
                                "%s.attention.sliding_window_pattern or shared_kv_layers?)",
                                fam->name,
                                st->n_layers,
                                st->d_model,
                                fam->name);
        return GEIST_E_UNSUPPORTED;
    }
    /* head_dim is metadata: a layer asking for more than the forward pass
     * holds (TRANSFORMER_HEAD_DIM_MAX) is refused here, before it can
     * overflow the attention kernels' per-head stack arrays. */
    for (size_t i = 0; i < st->n_layers + st->n_mtp_layers; i++) {
        const size_t hd = i < st->n_layers ? st->layers[i].head_dim
                                           : st->mtp_layers[i - st->n_layers].block.head_dim;
        if (hd > TRANSFORMER_HEAD_DIM_MAX) {
            geist_backend_set_error(be,
                                    GEIST_E_UNSUPPORTED,
                                    "transformer: %s layer %zu head_dim %zu exceeds the "
                                    "supported maximum %zu",
                                    fam->name,
                                    i,
                                    hd,
                                    TRANSFORMER_HEAD_DIM_MAX);
            return GEIST_E_UNSUPPORTED;
        }
    }
    return GEIST_OK;
}

/* A layer split (#49) holds one part of the model per state, which the
 * forward of the other part never sees. Refuse what crosses that line:
 * a table or KV cache layers on both sides read (PLE, shared KV), state
 * the split stage does not carry (DeltaNet recurrence, MTP blocks), and
 * load-time transforms over every layer (rotation, AWQ). */
[[nodiscard]] static enum geist_status split_supported(struct geist_backend                *be,
                                                       struct gguf_ctx                     *gguf,
                                                       const struct transformer_arch_state *st,
                                                       const struct geist_session_opts     *opts) {
    const char *why = nullptr;
    uint32_t    u   = 0;
    if (st->config.has_ple) {
        why = "per-layer embeddings";
    } else if (st->n_mtp_layers > 0) {
        why = "MTP blocks";
    } else if (gguf_get_meta_u32(gguf, "prism.hadamard.version", &u)) {
        why = "a rotated (prism.hadamard) basis";
    } else if (opts != nullptr && opts->awq_scales_path != nullptr) {
        why = "AWQ scales";
    }
    for (size_t li = 0; why == nullptr && li < st->n_layers; li++) {
        if (st->layers[li].is_kv_shared) {
            why = "shared KV caches";
        } else if (st->layers[li].mixer == GEIST_MIXER_DELTANET) {
            why = "DeltaNet layers";
        }
    }
    if (why == nullptr) {
        return GEIST_OK;
    }
    geist_backend_set_error(be,
                            GEIST_E_UNSUPPORTED,
                            "transformer: a layer split does not support %s (%s)",
                            why,
                            st->family != nullptr ? st->family->name : "?");
    return GEIST_E_UNSUPPORTED;
}

/* Shared body: takes ownership of an already-open `gguf` (closes it on
 * error; on success keeps it open for zero-copy weight aliasing or closes
 * it after copying, by mmap_alias_mode). The state holds and runs layers
 * [lo, min(hi, n_layers)); split_stage: only those, no embedding or head
 * (#49). The full model is lo = 0, hi = SIZE_MAX, !split_stage. */
[[nodiscard]] static enum geist_status state_create_range(struct geist_backend            *be,
                                                          struct gguf_ctx                 *gguf,
                                                          const struct geist_session_opts *opts,
                                                          size_t                           lo,
                                                          size_t                           hi,
                                                          bool split_stage,
                                                          bool host_weights,
                                                          struct transformer_arch_state **out) {
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
    st->backend         = be;
    st->split_stage     = split_stage;
    st->weights_on_host = host_weights;
    st->snapshot_id     = atomic_fetch_add(&next_snapshot_id, 1) + 1;
    st->gguf            = (struct gguf_ctx *) gguf;
    st->runtime_flags   = transformer_runtime_flags_from_env();
    /* Structural dims start at ZERO — every family populator fills them
     * from its GGUF metadata (or its own compiled-in defaults). A field
     * left at 0 fails loudly in populate_layers / weight wiring instead
     * of silently inheriting another family's geometry. */
    st->max_seq_len = (opts != nullptr && opts->max_seq_len > 0) ? opts->max_seq_len : 4096;
#if defined(GEIST_TARGET_PI5)
    st->m_max = 64; /* Pi 5: with activation packing (q4_K.c §10.7) a
                     * larger batch amortizes the per-block-row weight
                     * setup; m=128 regresses (working set too large).
                     * GEIST_M_MAX overrides. */
#else
    st->m_max = 64; /* 64 keeps scratch under ~30 MB while feeding the
                     * SGEMM path a decent batch. */
    if (be->desc != nullptr && be->desc->caps.preferred_m_max > 0) {
        st->m_max = be->desc->caps.preferred_m_max; /* backend-measured sweet spot */
    }
#endif
    const struct transformer_family *fam = nullptr;
    enum geist_status                gs  = geometry_from_metadata(be, gguf, st, &fam);
    if (gs != GEIST_OK) {
        void *p = st;
        safe_free(&p);
        gguf_close(gguf);
        return gs;
    }

    /* DeltaNet hybrids prefer SMALL prefill chunks: the chunked
     * delta-rule carries O(C^2) work per chunk (A/attn matrices +
     * substitution), so total prefill cost grows with the chunk size (on
     * Metal, 4B pp512, 64 beats 128 and 256 clearly); attention-only models
     * keep the backend's larger preferred value. GEIST_M_MAX, read below,
     * still wins.
     * Backends whose deltanet_mix sub-chunks internally (caps.dn_subchunk)
     * skip the cap: their DN cost is chunk-size-invariant, and the
     * surrounding GEMMs need the large batch for occupancy (#322). */
    if (st->config.dn_n_v_heads > 0 && st->m_max > 64 &&
        (be->desc == nullptr || !be->desc->caps.dn_subchunk)) {
        st->m_max = 64;
    }

    /* Per-model prefill knobs (prefill_tuning.h): the chunk gets the delta
     * for this family and weight size, and its OpenMP spin policy becomes
     * the process default — here, before weight packing
     * runs the first parallel region. GEIST_M_MAX and
     * GEIST_PREFILL_BLOCKTIME_MS still win. Weight bytes are the GGUF's
     * tensor bytes, the size every model states. */
    {
        size_t weight_bytes = 0;
        for (size_t i = 0; i < gguf_tensor_count(gguf); i++) {
            weight_bytes += gguf_tensor_at(gguf, i)->nbytes;
        }
        const size_t cap = be->desc != nullptr ? be->desc->caps.max_m : 0;
        const struct transformer_prefill_resolved pt = transformer_prefill_resolve(
                &fam->prefill_delta, weight_bytes, st->m_max, cap, nullptr);
        st->m_max          = pt.m_max;
        st->m_max_from_env = pt.m_max_from_env;
        transformer_prefill_apply_blocktime(&pt);
    }

    gs = geometry_layers(be, st, fam);
    if (gs != GEIST_OK) {
        transformer_state_destroy(st);
        return gs;
    }

    if (lo > st->n_layers) {
        geist_backend_set_error(be,
                                GEIST_E_INVALID_ARG,
                                "transformer: split_layers %zu exceeds the model's %zu layers",
                                lo,
                                st->n_layers);
        transformer_state_destroy(st);
        return GEIST_E_INVALID_ARG;
    }
    st->layer_lo = lo;
    st->layer_hi = hi < st->n_layers ? hi : st->n_layers;
    if (split_stage || lo > 0) {
        gs = split_supported(be, gguf, st, opts);
        if (gs != GEIST_OK) {
            transformer_state_destroy(st);
            return gs;
        }
    }

    /* Storage mode (mmap-alias default vs β-mode override). mmap-alias
     * keeps weight bytes aliased to the GGUF mmap and
     * demand-pages disk reads. This is the default because copying the
     * whole file resident blows RSS on memory-constrained targets: e.g.
     * Gemma 4 E2B Q4_K_M is 3.41 GB on disk but only ~1.49 GB is streamed
     * per decode token — the 1.93 GB `per_layer_token_embd` PLE table is
     * lookup-only (one row/token). β-mode copies that 1.93 GB resident,
     * pushing RSS to ~3.7 GB on a 4 GB Pi 5 → thrash. mmap-alias leaves it
     * disk-backed, paging only touched rows — as long as nothing binds the
     * table on a device: Metal pages in a bound buffer whole, so there the
     * host gathers lookup-only tables (caps.lookup_tables_on_host, #529).
     *
     * GEIST_WEIGHT_MMAP=0 forces β-mode (single backend arena, full
     * copy, mmap dropped post-load — full backend ownership, no retained
     * fd). Any other value (or unset) keeps the mmap-alias default. */
    /* GPU backends default to the backend-arena mode: norm weights and
     * embed tables must live inside a backend buffer to be GPU-bindable;
     * mmap pages are not (every touching op would flush + run on CPU). */
    bool mmap_alias_mode = !be->desc->caps.weights_need_backend_arena;
    {
        const char *env = getenv("GEIST_WEIGHT_MMAP");
        if (env != nullptr && env[0] != '\0') {
            mmap_alias_mode = (env[0] != '0');
        }
    }

    if (!mmap_alias_mode) {
        /* First arena chunk from the GGUF's tensor table (the whole model
         * without caps.weights_device_copy); every arena-resident tensor
         * bump-allocates from it and the arena grows past it. Through the
         * backend (see scratch pool): GPU backends can then bind norm
         * weights / embed tables that live inside it. */
        size_t            cap = 0;
        enum geist_status cs  = compute_weight_arena_capacity(be, gguf, st, &cap);
        if (cs == GEIST_OK) {
            cs = weight_arena_open(st, cap);
        }
        if (cs != GEIST_OK) {
            transformer_state_destroy(st);
            return cs;
        }
    }
    /* mmap-alias mode: leave weight_arena == nullptr. load_tensor_to_buffer
     * branches on this to pick its storage path. */

    /* A split stage carries no embedding, output norm or head. */
    enum geist_status s = st->split_stage ? GEIST_OK : load_globals(be, gguf, st);
    if (s != GEIST_OK) {
        transformer_state_destroy(st);
        return s;
    }

    for (size_t i = 0; i < (size_t) st->n_layers; i++) {
        st->layers[i].layer_idx = (int) i;
        if (!transformer_layer_owned(st, i)) {
            continue; /* the other part of a split holds it */
        }
        s = load_one_layer(st, gguf, &st->layers[i]);
        if (s != GEIST_OK) {
            transformer_state_destroy(st);
            return s;
        }
    }
    for (size_t i = 0; i < st->n_mtp_layers; i++) {
        s = load_mtp_layer(st, gguf, &st->mtp_layers[i]);
        if (s != GEIST_OK) {
            transformer_state_destroy(st);
            return s;
        }
    }

    /* prism.hadamard: validate the folded-weight set against the rotated
     * call sites before any session or plan exists. */
    s = transformer_rotation_load(st);
    if (s != GEIST_OK) {
        transformer_state_destroy(st);
        return s;
    }

    s = allocate_runtime_rope(st);
    if (s != GEIST_OK) {
        transformer_state_destroy(st);
        return s;
    }

    /* The default session (KV + scratch pool + arena + sampler) is built
     * lazily by transformer_default_session — see arch_state.h. Keep the
     * opts it will need: transformer_session_alloc reads kv_mode + m_max from
     * them; AUTO falls back to env / platform default. */
    if (opts != nullptr) {
        st->default_opts     = *opts;
        st->has_default_opts = true;
    }

    s = transformer_exec_plan_build(st);
    if (s != GEIST_OK) {
        transformer_state_destroy(st);
        return s;
    }

    /* Spec-head build at load: the model stays immutable afterwards, so
     * concurrent sessions never race a first-use build. A split stage has
     * no head to speed up. */
    if (!st->split_stage) {
        transformer_spec_head_init(st);
    }

    /* AWQ (optional): fold attn_norm/ffn_norm gammas and stash per-layer
     * inv-scales for o/down inputs. Runs after weights are mapped but
     * before any forward pass. In mmap-alias mode the folded norms get
     * buffers of their own (the GGUF pages are read-only, #72). */
    if (opts != nullptr) {
        if (st->rotation.active && opts->awq_scales_path != nullptr) {
            geist_backend_set_error(be,
                                    GEIST_E_UNSUPPORTED,
                                    "transformer: AWQ on a rotated (prism.hadamard) model");
            transformer_state_destroy(st);
            return GEIST_E_UNSUPPORTED;
        }
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
    if (!mmap_alias_mode && !be->desc->caps.weights_device_copy && st->gguf != nullptr) {
        gguf_close((struct gguf_ctx *) st->gguf);
        st->gguf = nullptr;
    }

    if (gains_wire(st) != GEIST_OK) {
        transformer_state_destroy(st);
        return GEIST_E_OOM;
    }

    *out = st;
    return GEIST_OK;
}

enum geist_status transformer_plan(struct geist_backend            *be,
                                   struct gguf_ctx                 *gguf,
                                   const struct geist_session_opts *opts,
                                   size_t                          *kv_bytes_per_token,
                                   size_t                          *model_bytes_per_token) {
    if (be == nullptr || gguf == nullptr || kv_bytes_per_token == nullptr ||
        model_bytes_per_token == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *kv_bytes_per_token    = 0;
    *model_bytes_per_token = 0;
    /* A host-only stand-in for the state: the geometry state_create derives,
     * without the weights, the RoPE tables or any backend buffer. */
    struct transformer_arch_state *st =
            heap_alloc_aligned(sizeof(*st), alignof(struct transformer_arch_state));
    if (st == nullptr) {
        return GEIST_E_OOM;
    }
    memset(st, 0, sizeof(*st));
    st->backend = be;
    st->gguf    = gguf; /* the populators read metadata through it, as in state_create;
                         * borrowed: the caller closes it */
    const struct transformer_family *fam = nullptr;
    enum geist_status                s   = geometry_from_metadata(be, gguf, st, &fam);
    if (s == GEIST_OK) {
        s = geometry_layers(be, st, fam);
    }
    if (s == GEIST_OK) {
        const struct transformer_kv_layout kl = transformer_kv_layout_resolve(st, opts);
        s = transformer_kv_bytes_for(kv_bytes_per_token, st, &kl);
    }
    if (s == GEIST_OK) {
        /* allocate_runtime_rope: cos and sin, for the sliding and the full rows. */
        size_t sl = 0, fl = 0, row = 0;
        rope_row_widths(st, &sl, &fl);
        if (ckd_add(&row, sl, fl) || ckd_mul(model_bytes_per_token, row, 2 * sizeof(float))) {
            s = GEIST_E_INVALID_ARG;
        }
    }
    if (s != GEIST_OK) {
        *kv_bytes_per_token    = 0;
        *model_bytes_per_token = 0;
    }
    safe_free((void **) &st->layers);
    safe_free((void **) &st->mtp_layers);
    safe_free((void **) &st);
    return s;
}

enum geist_status transformer_state_create_from_gguf(struct geist_backend            *be,
                                                     struct gguf_ctx                 *gguf,
                                                     const struct geist_session_opts *opts,
                                                     struct transformer_arch_state  **out) {
    return state_create_range(be, gguf, opts, 0, SIZE_MAX, false, false, out);
}

static bool split_requested(const struct geist_session_opts *opts) {
    return opts != nullptr && opts->split_backend != nullptr && opts->split_layers > 0;
}

size_t transformer_split_layers_fit(size_t       n_layers,
                                    const size_t layer_bytes[static n_layers],
                                    size_t       kv_bytes_per_layer,
                                    size_t       avail) {
    size_t n = 0, used = 0;
    while (n < n_layers && layer_bytes[n] <= SIZE_MAX - used - kv_bytes_per_layer &&
           used + layer_bytes[n] + kv_bytes_per_layer <= avail) {
        used += layer_bytes[n++] + kv_bytes_per_layer;
    }
    return n;
}

/* Planner (#78): split_layers for GEIST_SPLIT_AUTO. The GGUF's tensor table
 * gives each layer's weight bytes ("blk.<i>."), transformer_plan the KV
 * bytes per position on the split backend, opts->max_seq_len (else 4096)
 * the positions; the split backend's free memory, less a reserve for its
 * scratch, the x ring and the weight-reserve the backend keeps itself
 * (a sixteenth of the device and 512 MiB), takes as many leading layers as
 * fit. Unified memory takes them all. */
[[nodiscard]] static enum geist_status
plan_split_layers(struct gguf_ctx *gguf, const struct geist_session_opts *opts, size_t *out) {
    struct geist_backend *sbe = opts->split_backend;
    size_t                n   = 0;
    for (size_t i = 0; i < gguf_tensor_count(gguf); i++) {
        const char *name = gguf_tensor_at(gguf, i)->name;
        if (strncmp(name, "blk.", 4) == 0) {
            const size_t li = (size_t) strtoull(name + 4, nullptr, 10);
            n               = li + 1 > n ? li + 1 : n;
        }
    }
    size_t *bytes = n > 0 ? heap_calloc_aligned(n, sizeof(size_t), alignof(size_t)) : nullptr;
    if (bytes == nullptr) {
        return n > 0 ? GEIST_E_OOM : GEIST_E_FORMAT;
    }
    for (size_t i = 0; i < gguf_tensor_count(gguf); i++) {
        const struct gguf_tensor_t *t = gguf_tensor_at(gguf, i);
        if (strncmp(t->name, "blk.", 4) == 0) {
            const size_t li = (size_t) strtoull(t->name + 4, nullptr, 10);
            bytes[li] += t->nbytes;
        }
    }
    size_t                      kv_tok = 0, model_tok = 0;
    enum geist_status           s = transformer_plan(sbe, gguf, opts, &kv_tok, &model_tok);
    struct geist_backend_memory mem;
    if (s == GEIST_OK) {
        s = geist_backend_memory_info(sbe, &mem);
    }
    if (s == GEIST_OK) {
        const size_t ctx     = opts->max_seq_len > 0 ? opts->max_seq_len : 4096;
        const size_t reserve = (size_t) mem.total_bytes / 16 + ((size_t) 512 << 20);
        const size_t avail   = mem.free_bytes > reserve ? (size_t) mem.free_bytes - reserve : 0;
        size_t       kv      = 0;
        if (ckd_mul(&kv, kv_tok / n, ctx)) {
            kv = SIZE_MAX;
        }
        *out = mem.unified_memory ? n : transformer_split_layers_fit(n, bytes, kv, avail);
        if (getenv("GEIST_VERBOSE") != nullptr) {
            fprintf(stderr,
                    "geist: split planner: %zu of %zu layers on %s (%zu MiB free, %zu MiB "
                    "reserved, %zu MiB KV per layer at %zu positions)\n",
                    *out,
                    n,
                    geist_backend_name(sbe),
                    (size_t) (mem.free_bytes >> 20),
                    reserve >> 20,
                    kv >> 20,
                    ctx);
        }
    }
    safe_free((void **) &bytes);
    return s;
}

/* Layer split (#49): `st` (loaded on its backend from `gguf`, holding
 * layers [split_layers, n_layers)) gets the stage that holds [0,
 * split_layers) on opts->split_backend, loaded from `gguf2`, a second
 * handle on the same file. Both parts get the smaller prefill chunk:
 * every chunk crosses the boundary whole. Takes ownership of gguf2; on
 * failure destroys `st`. */
[[nodiscard]] static enum geist_status attach_split(struct transformer_arch_state   *st,
                                                    struct gguf_ctx                 *gguf2,
                                                    const struct geist_session_opts *opts) {
    struct transformer_arch_state *sub = nullptr;
    enum geist_status              s   = state_create_range(
            opts->split_backend, gguf2, opts, 0, opts->split_layers, true, false, &sub);
    if (s != GEIST_OK) {
        geist_backend_set_error(st->backend,
                                s,
                                "transformer: split stage on '%s': %s",
                                geist_backend_name(opts->split_backend),
                                geist_backend_errmsg(opts->split_backend));
        transformer_state_destroy(st);
        return s;
    }
    st->split = sub;
    if (sub->m_max < st->m_max) {
        st->m_max = sub->m_max;
    } else {
        sub->m_max = st->m_max;
    }
    if (getenv("GEIST_VERBOSE") != nullptr) {
        fprintf(stderr,
                "geist: layer split: layers 0..%zu on %s, layers %zu..%zu, embedding and head "
                "on %s\n",
                sub->layer_hi - 1,
                geist_backend_name(sub->backend),
                st->layer_lo,
                st->n_layers - 1,
                geist_backend_name(st->backend));
    }
    return GEIST_OK;
}

/* Offload (#78): the row threshold for the split model `st` with its stage
 * on `sbe`; 0 = off. GEIST_OFFLOAD_M_MIN=<rows> sets it, "0" turns it off.
 * Unset: 64 rows (from there the GPU path at least ties the split and
 * then wins, runs 1388/1396), where it can pay and fits: a discrete GPU
 * (with unified memory the whole model fits the GPU itself), and host
 * memory for the twins' copy of the load-backend layers' matrices with a
 * fifth to spare. Off when the split stage holds every layer: the twins
 * would have none, and each session's twin IO buffer would be refused. */
static size_t offload_m_min(const struct transformer_arch_state *st, struct geist_backend *sbe) {
    if (st->layer_lo >= st->n_layers) {
        return 0;
    }
    const char *v = getenv("GEIST_OFFLOAD_M_MIN");
    if (v != nullptr && *v != '\0') {
        return (size_t) strtoull(v, nullptr, 10);
    }
    struct geist_backend_memory mem;
    if (geist_backend_memory_info(sbe, &mem) != GEIST_OK || mem.unified_memory) {
        return 0;
    }
    size_t copy = 0;
    for (size_t li = st->layer_lo; li < st->n_layers; li++) {
        const struct transformer_layer_weights *L = &st->layers[li];
        copy += L->q_proj_w.raw_nbytes + L->k_proj_w.raw_nbytes + L->v_proj_w.raw_nbytes +
                L->o_proj_w.raw_nbytes + L->gate_proj_w.raw_nbytes + L->up_proj_w.raw_nbytes +
                L->down_proj_w.raw_nbytes;
    }
    const size_t avail = geist_mem_available();
    if (avail == 0 || copy > avail / 5 * 4) {
        if (getenv("GEIST_VERBOSE") != nullptr) {
            fprintf(stderr,
                    "geist: offload off: its copy of %zu MiB needs more host memory than the "
                    "%zu MiB available (GEIST_OFFLOAD_M_MIN overrides)\n",
                    copy >> 20,
                    avail >> 20);
        }
        return 0;
    }
    return 64;
}

static void offload_destroy(struct transformer_offload *o) {
    if (o == nullptr) {
        return;
    }
    if (getenv("GEIST_VERBOSE") != nullptr && o->twins != nullptr) {
        fprintf(stderr,
                "geist: offload: %llu prefill chunks ran on %s, %llu steps mirrored there\n",
                (unsigned long long) atomic_load(&o->chunks),
                geist_backend_name(o->twins->backend),
                (unsigned long long) atomic_load(&o->mirrored));
    }
    transformer_state_destroy(o->twins);
    void *p = o;
    safe_free(&p);
}

/* Offload (#78): `st`, a split model's load-backend part, gets its layers
 * [layer_lo, n_layers) a second time on the split backend, weights only,
 * loaded from `gguf3` (a third handle on the same file; owned). Their
 * matrices are GEIST_W_HOST: the backend keeps them in host memory and
 * streams them for large batches (#79), so the VRAM the split stage left
 * stays for the sessions' KV caches and scratch. On failure destroys `st`. */
[[nodiscard]] static enum geist_status attach_offload(struct transformer_arch_state   *st,
                                                      struct gguf_ctx                 *gguf3,
                                                      const struct geist_session_opts *opts,
                                                      size_t                           m_min) {
    struct transformer_offload *o =
            heap_calloc_aligned(1, sizeof *o, alignof(struct transformer_offload));
    enum geist_status s = o != nullptr ? GEIST_OK : GEIST_E_OOM;
    if (o == nullptr) {
        gguf_close(gguf3);
    } else {
        o->m_min = m_min;
        s        = state_create_range(
                opts->split_backend, gguf3, opts, st->layer_lo, SIZE_MAX, true, true, &o->twins);
    }
    if (s != GEIST_OK) {
        geist_backend_set_error(st->backend,
                                s,
                                "transformer: offload twins on '%s': %s",
                                geist_backend_name(opts->split_backend),
                                geist_backend_errmsg(opts->split_backend));
        offload_destroy(o);
        transformer_state_destroy(st);
        return s;
    }
    st->offload = o;
    if (o->twins->m_max < st->m_max) {
        st->m_max = o->twins->m_max;
    }
    if (getenv("GEIST_VERBOSE") != nullptr) {
        fprintf(stderr,
                "geist: offload: prefill chunks of %zu rows or more run layers %zu..%zu on %s\n",
                m_min,
                st->layer_lo,
                st->n_layers - 1,
                geist_backend_name(o->twins->backend));
    }
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
    struct geist_session_opts planned;
    if (split_requested(opts) && opts->split_layers == GEIST_SPLIT_AUTO) {
        planned                    = *opts;
        const enum geist_status ps = plan_split_layers(gguf, opts, &planned.split_layers);
        if (ps != GEIST_OK) {
            gguf_close(gguf);
            return ps;
        }
        opts = &planned;
    }
    if (!split_requested(opts)) {
        return transformer_state_create_from_gguf(be, gguf, opts, out);
    }
    enum geist_status s =
            state_create_range(be, gguf, opts, opts->split_layers, SIZE_MAX, false, false, out);
    if (s != GEIST_OK) {
        return s;
    }
    struct gguf_ctx *gguf2 = gguf_open(gguf_path, &err);
    if (gguf2 == nullptr) {
        geist_backend_set_error(be,
                                GEIST_E_FILE_NOT_FOUND,
                                "transformer: gguf_open(%s) for the split stage",
                                gguf_path);
        transformer_state_destroy(*out);
        *out = nullptr;
        return GEIST_E_FILE_NOT_FOUND;
    }
    s                  = attach_split(*out, gguf2, opts);
    const size_t m_min = s == GEIST_OK ? offload_m_min(*out, opts->split_backend) : 0;
    if (m_min > 0) {
        struct gguf_ctx *gguf3 = gguf_open(gguf_path, &err);
        s = gguf3 != nullptr ? attach_offload(*out, gguf3, opts, m_min) : GEIST_E_FILE_NOT_FOUND;
        if (gguf3 == nullptr) {
            transformer_state_destroy(*out);
        }
    }
    if (s != GEIST_OK) {
        *out = nullptr;
    }
    return s;
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
    struct geist_session_opts planned;
    if (split_requested(opts) && opts->split_layers == GEIST_SPLIT_AUTO) {
        planned                    = *opts;
        const enum geist_status ps = plan_split_layers(gguf, opts, &planned.split_layers);
        if (ps != GEIST_OK) {
            gguf_close(gguf);
            return ps;
        }
        opts = &planned;
    }
    if (!split_requested(opts)) {
        return transformer_state_create_from_gguf(be, gguf, opts, out);
    }
    enum geist_status s =
            state_create_range(be, gguf, opts, opts->split_layers, SIZE_MAX, false, false, out);
    if (s != GEIST_OK) {
        return s;
    }
    struct gguf_ctx *gguf2 = gguf_open_memory(data, size, &err);
    if (gguf2 == nullptr) {
        geist_backend_set_error(
                be, GEIST_E_FORMAT, "transformer: gguf_open_memory for the split stage");
        transformer_state_destroy(*out);
        *out = nullptr;
        return GEIST_E_FORMAT;
    }
    s                  = attach_split(*out, gguf2, opts);
    const size_t m_min = s == GEIST_OK ? offload_m_min(*out, opts->split_backend) : 0;
    if (m_min > 0) {
        struct gguf_ctx *gguf3 = gguf_open_memory(data, size, &err);
        s = gguf3 != nullptr ? attach_offload(*out, gguf3, opts, m_min) : GEIST_E_FORMAT;
        if (gguf3 == nullptr) {
            transformer_state_destroy(*out);
        }
    }
    if (s != GEIST_OK) {
        *out = nullptr;
    }
    return s;
}

void transformer_state_destroy(struct transformer_arch_state *st) {
    if (st == nullptr) {
        return;
    }
    struct geist_backend *be = st->backend;

    /* Gains first: every weight's gain_slot aliases this array, so it must
     * outlive nothing and die before the weights it points into. */
    safe_free((void **) &st->gains);
    st->n_gains = 0;

    /* Tear down the default session first. Engine-owned sessions must have
     * been destroyed via geist_session_destroy by now. */
    if (st->default_sess != nullptr) {
        transformer_session_free(st, st->default_sess);
        st->default_sess = nullptr;
    }
    /* Its sessions are gone with ours (split_sess). */
    transformer_state_destroy(st->split);
    st->split = nullptr;
    offload_destroy(st->offload);
    st->offload = nullptr;

    if (be != nullptr) {
        /* The layer array may be missing: its allocation failed. */
        for (size_t l = 0; st->layers != nullptr && l < (size_t) st->n_layers; l++) {
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
        for (size_t l = 0; st->mtp_layers != nullptr && l < st->n_mtp_layers; l++) {
            struct transformer_mtp_layer_weights *M = &st->mtp_layers[l];
            struct transformer_layer_weights     *L = &M->block;
            release_layer_weight_aux(L);
            release_weight_aux(&M->eh_proj_w);
            for (size_t b = 0; b < L->n_bufs; b++) {
                if (L->bufs[b] != nullptr) {
                    be->desc->vtbl->buffer_destroy(be, L->bufs[b]);
                }
            }
        }
        release_weight_aux(&st->embed_table_w);
        release_weight_aux(&st->model_proj_w);
        transformer_rotation_release(st);
        safe_free((void **) &st->spec_sketch);
        safe_free((void **) &st->spec_row_scale);
        safe_free((void **) &st->spec_dims);
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
    /* Release the weight arena once. The per-weight buffers were aliased
     * slices of it; their buffer_destroy above did not free the bytes. */
    for (size_t i = 0; i < st->n_weight_arena_chunks; i++) {
        st->backend->desc->vtbl->buffer_destroy(st->backend, st->weight_arena_chunks[i]);
    }
    {
        void *chunks = st->weight_arena_chunks;
        safe_free(&chunks);
    }
    st->weight_arena_chunks     = nullptr;
    st->n_weight_arena_chunks   = 0;
    st->cap_weight_arena_chunks = 0;
    st->weight_arena            = nullptr;
    st->weight_arena_capacity   = 0;
    st->weight_arena_used       = 0;
    safe_free(&st->rope_il_rows); /* same lifetime as the arena slices */
    st->rope_il_rows_capacity = 0;
    st->rope_il_rows_used     = 0;
    /* Their buffer handles are gone with the layers and globals. */
    for (size_t i = 0; i < st->n_host_weights; i++) {
        safe_free(&st->host_weights[i]);
    }
    {
        void *list = st->host_weights;
        safe_free(&list);
    }
    st->host_weights     = nullptr;
    st->n_host_weights   = 0;
    st->cap_host_weights = 0;
    transformer_exec_plan_destroy(st);
    if (st->layers != nullptr) {
        void *p_layers = st->layers;
        safe_free(&p_layers);
        st->layers = nullptr;
    }
    if (st->mtp_layers != nullptr) {
        void *p_mtp = st->mtp_layers;
        safe_free(&p_mtp);
        st->mtp_layers = nullptr;
    }
    void *p = st;
    safe_free(&p);
}

/* ---- Multi-session API ------------------------------------------------- */

[[nodiscard]] static enum geist_kv_mode
resolve_kv_mode_requested(const struct geist_backend      *be,
                          enum geist_kv_mode               family,
                          const struct geist_session_opts *opts) {
    enum geist_kv_mode m = (opts != nullptr) ? opts->kv_mode : GEIST_KV_AUTO;
    if (m != GEIST_KV_AUTO) {
        return m;
    }
    const char *env_kivi = getenv("GEIST_KV_KIVI");
    const char *env_int8 = getenv("GEIST_KV_INT8");
    const char *env_int4 = getenv("GEIST_KV_INT4");
    if (env_int4 != nullptr && env_int4[0] == '1') {
        return GEIST_KV_INT4;
    }
    if (env_kivi != nullptr && env_kivi[0] == '1') {
        return GEIST_KV_KIVI;
    }
    if (env_int8 != nullptr) {
        return (env_int8[0] == '1') ? GEIST_KV_INT8 : GEIST_KV_FP32;
    }
    /* The model family's preference, where it has one (transformer_family
     * .kv_auto): a precision need of the model, so it outranks the
     * backend's speed default. */
    if (family != GEIST_KV_AUTO) {
        return family;
    }
    /* Backend preference (caps.preferred_kv_mode): the backend owns its
     * platform/pipeline knowledge — GPUs keep dense KV on-device,
     * Apple-unified-memory CPU builds prefer FP32, Pi-class LPDDR
     * builds prefer INT8. No preference (= GEIST_KV_AUTO) → FP32. */
    if (be != nullptr && be->desc->caps.preferred_kv_mode != GEIST_KV_AUTO) {
        return be->desc->caps.preferred_kv_mode;
    }
    return GEIST_KV_FP32;
}

/* The requested mode, but never INT4 on a family where it misses the
 * quality bar (transformer_family.kv_int4_instead): that family gets its
 * replacement, and stderr says so once per process. */
[[nodiscard]] static enum geist_kv_mode resolve_kv_mode(const struct geist_backend      *be,
                                                        const struct transformer_family *family,
                                                        const struct geist_session_opts *opts) {
    const enum geist_kv_mode m = resolve_kv_mode_requested(
            be, family != nullptr ? family->kv_auto : GEIST_KV_AUTO, opts);
    if (m != GEIST_KV_INT4 || family == nullptr || family->kv_int4_instead == GEIST_KV_AUTO) {
        return m;
    }
    static atomic_bool told = false;
    if (!atomic_exchange(&told, true)) {
        fprintf(stderr,
                "geist: the INT4 KV cache misses the quality bar on %s models (#53); using %s "
                "instead\n",
                family->name,
                family->kv_int4_instead == GEIST_KV_INT8  ? "INT8"
                : family->kv_int4_instead == GEIST_KV_F16 ? "F16"
                                                          : "FP32");
    }
    return family->kv_int4_instead;
}

/* The KV cache layout a session gets: kv_mode, the GEIST_KV_* env and the
 * backend decide it. Shared by session_alloc and the plan, so the two
 * cannot disagree. */
struct transformer_kv_layout
transformer_kv_layout_resolve(const struct transformer_arch_state *state,
                              const struct geist_session_opts     *opts) {
    struct transformer_kv_layout kl = {};
    /* KV-mode resolution: opts override > env > backend/platform default. */
    const enum geist_kv_mode mode = resolve_kv_mode(state->backend, state->family, opts);
    kl.kivi                       = (mode == GEIST_KV_KIVI);
    kl.int8                       = (mode == GEIST_KV_INT8);
    /* Issue #61: packed 4-bit KV rides the INT8 storage path (buffer alloc +
     * ctx wiring), with half-size data buffers holding 2 values/byte. */
    kl.int4_packed = (mode == GEIST_KV_INT4);
    if (kl.int4_packed) {
        kl.int8 = true;
    }
    /* Issue #61: low-bit quality-sim reuses the INT8 storage path with an
     * N-bit quant grid (no packing, no memory win). GEIST_KV_QBITS=N (2..8)
     * forces INT8 storage on. Resolve before the rot flag so rotation sees
     * it. Ignored under the real packed-INT4 mode. */
    {
        int         qbits     = 0;
        const char *env_qbits = getenv("GEIST_KV_QBITS");
        long        q         = 0;
        if (!kl.int4_packed && geist_parse_long(env_qbits, &q)) {
            if (q >= 2 && q <= 8) {
                qbits = (q == 8) ? 0 : (int) q; /* 8-bit is the native path */
            }
        }
        kl.sim_qbits = qbits;
        if (qbits != 0) {
            kl.int8 = true;
            kl.kivi = false;
        }
    }
    /* Issue #61: Hadamard rotation, only meaningful on the INT8 storage path,
     * where it is on by default: INT4 without it is a quality cliff, and
     * INT8 without it misses the quality bar on QK-norm models, whose K and
     * Q rows carry outlier channels (#53: Qwen3-0.6B, teacher-forced
     * against FP32 KV, mean JS 0.016 nats and 82 % top-1 unrotated, 0.0013
     * and 96 % rotated). GEIST_KV_ROT=0 opts out, =1 forces on. */
    {
        const char *env_rot     = getenv("GEIST_KV_ROT");
        const bool  rot_default = kl.int8;
        const bool  rot_on      = (env_rot != nullptr) ? (env_rot[0] == '1') : rot_default;
        kl.rot                  = kl.int8 && rot_on;
    }
    /* F16 cache: explicit request (session, family), or AUTO-resolved
     * FP32 upgraded on a backend that does so by default
     * (caps.kv_f16_by_default). It needs attention that reads F16 K/V
     * (caps.kv_f16_attention) and the fused converting append (env
     * GEIST_KV_F16=0 forces FP32, =1 requests it under AUTO). Without
     * them F16 degrades to FP32. */
    {
        const bool  slot_ok = state->backend != nullptr &&
                              state->backend->desc->caps.kv_f16_attention &&
                              geist_backend_fused_tbl(state->backend)->kv_append_f16 != nullptr;
        const char *env_f16 = getenv("GEIST_KV_F16");
        bool        want    = mode == GEIST_KV_F16;
        if (mode == GEIST_KV_FP32 && (opts == nullptr || opts->kv_mode == GEIST_KV_AUTO) &&
            state->backend != nullptr) {
            want = env_f16 != nullptr ? env_f16[0] == '1'
                                      : state->backend->desc->caps.kv_f16_by_default;
        }
        kl.f16 = want && slot_ok && !(env_f16 != nullptr && env_f16[0] == '0');
    }
    return kl;
}

/* Whether this backend serves a SCRATCH buffer asked for device-local as
 * memory the host cannot map (#488) and can slice one by offset. Vulkan
 * does unless GEIST_VK_SCRATCH_DEVICE=0; every other backend either
 * has no buffer_create_view or maps everything. */
static bool backend_scratch_unmappable(struct geist_backend *be) {
    bool mapped = true;
    return be->desc->vtbl->buffer_create_view != nullptr &&
           transformer_buffer_probe_mappable(
                   be, GEIST_BUFFER_SCRATCH, GEIST_MEMORY_DEVICE, &mapped) &&
           !mapped;
}

/* Whether this session puts its scratch pool (#488), all but h_a, h_b and
 * logits, in device-local memory. Only where no host path maps one of
 * those slots: every op the forward runs on them must be the backend's, and
 * a host fallback the analysis missed must fail (buffer_map returns
 * nullptr; the Vulkan host views count a refusal) rather than read the
 * bytes over the bus. So, conservatively, not with:
 *   - a quantized or KIVI KV cache, or a rotated one: kv_store.c quantizes
 *     scratch_k / scratch_v and the INT8 attention reads scratch_q on the
 *     host;
 *   - per-layer embeddings without the on-device row lookup, an attention
 *     output gate without the backend's attn_qgate_split and sigmoid_mul,
 *     MTP heads, BitNet SubLN or per-projection norms, AWQ scales: each has
 *     a host loop over pool slots (layer.c, blocks.c, mtp.c,
 *     internal.h). A prism.hadamard rotation is fine: it only runs as the
 *     backend's hadamard_rotate (the model does not load otherwise). So is
 *     a DeltaNet mixer: its host oracle (layer_deltanet.c, taken when
 *     deltanet_mix declines) maps the dn_scratch_* slabs and the recurrent
 *     state, which are buffers of their own, not pool slots; everything
 *     around it on the pool (input norm, projections, residual add) is
 *     backend ops;
 *   - a backend without device buffer_copy or scale_f32, whose bound
 *     fallbacks are host memcpy / host loops (exec_plan.h).
 * Then the backend decides (backend_scratch_unmappable). */
/* The default chunk of a session whose scratch pool is device-local (see
 * transformer_session_alloc). */
constexpr size_t DEVICE_POOL_M_MAX = 512;

static bool scratch_device_wanted(const struct transformer_arch_session *sess) {
    const struct transformer_arch_state *st = sess->model;
    if (st->backend == nullptr || st->backend->desc == nullptr) {
        return false;
    }
    if (sess->kv_kivi_enabled || sess->kv_int8_enabled || sess->kv_rot_enabled ||
        sess->mtp_enabled) {
        return false;
    }
    /* PLE's host loops (layer.c: the row gather, the model_proj scale, the
     * add+scale combine) all sit behind !ple_lookup_scaled or
     * !prim_scale_f32; with both bound PLE runs on the backend (#488). A
     * host gather that uploads its rows (transformer_lookup_on_host) maps
     * no pool slot. */
    const bool ple_on_host = st->config.has_ple && !st->model_fusions.ple_lookup_scaled &&
                             !transformer_lookup_on_host(st->backend);
    /* qwen35's attention output gate (blocks.c): the query/gate split
     * writes scratch_q and the sigmoid gate scratch_attn on the host unless
     * the backend has the op. Vulkan's ops decline only on a shape the forward
     * never passes or a buffer without a VkBuffer behind it. */
    const struct geist_backend_fused *fused = geist_backend_fused_tbl(st->backend);
    const bool gate_on_host = st->config.has_attn_output_gate &&
                              (fused->attn_qgate_split == nullptr || fused->sigmoid_mul == nullptr);
    if (ple_on_host || gate_on_host || st->config.has_sub_ln ||
        st->config.has_projection_input_norms || !st->model_fusions.backend_buffer_copy ||
        !st->model_fusions.prim_scale_f32) {
        return false;
    }
    for (size_t li = 0; li < st->n_layers; li++) {
        const struct transformer_layer_weights *L = &st->layers[li];
        if (L->o_awq_inv_scale != nullptr || L->down_awq_inv_scale != nullptr) {
            return false;
        }
    }
    return backend_scratch_unmappable(st->backend);
}

/* Layer split (#49): the session on the split stage, with this session's
 * chunk, window and KV mode, and the host rows that carry the hidden state
 * across. KIVI drains every layer in lock-step from one counter, which two
 * sessions cannot share: refused. */
[[nodiscard]] static enum geist_status split_session_alloc(struct transformer_arch_session *sess,
                                                           const struct geist_session_opts *opts) {
    struct transformer_arch_state *st = sess->model;
    if (sess->kv_kivi_enabled) {
        geist_backend_set_error(st->backend,
                                GEIST_E_UNSUPPORTED,
                                "transformer_session_alloc: a layer split does not support KIVI");
        return GEIST_E_UNSUPPORTED;
    }
    struct geist_session_opts so = {0};
    so.max_seq_len               = sess->max_seq_len;
    so.m_max                     = sess->m_max;
    so.kv_mode                   = opts != nullptr ? opts->kv_mode : GEIST_KV_AUTO;
    sess->split_sess             = transformer_session_alloc(st->split, &so);
    if (sess->split_sess == nullptr) {
        geist_backend_set_error(st->backend,
                                GEIST_E_BACKEND,
                                "transformer_session_alloc: split stage on '%s': %s",
                                geist_backend_name(st->split->backend),
                                geist_backend_errmsg(st->split->backend));
        return GEIST_E_BACKEND;
    }
    if (sess->split_sess->kv_kivi_enabled) {
        geist_backend_set_error(st->backend,
                                GEIST_E_UNSUPPORTED,
                                "transformer_session_alloc: a layer split does not support KIVI");
        return GEIST_E_UNSUPPORTED;
    }
    size_t n = 0;
    if (ckd_mul(&n, sess->m_max, st->d_model)) {
        return GEIST_E_INVALID_ARG;
    }
    sess->split_rows = heap_alloc_array_aligned(float, n);
    if (sess->split_rows == nullptr) {
        geist_backend_set_error(st->backend, GEIST_E_OOM, "transformer_session_alloc: split rows");
        return GEIST_E_OOM;
    }
    /* An optimisation: without it (no buffer_copy, no memory for it) the
     * read-back maps the stage's scratch as before. */
    struct geist_backend            *sbe = st->split->backend;
    const struct geist_backend_vtbl *sv  = sbe->desc->vtbl;
    if (sv->buffer_copy != nullptr &&
        sv->buffer_create(
                sbe, n * sizeof(float), GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &sess->split_h_io) !=
                GEIST_OK) {
        sess->split_h_io = nullptr;
    }
    return GEIST_OK;
}

static void twin_session_free(struct transformer_arch_state   *state,
                              struct transformer_arch_session *sess) {
    if (state == nullptr || state->offload == nullptr) {
        return;
    }
    struct geist_backend            *tbe = state->offload->twins->backend;
    const struct geist_backend_vtbl *tv  = tbe->desc->vtbl;
    if (sess->twin_kv_io != nullptr) {
        tv->buffer_destroy(tbe, sess->twin_kv_io);
        sess->twin_kv_io = nullptr;
    }
    if (sess->twin_h_io != nullptr) {
        tv->buffer_destroy(tbe, sess->twin_h_io);
        sess->twin_h_io = nullptr;
    }
    transformer_session_free(state->offload->twins, sess->twin_sess);
    sess->twin_sess = nullptr;
}

/* Offload (#78): the session on the twins, with this session's chunk and
 * window and a dense KV cache, whose rows the load backend reads back: this
 * session's own mode when that is dense (FP32, F16), else F16 where the
 * backend appends it, else FP32; and the IO buffers the
 * chunk's K/V rows ([n_twin_layers, 2, m_max, kv_out], at most FP32) and
 * output rows come back through. */
[[nodiscard]] static enum geist_status twin_session_alloc(struct transformer_arch_session *sess) {
    struct transformer_arch_state *st = sess->model;
    struct transformer_arch_state *tw = st->offload->twins;
    struct geist_session_opts      so = {0};
    so.max_seq_len                    = sess->max_seq_len;
    so.m_max                          = sess->m_max;
    const bool dense =
            !sess->kv_int8_enabled && !sess->kv_kivi_enabled && !sess->kv_int4_packed_enabled;
    so.kv_mode = sess->kv_f16_enabled                                             ? GEIST_KV_F16
                 : dense                                                          ? GEIST_KV_FP32
                 : geist_backend_fused_tbl(tw->backend)->kv_append_f16 != nullptr ? GEIST_KV_F16
                                                                                  : GEIST_KV_FP32;
    sess->twin_sess = transformer_session_alloc(tw, &so);
    if (sess->twin_sess == nullptr) {
        geist_backend_set_error(st->backend,
                                GEIST_E_BACKEND,
                                "transformer_session_alloc: offload twins on '%s': %s",
                                geist_backend_name(tw->backend),
                                geist_backend_errmsg(tw->backend));
        return GEIST_E_BACKEND;
    }
    const struct transformer_arch_session *ts = sess->twin_sess;
    if (ts->kv_int8_enabled || ts->kv_kivi_enabled || ts->kv_int4_packed_enabled) {
        geist_backend_set_error(st->backend,
                                GEIST_E_UNSUPPORTED,
                                "transformer_session_alloc: offload needs a dense KV cache on '%s' "
                                "(a GEIST_KV_* override asked for a quantized one)",
                                geist_backend_name(tw->backend));
        return GEIST_E_UNSUPPORTED;
    }
    size_t kv_out = 0;
    for (size_t li = st->layer_lo; li < st->n_layers; li++) {
        kv_out = st->layers[li].kv_out > kv_out ? st->layers[li].kv_out : kv_out;
    }
    size_t kv_bytes = 0, h_bytes = 0;
    if (ckd_mul(&kv_bytes, st->n_layers - st->layer_lo, 2 * sess->m_max) ||
        ckd_mul(&kv_bytes, kv_bytes, kv_out * sizeof(float)) ||
        ckd_mul(&h_bytes, sess->m_max, st->d_model * sizeof(float))) {
        return GEIST_E_INVALID_ARG;
    }
    const struct geist_backend_vtbl *v = tw->backend->desc->vtbl;
    enum geist_status                s = v->buffer_create(
            tw->backend, kv_bytes, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &sess->twin_kv_io);
    if (s == GEIST_OK) {
        s = v->buffer_create(
                tw->backend, h_bytes, GEIST_BUFFER_IO, GEIST_MEMORY_AUTO, &sess->twin_h_io);
    }
    return s;
}

/* A session refused before its slot arrays exist: the handle and the
 * embedding accumulator are all it owns. */
static void session_free_shell(struct transformer_arch_session *sess) {
    void *acc = sess->embedding_acc;
    safe_free(&acc);
    void *p = sess;
    safe_free(&p);
}

/* The session handle with its embedding accumulator, model, chunk and
 * window. nullptr, with the backend error set, on a failed allocation or a
 * max_seq_len past the model's. */
static struct transformer_arch_session *session_shell(struct transformer_arch_state   *state,
                                                      const struct geist_session_opts *opts) {
    struct geist_backend            *be = state->backend;
    struct transformer_arch_session *sess =
            heap_alloc_aligned(sizeof(*sess), alignof(struct transformer_arch_session));
    if (sess == nullptr) {
        geist_backend_set_error(
                be, GEIST_E_OOM, "transformer_session_alloc: %zu bytes failed", sizeof(*sess));
        return nullptr;
    }
    memset(sess, 0, sizeof(*sess));
    /* Mean pooling sums across prefill chunks, so the accumulator outlives
     * the scratch the chunks reuse. Allocated once with the session -- 4 KB
     * at the widest geometry in tree. */
    sess->embedding_acc = heap_alloc_array_aligned(float, state->d_model);
    if (sess->embedding_acc == nullptr) {
        void *p = sess;
        safe_free(&p);
        geist_backend_set_error(be,
                                GEIST_E_OOM,
                                "transformer_session_alloc: embedding accumulator (%zu floats)",
                                (size_t) state->d_model);
        return nullptr;
    }
    sess->model       = state;
    sess->mtp_enabled = state->n_mtp_layers > 0 && env_flag_enabled("GEIST_MTP", false);
    sess->m_max       = (opts != nullptr && opts->m_max > 0) ? opts->m_max : state->m_max;
    /* Honor opts->max_seq_len up to the model's frozen bound. Sessions
     * may run SMALLER windows (their KV caches are sized to req_seq);
     * larger is rejected — the shared RoPE tables and geometry are
     * fixed at state_create so concurrent sessions can read them
     * without coordination. Load the model with the larger
     * max_seq_len instead. */
    const size_t req_seq =
            (opts != nullptr && opts->max_seq_len > 0) ? opts->max_seq_len : state->max_seq_len;
    if (req_seq > state->max_seq_len) {
        geist_backend_set_error(be,
                                GEIST_E_INVALID_ARG,
                                "transformer_session_alloc: max_seq_len %zu exceeds the "
                                "model's %zu (fixed at model load; reload the model with "
                                "a larger max_seq_len)",
                                req_seq,
                                state->max_seq_len);
        session_free_shell(sess);
        return nullptr;
    }
    sess->max_seq_len = req_seq;
    return sess;
}

/* The per-layer KV slot arrays: one zeroed allocation of KV_SLOT_ARRAYS *
 * n_layers pointers, partitioned; free_kv_caches frees it. */
[[nodiscard]] static bool alloc_kv_slot_arrays(struct transformer_arch_session *sess) {
    const size_t          n_layers = sess->model->n_layers;
    const size_t          kv_bytes = KV_SLOT_ARRAYS * n_layers * sizeof(struct geist_buffer *);
    struct geist_buffer **kv_block = heap_alloc_aligned(kv_bytes, alignof(struct geist_buffer *));
    if (kv_block == nullptr) {
        geist_backend_set_error(sess->model->backend,
                                GEIST_E_OOM,
                                "transformer_session_alloc: KV slot array "
                                "alloc failed (%zu × %zu bytes)",
                                (size_t) KV_SLOT_ARRAYS,
                                n_layers * sizeof(struct geist_buffer *));
        return false;
    }
    memset(kv_block, 0, kv_bytes);
    struct geist_buffer ***slots[KV_SLOT_ARRAYS];
    kv_slot_arrays(sess, slots);
    for (size_t i = 0; i < KV_SLOT_ARRAYS; i++) {
        *slots[i] = kv_block + i * n_layers;
    }
    return true;
}

/* The session's KV mode (transformer_kv_layout_resolve). The rotation
 * silently no-ops on a head_dim the FWHT can't handle (non-power-of-two, or
 * > 512) — packing/quant still run, so you get the unrotated (worse) cache
 * with no signal. Warn once so a misconfiguration reads as deliberate, not
 * a silent quality loss. The architecture layer has no log callback, so
 * this warning goes to stderr once per session. */
static void apply_kv_layout(struct transformer_arch_session *sess,
                            const struct geist_session_opts *opts) {
    const struct transformer_arch_state *state = sess->model;
    const struct transformer_kv_layout   kl    = transformer_kv_layout_resolve(state, opts);
    sess->kv_kivi_enabled                      = kl.kivi;
    sess->kv_int8_enabled                      = kl.int8;
    sess->kv_int4_packed_enabled               = kl.int4_packed;
    sess->kv_sim_qbits                         = kl.sim_qbits;
    sess->kv_rot_enabled                       = kl.rot;
    sess->kv_f16_enabled                       = kl.f16;
    if (sess->kv_rot_enabled) {
        for (size_t li = 0; li < state->n_layers; li++) {
            const size_t hd = state->layers[li].head_dim;
            if (!state->layers[li].is_kv_shared && (!fwht_supported(hd) || hd > 512)) {
                fprintf(stderr,
                        "geist: KV rotation requested but head_dim=%zu is not a power of two "
                        "<= 512; rotation disabled on affected layers (cache left unrotated).\n",
                        hd);
                break;
            }
        }
    }
    sess->kivi_residual_count = 0;
    sess->kivi_drained_count  = 0;
}

/* The default chunk (opts and GEIST_M_MAX win) fitted to the backend.
 * m_cap: caps.max_m, SIZE_MAX when uncapped.
 *
 * A pool that does not fit the backend's fast host-visible window lands in
 * system memory and every activation op then runs over PCIe (Bonsai 27B at
 * m 128 without resizable BAR: 3x slower prefill), so shrink to what fits,
 * never below 64. With a device-local pool only its host-mapped slots need
 * the window. */
static void fit_default_m_max(struct transformer_arch_session *sess, size_t m_cap) {
    const struct transformer_arch_state *state = sess->model;
    struct geist_backend                *be    = state->backend;
    if (state->m_max_from_env) {
        return;
    }
    if (be->desc != nullptr && be->desc->vtbl->fast_host_bytes != nullptr) {
        const size_t                    fast = be->desc->vtbl->fast_host_bytes(be);
        struct transformer_scratch_plan fit;
        transformer_scratch_plan_build(state, sess->m_max, &fit);
        size_t need = sess->scratch_device ? fit.host_bytes : fit.pool_bytes;
        /* slack for the per-session slabs outside the pool (DeltaNet, q gate) */
        while (sess->m_max > 64 && need + need / 4 > fast) {
            sess->m_max = sess->m_max / 2 < 64 ? 64 : sess->m_max / 2;
            transformer_scratch_plan_build(state, sess->m_max, &fit);
            need = sess->scratch_device ? fit.host_bytes : fit.pool_bytes;
        }
        /* ...and a host-visible pool of a small model grows while the
         * window still holds it, as a device-local pool does below: the
         * bigger chunk feeds the GEMMs. BitNet b1.58-large (sub-LN keeps
         * its pool host-visible) on an RTX 2080 Ti, pp512 at 128 -> 512:
         * 6830 -> 13160 t/s (#739). */
        while (!sess->scratch_device && sess->m_max < DEVICE_POOL_M_MAX &&
               sess->m_max * 2 <= m_cap) {
            transformer_scratch_plan_build(state, sess->m_max * 2, &fit);
            if (fit.pool_bytes + fit.pool_bytes / 4 > fast) {
                break;
            }
            sess->m_max *= 2;
        }
    }
    /* A device-local pool takes the BAR window out of the chunk's limits
     * (only h_a, h_b and the logits rows stay host-visible), and a bigger
     * chunk feeds the GEMMs: Gemma 4 E2B pp512 on an RTX 2080 Ti 1641 ->
     * 1813 t/s at 512, E4B 754 -> 839, Llama 3.2 3B 1229 -> 1298. Grow to
     * 512 when the device has room for the bigger pool, with half of what
     * is free left for the KV cache and everything else. Sessions that
     * keep a host-visible pool (quantized KV, ...) never get here: their
     * chunk stays what the BAR allows. */
    if (sess->scratch_device && sess->m_max < DEVICE_POOL_M_MAX && DEVICE_POOL_M_MAX <= m_cap) {
        struct geist_backend_memory     mem;
        struct transformer_scratch_plan now, big;
        transformer_scratch_plan_build(state, sess->m_max, &now);
        transformer_scratch_plan_build(state, DEVICE_POOL_M_MAX, &big);
        const size_t grow = big.pool_bytes - now.pool_bytes;
        if (geist_backend_memory_info(be, &mem) == GEIST_OK && !mem.unified_memory &&
            grow <= mem.free_bytes / 2) {
            sess->m_max = DEVICE_POOL_M_MAX;
        }
    }
}

struct transformer_arch_session *transformer_session_alloc(struct transformer_arch_state   *state,
                                                           const struct geist_session_opts *opts) {
    if (state == nullptr) {
        return nullptr;
    }
    struct geist_backend            *be   = state->backend;
    struct transformer_arch_session *sess = session_shell(state, opts);
    if (sess == nullptr) {
        return nullptr;
    }
    if (!alloc_kv_slot_arrays(sess)) {
        session_free_shell(sess);
        return nullptr;
    }
    /* KV-mode resolution: opts override > env > backend/platform default. */
    apply_kv_layout(sess, opts);

    /* Sampler defaults (greedy); apply_opts may override below. */
    sess->temperature = 0.0f;
    sess->top_p       = 1.0f;
    sess->top_k       = 0;
    sess->sampler_ws  = (struct geist_sampler_workspace) {0};
    sess->pen         = (struct geist_sampler_penalties) {0};
    geist_rng_seed(&sess->rng, 0xCAFEBABE1234ULL);
    /* One token id per position (#695): 4 bytes against the KV cache's
     * kilobytes per position. */
    sess->tok_hist = heap_alloc_array_aligned(geist_token_t, sess->max_seq_len);
    if (sess->tok_hist == nullptr) {
        geist_backend_set_error(be,
                                GEIST_E_OOM,
                                "transformer_session_alloc: token history (%zu positions)",
                                sess->max_seq_len);
        transformer_session_free(state, sess);
        return nullptr;
    }

    /* After the KV mode: whether the scratch pool can be device-local
     * depends on it (scratch_device_wanted). */
    sess->scratch_device = scratch_device_wanted(sess);
    /* caps.max_m is the backend's per-call row limit (CPU quant kernels
     * size stack arrays from it; batched-submit GPUs allow larger
     * batches). 0 = uncapped. */
    size_t m_cap =
            (be->desc != nullptr && be->desc->caps.max_m > 0) ? be->desc->caps.max_m : SIZE_MAX;
    /* Offload (#78): the load backend runs only chunks within its own limit
     * (transformer_prefill_chunk); the larger ones run on the split backend,
     * whose limit binds below. */
    if (state->offload != nullptr) {
        m_cap = SIZE_MAX;
    }
    /* Layer split (#49): every chunk runs on both backends, so the split
     * stage's row limit binds too, and the default chunk shrinks to fit. */
    if (state->split != nullptr) {
        const struct geist_backend *sbe = state->split->backend;
        if (sbe->desc != nullptr && sbe->desc->caps.max_m > 0 && sbe->desc->caps.max_m < m_cap) {
            m_cap = sbe->desc->caps.max_m;
        }
        if ((opts == nullptr || opts->m_max == 0) && sess->m_max > m_cap) {
            sess->m_max = m_cap;
        }
    }
    if (opts == nullptr || opts->m_max == 0) {
        fit_default_m_max(sess, m_cap);
    }
    if (sess->m_max == 0 || sess->m_max > m_cap) {
        geist_backend_set_error(
                be,
                GEIST_E_INVALID_ARG,
                "transformer_session_alloc: m_max=%zu outside supported range 1..%zu",
                sess->m_max,
                m_cap);
        transformer_session_free(state, sess);
        return nullptr;
    }

    enum geist_status s = allocate_runtime_session(sess);
    if (s == GEIST_OK && !transformer_spec_session_scratch_alloc(sess)) {
        s = GEIST_E_OOM;
    }
    if (s == GEIST_OK && state->split != nullptr) {
        s = split_session_alloc(sess, opts);
    }
    if (s == GEIST_OK && state->offload != nullptr) {
        /* An optimisation: a session whose twins do not fit (their KV cache
         * and scratch on a full card) runs as a plain split. */
        s = twin_session_alloc(sess);
        if (s != GEIST_OK && s != GEIST_E_UNSUPPORTED) {
            fprintf(stderr,
                    "geist: offload off for this session (%s): %s\n",
                    geist_status_to_string(s),
                    geist_backend_errmsg(state->offload->twins->backend));
            twin_session_free(state, sess);
            s = GEIST_OK;
        }
    }
    if (s == GEIST_OK && opts != nullptr) {
        s = transformer_session_apply_opts(sess, opts);
    }
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

    if (state != nullptr && state->split != nullptr && sess->split_h_io != nullptr) {
        state->split->backend->desc->vtbl->buffer_destroy(state->split->backend, sess->split_h_io);
        sess->split_h_io = nullptr;
    }
    if (sess->split_sess != nullptr) {
        transformer_session_free(state->split, sess->split_sess);
        sess->split_sess = nullptr;
    }
    void *split_rows = sess->split_rows;
    safe_free(&split_rows);
    sess->split_rows = nullptr;
    twin_session_free(state, sess);

    /* The family's own state first (arch_family.h). */
    if (state != nullptr && state->family != nullptr && state->family->session_free != nullptr) {
        state->family->session_free(sess);
    }

    void *acc = sess->embedding_acc;
    safe_free(&acc);
    sess->embedding_acc = nullptr;
    void *tail          = sess->kivi_pin_tail;
    safe_free(&tail);
    sess->kivi_pin_tail = nullptr;
    void *rows          = sess->lookup_rows;
    safe_free(&rows);
    sess->lookup_rows        = nullptr;
    sess->lookup_rows_floats = 0;

    free_kv_caches(state, be, sess);
    free_scratch(be, sess);
    geist_sampler_workspace_destroy(&sess->sampler_ws);
    geist_sampler_penalties_destroy(&sess->pen);
    safe_free((void **) &sess->tok_hist);
    transformer_spec_session_scratch_free(sess);

    void *sp = sess;
    safe_free(&sp);
}

struct transformer_arch_session *transformer_default_session(struct transformer_arch_state *state) {
    if (state == nullptr) {
        return nullptr;
    }
    if (state->default_sess == nullptr) {
        /* the spec-head sketch exists by now, so session_alloc also builds the
         * session's spec scratch */
        state->default_sess = transformer_session_alloc(
                state, state->has_default_opts ? &state->default_opts : nullptr);
    }
    return state->default_sess;
}

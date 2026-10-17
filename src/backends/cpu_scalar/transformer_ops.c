/*
 * src/backends/cpu_scalar/transformer_ops.c — rope_apply, embedding_lookup,
 * attention via the gemma4_kernels.c reference kernels.
 *
 * Layer: BACKEND.
 *
 * The gemma4_kernels.c rope/attention kernels are pure C, so they serve
 * directly as cpu_scalar's reference implementation.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "internal.h"

#include "gemma4_kernels.h"
#include "tensor_view.h"

#include <geist.h>
#include <geist_backend.h>

#include <stdint.h>
#include <string.h>

static float *get_f32_dense_ptr_full(const struct geist_tensor *t, size_t *out_n) {
    if (t == nullptr || t->dtype != GEIST_DTYPE_F32 || t->layout != GEIST_LAYOUT_DENSE ||
        t->buffer == nullptr || t->ndim < 1) {
        return nullptr;
    }
    size_t n = 1;
    for (int d = 0; d < t->ndim; d++) {
        if (t->shape[d] <= 0)
            return nullptr;
        n *= (size_t) t->shape[d];
    }
    *out_n = n;
    return (float *) ((uint8_t *) t->buffer->host + t->offset);
}

[[nodiscard]] enum geist_status cpu_scalar_rope_apply(struct geist_backend      *be,
                                                      struct geist_tensor       *x,
                                                      const struct geist_tensor *cos,
                                                      const struct geist_tensor *sin) {
    if (be == nullptr || x == nullptr || cos == nullptr || sin == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t       nx, nc, ns;
    float       *xp   = get_f32_dense_ptr_full(x, &nx);
    const float *cosp = get_f32_dense_ptr_full(cos, &nc);
    const float *sinp = get_f32_dense_ptr_full(sin, &ns);
    if (xp == nullptr || cosp == nullptr || sinp == nullptr || x->ndim != 3) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_scalar rope_apply: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
    /* The cos/sin row width is the rotated width (see rope_apply). */
    rope_apply((size_t) x->shape[0],
               (size_t) x->shape[1],
               (size_t) x->shape[2],
               (size_t) cos->shape[cos->ndim - 1],
               xp,
               cosp,
               sinp);
    return GEIST_OK;
}

[[nodiscard]] enum geist_status cpu_scalar_embedding_lookup(struct geist_backend      *be,
                                                            const struct geist_tensor *embed_table,
                                                            geist_token_t              token_id,
                                                            struct geist_tensor       *out) {
    if (be == nullptr || embed_table == nullptr || out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t       n_table, n_out;
    const float *tablep = get_f32_dense_ptr_full(embed_table, &n_table);
    float       *outp   = get_f32_dense_ptr_full(out, &n_out);
    if (tablep == nullptr || outp == nullptr || embed_table->ndim != 2) {
        return GEIST_E_INVALID_ARG;
    }
    int64_t vocab_size = embed_table->shape[0];
    int64_t d_model    = embed_table->shape[1];
    if (token_id < 0 || (int64_t) token_id >= vocab_size || n_out != (size_t) d_model) {
        geist_backend_set_error(be,
                                GEIST_E_INVALID_ARG,
                                "cpu_scalar embedding_lookup: token_id %d, d_model %lld, "
                                "n_out %zu",
                                (int) token_id,
                                (long long) d_model,
                                n_out);
        return GEIST_E_INVALID_ARG;
    }
    memcpy(outp, tablep + (size_t) token_id * (size_t) d_model, (size_t) d_model * sizeof(float));
    return GEIST_OK;
}

/* An F16 DENSE 3-D view's halves, or nullptr. */
static const uint16_t *get_f16_dense_3d(const struct geist_tensor *t, size_t *out_n) {
    if (t == nullptr || t->buffer == nullptr)
        return nullptr;
    return (const uint16_t *) geist_tensor_dense(
            t, GEIST_DTYPE_F16, sizeof(uint16_t), 3, t->buffer->host, t->buffer->bytes, out_n);
}

enum geist_status cpu_scalar_attention_f16(struct geist_backend      *be,
                                           const struct geist_tensor *q,
                                           const struct geist_tensor *k,
                                           const struct geist_tensor *v,
                                           size_t                     q_offset,
                                           size_t                     sliding_window,
                                           struct geist_tensor       *out) {
    size_t          nq = 0, nk = 0, nv = 0, no = 0;
    const float    *qp = get_f32_dense_ptr_full(q, &nq);
    const uint16_t *kp = get_f16_dense_3d(k, &nk);
    const uint16_t *vp = get_f16_dense_3d(v, &nv);
    float          *op = get_f32_dense_ptr_full(out, &no);
    if (qp == nullptr || kp == nullptr || vp == nullptr || op == nullptr || q->ndim != 3 ||
        out->ndim != 3 || k->shape[1] != v->shape[1] || k->shape[0] != v->shape[0] ||
        k->shape[2] != q->shape[2] || v->shape[2] != q->shape[2] ||
        q->shape[1] % k->shape[1] != 0) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "attention: bad F16 KV views");
        return GEIST_E_INVALID_ARG;
    }
    if (!attention_mqa_causal_kv_f16((size_t) q->shape[0],
                                     (size_t) k->shape[0],
                                     q_offset,
                                     (size_t) q->shape[1],
                                     (size_t) k->shape[1],
                                     (size_t) q->shape[2],
                                     sliding_window,
                                     qp,
                                     kp,
                                     vp,
                                     op)) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED, "attention: F16 KV head_dim too large");
        return GEIST_E_UNSUPPORTED;
    }
    return GEIST_OK;
}

enum geist_status cpu_scalar_kv_append_f16(struct geist_backend      *be,
                                           const struct geist_tensor *k_src,
                                           const struct geist_tensor *v_src,
                                           size_t                     q_position,
                                           struct geist_tensor       *k_cache,
                                           struct geist_tensor       *v_cache) {
    size_t       nks = 0, nvs = 0, nkc = 0, nvc = 0;
    const float *ks = get_f32_dense_ptr_full(k_src, &nks);
    const float *vs = get_f32_dense_ptr_full(v_src, &nvs);
    uint16_t    *kc = (uint16_t *) get_f16_dense_3d(k_cache, &nkc);
    uint16_t    *vc = (uint16_t *) get_f16_dense_3d(v_cache, &nvc);
    /* The source rows land at row q_position of the caches, whose views
     * end with them: the subtraction cannot wrap once both hold. */
    if (ks == nullptr || vs == nullptr || kc == nullptr || vc == nullptr || nks != nvs ||
        nkc != nvc || nks > nkc || k_src->shape[1] != k_cache->shape[1] ||
        k_src->shape[2] != k_cache->shape[2] ||
        q_position * (size_t) k_cache->shape[1] * (size_t) k_cache->shape[2] != nkc - nks) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "kv_append_f16: bad views");
        return GEIST_E_INVALID_ARG;
    }
    kv_rows_to_f16(nks, ks, kc + (nkc - nks));
    kv_rows_to_f16(nvs, vs, vc + (nvc - nvs));
    return GEIST_OK;
}

[[nodiscard]] enum geist_status cpu_scalar_attention(struct geist_backend      *be,
                                                     const struct geist_tensor *q,
                                                     const struct geist_tensor *k,
                                                     const struct geist_tensor *v,
                                                     size_t                     q_offset,
                                                     size_t                     sliding_window,
                                                     struct geist_tensor       *out) {
    if (be == nullptr || q == nullptr || k == nullptr || v == nullptr || out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    if (k->dtype == GEIST_DTYPE_F16) {
        return cpu_scalar_attention_f16(be, q, k, v, q_offset, sliding_window, out);
    }
    size_t       nq, nk, nv, no;
    const float *qp = get_f32_dense_ptr_full(q, &nq);
    const float *kp = get_f32_dense_ptr_full(k, &nk);
    const float *vp = get_f32_dense_ptr_full(v, &nv);
    float       *op = get_f32_dense_ptr_full(out, &no);
    if (qp == nullptr || kp == nullptr || vp == nullptr || op == nullptr || q->ndim != 3 ||
        k->ndim != 3 || v->ndim != 3) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_scalar attention: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
    size_t n_q        = (size_t) q->shape[0];
    size_t n_q_heads  = (size_t) q->shape[1];
    size_t head_dim   = (size_t) q->shape[2];
    size_t n_kv       = (size_t) k->shape[0];
    size_t n_kv_heads = (size_t) k->shape[1];
    attention_mqa_causal_kv(
            n_q, n_kv, q_offset, n_q_heads, n_kv_heads, head_dim, sliding_window, qp, kp, vp, op);
    return GEIST_OK;
}

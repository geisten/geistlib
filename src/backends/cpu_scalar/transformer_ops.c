/*
 * src/backends/cpu_scalar/transformer_ops.c — rope_apply, embedding_lookup,
 * attention via the gemma4_kernels.c reference kernels.
 *
 * Layer: BACKEND.
 *
 * The underlying kernels in gemma4_kernels.c are pure-C reference code
 * (no SIMD intrinsics for rope/attention). cpu_scalar can use them
 * directly without breaking its "scalar reference" contract — the
 * implementation is the algorithm reference. cpu_neon currently uses
 * the same kernels; a future NEON specialization can override only the
 * cpu_neon side.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "internal.h"

#include "gemma4_kernels.h"

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
    rope_apply((size_t) x->shape[0], (size_t) x->shape[1], (size_t) x->shape[2], xp, cosp, sinp);
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

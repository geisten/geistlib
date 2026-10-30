/*
 * src/backends/cpu_scalar/transformer_ops.c — rope_apply,
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
#include "host_buffer.h"
#include "host_ops.h"

#include <geist.h>
#include <geist_backend.h>

#include <stdint.h>
#include <string.h>

[[nodiscard]] enum geist_status cpu_scalar_rope_apply(struct geist_backend      *be,
                                                      struct geist_tensor       *x,
                                                      const struct geist_tensor *cos,
                                                      const struct geist_tensor *sin) {
    if (be == nullptr || x == nullptr || cos == nullptr || sin == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    size_t       nx = 0, nc = 0, ns = 0;
    float       *xp   = geist_host_tensor_f32(x, &nx);
    const float *cosp = geist_host_tensor_f32(cos, &nc);
    const float *sinp = geist_host_tensor_f32(sin, &ns);
    if (xp == nullptr || cosp == nullptr || sinp == nullptr || x->ndim != 3) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "cpu_scalar rope_apply: bad inputs");
        return GEIST_E_INVALID_ARG;
    }
    /* The cos/sin row width is the rotated width (see rope_apply). */
    const size_t seq = (size_t) x->shape[0], head_dim = (size_t) x->shape[2];
    const size_t n_rot = (size_t) cos->shape[cos->ndim - 1];
    if (!rope_apply_shape_ok(seq, head_dim, n_rot, nc, ns)) {
        geist_backend_set_error(
                be, GEIST_E_INVALID_ARG, "cpu_scalar rope_apply: cos/sin do not cover x");
        return GEIST_E_INVALID_ARG;
    }
    rope_apply(seq, (size_t) x->shape[1], head_dim, n_rot, xp, cosp, sinp);
    return GEIST_OK;
}

[[nodiscard]] enum geist_status cpu_scalar_attention(struct geist_backend      *be,
                                                     const struct geist_tensor *q,
                                                     const struct geist_tensor *k,
                                                     const struct geist_tensor *v,
                                                     size_t                     q_offset,
                                                     size_t                     sliding_window,
                                                     struct geist_tensor       *out) {
    struct geist_host_attn  a;
    const enum geist_status s = geist_host_attn_views(be, q, k, v, out, &a);
    if (s != GEIST_OK) {
        return s;
    }
    if (!a.f16) {
        attention_mqa_causal_kv(a.n_q,
                                a.n_kv,
                                q_offset,
                                a.n_q_heads,
                                a.n_kv_heads,
                                a.head_dim,
                                sliding_window,
                                a.q,
                                a.k,
                                a.v,
                                a.out);
    } else if (!attention_mqa_causal_kv_f16(a.n_q,
                                            a.n_kv,
                                            q_offset,
                                            a.n_q_heads,
                                            a.n_kv_heads,
                                            a.head_dim,
                                            sliding_window,
                                            a.q,
                                            a.k,
                                            a.v,
                                            a.out)) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED, "attention: F16 KV head_dim too large");
        return GEIST_E_UNSUPPORTED;
    }
    return GEIST_OK;
}

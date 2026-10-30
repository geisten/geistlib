/*
 * src/backends/common/host_ops.c — view checks and shared ops of the CPU
 * backends (see host_ops.h).
 *
 * Layer: BACKEND.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "host_ops.h"

#include "checked.h"
#include "gemma4_kernels.h"
#include "hadamard.h"

/* Reports a refused view under the backend's name and returns the code. */
static enum geist_status
refuse(struct geist_backend *be, enum geist_status code, const char *op, const char *what) {
    geist_backend_set_error(be, code, "%s %s: %s", geist_host_backend_name(be), op, what);
    return code;
}

enum geist_status geist_host_ew_views(struct geist_backend      *be,
                                      const char                *op,
                                      const unsigned             need,
                                      const struct geist_tensor *x,
                                      const struct geist_tensor *z,
                                      const struct geist_tensor *w,
                                      struct geist_tensor       *y,
                                      struct geist_host_ew      *v) {
    size_t nx = 0, nz = 0, nw = 0, ny = 0;
    *v = (struct geist_host_ew) {
            .x = geist_host_tensor_f32(x, &nx),
            .z = (need & GEIST_EW_Z) != 0 ? geist_host_tensor_f32(z, &nz) : nullptr,
            .w = (need & GEIST_EW_W) != 0 ? geist_host_tensor_f32(w, &nw) : nullptr,
            .y = geist_host_tensor_f32(y, &ny),
            .n = nx,
    };
    if (v->x == nullptr || v->y == nullptr || nx != ny ||
        ((need & GEIST_EW_Z) != 0 && (v->z == nullptr || nz != nx)) ||
        ((need & GEIST_EW_W) != 0 && v->w == nullptr)) {
        return refuse(be, GEIST_E_INVALID_ARG, op, "bad inputs");
    }
    if ((need & (GEIST_EW_FEAT | GEIST_EW_W)) != 0) {
        /* ndim >= 1 and every dimension >= 1, or the view had been refused */
        v->feat = (size_t) x->shape[x->ndim - 1];
        if (nx % v->feat != 0 || ((need & GEIST_EW_W) != 0 && nw != v->feat)) {
            return refuse(be, GEIST_E_INVALID_ARG, op, "bad inputs");
        }
    }
    return GEIST_OK;
}

enum geist_status geist_host_attn_views(struct geist_backend      *be,
                                        const struct geist_tensor *q,
                                        const struct geist_tensor *k,
                                        const struct geist_tensor *v,
                                        struct geist_tensor       *out,
                                        struct geist_host_attn    *a) {
    size_t     nq = 0, nk = 0, nv = 0, no = 0;
    const bool f16 = k != nullptr && k->dtype == GEIST_DTYPE_F16;
    *a             = (struct geist_host_attn) {
            .q   = geist_host_tensor(q, GEIST_DTYPE_F32, sizeof(float), 3, &nq),
            .k   = f16 ? (const void *) geist_host_tensor_f16_3d(k, &nk)
                       : geist_host_tensor(k, GEIST_DTYPE_F32, sizeof(float), 3, &nk),
            .v   = f16 ? (const void *) geist_host_tensor_f16_3d(v, &nv)
                       : geist_host_tensor(v, GEIST_DTYPE_F32, sizeof(float), 3, &nv),
            .out = geist_host_tensor(out, GEIST_DTYPE_F32, sizeof(float), 3, &no),
            .f16 = f16,
    };
    if (a->q == nullptr || a->k == nullptr || a->v == nullptr || a->out == nullptr) {
        return refuse(be, GEIST_E_INVALID_ARG, "attention", "bad views");
    }
    /* Every dimension is at least 1: the views refuse a non-positive one. */
    a->n_q        = (size_t) q->shape[0];
    a->n_q_heads  = (size_t) q->shape[1];
    a->head_dim   = (size_t) q->shape[2];
    a->n_kv       = (size_t) k->shape[0];
    a->n_kv_heads = (size_t) k->shape[1];
    if ((size_t) k->shape[2] != a->head_dim || (size_t) v->shape[0] != a->n_kv ||
        (size_t) v->shape[1] != a->n_kv_heads || (size_t) v->shape[2] != a->head_dim ||
        (size_t) out->shape[0] != a->n_q || (size_t) out->shape[1] != a->n_q_heads ||
        (size_t) out->shape[2] != a->head_dim || a->n_q_heads % a->n_kv_heads != 0) {
        return refuse(be, GEIST_E_INVALID_ARG, "attention", "shape mismatch");
    }
    return GEIST_OK;
}

enum geist_status geist_host_attn_kvq_views(struct geist_backend                 *be,
                                            const char                           *op,
                                            const struct geist_attention_kv_args *args,
                                            const bool                            int4_v,
                                            const enum geist_fused_op             fused_op,
                                            bool (*supported)(const struct geist_fusion_query *q),
                                            struct geist_host_attn_kvq *a) {
    size_t nq = 0, nk = 0, nv = 0, nks = 0, nvs = 0, no = 0;
    *a = (struct geist_host_attn_kvq) {
            .q = geist_host_tensor(args->q, GEIST_DTYPE_F32, sizeof(float), 3, &nq),
            .k = geist_host_tensor(args->k, GEIST_DTYPE_I8, 1, 3, &nk),
            .v = geist_host_tensor(args->v, int4_v ? GEIST_DTYPE_U8 : GEIST_DTYPE_I8, 1, 3, &nv),
            .k_scale = geist_host_tensor(args->k_scale, GEIST_DTYPE_F32, sizeof(float), 2, &nks),
            .v_scale = geist_host_tensor(args->v_scale, GEIST_DTYPE_F32, sizeof(float), 2, &nvs),
            .out     = geist_host_tensor(args->out, GEIST_DTYPE_F32, sizeof(float), 3, &no),
    };
    if (a->q == nullptr || a->k == nullptr || a->v == nullptr || a->k_scale == nullptr ||
        a->v_scale == nullptr || a->out == nullptr) {
        return refuse(be,
                      GEIST_E_INVALID_ARG,
                      op,
                      int4_v ? "q, scales and out must be F32, k I8 and v U8, DENSE views of the "
                               "rank given within their buffers"
                             : "q, scales and out must be F32, k and v I8, DENSE views of the "
                               "rank given within their buffers");
    }
    a->n_q                                = (size_t) args->q->shape[0];
    a->n_q_heads                          = (size_t) args->q->shape[1];
    a->head_dim                           = (size_t) args->q->shape[2];
    a->n_kv                               = (size_t) args->k->shape[0];
    a->n_kv_heads                         = (size_t) args->k->shape[1];
    const struct geist_fusion_query shape = {.op         = fused_op,
                                             .head_dim   = a->head_dim,
                                             .n_q_heads  = a->n_q_heads,
                                             .n_kv_heads = a->n_kv_heads};
    if (!supported(&shape)) {
        geist_backend_set_error(be,
                                GEIST_E_UNSUPPORTED,
                                "%s %s: head_dim %zu, %zu query heads on %zu KV heads",
                                geist_host_backend_name(be),
                                op,
                                a->head_dim,
                                a->n_q_heads,
                                a->n_kv_heads);
        return GEIST_E_UNSUPPORTED;
    }
    /* Matching shapes (half a byte per INT4 V value), and every query's
     * position in the cache: n_q and n_kv are >= 1 (the views refuse an
     * empty dimension), so the subtraction cannot wrap. */
    const size_t v_row = int4_v ? a->head_dim / 2 : a->head_dim;
    if ((size_t) args->k->shape[2] != a->head_dim || (size_t) args->v->shape[2] != v_row ||
        nv != (int4_v ? nk / 2 : nk) || (size_t) args->v->shape[0] != a->n_kv ||
        (size_t) args->v->shape[1] != a->n_kv_heads || no != nq ||
        (size_t) args->out->shape[0] != a->n_q || (size_t) args->out->shape[1] != a->n_q_heads ||
        nks != a->n_kv * a->n_kv_heads || nvs != a->n_kv * a->n_kv_heads ||
        (size_t) args->k_scale->shape[0] != a->n_kv ||
        (size_t) args->v_scale->shape[0] != a->n_kv || a->n_q > a->n_kv ||
        args->q_offset > a->n_kv - a->n_q) {
        geist_backend_set_error(be,
                                GEIST_E_INVALID_ARG,
                                "%s %s: shapes do not match, or %zu queries at %zu run past %zu "
                                "cached positions",
                                geist_host_backend_name(be),
                                op,
                                a->n_q,
                                args->q_offset,
                                a->n_kv);
        return GEIST_E_INVALID_ARG;
    }
    return GEIST_OK;
}

enum geist_status geist_host_kv_append_f16(struct geist_backend      *be,
                                           const struct geist_tensor *k_src,
                                           const struct geist_tensor *v_src,
                                           size_t                     q_position,
                                           struct geist_tensor       *k_cache,
                                           struct geist_tensor       *v_cache) {
    size_t       nks = 0, nvs = 0, nkc = 0, nvc = 0;
    const float *ks = geist_host_tensor_f32(k_src, &nks);
    const float *vs = geist_host_tensor_f32(v_src, &nvs);
    uint16_t    *kc = geist_host_tensor_f16_3d(k_cache, &nkc);
    uint16_t    *vc = geist_host_tensor_f16_3d(v_cache, &nvc);
    /* The source rows land at row q_position of the caches, whose views
     * end with them: the subtraction cannot wrap once both hold. */
    if (ks == nullptr || vs == nullptr || kc == nullptr || vc == nullptr || nks != nvs ||
        nkc != nvc || nks > nkc || k_src->shape[1] != k_cache->shape[1] ||
        k_src->shape[2] != k_cache->shape[2] ||
        q_position * (size_t) k_cache->shape[1] * (size_t) k_cache->shape[2] != nkc - nks) {
        return refuse(be, GEIST_E_INVALID_ARG, "kv_append_f16", "bad views");
    }
    kv_rows_to_f16(nks, ks, kc + (nkc - nks));
    kv_rows_to_f16(nvs, vs, vc + (nvc - nvs));
    return GEIST_OK;
}

enum geist_status geist_host_hadamard_rotate(struct geist_backend             *be,
                                             const struct geist_hadamard_args *args) {
    if (args == nullptr) {
        return refuse(be, GEIST_E_INVALID_ARG, "hadamard_rotate", "null args");
    }
    size_t       nx = 0, ns = 0, ny = 0;
    const float *xp = geist_host_tensor_f32(args->x, &nx);
    const float *sp = args->signs != nullptr ? geist_host_tensor_f32(args->signs, &ns) : nullptr;
    float       *yp = geist_host_tensor_f32(args->y, &ny);
    const enum geist_status s = (args->signs != nullptr && sp == nullptr)
                                        ? GEIST_E_INVALID_ARG
                                        : geist_hadamard_apply(args, nx, ns, ny, xp, sp, yp);
    return s == GEIST_OK ? s : refuse(be, s, "hadamard_rotate", "bad inputs");
}

/*
 * src/archs/transformer/forward/linear.c - resolved-weight linear
 * dispatcher shared by transformer forward stages.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "internal.h"

#include <geist_backend.h>

enum geist_status linear_w_or_legacy(struct geist_backend            *be,
                                     const struct geist_backend_vtbl *v,
                                     struct geist_buffer             *x_buf,
                                     struct geist_buffer             *y_buf,
                                     const struct geist_weight       *w,
                                     size_t                           seq,
                                     const struct geist_tensor       *t_x,
                                     const struct geist_tensor       *t_w,
                                     struct geist_tensor             *t_y) {
    (void) t_x;
    (void) t_w;
    (void) t_y;

    if (w == nullptr) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "linear_w: null geist_weight");
        return GEIST_E_INVALID_ARG;
    }
    if ((seq == 1 && w->linear_m1 == nullptr) || (seq > 1 && w->linear_mN == nullptr)) {
        geist_backend_set_error(be,
                                GEIST_E_UNSUPPORTED,
                                "linear_w: backend resolver installed no kernel "
                                "for dtype=%u, seq=%zu (legacy v->linear() path "
                                "retired in P2.e)",
                                (unsigned) w->dtype,
                                seq);
        return GEIST_E_UNSUPPORTED;
    }
    const float *xp = (const float *) v->buffer_map(x_buf);
    float       *yp = (float *) v->buffer_map(y_buf);
    if (xp == nullptr || yp == nullptr) {
        return GEIST_E_BACKEND;
    }
    /* Pass `be` so the kernel can reach its backend's workspace
     * (cpu_neon q8a scratch, etc.) without consulting file-scope TLS.
     * Engine guarantees `be->state` is valid for the lifetime of this
     * call — see the resolver fail-fast check at cpu_neon_resolve_weight.
     *
     * The kernels are void-returning; on allocation failure they leave `y`
     * unwritten and latch the error into be->err_code via
     * geist_backend_set_error. Clear it first, then surface it so the caller
     * never proceeds on stale output believing it succeeded. */
    be->err_code = GEIST_OK;
    if (seq == 1) {
        w->linear_m1(xp, w, be, yp);
    } else {
        w->linear_mN(xp, w, seq, be, yp);
    }
    v->buffer_unmap(x_buf);
    v->buffer_unmap(y_buf);
    return be->err_code;
}

enum geist_status linear_w_scaled_input_or_legacy(struct geist_backend            *be,
                                                  const struct geist_backend_vtbl *v,
                                                  struct geist_buffer             *x_buf,
                                                  struct geist_buffer             *y_buf,
                                                  const struct geist_weight       *w,
                                                  size_t                           seq,
                                                  size_t                           scale_n,
                                                  const float                     *scale,
                                                  const struct geist_tensor       *t_x,
                                                  const struct geist_tensor       *t_w,
                                                  struct geist_tensor             *t_y) {

    apply_per_channel_inv_scale_inplace(v, x_buf, seq, scale_n, scale);
    return linear_w_or_legacy(be, v, x_buf, y_buf, w, seq, t_x, t_w, t_y);
}

enum geist_status linear_w_pair_or_legacy(struct geist_backend            *be,
                                          const struct geist_backend_vtbl *v,
                                          struct geist_buffer             *x_buf,
                                          struct geist_buffer             *y0_buf,
                                          struct geist_buffer             *y1_buf,
                                          const struct geist_weight       *w0,
                                          const struct geist_weight       *w1,
                                          size_t                           seq,
                                          const struct geist_tensor       *t_x,
                                          const struct geist_tensor       *t_w0,
                                          const struct geist_tensor       *t_w1,
                                          struct geist_tensor             *t_y0,
                                          struct geist_tensor             *t_y1) {
    (void) t_x;
    (void) t_w0;
    (void) t_w1;
    (void) t_y0;
    (void) t_y1;

    if (w0 == nullptr || w1 == nullptr) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "linear_w_pair: null geist_weight");
        return GEIST_E_INVALID_ARG;
    }
    if ((seq == 1 && (w0->linear_m1 == nullptr || w1->linear_m1 == nullptr)) ||
        (seq > 1 && (w0->linear_mN == nullptr || w1->linear_mN == nullptr))) {
        geist_backend_set_error(be,
                                GEIST_E_UNSUPPORTED,
                                "linear_w_pair: resolver installed no paired "
                                "kernel for seq=%zu",
                                seq);
        return GEIST_E_UNSUPPORTED;
    }

    const float *xp  = (const float *) v->buffer_map(x_buf);
    float       *y0p = (float *) v->buffer_map(y0_buf);
    float       *y1p = (float *) v->buffer_map(y1_buf);
    if (xp == nullptr || y0p == nullptr || y1p == nullptr) {
        if (xp != nullptr) {
            v->buffer_unmap(x_buf);
        }
        if (y0p != nullptr) {
            v->buffer_unmap(y0_buf);
        }
        if (y1p != nullptr) {
            v->buffer_unmap(y1_buf);
        }
        return GEIST_E_BACKEND;
    }
    be->err_code = GEIST_OK; /* void kernels latch OOM here; see linear_w_or_legacy */
    if (seq == 1) {
        if (w0->linear_pair_m1 != nullptr && w0->linear_pair_m1 == w1->linear_pair_m1 &&
            w0->n_in == w1->n_in) {
            w0->linear_pair_m1(xp, w0, w1, be, y0p, y1p);
        } else {
            w0->linear_m1(xp, w0, be, y0p);
            w1->linear_m1(xp, w1, be, y1p);
        }
    } else {
        if (w0->linear_pair_mN != nullptr && w0->linear_pair_mN == w1->linear_pair_mN &&
            w0->n_in == w1->n_in) {
            w0->linear_pair_mN(xp, w0, w1, seq, be, y0p, y1p);
        } else {
            w0->linear_mN(xp, w0, seq, be, y0p);
            w1->linear_mN(xp, w1, seq, be, y1p);
        }
    }
    v->buffer_unmap(x_buf);
    v->buffer_unmap(y0_buf);
    v->buffer_unmap(y1_buf);
    return be->err_code;
}

enum geist_status linear_w_triple_or_legacy(struct geist_backend            *be,
                                            const struct geist_backend_vtbl *v,
                                            struct geist_buffer             *x_buf,
                                            struct geist_buffer             *y0_buf,
                                            struct geist_buffer             *y1_buf,
                                            struct geist_buffer             *y2_buf,
                                            const struct geist_weight       *w0,
                                            const struct geist_weight       *w1,
                                            const struct geist_weight       *w2,
                                            size_t                           seq,
                                            const struct geist_tensor       *t_x,
                                            const struct geist_tensor       *t_w0,
                                            const struct geist_tensor       *t_w1,
                                            const struct geist_tensor       *t_w2,
                                            struct geist_tensor             *t_y0,
                                            struct geist_tensor             *t_y1,
                                            struct geist_tensor             *t_y2) {
    (void) t_x;
    (void) t_w0;
    (void) t_w1;
    (void) t_w2;
    (void) t_y0;
    (void) t_y1;
    (void) t_y2;

    if (w0 == nullptr || w1 == nullptr || w2 == nullptr) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "linear_w_triple: null geist_weight");
        return GEIST_E_INVALID_ARG;
    }
    if ((seq == 1 &&
         (w0->linear_m1 == nullptr || w1->linear_m1 == nullptr || w2->linear_m1 == nullptr)) ||
        (seq > 1 &&
         (w0->linear_mN == nullptr || w1->linear_mN == nullptr || w2->linear_mN == nullptr))) {
        geist_backend_set_error(be,
                                GEIST_E_UNSUPPORTED,
                                "linear_w_triple: resolver installed no "
                                "kernel for seq=%zu",
                                seq);
        return GEIST_E_UNSUPPORTED;
    }

    const float *xp  = (const float *) v->buffer_map(x_buf);
    float       *y0p = (float *) v->buffer_map(y0_buf);
    float       *y1p = (float *) v->buffer_map(y1_buf);
    float       *y2p = (float *) v->buffer_map(y2_buf);
    if (xp == nullptr || y0p == nullptr || y1p == nullptr || y2p == nullptr) {
        if (xp != nullptr) {
            v->buffer_unmap(x_buf);
        }
        if (y0p != nullptr) {
            v->buffer_unmap(y0_buf);
        }
        if (y1p != nullptr) {
            v->buffer_unmap(y1_buf);
        }
        if (y2p != nullptr) {
            v->buffer_unmap(y2_buf);
        }
        return GEIST_E_BACKEND;
    }

    be->err_code = GEIST_OK; /* void kernels latch OOM here; see linear_w_or_legacy */
    if (seq == 1) {
        if (w0->linear_pair_m1 != nullptr && w0->linear_pair_m1 == w1->linear_pair_m1 &&
            w0->n_in == w1->n_in) {
            w0->linear_pair_m1(xp, w0, w1, be, y0p, y1p);
        } else {
            w0->linear_m1(xp, w0, be, y0p);
            w1->linear_m1(xp, w1, be, y1p);
        }
        w2->linear_m1(xp, w2, be, y2p);
    } else if (w1->linear_pair_mN != nullptr && w1->linear_pair_mN == w2->linear_pair_mN &&
               w1->n_in == w2->n_in) {
        w0->linear_mN(xp, w0, seq, be, y0p);
        w1->linear_pair_mN(xp, w1, w2, seq, be, y1p, y2p);
    } else {
        w0->linear_mN(xp, w0, seq, be, y0p);
        w1->linear_mN(xp, w1, seq, be, y1p);
        w2->linear_mN(xp, w2, seq, be, y2p);
    }

    v->buffer_unmap(x_buf);
    v->buffer_unmap(y0_buf);
    v->buffer_unmap(y1_buf);
    v->buffer_unmap(y2_buf);
    return be->err_code;
}

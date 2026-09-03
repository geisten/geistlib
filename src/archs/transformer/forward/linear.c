/*
 * src/archs/transformer/forward/linear.c - resolved-weight linear
 * dispatcher shared by transformer forward stages.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "internal.h"

#include <geist_backend.h>

/* ZO-tuning gain (GEIST_TUNE builds only). Scales one linear's output by
 * the model-owned, caller-writable gain that geist_model_gains hands out.
 *
 * Applied to the mapped output buffer AFTER the kernel rather than inside
 * each kernel invocation: the weight's own bytes stay frozen and read-only
 * (that is the whole point — the trits never move), and one helper covers
 * the single / pair / triple dispatchers alike.
 *
 * Not applied on the fused tensor path (fused->linear_t, GPU backends):
 * those write device memory and return before this point. The gains op
 * refuses such a backend outright — see op_gains in arch.c — so this is a
 * closed gap, not a silent one.
 *
 * Without GEIST_TUNE this compiles away entirely and the dispatchers are
 * byte-for-byte what they were. */
#ifdef GEIST_TUNE
static inline void apply_gain(size_t seq, const struct geist_weight *w, float *y) {
    if (w->gain_slot == nullptr) {
        return;
    }
    const float g = *w->gain_slot;
    if (g == 1.0f) {
        return;
    }
    const size_t n = (size_t) w->n_out * seq;
    for (size_t i = 0; i < n; i++) {
        y[i] *= g;
    }
}
#else
#define apply_gain(seq, w, y) ((void) 0)
#endif

enum geist_status linear_w_or_legacy(struct geist_backend            *be,
                                     const struct geist_backend_vtbl *v,
                                     struct geist_buffer             *x_buf,
                                     struct geist_buffer             *y_buf,
                                     const struct geist_weight       *w,
                                     size_t                           seq,
                                     const struct geist_tensor       *t_x,
                                     const struct geist_tensor       *t_w,
                                     struct geist_tensor             *t_y) {
    if (w == nullptr) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "linear_w: null geist_weight");
        return GEIST_E_INVALID_ARG;
    }
    const struct geist_backend_fused *fused = geist_backend_fused_tbl(be);
    /* Batched-submit backends (GPU) take the tensor path so the engine
     * never materializes host pointers here; UNSUPPORTED falls through to
     * the resolved host-pointer kernels below. */
    if (fused->linear_t != nullptr && t_x != nullptr && t_w != nullptr && t_y != nullptr) {
        enum geist_status ts = fused->linear_t(be, t_x, w, t_w, seq, t_y);
        if (ts != GEIST_E_UNSUPPORTED) {
            return ts;
        }
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
     * call — see the resolver fail-fast check at cpu_neon_resolve_weight. */
    if (seq == 1) {
        w->linear_m1(xp, w, be, yp);
    } else {
        w->linear_mN(seq, xp, w, be, yp);
    }
    apply_gain(seq, w, yp);
    v->buffer_unmap(x_buf);
    v->buffer_unmap(y_buf);
    return GEIST_OK;
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
    if (w0 == nullptr || w1 == nullptr) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "linear_w_pair: null geist_weight");
        return GEIST_E_INVALID_ARG;
    }
    const struct geist_backend_fused *fused = geist_backend_fused_tbl(be);
    /* Batched-submit backends: two tensor-path linears, no host pointers. */
    if (fused->linear_t != nullptr && t_x != nullptr && t_w0 != nullptr && t_w1 != nullptr &&
        t_y0 != nullptr && t_y1 != nullptr) {
        enum geist_status ts = fused->linear_t(be, t_x, w0, t_w0, seq, t_y0);
        if (ts == GEIST_OK) {
            ts = fused->linear_t(be, t_x, w1, t_w1, seq, t_y1);
        }
        if (ts != GEIST_E_UNSUPPORTED) {
            return ts;
        }
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
            w0->linear_pair_mN(seq, xp, w0, w1, be, y0p, y1p);
        } else {
            w0->linear_mN(seq, xp, w0, be, y0p);
            w1->linear_mN(seq, xp, w1, be, y1p);
        }
    }
    apply_gain(seq, w0, y0p);
    apply_gain(seq, w1, y1p);
    v->buffer_unmap(x_buf);
    v->buffer_unmap(y0_buf);
    v->buffer_unmap(y1_buf);
    return GEIST_OK;
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
    if (w0 == nullptr || w1 == nullptr || w2 == nullptr) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG, "linear_w_triple: null geist_weight");
        return GEIST_E_INVALID_ARG;
    }
    const struct geist_backend_fused *fused = geist_backend_fused_tbl(be);
    /* Batched-submit backends: three tensor-path linears, no host pointers.
     * w1/w2 (k/v projections) share dtype+shape — a fused pair matvec
     * reads the activations once for both when the backend offers it. */
    if (fused->linear_t != nullptr && t_x != nullptr && t_w0 != nullptr && t_w1 != nullptr &&
        t_w2 != nullptr && t_y0 != nullptr && t_y1 != nullptr && t_y2 != nullptr) {
        enum geist_status ts = fused->linear_t(be, t_x, w0, t_w0, seq, t_y0);
        if (ts == GEIST_OK && fused->linear_t_pair != nullptr &&
            fused->linear_t_pair(be, t_x, w1, t_w1, w2, t_w2, seq, t_y1, t_y2) == GEIST_OK) {
            return GEIST_OK;
        }
        if (ts == GEIST_OK) {
            ts = fused->linear_t(be, t_x, w1, t_w1, seq, t_y1);
        }
        if (ts == GEIST_OK) {
            ts = fused->linear_t(be, t_x, w2, t_w2, seq, t_y2);
        }
        if (ts != GEIST_E_UNSUPPORTED) {
            return ts;
        }
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
        w0->linear_mN(seq, xp, w0, be, y0p);
        w1->linear_pair_mN(seq, xp, w1, w2, be, y1p, y2p);
    } else {
        w0->linear_mN(seq, xp, w0, be, y0p);
        w1->linear_mN(seq, xp, w1, be, y1p);
        w2->linear_mN(seq, xp, w2, be, y2p);
    }
    apply_gain(seq, w0, y0p);
    apply_gain(seq, w1, y1p);
    apply_gain(seq, w2, y2p);

    v->buffer_unmap(x_buf);
    v->buffer_unmap(y0_buf);
    v->buffer_unmap(y1_buf);
    v->buffer_unmap(y2_buf);
    return GEIST_OK;
}

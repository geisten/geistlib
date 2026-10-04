/* Backend-only borrowed row tiles. Never slice an unknown repack layout. */
#pragma once
#include <geist_backend.h>
#include "quant.h"
#include "tensor_view.h"
#include <math.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

[[nodiscard]] static inline enum geist_status
geist_cpu_selected_rows(size_t                     n_tiles,
                        const geist_token_t        ids[static n_tiles],
                        const struct geist_tensor *x,
                        const struct geist_weight *w,
                        const struct geist_tensor *t_w,
                        struct geist_tensor       *y,
                        struct geist_backend      *be) {
    (void) t_w;
    if (w == nullptr || x == nullptr || y == nullptr || be == nullptr || x->buffer == nullptr ||
        y->buffer == nullptr || w->raw == nullptr || w->n_in <= 0 || w->n_out <= 0 || n_tiles == 0)
        return GEIST_E_INVALID_ARG;
    const size_t tile = w->linear_rows_tile;
    size_t       stride, tile_bytes, count, y_bytes;
    if (tile == 0 || w->linear_m1 == nullptr ||
        quant_raw_bytes((enum geist_dtype) w->dtype, (size_t) w->n_in, &stride) ||
        ckd_mul(&tile_bytes, stride, tile) || ckd_mul(&count, n_tiles, tile) ||
        ckd_mul(&y_bytes, count, sizeof(float)))
        return GEIST_E_INVALID_ARG;
    const bool packed = w->backend_layout == GEIST_W_LAYOUT_PQ2_0_X8_GEMV;
    if (packed ? tile != 8 || w->aux_fp32 == nullptr || w->aux_n <= 0
               : w->backend_layout != GEIST_W_LAYOUT_SOURCE ||
                         (w->flags & GEIST_W_AUX_BACKEND_REPACK) != 0)
        return GEIST_E_UNSUPPORTED;
    const struct geist_backend_vtbl *v  = be->desc->vtbl;
    void                            *xh = v->buffer_map(x->buffer), *yh = v->buffer_map(y->buffer);
    if (xh == nullptr || yh == nullptr) {
        if (xh != nullptr)
            v->buffer_unmap(x->buffer);
        if (yh != nullptr)
            v->buffer_unmap(y->buffer);
        return GEIST_E_BACKEND;
    }
    size_t       nx = 0, ny = 0;
    const float *xp = geist_tensor_f32_dense(x, xh, x->buffer->bytes, &nx);
    float       *yp = geist_tensor_f32_dense(y, yh, y->buffer->bytes, &ny);
    if (xp == nullptr || yp == nullptr || nx != (size_t) w->n_in || ny != count) {
        v->buffer_unmap(x->buffer);
        v->buffer_unmap(y->buffer);
        return GEIST_E_INVALID_ARG;
    }
    for (size_t i = 0; i < count; i++)
        yp[i] = NAN;
    /* A native tile is at most eight CPU rows. Keep the same row kernel,
     * but avoid launching an OpenMP team for every tiny tile. The ICV is
     * private to the calling task; restore it on success and failure. */
#ifdef _OPENMP
    const int prior_threads = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    enum geist_status status = GEIST_OK;
    for (size_t i = 0; i < n_tiles; i++) {
        size_t off;
        if (ids[i] < 0 || (size_t) ids[i] % tile != 0 || (size_t) ids[i] >= (size_t) w->n_out ||
            tile > (size_t) w->n_out - (size_t) ids[i] || ckd_mul(&off, (size_t) ids[i], stride) ||
            off > w->raw_nbytes || tile_bytes > w->raw_nbytes - off ||
            (packed && (off > (size_t) w->aux_n || tile_bytes > (size_t) w->aux_n - off))) {
            status = GEIST_E_INVALID_ARG;
            break;
        }
        struct geist_weight row = *w;
        row.raw                 = (const uint8_t *) w->raw + off;
        row.raw_nbytes          = tile_bytes;
        row.n_out               = (int32_t) tile;
        if (packed) {
            row.aux_fp32 = (const float *) ((const uint8_t *) w->aux_fp32 + off);
            row.aux_n    = (int32_t) tile_bytes;
        }
        row.linear_m1(xp, &row, be, yp + i * tile);
    }
#ifdef _OPENMP
    omp_set_num_threads(prior_threads);
#endif
    v->buffer_unmap(x->buffer);
    v->buffer_unmap(y->buffer);
    return status;
}

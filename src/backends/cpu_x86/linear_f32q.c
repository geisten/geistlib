/*
 * src/backends/cpu_x86/linear_f32q.c — F32 dense → W8A8 quantized linear.
 *
 * Layer: BACKEND (cpu_x86).
 *
 * Gemma 4's per-layer PLE projections (inp_gate 1536→256, proj 256→1536)
 * are stored F32, and skinny cblas sgemm is slow on them. They are quantized
 * to W8A8 (per-16-block asymmetric int8) at load and run on the VPDPBUSD
 * GEMM instead.
 *
 * W8A8 contract (kernel_w8a8.h): y = sum over 256-element groups g of
 * act_scale[g] * sum_{b in g} ( w_scale[b]*d_b - w_offset[b]*sum_a[b] ),
 * u_w in [0,255]. For an fp32 block with [min,max]:
 *   scale  = (max-min)/255,  u_w = round((w-min)/scale),
 *   w_scale = scale,  w_offset = -min
 * gives w[i] = scale*u_w[i] + min, exactly reconstructing the dot. Accuracy
 * is validated in test_f32q_unit.c (cosine vs the F32 reference).
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "linear_f32q.h"

#include "backend_state.h"
#include "checked.h"
#include "host_ops.h"
#include "kernel_w8a8.h"
#include "linear_ref.h"

#include "heap.h"

#include <geist_backend.h>

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static inline size_t f32q_blocks_per_row(size_t n_in) {
    return n_in / W8A8_BLOCK_ELEMS;
}

/* blob: [weights u8 : n_out*n_in] [w_scales f32 : n_out*nblk] [w_offsets f32],
 * row-major or interleaved (the same sizes: the interleave is a pure
 * permutation). */
static void f32q_pointers(const uint8_t  *blob,
                          size_t          n_in,
                          size_t          n_out,
                          const uint8_t **w_out,
                          const float   **s_out,
                          const float   **o_out) {
    const size_t nblk = f32q_blocks_per_row(n_in);
    *w_out            = blob;
    *s_out            = (const float *) (blob + n_out * n_in);
    *o_out            = *s_out + n_out * nblk;
}

/* The blob's size; false if it overflows size_t (n_in, n_out come from the
 * file). Every other product over n_in, n_out is smaller. */
[[nodiscard]] static bool f32q_blob_bytes(size_t n_in, size_t n_out, size_t *bytes) {
    size_t w_bytes = 0, n_sc = 0, sc_bytes = 0;
    return !ckd_mul(&w_bytes, n_out, n_in) && !ckd_mul(&n_sc, n_out, f32q_blocks_per_row(n_in)) &&
           !ckd_mul(&sc_bytes, n_sc, 2 * sizeof(float)) && !ckd_add(bytes, w_bytes, sc_bytes);
}

/* When n_out % 8 == 0 AND the host has AVX-512+VNNI, the blob holds the
 * lane-parallel W8x8 / W8x16 interleave instead of the row-major layout,
 * and decode and prefill both run w8x8_gemm / w8x16_gemm (8 or 16 output
 * rows per VPDPBUSD, no per-row hsum; decode as a chunk of one token). The
 * weight keeps the one layout both read, and a row takes the same kernel
 * alone and in a chunk (#476). w8x8_gemm/w8x16_gemm live in the
 * -mavx512vnni TU, so the VNNI check is a SIGILL guard, not a tuning
 * choice (same predicate at resolve, m1 and mN). */
static inline bool f32q_use_w8x8(size_t n_out) {
    return (n_out % W8X8_NROWS) == 0 && w8a8_isa_is_vnni();
}

void f32_to_w8a8_row(
        size_t n_in, const float *row, uint8_t *u_w, float *w_scales, float *w_offsets) {
    const size_t nblk = f32q_blocks_per_row(n_in);
    for (size_t b = 0; b < nblk; b++) {
        const float *wb = row + b * W8A8_BLOCK_ELEMS;
        float        mn = wb[0], mx = wb[0];
        for (size_t i = 1; i < W8A8_BLOCK_ELEMS; i++) {
            mn = wb[i] < mn ? wb[i] : mn;
            mx = wb[i] > mx ? wb[i] : mx;
        }
        const float scale = (mx > mn) ? (mx - mn) / 255.0f : 0.0f;
        const float inv   = (scale > 0.0f) ? 1.0f / scale : 0.0f;
        uint8_t    *ub    = u_w + b * W8A8_BLOCK_ELEMS;
        for (size_t i = 0; i < W8A8_BLOCK_ELEMS; i++) {
            int q = (int) lrintf((wb[i] - mn) * inv);
            q     = q < 0 ? 0 : (q > 255 ? 255 : q);
            ub[i] = (uint8_t) q;
        }
        w_scales[b]  = scale;
        w_offsets[b] = -mn;
    }
}

[[nodiscard]] enum geist_status cpu_x86_linear_f32q_resolve(struct geist_weight *w) {
    if (w == nullptr || w->raw == nullptr || w->n_in <= 0 || w->n_out <= 0) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t n_in  = (size_t) w->n_in;
    const size_t n_out = (size_t) w->n_out;
    size_t       bytes = 0;
    if (n_in % W8A8_BLOCK_ELEMS != 0 || !f32q_blob_bytes(n_in, n_out, &bytes)) {
        return GEIST_E_INVALID_ARG; /* caller keeps the cblas F32 path. */
    }
    const bool interleave = f32q_use_w8x8(n_out);

    uint8_t *blob = heap_alloc_aligned(bytes, OPTIMAL_ALIGNMENT);
    if (blob == nullptr) {
        return GEIST_E_OOM;
    }
    /* The interleave is repacked from a row-major staging copy. */
    void *staging = interleave ? heap_alloc_aligned(bytes, OPTIMAL_ALIGNMENT) : blob;
    if (staging == nullptr) {
        staging = blob;
        safe_free(&staging);
        return GEIST_E_OOM;
    }
    const uint8_t *rm = staging;
    const uint8_t *bw;
    const float   *bs, *bo;
    f32q_pointers(rm, n_in, n_out, &bw, &bs, &bo);
    const size_t nblk = f32q_blocks_per_row(n_in);
    const float *src  = (const float *) w->raw;
    for (size_t r = 0; r < n_out; r++) {
        f32_to_w8a8_row(n_in,
                        src + r * n_in,
                        (uint8_t *) bw + r * n_in,
                        (float *) bs + r * nblk,
                        (float *) bo + r * nblk);
    }

    if (interleave) {
        const uint8_t *iq;
        const float   *is, *io;
        f32q_pointers(blob, n_in, n_out, &iq, &is, &io);
        if (n_out % W8X16_NROWS == 0) {
            w8x16_repack(n_out, n_in, bw, bs, bo, (uint8_t *) iq, (float *) is, (float *) io);
        } else {
            w8x8_repack(n_out, n_in, bw, bs, bo, (uint8_t *) iq, (float *) is, (float *) io);
        }
        safe_free(&staging);
    }

    geist_weight_attach_aux(w, blob, bytes, 0, 0, 0);
    w->linear_m1 = cpu_x86_linear_f32q_m1;
    w->linear_mN = cpu_x86_linear_f32q_mN;
    return GEIST_OK;
}

/* m quantized activation rows through the blob's layout. */
static void f32q_gemm(size_t                     m,
                      const int8_t              *acts,
                      const int32_t             *sum_a,
                      const float               *act_scales,
                      const struct geist_weight *w,
                      float                     *y) {
    const size_t   n_in  = (size_t) w->n_in;
    const size_t   n_out = (size_t) w->n_out;
    const size_t   nblk  = f32q_blocks_per_row(n_in);
    const uint8_t *qs;
    const float   *sc, *of;
    f32q_pointers((const uint8_t *) w->aux_fp32, n_in, n_out, &qs, &sc, &of);
    if (!f32q_use_w8x8(n_out)) {
        w8a8_gemm(m, n_out, nblk, qs, sc, of, acts, sum_a, act_scales, y);
    } else if (n_out % W8X16_NROWS == 0) {
        w8x16_gemm(m, n_out, nblk, qs, sc, of, acts, sum_a, act_scales, y);
    } else {
        w8x8_gemm(m, n_out, nblk, qs, sc, of, acts, sum_a, act_scales, y);
    }
}

void cpu_x86_linear_f32q_m1(const float               *x,
                            const struct geist_weight *w,
                            struct geist_backend      *be,
                            float                     *y) {
    const size_t n_in     = (size_t) w->n_in;
    const size_t nblk     = n_in / W8A8_BLOCK_ELEMS;
    const size_t n_groups = w8a8_act_groups(nblk);

    /* One row of this thread's M>1 scratch (it has the scales slot). */
    struct cpu_x86_workspace *ws = nullptr;
    if (be != nullptr && be->state != nullptr) {
        ws = cpu_x86_ws_acquire_mN((struct cpu_x86_state *) be->state,
                                   n_in,
                                   nblk * sizeof(int32_t),
                                   n_groups * sizeof(float),
                                   0);
    }
    if (ws == nullptr) {
        geist_linear_ref(1, x, w, y); /* no scratch: the reference needs none */
        return;
    }
    w8a8_quantize_acts_row(n_in, x, ws->mN_acts, ws->mN_sum_a, ws->mN_scale);
    f32q_gemm(1, ws->mN_acts, ws->mN_sum_a, ws->mN_scale, w, y);
}

void cpu_x86_linear_f32q_mN(size_t                     m,
                            const float               *x,
                            const struct geist_weight *w,
                            struct geist_backend      *be,
                            float                     *y) {
    const size_t n_in = (size_t) w->n_in;
    const size_t nblk = n_in / W8A8_BLOCK_ELEMS;

    /* Per-thread workspace: int8 acts, 16-element sums, 256-element scales. */
    const size_t              n_groups   = w8a8_act_groups(nblk);
    size_t                    acts_bytes = 0, sum_elems = 0, sum_bytes = 0;
    size_t                    scale_elems = 0, scale_bytes = 0;
    struct cpu_x86_workspace *ws = nullptr;
    if (be != nullptr && be->state != nullptr && !ckd_mul(&acts_bytes, m, n_in) &&
        !ckd_mul(&sum_elems, m, nblk) && !ckd_mul(&sum_bytes, sum_elems, sizeof(int32_t)) &&
        !ckd_mul(&scale_elems, m, n_groups) && !ckd_mul(&scale_bytes, scale_elems, sizeof(float))) {
        ws = cpu_x86_ws_acquire_mN(
                (struct cpu_x86_state *) be->state, acts_bytes, sum_bytes, scale_bytes, 0);
    }
    if (ws == nullptr) {
        /* No prefill scratch: the M=1 kernel per row still writes y. */
        for (size_t row = 0; row < m; row++) {
            cpu_x86_linear_f32q_m1(x + row * n_in, w, be, y + row * (size_t) w->n_out);
        }
        return;
    }
    int8_t  *acts       = ws->mN_acts;
    int32_t *sum_a      = ws->mN_sum_a;
    float   *act_scales = ws->mN_scale;
    for (size_t j = 0; j < m; j++) {
        w8a8_quantize_acts_row(
                n_in, x + j * n_in, acts + j * n_in, sum_a + j * nblk, act_scales + j * n_groups);
    }
    f32q_gemm(m, acts, sum_a, act_scales, w, y);
}

/*
 * src/backends/cpu_x86/linear_f32q.c — F32 dense → W8A8 quantized linear.
 *
 * Layer: BACKEND (cpu_x86).
 *
 * Gemma 4's per-layer PLE projections (inp_gate 1536→256, proj 256→1536)
 * are stored F32 and were the dominant prefill gap: skinny cblas sgemm at
 * ~73 GFLOP/s (docs/LINUX_X86_PERF_PROFILE.md). Quantize them to W8A8
 * (per-16-block asymmetric int8) at load and run geist's VPDPBUSD GEMM
 * (~2600 GFLOP/s, OMP-parallel) instead.
 *
 * W8A8 decode contract: y = scale_x * sum_b ( w_scale[b]*u_w[b] -
 * w_offset[b]*sum_a[b] ), u_w in [0,255]. For an fp32 block with [min,max]:
 *   scale  = (max-min)/255,  u_w = round((w-min)/scale),
 *   w_scale = scale,  w_offset = -min
 * gives w[i] = scale*u_w[i] + min, exactly reconstructing the dot. Accuracy
 * is validated in test_f32q_unit.c (cosine vs the F32 reference).
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "linear_f32q.h"

#include "backend_state.h"
#include "kernel_w4a8.h" /* w4a8_quantize_acts_row */
#include "kernel_w8a8.h"

#include "heap.h"

#include <geist_backend.h>

#include <math.h>
#include <stddef.h>
#include <stdint.h>

static inline size_t f32q_blocks_per_row(size_t n_in) {
    return n_in / W8A8_BLOCK_ELEMS;
}

/* blob: [weights u8 : n_out*n_in] [w_scales f32 : n_out*nblk] [w_offsets f32]. */
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

static size_t f32q_rowmajor_bytes(size_t n_in, size_t n_out) {
    return n_out * n_in + 2 * n_out * f32q_blocks_per_row(n_in) * sizeof(float);
}

/* When n_out % 8 == 0 AND the host has AVX-512+VNNI, append a lane-parallel
 * W8x8 copy (same size, a pure permutation) and run w8x8_gemm at prefill —
 * 8 output rows per VPDPBUSD, reduced once per tile, vs w8a8_gemm's per-row
 * hsum. w8x8_gemm/w8x16_gemm live in the -mavx512vnni TU, so the VNNI check
 * is a SIGILL guard, not a tuning choice (same predicate at resolve and mN). */
static inline bool f32q_use_w8x8(size_t n_out) {
    return (n_out % W8X8_NROWS) == 0 && w8a8_isa_is_vnni();
}

static size_t f32q_blob_bytes(size_t n_in, size_t n_out) {
    const size_t rm = f32q_rowmajor_bytes(n_in, n_out);
    return f32q_use_w8x8(n_out) ? 2 * rm : rm;
}

/* Interleaved W8x8 region, appended after the row-major blob. */
static void f32q_w8x8_pointers(const uint8_t  *blob,
                               size_t          n_in,
                               size_t          n_out,
                               const uint8_t **qs,
                               const float   **s,
                               const float   **o) {
    const size_t   nblk = f32q_blocks_per_row(n_in);
    const uint8_t *base = blob + f32q_rowmajor_bytes(n_in, n_out);
    *qs                 = base;
    *s                  = (const float *) (base + n_out * n_in);
    *o                  = *s + n_out * nblk;
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
    if (n_in % W8A8_BLOCK_ELEMS != 0) {
        return GEIST_E_INVALID_ARG; /* caller keeps the cblas F32 path. */
    }

    uint8_t *blob = heap_alloc_aligned(f32q_blob_bytes(n_in, n_out), OPTIMAL_ALIGNMENT);
    if (blob == nullptr) {
        return GEIST_E_OOM;
    }
    const uint8_t *bw;
    const float   *bs, *bo;
    f32q_pointers(blob, n_in, n_out, &bw, &bs, &bo);
    const size_t nblk = f32q_blocks_per_row(n_in);
    const float *src  = (const float *) w->raw;
    for (size_t r = 0; r < n_out; r++) {
        f32_to_w8a8_row(n_in,
                        src + r * n_in,
                        (uint8_t *) bw + r * n_in,
                        (float *) bs + r * nblk,
                        (float *) bo + r * nblk);
    }

    /* Interleave for the lane-parallel prefill kernel (n_out % 8 == 0). */
    if (f32q_use_w8x8(n_out)) {
        const uint8_t *iq;
        const float   *is, *io;
        f32q_w8x8_pointers(blob, n_in, n_out, &iq, &is, &io);
        if (n_out % W8X16_NROWS == 0) {
            w8x16_repack(n_out, n_in, bw, bs, bo, (uint8_t *) iq, (float *) is, (float *) io);
        } else {
            w8x8_repack(n_out, n_in, bw, bs, bo, (uint8_t *) iq, (float *) is, (float *) io);
        }
    }

    w->aux_fp32 = (const float *) blob;
    w->aux_n    = (int32_t) f32q_blob_bytes(n_in, n_out);
    w->flags |= GEIST_W_AUX_HEAP_OWNED | GEIST_W_AUX_BACKEND_REPACK;
    w->linear_m1 = cpu_x86_linear_f32q_m1;
    w->linear_mN = cpu_x86_linear_f32q_mN;
    return GEIST_OK;
}

/* int8-quantize one activation row → st->acts_scratch + 16-block sum_a. */
static float
quant_act_row(size_t n_in, struct cpu_x86_state *st, const float *x, int8_t *acts, int32_t *sum_a) {
    const float  scale_x = w4a8_quantize_acts_row(n_in, x, acts, st->sum_a_scratch);
    const size_t nblk    = n_in / W8A8_BLOCK_ELEMS;
    for (size_t b = 0; b < nblk; b++) {
        int32_t s = 0;
        for (size_t i = 0; i < W8A8_BLOCK_ELEMS; i++)
            s += (int32_t) acts[b * W8A8_BLOCK_ELEMS + i];
        sum_a[b] = s;
    }
    return scale_x;
}

void cpu_x86_linear_f32q_m1(const float               *x,
                            const struct geist_weight *w,
                            struct geist_backend      *be,
                            float                     *y) {
    struct cpu_x86_state *st    = (struct cpu_x86_state *) be->state;
    const size_t          n_in  = (size_t) w->n_in;
    const size_t          n_out = (size_t) w->n_out;
    const uint8_t        *weights;
    const float          *w_scales, *w_offsets;
    f32q_pointers((const uint8_t *) w->aux_fp32, n_in, n_out, &weights, &w_scales, &w_offsets);

    const float scale_x = quant_act_row(n_in, st, x, st->acts_scratch, st->sum_a_scratch);
    w8a8_gemv(n_out,
              n_in / W8A8_BLOCK_ELEMS,
              weights,
              w_scales,
              w_offsets,
              st->acts_scratch,
              st->sum_a_scratch,
              scale_x,
              y);
}

void cpu_x86_linear_f32q_mN(const float               *x,
                            const struct geist_weight *w,
                            size_t                     m,
                            struct geist_backend      *be,
                            float                     *y) {
    (void) be;
    const size_t   n_in  = (size_t) w->n_in;
    const size_t   n_out = (size_t) w->n_out;
    const size_t   nblk  = n_in / W8A8_BLOCK_ELEMS;
    const uint8_t *weights;
    const float   *w_scales, *w_offsets;
    f32q_pointers((const uint8_t *) w->aux_fp32, n_in, n_out, &weights, &w_scales, &w_offsets);

    int8_t  *acts    = heap_alloc_aligned(m * n_in * sizeof(int8_t), OPTIMAL_ALIGNMENT);
    int32_t *sum_a   = heap_alloc_aligned(m * nblk * sizeof(int32_t), OPTIMAL_ALIGNMENT);
    float   *scale_x = heap_alloc_aligned(m * sizeof(float), OPTIMAL_ALIGNMENT);
    int32_t *tmp     = heap_alloc_aligned((n_in / 32 + 1) * sizeof(int32_t), OPTIMAL_ALIGNMENT);
    if (acts == nullptr || sum_a == nullptr || scale_x == nullptr || tmp == nullptr) {
        safe_free((void **) &acts);
        safe_free((void **) &sum_a);
        safe_free((void **) &scale_x);
        safe_free((void **) &tmp);
        return;
    }
    for (size_t j = 0; j < m; j++) {
        scale_x[j] = w4a8_quantize_acts_row(n_in, x + j * n_in, acts + j * n_in, tmp);
        int8_t  *a = acts + j * n_in;
        int32_t *s = sum_a + j * nblk;
        for (size_t b = 0; b < nblk; b++) {
            int32_t v = 0;
            for (size_t i = 0; i < W8A8_BLOCK_ELEMS; i++)
                v += (int32_t) a[b * W8A8_BLOCK_ELEMS + i];
            s[b] = v;
        }
    }
    if (f32q_use_w8x8(n_out)) {
        const uint8_t *iq;
        const float   *is, *io;
        f32q_w8x8_pointers((const uint8_t *) w->aux_fp32, n_in, n_out, &iq, &is, &io);
        if (n_out % W8X16_NROWS == 0) {
            w8x16_gemm(m, n_out, nblk, iq, is, io, acts, sum_a, scale_x, y);
        } else {
            w8x8_gemm(m, n_out, nblk, iq, is, io, acts, sum_a, scale_x, y);
        }
    } else {
        w8a8_gemm(m, n_out, nblk, weights, w_scales, w_offsets, acts, sum_a, scale_x, y);
    }
    safe_free((void **) &acts);
    safe_free((void **) &sum_a);
    safe_free((void **) &scale_x);
    safe_free((void **) &tmp);
}

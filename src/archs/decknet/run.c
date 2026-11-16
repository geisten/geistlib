/*
 * src/archs/decknet/run.c — the decknet forward pass (#532).
 *
 * Layer: ARCHITECTURE. One portable implementation, the oracle: it follows
 * the numpy reference (geistbox, software/geistbox/decknet.py: warp,
 * normalise, forward) operation by operation and agrees with it to rounding
 * (tests/test_decknet_unit.c). Built without -ffast-math and without
 * contraction (mk/common.mk): the sampling coordinates of a degenerate
 * homography are Inf or NaN, and the clamp has to see them.
 *
 * Where it does not compute as the reference does, and why (#533): a crop
 * is kept in integers. The reference multiplies the image by its gain and
 * samples F32; here a crop pixel is the bilinear sample of the camera bytes
 * in fixed point (decknet_blend, 1/65536 of a byte value, weights in 1/2^24)
 * and the gain is applied behind the kernels, which are linear. That is
 * within F32 rounding of the reference's pixel, and it makes the sum behind
 * a kernel an integer sum: exact, the same in any order and at any vector
 * width, so the compiler may vectorise it and a camera stream may keep it
 * from one frame to the next while the gain moves.
 *
 * The four operations:
 *   resample    bilinear sampling of a camera image through a homography
 *   mix         1x1 convolution on a pixel pair, ReLU, sum, saturation
 *   kernel_at   a small dense 2D kernel, evaluated only at the crop pixels
 *               a pooling row reads (about 1 pixel in 16 for the shipped
 *               file), which gives the same pooled values as a full map;
 *               for a stream that hands it prefix sums instead of a crop,
 *               from two of them per row of equal weights
 *   pool_*      sparse weighted pooling
 *
 * No allocation: every buffer is part of the handle (load.c).
 */
#define GEIST_INTERNAL_ARCH_LAYER
#include "decknet.h"

#include <math.h>
#include <string.h>

static_assert(__FINITE_MATH_ONLY__ == 0,
              "decknet needs -fno-finite-math-only (mk/common.mk): the resampler clamps NaN and "
              "Inf by comparison");

/* sat(x) = relu(x) - relu(x - 1). NaN gives 0. */
static inline float satf(float x) {
    return x > 0.0f ? (x < 1.0f ? x : 1.0f) : 0.0f;
}
static inline double satd(double x) {
    return x > 0.0 ? (x < 1.0 ? x : 1.0) : 0.0;
}
static inline double relud(double x) {
    return x > 0.0 ? x : 0.0;
}

/* ---- resample --------------------------------------------------------- */

/* out[y][x] for an out_h x out_w crop: the homography h maps the crop pixel
 * (x, y, 1) to the camera pixel (u, v, w); the image is sampled bilinearly
 * at (u / w - 0.5, v / w - 0.5), clamped to the image. */
static void resample(size_t         img_w,
                     size_t         img_h,
                     size_t         out_w,
                     size_t         out_h,
                     const float    h[static 9],
                     const uint8_t *img,
                     uint32_t      *out) {
    for (size_t y = 0; y < out_h; y++) {
        for (size_t x = 0; x < out_w; x++) {
            double   u, v;
            uint32_t w[3];
            decknet_sample_at(img_w, img_h, x, y, h, &u, &v);
            decknet_weights(u, v, w);
            const uint8_t *r0 = img + ((size_t) v * img_w + (size_t) u) * 3;
            const uint8_t *r1 = r0 + img_w * 3;
            for (size_t c = 0; c < 3; c++) {
                out[c] = decknet_blend(w, r0[c], r0[3 + c], r1[c], r1[3 + c]);
            }
            out[3] = 0;
            out += DECKNET_CROP_CH;
        }
    }
}

void decknet_strip(const struct geist_decknet *d, const uint32_t *q, float *out) {
    const double scale = decknet_scale(1.0f);
    for (size_t i = 0; i < d->bg_px; i++) {
        for (size_t c = 0; c < 3; c++) {
            out[i * 3 + c] = decknet_unit(q[i * DECKNET_CROP_CH + c], scale);
        }
    }
}

/* ---- mix -------------------------------------------------------------- */

/* Four F32 side by side: the 1x1 layer computes four of its rows at once,
 * each lane with the operations, in the order, a single row would take. */
typedef float   f4 [[gnu::vector_size(16)]];
typedef int32_t i4 [[gnu::vector_size(16)]];

static inline f4 f4_at(const float *p) {
    f4 v;
    memcpy(&v, p, sizeof v);
    return v;
}

/* sat(g * (sum over the rows of relu(w . [a, b]) - t)) for one pixel pair. */
static float mix(const struct decknet_mix *m,
                 float                     g,
                 float                     t,
                 const float               a[static 3],
                 const float               b[static 3]) {
    float sum = 0.0f;
    for (size_t r = 0; r < m->rows; r += 4) {
        const float *w = m->by_four + r * DECKNET_MIX_IN;
        const f4     v = f4_at(w) * a[0] + f4_at(w + 4) * a[1] + f4_at(w + 8) * a[2] +
                         f4_at(w + 12) * b[0] + f4_at(w + 16) * b[1] + f4_at(w + 20) * b[2];
        /* relu without a branch: the sign of a 1x1 unit is noise to a branch
         * predictor. A lane that is not above 0 (negative, NaN, or behind
         * the last row) becomes +0, and adding +0 to a sum that starts at
         * +0 and only grows leaves it as it is. Then row after row. */
        const i4 above = v > (f4) {0.0f, 0.0f, 0.0f, 0.0f};
        i4       bits;
        f4       relu;
        memcpy(&bits, &v, sizeof bits);
        bits &= above;
        memcpy(&relu, &bits, sizeof relu);
        sum = (((sum + relu[0]) + relu[1]) + relu[2]) + relu[3];
    }
    return satf(g * (sum - t));
}

/* The one unit of a region pool over mix() of the floor-plane crops: the
 * current image's, with the scale of its gain, and the empty deck's. mixv
 * has the 1x1 value of each row. */
static float pool_mix(const struct decknet_pool *p,
                      const struct decknet_mix  *m,
                      float                      g,
                      float                      t,
                      const double               scale[static 3],
                      const struct decknet_work *w,
                      float                     *mixv) {
    double acc = 0.0;
    for (size_t r = 0; r < p->n_rows; r++) {
        const size_t px = p->px[r];
        if (!w->kept || (w->stale != nullptr && w->stale[px] != 0)) {
            const uint32_t *q    = w->crop + px * DECKNET_CROP_CH;
            const float     a[3] = {decknet_unit(q[0], scale[0]),
                                    decknet_unit(q[1], scale[1]),
                                    decknet_unit(q[2], scale[2])};
            mixv[r]              = mix(m, g, t, a, w->crop_ref + px * 3);
        }
        acc += (double) (p->w[r] * mixv[r]);
    }
    return (float) acc;
}

/* ---- kernels ---------------------------------------------------------- */

/* A crop pixel as one value: four 32-bit lanes, added lane by lane. The
 * compilers' vector type, so that a pixel is one load and one add on every
 * target instead of what a vectoriser makes of a loop over a few pixels. */
typedef uint32_t q4 [[gnu::vector_size(16)]];
static_assert(sizeof(q4) == DECKNET_CROP_CH * sizeof(uint32_t));

static inline q4 q4_at(const uint32_t *p) {
    q4 v;
    memcpy(&v, p, sizeof v);
    return v;
}

/* Crop pixel px, from the crop or, with `prefix` (decknet_work.prefix),
 * from the two prefix sums around it. */
static inline q4 pixel_at(size_t px, const uint32_t *crop, const uint32_t *prefix) {
    return prefix != nullptr ? q4_at(prefix + (px + 1) * DECKNET_CROP_CH) -
                                       q4_at(prefix + px * DECKNET_CROP_CH)
                             : q4_at(crop + px * DECKNET_CROP_CH);
}

/* out[c]: the convolution of colour c of an h x w crop with kernel k at
 * pixel (x, y), "same" size with the edge pixels repeated, in crop units
 * (the gain is the caller's). A convolution, as the reference's FFT computes
 * it: tap (a, b) reads the pixel at (y + n - a, x + n - b), n = k / 2. The
 * shipped kernels are point-symmetric, for which a correlation is the same.
 *
 * Computed as the sum over the kernel's distinct weights of weight times the
 * sum of the pixels that have it. The pixel sums are integers, in 32 bits
 * within a part (k->part) and in 64 across the parts of a weight.
 *
 * The crop is `crop`, or with `prefix` its prefix sums
 * (decknet_work.prefix). From those a part's sum inside the crop is taken
 * span by span where the kernel has spans, as the difference of two prefix
 * sums: the same integer, from two reads per row of neighbours instead of
 * one per pixel. */
static void kernel_at(size_t                       w,
                      size_t                       h,
                      const struct decknet_kernel *k,
                      const uint32_t              *crop,
                      const uint32_t              *prefix,
                      size_t                       x,
                      size_t                       y,
                      double                       out[static 3]) {
    const size_t               n        = k->k / 2;
    const bool                 interior = y >= n && y + n < h && x >= n && x + n < w;
    const bool                 spans    = interior && prefix != nullptr && k->span != nullptr;
    const bool                 taps     = interior && prefix == nullptr;
    const uint32_t *const      at       = taps ? crop + (y * w + x) * DECKNET_CROP_CH : nullptr;
    const uint32_t *const      sum_at   = spans ? prefix + (y * w + x) * DECKNET_CROP_CH : nullptr;
    const struct decknet_tap  *tap      = k->tap;
    const struct decknet_span *span     = k->span;
    uint64_t                   sum[3]   = {0, 0, 0};
    double                     s[3]     = {0.0, 0.0, 0.0};
    for (size_t i = 0; i < k->n_parts; i++) {
        const struct decknet_part *part = &k->part[i];
        /* Four sums side by side: an integer sum is the same in any order,
         * and one chain of adds would wait for itself. */
        q4 a0 = {0, 0, 0, 0}, a1 = a0, a2 = a0, a3 = a0;
        if (spans) {
            size_t j = 0;
            for (; j + 2 <= part->n_spans; j += 2) {
                a0 += q4_at(sum_at + (ptrdiff_t) span[j].hi * (ptrdiff_t) DECKNET_CROP_CH);
                a1 += q4_at(sum_at + (ptrdiff_t) span[j].lo * (ptrdiff_t) DECKNET_CROP_CH);
                a2 += q4_at(sum_at + (ptrdiff_t) span[j + 1].hi * (ptrdiff_t) DECKNET_CROP_CH);
                a3 += q4_at(sum_at + (ptrdiff_t) span[j + 1].lo * (ptrdiff_t) DECKNET_CROP_CH);
            }
            if (j < part->n_spans) {
                a0 += q4_at(sum_at + (ptrdiff_t) span[j].hi * (ptrdiff_t) DECKNET_CROP_CH);
                a1 += q4_at(sum_at + (ptrdiff_t) span[j].lo * (ptrdiff_t) DECKNET_CROP_CH);
            }
            span += part->n_spans;
            a0 = a0 - a1; /* wraps as the prefix sums do */
            a1 = (q4) {0, 0, 0, 0};
            a2 = a2 - a3;
            a3 = a1;
        } else if (taps) {
            size_t j = 0;
            for (; j + 4 <= part->n_taps; j += 4) {
                a0 += q4_at(at + (ptrdiff_t) tap[j].off * (ptrdiff_t) DECKNET_CROP_CH);
                a1 += q4_at(at + (ptrdiff_t) tap[j + 1].off * (ptrdiff_t) DECKNET_CROP_CH);
                a2 += q4_at(at + (ptrdiff_t) tap[j + 2].off * (ptrdiff_t) DECKNET_CROP_CH);
                a3 += q4_at(at + (ptrdiff_t) tap[j + 3].off * (ptrdiff_t) DECKNET_CROP_CH);
            }
            for (; j < part->n_taps; j++) {
                a0 += q4_at(at + (ptrdiff_t) tap[j].off * (ptrdiff_t) DECKNET_CROP_CH);
            }
        } else {
            /* Row and column clamped to the crop, in signed arithmetic. */
            for (size_t j = 0; j < part->n_taps; j++) {
                const ptrdiff_t yy = (ptrdiff_t) y + tap[j].dy;
                const ptrdiff_t xx = (ptrdiff_t) x + tap[j].dx;
                const size_t    ya = yy < 0 ? 0 : ((size_t) yy < h ? (size_t) yy : h - 1);
                const size_t    xb = xx < 0 ? 0 : ((size_t) xx < w ? (size_t) xx : w - 1);
                a0 += pixel_at(ya * w + xb, crop, prefix);
            }
        }
        tap += part->n_taps;
        const q4 acc = (a0 + a1) + (a2 + a3);
        for (size_t c = 0; c < 3; c++) {
            sum[c] += acc[c];
        }
        if (part->last) {
            for (size_t c = 0; c < 3; c++) {
                s[c] += (double) part->value * (double) sum[c];
                sum[c] = 0;
            }
        }
    }
    out[0] = s[0];
    out[1] = s[1];
    out[2] = s[2];
}

/* Whether a stale cell holds a pixel the kernel reads around (x, y). */
static bool window_stale(size_t w, size_t h, size_t n, const uint8_t *stale, size_t x, size_t y) {
    const size_t cells_w = (w + DECKNET_CELL - 1) / DECKNET_CELL;
    const size_t y0      = (y < n ? 0 : y - n) / DECKNET_CELL;
    const size_t y1      = (y + n < h ? y + n : h - 1) / DECKNET_CELL;
    const size_t x0      = (x < n ? 0 : x - n) / DECKNET_CELL;
    const size_t x1      = (x + n < w ? x + n : w - 1) / DECKNET_CELL;
    for (size_t cy = y0; cy <= y1; cy++) {
        for (size_t cx = x0; cx <= x1; cx++) {
            if (stale[cy * cells_w + cx] != 0) {
                return true;
            }
        }
    }
    return false;
}

/* sums[i x 3 + c] = kernel_at pixel px[i]: for every i, or with w->kept for
 * those whose window holds a stale pixel. */
static void kernel_sums(const struct geist_decknet  *d,
                        const struct decknet_kernel *k,
                        size_t                       n,
                        const uint32_t               px[static n],
                        const struct decknet_work   *w,
                        double                      *sums) {
    if (w->kept && w->stale == nullptr) {
        return;
    }
    const uint32_t crop_w = (uint32_t) d->crop_w;
    for (size_t i = 0; i < n; i++) {
        const size_t y = px[i] / crop_w;
        const size_t x = px[i] % crop_w;
        if (!w->kept || window_stale(d->crop_w, d->crop_h, k->k / 2, w->stale, x, y)) {
            kernel_at(d->crop_w, d->crop_h, k, w->crop, w->prefix, x, y, sums + i * 3);
        }
    }
}

/* Blob energy of the plane's crop at the pixels the unit pools read:
 * e_bright = sum over RGB of relu(d), e_dark = sum of relu(-d), d the
 * convolution of the colour scaled by its gain. */
static void blob_energy(const struct geist_decknet *d,
                        struct decknet_ws          *ws,
                        const double                scale[static 3],
                        const struct decknet_work  *w) {
    kernel_sums(d, &d->blob, d->n_blob_px, d->blob_px, w, w->sums);
    for (size_t i = 0; i < d->n_blob_px; i++) {
        const size_t  px = d->blob_px[i];
        const double *c  = w->sums + i * 3;
        const double  c0 = scale[0] * c[0], c1 = scale[1] * c[1], c2 = scale[2] * c[2];
        ws->e_bright[px] = relud(c0) + relud(c1) + relud(c2);
        ws->e_dark[px]   = relud(-c0) + relud(-c1) + relud(-c2);
    }
}

/* ---- pool ------------------------------------------------------------- */

/* acc[u] = sum over the rows of unit u of w * x[px]. */
static void pool_values(size_t                     n_units,
                        const struct decknet_pool *p,
                        const double              *x,
                        double                     acc[static n_units]) {
    memset(acc, 0, n_units * sizeof acc[0]);
    for (size_t r = 0; r < p->n_rows; r++) {
        acc[p->unit[r]] += (double) p->w[r] * x[p->px[r]];
    }
}

/* out[u] = sat(g_unit * (pool.on(e)[u] - pool.off(e)[u] - t)) */
static void units(const struct geist_decknet *d,
                  struct decknet_ws          *ws,
                  float                       t,
                  const double               *e,
                  double                      out[static d->n_tips]) {
    pool_values(d->n_tips, &d->on, e, ws->acc_on);
    pool_values(d->n_tips, &d->off, e, ws->acc_off);
    for (size_t u = 0; u < d->n_tips; u++) {
        out[u] = satd((double) d->g_unit * (ws->acc_on[u] - ws->acc_off[u] - (double) t));
    }
}

static double mean(size_t n, const double x[static n]) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += x[i];
    }
    return sum / (double) n;
}

/* ---- the pass --------------------------------------------------------- */

/* Step 2: brightness of both images on the bare worktable strip. */
void decknet_gain(const struct geist_decknet *d, const float *bg, float g[static 3]) {
    /* F32 running sums in pixel order, on purpose: that is what the
     * reference's a.mean((0, 1)) on an F32 array computes, and its rounding
     * (about 1e-5 of the mean for the shipped strip) is part of the numbers
     * the thresholds were set on. Behind g_white an exact mean differs
     * from the reference by up to 4e-4 in a unit. */
    float sum[3] = {0.0f, 0.0f, 0.0f};
    for (size_t i = 0; i < d->bg_px; i++) {
        for (size_t c = 0; c < 3; c++) {
            sum[c] += bg[i * 3 + c];
        }
    }
    for (size_t c = 0; c < 3; c++) {
        g[c] = d->bg_nominal[c] / fmaxf(sum[c] / (float) d->bg_px, 1e-3f);
    }
}

void decknet_gains(struct geist_decknet *d,
                   const float          *bg_a,
                   const float          *bg_b,
                   float                 ga[static 3],
                   float                 gb[static 3]) {
    decknet_gain(d, bg_a, ga);
    decknet_gain(d, bg_b, gb);
    float gain_lo = INFINITY, gain_hi = -INFINITY;
    for (size_t c = 0; c < 3; c++) {
        const float gain  = ga[c] / gb[c];
        d->result.gain[c] = gain;
        gain_lo           = fminf(gain_lo, gain);
        gain_hi           = fmaxf(gain_hi, gain);
    }
    double changed = 0.0;
    for (size_t i = 0; i < d->bg_px; i++) {
        float a[3], b[3];
        for (size_t c = 0; c < 3; c++) {
            a[c] = bg_a[i * 3 + c] * ga[c];
            b[c] = bg_b[i * 3 + c] * gb[c];
        }
        changed += (double) mix(&d->change, d->g_change, d->t_change, a, b);
    }
    d->result.bg_changed = (float) (changed / (double) d->bg_px);
    /* Written so that a NaN on either side is "not checkable". */
    d->result.checkable =
            d->gain_min < gain_lo && gain_hi < d->gain_max && d->result.bg_changed < d->bg_rest_max;
    d->have_run = true;
    if (!d->result.checkable) {
        /* No slot state for a pair that cannot be compared: the last
         * checkable pair's must not survive as if it were this one's. */
        for (size_t s = 0; s < d->n_slots; s++) {
            struct geist_decknet_slot *o = &d->slot_out[s];
            o->occ_front = o->occ_rear = o->wall_front = o->rack = 0.0f;
            o->plate = o->deep = o->tube_n = o->tube_r = 0.0f;
        }
        memset(d->tips, 0, d->n_slots * d->n_tips * sizeof d->tips[0]);
        memset(d->tubes, 0, d->n_slots * d->tube.n_units * sizeof d->tubes[0]);
    }
}

/* Marks the k x k window conv_at reads around crop pixel px. */
static void mark_window(size_t w, size_t h, size_t side, size_t px, uint8_t *mark) {
    const size_t n = side / 2;
    const size_t y = px / w;
    const size_t x = px % w;
    for (size_t a = 0; a < side; a++) {
        const size_t ya = y + n < a ? 0 : (y + n - a < h ? y + n - a : h - 1);
        for (size_t b = 0; b < side; b++) {
            const size_t xb   = x + n < b ? 0 : (x + n - b < w ? x + n - b : w - 1);
            mark[ya * w + xb] = 1;
        }
    }
}

static void mark_rows(const struct decknet_pool *p, uint8_t *mark) {
    for (size_t r = 0; r < p->n_rows; r++) {
        mark[p->px[r]] = 1;
    }
}

void decknet_plane_pixels(const struct geist_decknet *d, enum decknet_plane plane, uint8_t *mark) {
    memset(mark, 0, d->crop_px);
    if (plane == DECKNET_PLANE_FLOOR) {
        mark_rows(&d->occ_front, mark);
        mark_rows(&d->occ_rear, mark);
        mark_rows(&d->wall_pool, mark);
        return;
    }
    /* Every other plane has its blob energy taken at blob_px. */
    for (size_t i = 0; i < d->n_blob_px; i++) {
        mark_window(d->crop_w, d->crop_h, d->blob.k, d->blob_px[i], mark);
    }
    if (plane == DECKNET_PLANE_HOLE) {
        for (size_t i = 0; i < d->n_tube_px; i++) {
            mark_window(d->crop_w, d->crop_h, d->tube_kernel.k, d->tube_px[i], mark);
        }
    }
    if (plane == DECKNET_PLANE_TIP) {
        mark_rows(&d->on, mark); /* white: the pixels themselves */
    }
}

void decknet_plane(struct geist_decknet      *d,
                   struct decknet_ws         *ws,
                   size_t                     s,
                   enum decknet_plane         plane,
                   const float                ga[static 3],
                   const struct decknet_work *w) {
    struct geist_decknet_slot *o     = &d->slot_out[s];
    float                     *tips  = d->tips + s * d->n_tips;
    float                     *tubes = d->tubes + s * d->tube.n_units;
    const double scale[3] = {decknet_scale(ga[0]), decknet_scale(ga[1]), decknet_scale(ga[2])};
    switch (plane) {
    case DECKNET_PLANE_FLOOR: {
        /* Step 3: change and wall against the empty deck. */
        float *front = w->mix, *rear = front + d->occ_front.n_rows;
        float *wall = rear + d->occ_rear.n_rows;
        o->occ_front =
                pool_mix(&d->occ_front, &d->change, d->g_change, d->t_change, scale, w, front);
        o->occ_rear = pool_mix(&d->occ_rear, &d->change, d->g_change, d->t_change, scale, w, rear);
        o->wall_front = pool_mix(&d->wall_pool, &d->wall, d->g_wall, d->t_wall, scale, w, wall);
        return;
    }
    case DECKNET_PLANE_PLATE:
        /* Steps 4 and 5, plate and deep-well planes: dark blobs on the raster. */
        blob_energy(d, ws, scale, w);
        units(d, ws, d->t_dark, ws->e_dark, ws->unit);
        o->plate = (float) mean(d->n_tips, ws->unit);
        return;
    case DECKNET_PLANE_DEEP:
        blob_energy(d, ws, scale, w);
        units(d, ws, d->t_dark, ws->e_dark, ws->unit);
        o->deep = (float) mean(d->n_tips, ws->unit);
        return;
    case DECKNET_PLANE_HOLE:
        /* Open holes, and step 6, the tubes. */
        blob_energy(d, ws, scale, w);
        units(d, ws, d->t_dark, ws->e_dark, ws->hole);
        kernel_sums(d, &d->tube_kernel, d->n_tube_px, d->tube_px, w, w->tube);
        for (size_t i = 0; i < d->n_tube_px; i++) {
            const double *c = w->tube + i * 3;
            ws->e_tube[d->tube_px[i]] =
                    relud(scale[0] * c[0]) + relud(scale[1] * c[1]) + relud(scale[2] * c[2]);
        }
        pool_values(d->tube.n_units, &d->tube, ws->e_tube, ws->acc_tube);
        for (size_t t = 0; t < d->tube.n_units; t++) {
            ws->acc_tube[t] = satd((double) d->g_unit * (ws->acc_tube[t] - (double) d->t_tube));
            tubes[t]        = (float) ws->acc_tube[t];
        }
        o->tube_n = (float) mean(d->n_tubes, ws->acc_tube);
        o->tube_r = (float) mean(d->n_tubes, ws->acc_tube + d->n_tubes);
        return;
    case DECKNET_PLANE_TIP: {
        /* Bright blob AND white AND no open hole (of the hole plane before). */
        blob_energy(d, ws, scale, w);
        units(d, ws, d->t_tip, ws->e_bright, ws->unit);
        memset(ws->acc_on, 0, d->n_tips * sizeof ws->acc_on[0]);
        for (size_t r = 0; r < d->on.n_rows; r++) {
            const q4    q  = pixel_at(d->on.px[r], w->crop, w->prefix);
            const float p0 = decknet_unit(q[0], scale[0]), p1 = decknet_unit(q[1], scale[1]);
            const float p2 = decknet_unit(q[2], scale[2]);
            ws->acc_on[d->on.unit[r]] += (double) (d->on.w[r] * fminf(fminf(p0, p1), p2));
        }
        double rack = 0.0;
        for (size_t u = 0; u < d->n_tips; u++) {
            const double white = satd((double) d->g_white * (ws->acc_on[u] - (double) d->t_white));
            const double tip   = relud(ws->unit[u] + white - 1.0 - ws->hole[u]);
            tips[u]            = (float) tip;
            rack += tip + relud(ws->hole[u] - tip); /* max(tip, hole) */
        }
        o->rack = (float) (rack / (double) d->n_tips);
        return;
    }
    }
}

void decknet_run(struct geist_decknet *d,
                 size_t                n_bytes,
                 const uint8_t         cur[static n_bytes],
                 const uint8_t         ref[static n_bytes]) {
    float ga[3], gb[3];
    resample(d->img_w, d->img_h, d->bg_w, d->bg_h, d->h_cur, cur, d->bg_q);
    decknet_strip(d, d->bg_q, d->bg_a);
    resample(d->img_w, d->img_h, d->bg_w, d->bg_h, d->h_ref, ref, d->bg_q);
    decknet_strip(d, d->bg_q, d->bg_b);
    decknet_gains(d, d->bg_a, d->bg_b, ga, gb);
    if (!d->result.checkable) {
        return;
    }
    const double scale_b[3] = {decknet_scale(gb[0]), decknet_scale(gb[1]), decknet_scale(gb[2])};
    const struct decknet_work work = {.crop     = d->ws.crop,
                                      .crop_ref = d->crop_b,
                                      .mix      = d->ws.mix,
                                      .sums     = d->ws.sums,
                                      .tube     = d->ws.tube_sums};
    for (size_t s = 0; s < d->n_slots; s++) {
        /* The floor plane of slot s is homography 1 + DECKNET_PLANES * s. */
        const size_t first = (1 + DECKNET_PLANES * s) * 9;
        resample(d->img_w, d->img_h, d->crop_w, d->crop_h, d->h_ref + first, ref, d->ws.crop);
        for (size_t i = 0; i < d->crop_px; i++) {
            for (size_t c = 0; c < 3; c++) {
                d->crop_b[i * 3 + c] =
                        decknet_unit(d->ws.crop[i * DECKNET_CROP_CH + c], scale_b[c]);
            }
        }
        for (size_t p = 0; p < DECKNET_PLANES; p++) {
            resample(d->img_w,
                     d->img_h,
                     d->crop_w,
                     d->crop_h,
                     d->h_cur + first + p * 9,
                     cur,
                     d->ws.crop);
            decknet_plane(d, &d->ws, s, (enum decknet_plane) p, ga, &work);
        }
    }
}

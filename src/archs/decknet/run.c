/*
 * src/archs/decknet/run.c — the decknet forward pass (#532).
 *
 * Layer: ARCHITECTURE. One portable implementation, the oracle: it follows
 * the numpy reference (geistbox, software/geistbox/decknet.py: warp,
 * normalise, forward) operation by operation, with the reference's
 * precision at each step (F32 pixels, F64 sampling coordinates, F64 sums
 * behind the kernels), so the two agree to rounding. Built without
 * -ffast-math and without contraction (mk/common.mk): the sampling
 * coordinates of a degenerate homography are Inf or NaN, and the clamp
 * below has to see them.
 *
 * The four operations:
 *   resample    bilinear sampling of a camera image through a homography
 *   mix         1x1 convolution on a pixel pair, ReLU, sum, saturation
 *   conv_at     a small dense 2D kernel, evaluated only at the crop pixels
 *               a pooling row reads (about 1 pixel in 16 for the shipped
 *               file), which gives the same pooled values as a full map
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

/* out[y][x][c] for an out_h x out_w crop: the homography h maps the crop
 * pixel (x, y, 1) to the camera pixel (u, v, w); the image, scaled to 0..1
 * and by gain[c], is sampled bilinearly at (u / w - 0.5, v / w - 0.5),
 * clamped to the image. `scale` is b / 255 for each byte value b. */
static void resample(size_t         img_w,
                     size_t         img_h,
                     size_t         out_w,
                     size_t         out_h,
                     const float    gain[static 3],
                     const float    h[static 9],
                     const float    scale[static 256],
                     const uint8_t *img,
                     float         *out) {
    for (size_t y = 0; y < out_h; y++) {
        for (size_t x = 0; x < out_w; x++) {
            double u, v;
            decknet_sample_at(img_w, img_h, x, y, h, &u, &v);
            const size_t   u0 = (size_t) u;
            const size_t   v0 = (size_t) v;
            const double   fu = u - (double) u0;
            const double   fv = v - (double) v0;
            const uint8_t *r0 = img + (v0 * img_w + u0) * 3;
            const uint8_t *r1 = r0 + img_w * 3;
            for (size_t c = 0; c < 3; c++) {
                const double a = (double) (scale[r0[c]] * gain[c]);
                const double b = (double) (scale[r0[3 + c]] * gain[c]);
                const double e = (double) (scale[r1[c]] * gain[c]);
                const double f = (double) (scale[r1[3 + c]] * gain[c]);
                *out++         = (float) ((a * (1.0 - fu) + b * fu) * (1.0 - fv) +
                                          (e * (1.0 - fu) + f * fu) * fv);
            }
        }
    }
}

/* ---- mix -------------------------------------------------------------- */

/* sat(g * (sum over the rows of relu(w . [a, b]) - t)) for one pixel pair. */
static float mix(const struct decknet_mix *m,
                 float                     g,
                 float                     t,
                 const float               a[static 3],
                 const float               b[static 3]) {
    float sum = 0.0f;
    for (size_t r = 0; r < m->rows; r++) {
        const float *w = m->w + r * DECKNET_MIX_IN;
        const float  v =
                w[0] * a[0] + w[1] * a[1] + w[2] * a[2] + w[3] * b[0] + w[4] * b[1] + w[5] * b[2];
        if (v > 0.0f) {
            sum += v;
        }
    }
    return satf(g * (sum - t));
}

/* The one unit of a region pool over mix() of the floor-plane crops a
 * (current) and b (empty deck). */
static float pool_mix(const struct decknet_pool *p,
                      const struct decknet_mix  *m,
                      float                      g,
                      float                      t,
                      const float               *a,
                      const float               *b) {
    double acc = 0.0;
    for (size_t r = 0; r < p->n_rows; r++) {
        const size_t at = (size_t) p->px[r] * 3;
        acc += (double) (p->w[r] * mix(m, g, t, a + at, b + at));
    }
    return (float) acc;
}

/* ---- conv ------------------------------------------------------------- */

/* out[c]: the convolution of channel c of an h x w crop with kernel k at
 * pixel px, "same" size with the edge pixels repeated. A convolution, as
 * the reference's FFT computes it: tap (a, b) reads the pixel at
 * (y + n - a, x + n - b), n = k / 2. The shipped kernels are point-symmetric,
 * for which a correlation is the same. */
static void conv_at(size_t                       w,
                    size_t                       h,
                    const struct decknet_kernel *k,
                    const float                 *crop,
                    size_t                       px,
                    double                       out[static 3]) {
    const size_t side = k->k;
    const size_t n    = side / 2;
    const size_t y    = px / w;
    const size_t x    = px % w;
    const float *kw   = k->w;
    double       s0 = 0.0, s1 = 0.0, s2 = 0.0;
    if (y >= n && y + n < h && x >= n && x + n < w) {
        for (size_t a = 0; a < side; a++) {
            /* row[0] is the pixel at column x - n, which tap side - 1 reads. */
            const float *row = crop + ((y + n - a) * w + (x - n)) * 3;
            for (size_t b = 0; b < side; b++) {
                const double kv = (double) *kw++;
                const float *p  = row + (side - 1 - b) * 3;
                s0 += kv * (double) p[0];
                s1 += kv * (double) p[1];
                s2 += kv * (double) p[2];
            }
        }
    } else {
        for (size_t a = 0; a < side; a++) {
            /* y + n - a clamped to [0, h - 1], without leaving size_t */
            const size_t ya = y + n < a ? 0 : (y + n - a < h ? y + n - a : h - 1);
            for (size_t b = 0; b < side; b++) {
                const size_t xb = x + n < b ? 0 : (x + n - b < w ? x + n - b : w - 1);
                const double kv = (double) *kw++;
                const float *p  = crop + (ya * w + xb) * 3;
                s0 += kv * (double) p[0];
                s1 += kv * (double) p[1];
                s2 += kv * (double) p[2];
            }
        }
    }
    out[0] = s0;
    out[1] = s1;
    out[2] = s2;
}

/* Blob energy of `crop` at the pixels the unit pools read:
 * e_bright = sum over RGB of relu(d), e_dark = sum of relu(-d). */
static void blob_energy(struct geist_decknet *d, const float *crop) {
    for (size_t i = 0; i < d->n_blob_px; i++) {
        const size_t px = d->blob_px[i];
        double       c[3];
        conv_at(d->crop_w, d->crop_h, &d->blob, crop, px, c);
        d->e_bright[px] = relud(c[0]) + relud(c[1]) + relud(c[2]);
        d->e_dark[px]   = relud(-c[0]) + relud(-c[1]) + relud(-c[2]);
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
static void units(struct geist_decknet *d, float t, const double *e, double out[static d->n_tips]) {
    pool_values(d->n_tips, &d->on, e, d->acc_on);
    pool_values(d->n_tips, &d->off, e, d->acc_off);
    for (size_t u = 0; u < d->n_tips; u++) {
        out[u] = satd((double) d->g_unit * (d->acc_on[u] - d->acc_off[u] - (double) t));
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

void decknet_plane(struct geist_decknet *d,
                   size_t                s,
                   enum decknet_plane    plane,
                   const float          *crop,
                   const float          *crop_ref) {
    struct geist_decknet_slot *o     = &d->slot_out[s];
    float                     *tips  = d->tips + s * d->n_tips;
    float                     *tubes = d->tubes + s * d->tube.n_units;
    switch (plane) {
    case DECKNET_PLANE_FLOOR:
        /* Step 3: change and wall against the empty deck. */
        o->occ_front =
                pool_mix(&d->occ_front, &d->change, d->g_change, d->t_change, crop, crop_ref);
        o->occ_rear = pool_mix(&d->occ_rear, &d->change, d->g_change, d->t_change, crop, crop_ref);
        o->wall_front = pool_mix(&d->wall_pool, &d->wall, d->g_wall, d->t_wall, crop, crop_ref);
        return;
    case DECKNET_PLANE_PLATE:
        /* Steps 4 and 5, plate and deep-well planes: dark blobs on the raster. */
        blob_energy(d, crop);
        units(d, d->t_dark, d->e_dark, d->unit);
        o->plate = (float) mean(d->n_tips, d->unit);
        return;
    case DECKNET_PLANE_DEEP:
        blob_energy(d, crop);
        units(d, d->t_dark, d->e_dark, d->unit);
        o->deep = (float) mean(d->n_tips, d->unit);
        return;
    case DECKNET_PLANE_HOLE:
        /* Open holes, and step 6, the tubes. */
        blob_energy(d, crop);
        units(d, d->t_dark, d->e_dark, d->hole);
        for (size_t i = 0; i < d->n_tube_px; i++) {
            const size_t px = d->tube_px[i];
            double       c[3];
            conv_at(d->crop_w, d->crop_h, &d->tube_kernel, crop, px, c);
            d->e_tube[px] = relud(c[0]) + relud(c[1]) + relud(c[2]);
        }
        pool_values(d->tube.n_units, &d->tube, d->e_tube, d->acc_tube);
        for (size_t t = 0; t < d->tube.n_units; t++) {
            d->acc_tube[t] = satd((double) d->g_unit * (d->acc_tube[t] - (double) d->t_tube));
            tubes[t]       = (float) d->acc_tube[t];
        }
        o->tube_n = (float) mean(d->n_tubes, d->acc_tube);
        o->tube_r = (float) mean(d->n_tubes, d->acc_tube + d->n_tubes);
        return;
    case DECKNET_PLANE_TIP: {
        /* Bright blob AND white AND no open hole (of the hole plane before). */
        blob_energy(d, crop);
        units(d, d->t_tip, d->e_bright, d->unit);
        memset(d->acc_on, 0, d->n_tips * sizeof d->acc_on[0]);
        for (size_t r = 0; r < d->on.n_rows; r++) {
            const float *p = crop + (size_t) d->on.px[r] * 3;
            d->acc_on[d->on.unit[r]] += (double) (d->on.w[r] * fminf(fminf(p[0], p[1]), p[2]));
        }
        double rack = 0.0;
        for (size_t u = 0; u < d->n_tips; u++) {
            const double white = satd((double) d->g_white * (d->acc_on[u] - (double) d->t_white));
            const double tip   = relud(d->unit[u] + white - 1.0 - d->hole[u]);
            tips[u]            = (float) tip;
            rack += tip + relud(d->hole[u] - tip); /* max(tip, hole) */
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
    static const float one[3] = {1.0f, 1.0f, 1.0f};
    float              scale[256];
    for (size_t b = 0; b < 256; b++) {
        scale[b] = (float) b / 255.0f;
    }
    float ga[3], gb[3];
    resample(d->img_w, d->img_h, d->bg_w, d->bg_h, one, d->h_cur, scale, cur, d->bg_a);
    resample(d->img_w, d->img_h, d->bg_w, d->bg_h, one, d->h_ref, scale, ref, d->bg_b);
    decknet_gains(d, d->bg_a, d->bg_b, ga, gb);
    if (!d->result.checkable) {
        return;
    }
    for (size_t s = 0; s < d->n_slots; s++) {
        /* The floor plane of slot s is homography 1 + DECKNET_PLANES * s. */
        const size_t first = (1 + DECKNET_PLANES * s) * 9;
        resample(d->img_w,
                 d->img_h,
                 d->crop_w,
                 d->crop_h,
                 gb,
                 d->h_ref + first,
                 scale,
                 ref,
                 d->crop_b);
        for (size_t p = 0; p < DECKNET_PLANES; p++) {
            resample(d->img_w,
                     d->img_h,
                     d->crop_w,
                     d->crop_h,
                     ga,
                     d->h_cur + first + p * 9,
                     scale,
                     cur,
                     d->crop_a);
            decknet_plane(d, s, (enum decknet_plane) p, d->crop_a, d->crop_b);
        }
    }
}

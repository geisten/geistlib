/*
 * src/archs/decknet/decknet.h — the geistbox deck-check network (#532).
 *
 * Layer: ARCHITECTURE. A fixed convolutional net with hand-written weights:
 * homography resample, 1x1 convolutions, two small dense 2D kernels, ReLU
 * and sparse weighted pooling. It has no decoder, tokenizer or session, so
 * it implements none of the <geist_arch.h> vtables; src/engine/decknet.c is
 * its only caller and defines the public entry points of <geist_decknet.h>.
 *
 * Defined in:
 *   src/archs/decknet/load.c — file contract, validation, workspace
 *   src/archs/decknet/run.c  — the forward pass
 */
#pragma once

#ifndef GEIST_INTERNAL_ARCH_LAYER
#error "decknet/decknet.h is internal to the architecture layer."
#endif

#include "gguf_reader.h"

#include <geist.h>
#include <geist_decknet.h>

#include <stddef.h>
#include <stdint.h>

/* The forward pass names its planes, in the order of the homographies of a
 * slot. */
enum decknet_plane : uint32_t {
    DECKNET_PLANE_FLOOR = 0,
    DECKNET_PLANE_PLATE = 1,
    DECKNET_PLANE_DEEP  = 2,
    DECKNET_PLANE_HOLE  = 3,
    DECKNET_PLANE_TIP   = 4,
};
static_assert(sizeof(enum decknet_plane) == sizeof(uint32_t));
constexpr size_t DECKNET_PLANES = 5;
/* Limits on what a file may declare. Pixel indices travel as F32 and are
 * exact below 2^24; the crop limit is far below that and bounds the
 * workspace a hostile file can ask for (about 250 MB). */
constexpr size_t DECKNET_MAX_SLOTS     = 64;
constexpr size_t DECKNET_MAX_PIXELS    = (size_t) 1 << 22U;
constexpr size_t DECKNET_MAX_IMAGE_DIM = 32768;
constexpr size_t DECKNET_MAX_KERNEL    = 255;
constexpr size_t DECKNET_MAX_MIX_ROWS  = 64;
constexpr size_t DECKNET_MIX_IN        = 6; /* current RGB, empty-deck RGB */

/* Sparse pooling rows, ascending by (unit, pixel): unit `unit[r]` adds
 * `w[r]` times the value at crop pixel `px[r]`. */
struct decknet_pool {
    size_t    n_rows;
    size_t    n_units;
    uint32_t *unit;
    uint32_t *px;
    float    *w;
};

/* A k x k kernel, k odd. */
struct decknet_kernel {
    size_t k;
    float *w;
};

/* A 1x1 convolution of `rows` units on the six channels of a pixel pair. */
struct decknet_mix {
    size_t rows;
    float *w; /* rows x DECKNET_MIX_IN */
};

struct geist_decknet {
    /* ---- the file ---- */
    char     slots[DECKNET_MAX_SLOTS + 1];
    size_t   n_slots;
    size_t   n_h; /* 1 + DECKNET_PLANES * n_slots */
    size_t   img_w, img_h, img_bytes;
    size_t   crop_w, crop_h, crop_px;
    size_t   bg_w, bg_h, bg_px;
    float    t_change, g_change, t_wall, g_wall, t_tip, t_dark, t_tube, g_unit, t_white, g_white;
    float    gain_min, gain_max, bg_rest_max, occ, smooth;
    float    planes_mm[DECKNET_PLANES];
    float    bg_nominal[3];
    float   *h_cur; /* n_h x 9, replaceable (geist_decknet_set_homographies) */
    float   *h_ref;
    uint64_t h_gen; /* counts the replacements; a stream's tables are of one value */
    struct decknet_mix    change, wall;
    struct decknet_kernel blob, tube_kernel;
    struct decknet_pool   on, off, tube, wall_pool, occ_front, occ_rear;
    /* Crop pixels a pooling row reads a convolution at, ascending. The
     * kernels are evaluated there and nowhere else. */
    size_t    n_blob_px, n_tube_px;
    uint32_t *blob_px, *tube_px;
    size_t    n_tips;  /* units of pool.on and pool.off */
    size_t    n_tubes; /* half the units of pool.tube */

    /* ---- workspace, sized at load ---- */
    float  *crop_a, *crop_b;                /* crop_px x 3 */
    float  *bg_a, *bg_b;                    /* bg_px x 3 */
    double *e_bright, *e_dark, *e_tube;     /* crop_px, written at blob_px / tube_px */
    double *acc_on, *acc_off, *unit, *hole; /* n_tips */
    double *acc_tube;                       /* 2 * n_tubes */

    /* ---- the last run ---- */
    bool                        have_run;
    struct geist_decknet_result result;
    struct geist_decknet_slot  *slot_out; /* n_slots; arrays point into the two below */
    float                      *tips;     /* n_slots x n_tips */
    float                      *tubes;    /* n_slots x 2 * n_tubes */
};

/* Builds a handle from an open GGUF. On failure *out is nullptr and `err`
 * (NUL-terminated, at most err_cap bytes) names the cause: GEIST_E_UNSUPPORTED
 * when general.architecture is not "decknet", GEIST_E_FORMAT when the file
 * breaks the contract, GEIST_E_OOM. Nothing of `g` is referenced afterwards. */
[[nodiscard]] enum geist_status decknet_create(size_t                 err_cap,
                                               const struct gguf_ctx *g,
                                               char                   err[static err_cap],
                                               struct geist_decknet **out);

/* Frees a handle and everything it owns; nullptr is a no-op. */
void decknet_destroy(struct geist_decknet *d);

/* One forward pass over two RGB8 images of d->img_bytes bytes each; writes
 * d->result and, for a checkable pair, every slot. Allocates nothing. */
void decknet_run(struct geist_decknet *d,
                 size_t                n_bytes,
                 const uint8_t         cur[static n_bytes],
                 const uint8_t         ref[static n_bytes]);

/* ---- the pass in pieces, for a caller that fills the crops (stream.c) ---- */

/* Where crop pixel (x, y) samples an img_w x img_h image through the
 * homography h: (u, v), clamped to [0, size - 1.001], so that the 2 x 2
 * neighbourhood at ((size_t) u, (size_t) v) is inside the image. NaN and
 * everything below 0 give 0, +Inf the upper bound. decknet_run and a
 * stream's sample tables agree bit for bit because both call this. */
static inline void decknet_sample_at(size_t      img_w,
                                     size_t      img_h,
                                     size_t      x,
                                     size_t      y,
                                     const float h[static 9],
                                     double     *u,
                                     double     *v) {
    const double j    = (double) x;
    const double i    = (double) y;
    const double p0   = j * (double) h[0] + i * (double) h[1] + (double) h[2];
    const double p1   = j * (double) h[3] + i * (double) h[4] + (double) h[5];
    const double p2   = j * (double) h[6] + i * (double) h[7] + (double) h[8];
    const double u_hi = (double) img_w - 1.001;
    const double v_hi = (double) img_h - 1.001;
    const double pu   = p0 / p2 - 0.5;
    const double pv   = p1 / p2 - 0.5;
    *u                = !(pu > 0.0) ? 0.0 : (pu < u_hi ? pu : u_hi);
    *v                = !(pv > 0.0) ? 0.0 : (pv < v_hi ? pv : v_hi);
}

/* The gain of one image from its worktable strip (d->bg_px x 3 floats,
 * sampled without gain): bg_nominal / max(mean, 1e-3) per colour. */
void decknet_gain(const struct geist_decknet *d, const float *bg, float g[static 3]);

/* Step 2 of the pass, from the two worktable strips (d->bg_px x 3 floats
 * each, sampled without gain): the gains ga and gb the crops of the current
 * and the empty-deck image are sampled with, and d->result. A pair that is
 * not checkable has its slot state cleared here and gets no decknet_plane. */
void decknet_gains(struct geist_decknet *d,
                   const float          *bg_a,
                   const float          *bg_b,
                   float                 ga[static 3],
                   float                 gb[static 3]);

/* Steps 3 to 6 for one plane of slot s, from its crop of the current image
 * (d->crop_px x 3 floats, sampled with ga); the floor plane also reads
 * `crop_ref`, the floor crop of the empty-deck image (sampled with gb), the
 * others ignore it. Of a crop only the pixels listed by decknet_plane_pixels
 * are read. The tip plane uses what the hole plane of the same slot left,
 * so the planes of a slot run in order. */
void decknet_plane(struct geist_decknet *d,
                   size_t                s,
                   enum decknet_plane    plane,
                   const float          *crop,
                   const float          *crop_ref);

/* mark[px] = 1 for each of the d->crop_px crop pixels decknet_plane reads of
 * a crop of that plane (of both crops, for the floor plane), 0 for the
 * others: the pixels of the pooling rows, and the kernel windows around the
 * pixels a convolution is evaluated at. */
void decknet_plane_pixels(const struct geist_decknet *d, enum decknet_plane plane, uint8_t *mark);

/* ---- stream (stream.c, #533) ------------------------------------------ */

/* One bilinear sample: crop pixel `px` reads the 2 x 2 camera pixels at
 * (x, y) with the fractions fu and fv. What decknet_sample_at gives for a
 * homography, kept so that a frame costs no coordinate arithmetic. */
struct decknet_sample {
    double   fu, fv;
    uint32_t px;
    uint16_t x, y; /* at most DECKNET_MAX_IMAGE_DIM - 2 */
};
static_assert(DECKNET_MAX_IMAGE_DIM <= (size_t) UINT16_MAX + 1);

struct geist_decknet_stream {
    struct geist_decknet *d; /* borrowed; the pass runs in its workspace */
    /* The crop pixels each plane reads, ascending (decknet_plane_pixels). */
    size_t    n_px[DECKNET_PLANES];
    uint32_t *px[DECKNET_PLANES];
    size_t    per_slot; /* sum of n_px */
    /* Sample tables of the current image's homographies. */
    struct decknet_sample *bg;  /* bg_px */
    struct decknet_sample *tab; /* n_slots x per_slot: slot by slot, plane by plane */
    /* The empty deck, sampled once: its strip, and the floor crop of each
     * slot with the gain already applied. */
    float   *bg_ref;    /* bg_px x 3 */
    float   *floor_ref; /* n_slots x crop_px x 3, written at px[DECKNET_PLANE_FLOOR] */
    float    scale[256];
    bool     have_ref;
    uint64_t h_gen; /* d->h_gen the tables and the reference were made at */
    bool     have_result;
    uint64_t timestamp, n_frames;
};

/* The public entry points of the same names without `geist_`, minus the
 * null checks (src/engine/decknet.c). */
[[nodiscard]] enum geist_status decknet_stream_create(struct geist_decknet         *d,
                                                      struct geist_decknet_stream **out);
void                            decknet_stream_destroy(struct geist_decknet_stream *s);
[[nodiscard]] enum geist_status decknet_stream_set_reference(struct geist_decknet_stream      *s,
                                                             const struct geist_decknet_frame *ref);
[[nodiscard]] enum geist_status decknet_stream_push_frame(struct geist_decknet_stream      *s,
                                                          const struct geist_decknet_frame *frame);

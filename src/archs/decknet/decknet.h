/*
 * src/archs/decknet/decknet.h — the geistbox deck-check network (#532).
 *
 * Layer: ARCHITECTURE. A fixed convolutional net with hand-written weights:
 * homography resample, 1x1 convolutions, two small dense 2D kernels, ReLU
 * and sparse weighted pooling. A crop is kept in integers (DECKNET_Q_ONE):
 * the bilinear sample of the camera bytes in fixed point, before any gain,
 * so that the kernels sum integers, exactly and in any order (#533).
 *
 * It has no decoder, tokenizer or session, so it implements none of the
 * <geist_arch.h> vtables; src/engine/decknet.c is its only caller and
 * defines the public entry points of <geist_decknet.h>.
 *
 * Defined in:
 *   src/archs/decknet/load.c — file contract, validation, workspace
 *   src/archs/decknet/run.c  — the forward pass
 *   src/archs/decknet/stream.c — the pass on a camera stream, on the CPU or
 *                                with its integer part on a GPU
 */
#pragma once

#ifndef GEIST_INTERNAL_ARCH_LAYER
#error "decknet/decknet.h is internal to the architecture layer."
#endif

#include "gguf_reader.h"

#include <geist.h>
#include <geist_decknet.h>
#include <geist_decknet_gpu.h>

#include <stdatomic.h>
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

/* A crop pixel is the camera byte value in 1/65536, so 255 << 16 at most:
 * the bilinear sample of four bytes with weights in 1/2^24, rounded once.
 * Four of them per pixel, R G B and a zero, so that a pixel is one vector.
 * The gain of an image is applied where a float is needed (decknet_unit). */
constexpr uint32_t DECKNET_Q_ONE      = 255u << 16U;
constexpr uint32_t DECKNET_WEIGHT     = 1u << 24U;
constexpr size_t   DECKNET_CROP_CH    = 4;
constexpr size_t   DECKNET_MAX_KERNEL = 255;
/* The product of a sample fits 32 bits, and so does the sum of
 * DECKNET_SUM_PIXELS crop pixels. */
constexpr size_t DECKNET_SUM_PIXELS = 256;
static_assert((uint64_t) DECKNET_SUM_PIXELS * DECKNET_Q_ONE <= UINT32_MAX);
static_assert((uint64_t) 255 * DECKNET_WEIGHT + 128 <= UINT32_MAX);
static_assert(DECKNET_MAX_KERNEL <= DECKNET_SUM_PIXELS);

/* A kernel tap as run.c reads it: the crop pixel at (y + dy, x + dx), which
 * is `off` pixels from (y, x) where the window lies inside the crop. */
struct decknet_tap {
    int32_t off;
    int16_t dy, dx;
};

/* Neighbouring taps of one kernel row, whose sum is the prefix sum `hi`
 * pixels from (y, x) less the one `lo` pixels from it
 * (decknet_work.prefix). */
struct decknet_span {
    int32_t lo, hi;
};

/* `n_taps` taps of one weight whose pixels sum in 32 bits. A weight with
 * more than DECKNET_SUM_PIXELS taps has several parts; `last` ends it, and
 * `value` is the weight. The same taps are `n_spans` spans. */
struct decknet_part {
    uint32_t n_taps, n_spans;
    bool     last;
    float    value;
};

/* A k x k kernel, k odd: its weights as the file has them, and the same
 * kernel with the taps ordered by weight (load.c), which is what run.c
 * evaluates: the sum over the distinct weights of weight * (integer sum of
 * its pixels). */
struct decknet_kernel {
    size_t               k;
    float               *w;
    size_t               n_parts;
    struct decknet_part *part;
    struct decknet_tap  *tap; /* k x k */
    /* The taps as rows of neighbours, part by part; nullptr for a kernel
     * that has too few neighbours of one weight for that to be less work
     * (DECKNET_SPAN_TAPS). */
    struct decknet_span *span;
};

/* A span is two reads, and a tap one; a kernel is evaluated by its spans
 * from this many taps per span on. */
constexpr size_t DECKNET_SPAN_TAPS = 3;

/* A 1x1 convolution of `rows` units on the six channels of a pixel pair. */
struct decknet_mix {
    size_t rows;
    float *w; /* rows x DECKNET_MIX_IN */
    /* The same weights four rows at a time, for run.c: per four rows
     * DECKNET_MIX_IN vectors of four, input by input; zero where the rows
     * end. */
    float *by_four;
};

/* What the planes of one slot are computed in: a crop, and what steps 3 to
 * 6 derive from it on the way to the units. The handle has one; a stream
 * that computes its slots on two threads has a second. */
struct decknet_ws {
    uint32_t *crop;                           /* (crop_px + 1) x DECKNET_CROP_CH */
    float    *mix;                            /* a 1x1 value per floor pooling row (n_mix) */
    double   *sums, *tube_sums;               /* 3 kernel sums per blob_px / tube_px */
    double   *e_bright, *e_dark, *e_tube;     /* crop_px, written at blob_px / tube_px */
    double   *acc_on, *acc_off, *unit, *hole; /* n_tips */
    double   *acc_tube;                       /* 2 * n_tubes */
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
    struct decknet_ws ws;
    uint32_t         *bg_q;        /* bg_px x DECKNET_CROP_CH */
    float            *crop_b;      /* crop_px x 3: the empty deck's floor, with its gain */
    float            *bg_a, *bg_b; /* bg_px x 3 */
    size_t            n_mix;       /* rows of occ_front, occ_rear and wall_pool */

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

/* A workspace for d's network. False, with *ws to be freed all the same, when
 * memory is short. decknet_ws_free tolerates a zeroed or half-built one. */
[[nodiscard]] bool decknet_ws_alloc(const struct geist_decknet *d, struct decknet_ws *ws);
void               decknet_ws_free(struct decknet_ws *ws);

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

/* The four weights of a bilinear sample at (u, v) as decknet_sample_at gives
 * them, in 1/2^24: w[0] for the pixel right of ((size_t) u, (size_t) v), w[1]
 * for the one below, w[2] for the one below right; the pixel itself has the
 * rest, so the four sum to 2^24 exactly and a flat image samples to itself. */
static inline void decknet_weights(double u, double v, uint32_t w[static 3]) {
    const double fu  = u - (double) (size_t) u;
    const double fv  = v - (double) (size_t) v;
    const double one = (double) DECKNET_WEIGHT;
    /* Rounded to nearest; each clamped to what the ones before it left, so
     * that three roundings up cannot exceed the whole. */
    const uint32_t w11 = (uint32_t) (fu * fv * one + 0.5);
    const uint32_t w01 = (uint32_t) (fu * (1.0 - fv) * one + 0.5);
    const uint32_t w10 = (uint32_t) ((1.0 - fu) * fv * one + 0.5);
    w[2]               = w11 < DECKNET_WEIGHT ? w11 : DECKNET_WEIGHT;
    w[0]               = w01 < DECKNET_WEIGHT - w[2] ? w01 : DECKNET_WEIGHT - w[2];
    w[1]               = w10 < DECKNET_WEIGHT - w[2] - w[0] ? w10 : DECKNET_WEIGHT - w[2] - w[0];
}

/* One colour of a crop pixel: the bytes b00 (the pixel), b01 (right of it),
 * b10 (below) and b11 with the weights of decknet_weights, rounded to
 * 1/65536 of a byte value. */
static inline uint32_t
decknet_blend(const uint32_t w[static 3], uint8_t b00, uint8_t b01, uint8_t b10, uint8_t b11) {
    const uint32_t w00 = DECKNET_WEIGHT - w[0] - w[1] - w[2];
    return (w00 * b00 + w[0] * b01 + w[1] * b10 + w[2] * b11 + 128) >> 8U;
}

/* What multiplies a crop value to give the image's 0..1 scale times `gain`. */
static inline double decknet_scale(float gain) {
    return (double) gain / (double) DECKNET_Q_ONE;
}

/* A crop value as the F32 the 1x1 layers and the gain check compute with. */
static inline float decknet_unit(uint32_t q, double scale) {
    return (float) ((double) q * scale);
}

/* out[i x 3 + c] = decknet_unit of the strip q (d->bg_px x DECKNET_CROP_CH)
 * without gain: what decknet_gain and decknet_gains take. */
void decknet_strip(const struct geist_decknet *d, const uint32_t *q, float *out);

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

/* The side of a square of crop pixels that is stale or not as a whole
 * (decknet_work.stale). */
constexpr size_t DECKNET_CELL = 16;

/* One plane of one slot: its crop, and the values the pass computes from the
 * crop alone before anything is pooled. A still pass and a stream without
 * gating compute them in the handle's workspace, plane after plane; a gated
 * stream keeps them per slot and plane and has only the stale ones
 * recomputed. */
struct decknet_work {
    /* The crop of the current image, d->crop_px x DECKNET_CROP_CH; nullptr
     * with `prefix`. */
    const uint32_t *crop;
    /* nullptr, or the same crop as a stream samples it for a plane whose
     * kernels have spans (never with `kept`): per run of crop pixels the
     * plane reads, at each pixel the sum of the run's pixels before it, and
     * at the pixel behind the run the sum of all of it; d->crop_px + 1
     * pixels. In 32 bits that wrap: the difference of two of them is the
     * sum of the pixels between, which fits, and a pixel is the difference
     * of the two around it. */
    const uint32_t *prefix;
    const float    *crop_ref; /* floor: the empty deck's crop x 3, with its gain */
    float          *mix;      /* floor: d->n_mix values, a 1x1 unit per pooling row */
    double         *sums;     /* other planes: 3 kernel sums per d->blob_px */
    double         *tube;     /* hole plane: 3 kernel sums per d->tube_px */
    /* false: mix, sums and tube are computed from the crop. true: they hold
     * what an earlier crop gave (mix: with the same gains), and only what
     * reads a pixel that differs from that crop's is computed again. */
    bool kept;
    /* With `kept`, the pixels that differ; nullptr when none does. Floor
     * plane: one flag per crop pixel. Other planes: one flag per
     * DECKNET_CELL x DECKNET_CELL square of the crop, in rows of
     * ceil(crop_w / DECKNET_CELL). */
    const uint8_t *stale;
};

/* Steps 3 to 6 for one plane of slot s, from its crop of the current image
 * (sampled without gain; `ga` is applied here); the floor plane also reads
 * `crop_ref`. Of a crop only the pixels listed by decknet_plane_pixels are
 * read. The tip plane uses what the hole plane of the same slot left in
 * `ws`, so the planes of a slot run in order in one workspace. Two slots may
 * be computed at once in two workspaces: a slot writes its own outputs. */
void decknet_plane(struct geist_decknet      *d,
                   struct decknet_ws         *ws,
                   size_t                     s,
                   enum decknet_plane         plane,
                   const float                ga[static 3],
                   const struct decknet_work *w);

/* sums[i x 3 + c] for n pixels a kernel is evaluated at, as decknet_plane
 * computes them from a crop, from the integer sums of the kernel's parts:
 * acc has, pixel by pixel and part by part, four 32-bit values, the sum of
 * the crop pixels of the part's taps (R G B and a zero), which is what a GPU
 * hands back (<geist_decknet_gpu.h>). A plane whose decknet_work has these
 * sums and `kept` without `stale` computes no kernel. */
void decknet_kernel_parts(const struct decknet_kernel *k,
                          size_t                       n,
                          const uint32_t              *acc,
                          double                      *sums);

/* mark[px] = 1 for each of the d->crop_px crop pixels decknet_plane reads of
 * a crop of that plane (of both crops, for the floor plane), 0 for the
 * others: the pixels of the pooling rows, and the kernel windows around the
 * pixels a convolution is evaluated at. */
void decknet_plane_pixels(const struct geist_decknet *d, enum decknet_plane plane, uint8_t *mark);

/* ---- stream (stream.c, #533) ------------------------------------------ */

/* One bilinear sample: a crop pixel reads the 2 x 2 camera pixels at (x, y)
 * with the weights w (decknet_weights). What decknet_sample_at gives for a
 * homography, kept so that a frame costs no coordinate arithmetic. Which
 * crop pixel it is the table does not say: decknet_run does. */
struct decknet_sample {
    uint32_t w[3];
    uint16_t x, y; /* at most DECKNET_MAX_IMAGE_DIM - 2 */
};
static_assert(DECKNET_MAX_IMAGE_DIM <= (size_t) UINT16_MAX + 1);
static_assert(sizeof(struct decknet_sample) == 16, "four 32-bit lanes, three of them the weights");
/* A GPU reads the same 16 bytes. */
static_assert(sizeof(struct decknet_sample) == sizeof(struct geist_decknet_gpu_sample) &&
              offsetof(struct decknet_sample, x) == offsetof(struct geist_decknet_gpu_sample, x) &&
              offsetof(struct decknet_sample, y) == offsetof(struct geist_decknet_gpu_sample, y));

/* The crop pixels px .. px + n - 1, whose samples a table has side by side:
 * what a plane reads of its crop is a few hundred such runs. */
struct decknet_run {
    uint32_t px, n;
};

/* The side of the squares of camera pixels a stream reads, converts,
 * compares and keeps as a whole. Part of the contract of a gated stream
 * (include/geist_decknet.h). */
constexpr size_t DECKNET_TILE = GEIST_DECKNET_TILE;
/* A gated frame in which more than 7 of 8 tiles were taken is computed
 * without the caches (stream.c, push_frame). */
constexpr size_t DECKNET_GATE_WHOLE_NUM = 7;
constexpr size_t DECKNET_GATE_WHOLE_DEN = 8;
/* A stream computes its slots on one thread or on two. */
constexpr size_t DECKNET_STREAM_THREADS = 2;
static_assert(DECKNET_MAX_SLOTS <= 64, "geist_decknet_stream.slot_tiles has a bit per slot");

/* A sample is computed from 16 bytes of each of its two rows, of which the
 * first six are its pixels, where that many are there to read. */
constexpr size_t DECKNET_SAMPLE_BYTES = 16;

/* Tiles [x0, x1] x [y0, y1]. */
struct decknet_box {
    uint16_t x0, x1, y0, y1;
};
constexpr size_t DECKNET_CHUNK = 256;

struct geist_decknet_stream {
    struct geist_decknet *d; /* borrowed; the pass runs in its workspace */
    /* The crop pixels each plane reads (decknet_plane_pixels), ascending, as
     * runs of neighbours: n_px pixels in n_runs runs. */
    size_t              n_px[DECKNET_PLANES], n_runs[DECKNET_PLANES];
    struct decknet_run *run[DECKNET_PLANES];
    size_t              per_slot; /* sum of n_px */
    /* Sample tables of the current image's homographies. */
    struct decknet_sample *bg;  /* bg_px */
    struct decknet_sample *tab; /* n_slots x per_slot: slot by slot, plane by plane */
    /* The empty deck, sampled once: its strip, and the floor crop of each
     * slot with the gain already applied. */
    float   *bg_ref;    /* bg_px x 3 */
    float   *floor_ref; /* n_slots x crop_px x 3, written at the floor plane's pixels */
    bool     have_ref;
    uint64_t h_gen; /* d->h_gen the tables and the reference were made at */
    bool     have_result;
    uint64_t timestamp, n_frames;
    size_t   n_tiles_changed, n_samples; /* of the last frame */

    /* ---- tiles ---- */
    size_t   tiles_w, tiles_h;
    size_t   n_tiles; /* tiles a sample reads */
    uint8_t *used;    /* tiles_w x tiles_h: a sample reads the tile */
    /* tiles_w x tiles_h: bit i is set where a sample of slot i reads the
     * tile. And n_slots + 1 rectangles: the tiles slot i reads lie in
     * read_box[i], those of the worktable strip in read_box[n_slots]. */
    uint64_t           *slot_tiles;
    struct decknet_box *read_box;
    uint8_t            *changed; /* tiles_w x tiles_h, gated: see tiles_gate */
    /* (tiles_w + 1) x (tiles_h + 1), gated: the number of `changed` tiles
     * left of and above a tile corner, so that "any in this rectangle" is
     * four reads. */
    uint32_t *changed_sum;
    /* Per DECKNET_CHUNK samples of `tab`, the rectangle of tiles their first
     * pixels lie in: a chunk with no changed tile in it is not looked at. */
    struct decknet_box *box;
    /* img_w x img_h x 3 and DECKNET_SAMPLE_BYTES behind them. For a gated
     * stream the kept tiles of any frame as RGB8, written at the used
     * tiles of an image of the frame's size. For an NV12 or YUYV frame that
     * is not gated, what the worktable strip and then each slot of the
     * first thread reads, as RGB8, packed (stream.c, pixels_box). */
    uint8_t *rgb;
    /* As much again: the same for the slots of the second thread, which
     * converts what its slots read while the first converts its own. */
    uint8_t *rgb2;

    /* ---- gating: allocated by the first set_gate that turns it on ---- */
    bool                      gate;
    uint32_t                  threshold;
    bool                      have_kept; /* the kept tiles are of a frame of kept_format */
    enum geist_decknet_format kept_format;
    uint8_t                  *kept; /* the kept tiles of an NV12 or YUYV frame, rows unpadded */
    bool      cached;    /* the caches below are of the kept tiles as they were before this frame */
    uint32_t *crop;      /* n_slots x DECKNET_PLANES crops */
    double   *sums;      /* n_slots x (DECKNET_PLANES - 1) x n_blob_px x 3 */
    double   *tube;      /* n_slots x n_tube_px x 3 */
    float    *mix;       /* n_slots x n_mix */
    float     mix_ga[3]; /* the gains `mix` was computed with */
    uint8_t  *stale;     /* per thread crop_px: decknet_work.stale, zero between planes */

    /* ---- a queue of one (decknet_stream_offer) ----
     * The frame that waits for the next pass, lent by the caller, and how
     * many were replaced while they waited. The only members a second
     * thread touches. */
    _Atomic(const struct geist_decknet_frame *) offered;
    atomic_uint_fast64_t                        n_dropped;

    /* ---- two threads (decknet_stream_set_threads) ---- */
    size_t            n_threads; /* 1 or DECKNET_STREAM_THREADS */
    bool              spans;     /* both kernels have spans: their planes are sampled as sums */
    struct decknet_ws ws2;       /* what the second one computes in; the first has d->ws */

    /* ---- a GPU (decknet_stream_set_gpu) ----
     * gpu_ops is nullptr while every frame is the CPU's. Otherwise the
     * backend's state for this stream and its host memory: the sample
     * table in the GPU's order, the frame, the pixels and the kernel sums
     * (stream.c, "a GPU"). */
    const struct geist_decknet_gpu_ops *gpu_ops;
    struct geist_decknet_gpu           *gpu;
    struct geist_decknet_gpu_memory     gpu_mem;
    size_t gpu_y0, gpu_y1; /* the rows of a frame a sample reads, [y0, y1) */
};

/* The public entry points of the same names without `geist_`, minus the
 * null checks (src/engine/decknet.c). */
[[nodiscard]] enum geist_status decknet_stream_create(struct geist_decknet         *d,
                                                      struct geist_decknet_stream **out);
void                            decknet_stream_destroy(struct geist_decknet_stream *s);
[[nodiscard]] enum geist_status
decknet_stream_set_gate(struct geist_decknet_stream *s, bool on, uint32_t threshold);
[[nodiscard]] enum geist_status decknet_stream_set_threads(struct geist_decknet_stream *s,
                                                           size_t                       n);
/* geist_decknet_stream_set_backend with the backend's decknet table, which
 * the engine looks up; nullptr for both goes back to the CPU. */
[[nodiscard]] enum geist_status decknet_stream_set_gpu(struct geist_decknet_stream        *s,
                                                       const struct geist_decknet_gpu_ops *ops,
                                                       struct geist_backend               *be);
[[nodiscard]] enum geist_status decknet_stream_set_reference(struct geist_decknet_stream      *s,
                                                             const struct geist_decknet_frame *ref);
[[nodiscard]] enum geist_status decknet_stream_push_frame(struct geist_decknet_stream      *s,
                                                          const struct geist_decknet_frame *frame);
[[nodiscard]] enum geist_status decknet_stream_offer(struct geist_decknet_stream       *s,
                                                     const struct geist_decknet_frame  *frame,
                                                     const struct geist_decknet_frame **dropped);
[[nodiscard]] enum geist_status
decknet_stream_push_offered(struct geist_decknet_stream       *s,
                            const struct geist_decknet_frame **done);
/* Frames dropped by decknet_stream_offer so far. */
uint64_t decknet_stream_dropped(const struct geist_decknet_stream *s);

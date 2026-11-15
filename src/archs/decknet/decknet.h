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

/* The forward pass names its planes: floor, plate, deep-well, hole, tip. */
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
    char   slots[DECKNET_MAX_SLOTS + 1];
    size_t n_slots;
    size_t n_h; /* 1 + DECKNET_PLANES * n_slots */
    size_t img_w, img_h, img_bytes;
    size_t crop_w, crop_h, crop_px;
    size_t bg_w, bg_h, bg_px;
    float  t_change, g_change, t_wall, g_wall, t_tip, t_dark, t_tube, g_unit, t_white, g_white;
    float  gain_min, gain_max, bg_rest_max, occ, smooth;
    float  planes_mm[DECKNET_PLANES];
    float  bg_nominal[3];
    float *h_cur; /* n_h x 9, replaceable (geist_decknet_set_homographies) */
    float *h_ref;
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

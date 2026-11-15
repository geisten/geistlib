/*
 * decknet: the geistbox deck-check network, a small fixed convolutional net
 * for images (#532). Two camera images in (the deck now, the empty deck),
 * one state vector per deck position out. All declarations are EXPERIMENTAL.
 *
 * Build with DECKNET=1. A library built without it has none of these
 * symbols, so a default build is byte-for-byte what it was; a program that
 * includes this header links only against a DECKNET=1 library.
 *
 * No tokenizer, no session, no backend handle: the forward pass is portable
 * C and is the same code on every platform. File contract, forward pass and
 * calibration: docs/DECKNET.md.
 */
#pragma once

#include <geist.h>

#ifdef __cplusplus
extern "C" {
#endif

struct geist_decknet;

/* @stability EXPERIMENTAL — which image's homographies a call replaces. */
enum geist_decknet_image : uint32_t {
    GEIST_DECKNET_IMAGE_CURRENT = 0, /* the deck now (`cur` of geist_decknet_run) */
    GEIST_DECKNET_IMAGE_EMPTY   = 1, /* the empty deck (`ref`) */
};
static_assert(sizeof(enum geist_decknet_image) == sizeof(uint32_t));

/* @stability EXPERIMENTAL — what the loaded file fixes. Pointers are borrowed
 * and valid until geist_decknet_destroy. */
struct geist_decknet_info {
    const char  *slots; /* NUL-terminated, one letter per deck position */
    size_t       n_slots;
    size_t       image_width; /* both images of geist_decknet_run, in pixels */
    size_t       image_height;
    size_t       n_planes;       /* rectified planes per slot (5) */
    const float *planes_mm;      /* n_planes plane heights above the slot floor */
    size_t       n_homographies; /* 1 + n_planes * n_slots */
    size_t       n_tips;         /* raster positions per slot (96) */
    size_t       n_tubes;        /* tube positions per orientation (8) */
    float        occ;            /* head threshold of the file: occupied from here */
    float        smooth;         /* head threshold of the file: no raster above it */
};

/* @stability EXPERIMENTAL — the image check of one run. When `checkable` is
 * false the pair cannot be compared (the brightness ratio `gain` left the
 * file's [gain_min, gain_max], or `bg_changed` reached bg_rest_max) and no
 * slot was evaluated. */
struct geist_decknet_result {
    bool  checkable;
    float gain[3];    /* RGB brightness, current over empty, on the bare worktable */
    float bg_changed; /* mean changed fraction of the bare worktable strip */
};

/* @stability EXPERIMENTAL — the units of one slot (0..1 for a file whose
 * pooling weights sum to 1 per unit, as the exporter writes them). The
 * arrays are borrowed: valid until the next geist_decknet_run on, or the
 * destruction of, this handle. Thresholding (occ, smooth, argmax over the
 * type units) is the caller's. */
struct geist_decknet_slot {
    float        occ_front;
    float        occ_rear;
    float        wall_front;
    float        rack;
    float        plate;
    float        deep;
    float        tube_n;
    float        tube_r;
    const float *tips;      /* n_tips */
    const float *tubes;     /* n_tubes */
    const float *tubes_rot; /* n_tubes, the rack turned by 180 degrees */
};

/* @stability EXPERIMENTAL
 * Loads a decknet GGUF (general.architecture = "decknet") and allocates the
 * whole workspace; no later call allocates. The file is validated against
 * the contract in docs/DECKNET.md and copied, so nothing of it is kept open.
 * On failure *out is nullptr and geist_last_create_error() names the cause:
 * GEIST_E_IO (the path cannot be opened or parsed as GGUF), GEIST_E_FORMAT
 * (memory that is not a GGUF, or a file that breaks the contract),
 * GEIST_E_UNSUPPORTED (another architecture), GEIST_E_OOM,
 * GEIST_E_INVALID_ARG. */
[[nodiscard]] enum geist_status geist_decknet_load(const char *path, struct geist_decknet **out);

/* @stability EXPERIMENTAL — the same from `size` bytes in memory, which the
 * caller may release as soon as the call returns. */
[[nodiscard]] enum geist_status
geist_decknet_load_from_memory(size_t size, const uint8_t *data, struct geist_decknet **out);

/* @stability EXPERIMENTAL — nullptr is a no-op. */
void geist_decknet_destroy(struct geist_decknet *d);

/* @stability EXPERIMENTAL — GEIST_E_INVALID_ARG for a null argument. */
[[nodiscard]] enum geist_status
geist_decknet_info(const struct geist_decknet *d, struct geist_decknet_info *out);

/* @stability EXPERIMENTAL
 * Calibration without a new file: replaces all `n` homographies of one
 * image, n * 9 floats, row-major 3 x 3 each, in the file's order ([0] the
 * bare worktable strip, then n_planes per slot). Each maps a crop pixel
 * (column, row, 1) to a camera pixel (u, v, w). Both images start with the
 * file's homographies. GEIST_E_INVALID_ARG, with nothing replaced, when n is
 * not n_homographies or a value is not finite. */
[[nodiscard]] enum geist_status geist_decknet_set_homographies(struct geist_decknet      *d,
                                                               size_t                     n,
                                                               enum geist_decknet_image   image,
                                                               const float               *h);

/* @stability EXPERIMENTAL
 * One forward pass. `cur` and `ref` are caller-owned RGB8 images, rows top
 * to bottom without padding, n_bytes = image_width * image_height * 3 each;
 * any other n_bytes is GEIST_E_INVALID_ARG. Lens distortion is removed by
 * the caller first. Returns GEIST_OK when the pass ran, also for a pair that
 * is not checkable (see geist_decknet_result). Allocates nothing. Never call
 * one handle from two threads at once. On failure *out is zero. */
[[nodiscard]] enum geist_status geist_decknet_run(struct geist_decknet        *d,
                                                  size_t                       n_bytes,
                                                  const uint8_t               *cur,
                                                  const uint8_t               *ref,
                                                  struct geist_decknet_result *out);

/* @stability EXPERIMENTAL
 * The units of slot `slot` (index into geist_decknet_info.slots) from the
 * last run. GEIST_E_INVALID_STATE, with *out zero, before the first run and
 * after a run whose pair was not checkable: such a pair has no slot state,
 * and zeros would read as an empty deck. GEIST_E_INVALID_ARG for a slot out
 * of range. */
[[nodiscard]] enum geist_status
geist_decknet_slot(const struct geist_decknet *d, size_t slot, struct geist_decknet_slot *out);

#ifdef __cplusplus
}
#endif

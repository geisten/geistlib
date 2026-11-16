/*
 * decknet: the geistbox deck-check network, a small fixed convolutional net
 * for images (#532). Two camera images in (the deck now, the empty deck),
 * one state vector per deck position out; as single images
 * (geist_decknet_run) or frame after frame from one camera
 * (geist_decknet_stream_*, #533). All declarations are EXPERIMENTAL.
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

/* ---- stream: one camera, frame after frame (#533) ----------------------- */

struct geist_decknet_stream;

/* @stability EXPERIMENTAL — how the bytes of a frame are laid out. Rows run
 * top to bottom, `stride` bytes apart.
 *   RGB8  3 bytes per pixel, R G B.
 *   NV12  a plane of Y, one byte per pixel, then at byte stride * height a
 *         plane of U V pairs, one pair per 2 x 2 pixels, its rows `stride`
 *         bytes apart as well. Width and height are even.
 *   YUYV  4 bytes per 2 pixels: Y0 U Y1 V. Width is even.
 * NV12 and YUYV are read as BT.601 limited range ("video range") with the
 * chroma of a pixel taken from its block without interpolation; the exact
 * integer formula is in docs/DECKNET.md. */
enum geist_decknet_format : uint32_t {
    GEIST_DECKNET_FORMAT_RGB8 = 0,
    GEIST_DECKNET_FORMAT_NV12 = 1,
    GEIST_DECKNET_FORMAT_YUYV = 2,
};
static_assert(sizeof(enum geist_decknet_format) == sizeof(uint32_t));

/* @stability EXPERIMENTAL — one frame in the caller's memory. It is read in
 * place during the call and never kept. `n_bytes` is the extent of `data`:
 * nothing at or behind data + n_bytes is read. */
struct geist_decknet_frame {
    enum geist_decknet_format format;
    size_t                    width; /* pixels; the model's image_width, image_height */
    size_t                    height;
    size_t                    stride; /* bytes from one row to the next */
    size_t                    n_bytes;
    const uint8_t            *data;
    uint64_t                  timestamp; /* the caller's; handed back with the result */
};

/* @stability EXPERIMENTAL — the side, in pixels, of the tiles a stream
 * divides a frame into, from its top left corner; the last tile of a row or
 * a column is what is left of the frame. */
constexpr size_t GEIST_DECKNET_TILE = 16;

/* @stability EXPERIMENTAL — what the last pushed frame gave. */
struct geist_decknet_stream_result {
    struct geist_decknet_result image;     /* as geist_decknet_run reports it */
    uint64_t                    timestamp; /* of that frame */
    uint64_t                    n_frames;  /* frames processed since the stream was created */
    /* What that frame cost. Of the frame's tiles the network reads n_tiles;
     * n_tiles_changed of them were taken from this frame (all of them
     * without gating), and n_samples bilinear samples were computed. */
    size_t n_tiles;
    size_t n_tiles_changed;
    size_t n_samples;
    /* Frames handed to geist_decknet_stream_offer since the stream was
     * created that a later one replaced before a pass took them. */
    uint64_t n_dropped;
};

/* @stability EXPERIMENTAL
 * A stream on `d`, which it borrows: `d` outlives the stream, and between
 * the creation and the destruction of the stream no geist_decknet_run is
 * called on it (the stream computes in the handle's workspace). Allocates
 * everything the stream will ever need; of the later calls on it only
 * geist_decknet_stream_set_gate allocates.
 * GEIST_E_INVALID_ARG for a null argument, GEIST_E_OOM. On failure *out is
 * nullptr. */
[[nodiscard]] enum geist_status
geist_decknet_stream_create(struct geist_decknet *d, struct geist_decknet_stream **out);

/* @stability EXPERIMENTAL — nullptr is a no-op. The model handle stays. */
void geist_decknet_stream_destroy(struct geist_decknet_stream *s);

/* @stability EXPERIMENTAL
 * Temporal gating: off after the create, `on` turns it on or off.
 *
 * A gated stream keeps the bytes of the tiles the network reads. A tile of
 * a pushed frame whose bytes differ from the kept ones by at most
 * `threshold`, as the sum of absolute differences, is not taken: the kept
 * bytes stand for it, and what the pass computed from them is reused. Any
 * other tile is taken and is the kept one from then on. The bytes of a tile
 * are those of its pixels in the frame's format: R G B for RGB8, Y0 U Y1 V
 * for YUYV, for NV12 its Y bytes and the U V pairs of its 2 x 2 blocks. So
 * the sum covers the colour as well, not the luma alone: threshold 0 means
 * "the same bytes".
 *
 * The result is exact with respect to the threshold: bit for bit that of
 * geist_decknet_run on the image whose tiles are the kept ones (for NV12 and
 * YUYV converted as geist_decknet_stream_push_frame says). With threshold 0
 * that image is the frame, and the result is that of a stream without
 * gating. The first frame after this call, after
 * geist_decknet_stream_set_reference and after a frame of another format is
 * taken whole.
 *
 * A frame in which more than 7 of 8 of the tiles the network reads were
 * taken is computed as a frame of a stream without gating, from the kept
 * tiles: the same bits, at about the cost of such a frame. What was
 * computed from earlier frames is not valid after it; the next frame that
 * mostly stands still is sampled whole to fill it again.
 *
 * The first call that turns gating on allocates what it keeps (the tiles,
 * and per slot and plane what it reuses); no frame allocates. GEIST_E_OOM
 * leaves the stream as it was. GEIST_E_INVALID_ARG for a null stream. */
[[nodiscard]] enum geist_status
geist_decknet_stream_set_gate(struct geist_decknet_stream *s, bool on, uint32_t threshold);

/* @stability EXPERIMENTAL
 * How many threads a pushed frame is computed on: 1 (after the create) or 2.
 * With 2, one half of the slots is computed on a second thread of the
 * library's thread pool while geist_decknet_stream_push_frame runs; the
 * result is the same bits. Allocates nothing. GEIST_E_INVALID_ARG for a null
 * stream or another count. */
[[nodiscard]] enum geist_status
geist_decknet_stream_set_threads(struct geist_decknet_stream *s, size_t n_threads);

/* @stability EXPERIMENTAL
 * The empty deck: the image every pushed frame is compared with (`ref` of
 * geist_decknet_run), in any of the formats. Also takes over the handle's
 * homographies as they are now: after geist_decknet_set_homographies the
 * reference is set again before the next frame. GEIST_E_INVALID_ARG for a
 * null argument or a frame whose geometry is refused (see
 * geist_decknet_stream_push_frame); the stream then keeps the reference it
 * had. */
[[nodiscard]] enum geist_status
geist_decknet_stream_set_reference(struct geist_decknet_stream      *s,
                                   const struct geist_decknet_frame *ref);

/* @stability EXPERIMENTAL
 * One forward pass on `frame` against the reference; returns when it is
 * done. Only the tiles the network samples are read, in the frame's own
 * format. The result is that of geist_decknet_run on the same two images:
 * bit for bit for RGB8, and for NV12 and YUYV bit for bit that of the frame
 * converted to RGB8 by the formula in docs/DECKNET.md.
 *
 * GEIST_E_INVALID_ARG, with nothing read and the last result kept, for a
 * null argument, an unknown format, a width or height that is not the
 * model's (or is odd where the format needs it even), a stride shorter than
 * a row, or n_bytes smaller than the frame's last row ends.
 * GEIST_E_INVALID_STATE before a reference is set and after the handle's
 * homographies were replaced. Never call one stream from two threads at
 * once (geist_decknet_stream_offer is the one function that may be). */
[[nodiscard]] enum geist_status
geist_decknet_stream_push_frame(struct geist_decknet_stream      *s,
                                const struct geist_decknet_frame *frame);

/* @stability EXPERIMENTAL
 * A queue of one, for a camera that may be faster than the pass: the thread
 * that captures hands each frame over here, and the thread that computes
 * takes the newest with geist_decknet_stream_push_offered.
 *
 * Never waits and never allocates: one atomic exchange. `frame` becomes the
 * waiting frame. If one was waiting already, the pass was slower than the
 * camera: that older frame is dropped, counted
 * (geist_decknet_stream_result.n_dropped) and handed back in *dropped;
 * otherwise *dropped is nullptr.
 *
 * Who owns a waiting frame. Nothing is copied: from this call on the struct
 * `frame` points to and the bytes it describes are lent to the stream, and
 * the caller neither writes nor frees nor reuses them until the stream
 * hands that pointer back. It does so exactly once: as *dropped of a later
 * offer, or as *done of the geist_decknet_stream_push_offered that computed
 * on it. A frame that is still out when the stream is destroyed is the
 * caller's again from then on.
 *
 * May be called from any thread, also while another thread is inside any
 * other function of the stream except geist_decknet_stream_destroy. The
 * geometry of the frame is checked here, as
 * geist_decknet_stream_push_frame checks it: GEIST_E_INVALID_ARG, with
 * nothing queued and *dropped nullptr, for a frame it would refuse and for
 * a null argument. */
[[nodiscard]] enum geist_status
geist_decknet_stream_offer(struct geist_decknet_stream       *s,
                           const struct geist_decknet_frame  *frame,
                           const struct geist_decknet_frame **dropped);

/* @stability EXPERIMENTAL
 * geist_decknet_stream_push_frame on the frame that is waiting, if one is;
 * returns when the pass is done. *done is that frame, the caller's again,
 * whatever the status; nullptr, with GEIST_OK and the last result kept,
 * when none was waiting. This call does not wait for a frame: when to call
 * it (a condition variable, a pipe, the capture loop's own event) is the
 * caller's. Called from the one thread that uses the stream, like every
 * function but geist_decknet_stream_offer. Statuses as
 * geist_decknet_stream_push_frame; GEIST_E_INVALID_ARG for a null
 * argument. */
[[nodiscard]] enum geist_status
geist_decknet_stream_push_offered(struct geist_decknet_stream       *s,
                                  const struct geist_decknet_frame **done);

/* @stability EXPERIMENTAL — the result of the last pushed frame.
 * GEIST_E_INVALID_STATE, with *out zero, before the first. */
[[nodiscard]] enum geist_status
geist_decknet_stream_result(const struct geist_decknet_stream  *s,
                            struct geist_decknet_stream_result *out);

/* @stability EXPERIMENTAL — the units of slot `slot` from the last pushed
 * frame; statuses and lifetime as geist_decknet_slot. */
[[nodiscard]] enum geist_status geist_decknet_stream_slot(const struct geist_decknet_stream *s,
                                                          size_t                             slot,
                                                          struct geist_decknet_slot         *out);

#ifdef __cplusplus
}
#endif

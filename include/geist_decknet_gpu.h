/*
 * decknet on a GPU: what a backend implements to take the integer part of a
 * camera stream's pass (#533). Backend-author territory, like
 * <geist_backend.h>; a program uses geist_decknet_stream_set_backend
 * (<geist_decknet.h>). All declarations are EXPERIMENTAL.
 *
 * The split. A crop pixel is an integer (the bilinear sample of four camera
 * bytes in fixed point) and a kernel is a sum of crop pixels per distinct
 * weight; both are exact in 32 bits that wrap, in any order, so a GPU
 * computes the bits the CPU computes. Everything behind them needs F64 (the
 * gains, the 1x1 units, the pooling) and stays on the CPU. Per frame:
 *
 *   stage 0   the samples [0, n_first): the worktable strip and the floor
 *             planes, which the CPU needs first (gains, 1x1 units);
 *   stage 1   the other samples, then both kernels' sums.
 *
 * The tables are 32-bit words in host byte order, so that the same bytes
 * serve a Metal buffer and a std430 block of a Vulkan port:
 *
 *   sample    four words: the weights w0 w1 w2 (in 1/2^24: of the pixel right
 *             of (x, y), of the one below, of the one below right; the pixel
 *             itself has the rest), then x | y << 16;
 *   dst       per sample, the pixel of the pixel memory it writes;
 *   pixels    four words a pixel: R G B in 1/65536 of a byte value, and 0:
 *               (b00 << 24) + 128 + w0 (b01 - b00) + w1 (b10 - b00)
 *                                 + w2 (b11 - b00), shifted right by 8,
 *             per colour, in 32 bits that wrap;
 *   taps      per kernel: n_parts words, the taps of each part, then per tap
 *             the signed words dy dx, part by part;
 *   sums      per kernel, crop of its list by crop, pixel of its list by
 *             pixel, part by part: four words, the sum over the part's taps
 *             of the crop pixel at (y + dy, x + dx), each clamped to the
 *             crop. Kernel 0's, then kernel 1's.
 *
 * The frame is the caller's bytes without row padding (stride = the bytes
 * of a row), converted per pixel where the format is a YUV one, by the
 * integer formula of docs/DECKNET.md.
 */
#pragma once

#include <geist.h>
#include <geist_decknet.h>

#ifdef __cplusplus
extern "C" {
#endif

struct geist_backend;
struct geist_decknet_gpu; /* a backend's state for one stream */

/* @stability EXPERIMENTAL */
struct geist_decknet_gpu_sample {
    uint32_t w[3];
    uint16_t x, y; /* one word, x in its low half */
};
static_assert(sizeof(struct geist_decknet_gpu_sample) == 16);

/* @stability EXPERIMENTAL — one kernel: where it is evaluated, and its taps.
 * The arrays are read during create and not kept. */
struct geist_decknet_gpu_kernel {
    size_t          n_parts;
    size_t          n_taps;
    const int32_t  *taps; /* n_parts + 2 * n_taps words */
    size_t          n_px;
    const uint32_t *px; /* crop pixels, row-major index */
    size_t          n_crops;
    const uint32_t *crop; /* crops: crop c starts at pixel c * crop_w * crop_h */
};

/* @stability EXPERIMENTAL — the pixels [first, first + n) of the pixel
 * memory. */
struct geist_decknet_gpu_span {
    uint32_t first, n;
};

/* @stability EXPERIMENTAL — what a stream needs of the GPU; fixed for the
 * life of the state. Every count fits 32 bits. */
struct geist_decknet_gpu_plan {
    size_t                          frame_bytes; /* the largest frame without padding */
    size_t                          crop_w, crop_h;
    size_t                          n_pixels;  /* of the pixel memory */
    size_t                          n_samples; /* of the sample table and of dst */
    size_t                          n_first;   /* samples of stage 0 */
    const uint32_t                 *dst;       /* n_samples; read during create, not kept */
    struct geist_decknet_gpu_kernel kernel[2];
    /* What the CPU reads of `pixels` after the wait for stage 0 and for
     * stage 1, and nothing else of them (it reads every sum): a backend whose
     * device does not work on host memory copies these spans back. Read
     * during create, not kept. */
    size_t                               n_read[2];
    const struct geist_decknet_gpu_span *read[2];
};

/* @stability EXPERIMENTAL — host memory of the state, valid until destroy.
 * `samples` and `frame` (frame_bytes, rounded up to whole words) are the
 * caller's to write; `pixels` and `sums` are read after the wait for the
 * stage that writes them, `pixels` within the plan's spans of that stage. */
struct geist_decknet_gpu_memory {
    struct geist_decknet_gpu_sample *samples;
    uint8_t                         *frame;
    const uint32_t                  *pixels;
    const uint32_t                  *sums;
};

/* @stability EXPERIMENTAL */
struct geist_decknet_gpu_ops {
    /* Buffers and pipelines for `plan`; allocates everything a frame will
     * need. GEIST_E_UNSUPPORTED, GEIST_E_OOM, GEIST_E_BACKEND; *out is
     * nullptr then. */
    enum geist_status (*create)(struct geist_backend                *be,
                                const struct geist_decknet_gpu_plan *plan,
                                struct geist_decknet_gpu           **out,
                                struct geist_decknet_gpu_memory     *mem);
    void (*destroy)(struct geist_decknet_gpu *gpu); /* nullptr is a no-op */
    /* After `samples` was written: a backend whose memory is not shared
     * with the device uploads it here. */
    enum geist_status (*samples_written)(struct geist_decknet_gpu *gpu);
    /* Both stages for the frame in `frame`, whose rows are `stride` bytes
     * apart and whose first plane has `height` rows. Does not wait. After
     * GEIST_OK the caller calls wait(0) and then wait(1), on this thread,
     * before anything else on this state. */
    enum geist_status (*submit)(struct geist_decknet_gpu *gpu,
                                enum geist_decknet_format format,
                                enum geist_decknet_colour colour,
                                size_t                    stride,
                                size_t                    height);
    /* Returns when the stage is done and what it wrote is readable.
     * wait(1) ends the submission, whatever it returns. */
    enum geist_status (*wait)(struct geist_decknet_gpu *gpu, size_t stage);
};

#ifdef __cplusplus
}
#endif

/*
 * image_pipeline — preprocessing for the Gemma 4 vision tower.
 *
 * Mirrors transformers.Gemma4ImageProcessor (image processor only — no
 * tokenization, no chat-template logic). Pipeline:
 *
 *   (1) decode bytes to RGB uint8 (caller's job — stb_image lives in
 *       the test harness, not here, so the same path works for video
 *       frames too)
 *   (2) aspect-preserving smart-resize (see image_pipeline_plan): pick
 *       target dims that are multiples of (patch_size * pool_kernel =
 *       48) and satisfy patches <= max_soft * pool_kernel² = 2520.
 *   (3) rescale: x /= 255  (do_rescale, rescale_factor = 1/255)
 *   (4) normalize: identity  (mean=0, std=1; happens in tower as 2*(x-0.5))
 *   (5) patchify: (3, H, W) → reshape→transpose→reshape →
 *       (n_patches, 16*16*3) with (kh, kw, c) innermost
 *   (6) position ids: (x, y) pairs per patch — direct index into the
 *       learned (2, position_embedding_size=10240, hidden) table.
 *       No bilinear interpolation: Gemma 4 vision uses separable
 *       axis-wise learned pos embeds (table[0, x] + table[1, y]).
 *
 * Reference: transformers/models/gemma4/image_processing_pil_gemma4.py
 * (get_aspect_ratio_preserving_size, convert_image_to_patches).
 *
 * Phase status:
 *   P1   — header + stubs
 *   P2 ⇐ THIS — bicubic + patchify + pos ids, parity vs HF (eps <= 1e-4)
 */
#ifndef IMAGE_PIPELINE_H
#define IMAGE_PIPELINE_H

#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Largest input height or width the pipeline can carry. Not a policy
 * number: stb_image_resize2 takes int dimensions AND an int row stride,
 * and the row stride is width * 3. Anything above this narrows to a
 * negative or wrapped int on the way into the resizer, which is where a
 * caller-supplied image geometry stops being representable rather than
 * merely large. Sizes that are representable but absurd fail honestly at
 * the allocator. */
constexpr size_t IMAGE_PIPELINE_MAX_DIM = (size_t) (INT_MAX / 3);

#define VISION_PATCH_SIZE 16
#define VISION_POOL_KERNEL 3
#define VISION_MAX_PATCHES (VISION_PATCH_SIZE * VISION_PATCH_SIZE) /* sanity bound */

/* Plan for one image. Filled by image_pipeline_plan(). Mirrors HF's
 * get_aspect_ratio_preserving_size — target dims are the largest that:
 *   (1) produce <= max_patches = max_soft * pool_kernel² patches,
 *   (2) have height and width divisible by pool_kernel * patch_size = 48.
 * Inputs in pixels (in_h, in_w); outputs in pixels (resized_h/w), patches
 * (grid_h/w), and pooled tokens (pool_h/w). */
struct image_plan {
    size_t in_h, in_w;           /* native input dims (pixels) */
    size_t resized_h, resized_w; /* multiples of 48 (= grid_*  * 16) */
    size_t grid_h, grid_w;       /* patch grid = resized_* / 16 */
    size_t pool_h, pool_w;       /* pooled grid = grid_*  / 3   */
    size_t soft_tokens;          /* pool_h * pool_w (always <= max_soft) */
};

/* Compute a plan. max_soft must be one of {70, 140, 280, 560, 1120}
 * (HF Gemma4 constraint; default 280). Returns false if the image is
 * degenerate (one dim rounds to zero with the other already at max). */
bool image_pipeline_plan(size_t in_h, size_t in_w, size_t max_soft, struct image_plan *out);

/* Preprocess RGB uint8 → fp32 patches.
 *   rgb_in:      (in_h, in_w, 3) row-major uint8
 *   plan:        from image_pipeline_plan
 *   out_patches: (grid_h * grid_w, 16 * 16 * 3) fp32 row-major.
 *                Patches in row-major (patch_y, patch_x) order.
 *                Inside each patch: (kh, kw, c) with c innermost,
 *                values rescaled by /255 (range [0,1]).
 *
 * Pipeline: bicubic resize via stb_image_resize2 (Catmull-Rom, matches
 * PIL bicubic) → /255 → CHW-equivalent patchify (yields HWC inside
 * each patch as HF's reshape+permute does).
 * Returns false on bad input. */
bool image_pipeline_preprocess(const uint8_t           *rgb_in,
                               const struct image_plan *plan,
                               float                   *out_patches);

/* Fill (x, y) int32 position-id pairs in row-major patch order matching
 * HF np.meshgrid(arange(pw), arange(ph), indexing='xy'):
 *   for patch_y in 0..grid_h-1:
 *     for patch_x in 0..grid_w-1:
 *       out_pos[i*2 + 0] = patch_x
 *       out_pos[i*2 + 1] = patch_y
 * These (x, y) pairs index the learned (2, position_embedding_size,
 * hidden) pos-embed table directly. */
void image_pipeline_position_ids(const struct image_plan *plan, int32_t *out_pos);

#endif

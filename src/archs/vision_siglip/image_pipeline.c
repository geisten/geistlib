/*
 * image_pipeline — P2 impl. Matches transformers.models.gemma4
 *   .image_processing_pil_gemma4.Gemma4ImageProcessorPil.
 *
 * Resize: stb_image_resize2 with STBIR_FILTER_CATMULLROM + STBIR_EDGE_CLAMP,
 *         which matches PIL.Image.BICUBIC (Keys cubic, a=-0.5) for both
 *         up- and downscale.
 *
 * Patchify: matches HF's
 *     image.reshape(C, H/16, 16, W/16, 16).transpose(1,3,2,4,0).reshape(N, -1)
 * but we operate on HWC uint8 directly, since the final patch layout is
 * (kh, kw, c) innermost — the same bytes you'd get by traversing the
 * input HWC image in (patch_y, patch_x, kh, kw, c) order.
 *
 * Position IDs: (x, y) row-major matching
 *     np.meshgrid(arange(grid_w), arange(grid_h), indexing='xy') →
 *     stack → reshape(n, 2).
 */
#include "image_pipeline.h"

#include "checked.h"
#include "heap.h"

#include "stb_image_resize2.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* Mirrors HF get_aspect_ratio_preserving_size. Returns false only on the
 * pathological 0×0 case (which HF raises on). */
bool image_pipeline_plan(size_t in_h, size_t in_w, size_t max_soft, struct image_plan *out) {
    if (out == nullptr || in_h == 0 || in_w == 0 || max_soft == 0) {
        return false;
    }
    /* Geometry comes from the caller of geist_session_attach_image; bound
     * it here, at the entry to the only code that turns it into byte
     * counts and int strides. */
    if (in_h > IMAGE_PIPELINE_MAX_DIM || in_w > IMAGE_PIPELINE_MAX_DIM) {
        return false;
    }

    const size_t patch       = VISION_PATCH_SIZE;  /* 16 */
    const size_t pool_k      = VISION_POOL_KERNEL; /* 3  */
    const size_t side_mult   = patch * pool_k;     /* 48 */
    const size_t max_patches = max_soft * pool_k * pool_k;
    const double total_px    = (double) in_h * (double) in_w;
    const double target_px   = (double) max_patches * (double) (patch * patch);
    const double factor      = sqrt(target_px / total_px);

    /* Snap the unit count to an integer when it's within ~1 ULP of a
     * whole number — Pi 5 builds use -ffast-math which can make
     * sqrt(9.0) = 2.99999... instead of exactly 3.0, which would
     * floor a 14.0 result to 13. Mac (Accelerate sqrt) and gcc strict
     * math both yield exact 14.0; this tolerance keeps the planner
     * grid stable across compiler flags. */
    const double units_h_d = (factor * (double) in_h) / (double) side_mult;
    const double units_w_d = (factor * (double) in_w) / (double) side_mult;
    size_t       units_h   = (size_t) units_h_d;
    size_t       units_w   = (size_t) units_w_d;
    if (units_h_d - (double) units_h > 1.0 - 1e-6)
        units_h++;
    if (units_w_d - (double) units_w > 1.0 - 1e-6)
        units_w++;
    size_t target_h = units_h * side_mult;
    size_t target_w = units_w * side_mult;

    const size_t max_side_length = (max_patches / (pool_k * pool_k)) * side_mult;

    if (target_h == 0 && target_w == 0) {
        return false;
    }
    if (target_h == 0) {
        target_h       = side_mult;
        size_t ratio_w = ((size_t) ((double) in_w / (double) in_h)) * side_mult;
        target_w       = ratio_w < max_side_length ? ratio_w : max_side_length;
    } else if (target_w == 0) {
        target_w       = side_mult;
        size_t ratio_h = ((size_t) ((double) in_h / (double) in_w)) * side_mult;
        target_h       = ratio_h < max_side_length ? ratio_h : max_side_length;
    }

    if (target_h * target_w > max_patches * patch * patch) {
        /* HF raises here; we treat as invalid plan. */
        return false;
    }

    *out = (struct image_plan) {
            .in_h      = in_h,
            .in_w      = in_w,
            .resized_h = target_h,
            .resized_w = target_w,
            .grid_h    = target_h / patch,
            .grid_w    = target_w / patch,
            .pool_h    = (target_h / patch) / pool_k,
            .pool_w    = (target_w / patch) / pool_k,
    };
    out->soft_tokens = out->pool_h * out->pool_w;
    return true;
}

bool image_pipeline_preprocess(const uint8_t           *rgb_in,
                               const struct image_plan *plan,
                               float                   *out_patches) {
    if (rgb_in == nullptr || plan == nullptr || out_patches == nullptr) {
        return false;
    }
    if (plan->resized_h == 0 || plan->resized_w == 0) {
        return false;
    }

    const size_t in_h     = plan->in_h;
    const size_t in_w     = plan->in_w;
    const size_t out_h    = plan->resized_h;
    const size_t out_w    = plan->resized_w;
    const size_t patch    = VISION_PATCH_SIZE;
    const size_t patch_px = patch * patch * 3;

    uint8_t       *resized = nullptr;
    const uint8_t *src;
    if (in_h == out_h && in_w == out_w) {
        src = rgb_in;
    } else {
        /* out_h/out_w are planner output, but the planner's inputs were
         * the caller's: check the product rather than assume it. */
        size_t resized_px    = 0;
        size_t resized_bytes = 0;
        if (ckd_mul(&resized_px, out_h, out_w) || ckd_mul(&resized_bytes, resized_px, 3u) ||
            out_h > IMAGE_PIPELINE_MAX_DIM || out_w > IMAGE_PIPELINE_MAX_DIM) {
            return false;
        }
        resized = heap_alloc_array_aligned(uint8_t, resized_bytes);
        if (resized == nullptr) {
            return false;
        }
        void *r = stbir_resize(rgb_in,
                               (int) in_w,
                               (int) in_h,
                               (int) (in_w * 3),
                               resized,
                               (int) out_w,
                               (int) out_h,
                               (int) (out_w * 3),
                               STBIR_RGB,
                               STBIR_TYPE_UINT8,
                               STBIR_EDGE_CLAMP,
                               STBIR_FILTER_CATMULLROM);
        if (r == nullptr) {
            safe_free((void **) &resized);
            return false;
        }
        src = resized;
    }

    /* Patchify HWC uint8 → (n_patches, kh*16 + kw*3 + c) fp32, /255.
     * Patch index i = patch_y * grid_w + patch_x. */
    const float  inv255     = 1.0f / 255.0f;
    const size_t grid_w     = plan->grid_w;
    const size_t row_stride = out_w * 3;

    for (size_t py = 0; py < plan->grid_h; py++) {
        for (size_t px = 0; px < grid_w; px++) {
            const size_t   patch_idx = py * grid_w + px;
            float         *dst       = out_patches + patch_idx * patch_px;
            const uint8_t *base      = src + (py * patch) * row_stride + (px * patch) * 3;
            for (size_t kh = 0; kh < patch; kh++) {
                const uint8_t *row = base + kh * row_stride;
                for (size_t kw = 0; kw < patch; kw++) {
                    const uint8_t *px3 = row + kw * 3;
                    float         *o   = dst + kh * patch * 3 + kw * 3;
                    o[0]               = (float) px3[0] * inv255;
                    o[1]               = (float) px3[1] * inv255;
                    o[2]               = (float) px3[2] * inv255;
                }
            }
        }
    }

    if (resized != nullptr) {
        safe_free((void **) &resized);
    }
    return true;
}

void image_pipeline_position_ids(const struct image_plan *plan, int32_t *out_pos) {
    if (plan == nullptr || out_pos == nullptr) {
        return;
    }
    /* HF: meshgrid(arange(pw), arange(ph), indexing='xy') then stack on
     * last dim → reshape (n, 2). With xy indexing, the inner loop is x
     * (column), outer loop is y (row), and values are (x, y) per patch. */
    const size_t grid_h = plan->grid_h;
    const size_t grid_w = plan->grid_w;
    for (size_t y = 0; y < grid_h; y++) {
        for (size_t x = 0; x < grid_w; x++) {
            const size_t i     = y * grid_w + x;
            out_pos[2 * i + 0] = (int32_t) x;
            out_pos[2 * i + 1] = (int32_t) y;
        }
    }
}

/*
 * vision_encoder — Gemma 4 vision tower (SigLIP-derived ViT).
 *
 * Public surface for the per-image / per-video forward pass. Owns the
 * mmap'd weights from vision_tower.safetensors. Stateless across calls
 * (no recurrence, no per-session state).
 *
 * Tower spec (from gemma-4-E2B-it config.json):
 *   16 layers, hidden=768, heads=12, head_dim=64, intermediate=3072,
 *   GELU-tanh activation, RMSNorm (eps=1e-6), 2D RoPE theta=100,
 *   patch_size=16, pooling_kernel_size=3, use_clipped_linears=true,
 *   position_embedding_size=10240 (learned, interpolated to grid).
 *   Projector: 768 → LM_hidden (2048 for Gemma 4 E2B), 280 soft tokens.
 *
 * Phase status:
 *   P1 ⇐ THIS — skeleton: open/close, stub forward returning 0
 *   P2        — preprocessing parity (bicubic, patchify, pos-embed interp)
 *   P3        — per-block tower parity
 *   P4        — pool + projector parity (end-to-end soft tokens)
 *   P6        — batched video path
 */
#ifndef VISION_ENCODER_H
#define VISION_ENCODER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

/* E2B soft-token dim (text-LM hidden, the projector's row count). The
 * per-checkpoint width — 2560 on E4B — comes from
 * vision_encoder_soft_dim(); size output buffers with that (#258). */
#define VISION_SOFT_TOKEN_DIM 1536
#define VISION_TOWER_HIDDEN 768 /* per-block ViT hidden_size */

/* Soft tokens per image (after kernel-3 avg-pool). */
#define VISION_SOFT_TOKENS_PER_IMAGE 280

/* Soft tokens per video frame (smaller pool to fit 32 frames in context). */
#define VISION_SOFT_TOKENS_PER_VIDEO_FRAME 70

struct VisionEncoder;

struct VisionEncoder *vision_encoder_create(const char *safetensors_path);
void                  vision_encoder_destroy(struct VisionEncoder *);

/* Soft-token width of THIS tower checkpoint (embedding_projection rows):
 * 1536 for E2B, 2560 for E4B. Size output buffers with this. */
size_t vision_encoder_soft_dim(const struct VisionEncoder *);

/* Full image-tower forward: RGB uint8 → patchify → ViT → pool → projector.
 *   rgb:     (height, width, 3) row-major uint8
 *   out:     soft tokens, (n_out, VISION_SOFT_TOKEN_DIM) fp32
 * Returns n_out (≤ VISION_SOFT_TOKENS_PER_IMAGE), or 0 on error.
 *
 * P3: returns 0 (only the tower forward is implemented; the pooler and
 * projector land in P4). For per-block parity testing, call
 * vision_encoder_run_tower() directly. */
size_t vision_encoder_run_image(
        const struct VisionEncoder *, size_t height, size_t width, const uint8_t *rgb, float *out);

/* Batched video-tower forward. */
size_t vision_encoder_run_video(const struct VisionEncoder *,
                                size_t         n_frames,
                                size_t         height,
                                size_t         width,
                                const uint8_t *frames,
                                float         *out);

/* P3 entry point: full tower forward on pre-processed patches.
 *   patches_in:   (n_patches, 16*16*3 = 768) fp32 — output of
 *                 image_pipeline_preprocess (values in [0, 1]).
 *   positions:    (n_patches, 2) int32 (x, y) — output of
 *                 image_pipeline_position_ids.
 *   n_patches:    number of patches.
 *   hidden_out:   (n_patches, VISION_TOWER_HIDDEN = 768) fp32.
 *                 Encoder last_hidden_state (no final norm — matches HF).
 * Returns 0 on error, 1 on success.
 *
 * If env var GEIST_VISION_DUMP_DIR is set to a directory, the function
 * also writes per-layer intermediate buffers to that dir as
 *   patch_embed_out.bin
 *   layer00.bin ... layer15.bin
 * (raw fp32 row-major, same layout as the HF parity dump). Used by the
 * P3 unit test to bisect parity failures. */
bool vision_encoder_run_tower(const struct VisionEncoder *,
                              const float   *patches_in,
                              const int32_t *positions,
                              size_t         n_patches,
                              float         *hidden_out);

#endif

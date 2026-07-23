/*
 * src/archs/vision_siglip/arch.h — Gemma 4 vision tower (SigLIP-derived).
 *
 * Layer: ARCHITECTURE. Implements the geist_arch_ops_vision vtable for
 * the Gemma 4 vision tower (16-layer ViT, RMSNorm, GELU-tanh, 2D RoPE
 * theta=100, kernel-3 avg-pool projector, 280 soft tokens per image).
 *
 * The vtable shape itself lives in <geist_arch.h> (engine-owned interface);
 * this header only exports the concrete descriptor.
 *
 * Defined in (Phase P1 skeleton, fleshed out P2-P8):
 *   src/archs/vision_siglip/arch.c            — descriptor, encode entry
 *   src/archs/vision_siglip/vision_encoder.c  — tower forward + weight load
 *   src/archs/vision_siglip/vision_kernels.c  — patch-embed, pool, 2D RoPE
 *   src/archs/vision_siglip/image_pipeline.c  — bicubic resize, patchify
 */
#ifndef GEIST_INTERNAL_ARCH_VISION_SIGLIP_H
#define GEIST_INTERNAL_ARCH_VISION_SIGLIP_H

#ifndef GEIST_INTERNAL_ARCH_LAYER
#error "vision_siglip/arch.h is internal to the architecture layer."
#endif

#include <geist.h>
#include <geist_arch.h>

/* Concrete descriptor for the Gemma 4 vision tower. */
extern const struct geist_arch_ops_vision geist_arch_vision_siglip;

#endif /* GEIST_INTERNAL_ARCH_VISION_SIGLIP_H */

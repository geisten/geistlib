/*
 * src/backends/vulkan/shaders/vk_limits.h — constants the host code and the
 * shaders must agree on (#474). Included by ops.c (through vk_internal.h) and
 * by the .comp files, so it holds nothing but object-like #defines and
 * comments: the subset of the preprocessor C and GLSL share. Every value is
 * an unsigned literal, which both languages accept in expressions, array
 * sizes and layout(local_size_x = ...).
 *
 * Changing a value here changes the SPIR-V: run `make vulkan-shaders`.
 *
 * Layer: BACKEND (vulkan, internal).
 */
#ifndef GEIST_VK_LIMITS_H
#define GEIST_VK_LIMITS_H

/* embed_lookup_scaled.comp: push.dtype codes (ops.c's vk_dtypes[].embed_code). */
#define VK_EMBED_DT_F32 0u
#define VK_EMBED_DT_F16 1u
#define VK_EMBED_DT_BF16 2u
#define VK_EMBED_DT_Q4_0 3u
#define VK_EMBED_DT_Q4_1 4u
#define VK_EMBED_DT_Q8_0 5u
#define VK_EMBED_DT_Q4_K 8u
#define VK_EMBED_DT_Q5_K 9u
#define VK_EMBED_DT_Q6_K 10u
#define VK_EMBED_DT_PQ2_0 11u

/* Output rows per workgroup of the linear kernels (vk_linear_gx/gy).
 * Matvecs (matvec_q4k/q5k/q6k/tq2_0, mv_legacy.glsl, ffn_norm_gate_up_q4k): */
#define VK_MV_ROWS_PER_WG 8u
/* matvec_pq2_0: one row per lane of a 32-lane slice. */
#define VK_MV_PQ2_0_ROWS_PER_WG 32u
/* ffn_gate_up_gelu_q4k: NUM_ROWS rows in one 32-lane workgroup. */
#define VK_FFN_GU_ROWS_PER_WG 4u
/* Register-tiled GEMMs: one output row per 32-lane subgroup, so the rows are
 * local_size_x / 32; the batch (token) rows per workgroup are the tile height.
 * mm_legacy.glsl and matmul_q4k: */
#define VK_MM_ROWS_PER_WG 8u
#define VK_MM_TOKENS_PER_WG 32u
/* matmul_q6k and matmul_f32: */
#define VK_MM_SMALL_ROWS_PER_WG 4u
#define VK_MM_SMALL_TOKENS_PER_WG 16u

/* hadamard_f32: the block is staged in shared memory. */
#define VK_HADAMARD_MAX_BLOCK 1024u

/* qkv_prep_f{16,32}: one head row in shared memory. */
#define VK_QKV_PREP_MAX_HEAD_DIM 512u

/* attention_f{16,32} and attn_part_f16: q staged in shared memory. */
#define VK_ATTN_MAX_HEAD_DIM 512u
/* attn_part_f16{,_g2}: lanes per workgroup (>= VK_ATTN_MAX_HEAD_DIM / 4: a
 * lane per 4-wide dim slice), the largest tile of key positions it scores at
 * once, and the most q-heads of a GQA group one workgroup serves (q staged
 * as MAX_GROUP x MAX_HEAD_DIM floats; the _g2 variant serves up to 2). */
#define VK_ATTN_PART_WG 128u
#define VK_ATTN_PART_TILE_MAX 64u
#define VK_ATTN_PART_MAX_GROUP 4u
/* attn_part_f16 dispatch size the host aims for (ops.c). */
#define VK_ATTN_PART_TARGET_WG 256u
/* attn_comb: lanes per workgroup, 4-wide dim slices per workgroup (y blocks
 * cover the head) and the spans whose merge factors it caches in shared
 * memory (more are recomputed). */
#define VK_ATTN_COMB_WG 256u
#define VK_ATTN_COMB_SLICES 32u
#define VK_ATTN_COMB_MAX_FAC 2048u

/* deltanet_delta_f32: q/k staged in shared memory (d_k), one lane per value
 * column (d_v). deltanet_conv_f32: the tap window lives in registers. */
#define VK_DN_MAX_DK 256u
#define VK_DN_MAX_DV 128u
#define VK_DN_MAX_CONV_K 8u
#define VK_DN_CONV_WG 128u
/* deltanet_scan_f32 (the fast path, #467): LANES adjacent lanes own one
 * value column with ROWS state rows each, so it takes d_k <= LANES * ROWS;
 * a workgroup covers WG / LANES columns. LANES is the subgroupClusteredAdd
 * cluster, so the device's subgroups must be at least that wide. 8 x 16
 * measured best on the RTX 2080 Ti (4 x 32 and 16 x 8 were 8 % and 19 %
 * slower). deltanet_norm_f32: one workgroup per head vector of up to
 * VK_DN_MAX_DK elements. */
#define VK_DN_SCAN_WG 128u
#define VK_DN_SCAN_LANES 8u
#define VK_DN_SCAN_ROWS 16u
#define VK_DN_NORM_WG 128u

#endif /* GEIST_VK_LIMITS_H */

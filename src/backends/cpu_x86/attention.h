/*
 * src/backends/cpu_x86/attention.h — cpu_x86 attention override.
 *
 * Layer: BACKEND (cpu_x86, internal). See attention.c.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_X86_ATTENTION_H
#define GEIST_INTERNAL_BACKEND_CPU_X86_ATTENTION_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_x86/attention.h is internal to the backend layer."
#endif

#include <geist.h>
#include <geist_backend.h>

#include <stddef.h>
#include <stdint.h>

[[nodiscard]] enum geist_status cpu_x86_attention(struct geist_backend      *be,
                                                  const struct geist_tensor *q,
                                                  const struct geist_tensor *k,
                                                  const struct geist_tensor *v,
                                                  size_t                     q_offset,
                                                  size_t                     sliding_window,
                                                  struct geist_tensor       *out);

/* ---- attention_int8.c: attention over the INT8 KV cache ---------------- */

/* fused->attention_kv_int8 (see geist_attention_kv_int8_args). */
[[nodiscard]] enum geist_status
cpu_x86_attention_kv_int8(struct geist_backend                      *be,
                          const struct geist_attention_kv_int8_args *args);

/* The probe's answer for GEIST_FUSED_ATTN_KV_INT8: any m, head_dim up to
 * 512, query heads a multiple of the KV heads. */
[[nodiscard]] bool cpu_x86_attention_kv_int8_supported(const struct geist_fusion_query *q);

/* Floats of partial results a split decode needs (0 < n_q_heads). */
[[nodiscard]] size_t cpu_x86_attention_kv_int8_part_floats(size_t n_q_heads, size_t head_dim);

/* The AVX2 kernel on host pointers, validated by the caller: q and out
 * [n_q, n_q_heads, head_dim], k and v [n_kv, n_kv_heads, head_dim], the
 * scales [n_kv, n_kv_heads]; `part` holds part_floats floats for a split
 * decode, or is nullptr (no split). */
void cpu_x86_attention_kv_int8_run(size_t        n_q,
                                   size_t        n_q_heads,
                                   size_t        head_dim,
                                   size_t        n_kv,
                                   size_t        n_kv_heads,
                                   size_t        part_floats,
                                   size_t        q_offset,
                                   size_t        sliding_window,
                                   const float  *q,
                                   const int8_t *k,
                                   const float  *k_scale,
                                   const int8_t *v,
                                   const float  *v_scale,
                                   float        *out,
                                   float        *part);

/* The same on AVX-512 VNNI (attention_int8_avx512_vnni.c): callable only
 * where avx512f, bw, dq, vl and vnni are all present. */
void cpu_x86_attention_kv_int8_run_avx512_vnni(size_t        n_q,
                                               size_t        n_q_heads,
                                               size_t        head_dim,
                                               size_t        n_kv,
                                               size_t        n_kv_heads,
                                               size_t        part_floats,
                                               size_t        q_offset,
                                               size_t        sliding_window,
                                               const float  *q,
                                               const int8_t *k,
                                               const float  *k_scale,
                                               const int8_t *v,
                                               const float  *v_scale,
                                               float        *out,
                                               float        *part);

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_ATTENTION_H */

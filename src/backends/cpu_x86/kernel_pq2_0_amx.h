/*
 * src/backends/cpu_x86/kernel_pq2_0_amx.h — PQ2_0 prefill GEMM, AMX-INT8.
 *
 * Layer: BACKEND (cpu_x86, internal). See kernel_pq2_0_amx.c.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_PQ2_0_AMX_H
#define GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_PQ2_0_AMX_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_x86/kernel_pq2_0_amx.h is internal to the backend layer."
#endif

#include <stddef.h>
#include <stdint.h>

/* Weight rows per group and tokens per tile: one C tile is 16 x 16. */
constexpr size_t PQ2_0_AMX_ROWS   = 16;
constexpr size_t PQ2_0_AMX_TOKENS = 16;

/* Packed activations of one 128-element block and one token tile: two B
 * tiles (the block's halves) of 16 rows x 64 bytes. */
constexpr size_t PQ2_0_AMX_TILE_BYTES = 2048;

/* Scratch per thread: PQ2_0_AMX_SCRATCH_FIXED bytes plus
 * PQ2_0_AMX_SCRATCH_PER_TILE per token tile, 64-byte aligned. */
constexpr size_t PQ2_0_AMX_SCRATCH_FIXED    = 8192;
constexpr size_t PQ2_0_AMX_SCRATCH_PER_TILE = 1024;

/* One block of n_tok quantized token rows (128 bytes each, contiguous, in
 * code order: linear_pq2_0.c's XQ layout) into tn token tiles at bt, tn *
 * PQ2_0_AMX_TILE_BYTES bytes. Rows n_tok .. 16 tn - 1 are zero.
 * 1 <= n_tok <= 16 tn. */
void pq2_0_amx_pack(size_t tn, size_t n_tok, const int8_t *xq, int32_t *bt);

/* Weight rows [16 g0, min(16 g1, n_out)) of y = W x for m tokens:
 *
 *   y[t * n_out + r] = inv[t] * sum_b d[r,b] * sum_k (code[r,b,k] - 1) * xq[t,b,k]
 *
 * w is the PQ2_0 weight as stored (n_out rows of nb blocks); bt holds the m
 * tokens packed block by block, tn = ceil(m / 16) tiles each
 * (pq2_0_amx_pack); scratch is 64-byte aligned, PQ2_0_AMX_SCRATCH_FIXED +
 * tn * PQ2_0_AMX_SCRATCH_PER_TILE bytes, private to the caller. Extents are
 * products over runtime dimensions, hence plain pointers.
 *
 * Only call on a host linear_pq2_0.c has cleared for AMX (cpuid, dispatcher
 * tier and the kernel's permission for the tile data): this TU is compiled
 * with -mamx-* and -mavx512*. */
void pq2_0_amx_gemm(size_t         nb,
                    size_t         n_out,
                    size_t         m,
                    size_t         tn,
                    size_t         g0,
                    size_t         g1,
                    const uint8_t *w,
                    const int32_t *bt,
                    const float   *inv,
                    uint8_t       *scratch,
                    float         *y);

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_PQ2_0_AMX_H */

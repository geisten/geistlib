/*
 * src/backends/cpu_x86/kernel_q4k_tiled.h — tiled int8 Q4_K GEMM for
 * prefill on AVX-512 VNNI (#18).
 *
 * Layer: BACKEND (cpu_x86).
 *
 * Y[M, N] = X[M, K] · W[N, K]ᵀ with W in the Q4_Kx8 repack and X quantized
 * to Q8_Kx4 (the same operands q4kx8_gemm_avx512 takes, so the integer
 * products are identical; only the float accumulation order differs).
 *
 * Per thread, windows of up to 256 weight rows × 256 activation rows are
 * unpacked one 256-element K block at a time into int8 tiles, and an 8x16
 * VPDPBUSD microkernel accumulates exact int32 subblock sums before the
 * scales are applied. Port of llama.cpp's tiled k-quant matmul
 * (ggml/src/ggml-cpu/tiled/, d834d44e6, MIT; see the .c file).
 */
#ifndef GEIST_CPU_X86_KERNEL_Q4K_TILED_H
#define GEIST_CPU_X86_KERNEL_Q4K_TILED_H

#include "backend_state.h"
#include "q4k_to_q4kx8.h"
#include "q8_kx4.h"

#include <stddef.h>

/* One thread's tile workspace in bytes (weight tile, activation tile,
 * 256x256 float accumulator). */
[[nodiscard]] size_t q4k_tiled_ws_bytes(void);

/* Y[M, N] = X · Wᵀ. M is a multiple of 4 (whole Q8_Kx4 groups), N a
 * multiple of 8 (whole Q4_Kx8 octets), K a multiple of 256. Runs on
 * geist_par_for; each thread takes its tile workspace from st. Returns
 * false, with Y unspecified, when a workspace cannot be allocated; the
 * caller then computes Y another way. The caller has checked
 * vnni_tiles_usable(). */
[[nodiscard]] bool q4k_tiled_gemm_avx512_vnni(size_t                     M,
                                              size_t                     N,
                                              size_t                     K,
                                              const struct block_q8_Kx4 *X,
                                              const struct block_q4_Kx8 *W,
                                              float                     *Y,
                                              struct cpu_x86_state      *st);

#endif /* GEIST_CPU_X86_KERNEL_Q4K_TILED_H */

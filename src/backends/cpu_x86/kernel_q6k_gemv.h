/*
 * src/backends/cpu_x86/kernel_q6k_gemv.h — native Q6_K decode GEMV.
 *
 * Layer: BACKEND (cpu_x86, internal). See kernel_q6k_gemv.c.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_Q6K_GEMV_H
#define GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_Q6K_GEMV_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_x86/kernel_q6k_gemv.h is internal to the backend layer."
#endif

#include <stddef.h>
#include <stdint.h>

/* Decode (M=1) GEMV over native Q6_K weights. q6k_raw is the original Q6_K
 * blob (N rows × K/256 block_q6_K_t); x is the fp32 activation row (length
 * K); y is the fp32 output (length N). K % 256 == 0, any length. AVX2. */
void q6k_gemv_m1(size_t N, size_t K, const float *x, const uint8_t *q6k_raw, float y[static N]);

/* Prefill (M>1) over the same native Q6_K weights: y[M][N] = x[M][K] W^T.
 * scratch holds the M quantized activation rows,
 * q6k_gemm_scratch_bytes(M, K) bytes, 4-byte aligned. Bit-identical per
 * output to q6k_gemv_m1 for K <= 16384. AVX2. */
size_t q6k_gemm_scratch_bytes(size_t M, size_t K);
void   q6k_gemm(size_t         M,
                size_t         N,
                size_t         K,
                const float   *x,
                const uint8_t *q6k_raw,
                void          *scratch,
                float         *y);

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_Q6K_GEMV_H */

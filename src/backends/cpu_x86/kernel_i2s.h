/*
 * src/backends/cpu_x86/kernel_i2s.h — BitNet b1.58 I2_S ternary GEMV.
 *
 * Layer: BACKEND (cpu_x86, internal).
 *
 * I2_S is Microsoft's official BitNet-2B-4T weight format: ternary weights
 * {-1,0,+1} packed 2 bits each (256-elem / 64-byte blocks), with ONE fp32
 * per-TENSOR scale at the tail (offset n_in*n_out/4). Within each byte the
 * four 2-bit fields are in REVERSE order: element 32*g+bb sits at shift
 * (6-2g); the two 32-byte halves cover elements [0,128) and [128,256).
 *
 * --- Algebra ----------------------------------------------------------------
 *
 *   trit = code - 1,  code ∈ {0,1,2}  (the stored 2-bit value)
 *   dot  = Σ_i trit_i * a_i  =  Σ_i code_i * a_i  −  Σ_i a_i
 *
 * So a biased u8×s8 VPDPBUSD over the raw codes, minus the per-token
 * activation sum (a single scalar, since there is no per-block scale),
 * gives the ternary dot. The per-tensor fp32 scale and the per-row
 * activation scale fold in once at the very end:
 *
 *   y[r] = tensor_scale * (1/act_scale) * ( Σ code·a  −  Σ a )
 *
 * Zen 5 has avx512_vnni (u8×s8 VPDPBUSD) but NOT avx_vnni_int8 (s8×s8),
 * which is exactly why the biased-u8 formulation is used.
 *
 * The hot path is allocation-free apart from a per-call activation scratch
 * (int8 quants in the VPDPBUSD-pairing permutation). Caller owns `y`.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_I2S_H
#define GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_I2S_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_x86/kernel_i2s.h is internal to the backend layer."
#endif

#include <stddef.h>
#include <stdint.h>

constexpr size_t I2S_BLOCK_ELEMS = 256;
constexpr size_t I2S_BLOCK_BYTES = 64;

/* Decode (M=1) ternary GEMV. n_in % 256 == 0. w_raw points at the packed
 * weight bytes (n_out rows × n_in/4 bytes); tensor_scale is the single
 * fp32 scale the caller read from w_raw + n_in*n_out/4. Dispatches to the
 * AVX-512+VNNI path when available, else the scalar reference. */
void i2s_gemv_m1(size_t        n_out,
                 size_t        n_in,
                 const float  *x,
                 const uint8_t w_raw[],
                 float         tensor_scale,
                 float         y[static n_out]);

/* Scalar reference (the oracle): same int8-quantized math the VNNI path
 * implements. Always available regardless of host ISA. */
void i2s_gemv_m1_scalar(size_t        n_out,
                        size_t        n_in,
                        const float  *x,
                        const uint8_t w_raw[],
                        float         tensor_scale,
                        float         y[static n_out]);

/* True iff i2s_gemv_m1 resolved to the AVX-512+VNNI tier. */
[[nodiscard]] int i2s_isa_is_vnni(void);

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_I2S_H */

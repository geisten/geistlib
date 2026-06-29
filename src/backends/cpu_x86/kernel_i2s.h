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

/* Prefill GEMM: M token rows × n_out output rows. x is [M, n_in] row-major;
 * y is [M, n_out] row-major (y[i*n_out + r]). Each weight row is read once
 * and reused across a token-tile (VPDPBUSD amortization). Dispatches to
 * AVX-512+VNNI when available, else the scalar reference. */
void i2s_gemm_mN(size_t        m,
                 size_t        n_out,
                 size_t        n_in,
                 const float  *x,
                 const uint8_t w_raw[],
                 float         tensor_scale,
                 float         y[]);

void i2s_gemm_mN_scalar(size_t        m,
                        size_t        n_out,
                        size_t        n_in,
                        const float  *x,
                        const uint8_t w_raw[],
                        float         tensor_scale,
                        float         y[]);

/* True iff i2s_gemv_m1 resolved to the AVX-512+VNNI tier. */
[[nodiscard]] int i2s_isa_is_vnni(void);

/* --- x4 row-interleaved layout (the fast path) ----------------------------
 *
 * Repack the native I2_S weights so 4 output rows are interleaved at 2-bit
 * granularity within each byte and columns are in natural order:
 *   x4[g*n_in + c] = code(4g+0,c)<<6 | code(4g+1,c)<<4
 *                  | code(4g+2,c)<<2 | code(4g+3,c)
 * for row-group g, column c. Still 0.25 B/wt (no expansion). A single
 * activation load then feeds 4 output rows (4× fewer act loads), and no
 * activation permute is needed. Requires n_out % 4 == 0 and n_in % 64 == 0.
 * Run once at model load; `x4` is n_out*n_in/4 bytes, caller-owned. */
void i2s_to_x4(size_t n_out, size_t n_in, const uint8_t w_raw[], uint8_t x4[]);

/* x4 decode GEMV (M=1) and prefill GEMM. `x4` is the i2s_to_x4 output;
 * `tensor_scale` is the per-tensor fp32 scale. AVX-512+VNNI only (caller
 * gates on i2s_isa_is_vnni() + the divisibility constraints). */
void i2s_x4_gemv_m1(size_t        n_out,
                    size_t        n_in,
                    const float  *x,
                    const uint8_t x4[],
                    float         tensor_scale,
                    float         y[static n_out]);

void i2s_x4_gemm_mN(size_t        m,
                    size_t        n_out,
                    size_t        n_in,
                    const float  *x,
                    const uint8_t x4[],
                    float         tensor_scale,
                    float         y[]);

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_I2S_H */

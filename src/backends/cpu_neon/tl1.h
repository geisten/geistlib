/*
 * src/backends/cpu_neon/tl1.h — W1.58 × A8 LUT-GEMV decode.
 *
 * Layer: BACKEND (cpu_neon, internal). NOT part of the public ABI.
 *
 * TL1 ("Table Lookup 1") is bitnet.cpp's Mac/ARM decode kernel for
 * ternary-quantized weights. It exchanges fp32 FMAs for int8 table
 * lookups (vqtbl1q_s8): instead of computing w·x as Σ trit_i * x_i,
 * pre-quantize x to int8, then precompute, per K-pair (a, b) of int8
 * activations, a 16-entry int8 LUT enumerating w0·a + w1·b for all 9
 * ternary (w0, w1) ∈ {-1,0,+1}² combinations. GEMV becomes a stream
 * of vqtbl1q_s8 lookups indexed by 4-bit weight nibbles.
 *
 * Provenance: bitnet.cpp's TL1 (utils/codegen_tl1.py, preset_kernels/
 * bitnet_b1_58-large/bitnet-lut-kernels-tl1.h). Instead of their
 * shape-specialized codegen, one kernel parameterized over
 * (M, K, BM=32, BBK=128) handles all BitNet shapes.
 *
 * Source format: TQ2_0 GGUF blocks (256 elements, fp16 scale, 64
 * packed 2-bit trit bytes), repacked at load time into TL1 tiles that
 * carry their rows' fp32 scales.
 *
 * Activation pipeline:
 *   (1) absmax-quant fp32 → int8 with one per-call scale `s = 127/max|x|`
 *   (2) per K-pair (a, b), enumerate nibble LUT → int8x16 (low/high split)
 *   (3) GEMV: load row's packed nibbles, vqtbl1q_s8 lookup, accumulate
 *       into int16x8, widen to int32 between BBK tiles
 *   (4) final scale: out = int32_acc * (act_scale_inv * row_scale)
 *
 * Tile parameters:
 *   BM = 32   output rows per outer tile
 *   BBK= 128  K positions per outer tile (= 64 K-pairs, accumulator int16)
 *
 * Storage per weight tensor (M rows × K cols), per (BM × BBK) tile:
 *   - packed:  BM × BBK/4 bytes        (M*K/4 in total)
 *   - scales:  BM fp32 = BM*4 bytes    (M*K/32 in total)

 *
 * Shape constraints: M % BM == 0, K % BBK == 0. Unaligned shapes
 * fall through to the existing W1.58 × A8 path.
 *
 * Nibble encoding (4 bits → ternary pair (w0, w1)):
 *   0: (-1, -1)    3: ( 0, -1)    6: (+1, -1)
 *   1: (-1,  0)    4: ( 0,  0)    7: (+1,  0)
 *   2: (-1, +1)    5: ( 0, +1)    8: (+1, +1)
 *   9..15: unused (encoder writes 4 = "0,0")
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_NEON_TL1_H
#define GEIST_INTERNAL_BACKEND_CPU_NEON_TL1_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_neon/tl1.h is internal to the backend layer."
#endif

#include <stddef.h>
#include <stdint.h>

struct geist_weight;
struct geist_backend;

#define TL1_BM 32
#define TL1_BBK 128

/* Returns total bytes for the packed TL1 representation of a (n_in, n_out)
 * weight, including the per-row scale tail. Zero if shape is unsupported. */
size_t tl1_pack_size_bytes(size_t n_in, size_t n_out);

/* Pack a TQ2_0 row-major weight (n_out rows × n_in elements, 256-block
 * TQ2_0 layout) into the TL1 tile format. `out` must hold
 * tl1_pack_size_bytes() bytes. Returns 0 on success, non-zero if shape
 * is unsupported. */
int tl1_pack_from_tq2_0(const void *tq2_0_rows, size_t n_in, size_t n_out, void *out);

/* M=1 decode kernel: y[n_out] = (W·x)[n_out] using the TL1 LUT path.
 * `w->aux_fp32` must point to a tl1_pack_from_tq2_0 buffer; `w->n_in`
 * and `w->n_out` must match the packed shape. */
void cpu_neon_w_tl1_m1(const float               *x,
                       const struct geist_weight *w,
                       struct geist_backend      *be,
                       float                     *y);

#endif /* GEIST_INTERNAL_BACKEND_CPU_NEON_TL1_H */

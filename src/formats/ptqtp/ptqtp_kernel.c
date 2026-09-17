/*
 * ptqtp_kernel.c — PTQTP kernel implementations.
 *
 * NEON path uses Cortex-A76 / Apple Silicon dotprod (vdotq_s32) plus
 * vqtbl1q / vqtbl2q for trit lookup. Scalar fallback for non-NEON.
 *
 * No allocations in the hot path; all buffers are caller-provided.
 * Thread parallelism via OpenMP over the output-row dimension. The kernels
 * never modify input buffers and never escape pointers.
 */
#include "ptqtp_kernel.h"

#include <string.h>

#if defined(__ARM_NEON) || defined(__NEON__)
#include <arm_neon.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

/* ---------------- 2-plane LUTs (joint nibble encoding) ----------------
 * Each weight encoded as 4-bit idx = (T1+1)*3 + (T2+1) ∈ {0..8}; bytes 9..15
 * unused so vqtbl1q with index>=9 returns 0 (not a problem since the
 * encoder never produces them). */
alignas(16) static const int8_t PTQTP_2P_T1_LUT[16] = {
        -1, -1, -1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};
alignas(16) static const int8_t PTQTP_2P_T2_LUT[16] = {
        -1, 0, 1, -1, 0, 1, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0};

/* ---------------- 3-plane LUTs (1-byte joint encoding) ----------------
 * Each weight encoded as idx = (T1+1)*9 + (T2+1)*3 + (T3+1) ∈ {0..26};
 * 27..31 unused (padding for vqtbl2q's 32-entry table). */
alignas(16) static const int8_t PTQTP_3P_T1_LUT[32] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0,
                                                       0,  0,  0,  0,  0,  0,  0,  1,  1,  1, 1,
                                                       1,  1,  1,  1,  1,  0,  0,  0,  0,  0};
alignas(16) static const int8_t PTQTP_3P_T2_LUT[32] = {-1, -1, -1, 0, 0, 0, 1, 1,  1,  -1, -1,
                                                       -1, 0,  0,  0, 1, 1, 1, -1, -1, -1, 0,
                                                       0,  0,  1,  1, 1, 0, 0, 0,  0,  0};
alignas(16) static const int8_t PTQTP_3P_T3_LUT[32] = {-1, 0,  1,  -1, 0,  1, -1, 0,  1, -1, 0,
                                                       1,  -1, 0,  1,  -1, 0, 1,  -1, 0, 1,  -1,
                                                       0,  1,  -1, 0,  1,  0, 0,  0,  0, 0};

#if defined(__ARM_NEON) || defined(__NEON__)
/* Reusable inner: NEON 2-plane decode of one group (group_size weights
 * out of [n_in, group_size]). Returns acc1, acc2 (int32, sum-across). */
static inline void ptqtp_2plane_group_neon(const uint8_t *g_trits,
                                           const int8_t  *g_x,
                                           size_t         group_byte_size,
                                           int32_t       *out_acc1,
                                           int32_t       *out_acc2) {
    const uint8x16_t lo_mask = vdupq_n_u8(0x0F);
    const int8x16_t  t1_lut  = vld1q_s8(PTQTP_2P_T1_LUT);
    const int8x16_t  t2_lut  = vld1q_s8(PTQTP_2P_T2_LUT);
    int32x4_t        v_acc1  = vdupq_n_s32(0);
    int32x4_t        v_acc2  = vdupq_n_s32(0);
    for (size_t k = 0; k < group_byte_size; k += 16) {
        const uint8x16_t  bytes   = vld1q_u8(g_trits + k);
        const uint8x16_t  idx_lo  = vandq_u8(bytes, lo_mask);
        const uint8x16_t  idx_hi  = vshrq_n_u8(bytes, 4);
        const int8x16_t   T1_even = vqtbl1q_s8(t1_lut, idx_lo);
        const int8x16_t   T2_even = vqtbl1q_s8(t2_lut, idx_lo);
        const int8x16_t   T1_odd  = vqtbl1q_s8(t1_lut, idx_hi);
        const int8x16_t   T2_odd  = vqtbl1q_s8(t2_lut, idx_hi);
        const int8x16x2_t xv      = vld2q_s8(g_x + k * 2);
        v_acc1                    = vdotq_s32(v_acc1, T1_even, xv.val[0]);
        v_acc2                    = vdotq_s32(v_acc2, T2_even, xv.val[0]);
        v_acc1                    = vdotq_s32(v_acc1, T1_odd, xv.val[1]);
        v_acc2                    = vdotq_s32(v_acc2, T2_odd, xv.val[1]);
    }
    *out_acc1 = vaddvq_s32(v_acc1);
    *out_acc2 = vaddvq_s32(v_acc2);
}
#endif

/* Scalar 2-plane decode of one group, used as fallback and reference.
 * [[maybe_unused]] because the NEON path is selected at compile time. */
[[maybe_unused]] static inline void ptqtp_2plane_group_scalar(const uint8_t *g_trits,
                                                              const int8_t  *g_x,
                                                              size_t         group_byte_size,
                                                              int32_t       *out_acc1,
                                                              int32_t       *out_acc2) {
    int32_t a1 = 0, a2 = 0;
    for (size_t k = 0; k < group_byte_size; k++) {
        const uint8_t b  = g_trits[k];
        const int8_t  xa = g_x[2 * k];
        const int8_t  xb = g_x[2 * k + 1];
        a1 += (int32_t) PTQTP_2P_T1_LUT[b & 0x0F] * xa + (int32_t) PTQTP_2P_T1_LUT[b >> 4] * xb;
        a2 += (int32_t) PTQTP_2P_T2_LUT[b & 0x0F] * xa + (int32_t) PTQTP_2P_T2_LUT[b >> 4] * xb;
    }
    *out_acc1 = a1;
    *out_acc2 = a2;
}

void ptqtp_gemv_2plane_fp32alpha(size_t         n_in,
                                 size_t         n_out,
                                 size_t         group_size,
                                 const int8_t  *x_q8,
                                 float          scale_x,
                                 const uint8_t *trits,
                                 const float   *alpha_fp32,
                                 float         *y) {
    if (group_size == 0 || n_in % group_size != 0 || group_size % 32 != 0)
        return;
    const size_t n_groups        = n_in / group_size;
    const size_t row_byte_stride = n_in / 2;
    const size_t group_byte_size = group_size / 2;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t n = 0; n < n_out; n++) {
        const uint8_t *row_trits = trits + n * row_byte_stride;
        const float   *row_alpha = alpha_fp32 + n * n_groups * 2;
        float          acc       = 0.0f;

        for (size_t g = 0; g < n_groups; g++) {
            int32_t acc1, acc2;
#if defined(__ARM_NEON) || defined(__NEON__)
            ptqtp_2plane_group_neon(row_trits + g * group_byte_size,
                                    x_q8 + g * group_size,
                                    group_byte_size,
                                    &acc1,
                                    &acc2);
#else
            ptqtp_2plane_group_scalar(row_trits + g * group_byte_size,
                                      x_q8 + g * group_size,
                                      group_byte_size,
                                      &acc1,
                                      &acc2);
#endif
            acc += row_alpha[g * 2 + 0] * (float) acc1 + row_alpha[g * 2 + 1] * (float) acc2;
        }
        y[n] = scale_x * acc;
    }
}

/* fp16α path: kept for disk-/bench-only callers that want to avoid the 2×
 * alpha-memory cost of the fp32 arena. The production runtime path goes
 * through ptqtp_gemv_2plane_fp32alpha after the loader pre-promotes alpha
 * once at model-load time (see gguf_ptqtp.c:259-274). On M1, fp16α is ~5 %
 * slower than fp32α; on Pi 5 the gap is in the same range (DRAM-bandwidth
 * margin masks part of it). Two attempts to close that gap by changing
 * the FP16→FP32 promotion (pre-promote at row start; direct __fp16 load
 * per group) produced no measurable improvement — the compiler already
 * register-allocates the pair buffer below. The remaining delta is the
 * FCVT instruction itself, which cannot be eliminated without going to
 * fp32α. Documented decision: use fp32α at runtime, leave this path
 * untouched. */
void ptqtp_gemv_2plane_fp16alpha(size_t          n_in,
                                 size_t          n_out,
                                 size_t          group_size,
                                 const int8_t   *x_q8,
                                 float           scale_x,
                                 const uint8_t  *trits,
                                 const uint16_t *alpha_fp16,
                                 float          *y) {
    if (group_size == 0 || n_in % group_size != 0 || group_size % 32 != 0)
        return;
    const size_t n_groups        = n_in / group_size;
    const size_t row_byte_stride = n_in / 2;
    const size_t group_byte_size = group_size / 2;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t n = 0; n < n_out; n++) {
        const uint8_t  *row_trits = trits + n * row_byte_stride;
        const uint16_t *row_alpha = alpha_fp16 + n * n_groups * 2;
        float           acc       = 0.0f;

        for (size_t g = 0; g < n_groups; g++) {
            int32_t acc1, acc2;
#if defined(__ARM_NEON) || defined(__NEON__)
            ptqtp_2plane_group_neon(row_trits + g * group_byte_size,
                                    x_q8 + g * group_size,
                                    group_byte_size,
                                    &acc1,
                                    &acc2);
            /* Convert 2 fp16 alpha values via hardware FCVT. We memcpy
             * the bit pattern into a `__fp16[2]` typed local instead of
             * casting `uint16_t*` to `__fp16*` — the latter is strict-
             * aliasing UB (review #12 / V8) because __fp16 is not a
             * character type and TBAA may reorder the init across the
             * load under -O3 -flto. memcpy gives the same FCVT codegen
             * without the UB. */
            __fp16 pair[2];
            memcpy(pair, &row_alpha[g * 2], 2 * sizeof(uint16_t));
            const float a1 = (float) pair[0];
            const float a2 = (float) pair[1];
#else
            ptqtp_2plane_group_scalar(row_trits + g * group_byte_size,
                                      x_q8 + g * group_size,
                                      group_byte_size,
                                      &acc1,
                                      &acc2);
            /* Manual fp16 → fp32 if vcvt is unavailable. */
            float a1, a2;
            {
                uint16_t h = row_alpha[g * 2 + 0];
                uint32_t s = (h >> 15) & 1u, e = (h >> 10) & 0x1Fu, f = h & 0x3FFu;
                uint32_t bits = (e == 0)      ? (f == 0 ? (s << 31) : 0)
                                : (e == 0x1F) ? ((s << 31) | 0x7F800000u | (f << 13))
                                              : ((s << 31) | ((e + 112u) << 23) | (f << 13));
                memcpy(&a1, &bits, 4);
            }
            {
                uint16_t h = row_alpha[g * 2 + 1];
                uint32_t s = (h >> 15) & 1u, e = (h >> 10) & 0x1Fu, f = h & 0x3FFu;
                uint32_t bits = (e == 0)      ? (f == 0 ? (s << 31) : 0)
                                : (e == 0x1F) ? ((s << 31) | 0x7F800000u | (f << 13))
                                              : ((s << 31) | ((e + 112u) << 23) | (f << 13));
                memcpy(&a2, &bits, 4);
            }
#endif
            acc += a1 * (float) acc1 + a2 * (float) acc2;
        }
        y[n] = scale_x * acc;
    }
}

void ptqtp_gemm_2plane_fp32alpha(size_t         M,
                                 size_t         n_in,
                                 size_t         n_out,
                                 size_t         group_size,
                                 const int8_t  *x_q8,
                                 const float   *scale_x,
                                 const uint8_t *trits,
                                 const float   *alpha_fp32,
                                 float         *y) {
    if (M == 0)
        return;
    if (M == 1) {
        ptqtp_gemv_2plane_fp32alpha(
                n_in, n_out, group_size, x_q8, scale_x[0], trits, alpha_fp32, y);
        return;
    }
    if (group_size == 0 || n_in % group_size != 0 || group_size % 32 != 0)
        return;
    const size_t n_groups        = n_in / group_size;
    const size_t row_byte_stride = n_in / 2;
    const size_t group_byte_size = group_size / 2;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t n = 0; n < n_out; n++) {
        const uint8_t *row_trits = trits + n * row_byte_stride;
        const float   *row_alpha = alpha_fp32 + n * n_groups * 2;

        /* VLA on stack for per-row M-accumulators. M is bounded by typical
         * prefill batch sizes (≤ a few thousand tokens). For M > stack
         * budget the GEMM kernel would need tiling — not needed here. */
        float acc_m[M];
        for (size_t m = 0; m < M; m++)
            acc_m[m] = 0.0f;

        for (size_t g = 0; g < n_groups; g++) {
            const float a1 = row_alpha[g * 2 + 0];
            const float a2 = row_alpha[g * 2 + 1];
            for (size_t m = 0; m < M; m++) {
                int32_t acc1, acc2;
#if defined(__ARM_NEON) || defined(__NEON__)
                ptqtp_2plane_group_neon(row_trits + g * group_byte_size,
                                        x_q8 + m * n_in + g * group_size,
                                        group_byte_size,
                                        &acc1,
                                        &acc2);
#else
                ptqtp_2plane_group_scalar(row_trits + g * group_byte_size,
                                          x_q8 + m * n_in + g * group_size,
                                          group_byte_size,
                                          &acc1,
                                          &acc2);
#endif
                acc_m[m] += a1 * (float) acc1 + a2 * (float) acc2;
            }
        }
        for (size_t m = 0; m < M; m++) {
            y[m * n_out + n] = scale_x[m] * acc_m[m];
        }
    }
}

/* ---------------- Bit-expansion LUT (256 → 8 bytes) ----------------
 * BIT_EXPAND_LUT[b][i] = (b >> i) & 1 for i ∈ [0, 8).
 * Used by the 5-bit packed kernel's NEON path to expand the high-bit stream;
 * the scalar path does not read it, so it only exists where NEON does (clang
 * rejects an unused internal table under -Werror).
 * 256 × 8 = 2048 bytes — comfortably fits in L1 (4 cache lines per entry
 * is misleading; the lookup pattern is byte-stream sequential so each
 * accessed entry brings its 8-byte payload in one read). */
#if defined(__ARM_NEON) || defined(__NEON__)
alignas(16) static const uint8_t BIT_EXPAND_LUT[256][8] = {
#define B0(b)                                                                          \
    ((uint8_t) ((b) & 1)), ((uint8_t) (((b) >> 1) & 1)), ((uint8_t) (((b) >> 2) & 1)), \
            ((uint8_t) (((b) >> 3) & 1)), ((uint8_t) (((b) >> 4) & 1)),                \
            ((uint8_t) (((b) >> 5) & 1)), ((uint8_t) (((b) >> 6) & 1)),                \
            ((uint8_t) (((b) >> 7) & 1))
#define B(b) {B0(b)}
#define R8(b) \
    B(b), B((b) + 1), B((b) + 2), B((b) + 3), B((b) + 4), B((b) + 5), B((b) + 6), B((b) + 7)
#define R64(b)                                                                                \
    R8(b), R8((b) + 8), R8((b) + 16), R8((b) + 24), R8((b) + 32), R8((b) + 40), R8((b) + 48), \
            R8((b) + 56)
        R64(0), R64(64), R64(128), R64(192)
#undef R64
#undef R8
#undef B
#undef B0
};

static_assert(sizeof(BIT_EXPAND_LUT) == 2048, "BIT_EXPAND_LUT must be 2 KiB");
#endif /* __ARM_NEON || __NEON__ */

void ptqtp_gemv_3plane_packed5_fp32alpha(size_t         n_in,
                                         size_t         n_out,
                                         size_t         group_size,
                                         const int8_t  *x_q8,
                                         float          scale_x,
                                         const uint8_t *trits,
                                         const float   *alpha_fp32,
                                         float         *y) {
    if (group_size == 0 || n_in % group_size != 0)
        return;
    if (n_in % 8 != 0 || group_size % 16 != 0)
        return;
    const size_t n_groups        = n_in / group_size;
    const size_t low_bytes_row   = n_in / 2; /* low nibble stream */
    const size_t hi_bytes_row    = n_in / 8; /* high-bit stream  */
    const size_t row_byte_stride = low_bytes_row + hi_bytes_row;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t n = 0; n < n_out; n++) {
        const uint8_t *row       = trits + n * row_byte_stride;
        const uint8_t *row_low   = row;
        const uint8_t *row_high  = row + low_bytes_row;
        const float   *row_alpha = alpha_fp32 + n * n_groups * 3;
        float          acc       = 0.0f;

        for (size_t g = 0; g < n_groups; g++) {
            const uint8_t *g_low  = row_low + g * (group_size / 2);
            const uint8_t *g_high = row_high + g * (group_size / 8);
            const int8_t  *g_x    = x_q8 + g * group_size;
            int32_t        acc1, acc2, acc3;

#if defined(__ARM_NEON) || defined(__NEON__)
            int8x16x2_t t1_lut, t2_lut, t3_lut;
            t1_lut.val[0]    = vld1q_s8(PTQTP_3P_T1_LUT);
            t1_lut.val[1]    = vld1q_s8(PTQTP_3P_T1_LUT + 16);
            t2_lut.val[0]    = vld1q_s8(PTQTP_3P_T2_LUT);
            t2_lut.val[1]    = vld1q_s8(PTQTP_3P_T2_LUT + 16);
            t3_lut.val[0]    = vld1q_s8(PTQTP_3P_T3_LUT);
            t3_lut.val[1]    = vld1q_s8(PTQTP_3P_T3_LUT + 16);
            int32x4_t v_acc1 = vdupq_n_s32(0);
            int32x4_t v_acc2 = vdupq_n_s32(0);
            int32x4_t v_acc3 = vdupq_n_s32(0);

            /* Per chunk: 16 weights = 8 low-nibble bytes + 2 hi-bit bytes.
             *
             * Decode strategy:
             *   nibbles[16]: split 8 source bytes into low/high nibbles, then
             *                vzip → weight-order [n0..n15].
             *   hi_bits[16]: BIT_EXPAND_LUT lookup × 2 → uint8x8 each, combine
             *                into a 16-lane vector. Lookup row maps a byte's
             *                bit i to lane i directly (weight order).
             *   idx = nibbles | (hi_bits << 4)
             */
            for (size_t k = 0; k < group_size; k += 16) {
                /* Low nibble stream → 16 nibbles in weight order. */
                const uint8x8_t   lo_pairs = vld1_u8(g_low + k / 2);
                const uint8x8_t   lo_even  = vand_u8(lo_pairs, vdup_n_u8(0x0F));
                const uint8x8_t   lo_odd   = vshr_n_u8(lo_pairs, 4);
                const uint8x8x2_t lo_zip   = vzip_u8(lo_even, lo_odd);
                const uint8x16_t  nibbles  = vcombine_u8(lo_zip.val[0], lo_zip.val[1]);

                /* High bit stream → 16 bits via 256-byte expansion LUT. */
                const uint8x8_t  hi_b0   = vld1_u8(BIT_EXPAND_LUT[g_high[k / 8 + 0]]);
                const uint8x8_t  hi_b1   = vld1_u8(BIT_EXPAND_LUT[g_high[k / 8 + 1]]);
                const uint8x16_t hi_bits = vcombine_u8(hi_b0, hi_b1);

                /* Combine: idx = nibble | (hi_bit << 4)  ∈ [0, 27). */
                const uint8x16_t idx = vorrq_u8(nibbles, vshlq_n_u8(hi_bits, 4));

                const int8x16_t T1 = vqtbl2q_s8(t1_lut, idx);
                const int8x16_t T2 = vqtbl2q_s8(t2_lut, idx);
                const int8x16_t T3 = vqtbl2q_s8(t3_lut, idx);
                const int8x16_t xv = vld1q_s8(g_x + k);
                v_acc1             = vdotq_s32(v_acc1, T1, xv);
                v_acc2             = vdotq_s32(v_acc2, T2, xv);
                v_acc3             = vdotq_s32(v_acc3, T3, xv);
            }
            acc1 = vaddvq_s32(v_acc1);
            acc2 = vaddvq_s32(v_acc2);
            acc3 = vaddvq_s32(v_acc3);
#else
            acc1 = 0;
            acc2 = 0;
            acc3 = 0;
            for (size_t k = 0; k < group_size; k++) {
                const uint8_t lo_byte = g_low[k / 2];
                const uint8_t nibble  = (uint8_t) ((lo_byte >> ((k & 1u) * 4u)) & 0x0Fu);
                const uint8_t hi_byte = g_high[k / 8];
                const uint8_t bit     = (uint8_t) ((hi_byte >> (k & 7u)) & 0x01u);
                const uint8_t idx     = (uint8_t) (nibble | (bit << 4u));
                const int8_t  xv      = g_x[k];
                acc1 += (int32_t) PTQTP_3P_T1_LUT[idx] * xv;
                acc2 += (int32_t) PTQTP_3P_T2_LUT[idx] * xv;
                acc3 += (int32_t) PTQTP_3P_T3_LUT[idx] * xv;
            }
#endif
            acc += row_alpha[g * 3 + 0] * (float) acc1 + row_alpha[g * 3 + 1] * (float) acc2 +
                   row_alpha[g * 3 + 2] * (float) acc3;
        }
        y[n] = scale_x * acc;
    }
}

void ptqtp_gemv_3plane_fp32alpha(size_t         n_in,
                                 size_t         n_out,
                                 size_t         group_size,
                                 const int8_t  *x_q8,
                                 float          scale_x,
                                 const uint8_t *trits,
                                 const float   *alpha_fp32,
                                 float         *y) {
    if (group_size == 0 || n_in % group_size != 0 || group_size % 16 != 0)
        return;
    const size_t n_groups        = n_in / group_size;
    const size_t row_byte_stride = n_in; /* 1 byte per weight */

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t n = 0; n < n_out; n++) {
        const uint8_t *row_trits = trits + n * row_byte_stride;
        const float   *row_alpha = alpha_fp32 + n * n_groups * 3;
        float          acc       = 0.0f;

        for (size_t g = 0; g < n_groups; g++) {
            const uint8_t *g_trits = row_trits + g * group_size;
            const int8_t  *g_x     = x_q8 + g * group_size;
            int32_t        acc1, acc2, acc3;

#if defined(__ARM_NEON) || defined(__NEON__)
            int8x16x2_t t1_lut, t2_lut, t3_lut;
            t1_lut.val[0]    = vld1q_s8(PTQTP_3P_T1_LUT);
            t1_lut.val[1]    = vld1q_s8(PTQTP_3P_T1_LUT + 16);
            t2_lut.val[0]    = vld1q_s8(PTQTP_3P_T2_LUT);
            t2_lut.val[1]    = vld1q_s8(PTQTP_3P_T2_LUT + 16);
            t3_lut.val[0]    = vld1q_s8(PTQTP_3P_T3_LUT);
            t3_lut.val[1]    = vld1q_s8(PTQTP_3P_T3_LUT + 16);
            int32x4_t v_acc1 = vdupq_n_s32(0);
            int32x4_t v_acc2 = vdupq_n_s32(0);
            int32x4_t v_acc3 = vdupq_n_s32(0);
            for (size_t k = 0; k < group_size; k += 16) {
                const uint8x16_t bytes = vld1q_u8(g_trits + k);
                const int8x16_t  T1    = vqtbl2q_s8(t1_lut, bytes);
                const int8x16_t  T2    = vqtbl2q_s8(t2_lut, bytes);
                const int8x16_t  T3    = vqtbl2q_s8(t3_lut, bytes);
                const int8x16_t  xv    = vld1q_s8(g_x + k);
                v_acc1                 = vdotq_s32(v_acc1, T1, xv);
                v_acc2                 = vdotq_s32(v_acc2, T2, xv);
                v_acc3                 = vdotq_s32(v_acc3, T3, xv);
            }
            acc1 = vaddvq_s32(v_acc1);
            acc2 = vaddvq_s32(v_acc2);
            acc3 = vaddvq_s32(v_acc3);
#else
            acc1 = 0;
            acc2 = 0;
            acc3 = 0;
            for (size_t k = 0; k < group_size; k++) {
                const uint8_t b  = g_trits[k];
                const int8_t  xv = g_x[k];
                acc1 += (int32_t) PTQTP_3P_T1_LUT[b] * xv;
                acc2 += (int32_t) PTQTP_3P_T2_LUT[b] * xv;
                acc3 += (int32_t) PTQTP_3P_T3_LUT[b] * xv;
            }
#endif
            acc += row_alpha[g * 3 + 0] * (float) acc1 + row_alpha[g * 3 + 1] * (float) acc2 +
                   row_alpha[g * 3 + 2] * (float) acc3;
        }
        y[n] = scale_x * acc;
    }
}

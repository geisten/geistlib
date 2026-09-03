/*
 * bench_5trit_probe — head-to-head: current 4-bpw joint encoding vs
 * 5-trit-per-byte separate-plane encoding for one PTQTP tensor.
 *
 * Loads a real PTQTP tensor, repacks T1/T2 as 5-trits-per-byte, then
 * runs both kernels in tight loops and reports GB/s. Decision: if the
 * 5-trit kernel hits >18% faster (matching the bandwidth saving), it's
 * worth a full integration. If decode cost eats most of the saving,
 * abandon.
 *
 * Smart decode: multiplication by reciprocal (vmovl_u8 + vmulq + vshrn)
 * to extract trit positions without LUT lookups. ~22 vops per 16-byte
 * input chunk to get 80 trits, vs naive LUT approach at ~115 vops.
 *
 * Usage:
 *   bench_5trit_probe <ptqtp.bin> [tensor_name]
 *     default tensor: blk.0.ffn_gate.weight
 */
#include "gguf_ptqtp.h"
#include "quant.h"
#include "ptqtp_kernel.h"

#include <alloca.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return (double) tv.tv_sec * 1000.0 + (double) tv.tv_usec / 1000.0;
}

/* Decode 4 bits → trit. Mirror of gguf_quant.c. */
static const int8_t T1_LUT[16] = {-1, -1, -1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};
static const int8_t T2_LUT[16] = {-1, 0, 1, -1, 0, 1, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0};

/* Repack a row of joint-encoded trits → separate 5-trit-per-byte planes.
 * Output: T1_packed[ceil(n_in/5)], T2_packed[ceil(n_in/5)].
 * The last byte may be partially filled; we pad input with 0 (T=0). */
static void
repack_row_to_5trit(const uint8_t *joint, size_t n_in, uint8_t *T1_packed, uint8_t *T2_packed) {
    /* First decode joint into per-weight T1[], T2[] arrays. */
    int8_t *T1 = (int8_t *) alloca(((n_in + 4) / 5) * 5);
    int8_t *T2 = (int8_t *) alloca(((n_in + 4) / 5) * 5);
    for (size_t i = 0; i < n_in / 2; i++) {
        const uint8_t b = joint[i];
        T1[2 * i]       = T1_LUT[b & 0x0F];
        T2[2 * i]       = T2_LUT[b & 0x0F];
        T1[2 * i + 1]   = T1_LUT[b >> 4];
        T2[2 * i + 1]   = T2_LUT[b >> 4];
    }
    /* Pad to multiple of 5. */
    size_t padded = ((n_in + 4) / 5) * 5;
    for (size_t i = n_in; i < padded; i++) {
        T1[i] = 0;
        T2[i] = 0;
    }

    /* Pack 5 trits per byte: byte = t0 + 3*t1 + 9*t2 + 27*t3 + 81*t4
     * where each ti is shifted to {0,1,2}. */
    const size_t n_packed = padded / 5;
    for (size_t i = 0; i < n_packed; i++) {
        const int s  = i * 5;
        T1_packed[i] = (uint8_t) ((T1[s] + 1) + 3 * (T1[s + 1] + 1) + 9 * (T1[s + 2] + 1) +
                                  27 * (T1[s + 3] + 1) + 81 * (T1[s + 4] + 1));
        T2_packed[i] = (uint8_t) ((T2[s] + 1) + 3 * (T2[s + 1] + 1) + 9 * (T2[s + 2] + 1) +
                                  27 * (T2[s + 3] + 1) + 81 * (T2[s + 4] + 1));
    }
}

/* Smart 5-trit kernel. Decodes 16 input bytes (80 trits per plane) using
 * multiplication-by-reciprocal extraction. Per-row, per-group accumulation.
 * Group size assumed 128 (so 26 packed bytes per plane per group, but we
 * process the first 25 bytes vectorized = 80 trits, plus tail). */
static void linear_5trit_decode(size_t          n_in,
                                size_t          n_out,
                                size_t          group_size,
                                const int8_t   *x_q8,
                                float           scale_x,
                                const uint8_t  *T1_packed,
                                const uint8_t  *T2_packed,
                                const uint16_t *alpha,
                                float          *y) {
    if (n_in % group_size != 0)
        return;
    const size_t n_groups = n_in / group_size;
    /* Bytes per row per plane = ceil(n_in / 5). */
    const size_t row_bytes_per_plane = (n_in + 4) / 5;
    /* Bytes per group per plane = ceil(group_size / 5). For group_size=128
     * this is 26; we process 25 (=5 chunks of 5) vectorized then 1 tail. */

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t n = 0; n < n_out; n++) {
        const uint8_t  *row_t1    = T1_packed + n * row_bytes_per_plane;
        const uint8_t  *row_t2    = T2_packed + n * row_bytes_per_plane;
        const uint16_t *row_alpha = alpha + n * n_groups * 2;
        float           acc       = 0.0f;

        size_t weight_off = 0; /* index into x_q8 */
        for (size_t g = 0; g < n_groups; g++) {
            int32_t      acc1 = 0, acc2 = 0;
            const size_t pack_g_start = g * ((group_size + 4) / 5);
            /* Per group: process 128 weights = 26 packed bytes. We do the
             * first 25 packed bytes (= 125 trits) via 16-byte chunks, then
             * the tail 1 byte (= 5 trits, 3 of which we use to reach 128). */
            size_t k = 0;
#if defined(__ARM_NEON)
            /* Process 16 packed bytes at a time → 80 trits per plane.
             * For group_size=128 we have 25 effective packed bytes per
             * plane (last one partial), so one 16-byte chunk + tail.
             *
             * Decode strategy per 16 input bytes:
             *   v0 = packed bytes (uint8x16)
             *   trit_pos[0] = (v0 % 3) - 1
             *   v1 = v0 / 3
             *   trit_pos[1] = (v1 % 3) - 1
             *   ... (5 positions)
             *
             * Division by 3 (integer): use multiplication by 0xAB then
             * shift. For uint8 b in [0, 243]:
             *   b/3 = (b * 0xAB) >> 9   approximately but exactly for b<243
             * Use uint16 widening to keep precision.
             *
             * Mod 3: q = b/3, m = b - 3*q. */
            const uint8x16_t v_three = vdupq_n_u8(3);
            const uint8x16_t v_one_i = vdupq_n_u8(1);

            for (; k + 16 <= 25; k += 16) {
                /* Load 16 packed bytes for both planes. */
                uint8x16_t b1 = vld1q_u8(row_t1 + pack_g_start + k);
                uint8x16_t b2 = vld1q_u8(row_t2 + pack_g_start + k);

                /* Process trits per position 0..4. After each, divide by 3. */
                int8x16_t trits1[5], trits2[5];
                for (int pos = 0; pos < 5; pos++) {
                    /* mod 3 */
                    /* div by 3: (b * 0xAB) >> 9, valid for b in [0,255]. */
                    uint16x8_t b1_lo = vmull_u8(vget_low_u8(b1), vdup_n_u8(0xAB));
                    uint16x8_t b1_hi = vmull_u8(vget_high_u8(b1), vdup_n_u8(0xAB));
                    uint8x16_t q1    = vcombine_u8(vmovn_u16(vshrq_n_u16(b1_lo, 9)),
                                                   vmovn_u16(vshrq_n_u16(b1_hi, 9)));
                    uint8x16_t m1    = vsubq_u8(b1, vmulq_u8(q1, v_three));
                    trits1[pos]      = vreinterpretq_s8_u8(vsubq_u8(m1, v_one_i));
                    b1               = q1;

                    uint16x8_t b2_lo = vmull_u8(vget_low_u8(b2), vdup_n_u8(0xAB));
                    uint16x8_t b2_hi = vmull_u8(vget_high_u8(b2), vdup_n_u8(0xAB));
                    uint8x16_t q2    = vcombine_u8(vmovn_u16(vshrq_n_u16(b2_lo, 9)),
                                                   vmovn_u16(vshrq_n_u16(b2_hi, 9)));
                    uint8x16_t m2    = vsubq_u8(b2, vmulq_u8(q2, v_three));
                    trits2[pos]      = vreinterpretq_s8_u8(vsubq_u8(m2, v_one_i));
                    b2               = q2;
                }

                /* Now we have 5 vectors of 16 trits each = 80 trits per plane.
                 * Need to dot with x_q8[weight_off + pos*16 .. ]. The trits
                 * come out interleaved by position: trit[0] for input bytes
                 * 0..15 corresponds to weights 0, 5, 10, ..., 75.
                 *
                 * That's a STRIDED access pattern. We need to either gather
                 * x_q8 strided or rearrange the trits. Easier: process 5
                 * vdot calls, where vdot[pos] takes trits[pos] (16 trits at
                 * positions p, p+5, p+10, ...) and the corresponding x_q8.
                 * x_q8 strided: x_q8[w_off + 5*i + pos] for i=0..15.
                 *
                 * For position pos, we need x_q8 vector with elements
                 * (x_q8[w_off + 0*5+pos], x_q8[w_off + 1*5+pos], ..., x_q8[w_off + 15*5+pos]).
                 * That's a strided gather, not a contiguous load.
                 *
                 * NEON doesn't have strided loads. We'd need vld5q_s8 (stride-5
                 * deinterleave) which doesn't exist (vld2/3/4 only).
                 *
                 * Workaround: use scalar gather. Or change packing layout. */
                /* SCALAR fallback for now to validate correctness. */
                int8_t trits1_buf[80], trits2_buf[80];
                for (int pos = 0; pos < 5; pos++) {
                    vst1q_s8(trits1_buf + pos * 16, trits1[pos]);
                    vst1q_s8(trits2_buf + pos * 16, trits2[pos]);
                }
                /* Reconstruct in-order: trit at weight w corresponds to
                 * input byte w/5 (= one of bytes 0..15) and position w%5. */
                for (int w = 0; w < 80; w++) {
                    const int    byte_idx = w / 5; /* 0..15 */
                    const int    pos      = w % 5; /* 0..4 */
                    const int8_t t1       = trits1_buf[pos * 16 + byte_idx];
                    const int8_t t2       = trits2_buf[pos * 16 + byte_idx];
                    const int8_t xv       = x_q8[weight_off + w];
                    acc1 += (int32_t) t1 * xv;
                    acc2 += (int32_t) t2 * xv;
                }
                weight_off += 80;
            }
#endif
            /* Tail: scalar decode for remaining packed bytes. */
            const size_t bytes_per_group = (group_size + 4) / 5;
            for (; k < bytes_per_group; k++) {
                uint8_t b1 = row_t1[pack_g_start + k];
                uint8_t b2 = row_t2[pack_g_start + k];
                for (int pos = 0; pos < 5 && weight_off < (g + 1) * group_size; pos++) {
                    const int8_t t1 = (int8_t) (b1 % 3) - 1;
                    b1 /= 3;
                    const int8_t t2 = (int8_t) (b2 % 3) - 1;
                    b2 /= 3;
                    const int8_t xv = x_q8[weight_off];
                    acc1 += (int32_t) t1 * xv;
                    acc2 += (int32_t) t2 * xv;
                    weight_off++;
                }
            }

            /* Apply alpha for this group. */
            __fp16 a1h, a2h;
            memcpy(&a1h, &row_alpha[g * 2 + 0], 2);
            memcpy(&a2h, &row_alpha[g * 2 + 1], 2);
            acc += (float) a1h * (float) acc1 + (float) a2h * (float) acc2;
        }
        y[n] = scale_x * acc;
    }
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s <ptqtp.bin> [tensor_name]\n", argv[0]);
        return 2;
    }
    const char       *err = nullptr;
    struct ptqtp_ctx *ctx = ptqtp_open(argv[1], &err);
    if (!ctx) {
        fprintf(stderr, "ptqtp_open: %s\n", err ? err : "?");
        return 1;
    }
    const char                  *name = (argc == 3) ? argv[2] : "blk.0.ffn_gate.weight";
    const struct ptqtp_tensor_t *t    = ptqtp_get_tensor(ctx, name);
    if (!t) {
        fprintf(stderr, "tensor not found\n");
        return 1;
    }

    const uint32_t gs              = ptqtp_group_size(ctx);
    const size_t   n_in            = t->n_in;
    const size_t   n_out           = t->n_out;
    const size_t   bytes_per_plane = ((n_in + 4) / 5) * n_out;
    printf("Tensor %s: %zu × %zu (gs=%u)\n", name, n_out, n_in, gs);
    printf("  Current 4-bpw joint: %zu bytes trits + alpha fp16\n", n_out * n_in / 2);
    printf("  5-trit packed:       %zu bytes T1 + %zu bytes T2 + alpha\n",
           bytes_per_plane,
           bytes_per_plane);
    printf("  Bandwidth saving:    %.1f%%\n",
           100.0 * (1.0 - (2.0 * bytes_per_plane) / (double) (n_out * n_in / 2)));

    /* Pack the entire tensor as 5-trit. */
    uint8_t *T1_packed = (uint8_t *) aligned_alloc(64, bytes_per_plane);
    uint8_t *T2_packed = (uint8_t *) aligned_alloc(64, bytes_per_plane);
    for (size_t r = 0; r < n_out; r++) {
        repack_row_to_5trit(t->trits + r * (n_in / 2),
                            n_in,
                            T1_packed + r * ((n_in + 4) / 5),
                            T2_packed + r * ((n_in + 4) / 5));
    }

    /* Setup. */
    int8_t *x_q8  = (int8_t *) aligned_alloc(64, n_in);
    float  *y_old = (float *) aligned_alloc(64, n_out * sizeof(float));
    float  *y_new = (float *) aligned_alloc(64, n_out * sizeof(float));
    float  *x     = (float *) aligned_alloc(64, n_in * sizeof(float));
    for (size_t i = 0; i < n_in; i++)
        x[i] = ((float) i * 0.0137f) - 7.3f;
    float scale_x = quantize_x_int8_sym(n_in, x, x_q8);
    free(x);

    /* Bench current kernel. */
    {
        const double tw = now_ms();
        ptqtp_gemv_2plane_fp16alpha(n_in, n_out, gs, x_q8, scale_x, t->trits, t->alpha, y_old);
        const double single_ms = now_ms() - tw;
        int          n_iter    = (int) (3000.0 / (single_ms + 0.001)) + 5;
        if (n_iter > 100000)
            n_iter = 100000;
        const double t0 = now_ms();
        for (int it = 0; it < n_iter; it++) {
            ptqtp_gemv_2plane_fp16alpha(n_in, n_out, gs, x_q8, scale_x, t->trits, t->alpha, y_old);
        }
        const double dt_ms = (now_ms() - t0) / n_iter;
        const size_t bytes = n_out * n_in / 2 + n_out * (n_in / gs) * 2 * sizeof(uint16_t);
        printf("\n  [4-bpw  ] %.3f ms/call  %.2f GB/s  (%d it)\n",
               dt_ms,
               (double) bytes / (dt_ms * 1e6),
               n_iter);
    }

    /* Bench 5-trit kernel. */
    {
        const double tw = now_ms();
        linear_5trit_decode(n_in, n_out, gs, x_q8, scale_x, T1_packed, T2_packed, t->alpha, y_new);
        const double single_ms = now_ms() - tw;
        int          n_iter    = (int) (3000.0 / (single_ms + 0.001)) + 5;
        if (n_iter > 100000)
            n_iter = 100000;
        const double t0 = now_ms();
        for (int it = 0; it < n_iter; it++) {
            linear_5trit_decode(
                    n_in, n_out, gs, x_q8, scale_x, T1_packed, T2_packed, t->alpha, y_new);
        }
        const double dt_ms = (now_ms() - t0) / n_iter;
        const size_t bytes = 2 * bytes_per_plane + n_out * (n_in / gs) * 2 * sizeof(uint16_t);
        printf("  [5-trit ] %.3f ms/call  %.2f GB/s  (%d it)\n",
               dt_ms,
               (double) bytes / (dt_ms * 1e6),
               n_iter);
    }

    /* Correctness: cos sim between y_old and y_new. */
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n_out; i++) {
        dot += (double) y_old[i] * y_new[i];
        na += (double) y_old[i] * y_old[i];
        nb += (double) y_new[i] * y_new[i];
    }
    printf("\n  cos_sim(old, new) = %.7f\n", dot / (na > 0 && nb > 0 ? sqrt(na) * sqrt(nb) : 1.0));

    free(T1_packed);
    free(T2_packed);
    free(x_q8);
    free(y_old);
    free(y_new);
    ptqtp_close(ctx);
    return 0;
}

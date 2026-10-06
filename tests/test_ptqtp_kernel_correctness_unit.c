/*
 * test_ptqtp_kernel_correctness — the 2-plane PTQTP GEMV kernels.
 *
 * Verifies that ptqtp_gemv_2plane_fp16alpha and ptqtp_gemv_2plane_fp32alpha
 * produce equivalent output on a deterministic 2-plane test case, plus a
 * scalar reference computation, so a change to the inner-loop α handling
 * shows up here.
 *
 * Tolerances:
 *   fp16α vs scalar reference  : rtol=1e-2 atol=1e-3 (fp16 alpha loses ~3
 *                                 significant bits per per-group factor)
 *   fp32α vs scalar reference  : rtol=1e-5 atol=1e-4 (FP32 throughout, only
 *                                 accumulation-order rounding noise)
 *   fp16α vs fp32α             : rtol=1e-2 atol=1e-3 (same fp16-quantized α)
 *
 * Exit codes follow geist test convention: 0 PASS, 99 ERROR, !=0 FAIL.
 */
#include "ptqtp_kernel.h"
#include "test_helpers.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Reference 2-plane decoder, scalar, FP32 throughout for comparison. */
static void ref_gemv_2plane(size_t        n_in,
                            size_t        n_out,
                            size_t        group_size,
                            const int8_t *x_q8,
                            float         scale_x,
                            const int8_t *T1_flat,
                            const int8_t *T2_flat,
                            const float  *alpha_fp32,
                            float        *y_out) {
    const size_t n_groups = n_in / group_size;
    for (size_t n = 0; n < n_out; n++) {
        float acc = 0.0f;
        for (size_t g = 0; g < n_groups; g++) {
            int32_t a1 = 0, a2 = 0;
            for (size_t k = 0; k < group_size; k++) {
                const size_t  col = g * group_size + k;
                const int32_t x_v = (int32_t) x_q8[col];
                a1 += (int32_t) T1_flat[n * n_in + col] * x_v;
                a2 += (int32_t) T2_flat[n * n_in + col] * x_v;
            }
            acc += alpha_fp32[n * n_groups * 2 + g * 2 + 0] * (float) a1 +
                   alpha_fp32[n * n_groups * 2 + g * 2 + 1] * (float) a2;
        }
        y_out[n] = scale_x * acc;
    }
}

/* Encode (T1, T2) into 4-bit joint nibble, 2 weights per byte.
 * idx = (T1+1)*3 + (T2+1), the code the 2-plane LUTs in
 * src/formats/ptqtp/ptqtp_kernel.c decode. */
static void
pack_trits(size_t n_out, size_t n_in, const int8_t *T1, const int8_t *T2, uint8_t *trits_out) {
    for (size_t n = 0; n < n_out; n++) {
        for (size_t j = 0; j < n_in; j += 2) {
            const uint8_t idx_lo =
                    (uint8_t) (((int) T1[n * n_in + j] + 1) * 3 + ((int) T2[n * n_in + j] + 1));
            const uint8_t idx_hi            = (uint8_t) (((int) T1[n * n_in + j + 1] + 1) * 3 +
                                                         ((int) T2[n * n_in + j + 1] + 1));
            trits_out[n * n_in / 2 + j / 2] = (uint8_t) ((idx_hi << 4) | idx_lo);
        }
    }
}

/* Pack two fp32 alphas as fp16. Bit-pattern conversion for portable build —
 * uses the same path the loader does. */
static uint16_t fp32_to_fp16_round_to_nearest(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    const uint32_t sign = (bits >> 31) & 0x1u;
    const int32_t  exp  = (int32_t) ((bits >> 23) & 0xFFu) - 127;
    const uint32_t frac = bits & 0x7FFFFFu;
    if (exp > 15) {
        return (uint16_t) ((sign << 15) | 0x7C00u);
    } /* +-Inf */
    if (exp < -14) {
        return (uint16_t) (sign << 15);
    } /* underflow */
    const uint16_t e16 = (uint16_t) (exp + 15);
    const uint16_t f16 = (uint16_t) (frac >> 13);
    return (uint16_t) ((sign << 15) | (e16 << 10) | f16);
}

static int run_one_shape(size_t n_in, size_t n_out, size_t group_size, unsigned seed) {
    if (n_in % group_size != 0) {
        fprintf(stderr, "  bad shape n_in=%zu gs=%zu\n", n_in, group_size);
        return GEIST_TEST_ERROR;
    }
    const size_t n_groups   = n_in / group_size;
    const size_t trit_bytes = n_out * n_in / 2;

    int8_t   *T1        = (int8_t *) malloc(n_out * n_in);
    int8_t   *T2        = (int8_t *) malloc(n_out * n_in);
    uint8_t  *trits     = (uint8_t *) aligned_alloc(64, trit_bytes);
    int8_t   *x_q8      = (int8_t *) aligned_alloc(64, n_in);
    float    *alpha_f32 = (float *) aligned_alloc(64, n_out * n_groups * 2 * sizeof(float));
    uint16_t *alpha_f16 = (uint16_t *) aligned_alloc(64, n_out * n_groups * 2 * sizeof(uint16_t));
    float    *y_ref     = (float *) aligned_alloc(64, n_out * sizeof(float));
    float    *y_fp32a   = (float *) aligned_alloc(64, n_out * sizeof(float));
    float    *y_fp16a   = (float *) aligned_alloc(64, n_out * sizeof(float));
    if (!T1 || !T2 || !trits || !x_q8 || !alpha_f32 || !alpha_f16 || !y_ref || !y_fp32a ||
        !y_fp16a) {
        fprintf(stderr, "  alloc fail\n");
        return GEIST_TEST_ERROR;
    }

    srand(seed);
    for (size_t i = 0; i < n_out * n_in; i++) {
        T1[i] = (int8_t) ((rand() % 3) - 1);
        T2[i] = (int8_t) ((rand() % 3) - 1);
    }
    pack_trits(n_out, n_in, T1, T2, trits);
    for (size_t i = 0; i < n_in; i++) {
        x_q8[i] = (int8_t) ((rand() % 256) - 128);
    }
    /* Alpha range matches realistic weight stats: ~ [-0.05, +0.05]. */
    for (size_t i = 0; i < n_out * n_groups * 2; i++) {
        const float a = ((float) rand() / (float) RAND_MAX - 0.5f) * 0.1f;
        alpha_f32[i]  = a;
        /* Round-trip through fp16 so fp16α path sees the exact same value
         * after promotion. Avoids spurious fp16-quantization deltas in the
         * fp16α-vs-fp32α gate. */
        alpha_f16[i] = fp32_to_fp16_round_to_nearest(a);
        uint16_t h   = alpha_f16[i];
        uint32_t s = (uint32_t) (h >> 15) & 1u, e = (uint32_t) (h >> 10) & 0x1Fu,
                 f    = (uint32_t) h & 0x3FFu;
        uint32_t bits = (e == 0)      ? (s << 31)
                        : (e == 0x1F) ? ((s << 31) | 0x7F800000u | (f << 13))
                                      : ((s << 31) | ((e + 112u) << 23) | (f << 13));
        memcpy(&alpha_f32[i], &bits, 4); /* now exactly the fp16 value */
    }
    const float scale_x = 0.005f;

    ref_gemv_2plane(n_in, n_out, group_size, x_q8, scale_x, T1, T2, alpha_f32, y_ref);
    ptqtp_gemv_2plane_fp32alpha(n_in, n_out, group_size, x_q8, scale_x, trits, alpha_f32, y_fp32a);
    ptqtp_gemv_2plane_fp16alpha(n_in, n_out, group_size, x_q8, scale_x, trits, alpha_f16, y_fp16a);

    int       fails = 0;
    ptrdiff_t i;
    if ((i = geist_fp32_close_array(y_fp32a, y_ref, n_out, 1e-5f, 1e-4f)) >= 0) {
        fprintf(stderr,
                "  fp32α vs ref FAIL at %td: got %g, want %g (delta %g)\n",
                i,
                y_fp32a[i],
                y_ref[i],
                y_fp32a[i] - y_ref[i]);
        fails++;
    }
    if ((i = geist_fp32_close_array(y_fp16a, y_ref, n_out, 1e-4f, 1e-4f)) >= 0) {
        fprintf(stderr,
                "  fp16α vs ref FAIL at %td: got %g, want %g (delta %g)\n",
                i,
                y_fp16a[i],
                y_ref[i],
                y_fp16a[i] - y_ref[i]);
        fails++;
    }
    if ((i = geist_fp32_close_array(y_fp16a, y_fp32a, n_out, 1e-5f, 1e-5f)) >= 0) {
        fprintf(stderr,
                "  fp16α vs fp32α FAIL at %td: got %g, want %g (delta %g)\n",
                i,
                y_fp16a[i],
                y_fp32a[i],
                y_fp16a[i] - y_fp32a[i]);
        fails++;
    }

    free(T1);
    free(T2);
    free(trits);
    free(x_q8);
    free(alpha_f32);
    free(alpha_f16);
    free(y_ref);
    free(y_fp32a);
    free(y_fp16a);
    return fails ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
}

int main(void) {
    /* Three representative shapes: small (1 group), medium (8 groups),
     * Gemma-class (12 groups). */
    const struct {
        size_t      n_in, n_out, gs;
        const char *label;
    } cases[] = {
            {128, 16, 128, "tiny      (128×16,   1 group)"},
            {1024, 32, 128, "medium    (1024×32,  8 groups)"},
            {1536, 64, 128, "gemma4-q  (1536×64, 12 groups)"},
    };
    const size_t n_cases = sizeof(cases) / sizeof(cases[0]);

    int worst = GEIST_TEST_PASS;
    for (size_t c = 0; c < n_cases; c++) {
        printf("  %s\n", cases[c].label);
        const int rc =
                run_one_shape(cases[c].n_in, cases[c].n_out, cases[c].gs, (unsigned) (42 + c));
        if (rc != GEIST_TEST_PASS) {
            worst = rc;
        }
    }
    if (worst == GEIST_TEST_PASS) {
        printf("  PASS — all 2-plane PTQTP kernel variants match scalar ref\n");
    }
    return worst;
}

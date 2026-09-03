/* Both-branches parity matrix for every calibrated tunable: whatever
 * value a calibration ever applies, BOTH kernel paths behind the knob
 * must agree numerically. Each knob's branches are forced via its env
 * override (read at backend create), the same synthetic panel is
 * resolved under each, and the mN outputs are compared within the
 * W*A8-vs-f32 tolerance the existing parity tests use. This is what
 * makes calibration quality-safe by construction: it may only ever
 * pick between paths CI has proven equivalent — on every CI µarch. */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>

#include <stdlib.h>
#include <string.h>

static int  g_fail = 0;
static void check(bool ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        g_fail = 1;
    }
}

enum { N_IN = 1024, N_OUT = 128, M = 33 }; /* odd m: tile tails included */

static void fill_blocks(uint8_t *dst, size_t rows, size_t block_bytes, size_t blocks_per_row) {
    uint32_t lcg = 0xBEEF5u;
    for (size_t r = 0; r < rows; r++) {
        for (size_t b = 0; b < blocks_per_row; b++) {
            uint8_t *blk = dst + (r * blocks_per_row + b) * block_bytes;
            for (size_t i = 0; i < block_bytes; i++) {
                lcg    = lcg * 1664525u + 1013904223u;
                blk[i] = (uint8_t) (lcg >> 24);
            }
            blk[0] = 0x00; /* d = fp16 1.0, dmin/scales_h zeroed */
            blk[1] = 0x3C;
            blk[2] = 0x00;
            blk[3] = 0x00;
        }
    }
}

/* Resolve the panel under the given env setting and run mN once. */
static bool run_variant(const char      *env_name,
                        const char      *env_value,
                        enum geist_dtype dtype,
                        const uint8_t   *wbuf,
                        size_t           wbytes,
                        const float     *x,
                        float           *y) {
    setenv(env_name, env_value, 1);
    struct geist_backend *be = nullptr;
    const bool            ok = geist_backend_create("cpu_neon", nullptr, nullptr, &be) == GEIST_OK;
    unsetenv(env_name);
    if (!ok) {
        return false;
    }
    struct geist_weight w = {
            .raw        = wbuf,
            .raw_nbytes = wbytes,
            .n_in       = N_IN,
            .n_out      = N_OUT,
            .dtype      = dtype,
    };
    bool ran = be->desc->vtbl->resolve_weight(be, &w) == GEIST_OK && w.linear_mN != nullptr;
    if (ran) {
        w.linear_mN(M, x, &w, be, y);
    }
    if ((w.flags & GEIST_W_AUX_HEAP_OWNED) != 0 && w.aux_fp32 != nullptr) {
        free((void *) (uintptr_t) w.aux_fp32);
    }
    geist_backend_destroy(be);
    return ran;
}

static void compare(const char *knob, const float *a, const float *b) {
    float max_d = 0.0f, mag = 0.0f;
    for (size_t i = 0; i < (size_t) M * N_OUT; i++) {
        const float d = a[i] - b[i];
        max_d         = fabsf(d) > max_d ? fabsf(d) : max_d;
        mag           = fabsf(a[i]) > mag ? fabsf(a[i]) : mag;
    }
    char msg[128];
    snprintf(msg, sizeof msg, "%s: both branches agree (max|d| within a8 tolerance)", knob);
    check(max_d <= 2e-2f * (mag > 1.0f ? mag : 1.0f), msg);
}

static void
matrix_case(const char *env_name, enum geist_dtype dtype, size_t block_bytes, size_t block_elems) {
    const size_t blocks_per_row = N_IN / block_elems;
    const size_t wbytes         = N_OUT * blocks_per_row * block_bytes;
    uint8_t     *wbuf           = malloc(wbytes);
    float       *x              = malloc((size_t) M * N_IN * sizeof(float));
    float       *y0             = malloc((size_t) M * N_OUT * sizeof(float));
    float       *y1             = malloc((size_t) M * N_OUT * sizeof(float));
    if (wbuf == nullptr || x == nullptr || y0 == nullptr || y1 == nullptr) {
        check(false, "alloc");
        return;
    }
    fill_blocks(wbuf, N_OUT, block_bytes, blocks_per_row);
    for (size_t i = 0; i < (size_t) M * N_IN; i++) {
        x[i] = (float) ((int) (i % 23u) - 11) * 0.125f;
    }
    if (run_variant(env_name, "0", dtype, wbuf, wbytes, x, y0) &&
        run_variant(env_name, "1", dtype, wbuf, wbytes, x, y1)) {
        compare(env_name, y0, y1);
    } else {
        check(false, env_name);
    }
    free(wbuf);
    free(x);
    free(y0);
    free(y1);
}

int main(void) {
    struct geist_backend *probe = nullptr;
    if (geist_backend_create("cpu_neon", nullptr, nullptr, &probe) != GEIST_OK) {
        printf("test_calibration_paths_unit: SKIP (no cpu_neon backend)\n");
        return 0;
    }
    geist_backend_destroy(probe);

    matrix_case("GEIST_Q5K_NATIVE_MN", GEIST_DTYPE_Q5_K, 176, 256);
    matrix_case("GEIST_Q8_0_NATIVE_MN", GEIST_DTYPE_Q8_0, 34, 32);
    matrix_case("GEIST_Q4_01_NATIVE_MN", GEIST_DTYPE_Q4_0, 18, 32);
    matrix_case("GEIST_IQ4XS_NATIVE_MN", GEIST_DTYPE_IQ4_XS, 136, 256);
    /* qk_sgemm_threshold: both extremes of the Q4_K prefill route. */
    {
        const size_t wbytes = N_OUT * (N_IN / 256) * 144;
        uint8_t     *wbuf   = malloc(wbytes);
        float       *x      = malloc((size_t) M * N_IN * sizeof(float));
        float       *y0     = malloc((size_t) M * N_OUT * sizeof(float));
        float       *y1     = malloc((size_t) M * N_OUT * sizeof(float));
        fill_blocks(wbuf, N_OUT, 144, N_IN / 256);
        for (size_t i = 0; i < (size_t) M * N_IN; i++) {
            x[i] = (float) ((int) (i % 23u) - 11) * 0.125f;
        }
        if (run_variant("GEIST_QK_SGEMM_THRESHOLD", "1", GEIST_DTYPE_Q4_K, wbuf, wbytes, x, y0) &&
            run_variant(
                    "GEIST_QK_SGEMM_THRESHOLD", "100000", GEIST_DTYPE_Q4_K, wbuf, wbytes, x, y1)) {
            compare("GEIST_QK_SGEMM_THRESHOLD", y0, y1);
        } else {
            check(false, "GEIST_QK_SGEMM_THRESHOLD");
        }
        free(wbuf);
        free(x);
        free(y0);
        free(y1);
    }

    if (g_fail == 0) {
        printf("test_calibration_paths_unit: all checks passed\n");
    }
    return g_fail == 0 ? 0 : 1;
}

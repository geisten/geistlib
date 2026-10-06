/*
 * test_linear_ref_unit — geist_linear_ref, the linear kernel that cannot
 * fail, and cpu_scalar's kernels, which are it.
 *
 * geist_linear_ref (backends/common/linear_ref.c) decodes a weight row a
 * 1024-element tile at a time into a stack buffer and dots it in double.
 * For every dtype it decodes, against a whole-row decode with the quant.h
 * row codecs and a double dot written here:
 *   - rows of 2 whole tiles (2048), and rows of a tile and one block, whose
 *     last tile is a single block;
 *   - m = 1, 3 and 70 (more rows than it accumulates at once, 64).
 * The results agree to 1 ulp: -ffast-math lets the compiler reassociate
 * the double sums, which may round the float result the other way.
 *
 * cpu_scalar's linear_m1 / linear_mN for those dtypes must give the same
 * bits as geist_linear_ref and allocate nothing (geist_weight.h has them
 * allocation-free), so a failed allocation can never leave y unwritten.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>

#include "heap.h"
#include "linear_ref.h"
#include "quant.h"
#include "quant_fixtures.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

constexpr size_t N_OUT = 5;

/* Row j of the weight, whole, decoded here. */
static void decode_row(size_t d, size_t n_in, const uint8_t *raw, size_t j, float *out) {
    const size_t e0 = j * n_in;
    switch (DTYPES[d].dt) {
    case GEIST_DTYPE_F32:
        memcpy(out, raw + 4 * e0, n_in * 4);
        return;
    case GEIST_DTYPE_F16:
    case GEIST_DTYPE_BF16:
        for (size_t i = 0; i < n_in; i++) {
            const uint16_t h = (uint16_t) (raw[2 * (e0 + i)] | (raw[2 * (e0 + i) + 1] << 8));
            if (DTYPES[d].dt == GEIST_DTYPE_F16) {
                out[i] = fp16_to_fp32(h);
            } else {
                const uint32_t b = (uint32_t) h << 16;
                memcpy(&out[i], &b, 4);
            }
        }
        return;
    case GEIST_DTYPE_I2_S: {
        float scale;
        memcpy(&scale, raw + i2_s_scale_offset(n_in * N_OUT), sizeof scale);
        for (size_t e = 0; e < n_in; e++) {
            /* Element e of a 256-block: byte h*32 + e%32 of its 64, h =
             * (e%256)/128, at shift 6 - 2*((e%128)/32). */
            const size_t  bl = e / 256, r = e % 256;
            const uint8_t byte = raw[j * (n_in / 4) + bl * 64 + (r / 128) * 32 + r % 32];
            const int     trit = (int) ((byte >> (6 - 2 * ((r % 128) / 32))) & 3) - 1;
            out[e]             = (float) trit * scale;
        }
        return;
    }
    default:
        DTYPES[d].row(n_in, raw + e0 / DTYPES[d].blk * DTYPES[d].bytes, out);
        return;
    }
}

/* ulps between two finite floats of the same sign; a large number when
 * their signs differ and they are not both zero. */
static uint32_t ulps(float a, float b) {
    if (a == b) {
        return 0;
    }
    int32_t ia, ib;
    memcpy(&ia, &a, 4);
    memcpy(&ib, &b, 4);
    if ((ia < 0) != (ib < 0)) {
        return UINT32_MAX;
    }
    return (uint32_t) (ia > ib ? ia - ib : ib - ia);
}

static int run_case(size_t d, size_t n_in, size_t m, struct geist_backend *scalar) {
    int      fails = 0;
    char     msg[192];
    size_t   bytes = 0;
    uint8_t *raw   = make_weight(d, n_in, N_OUT, &bytes);
    float   *x     = malloc(m * n_in * sizeof *x);
    float   *want  = malloc(m * N_OUT * sizeof *want);
    float   *got   = malloc(m * N_OUT * sizeof *got);
    float   *ker   = malloc(m * N_OUT * sizeof *ker);
    float   *row   = malloc(n_in * sizeof *row);
    if (raw == nullptr || x == nullptr || want == nullptr || got == nullptr || ker == nullptr ||
        row == nullptr) {
        fprintf(stderr, "FAIL: %s: test allocation\n", DTYPES[d].name);
        fails = 1;
        goto out;
    }
    for (size_t i = 0; i < m * n_in; i++) {
        x[i] = next_f();
    }
    for (size_t j = 0; j < N_OUT; j++) {
        decode_row(d, n_in, raw, j, row);
        for (size_t i = 0; i < m; i++) {
            double acc = 0.0;
            for (size_t k = 0; k < n_in; k++) {
                acc += (double) x[i * n_in + k] * (double) row[k];
            }
            want[i * N_OUT + j] = (float) acc;
        }
    }

    const struct geist_weight w = {.raw        = raw,
                                   .raw_nbytes = bytes,
                                   .n_in       = (int32_t) n_in,
                                   .n_out      = (int32_t) N_OUT,
                                   .dtype      = (uint16_t) DTYPES[d].dt};
    snprintf(msg, sizeof msg, "%s decodes", DTYPES[d].name);
    fails += geist_expect(geist_linear_ref_decodes(w.dtype), msg);
    geist_linear_ref(m, x, &w, got);
    uint32_t worst = 0;
    for (size_t i = 0; i < m * N_OUT; i++) {
        const uint32_t u = ulps(got[i], want[i]);
        worst            = u > worst ? u : worst;
    }
    snprintf(msg,
             sizeof msg,
             "%s n_in=%zu m=%zu: reference within 1 ulp of a whole-row decode (worst %u)",
             DTYPES[d].name,
             n_in,
             m,
             worst);
    fails += geist_expect(worst <= 1, msg);

    if (scalar != nullptr) {
        struct geist_weight ws = w;
        const bool          ok = scalar->desc->vtbl->resolve_weight(scalar, &ws) == GEIST_OK;
        snprintf(msg, sizeof msg, "cpu_scalar resolves %s", DTYPES[d].name);
        fails += geist_expect(ok, msg);
        if (ok) {
            const uint64_t a0 = heap_alloc_count();
            if (m == 1) {
                ws.linear_m1(x, &ws, scalar, ker);
            } else {
                ws.linear_mN(m, x, &ws, scalar, ker);
            }
            const uint64_t a1 = heap_alloc_count();
            snprintf(msg,
                     sizeof msg,
                     "cpu_scalar %s m=%zu: the same bits as the reference",
                     DTYPES[d].name,
                     m);
            fails += geist_expect(memcmp(ker, got, m * N_OUT * sizeof *ker) == 0, msg);
            snprintf(msg,
                     sizeof msg,
                     "cpu_scalar %s m=%zu: no allocation (got %llu)",
                     DTYPES[d].name,
                     m,
                     (unsigned long long) (a1 - a0));
            fails += geist_expect(a1 == a0, msg);
        }
    }
out:
    free(raw);
    free(x);
    free(want);
    free(got);
    free(ker);
    free(row);
    return fails;
}

int main(void) {
    struct geist_backend *scalar = nullptr;
    if (geist_backend_create("cpu_scalar", nullptr, nullptr, &scalar) != GEIST_OK) {
        scalar = nullptr; /* not in this build: the reference is still checked */
    }
    int fails = 0;
    for (size_t d = 0; d < N_DTYPES; d++) {
        /* Two whole tiles; a tile and one block. */
        const size_t widths[] = {2048, 1024 + DTYPES[d].blk};
        const size_t ms[]     = {1, 3, 70};
        for (size_t wi = 0; wi < 2; wi++) {
            for (size_t mi = 0; mi < 3; mi++) {
                fails += run_case(d, widths[wi], ms[mi], scalar);
            }
        }
    }
    if (scalar != nullptr) {
        geist_backend_destroy(scalar);
    }
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: geist_linear_ref matches a whole-row decode for every dtype it decodes; "
           "cpu_scalar's kernels are it and allocate nothing\n");
    return GEIST_TEST_PASS;
}

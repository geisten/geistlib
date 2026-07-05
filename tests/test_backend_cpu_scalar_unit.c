/*
 * test_backend_cpu_scalar_unit — walking-skeleton smoke test for the
 * cpu_scalar backend (Phase B-2a).
 *
 * Verifies:
 *   - geist_backend_create("cpu_scalar") succeeds
 *   - backend name + version helpers report sensible values
 *   - buffer_create / upload / download / map / destroy round-trip
 *   - geist_backend_destroy frees the handle without leaking
 *
 * Real op tests (linear etc.) land in B-2b once the kernels are ported.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_util.h>
#include <geist_backend.h>
#include <geist_weight.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int check(bool cond, const char *what) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", what);
        return 1;
    }
    return 0;
}

int main(void) {
    int fails = 0;

    /* Version sanity. */
    int major = -1, minor = -1, patch = -1;
    geist_version_components(&major, &minor, &patch);
    fails += check(major == GEIST_VERSION_MAJOR, "version major matches macro");
    fails += check(minor == GEIST_VERSION_MINOR, "version minor matches macro");
    fails += check(patch == GEIST_VERSION_PATCH, "version patch matches macro");
    fails += check(strcmp(geist_version_string(), GEIST_VERSION_STRING) == 0,
                   "version string matches macro");

    /* Explicit cpu_scalar — guaranteed available since it's the default backend. */
    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create("cpu_scalar", nullptr, nullptr, &be);
    fails += check(s == GEIST_OK, "geist_backend_create(\"cpu_scalar\") returns OK");
    fails += check(be != nullptr, "backend handle is non-null");
    if (s != GEIST_OK || be == nullptr) {
        fprintf(stderr, "  create-time error: %s\n", geist_last_create_error());
        return GEIST_TEST_FAIL;
    }

    fails += check(strcmp(geist_backend_name(be), "cpu_scalar") == 0,
                   "explicit name returns cpu_scalar");
    fails += check(geist_backend_errcode(be) == GEIST_OK, "errcode is OK after create");

    /* Buffer round-trip: 64-byte ACTIVATION buffer. */
    struct geist_buffer *buf = nullptr;
    s = be->desc->vtbl->buffer_create(be, 64, GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &buf);
    fails += check(s == GEIST_OK, "buffer_create OK");
    fails += check(buf != nullptr, "buffer handle non-null");

    if (buf != nullptr) {
        uint8_t src[64];
        uint8_t dst[64];
        for (size_t i = 0; i < 64; i++) {
            src[i] = (uint8_t) (i * 3 + 7);
        }
        s = be->desc->vtbl->buffer_upload(buf, 64, src);
        fails += check(s == GEIST_OK, "buffer_upload OK");

        /* map should give back a non-null host pointer for CPU backend. */
        void *host = be->desc->vtbl->buffer_map(buf);
        fails += check(host != nullptr, "buffer_map returns host pointer");
        if (host != nullptr) {
            fails += check(memcmp(host, src, 64) == 0, "host alias matches uploaded data");
        }
        be->desc->vtbl->buffer_unmap(buf);

        s = be->desc->vtbl->buffer_download(64, dst, buf);
        fails += check(s == GEIST_OK, "buffer_download OK");
        fails += check(memcmp(dst, src, 64) == 0, "downloaded data matches");

        be->desc->vtbl->buffer_destroy(be, buf);
    }

    /* Capability query: F32 DENSE LINEAR is NATIVE; an unknown (CUSTOM)
     * block-quantized dtype the resolver has no kernel for is NONE. */
    struct geist_op_support_query q_f32 = {
            .op          = GEIST_OP_LINEAR,
            .input_count = 2,
            .inputs =
                    {
                            {.dtype = GEIST_DTYPE_F32, .layout = GEIST_LAYOUT_DENSE},
                            {.dtype = GEIST_DTYPE_F32, .layout = GEIST_LAYOUT_DENSE},
                    },
            .output_count = 1,
            .outputs      = {{.dtype = GEIST_DTYPE_F32, .layout = GEIST_LAYOUT_DENSE}},
    };
    fails += check(geist_backend_supports_op(be, &q_f32) == GEIST_SUPPORT_NATIVE,
                   "supports_op(LINEAR, F32 DENSE) is NATIVE");

    struct geist_op_support_query q_q4k = {
            .op          = GEIST_OP_LINEAR,
            .input_count = 2,
            .inputs =
                    {
                            {.dtype = GEIST_DTYPE_F32, .layout = GEIST_LAYOUT_DENSE},
                            {.dtype = GEIST_DTYPE_CUSTOM, .layout = GEIST_LAYOUT_BLOCK_QUANTIZED},
                    },
    };
    fails += check(geist_backend_supports_op(be, &q_q4k) == GEIST_SUPPORT_NONE,
                   "supports_op(LINEAR, unknown CUSTOM dtype) is NONE");

    /* Real F32 DENSE matmul: y = x @ W^T
     *   x: (2, 3) =  [[1, 2, 3], [4, 5, 6]]
     *   W: (4, 3) =  [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]
     *   Expected y: (2, 4) =  [[1, 2, 3,  6], [4, 5, 6, 15]] */
    const size_t         M = 2, KK = 3, N = 4;
    struct geist_buffer *bx = nullptr, *bw = nullptr, *by = nullptr;
    s = be->desc->vtbl->buffer_create(
            be, M * KK * sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &bx);
    fails += check(s == GEIST_OK, "buffer_create x OK");
    s = be->desc->vtbl->buffer_create(
            be, N * KK * sizeof(float), GEIST_BUFFER_WEIGHT, GEIST_MEMORY_AUTO, &bw);
    fails += check(s == GEIST_OK, "buffer_create w OK");
    s = be->desc->vtbl->buffer_create(
            be, M * N * sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &by);
    fails += check(s == GEIST_OK, "buffer_create y OK");

    if (bx && bw && by) {
        const float xdata[]          = {1, 2, 3, 4, 5, 6};
        const float wdata[]          = {1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1};
        const float ydata_expected[] = {1, 2, 3, 6, 4, 5, 6, 15};

        s = be->desc->vtbl->buffer_upload(bx, sizeof(xdata), (const uint8_t *) xdata);
        fails += check(s == GEIST_OK, "upload x OK");
        s = be->desc->vtbl->buffer_upload(bw, sizeof(wdata), (const uint8_t *) wdata);
        fails += check(s == GEIST_OK, "upload w OK");

        /* P2-final: linear vtbl slot is gone. Route through the resolver
         * and call the installed wkr.linear_mN function pointer. */
        void *w_host = be->desc->vtbl->buffer_map(bw);
        fails += check(w_host != nullptr, "buffer_map w OK");
        struct geist_weight wkr = {
                .raw   = w_host,
                .n_in  = (int32_t) KK,
                .n_out = (int32_t) N,
                .dtype = (uint16_t) GEIST_DTYPE_F32,
        };
        s = be->desc->vtbl->resolve_weight(be, &wkr);
        fails += check(s == GEIST_OK && wkr.linear_mN != nullptr,
                       "resolve_weight(F32 DENSE) installs linear_mN");
        be->desc->vtbl->buffer_unmap(bw);

        void *x_host = be->desc->vtbl->buffer_map(bx);
        void *y_host = be->desc->vtbl->buffer_map(by);
        fails += check(x_host != nullptr && y_host != nullptr, "map x,y OK");
        if (wkr.linear_mN != nullptr && x_host != nullptr && y_host != nullptr) {
            wkr.linear_mN((const float *) x_host, &wkr, M, be, (float *) y_host);
        }
        be->desc->vtbl->buffer_unmap(bx);
        be->desc->vtbl->buffer_unmap(by);

        float yres[8];
        s = be->desc->vtbl->buffer_download(sizeof(yres), (uint8_t *) yres, by);
        fails += check(s == GEIST_OK, "download y OK");

        for (size_t i = 0; i < 8; i++) {
            if (!geist_fp32_close(yres[i], ydata_expected[i], 1e-6f, 1e-7f)) {
                fprintf(stderr,
                        "  y[%zu] = %.6f, expected %.6f\n",
                        i,
                        (double) yres[i],
                        (double) ydata_expected[i]);
                fails++;
            }
        }
        if (fails == 0) {
            printf("  matmul output matches reference (8 elements)\n");
        }

        be->desc->vtbl->buffer_destroy(be, bx);
        be->desc->vtbl->buffer_destroy(be, bw);
        be->desc->vtbl->buffer_destroy(be, by);
    }

    /* Bad-name lookup. */
    struct geist_backend *be_bad = nullptr;
    s = geist_backend_create("does_not_exist", nullptr, nullptr, &be_bad);
    fails += check(s == GEIST_E_NOT_FOUND, "unknown backend name returns NOT_FOUND");
    fails += check(be_bad == nullptr, "out param zeroed on error");
    fails += check(strstr(geist_last_create_error(), "does_not_exist") != nullptr,
                   "create-time error mentions requested name");

    geist_backend_destroy(be);

    if (fails == 0) {
        printf("PASS: backend cpu_scalar walking skeleton (%s)\n", geist_version_string());
        return GEIST_TEST_PASS;
    }
    fprintf(stderr, "FAILED: %d check(s) failed\n", fails);
    return GEIST_TEST_FAIL;
}

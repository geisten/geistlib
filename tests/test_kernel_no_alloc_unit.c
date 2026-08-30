/*
 * test_kernel_no_alloc_unit — the allocation-free linear-kernel contract
 * (issue #336).
 *
 * geist_weight.h states it plainly:
 *
 *     Hot-path contract:
 *       - linear_m1 / linear_mN must be allocation-free.
 *
 * It was not true. The resolver wrappers discarded their `be` argument
 * and called the convenience entry points in quant.h, which malloc a
 * per-call activation buffer — some of them dereferencing the result
 * without checking it first.
 *
 * The fix routes them through the per-thread cpu_neon workspace, which
 * grows once and is reused. This test is what keeps it that way: run a
 * kernel many times over growing shapes and assert the workspace
 * capacity settles — i.e. the steady state allocates nothing.
 *
 * Observing "no malloc" portably is not possible without interposition,
 * so the observable proxy is the workspace itself: capacity must stop
 * changing, and the buffer pointer must stop moving, after the first
 * call at the largest shape.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>

#include <stdio.h>
#include <math.h>
#include <string.h>

#if !defined(GEIST_BACKEND_CPU_NEON)
int main(void) {
    printf("SKIP: cpu_neon backend not in this build\n");
    return GEIST_TEST_SKIP;
}
#else

#define GEIST_INTERNAL_BACKEND_LAYER
#include "../src/backends/cpu_neon/internal.h"

#include "quant.h"

static int g_fail = 0;

#define CHECK(cond, msg)                          \
    do {                                          \
        if (!(cond)) {                            \
            fprintf(stderr, "FAIL: %s\n", (msg)); \
            g_fail = 1;                           \
        }                                         \
    } while (0)

/* Weight blob for [n_out, n_in], contents irrelevant — this test is about
 * allocation behaviour, not numerics (the parity suites cover that). */
static void *make_blob(size_t n_in, size_t n_out, size_t blk_elems, size_t blk_bytes) {
    const size_t nb    = n_in / blk_elems;
    const size_t bytes = n_out * nb * blk_bytes;
    uint8_t     *blob  = heap_alloc_array_aligned(uint8_t, bytes);
    if (blob != nullptr) {
        for (size_t i = 0; i < bytes; i++) {
            blob[i] = (uint8_t) (i * 31u + 7u);
        }
    }
    return blob;
}

struct dtype_case {
    const char *name;
    uint16_t    dtype;
    size_t      blk_elems;
    size_t      blk_bytes;
};

/* Every quantized dtype cpu_neon resolves to a native W*A8 kernel. A dtype
 * the host does not resolve is reported and skipped, not silently passed —
 * on macOS some M>1 paths are rebound to the dequant + SGEMM trampoline
 * (apply_resolver_post_hooks), which is why the assertion below covers the
 * WHOLE workspace and not just the activation scratch: the trampoline has
 * its own fields, and they must be just as stable. */
static const struct dtype_case CASES[] = {
        {"Q3_K", (uint16_t) GEIST_DTYPE_Q3_K, Q3_K_BLOCK_ELEMS, Q3_K_BLOCK_BYTES},
        {"Q5_K", (uint16_t) GEIST_DTYPE_Q5_K, Q5_K_BLOCK_ELEMS, Q5_K_BLOCK_BYTES},
        {"Q8_0", (uint16_t) GEIST_DTYPE_Q8_0, Q8_0_BLOCK_ELEMS, Q8_0_BLOCK_BYTES},
        {"Q4_0", (uint16_t) GEIST_DTYPE_Q4_0, Q4_0_BLOCK_ELEMS, Q4_0_BLOCK_BYTES},
        {"Q4_1", (uint16_t) GEIST_DTYPE_Q4_1, Q4_1_BLOCK_ELEMS, Q4_1_BLOCK_BYTES},
        {"IQ4_XS", (uint16_t) GEIST_DTYPE_IQ4_XS, IQ4_XS_BLOCK_ELEMS, IQ4_XS_BLOCK_BYTES},
        {"IQ2_S", (uint16_t) GEIST_DTYPE_IQ2_S, IQ2_S_BLOCK_ELEMS, IQ2_S_BLOCK_BYTES},
        {"IQ3_S", (uint16_t) GEIST_DTYPE_IQ3_S, IQ3_S_BLOCK_ELEMS, IQ3_S_BLOCK_BYTES},
};

constexpr size_t N_IN  = 512;
constexpr size_t N_OUT = 64;
constexpr size_t M_MAX = 8;

/* Returns true if the dtype was exercised, false if the host does not
 * resolve it (reported by the caller). */
static bool run_case(struct geist_backend *be, const struct dtype_case *c, float *x, float *y) {
    void *w_raw = make_blob(N_IN, N_OUT, c->blk_elems, c->blk_bytes);
    if (w_raw == nullptr) {
        CHECK(false, "blob allocation");
        return false;
    }
    struct geist_weight w = {
            .raw        = w_raw,
            .raw_nbytes = (N_IN / c->blk_elems) * c->blk_bytes * N_OUT,
            .n_in       = (int32_t) N_IN,
            .n_out      = (int32_t) N_OUT,
            .dtype      = c->dtype,
    };
    void *freeme = w_raw;
    if (be->desc->vtbl->resolve_weight(be, &w) != GEIST_OK || w.linear_m1 == nullptr ||
        w.linear_mN == nullptr) {
        safe_free(&freeme);
        return false;
    }

    struct cpu_neon_state     *st = (struct cpu_neon_state *) be->state;
    struct cpu_neon_workspace *ws = cpu_neon_ws(st);
    if (ws == nullptr) {
        CHECK(false, "workspace available");
        safe_free(&freeme);
        return false;
    }

    /* Warm up BOTH paths at their largest shape before snapshotting: the
     * workspace is only at its high-water mark once each path has run
     * once. m1 and mN do not necessarily share fields — a dtype whose mN
     * is rebound to the dequant trampoline grows nothing the m1 kernel
     * needs, so snapshotting after mN alone would blame the first m1 call
     * for a growth that is legitimately its first. */
    w.linear_mN(M_MAX, x, &w, be, y);
    w.linear_m1(x, &w, be, y);
    const struct cpu_neon_workspace snapshot = *ws;

    /* Steady state: 200 decode calls and every prefill shape up to the
     * high-water mark. None of them may move a buffer or change a
     * capacity — that is the allocation-free contract, observable. */
    for (int rep = 0; rep < 200; rep++) {
        w.linear_m1(x, &w, be, y);
    }
    for (size_t m = 1; m <= M_MAX; m++) {
        w.linear_mN(m, x, &w, be, y);
    }
    if (memcmp(&snapshot, ws, sizeof snapshot) != 0) {
        const unsigned char *a = (const unsigned char *) &snapshot;
        const unsigned char *b = (const unsigned char *) ws;
        for (size_t k = 0; k < sizeof snapshot; k++)
            if (a[k] != b[k]) {
                fprintf(stderr, "  diff bei Offset %zu\n", k);
                break;
            }
        fprintf(stderr, "FAIL: %s: workspace changed in steady state\n", c->name);
        g_fail = 1;
    }

    /* And the kernel actually ran: poison the output, call once more, and
     * require every element to have been written. Not a numeric check —
     * the blob is random bytes, so its fp16 scales decode to whatever they
     * decode to; parity is the parity suites' job. This only rules out
     * "the workspace looked stable because nothing was executed". */
    constexpr float POISON = -7.5e30f;
    for (size_t i = 0; i < N_OUT; i++) {
        y[i] = POISON;
    }
    w.linear_m1(x, &w, be, y);
    size_t written = 0;
    for (size_t i = 0; i < N_OUT; i++) {
        written += (y[i] != POISON);
    }
    if (written != N_OUT) {
        fprintf(stderr, "FAIL: %s: kernel did not write its whole output row\n", c->name);
        g_fail = 1;
    }
    if (memcmp(&snapshot, ws, sizeof snapshot) != 0) {
        fprintf(stderr, "FAIL: %s: final call changed the workspace\n", c->name);
        g_fail = 1;
    }
    safe_free(&freeme);
    return true;
}

int main(void) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create("cpu_neon", nullptr, nullptr, &be) != GEIST_OK || be == nullptr) {
        printf("SKIP: no cpu_neon backend: %s\n", geist_last_create_error());
        return GEIST_TEST_SKIP;
    }
    float *x = heap_alloc_array_aligned(float, M_MAX *N_IN);
    float *y = heap_alloc_array_aligned(float, M_MAX *N_OUT);
    if (x == nullptr || y == nullptr) {
        geist_backend_destroy(be);
        return GEIST_TEST_ERROR;
    }
    for (size_t i = 0; i < M_MAX * N_IN; i++) {
        x[i] = (float) ((i % 61) - 30) * 0.05f;
    }

    size_t ran = 0;
    for (size_t i = 0; i < sizeof CASES / sizeof CASES[0]; i++) {
        if (run_case(be, &CASES[i], x, y)) {
            ran++;
        } else {
            printf("  (%s not resolved on this host — skipped)\n", CASES[i].name);
        }
    }
    CHECK(ran > 0, "no dtype was exercised at all");

    safe_free((void **) &x);
    safe_free((void **) &y);
    geist_backend_destroy(be);

    if (g_fail) {
        return GEIST_TEST_FAIL;
    }
    printf("PASS: %zu dtypes, linear_m1/linear_mN allocate nothing in steady state "
           "(200 decode + %zu prefill shapes each, whole workspace stable)\n",
           ran,
           M_MAX);
    return GEIST_TEST_PASS;
}

#endif /* GEIST_BACKEND_CPU_NEON */

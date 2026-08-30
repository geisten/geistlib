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

/* Q3_K weight blob for [n_out, n_in], contents irrelevant — this test is
 * about allocation behaviour, not numerics (the parity suites cover that).
 *
 * Q3_K on purpose: Q8_0's and Q4_0's M>1 paths are rebound to the dequant
 * + SGEMM trampoline on hosts where Accelerate wins the crossover
 * (apply_resolver_post_hooks), so on macOS they would not exercise the
 * kernel under test at all. Q3_K keeps its native kernel on every host. */
static void *make_q3_K(size_t n_in, size_t n_out) {
    const size_t nb    = n_in / Q3_K_BLOCK_ELEMS;
    const size_t bytes = n_out * nb * Q3_K_BLOCK_BYTES;
    uint8_t     *blob  = heap_alloc_array_aligned(uint8_t, bytes);
    if (blob != nullptr) {
        for (size_t i = 0; i < bytes; i++) {
            blob[i] = (uint8_t) (i * 31u + 7u);
        }
    }
    return blob;
}

int main(void) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create("cpu_neon", nullptr, nullptr, &be) != GEIST_OK || be == nullptr) {
        printf("SKIP: no cpu_neon backend: %s\n", geist_last_create_error());
        return GEIST_TEST_SKIP;
    }

    /* One declarator each — gcc rejects the compound form (AGENT.md §1). */
    constexpr size_t N_IN  = 512;
    constexpr size_t N_OUT = 64;
    constexpr size_t M_MAX = 8;
    void            *w_raw = make_q3_K(N_IN, N_OUT);
    float           *x     = heap_alloc_array_aligned(float, M_MAX *N_IN);
    float           *y     = heap_alloc_array_aligned(float, M_MAX *N_OUT);
    if (w_raw == nullptr || x == nullptr || y == nullptr) {
        geist_backend_destroy(be);
        return GEIST_TEST_ERROR;
    }
    for (size_t i = 0; i < M_MAX * N_IN; i++) {
        x[i] = (float) ((i % 61) - 30) * 0.05f;
    }

    struct geist_weight w = {
            .raw        = w_raw,
            .raw_nbytes = (N_IN / Q3_K_BLOCK_ELEMS) * Q3_K_BLOCK_BYTES * N_OUT,
            .n_in       = (int32_t) N_IN,
            .n_out      = (int32_t) N_OUT,
            .dtype      = (uint16_t) GEIST_DTYPE_Q3_K,
    };
    CHECK(be->desc->vtbl->resolve_weight(be, &w) == GEIST_OK, "resolve_weight(Q3_K)");
    CHECK(w.linear_m1 != nullptr && w.linear_mN != nullptr, "kernels installed");
    if (g_fail) {
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    struct cpu_neon_state     *st = (struct cpu_neon_state *) be->state;
    struct cpu_neon_workspace *ws = cpu_neon_ws(st);
    CHECK(ws != nullptr, "workspace available");
    if (ws == nullptr) {
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    /* Warm up at the LARGEST shape first: after this the workspace is at
     * its high-water mark and nothing below it may reallocate. */
    w.linear_mN(M_MAX, x, &w, be, y);
    const int8_t *xq_after_warm  = ws->act_xq;
    const size_t  cap_after_warm = ws->act_xq_cap;
    CHECK(xq_after_warm != nullptr && cap_after_warm >= M_MAX * N_IN,
          "workspace grew to the high-water shape");

    /* Steady state: 200 decode calls and every prefill shape up to the
     * high-water mark. None of them may move the buffer or change its
     * capacity — that is the allocation-free contract, observable. */
    for (int rep = 0; rep < 200; rep++) {
        w.linear_m1(x, &w, be, y);
    }
    for (size_t m = 1; m <= M_MAX; m++) {
        w.linear_mN(m, x, &w, be, y);
    }
    CHECK(ws->act_xq == xq_after_warm, "activation scratch was reallocated in steady state");
    CHECK(ws->act_xq_cap == cap_after_warm, "activation scratch capacity changed in steady state");

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
    CHECK(written == N_OUT, "kernel did not write its whole output row");
    CHECK(ws->act_xq == xq_after_warm, "final call reallocated the scratch");

    void *p = w_raw;
    safe_free(&p);
    safe_free((void **) &x);
    safe_free((void **) &y);
    geist_backend_destroy(be);

    if (g_fail) {
        return GEIST_TEST_FAIL;
    }
    printf("PASS: Q3_K linear_m1/linear_mN allocate nothing in steady state "
           "(200 decode + 8 prefill shapes, workspace pointer and capacity stable)\n");
    return GEIST_TEST_PASS;
}

#endif /* GEIST_BACKEND_CPU_NEON */

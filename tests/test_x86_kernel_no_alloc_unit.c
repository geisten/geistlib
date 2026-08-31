/*
 * test_x86_kernel_no_alloc_unit — the allocation-free linear-kernel
 * contract on cpu_x86 (issue #336, batch 3).
 *
 * The x86 M>1 kernels allocated their activation scratch on every call:
 * three raw `malloc`s in the I2S pair (also giving 16-byte alignment to
 * buffers that feed AVX-512 loads), three or four `heap_alloc_aligned`s in
 * the Q4_K / Q6_K / F32 prefill paths — per projection, per layer, per
 * chunk. They now take the per-thread backend workspace.
 *
 * Where the cpu_neon sibling test has to use the workspace's own capacity
 * as a proxy ("observing no malloc portably is not possible"), this one
 * observes the allocator directly: heap.h counts successful allocations
 * (#331), so the assertion is the real thing — zero allocations across the
 * measured loop. Both the poison check and the parity suites answer the
 * other half, "did the kernel actually run".
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>

#include <stdio.h>
#include <string.h>

#if !defined(GEIST_BACKEND_CPU_X86)
int main(void) {
    printf("SKIP: cpu_x86 backend not in this build\n");
    return GEIST_TEST_SKIP;
}
#else

#include "heap.h"
#include "quant.h"

static int g_fail = 0;

#define CHECK(cond, msg)                          \
    do {                                          \
        if (!(cond)) {                            \
            fprintf(stderr, "FAIL: %s\n", (msg)); \
            g_fail = 1;                           \
        }                                         \
    } while (0)

constexpr size_t N_IN  = 512;
constexpr size_t N_OUT = 64;
constexpr size_t M_MAX = 8;

struct dtype_case {
    const char *name;
    uint16_t    dtype;
    /* Total weight-blob bytes for N_IN x N_OUT. I2_S carries a trailing
     * per-tensor f32 scale; the dense dtypes are plain element arrays. */
    size_t (*nbytes)(void);
};

static size_t bytes_i2s(void) {
    return i2_s_scale_offset(N_IN * N_OUT) + sizeof(float);
}
static size_t bytes_q4k(void) {
    return (N_IN / Q4_K_BLOCK_ELEMS) * Q4_K_BLOCK_BYTES * N_OUT;
}
static size_t bytes_q6k(void) {
    return (N_IN / Q6_K_BLOCK_ELEMS) * Q6_K_BLOCK_BYTES * N_OUT;
}
static size_t bytes_f32(void) {
    return N_IN * N_OUT * sizeof(float);
}

static const struct dtype_case CASES[] = {
        {"I2_S", (uint16_t) GEIST_DTYPE_I2_S, bytes_i2s},
        {"Q4_K", (uint16_t) GEIST_DTYPE_Q4_K, bytes_q4k},
        {"Q6_K", (uint16_t) GEIST_DTYPE_Q6_K, bytes_q6k},
        {"F32", (uint16_t) GEIST_DTYPE_F32, bytes_f32},
};

/* Deterministic bytes. Contents are irrelevant — this test is about
 * allocation behaviour, not numerics (the parity suites cover that) — but
 * the F32 case is read as floats, so keep the pattern away from NaN/Inf
 * exponents by masking the high byte of every 4-byte group. */
static void *make_blob(size_t bytes) {
    uint8_t *blob = heap_alloc_array_aligned(uint8_t, bytes);
    if (blob == nullptr) {
        return nullptr;
    }
    for (size_t i = 0; i < bytes; i++) {
        blob[i] = (i % 4 == 3) ? (uint8_t) 0x3Du : (uint8_t) (i * 31u + 7u);
    }
    return blob;
}

/* Returns true if the dtype was exercised, false if this build/host does
 * not resolve it to an x86 kernel (reported by the caller, never silently
 * counted as a pass). */
static bool run_case(struct geist_backend *be, const struct dtype_case *c, float *x, float *y) {
    const size_t nbytes = c->nbytes();
    void        *w_raw  = make_blob(nbytes);
    if (w_raw == nullptr) {
        CHECK(false, "blob allocation");
        return false;
    }
    struct geist_weight w = {
            .raw        = w_raw,
            .raw_nbytes = nbytes,
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

    /* Warm up BOTH paths at their largest shape first: the workspace is
     * only at its high-water mark once each has run once, and the resolver
     * may still be lazily building an aux blob on the first call. */
    w.linear_mN(M_MAX, x, &w, be, y);
    w.linear_m1(x, &w, be, y);

    /* Steady state: 200 decode calls plus every prefill shape up to the
     * high-water mark. Not one of them may reach the allocator. */
    const uint64_t before = heap_alloc_count();
    for (int rep = 0; rep < 200; rep++) {
        w.linear_m1(x, &w, be, y);
    }
    for (size_t m = 1; m <= M_MAX; m++) {
        w.linear_mN(m, x, &w, be, y);
    }
    const uint64_t allocs = heap_alloc_count() - before;
    if (allocs != 0) {
        fprintf(stderr,
                "FAIL: %s: %llu heap allocations in steady state (expected 0)\n",
                c->name,
                (unsigned long long) allocs);
        g_fail = 1;
    }

    /* And the kernel actually ran: poison the output, call once more, and
     * require every element to have been written. Rules out "no allocation
     * because nothing executed" — the failure mode a counter cannot see. */
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

    printf("  %-6s m1 x200 + mN m=1..%zu: %llu allocations\n",
           c->name,
           M_MAX,
           (unsigned long long) allocs);

    if ((w.flags & GEIST_W_AUX_HEAP_OWNED) != 0 && w.aux_fp32 != nullptr) {
        void *aux = (void *) (uintptr_t) w.aux_fp32;
        safe_free(&aux);
    }
    safe_free(&freeme);
    return true;
}

int main(void) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create("cpu_x86", nullptr, nullptr, &be) != GEIST_OK || be == nullptr) {
        printf("SKIP: cpu_x86 backend did not register on this host\n");
        return GEIST_TEST_SKIP;
    }

    float *x = heap_alloc_array_aligned(float, M_MAX *N_IN);
    float *y = heap_alloc_array_aligned(float, M_MAX *N_OUT);
    if (x == nullptr || y == nullptr) {
        fprintf(stderr, "ERROR: activation allocation failed\n");
        return GEIST_TEST_ERROR;
    }
    for (size_t i = 0; i < M_MAX * N_IN; i++) {
        x[i] = (float) ((i * 37u) % 101u) * 0.01f - 0.5f;
    }

    size_t exercised = 0;
    printf("cpu_x86 allocation-free linear contract:\n");
    for (size_t i = 0; i < sizeof CASES / sizeof *CASES; i++) {
        if (run_case(be, &CASES[i], x, y)) {
            exercised++;
        } else {
            printf("  %-6s not resolved to an x86 kernel here — skipped\n", CASES[i].name);
        }
    }

    safe_free((void **) &x);
    safe_free((void **) &y);
    geist_backend_destroy(be);

    /* A run that resolved nothing proves nothing; fail rather than pass an
     * empty suite. */
    if (exercised == 0) {
        fprintf(stderr, "FAIL: no dtype resolved to an x86 kernel — nothing was tested\n");
        return GEIST_TEST_FAIL;
    }
    if (g_fail != 0) {
        return GEIST_TEST_FAIL;
    }
    printf("PASS: %zu dtype(s) allocation-free in steady state\n", exercised);
    return GEIST_TEST_PASS;
}

#endif /* GEIST_BACKEND_CPU_X86 */

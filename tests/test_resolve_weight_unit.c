/*
 * test_resolve_weight_unit — smoke test for backend->resolve_weight.
 *
 * Validates the load-time kernel-resolution API introduced in P1.1.b.
 * No GGUF needed; we synthesize geist_weight handles with raw=non-null
 * stubs and the supported dtypes, and verify the backend populates
 * linear_m1 / linear_mN as expected. For unsupported dtypes we expect
 * GEIST_E_UNSUPPORTED.
 *
 * Also gates cpu_neon_linear_support against the resolver: for every dtype the two
 * must agree on whether the backend can do it at all. They drifted apart
 * once (Q5_K / IQ2_S / IQ3_S advertised EMULATED and TQ2_0 / I2_S
 * advertised NONE while all five had native kernels), which is invisible
 * without a check like this because nothing in the engine consults
 * the capability answer — only the resolver and this gate do.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>
#include <geist_util.h>

#include "quant.h"

#define GEIST_INTERNAL_BACKEND_LAYER /* cpu_neon_linear_support (internal.h) */
#include "../src/backends/cpu_neon/internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int expect_resolved(struct geist_backend *be,
                           enum geist_dtype      dtype,
                           const char           *name,
                           bool                  expect_mN) {
    /* Stub raw pointer — we never call the kernel here, only resolve. */
    const size_t raw_bytes =
            dtype == GEIST_DTYPE_Q4_K ? (1536 / Q4_K_BLOCK_ELEMS) * 1536 * Q4_K_BLOCK_BYTES : 64;
    void *raw = calloc(1, raw_bytes);
    if (raw == nullptr) {
        fprintf(stderr, "  [%s] raw stub allocation failed\n", name);
        return 1;
    }
    struct geist_weight w = {
            .raw   = raw,
            .n_in  = 1536,
            .n_out = 1536,
            .dtype = (uint16_t) dtype,
    };
    const enum geist_status s = be->desc->vtbl->resolve_weight(be, &w);
    if (s != GEIST_OK) {
        fprintf(stderr, "  [%s] resolve_weight failed: %s\n", name, geist_status_to_string(s));
        free(raw);
        return 1;
    }
    if (w.linear_m1 == nullptr) {
        fprintf(stderr, "  [%s] linear_m1 still null after resolve\n", name);
        free(raw);
        return 1;
    }
    if (expect_mN && w.linear_mN == nullptr) {
        fprintf(stderr, "  [%s] linear_mN expected non-null but is null\n", name);
        free(raw);
        return 1;
    }
    if ((w.flags & GEIST_W_AUX_HEAP_OWNED) != 0 && w.aux_fp32 != nullptr) {
        void *aux = (void *) w.aux_fp32;
        free(aux);
    }
    free(raw);
    return 0;
}

/* cpu_neon_linear_support and resolve_weight must agree on whether this backend can
 * handle `dtype`. Asserts the equivalence rather than a fixed answer, so
 * it holds on hosts with and without dotprod (TQ2_0 / I2_S are ISA-gated
 * in the kernel table). */
static int
expect_support_agrees(struct geist_backend *be, enum geist_dtype dtype, const char *name) {
    void *raw = calloc(1, 64);
    if (raw == nullptr) {
        fprintf(stderr, "  [%s] raw stub allocation failed\n", name);
        return 1;
    }
    struct geist_weight w = {
            .raw   = raw,
            .n_in  = 1536,
            .n_out = 1536,
            .dtype = (uint16_t) dtype,
    };
    const bool resolves = be->desc->vtbl->resolve_weight(be, &w) == GEIST_OK;
    if ((w.flags & GEIST_W_AUX_HEAP_OWNED) != 0 && w.aux_fp32 != nullptr) {
        void *aux = (void *) w.aux_fp32;
        free(aux);
    }
    free(raw);

    const bool advertised = cpu_neon_linear_support(be, dtype) != CPU_NEON_SUPPORT_NONE;
    if (resolves != advertised) {
        fprintf(stderr,
                "  [%s] linear_support says %s but resolve_weight %s\n",
                name,
                advertised ? "supported" : "NONE",
                resolves ? "succeeded" : "failed");
        return 1;
    }
    return 0;
}

int main(void) {
    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create("cpu_neon", nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        fprintf(stderr, "no cpu_neon backend available: %s\n", geist_last_create_error());
        return GEIST_TEST_SKIP;
    }
    if (be->desc->vtbl->resolve_weight == nullptr) {
        fprintf(stderr, "backend does not implement resolve_weight\n");
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    int fails = 0;
    /* Supported (M=1 + M>1). */
    fails += expect_resolved(be, GEIST_DTYPE_Q3_K, "Q3_K", true);
    fails += expect_resolved(be, GEIST_DTYPE_Q4_K, "Q4_K", true);
    fails += expect_resolved(be, GEIST_DTYPE_Q6_K, "Q6_K", true);
    fails += expect_resolved(be, GEIST_DTYPE_IQ2_S, "IQ2_S", true);
    fails += expect_resolved(be, GEIST_DTYPE_IQ3_S, "IQ3_S", true);
    /* P2: Q8_0 M>1 now covered via dequant-and-cblas trampoline. */
    fails += expect_resolved(be, GEIST_DTYPE_Q8_0, "Q8_0", true);
    /* F32 dense supported via cblas trampolines (P1.1.e). */
    fails += expect_resolved(be, GEIST_DTYPE_F32, "F32", true);
    /* P2: Q5_K / F16 / BF16 now resolve via dequant-and-cblas
     * trampolines — the legacy v->linear() vtable fallback is no
     * longer used for these formats. */
    fails += expect_resolved(be, GEIST_DTYPE_Q5_K, "Q5_K", true);
    fails += expect_resolved(be, GEIST_DTYPE_F16, "F16", true);
    fails += expect_resolved(be, GEIST_DTYPE_BF16, "BF16", true);

    /* linear_support must not carry a dtype list of its own. TQ2_0 and I2_S
     * are the rows that were silently reported NONE; GEIST_DTYPE_CUSTOM
     * is the negative case that must stay NONE on both sides. */
    fails += expect_support_agrees(be, GEIST_DTYPE_Q3_K, "Q3_K");
    fails += expect_support_agrees(be, GEIST_DTYPE_Q4_K, "Q4_K");
    fails += expect_support_agrees(be, GEIST_DTYPE_Q5_K, "Q5_K");
    fails += expect_support_agrees(be, GEIST_DTYPE_Q6_K, "Q6_K");
    fails += expect_support_agrees(be, GEIST_DTYPE_Q8_0, "Q8_0");
    fails += expect_support_agrees(be, GEIST_DTYPE_IQ2_S, "IQ2_S");
    fails += expect_support_agrees(be, GEIST_DTYPE_IQ3_S, "IQ3_S");
    fails += expect_support_agrees(be, GEIST_DTYPE_TQ2_0, "TQ2_0");
    fails += expect_support_agrees(be, GEIST_DTYPE_I2_S, "I2_S");
    fails += expect_support_agrees(be, GEIST_DTYPE_F16, "F16");
    fails += expect_support_agrees(be, GEIST_DTYPE_BF16, "BF16");
    fails += expect_support_agrees(be, GEIST_DTYPE_F32, "F32");
    fails += expect_support_agrees(be, GEIST_DTYPE_CUSTOM, "CUSTOM (unsupported)");

    geist_backend_destroy(be);

    if (fails > 0) {
        fprintf(stderr, "FAIL: %d resolve_weight assertion(s)\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: cpu_neon resolve_weight covers Q3_K/Q4_K/Q5_K/Q6_K/Q8_0/"
           "IQ2_S/IQ3_S/F32/F16/BF16 (M=1 and M>1); linear_support agrees "
           "with the resolver on every dtype.\n");
    return GEIST_TEST_PASS;
}

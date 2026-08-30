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

#include <stdio.h>

/* The whole file gates cpu_neon's resolver against its kernel table —
 * it references backend-internal symbols (cpu_neon_linear_support) that
 * only exist when the neon backend is compiled in. Builds without it
 * (linux x86 / scalar-only CI) compile a SKIP stub instead. */
#if !defined(GEIST_BACKEND_CPU_NEON)

int main(void) {
    printf("SKIP: cpu_neon backend not in this build\n");
    return GEIST_TEST_SKIP;
}

#else

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

constexpr size_t STUB_N_IN  = 1536;
constexpr size_t STUB_N_OUT = 1536;

/* Format-correct storage for a [n_out, n_in] tensor of `dtype`, sized by
 * the same helper the resolver validates against. The old fixture handed
 * every dtype a flat 64-byte buffer; the Q4_K predecode hook then repacked
 * the whole tensor out of it. Under-sizing is now its own test case
 * (expect_short_buffer_rejected) rather than an accident here. */
static size_t stub_raw_bytes(enum geist_dtype dtype) {
    size_t need = 0;
    if (quant_raw_bytes(dtype, STUB_N_IN * STUB_N_OUT, &need)) {
        return 64; /* no fixed raw layout (CUSTOM): any non-zero stub */
    }
    return need;
}

static int expect_resolved(struct geist_backend *be,
                           enum geist_dtype      dtype,
                           const char           *name,
                           bool                  expect_mN) {
    const size_t raw_bytes = stub_raw_bytes(dtype);
    void        *raw       = calloc(1, raw_bytes);
    if (raw == nullptr) {
        fprintf(stderr, "  [%s] raw stub allocation failed\n", name);
        return 1;
    }
    struct geist_weight w = {
            .raw        = raw,
            .raw_nbytes = raw_bytes,
            .n_in       = (int32_t) STUB_N_IN,
            .n_out      = (int32_t) STUB_N_OUT,
            .dtype      = (uint16_t) dtype,
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
    const size_t raw_bytes = stub_raw_bytes(dtype);
    void        *raw       = calloc(1, raw_bytes);
    if (raw == nullptr) {
        fprintf(stderr, "  [%s] raw stub allocation failed\n", name);
        return 1;
    }
    struct geist_weight w = {
            .raw        = raw,
            .raw_nbytes = raw_bytes,
            .n_in       = (int32_t) STUB_N_IN,
            .n_out      = (int32_t) STUB_N_OUT,
            .dtype      = (uint16_t) dtype,
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

/* A source buffer one byte short of what (dtype, n_in, n_out) requires must
 * be refused before any repack reads it — not repacked out of whatever
 * follows the allocation. Runs for every dtype with a fixed raw layout;
 * ASan is what makes this test worth having. */
static int
expect_short_buffer_rejected(struct geist_backend *be, enum geist_dtype dtype, const char *name) {
    size_t need = 0;
    if (quant_raw_bytes(dtype, STUB_N_IN * STUB_N_OUT, &need) || need == 0) {
        return 0; /* no fixed raw layout: nothing to be short of */
    }
    void *raw = calloc(1, need - 1);
    if (raw == nullptr) {
        fprintf(stderr, "  [%s] short stub allocation failed\n", name);
        return 1;
    }
    struct geist_weight w = {
            .raw        = raw,
            .raw_nbytes = need - 1,
            .n_in       = (int32_t) STUB_N_IN,
            .n_out      = (int32_t) STUB_N_OUT,
            .dtype      = (uint16_t) dtype,
    };
    const enum geist_status s   = be->desc->vtbl->resolve_weight(be, &w);
    int                     bad = 0;
    if (s == GEIST_OK) {
        fprintf(stderr,
                "  [%s] resolve_weight accepted a source %zu bytes short of %zu\n",
                name,
                (size_t) 1,
                need);
        bad = 1;
    }
    if (w.linear_m1 != nullptr || w.linear_mN != nullptr) {
        fprintf(stderr, "  [%s] rejected resolve still installed a kernel\n", name);
        bad = 1;
    }
    if ((w.flags & GEIST_W_AUX_HEAP_OWNED) != 0 && w.aux_fp32 != nullptr) {
        void *aux = (void *) w.aux_fp32;
        free(aux);
        fprintf(stderr, "  [%s] rejected resolve still allocated aux\n", name);
        bad = 1;
    }
    free(raw);
    return bad;
}

/* The extent check is only as good as quant_raw_bytes' coverage: a dtype
 * the kernel table can install but the size table does not know passes
 * validation unchecked. That gap opens silently whenever a new dtype
 * lands (IQ4_NL and IQ4_XS did exactly that), so assert the two tables
 * agree rather than trusting them to be edited together.
 *
 * Sweeps the whole enum: nothing to keep in sync here either. */
static int expect_extent_known_for_every_supported_dtype(struct geist_backend *be) {
    int bad = 0;
    for (int d = 0; d <= (int) GEIST_DTYPE_CUSTOM; d++) {
        const enum geist_dtype dt = (enum geist_dtype) d;
        if (cpu_neon_linear_support(be, dt) == CPU_NEON_SUPPORT_NONE) {
            continue;
        }
        size_t need = 0;
        if (quant_raw_bytes(dt, STUB_N_IN * STUB_N_OUT, &need) || need == 0) {
            fprintf(stderr,
                    "  [dtype %d] cpu_neon installs a kernel for it, but quant_raw_bytes "
                    "cannot size it — resolve_weight cannot check its source extent\n",
                    d);
            bad = 1;
        }
    }
    return bad;
}

/* raw_nbytes is not optional: a caller that forgets it gets an argument
 * error, not an unvalidated repack. */
static int expect_missing_extent_rejected(struct geist_backend *be) {
    void *raw = calloc(1, stub_raw_bytes(GEIST_DTYPE_Q4_K));
    if (raw == nullptr) {
        return 1;
    }
    struct geist_weight w = {
            .raw   = raw,
            .n_in  = (int32_t) STUB_N_IN,
            .n_out = (int32_t) STUB_N_OUT,
            .dtype = (uint16_t) GEIST_DTYPE_Q4_K,
    };
    const enum geist_status s = be->desc->vtbl->resolve_weight(be, &w);
    free(raw);
    if (s != GEIST_E_INVALID_ARG) {
        fprintf(stderr,
                "  [raw_nbytes=0] expected GEIST_E_INVALID_ARG, got %s\n",
                geist_status_to_string(s));
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

    /* Short-source rejection, for every repacked dtype. */
    fails += expect_short_buffer_rejected(be, GEIST_DTYPE_Q3_K, "Q3_K short");
    fails += expect_short_buffer_rejected(be, GEIST_DTYPE_Q4_K, "Q4_K short");
    fails += expect_short_buffer_rejected(be, GEIST_DTYPE_Q5_K, "Q5_K short");
    fails += expect_short_buffer_rejected(be, GEIST_DTYPE_Q6_K, "Q6_K short");
    fails += expect_short_buffer_rejected(be, GEIST_DTYPE_Q8_0, "Q8_0 short");
    fails += expect_short_buffer_rejected(be, GEIST_DTYPE_IQ2_S, "IQ2_S short");
    fails += expect_short_buffer_rejected(be, GEIST_DTYPE_IQ3_S, "IQ3_S short");
    fails += expect_short_buffer_rejected(be, GEIST_DTYPE_TQ2_0, "TQ2_0 short");
    fails += expect_short_buffer_rejected(be, GEIST_DTYPE_I2_S, "I2_S short");
    fails += expect_short_buffer_rejected(be, GEIST_DTYPE_F16, "F16 short");
    fails += expect_short_buffer_rejected(be, GEIST_DTYPE_BF16, "BF16 short");
    fails += expect_short_buffer_rejected(be, GEIST_DTYPE_F32, "F32 short");
    fails += expect_short_buffer_rejected(be, GEIST_DTYPE_IQ4_NL, "IQ4_NL short");
    fails += expect_short_buffer_rejected(be, GEIST_DTYPE_IQ4_XS, "IQ4_XS short");
    fails += expect_short_buffer_rejected(be, GEIST_DTYPE_Q4_0, "Q4_0 short");
    fails += expect_missing_extent_rejected(be);
    fails += expect_extent_known_for_every_supported_dtype(be);

    geist_backend_destroy(be);

    if (fails > 0) {
        fprintf(stderr, "FAIL: %d resolve_weight assertion(s)\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: cpu_neon resolve_weight covers Q3_K/Q4_K/Q5_K/Q6_K/Q8_0/"
           "IQ2_S/IQ3_S/F32/F16/BF16 (M=1 and M>1); linear_support agrees "
           "with the resolver on every dtype; short and unset source extents "
           "are refused.\n");
    return GEIST_TEST_PASS;
}

#endif /* GEIST_BACKEND_CPU_NEON */

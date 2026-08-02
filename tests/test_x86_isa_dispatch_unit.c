/*
 * test_x86_isa_dispatch_unit — proof that the requested ISA tier was
 * actually selected (#184).
 *
 * The x86 dispatcher probes CPUID and applies the GEIST_FORCE_ISA clamp
 * (down-only). On a runner without AVX-512 the AVX-512 kernels are built
 * but silently not run — a green suite proves the fallback, not the
 * shipped fast path. This test turns "which tier ran?" into an assertable
 * fact:
 *
 *   - always prints the probed tier and the relevant CPUID features;
 *   - when GEIST_EXPECT_ISA={scalar,avx2,avx512,avx512_vnni} is set, the
 *     selected tier must match EXACTLY — a silent downgrade (e.g. SDE not
 *     active, wrong runner label, clamp bug) is a hard failure, not a skip.
 *
 * The AVX-512 CI leg runs this under Intel SDE with
 * GEIST_EXPECT_ISA=avx512_vnni before any kernel test, so the leg cannot
 * turn green on a CPU (real or emulated) that did not expose the features.
 */
#include "test_helpers.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(GEIST_BACKEND_CPU_X86) && (defined(__x86_64__) || defined(_M_X64))

#define GEIST_INTERNAL_BACKEND_LAYER
#include "src/backends/cpu_x86/kernel_w4a8.h"

/* __builtin_cpu_supports demands a string LITERAL, hence a macro. */
#define FEAT(s) (__builtin_cpu_supports(s) ? "yes" : "no")

int main(void) {
    printf("cpu features: avx2=%s avx512f=%s avx512bw=%s avx512dq=%s avx512vl=%s "
           "avx512vnni=%s avx512bf16=%s\n",
           FEAT("avx2"),
           FEAT("avx512f"),
           FEAT("avx512bw"),
           FEAT("avx512dq"),
           FEAT("avx512vl"),
           FEAT("avx512vnni"),
           FEAT("avx512bf16"));

    const enum w4a8_isa tier = w4a8_dispatcher_tier();
    printf("dispatcher tier: %s (GEIST_FORCE_ISA=%s)\n",
           w4a8_isa_name(tier),
           getenv("GEIST_FORCE_ISA") != nullptr ? getenv("GEIST_FORCE_ISA") : "<unset>");

    const char *expect = getenv("GEIST_EXPECT_ISA");
    if (expect == nullptr || expect[0] == '\0') {
        printf("PASS: dispatcher tier reported (no expectation set)\n");
        return GEIST_TEST_PASS;
    }

    enum w4a8_isa want;
    if (strcmp(expect, "scalar") == 0) {
        want = W4A8_ISA_SCALAR;
    } else if (strcmp(expect, "avx2") == 0) {
        want = W4A8_ISA_AVX2;
    } else if (strcmp(expect, "avx512") == 0) {
        want = W4A8_ISA_AVX512;
    } else if (strcmp(expect, "avx512_vnni") == 0) {
        want = W4A8_ISA_AVX512_VNNI;
    } else {
        fprintf(stderr, "FAIL: unknown GEIST_EXPECT_ISA '%s'\n", expect);
        return GEIST_TEST_ERROR;
    }

    /* BF16 aliases VNNI in the current dispatcher; accept it for a VNNI
     * expectation, never for anything lower. */
    const bool ok = tier == want || (want == W4A8_ISA_AVX512_VNNI && tier == W4A8_ISA_AVX512_BF16);
    if (!ok) {
        fprintf(stderr,
                "FAIL: expected tier %s but the dispatcher selected %s — "
                "the requested ISA did not actually run\n",
                expect,
                w4a8_isa_name(tier));
        return GEIST_TEST_FAIL;
    }
    printf("PASS: dispatcher selected %s as demanded\n", w4a8_isa_name(tier));
    return GEIST_TEST_PASS;
}

#else /* !cpu_x86 build — the dispatcher does not exist here. */

int main(void) {
    fprintf(stderr, "SKIP: x86 ISA dispatch test needs a cpu_x86 build\n");
    return GEIST_TEST_SKIP;
}

#endif

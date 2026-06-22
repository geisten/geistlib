/* tests/test_helpers.h — shared test helpers for geist test binaries.
 *
 * Conventions:
 *   - Each test is its own binary with main(), exits with one of:
 *       0   PASS
 *       77  SKIPPED (with reason on stdout)
 *       99  ERROR  (test harness broke, not the code-under-test)
 *       1   FAIL   (or any other non-zero)
 *   - PASS/FAIL/SKIP status MUST be communicated via exit code, not stdout.
 *     stdout is for informational output and failure detail only.
 *   - Tests requiring a GGUF model look for it via GEIST_GGUF_PATH env-var
 *     and skip cleanly if not found.
 *
 * See tests/README.md for the full test-conventions documentation.
 */
#ifndef GEIST_TEST_HELPERS_H
#define GEIST_TEST_HELPERS_H

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Checked fixture I/O. glibc marks fread/fwrite warn_unused_result under
 * _FORTIFY_SOURCE (Ubuntu CI), so an ignored return is a -Werror=unused-result
 * build failure. These wrappers also make a broken fixture fail loudly (abort)
 * instead of feeding garbage downstream. static inline -> no unused-function
 * warning in TUs that don't use them. */
static inline size_t xfread(void *p, size_t sz, size_t n, FILE *f) {
    size_t got = fread(p, sz, n, f);
    if (got != n) {
        fprintf(stderr, "xfread: short read (%zu of %zu items)\n", got, n);
        abort();
    }
    return got;
}
static inline size_t xfwrite(const void *p, size_t sz, size_t n, FILE *f) {
    size_t put = fwrite(p, sz, n, f);
    if (put != n) {
        fprintf(stderr, "xfwrite: short write (%zu of %zu items)\n", put, n);
        abort();
    }
    return put;
}

/* ---- Exit codes (automake-compatible) ----------------------------------- */

#define GEIST_TEST_PASS 0
#define GEIST_TEST_FAIL 1
#define GEIST_TEST_SKIP 77
#define GEIST_TEST_ERROR 99

/* Counter-style expectation: prints FAIL and returns 1 on a failed check, so a
 * test can accumulate  fails += geist_expect(cond, "what");  and finish with
 * return fails ? GEIST_TEST_FAIL : GEIST_TEST_PASS. static inline -> no
 * unused-function warning in TUs that don't use it. */
static inline int geist_expect(int cond, const char *what) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", what);
        return 1;
    }
    return 0;
}

/* ---- Skip helpers ------------------------------------------------------- */

/* Print SKIP reason to stdout and exit 77. Use from main() when the test
 * cannot run on this hardware/config (e.g. NEON-only test on x86 host,
 * GGUF-bound test without model file). */
#define GEIST_SKIP(reason)                       \
    do {                                         \
        fprintf(stdout, "SKIP: %s\n", (reason)); \
        exit(GEIST_TEST_SKIP);                   \
    } while (0)

#define GEIST_SKIP_IF(cond, reason) \
    do {                            \
        if ((cond)) {               \
            GEIST_SKIP(reason);     \
        }                           \
    } while (0)

/* Skip with a usage message when required CLI args are missing. The runner
 * already prefixes the binary name, so usage_msg should describe the args
 * only (e.g. "<tokenizer.bin>"). */
#define GEIST_REQUIRE_ARGS(argc, required, usage_msg) \
    GEIST_SKIP_IF((argc) < (required), "Missing test data. Usage: " usage_msg)

/* ---- GGUF model discovery ----------------------------------------------- */

/* Returns a path to the GGUF model file, or nullptr if not found.
 * Search order:
 *   1. $GEIST_GGUF_PATH env-var (if set)
 *   2. ./gemma-4-e2b-it-q3_k_m.gguf
 *   3. ../models/gemma-4-e2b-it-q3_k_m.gguf
 *   4. ../../gemma-4-E2B-it/gemma-4-e2b-it-q3_k_m.gguf
 *
 * Caller does NOT free the returned pointer (it is either env or static). */
static inline const char *geist_test_find_gguf(void) {
    const char *env = getenv("GEIST_GGUF_PATH");
    if (env && env[0]) {
        FILE *f = fopen(env, "rb");
        if (f) {
            fclose(f);
            return env;
        }
    }
    static const char *candidates[] = {
            "./gemma-4-e2b-it-q3_k_m.gguf",
            "../models/gemma-4-e2b-it-q3_k_m.gguf",
            "../../gemma-4-E2B-it/gemma-4-e2b-it-q3_k_m.gguf",
            nullptr,
    };
    for (size_t i = 0; candidates[i] != nullptr; i++) {
        FILE *f = fopen(candidates[i], "rb");
        if (f) {
            fclose(f);
            return candidates[i];
        }
    }
    return nullptr;
}

/* clang-tidy: bugprone-macro-parentheses doesn't understand that `varname` here is a
 * declaration identifier — wrapping it in parens would yield invalid C. Suppress. */
#define GEIST_REQUIRE_GGUF(varname) /* NOLINT(bugprone-macro-parentheses) */ \
    const char *varname = geist_test_find_gguf();                            \
    GEIST_SKIP_IF((varname) == nullptr,                                      \
                  "GGUF model not found. Set GEIST_GGUF_PATH or place model in ./, ../models/")

/* ---- Floating-point comparison ------------------------------------------ */

/* Combined relative + absolute tolerance check (numpy.allclose convention):
 *   |a - b| <= atol + rtol * |b|
 *
 * Defaults to use:
 *   FP32 reference paths:    rtol=1e-5,  atol=1e-7
 *   Quantized W3A8 / W4A8:   rtol=1e-3,  atol=1e-2  (lossy by design) */
static inline bool geist_fp32_close(float a, float b, float rtol, float atol) {
    return fabsf(a - b) <= atol + rtol * fabsf(b);
}

/* Vector-equivalent: returns the index of the first element that fails
 * the closeness test, or -1 if all elements pass. */
static inline ptrdiff_t
geist_fp32_close_array(const float *a, const float *b, size_t n, float rtol, float atol) {
    for (size_t i = 0; i < n; i++) {
        if (!geist_fp32_close(a[i], b[i], rtol, atol)) {
            return (ptrdiff_t) i;
        }
    }
    return -1;
}

#endif /* GEIST_TEST_HELPERS_H */

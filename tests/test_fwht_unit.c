/*
 * test_fwht_unit — verifies the orthonormal FWHT used to rotate attention
 * Q/K/V before INT8 KV-cache quantization (issue #61).
 *
 * Four properties that the rotation trick depends on:
 *   1. Self-inverse:  H(H(x)) == x. One matrix serves forward (Q/K/V) and
 *      backward (attention output) rotation.
 *   2. Dot-product preserving:  (Hx)·(Hy) == x·y. This is why rotating Q and
 *      K by the same H leaves the QK attention scores unchanged.
 *   3. Outlier suppression:  a spiky vector's peak magnitude shrinks after
 *      rotation, so symmetric INT8 quant (scale = amax/127) wastes fewer
 *      levels on the outlier — the whole reason to rotate before quantizing.
 *   4. The butterfly order, to the bit:  the passes of len 1, 2, 4, ... in
 *      turn, the last one scaled by 1/sqrt(n). The Vulkan and Metal ports
 *      run that order and their tests expect the host's bits; fwht.c
 *      spells out the first passes to vectorize them, which must not move
 *      a bit.
 *
 * Deterministic — fixed seed, no model needed.
 */
#include "fwht.h"
#include "test_helpers.h"

#include <math.h>
#include <string.h>

/* Deterministic N(0,1) via Box-Muller over a tiny LCG. */
static float gauss(uint32_t *seed) {
    uint32_t a  = (*seed = (*seed) * 1103515245u + 12345u);
    uint32_t b  = (*seed = (*seed) * 1103515245u + 12345u);
    float    u1 = ((float) (a & 0xffffff) + 1.0f) / (float) 0x1000000;
    float    u2 = ((float) (b & 0xffffff)) / (float) 0x1000000;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

static float dot(const float *a, const float *b, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; i++)
        s += (double) a[i] * (double) b[i];
    return (float) s;
}

static float absmax(const float *a, size_t n) {
    float m = 0.0f;
    for (size_t i = 0; i < n; i++)
        m = fabsf(a[i]) > m ? fabsf(a[i]) : m;
    return m;
}

/* Property 1: H(H(x)) reproduces x. */
static int test_self_inverse(void) {
    enum { N = 256 };
    float    x[N], y[N];
    uint32_t seed = 0x1234u;
    for (size_t i = 0; i < N; i++)
        x[i] = y[i] = gauss(&seed);
    fwht_orthonormal(N, y);
    fwht_orthonormal(N, y);
    for (size_t i = 0; i < N; i++) {
        if (fabsf(y[i] - x[i]) > 1e-4f) {
            printf("self-inverse: y[%zu]=%.6f != x=%.6f\n", i, y[i], x[i]);
            return 1;
        }
    }
    return 0;
}

/* Property 2: rotation preserves the dot product used by QK scores. */
static int test_dot_preserved(void) {
    enum { N = 128 };
    float    a[N], b[N], ra[N], rb[N];
    uint32_t seed = 0x9e37u;
    for (size_t i = 0; i < N; i++) {
        a[i] = ra[i] = gauss(&seed);
        b[i] = rb[i] = gauss(&seed);
    }
    fwht_orthonormal(N, ra);
    fwht_orthonormal(N, rb);
    const float d0 = dot(a, b, N);
    const float d1 = dot(ra, rb, N);
    if (fabsf(d0 - d1) > 1e-3f * (1.0f + fabsf(d0))) {
        printf("dot not preserved: %.6f vs %.6f\n", d0, d1);
        return 1;
    }
    return 0;
}

/* Property 3: a single big outlier is spread out, dropping the peak
 * magnitude that symmetric INT8 quant has to reach. */
static int test_outlier_suppressed(void) {
    enum { N = 256 };
    float    x[N];
    uint32_t seed = 0x5eedu;
    for (size_t i = 0; i < N; i++)
        x[i] = 0.1f * gauss(&seed);
    x[42]             = 20.0f; /* the outlier */
    const float peak0 = absmax(x, N);
    fwht_orthonormal(N, x);
    const float peak1 = absmax(x, N);
    /* 20.0 spread over sqrt(256)=16 → peak ~1.25, far below 20. Assert a
     * clear drop rather than the exact figure. */
    if (!(peak1 < 0.5f * peak0)) {
        printf("outlier not suppressed: peak %.4f -> %.4f\n", peak0, peak1);
        return 1;
    }
    return 0;
}

/* Property 4: the passes of a plain loop, in order, give the same bits.
 * The scale s is the library's own: -ffast-math lets gcc take 1/sqrtf(n)
 * through RSQRTSS and a Newton step, which two translation units need not
 * round alike. */
static void fwht_reference(size_t n, float s, float a[static n]) {
    if (n < 2)
        return;
    const size_t half = n >> 1;
    for (size_t len = 1; len < half; len <<= 1)
        for (size_t i = 0; i < n; i += 2 * len)
            for (size_t j = i; j < i + len; j++) {
                const float x = a[j];
                const float y = a[j + len];
                a[j]          = x + y;
                a[j + len]    = x - y;
            }
    for (size_t j = 0; j < half; j++) {
        const float x = a[j];
        const float y = a[j + half];
        a[j]          = (x + y) * s;
        a[j + half]   = (x - y) * s;
    }
}

static int test_butterfly_order(void) {
    enum { N_MAX = 8192 };
    static float x[N_MAX], ref[N_MAX];
    uint32_t     seed = 0xb17eu;
    for (size_t n = 1; n <= N_MAX; n <<= 1) {
        /* H e_0 is all ones before the scale, so every element is s. */
        for (size_t i = 0; i < n; i++)
            x[i] = i == 0 ? 1.0f : 0.0f;
        fwht_orthonormal(n, x);
        const float s = x[0];
        for (size_t i = 0; i < n; i++)
            x[i] = ref[i] = gauss(&seed);
        fwht_orthonormal(n, x);
        fwht_reference(n, s, ref);
        if (memcmp(x, ref, n * sizeof x[0]) != 0) {
            printf("butterfly order: n=%zu differs from the plain passes\n", n);
            return 1;
        }
    }
    return 0;
}

int main(void) {
    if (fwht_supported(256) && fwht_supported(128) && !fwht_supported(0) && !fwht_supported(96) &&
        test_self_inverse() == 0 && test_dot_preserved() == 0 && test_outlier_suppressed() == 0 &&
        test_butterfly_order() == 0) {
        printf("PASS: FWHT is self-inverse, dot-preserving, suppresses outliers, and runs the "
               "plain passes' butterflies to the bit\n");
        return GEIST_TEST_PASS;
    }
    return GEIST_TEST_FAIL;
}

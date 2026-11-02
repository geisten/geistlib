/*
 * test_moe_route_unit - the mixture-of-experts router's top-k (#48).
 *
 * moe_route_topk against a double-precision reference on the router edge
 * cases #48 lists: ties in router scores, top-k = number of experts, an
 * expert never selected, all-zero router input, and refusals (k = 0, k > n,
 * non-finite logits). moe_route: a router bias and GPT-OSS's
 * softmax over the selected (#243). No model, no fixture.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "src/archs/transformer/forward/blocks.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

static int fails;

static void expect(bool ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        fails++;
    }
}

/* Checks ids against `want` and w against softmax (renormalised when norm). */
static void
check(const char *what, size_t n, size_t k, bool norm, const float *logits, const uint32_t *want) {
    uint32_t ids[16];
    float    w[16];
    if (!moe_route_topk(n, k, norm, logits, ids, w)) {
        expect(false, what);
        return;
    }
    double mx = logits[0], z = 0.0, zs = 0.0, sum = 0.0;
    for (size_t i = 0; i < n; i++)
        mx = fmax(mx, logits[i]);
    for (size_t i = 0; i < n; i++)
        z += exp(logits[i] - mx);
    for (size_t j = 0; j < k; j++)
        zs += exp(logits[want[j]] - mx);
    for (size_t j = 0; j < k; j++) {
        const double ref = exp(logits[want[j]] - mx) / (norm ? zs : z);
        if (ids[j] != want[j] || fabs(w[j] - ref) > 1e-6) {
            fprintf(stderr,
                    "  %s: slot %zu id %u w %.8f, want id %u w %.8f\n",
                    what,
                    j,
                    ids[j],
                    (double) w[j],
                    want[j],
                    ref);
            expect(false, what);
            return;
        }
        sum += w[j];
    }
    if (norm)
        expect(fabs(sum - 1.0) < 1e-6, "renormalised weights sum to 1");
}

/* moe_route, GPT-OSS's router (#243): the bias is added to the logits
 * before the selection, and SOFTMAX_WEIGHT weights the selected by the
 * softmax over their biased logits alone. */
static void check_softmax_weight(void) {
    const float logits[6] = {0.1f, 2.0f, -1.0f, 3.5f, 0.0f, 1.9f};
    const float bias[6]   = {0.0f, -1.5f, 0.0f, 0.0f, 2.2f, 0.0f};
    uint32_t    ids[4];
    float       w[4];
    /* Biased: {0.1, 0.5, -1, 3.5, 2.2, 1.9} -> 3, 4, 5. */
    const uint32_t          want[3] = {3, 4, 5};
    const struct moe_router sw      = {.gating = GEIST_MOE_GATING_SOFTMAX_WEIGHT, .bias = bias};
    const struct moe_router sb = {.gating = GEIST_MOE_GATING_SOFTMAX, .norm = true, .bias = bias};
    const struct moe_router sp = {.gating = GEIST_MOE_GATING_SOFTMAX};
    if (!moe_route(6, 3, &sw, logits, ids, w)) {
        expect(false, "softmax_weight routes");
        return;
    }
    double b[3], z = 0.0;
    for (size_t j = 0; j < 3; j++) {
        b[j] = (double) logits[want[j]] + (double) bias[want[j]];
        z += exp(b[j] - b[0]);
    }
    for (size_t j = 0; j < 3; j++) {
        expect(ids[j] == want[j], "softmax_weight selects by the biased logits");
        expect(fabs(w[j] - exp(b[j] - b[0]) / z) < 1e-6,
               "softmax_weight weights are the softmax over the selected");
    }
    /* SOFTMAX with a bias: moe_route_topk on the biased logits. */
    const float biased[6] = {0.1f, 0.5f, -1.0f, 3.5f, 2.2f, 1.9f};
    uint32_t    ids2[3];
    float       w2[3];
    expect(moe_route(6, 3, &sb, logits, ids, w) && moe_route_topk(6, 3, true, biased, ids2, w2) &&
                   ids[0] == ids2[0] && ids[1] == ids2[1] && ids[2] == ids2[2] &&
                   fabsf(w[0] - w2[0]) < 1e-6f && fabsf(w[2] - w2[2]) < 1e-6f,
           "softmax with a bias is the top-k of the biased logits");
    /* Without a bias, SOFTMAX is moe_route_topk itself. */
    expect(moe_route(6, 2, &sp, logits, ids, w) && moe_route_topk(6, 2, false, logits, ids2, w2) &&
                   ids[0] == ids2[0] && ids[1] == ids2[1] && w[0] == w2[0] && w[1] == w2[1],
           "softmax without a bias is moe_route_topk");
    expect(!moe_route(6, 0, &sw, logits, ids, w), "softmax_weight refuses k = 0");
}

int main(void) {
    check_softmax_weight();
    /* Distinct scores: the k largest, largest first. */
    const float a[6] = {0.1f, 2.0f, -1.0f, 3.5f, 0.0f, 1.9f};
    check("distinct, k=3, norm", 6, 3, true, a, (const uint32_t[]) {3, 1, 5});
    check("distinct, k=3, no norm", 6, 3, false, a, (const uint32_t[]) {3, 1, 5});

    /* Ties: the lower expert id first, also across the cut at k. */
    const float t[6] = {1.0f, 2.0f, 2.0f, 1.0f, 2.0f, 1.0f};
    check("ties, k=2", 6, 2, true, t, (const uint32_t[]) {1, 2});
    check("ties across the cut, k=4", 6, 4, true, t, (const uint32_t[]) {1, 2, 4, 0});

    /* k = n: every expert, in rank order; renormalised it is the softmax. */
    check("k = n", 6, 6, true, a, (const uint32_t[]) {3, 1, 5, 0, 4, 2});

    /* All-zero input gives all-zero logits: the first k ids, equal weights. */
    const float z[8] = {0};
    check("all-zero logits", 8, 4, true, z, (const uint32_t[]) {0, 1, 2, 3});

    /* An expert that is never best is never selected. */
    const float nv[4] = {5.0f, 4.0f, -50.0f, 3.0f};
    check("never-selected expert", 4, 3, true, nv, (const uint32_t[]) {0, 1, 3});

    /* Large logits must not overflow the softmax. */
    const float big[3] = {1000.0f, 999.0f, 998.0f};
    check("large logits", 3, 2, true, big, (const uint32_t[]) {0, 1});

    /* Refusals. */
    uint32_t ids[8];
    float    w[8];
    expect(!moe_route_topk(4, 0, true, nv, ids, w), "k = 0 refused");
    expect(!moe_route_topk(4, 5, true, (const float[5]) {0}, ids, w), "k > n refused");
    expect(!moe_route_topk(3, 2, true, (const float[]) {0.0f, NAN, 1.0f}, ids, w),
           "NaN logit refused");
    expect(!moe_route_topk(3, 2, true, (const float[]) {0.0f, INFINITY, 1.0f}, ids, w),
           "infinite logit refused");

    if (fails) {
        fprintf(stderr, "FAIL: %d router check(s)\n", fails);
        return 1;
    }
    printf("PASS: MoE router top-k matches the softmax reference on ties, k = n, "
           "zero input, an unused expert, and refuses k = 0, k > n, non-finite logits\n");
    return 0;
}

/*
 * test_bitnet_arch_unit — the FFN-activation selection logic, no model.
 *
 * geist_ffn_activation_select decides which FFN nonlinearity a model runs from
 * its general.architecture + an optional feed_forward_activation override. This
 * is the exact decision that, when it wrongly defaulted BitNet b1.58 2B-4T to
 * SwiGLU instead of gated squared-ReLU, dropped its MMLU to chance (25.5% ->
 * 50%). Pin it here so that regression can't return silently. No assert() —
 * checks set a flag, the exit code carries PASS/FAIL.
 */
#define _POSIX_C_SOURCE 200809L
#define GEIST_INTERNAL_ARCH_LAYER /* arch_config.h is layer-internal */

#include "test_helpers.h"

#include "../src/archs/transformer/arch_config.h"

#include <stdio.h>

/* string literal -> (ptr, len) pair matching geist_ffn_activation_select's args */
#define S(lit) (lit), (sizeof(lit) - 1)

static int                            fails = 0;
static enum geist_ffn_activation_kind sel(const char *a, size_t al, const char *k, size_t kl) {
    return geist_ffn_activation_select(a, al, k, kl);
}

int main(void) {
    /* Arch-keyed default with NO explicit key (act = nullptr) — the BitNet bug. */
    fails += geist_expect(sel(S("bitnet-b1.58"), nullptr, 0) == GEIST_FFN_GATED_SQUARED_RELU,
                          "bitnet-b1.58 default -> gated squared-ReLU (the 2B-4T fix)");
    fails += geist_expect(sel(S("bitnet"), nullptr, 0) == GEIST_FFN_SWIGLU,
                          "community bitnet default -> SwiGLU");
    fails += geist_expect(sel(S("llama"), nullptr, 0) == GEIST_FFN_SWIGLU, "llama -> SwiGLU");
    fails += geist_expect(sel(S("gemma4"), nullptr, 0) == GEIST_FFN_SWIGLU,
                          "non-bitnet arch -> SwiGLU default");
    fails += geist_expect(sel(nullptr, 0, nullptr, 0) == GEIST_FFN_SWIGLU,
                          "null arch -> SwiGLU (safe default)");
    fails += geist_expect(sel(S("bitnet-b1.58-xl"), nullptr, 0) == GEIST_FFN_SWIGLU,
                          "near-miss arch is not bitnet-b1.58 (exact match only)");

    /* An explicit feed_forward_activation key overrides the arch default. */
    fails += geist_expect(sel(S("bitnet-b1.58"), S("swiglu")) == GEIST_FFN_SWIGLU,
                          "explicit swiglu overrides the bitnet-b1.58 default");
    fails += geist_expect(sel(S("bitnet"), S("relu2")) == GEIST_FFN_GATED_SQUARED_RELU,
                          "explicit relu2 overrides community default");
    fails += geist_expect(sel(S("bitnet"), S("gated_squared_relu")) == GEIST_FFN_GATED_SQUARED_RELU,
                          "explicit gated_squared_relu");
    fails += geist_expect(sel(S("gemma4"), S("geglu")) == GEIST_FFN_GEGLU, "explicit geglu");
    fails += geist_expect(sel(S("bitnet-b1.58"), S("squared_relu")) == GEIST_FFN_SQUARED_RELU,
                          "explicit (gateless) squared_relu overrides");
    fails += geist_expect(sel(S("bitnet-b1.58"), S("bogus")) == GEIST_FFN_GATED_SQUARED_RELU,
                          "unknown activation string falls back to the arch default");

    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("bitnet arch: FFN-activation selection pass\n");
    return GEIST_TEST_PASS;
}

/*
 * test_bitnet_arch_unit — the FFN-activation selection logic, no model.
 *
 * geist_ffn_activation_select decides which FFN nonlinearity a model runs
 * from its family's default and an optional feed_forward_activation
 * override; the two BitNet families name their defaults. Defaulting BitNet
 * b1.58 2B-4T to SwiGLU instead of gated squared-ReLU drops its MMLU to
 * chance, so the decision is pinned here, through the family registry on
 * GGUFs that carry only general.architecture (and the override). No
 * assert() — checks set a flag, the exit code carries PASS/FAIL.
 */
#define _POSIX_C_SOURCE 200809L
#define GEIST_INTERNAL_ARCH_LAYER /* arch_config.h is layer-internal */

#include "test_helpers.h"

#include "../src/archs/transformer/arch_family.h"
#include "../src/io/gguf_reader.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* string literal -> (len, ptr) pair matching geist_ffn_activation_select's args */
#define S(lit) (sizeof(lit) - 1), (lit)

static int fails = 0;
static enum geist_ffn_activation_kind
sel(enum geist_ffn_activation_kind dflt, size_t kl, const char *k) {
    return geist_ffn_activation_select(kl, dflt, k);
}

static size_t put_u32(uint8_t *p, uint32_t v) {
    memcpy(p, &v, 4);
    return 4;
}

static size_t put_u64(uint8_t *p, uint64_t v) {
    memcpy(p, &v, 8);
    return 8;
}

static size_t put_str(uint8_t *p, const char *s) {
    const uint64_t n = strlen(s);
    size_t         o = put_u64(p, n);
    memcpy(p + o, s, n);
    return o + n;
}

static size_t kv_str(uint8_t *p, const char *key, const char *v) {
    size_t o = put_str(p, key);
    o += put_u32(p + o, 8); /* string */
    return o + put_str(p + o, v);
}

/* The activation the family registered for `arch` populates from a GGUF
 * with only general.architecture, one dummy tensor and, when `act` is set,
 * <arch>.feed_forward_activation. -1 when the arch selects no family. */
static int family_activation(const char *arch, const char *act) {
    uint8_t buf[512];
    size_t  o = 0;
    memcpy(buf, "GGUF", 4);
    o += 4;
    o += put_u32(buf + o, 3);                      /* version */
    o += put_u64(buf + o, 1);                      /* n_tensors */
    o += put_u64(buf + o, act != nullptr ? 2 : 1); /* n_kv */
    o += kv_str(buf + o, "general.architecture", arch);
    if (act != nullptr) {
        char key[64];
        snprintf(key, sizeof key, "%s.feed_forward_activation", arch);
        o += kv_str(buf + o, key, act);
    }
    o += put_str(buf + o, "t"); /* one dummy tensor: name, dims, dtype, offset */
    o += put_u32(buf + o, 1);
    o += put_u64(buf + o, 1);
    o += put_u32(buf + o, 0); /* f32 */
    o += put_u64(buf + o, 0);
    while (o % 32 != 0) {
        buf[o++] = 0;
    }
    memset(buf + o, 0, 4); /* its 4 data bytes */
    o += 4;
    struct gguf_ctx *g = gguf_open_memory(buf, o, nullptr);
    if (g == nullptr) {
        return -1;
    }
    static struct transformer_arch_state st;
    memset(&st, 0, sizeof st);
    const struct transformer_family *fam = transformer_family_select(g);
    int                              out = -1;
    if (fam != nullptr) {
        fam->populate(g, &st);
        out = (int) st.config.ffn_activation;
    }
    gguf_close(g);
    return out;
}

int main(void) {
    /* The family defaults, through the registry. */
    fails += geist_expect(family_activation("bitnet-b1.58", nullptr) ==
                                  (int) GEIST_FFN_GATED_SQUARED_RELU,
                          "bitnet-b1.58 default -> gated squared-ReLU (the 2B-4T fix)");
    fails += geist_expect(family_activation("bitnet", nullptr) == (int) GEIST_FFN_SWIGLU,
                          "community bitnet default -> SwiGLU");
    fails += geist_expect(family_activation("bitnet-b1.58", "swiglu") == (int) GEIST_FFN_SWIGLU,
                          "an explicit key overrides the bitnet-b1.58 default");
    fails += geist_expect(family_activation("bitnet-b1.58-xl", nullptr) == -1,
                          "near-miss arch selects no family (exact match only)");

    /* An explicit feed_forward_activation key overrides the default. */
    fails += geist_expect(sel(GEIST_FFN_SWIGLU, 0, nullptr) == GEIST_FFN_SWIGLU,
                          "no key -> the default");
    fails += geist_expect(sel(GEIST_FFN_GATED_SQUARED_RELU, S("swiglu")) == GEIST_FFN_SWIGLU,
                          "explicit swiglu overrides a gated squared-ReLU default");
    fails += geist_expect(sel(GEIST_FFN_SWIGLU, S("relu2")) == GEIST_FFN_GATED_SQUARED_RELU,
                          "explicit relu2 overrides a SwiGLU default");
    fails += geist_expect(sel(GEIST_FFN_SWIGLU, S("gated_squared_relu")) ==
                                  GEIST_FFN_GATED_SQUARED_RELU,
                          "explicit gated_squared_relu");
    fails += geist_expect(sel(GEIST_FFN_SWIGLU, S("geglu")) == GEIST_FFN_GEGLU, "explicit geglu");
    fails += geist_expect(sel(GEIST_FFN_GATED_SQUARED_RELU, S("squared_relu")) ==
                                  GEIST_FFN_SQUARED_RELU,
                          "explicit (gateless) squared_relu overrides");
    fails += geist_expect(sel(GEIST_FFN_GATED_SQUARED_RELU, S("bogus")) ==
                                  GEIST_FFN_GATED_SQUARED_RELU,
                          "unknown activation string falls back to the default");

    /* Which activations need the relu_squared primitive (#352). A backend
     * without it cannot run these, and exec_plan_build refuses at load
     * instead of calling a null pointer once per layer — so a new gated
     * variant added to the enum must be added to this predicate too, and
     * this is where that is caught. */
    fails += geist_expect(geist_ffn_needs_relu_squared(GEIST_FFN_GATED_SQUARED_RELU),
                          "gated squared ReLU needs the relu_squared prim");
    fails += geist_expect(geist_ffn_needs_relu_squared(GEIST_FFN_SQUARED_RELU),
                          "gateless squared ReLU needs the relu_squared prim");
    fails += geist_expect(!geist_ffn_needs_relu_squared(GEIST_FFN_SWIGLU),
                          "SwiGLU does not need the relu_squared prim");
    fails += geist_expect(!geist_ffn_needs_relu_squared(GEIST_FFN_GEGLU),
                          "GEGLU does not need the relu_squared prim");
    /* And the two are wired together: every activation string that selects
     * a squared-ReLU kind must answer true. */
    fails += geist_expect(
            geist_ffn_needs_relu_squared(
                    (enum geist_ffn_activation_kind) family_activation("bitnet-b1.58", nullptr)),
            "the bitnet-b1.58 default requires relu_squared");
    fails += geist_expect(geist_ffn_needs_relu_squared(sel(GEIST_FFN_SWIGLU, S("relu2"))),
                          "a SwiGLU default with activation=relu2 requires it too");

    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("bitnet arch: FFN-activation selection + relu_squared requirement pass\n");
    return GEIST_TEST_PASS;
}

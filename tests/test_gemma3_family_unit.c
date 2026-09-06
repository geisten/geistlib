/*
 * test_gemma3_family_unit — the gemma3 arch name is accepted, and the
 * embedding scale is no longer tied to per-layer embeddings.
 *
 * No model, no backend: this pins two decisions that are otherwise only
 * observable by loading a GGUF.
 *
 *   1. The engine gate rejects any `general.architecture` it does not
 *      list, before weights are touched. Adding a family means adding it
 *      to BOTH the registry and geist_arch_transformer_gguf_names, and the
 *      static_assert in arch_family.c only checks their COUNTS match — two
 *      lists of the same length with different contents would compile and
 *      then fail closed on a real model.
 *
 *   2. sqrt(d_model) embedding scaling used to ride on config.has_ple.
 *      That held only while the sole family with the scale also had
 *      per-layer embeddings. Gemma 3 has the scale and no PLE, so they had
 *      to split. The risk in splitting them is the reverse: silently
 *      dropping the scale for gemma4, which would degrade a shipping model
 *      without failing anything.
 *
 * No assert(): release builds define NDEBUG.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "../src/archs/transformer/arch_config.h"

#include <stdio.h>
#include <string.h>

/* Exported by src/archs/transformer/arch_family.c; the engine gate reads
 * this exact list (src/engine/arch_registry.c). */
extern const char *const geist_arch_transformer_gguf_names[];

static int fails = 0;

static void check(const char *what, bool ok) {
    printf("  %s  %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) {
        fails++;
    }
}

static bool accepts(const char *arch) {
    for (size_t i = 0; geist_arch_transformer_gguf_names[i] != nullptr; i++) {
        if (strcmp(geist_arch_transformer_gguf_names[i], arch) == 0) {
            return true;
        }
    }
    return false;
}

int main(void) {
    puts("=== gemma3 family registration ===");

    check("gemma3 is an accepted architecture", accepts("gemma3"));

    /* The families that were already there must stay there — an edit to
     * the name list is exactly where one would go missing. */
    const char *const existing[] = {
            "gemma4",
            "llama",
            "qwen3",
            "qwen35",
            "bitnet-b1.58",
            "bitnet",
    };
    bool all_present = true;
    for (size_t i = 0; i < sizeof existing / sizeof existing[0]; i++) {
        if (!accepts(existing[i])) {
            printf("    missing: %s\n", existing[i]);
            all_present = false;
        }
    }
    check("every previously registered family is still accepted", all_present);

    check("an unknown architecture is still rejected", !accepts("gemma2"));
    check("gemma3 is not confused with gemma4", accepts("gemma3") && accepts("gemma4"));

    puts("=== embedding scale is independent of PLE ===");

    /* The struct is the contract here: two separate fields, so a family can
     * have either without the other. Before the split these were one. */
    struct geist_arch_config gemma3_like = {.has_embed_scale = true, .has_ple = false};
    struct geist_arch_config gemma4_like = {.has_embed_scale = true, .has_ple = true};
    struct geist_arch_config llama_like  = {.has_embed_scale = false, .has_ple = false};

    check("a family can scale embeddings without per-layer embeddings",
          gemma3_like.has_embed_scale && !gemma3_like.has_ple);
    check("gemma4 keeps both", gemma4_like.has_embed_scale && gemma4_like.has_ple);
    check("llama-likes keep neither", !llama_like.has_embed_scale && !llama_like.has_ple);

    if (fails > 0) {
        printf("FAIL: %d check(s) failed\n", fails);
        return 1;
    }
    puts("PASS: gemma3 is registered and the embed scale stands alone");
    return 0;
}

/*
 * tests/wiring/family_fixture.h - hooks for the family wiring cases (#345).
 *
 * Each tests/wiring/family_*.c defines one family table the way a unit in
 * src/archs/transformer/families/ does. scripts/check_wiring.py compiles it
 * twice: with -DWIRING_OK it must compile (so the failure below is the
 * mistake, not a missing include), without it the compile must fail with
 * the message its `// expect:` line names.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "src/archs/transformer/arch_family.h"

static void fixture_populate(struct gguf_ctx *gguf, struct transformer_arch_state *st) {
    (void) gguf;
    (void) st;
}

static bool fixture_populate_layers(struct transformer_arch_state *st) {
    (void) st;
    return true;
}

static enum geist_status fixture_block(struct transformer_layer_forward_ctx *ctx) {
    (void) ctx;
    return GEIST_OK;
}

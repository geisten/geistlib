/*
 * test_split_planner_unit — the layer split planner's fit (#78,
 * GEIST_SPLIT_AUTO): how many leading layers fit the split backend's
 * memory, each taking its weights and its KV cache.
 */
#define GEIST_INTERNAL_ARCH_LAYER
#include "src/archs/transformer/arch_state.h"
#include "test_helpers.h"

#include <stdint.h>
#include <stdio.h>

int main(void) {
    int          fails   = 0;
    const size_t even[]  = {100, 100, 100, 100};
    const size_t mixed[] = {50, 300, 10, 10};
    const struct {
        const size_t *bytes;
        size_t        n, kv, avail, want;
        const char   *what;
    } C[] = {
            {even, 4, 0, 1000, 4, "everything fits"},
            {even, 4, 0, 399, 3, "one byte short of four"},
            {even, 4, 0, 400, 4, "exactly four"},
            {even, 4, 25, 400, 3, "the KV cache takes a layer's room"},
            {even, 4, 0, 0, 0, "no memory"},
            {even, 4, 0, 99, 0, "less than one layer"},
            {mixed, 4, 0, 349, 1, "leading layers only: a big second layer stops it"},
            {mixed, 4, SIZE_MAX, SIZE_MAX, 0, "a KV cache that overflows"},
            {even, 0, 0, 1000, 0, "no layers"},
    };
    for (size_t i = 0; i < sizeof C / sizeof C[0]; i++) {
        const size_t got = transformer_split_layers_fit(
                C[i].n, C[i].n > 0 ? C[i].bytes : even, C[i].kv, C[i].avail);
        if (got != C[i].want) {
            fprintf(stderr, "FAIL: %s: %zu, want %zu\n", C[i].what, got, C[i].want);
            fails++;
        }
    }
    printf("%s: split planner fit\n", fails ? "FAIL" : "PASS");
    return fails ? GEIST_TEST_FAIL : 0;
}

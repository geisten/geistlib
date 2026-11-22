/*
 * test_kv_shared_prefill_int - a prefill that leaves the KV-shared layers out
 * for its cache rows gives the bits of one that runs them for every row
 * (#591).
 *
 * Gemma 4's last layers are KV-shared: they read an earlier layer's cache
 * and keep nothing per position (E2B 20 of 35, E4B 18 of 42). A prefill reads
 * one row's output, the last one's, so the rows of every chunk before the
 * last run layers [0, n_cache_layers) only, and on a GPU backend the first
 * rows of a last chunk of 192 or more too (transformer_prefill_pass_rows).
 * The oracle is the same model loaded with GEIST_KV_SHARED_FULL=1, which
 * runs every layer for every row as before the change.
 *
 * Both are driven through the same calls; after each, the two sessions must
 * agree bit for bit on the logits, the pending token, the length, the token
 * history and every K/V row of every layer that owns a cache:
 *
 *   - one prefill call at lengths around every boundary of the plan (a
 *     last chunk too short to split, the shortest that splits, one chunk,
 *     one chunk and a short last one, several chunks), at the model's chunk
 *     and at a session chunk of 16, then 6 decode steps, the state after
 *     each;
 *   - two prefill calls in a row; a prefill, a truncate and a prefill;
 *   - the same prompt again after a reset, and a token fed at a position a
 *     decode step was truncated from: the logits of the first time (a
 *     session's earlier pass must not show in a later one; the oracle is
 *     only an oracle if it is the same whatever ran before);
 *   - a snapshot after the prefill, restored after decoding on;
 *   - a pinned prefix, a turn, a reset and the turn again;
 *   - a verify pass, which returns a token for every row: no cache rows
 *     there, and the tokens agree;
 *   - a soft-token prefill (the audio and image path);
 *   - a decision readout of the last row;
 *   - each KV representation the backend takes (FP32, F16, INT8, INT4,
 *     KIVI; KIVI does not split its last chunk, its drain would move).
 *
 * The session that leaves rows out counts them (n_cache_rows): each case
 * checks the count against the plan, and the oracle's stays 0, so a pass
 * here means rows were left out and nothing moved, not that nothing was
 * left out.
 *
 * GEIST_BACKEND picks the backend (default "auto", and the CPU backend too
 * where "auto" is a GPU). Skips on a model without KV-shared layers.
 */
#define _POSIX_C_SOURCE 200809L
#define GEIST_INTERNAL_ARCH_LAYER
#define GEIST_INTERNAL_ENGINE_LAYER

#include "kv_shared_prefill.h"

int main(void) {
    GEIST_REQUIRE_GGUF(model_path);
    const struct kvs_source src = {.path = model_path};
    size_t                  ran = 0;
    kvs_ids();
    const char *env = getenv("GEIST_BACKEND");
    if (env != nullptr && env[0] != '\0') {
        return kvs_run_backend(&src, env, &ran) ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
    }
    int                   fails = kvs_run_backend(&src, "auto", &ran);
    struct geist_backend *be    = nullptr;
    if (geist_backend_create("auto", nullptr, nullptr, &be) == GEIST_OK) {
        const bool gpu = strncmp(geist_backend_name(be), "cpu_", 4) != 0;
        geist_backend_destroy(be);
        if (gpu) {
            fails += kvs_run_backend(&src, GEIST_TEST_CPU_BACKEND, &ran);
        }
    }
    return fails ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
}

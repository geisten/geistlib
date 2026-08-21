/*
 * test_qwen3_load_int — #275 smoke test for the qwen3 family path.
 *
 * Loads a qwen3 GGUF (default: gguf_artifacts/qwen3-0.6b-q8_0.gguf,
 * override with GEIST_QWEN3_GGUF_PATH) and checks that:
 *
 *   - transformer_family_select picks FAMILY_QWEN3
 *   - the populator strips the Gemma defaults (no PLE, no softcap,
 *     no KV sharing, no gemma attn norms) but keeps has_qk_norms
 *   - head_dim comes from qwen3.attention.key_length (128), NOT from
 *     d_model / n_heads (which would give 64) — q_out (2048) exceeds
 *     d_model (1024), the geometry #275 exists to pin
 *   - per-layer q_norm / k_norm weights actually loaded (non-null)
 *   - state_create completes with every tensor wired
 *
 * SKIPs cleanly when the fixture is absent (make fetch-qwen3-model).
 */
#include "test_helpers.h"

#define GEIST_INTERNAL_ARCH_LAYER
#include "src/archs/transformer/arch_state.h"

#define GEIST_INTERNAL_ENGINE_LAYER
#include "src/io/gguf_reader.h"

#include <geist.h>
#include <geist_backend.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *resolve_path(void) {
    const char *env = getenv("GEIST_QWEN3_GGUF_PATH");
    if (env != nullptr && env[0] != '\0')
        return env;
    static const char *candidates[] = {
            "gguf_artifacts/qwen3-0.6b-q8_0.gguf",
            "./qwen3-0.6b-q8_0.gguf",
            nullptr,
    };
    for (size_t i = 0; candidates[i] != nullptr; i++) {
        FILE *f = fopen(candidates[i], "rb");
        if (f != nullptr) {
            fclose(f);
            return candidates[i];
        }
    }
    return nullptr;
}

int main(void) {
    const char *path = resolve_path();
    if (path == nullptr) {
        GEIST_SKIP_FIXTURE("no qwen3 GGUF. Run `make fetch-qwen3-model`, or set "
                           "GEIST_QWEN3_GGUF_PATH");
    }

    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create("cpu_neon", nullptr, nullptr, &be);
    if (s != GEIST_OK)
        s = geist_backend_create("cpu_scalar", nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        fprintf(stderr, "backend create failed: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }

    struct transformer_arch_state *st = nullptr;
    s                                 = transformer_state_create(be, path, nullptr, &st);
    if (s != GEIST_OK) {
        fprintf(stderr,
                "state_create FAIL: %s — %s\n",
                geist_status_to_string(s),
                geist_backend_errmsg(be));
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    int fails = 0;

    /* Family + geometry assertions — Qwen3-0.6B values. */
    if (strcmp(st->config.family, "qwen3") != 0) {
        fprintf(stderr, "FAIL: family=%s, expected qwen3\n", st->config.family);
        fails++;
    }
    if (st->n_layers != 28) {
        fprintf(stderr, "FAIL: n_layers=%zu\n", st->n_layers);
        fails++;
    }
    if (st->d_model != 1024) {
        fprintf(stderr, "FAIL: d_model=%zu\n", st->d_model);
        fails++;
    }
    if (st->vocab_size != 151936) {
        fprintf(stderr,
                "FAIL: vocab=%zu (tokenizer.ggml.tokens fallback broken?)\n",
                st->vocab_size);
        fails++;
    }
    if (st->n_q_heads != 16 || st->n_kv_heads != 8) {
        fprintf(stderr, "FAIL: GQA %zu:%zu, expected 16:8\n", st->n_q_heads, st->n_kv_heads);
        fails++;
    }
    if (st->hidden_per_layer != 0 || st->ple_out != 0) {
        fprintf(stderr, "FAIL: PLE dims nonzero for qwen3\n");
        fails++;
    }

    /* Config flags: QK-norm alone, none of the Gemma extras. */
    if (!st->config.has_qk_norms) {
        fprintf(stderr, "FAIL: has_qk_norms should be true\n");
        fails++;
    }
    if (st->config.has_gemma_attn_norms) {
        fprintf(stderr, "FAIL: has_gemma_attn_norms should be false\n");
        fails++;
    }
    if (st->config.has_ple || st->config.logit_softcap != 0.0f) {
        fprintf(stderr, "FAIL: PLE/softcap not stripped\n");
        fails++;
    }
    if (st->config.kv_sliding_src != -1 || st->config.kv_full_src != -1) {
        fprintf(stderr, "FAIL: KV-share srcs not -1\n");
        fails++;
    }
    if (st->config.rope_interleaved) {
        fprintf(stderr, "FAIL: rope_interleaved should be false (NEOX, no pre-permute)\n");
        fails++;
    }

    /* Per-layer geometry: head_dim 128 from metadata (NOT 1024/16=64),
     * q_out 2048 > d_model 1024, uniform full-attn, and the QK-norm
     * weights actually present. */
    for (size_t i = 0; i < st->n_layers; i++) {
        const struct transformer_layer_weights *L = &st->layers[i];
        if (!L->is_full || L->is_kv_shared || L->head_dim != 128 || L->q_out != 2048 ||
            L->kv_out != 1024 || L->intermediate != 3072 || L->sliding_window != 0 ||
            L->n_rotated_dims != 128 || L->rope_theta != 1000000.0f) {
            fprintf(stderr,
                    "FAIL: layer[%zu] geometry (is_full=%d kv_shared=%d hd=%zu q=%zu kv=%zu "
                    "inter=%zu win=%zu rot=%d theta=%g)\n",
                    i,
                    L->is_full,
                    L->is_kv_shared,
                    L->head_dim,
                    L->q_out,
                    L->kv_out,
                    L->intermediate,
                    L->sliding_window,
                    L->n_rotated_dims,
                    (double) L->rope_theta);
            fails++;
            break;
        }
        if (L->q_norm.buffer == nullptr || L->k_norm.buffer == nullptr) {
            fprintf(stderr, "FAIL: layer[%zu] q_norm/k_norm not loaded\n", i);
            fails++;
            break;
        }
    }

    printf("loaded: %s  (%zu layers, d=%zu, vocab=%zu, GQA %zu:%zu, head_dim=%zu)\n",
           st->config.family,
           st->n_layers,
           st->d_model,
           st->vocab_size,
           st->n_q_heads,
           st->n_kv_heads,
           st->layers[0].head_dim);

    transformer_state_destroy(st);
    geist_backend_destroy(be);

    if (fails > 0) {
        fprintf(stderr, "%d assertion(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS\n");
    return GEIST_TEST_PASS;
}

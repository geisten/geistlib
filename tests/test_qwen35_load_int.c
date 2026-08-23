/*
 * test_qwen35_load_int — #281 smoke test for the qwen35 hybrid family.
 *
 * Loads a qwen35 GGUF (default: gguf_artifacts/qwen3.5-0.8b-q8_0.gguf,
 * override with GEIST_QWEN35_GGUF_PATH) and checks that:
 *
 *   - transformer_family_select picks FAMILY_QWEN35
 *   - the per-layer mixer schedule matches full_attention_interval=4:
 *     layers 3,7,11,15,19,23 are attention, the rest gated DeltaNet
 *   - DeltaNet geometry reaches the config (16x128 k/v heads, conv 4)
 *   - attention layers carry QK-norms + the 2x joint q+gate projection;
 *     DeltaNet layers carry the dn_* tensor set and no q_proj
 *   - state_create completes with every tensor wired
 *
 * SKIPs cleanly when the fixture is absent (make fetch-qwen35-model).
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
    const char *env = getenv("GEIST_QWEN35_GGUF_PATH");
    if (env != nullptr && env[0] != '\0')
        return env;
    static const char *candidates[] = {
            "gguf_artifacts/qwen3.5-0.8b-q8_0.gguf",
            "./qwen3.5-0.8b-q8_0.gguf",
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
        GEIST_SKIP_FIXTURE("no qwen35 GGUF. Run `make fetch-qwen35-model`, or set "
                           "GEIST_QWEN35_GGUF_PATH");
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

    if (strcmp(st->config.family, "qwen35") != 0) {
        fprintf(stderr, "FAIL: family=%s\n", st->config.family);
        fails++;
    }
    if (st->n_layers != 24 || st->d_model != 1024 || st->vocab_size != 248320) {
        fprintf(stderr,
                "FAIL: layers=%zu d=%zu vocab=%zu\n",
                st->n_layers,
                st->d_model,
                st->vocab_size);
        fails++;
    }
    if (st->n_q_heads != 8 || st->n_kv_heads != 2) {
        fprintf(stderr, "FAIL: GQA %zu:%zu\n", st->n_q_heads, st->n_kv_heads);
        fails++;
    }
    if (!st->config.has_attn_output_gate || !st->config.has_qk_norms ||
        st->config.has_gemma_attn_norms) {
        fprintf(stderr, "FAIL: config flags\n");
        fails++;
    }
    if (st->config.dn_n_k_heads != 16 || st->config.dn_n_v_heads != 16 ||
        st->config.dn_head_k != 128 || st->config.dn_head_v != 128 ||
        st->config.dn_conv_kernel != 4) {
        fprintf(stderr,
                "FAIL: dn geometry %zu/%zu/%zu/%zu conv=%zu\n",
                st->config.dn_n_k_heads,
                st->config.dn_n_v_heads,
                st->config.dn_head_k,
                st->config.dn_head_v,
                st->config.dn_conv_kernel);
        fails++;
    }

    size_t n_attn = 0, n_dn = 0;
    for (size_t i = 0; i < st->n_layers; i++) {
        const struct transformer_layer_weights *L           = &st->layers[i];
        const bool                              expect_attn = ((i + 1) % 4 == 0);
        if ((L->mixer == GEIST_MIXER_ATTN) != expect_attn) {
            fprintf(stderr, "FAIL: layer[%zu] mixer schedule\n", i);
            fails++;
            break;
        }
        if (L->mixer == GEIST_MIXER_ATTN) {
            n_attn++;
            if (L->head_dim != 256 || L->n_rotated_dims != 64 || L->q_out != 2048 ||
                L->kv_out != 512 || L->rope_theta != 10000000.0f) {
                fprintf(stderr, "FAIL: attn layer[%zu] geometry\n", i);
                fails++;
                break;
            }
            if (L->q_norm.buffer == nullptr || L->k_norm.buffer == nullptr ||
                L->q_proj.buffer == nullptr) {
                fprintf(stderr, "FAIL: attn layer[%zu] tensors\n", i);
                fails++;
                break;
            }
        } else {
            n_dn++;
            if (L->dn_qkv.buffer == nullptr || L->dn_z.buffer == nullptr ||
                L->dn_conv.buffer == nullptr || L->dn_a.buffer == nullptr ||
                L->dn_norm.buffer == nullptr || L->dn_out.buffer == nullptr) {
                fprintf(stderr, "FAIL: deltanet layer[%zu] tensors\n", i);
                fails++;
                break;
            }
            if (L->q_proj.buffer != nullptr) {
                fprintf(stderr, "FAIL: deltanet layer[%zu] has q_proj\n", i);
                fails++;
                break;
            }
        }
    }
    if (n_attn != 6 || n_dn != 18) {
        fprintf(stderr, "FAIL: schedule counts attn=%zu dn=%zu\n", n_attn, n_dn);
        fails++;
    }

    printf("loaded: %s  (%zu layers: %zu attn + %zu deltanet, d=%zu, vocab=%zu)\n",
           st->config.family,
           st->n_layers,
           n_attn,
           n_dn,
           st->d_model,
           st->vocab_size);

    transformer_state_destroy(st);
    geist_backend_destroy(be);

    if (fails > 0) {
        fprintf(stderr, "%d assertion(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS\n");
    return GEIST_TEST_PASS;
}

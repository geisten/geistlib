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

    const char      *gguf_err = nullptr;
    struct gguf_ctx *gguf     = gguf_open(path, &gguf_err);
    if (gguf == nullptr) {
        fprintf(stderr, "gguf_open FAIL: %s\n", gguf_err != nullptr ? gguf_err : "?");
        return GEIST_TEST_ERROR;
    }
    uint32_t total_layers = 0, n_mtp = 0, d_model = 0, n_q_heads = 0, n_kv_heads = 0;
    uint32_t head_dim = 0, intermediate = 0, rot = 0, interval = 0;
    uint32_t dn_k_heads = 0, dn_v_heads = 0, dn_head_k = 0, dn_inner = 0, dn_conv = 0;
    if (!gguf_get_meta_u32(gguf, "qwen35.block_count", &total_layers) ||
        !gguf_get_meta_u32(gguf, "qwen35.embedding_length", &d_model) ||
        !gguf_get_meta_u32(gguf, "qwen35.attention.head_count", &n_q_heads) ||
        !gguf_get_meta_u32(gguf, "qwen35.attention.head_count_kv", &n_kv_heads) ||
        !gguf_get_meta_u32(gguf, "qwen35.attention.key_length", &head_dim) ||
        !gguf_get_meta_u32(gguf, "qwen35.feed_forward_length", &intermediate) ||
        !gguf_get_meta_u32(gguf, "qwen35.rope.dimension_count", &rot) ||
        !gguf_get_meta_u32(gguf, "qwen35.full_attention_interval", &interval) ||
        !gguf_get_meta_u32(gguf, "qwen35.ssm.group_count", &dn_k_heads) ||
        !gguf_get_meta_u32(gguf, "qwen35.ssm.time_step_rank", &dn_v_heads) ||
        !gguf_get_meta_u32(gguf, "qwen35.ssm.state_size", &dn_head_k) ||
        !gguf_get_meta_u32(gguf, "qwen35.ssm.inner_size", &dn_inner) ||
        !gguf_get_meta_u32(gguf, "qwen35.ssm.conv_kernel", &dn_conv)) {
        fprintf(stderr, "required qwen35 geometry metadata missing\n");
        gguf_close(gguf);
        return GEIST_TEST_ERROR;
    }
    gguf_get_meta_u32(gguf, "qwen35.nextn_predict_layers", &n_mtp);
    gguf_close(gguf);

    const char           *backend_name     = getenv("GEIST_QWEN35_BACKEND");
    const bool            backend_explicit = backend_name != nullptr && backend_name[0] != '\0';
    struct geist_backend *be               = nullptr;
    enum geist_status     s                = geist_backend_create(
            backend_explicit ? backend_name : "cpu_neon", nullptr, nullptr, &be);
    if (s != GEIST_OK && !backend_explicit)
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
    if (total_layers <= n_mtp || st->n_layers != total_layers - n_mtp ||
        st->n_mtp_layers != n_mtp || st->d_model != d_model || st->vocab_size != 248320) {
        fprintf(stderr,
                "FAIL: layers=%zu mtp=%zu d=%zu vocab=%zu\n",
                st->n_layers,
                st->n_mtp_layers,
                st->d_model,
                st->vocab_size);
        fails++;
    }
    if (st->n_q_heads != n_q_heads || st->n_kv_heads != n_kv_heads) {
        fprintf(stderr, "FAIL: GQA %zu:%zu\n", st->n_q_heads, st->n_kv_heads);
        fails++;
    }
    if (!st->config.has_attn_output_gate || !st->config.has_qk_norms ||
        st->config.has_gemma_attn_norms) {
        fprintf(stderr, "FAIL: config flags\n");
        fails++;
    }
    if (st->config.dn_n_k_heads != dn_k_heads || st->config.dn_n_v_heads != dn_v_heads ||
        st->config.dn_head_k != dn_head_k || st->config.dn_head_v != dn_inner / dn_v_heads ||
        st->config.dn_conv_kernel != dn_conv) {
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
        const bool                              expect_attn = ((i + 1) % interval == 0);
        if ((L->mixer == GEIST_MIXER_ATTN) != expect_attn) {
            fprintf(stderr, "FAIL: layer[%zu] mixer schedule\n", i);
            fails++;
            break;
        }
        if (L->mixer == GEIST_MIXER_ATTN) {
            n_attn++;
            if (L->head_dim != head_dim || L->n_rotated_dims != (int) rot ||
                L->q_out != (size_t) n_q_heads * head_dim ||
                L->kv_out != (size_t) n_kv_heads * head_dim || L->intermediate != intermediate ||
                L->rope_theta != 10000000.0f) {
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
    const size_t expected_attn = st->n_layers / interval;
    if (n_attn != expected_attn || n_dn != st->n_layers - expected_attn) {
        fprintf(stderr, "FAIL: schedule counts attn=%zu dn=%zu\n", n_attn, n_dn);
        fails++;
    }

    if (n_mtp == 0) {
        if (st->mtp_layers != nullptr) {
            fprintf(stderr, "FAIL: model without MTP allocated MTP layers\n");
            fails++;
        }
    } else {
        if (st->mtp_layers == nullptr) {
            fprintf(stderr, "FAIL: MTP layer array missing\n");
            fails++;
        } else {
            for (size_t i = 0; i < st->n_mtp_layers; i++) {
                const struct transformer_mtp_layer_weights *M = &st->mtp_layers[i];
                const struct transformer_layer_weights     *L = &M->block;
                if (L->layer_idx != (int) (st->n_layers + i) || L->mixer != GEIST_MIXER_ATTN ||
                    !L->is_full || L->head_dim != head_dim || L->intermediate != intermediate) {
                    fprintf(stderr, "FAIL: MTP layer[%zu] geometry\n", i);
                    fails++;
                }
                if (L->q_proj.buffer == nullptr || L->k_proj.buffer == nullptr ||
                    L->v_proj.buffer == nullptr || L->o_proj.buffer == nullptr ||
                    L->gate_proj.buffer == nullptr || L->up_proj.buffer == nullptr ||
                    L->down_proj.buffer == nullptr || M->eh_proj.buffer == nullptr ||
                    M->enorm.buffer == nullptr || M->hnorm.buffer == nullptr ||
                    M->shared_head_norm.buffer == nullptr) {
                    fprintf(stderr, "FAIL: MTP layer[%zu] tensors\n", i);
                    fails++;
                }
                if (M->eh_proj.ndim != 2 || M->eh_proj.shape[0] != (int64_t) d_model ||
                    M->eh_proj.shape[1] != (int64_t) (2 * d_model) ||
                    M->enorm.shape[0] != d_model || M->hnorm.shape[0] != d_model ||
                    M->shared_head_norm.shape[0] != d_model) {
                    fprintf(stderr, "FAIL: MTP layer[%zu] tensor shapes\n", i);
                    fails++;
                }
            }
        }
    }

    printf("loaded: %s  (%zu layers: %zu attn + %zu deltanet, %zu MTP, d=%zu, vocab=%zu)\n",
           st->config.family,
           st->n_layers,
           n_attn,
           n_dn,
           st->n_mtp_layers,
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

/*
 * src/archs/transformer/arch_family.c — family-populator implementations
 * and the `general.architecture` → family registry.
 *
 * Layer: ARCHITECTURE.
 *
 * Registry: gemma4, gemma3, llama, qwen3, qwen35, bitnet-b1.58, bitnet — a
 * static array searched linearly.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "arch_family.h"
#include "arch_state.h"

#include "gguf_reader.h"

#include <stddef.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

/* ---- BitNet embedding metadata ----------------------------------------- */
/*
 * The two facts that turn a stock qwen3 / gemma3 GGUF into an embedding
 * model: an RMSNorm on every projection's input, and a pooling. Both
 * populators read them, and there are two spellings in the wild.
 *
 *   tools/convert_bitnet_embedding.py writes
 *       bitnet.embedding.projection_input_norms  (bool)
 *       bitnet.embedding.pooling                 ("mean" | "last_token")
 *
 *   Microsoft's own published GGUFs (bitnet-embeddings-*-i2_s.gguf on the
 *   Hugging Face model pages) write NEITHER. They carry the per-projection
 *   norms as tensors only, and the pooling as gguf-py's standard numeric
 *   `{arch}.pooling_type`.
 *
 * Both spellings are read here; neither weakens the guarantee that the file
 * has to actually contain what the metadata promises.
 */
void populate_bitnet_embedding_meta(struct gguf_ctx               *gguf,
                                    const char                    *arch,
                                    struct transformer_arch_state *st) {
    /* Projection-input norms. The TENSOR is the evidence — metadata must
     * not promise norms the file lacks, so the key is not required. An
     * explicit `false` still wins, so a converter can opt out. */
    const bool have_norm_tensors = gguf_get_tensor(gguf, "blk.0.attn_q_norm_in.weight") != nullptr;
    bool       proj_norms        = true;
    (void) gguf_get_meta_bool(gguf, "bitnet.embedding.projection_input_norms", &proj_norms);
    st->config.has_projection_input_norms = have_norm_tensors && proj_norms;

    /* Pooling. Its absence is what makes a stock GGUF generative, so a
     * missing key in BOTH spellings means NONE, not a failure. A key that
     * is present but names a pooling this build cannot perform leaves
     * pooling_unsupported set, which populate_layers refuses — better a
     * clear load failure than a vector pooled the wrong way. */
    size_t      plen = 0;
    const char *pstr = gguf_get_meta_string(gguf, "bitnet.embedding.pooling", &plen);
    if (pstr != nullptr) {
        if (!geist_pooling_select(plen, pstr, &st->config.pooling)) {
            st->config.pooling             = GEIST_POOLING_NONE;
            st->config.pooling_unsupported = true;
        }
        return;
    }
    char key[96];
    snprintf(key, sizeof key, "%s.pooling_type", arch);
    uint32_t pooling_type = 0;
    if (!gguf_get_meta_u32(gguf, key, &pooling_type)) {
        st->config.pooling = GEIST_POOLING_NONE; /* a generative model */
        return;
    }
    if (!geist_pooling_from_gguf_type(pooling_type, &st->config.pooling)) {
        st->config.pooling             = GEIST_POOLING_NONE;
        st->config.pooling_unsupported = true;
    }
}

/* ---- Shared metadata and geometry -------------------------------------- */

bool family_meta_u32(
        struct gguf_ctx *g, const char *arch, const char *alt, const char *key, uint32_t *out) {
    char k[96];
    snprintf(k, sizeof k, "%s.%s", arch, key);
    if (gguf_get_meta_u32(g, k, out)) {
        return true;
    }
    if (alt == nullptr) {
        return false;
    }
    snprintf(k, sizeof k, "%s.%s", alt, key);
    return gguf_get_meta_u32(g, k, out);
}

bool family_meta_f32(
        struct gguf_ctx *g, const char *arch, const char *alt, const char *key, float *out) {
    char k[96];
    snprintf(k, sizeof k, "%s.%s", arch, key);
    if (gguf_get_meta_f32(g, k, out)) {
        return true;
    }
    if (alt == nullptr) {
        return false;
    }
    snprintf(k, sizeof k, "%s.%s", alt, key);
    return gguf_get_meta_f32(g, k, out);
}

void family_populate_common(struct gguf_ctx               *g,
                            const char                    *arch,
                            const char                    *alt,
                            struct transformer_arch_state *st) {
    uint32_t u;
    if (family_meta_u32(g, arch, alt, "block_count", &u)) {
        st->n_layers = u;
    }
    if (family_meta_u32(g, arch, alt, "embedding_length", &u)) {
        st->d_model = u;
    }
    if (family_meta_u32(g, arch, alt, "attention.head_count", &u)) {
        st->n_q_heads = u;
    }
    if (family_meta_u32(g, arch, alt, "attention.head_count_kv", &u)) {
        st->n_kv_heads = u;
    }
    (void) family_meta_f32(g, arch, alt, "attention.layer_norm_rms_epsilon", &st->config.rms_eps);
}

size_t family_vocab_size(struct gguf_ctx *g, const char *arch, const char *alt) {
    uint32_t u;
    if (family_meta_u32(g, arch, alt, "vocab_size", &u)) {
        return u;
    }
    uint32_t       elem_vt = 0;
    uint64_t       count   = 0;
    const uint8_t *payload = nullptr;
    if (gguf_get_meta_array_info(g, "tokenizer.ggml.tokens", &elem_vt, &count, &payload)) {
        return (size_t) count;
    }
    return 0;
}

void family_uniform_layer(const struct transformer_arch_state *st,
                          size_t                               i,
                          size_t                               head_dim,
                          size_t                               intermediate,
                          float                                rope_theta,
                          int                                  n_rotated,
                          struct transformer_layer_weights    *L) {
    L->layer_idx      = (int) i;
    L->mixer          = GEIST_MIXER_ATTN;
    L->is_full        = true;
    L->is_kv_shared   = false;
    L->head_dim       = head_dim;
    L->q_out          = st->n_q_heads * head_dim;
    L->kv_out         = st->n_kv_heads * head_dim;
    L->intermediate   = intermediate;
    L->sliding_window = 0;
    L->rope_theta     = rope_theta;
    L->n_rotated_dims = n_rotated;
}

void family_uniform_layers(struct transformer_arch_state *st,
                           size_t                         head_dim,
                           size_t                         intermediate,
                           float                          rope_theta) {
    for (size_t i = 0; i < st->n_layers; i++) {
        family_uniform_layer(
                st, i, head_dim, intermediate, rope_theta, (int) head_dim, &st->layers[i]);
    }
}

/* ---- Registry --------------------------------------------------------- */

/* Only the families the build selected (make FAMILIES=..., mk/common.mk).
 * A build that selects none (no -DGEIST_FAMILY_*, e.g. another build system)
 * gets all of them. */
#if !defined(GEIST_FAMILY_BITNET) && !defined(GEIST_FAMILY_GEMMA3) &&    \
        !defined(GEIST_FAMILY_GEMMA4) && !defined(GEIST_FAMILY_LLAMA) && \
        !defined(GEIST_FAMILY_QWEN3) && !defined(GEIST_FAMILY_QWEN35) && \
        !defined(GEIST_FAMILY_QWEN3MOE)
#define GEIST_FAMILY_BITNET 1
#define GEIST_FAMILY_GEMMA3 1
#define GEIST_FAMILY_GEMMA4 1
#define GEIST_FAMILY_LLAMA 1
#define GEIST_FAMILY_QWEN3 1
#define GEIST_FAMILY_QWEN35 1
#define GEIST_FAMILY_QWEN3MOE 1
#endif
static const struct transformer_family *const REGISTRY[] = {
#if defined(GEIST_FAMILY_GEMMA4)
        &transformer_family_gemma4, /* families/gemma4.c */
#endif
#if defined(GEIST_FAMILY_GEMMA3)
        &transformer_family_gemma3, /* families/gemma3.c */
#endif
#if defined(GEIST_FAMILY_LLAMA)
        &transformer_family_llama, /* families/llama.c */
#endif
#if defined(GEIST_FAMILY_QWEN3)
        &transformer_family_qwen3, /* families/qwen3.c */
#endif
#if defined(GEIST_FAMILY_QWEN35)
        &transformer_family_qwen35, /* families/qwen35.c */
#endif
#if defined(GEIST_FAMILY_BITNET)
        &transformer_family_bitnet_b158, /* families/bitnet.c */
        &transformer_family_bitnet,
#endif
#if defined(GEIST_FAMILY_QWEN3MOE)
        &transformer_family_qwen3moe, /* families/qwen3moe.c */
#endif
};
static const size_t REGISTRY_N = sizeof REGISTRY / sizeof REGISTRY[0];

/* Engine-facing match list: the exact `general.architecture` values this
 * architecture accepts, NULL-terminated. Referenced by desc_transformer in
 * src/engine/arch_registry.c — the engine gate rejects any GGUF whose arch
 * string is not listed here, before weights are touched. Must mirror
 * REGISTRY above (count enforced below; a string mismatch fails closed at
 * transformer_family_select, never loads the wrong family). */
const char *const geist_arch_transformer_gguf_names[] = {
#if defined(GEIST_FAMILY_GEMMA4)
        "gemma4",
#endif
#if defined(GEIST_FAMILY_GEMMA3)
        "gemma3",
#endif
#if defined(GEIST_FAMILY_LLAMA)
        "llama",
#endif
#if defined(GEIST_FAMILY_QWEN3)
        "qwen3",
#endif
#if defined(GEIST_FAMILY_QWEN35)
        "qwen35",
#endif
#if defined(GEIST_FAMILY_BITNET)
        "bitnet-b1.58",
        "bitnet",
#endif
#if defined(GEIST_FAMILY_QWEN3MOE)
        "qwen3moe",
#endif
        nullptr,
};
static_assert(sizeof geist_arch_transformer_gguf_names /
                                      sizeof geist_arch_transformer_gguf_names[0] -
                              1 ==
                      sizeof REGISTRY / sizeof REGISTRY[0],
              "gguf name list must mirror REGISTRY");

const struct transformer_family *transformer_family_select(struct gguf_ctx *gguf) {
    size_t      arch_len = 0;
    const char *arch     = gguf_get_meta_string(gguf, "general.architecture", &arch_len);
    if (arch != nullptr) {
        for (size_t i = 0; i < REGISTRY_N; i++) {
            const struct transformer_family *f    = REGISTRY[i];
            const size_t                     flen = strlen(f->name);
            if (flen == arch_len && memcmp(arch, f->name, flen) == 0) {
                return f;
            }
        }
    }
    /* Fail closed: no silent Gemma-4 fallback. The engine gate should
     * have rejected the GGUF already; reaching this means a caller
     * bypassed it or the name lists diverged. */
    return nullptr;
}

/*
 * src/engine/model.c — geist_model_load facade.
 *
 * Layer: ENGINE.
 *
 * Dispatches to arch_ops based on the GGUF's general.architecture metadata.
 * Transformer goes to src/archs/transformer (which since B-4e runs entirely
 * through backend->vtbl ops); future architectures (Mamba, etc.) plug in
 * by registering an arch_ops descriptor in arch_registry.c.
 */
#define GEIST_INTERNAL_ENGINE_LAYER

#include "model.h"
#include "arch_registry.h"
#include "error.h"

#include <geist_arch.h> /* arch_ops vtables the engine dispatches through */

#include "heap.h"
#include "sp_bpe_tokenizer.h"
#include "gguf_tokenizer.h"
#include "gguf_reader.h"

#include <geist.h>
#include <geist_util.h> /* eos/bos/token_by_text live here as of 0.2.0 */

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Per-model engine-side state. The text_decoder.arch_meta in
 * struct geist_model holds the arch-specific state — transformer_arch_state
 * for the transformer arch; future archs will store their own state types.
 * This struct holds engine-only side data that's architecture-agnostic. */
struct model_engine_state {
    char *path;
    /* Two tokenizer paths, exactly one populated:
     *  - `sp_tok` is loaded from an external tokenizer.bin (Gemma path).
     *  - `gguf_tok` is loaded from the GGUF-embedded vocab + merges
     *    (Llama / Mistral path, P1.6). Owns heap-arena copies of the
     *    token strings so the gguf_ctx can close after load.
     * set_prompt / tokenize dispatch on whichever is non-null. */
    struct sp_bpe_tokenizer *sp_tok;
    struct gguf_tokenizer   *gguf_tok;
    char                    *arch; /* general.architecture, owned copy (nullptr if absent) */
};

/* Read general.architecture from an open GGUF into an owned NUL-terminated copy
 * (GGUF strings are length-prefixed, not terminated). Returns nullptr if absent
 * or OOM — callers fall back to "transformer". */
static char *model_arch_copy(struct gguf_ctx *tg) {
    size_t      alen = 0;
    const char *as   = gguf_get_meta_string(tg, "general.architecture", &alen);
    if (as == nullptr || alen == 0) {
        return nullptr;
    }
    char *copy = heap_alloc_aligned(alen + 1, alignof(char));
    if (copy != nullptr) {
        memcpy(copy, as, alen);
        copy[alen] = '\0';
    }
    return copy;
}

/* Find tokenizer.bin near the GGUF: <dir>/tokenizer.bin, then ./tokenizer.bin,
 * then ../<sibling-model-dir>/tokenizer.bin heuristically, then env. */
static char *find_tokenizer_path(const char *gguf_path) {
    /* 1. Env override */
    const char *env = getenv("GEIST_TOKENIZER_PATH");
    if (env != nullptr && env[0] != '\0') {
        FILE *f = fopen(env, "rb");
        if (f != nullptr) {
            fclose(f);
            return strdup(env);
        }
    }

    /* 2. Same directory as the GGUF */
    const char *slash = strrchr(gguf_path, '/');
    if (slash != nullptr) {
        size_t dir_len       = (size_t) (slash - gguf_path);
        size_t candidate_len = dir_len + strlen("/tokenizer.bin") + 1;
        char  *candidate     = heap_alloc_aligned(candidate_len, alignof(char));
        if (candidate != nullptr) {
            memcpy(candidate, gguf_path, dir_len);
            memcpy(candidate + dir_len, "/tokenizer.bin", strlen("/tokenizer.bin") + 1);
            FILE *f = fopen(candidate, "rb");
            if (f != nullptr) {
                fclose(f);
                return candidate;
            }
            safe_free((void **) &candidate);
        }
    }

    /* 3. Project default for Gemma 4 development setup. */
    static const char *fallbacks[] = {
            "./tokenizer.bin",
            "../gemma-4-E2B-it/tokenizer.bin",
            "../../gemma-4-E2B-it/tokenizer.bin",
            nullptr,
    };
    for (size_t i = 0; fallbacks[i] != nullptr; i++) {
        FILE *f = fopen(fallbacks[i], "rb");
        if (f != nullptr) {
            fclose(f);
            return strdup(fallbacks[i]);
        }
    }
    return nullptr;
}

[[nodiscard]] enum geist_status
geist_model_load(const char *path, struct geist_backend *be, struct geist_model **out) {
    if (out == nullptr) {
        geist_error_set_create_time(GEIST_E_INVALID_ARG, "geist_model_load", "out is null");
        return GEIST_E_INVALID_ARG;
    }
    *out = nullptr;
    if (path == nullptr || path[0] == '\0' || be == nullptr) {
        geist_error_set_create_time(
                GEIST_E_INVALID_ARG, "geist_model_load", "path or backend is null/empty");
        return GEIST_E_INVALID_ARG;
    }

    /* Resolve architecture. Today the registry has exactly one entry
     * (transformer) — geist_arch_registry_lookup returns it as the
     * fallback regardless of the GGUF arch string. */
    const struct geist_arch_descriptor *desc = geist_arch_registry_lookup("gemma");
    if (desc == nullptr || desc->decoder_ops == nullptr) {
        geist_error_set_create_time(GEIST_E_UNSUPPORTED,
                                    "geist_model_load",
                                    "no decoder architecture compiled into this build");
        return GEIST_E_UNSUPPORTED;
    }

    /* Allocate the arch_state via the decoder's state_create. For the
     * transformer arch this opens the GGUF and loads weights into
     * backend-owned buffers; for future archs it'll be SSM-state or
     * whatever the arch needs. */
    void *arch_state = desc->decoder_ops->state_create(be, path, nullptr);
    if (arch_state == nullptr) {
        geist_error_set_create_time(GEIST_E_IO,
                                    "geist_model_load",
                                    "decoder state_create failed for '%s' (file missing or "
                                    "malformed, or out of memory?)",
                                    path);
        return GEIST_E_IO;
    }

    struct geist_model *m = heap_alloc_aligned(sizeof(*m), alignof(struct geist_model));
    if (m == nullptr) {
        desc->decoder_ops->state_destroy(arch_state);
        geist_error_set_create_time(
                GEIST_E_OOM, "geist_model_load", "failed to allocate model handle");
        return GEIST_E_OOM;
    }
    struct model_engine_state *eng =
            heap_alloc_aligned(sizeof(*eng), alignof(struct model_engine_state));
    if (eng == nullptr) {
        safe_free((void **) &m);
        desc->decoder_ops->state_destroy(arch_state);
        geist_error_set_create_time(
                GEIST_E_OOM, "geist_model_load", "failed to allocate engine-side state");
        return GEIST_E_OOM;
    }

    size_t path_len  = strlen(path);
    char  *path_copy = heap_alloc_aligned(path_len + 1, alignof(char));
    if (path_copy == nullptr) {
        safe_free((void **) &eng);
        safe_free((void **) &m);
        desc->decoder_ops->state_destroy(arch_state);
        geist_error_set_create_time(
                GEIST_E_OOM, "geist_model_load", "failed to allocate path string");
        return GEIST_E_OOM;
    }
    memcpy(path_copy, path, path_len + 1);

    /* Best-effort tokenizer load. Failure is non-fatal — caller can still
     * use geist_session_prefill_tokens with pre-tokenized IDs.
     *
     * Two paths (P1.6). Try GGUF-embedded first since it's the model's
     * own tokenizer; only fall back to external `tokenizer.bin` if the
     * GGUF doesn't ship a usable one (gguf_tokenizer_load_copy refuses
     * non-gpt2 models, so Gemma 4 sentencepiece-LLama variants flow to
     * the external sp_bpe path):
     *  1. GGUF-embedded vocab + merges — Llama, Mistral, SmolLM2, etc.
     *     gguf_tokenizer handles GPT-2-style byte-level BPE.
     *  2. External `tokenizer.bin` — Gemma 4 layout.
     *     sp_bpe_tokenizer handles SentencePiece-BPE. */
    struct sp_bpe_tokenizer *sp_tok    = nullptr;
    struct gguf_tokenizer   *gguf_tok  = nullptr;
    char                    *arch_copy = nullptr;
    {
        const char      *terr = nullptr;
        struct gguf_ctx *tg   = gguf_open(path, &terr);
        if (tg != nullptr) {
            arch_copy = model_arch_copy(tg);
            gguf_tok  = heap_alloc_aligned(sizeof(*gguf_tok), alignof(struct gguf_tokenizer));
            if (gguf_tok != nullptr) {
                if (!gguf_tokenizer_load_copy(gguf_tok, tg)) {
                    void *p = gguf_tok;
                    safe_free(&p);
                    gguf_tok = nullptr;
                }
            }
            gguf_close(tg);
        }
    }
    if (gguf_tok == nullptr) {
        char *tok_path = find_tokenizer_path(path);
        if (tok_path != nullptr) {
            if (!sp_bpe_tokenizer_load(&sp_tok, tok_path)) {
                sp_tok = nullptr;
            }
            safe_free((void **) &tok_path);
        }
    }

    /* aux_search_root: directory containing the GGUF, used by both audio
     * and vision encoders to locate their standalone safetensors files. */
    char *aux_root = nullptr;
    {
        const char *slash = strrchr(path, '/');
        if (slash != nullptr) {
            size_t dir_len = (size_t) (slash - path);
            aux_root       = heap_alloc_aligned(dir_len + 1, alignof(char));
            if (aux_root != nullptr) {
                memcpy(aux_root, path, dir_len);
                aux_root[dir_len] = '\0';
            }
        }
    }

    const char *text_only_env = getenv("GEIST_TEXT_ONLY");
    const bool  text_only     = text_only_env != nullptr && text_only_env[0] == '1';

    /* Best-effort load of the audio encoder. The Conformer needs a
     * safetensors file (not part of the GGUF) + mel constants. Failure is
     * non-fatal — text-only sessions keep working, attach_audio reports
     * GEIST_E_NOT_FOUND. */
    void *audio_state = nullptr;
    if (!text_only && desc->audio_encoder_ops != nullptr &&
        desc->audio_encoder_ops->state_create != nullptr) {
        audio_state = desc->audio_encoder_ops->state_create(be, aux_root);
    }

    /* Best-effort load of the vision encoder (Gemma 4 vision tower).
     * Looks for vision_tower.safetensors alongside the audio tower.
     * Failure is non-fatal — vision-less sessions keep working;
     * attach_image / attach_video will report GEIST_E_NOT_FOUND. */
    void *vision_state = nullptr;
    if (!text_only && desc->vision_encoder_ops != nullptr &&
        desc->vision_encoder_ops->state_create != nullptr) {
        vision_state = desc->vision_encoder_ops->state_create(be, aux_root);
    }

    if (aux_root != nullptr) {
        safe_free((void **) &aux_root);
    }

    *eng = (struct model_engine_state) {
            .path     = path_copy,
            .sp_tok   = sp_tok,
            .gguf_tok = gguf_tok,
            .arch     = arch_copy,
    };
    *m = (struct geist_model) {
            .text_decoder   = {.arch_ops = desc->decoder_ops, .arch_meta = arch_state},
            .audio_encoder  = {.arch_ops =
                                       audio_state != nullptr ? desc->audio_encoder_ops : nullptr,
                               .arch_meta = audio_state},
            .vision_encoder = {.arch_ops =
                                       vision_state != nullptr ? desc->vision_encoder_ops : nullptr,
                               .arch_meta = vision_state},
            .weights        = eng, /* engine-side state piggybacks here */
            .tokenizer      = sp_tok != nullptr ? (void *) sp_tok : (void *) gguf_tok,
            .backend        = be,
    };
    *out = m;
    return GEIST_OK;
}

[[nodiscard]] enum geist_status geist_model_load_from_memory(const void           *data,
                                                             size_t                size,
                                                             struct geist_backend *be,
                                                             struct geist_model  **out) {
    if (out == nullptr) {
        geist_error_set_create_time(
                GEIST_E_INVALID_ARG, "geist_model_load_from_memory", "out is null");
        return GEIST_E_INVALID_ARG;
    }
    *out = nullptr;
    if (data == nullptr || size == 0 || be == nullptr) {
        geist_error_set_create_time(GEIST_E_INVALID_ARG,
                                    "geist_model_load_from_memory",
                                    "data/size/backend is null or empty");
        return GEIST_E_INVALID_ARG;
    }

    const struct geist_arch_descriptor *desc = geist_arch_registry_lookup("gemma");
    if (desc == nullptr || desc->decoder_ops == nullptr ||
        desc->decoder_ops->state_create_from_memory == nullptr) {
        geist_error_set_create_time(
                GEIST_E_UNSUPPORTED,
                "geist_model_load_from_memory",
                "no decoder architecture supports memory loading in this build");
        return GEIST_E_UNSUPPORTED;
    }

    /* Weights are aliased zero-copy from `data` — it must outlive the model
     * (for an embedded blob it lives in .rodata, i.e. forever). No aux files
     * (tokenizer.bin / vision / audio safetensors) are searched: a memory blob
     * has no directory. The GGUF must carry its own tokenizer. */
    void *arch_state = desc->decoder_ops->state_create_from_memory(be, data, size, nullptr);
    if (arch_state == nullptr) {
        geist_error_set_create_time(GEIST_E_FORMAT,
                                    "geist_model_load_from_memory",
                                    "decoder state_create_from_memory failed "
                                    "(malformed GGUF or out of memory?)");
        return GEIST_E_FORMAT;
    }

    struct geist_model *m = heap_alloc_aligned(sizeof(*m), alignof(struct geist_model));
    if (m == nullptr) {
        desc->decoder_ops->state_destroy(arch_state);
        geist_error_set_create_time(
                GEIST_E_OOM, "geist_model_load_from_memory", "failed to allocate model handle");
        return GEIST_E_OOM;
    }
    struct model_engine_state *eng =
            heap_alloc_aligned(sizeof(*eng), alignof(struct model_engine_state));
    if (eng == nullptr) {
        safe_free((void **) &m);
        desc->decoder_ops->state_destroy(arch_state);
        geist_error_set_create_time(GEIST_E_OOM,
                                    "geist_model_load_from_memory",
                                    "failed to allocate engine-side state");
        return GEIST_E_OOM;
    }

    /* GGUF-embedded tokenizer only (no sibling tokenizer.bin to find). */
    struct gguf_tokenizer *gguf_tok  = nullptr;
    char                  *arch_copy = nullptr;
    {
        const char      *terr = nullptr;
        struct gguf_ctx *tg   = gguf_open_memory(data, size, &terr);
        if (tg != nullptr) {
            arch_copy = model_arch_copy(tg);
            gguf_tok  = heap_alloc_aligned(sizeof(*gguf_tok), alignof(struct gguf_tokenizer));
            if (gguf_tok != nullptr && !gguf_tokenizer_load_copy(gguf_tok, tg)) {
                void *p = gguf_tok;
                safe_free(&p);
                gguf_tok = nullptr;
            }
            gguf_close(tg);
        }
    }

    *eng = (struct model_engine_state) {
            .path     = nullptr, /* embedded — no file path */
            .sp_tok   = nullptr,
            .gguf_tok = gguf_tok,
            .arch     = arch_copy,
    };
    *m = (struct geist_model) {
            .text_decoder   = {.arch_ops = desc->decoder_ops, .arch_meta = arch_state},
            .audio_encoder  = {.arch_ops = nullptr, .arch_meta = nullptr},
            .vision_encoder = {.arch_ops = nullptr, .arch_meta = nullptr},
            .weights        = eng,
            .tokenizer      = (void *) gguf_tok,
            .backend        = be,
    };
    *out = m;
    return GEIST_OK;
}

void geist_model_destroy(struct geist_model *m) {
    if (m == nullptr) {
        return;
    }
    /* Tear down the decoder arch state. */
    if (m->text_decoder.arch_ops != nullptr && m->text_decoder.arch_meta != nullptr) {
        m->text_decoder.arch_ops->state_destroy(m->text_decoder.arch_meta);
    }
    /* Tear down the audio encoder state (best-effort loaded). */
    if (m->audio_encoder.arch_ops != nullptr && m->audio_encoder.arch_meta != nullptr) {
        m->audio_encoder.arch_ops->state_destroy(m->audio_encoder.arch_meta);
    }
    /* Tear down the vision encoder state (best-effort loaded). */
    if (m->vision_encoder.arch_ops != nullptr && m->vision_encoder.arch_meta != nullptr) {
        m->vision_encoder.arch_ops->state_destroy(m->vision_encoder.arch_meta);
    }
    /* Tear down engine-side state (path string, tokenizer). */
    struct model_engine_state *eng = (struct model_engine_state *) m->weights;
    if (eng != nullptr) {
        if (eng->sp_tok != nullptr) {
            sp_bpe_tokenizer_free(eng->sp_tok);
            eng->sp_tok = nullptr;
        }
        if (eng->gguf_tok != nullptr) {
            gguf_tokenizer_unload(eng->gguf_tok);
            void *p = eng->gguf_tok;
            safe_free(&p);
            eng->gguf_tok = nullptr;
        }
        if (eng->path != nullptr) {
            safe_free((void **) &eng->path);
        }
        if (eng->arch != nullptr) {
            safe_free((void **) &eng->arch);
        }
        safe_free((void **) &eng);
    }
    safe_free((void **) &m);
}

const char *geist_model_errmsg(const struct geist_model *m) {
    (void) m; /* TODO B-4d: per-handle err slot */
    return "(model errmsg not yet stored per-handle)";
}

static const struct model_engine_state *model_engine(const struct geist_model *m); /* fwd */

const char *geist_model_arch(const struct geist_model *m) {
    const struct model_engine_state *eng = model_engine(m);
    /* The GGUF's general.architecture ("gemma4", "bitnet-b1.58", "llama", …),
     * captured at load. Falls back to "transformer" when the key is absent. */
    return (eng != nullptr && eng->arch != nullptr) ? eng->arch : "transformer";
}

/* Special-token accessors — read the ids the tokenizer parsed from the GGUF
 * metadata (or tokenizer.bin). Both tokenizer paths default unset ids to -1,
 * which is GEIST_TOKEN_NONE. */
static const struct model_engine_state *model_engine(const struct geist_model *m) {
    if (m == nullptr || m->weights == nullptr) {
        return nullptr;
    }
    return (const struct model_engine_state *) m->weights;
}

geist_token_t geist_model_eos_token(const struct geist_model *m) {
    const struct model_engine_state *eng = model_engine(m);
    if (eng == nullptr) {
        return GEIST_TOKEN_NONE;
    }
    if (eng->gguf_tok != nullptr) {
        return (geist_token_t) eng->gguf_tok->eos_id;
    }
    if (eng->sp_tok != nullptr) {
        return (geist_token_t) sp_bpe_tokenizer_eos_id(eng->sp_tok);
    }
    return GEIST_TOKEN_NONE;
}

geist_token_t geist_model_bos_token(const struct geist_model *m) {
    const struct model_engine_state *eng = model_engine(m);
    if (eng == nullptr) {
        return GEIST_TOKEN_NONE;
    }
    if (eng->gguf_tok != nullptr) {
        return (geist_token_t) eng->gguf_tok->bos_id;
    }
    if (eng->sp_tok != nullptr) {
        return (geist_token_t) sp_bpe_tokenizer_bos_id(eng->sp_tok);
    }
    return GEIST_TOKEN_NONE;
}

geist_token_t geist_model_token_by_text(const struct geist_model *m, const char *text) {
    const struct model_engine_state *eng = model_engine(m);
    if (eng == nullptr || text == nullptr) {
        return GEIST_TOKEN_NONE;
    }
    /* Exact-vocab lookup is the GGUF-embedded tokenizer path (what a chat app
     * loading the .gguf directly uses). The external SentencePiece path has no
     * reverse lookup exposed; callers there rely on eos/bos. */
    if (eng->gguf_tok != nullptr) {
        return (geist_token_t) gguf_tokenizer_id_for_text(eng->gguf_tok, text);
    }
    return GEIST_TOKEN_NONE;
}

struct sp_bpe_tokenizer *geist_model_internal_tokenizer(struct geist_model *m) {
    if (m == nullptr || m->weights == nullptr) {
        return nullptr;
    }
    return ((struct model_engine_state *) m->weights)->sp_tok;
}

struct gguf_tokenizer *geist_model_internal_gguf_tokenizer(struct geist_model *m) {
    if (m == nullptr || m->weights == nullptr) {
        return nullptr;
    }
    return ((struct model_engine_state *) m->weights)->gguf_tok;
}

void *geist_model_internal_arch_meta(struct geist_model *m) {
    if (m == nullptr)
        return nullptr;
    return m->text_decoder.arch_meta;
}

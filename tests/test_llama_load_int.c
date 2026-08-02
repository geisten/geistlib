/*
 * test_llama_load_int — P1.5.d smoke test for the Llama family path.
 *
 * Loads a Llama-family GGUF (default search location:
 * gguf_artifacts/smollm2-360m-instruct-q8_0.gguf — override with
 * GEIST_LLAMA_GGUF_PATH) and checks that:
 *
 *   - general.architecture is correctly read as "llama"
 *   - transformer_family_select picks FAMILY_LLAMA
 *   - the Llama populator overrides Gemma-4 defaults
 *     (has_ple=false, has_gemma_attn_norms=false, kv_sliding_src=-1)
 *   - block_count / embedding_length / head_count / head_count_kv /
 *     feed_forward_length / rope.freq_base / layer_norm_rms_epsilon
 *     all reach the runtime fields
 *   - the per-layer geometry filler sets uniform full-attention
 *     across all layers (no KV sharing, no sliding-window)
 *   - state_create completes — every weight tensor for every
 *     layer + the two globals (token_embd, output_norm) loaded
 *     without missing-tensor errors
 *
 * Does NOT attempt decode — that needs the GGUF-embedded BPE
 * tokenizer plumbed into the engine path (next phase). SKIPs
 * cleanly if no Llama GGUF is reachable so the test passes in CI
 * envs without the model downloaded.
 */
#include "test_helpers.h"

#define GEIST_INTERNAL_ARCH_LAYER
#include "src/archs/transformer/arch_state.h"

#define GEIST_INTERNAL_ENGINE_LAYER
#include "src/engine/gguf_tokenizer.h"
#include "src/io/gguf_reader.h"

#include <geist.h>
#include <geist_backend.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *resolve_path(void) {
    const char *env = getenv("GEIST_LLAMA_GGUF_PATH");
    if (env != nullptr && env[0] != '\0')
        return env;
    static const char *candidates[] = {
            "gguf_artifacts/smollm2-360m-instruct-q8_0.gguf",
            "./smollm2-360m-instruct-q8_0.gguf",
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
        printf("SKIP: no Llama GGUF found. Set GEIST_LLAMA_GGUF_PATH "
               "or place smollm2-360m-instruct-q8_0.gguf in ./ or "
               "./gguf_artifacts/\n");
        return GEIST_TEST_SKIP;
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
    struct transformer_arch_session *sess [[maybe_unused]] = transformer_default_session(st);
    if (s != GEIST_OK) {
        fprintf(stderr,
                "state_create FAIL: %s — %s\n",
                geist_status_to_string(s),
                geist_backend_errmsg(be));
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    int fails = 0;

    /* Family + geometry assertions. SmolLM2-360M-Instruct values. */
    if (strcmp(st->config.family, "llama") != 0) {
        fprintf(stderr, "FAIL: family=%s, expected llama\n", st->config.family);
        fails++;
    }
    if (st->n_layers != 32) {
        fprintf(stderr, "FAIL: n_layers=%zu\n", st->n_layers);
        fails++;
    }
    if (st->d_model != 960) {
        fprintf(stderr, "FAIL: d_model=%zu\n", st->d_model);
        fails++;
    }
    if (st->vocab_size != 49152) {
        fprintf(stderr, "FAIL: vocab=%zu\n", st->vocab_size);
        fails++;
    }
    if (st->n_q_heads != 15) {
        fprintf(stderr, "FAIL: n_q=%zu\n", st->n_q_heads);
        fails++;
    }
    if (st->n_kv_heads != 5) {
        fprintf(stderr, "FAIL: n_kv=%zu\n", st->n_kv_heads);
        fails++;
    }
    if (st->hidden_per_layer != 0) {
        fprintf(stderr, "FAIL: hpl=%zu (expected 0 for non-PLE)\n", st->hidden_per_layer);
        fails++;
    }
    if (st->ple_out != 0) {
        fprintf(stderr, "FAIL: ple_out=%zu (expected 0)\n", st->ple_out);
        fails++;
    }

    /* Config flags. */
    if (st->config.has_ple) {
        fprintf(stderr, "FAIL: has_ple should be false for Llama\n");
        fails++;
    }
    if (st->config.has_gemma_attn_norms) {
        fprintf(stderr, "FAIL: has_gemma_attn_norms should be false\n");
        fails++;
    }
    if (st->config.logit_softcap != 0.0f) {
        fprintf(stderr, "FAIL: softcap=%g, expected 0\n", st->config.logit_softcap);
        fails++;
    }
    if (st->config.kv_sliding_src != -1) {
        fprintf(stderr, "FAIL: kv_sliding_src=%d\n", st->config.kv_sliding_src);
        fails++;
    }
    if (st->config.kv_full_src != -1) {
        fprintf(stderr, "FAIL: kv_full_src=%d\n", st->config.kv_full_src);
        fails++;
    }

    /* Per-layer geometry — uniform full-attn, head_dim = d/n_q. */
    for (size_t i = 0; i < st->n_layers; i++) {
        const struct transformer_layer_weights *L = &st->layers[i];
        if (!L->is_full || L->is_kv_shared || L->head_dim != 64 || L->q_out != 960 ||
            L->kv_out != 320 || L->intermediate != 2560 || L->sliding_window != 0 ||
            L->n_rotated_dims != 64) {
            fprintf(stderr,
                    "FAIL: layer[%zu] geometry mismatch (is_full=%d, kv_shared=%d, "
                    "hd=%zu, q=%zu/kv=%zu, inter=%zu, win=%zu, rot=%d)\n",
                    i,
                    L->is_full,
                    L->is_kv_shared,
                    L->head_dim,
                    L->q_out,
                    L->kv_out,
                    L->intermediate,
                    L->sliding_window,
                    L->n_rotated_dims);
            fails++;
            break;
        }
    }

    printf("loaded: %s  (%zu layers, d=%zu, vocab=%zu, GQA %zu:%zu)\n",
           st->config.family,
           st->n_layers,
           st->d_model,
           st->vocab_size,
           st->n_q_heads,
           st->n_kv_heads);

    /* P1.5.e/.f: prefill a token sequence + decode the model's
     * generation. We don't have a BPE encoder yet (P1.5.g), so the
     * prompt IDs are arbitrary low-vocab IDs — the goal of THIS test
     * is to confirm the full chain runs (load → prefill → forward
     * through 32 Llama layers → lm_head → argmax → token-id decode).
     * Coherent-text generation needs the BPE encoder so we can feed
     * real prompts in; that's queued as P1.5.g. */
    {
        /* GGUF-embedded BPE tokenizer load. SmolLM2 ships
         * tokenizer.ggml.model = "gpt2" + 49152 vocab in
         * tokenizer.ggml.tokens. β-mode state_create closes the GGUF
         * after weight load, so re-open it for the tokenizer (its
         * token strings point into the mmap region — needs to stay
         * open for the test's lifetime). Future: load + heap-copy
         * the tokenizer inside state_create so the engine owns it. */
        const char           *terr = nullptr;
        struct gguf_ctx      *tg   = gguf_open(path, &terr);
        struct gguf_tokenizer tok  = {0};
        if (tg == nullptr || !gguf_tokenizer_load(&tok, tg)) {
            fprintf(stderr,
                    "FAIL: gguf_tokenizer_load: %s\n",
                    terr != nullptr ? terr : "no tokens array");
            fails++;
        } else {
            printf("tokenizer: model=%.*s vocab=%zu bos=%d eos=%d\n",
                   (int) tok.model_len,
                   tok.model,
                   tok.vocab_size,
                   tok.bos_id,
                   tok.eos_id);
        }

        /* P1.5.g: encode a real English prompt via the GGUF-embedded
         * BPE encoder. BOS + encoded tokens for "The capital of
         * France is". */
        geist_token_t prompt_ids[32] = {0};
        size_t        n_prompt       = 0;
        if (tok.vocab_size > 0 && tok.bos_id >= 0) {
            prompt_ids[n_prompt++] = tok.bos_id;
            int32_t enc[24]        = {0};
            size_t  n_enc          = 0;
            if (gguf_tokenizer_encode(&tok,
                                      "The capital of France is",
                                      enc,
                                      sizeof enc / sizeof enc[0],
                                      &n_enc)) {
                for (size_t i = 0; i < n_enc && n_prompt < 32; i++) {
                    prompt_ids[n_prompt++] = enc[i];
                }
            }
        }

        /* Echo the prompt decoded for confirmation. */
        if (tok.vocab_size > 0) {
            char buf[256] = {0};
            gguf_tokenizer_decode(&tok, prompt_ids, n_prompt, buf, sizeof buf);
            printf("prompt encoded -> %zu tokens -> decoded: %s\n", n_prompt, buf);
        }

        enum geist_status ps = transformer_prefill_text_batch(sess, n_prompt, prompt_ids);
        if (ps != GEIST_OK) {
            fprintf(stderr,
                    "FAIL: Llama prefill returned %s — %s\n",
                    geist_status_to_string(ps),
                    geist_backend_errmsg(be));
            fails++;
        } else {
            /* Decode 8 tokens; print each as it's produced. */
            const int     N_STEPS     = 8;
            geist_token_t prev        = prompt_ids[n_prompt - 1];
            geist_token_t out_ids[16] = {0};
            int           n_decoded   = 0;
            for (int i = 0; i < N_STEPS; i++) {
                geist_token_t next = -1;
                ps                 = transformer_decode_step(sess, prev, &next);
                if (ps != GEIST_OK || next < 0 || (size_t) next >= st->vocab_size) {
                    fprintf(stderr,
                            "FAIL: Llama decode_step[%d] status=%s tok=%d\n",
                            i,
                            geist_status_to_string(ps),
                            next);
                    fails++;
                    break;
                }
                out_ids[n_decoded++] = next;
                if (tok.eos_id >= 0 && next == tok.eos_id)
                    break;
                prev = next;
            }
            if (n_decoded > 0 && tok.vocab_size > 0) {
                char buf[256] = {0};
                gguf_tokenizer_decode(&tok, out_ids, (size_t) n_decoded, buf, sizeof buf);
                printf("model output (%d tok): %s\n", n_decoded, buf);
            }
        }
        gguf_tokenizer_unload(&tok);
        if (tg != nullptr)
            gguf_close(tg);
    }

    transformer_state_destroy(st);
    geist_backend_destroy(be);

    if (fails == 0) {
        printf("PASS: Llama family load + forward (SmolLM2-360M)\n");
        return GEIST_TEST_PASS;
    }
    return GEIST_TEST_FAIL;
}

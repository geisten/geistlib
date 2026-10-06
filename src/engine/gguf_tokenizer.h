/*
 * src/engine/gguf_tokenizer.h — encoder/decoder for the tokenizer embedded
 * in a GGUF file (`tokenizer.ggml.*` metadata).
 *
 * Layer: ENGINE (internal).
 *
 * Decode allocates nothing: it walks the token strings and writes into a
 * caller-provided buffer.
 */
#ifndef GEIST_INTERNAL_ENGINE_GGUF_TOKENIZER_H
#define GEIST_INTERNAL_ENGINE_GGUF_TOKENIZER_H

#ifndef GEIST_INTERNAL_ENGINE_LAYER
#error "src/engine/gguf_tokenizer.h is internal to the engine layer."
#endif

#include "gguf_reader.h"

#include <geist.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Tokenization algorithm, picked from `tokenizer.ggml.model` at load.
 *   GPT2    — byte-level BPE (Ġ space marker, byte↔codepoint map).
 *   SPM     — SentencePiece-style BPE (▁ space marker, <0xXX> byte fallback),
 *             driven by the embedded merges. Covers Gemma / Mistral.
 *   UNIGRAM — SentencePiece unigram (▁ marker, <0xXX> fallback), driven by the
 *             per-token scores (no merges): a greedy pairwise merge whose
 *             priority is the merged token's score, llama.cpp's SPM algorithm,
 *             not a Viterbi search (see unigram_chunk_to_ids). Covers the classic
 *             Llama/SentencePiece vocab — incl. BitNet b1.58, which ships
 *             tokenizer.ggml.model="llama" with scores but no merges.
 * UNSUPPORTED leaves the encode/decode paths refusing so the engine falls
 * back to an external tokenizer.bin. */
enum gguf_tokenizer_mode {
    GGUF_TOK_MODE_UNSUPPORTED = 0,
    GGUF_TOK_MODE_GPT2,
    GGUF_TOK_MODE_SPM,
    GGUF_TOK_MODE_UNIGRAM,
};

struct gguf_tokenizer {
    /* Pointer-and-length pair for each vocab entry. The pointer
     * indexes into the GGUF mmap; valid for the ctx's lifetime. */
    const char **token_str;
    size_t      *token_len;
    size_t       vocab_size;

    /* UNIGRAM: per-token scores from tokenizer.ggml.scores (rank-like, ≈ -id),
     * owned heap copy (4 B × vocab, always copied at load). null for gpt2. */
    float *scores;

    /* BPE merges. Each entry encodes "left right" as a
     * single GGUF string; we split on the first space at load time.
     * Rank = array index (0 = highest priority). */
    const char **merge_left;
    size_t      *merge_left_len;
    const char **merge_right;
    size_t      *merge_right_len;
    size_t       n_merges;

    /* Special token IDs from `tokenizer.ggml.<bos|eos|unknown>_token_id`.
     * -1 when the corresponding key is absent. */
    int32_t bos_id;
    /* tokenizer.ggml.add_bos_token: whether encode should PREPEND bos.
     * Qwen-family GGUFs carry a bos_id but set this false — blindly
     * prepending desynced the qwen35 recurrent state (#281). Absent:
     * prepend iff bos_id is set and the mode is not GPT2. */
    bool add_bos;

    /* tokenizer.ggml.add_eos_token: whether the model expects eos APPENDED.
     * No generative GGUF in tree sets it; the BitNet embedding checkpoints
     * do, and pooling over a sequence that is one token short of upstream's
     * shifts the vector. Absent = false. */
    bool    add_eos;
    int32_t eos_id;
    int32_t unk_id;

    /* tokenizer.ggml.model ("gpt2", "llama", ...); `mode` is resolved from
     * it. GGUF strings are length-prefixed, NOT NUL-terminated — use
     * `model_len` for any comparison. */
    const char *model; /* points into GGUF mmap; not owned */
    size_t      model_len;

    /* Resolved algorithm + SPM-only state (set at load). */
    enum gguf_tokenizer_mode mode;
    bool                     add_space_prefix; /* SPM: prepend ▁ to the input (add_dummy_prefix) */
    /* GPT-2 mode: tokenizer.ggml.pre == "qwen2" — encode chunks via the
     * qwen2 pretokenizer scanner instead of the simple word splitter
     * (contractions, single digits, punctuation runs, the one-codepoint
     * prefix rule; #275). Also set for pre == "qwen35". */
    bool    pre_qwen2;
    int32_t spm_byte_id[256]; /* SPM byte fallback: byte → "<0xXX>" vocab id, -1 if absent */

    /* Open-addressed hash indices over the vocab + merges arrays. A slot
     * holds the array index; -1 (vocab) or SIZE_MAX (merges) marks empty.
     * Power-of-2 sized, load factor ~0.5, mask = size - 1. Built at load;
     * null when allocation failed (lookups then scan linearly). */
    int32_t *vocab_hash;
    size_t   vocab_hash_mask;
    size_t  *merge_hash;
    size_t   merge_hash_mask;

    /* Owned heap arenas. Non-null when the tokenizer was loaded
     * in copy mode (gguf_tokenizer_load_copy) — then token_str /
     * merge_*_str point into these instead of into the GGUF mmap,
     * letting the caller close the gguf_ctx after load.
     *
     * Layout: a single flat byte buffer holding the concatenated
     * strings; each token/merge slot's (ptr, len) indexes into it. */
    char *vocab_arena; /* not null = copy mode for vocab */
    char *merge_arena; /* not null = copy mode for merges */
    char *model_arena; /* not null = copy mode for model name */

    /* Special / added tokens — entries in `tokenizer.ggml.tokens` whose
     * `tokenizer.ggml.token_type` is CONTROL(3) or USER_DEFINED(4).
     * Sorted longest-first so the encoder can do leftmost-longest pre-
     * scan: at each text position try every special's exact string
     * first, and only fall through to byte-level BPE when none match.
     * Without this Llama3 specials like `<|begin_of_text|>` get
     * shredded into `< | begin _ of _ text | >`. */
    struct gguf_tokenizer_special {
        const char *text;
        size_t      len;
        int32_t     id;
    }     *specials;
    size_t n_specials;
};

/* Load token table + special-ID metadata. Allocates two pointer
 * arrays sized to vocab; freed in gguf_tokenizer_unload. Returns
 * false (and zeroes *tok) if the GGUF lacks the required keys
 * (`tokenizer.ggml.tokens` and `tokenizer.ggml.model`). */
[[nodiscard]] bool gguf_tokenizer_load(struct gguf_tokenizer *tok, const struct gguf_ctx *ctx);

/* Copy-mode loader. Like gguf_tokenizer_load, but copies every vocab,
 * merge and model-name byte into heap arenas, so the caller may close the
 * gguf_ctx afterwards. Also returns false for GGUF_TOK_MODE_UNSUPPORTED. */
[[nodiscard]] bool gguf_tokenizer_load_copy(struct gguf_tokenizer *tok, const struct gguf_ctx *ctx);

void gguf_tokenizer_unload(struct gguf_tokenizer *tok);

/* Exact vocab lookup: token id for the surface string `text`, or -1 if it is
 * not a single vocab entry. Backs the public geist_model_token_by_text. */
int32_t gguf_tokenizer_id_for_text(const struct gguf_tokenizer *tok, const char *text);

/* Decode `n` token IDs into `out` (UTF-8 bytes, no trailing NUL). The
 * caller-supplied buffer must hold `out_cap` bytes. Returns the
 * number of bytes written; if `out_cap` is too small, writes as much
 * as fits and returns the would-be total. Token IDs outside the
 * vocab are written as `<unk>` placeholders. */
size_t gguf_tokenizer_decode(
        const struct gguf_tokenizer *tok, const int32_t *ids, size_t n, char *out, size_t out_cap);

/* Encode a UTF-8 text string into token IDs.
 *
 *   text       NUL-terminated UTF-8 input.
 *   cap        Room in out_ids.
 *   out_ids    Caller-allocated array, `cap` slots; may be null when cap is 0.
 *   n_out      Number of IDs written; 0 on every failure.
 *
 * GEIST_OK when every ID fit. A text that needs more than `cap` IDs is
 * GEIST_E_INVALID_ARG with nothing written, never a shortened encoding
 * (AGENT.md §5); GEIST_E_OOM when scratch allocation fails;
 * GEIST_E_UNSUPPORTED for a tokenizer mode that cannot encode. For
 * tokenizer.ggml.model == "gpt2" the encoder:
 *   1. Pre-tokenizes the input into word chunks (simplified GPT-2-
 *      style: leading-space-aware runs of letters / digits / other-
 *      punctuation / whitespace).
 *   2. Maps each chunk's bytes to the GPT-2 unicode codepoints
 *      (forward of the decoder's inverse map).
 *   3. Applies BPE merges from `tokenizer.ggml.merges` greedily,
 *      lowest rank first, until no more merges apply.
 *   4. Looks up each final symbol in the vocab to get its ID.
 *
 * The pre-tokenizer is a simplification — it doesn't implement the
 * full GPT-2 regex, so some inputs (apostrophes, complex Unicode)
 * may tokenize slightly differently from a reference encoder. ASCII
 * English text encodes correctly. */
[[nodiscard]] enum geist_status gguf_tokenizer_encode(const struct gguf_tokenizer *tok,
                                                      const char                  *text,
                                                      size_t                       cap,
                                                      int32_t                     *out_ids,
                                                      size_t                      *n_out);

#endif /* GEIST_INTERNAL_ENGINE_GGUF_TOKENIZER_H */

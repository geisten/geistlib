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
 * back to an external tokenizer.bin, which gguf_tokenizer_load_bin reads into
 * the same struct in SPM mode. */
enum gguf_tokenizer_mode {
    GGUF_TOK_MODE_UNSUPPORTED = 0,
    GGUF_TOK_MODE_GPT2,
    GGUF_TOK_MODE_SPM,
    GGUF_TOK_MODE_UNIGRAM,
};

/* The regex-sequence pretokenizers (gguf_tokenizer_seq_split). */
enum gguf_pre_seq {
    GGUF_PRE_NONE = 0,
    GGUF_PRE_DEEPSEEK_LLM, /* tokenizer.ggml.pre "deepseek-llm" (DeepSeek-V2) */
    GGUF_PRE_DEEPSEEK_V3,  /* "deepseek-v3" (DeepSeek-V3, R1) */
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
    /* Whether the model expects BOS prepended, decided as llama.cpp does
     * (gguf_tokenizer_add_bos_rule): tokenizer.ggml.add_bos_token when the
     * file has it, else the default of its vocabulary and pretokenizer —
     * true for Llama 3's byte-BPE ("llama-bpe", #385), false for Qwen's,
     * whose GGUFs carry a bos_id that must not be prepended (#281). False
     * when the file has neither the flag nor a bos_id. */
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
    /* GPT-2 mode: the pretokenizer is llama.cpp's llama3 / qwen2 regex
     * (gguf_tokenizer_llama3_chunk_end: contractions, the one-codepoint
     * prefix rule, digit groups, punctuation runs; #275, #399) instead of
     * the simple word splitter. pre_digits is the most digits one chunk takes: 3 for Llama 3
     * ("llama-bpe", "llama3", ...), 1 for "qwen2" / "qwen35"; 0 when the
     * file uses another pretokenizer. pre_marks: letter runs take
     * combining marks too ("qwen35"). pre_ignore_merges: a chunk that is a
     * vocab entry is that one token, as llama.cpp's ignore_merges does for
     * Llama 3. */
    uint8_t pre_digits;
    bool    pre_marks;
    bool    pre_ignore_merges;
    /* GPT-2 mode: tokenizer.ggml.pre == "gpt-4o" or "llama4" (#243) —
     * gguf_tokenizer_gpt4o_chunk_end splits the text. */
    bool pre_gpt4o;
    /* GPT-2 mode: llama.cpp runs the GPT-2 regex for the pretokenizer
     * ("gpt-2", "olmo", ...; gguf_tokenizer_gpt2_chunk_end, #403).
     * pre_split_digits: after its \p{N} split ("smollm", "starcoder", ...). */
    bool pre_gpt2;
    bool pre_split_digits;
    /* GPT-2 mode: a pretokenizer llama.cpp runs as a sequence of regexes
     * (gguf_tokenizer_seq_split): "deepseek-llm", "deepseek-v3" (#242).
     * GGUF_PRE_NONE otherwise. */
    enum gguf_pre_seq pre_seq;
    int32_t spm_byte_id[256]; /* SPM byte fallback: byte → "<0xXX>" vocab id, -1 if absent */

    /* Loaded from a tokenizer.bin (gguf_tokenizer_load_bin), SPM mode without
     * the ▁ prefix. Keeps what that path has always answered (#263):
     *   - encode splits symbols by the UTF-8 lead byte alone (a truncated
     *     sequence swallows the bytes after it), not by valid sequences;
     *   - geist_session_token_to_str returns the stored piece undecoded, and
     *     nullptr past the vocab;
     *   - geist_model_token_by_text finds nothing, DRY sequence breakers are
     *     not resolved, and a draft model is accepted only if it is the same
     *     model. */
    bool tokenizer_bin;

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

/* Load a tokenizer.bin ("GMTK" v1: header, u16-length-prefixed tokens, merge
 * pairs in rank order, then (id, text) special tokens) into *tok, owning a
 * heap copy of the file. Prints why on stderr and returns false (with *tok
 * unloaded) for a file it cannot read or that its counts overrun. */
[[nodiscard]] bool gguf_tokenizer_load_bin(struct gguf_tokenizer *tok, const char *path);

void gguf_tokenizer_unload(struct gguf_tokenizer *tok);

/* End (exclusive, a byte offset) of the gpt-4o pretokenizer chunk that
 * starts at byte i < tlen of text: llama.cpp's LLAMA_VOCAB_PRE_TYPE_GPT4O
 * regex over llama.cpp's letter / number / whitespace classes. Always > i.
 * Exposed for tests. */
[[nodiscard]] size_t
gguf_tokenizer_gpt4o_chunk_end(size_t tlen, size_t i, const char text[static tlen]);

/* End (exclusive, a byte offset) of the llama3 / qwen2 pretokenizer chunk
 * that starts at byte i < tlen of text (llama.cpp's
 * unicode_regex_split_custom_llama3 / _qwen2 / _qwen35): max_digits is 3
 * for Llama 3's \p{N}{1,3}, 1 for Qwen's \p{N}; marks puts \p{M} into the
 * letter class (qwen35). Always > i. Exposed for tests. */
[[nodiscard]] size_t gguf_tokenizer_llama3_chunk_end(
        size_t tlen, size_t i, size_t max_digits, bool marks, const char text[static tlen]);

/* End (exclusive, a byte offset) of the GPT-2 pretokenizer chunk that
 * starts at byte i < tlen of text (llama.cpp's
 * unicode_regex_split_custom_gpt2). split_digits: the text went through
 * llama.cpp's "\p{N}" regex first, so every \p{N} codepoint is a chunk of
 * its own and ends the fragment the regex sees (smollm). Always > i.
 * Exposed for tests. */
[[nodiscard]] size_t gguf_tokenizer_gpt2_chunk_end(size_t     tlen,
                                                   size_t     i,
                                                   bool       split_digits,
                                                   const char text[static tlen]);

/* Split n codepoints as llama.cpp's regex sequence for `kind` does
 * (GGUF_PRE_DEEPSEEK_LLM or _V3): the piece lengths, in codepoints and in
 * order, land in pieces[0, return value); pieces[n, 2n) is scratch. Every
 * piece is non-empty and they sum to n. Exposed for tests. */
[[nodiscard]] size_t gguf_tokenizer_seq_split(enum gguf_pre_seq kind,
                                              size_t            n,
                                              const uint32_t    cps[static n],
                                              size_t            pieces[static 2 * n]);

/* Whether a GGUF tokenizer adds BOS, by llama.cpp's rule
 * (llama_vocab::impl::load in llama-vocab.cpp; llama-perplexity puts BOS at
 * the start of every chunk exactly when this is true):
 *   1. the default of the vocabulary type, from tokenizer.ggml.model:
 *      "llama" (SPM) and "bert" (WPM) true; byte-BPE ("gpt2", "gemma4", ...)
 *      true only for the pretokenizers (tokenizer.ggml.pre) whose branch sets
 *      it — "llama3", "llama-v3", "llama-bpe", "falcon3", "falcon-h1",
 *      "pixtral", "midm-2.0", "lfm2", "jina-v5-nano", "tekken", "chameleon";
 *      "t5", "rwkv", "plamo2" false;
 *   2. tokenizer.ggml.add_bos_token, when present, replaces that default;
 *   3. a Gemma 4 byte-BPE vocabulary (model "gemma4", or pre "gemma4" /
 *      "granite-embed-multi-311m") adds BOS even when the flag says false.
 * model[0, model_len) and pre[0, pre_len) are the metadata strings (not
 * NUL-terminated; pre null when absent). flag is the add_bos_token value
 * (0/1), or -1 when the key is absent. fallback is the default for a model
 * name llama.cpp does not know (it refuses to load those). */
[[nodiscard]] bool gguf_tokenizer_add_bos_rule(size_t      model_len,
                                               size_t      pre_len,
                                               int         flag,
                                               bool        fallback,
                                               const char *model,
                                               const char *pre);

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
 *   1. Pre-tokenizes the input into word chunks: llama.cpp's regex for
 *      the llama3 ("llama-bpe", ...), qwen2, qwen35, gpt-4o / llama4,
 *      deepseek-llm / deepseek-v3 and GPT-2 ("gpt-2", "smollm", ...)
 *      pretokenizers; for any other
 *      tokenizer.ggml.pre a simplified GPT-2 split (leading-space-aware
 *      runs of letters / digits / single punctuation bytes / whitespace),
 *      which differs from llama.cpp on digits, punctuation runs,
 *      apostrophes and non-ASCII text.
 *   2. Maps each chunk's bytes to the GPT-2 unicode codepoints
 *      (forward of the decoder's inverse map).
 *   3. Applies BPE merges from `tokenizer.ggml.merges` greedily,
 *      lowest rank first, until no more merges apply (a llama3 chunk
 *      that is itself a vocab entry is that token, ignore_merges).
 *   4. Looks up each final symbol in the vocab to get its ID. */
[[nodiscard]] enum geist_status gguf_tokenizer_encode(const struct gguf_tokenizer *tok,
                                                      const char                  *text,
                                                      size_t                       cap,
                                                      int32_t                     *out_ids,
                                                      size_t                      *n_out);

#endif /* GEIST_INTERNAL_ENGINE_GGUF_TOKENIZER_H */

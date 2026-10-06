/*
 * SentencePiece-style BPE tokenizer for HF tokenizers v4+ format
 * (Gemma, Llama, Mistral, Qwen, ...).
 *
 * Differences from GPT-2-style byte-level BPE:
 *   - Normalizer: " " (0x20) -> "▁" (U+2581, UTF-8 0xE2 0x96 0x81)
 *     not " " -> "Ġ" (U+0120, GPT-2 byte map)
 *   - No pre-tokenizer split at space/word/digit boundaries — BPE
 *     operates over the entire normalized chunk between specials
 *   - Reads a custom binary format (tokenizer.bin) produced by
 *     convert_tokenizer.py — no JSON parsing in C
 *
 * Algorithm:
 *   1. Scan input for longest-match special tokens. Split into
 *      [chunk_0, special_0, chunk_1, special_1, ..., chunk_N].
 *   2. For each chunk: normalize " " -> "▁", then rank-ordered BPE merges.
 *   3. Concatenate token IDs in order.
 */
#ifndef SP_BPE_TOKENIZER_H
#define SP_BPE_TOKENIZER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct sp_bpe_tokenizer;

[[nodiscard]] bool sp_bpe_tokenizer_load(struct sp_bpe_tokenizer **out, const char *bin_path);

void sp_bpe_tokenizer_free(struct sp_bpe_tokenizer *tok);

uint32_t sp_bpe_tokenizer_vocab_size(const struct sp_bpe_tokenizer *tok);
uint32_t sp_bpe_tokenizer_bos_id(const struct sp_bpe_tokenizer *tok);
uint32_t sp_bpe_tokenizer_eos_id(const struct sp_bpe_tokenizer *tok);
uint32_t sp_bpe_tokenizer_pad_id(const struct sp_bpe_tokenizer *tok);
uint32_t sp_bpe_tokenizer_unk_id(const struct sp_bpe_tokenizer *tok);
uint32_t sp_bpe_tokenizer_specials_count(const struct sp_bpe_tokenizer *tok);

/* Encode a NUL-terminated UTF-8 string into token IDs.
 * Caller frees *tokens_out via safe_free((void **) &tokens_out). */
[[nodiscard]] bool sp_bpe_tokenizer_encode(const struct sp_bpe_tokenizer *tok,
                                           const char                    *text,
                                           uint32_t                     **tokens_out,
                                           size_t                        *count_out);

/* Look up token text by ID. Returns text+len pair or (nullptr, 0) on error.
 * Text points into the mmap region — do not free. */
const char *
sp_bpe_tokenizer_id_to_text(const struct sp_bpe_tokenizer *tok, uint32_t id, size_t *len_out);

#endif

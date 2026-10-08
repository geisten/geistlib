/*
 * src/engine/dry_breakers.h — DRY sequence breakers (#695): strings to the
 * token sequences the sampler matches.
 *
 * Layer: ENGINE (needs the tokenizer, which the architecture does not see).
 */
#ifndef GEIST_INTERNAL_DRY_BREAKERS_H
#define GEIST_INTERNAL_DRY_BREAKERS_H

#ifndef GEIST_INTERNAL_ENGINE_LAYER
#error "dry_breakers.h is internal — define GEIST_INTERNAL_ENGINE_LAYER."
#endif

#include <geist.h>

#include <stddef.h>

struct gguf_tokenizer;

/* llama.cpp's defaults (common/common.h, dry_sequence_breakers). */
extern const char *const geist_dry_default_breakers[4];

/* llama.cpp's get_overlapping_token_sequences for each of the n_strs
 * strings (null and empty ones skipped, each cut to 40 bytes): every token
 * whose text contains the string is a single-token breaker; a token whose
 * text ends in a prefix of the string is a head whose tail is the rest of
 * the string, tokenized and cut to 20 tokens. Writes the sequences packed
 * as set_dry_breakers takes them (geist_arch.h) to a heap block in *out,
 * its length in *n_words; *out is nullptr when nothing matched. */
[[nodiscard]] enum geist_status geist_dry_breakers_resolve(size_t                      *n_words,
                                                           geist_token_t              **out,
                                                           const struct gguf_tokenizer *tok,
                                                           size_t                       n_strs,
                                                           const char *const           *strs);

#endif /* GEIST_INTERNAL_DRY_BREAKERS_H */

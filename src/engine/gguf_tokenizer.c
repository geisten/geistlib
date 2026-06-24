/*
 * src/engine/gguf_tokenizer.c — see gguf_tokenizer.h.
 *
 * GPT-2 byte-level BPE decode: the original byte-encoder maps the 256
 * bytes onto 256 unicode codepoints, choosing the 188 printable ASCII
 * + Latin-1 bytes to stay identity and assigning the remaining 68
 * "ugly" bytes (0..32, 127..160, 173) to unicode codepoints starting
 * at 256. The token strings stored in the GGUF are UTF-8 encodings of
 * those mapped codepoints; decode reverses the mapping byte-by-byte.
 *
 * SmolLM2 / Llama 3 / GPT-NeoX / GPT-2 all use the same scheme; the
 * mapping table is computed once at module init.
 */
#define GEIST_INTERNAL_ENGINE_LAYER

#include "gguf_tokenizer.h"

#include "heap.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Build the gpt2 codepoint → byte inverse map. Walks the same logic
 * as huggingface_hub's bytes_to_unicode(): 188 identity bytes plus
 * 68 remapped bytes at codepoints 256..323. Returns -1 for codepoints
 * outside the map (e.g. real multi-byte unicode that the user
 * embedded in a token name; rare). */
static int gpt2_codepoint_to_byte(uint32_t cp) {
    /* Identity-mapped byte ranges (printable ASCII + Latin-1). */
    if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) || (cp >= 174 && cp <= 255)) {
        return (int) cp;
    }
    /* Remapped bytes live at codepoints 256.. in insertion order:
     *  - bytes 0..32       → 256..288   (33 entries)
     *  - bytes 127..160    → 289..322   (34 entries)
     *  - byte  173         → 323
     * Total 68 entries. */
    if (cp >= 256 && cp <= 288)
        return (int) (cp - 256); /* 0..32 */
    if (cp >= 289 && cp <= 322)
        return (int) (cp - 289 + 127); /* 127..160 */
    if (cp == 323)
        return 173;
    return -1;
}

/* Forward of gpt2_codepoint_to_byte: maps each of the 256 byte values
 * to a codepoint in [0, 323]. P1.5.g (encoder). */
static uint32_t gpt2_byte_to_codepoint(uint8_t b) {
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174) {
        return b; /* identity (174..255 is the open-ended upper range) */
    }
    if (b <= 32)
        return 256u + b; /* 0..32 → 256..288 */
    if (b >= 127 && b <= 160)
        return 289u + (b - 127); /* 127..160 → 289..322 */
    return 323;                  /* b == 173 */
}

/* Encode a single codepoint as UTF-8 into `out` (≥ 4 bytes). Returns
 * the byte count written. */
static size_t utf8_encode_one(uint32_t cp, char *out) {
    if (cp < 0x80) {
        out[0] = (char) cp;
        return 1;
    }
    if (cp < 0x800) {
        out[0] = (char) (0xC0 | (cp >> 6));
        out[1] = (char) (0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (char) (0xE0 | (cp >> 12));
        out[1] = (char) (0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char) (0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (char) (0xF0 | (cp >> 18));
    out[1] = (char) (0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char) (0x80 | ((cp >> 6) & 0x3F));
    out[3] = (char) (0x80 | (cp & 0x3F));
    return 4;
}

/* Decode a single UTF-8 codepoint starting at `s`. Returns the
 * codepoint and writes the byte count consumed into *advance.
 * Returns -1 on malformed UTF-8 (and sets *advance = 1 to keep
 * progress). */
static int32_t utf8_decode_one(const char *s, size_t n, size_t *advance) {
    if (n == 0) {
        *advance = 0;
        return -1;
    }
    const unsigned char b0 = (unsigned char) s[0];
    if (b0 < 0x80) {
        *advance = 1;
        return b0;
    }
    if ((b0 & 0xE0) == 0xC0 && n >= 2 && (s[1] & 0xC0) == 0x80) {
        *advance = 2;
        return ((int32_t) (b0 & 0x1F) << 6) | (int32_t) (s[1] & 0x3F);
    }
    if ((b0 & 0xF0) == 0xE0 && n >= 3 && (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80) {
        *advance = 3;
        return ((int32_t) (b0 & 0x0F) << 12) | ((int32_t) (s[1] & 0x3F) << 6) |
               (int32_t) (s[2] & 0x3F);
    }
    if ((b0 & 0xF8) == 0xF0 && n >= 4 && (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80 &&
        (s[3] & 0xC0) == 0x80) {
        *advance = 4;
        return ((int32_t) (b0 & 0x07) << 18) | ((int32_t) (s[1] & 0x3F) << 12) |
               ((int32_t) (s[2] & 0x3F) << 6) | (int32_t) (s[3] & 0x3F);
    }
    *advance = 1;
    return -1;
}

/* Forward decls for hash helpers defined further down (P1.5.h). */
static uint64_t fnv1a64(const char *p, size_t n);
static uint64_t fnv1a64_pair(const char *a, size_t alen, const char *b, size_t blen);
static size_t   next_pow2(size_t n);
static int32_t  vocab_lookup(const struct gguf_tokenizer *tok, const char *bytes, size_t len);

/* SentencePiece space marker U+2581 (▁) in UTF-8. */
static const char SPM_MARKER[3] = {(char) 0xE2, (char) 0x96, (char) 0x81};
#define SPM_MARKER_LEN 3

[[nodiscard]] bool gguf_tokenizer_load(struct gguf_tokenizer *tok, const struct gguf_ctx *ctx) {
    if (tok == nullptr)
        return false;
    *tok        = (struct gguf_tokenizer) {0};
    tok->bos_id = -1;
    tok->eos_id = -1;
    tok->unk_id = -1;

    tok->model = gguf_get_meta_string(ctx, "tokenizer.ggml.model", &tok->model_len);
    if (tok->model == nullptr)
        return false;

    uint32_t       elem_vt;
    uint64_t       count;
    const uint8_t *p;
    if (!gguf_get_meta_array_info(ctx, "tokenizer.ggml.tokens", &elem_vt, &count, &p))
        return false;
    if (elem_vt != GGUF_META_VT_STRING || count == 0)
        return false;

    tok->token_str = heap_alloc_array_aligned(const char *, (size_t) count);
    tok->token_len = heap_alloc_array_aligned(size_t, (size_t) count);
    if (tok->token_str == nullptr || tok->token_len == nullptr) {
        gguf_tokenizer_unload(tok);
        return false;
    }
    tok->vocab_size = (size_t) count;

    for (uint64_t i = 0; i < count; i++) {
        uint64_t slen;
        memcpy(&slen, p, 8);
        p += 8;
        tok->token_str[i] = (const char *) p;
        tok->token_len[i] = (size_t) slen;
        p += slen;
    }

    /* P1.5.g: load tokenizer.ggml.merges. Each entry is "left right"
     * (one space separator). Split on first space at load time so
     * the encoder can compare both halves without re-parsing. */
    if (gguf_get_meta_array_info(ctx, "tokenizer.ggml.merges", &elem_vt, &count, &p) &&
        elem_vt == GGUF_META_VT_STRING && count > 0) {
        tok->merge_left      = heap_alloc_array_aligned(const char *, (size_t) count);
        tok->merge_left_len  = heap_alloc_array_aligned(size_t, (size_t) count);
        tok->merge_right     = heap_alloc_array_aligned(const char *, (size_t) count);
        tok->merge_right_len = heap_alloc_array_aligned(size_t, (size_t) count);
        if (tok->merge_left == nullptr || tok->merge_left_len == nullptr ||
            tok->merge_right == nullptr || tok->merge_right_len == nullptr) {
            gguf_tokenizer_unload(tok);
            return false;
        }
        tok->n_merges = (size_t) count;
        for (uint64_t i = 0; i < count; i++) {
            uint64_t slen;
            memcpy(&slen, p, 8);
            p += 8;
            const char *s = (const char *) p;
            p += slen;
            size_t sp = 0;
            while (sp < slen && s[sp] != ' ')
                sp++;
            tok->merge_left[i]     = s;
            tok->merge_left_len[i] = sp;
            if (sp < slen) {
                tok->merge_right[i]     = s + sp + 1;
                tok->merge_right_len[i] = slen - sp - 1;
            } else {
                tok->merge_right[i]     = nullptr;
                tok->merge_right_len[i] = 0;
            }
        }
    }

    /* tokenizer.ggml.scores — per-token log-probs for the unigram model. Copied
     * into an owned heap array (small, 4 B × vocab) so it survives a non-copy
     * load too; only present/needed for the unigram (no-merges) path. */
    if (gguf_get_meta_array_info(ctx, "tokenizer.ggml.scores", &elem_vt, &count, &p) &&
        elem_vt == GGUF_META_VT_F32 && count == tok->vocab_size) {
        tok->scores = heap_alloc_array_aligned(float, tok->vocab_size);
        if (tok->scores != nullptr)
            memcpy(tok->scores, p, tok->vocab_size * sizeof(float));
    }

    uint32_t u;
    if (gguf_get_meta_u32(ctx, "tokenizer.ggml.bos_token_id", &u))
        tok->bos_id = (int32_t) u;
    if (gguf_get_meta_u32(ctx, "tokenizer.ggml.eos_token_id", &u))
        tok->eos_id = (int32_t) u;
    if (gguf_get_meta_u32(ctx, "tokenizer.ggml.unknown_token_id", &u))
        tok->unk_id = (int32_t) u;

    /* Special / added tokens. GGUF convention: tokenizer.ggml.token_type
     * is a parallel uint32 array. type 3 = CONTROL (Llama3's
     * <|begin_of_text|>, <|eot_id|>, etc.), type 4 = USER_DEFINED. We
     * collect both, sort by len descending so leftmost-longest matching
     * in encode is a simple linear scan. */
    {
        uint32_t       tt_vt = 0;
        uint64_t       tt_n  = 0;
        const uint8_t *tt_p  = nullptr;
        if (gguf_get_meta_array_info(ctx, "tokenizer.ggml.token_type", &tt_vt, &tt_n, &tt_p) &&
            (tt_vt == GGUF_META_VT_U32 || tt_vt == GGUF_META_VT_I32) && tt_n == tok->vocab_size) {
            size_t n_sp = 0;
            for (size_t i = 0; i < tok->vocab_size; i++) {
                int32_t t;
                memcpy(&t, tt_p + i * 4, 4);
                if (t == 3 || t == 4)
                    n_sp++;
            }
            if (n_sp > 0) {
                tok->specials = heap_alloc_array_aligned(struct gguf_tokenizer_special, n_sp);
                if (tok->specials != nullptr) {
                    size_t k = 0;
                    for (size_t i = 0; i < tok->vocab_size; i++) {
                        int32_t t;
                        memcpy(&t, tt_p + i * 4, 4);
                        if (t != 3 && t != 4)
                            continue;
                        tok->specials[k].text = tok->token_str[i];
                        tok->specials[k].len  = tok->token_len[i];
                        tok->specials[k].id   = (int32_t) i;
                        k++;
                    }
                    tok->n_specials = k;
                    /* Sort by len desc — leftmost-longest match at encode. */
                    for (size_t a = 1; a < k; a++) {
                        struct gguf_tokenizer_special tmp = tok->specials[a];
                        size_t                        b   = a;
                        while (b > 0 && tok->specials[b - 1].len < tmp.len) {
                            tok->specials[b] = tok->specials[b - 1];
                            b--;
                        }
                        tok->specials[b] = tmp;
                    }
                }
            }
        }
    }

    /* P1.5.h: build the vocab + merges hash indices. Both tables are
     * pow-of-2 sized with load factor ~0.5; failure to allocate is
     * non-fatal — the encoder's linear-scan fallback kicks in. */
    {
        const size_t vsize = next_pow2(tok->vocab_size * 2);
        tok->vocab_hash    = heap_alloc_array_aligned(int32_t, vsize);
        if (tok->vocab_hash != nullptr) {
            tok->vocab_hash_mask = vsize - 1;
            for (size_t s = 0; s < vsize; s++)
                tok->vocab_hash[s] = -1;
            for (size_t id = 0; id < tok->vocab_size; id++) {
                uint64_t h = fnv1a64(tok->token_str[id], tok->token_len[id]);
                size_t   k = (size_t) h & tok->vocab_hash_mask;
                while (tok->vocab_hash[k] >= 0) {
                    k = (k + 1) & tok->vocab_hash_mask;
                }
                tok->vocab_hash[k] = (int32_t) id;
            }
        }
    }
    if (tok->n_merges > 0) {
        const size_t msize = next_pow2(tok->n_merges * 2);
        tok->merge_hash    = heap_alloc_array_aligned(size_t, msize);
        if (tok->merge_hash != nullptr) {
            tok->merge_hash_mask = msize - 1;
            for (size_t s = 0; s < msize; s++)
                tok->merge_hash[s] = SIZE_MAX;
            for (size_t m = 0; m < tok->n_merges; m++) {
                uint64_t h = fnv1a64_pair(tok->merge_left[m],
                                          tok->merge_left_len[m],
                                          tok->merge_right[m],
                                          tok->merge_right_len[m]);
                size_t   k = (size_t) h & tok->merge_hash_mask;
                while (tok->merge_hash[k] != SIZE_MAX) {
                    k = (k + 1) & tok->merge_hash_mask;
                }
                tok->merge_hash[k] = m;
            }
        }
    }

    /* Resolve the tokenization algorithm. "gpt2" is byte-level BPE; any
     * other model that ships merges (Gemma/Llama/Mistral SentencePiece,
     * model="gemma4"/"llama"/…) is driven through the same merge engine
     * with ▁ normalization + <0xXX> byte fallback. A model without merges
     * (pure unigram, or wordpiece) is unsupported here — load_copy then
     * refuses so the engine falls back to an external tokenizer.bin. */
    if (tok->model_len == 4 && memcmp(tok->model, "gpt2", 4) == 0) {
        tok->mode = GGUF_TOK_MODE_GPT2;
    } else if (tok->n_merges > 0) {
        tok->mode = GGUF_TOK_MODE_SPM;
    } else if (tok->scores != nullptr) {
        /* No merges but per-token scores -> SentencePiece unigram (Llama/BitNet). */
        tok->mode = GGUF_TOK_MODE_UNIGRAM;
    } else {
        tok->mode = GGUF_TOK_MODE_UNSUPPORTED;
    }

    if (tok->mode == GGUF_TOK_MODE_SPM || tok->mode == GGUF_TOK_MODE_UNIGRAM) {
        /* SentencePiece add_dummy_prefix — defaults to true when absent. */
        bool b;
        tok->add_space_prefix =
                gguf_get_meta_bool(ctx, "tokenizer.ggml.add_space_prefix", &b) ? b : true;
        /* Byte-fallback map: "<0xXX>" → vocab id (-1 if not present). */
        for (int i = 0; i < 256; i++) {
            char buf[8];
            int  n              = snprintf(buf, sizeof buf, "<0x%02X>", (unsigned) i);
            tok->spm_byte_id[i] = vocab_lookup(tok, buf, (size_t) n);
        }
    }
    return true;
}

void gguf_tokenizer_unload(struct gguf_tokenizer *tok) {
    if (tok == nullptr)
        return;
    if (tok->token_str != nullptr) {
        void *p = tok->token_str;
        safe_free(&p);
        tok->token_str = nullptr;
    }
    if (tok->token_len != nullptr) {
        void *p = tok->token_len;
        safe_free(&p);
        tok->token_len = nullptr;
    }
    if (tok->scores != nullptr) {
        void *p = tok->scores;
        safe_free(&p);
        tok->scores = nullptr;
    }
    if (tok->merge_left != nullptr) {
        void *p = tok->merge_left;
        safe_free(&p);
        tok->merge_left = nullptr;
    }
    if (tok->merge_left_len != nullptr) {
        void *p = tok->merge_left_len;
        safe_free(&p);
        tok->merge_left_len = nullptr;
    }
    if (tok->merge_right != nullptr) {
        void *p = tok->merge_right;
        safe_free(&p);
        tok->merge_right = nullptr;
    }
    if (tok->merge_right_len != nullptr) {
        void *p = tok->merge_right_len;
        safe_free(&p);
        tok->merge_right_len = nullptr;
    }
    if (tok->vocab_hash != nullptr) {
        void *p = tok->vocab_hash;
        safe_free(&p);
        tok->vocab_hash      = nullptr;
        tok->vocab_hash_mask = 0;
    }
    if (tok->merge_hash != nullptr) {
        void *p = tok->merge_hash;
        safe_free(&p);
        tok->merge_hash      = nullptr;
        tok->merge_hash_mask = 0;
    }
    if (tok->vocab_arena != nullptr) {
        void *p = tok->vocab_arena;
        safe_free(&p);
        tok->vocab_arena = nullptr;
    }
    if (tok->merge_arena != nullptr) {
        void *p = tok->merge_arena;
        safe_free(&p);
        tok->merge_arena = nullptr;
    }
    if (tok->model_arena != nullptr) {
        void *p = tok->model_arena;
        safe_free(&p);
        tok->model_arena = nullptr;
    }
    if (tok->specials != nullptr) {
        void *p = tok->specials;
        safe_free(&p);
        tok->specials = nullptr;
    }
    tok->vocab_size = 0;
    tok->n_merges   = 0;
    tok->n_specials = 0;
}

/* P1.6: copy-mode loader. Calls the mmap-pointing loader first, then
 * walks every vocab + merge string and memcpys the bytes into heap
 * arenas, rewriting the pointer slots to indirect through them. The
 * model name string is copied too. After this returns the caller can
 * close the GGUF — the tokenizer no longer depends on the mmap. */
[[nodiscard]] bool gguf_tokenizer_load_copy(struct gguf_tokenizer *tok,
                                            const struct gguf_ctx *ctx) {
    if (!gguf_tokenizer_load(tok, ctx))
        return false;
    /* gpt2 (byte-level BPE) and merge-driven SentencePiece (▁ BPE) are both
     * implemented for encode + decode. A model with neither (pure unigram /
     * wordpiece) is unsupported — refuse so the engine falls back to the
     * external sp_bpe (tokenizer.bin) path. */
    if (tok->mode == GGUF_TOK_MODE_UNSUPPORTED) {
        gguf_tokenizer_unload(tok);
        return false;
    }

    /* Vocab arena: sum of all token lengths. */
    size_t vbytes = 0;
    for (size_t i = 0; i < tok->vocab_size; i++)
        vbytes += tok->token_len[i];
    tok->vocab_arena = heap_alloc_array_aligned(char, vbytes ? vbytes : 1);
    if (tok->vocab_arena == nullptr) {
        gguf_tokenizer_unload(tok);
        return false;
    }
    size_t off = 0;
    for (size_t i = 0; i < tok->vocab_size; i++) {
        memcpy(tok->vocab_arena + off, tok->token_str[i], tok->token_len[i]);
        tok->token_str[i] = tok->vocab_arena + off;
        off += tok->token_len[i];
    }
    /* Specials alias into token_str[id]; re-point to the freshly-copied
     * vocab arena bytes so they survive the gguf_ctx close. */
    for (size_t s = 0; s < tok->n_specials; s++) {
        int32_t id = tok->specials[s].id;
        if (id >= 0 && (size_t) id < tok->vocab_size) {
            tok->specials[s].text = tok->token_str[id];
        }
    }

    /* Merges arena: sum of all left + right lengths. */
    if (tok->n_merges > 0) {
        size_t mbytes = 0;
        for (size_t m = 0; m < tok->n_merges; m++) {
            mbytes += tok->merge_left_len[m] + tok->merge_right_len[m];
        }
        tok->merge_arena = heap_alloc_array_aligned(char, mbytes ? mbytes : 1);
        if (tok->merge_arena == nullptr) {
            gguf_tokenizer_unload(tok);
            return false;
        }
        off = 0;
        for (size_t m = 0; m < tok->n_merges; m++) {
            memcpy(tok->merge_arena + off, tok->merge_left[m], tok->merge_left_len[m]);
            tok->merge_left[m] = tok->merge_arena + off;
            off += tok->merge_left_len[m];
            memcpy(tok->merge_arena + off, tok->merge_right[m], tok->merge_right_len[m]);
            tok->merge_right[m] = tok->merge_arena + off;
            off += tok->merge_right_len[m];
        }
    }

    /* Model name (e.g. "gpt2"). */
    tok->model_arena = heap_alloc_array_aligned(char, tok->model_len ? tok->model_len : 1);
    if (tok->model_arena == nullptr) {
        gguf_tokenizer_unload(tok);
        return false;
    }
    memcpy(tok->model_arena, tok->model, tok->model_len);
    tok->model = tok->model_arena;

    /* Hashes were built against the original (mmap) pointers; they
     * compare bytes via memcmp at probe time so they still match the
     * rewritten arena pointers — same bytes, different address.
     * No rebuild needed. */
    return true;
}

/* Append a single byte to `out` honoring the cap. Returns the
 * post-append running-total length (the value that would-fit-if-
 * the-cap-were-unlimited). */
static inline size_t append_byte(char *out, size_t cap, size_t used, char b) {
    if (out != nullptr && used < cap)
        out[used] = b;
    return used + 1;
}

/* Parse a SentencePiece byte-fallback token "<0xXX>" → 0..255, or -1. */
static int spm_byte_token_value(const char *s, size_t slen) {
    if (slen != 6 || s[0] != '<' || s[1] != '0' || s[2] != 'x' || s[5] != '>')
        return -1;
    int hi = 0, lo = 0;
    for (int k = 3; k <= 4; k++) {
        const char c = s[k];
        int        v;
        if (c >= '0' && c <= '9')
            v = c - '0';
        else if (c >= 'A' && c <= 'F')
            v = c - 'A' + 10;
        else if (c >= 'a' && c <= 'f')
            v = c - 'a' + 10;
        else
            return -1;
        if (k == 3)
            hi = v;
        else
            lo = v;
    }
    return (hi << 4) | lo;
}

size_t gguf_tokenizer_decode(
        const struct gguf_tokenizer *tok, const int32_t *ids, size_t n, char *out, size_t out_cap) {
    if (tok == nullptr || ids == nullptr)
        return 0;
    const bool gpt2  = tok->mode == GGUF_TOK_MODE_GPT2;
    size_t     total = 0;
    for (size_t i = 0; i < n; i++) {
        int32_t     id = ids[i];
        const char *s;
        size_t      slen;
        if (id < 0 || (size_t) id >= tok->vocab_size) {
            static const char placeholder[] = "<unk>";
            s                               = placeholder;
            slen                            = sizeof placeholder - 1;
        } else {
            s    = tok->token_str[id];
            slen = tok->token_len[id];
        }
        if (tok->mode == GGUF_TOK_MODE_SPM || tok->mode == GGUF_TOK_MODE_UNIGRAM) {
            /* SentencePiece (BPE or unigram): <0xXX> → raw byte; ▁ (U+2581) →
             * space; the rest verbatim. (add_space_prefix's leading ▁, if any,
             * decodes to a leading space — symmetric with encode.) */
            const int bval = spm_byte_token_value(s, slen);
            if (bval >= 0) {
                total = append_byte(out, out_cap, total, (char) bval);
                continue;
            }
            for (size_t k = 0; k < slen;) {
                if (k + SPM_MARKER_LEN <= slen && memcmp(s + k, SPM_MARKER, SPM_MARKER_LEN) == 0) {
                    total = append_byte(out, out_cap, total, ' ');
                    k += SPM_MARKER_LEN;
                } else {
                    total = append_byte(out, out_cap, total, s[k]);
                    k++;
                }
            }
            continue;
        }
        if (!gpt2) {
            /* Unsupported model: emit raw bytes verbatim. */
            for (size_t k = 0; k < slen; k++) {
                total = append_byte(out, out_cap, total, s[k]);
            }
            continue;
        }
        /* GPT-2: walk UTF-8 codepoints, map each back to a single byte. */
        size_t k = 0;
        while (k < slen) {
            size_t  adv;
            int32_t cp = utf8_decode_one(s + k, slen - k, &adv);
            if (cp < 0) {
                k += adv ? adv : 1;
                continue;
            }
            int b = gpt2_codepoint_to_byte((uint32_t) cp);
            if (b < 0) {
                /* Codepoint outside the map (rare; user-defined unicode
                 * in a token name). Emit the codepoint's UTF-8 bytes
                 * verbatim so the user sees something readable. */
                for (size_t j = 0; j < adv; j++) {
                    total = append_byte(out, out_cap, total, s[k + j]);
                }
            } else {
                total = append_byte(out, out_cap, total, (char) b);
            }
            k += adv;
        }
    }
    /* NUL-terminate when there's room. */
    if (out != nullptr && total < out_cap)
        out[total] = '\0';
    else if (out != nullptr && out_cap > 0)
        out[out_cap - 1] = '\0';
    return total;
}

/* ============================================================== */
/* Encoder (P1.5.g) + hash indices (P1.5.h)                        */
/* ============================================================== */

/* FNV-1a 64-bit. Stable across builds; we just need a good
 * distribution over short byte strings. */
static uint64_t fnv1a64(const char *p, size_t n) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= (uint8_t) p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

/* Hash a (left, right) pair. The separator byte ensures
 * ("ab", "") and ("a", "b") hash distinctly. */
static uint64_t fnv1a64_pair(const char *a, size_t alen, const char *b, size_t blen) {
    uint64_t h = fnv1a64(a, alen);
    h ^= 0xFFu;
    h *= 0x100000001b3ULL;
    for (size_t i = 0; i < blen; i++) {
        h ^= (uint8_t) b[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

static size_t next_pow2(size_t n) {
    size_t p = 1;
    while (p < n)
        p <<= 1;
    return p;
}

/* O(1) vocab lookup. Returns -1 when no entry matches. */
static int32_t vocab_lookup(const struct gguf_tokenizer *tok, const char *bytes, size_t len) {
    if (tok->vocab_hash == nullptr || tok->vocab_hash_mask == 0) {
        /* Fallback to linear scan if the hash table wasn't built. */
        for (size_t i = 0; i < tok->vocab_size; i++) {
            if (tok->token_len[i] == len && memcmp(tok->token_str[i], bytes, len) == 0) {
                return (int32_t) i;
            }
        }
        return -1;
    }
    uint64_t h = fnv1a64(bytes, len);
    size_t   i = (size_t) h & tok->vocab_hash_mask;
    while (true) {
        const int32_t id = tok->vocab_hash[i];
        if (id < 0)
            return -1;
        if (tok->token_len[id] == len && memcmp(tok->token_str[id], bytes, len) == 0) {
            return id;
        }
        i = (i + 1) & tok->vocab_hash_mask;
    }
}

int32_t gguf_tokenizer_id_for_text(const struct gguf_tokenizer *tok, const char *text) {
    if (tok == nullptr || text == nullptr) {
        return -1;
    }
    return vocab_lookup(tok, text, strlen(text));
}

/* O(1) merge-rank lookup. Returns SIZE_MAX when no merge has
 * (left, right) as its operands. */
static size_t merge_lookup(const struct gguf_tokenizer *tok,
                           const char                  *lstr,
                           size_t                       llen,
                           const char                  *rstr,
                           size_t                       rlen) {
    if (tok->merge_hash == nullptr || tok->merge_hash_mask == 0) {
        for (size_t m = 0; m < tok->n_merges; m++) {
            if (tok->merge_left_len[m] == llen && tok->merge_right_len[m] == rlen &&
                memcmp(tok->merge_left[m], lstr, llen) == 0 &&
                memcmp(tok->merge_right[m], rstr, rlen) == 0) {
                return m;
            }
        }
        return SIZE_MAX;
    }
    uint64_t h = fnv1a64_pair(lstr, llen, rstr, rlen);
    size_t   i = (size_t) h & tok->merge_hash_mask;
    while (true) {
        const size_t m = tok->merge_hash[i];
        if (m == SIZE_MAX)
            return SIZE_MAX;
        if (tok->merge_left_len[m] == llen && tok->merge_right_len[m] == rlen &&
            memcmp(tok->merge_left[m], lstr, llen) == 0 &&
            memcmp(tok->merge_right[m], rstr, rlen) == 0) {
            return m;
        }
        i = (i + 1) & tok->merge_hash_mask;
    }
}

/* One symbol in the BPE merge loop: a slice of the chunk's UTF-8
 * byte buffer plus a doubly-linked-list pointer pair (sym indices). */
struct bpe_sym {
    size_t off;
    size_t len;
    int    prev; /* -1 = head */
    int    next; /* -1 = tail */
};

/* Find the lowest-rank merge in the symbol list. Returns the index
 * of the LEFT symbol of the winning pair, or -1 when no adjacent
 * pair has a merge entry. P1.5.h: uses the merge hash index — O(1)
 * per pair instead of O(N_merges). */
static int find_best_merge(const struct gguf_tokenizer *tok,
                           const char                  *buf,
                           const struct bpe_sym        *syms,
                           int                          head) {
    int    best_left = -1;
    size_t best_rank = SIZE_MAX;
    int    i         = head;
    while (i >= 0 && syms[i].next >= 0) {
        const int    j = syms[i].next;
        const size_t m =
                merge_lookup(tok, buf + syms[i].off, syms[i].len, buf + syms[j].off, syms[j].len);
        if (m < best_rank) {
            best_rank = m;
            best_left = i;
        }
        i = j;
    }
    return best_left;
}

/* Apply one BPE step on a pre-tokenized chunk's byte buffer, emitting
 * vocab IDs into out (returns count). Internal helper for encode.
 * When byte_fallback is set (SPM), a symbol missing from the vocab is
 * emitted as one "<0xXX>" token per byte (spm_byte_id), then unk_id;
 * otherwise (gpt2) a missing symbol emits a single unk_id. */
static size_t bpe_chunk_to_ids(const struct gguf_tokenizer *tok,
                               const char                  *buf,
                               size_t                       buf_len,
                               int32_t                     *out,
                               size_t                       cap,
                               bool                         byte_fallback) {
    if (buf_len == 0 || cap == 0)
        return 0;

    /* Decompose into one symbol per UTF-8 codepoint. Cap symbols at
     * buf_len since each codepoint takes >= 1 byte. */
    struct bpe_sym *syms = heap_alloc_array_aligned(struct bpe_sym, buf_len);
    if (syms == nullptr)
        return 0;
    int    n_syms = 0;
    size_t k      = 0;
    while (k < buf_len) {
        size_t adv;
        (void) utf8_decode_one(buf + k, buf_len - k, &adv);
        if (adv == 0)
            adv = 1;
        if (k + adv > buf_len)
            adv = buf_len - k;
        syms[n_syms].off  = k;
        syms[n_syms].len  = adv;
        syms[n_syms].prev = n_syms - 1;
        syms[n_syms].next = n_syms + 1;
        n_syms++;
        k += adv;
    }
    if (n_syms == 0) {
        void *p = syms;
        safe_free(&p);
        return 0;
    }
    syms[n_syms - 1].next = -1;

    /* Greedy merge loop. Each iteration finds the global lowest-rank
     * adjacent pair and merges. Stops when no merge exists. */
    int head = 0;
    while (true) {
        int li = find_best_merge(tok, buf, syms, head);
        if (li < 0)
            break;
        int ri = syms[li].next;
        syms[li].len += syms[ri].len;
        syms[li].next = syms[ri].next;
        if (syms[ri].next >= 0)
            syms[syms[ri].next].prev = li;
    }

    /* Walk the surviving symbols and look up each in the vocab. */
    size_t n_out = 0;
    int    i     = head;
    while (i >= 0 && n_out < cap) {
        int32_t id = vocab_lookup(tok, buf + syms[i].off, syms[i].len);
        if (id >= 0) {
            out[n_out++] = id;
        } else if (byte_fallback) {
            /* SPM: emit one <0xXX> token per raw byte, falling back to unk. */
            for (size_t b = 0; b < syms[i].len && n_out < cap; b++) {
                int32_t bid = tok->spm_byte_id[(uint8_t) buf[syms[i].off + b]];
                if (bid < 0)
                    bid = tok->unk_id;
                if (bid >= 0)
                    out[n_out++] = bid;
            }
        } else if (tok->unk_id >= 0) {
            out[n_out++] = tok->unk_id;
        }
        i = syms[i].next;
    }
    void *p = syms;
    safe_free(&p);
    return n_out;
}

/* Simplified GPT-2 pre-tokenizer. Walks the input UTF-8 byte stream
 * and emits one chunk per "word", where a word is:
 *   - A run of whitespace (kept attached to the NEXT non-ws char's
 *     leading position — i.e. " hello" becomes "Ġhello" after byte
 *     mapping).
 *   - A run of letters (ASCII A-Z, a-z + any byte >= 0x80, i.e.
 *     UTF-8 continuation bytes treat all non-ASCII as letter-like).
 *   - A run of digits.
 *   - A run of other punctuation / symbols.
 *
 * This isn't the full regex llama.cpp uses, but matches well for
 * plain English text. Apostrophe contractions and Unicode categories
 * differ. Each chunk gets passed through the byte→codepoint forward
 * map + utf8-encoded into a chunk_buf, then handed to bpe_chunk_to_ids. */
static bool is_ascii_letter(unsigned char b) {
    return (b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z') || b >= 0x80;
}
static bool is_ascii_digit(unsigned char b) {
    return b >= '0' && b <= '9';
}
static bool is_ascii_space(unsigned char b) {
    return b == ' ' || b == '\t' || b == '\n' || b == '\r';
}

/* SentencePiece *unigram* encode of one normalized chunk. Same greedy pairwise
 * merge as SPM-BPE, but with no merges table: a pair may merge iff its
 * concatenation is a vocab token, and the merge PRIORITY is that token's score
 * (highest score merges first). This is llama.cpp's SPM algorithm — the scores
 * here are rank-like (score ≈ -(id), so a lower-id/"more common" token wins),
 * which is why a score-MAXIMIZING lattice would wrongly collapse to the
 * score-0 byte tokens; the pairwise merge is the correct reading. Unmatched
 * symbols fall back to <0xXX> byte tokens, then unk. Returns ids written. */
static size_t unigram_chunk_to_ids(const struct gguf_tokenizer *tok,
                                   const char                  *buf,
                                   size_t                       buf_len,
                                   int32_t                     *out,
                                   size_t                       cap) {
    if (buf_len == 0 || cap == 0)
        return 0;
    struct bpe_sym *syms = heap_alloc_array_aligned(struct bpe_sym, buf_len);
    if (syms == nullptr)
        return 0;
    int    n_syms = 0;
    size_t k      = 0;
    while (k < buf_len) { /* one symbol per UTF-8 codepoint */
        size_t adv;
        (void) utf8_decode_one(buf + k, buf_len - k, &adv);
        if (adv == 0)
            adv = 1;
        if (k + adv > buf_len)
            adv = buf_len - k;
        syms[n_syms].off  = k;
        syms[n_syms].len  = adv;
        syms[n_syms].prev = n_syms - 1;
        syms[n_syms].next = n_syms + 1;
        n_syms++;
        k += adv;
    }
    if (n_syms == 0) {
        void *p = syms;
        safe_free(&p);
        return 0;
    }
    syms[n_syms - 1].next = -1;

    /* Greedy merge: each step joins the adjacent pair whose concatenation is a
     * vocab token with the highest score. Stops when no pair forms a token. */
    int head = 0;
    while (true) {
        int   best_left  = -1;
        float best_score = 0.0f;
        for (int i = head; i >= 0 && syms[i].next >= 0; i = syms[i].next) {
            const int     j  = syms[i].next;
            const int32_t id = vocab_lookup(tok, buf + syms[i].off, syms[i].len + syms[j].len);
            if (id < 0)
                continue;
            const float sc = tok->scores[id];
            if (best_left < 0 || sc > best_score) {
                best_score = sc;
                best_left  = i;
            }
        }
        if (best_left < 0)
            break;
        const int ri = syms[best_left].next;
        syms[best_left].len += syms[ri].len;
        syms[best_left].next = syms[ri].next;
        if (syms[ri].next >= 0)
            syms[syms[ri].next].prev = best_left;
    }

    size_t n_out = 0;
    for (int i = head; i >= 0 && n_out < cap; i = syms[i].next) {
        int32_t id = vocab_lookup(tok, buf + syms[i].off, syms[i].len);
        if (id >= 0) {
            out[n_out++] = id;
        } else { /* SPM byte fallback: one <0xXX> per raw byte, else unk */
            for (size_t b = 0; b < syms[i].len && n_out < cap; b++) {
                int32_t bid = tok->spm_byte_id[(uint8_t) buf[syms[i].off + b]];
                if (bid < 0)
                    bid = tok->unk_id;
                if (bid >= 0)
                    out[n_out++] = bid;
            }
        }
    }
    void *p = syms;
    safe_free(&p);
    return n_out;
}

/* Encode one normalized chunk with the active SentencePiece algorithm. */
static size_t spm_chunk_to_ids(
        const struct gguf_tokenizer *tok, const char *buf, size_t len, int32_t *out, size_t cap) {
    if (tok->mode == GGUF_TOK_MODE_UNIGRAM)
        return unigram_chunk_to_ids(tok, buf, len, out, cap);
    return bpe_chunk_to_ids(tok, buf, len, out, cap, true);
}

/* SentencePiece encode: there is no GPT-2 pre-tokenizer split — the algorithm
 * runs over each whole inter-special chunk after ▁ normalization (space → ▁,
 * plus an optional leading ▁ when add_space_prefix). Unknown symbols fall
 * back to <0xXX> byte tokens. Covers both the merge-driven (SPM) and unigram
 * paths via spm_chunk_to_ids. */
static bool encode_spm(const struct gguf_tokenizer *tok,
                       const char                  *text,
                       int32_t                     *out_ids,
                       size_t                       cap,
                       size_t                      *n_out) {
    const size_t tlen = strlen(text);
    if (tlen == 0)
        return true;

    /* Scratch for one normalized chunk: worst case every byte is a space
     * (→ ▁, 3 bytes) plus an optional leading ▁. Reused across chunks. */
    char *buf = heap_alloc_array_aligned(char, tlen *SPM_MARKER_LEN + SPM_MARKER_LEN);
    if (buf == nullptr)
        return false;

    size_t i = 0, chunk_start = 0;
    bool   first_chunk = true;

/* Normalize text[chunk_start, end) into buf and BPE-encode it. */
#define SPM_FLUSH(end)                                                                \
    do {                                                                              \
        size_t w_ = 0;                                                                \
        if (first_chunk && tok->add_space_prefix) {                                   \
            memcpy(buf + w_, SPM_MARKER, SPM_MARKER_LEN);                             \
            w_ += SPM_MARKER_LEN;                                                     \
        }                                                                             \
        for (size_t r_ = chunk_start; r_ < (end); r_++) {                             \
            if (text[r_] == ' ') {                                                    \
                memcpy(buf + w_, SPM_MARKER, SPM_MARKER_LEN);                         \
                w_ += SPM_MARKER_LEN;                                                 \
            } else {                                                                  \
                buf[w_++] = text[r_];                                                 \
            }                                                                         \
        }                                                                             \
        first_chunk = false;                                                          \
        if (w_ > 0 && *n_out < cap) {                                                 \
            *n_out += spm_chunk_to_ids(tok, buf, w_, out_ids + *n_out, cap - *n_out); \
        }                                                                             \
    } while (0)

    while (i < tlen) {
        bool matched = false;
        for (size_t s = 0; s < tok->n_specials && *n_out < cap; s++) {
            const size_t slen = tok->specials[s].len;
            if (slen == 0 || i + slen > tlen)
                continue;
            if (memcmp(text + i, tok->specials[s].text, slen) == 0) {
                if (chunk_start < i)
                    SPM_FLUSH(i);
                /* SPM_FLUSH may have filled the buffer to cap; guard the
                 * special-token write so it truncates instead of overflowing. */
                if (*n_out < cap)
                    out_ids[(*n_out)++] = tok->specials[s].id;
                i += slen;
                chunk_start = i;
                matched     = true;
                break;
            }
        }
        if (!matched)
            i++;
    }
    if (chunk_start < tlen)
        SPM_FLUSH(tlen);
#undef SPM_FLUSH

    void *p = buf;
    safe_free(&p);
    return true;
}

[[nodiscard]] bool gguf_tokenizer_encode(const struct gguf_tokenizer *tok,
                                         const char                  *text,
                                         int32_t                     *out_ids,
                                         size_t                       cap,
                                         size_t                      *n_out) {
    if (tok == nullptr || text == nullptr || out_ids == nullptr || n_out == nullptr)
        return false;
    *n_out = 0;
    if (tok->mode == GGUF_TOK_MODE_SPM || tok->mode == GGUF_TOK_MODE_UNIGRAM)
        return encode_spm(tok, text, out_ids, cap, n_out);
    if (tok->mode != GGUF_TOK_MODE_GPT2)
        return false;

    const size_t tlen = strlen(text);
    if (tlen == 0)
        return true;

    /* Worst-case chunk size = entire input byte run × 4 (codepoint
     * 256+ encodes as 2 UTF-8 bytes, so up to 2× expansion). */
    char *chunk_buf = heap_alloc_array_aligned(char, tlen * 4 + 16);
    if (chunk_buf == nullptr)
        return false;

    size_t i = 0;
    while (i < tlen) {
        /* Leftmost-longest special-token match at current position. The
         * specials array is sorted by len desc, so the first hit IS the
         * longest possible match at this offset. */
        bool matched_special = false;
        for (size_t s = 0; s < tok->n_specials && *n_out < cap; s++) {
            size_t slen = tok->specials[s].len;
            if (slen == 0 || i + slen > tlen)
                continue;
            if (memcmp(text + i, tok->specials[s].text, slen) == 0) {
                out_ids[(*n_out)++] = tok->specials[s].id;
                i += slen;
                matched_special = true;
                break;
            }
        }
        if (matched_special)
            continue;

        size_t start = i;

        /* Detect chunk kind from first non-space byte. */
        while (i < tlen && is_ascii_space((unsigned char) text[i]))
            i++;
        /* Now i is the first non-space (or tlen). The leading-space
         * run (start..i) is appended as Ġ-prefix on the following
         * word, GPT-2 style. */
        const size_t lead_ws_start = start, lead_ws_end = i;
        if (i >= tlen) {
            /* Trailing whitespace becomes its own chunk. */
            if (lead_ws_end > lead_ws_start) {
                size_t cb_used = 0;
                for (size_t b = lead_ws_start; b < lead_ws_end; b++) {
                    uint32_t cp = gpt2_byte_to_codepoint((unsigned char) text[b]);
                    cb_used += utf8_encode_one(cp, chunk_buf + cb_used);
                }
                size_t produced = bpe_chunk_to_ids(
                        tok, chunk_buf, cb_used, out_ids + *n_out, cap - *n_out, false);
                *n_out += produced;
            }
            break;
        }

        /* Classify the run by first byte. */
        unsigned char c         = (unsigned char) text[i];
        size_t        run_start = i;
        if (is_ascii_letter(c)) {
            while (i < tlen && is_ascii_letter((unsigned char) text[i]))
                i++;
        } else if (is_ascii_digit(c)) {
            while (i < tlen && is_ascii_digit((unsigned char) text[i]))
                i++;
        } else {
            /* Punctuation / symbol — match one byte at a time, but
             * group multiple of the same byte. Simpler: single byte. */
            i++;
        }

        /* Emit chunk: leading whitespace + run (all bytes get
         * byte→codepoint→UTF-8 mapping). */
        size_t cb_used = 0;
        for (size_t b = lead_ws_start; b < i; b++) {
            uint32_t cp = gpt2_byte_to_codepoint((unsigned char) text[b]);
            cb_used += utf8_encode_one(cp, chunk_buf + cb_used);
        }
        size_t produced =
                bpe_chunk_to_ids(tok, chunk_buf, cb_used, out_ids + *n_out, cap - *n_out, false);
        *n_out += produced;
        if (*n_out >= cap)
            break;
        (void) run_start;
    }

    void *p = chunk_buf;
    safe_free(&p);
    return true;
}

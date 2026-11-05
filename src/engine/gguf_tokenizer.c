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
#include "pair_merge.h"
#include "unicode_class.h"

#include "checked.h"
#include "fnv1a.h"
#include "heap.h"
#include "cleanup.h"

#include "file_map.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

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
 * to a codepoint in [0, 323]. */
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

/* The sequence length a UTF-8 lead byte announces (1..4), 1 for a byte that
 * cannot lead one. Continuation bytes are not looked at (tokenizer_bin). */
static size_t utf8_lead_len(uint8_t b) {
    if (b < 0x80)
        return 1;
    if ((b & 0xE0) == 0xC0)
        return 2;
    if ((b & 0xF0) == 0xE0)
        return 3;
    if ((b & 0xF8) == 0xF0)
        return 4;
    return 1;
}

/* Hash helpers, defined further down. */
static uint64_t fnv1a64(const char *p, size_t n);
static uint64_t fnv1a64_pair(const char *a, size_t alen, const char *b, size_t blen);
static size_t   next_pow2(size_t n);
static int32_t  vocab_lookup(const struct gguf_tokenizer *tok, const char *bytes, size_t len);

/* SentencePiece space marker U+2581 (▁) in UTF-8. */
static const char SPM_MARKER[3]  = {(char) 0xE2, (char) 0x96, (char) 0x81};
constexpr int     SPM_MARKER_LEN = 3;

/* Sort the specials by length, longest first and stable among equals, so the
 * first match in the encoders' linear scan is the leftmost-longest one. */
static void sort_specials(struct gguf_tokenizer *tok) {
    for (size_t a = 1; a < tok->n_specials; a++) {
        struct gguf_tokenizer_special tmp = tok->specials[a];
        size_t                        b   = a;
        while (b > 0 && tok->specials[b - 1].len < tmp.len) {
            tok->specials[b] = tok->specials[b - 1];
            b--;
        }
        tok->specials[b] = tmp;
    }
}

/* special_first from both runs of the specials. */
static void index_special_first(struct gguf_tokenizer *tok) {
    memset(tok->special_first, 0, sizeof tok->special_first);
    for (size_t s = 0; s < tok->n_specials + tok->n_user_specials; s++)
        if (tok->specials[s].len > 0)
            tok->special_first[(unsigned char) tok->specials[s].text[0]] |=
                    s < tok->n_specials ? 1 : 2;
}

/* Control-looking strings llama.cpp marks CONTROL whatever their token_type
 * (llama_vocab::impl::load, "control-looking token ... will be overridden"):
 * its end-of-generation list (which holds the EOT / EOM lists) always, and
 * a FIM role's names when the file has no id for that role. Qwen3 has "</s>"
 * as a NORMAL token, Gemma 4 "<eos>" (#404). */
#define FW "\xef\xbd\x9c" /* U+FF5C, DeepSeek's fullwidth bar */
#define LB "\xe2\x96\x81" /* U+2581 */
static const char *const FIM_KEYS[][2] = {
        {nullptr, nullptr},
        {"tokenizer.ggml.fim_pre_token_id", "tokenizer.ggml.prefix_token_id"},
        {"tokenizer.ggml.fim_suf_token_id", "tokenizer.ggml.suffix_token_id"},
        {"tokenizer.ggml.fim_mid_token_id", "tokenizer.ggml.middle_token_id"},
        {"tokenizer.ggml.fim_pad_token_id", nullptr},
        {"tokenizer.ggml.fim_rep_token_id", nullptr},
        {"tokenizer.ggml.fim_sep_token_id", nullptr},
};
static const struct {
    const char *text;
    uint8_t     fim; /* FIM_KEYS row; 0 = always */
} CONTROL_LOOKING[] = {
        {"<|eot_id|>", 0},
        {"<|im_end|>", 0},
        {"<|end|>", 0},
        {"<|return|>", 0},
        {"<|call|>", 0},
        {"<|flush|>", 0},
        {"<|calls|>", 0},
        {"<end_of_turn>", 0},
        {"<|endoftext|>", 0},
        {"</s>", 0},
        {"<|eom_id|>", 0},
        {"<EOT>", 0},
        {"_<EOT>", 0},
        {"[EOT]", 0},
        {"[EOS]", 0},
        {"<|end_of_text|>", 0},
        {"<end_of_utterance>", 0},
        {"<eos>", 0},
        {"<turn|>", 0},
        {"<|tool_response>", 0},
        {"<" FW "end" LB "of" LB "sentence" FW ">", 0},
        {"[e~[", 0},
        {"<|fim_prefix|>", 1},
        {"<fim-prefix>", 1},
        {"<fim_prefix>", 1},
        {"<" FW "fim" LB "begin" FW ">", 1},
        {"<PRE>", 1},
        {LB "<PRE>", 1},
        {"<|code_prefix|>", 1},
        {"<|prefix|>", 1},
        {"<|fim_suffix|>", 2},
        {"<fim-suffix>", 2},
        {"<fim_suffix>", 2},
        {"<" FW "fim" LB "hole" FW ">", 2},
        {"<SUF>", 2},
        {LB "<SUF>", 2},
        {"<|code_suffix|>", 2},
        {"<|suffix|>", 2},
        {"<|fim_middle|>", 3},
        {"<fim-middle>", 3},
        {"<fim_middle>", 3},
        {"<" FW "fim" LB "end" FW ">", 3},
        {"<MID>", 3},
        {LB "<MID>", 3},
        {"<|code_middle|>", 3},
        {"<|middle|>", 3},
        {"<|fim_pad|>", 4},
        {"<fim-pad>", 4},
        {"<fim_pad>", 4},
        {"<PAD>", 4},
        {"[PAD]", 4},
        {"<|fim_repo|>", 5},
        {"<|repo_name|>", 5},
        {"<fim-repo>", 5},
        {"<REPO>", 5},
        {"<reponame>", 5},
        {"<|file_sep|>", 6},
};
#undef FW
#undef LB

/* token_type tt[i] (null: absent, NORMAL). */
static int32_t token_type_at(const uint8_t *tt, size_t i) {
    int32_t t = 1;
    if (tt != nullptr)
        memcpy(&t, tt + i * 4, 4);
    return t;
}

/* Whether token_type tt (null: absent) makes vocab entry i a special token:
 * UNKNOWN(2), CONTROL(3) or USER_DEFINED(4). */
static bool special_type(const uint8_t *tt, size_t i) {
    const int32_t t = token_type_at(tt, i);
    return t == 2 || t == 3 || t == 4;
}

/* The special tokens: the vocab entries llama.cpp's tokenizer matches in the
 * text before pretokenizing (cache_special_tokens, parse_special on). Those
 * are the special_type ones and the CONTROL_LOOKING entries, but not "</s>" in a vocab that has
 * "<|tool_response>" (llama.cpp's Gemma 4 workaround makes it NORMAL). Sorted
 * longest first, so leftmost-longest matching in encode is a linear scan;
 * without them Llama 3's <|begin_of_text|> is shredded into
 * `< | begin _ of _ text | >`. Needs the vocab index. Leaves no specials when
 * the list cannot be allocated, as before #404.
 *
 * The `user` ones (#409) are the entries llama.cpp still matches with
 * parse_special=false: attr USER_DEFINED without CONTROL or UNKNOWN, which
 * is token_type 4 unless the string is control-looking (the promotion ORs
 * CONTROL in). Not mirrored: llama.cpp's gpt-oss hack, which makes
 * "<|start|>", "<|channel|>", "<|message|>", "<|constrain|>" and (with
 * "<|return|>" and "<|call|>") "<|end|>" USER_DEFINED, so that they are
 * matched in plain text too; here they stay role markers that plain text
 * cannot spell. */
static void load_specials(struct gguf_tokenizer *tok, const struct gguf_ctx *ctx) {
    uint32_t       tt_vt = 0, u;
    uint64_t       tt_n  = 0;
    const uint8_t *tt    = nullptr;
    if (!gguf_get_meta_array_info(ctx, "tokenizer.ggml.token_type", &tt_vt, &tt_n, &tt) ||
        (tt_vt != GGUF_META_VT_U32 && tt_vt != GGUF_META_VT_I32) || tt_n != tok->vocab_size)
        tt = nullptr;
    const int32_t s_close =
            vocab_lookup(tok, "<|tool_response>", 16) >= 0 ? vocab_lookup(tok, "</s>", 4) : -1;
    constexpr size_t N_LOOKING = sizeof CONTROL_LOOKING / sizeof CONTROL_LOOKING[0];
    int32_t          promoted[N_LOOKING]; /* every control-looking entry */
    int32_t          looking[N_LOOKING];  /* the ones not special by type */
    size_t           n_promoted = 0, n_looking = 0, n_sp = 0, n_user = 0;
    for (size_t c = 0; c < N_LOOKING; c++) {
        const char *const *keys   = FIM_KEYS[CONTROL_LOOKING[c].fim];
        bool               has_id = false;
        for (size_t k = 0; k < 2; k++)
            has_id = has_id || (keys[k] != nullptr && gguf_get_meta_u32(ctx, keys[k], &u) &&
                                u < tok->vocab_size);
        const char   *text = CONTROL_LOOKING[c].text;
        const int32_t id   = has_id ? -1 : vocab_lookup(tok, text, strlen(text));
        if (id < 0 || id == s_close)
            continue;
        promoted[n_promoted++] = id;
        if (!special_type(tt, (size_t) id))
            looking[n_looking++] = id;
    }
    for (size_t i = 0; tt != nullptr && i < tok->vocab_size; i++) {
        if (!special_type(tt, i) || (int32_t) i == s_close)
            continue;
        n_sp++;
        bool user = token_type_at(tt, i) == 4;
        for (size_t c = 0; user && c < n_promoted; c++)
            user = promoted[c] != (int32_t) i;
        n_user += user;
    }
    if (n_sp + n_looking == 0)
        return;
    tok->specials =
            heap_alloc_array_aligned(struct gguf_tokenizer_special, n_sp + n_looking + n_user);
    if (tok->specials == nullptr)
        return;
    for (size_t i = 0; i < tok->vocab_size + n_looking; i++) {
        const size_t id = i < tok->vocab_size ? i : (size_t) looking[i - tok->vocab_size];
        if (i < tok->vocab_size && (!special_type(tt, i) || (int32_t) i == s_close))
            continue;
        bool user = i < tok->vocab_size && token_type_at(tt, i) == 4;
        for (size_t c = 0; user && c < n_promoted; c++)
            user = promoted[c] != (int32_t) id;
        tok->specials[tok->n_specials++] =
                (struct gguf_tokenizer_special) {.text = tok->token_str[id],
                                                 .len  = tok->token_len[id],
                                                 .id   = (int32_t) id,
                                                 .user = user};
    }
    sort_specials(tok);
    /* The user run, in the same longest-first order. */
    for (size_t s = 0; s < tok->n_specials; s++)
        if (tok->specials[s].user)
            tok->specials[tok->n_specials + tok->n_user_specials++] = tok->specials[s];
    index_special_first(tok);
}

/* Build the vocab + merges hash indices over token_str / merge_*. Both
 * tables are pow-of-2 sized with load factor ~0.5; a lookup that collides
 * with an earlier equal entry finds the earlier one (the lower id or rank).
 * False when a table cannot be allocated. */
static bool build_indices(struct gguf_tokenizer *tok) {
    const size_t vsize = next_pow2(tok->vocab_size * 2);
    const size_t msize = tok->n_merges > 0 ? next_pow2(tok->n_merges * 2) : 0;
    tok->vocab_hash    = heap_alloc_array_aligned(int32_t, vsize);
    tok->merge_hash    = msize > 0 ? heap_alloc_array_aligned(size_t, msize) : nullptr;
    if (tok->vocab_hash == nullptr || (msize > 0 && tok->merge_hash == nullptr)) {
        return false;
    }
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
    if (msize > 0) {
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
    return true;
}

/* SentencePiece byte-fallback map: "<0xXX>" → vocab id (-1 if not present). */
static void build_spm_byte_map(struct gguf_tokenizer *tok) {
    for (int i = 0; i < 256; i++) {
        char buf[8];
        int  n              = snprintf(buf, sizeof buf, "<0x%02X>", (unsigned) i);
        tok->spm_byte_id[i] = vocab_lookup(tok, buf, (size_t) n);
    }
}

/* s[0, n) is one of the NUL-terminated names in list (null-terminated). */
static bool name_in(size_t n, const char *s, const char *const *list) {
    if (s == nullptr)
        return false;
    for (; *list != nullptr; list++) {
        if (strlen(*list) == n && memcmp(s, *list, n) == 0)
            return true;
    }
    return false;
}

[[nodiscard]] bool gguf_tokenizer_add_bos_rule(size_t      model_len,
                                               size_t      pre_len,
                                               int         flag,
                                               bool        fallback,
                                               const char *model,
                                               const char *pre) {
    /* llama_vocab::impl::load: the vocabulary type comes from the model name. */
    static const char *const SPM[]  = {"llama", nullptr};
    static const char *const WPM[]  = {"bert", nullptr};
    static const char *const BPE[]  = {"gpt2", "hybriddna", "whitespace", "gemma4", nullptr};
    static const char *const NONE[] = {"t5", "rwkv", "plamo2", "test", "no_vocab", "none", nullptr};
    /* The BPE pretokenizers whose branch sets add_bos = true. */
    static const char *const BPE_BOS[] = {"llama3",
                                          "llama-v3",
                                          "llama-bpe",
                                          "falcon3",
                                          "falcon-h1",
                                          "pixtral",
                                          "midm-2.0",
                                          "lfm2",
                                          "jina-v5-nano",
                                          "tekken",
                                          "chameleon",
                                          nullptr};
    /* LLAMA_VOCAB_PRE_TYPE_GEMMA4; model "gemma4" sets this pre itself. */
    static const char *const GEMMA4[]       = {"gemma4", "granite-embed-multi-311m", nullptr};
    static const char *const GEMMA4_MODEL[] = {"gemma4", nullptr};

    const bool is_bpe = name_in(model_len, model, BPE);
    bool       add_bos;
    if (name_in(model_len, model, SPM) || name_in(model_len, model, WPM)) {
        add_bos = true;
    } else if (is_bpe) {
        add_bos = name_in(pre_len, pre, BPE_BOS);
    } else if (name_in(model_len, model, NONE)) {
        add_bos = false; /* UGM, RWKV and the rest: the member default */
    } else {
        add_bos = fallback; /* llama.cpp refuses to load this vocabulary */
    }
    if (flag >= 0) /* tokenizer.ggml.add_bos_token overrides the default */
        add_bos = flag != 0;
    /* ... except for Gemma 4, which llama.cpp forces on (its PR 21500). */
    const bool gemma4 =
            is_bpe && (name_in(model_len, model, GEMMA4_MODEL) || name_in(pre_len, pre, GEMMA4));
    return add_bos || gemma4;
}

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

    struct gguf_str_iter tokens = {.p = p, .left = count};
    for (size_t i = 0; i < tok->vocab_size; i++) {
        gguf_str_next(&tokens, &tok->token_str[i], &tok->token_len[i]);
    }

    /* tokenizer.ggml.merges: each entry is "left right"
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
        tok->n_merges               = (size_t) count;
        struct gguf_str_iter merges = {.p = p, .left = count};
        const char          *s;
        size_t               slen;
        for (size_t i = 0; gguf_str_next(&merges, &s, &slen); i++) {
            size_t sp = 0;
            while (sp < slen && s[sp] != ' ')
                sp++;
            /* No space leaves the right half empty. It points at the end of
             * the string, never at null: load_copy memcpys it and
             * merge_lookup memcmps it, and both need a valid pointer even
             * for 0 bytes. */
            const size_t r          = sp < slen ? sp + 1 : slen;
            tok->merge_left[i]      = s;
            tok->merge_left_len[i]  = sp;
            tok->merge_right[i]     = s + r;
            tok->merge_right_len[i] = slen - r;
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

    if (!build_indices(tok)) {
        gguf_tokenizer_unload(tok);
        return false;
    }
    load_specials(tok, ctx);

    /* Resolve the tokenization algorithm. "gpt2" is byte-level BPE; any
     * other model that ships merges (Gemma/Llama/Mistral SentencePiece,
     * model="gemma4"/"llama"/…) is driven through the same merge engine
     * with ▁ normalization + <0xXX> byte fallback. Without merges, per-token
     * scores select unigram; with neither the mode is unsupported and
     * load_copy refuses, so the engine falls back to tokenizer.bin. */
    if (tok->model_len == 4 && memcmp(tok->model, "gpt2", 4) == 0) {
        tok->mode           = GGUF_TOK_MODE_GPT2;
        size_t      pre_len = 0;
        const char *pre     = gguf_get_meta_string(ctx, "tokenizer.ggml.pre", &pre_len);
        /* The pretokenizers llama.cpp runs the llama3 regex for
         * (LLAMA_VOCAB_PRE_TYPE_LLAMA3, with ignore_merges), and qwen2's,
         * which differs by single digits; qwen35 adds \p{M} to the letter
         * classes (#281). */
        static const char *const LLAMA3[] = {"llama3",
                                             "llama-v3",
                                             "llama-bpe",
                                             "falcon3",
                                             "falcon-h1",
                                             "pixtral",
                                             "midm-2.0",
                                             "lfm2",
                                             "jina-v5-nano",
                                             nullptr};
        static const char *const QWEN2[]  = {"qwen2", "qwen35", nullptr};
        static const char *const QWEN35[] = {"qwen35", nullptr};
        tok->pre_ignore_merges            = name_in(pre_len, pre, LLAMA3);
        tok->pre_digits = tok->pre_ignore_merges ? 3 : name_in(pre_len, pre, QWEN2) ? 1 : 0;
        tok->pre_marks  = name_in(pre_len, pre, QWEN35);
        /* The GPT-2 regex: LLAMA_VOCAB_PRE_TYPE_GPT2 and the types that share
         * its regex; the SMOLLM group runs "\p{N}" before it (#403). */
        static const char *const GPT2[] = {
                "gpt-2",      "phi-2",        "jina-es",     "jina-de",         "gigachat",
                "jina-v2-es", "jina-v2-de",   "a.x-4.0",     "mellum",          "modern-bert",
                "jina-v1-en", "jina-v2-code", "roberta-bpe", "exaone4",         "mpt",
                "olmo",       "jais",         "trillion",    "granite-docling", nullptr};
        static const char *const SMOLLM[] = {"smollm",
                                             "starcoder",
                                             "refact",
                                             "command-r",
                                             "codeshell",
                                             "exaone",
                                             "minerva-7b",
                                             "mellum2",
                                             nullptr};
        tok->pre_split_digits             = name_in(pre_len, pre, SMOLLM);
        tok->pre_gpt2                     = tok->pre_split_digits || name_in(pre_len, pre, GPT2);
        /* llama.cpp runs one regex for both (LLAMA_VOCAB_PRE_TYPE_GPT4O, #243). */
        tok->pre_gpt4o = pre != nullptr && ((pre_len == 6 && memcmp(pre, "gpt-4o", 6) == 0) ||
                                            (pre_len == 6 && memcmp(pre, "llama4", 6) == 0));
        /* Regex sequences (#242). */
        if (pre != nullptr && pre_len == 12 && memcmp(pre, "deepseek-llm", 12) == 0)
            tok->pre_seq = GGUF_PRE_DEEPSEEK_LLM;
        else if (pre != nullptr && pre_len == 11 && memcmp(pre, "deepseek-v3", 11) == 0)
            tok->pre_seq = GGUF_PRE_DEEPSEEK_V3;
        else
            tok->pre_seq = GGUF_PRE_NONE;
    } else if (tok->n_merges > 0) {
        tok->mode = GGUF_TOK_MODE_SPM;
        /* llama.cpp's BPE vocabulary types (Gemma 4) read the text as
         * codepoints, a malformed byte as U+FFFD; its SPM ("llama") keeps
         * the raw bytes (#427). */
        static const char *const BPE_SPM[] = {"gemma4", nullptr};
        tok->spm_fffd                      = name_in(tok->model_len, tok->model, BPE_SPM);
    } else if (tok->scores != nullptr) {
        /* No merges but per-token scores -> SentencePiece unigram (Llama/BitNet). */
        tok->mode = GGUF_TOK_MODE_UNIGRAM;
    } else {
        tok->mode = GGUF_TOK_MODE_UNSUPPORTED;
    }

    /* BOS prepend policy: llama.cpp's (gguf_tokenizer_add_bos_rule). A
     * byte-BPE file without the flag prepends only for the pretokenizers
     * llama.cpp names (Llama 3's "llama-bpe", ...), so Qwen3.8 GGUFs, which
     * carry a bos_id with no flag, still get none: prepending desynced the
     * qwen35 recurrent stack into repetition loops (#281). Without a BOS id
     * there is nothing to prepend, whatever the default says (#385). */
    {
        bool b;
        if (gguf_get_meta_bool(ctx, "tokenizer.ggml.add_eos_token", &b)) {
            tok->add_eos = b;
        }
        size_t      pre_len  = 0;
        const char *pre      = gguf_get_meta_string(ctx, "tokenizer.ggml.pre", &pre_len);
        const bool  has_flag = gguf_get_meta_bool(ctx, "tokenizer.ggml.add_bos_token", &b);
        tok->add_bos =
                gguf_tokenizer_add_bos_rule(tok->model_len,
                                            pre_len,
                                            has_flag ? (b ? 1 : 0) : -1,
                                            tok->bos_id >= 0 && tok->mode != GGUF_TOK_MODE_GPT2,
                                            tok->model,
                                            pre);
        if (!has_flag && tok->bos_id < 0)
            tok->add_bos = false;
    }

    if (tok->mode == GGUF_TOK_MODE_SPM || tok->mode == GGUF_TOK_MODE_UNIGRAM) {
        /* SentencePiece add_dummy_prefix — defaults to true when absent. */
        bool b;
        tok->add_space_prefix =
                gguf_get_meta_bool(ctx, "tokenizer.ggml.add_space_prefix", &b) ? b : true;
        build_spm_byte_map(tok);
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
    tok->vocab_size      = 0;
    tok->n_merges        = 0;
    tok->n_specials      = 0;
    tok->n_user_specials = 0;
}

/* Copy-mode loader. Calls the mmap-pointing loader first, then
 * walks every vocab + merge string and memcpys the bytes into heap
 * arenas, rewriting the pointer slots to indirect through them. The
 * model name string is copied too. After this returns the caller can
 * close the GGUF — the tokenizer no longer depends on the mmap. */
[[nodiscard]] bool gguf_tokenizer_load_copy(struct gguf_tokenizer *tok,
                                            const struct gguf_ctx *ctx) {
    if (!gguf_tokenizer_load(tok, ctx))
        return false;
    /* Refuse an unsupported mode so the engine falls back to an external
     * tokenizer.bin (gguf_tokenizer_load_bin). */
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
    /* Specials (both runs) alias into token_str[id]; re-point to the
     * freshly-copied vocab arena bytes so they survive the gguf_ctx close. */
    for (size_t s = 0; s < tok->n_specials + tok->n_user_specials; s++) {
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

/* ---- tokenizer.bin ------------------------------------------------------ */

constexpr unsigned TOKENIZER_BIN_MAGIC   = 0x4B544D47u; /* "GMTK" little-endian */
constexpr unsigned TOKENIZER_BIN_VERSION = 1u;

static uint32_t rd_u32(const uint8_t *p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

static uint16_t rd_u16(const uint8_t *p) {
    uint16_t v;
    memcpy(&v, p, 2);
    return v;
}

/* A u16-length-prefixed string at *p; false when the file ends first. The
 * length is checked by subtraction before p moves (AGENT.md §4). */
static bool rd_str(const uint8_t **p, const uint8_t *end, const char **s, size_t *len) {
    if ((size_t) (end - *p) < 2) {
        return false;
    }
    const uint16_t n = rd_u16(*p);
    *p += 2;
    if (n > (size_t) (end - *p)) {
        return false;
    }
    *s   = (const char *) *p;
    *len = n;
    *p += n;
    return true;
}

/* Parse the file image [p, end) (the vocab arena) into tok. Every count is
 * checked against the bytes left before its table is allocated: an entry
 * takes at least 2 (token), 4 (merge) or 6 (special) bytes. */
static bool parse_tokenizer_bin(struct gguf_tokenizer *tok, const uint8_t *p, const uint8_t *end) {
    if ((size_t) (end - p) < 36) {
        fprintf(stderr, "tokenizer.bin: file too small for header\n");
        return false;
    }
    const uint32_t magic = rd_u32(p), version = rd_u32(p + 4);
    if (magic != TOKENIZER_BIN_MAGIC) {
        fprintf(stderr, "tokenizer.bin: bad magic %08x\n", magic);
        return false;
    }
    if (version != TOKENIZER_BIN_VERSION) {
        fprintf(stderr, "tokenizer.bin: unsupported version %u\n", version);
        return false;
    }
    const uint32_t n_tok = rd_u32(p + 8), n_merges = rd_u32(p + 12), n_specials = rd_u32(p + 16);
    tok->bos_id = (int32_t) rd_u32(p + 20);
    tok->eos_id = (int32_t) rd_u32(p + 24);
    /* p + 28: pad id, which nothing reads. */
    tok->unk_id = (int32_t) rd_u32(p + 32);
    p += 36;

    if (n_tok > (size_t) (end - p) / 2) {
        goto truncated;
    }
    /* An empty vocab fails here too (a zero count allocates nothing). */
    tok->token_str = heap_alloc_array_aligned(const char *, n_tok);
    tok->token_len = heap_alloc_array_aligned(size_t, n_tok);
    if (tok->token_str == nullptr || tok->token_len == nullptr) {
        goto oom;
    }
    tok->vocab_size = n_tok;
    for (size_t i = 0; i < n_tok; i++) {
        if (!rd_str(&p, end, &tok->token_str[i], &tok->token_len[i])) {
            goto truncated;
        }
    }

    if (n_merges > (size_t) (end - p) / 4) {
        goto truncated;
    }
    if (n_merges > 0) {
        tok->merge_left      = heap_alloc_array_aligned(const char *, n_merges);
        tok->merge_left_len  = heap_alloc_array_aligned(size_t, n_merges);
        tok->merge_right     = heap_alloc_array_aligned(const char *, n_merges);
        tok->merge_right_len = heap_alloc_array_aligned(size_t, n_merges);
        if (tok->merge_left == nullptr || tok->merge_left_len == nullptr ||
            tok->merge_right == nullptr || tok->merge_right_len == nullptr) {
            goto oom;
        }
        tok->n_merges = n_merges;
    }
    for (size_t m = 0; m < n_merges; m++) {
        if (!rd_str(&p, end, &tok->merge_left[m], &tok->merge_left_len[m]) ||
            !rd_str(&p, end, &tok->merge_right[m], &tok->merge_right_len[m])) {
            goto truncated;
        }
    }

    if (n_specials > (size_t) (end - p) / 6) {
        goto truncated;
    }
    if (n_specials > 0) {
        tok->specials = heap_alloc_array_aligned(struct gguf_tokenizer_special, n_specials);
        if (tok->specials == nullptr) {
            goto oom;
        }
    }
    for (size_t i = 0; i < n_specials; i++) {
        if ((size_t) (end - p) < 4) {
            goto truncated;
        }
        const int32_t id = (int32_t) rd_u32(p);
        p += 4;
        if (!rd_str(&p, end, &tok->specials[i].text, &tok->specials[i].len)) {
            goto truncated;
        }
        tok->specials[i].id   = id;
        tok->specials[i].user = false; /* no token types: none is plain text */
        tok->n_specials       = i + 1;
    }
    return true;

truncated:
    fprintf(stderr, "tokenizer.bin: truncated file\n");
    return false;
oom:
    fprintf(stderr, "tokenizer.bin: allocation failed\n");
    return false;
}

[[nodiscard]] bool gguf_tokenizer_load_bin(struct gguf_tokenizer *tok, const char *path) {
    if (tok == nullptr || path == nullptr) {
        return false;
    }
    *tok = (struct gguf_tokenizer) {0};
    struct io_map m;
    const char   *err = nullptr;
    if (!io_map_ro(path, MAP_PRIVATE, &m, &err)) {
        perror(err); /* "open() failed: No such file or directory" */
        return false;
    }
    /* The strings stay in a heap copy of the file, so the mapping goes now. */
    tok->vocab_arena = heap_alloc_array_aligned(char, m.size);
    if (tok->vocab_arena != nullptr) {
        memcpy(tok->vocab_arena, m.map, m.size);
    }
    munmap(m.map, m.size);
    close(m.fd);
    if (tok->vocab_arena == nullptr) {
        fprintf(stderr, "tokenizer.bin: allocation failed\n");
        return false;
    }
    const uint8_t *p = (const uint8_t *) tok->vocab_arena;
    if (!parse_tokenizer_bin(tok, p, p + m.size)) {
        gguf_tokenizer_unload(tok);
        return false;
    }
    if (!build_indices(tok)) {
        fprintf(stderr, "tokenizer.bin: allocation failed\n");
        gguf_tokenizer_unload(tok);
        return false;
    }
    sort_specials(tok);
    index_special_first(tok);
    tok->mode          = GGUF_TOK_MODE_SPM;
    tok->tokenizer_bin = true;
    tok->add_bos       = true; /* the Gemma models it serves expect BOS (#263) */
    build_spm_byte_map(tok);
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

/* ---- Encoder + hash indices ---- */

/* FNV-1a 64-bit. Stable across builds; we just need a good
 * distribution over short byte strings. */
static uint64_t fnv1a64(const char *p, size_t n) {
    return geist_fnv1a(GEIST_FNV1A_INIT, n, p);
}

/* Hash a (left, right) pair. The separator byte ensures
 * ("ab", "") and ("a", "b") hash distinctly. */
static uint64_t fnv1a64_pair(const char *a, size_t alen, const char *b, size_t blen) {
    return geist_fnv1a(geist_fnv1a(fnv1a64(a, alen), 1, "\xff"), blen, b);
}

static size_t next_pow2(size_t n) {
    size_t p = 1;
    while (p < n)
        p <<= 1;
    return p;
}

/* O(1) vocab lookup (the load built the table). Returns -1 when no entry
 * matches. */
static int32_t vocab_lookup(const struct gguf_tokenizer *tok, const char *bytes, size_t len) {
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
 * (left, right) as its operands; the table is absent only without merges. */
static size_t merge_lookup(const struct gguf_tokenizer *tok,
                           const char                  *lstr,
                           size_t                       llen,
                           const char                  *rstr,
                           size_t                       rlen) {
    if (tok->merge_hash == nullptr) {
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

/* Find the lowest-rank merge in the symbol list. Returns the index
 * of the LEFT symbol of the winning pair, or -1 when no adjacent
 * pair has a merge entry. */
static int find_best_merge(const struct gguf_tokenizer *tok,
                           const char                  *buf,
                           const struct pair_merge_sym *syms,
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

/* Symbols up to which a chunk rescans with find_best_merge instead of
 * running the merge heap: gpt2 hands over one pre-tokenized word at a
 * time, and for a word of a few symbols the rescan costs no more lookups
 * and none of the heap's bookkeeping. The heap wins from ~9 symbols up
 * (bench_tokenizer, letter runs). */
constexpr int BPE_RESCAN_MAX_SYMS = 8;

/* Merge-engine key (pair_merge.h) for BPE: the merge's rank, lowest
 * first. */
static bool bpe_merge_key(
        const void *ctx, const char *buf, size_t off, size_t llen, size_t rlen, uint64_t *key) {
    const size_t m = merge_lookup(ctx, buf + off, llen, buf + off + llen, rlen);
    *key           = m;
    return m != SIZE_MAX;
}

/* Apply one BPE step on a pre-tokenized chunk's byte buffer, emitting
 * vocab IDs into out. Returns the count, or SIZE_MAX when scratch
 * allocation fails — a caller that adds the return value blindly would
 * otherwise turn dropped tokens into a silently shorter encoding.
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
    if (tok->pre_ignore_merges) { /* llama.cpp's ignore_merges (Llama 3) */
        const int32_t id = vocab_lookup(tok, buf, buf_len);
        if (id >= 0) {
            out[0] = id;
            return 1;
        }
    }
    /* One codepoint has no pair to merge: skip the scratch (smollm's
     * digits and lone spaces, #403). */
    size_t adv1;
    (void) utf8_decode_one(buf, buf_len, &adv1);
    if (adv1 == buf_len && !tok->tokenizer_bin) {
        const int32_t id = vocab_lookup(tok, buf, buf_len);
        if (id >= 0) {
            out[0] = id;
            return 1;
        }
    }

    /* Decompose into one symbol per UTF-8 codepoint. Cap symbols at
     * buf_len since each codepoint takes >= 1 byte. */
    struct pair_merge_sym *syms = heap_alloc_array_aligned(struct pair_merge_sym, buf_len);
    if (syms == nullptr)
        return SIZE_MAX; /* Allocation failure, never an empty chunk. */
    /* A local, so the stores to syms (-fno-strict-aliasing) do not reload it. */
    const bool lead_only = tok->tokenizer_bin;
    int        n_syms    = 0;
    size_t     k         = 0;
    while (k < buf_len) {
        size_t adv;
        if (lead_only) {
            adv = utf8_lead_len((uint8_t) buf[k]);
        } else {
            (void) utf8_decode_one(buf + k, buf_len - k, &adv);
        }
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

    /* Greedy merges, lowest rank first, leftmost among equal ranks, until
     * no adjacent pair has a merge: a rescan per merge for a short chunk,
     * the merge heap (pair_merge.h, O(n log n)) for a longer one. */
    int head = 0;
    if (n_syms <= BPE_RESCAN_MAX_SYMS) {
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
    } else if (!pair_merge_run((size_t) n_syms, syms, buf, bpe_merge_key, tok)) {
        void *p = syms;
        safe_free(&p);
        return SIZE_MAX;
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

/* ---- llama3 / qwen2 pretokenizer scanner (#275, #399) ------------------ *
 *
 * Reproduces llama.cpp's regex for "llama-bpe" (Llama 3) and "qwen2" over
 * UTF-8 codepoints (unicode_regex_split_custom_llama3 / _qwen2 in
 * unicode.cpp):
 *
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d)        contractions
 *   |[^\r\n\p{L}\p{N}]?\p{L}+           one-cp prefix + letter run
 *   |\p{N}{1,3}                         llama3: up to 3 digits; qwen2: \p{N}
 *   | ?[^\s\p{L}\p{N}]+[\r\n]*          punct run (optional lead space)
 *   |\s*[\r\n]+                          newline groups
 *   |\s+(?!\S)                           trailing-ws (run minus last cp)
 *   |\s+
 *
 * "qwen35" adds \p{M} to the letter run and removes it from the punct run,
 * so a mark classifies as a letter there. \p{L} / \p{N} and \s are
 * llama.cpp's own (unicode_class.h and its whitespace list). */
uint32_t gguf_tokenizer_utf8_cp(size_t n, size_t i, const char text[static n], size_t *adv) {
    static const int32_t MIN_CP[5] = {0, 0, 0x80, 0x800, 0x10000};
    const int32_t        cp        = utf8_decode_one(text + i, n - i, adv);
    if (cp < MIN_CP[*adv] || cp > 0x10FFFF) {
        *adv = 1;
        return 0xFFFD;
    }
    return (uint32_t) cp;
}
/* gguf_tokenizer_utf8_cp in the scanners' argument order. */
static uint32_t q2_cp_at(const char *text, size_t tlen, size_t i, size_t *adv) {
    return gguf_tokenizer_utf8_cp(tlen, i, text, adv);
}

/* Whether text[i], i < tlen, is a byte llama.cpp reads as U+FFFD
 * (q2_cp_at); *adv is the bytes its codepoint takes. */
static bool utf8_bad_at(const char *text, size_t tlen, size_t i, size_t *adv) {
    *adv = 1;
    return (unsigned char) text[i] >= 0x80 && q2_cp_at(text, tlen, i, adv) == 0xFFFD && *adv == 1;
}

/* U+FFFD, which llama.cpp writes for a malformed byte when it rebuilds a
 * word from its codepoints (unicode_regex_split). */
static const char UTF8_FFFD[3] = {(char) 0xEF, (char) 0xBF, (char) 0xBD};

/* text[from, to) in GPT-2's byte-level form, as llama.cpp's
 * unicode_regex_split + unicode_byte_encoding_process build it: every
 * malformed byte replaced by U+FFFD's three bytes (#427), each byte then
 * mapped to its GPT-2 codepoint and written to out as UTF-8 — at most 6
 * bytes per byte of text. Returns the bytes written. */
static size_t gpt2_chunk_bytes(size_t to, size_t from, const char text[static to], char *out) {
    size_t w = 0;
    for (size_t b = from; b < to;) {
        size_t       adv;
        const bool   bad = utf8_bad_at(text, to, b, &adv);
        const char  *src = bad ? UTF8_FFFD : text + b;
        const size_t n   = bad ? sizeof UTF8_FFFD : adv;
        for (size_t k = 0; k < n; k++)
            w += utf8_encode_one(gpt2_byte_to_codepoint((unsigned char) src[k]), out + w);
        b += adv;
    }
    return w;
}
static bool q2_is_ws(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0x0B || cp == 0x0C ||
           cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 ||
           cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

enum q2_kind : uint32_t {
    Q2_OTHER, /* [^\s\p{L}\p{N}] */
    Q2_WS,    /* \s */
    Q2_LETTER,
    Q2_DIGIT, /* \p{N} */
};
static_assert(sizeof(enum q2_kind) == sizeof(uint32_t));

static enum q2_kind q2_kind_at(const char *text, size_t tlen, size_t i, bool marks, size_t *adv) {
    const unsigned char b = (unsigned char) text[i];
    if (b < 0x80) { /* without the decode: most of any text (#403) */
        *adv = 1;
        if (b == ' ' || (b >= 0x09 && b <= 0x0D))
            return Q2_WS;
        if ((b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z'))
            return Q2_LETTER;
        return b >= '0' && b <= '9' ? Q2_DIGIT : Q2_OTHER;
    }
    const uint32_t cp = q2_cp_at(text, tlen, i, adv);
    if (cp == 0x85 || q2_is_ws(cp))
        return Q2_WS;
    switch (unicode_class_of(cp)) {
    case UNICODE_CLASS_LETTER:
        return Q2_LETTER;
    case UNICODE_CLASS_MARK:
        return marks ? Q2_LETTER : Q2_OTHER;
    case UNICODE_CLASS_NUMBER:
        return Q2_DIGIT;
    case UNICODE_CLASS_OTHER:
    case UNICODE_CLASS_PUNCT:
    case UNICODE_CLASS_SYMBOL:
        break;
    }
    return Q2_OTHER;
}

/* End of the run of `kind` codepoints starting at j. */
static size_t q2_run_end(const char *text, size_t tlen, size_t j, bool marks, enum q2_kind kind) {
    size_t adv;
    while (j < tlen && q2_kind_at(text, tlen, j, marks, &adv) == kind)
        j += adv;
    return j;
}

/* Consume `[^\s\p{L}\p{N}]+[\r\n]*` starting at the first punct cp. */
static size_t q2_punct_run_end(const char *text, size_t tlen, size_t first, bool marks) {
    size_t j = q2_run_end(text, tlen, first, marks, Q2_OTHER);
    while (j < tlen && (text[j] == '\n' || text[j] == '\r'))
        j++;
    return j;
}

/* End (exclusive) of the pretokenizer match starting at i < tlen, where
 * tlen ends the fragment (the text up to the next special token). Follows
 * the regex alternation order exactly; never returns i. */
size_t gguf_tokenizer_llama3_chunk_end(
        size_t tlen, size_t i, size_t max_digits, bool marks, const char text[static tlen]) {
    size_t             adv0;
    const enum q2_kind k0 = q2_kind_at(text, tlen, i, marks, &adv0);

    /* 1. contractions, case-insensitive */
    if (text[i] == '\'' && i + 1 < tlen) {
        const char n1 = (char) (text[i + 1] | 0x20);
        if (n1 == 's' || n1 == 't' || n1 == 'm' || n1 == 'd')
            return i + 2;
        if (i + 2 < tlen) {
            const char n2 = (char) (text[i + 2] | 0x20);
            if ((n1 == 'r' && n2 == 'e') || (n1 == 'v' && n2 == 'e') || (n1 == 'l' && n2 == 'l'))
                return i + 3;
        }
    }

    /* 2. letter run, after one optional codepoint that is no \r, \n or digit */
    if (k0 == Q2_LETTER)
        return q2_run_end(text, tlen, i + adv0, marks, Q2_LETTER);
    const bool   more = i + adv0 < tlen;
    size_t       adv1 = 0;
    enum q2_kind k1   = more ? q2_kind_at(text, tlen, i + adv0, marks, &adv1) : Q2_WS;
    if (k1 == Q2_LETTER && k0 != Q2_DIGIT && text[i] != '\r' && text[i] != '\n')
        return q2_run_end(text, tlen, i + adv0 + adv1, marks, Q2_LETTER);

    /* 3. digits */
    if (k0 == Q2_DIGIT) {
        size_t j = i + adv0, adv;
        for (size_t n = 1; n < max_digits && j < tlen; n++) {
            if (q2_kind_at(text, tlen, j, marks, &adv) != Q2_DIGIT)
                break;
            j += adv;
        }
        return j;
    }

    /* 4. punct run, after an optional ASCII space */
    if (k0 == Q2_OTHER)
        return q2_punct_run_end(text, tlen, i, marks);
    if (text[i] == ' ' && k1 == Q2_OTHER && more)
        return q2_punct_run_end(text, tlen, i + 1, marks);

    /* Whitespace: scan the run; remember the end of the last \r|\n. */
    size_t j = i, last_nl_end = 0, n_cps = 0, last_cp_start = i;
    while (j < tlen) {
        size_t adv;
        if (q2_kind_at(text, tlen, j, marks, &adv) != Q2_WS)
            break;
        if (text[j] == '\n' || text[j] == '\r')
            last_nl_end = j + 1;
        last_cp_start = j;
        j += adv;
        n_cps++;
    }
    if (last_nl_end > 0)
        return last_nl_end; /* 5. \s*[\r\n]+ */
    if (n_cps >= 2 && j < tlen)
        return last_cp_start; /* 6. \s+(?!\S): run minus last cp */
    return j;                 /* 7. \s+ */
}

/* ---- GPT-2 pretokenizer scanner (#403) --------------------------------- *
 *
 * llama.cpp's unicode_regex_split_custom_gpt2, with the classes of the
 * llama3 scanner above (\p{M} is punctuation here):
 *
 *   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
 *
 * split_digits: llama.cpp ran "\p{N}" over the text first ("smollm" and the
 * rest of its LLAMA_VOCAB_PRE_TYPE_SMOLLM group), so each \p{N} codepoint
 * is a fragment of its own. Inside the regex a digit then acts as the
 * fragment's end: " 7" is " " "7", and "a  7" keeps its whitespace whole. */
size_t gguf_tokenizer_gpt2_chunk_end(size_t     tlen,
                                     size_t     i,
                                     bool       split_digits,
                                     const char text[static tlen]) {
    size_t             adv0;
    const enum q2_kind k0 = q2_kind_at(text, tlen, i, false, &adv0);
    if (split_digits && k0 == Q2_DIGIT)
        return i + adv0;

    /* contractions, lower case only */
    if (text[i] == '\'' && i + 1 < tlen) {
        const char n1 = text[i + 1];
        if (n1 == 's' || n1 == 't' || n1 == 'm' || n1 == 'd')
            return i + 2;
        if (i + 2 < tlen) {
            const char n2 = text[i + 2];
            if ((n1 == 'r' && n2 == 'e') || (n1 == 'v' && n2 == 'e') || (n1 == 'l' && n2 == 'l'))
                return i + 3;
        }
    }

    /* letter, number or punct run, after an optional ASCII space */
    if (k0 != Q2_WS)
        return q2_run_end(text, tlen, i + adv0, false, k0);
    if (text[i] == ' ' && i + 1 < tlen) {
        size_t             adv1;
        const enum q2_kind k1 = q2_kind_at(text, tlen, i + 1, false, &adv1);
        if (k1 != Q2_WS && !(split_digits && k1 == Q2_DIGIT))
            return q2_run_end(text, tlen, i + 1 + adv1, false, k1);
    }

    /* whitespace run: \s+(?!\S) leaves its last cp when \S follows */
    size_t       j = i, last = i, n = 0, adv = 0;
    enum q2_kind k = Q2_WS;
    while (j < tlen && (k = q2_kind_at(text, tlen, j, false, &adv)) == Q2_WS) {
        last = j;
        j += adv;
        n++;
    }
    if (n > 1 && j < tlen && !(split_digits && k == Q2_DIGIT))
        return last;
    return j;
}

/* ---- gpt-4o pretokenizer scanner (#243) -------------------------------- *
 *
 * tokenizer.ggml.pre "gpt-4o" (GPT-OSS's o200k_harmony) and "llama4". The
 * regex llama.cpp runs for them (llama-vocab.cpp, LLAMA_VOCAB_PRE_TYPE_GPT4O;
 * tokenizer.json's \p{Lu}/\p{Ll} classes rewritten as "a letter that is not
 * a-z" / "a letter that is not A-Z"):
 *
 *   1. [^\r\n\p{L}\p{N}]?U*L+C?      U = letter not a-z, L = letter not A-Z,
 *   2. [^\r\n\p{L}\p{N}]?U+L*C?      C = 's 't 're 've 'm 'll 'd (any case)
 *   3. \p{N}{1,3}
 *   4.  ?[^\s\p{L}\p{N}]+[\r\n/]*
 *   5. \s*[\r\n]+
 *   6. \s+(?!\S)
 *   7. \s+
 *
 * taken in that order at each position, as std::regex does (first matching
 * alternative, greedy with backtracking). It runs on llama.cpp's collapsed
 * text, so this scanner classifies the way that text does: a non-ASCII
 * letter is both U and L, a combining mark is neither letter nor number,
 * and the letter / number / whitespace sets are llama.cpp's own
 * (unicode_class.h; the whitespace list below). Hence "ÉCOLE" splits after
 * É, as it does in llama.cpp (HF, with real case classes, keeps it whole). */
enum g4_kind : uint32_t {
    G4_OTHER, /* [^\s\p{L}\p{N}] */
    G4_SPACE, /* \s other than \r \n */
    G4_NL,    /* \r \n */
    G4_UPPER, /* A-Z */
    G4_LOWER, /* a-z */
    G4_ALPHA, /* a non-ASCII letter: both U and L */
    G4_DIGIT, /* \p{N} */
};
static_assert(sizeof(enum g4_kind) == sizeof(uint32_t));

static enum g4_kind g4_kind_at(const char *text, size_t tlen, size_t i, size_t *adv) {
    const uint32_t cp = q2_cp_at(text, tlen, i, adv);
    if (cp < 0x80) {
        if (cp == '\r' || cp == '\n')
            return G4_NL;
        if (cp == ' ' || cp == '\t' || cp == 0x0B || cp == 0x0C)
            return G4_SPACE;
        if (cp >= 'A' && cp <= 'Z')
            return G4_UPPER;
        if (cp >= 'a' && cp <= 'z')
            return G4_LOWER;
        return cp >= '0' && cp <= '9' ? G4_DIGIT : G4_OTHER;
    }
    if (cp == 0x85 || q2_is_ws(cp))
        return G4_SPACE;
    switch (unicode_class_of(cp)) {
    case UNICODE_CLASS_LETTER:
        return G4_ALPHA;
    case UNICODE_CLASS_NUMBER:
        return G4_DIGIT;
    case UNICODE_CLASS_OTHER:
    case UNICODE_CLASS_PUNCT:
    case UNICODE_CLASS_MARK:
    case UNICODE_CLASS_SYMBOL:
        break;
    }
    return G4_OTHER;
}

static bool g4_is_u(enum g4_kind k) {
    return k == G4_UPPER || k == G4_ALPHA;
}
static bool g4_is_l(enum g4_kind k) {
    return k == G4_LOWER || k == G4_ALPHA;
}
static bool g4_is_letter(enum g4_kind k) {
    return k == G4_UPPER || k == G4_LOWER || k == G4_ALPHA;
}

/* End of an optional contraction at j ('s 't 're 've 'm 'll 'd, any case). */
static size_t g4_contraction_end(const char *text, size_t tlen, size_t j) {
    if (j + 1 >= tlen || text[j] != '\'')
        return j;
    const char a = (char) (text[j + 1] | 0x20);
    if (a == 's' || a == 't' || a == 'm' || a == 'd')
        return j + 2;
    if (j + 2 < tlen) {
        const char b = (char) (text[j + 2] | 0x20);
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l'))
            return j + 3;
    }
    return j;
}

/* Alternatives 1 and 2 from the letter at s. */
static size_t g4_letters_end(const char *text, size_t tlen, size_t s) {
    /* U* greedy, remembering the last cp that is also an L. */
    size_t j = s, last_l = SIZE_MAX, last_l_adv = 0, adv;
    while (j < tlen) {
        const enum g4_kind k = g4_kind_at(text, tlen, j, &adv);
        if (!g4_is_u(k))
            break;
        if (g4_is_l(k))
            last_l = j, last_l_adv = adv;
        j += adv;
    }
    if (j < tlen && g4_is_l(g4_kind_at(text, tlen, j, &adv))) {
        /* 1. L+ right after the whole U run. */
        while (j < tlen && g4_is_l(g4_kind_at(text, tlen, j, &adv)))
            j += adv;
        return g4_contraction_end(text, tlen, j);
    }
    if (last_l != SIZE_MAX) /* 1. backtracked: U* ends before the last L of the run */
        return g4_contraction_end(text, tlen, last_l + last_l_adv);
    return g4_contraction_end(text, tlen, j); /* 2. U+ (j > s: s is a letter, not an L) */
}

size_t gguf_tokenizer_gpt4o_chunk_end(size_t tlen, size_t i, const char text[static tlen]) {
    size_t             adv0, adv;
    const enum g4_kind k0 = g4_kind_at(text, tlen, i, &adv0);

    if (g4_is_letter(k0))
        return g4_letters_end(text, tlen, i);
    const bool         more = i + adv0 < tlen;
    const enum g4_kind k1   = more ? g4_kind_at(text, tlen, i + adv0, &adv) : G4_NL;
    /* 1./2. with the one-codepoint prefix [^\r\n\p{L}\p{N}]. */
    if ((k0 == G4_OTHER || k0 == G4_SPACE) && more && g4_is_letter(k1))
        return g4_letters_end(text, tlen, i + adv0);
    if (k0 == G4_DIGIT) { /* 3. */
        size_t j = i + adv0;
        for (int n = 1; n < 3 && j < tlen && g4_kind_at(text, tlen, j, &adv) == G4_DIGIT; n++)
            j += adv;
        return j;
    }
    /* 4. punctuation run, after an optional ASCII space. */
    size_t p = SIZE_MAX;
    if (k0 == G4_OTHER)
        p = i;
    else if (text[i] == ' ' && more && k1 == G4_OTHER)
        p = i + 1;
    if (p != SIZE_MAX) {
        while (p < tlen && g4_kind_at(text, tlen, p, &adv) == G4_OTHER)
            p += adv;
        while (p < tlen && (text[p] == '\r' || text[p] == '\n' || text[p] == '/'))
            p++;
        return p;
    }
    /* Whitespace (k0 is G4_SPACE or G4_NL). */
    size_t j = i, last_nl_end = 0, n_cps = 0, last_start = i;
    while (j < tlen) {
        const enum g4_kind k = g4_kind_at(text, tlen, j, &adv);
        if (k != G4_SPACE && k != G4_NL)
            break;
        if (k == G4_NL)
            last_nl_end = j + adv;
        last_start = j;
        j += adv;
        n_cps++;
    }
    if (last_nl_end > 0)
        return last_nl_end; /* 5. */
    if (j >= tlen || n_cps < 2)
        return j;      /* 6. at the end, else 7. one codepoint */
    return last_start; /* 6. the run but its last codepoint */
}

/* Merge-engine key for unigram: the joined piece's score, highest first.
 * The float's bits made to sort as unsigned (sign bit set: all bits
 * flipped; clear: sign bit set) and then reversed. -0 is made +0 first so
 * the two tie, as a float compare would (fast-math may not tell them apart). */
static bool unigram_merge_key(
        const void *ctx, const char *buf, size_t off, size_t llen, size_t rlen, uint64_t *key) {
    const struct gguf_tokenizer *tok = ctx;
    const int32_t                id  = vocab_lookup(tok, buf + off, llen + rlen);
    if (id < 0)
        return false;
    uint32_t u;
    memcpy(&u, &tok->scores[id], sizeof u);
    if ((u << 1) == 0)
        u = 0;
    u    = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    *key = UINT32_MAX - u;
    return true;
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
    struct pair_merge_sym *syms = heap_alloc_array_aligned(struct pair_merge_sym, buf_len);
    if (syms == nullptr)
        return SIZE_MAX; /* Allocation failure, never an empty chunk. */
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
     * vocab token with the highest score, the leftmost among equal scores.
     * Stops when no pair forms a token (pair_merge.h, O(n log n)). */
    int head = 0;
    if (!pair_merge_run((size_t) n_syms, syms, buf, unigram_merge_key, tok)) {
        void *p = syms;
        safe_free(&p);
        return SIZE_MAX;
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

/* The special tokens one encode matches (#409): all of tok's, except that
 * inside the caller's plain ranges only the `user` ones are, and a special
 * that reaches into a plain range is not matched at all. plain is null when
 * there are no plain ranges (gguf_tokenizer_encode), else one byte per text
 * byte, 1 inside a range. */
struct special_scan {
    const struct gguf_tokenizer_special *all, *user;
    size_t                               n_all, n_user;
    const uint8_t                       *first; /* tok->special_first */
    const uint8_t                       *plain;
};

/* The longest special token the scan matches at text[p], p < tlen: its
 * length, and its id in *id; 0 when none. specials are sorted longest
 * first, so the first allowed hit is the leftmost-longest match. */
static inline size_t special_at(const struct special_scan *sc,
                                size_t                     tlen,
                                size_t                     p,
                                const char                 text[static tlen],
                                int32_t                   *id) {
    const bool in_plain = sc->plain != nullptr && sc->plain[p] != 0;
    if ((sc->first[(unsigned char) text[p]] & (in_plain ? 2 : 1)) == 0)
        return 0;
    const struct gguf_tokenizer_special *sp = in_plain ? sc->user : sc->all;
    const size_t                         n  = in_plain ? sc->n_user : sc->n_all;
    for (size_t s = 0; s < n; s++) {
        const size_t slen = sp[s].len;
        if (slen == 0 || slen > tlen - p || text[p] != sp[s].text[0] ||
            memcmp(text + p, sp[s].text, slen) != 0)
            continue;
        if (!sp[s].user && sc->plain != nullptr && memchr(sc->plain + p, 1, slen) != nullptr)
            continue; /* reaches into a plain range */
        *id = sp[s].id;
        return slen;
    }
    return 0;
}

/* SentencePiece encode: there is no GPT-2 pre-tokenizer split — the algorithm
 * runs over each whole inter-special chunk after ▁ normalization (space → ▁,
 * plus an optional leading ▁ when add_space_prefix). Unknown symbols fall
 * back to <0xXX> byte tokens. Covers both the merge-driven (SPM) and unigram
 * paths via spm_chunk_to_ids. */
static bool encode_spm(const struct gguf_tokenizer *tok,
                       const struct special_scan   *sc,
                       const char                  *text,
                       int32_t                     *out_ids,
                       size_t                       cap,
                       size_t                      *n_out) {
    const size_t tlen = strlen(text);
    if (tlen == 0)
        return true;

    /* Scratch for one normalized chunk: worst case every byte is a space
     * (→ ▁, 3 bytes) plus an optional leading ▁. Reused across chunks. */
    GEIST_CLEANUP(heap_freep) char *buf =
            heap_alloc_array_aligned(char, tlen *SPM_MARKER_LEN + SPM_MARKER_LEN);
    if (buf == nullptr)
        return false;

    size_t i = 0, chunk_start = 0;
    bool   first_chunk = true;

/* Normalize text[chunk_start, end) into buf and BPE-encode it. */
#define SPM_FLUSH(end)                                                                         \
    do {                                                                                       \
        size_t w_ = 0;                                                                         \
        if (first_chunk && tok->add_space_prefix) {                                            \
            memcpy(buf + w_, SPM_MARKER, SPM_MARKER_LEN);                                      \
            w_ += SPM_MARKER_LEN;                                                              \
        }                                                                                      \
        for (size_t r_ = chunk_start, adv_ = 1; r_ < (end); r_ += adv_) {                      \
            adv_ = 1;                                                                          \
            if (text[r_] == ' ') {                                                             \
                memcpy(buf + w_, SPM_MARKER, SPM_MARKER_LEN);                                  \
                w_ += SPM_MARKER_LEN;                                                          \
            } else if (tok->spm_fffd && utf8_bad_at(text, (end), r_, &adv_)) {                 \
                memcpy(buf + w_, UTF8_FFFD, sizeof UTF8_FFFD);                                 \
                w_ += sizeof UTF8_FFFD;                                                        \
            } else { /* adv_: the codepoint's bytes when spm_fffd, else 1 */                   \
                memcpy(buf + w_, text + r_, adv_);                                             \
                w_ += adv_;                                                                    \
            }                                                                                  \
        }                                                                                      \
        first_chunk = false;                                                                   \
        if (w_ > 0 && *n_out < cap) {                                                          \
            size_t produced_ = spm_chunk_to_ids(tok, buf, w_, out_ids + *n_out, cap - *n_out); \
            if (produced_ == SIZE_MAX)                                                         \
                return false;                                                                  \
            *n_out += produced_;                                                               \
        }                                                                                      \
    } while (0)

    while (i < tlen) {
        int32_t      id   = -1;
        const size_t slen = *n_out < cap ? special_at(sc, tlen, i, text, &id) : 0;
        if (slen == 0) {
            i++;
            continue;
        }
        if (chunk_start < i)
            SPM_FLUSH(i);
        /* SPM_FLUSH may have filled the buffer to cap; guard the
         * special-token write so it truncates instead of overflowing. */
        if (*n_out < cap)
            out_ids[(*n_out)++] = id;
        i += slen;
        chunk_start = i;
    }
    if (chunk_start < tlen)
        SPM_FLUSH(tlen);
#undef SPM_FLUSH
    return true;
}

/* ---- Regex-sequence pretokenizers: deepseek-llm, deepseek-v3 (#242) ---- *
 *
 * llama.cpp runs these as a list of regexes (llama-vocab.cpp,
 * LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM / DEEPSEEK3_LLM), each over the pieces
 * the previous one left (unicode_regex_split): inside a piece, every
 * leftmost match becomes a piece and so does every gap between matches.
 * A regex with a \p{...} class runs on llama.cpp's collapsed text (a
 * non-ASCII codepoint becomes its class: letter, number, punctuation, mark,
 * symbol, whitespace or other); one without runs as std::wregex on the
 * codepoints themselves, non-ASCII whitespace replaced by \v.
 *
 * deepseek-llm:
 *   1. [\r\n]
 *   2. \s?[A-Za-zµÀ-Ö...]+      a fixed Latin/Greek/Cyrillic/... range list
 *   3. \s?[!-/:-~！-／：-～‘-‟　-。]+
 *   4. \s+$
 *   5. [一-龥ࠀ-一가-퟿]+
 *   6. \p{N}+
 * deepseek-v3:
 *   1. \p{N}{1,3}
 *   2. [一-龥぀-ゟ゠-ヿ]+
 *   3. [!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+
 *      |[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*
 *      |\s*[\r\n]+|\s+(?!\S)|\s+
 * Each matcher below answers "the match that starts exactly at p" with the
 * regex's own alternative order and backtracking; the driver searches
 * left to right as std::regex_iterator does. */

struct cp_range {
    uint32_t lo, hi;
};

static const struct cp_range DSL_LETTERS[] = {
        {0x41, 0x5A},       {0x61, 0x7A},       {0xB5, 0xB5},       {0xC0, 0xD6},
        {0xD8, 0xF6},       {0xF8, 0x1BA},      {0x1BC, 0x1BF},     {0x1C4, 0x293},
        {0x295, 0x2AF},     {0x370, 0x373},     {0x376, 0x377},     {0x37B, 0x37D},
        {0x37F, 0x37F},     {0x386, 0x386},     {0x388, 0x38A},     {0x38C, 0x38C},
        {0x38E, 0x3A1},     {0x3A3, 0x3F5},     {0x3F7, 0x481},     {0x48A, 0x52F},
        {0x531, 0x556},     {0x10A0, 0x10C5},   {0x13A0, 0x13F5},   {0x13F8, 0x13FD},
        {0x1C90, 0x1CBA},   {0x1CBD, 0x1CBF},   {0x1D00, 0x1D2B},   {0x1D6B, 0x1D77},
        {0x1D79, 0x1D9A},   {0x1E00, 0x1F15},   {0x1F18, 0x1F1D},   {0x1F20, 0x1F45},
        {0x1F48, 0x1F4D},   {0x1F50, 0x1F57},   {0x1F59, 0x1F59},   {0x1F5B, 0x1F5B},
        {0x1F5D, 0x1F5D},   {0x1F5F, 0x1F7D},   {0x1F80, 0x1FB4},   {0x1FB6, 0x1FBC},
        {0x1FBE, 0x1FBE},   {0x1FC2, 0x1FC4},   {0x1FC6, 0x1FCC},   {0x1FD0, 0x1FD3},
        {0x1FD6, 0x1FDB},   {0x1FE0, 0x1FEC},   {0x1FF2, 0x1FF4},   {0x1FF6, 0x1FFC},
        {0x2102, 0x2102},   {0x2107, 0x2107},   {0x210A, 0x2113},   {0x2115, 0x2115},
        {0x2119, 0x211D},   {0x2124, 0x2124},   {0x2126, 0x2126},   {0x2128, 0x2128},
        {0x212A, 0x212D},   {0x212F, 0x2134},   {0x2139, 0x2139},   {0x213C, 0x213F},
        {0x2145, 0x2149},   {0x214E, 0x214E},   {0x2183, 0x2184},   {0x2C00, 0x2C7B},
        {0x2C7E, 0x2CE4},   {0x2CEB, 0x2CEE},   {0x2CF2, 0x2CF3},   {0xA640, 0xA66D},
        {0xA680, 0xA69B},   {0xA722, 0xA76F},   {0xA771, 0xA787},   {0xA78B, 0xA78E},
        {0xAB70, 0xABBF},   {0xFB00, 0xFB06},   {0xFB13, 0xFB17},   {0xFF21, 0xFF3A},
        {0xFF41, 0xFF5A},   {0x10400, 0x1044F}, {0x104B0, 0x104D3}, {0x104D8, 0x104FB},
        {0x10C80, 0x10CB2}, {0x10CC0, 0x10CF2}, {0x118A0, 0x118DF}, {0x1E900, 0x1E943},
};
static const struct cp_range DSL_PUNCT[] = {
        {0x21, 0x2F},
        {0x3A, 0x7E},
        {0xFF01, 0xFF0F},
        {0xFF1A, 0xFF5E},
        {0x2018, 0x201F},
        {0x3000, 0x3002},
};
static const struct cp_range DSL_CJK[]  = {{0x800, 0x9FA5}, {0xAC00, 0xD7FF}};
static const struct cp_range DSV3_CJK[] = {{0x3040, 0x30FF}, {0x4E00, 0x9FA5}};

static bool cp_in(size_t n, const struct cp_range r[static n], uint32_t cp) {
    for (size_t i = 0; i < n; i++)
        if (cp >= r[i].lo && cp <= r[i].hi)
            return true;
    return false;
}
#define CP_IN(table, cp) cp_in(sizeof(table) / sizeof((table)[0]), (table), (cp))

/* \s as both regex paths see it: ASCII \t \n \v \f \r and space, and
 * llama.cpp's non-ASCII whitespace (which it rewrites to \v). */
static bool seq_ws(uint32_t cp) {
    return (cp >= 0x09 && cp <= 0x0D) || cp == ' ' || (cp >= 0x80 && (cp == 0x85 || q2_is_ws(cp)));
}

/* The codepoint a wregex sees: non-ASCII whitespace became \v. */
static uint32_t seq_wcp(uint32_t cp) {
    return cp >= 0x80 && seq_ws(cp) ? 0x0B : cp;
}

/* The collapsed classes (\p{L} \p{N} \p{P} \p{M} \p{S}); ASCII by
 * llama.cpp's k_ucat_map, the rest by unicode_class.h. */
static enum unicode_class seq_class(uint32_t cp) {
    if (cp < 0x80) {
        if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z'))
            return UNICODE_CLASS_LETTER;
        if (cp >= '0' && cp <= '9')
            return UNICODE_CLASS_NUMBER;
        if (cp == '$' || cp == '+' || (cp >= '<' && cp <= '>') || cp == '^' || cp == '`' ||
            cp == '|' || cp == '~')
            return UNICODE_CLASS_SYMBOL;
        if (cp >= 0x21 && cp <= 0x7E)
            return UNICODE_CLASS_PUNCT;
        return UNICODE_CLASS_OTHER;
    }
    return seq_ws(cp) ? UNICODE_CLASS_OTHER : unicode_class_of(cp);
}

/* One regex: the end of the match starting at p in [p, end), or p. */
typedef size_t (*seq_match_fn)(const uint32_t *cps, size_t end, size_t p);

static size_t seq_run(const uint32_t *cps, size_t end, size_t j, bool (*in)(uint32_t)) {
    while (j < end && in(cps[j]))
        j++;
    return j;
}

static bool dsl_letter(uint32_t cp) {
    return CP_IN(DSL_LETTERS, seq_wcp(cp));
}
static bool dsl_punct(uint32_t cp) {
    return CP_IN(DSL_PUNCT, seq_wcp(cp));
}
static bool dsl_cjk(uint32_t cp) {
    return CP_IN(DSL_CJK, seq_wcp(cp));
}
static bool dsv3_cjk(uint32_t cp) {
    return CP_IN(DSV3_CJK, seq_wcp(cp));
}
static bool seq_num(uint32_t cp) {
    return seq_class(cp) == UNICODE_CLASS_NUMBER;
}
static bool seq_lm(uint32_t cp) {
    const enum unicode_class c = seq_class(cp);
    return c == UNICODE_CLASS_LETTER || c == UNICODE_CLASS_MARK;
}
static bool seq_ps(uint32_t cp) {
    const enum unicode_class c = seq_class(cp);
    return c == UNICODE_CLASS_PUNCT || c == UNICODE_CLASS_SYMBOL;
}
static bool seq_nl(uint32_t cp) {
    return cp == '\r' || cp == '\n';
}
static bool seq_ascii_alpha(uint32_t cp) {
    return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
}

/* [\r\n] */
static size_t dsl_m1(const uint32_t *cps, size_t end, size_t p) {
    (void) end;
    return seq_nl(cps[p]) ? p + 1 : p;
}

/* \s?[class]+ */
static size_t seq_space_run(const uint32_t *cps, size_t end, size_t p, bool (*in)(uint32_t)) {
    if (seq_ws(cps[p]) && p + 1 < end && in(cps[p + 1]))
        return seq_run(cps, end, p + 1, in);
    return in(cps[p]) ? seq_run(cps, end, p, in) : p;
}
static size_t dsl_m2(const uint32_t *cps, size_t end, size_t p) {
    return seq_space_run(cps, end, p, dsl_letter);
}
static size_t dsl_m3(const uint32_t *cps, size_t end, size_t p) {
    return seq_space_run(cps, end, p, dsl_punct);
}

/* \s+$ */
static size_t dsl_m4(const uint32_t *cps, size_t end, size_t p) {
    const size_t q = seq_run(cps, end, p, seq_ws);
    return q > p && q == end ? q : p;
}

static size_t dsl_m5(const uint32_t *cps, size_t end, size_t p) {
    return seq_run(cps, end, p, dsl_cjk);
}
static size_t dsl_m6(const uint32_t *cps, size_t end, size_t p) {
    return seq_run(cps, end, p, seq_num);
}

/* \p{N}{1,3} */
static size_t dsv3_m1(const uint32_t *cps, size_t end, size_t p) {
    size_t j = p;
    while (j < end && j - p < 3 && seq_num(cps[j]))
        j++;
    return j;
}
static size_t dsv3_m2(const uint32_t *cps, size_t end, size_t p) {
    return seq_run(cps, end, p, dsv3_cjk);
}

static size_t dsv3_m3(const uint32_t *cps, size_t end, size_t p) {
    const uint32_t c0   = cps[p];
    const bool     more = p + 1 < end;
    /* [ASCII punctuation][A-Za-z]+ */
    if (c0 < 0x80 && seq_ps(c0) && more && seq_ascii_alpha(cps[p + 1]))
        return seq_run(cps, end, p + 1, seq_ascii_alpha);
    /* [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+ */
    const enum unicode_class k0 = seq_class(c0);
    const bool prefix = !seq_nl(c0) && k0 != UNICODE_CLASS_LETTER && k0 != UNICODE_CLASS_PUNCT &&
                        k0 != UNICODE_CLASS_SYMBOL;
    if (prefix && more && seq_lm(cps[p + 1]))
        return seq_run(cps, end, p + 1, seq_lm);
    if (seq_lm(c0))
        return seq_run(cps, end, p, seq_lm);
    /*  ?[\p{P}\p{S}]+[\r\n]* */
    size_t j = c0 == ' ' && more && seq_ps(cps[p + 1]) ? p + 1 : p;
    if (seq_ps(cps[j]))
        return seq_run(cps, end, seq_run(cps, end, j, seq_ps), seq_nl);
    if (!seq_ws(c0))
        return p;
    /* \s*[\r\n]+ ends after the run's last \r or \n; \s+(?!\S) leaves the
     * run's last codepoint unless the run ends the piece; \s+ */
    const size_t q    = seq_run(cps, end, p, seq_ws);
    size_t       last = p;
    for (size_t k = p; k < q; k++)
        if (seq_nl(cps[k]))
            last = k + 1;
    if (last > p)
        return last;
    if (q == end || q - p < 2)
        return q;
    return q - 1;
}

static const seq_match_fn SEQ_DEEPSEEK_LLM[] = {dsl_m1, dsl_m2, dsl_m3, dsl_m4, dsl_m5, dsl_m6};
static const seq_match_fn SEQ_DEEPSEEK_V3[]  = {dsv3_m1, dsv3_m2, dsv3_m3};

size_t gguf_tokenizer_seq_split(enum gguf_pre_seq kind,
                                size_t            n,
                                const uint32_t    cps[static n],
                                size_t            pieces[static 2 * n]) {
    const seq_match_fn *fns   = kind == GGUF_PRE_DEEPSEEK_V3 ? SEQ_DEEPSEEK_V3 : SEQ_DEEPSEEK_LLM;
    const size_t        n_fns = kind == GGUF_PRE_DEEPSEEK_V3
                                        ? sizeof SEQ_DEEPSEEK_V3 / sizeof SEQ_DEEPSEEK_V3[0]
                                        : sizeof SEQ_DEEPSEEK_LLM / sizeof SEQ_DEEPSEEK_LLM[0];
    size_t             *cur = pieces, *nxt = pieces + n, n_cur = 1;
    cur[0] = n;
    for (size_t f = 0; f < n_fns; f++) {
        size_t n_nxt = 0, start = 0;
        for (size_t k = 0; k < n_cur; k++) {
            const size_t end = start + cur[k];
            size_t       gap = start;
            for (size_t p = start; p < end;) {
                const size_t m = fns[f](cps, end, p);
                if (m == p) {
                    p++;
                    continue;
                }
                if (p > gap)
                    nxt[n_nxt++] = p - gap;
                nxt[n_nxt++] = m - p;
                gap = p = m;
            }
            if (end > gap)
                nxt[n_nxt++] = end - gap;
            start = end;
        }
        size_t *t = cur;
        cur = nxt, nxt = t, n_cur = n_nxt;
    }
    if (cur != pieces)
        memcpy(pieces, cur, n_cur * sizeof pieces[0]);
    return n_cur;
}

/* Writes the first min(needed, cap) IDs and returns true either way — false
 * only on allocation failure (*n_out then counts the IDs written before it),
 * once gguf_tokenizer_encode has refused modes that cannot encode. A full
 * buffer is therefore ambiguous; the caller resolves it. */
static bool encode_upto(const struct gguf_tokenizer *tok,
                        const struct special_scan   *sc,
                        const char                  *text,
                        int32_t                     *out_ids,
                        size_t                       cap,
                        size_t                      *n_out) {
    *n_out = 0;
    if (tok->mode == GGUF_TOK_MODE_SPM || tok->mode == GGUF_TOK_MODE_UNIGRAM)
        return encode_spm(tok, sc, text, out_ids, cap, n_out);
    if (tok->mode != GGUF_TOK_MODE_GPT2)
        return false;

    const size_t tlen = strlen(text);
    if (tlen == 0)
        return true;

    /* Worst-case chunk size: 6 bytes per byte of text — a malformed byte
     * becomes U+FFFD's three bytes, each mapped to a codepoint that UTF-8
     * writes in two (gpt2_chunk_bytes). */
    size_t chunk_cap;
    if (ckd_mul(&chunk_cap, tlen, (size_t) 6) || ckd_add(&chunk_cap, chunk_cap, (size_t) 16))
        return false;
    GEIST_CLEANUP(heap_freep) char *chunk_buf = heap_alloc_array_aligned(char, chunk_cap);
    if (chunk_buf == nullptr)
        return false;
    /* Regex-sequence pretokenizers (#242): a fragment's codepoints, their
     * byte offsets (plus the end) and the piece lengths with their scratch. */
    const bool                          seq = tok->pre_seq != GGUF_PRE_NONE;
    GEIST_CLEANUP(heap_freep) uint32_t *seq_cps =
            seq ? heap_alloc_array_aligned(uint32_t, tlen) : nullptr;
    GEIST_CLEANUP(heap_freep) size_t *seq_off =
            seq ? heap_alloc_array_aligned(size_t, tlen + 1) : nullptr;
    GEIST_CLEANUP(heap_freep) size_t *seq_pieces =
            seq ? heap_alloc_array_aligned(size_t, 2 * tlen) : nullptr;
    if (seq && (seq_cps == nullptr || seq_off == nullptr || seq_pieces == nullptr))
        return false;

    size_t i = 0;
    /* Scanner paths: start of the next special after i; none starts before
     * it, so the match below is skipped until i reaches it (#403). */
    size_t next_special = 0;
    while (i < tlen) {
        /* Leftmost-longest special-token match at current position. */
        int32_t      sp_id = -1;
        const size_t sp_len =
                *n_out < cap && i >= next_special ? special_at(sc, tlen, i, text, &sp_id) : 0;
        if (sp_len != 0) {
            out_ids[(*n_out)++] = sp_id;
            i += sp_len;
            continue;
        }

        if (tok->pre_seq != GGUF_PRE_NONE) {
            /* The fragment up to the next special token, split as a whole
             * (llama.cpp splits on specials first). */
            size_t frag_end = tlen;
            for (size_t p = i + 1; p < tlen && frag_end == tlen; p++)
                if (special_at(sc, tlen, p, text, &sp_id) != 0)
                    frag_end = p;
            size_t n_cp = 0;
            for (size_t b = i; b < frag_end; n_cp++) {
                size_t adv;
                seq_cps[n_cp] = q2_cp_at(text, frag_end, b, &adv);
                seq_off[n_cp] = b;
                b += adv;
            }
            seq_off[n_cp] = frag_end;
            const size_t n_pieces =
                    gguf_tokenizer_seq_split(tok->pre_seq, n_cp, seq_cps, seq_pieces);
            size_t at = 0;
            for (size_t k = 0; k < n_pieces && *n_out < cap; k++) {
                const size_t cb_used =
                        gpt2_chunk_bytes(seq_off[at + seq_pieces[k]], seq_off[at], text, chunk_buf);
                at += seq_pieces[k];
                size_t produced = bpe_chunk_to_ids(
                        tok, chunk_buf, cb_used, out_ids + *n_out, cap - *n_out, false);
                if (produced == SIZE_MAX)
                    return false;
                *n_out += produced;
            }
            i = frag_end;
            if (*n_out >= cap)
                break;
            continue;
        }

        if (tok->pre_gpt4o || tok->pre_digits != 0 || tok->pre_gpt2) {
            /* gpt-4o, llama3 / qwen2 and GPT-2 chunks run on the text up to the
             * next special token, as llama.cpp splits on specials first and
             * runs the regex on each fragment (a lookahead at the
             * fragment's end sees the end; "es?<|im_end|>" cannot swallow
             * the marker). next_special stays valid until i passes it. */
            if (next_special <= i) {
                next_special = tlen;
                for (size_t p = i + 1; p < tlen && next_special == tlen; p++)
                    if (special_at(sc, tlen, p, text, &sp_id) != 0)
                        next_special = p;
            }
            const size_t end =
                    tok->pre_gpt4o ? gguf_tokenizer_gpt4o_chunk_end(next_special, i, text)
                    : tok->pre_gpt2
                            ? gguf_tokenizer_gpt2_chunk_end(
                                      next_special, i, tok->pre_split_digits, text)
                            : gguf_tokenizer_llama3_chunk_end(
                                      next_special, i, tok->pre_digits, tok->pre_marks, text);
            const size_t cb_used  = gpt2_chunk_bytes(end, i, text, chunk_buf);
            size_t       produced = bpe_chunk_to_ids(
                    tok, chunk_buf, cb_used, out_ids + *n_out, cap - *n_out, false);
            if (produced == SIZE_MAX)
                return false;
            *n_out += produced;
            i = end;
            if (*n_out >= cap)
                break;
            continue;
        }

        size_t start = i;

        /* Detect chunk kind from first non-space byte. */
        while (i < tlen && is_ascii_space((unsigned char) text[i]))
            i++;

        /* A special token directly after the whitespace run: emit the
         * run as its own chunk and let the outer loop match the special
         * — otherwise its first byte gets shredded into the next punct
         * chunk ("\n<|im_start|>" would become "\n<" + "|" + ..., #275). */
        if (i > start && i < tlen) {
            if (special_at(sc, tlen, i, text, &sp_id) != 0) {
                const size_t cb_used  = gpt2_chunk_bytes(i, start, text, chunk_buf);
                size_t       produced = bpe_chunk_to_ids(
                        tok, chunk_buf, cb_used, out_ids + *n_out, cap - *n_out, false);
                if (produced == SIZE_MAX)
                    return false;
                *n_out += produced;
                if (*n_out >= cap)
                    break;
                continue; /* outer loop matches the special at i */
            }
        }
        /* Now i is the first non-space (or tlen). The leading-space
         * run (start..i) is appended as Ġ-prefix on the following
         * word, GPT-2 style. */
        const size_t lead_ws_start = start, lead_ws_end = i;
        if (i >= tlen) {
            /* Trailing whitespace becomes its own chunk. */
            if (lead_ws_end > lead_ws_start) {
                const size_t cb_used =
                        gpt2_chunk_bytes(lead_ws_end, lead_ws_start, text, chunk_buf);
                size_t produced = bpe_chunk_to_ids(
                        tok, chunk_buf, cb_used, out_ids + *n_out, cap - *n_out, false);
                if (produced == SIZE_MAX)
                    return false;
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
            /* Punctuation / symbol: one byte per chunk. */
            i++;
        }

        /* Emit chunk: leading whitespace + run (all bytes get
         * byte→codepoint→UTF-8 mapping). */
        const size_t cb_used = gpt2_chunk_bytes(i, lead_ws_start, text, chunk_buf);
        size_t       produced =
                bpe_chunk_to_ids(tok, chunk_buf, cb_used, out_ids + *n_out, cap - *n_out, false);
        if (produced == SIZE_MAX)
            return false;
        *n_out += produced;
        if (*n_out >= cap)
            break;
        (void) run_start;
    }
    return true;
}

[[nodiscard]] enum geist_status gguf_tokenizer_encode(const struct gguf_tokenizer *tok,
                                                      const char                  *text,
                                                      size_t                       cap,
                                                      int32_t                     *out_ids,
                                                      size_t                      *n_out) {
    return gguf_tokenizer_encode_plain(tok, text, 0, nullptr, cap, out_ids, n_out);
}

[[nodiscard]] enum geist_status gguf_tokenizer_encode_plain(const struct gguf_tokenizer   *tok,
                                                            const char                    *text,
                                                            size_t                         n_plain,
                                                            const struct geist_text_range *plain,
                                                            size_t                         cap,
                                                            int32_t                       *out_ids,
                                                            size_t                        *n_out) {
    if (n_out == nullptr)
        return GEIST_E_INVALID_ARG;
    *n_out = 0;
    /* One spare slot: encode_upto fills whatever room it gets, so only a
     * filled spare proves the text needs more than `cap` IDs. */
    size_t room;
    if (tok == nullptr || text == nullptr || (cap > 0 && out_ids == nullptr) ||
        (n_plain > 0 && plain == nullptr) || ckd_add(&room, cap, 1))
        return GEIST_E_INVALID_ARG;
    /* Ascending, disjoint, within the text; covered: some range is not empty. */
    const size_t tlen    = n_plain > 0 ? strlen(text) : 0;
    bool         covered = false;
    for (size_t k = 0; k < n_plain; k++) {
        if (plain[k].begin > plain[k].end || plain[k].end > tlen ||
            (k > 0 && plain[k].begin < plain[k - 1].end))
            return GEIST_E_INVALID_ARG;
        covered = covered || plain[k].end > plain[k].begin;
    }
    if (tok->mode == GGUF_TOK_MODE_UNSUPPORTED)
        return GEIST_E_UNSUPPORTED;
    GEIST_CLEANUP(heap_freep) uint8_t *mask =
            covered ? heap_calloc_array_aligned(uint8_t, tlen) : nullptr;
    if (covered && mask == nullptr)
        return GEIST_E_OOM;
    for (size_t k = 0; mask != nullptr && k < n_plain; k++)
        memset(mask + plain[k].begin, 1, plain[k].end - plain[k].begin);
    const struct special_scan sc = {
            .all    = tok->specials,
            .n_all  = tok->n_specials,
            .user   = tok->specials != nullptr ? tok->specials + tok->n_specials : nullptr,
            .n_user = tok->n_user_specials,
            .first  = tok->special_first,
            .plain  = mask};
    int32_t *ids = heap_alloc_array_aligned(int32_t, room);
    if (ids == nullptr)
        return GEIST_E_OOM;
    size_t            n  = 0;
    enum geist_status st = encode_upto(tok, &sc, text, ids, room, &n) ? GEIST_OK : GEIST_E_OOM;
    if (st == GEIST_OK && n > cap)
        st = GEIST_E_INVALID_ARG;
    if (st == GEIST_OK && n > 0) {
        memcpy(out_ids, ids, n * sizeof ids[0]);
        *n_out = n;
    }
    void *p = ids;
    safe_free(&p);
    return st;
}

/* Dump everything a model's tokenizer answers through the public API, for a
 * diff between two revisions (tools/tokenizer_identity.py).
 *
 * Usage: dump_geist_tokens <model.gguf> <corpus.hex> [--no-prefill] [--ids N]
 *
 * corpus.hex holds one input per line, hex-encoded (so newlines, \r and
 * invalid UTF-8 survive). Per input: geist_session_tokenize's ids and status,
 * the status of a tokenize one slot short, and — unless --no-prefill —
 * geist_session_set_prompt's status, the session length after it and an
 * FNV-1a hash of the logits it left (ids that differ change both). Then, once
 * per model: bos/eos/add_bos/add_eos, geist_model_token_by_text for every
 * input and a few fixed strings, geist_session_token_to_str for ids -1 ..
 * N + 1 (N = --ids, 300000 by default), and whether a second load of the same
 * file is accepted as a speculative draft. A tokenizer.bin is found the way
 * the engine finds it (GEIST_TOKENIZER_PATH, then beside the GGUF).
 * GEIST_DUMP_BACKEND=<name> selects the backend ("cpu_scalar" by default).
 */
#include "tool_common.h"

#include <stdint.h>
#include <string.h>

static void put_hex(const char *s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        printf("%02x", (unsigned char) s[i]);
    }
}

static uint64_t fnv1a(const void *p, size_t n) {
    const unsigned char *b = p;
    uint64_t             h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) {
        h = (h ^ b[i]) * 1099511628211ULL;
    }
    return h;
}

static int hexval(int c) {
    if (c >= '0' && c <= '9')
        return c - '0';
    if (c >= 'a' && c <= 'f')
        return c - 'a' + 10;
    if (c >= 'A' && c <= 'F')
        return c - 'A' + 10;
    return -1;
}

/* Decodes the hex line [p, end) in place into a NUL-terminated string. */
static char *unhex(char *p, const char *end) {
    char *w = p;
    for (const char *r = p; r + 1 < end; r += 2) {
        const int hi = hexval(r[0]), lo = hexval(r[1]);
        if (hi < 0 || lo < 0)
            break;
        *w++ = (char) (hi << 4 | lo);
    }
    *w = '\0';
    return p;
}

static void dump_entry(struct geist_session *s, size_t idx, const char *text, bool prefill) {
    printf("entry %zu ", idx);
    put_hex(text, strlen(text));
    printf("\n");
    const size_t   cap = strlen(text) * 3 + 16;
    geist_token_t *ids = malloc(cap * sizeof *ids);
    size_t         n   = 0;
    if (ids == nullptr) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    enum geist_status st = geist_session_tokenize(s, text, cap, ids, &n);
    printf("  tokenize=%d n=%zu:", (int) st, n);
    for (size_t i = 0; i < n; i++) {
        printf(" %d", ids[i]);
    }
    printf("\n");
    if (st == GEIST_OK && n > 0) {
        size_t n2 = 99;
        st        = geist_session_tokenize(s, text, n - 1, ids, &n2);
        printf("  short=%d n=%zu\n", (int) st, n2);
    }
    free(ids);
    if (prefill) {
        (void) geist_session_reset(s);
        st                   = geist_session_set_prompt(s, text);
        size_t       n_vocab = 0;
        const float *logits  = st == GEIST_OK ? geist_session_peek_logits(&n_vocab, s) : nullptr;
        printf("  set_prompt=%d len=%zu logits=%016llx\n",
               (int) st,
               geist_session_length(s),
               logits != nullptr ? (unsigned long long) fnv1a(logits, n_vocab * sizeof *logits)
                                 : 0ULL);
        (void) geist_session_reset(s);
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <corpus.hex> [--no-prefill] [--ids N]\n", argv[0]);
        return 2;
    }
    bool prefill = true;
    long max_id  = 300000;
    for (int a = 3; a < argc; a++) {
        if (strcmp(argv[a], "--no-prefill") == 0) {
            prefill = false;
        } else if (strcmp(argv[a], "--ids") == 0 && a + 1 < argc) {
            max_id = strtol(argv[++a], nullptr, 10);
        } else {
            fprintf(stderr, "unknown argument %s\n", argv[a]);
            return 2;
        }
    }
    size_t corpus_n = 0;
    char  *corpus   = tool_read_file(argv[2], &corpus_n);
    if (corpus == nullptr) {
        fprintf(stderr, "cannot read %s\n", argv[2]);
        return 2;
    }
    const char *bn = getenv("GEIST_DUMP_BACKEND");
    if (bn == nullptr || bn[0] == '\0')
        bn = "cpu_scalar";
    struct tool_engine e = {0};
    if (!tool_open(bn, nullptr, argv[1], nullptr, nullptr, &e)) {
        tool_close(&e);
        free(corpus);
        return 1;
    }
    struct geist_model *m = e.model;
    printf("bos=%d eos=%d add_bos=%d add_eos=%d\n",
           geist_model_bos_token(m),
           geist_model_eos_token(m),
           geist_model_add_bos(m),
           geist_model_add_eos(m));

    static const char *const FIXED[] = {
            "",
            " ",
            "a",
            "\xe2\x96\x81",
            "<bos>",
            "<eos>",
            "<unk>",
            "<pad>",
            "<mask>",
            "<0x0A>",
            "<start_of_turn>",
            "<end_of_turn>",
            "<|image|>",
            "<|eot|>",
            "<|im_end|>",
            "Hello",
            "\xe2\x96\x81the",
            "<|begin_of_text|>",
            "<s>",
            "</s>",
            "<|endoftext|>",
    };
    for (size_t i = 0; i < sizeof FIXED / sizeof FIXED[0]; i++) {
        printf("by_text ");
        put_hex(FIXED[i], strlen(FIXED[i]));
        printf(" %d\n", geist_model_token_by_text(m, FIXED[i]));
    }

    dump_entry(e.sess, 0, "", prefill); /* the empty input, which a line cannot hold */
    size_t idx = 1;
    for (char *p = corpus, *end = corpus + corpus_n; p < end;) {
        char *nl = memchr(p, '\n', (size_t) (end - p));
        char *le = nl != nullptr ? nl : end;
        if (le > p) {
            const char *text = unhex(p, le);
            dump_entry(e.sess, idx, text, prefill);
            printf("  by_text=%d\n", geist_model_token_by_text(m, text));
            idx++;
        }
        p = le + 1;
    }

    for (long id = -1; id <= max_id + 1; id++) {
        const char *str = geist_session_token_to_str(e.sess, (geist_token_t) id);
        printf("str %ld ", id);
        if (str == nullptr) {
            printf("NULL\n");
        } else {
            put_hex(str, strlen(str));
            printf("\n");
        }
    }

    /* A second load of the same file as a draft: accepted iff the engine
     * compares the vocabularies. */
    struct geist_model   *m2 = nullptr;
    struct geist_session *s2 = nullptr;
    if (geist_model_load(argv[1], e.be, &m2) == GEIST_OK &&
        geist_session_create(m2, e.be, nullptr, &s2) == GEIST_OK) {
        printf("set_draft=%d\n", (int) geist_session_set_draft(e.sess, s2));
        (void) geist_session_set_draft(e.sess, nullptr);
    } else {
        printf("set_draft=load-failed\n");
    }
    geist_session_destroy(s2);
    geist_model_destroy(m2);
    tool_close(&e);
    free(corpus);
    return 0;
}

/*
 * eval_geist — REPL for benchmark evaluation against the geist v2 API.
 *
 * Loads the model ONCE and processes commands from stdin until EOF.
 * Each command is one line, results are one line on stdout.
 *
 * Commands:
 *   DECODE <n_prompt> <id0> <id1> ... <id_{n-1}> <n_decode>
 *     -> "OK <n> <out0> <out1> ... <out_{n-1}>"  (greedy decoded tokens)
 *
 *   SCORE <n_prompt> <id0> ... <id_{n-1}> <n_cont> <c0> ... <c_{m-1}>
 *     -> "OK <total_logprob> <lp0> <lp1> ... <lp_{m-1}>"
 *     where lp_i = log p(c_i | prompt + c_0..c_{i-1})
 *     Used for multiple-choice: score each option, pick max.
 *
 *   SCOREALT <n_prompt> <id0> ... <id_{n-1}> <n_alt> <a0> ... <a_{n-1}>
 *     -> "OK <lp0> <lp1> ... <lp_{n-1}>"
 *     Same prompt, K single-token candidates. Single prefill amortized.
 *
 *   TOK <text> -> "OK <n> <id0> ... <id_{n-1}>"
 *     Tokenize text with the model's own GGUF tokenizer (\n/\t/\\ escapes).
 *     Lets eval harnesses stay self-contained — no external HF tokenizer.
 *
 *   GEN <max_new> <text> -> "OK <generated text>"
 *     Greedy-generate from the prompt and return the decoded surface text
 *     (\n/\t/\\ escaped). Stops at max_new or an end-of-turn / EOS marker.
 *     Used by the MMLU (TOK/SCOREALT) and tool-calling/JSON (GEN) benchmarks.
 *
 *   RESET -> "OK"   (clears KV cache)
 *   QUIT  -> exit
 *
 * Tokens are IDs (int32). The MMLU harness (tools/eval_mmlu.py) uses TOK/GEN so
 * no external HF tokenizer is needed; tools/eval_runner.py optionally uses an HF
 * tokenizer for reference parity.
 *
 * Usage:
 *   eval_geist <gguf> [--awq <scales.bin>]
 */
#include <geist.h>
#include <geist_util.h>
#include <geist_backend.h>

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Read whitespace-separated int32 list of length n into out. */
static int read_ids(int n, geist_token_t *out, char **cursor) {
    for (int i = 0; i < n; i++) {
        char *end;
        long  v = strtol(*cursor, &end, 10);
        if (end == *cursor)
            return -1;
        out[i]  = (geist_token_t) v;
        *cursor = end;
    }
    return 0;
}

/* log_softmax then return log_p[token]. */
static double logprob_at(const float *logits, size_t n_logits, geist_token_t token) {
    float maxv = logits[0];
    for (size_t i = 1; i < n_logits; i++)
        if (logits[i] > maxv)
            maxv = logits[i];
    double sum = 0.0;
    for (size_t i = 0; i < n_logits; i++)
        sum += exp((double) (logits[i] - maxv));
    return (double) (logits[token] - maxv) - log(sum);
}

static int cmd_decode(struct geist_session *s, char *args) {
    char *p = args;
    char *end;
    long  n_prompt = strtol(p, &end, 10);
    if (end == p || n_prompt <= 0) {
        puts("ERR bad n_prompt");
        return -1;
    }
    p = end;

    geist_token_t *ids = (geist_token_t *) malloc((size_t) n_prompt * sizeof(*ids));
    if (read_ids((int) n_prompt, ids, &p) < 0) {
        free(ids);
        puts("ERR bad ids");
        return -1;
    }

    long n_decode = strtol(p, &end, 10);
    if (end == p || n_decode < 0) {
        free(ids);
        puts("ERR bad n_decode");
        return -1;
    }

    if (geist_session_reset(s) != GEIST_OK ||
        geist_session_prefill_tokens(s, (size_t) n_prompt, ids) != GEIST_OK) {
        free(ids);
        puts("ERR prefill failed");
        return -1;
    }
    free(ids);

    geist_token_t *out = (geist_token_t *) malloc((size_t) n_decode * sizeof(*out));
    for (long i = 0; i < n_decode; i++) {
        if (geist_session_decode_step(s, &out[i]) != GEIST_OK)
            out[i] = -1;
    }

    fputs("OK ", stdout);
    printf("%ld", n_decode);
    for (long i = 0; i < n_decode; i++)
        printf(" %d", (int) out[i]);
    putchar('\n');
    fflush(stdout);
    free(out);
    return 0;
}

/* SCOREALT — prefill prompt once, return log_p for each of K candidate
 * single-token alternatives. Avoids re-prefilling the prompt N times for
 * multi-choice scoring (MMLU). */
static int cmd_scorealt(struct geist_session *s, char *args) {
    char *p = args;
    char *end;
    long  n_prompt = strtol(p, &end, 10);
    if (end == p || n_prompt <= 0) {
        puts("ERR bad n_prompt");
        return -1;
    }
    p = end;

    geist_token_t *ids = (geist_token_t *) malloc((size_t) n_prompt * sizeof(*ids));
    if (read_ids((int) n_prompt, ids, &p) < 0) {
        free(ids);
        puts("ERR bad ids");
        return -1;
    }

    long n_alt = strtol(p, &end, 10);
    if (end == p || n_alt <= 0) {
        free(ids);
        puts("ERR bad n_alt");
        return -1;
    }
    p                   = end;
    geist_token_t *alts = (geist_token_t *) malloc((size_t) n_alt * sizeof(*alts));
    if (read_ids((int) n_alt, alts, &p) < 0) {
        free(ids);
        free(alts);
        puts("ERR bad alts");
        return -1;
    }

    if (geist_session_reset(s) != GEIST_OK ||
        geist_session_prefill_tokens(s, (size_t) n_prompt, ids) != GEIST_OK) {
        free(ids);
        free(alts);
        puts("ERR prefill failed");
        return -1;
    }
    free(ids);

    size_t       n_logits = 0;
    const float *logits   = geist_session_peek_logits(s, &n_logits);
    if (logits == nullptr || n_logits == 0) {
        free(alts);
        puts("ERR no logits");
        return -1;
    }

    fputs("OK", stdout);
    for (long i = 0; i < n_alt; i++)
        printf(" %.6f", logprob_at(logits, n_logits, alts[i]));
    putchar('\n');
    fflush(stdout);
    free(alts);
    return 0;
}

static int cmd_score(struct geist_session *s, char *args) {
    char *p = args;
    char *end;
    long  n_prompt = strtol(p, &end, 10);
    if (end == p || n_prompt <= 0) {
        puts("ERR bad n_prompt");
        return -1;
    }
    p = end;

    geist_token_t *ids = (geist_token_t *) malloc((size_t) n_prompt * sizeof(*ids));
    if (read_ids((int) n_prompt, ids, &p) < 0) {
        free(ids);
        puts("ERR bad ids");
        return -1;
    }

    long n_cont = strtol(p, &end, 10);
    if (end == p || n_cont <= 0) {
        free(ids);
        puts("ERR bad n_cont");
        return -1;
    }
    p                   = end;
    geist_token_t *cont = (geist_token_t *) malloc((size_t) n_cont * sizeof(*cont));
    if (read_ids((int) n_cont, cont, &p) < 0) {
        free(ids);
        free(cont);
        puts("ERR bad cont");
        return -1;
    }

    if (geist_session_reset(s) != GEIST_OK ||
        geist_session_prefill_tokens(s, (size_t) n_prompt, ids) != GEIST_OK) {
        free(ids);
        free(cont);
        puts("ERR prefill failed");
        return -1;
    }
    free(ids);

    double  total_lp = 0.0;
    double *per_tok  = (double *) malloc((size_t) n_cont * sizeof(*per_tok));
    for (long i = 0; i < n_cont; i++) {
        size_t       n_logits = 0;
        const float *logits   = geist_session_peek_logits(s, &n_logits);
        if (logits == nullptr) {
            free(cont);
            free(per_tok);
            puts("ERR no logits");
            return -1;
        }
        per_tok[i] = logprob_at(logits, n_logits, cont[i]);
        total_lp += per_tok[i];
        /* Advance KV by feeding the actual continuation token, not the argmax. */
        if (geist_session_prefill_tokens(s, 1, &cont[i]) != GEIST_OK) {
            free(cont);
            free(per_tok);
            puts("ERR prefill 1 failed");
            return -1;
        }
    }

    fputs("OK ", stdout);
    printf("%.6f", total_lp);
    for (long i = 0; i < n_cont; i++)
        printf(" %.6f", per_tok[i]);
    putchar('\n');
    fflush(stdout);
    free(cont);
    free(per_tok);
    return 0;
}

/* The REPL is line-oriented, but prompts are multi-line; decode \n / \t / \\
 * escapes so a whole prompt fits on one line. Returns the NUL-terminated
 * result length (clamped to cap-1). */
static size_t unescape(const char *src, char *dst, size_t cap) {
    size_t w = 0;
    for (const char *p = src; *p && w + 1 < cap; p++) {
        if (*p == '\\' && p[1]) {
            p++;
            dst[w++] = (*p == 'n') ? '\n' : (*p == 't') ? '\t' : *p;
        } else {
            dst[w++] = *p;
        }
    }
    dst[w] = '\0';
    return w;
}

/* Append `piece` to dst (size cap, *w the cursor), escaping \n and \\ so the
 * generated text rides back on one REPL line. */
static void append_escaped(const char *piece, char *dst, size_t *w, size_t cap) {
    for (const char *c = piece; *c && *w + 2 < cap; c++) {
        if (*c == '\n') {
            dst[(*w)++] = '\\';
            dst[(*w)++] = 'n';
        } else if (*c == '\\') {
            dst[(*w)++] = '\\';
            dst[(*w)++] = '\\';
        } else {
            dst[(*w)++] = *c;
        }
    }
}

/* TOK <text> -> "OK <n> <id0> ... <id_{n-1}>"
 * Tokenize text with the model's own GGUF tokenizer. Lets evaluation
 * harnesses (e.g. MMLU) stay self-contained — no external HF tokenizer, and no
 * tokenizer-mismatch risk between scoring and the model's vocabulary. */
static int cmd_tok(struct geist_session *s, const char *text) {
    enum { TOK_CAP = 8192 };
    static geist_token_t out[TOK_CAP];
    static char          unesc[1 << 16];
    unescape(text, unesc, sizeof unesc);
    size_t            n  = 0;
    enum geist_status st = geist_session_tokenize(s, unesc, TOK_CAP, out, &n);
    if (st != GEIST_OK) {
        printf("ERR tokenize %s\n", geist_status_to_string(st));
        fflush(stdout);
        return -1;
    }
    printf("OK %zu", n);
    for (size_t i = 0; i < n; i++)
        printf(" %d", (int) out[i]);
    putchar('\n');
    fflush(stdout);
    return 0;
}

/* GEN <max_new> <text> -> "OK <escaped generated text>"
 * Greedy-generate from the prompt and return the decoded surface text. Stops at
 * max_new tokens or an end-of-turn / EOS marker. Used by the tool-calling / JSON
 * generation benchmark; unlike the simple_generate demo it does NOT stop at
 * arbitrary bracketed tokens, since tool calls legitimately contain specials
 * such as <|tool_call>. */
static int cmd_gen(struct geist_session *s, char *args) {
    char *p       = args;
    long  max_new = strtol(p, &p, 10);
    if (p == args || max_new <= 0 || max_new > 8192) {
        puts("ERR bad max_new");
        fflush(stdout);
        return -1;
    }
    while (*p == ' ')
        p++;

    enum { TOK_CAP = 8192 };
    static char prompt[1 << 16];
    unescape(p, prompt, sizeof prompt);

    static geist_token_t ids[TOK_CAP];
    size_t               n = 0;
    if (geist_session_tokenize(s, prompt, TOK_CAP, ids, &n) != GEIST_OK ||
        geist_session_reset(s) != GEIST_OK || geist_session_prefill_tokens(s, n, ids) != GEIST_OK) {
        puts("ERR gen prefill");
        fflush(stdout);
        return -1;
    }

    static char out[1 << 16];
    size_t      w = 0;
    for (long i = 0; i < max_new; i++) {
        geist_token_t tok = 0;
        if (geist_session_decode_step(s, &tok) != GEIST_OK)
            break;
        const char *piece = geist_session_token_to_str(s, tok);
        if (piece == nullptr)
            break; /* control token with no surface form */
        if (strcmp(piece, "<eos>") == 0 || strcmp(piece, "<turn|>") == 0 ||
            strcmp(piece, "<end_of_turn>") == 0)
            break; /* end of the assistant turn */
        append_escaped(piece, out, &w, sizeof out);
    }
    out[w] = '\0';
    printf("OK %s\n", out);
    fflush(stdout);
    return 0;
}

int main(int argc, char **argv) {
    const char *gguf_path = nullptr;
    const char *awq_path  = nullptr;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--awq") == 0 && i + 1 < argc) {
            awq_path = argv[++i];
        } else if (gguf_path == nullptr) {
            gguf_path = argv[i];
        }
    }
    if (gguf_path == nullptr) {
        fprintf(stderr, "Usage: %s <gguf> [--awq <scales.bin>]\n", argv[0]);
        return 2;
    }

    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create("auto", nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        fprintf(stderr, "backend create failed: %s\n", geist_last_create_error());
        return 1;
    }

    struct geist_model *model = nullptr;
    s                         = geist_model_load(gguf_path, be, &model);
    if (s != GEIST_OK) {
        fprintf(stderr, "geist_model_load failed: %s\n", geist_last_create_error());
        geist_backend_destroy(be);
        return 1;
    }

    struct geist_session_opts opts = {
            .temperature     = 0.0f, /* greedy */
            .awq_scales_path = awq_path,
    };
    struct geist_session *sess = nullptr;
    s                          = geist_session_create(model, be, &opts, &sess);
    if (s != GEIST_OK) {
        fprintf(stderr, "geist_session_create failed\n");
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return 1;
    }

    fprintf(stderr, "READY\n");
    fflush(stderr);
    /* Also signal ready on stdout so subprocess wrapper can sync. */
    puts("READY");
    fflush(stdout);

    char line[1 << 20]; /* 1 MB buffer for very long prompts */
    while (fgets(line, sizeof(line), stdin)) {
        size_t n = strlen(line);
        while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r'))
            line[--n] = '\0';
        if (n == 0)
            continue;

        char *cmd = line;
        while (*cmd && isspace((unsigned char) *cmd))
            cmd++;
        char *args = cmd;
        while (*args && !isspace((unsigned char) *args))
            args++;
        if (*args) {
            *args = '\0';
            args++;
        }

        if (strcmp(cmd, "DECODE") == 0)
            cmd_decode(sess, args);
        else if (strcmp(cmd, "SCORE") == 0)
            cmd_score(sess, args);
        else if (strcmp(cmd, "SCOREALT") == 0)
            cmd_scorealt(sess, args);
        else if (strcmp(cmd, "TOK") == 0)
            cmd_tok(sess, args);
        else if (strcmp(cmd, "GEN") == 0)
            cmd_gen(sess, args);
        else if (strcmp(cmd, "RESET") == 0) {
            (void) geist_session_reset(sess);
            puts("OK");
            fflush(stdout);
        } else if (strcmp(cmd, "QUIT") == 0)
            break;
        else {
            printf("ERR unknown cmd '%s'\n", cmd);
            fflush(stdout);
        }
    }

    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return 0;
}

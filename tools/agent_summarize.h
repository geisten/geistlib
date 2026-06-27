/*
 * agent_summarize.h — a summarize_file geist_tool: read a text file and return a
 * summary. Unlike a dumb reader, this tool OWNS the summarization: a deterministic
 * C tool can't summarize, so it runs the model itself over a private sub-session
 * (separate from the agent's, so the agent loop's state is untouched). The agent
 * turn then just relays the finished summary.
 *
 * Pipeline (decided via /grill-me):
 *   1. confine — reject absolute / parent-escaping paths; read under `root`.
 *   2. read    — up to SUMM_RAW_CAP bytes (bigger files are summarized truncated).
 *   3. prep    — per-extension dispatch (extensible): .html/.htm -> strip tags;
 *                everything else -> raw. Add a row to SUMM_PREPROC for new types.
 *   4. chunk   — ~SUMM_CHUNK bytes on paragraph/line boundaries (hard-split only a
 *                pathological long line).
 *   5. refine  — rolling summary: running = summarize(chunk0); then
 *                running = summarize(running + chunk_i). Linear, bounded context,
 *                no map-reduce overflow / recursion. Each step is greedy, capped
 *                at SUMM_GEN_TOKENS, template-framed, and stopped by the anti-loop
 *                cap (agent_tail_loop) — so summaries stay short and don't ramble.
 *
 * ponytail: SUMM_RAW_CAP bounds the read (no streaming yet — that's the upgrade
 *   for arbitrarily large files); static work buffers (single-threaded agent).
 */
#ifndef GEIST_AGENT_SUMMARIZE_H
#define GEIST_AGENT_SUMMARIZE_H

#include <geist.h>
#include <geist_util.h>

#include "agent.h"
#include "agent_webfetch.h" /* webfetch_strip_html */

#include <stdio.h>
#include <string.h>

enum {
    SUMM_RAW_CAP    = 1 << 18, /* 256 KB read/preprocess cap */
    SUMM_CHUNK      = 2048,    /* target bytes per chunk */
    SUMM_RUN_CAP    = 2048,    /* running-summary buffer */
    SUMM_GEN_TOKENS = 256,     /* max tokens per refine step */
    SUMM_PROMPT_CAP = SUMM_CHUNK + SUMM_RUN_CAP + 512,
};

/* ---- extensible per-extension preprocessing ---------------------------- */
typedef size_t (*summ_preproc)(size_t n, const char in[static n], size_t cap, char out[static cap]);

static inline size_t
summ_pp_html(size_t n, const char in[static n], size_t cap, char out[static cap]) {
    return webfetch_strip_html(n, in, cap, out);
}

static const struct {
    const char  *ext;
    summ_preproc fn;
} SUMM_PREPROC[] = {
    {".html", summ_pp_html},
    {".htm", summ_pp_html},
    /* add more types here: {".pdf", summ_pp_pdf}, {".json", summ_pp_json}, … */
};

/* Reject absolute paths and any "../" escape; the tool reads only under root. */
static inline int summ_path_ok(const char *path) {
    return path[0] != '/' && strstr(path, "..") == nullptr;
}

/* End offset of the next chunk: ~target bytes from start, backed up to the last
 * paragraph (blank line) or line boundary; a single overlong line hard-splits. */
static inline size_t summ_next_chunk(const char *t, size_t len, size_t start, size_t target) {
    if (start + target >= len) {
        return len;
    }
    size_t lim = start + target;
    for (size_t i = lim; i > start; i--) {
        if (t[i] == '\n' && i + 1 < len && t[i + 1] == '\n') {
            return i + 2; /* paragraph boundary — after the blank line */
        }
    }
    for (size_t i = lim; i > start; i--) {
        if (t[i] == '\n') {
            return i + 1; /* line boundary */
        }
    }
    return lim; /* hard split */
}

/* One greedy, bounded summary on the sub-session, framed with the model's chat
 * template. Stops on EOS / the template stop / a degeneration loop. */
static inline size_t summ_generate(struct geist_session             *s,
                                   const struct geist_chat_template *t,
                                   geist_token_t                     eos,
                                   geist_token_t                     eot,
                                   const char                       *prompt,
                                   size_t                            cap,
                                   char                              out[static cap]) {
    static char framed[SUMM_PROMPT_CAP + 128];
    snprintf(framed, sizeof framed, "%s%s%s%s", t->user_open, prompt, t->turn_close, t->model_open);
    if (geist_session_reset(s) != GEIST_OK || geist_session_set_prompt(s, framed) != GEIST_OK) {
        out[0] = '\0';
        return 0;
    }
    size_t w = 0;
    for (int i = 0; i < SUMM_GEN_TOKENS; i++) {
        geist_token_t tok = 0;
        if (geist_session_decode_step(s, &tok) != GEIST_OK) {
            break;
        }
        if (tok == eos || (eot != GEIST_TOKEN_NONE && tok == eot)) {
            break;
        }
        const char *p  = geist_session_token_to_str(s, tok);
        size_t      pl = p ? strlen(p) : 0;
        if (pl == 0 || (pl >= 2 && p[0] == '<' && p[pl - 1] == '>')) {
            break; /* control marker */
        }
        if (w + pl + 1 >= cap) {
            break;
        }
        memcpy(out + w, p, pl);
        w += pl;
        size_t lp = agent_tail_loop(out, w);
        if (lp > 0) {
            w -= 2 * lp; /* drop the degenerate run, keep one copy */
            break;
        }
    }
    out[w] = '\0';
    for (size_t m = 0; t->leak[m] != nullptr; m++) { /* cut any leaked turn marker */
        char *h = strstr(out, t->leak[m]);
        if (h != nullptr) {
            w      = (size_t) (h - out);
            out[w] = '\0';
        }
    }
    return w;
}

/* A refine step must never replace a good running summary with worse output: a
 * weak model sometimes echoes the "Summary so far:" scaffold or returns nothing.
 * Reject those (keep the prior summary); accept anything with real content. */
static inline int summ_refine_usable(size_t n, const char *s) {
    if (n == 0) {
        return 0;
    }
    while (*s == ' ' || *s == '\n' || *s == '\t' || *s == '\r') {
        s++;
    }
    if (*s == '\0') {
        return 0;
    }
    return strncmp(s, "Summary so far", 14) != 0; /* scaffold echo */
}

/* Skip a leading echo of the refine scaffold ("Summary so far:" / "Updated
 * summary:"), with optional surrounding markdown '*' and whitespace, looping in
 * case both appear. Returns a pointer into s past the header(s); s unchanged when
 * the output doesn't lead with one (so non-echoing models are untouched). */
static inline const char *summ_skip_scaffold(const char *s) {
    static const char *const hdr[] = {"Summary so far:", "Updated summary:"};
    for (;;) {
        const char *p = s;
        while (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r' || *p == '*') {
            p++;
        }
        int matched = 0;
        for (size_t i = 0; i < sizeof hdr / sizeof *hdr; i++) {
            size_t len = strlen(hdr[i]);
            if (strncmp(p, hdr[i], len) == 0) {
                p += len;
                matched = 1;
                break;
            }
        }
        if (!matched) {
            return s; /* no scaffold prefix -> leave the output as-is */
        }
        while (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r' || *p == '*') {
            p++;
        }
        s = p;
    }
}

/* Core: read `path` under `root`, preprocess by type, refine-summarize, write the
 * summary into out. Always returns GEIST_OK with a human-readable error string in
 * out on failure (so the agent observes it rather than aborting). */
static inline enum geist_status summarize_file(struct geist_model   *model,
                                               struct geist_backend *be,
                                               const char           *root,
                                               const char           *path,
                                               size_t                out_cap,
                                               char                  out[static out_cap],
                                               size_t               *out_len) {
#define SUMM_ERR(...)                                          \
    do {                                                      \
        size_t n_ = (size_t) snprintf(out, out_cap, __VA_ARGS__); \
        if (out_len) {                                        \
            *out_len = n_;                                    \
        }                                                     \
        return GEIST_OK;                                      \
    } while (0)

    if (!summ_path_ok(path)) {
        SUMM_ERR("error: path \"%s\" is not allowed", path);
    }
    char full[1100];
    snprintf(full, sizeof full, "%s/%s", (root && root[0]) ? root : ".", path);

    static char raw[SUMM_RAW_CAP];
    FILE       *f = fopen(full, "r");
    if (f == nullptr) {
        SUMM_ERR("error: cannot open \"%s\"", path);
    }
    size_t rn = fread(raw, 1, sizeof raw - 1, f);
    raw[rn]   = '\0';
    fclose(f);

    const char *text = raw;
    size_t      tn   = rn;
    const char *dot  = strrchr(path, '.');
    if (dot != nullptr) {
        for (size_t i = 0; i < sizeof SUMM_PREPROC / sizeof SUMM_PREPROC[0]; i++) {
            if (strcmp(dot, SUMM_PREPROC[i].ext) == 0) {
                static char clean[SUMM_RAW_CAP];
                tn   = SUMM_PREPROC[i].fn(rn, raw, sizeof clean, clean);
                text = clean;
                break;
            }
        }
    }
    if (tn == 0) {
        SUMM_ERR("(empty file)");
    }

    struct geist_session_opts so  = {0}; /* greedy, deterministic */
    struct geist_session     *sub = nullptr;
    if (geist_session_create(model, be, &so, &sub) != GEIST_OK) {
        SUMM_ERR("error: could not create the summarizer session");
    }
    struct geist_chat_template tmpl = geist_chat_template_for_model(model);
    geist_token_t              eos  = geist_model_eos_token(model);
    geist_token_t eot = tmpl.stop[0] ? geist_model_token_by_text(model, tmpl.stop) : GEIST_TOKEN_NONE;

    static char running[SUMM_RUN_CAP];
    static char fresh[SUMM_RUN_CAP];
    static char prompt[SUMM_PROMPT_CAP];
    running[0]    = '\0';
    size_t run_n  = 0;
    size_t pos    = 0;
    int    first  = 1;
    while (pos < tn) {
        size_t end = summ_next_chunk(text, tn, pos, SUMM_CHUNK);
        if (first) {
            snprintf(prompt,
                     sizeof prompt,
                     "Summarize the following text concisely:\n%.*s",
                     (int) (end - pos),
                     text + pos);
        } else {
            snprintf(prompt,
                     sizeof prompt,
                     "Summary so far:\n%s\n\nRefine it to also cover this additional text:\n"
                     "%.*s\n\nUpdated summary:",
                     running,
                     (int) (end - pos),
                     text + pos);
        }
        /* Generate into a scratch buffer, then accept only if it's an improvement
         * to keep — else the prior running summary survives a degenerate step. */
        size_t fn = summ_generate(sub, &tmpl, eos, eot, prompt, sizeof fresh, fresh);
        if (run_n == 0 || summ_refine_usable(fn, fresh)) {
            memcpy(running, fresh, fn);
            running[fn] = '\0';
            run_n       = fn;
        }
        first = 0;
        pos   = end;
    }
    geist_session_destroy(sub);

    /* Some models (e.g. Gemma) prefix the answer with the refine scaffold itself
     * ("**Summary so far:**\n\n<real summary>"). Strip a leading scaffold header
     * if present — content-based, so a model that doesn't emit it is untouched. */
    const char *start = summ_skip_scaffold(running);
    run_n             = run_n - (size_t) (start - running);

    size_t n = run_n < out_cap ? run_n : out_cap - 1;
    memcpy(out, start, n);
    out[n] = '\0';
    if (out_len) {
        *out_len = n;
    }
    return GEIST_OK;
#undef SUMM_ERR
}

/* Tool ctx: the model + backend the sub-session runs on, and the read root. The
 * caller must build this AFTER loading the model and keep it alive. */
struct summarize_ctx {
    struct geist_model   *model;
    struct geist_backend *be;
    const char           *root; /* read confinement root; nullptr/"" -> cwd */
};

static inline enum geist_status summarize_invoke(void      *ctx,
                                                 size_t     args_len,
                                                 const char args[static args_len],
                                                 size_t     out_cap,
                                                 char       out[static out_cap],
                                                 size_t    *out_len) {
    (void) args_len;
    struct summarize_ctx *c = (struct summarize_ctx *) ctx;
    char                  path[1024];
    if (!agent_json_str(args, "path", sizeof path, path) || path[0] == '\0') {
        size_t n = (size_t) snprintf(out, out_cap, "error: missing \"path\"");
        if (out_len) {
            *out_len = n;
        }
        return GEIST_OK;
    }
    return summarize_file(c->model, c->be, c->root, path, out_cap, out, out_len);
}

static inline struct geist_tool summarize_file_tool(struct summarize_ctx *ctx) {
    return (struct geist_tool) {
            .name        = "summarize_file",
            .description = "eine Textdatei lesen und ihren Inhalt zusammenfassen",
            .args_schema = "{\"path\": string}",
            .invoke      = summarize_invoke,
            .ctx         = ctx,
    };
}

#endif /* GEIST_AGENT_SUMMARIZE_H */

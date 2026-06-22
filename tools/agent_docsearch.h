/*
 * agent_docsearch.h — a doc_search geist_tool: keyword search over a directory
 * of text files (BGB, manuals, ...). ctx is the directory path (const char *).
 *
 * Header-only. Returns up to MAX_HITS matching lines as "[file] line" so the
 * model can answer from the retrieved passages (local RAG, no embeddings).
 * ponytail: case-insensitive substring scan, line-granular, no ranking — fine
 * for a few MB of docs; add BM25 / an index when the corpus is large, and a
 * text-extraction step (pdftotext, ...) since this reads plain text only.
 */
#ifndef GEIST_AGENT_DOCSEARCH_H
#define GEIST_AGENT_DOCSEARCH_H

#include <geist.h>

#include "agent.h"

#include <ctype.h>
#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { DOCSEARCH_MAX_HITS = 5, DOCSEARCH_LINE_CAP = 1024 };

static inline int ci_contains(const char *hay, const char *needle) {
    size_t nl = strlen(needle);
    if (nl == 0) {
        return 0;
    }
    for (const char *h = hay; *h; h++) {
        size_t i = 0;
        while (h[i] && i < nl &&
               tolower((unsigned char) h[i]) == tolower((unsigned char) needle[i])) {
            i++;
        }
        if (i == nl) {
            return 1;
        }
    }
    return 0;
}

/* A line matches if every "significant" query word (>= 3 chars) appears in it,
 * case-insensitive (AND). Short words (the, on, is, …) are skipped so a natural
 * question like "warranty on the X200" matches a line that lacks "on"/"the". If
 * the query has no >=3-char word, fall back to a whole-query substring.
 * ponytail: length-3 heuristic, not a real stop-word list or stemmer; no
 * ranking — upgrade to BM25/an index when the corpus grows. */
static inline int docsearch_line_matches(const char *line, const char *query) {
    int         significant = 0;
    const char *p           = query;
    char        word[128];
    while (*p) {
        while (*p == ' ') {
            p++;
        }
        size_t w = 0;
        while (*p && *p != ' ') {
            if (w + 1 < sizeof word) {
                word[w++] = *p;
            }
            p++;
        }
        word[w] = '\0';
        if (w < 3) {
            continue; /* skip short / stop-ish words */
        }
        significant++;
        if (!ci_contains(line, word)) {
            return 0; /* a significant word is missing */
        }
    }
    return significant ? 1 : ci_contains(line, query);
}

/* geist_tool invoke: ctx = directory path. args = {"query": "..."}. */
static inline enum geist_status docsearch_invoke(void      *ctx,
                                                 size_t     args_len,
                                                 const char args[static args_len],
                                                 size_t     out_cap,
                                                 char       out[static out_cap],
                                                 size_t    *out_len) {
    (void) args_len;
    const char *dir = (const char *) ctx;
    char        query[256];
    if (!agent_json_str(args, "query", sizeof query, query)) {
        size_t n = (size_t) snprintf(out, out_cap, "error: missing \"query\"");
        if (out_len) {
            *out_len = n;
        }
        return GEIST_OK; /* a usable observation, not a hard failure */
    }

    DIR *d = opendir(dir);
    if (!d) {
        size_t n = (size_t) snprintf(out, out_cap, "error: cannot open doc dir");
        if (out_len) {
            *out_len = n;
        }
        return GEIST_OK;
    }

    size_t         w    = 0;
    int            hits = 0;
    struct dirent *de;
    while ((de = readdir(d)) != nullptr && hits < DOCSEARCH_MAX_HITS) {
        if (de->d_name[0] == '.') {
            continue; /* skip dotfiles / . / .. */
        }
        char path[1024];
        snprintf(path, sizeof path, "%s/%s", dir, de->d_name);
        FILE *f = fopen(path, "r");
        if (!f) {
            continue;
        }
        char line[DOCSEARCH_LINE_CAP];
        while (fgets(line, sizeof line, f) && hits < DOCSEARCH_MAX_HITS) {
            if (docsearch_line_matches(line, query)) {
                line[strcspn(line, "\n")] = '\0';
                int k = snprintf(out + w, out_cap - w, "[%s] %s\n", de->d_name, line);
                if (k < 0 || (size_t) k >= out_cap - w) {
                    break; /* out of room */
                }
                w += (size_t) k;
                hits++;
            }
        }
        fclose(f);
    }
    closedir(d);

    if (hits == 0) {
        w = (size_t) snprintf(out, out_cap, "no matches for \"%s\"", query);
    }
    if (out_len) {
        *out_len = w;
    }
    return GEIST_OK;
}

/* Ready-made whitelist entry; bind .ctx to your doc directory. */
static inline struct geist_tool docsearch_tool(const char *doc_dir) {
    return (struct geist_tool) {
            .name        = "doc_search",
            .args_schema = "{\"query\": string}",
            .invoke      = docsearch_invoke,
            .ctx         = (void *) (intptr_t) doc_dir, /* borrowed, host-owned */
    };
}

#endif /* GEIST_AGENT_DOCSEARCH_H */

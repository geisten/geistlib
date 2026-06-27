/*
 * agent_docsearch.h — a doc_search geist_tool: keyword search over a directory
 * of text files (BGB, manuals, ...). ctx is the directory path (const char *).
 *
 * Header-only. Returns up to MAX_HITS matching *paragraphs* as "[file] text",
 * so the model can answer from real passages (local RAG, no embeddings).
 * Matching is overlap-scored, not all-words-AND: a paragraph is returned when it
 * shares enough significant (>= 3-char) query words with the query — a whole
 * question ("how long is the warranty on the X200?") still hits the answer
 * paragraph, which shares "warranty/X200/..." but not "how/long". Recall over
 * precision: surface candidates, let the model judge.
 * ponytail: count-overlap ranking + a length-3 stop-word heuristic, paragraph-
 * granular; not BM25/stemming, no global ranking across files, reads plain text
 * only. Upgrade to BM25 + an index (and PDF extraction) when the corpus grows.
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

enum {
    DOCSEARCH_MAX_HITS  = 4,
    DOCSEARCH_FILE_CAP  = 1 << 16, /* read at most 64 KB per file */
    DOCSEARCH_PARA_CAP  = 1024,    /* score + show at most this much of a paragraph */
    DOCSEARCH_MAX_WORDS = 16,      /* significant query words considered */
};

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

/* Collect significant (>= 3-char) query words into words[]. Returns the count. */
static inline size_t docsearch_words(const char *query, size_t maxw, char words[static maxw][64]) {
    size_t      nw = 0;
    const char *p  = query;
    while (*p && nw < maxw) {
        while (*p == ' ') {
            p++;
        }
        size_t w = 0;
        char   tmp[64];
        while (*p && *p != ' ') {
            if (w + 1 < sizeof tmp) {
                tmp[w++] = *p;
            }
            p++;
        }
        tmp[w] = '\0';
        if (w >= 3) {
            snprintf(words[nw++], 64, "%s", tmp);
        }
    }
    return nw;
}

/* Number of distinct query words present in text (the overlap score). */
static inline int docsearch_score(const char *text, size_t nw, char words[static nw][64]) {
    int s = 0;
    for (size_t i = 0; i < nw; i++) {
        if (ci_contains(text, words[i])) {
            s++;
        }
    }
    return s;
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

    char   words[DOCSEARCH_MAX_WORDS][64];
    size_t nw = docsearch_words(query, DOCSEARCH_MAX_WORDS, words);
    /* need this many distinct words to count a paragraph; <2 sig words -> 1. */
    int threshold = nw >= 2 ? 2 : 1;
    if (nw == 0) { /* no significant word: fall back to the whole query */
        snprintf(words[0], 64, "%.63s", query);
        nw        = 1;
        threshold = 1;
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
    static char    fbuf[DOCSEARCH_FILE_CAP];
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
        size_t fn = fread(fbuf, 1, sizeof fbuf - 1, f);
        fbuf[fn]  = '\0';
        fclose(f);

        /* walk blank-line-separated paragraphs */
        const char *p = fbuf;
        while (*p && hits < DOCSEARCH_MAX_HITS) {
            while (*p == '\n') {
                p++; /* skip blank lines between paragraphs */
            }
            const char *e = p;
            while (*e && !(e[0] == '\n' && (e[1] == '\n' || e[1] == '\0'))) {
                e++;
            }
            /* copy paragraph (capped), collapsing newlines/runs of space */
            char   para[DOCSEARCH_PARA_CAP];
            size_t pw = 0;
            int    sp = 0;
            for (const char *q = p; q < e && pw + 1 < sizeof para; q++) {
                char c = (*q == '\n' || *q == '\t' || *q == '\r') ? ' ' : *q;
                if (c == ' ') {
                    if (sp) {
                        continue;
                    }
                    sp = 1;
                } else {
                    sp = 0;
                }
                para[pw++] = c;
            }
            while (pw > 0 && para[pw - 1] == ' ') {
                pw--;
            }
            para[pw] = '\0';

            if (pw > 0 && docsearch_score(para, nw, words) >= threshold) {
                int k = snprintf(out + w, out_cap - w, "[%s] %s\n", de->d_name, para);
                if (k < 0 || (size_t) k >= out_cap - w) {
                    break; /* out of room */
                }
                w += (size_t) k;
                hits++;
            }
            p = (*e) ? e + 1 : e;
        }
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
            .description = "search the local documents for a query",
            .args_schema = "{\"query\": string}",
            .invoke      = docsearch_invoke,
            .ctx         = (void *) (intptr_t) doc_dir, /* borrowed, host-owned */
    };
}

#endif /* GEIST_AGENT_DOCSEARCH_H */

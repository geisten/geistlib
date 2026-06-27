/*
 * agent_websearch.h — a web_search geist_tool (Unix/desktop): run a query against
 * DuckDuckGo's no-JS HTML endpoint and return the top results as "title + URL"
 * lines, which the agent can then hand to web_fetch. Reuses webfetch_curl (no
 * shell: curl via fork + execvp; scheme/size/time caps), so the same trust
 * boundary applies. ctx is an optional endpoint override (a SearXNG base URL
 * ending in "?q="); nullptr = DuckDuckGo.
 *
 * Unlike web_fetch the *host* is fixed (the search engine), so this tool is the
 * low-risk half of web access — but the results it returns are attacker-influenced
 * text, so treat any URL the model later fetches as untrusted (web_fetch's host
 * allowlist is the real gate).
 * ponytail: scrapes the HTML result anchors — if DuckDuckGo changes its markup or
 *   blocks the "geist-agent" UA, point ctx at a SearXNG instance instead. The
 *   parser is markup-shape, not a full HTML parser; that's the known ceiling.
 */
#ifndef GEIST_AGENT_WEBSEARCH_H
#define GEIST_AGENT_WEBSEARCH_H

#include <geist.h>

#include "agent.h"
#include "agent_webfetch.h" /* webfetch_curl, webfetch_strip_html */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { WEBSEARCH_MAX_RESULTS = 5 };
enum { WEBSEARCH_RAW_CAP = 1 << 17 }; /* 128 KB: the top results sit early in the page */

/* Percent-encode `q` onto the end of `base` (the search endpoint up to "?q="). */
static inline void
websearch_build_url(const char *base, const char *q, size_t cap, char out[static cap]) {
    size_t w = (size_t) snprintf(out, cap, "%s", base);
    for (const char *p = q; *p && w + 4 < cap; p++) {
        unsigned char c = (unsigned char) *p;
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' ||
            c == '_' || c == '.' || c == '~') {
            out[w++] = (char) c;
        } else {
            w += (size_t) snprintf(out + w, cap - w, "%%%02X", c);
        }
    }
    out[w < cap ? w : cap - 1] = '\0';
}

/* Percent-decode s[0..len) into out (also '+' -> space); for the DDG redirect's
 * uddg= real-URL parameter. */
static inline void
websearch_pct_decode(const char *s, size_t len, size_t cap, char out[static cap]) {
    size_t w = 0;
    for (size_t i = 0; i < len && w + 1 < cap; i++) {
        if (s[i] == '%' && i + 2 < len) {
            char        hi = s[i + 1], lo = s[i + 2];
            const char *hex = "0123456789abcdef";
            const char *ph  = strchr(hex, hi | 0x20), *pl = strchr(hex, lo | 0x20);
            if (ph && pl) {
                out[w++] = (char) ((ph - hex) * 16 + (pl - hex));
                i += 2;
                continue;
            }
        }
        out[w++] = s[i] == '+' ? ' ' : s[i];
    }
    out[w] = '\0';
}

/* Pull the real URL out of a DDG result href: "//duckduckgo.com/l/?uddg=<ENC>&..".
 * Decodes the uddg= value; if there's no uddg= (already a plain href) copies it
 * verbatim. Returns 1 on success. */
static inline int websearch_real_url(const char *href, size_t hlen, size_t cap, char out[static cap]) {
    const char *u = strstr(href, "uddg=");
    if (u && u < href + hlen) {
        u += 5;
        const char *end = u;
        while (end < href + hlen && *end != '&') {
            end++;
        }
        websearch_pct_decode(u, (size_t) (end - u), cap, out);
        return out[0] != '\0';
    }
    size_t n = hlen < cap - 1 ? hlen : cap - 1;
    memcpy(out, href, n);
    out[n] = '\0';
    return n > 0;
}

/* Parse up to WEBSEARCH_MAX_RESULTS "result__a" anchors out of a DDG HTML page
 * into "N. <title>\n   <url>\n" lines. Returns bytes written. */
static inline size_t
websearch_parse(const char *html, size_t cap, char out[static cap]) {
    size_t      w   = 0;
    int         hit = 0;
    const char *p   = html;
    while (hit < WEBSEARCH_MAX_RESULTS && (p = strstr(p, "result__a")) != nullptr) {
        const char *href = strstr(p, "href=\"");
        if (!href) {
            break;
        }
        href += 6;
        const char *hend = strchr(href, '"');
        if (!hend) {
            break;
        }
        const char *tstart = strchr(hend, '>'); /* title text starts after ">" */
        const char *tend   = tstart ? strstr(tstart, "</a>") : nullptr;
        if (!tstart || !tend) {
            break;
        }
        tstart++;

        char url[1024], title[512];
        websearch_real_url(href, (size_t) (hend - href), sizeof url, url);
        webfetch_strip_html((size_t) (tend - tstart), tstart, sizeof title, title);
        if (title[0] && url[0]) {
            w += (size_t) snprintf(out + w, w < cap ? cap - w : 0, "%d. %s\n   %s\n", hit + 1, title,
                                   url);
            hit++;
        }
        p = tend;
    }
    if (w == 0) {
        /* DuckDuckGo serves an "anomaly" interstitial when it rate-limits a
         * client — distinguish that from a genuinely empty result set. */
        w = (strstr(html, "anomaly") || strstr(html, "https://duckduckgo.com/help"))
                    ? (size_t) snprintf(
                              out, cap,
                              "error: the search engine rate-limited this request (try again "
                              "later, or point web_search at a SearXNG instance)")
                    : (size_t) snprintf(out, cap, "(no results)");
    }
    out[w < cap ? w : cap - 1] = '\0';
    return w < cap ? w : cap - 1;
}

/* geist_tool invoke: ctx = endpoint base up to "?q=" (or nullptr = DuckDuckGo).
 * args = {"query": "..."}. */
static inline enum geist_status websearch_invoke(void      *ctx,
                                                 size_t     args_len,
                                                 const char args[static args_len],
                                                 size_t     out_cap,
                                                 char       out[static out_cap],
                                                 size_t    *out_len) {
    (void) args_len;
    const char *base = ctx ? (const char *) ctx : "https://html.duckduckgo.com/html/?q=";
    char        query[512];
    size_t      n = 0;
    if (!agent_json_str(args, "query", sizeof query, query) || query[0] == '\0') {
        n = (size_t) snprintf(out, out_cap, "error: missing \"query\"");
    } else {
        char url[2048];
        websearch_build_url(base, query, sizeof url, url);
        static char raw[WEBSEARCH_RAW_CAP]; /* ponytail: bounded, single-threaded agent */
        long        got = webfetch_curl(url, sizeof raw, raw);
        if (got < 0) {
            n = (size_t) snprintf(out, out_cap, "error: search failed (curl missing or HTTP error)");
        } else {
            n = websearch_parse(raw, out_cap, out);
        }
    }
    if (out_len) {
        *out_len = n;
    }
    return GEIST_OK; /* errors are usable observations, not hard failures */
}

/* Ready-made whitelist entry; endpoint nullptr = DuckDuckGo HTML. */
static inline struct geist_tool websearch_tool(const char *endpoint) {
    return (struct geist_tool) {
            .name        = "web_search",
            .description = "das Web nach einer Anfrage durchsuchen und Trefferlinks zurückgeben",
            .args_schema = "{\"query\": string}",
            .invoke      = websearch_invoke,
            .ctx         = (void *) (intptr_t) endpoint,
    };
}

#endif /* GEIST_AGENT_WEBSEARCH_H */

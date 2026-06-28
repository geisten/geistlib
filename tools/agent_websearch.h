/*
 * agent_websearch.h — a web_search geist_tool (Unix/desktop): run a query and
 * return the top results as "title + URL" lines, which the agent can then hand to
 * web_fetch. Reuses webfetch_curl (no shell: curl via fork + execvp;
 * scheme/size/time caps), so the same trust boundary applies.
 *
 * ctx is the endpoint base up to "?q=" (nullptr = DuckDuckGo's no-JS HTML page).
 * The response shape is auto-detected: a body starting with '{' is parsed as
 * SearXNG JSON ({"results":[{"url","title",..}]}), anything else as DuckDuckGo
 * HTML — so pointing ctx at "https://<host>/search?format=json&q=" switches to a
 * self-hosted SearXNG with no code or flag change (and sidesteps DDG throttling).
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
websearch_parse_html(const char *html, size_t cap, char out[static cap]) {
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

/* Copy the JSON string value starting at `s` (just past the opening quote) into
 * out, decoding the escapes SearXNG actually emits (\" \\ \/ \n \t). Returns a
 * pointer just past the closing quote, or nullptr if unterminated. */
static inline const char *websearch_json_str(const char *s, size_t cap, char out[static cap]) {
    size_t w = 0;
    while (*s && *s != '"' && w + 1 < cap) {
        if (*s == '\\' && s[1]) {
            s++;
            switch (*s) {
            case 'n': out[w++] = '\n'; break;
            case 't': out[w++] = '\t'; break;
            default: out[w++] = *s; break; /* \" \\ \/ -> the literal char */
            }
            s++;
        } else {
            out[w++] = *s++;
        }
    }
    out[w] = '\0';
    return *s == '"' ? s + 1 : nullptr;
}

/* Parse a SearXNG JSON response ({"results":[{"url":..,"title":..,..},..]}) into
 * the same "N. <title>\n   <url>\n" lines. Pairs each "title" with the "url" in
 * the same result object (whichever key appears first), up to MAX_RESULTS.
 * ponytail: a flat key scan, not a real JSON parser — fine for SearXNG's shape;
 * a "content" value containing a literal "url": would mis-pair (not observed). */
static inline size_t websearch_parse_json(const char *json, size_t cap, char out[static cap]) {
    size_t      w   = 0;
    int         hit = 0;
    char        url[1024]  = {0};
    char        title[512] = {0};
    const char *p          = json;
    while (hit < WEBSEARCH_MAX_RESULTS && *p) {
        const char *uq = strstr(p, "\"url\"");
        const char *tq = strstr(p, "\"title\"");
        if (!uq && !tq) {
            break;
        }
        /* take whichever key comes first; step past key + ':' + optional spaces to
         * the value's opening quote (SearXNG may pretty-print "url": "...") */
        int         is_url = uq && (!tq || uq < tq);
        const char *c      = (is_url ? uq + 5 : tq + 7); /* past "url" / "title" */
        while (*c && *c != ':' && *c != '"') {
            c++;
        }
        if (*c != ':') { /* malformed / not this key's value -> skip past the key */
            p = c;
            continue;
        }
        c++;
        while (*c == ' ' || *c == '\n' || *c == '\t' || *c == '\r') {
            c++;
        }
        if (*c != '"') { /* value isn't a string (null/number) -> skip */
            p = c;
            continue;
        }
        p = websearch_json_str(c + 1, is_url ? sizeof url : sizeof title, is_url ? url : title);
        if (!p) {
            break;
        }
        if (url[0] && title[0]) {
            w += (size_t) snprintf(out + w, w < cap ? cap - w : 0, "%d. %s\n   %s\n", hit + 1, title,
                                   url);
            hit++;
            url[0] = title[0] = '\0';
        }
    }
    if (w == 0) {
        w = (size_t) snprintf(out, cap, "(no results)");
    }
    out[w < cap ? w : cap - 1] = '\0';
    return w < cap ? w : cap - 1;
}

/* Dispatch by response shape: a JSON body (SearXNG ?format=json) starts with '{';
 * anything else is treated as DuckDuckGo HTML. So one tool serves both engines
 * with no config flag — the endpoint (ctx) alone decides what comes back. */
static inline size_t websearch_parse(const char *body, size_t cap, char out[static cap]) {
    const char *p = body;
    while (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r') {
        p++;
    }
    return *p == '{' ? websearch_parse_json(body, cap, out) : websearch_parse_html(body, cap, out);
}

/* geist_tool invoke: ctx = endpoint base up to "?q=" (or nullptr = DuckDuckGo).
 * For SearXNG pass "https://<host>/search?format=json&q=". args = {"query": ".."}. */
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

/*
 * agent_stocks.h — stock_movers geist_tool: today's top stock gainers/losers.
 *
 * The "which stock performed best today?" use case done the geist way: the
 * TOOL is deterministic (a finance screener API returns the list already
 * sorted by percent change), the model only routes — no scraping a HTML table
 * and no asking a 2B model to compute an argmax over it.
 *
 *   stock_movers: args {"query": "..."}  free text; a losers word in it
 *                 (verlier/loser/worst/schlecht/lost) flips the direction,
 *                 anything else means gainers. The value is lifted from the
 *                 request by the forced path, so the whole user question IS
 *                 the query.
 *
 * Data source: the Yahoo Finance predefined screener (day_gainers/day_losers,
 * JSON, keyless). ctx overrides the endpoint prefix (a test server, another
 * screener with the same shape); nullptr = Yahoo. Unix/desktop only — it
 * fetches via webfetch_curl (fork + execvp, no shell); an embedded host
 * supplies its own tool over the platform HTTP client.
 *
 * ponytail: fields are scanned with strstr (symbol + regularMarketChangePercent,
 * bare number or {"raw":...}) — move to a real JSON walker if the API shape
 * ever grows ambiguous. US market data; the API decides what "today" means.
 */
#ifndef GEIST_AGENT_STOCKS_H
#define GEIST_AGENT_STOCKS_H

#include <geist.h>

#include "agent.h"
#include "agent_webfetch.h" /* webfetch_curl */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#define STOCKS_DEFAULT_ENDPOINT \
    "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
enum { STOCKS_RAW_CAP = 1 << 16, STOCKS_TOP_N = 10 };

/* True if the query asks for the losers side (verlier/loser/lost/worst/
 * schlecht — word-start, case-insensitive); default is gainers. PURE. */
static inline int stocks_wants_losers(const char *query) {
    static const char *const w[] = {"verlier", "verlor", "loser", "lost", "worst", "schlecht"};
    for (size_t v = 0; v < sizeof w / sizeof *w; v++) {
        size_t wl = strlen(w[v]);
        for (const char *p = query; *p; p++) {
            if ((p == query || p[-1] == ' ' || p[-1] == '\t') && strncasecmp(p, w[v], wl) == 0) {
                return 1;
            }
        }
    }
    return 0;
}

/* Format the screener JSON into "1. SYMBOL +12.34%" lines, sorted by percent
 * change (descending for gainers, ascending for losers) — the API's own order
 * is only approximately sorted, and rank 1 IS the answer to "which stock
 * performed best". Scans "symbol":"..." and the following
 * "regularMarketChangePercent": value — a bare number or a {"raw":N,...}
 * object. Returns bytes written, 0 if nothing parsed. PURE. */
static inline size_t stocks_format(const char *raw, int losers, size_t cap, char out[static cap]) {
    char        sym[STOCKS_TOP_N][16];
    double      pct[STOCKS_TOP_N];
    int         n = 0;
    const char *p = raw;
    while (n < STOCKS_TOP_N && (p = strstr(p, "\"symbol\":\"")) != nullptr) {
        p += strlen("\"symbol\":\"");
        size_t sl = 0;
        while (*p && *p != '"' && sl + 1 < sizeof sym[0]) {
            sym[n][sl++] = *p++;
        }
        sym[n][sl]    = '\0';
        const char *c = strstr(p, "\"regularMarketChangePercent\":");
        if (c == nullptr) {
            break;
        }
        c += strlen("\"regularMarketChangePercent\":");
        if (*c == '{') { /* {"raw":N,"fmt":"..."} variant */
            const char *r = strstr(c, "\"raw\":");
            if (r == nullptr) {
                break;
            }
            c = r + strlen("\"raw\":");
        }
        pct[n++] = strtod(c, nullptr);
        p        = c;
    }
    for (int i = 0; i < n; i++) { /* insertion sort; n <= 10 */
        for (int j = i; j > 0 && (losers ? pct[j] < pct[j - 1] : pct[j] > pct[j - 1]); j--) {
            double td = pct[j];
            pct[j] = pct[j - 1], pct[j - 1] = td;
            char ts[16];
            memcpy(ts, sym[j], sizeof ts);
            memcpy(sym[j], sym[j - 1], sizeof ts);
            memcpy(sym[j - 1], ts, sizeof ts);
        }
    }
    size_t w = 0;
    for (int i = 0; i < n; i++) {
        int k = snprintf(out + w, cap - w, "%d. %s %+.2f%%\n", i + 1, sym[i], pct[i]);
        if (k < 0 || (size_t) k >= cap - w) {
            break;
        }
        w += (size_t) k;
    }
    return w;
}

/* geist_tool invoke: ctx = endpoint prefix override (or nullptr = Yahoo). */
static inline enum geist_status stocks_invoke(void      *ctx,
                                              size_t     args_len,
                                              const char args[static args_len],
                                              size_t     out_cap,
                                              char       out[static out_cap],
                                              size_t    *out_len) {
    char query[512];
    agent_json_str(args, "query", sizeof query, query);
    const int   losers   = stocks_wants_losers(query);
    const char *endpoint = ctx ? (const char *) ctx : STOCKS_DEFAULT_ENDPOINT;

    char url[1024];
    snprintf(url,
             sizeof url,
             "%s?scrIds=%s&count=%d",
             endpoint,
             losers ? "day_losers" : "day_gainers",
             STOCKS_TOP_N);

    static char raw[STOCKS_RAW_CAP];
    if (webfetch_curl(url, sizeof raw, raw) <= 0) {
        return agent_obs(out_cap, out, out_len, "error: stock data unavailable (fetch failed)");
    }
    size_t w = (size_t) snprintf(out,
                                 out_cap,
                                 "Today's %s (percent change):\n",
                                 losers ? "worst performing stocks" : "top performing stocks");
    size_t n = stocks_format(raw, losers, out_cap - w, out + w);
    if (n == 0) {
        return agent_obs(out_cap, out, out_len, "error: unexpected screener response");
    }
    return agent_ret(out_len, w + n);
}

/* Ready-made whitelist entry; endpoint nullptr = the Yahoo screener. */
static inline struct geist_tool stock_movers_tool(const char *endpoint) {
    return (struct geist_tool) {
            .name        = "stock_movers",
            .description = "die Aktien-Tagesgewinner oder -verlierer der Börse abrufen",
            .args_schema = "{\"query\": string}",
            .invoke      = stocks_invoke,
            .ctx         = (void *) (intptr_t) endpoint,
    };
}

#endif /* GEIST_AGENT_STOCKS_H */

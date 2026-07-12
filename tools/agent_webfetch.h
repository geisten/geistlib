/*
 * agent_webfetch.h — a web_fetch geist_tool (Unix/desktop): fetch an http(s)
 * URL via curl and return the tag-stripped text. ctx is an optional host
 * allowlist (comma-separated; nullptr = any http/https host).
 *
 * web_fetch is the agent's most dangerous tool — an untrusted model picks the
 * URL — so the trust boundary is defended several ways, NOT simplified away:
 *   - no shell: curl is run via fork + execvp with the URL as a separate argv
 *     element, so a URL like  http://x;rm -rf ~  cannot inject a command;
 *   - scheme gate: only literal http:// / https:// (no file:, data:, gopher:);
 *   - host allowlist (ctx): exact host or a dot-bounded subdomain;
 *   - curl --proto/--proto-redir =http,https so a redirect can't escape to
 *     file:// or another scheme; size + time caps.
 * ponytail: does NOT block private/link-local IPs (SSRF) — for an open fetch
 *   add IP filtering; the host allowlist is the real mitigation. Keeps text
 *   inside <script>/<style>; swap in lynx -dump if you need clean extraction.
 *   iOS/Android: curl + fork aren't available in the sandbox — the host app
 *   supplies its own web_fetch via the platform HTTP client; this header is the
 *   Unix tool.
 */
#ifndef GEIST_AGENT_WEBFETCH_H
#define GEIST_AGENT_WEBFETCH_H

#include <geist.h>

#include "agent.h"

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

enum { WEBFETCH_RAW_CAP = 1 << 16 }; /* read at most 64 KB of body */

/* http:// or https:// only (lowercase — models emit lowercase; stricter=safer). */
static inline int webfetch_scheme_ok(const char *url) {
    return strncmp(url, "http://", 7) == 0 || strncmp(url, "https://", 8) == 0;
}

/* Host between "://" and the next '/'|':'|'?'|'#'. Returns 1 on success.
 * Any "userinfo@" prefix is skipped so the allowlist validates the host curl
 * actually connects to — else "http://allowed.com:@evil/" would pass the gate
 * (host read as "allowed.com") while curl connects to "evil". */
static inline int webfetch_host(const char *url, size_t cap, char host[static cap]) {
    const char *s = strstr(url, "://");
    if (!s) {
        return 0;
    }
    s += 3;
    for (const char *at = s; *at && *at != '/' && *at != '?' && *at != '#'; at++) {
        if (*at == '@') {
            s = at + 1; /* last '@' in the authority wins, as curl parses it */
        }
    }
    size_t w = 0;
    while (*s && *s != '/' && *s != ':' && *s != '?' && *s != '#' && w + 1 < cap) {
        host[w++] = *s++;
    }
    host[w] = '\0';
    return w > 0;
}

/* allow nullptr/"" = any. Else host must equal an entry or be a dot-bounded
 * subdomain of it (so "example.com" allows "www.example.com" but NOT
 * "notexample.com" or "example.com.evil.com"). */
static inline int webfetch_host_allowed(const char *host, const char *allow) {
    if (!allow || !allow[0]) {
        return 1;
    }
    size_t      hl = strlen(host);
    const char *p  = allow;
    while (*p) {
        const char *comma = strchr(p, ',');
        size_t      el    = comma ? (size_t) (comma - p) : strlen(p);
        while (el && *p == ' ') { /* trim */
            p++;
            el--;
        }
        while (el && p[el - 1] == ' ') {
            el--;
        }
        if (el) {
            if (hl == el && strncmp(host, p, el) == 0) {
                return 1; /* exact */
            }
            if (hl > el + 1 && host[hl - el - 1] == '.' && strncmp(host + hl - el, p, el) == 0) {
                return 1; /* dot-bounded subdomain */
            }
        }
        if (!comma) {
            break;
        }
        p = comma + 1;
    }
    return 0;
}

/* If s[0..max) begins with an HTML entity (&name; or &#NN; / &#xHH;), set *ch to
 * its character and return the bytes consumed (incl. '&' and ';'); else return 0.
 * Codepoints >127 are dropped (*ch=0, still consumed) except nbsp -> space, so the
 * decode never emits raw high bytes into the agent's text. */
static inline size_t webfetch_decode_entity(const char *s, size_t max, char *ch) {
    if (max < 3 || s[0] != '&') {
        return 0;
    }
    size_t semi = 0;
    for (size_t k = 1; k < max && k <= 10; k++) { /* entities are short; bail early */
        if (s[k] == ';') {
            semi = k;
            break;
        }
        if (s[k] == '&' || s[k] == ' ' || s[k] == '<') {
            return 0; /* a bare '&' in running text, not an entity */
        }
    }
    if (semi < 2) {
        return 0;
    }
    if (s[1] == '#') { /* numeric: &#39; (decimal) or &#x27; (hex) */
        char *end = nullptr;
        int   hex = (s[2] == 'x' || s[2] == 'X');
        long  v   = strtol(s + (hex ? 3 : 2), &end, hex ? 16 : 10);
        if (end != s + semi) {
            return 0;
        }
        *ch = (v > 0 && v < 128) ? (char) v : (v == 160 ? ' ' : 0);
        return semi + 1;
    }
    static const struct {
        const char *name;
        char        ch;
    } ents[] = {
            {"amp", '&'}, {"lt", '<'}, {"gt", '>'}, {"quot", '"'}, {"apos", '\''}, {"nbsp", ' '}};
    size_t nlen = semi - 1;
    for (size_t e = 0; e < sizeof ents / sizeof ents[0]; e++) {
        if (strlen(ents[e].name) == nlen && strncmp(s + 1, ents[e].name, nlen) == 0) {
            *ch = ents[e].ch;
            return semi + 1;
        }
    }
    return 0;
}

/* Drop everything between '<' and '>', decode HTML entities (&amp; -> &, …), and
 * collapse runs of whitespace. */
static inline size_t
webfetch_strip_html(size_t n, const char in[static n], size_t cap, char out[static cap]) {
    size_t w     = 0;
    int    intag = 0, sp = 0;
    for (size_t i = 0; i < n && w + 1 < cap; i++) {
        char c = in[i];
        if (c == '<') {
            intag = 1;
        } else if (c == '>') {
            intag = 0;
        } else if (!intag) {
            char   ent  = 0;
            size_t used = c == '&' ? webfetch_decode_entity(in + i, n - i, &ent) : 0;
            if (used) {
                i += used - 1; /* the loop's i++ steps past the ';' */
                c = ent;       /* fall through with the decoded char (0 = drop) */
                if (c == '\0') {
                    continue;
                }
            }
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                if (!sp && w > 0) {
                    out[w++] = ' ';
                    sp       = 1;
                }
            } else {
                out[w++] = c;
                sp       = 0;
            }
        }
    }
    while (w > 0 && out[w - 1] == ' ') {
        w--;
    }
    out[w] = '\0';
    return w;
}

/* Run curl with the URL as a literal argv (no shell). Returns bytes read into
 * raw, or -1 if curl is missing / failed / the fetch errored. */
static inline long webfetch_curl(const char *url, size_t cap, char raw[static cap]) {
    int fds[2];
    if (pipe(fds) != 0) {
        return -1;
    }
    pid_t pid = fork();
    if (pid < 0) {
        close(fds[0]);
        close(fds[1]);
        return -1;
    }
    if (pid == 0) { /* child: curl stdout -> pipe, stderr -> /dev/null */
        dup2(fds[1], STDOUT_FILENO);
        int nul = open("/dev/null", O_WRONLY);
        if (nul >= 0) {
            dup2(nul, STDERR_FILENO);
        }
        close(fds[0]);
        close(fds[1]);
        char *const argv[] = {(char *) "curl",
                              (char *) "-sSL",
                              (char *) "--max-time",
                              (char *) "10",
                              (char *) "--max-filesize",
                              (char *) "5000000",
                              (char *) "--proto",
                              (char *) "=http,https",
                              (char *) "--proto-redir",
                              (char *) "=http,https",
                              (char *) "-A",
                              (char *) "geist-agent",
                              (char *) url,
                              nullptr};
        execvp("curl", argv);
        _exit(127); /* curl not found */
    }
    close(fds[1]);
    size_t  n = 0;
    ssize_t r;
    while (n + 1 < cap && (r = read(fds[0], raw + n, cap - 1 - n)) > 0) {
        n += (size_t) r;
    }
    raw[n] = '\0';
    close(fds[0]);
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        return -1;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        return -1; /* curl missing or fetch failed */
    }
    return (long) n;
}

/* geist_tool invoke: ctx = host allowlist (or nullptr). args = {"url": "..."}. */
static inline enum geist_status webfetch_invoke(void      *ctx,
                                                size_t     args_len,
                                                const char args[static args_len],
                                                size_t     out_cap,
                                                char       out[static out_cap],
                                                size_t    *out_len) {
    (void) args_len;
    const char *allow = (const char *) ctx;
    char        url[1024];
    size_t      n = 0;

    if (!agent_json_str(args, "url", sizeof url, url)) {
        n = (size_t) snprintf(out, out_cap, "error: missing \"url\"");
    } else if (!webfetch_scheme_ok(url)) {
        n = (size_t) snprintf(out, out_cap, "error: only http/https URLs are allowed");
    } else {
        char host[256];
        if (!webfetch_host(url, sizeof host, host)) {
            n = (size_t) snprintf(out, out_cap, "error: malformed URL");
        } else if (!webfetch_host_allowed(host, allow)) {
            n = (size_t) snprintf(out, out_cap, "error: host \"%s\" is not allowed", host);
        } else {
            static char raw[WEBFETCH_RAW_CAP]; /* ponytail: bounded, single-threaded agent */
            long        got = webfetch_curl(url, sizeof raw, raw);
            if (got < 0) {
                n = (size_t) snprintf(
                        out, out_cap, "error: fetch failed (curl missing or HTTP error)");
            } else {
                n = webfetch_strip_html((size_t) got, raw, out_cap, out);
                if (n == 0) {
                    n = (size_t) snprintf(out, out_cap, "(no text content)");
                }
            }
        }
    }
    if (out_len) {
        *out_len = n;
    }
    return GEIST_OK; /* errors are usable observations, not hard failures */
}

/* Ready-made whitelist entry; allow_hosts nullptr = any http/https host. */
static inline struct geist_tool webfetch_tool(const char *allow_hosts) {
    return (struct geist_tool) {
            .name        = "web_fetch",
            .description = "eine Webseite per URL abrufen und ihren Text zurückgeben",
            .args_schema = "{\"url\": string}",
            .invoke      = webfetch_invoke,
            .ctx         = (void *) (intptr_t) allow_hosts,
    };
}

#endif /* GEIST_AGENT_WEBFETCH_H */

/*
 * test_websearch_unit — the pure (no-network) core of web_search: URL building,
 * percent-decoding the DuckDuckGo redirect, and parsing result anchors out of a
 * representative HTML snippet. The network path (webfetch_curl) is not exercised
 * here. No assert() — checks set a flag, the exit code carries PASS/FAIL.
 */
#define _POSIX_C_SOURCE 200809L

#include "test_helpers.h"

#include "../tools/agent_websearch.h"

#include <stdio.h>
#include <string.h>

static int fails = 0;

static void test_url(void) {
    char url[256];
    websearch_build_url("https://x/?q=", "fußball wm 2026", sizeof url, url);
    fails += geist_expect(strcmp(url, "https://x/?q=fu%C3%9Fball%20wm%202026") == 0,
                          "url: query percent-encoded (space + UTF-8)");
}

static void test_pct_decode(void) {
    char out[256];
    websearch_pct_decode("https%3A%2F%2Fde.wikipedia.org%2Fwiki%2FWM",
                         strlen("https%3A%2F%2Fde.wikipedia.org%2Fwiki%2FWM"),
                         sizeof out,
                         out);
    fails += geist_expect(strcmp(out, "https://de.wikipedia.org/wiki/WM") == 0,
                          "decode: uddg real URL decoded");
}

static void test_real_url(void) {
    char        out[256];
    const char *href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fp&rut=abc";
    websearch_real_url(href, strlen(href), sizeof out, out);
    fails += geist_expect(strcmp(out, "https://example.org/p") == 0,
                          "real_url: extracts + decodes uddg, drops &rut");

    const char *plain = "https://plain.example/x";
    websearch_real_url(plain, strlen(plain), sizeof out, out);
    fails += geist_expect(strcmp(out, "https://plain.example/x") == 0,
                          "real_url: a plain href is kept verbatim");
}

static void test_parse(void) {
    /* Two DuckDuckGo-shaped result anchors; the second title has a nested tag. */
    const char *html = "<div><a rel=\"nofollow\" class=\"result__a\" "
                       "href=\"//duckduckgo.com/l/"
                       "?uddg=https%3A%2F%2Fde.wikipedia.org%2Fwiki%2FWM_2026&rut=x\">"
                       "WM 2026 – Wikipedia</a></div>"
                       "<div><a class=\"result__a\" "
                       "href=\"//duckduckgo.com/l/?uddg=https%3A%2F%2Ffifa.com%2Fwc\">"
                       "FIFA <b>World Cup</b> 2026</a></div>";
    char        out[1024];
    size_t      n = websearch_parse(html, sizeof out, out);
    fails += geist_expect(n > 0, "parse: non-empty");
    fails += geist_expect(strstr(out, "WM 2026 – Wikipedia") != nullptr, "parse: first title");
    fails += geist_expect(strstr(out, "https://de.wikipedia.org/wiki/WM_2026") != nullptr,
                          "parse: first URL decoded");
    fails += geist_expect(strstr(out, "FIFA World Cup 2026") != nullptr,
                          "parse: second title with nested tag stripped");
    fails += geist_expect(strstr(out, "https://fifa.com/wc") != nullptr, "parse: second URL");

    char empty[128];
    websearch_parse("<html>no results here</html>", sizeof empty, empty);
    fails += geist_expect(strcmp(empty, "(no results)") == 0, "parse: no anchors -> (no results)");

    char blocked[128];
    websearch_parse(
            "<html>If this error persists... anomaly detected</html>", sizeof blocked, blocked);
    fails += geist_expect(strstr(blocked, "rate-limited") != nullptr,
                          "parse: DDG anomaly page -> rate-limit message, not (no results)");
}

int main(void) {
    test_url();
    test_pct_decode();
    test_real_url();
    test_parse();
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("web_search: url build + uddg decode + result parsing pass\n");
    return GEIST_TEST_PASS;
}

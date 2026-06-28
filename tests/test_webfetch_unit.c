/*
 * test_webfetch_unit — web_fetch's deterministic, security-critical parts.
 * No network, no curl. Exhaustively covers the trust-boundary checks (scheme
 * gate, host allowlist incl. the subdomain / look-alike traps), the URL parse,
 * the args extraction, and the HTML stripper. The real curl fetch is in the
 * _int test. No assert() — checks set a flag, exit code carries PASS/FAIL.
 */
#define _POSIX_C_SOURCE 200809L

#include "test_helpers.h"

#include "../tools/agent_webfetch.h"

#include <stdio.h>
#include <string.h>

static int  fails = 0;
static void test_scheme(void) {
    fails += geist_expect(webfetch_scheme_ok("http://example.com"), "scheme: http ok");
    fails += geist_expect(webfetch_scheme_ok("https://example.com"), "scheme: https ok");
    fails += geist_expect(!webfetch_scheme_ok("file:///etc/passwd"), "scheme: file:// rejected");
    fails += geist_expect(!webfetch_scheme_ok("ftp://host/x"), "scheme: ftp rejected");
    fails += geist_expect(!webfetch_scheme_ok("data:text/html,<b>x</b>"), "scheme: data: rejected");
    fails += geist_expect(!webfetch_scheme_ok("gopher://h"), "scheme: gopher rejected");
    fails +=
            geist_expect(!webfetch_scheme_ok("javascript:alert(1)"), "scheme: javascript rejected");
    fails += geist_expect(!webfetch_scheme_ok("HTTP://example.com"),
                          "scheme: uppercase rejected (strict)");
    fails += geist_expect(!webfetch_scheme_ok("//example.com"), "scheme: schemeless rejected");
    fails += geist_expect(!webfetch_scheme_ok(""), "scheme: empty rejected");
}

static void test_host(void) {
    char h[256];
    fails += geist_expect(webfetch_host("http://example.com/path?q=1", sizeof h, h) &&
                                  strcmp(h, "example.com") == 0,
                          "host: path stripped");
    fails += geist_expect(webfetch_host("https://a.b.c:8443/x", sizeof h, h) &&
                                  strcmp(h, "a.b.c") == 0,
                          "host: port stripped");
    fails += geist_expect(webfetch_host("http://h?q", sizeof h, h) && strcmp(h, "h") == 0,
                          "host: query stripped");
    fails += geist_expect(webfetch_host("http://h#frag", sizeof h, h) && strcmp(h, "h") == 0,
                          "host: fragment stripped");
    fails += geist_expect(!webfetch_host("notaurl", sizeof h, h), "host: no scheme -> fail");
}

static void test_allowlist(void) {
    /* nullptr / "" = any host allowed */
    fails += geist_expect(webfetch_host_allowed("anything.com", nullptr), "allow: null = any");
    fails += geist_expect(webfetch_host_allowed("anything.com", ""), "allow: empty = any");

    const char *al = "example.com, intranet.local";
    fails += geist_expect(webfetch_host_allowed("example.com", al), "allow: exact match");
    fails += geist_expect(webfetch_host_allowed("www.example.com", al), "allow: subdomain ok");
    fails += geist_expect(webfetch_host_allowed("a.b.example.com", al), "allow: deep subdomain ok");
    fails += geist_expect(webfetch_host_allowed("intranet.local", al),
                          "allow: second entry, spaces trimmed");
    fails += geist_expect(!webfetch_host_allowed("notexample.com", al),
                          "allow: look-alike rejected");
    fails += geist_expect(!webfetch_host_allowed("example.com.evil.com", al),
                          "allow: suffix-spoof rejected");
    fails += geist_expect(!webfetch_host_allowed("example.org", al), "allow: other TLD rejected");
    fails += geist_expect(!webfetch_host_allowed("evil.com", al), "allow: unrelated rejected");
}

static void test_strip(void) {
    char out[1024];
    webfetch_strip_html(
            strlen("<p>Hello <b>world</b></p>"), "<p>Hello <b>world</b></p>", sizeof out, out);
    fails += geist_expect(strcmp(out, "Hello world") == 0, "strip: tags removed, text kept");

    const char *attr = "<a href=\"http://x\">link</a>";
    webfetch_strip_html(strlen(attr), attr, sizeof out, out);
    fails += geist_expect(strcmp(out, "link") == 0, "strip: attributes (incl. <) dropped");

    const char *ws = "  Lots\n\n  of   \t whitespace  ";
    webfetch_strip_html(strlen(ws), ws, sizeof out, out);
    fails += geist_expect(strcmp(out, "Lots of whitespace") == 0,
                          "strip: whitespace collapsed + trimmed");

    webfetch_strip_html(strlen("plain text"), "plain text", sizeof out, out);
    fails += geist_expect(strcmp(out, "plain text") == 0, "strip: no-tag passthrough");

    webfetch_strip_html(
            strlen("<div><span></span></div>"), "<div><span></span></div>", sizeof out, out);
    fails += geist_expect(out[0] == '\0', "strip: tags-only -> empty");

    /* HTML entities are decoded (the &amp; that showed up in web_search titles) */
    const char *ent = "Tickets &amp; Travel &lt;b&gt; &quot;x&quot; &#39;y&#39; a&nbsp;b";
    webfetch_strip_html(strlen(ent), ent, sizeof out, out);
    fails += geist_expect(strcmp(out, "Tickets & Travel <b> \"x\" 'y' a b") == 0,
                          "strip: HTML entities decoded");

    /* a bare ampersand in running text is left alone (not an entity) */
    const char *amp = "Q &amp; A but R & D and AT&T";
    webfetch_strip_html(strlen(amp), amp, sizeof out, out);
    fails += geist_expect(strcmp(out, "Q & A but R & D and AT&T") == 0,
                          "strip: bare '&' passes through");

    /* truncation must stay in-bounds and NUL-terminate */
    char small[8];
    webfetch_strip_html(strlen("abcdefghijklmnop"), "abcdefghijklmnop", sizeof small, small);
    fails += geist_expect(strlen(small) < sizeof small, "strip: respects out cap");
}

static void test_args(void) {
    char url[1024];
    fails += geist_expect(
            agent_json_str("{\"url\":\"http://example.com/x\"}", "url", sizeof url, url) &&
                    strcmp(url, "http://example.com/x") == 0,
            "args: url extracted");
    fails += geist_expect(!agent_json_str("{\"query\":\"x\"}", "url", sizeof url, url),
                          "args: missing url -> 0");
}

int main(void) {
    test_scheme();
    test_host();
    test_allowlist();
    test_strip();
    test_args();
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("web_fetch: scheme + allowlist + strip + args pass\n");
    return GEIST_TEST_PASS;
}

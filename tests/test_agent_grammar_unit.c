/*
 * test_agent_grammar_unit — the pure (no-model) core of the grammar constraint.
 *
 * Two slices of item 3:
 *  - name: agent_name_is_prefix / agent_name_complete decide, for the chars
 *    emitted so far, whether the partial still matches a whitelist name and
 *    whether it is complete — the invariant that lets the model-driven decode
 *    (agent_decode_name_constrained, exercised in the _int test) only ever spell
 *    a whitelisted name.
 *  - args: agent_schema_keys parses a tool's declared keys and
 *    agent_args_normalize re-keys a mis-keyed string arg onto the schema key.
 * No assert() — checks set a flag, the exit code carries PASS/FAIL.
 */
#define _POSIX_C_SOURCE 200809L

#include "test_helpers.h"

#include "../tools/agent.h"

#include <stdio.h>
#include <string.h>

static int fails = 0;

static void test_matchers(void) {
    struct geist_tool tools[] = {
            {.name = "doc_search", .args_schema = "{}", .invoke = nullptr, .ctx = nullptr},
            {.name = "web_fetch", .args_schema = "{}", .invoke = nullptr, .ctx = nullptr},
    };
    static struct geist_agent a;
    a.tools   = tools;
    a.n_tools = sizeof tools / sizeof *tools;

    /* is_prefix: empty matches anything; real prefixes match; off-list don't */
    fails += geist_expect(agent_name_is_prefix(&a, ""), "prefix: empty matches");
    fails += geist_expect(agent_name_is_prefix(&a, "doc"), "prefix: 'doc' matches doc_search");
    fails += geist_expect(agent_name_is_prefix(&a, "web_fetch"), "prefix: full name matches");
    fails += geist_expect(!agent_name_is_prefix(&a, "doc_searx"), "prefix: typo does not match");
    fails += geist_expect(!agent_name_is_prefix(&a, "rm"), "prefix: unrelated does not match");
    fails += geist_expect(!agent_name_is_prefix(&a, "doc_search_extra"),
                          "prefix: longer-than-name does not match");

    /* complete: exact name -> index; partial / off-list -> -1 */
    fails += geist_expect(agent_name_complete(&a, "doc_search") == 0, "complete: first tool index");
    fails += geist_expect(agent_name_complete(&a, "web_fetch") == 1, "complete: second tool index");
    fails +=
            geist_expect(agent_name_complete(&a, "doc") == -1, "complete: partial is not complete");
    fails += geist_expect(agent_name_complete(&a, "rm_rf") == -1, "complete: off-list is -1");

    /* the decode invariant: a name built char-by-char stays a prefix at every
     * step and only the full string completes — so the decode can never land on
     * a non-whitelisted name. */
    const char *target                        = "doc_search";
    char        partial[GEIST_AGENT_NAME_CAP] = {0};
    int         all_prefix = 1, early_complete = 0;
    for (size_t i = 0; i < strlen(target); i++) {
        partial[i]     = target[i];
        partial[i + 1] = '\0';
        all_prefix &= agent_name_is_prefix(&a, partial);
        if (i + 1 < strlen(target) && agent_name_complete(&a, partial) >= 0) {
            early_complete = 1;
        }
    }
    fails += geist_expect(all_prefix, "decode-invariant: every step stays a prefix");
    fails += geist_expect(!early_complete, "decode-invariant: completes only at the full name");
    fails += geist_expect(agent_name_complete(&a, partial) == 0,
                          "decode-invariant: final partial is the whitelist tool");
}

static void test_schema_keys(void) {
    char   keys[4][GEIST_AGENT_NAME_CAP];
    size_t n;

    n = agent_schema_keys("{\"query\": string}", 4, keys);
    fails += geist_expect(n == 1 && strcmp(keys[0], "query") == 0, "schema: single key");

    n = agent_schema_keys("{\"url\": string, \"limit\": int}", 4, keys);
    fails += geist_expect(n == 2 && strcmp(keys[0], "url") == 0 && strcmp(keys[1], "limit") == 0,
                          "schema: two keys in order");

    n = agent_schema_keys("{}", 4, keys);
    fails += geist_expect(n == 0, "schema: no keys");
}

static void test_args_normalize(void) {
    char args[GEIST_AGENT_ARGS_CAP];

    /* wrong key, right value -> re-keyed to the schema key */
    snprintf(args, sizeof args, "{\"contents\":\"how long is the warranty\"}");
    fails += geist_expect(agent_args_normalize("{\"query\": string}", sizeof args, args) == 1 &&
                                  strcmp(args, "{\"query\":\"how long is the warranty\"}") == 0,
                          "normalize: re-keys a mis-keyed string arg");

    /* already the schema key -> untouched */
    snprintf(args, sizeof args, "{\"query\":\"rent\"}");
    agent_args_normalize("{\"query\": string}", sizeof args, args);
    fails += geist_expect(strcmp(args, "{\"query\":\"rent\"}") == 0,
                          "normalize: happy path untouched");

    /* a value with a quote survives via escaping */
    snprintf(args, sizeof args, "{\"q\":\"say \\\"hi\\\"\"}");
    agent_args_normalize("{\"query\": string}", sizeof args, args);
    fails += geist_expect(strcmp(args, "{\"query\":\"say \\\"hi\\\"\"}") == 0,
                          "normalize: escapes quotes in the re-keyed value");

    /* empty schema -> nothing to enforce, args left as-is */
    snprintf(args, sizeof args, "{\"x\":\"y\"}");
    fails += geist_expect(agent_args_normalize("{}", sizeof args, args) == 1,
                          "normalize: empty schema is a no-op");

    /* multi-key schema -> left untouched (ambiguous to re-key) */
    snprintf(args, sizeof args, "{\"wrong\":\"v\"}");
    fails += geist_expect(
            agent_args_normalize("{\"a\": string, \"b\": string}", sizeof args, args) == 0,
            "normalize: multi-key schema is left to the model");
}

static void test_tail_loop(void) {
    /* No repetition -> 0 (a real answer / one-line tool call is left intact). */
    fails += geist_expect(agent_tail_loop("The warranty is 24 months.", 26) == 0,
                          "loop: clean text is not a loop");
    fails += geist_expect(
            agent_tail_loop("{\"tool\":\"doc_search\",\"args\":{\"query\":\"x\"}}", 41) == 0,
            "loop: a valid tool call is not a loop");

    /* Three identical consecutive >=3-byte blocks -> detected, returns P. */
    const char *r = "ans [File: a.pdf][File: a.pdf][File: a.pdf]";
    fails += geist_expect(agent_tail_loop(r, strlen(r)) == strlen("[File: a.pdf]"),
                          "loop: triple-repeated block detected (P = block len)");

    /* Two repeats is not enough (avoids over-eager cutting). */
    const char *r2 = "ans [File: a.pdf][File: a.pdf]";
    fails += geist_expect(agent_tail_loop(r2, strlen(r2)) == 0, "loop: two repeats is not a loop");

    /* Short period: "ababab" -> P=2 is below the min(3); "xyzxyzxyz" -> P=3. */
    fails += geist_expect(agent_tail_loop("ababab", 6) == 0, "loop: period<3 ignored (e.g. '...')");
    fails += geist_expect(agent_tail_loop("xyzxyzxyz", 9) == 3, "loop: period-3 triple detected");
}

static void test_degenerate(void) {
    fails += geist_expect(agent_answer_degenerate("{"), "degenerate: lone brace");
    fails += geist_expect(agent_answer_degenerate("  {} \n"), "degenerate: punctuation only");
    fails += geist_expect(agent_answer_degenerate(""), "degenerate: empty");
    fails += geist_expect(!agent_answer_degenerate("error"), "degenerate: a real word is not");
    fails += geist_expect(!agent_answer_degenerate("notes.txt\nreport.pdf"),
                          "degenerate: a listing is not");
    fails += geist_expect(!agent_answer_degenerate("ok"), "degenerate: two letters is enough");
}

static void test_locator(void) {
    fails += geist_expect(agent_key_is_locator("path") && agent_key_is_locator("url") &&
                                  !agent_key_is_locator("query"),
                          "locator: path/url yes, query no");

    char        out[512];
    size_t      n;
    const char *r1 = "Fasse die Datei /tmp/note.txt zusammen";
    n              = agent_extract_locator(strlen(r1), r1, sizeof out, out);
    fails += geist_expect(n == strlen("/tmp/note.txt") && strcmp(out, "/tmp/note.txt") == 0,
                          "locator: lifts the path verbatim (no space mangling)");

    const char *r2 = "fetch http://example.com/p now";
    n              = agent_extract_locator(strlen(r2), r2, sizeof out, out);
    fails += geist_expect(strcmp(out, "http://example.com/p") == 0, "locator: lifts a URL");

    const char *r3 = "show me the current directory";
    fails += geist_expect(agent_extract_locator(strlen(r3), r3, sizeof out, out) == 0,
                          "locator: no slash word -> none (free-decode fallback)");
}

static void test_route_tiebreak(void) {
    /* names-a-file: an extension word (even without a slash) is detected; a bare
     * directory request / path is not. */
    const char *f1 = "Fasse die Datei note.txt zusammen";
    const char *f2 = "Zeige mir den Inhalt des aktuellen Ordners";
    const char *f3 = "list /home/germar/docs";
    fails +=
            geist_expect(agent_request_names_file(strlen(f1), f1), "names_file: note.txt detected");
    fails += geist_expect(!agent_request_names_file(strlen(f2), f2),
                          "names_file: no extension -> no");
    fails += geist_expect(!agent_request_names_file(strlen(f3), f3),
                          "names_file: bare dir path -> no (no extension)");

    fails += geist_expect(agent_desc_is_dir("die Dateien in einem Verzeichnis auflisten"),
                          "desc_is_dir: Verzeichnis tool");
    fails += geist_expect(!agent_desc_is_dir("eine Textdatei lesen und zusammenfassen"),
                          "desc_is_dir: file tool is not a dir tool");
}

int main(void) {
    test_matchers();
    test_schema_keys();
    test_args_normalize();
    test_tail_loop();
    test_degenerate();
    test_locator();
    test_route_tiebreak();
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("agent grammar: name matchers + schema keys + args re-keying + loop-stop pass\n");
    return GEIST_TEST_PASS;
}

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
                          "locator: no path-like word -> none");

    /* a bare filename with an extension (no slash) is a locator too */
    const char *r4 = "Fasse die Datei note.txt zusammen";
    n              = agent_extract_locator(strlen(r4), r4, sizeof out, out);
    fails += geist_expect(strcmp(out, "note.txt") == 0, "locator: lifts a bare filename.ext");

    const char *r5 = "summarize wm2026_de.txt please";
    n              = agent_extract_locator(strlen(r5), r5, sizeof out, out);
    fails += geist_expect(strcmp(out, "wm2026_de.txt") == 0, "locator: lifts filename with digits");

    /* slug-shaped words (mind.h note names) are locators too — but a real
     * path/extension word always wins over a slug-shaped one */
    fails += geist_expect(agent_key_is_locator("slug"), "locator: slug is a locator key");
    const char *r6 = "Rufe die Notiz pi-serial ab";
    n              = agent_extract_locator(strlen(r6), r6, sizeof out, out);
    fails += geist_expect(strcmp(out, "pi-serial") == 0, "locator: lifts a slug word");
    const char *r7 = "summarize the on-device report.md";
    n              = agent_extract_locator(strlen(r7), r7, sizeof out, out);
    fails += geist_expect(strcmp(out, "report.md") == 0, "locator: extension beats slug shape");
    const char *r8 = "Rufe die Notiz ab";
    n              = agent_extract_locator(strlen(r8), r8, sizeof out, out);
    fails += geist_expect(n == 0, "locator: no slug word -> nothing lifted");
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

    /* has-url: a literal http(s):// scheme is detected; a bare domain or a
     * mid-word "http" is not. */
    const char *u1 = "Fetch https://example.com/bitnet.html and tell me what it says";
    const char *u2 = "Hole http://example.com/kv.html";
    const char *u3 = "Search the web for BitNet ternary models";
    const char *u4 = "is httpd a web server?";
    fails += geist_expect(agent_request_has_url(strlen(u1), u1), "has_url: https:// detected");
    fails += geist_expect(agent_request_has_url(strlen(u2), u2), "has_url: http:// detected");
    fails += geist_expect(!agent_request_has_url(strlen(u3), u3), "has_url: no scheme -> no");
    fails += geist_expect(!agent_request_has_url(strlen(u4), u4), "has_url: httpd -> no");

    fails += geist_expect(agent_schema_wants_url("{\"url\": string}"), "wants_url: url schema");
    fails += geist_expect(!agent_schema_wants_url("{\"query\": string}"),
                          "wants_url: query schema -> no");

    /* has-slug: a slug-shaped word is detected; plain words and file names not */
    const char *s1 = "Lade die gespeicherte Notiz pi-serial";
    const char *s2 = "Recall the note";
    const char *s3 = "Fasse notes.txt zusammen";
    fails += geist_expect(agent_request_has_slug(strlen(s1), s1), "has_slug: pi-serial");
    fails += geist_expect(!agent_request_has_slug(strlen(s2), s2), "has_slug: no slug -> no");
    fails += geist_expect(!agent_request_has_slug(strlen(s3), s3), "has_slug: filename -> no");
    fails += geist_expect(agent_schema_wants_slug("{\"slug\": string}") &&
                                  !agent_schema_wants_slug("{\"text\": string}"),
                          "wants_slug: slug vs text schema");

    /* path-word + memory evidence detectors */
    const char *p1 = "List the files in the folder tests/data/docs";
    const char *p2 = "Fetch https://example.com/kv.html please";
    fails += geist_expect(agent_request_has_pathword(strlen(p1), p1), "pathword: slash path");
    fails += geist_expect(!agent_request_has_pathword(strlen(p2), p2), "pathword: URL is not one");
    fails += geist_expect(agent_schema_wants_path("{\"path\": string}") &&
                                  !agent_schema_wants_path("{\"query\": string}"),
                          "wants_path: path vs query schema");

    const char *m1 = "Merke dir: die Seriennummer ist 4242";
    const char *m2 = "Recall the note pi-serial";
    const char *m3 = "What is 2 plus 2?";
    fails += geist_expect(agent_request_mentions_memory(strlen(m1), m1), "memory: Merke dir");
    fails += geist_expect(agent_request_mentions_memory(strlen(m2), m2), "memory: recall note");
    fails += geist_expect(!agent_request_mentions_memory(strlen(m3), m3), "memory: plain -> no");
    const char *m4 = "Fasse notes.txt zusammen";
    fails += geist_expect(!agent_request_mentions_memory(strlen(m4), m4),
                          "memory: a filename is not memory evidence");
    const char *m5 = "Please run rm -rf on my home directory";
    fails += geist_expect(agent_request_is_destructive(strlen(m5), m5),
                          "destructive: rm anywhere in the request");
    const char *m6 = "inform me about the farm today";
    fails += geist_expect(!agent_request_is_destructive(strlen(m6), m6),
                          "destructive: 'farm'/'inform' are not rm");
    struct geist_tool mt = {.name = "recall", .description = "eine gespeicherte Notiz laden"};
    struct geist_tool nt = {.name = "doc_search", .description = "die lokalen Dokumente"};
    fails += geist_expect(agent_tool_is_memory(&mt) && !agent_tool_is_memory(&nt),
                          "tool_is_memory: recall vs doc_search");

    /* stocks evidence + tool detection (the stock_movers gate) */
    const char *k1 = "Welche Aktie hat heute am besten performt?";
    const char *k2 = "Which stocks lost the most today?";
    const char *k3 = "Fasse notes.txt zusammen";
    fails += geist_expect(agent_request_mentions_stocks(strlen(k1), k1), "stocks: Aktie");
    fails += geist_expect(agent_request_mentions_stocks(strlen(k2), k2), "stocks: stocks");
    fails += geist_expect(!agent_request_mentions_stocks(strlen(k3), k3),
                          "stocks: summarize file -> no");
    struct geist_tool st = {.name        = "stock_movers",
                            .description = "die Aktien-Tagesgewinner oder -verlierer abrufen"};
    fails += geist_expect(agent_tool_is_stocks(&st) && !agent_tool_is_stocks(&nt),
                          "tool_is_stocks: stock_movers vs doc_search");

    /* mentions-docs: a docs/Dokumente word at a word start is detected; requests
     * without one (or with doc only mid-word) are not. */
    const char *d1 = "Search the docs for kv cache quantization";
    const char *d2 = "Wo steht in den Dokumenten etwas zu NEON Kerneln?";
    const char *d3 = "Search for kv cache";
    const char *d4 = "run maxdoc now";
    fails += geist_expect(agent_request_mentions_docs(strlen(d1), d1), "mentions_docs: docs");
    fails += geist_expect(agent_request_mentions_docs(strlen(d2), d2), "mentions_docs: Dokumenten");
    fails += geist_expect(!agent_request_mentions_docs(strlen(d3), d3),
                          "mentions_docs: no doc word -> no");
    fails += geist_expect(!agent_request_mentions_docs(strlen(d4), d4),
                          "mentions_docs: mid-word doc -> no");

    struct geist_tool dt = {.name        = "doc_search",
                            .description = "die lokalen Dokumente nach einer Anfrage durchsuchen"};
    struct geist_tool lt = {.name        = "list_dir",
                            .description = "die Dateien in einem Verzeichnis auflisten"};
    fails += geist_expect(agent_tool_is_docs(&dt), "tool_is_docs: doc_search");
    fails += geist_expect(!agent_tool_is_docs(&lt), "tool_is_docs: list_dir -> no");

    /* destructive-verb guard: an opening imperative (optionally after one
     * politeness word) is detected; mid-request destructive words are not. */
    const char *x1 = "Delete report.md";
    const char *x2 = "Bitte lösche notes.txt";
    const char *x3 = "rm -rf everything";
    const char *x4 = "Search the docs for how to remove noise";
    const char *x5 = "List the files in the current directory";
    fails += geist_expect(agent_request_is_destructive(strlen(x1), x1), "destructive: Delete");
    fails +=
            geist_expect(agent_request_is_destructive(strlen(x2), x2), "destructive: Bitte lösche");
    fails += geist_expect(agent_request_is_destructive(strlen(x3), x3), "destructive: rm");
    fails += geist_expect(!agent_request_is_destructive(strlen(x4), x4),
                          "destructive: mid-request remove -> no");
    fails += geist_expect(!agent_request_is_destructive(strlen(x5), x5), "destructive: List -> no");

    struct geist_tool  del = {.name = "delete_file", .description = "eine Datei löschen"};
    struct geist_agent ad  = {.tools = &del, .n_tools = 1};
    struct geist_agent al  = {.tools = &lt, .n_tools = 1};
    fails += geist_expect(agent_tools_cover_destruction(&ad), "cover_destruction: delete_file");
    fails += geist_expect(!agent_tools_cover_destruction(&al), "cover_destruction: list_dir -> no");
}

static void test_recipes(void) {
    struct geist_tool tools[] = {
            {.name = "web_search", .args_schema = "{\"query\": string}"},
            {.name = "web_fetch", .args_schema = "{\"url\": string}"},
            {.name = "doc_search", .args_schema = "{\"query\": string}"},
            {.name = "summarize_file", .args_schema = "{\"path\": string}"},
    };
    static struct geist_agent a;
    a.tools   = tools;
    a.n_tools = sizeof tools / sizeof *tools;

    /* a cue word continues the routed tool into the recipe's second step */
    const char *r1 = "Find a page about BitNet on the web and read it to me";
    const char *r2 = "Suche im Web nach BitNet 2B und lies die erste Trefferseite";
    const char *r3 = "Search the web for BitNet ternary models"; /* no cue */
    const char *r4 = "Which doc covers kv cache? Find it and summarize that file.";
    fails += geist_expect(agent_recipe_next(&a, 0, strlen(r1), r1) == 1,
                          "recipe: search+read -> web_fetch");
    fails += geist_expect(agent_recipe_next(&a, 0, strlen(r2), r2) == 1,
                          "recipe: German lies -> web_fetch");
    fails += geist_expect(agent_recipe_next(&a, 0, strlen(r3), r3) == -1,
                          "recipe: no cue -> single-shot");
    const char *r5 = "Suche den Artikel im Web und fasse zusammen was drinsteht";
    fails += geist_expect(agent_recipe_next(&a, 0, strlen(r5), r5) == 1,
                          "recipe: web+fasse -> web_fetch");
    fails += geist_expect(agent_recipe_next(&a, 2, strlen(r4), r4) == 3,
                          "recipe: doc_search+summarize -> summarize_file");
    fails += geist_expect(agent_recipe_next(&a, 3, strlen(r4), r4) == -1,
                          "recipe: summarize_file is no recipe start");

    /* step-1 locator lifted from the step-0 observation */
    char        out[256];
    const char *hits = "1. BitNet 2B4T — https://example.com/bitnet.html\n"
                       "2. KV cache — https://example.com/kv.html\n";
    fails += geist_expect(
            agent_obs_locator(1, strlen(hits), hits, strlen(r1), r1, sizeof out, out) > 0 &&
                    strcmp(out, "https://example.com/bitnet.html") == 0,
            "obs_locator: first URL is the top hit");

    /* doc_search hits are already query-filtered — the FIRST hit is the pick;
     * the surrounding [brackets] of the hit format are stripped. */
    const char *para     = "[notes.txt] NEON kernels on the A76\n"
                           "[kv-cache.md] The KV cache is stored packed 4-bit\n";
    const char *req_neon = "Finde das Dokument ueber NEON und fasse es zusammen";
    fails += geist_expect(
            agent_obs_locator(0, strlen(para), para, strlen(req_neon), req_neon, sizeof out, out) >
                            0 &&
                    strcmp(out, "notes.txt") == 0,
            "obs_locator: first bracketed hit, brackets stripped");

    const char *listing = "report.md\nnotes.txt\ntodo.txt\n";
    const char *req_rep = "Look at the files in this folder, then summarize the report";
    fails += geist_expect(
            agent_obs_locator(
                    0, strlen(listing), listing, strlen(req_rep), req_rep, sizeof out, out) > 0 &&
                    strcmp(out, "report.md") == 0,
            "obs_locator: listing stem match picks report.md");

    const char *nohit = "no matches for \"xyz\"";
    fails += geist_expect(
            agent_obs_locator(0, strlen(nohit), nohit, strlen(r4), r4, sizeof out, out) == 0,
            "obs_locator: nothing liftable -> 0");
}

static void test_home_detectors(void) {
    /* home evidence: German COMPOUNDS carry the noun mid-word -> substring */
    const char *c1 = "Mach das Flurlicht an";
    const char *c2 = "Schalte die Wohnzimmerlampe ein";
    const char *c3 = "Was ist 2 plus 2?";
    fails += geist_expect(agent_request_mentions_home(strlen(c1), c1),
                          "home: compound Flurlicht matches");
    fails += geist_expect(agent_request_mentions_home(strlen(c2), c2),
                          "home: compound Wohnzimmerlampe matches");
    fails += geist_expect(!agent_request_mentions_home(strlen(c3), c3),
                          "home: plain math -> no evidence");

    /* sentence shape: imperative vs question, filler skipped */
    const char *i1 = "Dimme das Licht auf 40 Prozent";
    const char *i2 = "Und jetzt schliesse ihn wieder";
    const char *i3 = "Unlock the front door";
    const char *q1 = "Ist das Fenster im Bad offen?";
    const char *q2 = "Wie warm ist es im Bad?";
    fails += geist_expect(agent_request_is_imperative(strlen(i1), i1), "shape: Dimme imperative");
    fails += geist_expect(agent_request_is_imperative(strlen(i2), i2),
                          "shape: filler skipped, schliesse imperative");
    fails += geist_expect(agent_request_is_imperative(strlen(i3), i3), "shape: unlock imperative");
    fails += geist_expect(!agent_request_is_imperative(strlen(q1), q1),
                          "shape: question is not imperative");
    fails += geist_expect(agent_request_is_question(strlen(q1), q1), "shape: Ist question");
    fails += geist_expect(agent_request_is_question(strlen(q2), q2), "shape: Wie question");
    fails += geist_expect(!agent_request_is_question(strlen(i1), i1),
                          "shape: imperative is not a question");

    /* home tool predicates split on the name */
    struct geist_tool cmd = {.name        = "control_device",
                             .description = "ein Hausger\u00e4t schalten oder stellen"};
    struct geist_tool st  = {.name        = "home_status",
                             .description = "den Zustand eines Ger\u00e4ts abfragen"};
    fails += geist_expect(agent_tool_is_home(&cmd) && agent_tool_is_home(&st),
                          "home: both tools detected as home");
    fails += geist_expect(agent_pred_home_command(&cmd) && !agent_pred_home_command(&st),
                          "home: command predicate");
    fails += geist_expect(agent_pred_home_status(&st) && !agent_pred_home_status(&cmd),
                          "home: status predicate");
}

int main(void) {
    test_matchers();
    test_schema_keys();
    test_args_normalize();
    test_tail_loop();
    test_degenerate();
    test_locator();
    test_route_tiebreak();
    test_recipes();
    test_home_detectors();
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("agent grammar: name matchers + schema keys + args re-keying + loop-stop pass\n");
    return GEIST_TEST_PASS;
}

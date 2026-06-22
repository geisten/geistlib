/*
 * test_agent_unit — the agent's deterministic parts, no model.
 *
 * Covers the tool-call parser (valid, fenced/prose-wrapped, no-call, missing
 * args), the whitelist gate (agent_find), and the doc_search tool over a
 * throwaway doc dir. The model-driven run loop is exercised in the _int test.
 * No assert() — checks set a flag, the exit code carries PASS/FAIL.
 */
#define _POSIX_C_SOURCE 200809L

#include "test_helpers.h"

#include "../tools/agent.h"
#include "../tools/agent_docsearch.h"

#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#define DOC_DIR "./.agent_unit_test"

static int  fails = 0;
static void test_parser(void) {
    char name[GEIST_AGENT_NAME_CAP], args[GEIST_AGENT_ARGS_CAP];

    const char *clean = "{\"tool\":\"doc_search\",\"args\":{\"query\":\"rent\"}}";
    fails += geist_expect(
            agent_parse_call(strlen(clean), clean, sizeof name, name, sizeof args, args) == 1,
            "parse: finds a clean call");
    fails += geist_expect(strcmp(name, "doc_search") == 0, "parse: extracts the tool name");
    fails += geist_expect(strstr(args, "\"query\"") != nullptr, "parse: extracts the args object");

    /* small models wrap calls in prose / ```json fences — must still parse */
    const char *fenced = "Sure, let me look.\n```json\n"
                         "{\"tool\":\"doc_search\", \"args\": {\"query\":\"x\"}}\n```";
    fails += geist_expect(
            agent_parse_call(strlen(fenced), fenced, sizeof name, name, sizeof args, args) == 1,
            "parse: tolerates prose + fences");
    fails += geist_expect(strcmp(name, "doc_search") == 0, "parse: name through the fence");

    /* a plain-text answer is NOT a tool call -> final answer */
    const char *answer = "The rent is due on the first of the month.";
    fails += geist_expect(
            agent_parse_call(strlen(answer), answer, sizeof name, name, sizeof args, args) == 0,
            "parse: plain text is not a call");

    /* missing args -> name parses, args defaults to {} */
    const char *noargs = "{\"tool\":\"now\"}";
    fails += geist_expect(
            agent_parse_call(strlen(noargs), noargs, sizeof name, name, sizeof args, args) == 1,
            "parse: call without args");
    fails += geist_expect(strcmp(name, "now") == 0 && strcmp(args, "{}") == 0,
                          "parse: args defaults to {}");
}

static void test_whitelist(void) {
    struct geist_tool         tools[] = {docsearch_tool(DOC_DIR)};
    static struct geist_agent ag; /* large struct -> static, not stack */
    ag.tools   = tools;
    ag.n_tools = 1;
    fails += geist_expect(agent_find(&ag, "doc_search") != nullptr, "whitelist: known tool found");
    fails += geist_expect(agent_find(&ag, "rm_rf") == nullptr, "whitelist: unknown tool rejected");
    fails += geist_expect(agent_find(&ag, "doc_searc") == nullptr, "whitelist: no prefix match");
}

static void test_docsearch(void) {
    mkdir(DOC_DIR, 0755);
    FILE *f = fopen(DOC_DIR "/bgb.txt", "w");
    if (f) {
        fputs("§ 573c Kuendigungsfristen\n"
              "Die Kuendigung ist spaetestens am dritten Werktag zulaessig.\n"
              "Ein unkuendbarer Mietvertrag ist die Ausnahme.\n",
              f);
        fclose(f);
    }

    char   out[GEIST_AGENT_OBS_CAP];
    size_t n = 0;
    /* a hit, case-insensitive */
    enum geist_status st = docsearch_invoke((void *) (intptr_t) DOC_DIR,
                                            strlen("{\"query\":\"UNKUENDBAR\"}"),
                                            "{\"query\":\"UNKUENDBAR\"}",
                                            sizeof out,
                                            out,
                                            &n);
    fails += geist_expect(st == GEIST_OK, "docsearch: returns OK");
    fails += geist_expect(strstr(out, "unkuendbar") != nullptr,
                          "docsearch: finds the matching line");
    fails += geist_expect(strstr(out, "bgb.txt") != nullptr, "docsearch: tags the source file");

    /* no hit */
    docsearch_invoke((void *) (intptr_t) DOC_DIR,
                     strlen("{\"query\":\"zebra\"}"),
                     "{\"query\":\"zebra\"}",
                     sizeof out,
                     out,
                     &n);
    fails += geist_expect(strstr(out, "no matches") != nullptr, "docsearch: reports no matches");

    /* missing query field */
    docsearch_invoke((void *) (intptr_t) DOC_DIR, strlen("{}"), "{}", sizeof out, out, &n);
    fails += geist_expect(strstr(out, "missing") != nullptr, "docsearch: flags a missing query");

    remove(DOC_DIR "/bgb.txt");
    remove(DOC_DIR);
}

int main(void) {
    test_parser();
    test_whitelist();
    test_docsearch();
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("agent: parser + whitelist + doc_search pass\n");
    return GEIST_TEST_PASS;
}

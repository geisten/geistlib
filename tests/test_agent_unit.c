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

static int               fails = 0;
static enum geist_status counted_tool(void      *ctx,
                                      size_t     args_len,
                                      const char args[static args_len],
                                      size_t     out_cap,
                                      char       out[static out_cap],
                                      size_t    *out_len) {
    (void) args_len;
    (void) args;
    (*(unsigned *) ctx)++;
    return agent_obs(out_cap, out, out_len, "executed");
}

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

    const char *nested = "{\"tool\":\"complex\",\"args\":{\"text\":\"keep } here\","
                         "\"target\":{\"ids\":[\"a\",\"b\"]}}}";
    fails += geist_expect(
            agent_parse_call(strlen(nested), nested, sizeof name, name, sizeof args, args) == 1 &&
                    strcmp(name, "complex") == 0 && strstr(args, "keep } here") != nullptr &&
                    strstr(args, "\"ids\"") != nullptr,
            "parse: nested args and quoted braces remain intact");

    const char *decoy = "{\"wrapper\":{\"tool\":\"rm_rf\",\"args\":{}}}";
    fails += geist_expect(
            agent_parse_call(strlen(decoy), decoy, sizeof name, name, sizeof args, args) == 0,
            "parse: nested tool decoy is not a call");

    const char *duplicate = "{\"tool\":\"safe\",\"tool\":\"unsafe\",\"args\":{}}";
    fails += geist_expect(
            agent_parse_call(strlen(duplicate), duplicate, sizeof name, name, sizeof args, args) ==
                    0,
            "parse: duplicate call keys fail closed");
}

static void test_dynamic_dispatch_gate(void) {
    unsigned          calls = 0u;
    struct geist_tool tool  = {
            .name        = "SetLevel",
            .args_schema = "{\"level\":integer}",
            .description = "Set a level",
            .parameters_schema =
                    "{\"type\":\"object\",\"properties\":{\"level\":{\"type\":\"integer\","
                    "\"minimum\":0,\"maximum\":100}},\"required\":[\"level\"]}",
            .invoke = counted_tool,
            .ctx    = &calls,
    };
    char        observation[256];
    size_t      observation_len = 0u;
    const char *invalid         = "{\"level\":101}";
    fails += geist_expect(agent_tool_invoke_checked(&tool,
                                                    strlen(invalid),
                                                    invalid,
                                                    sizeof observation,
                                                    observation,
                                                    &observation_len) == GEIST_OK &&
                                  calls == 0u && strstr(observation, "range") != nullptr,
                          "dynamic dispatch: invalid args never reach host callback");
    const char *valid = "{\"level\":42}";
    fails += geist_expect(agent_tool_invoke_checked(&tool,
                                                    strlen(valid),
                                                    valid,
                                                    sizeof observation,
                                                    observation,
                                                    &observation_len) == GEIST_OK &&
                                  calls == 1u && strcmp(observation, "executed") == 0,
                          "dynamic dispatch: valid args invoke exactly once");
}

static void test_whitelist(void) {
    struct geist_tool         tools[] = {docsearch_tool(DOC_DIR)};
    static struct geist_agent ag; /* large struct -> static, not stack */
    ag.tools   = tools;
    ag.n_tools = 1;
    fails += geist_expect(agent_find(&ag, "doc_search") != nullptr, "whitelist: known tool found");
    fails += geist_expect(agent_find(&ag, "rm_rf") == nullptr, "whitelist: unknown tool rejected");
    fails += geist_expect(agent_find(&ag, "doc_searc") == nullptr, "whitelist: no prefix match");
    ag.n_tools = 0u;
    fails += geist_expect(agent_select_tool(&ag, 1u, "x") == -1,
                          "routing: empty offered set replies without a call");
    const float         scores[]  = {1.0f, 0.8f, 0.9f};
    const unsigned char allowed[] = {0u, 0u};
    fails += geist_expect(!agent_route_confident(2u, 0, scores, allowed, 0.35f),
                          "routing: close tool/reply race requests clarification");
    const float clear_scores[] = {1.5f, 0.8f, 0.9f};
    fails += geist_expect(agent_route_confident(2u, 0, clear_scores, allowed, 0.35f),
                          "routing: clear winner may execute");
}

static void run_search(const char *query, char *out, size_t cap) {
    char   req[512];
    int    k = snprintf(req, sizeof req, "{\"query\":\"%s\"}", query);
    size_t n = 0;
    docsearch_invoke((void *) (intptr_t) DOC_DIR, (size_t) k, req, cap, out, &n);
}

static void test_docsearch(void) {
    mkdir(DOC_DIR, 0755);
    /* three blank-line-separated paragraphs: the answer is in para 2/3, not 1. */
    FILE *f = fopen(DOC_DIR "/bgb.txt", "w");
    if (f) {
        fputs("# Mietrecht\nAllgemeine Hinweise zum Wohnraum.\n\n"
              "Die Kuendigungsfrist betraegt drei Monate fuer den Mieter.\n\n"
              "Ein unkuendbarer Mietvertrag ist die Ausnahme.\n",
              f);
        fclose(f);
    }

    char out[GEIST_AGENT_OBS_CAP];

    /* single keyword (case-insensitive) -> the paragraph that contains it */
    run_search("UNKUENDBAR", out, sizeof out);
    fails += geist_expect(strstr(out, "unkuendbarer") != nullptr, "docsearch: single-word hit");
    fails += geist_expect(strstr(out, "bgb.txt") != nullptr, "docsearch: tags the source file");

    /* a whole QUESTION: shares warranty-style key terms with para 2 but also
     * carries interrogatives (wie/lang) the answer lacks -> overlap score still
     * clears the threshold (this is the line-AND regression this fixes). */
    run_search("wie lang ist die Kuendigungsfrist fuer den Mieter", out, sizeof out);
    fails += geist_expect(strstr(out, "drei Monate") != nullptr,
                          "docsearch: question hits answer paragraph");

    /* returns the matching paragraph, not paragraph 1 */
    fails += geist_expect(strstr(out, "Allgemeine Hinweise") == nullptr,
                          "docsearch: returns the relevant paragraph, not the first");

    run_search("zebra giraffe", out, sizeof out);
    fails += geist_expect(strstr(out, "no matches") != nullptr, "docsearch: reports no matches");

    /* one key term present but below threshold (2) -> no match */
    run_search("Mietvertrag zebra", out, sizeof out);
    fails += geist_expect(strstr(out, "no matches") != nullptr,
                          "docsearch: below-threshold -> miss");

    /* missing query field */
    docsearch_invoke((void *) (intptr_t) DOC_DIR, strlen("{}"), "{}", sizeof out, out, nullptr);
    fails += geist_expect(strstr(out, "missing") != nullptr, "docsearch: flags a missing query");

    remove(DOC_DIR "/bgb.txt");
    remove(DOC_DIR);
}

int main(void) {
    test_parser();
    test_dynamic_dispatch_gate();
    test_whitelist();
    test_docsearch();
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("agent: parser + whitelist + doc_search pass\n");
    return GEIST_TEST_PASS;
}

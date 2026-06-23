/*
 * test_agent_grammar_unit — the whitelist-grammar matchers, no model.
 *
 * agent_name_is_prefix / agent_name_complete are the pure core of the
 * constrained tool-name decode (item 3): they decide, for the chars emitted so
 * far, whether the partial still matches a whitelist name and whether it is a
 * complete one. The model-driven decode (agent_decode_name_constrained) is
 * exercised in the _int test; here we pin the matcher invariants that guarantee
 * the decode can only ever spell a whitelisted name. No assert() — checks set a
 * flag, the exit code carries PASS/FAIL.
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

int main(void) {
    test_matchers();
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("agent grammar: prefix + complete matchers pass\n");
    return GEIST_TEST_PASS;
}

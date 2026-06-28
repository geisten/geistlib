/*
 * test_agent_main_unit — the agent CLI engine's deterministic parts, no model.
 * Covers agent_main_parse_args (happy paths + every reject: no model, unknown
 * flag, -n without/invalid value, extra positional, help) and the per-app
 * system-prompt threading into agent_system_prompt. No assert() — exit codes.
 */
#include "test_helpers.h"

#include "../tools/agent.h"
#include "../tools/agent_main.h"

#include <stdio.h>
#include <string.h>

static int fails = 0;

/* A trivial tool so the whitelist/prompt have something to list (no POSIX). */
static enum geist_status stub_invoke(void      *ctx,
                                     size_t     alen,
                                     const char a[static alen],
                                     size_t     ocap,
                                     char       o[static ocap],
                                     size_t    *olen) {
    (void) ctx;
    (void) alen;
    (void) a;
    size_t n = (size_t) snprintf(o, ocap, "stub");
    if (olen) {
        *olen = n;
    }
    return GEIST_OK;
}
static const struct geist_tool STUB = {
        .name = "echo", .args_schema = "{\"x\": string}", .invoke = stub_invoke, .ctx = nullptr};

/* Parse a fixed argv array (no GNU statement-expressions -> -Wpedantic clean). */
static enum agent_main_parse parse(struct agent_main_opts *o, int argc, char **argv) {
    return agent_main_parse_args(argc, argv, /*want_model=*/true, o);
}

static void test_parse(void) {
    struct agent_main_opts o;

    char *a_model[] = {"p", "m.gguf"};
    fails += geist_expect(parse(&o, 2, a_model) == AGENT_MAIN_RUN && o.model &&
                                  strcmp(o.model, "m.gguf") == 0 && o.question == nullptr &&
                                  o.max_steps == 0,
                          "parse: model only -> RUN, REPL");

    char *a_q[] = {"p", "m.gguf", "what?"};
    fails += geist_expect(parse(&o, 3, a_q) == AGENT_MAIN_RUN && o.question &&
                                  strcmp(o.question, "what?") == 0,
                          "parse: model + question");

    char *a_n[] = {"p", "m.gguf", "-n", "5"};
    fails += geist_expect(parse(&o, 4, a_n) == AGENT_MAIN_RUN && o.max_steps == 5,
                          "parse: -n sets max_steps");
    char *a_nfirst[] = {"p", "-n", "3", "m.gguf"};
    fails += geist_expect(parse(&o, 4, a_nfirst) == AGENT_MAIN_RUN && o.max_steps == 3,
                          "parse: -n before positional");

    char *a_h[]  = {"p", "-h"};
    char *a_hh[] = {"p", "--help"};
    fails += geist_expect(parse(&o, 2, a_h) == AGENT_MAIN_HELP, "parse: -h -> HELP");
    fails += geist_expect(parse(&o, 2, a_hh) == AGENT_MAIN_HELP, "parse: --help -> HELP");

    char *a_none[]  = {"p"};
    char *a_bad[]   = {"p", "--bogus", "m.gguf"};
    char *a_noval[] = {"p", "m.gguf", "-n"};
    char *a_zero[]  = {"p", "m.gguf", "-n", "0"};
    char *a_nan[]   = {"p", "m.gguf", "-n", "x"};
    char *a_extra[] = {"p", "m.gguf", "q", "extra"};
    fails += geist_expect(parse(&o, 1, a_none) == AGENT_MAIN_BADARGS, "parse: no model -> BADARGS");
    fails += geist_expect(parse(&o, 3, a_bad) == AGENT_MAIN_BADARGS,
                          "parse: unknown flag -> BADARGS");
    fails += geist_expect(parse(&o, 3, a_noval) == AGENT_MAIN_BADARGS,
                          "parse: -n without value -> BADARGS");
    fails += geist_expect(parse(&o, 4, a_zero) == AGENT_MAIN_BADARGS, "parse: -n 0 -> BADARGS");
    fails += geist_expect(parse(&o, 4, a_nan) == AGENT_MAIN_BADARGS,
                          "parse: -n non-numeric -> BADARGS");
    fails += geist_expect(parse(&o, 4, a_extra) == AGENT_MAIN_BADARGS,
                          "parse: extra positional -> BADARGS");
}

static void test_system_prompt(void) {
    static struct geist_agent ag; /* large -> static, no model needed for this */
    ag.tools   = &STUB;
    ag.n_tools = 1;

    char buf[2048];

    ag.system_prompt = "ROLE_MARKER_42";
    agent_system_prompt(&ag, sizeof buf, buf);
    fails += geist_expect(strstr(buf, "ROLE_MARKER_42") != nullptr,
                          "system_prompt: custom role appears");
    fails += geist_expect(strstr(buf, "echo") != nullptr, "system_prompt: tool name listed");
    fails += geist_expect(strstr(buf, "{\"tool\"") != nullptr,
                          "system_prompt: tool-call format shown");

    ag.system_prompt = nullptr;
    agent_system_prompt(&ag, sizeof buf, buf);
    fails += geist_expect(strstr(buf, "task agent") != nullptr,
                          "system_prompt: nullptr -> default role");
}

int main(void) {
    test_parse();
    test_system_prompt();
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("agent_main: arg parse + system-prompt threading pass\n");
    return GEIST_TEST_PASS;
}

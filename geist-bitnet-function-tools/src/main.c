/*
 * main.c — geist-fn: BitNet b1.58 2B-4T + native C function tools.
 *
 * Loads the BitNet GGUF through the imported engine (geistenlib.a), wires the
 * function-tool whitelist (src/function_tools.h) into the header-only agent
 * layer, and runs one request through the bounded tool loop:
 *
 *   ./geist-fn -m models/bitnet-2b4t.i2_s.gguf "Add 5 and 7"
 *   ./geist-fn -m models/bitnet-2b4t.i2_s.gguf --free "What is geist?"
 *   ./geist-fn --selftest        # registry + schema validation, no model
 *
 * Forced calls are ON by default (--free disables them): BitNet 2B-4T is not
 * tool-trained, so routing + grammar-forcing is what makes calls reliable —
 * see docs/agent.md ("Tool selection & forced calls").
 */
#include "function_tools.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* One tool invocation through the SAME validation boundary the agent uses
 * (agent_tool_invoke_checked: dynamic-tools-v1 schema check, then the C
 * function). expect==nullptr means "must contain error". */
static int selftest_case(const struct geist_tool *tool, const char *args,
                         const char *expect) {
    char   out[GEIST_AGENT_OBS_CAP];
    size_t n  = 0;
    enum geist_status st =
            agent_tool_invoke_checked(tool, strlen(args), args, sizeof out, out, &n);
    const char *want = expect ? expect : "error";
    int ok = st == GEIST_OK && strstr(out, want) != nullptr;
    printf("  %-4s %s(%s) -> %s\n", ok ? "ok" : "FAIL", tool->name, args, out);
    return ok;
}

static int selftest(void) {
    printf("function-tool registry selftest (%d tools, no model)\n",
           (int) FUNCTION_TOOLS_N);
    int ok = 1;
    const struct geist_tool *calc = &FUNCTION_TOOLS[0];
    const struct geist_tool *clk  = &FUNCTION_TOOLS[1];
    const struct geist_tool *sys  = &FUNCTION_TOOLS[2];

    ok &= selftest_case(calc, "{\"op\":\"add\",\"a\":5,\"b\":7}", "12");
    ok &= selftest_case(calc, "{\"op\":\"multiply\",\"a\":-3,\"b\":2.5}", "-7.5");
    ok &= selftest_case(calc, "{\"op\":\"divide\",\"a\":1,\"b\":0}", nullptr);
    /* schema gate: "pow" violates the enum -> rejected BEFORE the C function */
    ok &= selftest_case(calc, "{\"op\":\"pow\",\"a\":2,\"b\":8}", nullptr);
    ok &= selftest_case(clk, "{}", "UTC");
    ok &= selftest_case(clk, "{\"zone\":\"local\"}", "local");
    ok &= selftest_case(sys, "{}", "cores");

    printf(ok ? "selftest: all cases passed\n" : "selftest: FAILURES above\n");
    return ok ? 0 : 1;
}

static int usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s --selftest\n"
            "       %s -m <model.gguf> [--steps N] [--free] \"<request>\"\n",
            argv0, argv0);
    return 2;
}

int main(int argc, char **argv) {
    const char *model_path = nullptr;
    const char *request    = nullptr;
    size_t      max_steps  = 0; /* 0 -> agent default (8) */
    bool        force      = true;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--selftest") == 0) {
            return selftest();
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            max_steps = (size_t) atoi(argv[++i]);
        } else if (strcmp(argv[i], "--free") == 0) {
            force = false;
        } else if (argv[i][0] != '-') {
            request = argv[i];
        } else {
            return usage(argv[0]);
        }
    }
    if (model_path == nullptr || request == nullptr) {
        return usage(argv[0]);
    }

    struct geist_backend *be = nullptr;
    if (geist_backend_create("auto", nullptr, nullptr, &be) != GEIST_OK) {
        fprintf(stderr, "backend_create failed: %s\n", geist_last_create_error());
        return 1;
    }
    struct geist_model *model = nullptr;
    if (geist_model_load(model_path, be, &model) != GEIST_OK) {
        fprintf(stderr, "model_load failed: %s\n", geist_last_create_error());
        geist_backend_destroy(be);
        return 1;
    }
    fprintf(stderr, "loaded %s (arch: %s, backend: %s)\n", model_path,
            geist_model_arch(model), geist_backend_name(be));

    struct geist_session_opts opts = {0}; /* greedy decode */
    struct geist_session     *sess = nullptr;
    if (geist_session_create(model, be, &opts, &sess) != GEIST_OK) {
        fprintf(stderr, "session_create failed\n");
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return 1;
    }

    /* The agent struct is large — static storage, not the stack. */
    static struct geist_agent agent;
    geist_agent_init(&agent, model, sess, FUNCTION_TOOLS_N, FUNCTION_TOOLS,
                     max_steps,
                     "You are a precise local assistant with function tools. "
                     "Use a tool when it answers the request; otherwise reply "
                     "directly.");
    agent.force_call             = force;
    agent.clarify_low_confidence = true;
    agent.on_event               = agent_event_print; /* live trace on stderr */
    agent.on_event_ctx           = stderr;

    char   resp[4096];
    size_t resp_len = 0;
    enum geist_status st = geist_agent_run(&agent, strlen(request), request,
                                           sizeof resp, resp, &resp_len);
    if (st != GEIST_OK) {
        fprintf(stderr, "agent_run failed (status %d)\n", (int) st);
    } else {
        printf("%s\n", resp);
    }

    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return st == GEIST_OK ? 0 : 1;
}

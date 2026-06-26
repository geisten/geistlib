/* test_force_call_int — show what agent_force_call produces on a real model. */
#define _POSIX_C_SOURCE 200809L
#include "test_helpers.h"
#include "../tools/agent.h"
#include "../tools/agent_listdir.h"
#include <geist.h>
#include <stdio.h>
#include <string.h>

static struct geist_agent agent;

int main(void) {
    GEIST_REQUIRE_GGUF(model_path);
    struct geist_backend *be = nullptr;
    if (geist_backend_create("auto", nullptr, nullptr, &be) != GEIST_OK)
        GEIST_SKIP("backend");
    struct geist_model *model = nullptr;
    if (geist_model_load(model_path, be, &model) != GEIST_OK) {
        geist_backend_destroy(be);
        GEIST_SKIP("load");
    }
    struct geist_session_opts o = {0};
    struct geist_session     *s = nullptr;
    if (geist_session_create(model, be, &o, &s) != GEIST_OK) {
        GEIST_SKIP("session");
    }

    struct geist_tool tools[] = {listdir_tool()};
    geist_agent_init(&agent, model, s, 1, tools, 4, "You are a file assistant.");
    fprintf(stderr, "template=%s\n", agent.tmpl.name);

    /* Build the transcript exactly as run() does, then force a call. */
    agent.tlen      = agent_system_prompt(&agent, sizeof agent.transcript, agent.transcript);
    const char *req = "Zeige mir den Inhalt des aktuellen Ordners";
    agent.tlen += (size_t) snprintf(agent.transcript + agent.tlen,
                                    sizeof agent.transcript - agent.tlen,
                                    "%s%s%s",
                                    req,
                                    agent.tmpl.turn_close,
                                    agent.tmpl.model_open);
    geist_session_reset(s);
    geist_session_set_prompt(s, agent.transcript);

    char   turn[GEIST_AGENT_TURN_CAP];
    size_t n = agent_force_call(&agent, sizeof turn, turn);
    fprintf(stderr, "FORCED CALL (%zu bytes): %s\n", n, turn);

    char name[GEIST_AGENT_NAME_CAP], args[GEIST_AGENT_ARGS_CAP];
    int  parsed = agent_parse_call(n, turn, sizeof name, name, sizeof args, args);
    fprintf(stderr, "parsed=%d name=\"%s\" args=%s\n", parsed, name, args);
    int rc = (parsed && strcmp(name, "list_dir") == 0) ? GEIST_TEST_PASS : GEIST_TEST_FAIL;

    geist_session_destroy(s);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return rc;
}

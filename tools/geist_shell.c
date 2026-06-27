/*
 * geist_shell — a small "file assistant" agent CLI exposing two host
 * capabilities to the model: list a directory (list_dir) and read+summarize a
 * text file (summarize_file). Shows how function tooling is wired and that the
 * security model holds — both tools are opendir/read only, no shell.
 *
 *   geist_shell <model.gguf> "Fasse die Datei README.md zusammen"
 *   geist_shell <model.gguf>                 # interactive REPL
 *
 * GEIST_FORCE_CALL=1 grammar-forces turn 0 into a tool call (lets a model that
 * isn't tool-trained still drive the tools — see agent_force_call).
 *
 * Unlike the other demos this does its own model load (rather than calling
 * geist_agent_main): summarize_file's ctx needs the loaded model + backend so
 * its sub-session can run, so the tool table is built AFTER the model exists.
 */
#define _POSIX_C_SOURCE 200809L

#include "agent_listdir.h"
#include "agent_main.h" /* arg parsing + agent_main_ask, reused */
#include "agent_summarize.h"
#include "agent_webfetch.h"
#include "agent_websearch.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *SHELL_SYSTEM =
        "You are a file and web assistant. To see a directory's contents reply with "
        "{\"tool\":\"list_dir\",\"args\":{\"path\":\".\"}}. To summarize a file reply with "
        "{\"tool\":\"summarize_file\",\"args\":{\"path\":\"<file>\"}}. To search the web reply with "
        "{\"tool\":\"web_search\",\"args\":{\"query\":\"<query>\"}}. To read a web page reply with "
        "{\"tool\":\"web_fetch\",\"args\":{\"url\":\"<url>\"}}. After the tool result, "
        "answer the user in one or two sentences.";

int main(int argc, char **argv) {
    const char            *prog = argc > 0 ? argv[0] : "geist_shell";
    struct agent_main_opts opts;
    switch (agent_main_parse_args(argc, argv, &opts)) {
    case AGENT_MAIN_HELP:
        agent_main_usage(stdout, prog);
        return 0;
    case AGENT_MAIN_BADARGS:
        agent_main_usage(stderr, prog);
        return 2;
    case AGENT_MAIN_RUN:
        break;
    }

    struct geist_backend *be = nullptr;
    if (geist_backend_create("auto", nullptr, nullptr, &be) != GEIST_OK) {
        fprintf(stderr, "geist_shell: backend_create failed: %s\n", geist_last_create_error());
        return 1;
    }
    struct geist_model *model = nullptr;
    if (geist_model_load(opts.model, be, &model) != GEIST_OK) {
        fprintf(stderr, "geist_shell: model_load failed: %s\n", geist_last_create_error());
        geist_backend_destroy(be);
        return 1;
    }
    struct geist_session_opts sopts = {0}; /* greedy, deterministic */
    struct geist_session     *sess  = nullptr;
    if (geist_session_create(model, be, &sopts, &sess) != GEIST_OK) {
        fprintf(stderr, "geist_shell: session_create failed\n");
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return 1;
    }

    /* summarize_file's sub-session runs on this model+backend; ctx must outlive
     * the agent. root="." -> reads are confined to the current directory tree. */
    static struct summarize_ctx sctx;
    sctx                      = (struct summarize_ctx) {.model = model, .be = be, .root = "."};
    /* web_search hits a fixed engine (low risk); web_fetch lets the model pick the
     * host — nullptr allowlist = any http/https, fine for a local demo, tighten via
     * webfetch_tool("example.com,...") for an exposed deployment. */
    struct geist_tool tools[] = {listdir_tool(), summarize_file_tool(&sctx), websearch_tool(nullptr),
                                 webfetch_tool(nullptr)};
    static struct geist_agent agent;
    geist_agent_init(&agent,
                     model,
                     sess,
                     sizeof tools / sizeof *tools,
                     tools,
                     opts.max_steps,
                     SHELL_SYSTEM);
    const char *fc   = getenv("GEIST_FORCE_CALL");
    agent.force_call = fc != nullptr && fc[0] == '1';

    int rc = 0;
    if (opts.question) {
        rc = agent_main_ask(&agent, opts.question);
    } else {
        static char line[AGENT_MAIN_LINE_CAP];
        fputs("> ", stdout);
        fflush(stdout);
        while (fgets(line, sizeof line, stdin)) {
            line[strcspn(line, "\n")] = '\0';
            if (line[0] == '\0') {
                fputs("> ", stdout);
                fflush(stdout);
                continue;
            }
            if (strcmp(line, "/quit") == 0 || strcmp(line, "/exit") == 0) {
                break;
            }
            rc |= agent_main_ask(&agent, line);
            fputs("> ", stdout);
            fflush(stdout);
        }
    }

    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return rc;
}

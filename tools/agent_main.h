/*
 * agent_main.h — the reusable agent CLI engine. An app defines its tools + a
 * system prompt and forwards argc/argv; this does the rest (arg parse, model
 * load, one-shot or REPL, geist_agent_run, cleanup). Header-only so a separate
 * app repo just links libgeist.a and writes a ~15-line main:
 *
 *     int main(int argc, char **argv) {
 *         struct geist_tool tools[] = { docsearch_tool(getenv("GEIST_DOCS")) };
 *         return geist_agent_main(argc, argv, "Answer from the local docs.",
 *                                 sizeof tools / sizeof *tools, tools);
 *     }
 *
 * Plain ISO C (no POSIX) so it needs no feature-test macro; the app's tools may.
 */
#ifndef GEIST_AGENT_MAIN_H
#define GEIST_AGENT_MAIN_H

#include <geist.h>
#include <geist_util.h>

#include "agent.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { AGENT_MAIN_RESP_CAP = 1 << 13, AGENT_MAIN_LINE_CAP = 8192 };

struct agent_main_opts {
    const char *model;     /* required (first positional) */
    const char *question;  /* second positional; nullptr -> interactive REPL */
    size_t      max_steps; /* -n/--max-steps; 0 -> the agent's default */
};

enum agent_main_parse { AGENT_MAIN_RUN = 0, AGENT_MAIN_HELP = 1, AGENT_MAIN_BADARGS = 2 };

static inline void agent_main_usage(FILE *o, const char *prog) {
    fprintf(o,
            "usage: %s <model.gguf> [\"question\"] [-n max_steps]\n"
            "  no question -> interactive REPL (one request per line; /quit to exit)\n",
            prog);
}

/* Pure parse: no exit(), no output, so it is unit-testable. Fills *opts and
 * returns RUN / HELP / BADARGS. */
[[nodiscard]] static inline enum agent_main_parse
agent_main_parse_args(int argc, char **argv, struct agent_main_opts *opts) {
    opts->model     = nullptr;
    opts->question  = nullptr;
    opts->max_steps = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (strcmp(a, "-h") == 0 || strcmp(a, "--help") == 0) {
            return AGENT_MAIN_HELP;
        } else if (strcmp(a, "-n") == 0 || strcmp(a, "--max-steps") == 0) {
            if (i + 1 >= argc) {
                return AGENT_MAIN_BADARGS;
            }
            char *end = nullptr;
            long  n   = strtol(argv[++i], &end, 10);
            if (end == argv[i] || *end != '\0' || n <= 0) {
                return AGENT_MAIN_BADARGS;
            }
            opts->max_steps = (size_t) n;
        } else if (a[0] == '-' && a[1] != '\0') {
            return AGENT_MAIN_BADARGS; /* unknown flag */
        } else if (!opts->model) {
            opts->model = a;
        } else if (!opts->question) {
            opts->question = a;
        } else {
            return AGENT_MAIN_BADARGS; /* extra positional */
        }
    }
    return opts->model ? AGENT_MAIN_RUN : AGENT_MAIN_BADARGS;
}

/* Run one request and print the answer + newline. Returns 0 on success. */
static inline int agent_main_ask(struct geist_agent *agent, const char *req) {
    static char resp[AGENT_MAIN_RESP_CAP];
    size_t      n = 0;
    if (geist_agent_run(agent, strlen(req), req, sizeof resp, resp, &n) != GEIST_OK) {
        fprintf(stderr, "agent: run failed\n");
        return 1;
    }
    fwrite(resp, 1, n, stdout);
    putchar('\n');
    fflush(stdout);
    return 0;
}

/* The reusable agent CLI. tools[] + system_prompt must outlive the call. */
[[nodiscard]] static inline int geist_agent_main(int                     argc,
                                                 char                  **argv,
                                                 const char             *system_prompt,
                                                 size_t                  n_tools,
                                                 const struct geist_tool tools[static n_tools]) {
    const char            *prog = argc > 0 ? argv[0] : "geist_agent";
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
        fprintf(stderr, "agent: backend_create failed: %s\n", geist_last_create_error());
        return 1;
    }
    struct geist_model *model = nullptr;
    if (geist_model_load(opts.model, be, &model) != GEIST_OK) {
        fprintf(stderr, "agent: model_load failed: %s\n", geist_last_create_error());
        geist_backend_destroy(be);
        return 1;
    }
    struct geist_session_opts sopts = {0}; /* greedy, deterministic */
    struct geist_session     *sess  = nullptr;
    if (geist_session_create(model, be, &sopts, &sess) != GEIST_OK) {
        fprintf(stderr, "agent: session_create failed\n");
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return 1;
    }

    static struct geist_agent agent; /* large -> static, not a deep stack */
    geist_agent_init(&agent, model, sess, n_tools, tools, opts.max_steps, system_prompt);

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

#endif /* GEIST_AGENT_MAIN_H */

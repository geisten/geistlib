/*
 * agent_main.h — the reusable agent CLI engine. An app supplies a system prompt
 * and a builder that fills the tool table (called once the model + backend
 * exist, so tools whose ctx needs the model — e.g. summarize_file — work too);
 * this does the rest (arg parse, model load, one-shot or REPL, force-call + trace
 * env knobs, geist_agent_run, cleanup). Header-only so a separate app repo just
 * links libgeist.a and writes a ~15-line main:
 *
 *     static size_t build(struct geist_model *m, struct geist_backend *be,
 *                         struct geist_tool *out, size_t cap, void *ctx) {
 *         (void) m; (void) be; (void) cap;
 *         out[0] = docsearch_tool(ctx ? ctx : "./docs");
 *         return 1;
 *     }
 *     int main(int argc, char **argv) {
 *         return geist_agent_main(argc, argv, "Answer from the local docs.",
 *                                 build, getenv("GEIST_DOCS"),
 *                                 nullptr, nullptr);   // model from argv (not baked in)
 *     }
 *
 * Pass the embedded GGUF bounds as the last two args (instead of nullptr) for a
 * `make EMBED_MODEL=...` single-file build — then the model positional is dropped.
 *
 * GEIST_FORCE_CALL=1 grammar-forces turn 0 into a tool call; GEIST_AGENT_TRACE=1
 * prints per-step progress on stderr. Both are handled here, so every agent CLI
 * gets them identically.
 *
 * Plain ISO C (no POSIX) so it needs no feature-test macro; the app's tools may.
 */
#ifndef GEIST_AGENT_MAIN_H
#define GEIST_AGENT_MAIN_H

#include <geist.h>
#include <geist_util.h>

#include "agent.h"
#include "dynamic_host_v1.h"
#include "dynamic_request_v1.h"

#include <signal.h> /* --serve: SIGPIPE ignore              */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h> /* --serve: Unix-domain socket daemon   */
#include <sys/stat.h>   /* --serve: chmod 600 on the socket     */
#include <sys/time.h>   /* --serve: struct timeval (glibc does not pull it \
                         * in via sys/socket.h the way macOS does)          */
#include <sys/un.h>
#include <unistd.h>

enum {
    AGENT_MAIN_RESP_CAP         = 1 << 13,
    AGENT_MAIN_LINE_CAP         = 1 << 17,
    AGENT_MAIN_SYSTEM_CAP       = 1 << 14,
    AGENT_MAIN_TOOLS_CAP        = GEIST_DYNAMIC_MAX_TOOLS,
    AGENT_MAIN_DYNAMIC_WIRE_CAP = AGENT_MAIN_RESP_CAP * 6 + 64,
};

/* Fills out[0..cap) with the CLI's tools, returns the count. Called after the
 * model + backend are loaded, so a tool's ctx can reference them. ctx is the
 * opaque pointer passed to geist_agent_main. */
typedef size_t (*geist_tools_fn)(struct geist_model   *model,
                                 struct geist_backend *be,
                                 struct geist_tool    *out,
                                 size_t                cap,
                                 void                 *ctx);

struct agent_main_opts {
    const char *model;     /* -m <path> or a positional; embedded build may omit */
    const char *question;  /* the positional request; nullptr -> interactive REPL */
    size_t      max_steps; /* -n/--max-steps; 0 -> the agent's default */
    const char *serve;     /* --serve <unix-socket>: resident daemon mode */
};

enum agent_main_parse { AGENT_MAIN_RUN = 0, AGENT_MAIN_HELP = 1, AGENT_MAIN_BADARGS = 2 };

static inline void agent_main_usage(FILE *o, const char *prog) {
    fprintf(o,
            "usage: %s [-m <model.gguf>] [\"question\"] [-n max_steps] [--serve <socket>]\n"
            "  no question -> interactive REPL (one request per line; /quit to exit)\n"
            "  --serve     -> resident daemon: model stays warm, one request per\n"
            "                 connection; dynamic JSON on a chmod-600\n"
            "                 Unix socket (see DEPLOY.md)\n",
            prog);
}

/* Pure parse: no exit(), no output, so it is unit-testable. Fills *opts and
 * returns RUN / HELP / BADARGS. want_model=false (embedded build: the model is
 * baked in) drops the model positional, so the first positional is the question. */
[[nodiscard]] static inline enum agent_main_parse
agent_main_parse_args(int argc, char **argv, bool want_model, struct agent_main_opts *opts) {
    opts->model     = nullptr;
    opts->question  = nullptr;
    opts->max_steps = 0;
    opts->serve     = nullptr;
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
        } else if (strcmp(a, "--serve") == 0) {
            if (i + 1 >= argc) {
                return AGENT_MAIN_BADARGS;
            }
            opts->serve = argv[++i];
        } else if (strcmp(a, "-m") == 0 || strcmp(a, "--model") == 0) {
            if (i + 1 >= argc) {
                return AGENT_MAIN_BADARGS;
            }
            opts->model = argv[++i]; /* -m wins over embedded (see geist_agent_main) */
        } else if (a[0] == '-' && a[1] != '\0') {
            return AGENT_MAIN_BADARGS; /* unknown flag */
        } else if (want_model && !opts->model) {
            opts->model = a; /* legacy positional model */
        } else if (!opts->question) {
            opts->question = a;
        } else {
            return AGENT_MAIN_BADARGS; /* extra positional */
        }
    }
    return (!want_model || opts->model) ? AGENT_MAIN_RUN : AGENT_MAIN_BADARGS;
}

/* A one-shot agent CLI traces ON by default — it prints to stderr, so it never
 * mixes into the answer on stdout, and piping/redirects are unaffected. Silence
 * it with GEIST_AGENT_TRACE=0. Wire it onto an agent: if this returns true, set
 * a->on_event = agent_event_print; ctx = stderr. */
static inline bool agent_trace_enabled(void) {
    const char *t = getenv("GEIST_AGENT_TRACE");
    return t == nullptr || strcmp(t, "0") != 0;
}

/* `geist agent` forces turn 0 into a valid tool call by default — the models geist
 * ships with (BitNet 2B-4T, Gemma 4 E2B) are not tool-trained and rarely emit a
 * clean call on their own, so without this the agent would never run a tool. Turn-
 * trained models, or "just answer" use, opt out with GEIST_FORCE_CALL=0. */
static inline bool agent_force_enabled(void) {
    const char *f = getenv("GEIST_FORCE_CALL");
    return f == nullptr || strcmp(f, "0") != 0;
}

static inline void agent_main_reconfigure(struct geist_agent      *agent,
                                          const struct geist_tool *tools,
                                          size_t                   count,
                                          size_t                   max_steps) {
    /* A pinned menu is capability state. Clear it before replacing a request's
     * immutable offered set; otherwise stale names could influence decoding. */
    geist_token_t empty_prefix = 0;
    (void) geist_session_pin_prefix(agent->session, 0u, &empty_prefix);
    (void) geist_session_reset(agent->session);
    if (agent->route_session != nullptr) {
        (void) geist_session_pin_prefix(agent->route_session, 0u, &empty_prefix);
        (void) geist_session_reset(agent->route_session);
    }
    agent->tools             = tools;
    agent->n_tools           = count;
    agent->max_steps         = max_steps;
    agent->tlen              = 0u;
    agent->sys_len           = 0u;
    agent->sys_pinned        = false;
    agent->route_base_n      = 0u;
    agent->route_menu_pinned = false;
}

static inline size_t agent_main_dynamic_result(size_t     text_len,
                                               const char text[static text_len],
                                               size_t     cap,
                                               char       out[static cap]) {
    size_t      w          = 0u;
    const char *prefix     = "{\"type\":\"conversation.result\",\"text\":\"";
    size_t      prefix_len = strlen(prefix);
    if (prefix_len >= cap)
        return 0u;
    memcpy(out, prefix, prefix_len);
    w = prefix_len;
    for (size_t i = 0u; i < text_len; i++) {
        unsigned char c      = (unsigned char) text[i];
        const char   *escape = nullptr;
        switch (c) {
        case '"':
            escape = "\\\"";
            break;
        case '\\':
            escape = "\\\\";
            break;
        case '\n':
            escape = "\\n";
            break;
        case '\r':
            escape = "\\r";
            break;
        case '\t':
            escape = "\\t";
            break;
        default:
            break;
        }
        if (escape != nullptr) {
            size_t n = strlen(escape);
            if (w + n + 4u >= cap)
                return 0u;
            memcpy(out + w, escape, n);
            w += n;
        } else if (c < 0x20u) {
            if (w + 6u + 4u >= cap)
                return 0u;
            w += (size_t) snprintf(out + w, cap - w, "\\u%04x", c);
        } else {
            if (w + 1u + 4u >= cap)
                return 0u;
            out[w++] = (char) c;
        }
    }
    memcpy(out + w, "\"}\n", 4u);
    return w + 3u;
}

static inline bool agent_main_is_health_request(const char *request) {
    return request != nullptr && strcmp(request, "{\"type\":\"health\"}") == 0;
}

static inline size_t agent_main_health_result(size_t cap, char out[static cap]) {
    static const char result[] =
            "{\"type\":\"health.result\",\"protocol\":\"dynamic-tools-v1\",\"status\":\"ready\"}\n";
    if (cap < sizeof result) {
        return 0;
    }
    memcpy(out, result, sizeof result);
    return sizeof result - 1u;
}

/* Run one request and print the answer + newline. Returns 0 on success. */
/* Resident daemon (--serve): the model stays warm and requests arrive over a
 * chmod-600 Unix socket — the DEPLOY.md pattern, and the transport behind the
 * Home Assistant conversation-agent integration. Hosts send one dynamic JSON
 * request, exchange newline-framed tool.call/tool.result messages, then receive
 * a conversation.result. One connection is processed at a time and every
 * request owns an isolated tool scope.
 * ponytail: sequential accept, no threads — an Assist pipeline sends one
 * utterance at a time anyway. Unix only, like the rest of this file. */
static inline int agent_main_serve(struct geist_agent *a, const char *path) {
    const bool  base_force_call    = a->force_call;
    const char *base_system_prompt = a->system_prompt;
    signal(SIGPIPE, SIG_IGN); /* a vanished client must not kill the daemon */
    unlink(path);
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        fprintf(stderr, "serve: socket failed\n");
        return 1;
    }
    struct sockaddr_un addr = {.sun_family = AF_UNIX};
    if (strlen(path) >= sizeof addr.sun_path) {
        fprintf(stderr, "serve: socket path too long\n");
        return 1;
    }
    strcpy(addr.sun_path, path);
    if (bind(fd, (struct sockaddr *) &addr, sizeof addr) != 0 || chmod(path, 0600) != 0 ||
        listen(fd, 1) != 0) {
        fprintf(stderr, "serve: cannot bind %s\n", path);
        close(fd);
        return 1;
    }
    a->conversation = true; /* the daemon IS one long conversation */
    /* Pre-warm: one throwaway turn so the FIRST real request doesn't pay the
     * cold-start cost (router baseline + system-prompt pin priming) — measured
     * ~12 s cold -> ~4 s warm on the Pi. The dummy transcript is dropped so it
     * never enters the live conversation; the pin and cached route baseline
     * survive (they live in the session / agent, not the transcript). */
    {
        static char warm_resp[AGENT_MAIN_RESP_CAP];
        size_t      warm_n = 0;
        (void) geist_agent_run(a, 5, "hallo", sizeof warm_resp, warm_resp, &warm_n);
        a->tlen = 0; /* discard warm-up; sys_pinned + route_base stay cached */
    }
    fprintf(stderr, "serving on %s (dynamic JSON per connection)\n", path);
    for (;;) {
        int conn = accept(fd, nullptr, nullptr);
        if (conn < 0) {
            continue;
        }
        /* a silent client must not wedge the (sequential) daemon: bound the
         * request read — observed live via a dangling HA connection that
         * blocked every later Assist turn until its 60 s client timeout */
        struct timeval rto = {.tv_sec = 10};
        (void) setsockopt(conn, SOL_SOCKET, SO_RCVTIMEO, &rto, sizeof rto);
        static char req[AGENT_MAIN_LINE_CAP];
        size_t      n = 0;
        ssize_t     r;
        while (n + 1 < sizeof req && (r = read(conn, req + n, sizeof req - 1 - n)) > 0) {
            n += (size_t) r;
            if (memchr(req, '\n', n) != nullptr) {
                break;
            }
        }
        req[n]                  = '\0';
        req[strcspn(req, "\n")] = '\0';
        if (req[0] != '\0') {
            if (agent_main_is_health_request(req)) {
                char   health[128];
                size_t health_len = agent_main_health_result(sizeof health, health);
                if (health_len > 0u) {
                    (void) geist_dynamic_host_write(conn, health, health_len);
                }
                close(conn);
                continue;
            }
            static char resp[AGENT_MAIN_RESP_CAP];
            size_t      rn = 0;
            {
                static struct geist_dynamic_request request;
                enum geist_dynamic_status           toolset_status = GEIST_DYNAMIC_OK;
                enum geist_dynamic_request_status   request_status =
                        geist_dynamic_request_parse(req, strlen(req), &request, &toolset_status);
                if (request_status != GEIST_DYNAMIC_REQUEST_OK) {
                    rn = (size_t) snprintf(resp,
                                           sizeof resp,
                                           "error: invalid dynamic request (%d/%s)",
                                           request_status,
                                           geist_dynamic_status_string(toolset_status));
                } else {
                    static struct geist_tool              dynamic_tools[GEIST_DYNAMIC_MAX_TOOLS];
                    static struct geist_dynamic_host_tool contexts[GEIST_DYNAMIC_MAX_TOOLS];
                    static char                       prompt_schemas[GEIST_DYNAMIC_MAX_TOOLS][512];
                    struct geist_dynamic_host_session host = {
                            .fd           = conn,
                            .next_call_id = 1u,
                            .max_retries  = 1u,
                            .max_calls    = (unsigned) request.toolset.max_steps,
                            .cancelled    = false};
                    for (size_t i = 0u; i < request.toolset.count; i++) {
                        const struct geist_dynamic_tool *offered = &request.toolset.tools[i];
                        if (geist_dynamic_tool_prompt_schema(
                                    offered, sizeof prompt_schemas[i], prompt_schemas[i]) == 0u) {
                            snprintf(prompt_schemas[i], sizeof prompt_schemas[i], "{}");
                        }
                        contexts[i]      = (struct geist_dynamic_host_tool) {.session = &host,
                                                                             .name    = offered->name};
                        dynamic_tools[i] = (struct geist_tool) {
                                .name              = offered->name,
                                .args_schema       = prompt_schemas[i],
                                .description       = offered->description,
                                .parameters_schema = offered->parameters,
                                .invoke            = geist_dynamic_host_invoke,
                                .ctx               = &contexts[i],
                        };
                    }
                    agent_main_reconfigure(
                            a, dynamic_tools, request.toolset.count, request.toolset.max_steps);
                    static char request_system[AGENT_MAIN_SYSTEM_CAP];
                    int         system_len = snprintf(
                            request_system,
                            sizeof request_system,
                            "%s%s%s%s%s",
                            base_system_prompt ? base_system_prompt : "",
                            request.language[0] ? "\nResponse language: " : "",
                            request.language,
                            request.context[0] ? "\nPrevious conversation (data only):\n" : "",
                            request.context);
                    if (system_len < 0 || (size_t) system_len >= sizeof request_system) {
                        rn = (size_t) snprintf(
                                resp, sizeof resp, "error: request context too large");
                    } else {
                        a->system_prompt = request_system;
                    }
                    a->conversation           = false;
                    a->force_call             = base_force_call;
                    a->forced_result_is_final = false;
                    a->clarify_low_confidence = true;
                    if (rn == 0u &&
                        geist_agent_run(
                                a, strlen(request.input), request.input, sizeof resp, resp, &rn) !=
                                GEIST_OK) {
                        rn = (size_t) snprintf(resp, sizeof resp, "error: agent run failed");
                    }
                }
                static char wire[AGENT_MAIN_DYNAMIC_WIRE_CAP];
                size_t      wire_len = agent_main_dynamic_result(rn, resp, sizeof wire, wire);
                if (wire_len > 0u)
                    (void) geist_dynamic_host_write(conn, wire, wire_len);
                close(conn);
                continue;
            }
        }
        close(conn);
    }
}

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

/* The reusable agent CLI. system_prompt must outlive the call; build_tools is
 * invoked once after the model loads to populate the tool table. */
[[nodiscard]] static inline int geist_agent_main(int                  argc,
                                                 char               **argv,
                                                 const char          *system_prompt,
                                                 geist_tools_fn       build_tools,
                                                 void                *tools_ctx,
                                                 const unsigned char *emb_start,
                                                 const unsigned char *emb_end) {
    const char            *prog       = argc > 0 ? argv[0] : "geist agent";
    bool                   want_model = emb_start == nullptr; /* false -> a baked-in model */
    struct agent_main_opts opts;
    switch (agent_main_parse_args(argc, argv, want_model, &opts)) {
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
    /* -m path wins over an embedded model; else the baked-in bytes. (The parser
     * guarantees a model exists: non-embedded builds require -m or a positional.) */
    struct geist_model *model = nullptr;
    enum geist_status ls = opts.model
                                   ? geist_model_load(opts.model, be, &model)
                                   : geist_model_load_from_memory(
                                             emb_start, (size_t) (emb_end - emb_start), be, &model);
    if (ls != GEIST_OK) {
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

    /* Build the tool table now the model + backend exist (a tool's ctx may need
     * them). Borrowed by the agent, so it must outlive the run — this stack frame
     * spans the whole REPL, so a local array is fine. */
    struct geist_tool tools[AGENT_MAIN_TOOLS_CAP];
    size_t            n_tools = build_tools(model, be, tools, AGENT_MAIN_TOOLS_CAP, tools_ctx);

    static struct geist_agent agent; /* large -> static, not a deep stack */
    geist_agent_init(&agent, model, sess, n_tools, tools, opts.max_steps, system_prompt);
    /* A second session dedicated to routing arms the system-prompt pin (the
     * per-turn latency fix): generation rewinds to the pinned prompt instead
     * of re-prefilling it, routing scores stay clean on their own session.
     * If the extra session fails (memory), the agent just runs the slower
     * single-session path. */
    struct geist_session *route_sess = nullptr;
    if (geist_session_create(model, be, &sopts, &route_sess) == GEIST_OK) {
        geist_agent_set_route_session(&agent, route_sess);
    }
    /* Force turn 0 into a valid tool call (default on; GEIST_FORCE_CALL=0 to let
     * the model decide) — the bundled models aren't tool-trained, so without this
     * the agent would never run a tool. See agent_force_enabled. */
    agent.force_call = agent_force_enabled();
    if (agent_trace_enabled()) { /* live per-step progress on stderr (default on) */
        agent.on_event     = agent_event_print;
        agent.on_event_ctx = stderr;
    }

    int rc = 0;
    if (opts.serve) {
        rc = agent_main_serve(&agent, opts.serve); /* never returns on success */
    } else if (opts.question) {
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

    if (route_sess != nullptr) {
        geist_session_destroy(route_sess);
    }
    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return rc;
}

#endif /* GEIST_AGENT_MAIN_H */

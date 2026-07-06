/* bench_agent_eval — agent-layer reliability eval: routing / args / chains.
 *
 * Measures the HARNESS layer (tools/agent.h) mechanically, per stage:
 *   route  first tool the model CALLED == expected  ("none" = no call)
 *   args   expected arg key present, value contains the expected substring
 *   chain  exact sequence of EXECUTED tools == expected chain
 * No answer-quality judging — that is a manual spot-check by design.
 *
 * Corpus: tests/data/agent_eval/cases.jsonl (flat JSONL; fields id, cat,
 * req, tool, arg, want, chain — see the file). Deterministic: greedy decode,
 * fixture doc dir, and the web tools are in-process stubs carrying the real
 * name/description/schema so routing sees the production tool surface.
 * ponytail: in-process web stubs, not an HTTP replay server — same
 * determinism, zero infra; upgrade to a local SearXNG-JSON fixture server if
 * stub texts ever diverge from real curl output in a way that shifts routing.
 *
 * Runs each case in both agent modes (force_call on/off) as separate columns.
 * Exit 0 always — baseline mode; thresholds are fixed AFTER the first
 * baseline run. 77 without a model, 99 if the harness itself is broken.
 *
 * Usage: bench_agent_eval [--mode forced|free|both] [--verbose] [cases.jsonl]
 */
#define _POSIX_C_SOURCE 200809L
#include "test_helpers.h"
#include "../tools/agent.h"
#include "../tools/agent_docsearch.h"
#include "../tools/agent_listdir.h"
#include "../tools/agent_summarize.h"
#include "../tools/agent_webfetch.h"
#include "../tools/agent_websearch.h"
#include <geist.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <time.h>

enum { EV_MAX_CASES = 64, EV_MAX_STEPS = 4 };

#define EV_DOCS_DIR "tests/data/agent_eval/docs"
#define EV_CASES "tests/data/agent_eval/cases.jsonl"

struct ev_case {
    char id[64];
    char cat[16]; /* single | chain | ambig | neg */
    char req[512];
    char tool[GEIST_AGENT_NAME_CAP]; /* expected first call; "none" = no call */
    char arg[32];                    /* expected arg key, "" = don't check */
    char want[128];                  /* substring expected in the arg value */
    char chain[256];                 /* comma-joined executed tools; "" for none */
};

static struct ev_case     ev_cases[EV_MAX_CASES];
static struct geist_agent ev_agent; /* large — keep off the stack */

/* ---- per-run capture via the agent's on_event hook ---------------------- */

static struct {
    char first_tool[GEIST_AGENT_NAME_CAP]; /* first CALLING = the routing decision */
    char first_args[GEIST_AGENT_ARGS_CAP];
    char got_chain[256]; /* comma-joined RUNNING tools = what actually executed */
    char answer[240];
} cap;

static void ev_capture(void *ctx, const struct geist_agent_event *ev) {
    (void) ctx;
    switch (ev->phase) {
    case GEIST_AGENT_CALLING:
        if (!cap.first_tool[0] && ev->tool) {
            snprintf(cap.first_tool, sizeof cap.first_tool, "%s", ev->tool);
            snprintf(cap.first_args, sizeof cap.first_args, "%s", ev->detail ? ev->detail : "");
        }
        break;
    case GEIST_AGENT_RUNNING: {
        size_t w = strlen(cap.got_chain);
        snprintf(cap.got_chain + w, sizeof cap.got_chain - w, "%s%s", w ? "," : "", ev->tool);
        break;
    }
    case GEIST_AGENT_ANSWERING:
        if (ev->detail)
            snprintf(cap.answer, sizeof cap.answer, "%s", ev->detail);
        break;
    default:
        break;
    }
}

/* ---- deterministic web stubs (production name/description/schema) ------- */

static enum geist_status ev_stub_search(void      *ctx,
                                        size_t     args_len,
                                        const char args[static args_len],
                                        size_t     out_cap,
                                        char       out[static out_cap],
                                        size_t    *out_len) {
    (void) ctx;
    (void) args;
    size_t w = (size_t) snprintf(
            out,
            out_cap,
            "1. BitNet b1.58 2B4T - ternary LLM — https://example.com/bitnet.html\n"
            "2. KV cache quantization overview — https://example.com/kv.html\n"
            "3. Raspberry Pi 5 inference benchmarks — https://example.com/pi5.html\n");
    return agent_ret(out_len, w);
}

static enum geist_status ev_stub_fetch(void      *ctx,
                                       size_t     args_len,
                                       const char args[static args_len],
                                       size_t     out_cap,
                                       char       out[static out_cap],
                                       size_t    *out_len) {
    (void) ctx;
    char url[512];
    agent_json_str(args, "url", sizeof url, url);
    size_t w =
            (size_t) snprintf(out,
                              out_cap,
                              "Page %s: BitNet b1.58 2B4T is a native 1-bit large language model "
                              "with ternary weights. It targets efficient on-device inference and "
                              "matches full-precision models of similar size on many benchmarks.",
                              url[0] ? url : "(no url)");
    return agent_ret(out_len, w);
}

/* ---- corpus -------------------------------------------------------------- */

static size_t ev_load(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f)
        return 0;
    char   line[2048];
    size_t n = 0;
    while (n < EV_MAX_CASES && fgets(line, sizeof line, f)) {
        if (line[0] != '{')
            continue; /* blank / comment */
        struct ev_case *c = &ev_cases[n];
        if (!agent_json_str(line, "id", sizeof c->id, c->id))
            continue;
        agent_json_str(line, "cat", sizeof c->cat, c->cat);
        agent_json_str(line, "req", sizeof c->req, c->req);
        agent_json_str(line, "tool", sizeof c->tool, c->tool);
        agent_json_str(line, "arg", sizeof c->arg, c->arg);
        agent_json_str(line, "want", sizeof c->want, c->want);
        /* default chain: the expected tool alone; "none" -> empty chain */
        if (!agent_json_str(line, "chain", sizeof c->chain, c->chain) &&
            strcmp(c->tool, "none") != 0)
            snprintf(c->chain, sizeof c->chain, "%s", c->tool);
        n++;
    }
    fclose(f);
    return n;
}

static int ev_ci_contains(const char *hay, const char *needle) {
    size_t n = strlen(needle);
    if (n == 0)
        return 1;
    for (; *hay; hay++)
        if (strncasecmp(hay, needle, n) == 0)
            return 1;
    return 0;
}

/* ---- scoring ------------------------------------------------------------- */

struct ev_tally {
    int total, route, args_app, args_ok, chain, pass;
};

static const char *EV_CATS[] = {"single", "chain", "ambig", "neg"};

static int ev_cat_idx(const char *cat) {
    for (int i = 0; i < 4; i++)
        if (strcmp(cat, EV_CATS[i]) == 0)
            return i;
    return 0;
}

static void ev_run_mode(struct geist_model   *model,
                        struct geist_session *session,
                        struct geist_tool    *tools,
                        size_t                n_tools,
                        size_t                n_cases,
                        bool                  forced,
                        bool                  verbose) {
    const char *mode = forced ? "forced" : "free";
    geist_agent_init(&ev_agent, model, session, n_tools, tools, EV_MAX_STEPS, nullptr);
    ev_agent.force_call = forced;
    ev_agent.on_event   = ev_capture;

    struct ev_tally t[4] = {0};
    time_t          t0   = time(nullptr);

    for (size_t i = 0; i < n_cases; i++) {
        const struct ev_case *c = &ev_cases[i];
        memset(&cap, 0, sizeof cap);

        static char       resp[4096];
        size_t            resp_len = 0;
        enum geist_status st =
                geist_agent_run(&ev_agent, strlen(c->req), c->req, sizeof resp, resp, &resp_len);

        bool none_expected = strcmp(c->tool, "none") == 0;
        bool route_ok = (st == GEIST_OK) && (none_expected ? cap.first_tool[0] == '\0'
                                                           : strcmp(cap.first_tool, c->tool) == 0);
        bool chain_ok = (st == GEIST_OK) && strcmp(cap.got_chain, c->chain) == 0;
        bool args_app = c->arg[0] != '\0';
        bool args_ok  = false;
        if (args_app && route_ok) {
            char val[512];
            args_ok = agent_json_str(cap.first_args, c->arg, sizeof val, val) &&
                      ev_ci_contains(val, c->want);
        }
        bool pass = route_ok && chain_ok && (!args_app || args_ok);

        struct ev_tally *y = &t[ev_cat_idx(c->cat)];
        y->total++;
        y->route += route_ok;
        y->args_app += args_app;
        y->args_ok += args_ok;
        y->chain += chain_ok;
        y->pass += pass;

        printf("[%s] %-10s %-6s route=%-4s args=%-4s chain=%-4s got=[%s]%s\n",
               mode,
               c->id,
               c->cat,
               route_ok ? "ok" : "FAIL",
               args_app ? (args_ok ? "ok" : "FAIL") : "-",
               chain_ok ? "ok" : "FAIL",
               cap.got_chain[0] ? cap.got_chain : "none",
               st == GEIST_OK ? "" : " (run error)");
        if (verbose)
            printf("        req: %s\n        called: [%s]\n        args: %s\n        answer: %s\n",
                   c->req,
                   cap.first_tool,
                   cap.first_args[0] ? cap.first_args : "-",
                   cap.answer[0] ? cap.answer : "-");
        fflush(stdout);
    }

    struct ev_tally all = {0};
    printf("\n== mode=%s (%ld s) ==\n cat     pass    route   args    chain\n",
           mode,
           (long) (time(nullptr) - t0));
    for (int i = 0; i < 4; i++) {
        if (t[i].total == 0)
            continue;
        printf(" %-7s %2d/%-4d %2d/%-4d",
               EV_CATS[i],
               t[i].pass,
               t[i].total,
               t[i].route,
               t[i].total);
        if (t[i].args_app)
            printf(" %2d/%-4d", t[i].args_ok, t[i].args_app);
        else
            printf(" %-7s", "-");
        printf(" %2d/%d\n", t[i].chain, t[i].total);
        all.total += t[i].total;
        all.route += t[i].route;
        all.args_app += t[i].args_app;
        all.args_ok += t[i].args_ok;
        all.chain += t[i].chain;
        all.pass += t[i].pass;
    }
    /* machine-readable line for CI diffing */
    printf("SUMMARY mode=%s pass=%d/%d route=%d/%d args=%d/%d chain=%d/%d\n\n",
           mode,
           all.pass,
           all.total,
           all.route,
           all.total,
           all.args_ok,
           all.args_app,
           all.chain,
           all.total);
}

int main(int argc, char **argv) {
    const char *cases_path = EV_CASES;
    bool        verbose = false, run_forced = true, run_free = true;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            i++;
            run_forced = strcmp(argv[i], "free") != 0;
            run_free   = strcmp(argv[i], "forced") != 0;
        } else {
            cases_path = argv[i];
        }
    }

    size_t n_cases = ev_load(cases_path);
    if (n_cases == 0) {
        fprintf(stderr, "no cases in %s (run from the repo root)\n", cases_path);
        return GEIST_TEST_ERROR;
    }

    GEIST_REQUIRE_GGUF(model_path);
    struct geist_backend *be = nullptr;
    if (geist_backend_create("auto", nullptr, nullptr, &be) != GEIST_OK)
        GEIST_SKIP("backend");
    struct geist_model *model = nullptr;
    if (geist_model_load(model_path, be, &model) != GEIST_OK)
        GEIST_SKIP("load");
    struct geist_session_opts o       = {0};
    struct geist_session     *session = nullptr;
    if (geist_session_create(model, be, &o, &session) != GEIST_OK)
        GEIST_SKIP("session");

    static struct summarize_ctx sumctx;
    sumctx = (struct summarize_ctx) {.model = model, .be = be, .root = EV_DOCS_DIR};

    struct geist_tool tools[] = {
            listdir_tool(),
            docsearch_tool(EV_DOCS_DIR),
            summarize_file_tool(&sumctx),
            websearch_tool(nullptr),
            webfetch_tool(nullptr),
    };
    tools[3].invoke = ev_stub_search; /* keep production name/desc/schema, */
    tools[4].invoke = ev_stub_fetch;  /* stub only the side effect         */
    tools[3].ctx = tools[4].ctx = nullptr;

    printf("agent eval: %zu cases, model=%s\n\n", n_cases, model_path);
    size_t n_tools = sizeof tools / sizeof tools[0];
    if (run_forced)
        ev_run_mode(model, session, tools, n_tools, n_cases, true, verbose);
    if (run_free)
        ev_run_mode(model, session, tools, n_tools, n_cases, false, verbose);

    geist_session_destroy(session);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return GEIST_TEST_PASS; /* report-only: thresholds land after the baseline run */
}

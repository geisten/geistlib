/* test_agent_masked_int — grammar-masked call decoding on a real model.
 *
 * Exercises agent_generate_call_masked directly (the free-mode path when
 * greedy opens a call): the produced call must ALWAYS be parseable, name an
 * on-whitelist tool, and key its args strictly from the tool's schema — for a
 * single-key and a multi-key tool. Values are the model's (not asserted).
 * Masked values contain no quotes by construction, so the quoted strings in
 * args alternate key, value, key, value.
 */
#define _POSIX_C_SOURCE 200809L
#include "test_helpers.h"
#include "../tools/agent.h"
#include <geist.h>
#include <stdio.h>
#include <string.h>

static struct geist_agent agent;
static int                fails = 0;

/* echo stub — never dispatched here, the test stops at the decoded call */
static enum geist_status stub_invoke(void      *ctx,
                                     size_t     args_len,
                                     const char args[static args_len],
                                     size_t     out_cap,
                                     char       out[static out_cap],
                                     size_t    *out_len) {
    (void) ctx;
    (void) args;
    return agent_obs(out_cap, out, out_len, "ok");
}

/* Open a fresh model turn for `req` and decode one masked call into out. */
static size_t masked_call(const char *req, size_t cap, char *out) {
    agent.tlen = agent_system_prompt(&agent, sizeof agent.transcript, agent.transcript);
    agent.tlen += (size_t) snprintf(agent.transcript + agent.tlen,
                                    sizeof agent.transcript - agent.tlen,
                                    "%s%s%s",
                                    req,
                                    agent.tmpl.turn_close,
                                    agent.tmpl.model_open);
    if (geist_session_reset(agent.session) != GEIST_OK ||
        geist_session_set_prompt(agent.session, agent.transcript) != GEIST_OK) {
        return 0;
    }
    return agent_generate_call_masked(&agent, cap, out);
}

/* The decoded call parses, names a whitelisted tool, and every args key is a
 * schema key of that tool, each used at most once. */
static void check_call(const char *label, size_t tn, const char *turn) {
    printf("%s: %s\n", label, tn ? turn : "(no call)");
    fails += geist_expect(tn > 0, "masked decode produced a call");
    if (tn == 0) {
        return;
    }
    char name[GEIST_AGENT_NAME_CAP];
    char args[GEIST_AGENT_ARGS_CAP];
    fails += geist_expect(agent_parse_call(tn, turn, sizeof name, name, sizeof args, args),
                          "masked call parses");
    const struct geist_tool *t = nullptr;
    for (size_t i = 0; i < agent.n_tools; i++) {
        if (strcmp(agent.tools[i].name, name) == 0) {
            t = &agent.tools[i];
        }
    }
    fails += geist_expect(t != nullptr, "masked call names a whitelisted tool");
    if (!t) {
        return;
    }
    char   keys[4][GEIST_AGENT_NAME_CAP];
    size_t nk      = agent_schema_keys(t->args_schema, 4, keys);
    int    seen[4] = {0};
    int    idx     = 0; /* quoted strings alternate key (even), value (odd) */
    for (const char *p = strchr(args, '"'); p; idx++) {
        p++;
        const char *e = strchr(p, '"');
        if (!e) {
            break;
        }
        if (idx % 2 == 0) {
            int ok = 0;
            for (size_t i = 0; i < nk; i++) {
                if ((size_t) (e - p) == strlen(keys[i]) && strncmp(p, keys[i], e - p) == 0) {
                    ok      = !seen[i];
                    seen[i] = 1;
                }
            }
            fails += geist_expect(ok, "args key is a schema key, used once");
        }
        p = strchr(e + 1, '"');
    }
    fails += geist_expect(idx >= 2, "at least one key/value pair");
}

int main(void) {
    GEIST_REQUIRE_GGUF(model_path);
    struct geist_backend *be = nullptr;
    if (geist_backend_create("auto", nullptr, nullptr, &be) != GEIST_OK)
        GEIST_SKIP("backend");
    struct geist_model *model = nullptr;
    if (geist_model_load(model_path, be, &model) != GEIST_OK)
        GEIST_SKIP("load");
    struct geist_session_opts o = {0};
    struct geist_session     *s = nullptr;
    if (geist_session_create(model, be, &o, &s) != GEIST_OK)
        GEIST_SKIP("session");

    struct geist_tool tools[] = {
            {.name        = "doc_search",
             .description = "die lokalen Dokumente nach einer Anfrage durchsuchen",
             .args_schema = "{\"query\": string}",
             .invoke      = stub_invoke},
            {.name        = "write_note",
             .description = "eine Notiz mit Titel und Text speichern",
             .args_schema = "{\"title\": string, \"text\": string}",
             .invoke      = stub_invoke},
    };
    geist_agent_init(&agent, model, s, 2, tools, 4, nullptr);

    char   turn[GEIST_AGENT_TURN_CAP];
    size_t tn;

    /* single-key tool: key is prefilled, value decoded */
    tn = masked_call("Search the docs for kv cache quantization", sizeof turn, turn);
    check_call("single-key", tn, turn);

    /* multi-key tool: keys constrained to the schema, each used at most once */
    tn = masked_call(
            "Save a note titled shopping with the text buy milk and bread", sizeof turn, turn);
    check_call("multi-key", tn, turn);

    geist_session_destroy(s);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    if (fails) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("agent masked-call grammar: parse + whitelist + schema keys pass\n");
    return GEIST_TEST_PASS;
}

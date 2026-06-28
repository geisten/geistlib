/*
 * test_agent_int — the full agent run loop against a real model (model-gated).
 *
 * Builds a one-tool agent (doc_search over a throwaway doc dir), runs a request,
 * and asserts MECHANICS, not content (greedy small-model output varies): the
 * loop returns OK with a non-empty answer inside max_steps and never runs an
 * off-whitelist tool. SKIPs cleanly without GEIST_GGUF_PATH. No assert().
 */
#define _POSIX_C_SOURCE 200809L

#include "test_helpers.h"

#include "../tools/agent.h"
#include "../tools/agent_docsearch.h"

#include <geist.h>

#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#define DOC_DIR "./.agent_int_test"

static struct geist_agent agent; /* 32 KB+ -> not on the stack */

int main(void) {
    GEIST_REQUIRE_GGUF(model_path);

    mkdir(DOC_DIR, 0755);
    FILE *f = fopen(DOC_DIR "/note.txt", "w");
    if (f) {
        fputs("The warranty period for the X200 blender is 24 months.\n", f);
        fclose(f);
    }

    struct geist_backend *be = nullptr;
    if (geist_backend_create("auto", nullptr, nullptr, &be) != GEIST_OK) {
        GEIST_SKIP("backend_create failed");
    }
    struct geist_model *model = nullptr;
    if (geist_model_load(model_path, be, &model) != GEIST_OK) {
        geist_backend_destroy(be);
        GEIST_SKIP("model_load failed (set GEIST_GGUF_PATH)");
    }
    struct geist_session_opts opts = {0}; /* greedy, deterministic */
    struct geist_session     *sess = nullptr;
    if (geist_session_create(model, be, &opts, &sess) != GEIST_OK) {
        geist_model_destroy(model);
        geist_backend_destroy(be);
        GEIST_SKIP("session_create failed");
    }

    struct geist_tool tools[] = {docsearch_tool(DOC_DIR)};
    geist_agent_init(&agent, model, sess, 1, tools, 4 /* max_steps */, nullptr /* default role */);

    int               rc  = GEIST_TEST_PASS;
    const char       *req = "How long is the warranty on the X200 blender?";
    char              resp[2048];
    size_t            rn = 0;
    enum geist_status st = geist_agent_run(&agent, strlen(req), req, sizeof resp, resp, &rn);

    fprintf(stderr, "agent_run -> st=%d, %zu bytes: \"%.200s\"\n", (int) st, rn, resp);
    if (st != GEIST_OK) {
        fprintf(stderr, "FAIL: agent_run did not return OK\n");
        rc = GEIST_TEST_FAIL;
    } else if (rn == 0 || resp[0] == '\0') {
        fprintf(stderr, "FAIL: agent_run produced an empty response\n");
        rc = GEIST_TEST_FAIL;
    }

    if (rc == GEIST_TEST_PASS) {
        printf("agent run loop works end to end\n");
    }

    /* constrained tool-name decode (item 3): drive it directly against the real
     * model with a 2-tool whitelist. Whatever the logits prefer, the result must
     * be a valid whitelist index spelling a real tool name — the guarantee that
     * an off-list name can never reach dispatch. */
    if (rc == GEIST_TEST_PASS) {
        struct geist_tool         two[] = {docsearch_tool(DOC_DIR),
                                           {.name        = "web_fetch",
                                            .args_schema = "{\"url\": string}",
                                            .invoke      = docsearch_invoke,
                                            .ctx         = (void *) (intptr_t) DOC_DIR}};
        static struct geist_agent ca;
        geist_agent_init(&ca, model, sess, 2, two, 4, nullptr);
        ca.tlen = agent_system_prompt(&ca, sizeof ca.transcript, ca.transcript);
        ca.tlen +=
                (size_t) snprintf(ca.transcript + ca.tlen,
                                  sizeof ca.transcript - ca.tlen,
                                  "Fetch https://example.com<end_of_turn>\n<start_of_turn>model\n");
        geist_token_t ids[GEIST_AGENT_NAME_CAP];
        size_t        nid = 0;
        int           idx = -1;
        if (geist_session_reset(sess) == GEIST_OK &&
            geist_session_set_prompt(sess, ca.transcript) == GEIST_OK &&
            geist_session_tokenize(sess, "{\"tool\":\"", sizeof ids / sizeof *ids, ids, &nid) ==
                    GEIST_OK &&
            geist_session_prefill_tokens(sess, nid, ids) == GEIST_OK) {
            idx = agent_decode_name_constrained(&ca);
        }
        fprintf(stderr,
                "constrained name decode -> idx=%d (%s)\n",
                idx,
                idx >= 0 ? two[idx].name : "<none>");
        if (idx < 0 || idx >= 2) {
            fprintf(stderr, "FAIL: constrained decode did not yield a whitelist tool\n");
            rc = GEIST_TEST_FAIL;
        } else {
            printf("constrained tool-name decode yields a whitelisted tool\n");
        }
    }

    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    remove(DOC_DIR "/note.txt");
    remove(DOC_DIR);
    return rc;
}

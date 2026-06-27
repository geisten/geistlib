/*
 * test_agent_summarize_route_int — the whole routing chain on a real document.
 *
 * Runs "Fasse die Datei wm2026_de.txt zusammen" through geist_agent_run with the
 * SAME two tools geist_shell ships (list_dir + summarize_file) and force_call on,
 * over a committed fixture (tests/data/wm2026_de.txt, a German Wikipedia article).
 * This is the end-to-end exercise of everything the router work added: name
 * scoring + PMI calibration + the file-name tie-breaker must pick summarize_file
 * (NOT list_dir) for a German "Fasse … zusammen" request that names a file, and
 * the forced single-shot call must then produce a real summary.
 *
 * Asserts the model-independent mechanics: agent_select_tool routes to
 * summarize_file, and the run returns OK with a non-empty answer shorter than the
 * 8 KB input (it summarized, didn't echo or list). We don't assert the wording.
 * Runs from the repo root (make test-int); SKIPs cleanly without GEIST_GGUF_PATH
 * or if the fixture isn't reachable. No assert().
 */
#define _POSIX_C_SOURCE 200809L

#include "test_helpers.h"

#include "../tools/agent.h"
#include "../tools/agent_listdir.h"
#include "../tools/agent_summarize.h"

#include <geist.h>

#include <stdio.h>
#include <string.h>

#define FIXTURE_DIR "tests/data"
#define FIXTURE_NAME "wm2026_de.txt"

static struct geist_agent agent;

int main(void) {
    GEIST_REQUIRE_GGUF(model_path);

    /* The fixture must be reachable from the cwd (repo root under make test-int);
     * measure its size for the "summary is shorter" check. */
    long  input_len = 0;
    FILE *fx        = fopen(FIXTURE_DIR "/" FIXTURE_NAME, "rb");
    if (fx == nullptr) {
        GEIST_SKIP("fixture " FIXTURE_DIR "/" FIXTURE_NAME " not reachable (run from repo root)");
    }
    fseek(fx, 0, SEEK_END);
    input_len = ftell(fx);
    fclose(fx);

    struct geist_backend *be = nullptr;
    if (geist_backend_create("auto", nullptr, nullptr, &be) != GEIST_OK) {
        GEIST_SKIP("backend_create failed");
    }
    struct geist_model *model = nullptr;
    if (geist_model_load(model_path, be, &model) != GEIST_OK) {
        geist_backend_destroy(be);
        GEIST_SKIP("model_load failed (set GEIST_GGUF_PATH)");
    }
    struct geist_session_opts opts = {0};
    struct geist_session     *sess = nullptr;
    if (geist_session_create(model, be, &opts, &sess) != GEIST_OK) {
        geist_model_destroy(model);
        geist_backend_destroy(be);
        GEIST_SKIP("session_create failed");
    }

    /* The exact two tools geist_shell ships; summarize confined to the fixture dir. */
    static struct summarize_ctx sctx;
    sctx = (struct summarize_ctx) {.model = model, .be = be, .root = FIXTURE_DIR};
    struct geist_tool tools[] = {listdir_tool(), summarize_file_tool(&sctx)};
    geist_agent_init(&agent,
                     model,
                     sess,
                     2,
                     tools,
                     4,
                     "Du bist ein Datei-Assistent. Nutze die passenden Werkzeuge.");
    agent.force_call = true;

    const char *req = "Fasse die Datei " FIXTURE_NAME " zusammen";

    /* 1) Routing: the chain must pick summarize_file (index 1), not list_dir. */
    int routed = agent_select_tool(&agent, strlen(req), req);
    fprintf(stderr,
            "routed -> %d (%s)\n",
            routed,
            routed >= 0 && routed < 2 ? tools[routed].name : "?");

    /* 2) End to end: force + run must yield a real summary (single-shot returns
     * the tool's observation as the answer). */
    char              resp[2048];
    size_t            rn = 0;
    enum geist_status st = geist_agent_run(&agent, strlen(req), req, sizeof resp, resp, &rn);
    fprintf(stderr,
            "run st=%d, input=%ld, answer=%zu bytes: %.200s\n",
            (int) st,
            input_len,
            rn,
            resp);

    int rc = GEIST_TEST_PASS;
    if (routed != 1) {
        fprintf(stderr,
                "FAIL: request routed to %s, expected summarize_file\n",
                routed >= 0 && routed < 2 ? tools[routed].name : "?");
        rc = GEIST_TEST_FAIL;
    } else if (st != GEIST_OK) {
        fprintf(stderr, "FAIL: agent_run did not return OK\n");
        rc = GEIST_TEST_FAIL;
    } else if (rn == 0 || resp[0] == '\0') {
        fprintf(stderr, "FAIL: empty answer\n");
        rc = GEIST_TEST_FAIL;
    } else if ((long) rn >= input_len) {
        fprintf(stderr, "FAIL: answer not shorter than the input (no summarization)\n");
        rc = GEIST_TEST_FAIL;
    } else if (strstr(resp, FIXTURE_NAME) != nullptr) {
        fprintf(stderr,
                "FAIL: answer contains the file name -> looks like a listing, not a summary\n");
        rc = GEIST_TEST_FAIL;
    }
    if (rc == GEIST_TEST_PASS) {
        printf("agent route: 'Fasse die Datei %s zusammen' -> summarize_file -> summary\n",
               FIXTURE_NAME);
    }

    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return rc;
}

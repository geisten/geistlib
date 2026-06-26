/*
 * test_agent_listdir_int — the full agent tool loop end to end (model-gated).
 *
 * Runs the request "Zeige mir den Inhalt des aktuellen Ordners" through
 * geist_agent_run with the real list_dir tool, in a throwaway directory holding
 * known files. force_call makes turn 0 a guaranteed list_dir call on ANY model
 * (no tool training needed), so this is deterministic; we then assert the loop
 * actually DISPATCHED list_dir and fed the real directory listing back as an
 * observation — i.e. the command ran and its output entered the conversation.
 *
 * We assert on the transcript (the observation), not the model's final prose:
 * a small model's summary is unreliable, but the dispatched tool's output is
 * exact. The tool here lists the cwd deterministically (ignoring the model's
 * forced "path" value) — that value is build-dependent (gcc/clang -ffast-math
 * differ in the greedy pick), so the test pins the LOOP MECHANICS (forced call
 * -> dispatched -> real listing observed), not the model's argument choice.
 * SKIPs cleanly without GEIST_GGUF_PATH. No assert().
 */
#define _POSIX_C_SOURCE 200809L

#include "test_helpers.h"

#include "../tools/agent.h"
#include "../tools/agent_listdir.h"

#include <geist.h>

#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define TMPDIR ".agent_listdir_int_test"

static struct geist_agent agent;

/* Real list_dir logic, but on a FIXED "." (cwd) — ignores the model's forced
 * path so the listing is deterministic regardless of which value greedy decode
 * picked on this build. */
static enum geist_status spy_listdir(void      *ctx,
                                     size_t     args_len,
                                     const char args[static args_len],
                                     size_t     out_cap,
                                     char       out[static out_cap],
                                     size_t    *out_len) {
    (void) ctx;
    (void) args_len;
    (void) args;
    const char *cwd = "{\"path\":\".\"}";
    return listdir_invoke(nullptr, strlen(cwd), cwd, out_cap, out, out_len);
}

int main(void) {
    GEIST_REQUIRE_GGUF(model_path);

    /* Load the model BEFORE chdir — model_path may be relative (CI passes
     * gguf_artifacts/...). */
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

    /* A throwaway "current folder" with known entries. */
    char cwd0[4096];
    if (getcwd(cwd0, sizeof cwd0) == nullptr) {
        GEIST_SKIP("getcwd failed");
    }
    mkdir(TMPDIR, 0755);
    FILE *f;
    if ((f = fopen(TMPDIR "/alpha.txt", "w")) != nullptr) {
        fclose(f);
    }
    if ((f = fopen(TMPDIR "/beta.md", "w")) != nullptr) {
        fclose(f);
    }
    if (chdir(TMPDIR) != 0) {
        GEIST_SKIP("chdir failed");
    }

    struct geist_tool tools[] = {{.name        = "list_dir",
                                  .args_schema = "{\"path\": string}",
                                  .invoke      = spy_listdir,
                                  .ctx         = nullptr}};
    geist_agent_init(
            &agent,
            model,
            sess,
            1,
            tools,
            4,
            "You are a file assistant. Use the list_dir tool to see the current directory.");
    agent.force_call = true; /* force turn 0 into a list_dir call — works on any model */

    char              resp[2048];
    size_t            rn  = 0;
    const char       *req = "Zeige mir den Inhalt des aktuellen Ordners";
    enum geist_status st  = geist_agent_run(&agent, strlen(req), req, sizeof resp, resp, &rn);

    if (chdir(cwd0) != 0) { /* restore before asserting / cleanup (best effort) */
        fprintf(stderr, "warning: could not restore cwd to %s\n", cwd0);
    }

    fprintf(stderr, "run st=%d, answer=\"%.120s\"\n", (int) st, resp);

    int rc = GEIST_TEST_PASS;
    /* The loop must have: returned OK, emitted a list_dir call, and fed the real
     * listing back as an observation (assumes the forced path is cwd-relative,
     * which greedy decoding yields — empty or "." both list the cwd). */
    if (st != GEIST_OK) {
        fprintf(stderr, "FAIL: agent_run did not return OK\n");
        rc = GEIST_TEST_FAIL;
    } else if (strstr(agent.transcript, "list_dir") == nullptr) {
        fprintf(stderr, "FAIL: no list_dir call in the transcript\n");
        rc = GEIST_TEST_FAIL;
    } else if (strstr(agent.transcript, "alpha.txt") == nullptr ||
               strstr(agent.transcript, "beta.md") == nullptr) {
        fprintf(stderr, "FAIL: directory contents not in the observation\n");
        rc = GEIST_TEST_FAIL;
    }
    if (rc == GEIST_TEST_PASS) {
        printf("agent list_dir loop: forced call -> dispatched -> contents observed\n");
    }

    remove(TMPDIR "/alpha.txt");
    remove(TMPDIR "/beta.md");
    remove(TMPDIR);
    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return rc;
}

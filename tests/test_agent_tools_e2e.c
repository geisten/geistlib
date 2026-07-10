/*
 * test_agent_tools_e2e — drive the built `geist agent` subcommand end to end
 * across a range of request shapes (model-gated).
 *
 * One `geist agent` process per scenario, real model load, the real whitelist-
 * gated loop. This asserts the WIRING and MECHANICS — that the subcommand loads
 * the model and runs the loop to a non-empty answer (exit 0) for each kind of
 * request, with the relevant env knobs (GEIST_FORCE_CALL, GEIST_DOCS,
 * GEIST_MIND_DIR) — NOT which tool the model routes to or the exact text.
 *
 * Why not assert the routed tool / its output? Routing is the model's choice
 * (agent_select_tool scores tool names from the logits), and with the full
 * 7-tool set those logits — hence the pick — differ across CPU backends
 * (Accelerate/AMX vs NEON), so a "routed to list_dir" assertion is not portable.
 * The tools themselves are verified deterministically, backend-independent, by
 * the unit/int tests: test_listdir_unit, test_agent_summarize_unit/_int,
 * test_agent_memory_unit, test_webfetch_int, test_websearch_unit, and the
 * 2-tool routing in test_agent_summarize_route_int. This e2e covers the layer
 * above them: the CLI subcommand + engine actually run for each request shape.
 *
 * SKIPs cleanly without a GGUF or if the geist binary can't be located.
 */
/* _XOPEN_SOURCE, not _POSIX_C_SOURCE: glibc guards realpath() behind the
 * X/Open extensions, so plain POSIX 2008 leaves it undeclared on Linux
 * (macOS declares it unconditionally — which is why this only breaks on the
 * Pi). XOPEN 700 is a superset of POSIX.1-2008. */
#define _XOPEN_SOURCE 700

#include "test_helpers.h"

#include "../tools/mind.h" /* mind_slurp + mind_remember for the recall fixture */

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define DIR_ "./.agent_tools_e2e"
#define OUTFILE DIR_ "/out.txt"
#define ERRFILE DIR_ "/err.txt"

static char g_bin[PATH_MAX + 32]; /* realpath'd dir + "/tools/geist" */
static char g_model[PATH_MAX];    /* absolute path to the GGUF (realpath target) */
static char g_dir[PATH_MAX];      /* absolute path to the fixture dir */
static int  fails = 0;

/* Run `[env] geist agent <model> "<req>" -n <n>`, capturing stdout->OUTFILE and
 * the trace (stderr)->ERRFILE. Returns the child's exit status. */
static int run_agent(const char *env, const char *req, int n) {
    char cmd[1 << 15];
    snprintf(cmd,
             sizeof cmd,
             "%s '%s' agent '%s' '%s' -n %d > '%s' 2> '%s'",
             env,
             g_bin,
             g_model,
             req,
             n,
             OUTFILE,
             ERRFILE);
    return system(cmd);
}

/* Assert one scenario ran end to end: the child exited 0 and produced a
 * non-empty answer on stdout. (Which tool ran / the exact text is the model's
 * backend-sensitive choice — see the file header — so it is not asserted here.) */
static void expect_scenario(const char *label, int rc) {
    static char out[1 << 16];
    mind_slurp(OUTFILE, out, sizeof out);
    char what[256];
    snprintf(what, sizeof what, "%s: exits 0", label);
    fails += geist_expect(rc == 0, what);
    snprintf(what, sizeof what, "%s: produced a non-empty answer", label);
    fails += geist_expect(out[0] != '\0', what);
}

int main(int argc, char **argv) {
    (void) argc;
    GEIST_REQUIRE_GGUF(model_path);
    if (!realpath(model_path, g_model)) {
        GEIST_SKIP("cannot resolve the model path");
    }

    /* locate the geist binary: bin/<t>/<m>/tests/<this> -> .../tools/geist */
    char self[PATH_MAX];
    if (!realpath(argv[0], self)) {
        GEIST_SKIP("cannot resolve argv[0]");
    }
    char *t = strstr(self, "/tests/");
    if (!t) {
        GEIST_SKIP("cannot locate geist from argv[0]");
    }
    *t = '\0';
    snprintf(g_bin, sizeof g_bin, "%s/tools/geist", self);
    FILE *bf = fopen(g_bin, "rb");
    if (!bf) {
        GEIST_SKIP("geist binary not built next to this test");
    }
    fclose(bf);

    /* fixture: a tidy dir with a file to summarize, a doc with a token, and a
     * pre-seeded mind dir for the recall scenario. */
    mkdir(DIR_, 0755);
    if (!realpath(DIR_, g_dir)) {
        GEIST_SKIP("cannot resolve the fixture dir");
    }
    FILE *f;
    if ((f = fopen(DIR_ "/report.md", "w"))) {
        fputs("# Q3 report\nWe migrate billing to the new ledger service in Q3.\n", f);
        fclose(f);
    }
    if ((f = fopen(DIR_ "/facts.md", "w"))) {
        fputs("# Facts\nThe rarest secret fruit in the vault is the pineapple.\n", f);
        fclose(f);
    }

    char mind_recall_dir[PATH_MAX + 32];
    snprintf(mind_recall_dir, sizeof mind_recall_dir, "%s/mind_recall", g_dir);
    setenv("GEIST_MIND_DIR", mind_recall_dir, 1);
    mind_remember("Recall Fixture", "the vault password is hunter2");

    char mind_write_dir[PATH_MAX + 32];
    snprintf(mind_write_dir, sizeof mind_write_dir, "%s/mind_write", g_dir);

    char env[PATH_MAX * 2];
    char req[PATH_MAX + 256];
    char forced[] = "GEIST_FORCE_CALL=1 GEIST_AGENT_TRACE=1";

    /* 1. a directory-listing request (forced tool call) */
    snprintf(req, sizeof req, "List the files in the directory %s", g_dir);
    expect_scenario("list_dir", run_agent(forced, req, 2));

    /* 2. a file-summary request (forced) */
    snprintf(req, sizeof req, "Summarize the file %s/report.md", g_dir);
    expect_scenario("summarize_file", run_agent(forced, req, 4));

    /* 3. a local-document search (GEIST_DOCS set, forced) */
    snprintf(env, sizeof env, "%s GEIST_DOCS='%s'", forced, g_dir);
    expect_scenario("doc_search", run_agent(env, "Search the documents for pineapple", 2));

    /* 4. a save-a-note request (fresh GEIST_MIND_DIR, forced) */
    snprintf(env, sizeof env, "%s GEIST_MIND_DIR='%s'", forced, mind_write_dir);
    expect_scenario("remember", run_agent(env, "Remember that the magic number is 42", 2));

    /* 5. a recall request against the pre-seeded note (forced) */
    snprintf(env, sizeof env, "%s GEIST_MIND_DIR='%s'", forced, mind_recall_dir);
    expect_scenario("recall", run_agent(env, "recall-fixture", 2));

    /* 6. a plain question with forcing off (GEIST_FORCE_CALL=0): the model answers
     * directly, no tool. Forcing is on by default, so opt out explicitly here. */
    expect_scenario("plain_answer",
                    run_agent("GEIST_FORCE_CALL=0", "What is the capital of France?", 8));

    /* cleanup */
    remove(DIR_ "/report.md");
    remove(DIR_ "/facts.md");
    remove(OUTFILE);
    remove(ERRFILE);
    char rmcmd[PATH_MAX + 32];
    snprintf(rmcmd, sizeof rmcmd, "rm -rf '%s'", g_dir);
    if (system(rmcmd) != 0) {
        /* non-fatal: a leftover fixture dir doesn't affect the verdict */
    }

    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("geist agent e2e: subcommand runs end to end across 6 request shapes\n");
    return GEIST_TEST_PASS;
}

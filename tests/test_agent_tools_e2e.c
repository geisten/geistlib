/*
 * test_agent_tools_e2e — drive the built `geist agent` subcommand across every
 * deterministic (non-network) tool scenario end to end (model-gated).
 *
 * One `geist agent` process per scenario, real model load, the real whitelist-
 * gated loop. With GEIST_FORCE_CALL=1 the routed tool is forced, so routing is
 * deterministic and we can assert BOTH the routed tool (from the GEIST_AGENT_TRACE
 * stream on stderr) and the tool's effect (stdout / on-disk side effect):
 *
 *   list_dir       -> lists a fixture dir, output names a known file
 *   summarize_file -> routes to summarize_file for a named file, answers non-empty
 *   doc_search     -> finds a unique token planted in a GEIST_DOCS folder
 *   remember       -> writes a note + INDEX.md under GEIST_MIND_DIR
 *   recall         -> loads a pre-written note's body back out
 *   plain answer   -> no force, no tool: the model answers directly, exits 0
 *
 * The network tools (web_search / web_fetch) are covered without the internet by
 * test_websearch_unit + test_webfetch_int (loopback server); driving them through
 * the model would need live DuckDuckGo, so they are intentionally out of scope here.
 *
 * SKIPs cleanly without a GGUF or if the geist binary can't be located. Asserts
 * MECHANICS + routing, not generated content. No assert() — exit code carries it.
 */
#define _POSIX_C_SOURCE 200809L

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

static char g_bin[PATH_MAX];   /* absolute path to the geist binary */
static char g_model[PATH_MAX]; /* absolute path to the GGUF */
static char g_dir[PATH_MAX];   /* absolute path to the fixture dir */
static int  fails = 0;

/* Run `[env] geist agent <model> "<req>" -n <n>`, capturing stdout->OUTFILE and
 * the trace (stderr)->ERRFILE. Returns the child's exit status. */
static int run_agent(const char *env, const char *req, int n) {
    char cmd[1 << 14];
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

/* Assert one scenario: child exited 0, the trace named `tool` (nullptr to skip),
 * and stdout contains `needle` (nullptr to skip). */
static void expect_scenario(const char *label, int rc, const char *tool, const char *needle) {
    static char out[1 << 16], err[1 << 16];
    mind_slurp(OUTFILE, out, sizeof out);
    mind_slurp(ERRFILE, err, sizeof err);
    char what[256];
    snprintf(what, sizeof what, "%s: exits 0", label);
    fails += geist_expect(rc == 0, what);
    if (tool) {
        snprintf(what, sizeof what, "%s: routed to %s (trace)", label, tool);
        fails += geist_expect(strstr(err, tool) != nullptr, what);
    }
    if (needle) {
        snprintf(what, sizeof what, "%s: output contains \"%s\"", label, needle);
        fails += geist_expect(strstr(out, needle) != nullptr, what);
    }
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

    /* fixture: a tidy dir with a file to summarize, a doc with a unique token,
     * and two empty mind dirs (one to write into, one pre-seeded for recall). */
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

    /* seed a note for the recall scenario in a dedicated mind dir. */
    char mind_recall_dir[PATH_MAX];
    snprintf(mind_recall_dir, sizeof mind_recall_dir, "%s/mind_recall", g_dir);
    setenv("GEIST_MIND_DIR", mind_recall_dir, 1);
    mind_remember("Recall Fixture", "the vault password is hunter2");

    char env[PATH_MAX * 2];
    char req[PATH_MAX + 256];

    /* 1. list_dir — force the listing of the fixture dir; output names report.md */
    snprintf(req, sizeof req, "List the files in the directory %s", g_dir);
    expect_scenario("list_dir", run_agent("GEIST_FORCE_CALL=1 GEIST_AGENT_TRACE=1", req, 2),
                    "list_dir", "report.md");

    /* 2. summarize_file — force; routes to summarize_file for the named file */
    snprintf(req, sizeof req, "Summarize the file %s/report.md", g_dir);
    expect_scenario("summarize_file", run_agent("GEIST_FORCE_CALL=1 GEIST_AGENT_TRACE=1", req, 4),
                    "summarize_file", nullptr); /* summary content varies */

    /* 3. doc_search — the unique token "pineapple" planted in facts.md */
    snprintf(env, sizeof env, "GEIST_FORCE_CALL=1 GEIST_AGENT_TRACE=1 GEIST_DOCS='%s'", g_dir);
    expect_scenario("doc_search", run_agent(env, "Search the documents for pineapple", 2),
                    "doc_search", "pineapple");

    /* 4. remember — force; a note + INDEX.md land under a fresh mind dir */
    char mind_write_dir[PATH_MAX];
    snprintf(mind_write_dir, sizeof mind_write_dir, "%s/mind_write", g_dir);
    snprintf(env, sizeof env, "GEIST_FORCE_CALL=1 GEIST_AGENT_TRACE=1 GEIST_MIND_DIR='%s'",
             mind_write_dir);
    expect_scenario("remember", run_agent(env, "Remember that the magic number is 42", 2),
                    "remember", nullptr);
    char ipath[PATH_MAX], buf[8192];
    snprintf(ipath, sizeof ipath, "%s/INDEX.md", mind_write_dir);
    fails += geist_expect(mind_slurp(ipath, buf, sizeof buf) > 0, "remember: wrote INDEX.md on disk");

    /* 5. recall — load the pre-seeded note's body back out. A forced single-arg
     * call lifts the whole request as the slug (a slug isn't a path-like locator),
     * so the request IS the slug: "recall-fixture". */
    snprintf(env, sizeof env, "GEIST_FORCE_CALL=1 GEIST_AGENT_TRACE=1 GEIST_MIND_DIR='%s'",
             mind_recall_dir);
    expect_scenario("recall", run_agent(env, "recall-fixture", 2), "recall", "hunter2");

    /* 6. plain answer — no force, no tool needed: the model answers, exits 0 */
    expect_scenario("plain_answer", run_agent("", "What is the capital of France?", 8), nullptr,
                    nullptr);
    mind_slurp(OUTFILE, buf, sizeof buf);
    fails += geist_expect(buf[0] != '\0', "plain_answer: non-empty answer");

    /* cleanup */
    remove(DIR_ "/report.md");
    remove(DIR_ "/facts.md");
    remove(OUTFILE);
    remove(ERRFILE);
    /* mind dirs hold generated note files of unknown name; remove best-effort */
    char rmcmd[PATH_MAX + 32];
    snprintf(rmcmd, sizeof rmcmd, "rm -rf '%s'", g_dir);
    if (system(rmcmd) != 0) {
        /* non-fatal: leftover fixture dir doesn't affect the verdict */
    }

    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("geist agent e2e: list_dir, summarize, doc_search, remember, recall, plain answer\n");
    return GEIST_TEST_PASS;
}

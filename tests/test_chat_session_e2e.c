/*
 * test_chat_session_e2e — drive the built `geist chat` subcommand through a full
 * scripted conversation end to end (model-gated), in ONE model load.
 *
 * Unlike test_chat_e2e (palace slash commands only, no generation), this exercises
 * the conversation engine + memory together:
 *
 *   - multi-turn memory: "My name is Aramis." then "What is my name?" -> the
 *     answer echoes "Aramis" (proves the transcript carries across turns, i.e.
 *     geist_agent_run's conversation mode);
 *   - /remember + /notes: a note is written and the index lists it;
 *   - /help: prints the command list;
 *   - /recall: queues a note for the next turn and acknowledges;
 *   - an empty line is a no-op prompt (no crash);
 *   - /quit exits 0; the note + INDEX.md exist on disk afterwards.
 *
 * Greedy decode is deterministic, so the "Aramis" echo is reproducible for a
 * given model (not a coin flip). Asserts MECHANICS, not free-form phrasing.
 *
 * SKIPs cleanly without a GGUF or if the geist binary can't be located.
 */
#define _POSIX_C_SOURCE 200809L

#include "test_helpers.h"

#include "../tools/mind.h" /* mind_slurp */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define E2E_DIR "./.chat_session_e2e"
#define SCRIPT E2E_DIR "/script.txt"
#define OUTFILE E2E_DIR "/out.txt"

static int fails = 0;

int main(int argc, char **argv) {
    (void) argc;
    GEIST_REQUIRE_GGUF(model_path);

    /* bin/<t>/<m>/tests/test_chat_session_e2e -> bin/<t>/<m>/tools/geist */
    char self[1024];
    snprintf(self, sizeof self, "%s", argv[0]);
    char *t = strstr(self, "/tests/");
    if (!t) {
        GEIST_SKIP("cannot locate geist from argv[0]");
    }
    *t = '\0';
    char bin[1100];
    snprintf(bin, sizeof bin, "%s/tools/geist", self);
    FILE *bf = fopen(bin, "rb");
    if (!bf) {
        GEIST_SKIP("geist binary not built next to this test");
    }
    fclose(bf);

    mkdir(E2E_DIR, 0755);
    setenv("GEIST_MIND_DIR", E2E_DIR, 1); /* the child inherits this */

    /* A scripted session: two chat turns that test memory, then the slash
     * commands, an empty line (no-op), a /recall, and /quit. */
    FILE *s = fopen(SCRIPT, "w");
    if (!s) {
        GEIST_SKIP("cannot write the REPL script");
    }
    fputs("My name is Aramis.\n"
          "What is my name?\n"
          "/remember Capital fact | Paris is the capital of France.\n"
          "/notes\n"
          "/help\n"
          "\n"
          "/recall capital-fact\n"
          "/quit\n",
          s);
    fclose(s);

    char cmd[4096];
    snprintf(cmd,
             sizeof cmd,
             "'%s' chat '%s' < '%s' > '%s' 2>/dev/null",
             bin,
             model_path,
             SCRIPT,
             OUTFILE);
    int rc = system(cmd);
    fails += geist_expect(rc == 0, "geist chat exited 0");

    static char out[1 << 16];
    mind_slurp(OUTFILE, out, sizeof out);

    /* conversation memory: turn 2 recalls the name from turn 1. */
    fails += geist_expect(strstr(out, "Aramis") != nullptr,
                          "multi-turn: 'What is my name?' echoes Aramis from the prior turn");
    /* /notes printed the freshly remembered index entry. */
    fails += geist_expect(strstr(out, "Capital fact") != nullptr, "/notes lists the new note");
    /* /help printed the command list. */
    fails += geist_expect(strstr(out, "/remember") != nullptr, "/help prints commands");
    /* /recall acknowledged the slug. */
    fails += geist_expect(strstr(out, "recalled capital-fact") != nullptr, "/recall acknowledged");

    /* on-disk side effects of /remember. */
    static char buf[8192];
    mind_slurp(E2E_DIR "/capital-fact.md", buf, sizeof buf);
    fails += geist_expect(strstr(buf, "Paris is the capital of France") != nullptr,
                          "note file written with the body");
    mind_slurp(E2E_DIR "/INDEX.md", buf, sizeof buf);
    fails += geist_expect(strstr(buf, "[Capital fact](capital-fact.md)") != nullptr,
                          "INDEX.md links the note");

    remove(E2E_DIR "/capital-fact.md");
    remove(E2E_DIR "/INDEX.md");
    remove(SCRIPT);
    remove(OUTFILE);
    remove(E2E_DIR);

    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("geist chat e2e: multi-turn memory + slash commands in one session\n");
    return GEIST_TEST_PASS;
}

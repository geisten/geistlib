/*
 * test_chat_e2e — drive the built geist_chat binary end to end (model-gated).
 *
 * The real tool, real model load, real REPL: feed a script of palace commands
 * on stdin and check both the on-disk side effects (note file + INDEX.md) and
 * the REPL's stdout (/notes, /recall). No chat turn — palace commands only, so
 * it stays fast (no generation). The geist_chat path is derived from argv[0]
 * (sibling tools/ dir of this test's bin dir).
 *
 * SKIPs cleanly without a GGUF or if the geist_chat binary can't be located.
 * No assert() — exit codes carry PASS/FAIL/SKIP.
 */
#define _POSIX_C_SOURCE 200809L

#include "test_helpers.h"

#include "../tools/mind.h" /* mind_slurp */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define E2E_DIR "./.mind_e2e_test"
#define SCRIPT E2E_DIR "/script.txt"
#define OUTFILE E2E_DIR "/out.txt"

static int fails = 0;

int main(int argc, char **argv) {
    (void) argc;
    GEIST_REQUIRE_GGUF(model_path);

    /* geist_chat lives in the sibling tools/ dir of this test's bin dir:
     * bin/<t>/<m>/tests/test_chat_e2e -> bin/<t>/<m>/tools/geist_chat */
    char self[1024];
    snprintf(self, sizeof self, "%s", argv[0]);
    char *t = strstr(self, "/tests/");
    if (!t) {
        GEIST_SKIP("cannot locate geist_chat from argv[0]");
    }
    *t = '\0';
    char bin[1100];
    snprintf(bin, sizeof bin, "%s/tools/geist_chat", self);
    FILE *bf = fopen(bin, "rb");
    if (!bf) {
        GEIST_SKIP("geist_chat binary not built next to this test");
    }
    fclose(bf);

    mkdir(E2E_DIR, 0755);
    setenv("GEIST_MIND_DIR", E2E_DIR, 1); /* the child inherits this */

    FILE *s = fopen(SCRIPT, "w");
    if (!s) {
        GEIST_SKIP("cannot write the REPL script");
    }
    fputs("/remember E2E Note | stored by the e2e test\n"
          "/notes\n"
          "/recall e2e-note\n"
          "/quit\n",
          s);
    fclose(s);

    char cmd[4096];
    snprintf(cmd,
             sizeof cmd,
             "'%s' '%s' < '%s' > '%s' 2>/dev/null",
             bin,
             model_path,
             SCRIPT,
             OUTFILE);
    int rc = system(cmd);
    fails += geist_expect(rc == 0, "geist_chat exited 0");

    char buf[8192];
    /* on-disk side effects */
    mind_slurp(E2E_DIR "/e2e-note.md", buf, sizeof buf);
    fails += geist_expect(strstr(buf, "stored by the e2e test") != nullptr,
                          "note file written with the body");
    mind_slurp(E2E_DIR "/INDEX.md", buf, sizeof buf);
    fails += geist_expect(strstr(buf, "[E2E Note](e2e-note.md)") != nullptr,
                          "INDEX.md links the note");
    /* REPL stdout proves the commands ran */
    mind_slurp(OUTFILE, buf, sizeof buf);
    fails += geist_expect(strstr(buf, "[E2E Note](e2e-note.md)") != nullptr,
                          "/notes printed the index");
    fails += geist_expect(strstr(buf, "recalled e2e-note into context") != nullptr,
                          "/recall acknowledged");

    remove(E2E_DIR "/e2e-note.md");
    remove(E2E_DIR "/INDEX.md");
    remove(SCRIPT);
    remove(OUTFILE);
    remove(E2E_DIR);

    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("geist_chat e2e: palace commands work end to end\n");
    return GEIST_TEST_PASS;
}

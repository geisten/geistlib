/*
 * test_agent_main_e2e — drive the built geist_agent CLI end to end (model-gated).
 * Real binary, real model, the whitelist-gated loop over a throwaway doc dir
 * (GEIST_DOCS). Asserts MECHANICS (exits 0, prints a non-empty answer within a
 * small step budget), not content. SKIPs without a GGUF or the binary. No
 * assert() — exit codes carry PASS/FAIL/SKIP.
 */
#define _POSIX_C_SOURCE 200809L /* mkdir, setenv, system */

#include "test_helpers.h"

#include "../tools/mind.h" /* mind_slurp */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define E2E_DIR "./.agent_main_e2e"
#define OUTFILE E2E_DIR "/out.txt"

static int fails = 0;

int main(int argc, char **argv) {
    (void) argc;
    GEIST_REQUIRE_GGUF(model_path);

    /* geist_agent is the sibling tools/ binary of this test's bin dir. */
    char self[1024];
    snprintf(self, sizeof self, "%s", argv[0]);
    char *t = strstr(self, "/tests/");
    if (!t) {
        GEIST_SKIP("cannot locate geist_agent from argv[0]");
    }
    *t = '\0';
    char bin[1100];
    snprintf(bin, sizeof bin, "%s/tools/geist_agent", self);
    FILE *bf = fopen(bin, "rb");
    if (!bf) {
        GEIST_SKIP("geist_agent binary not built next to this test");
    }
    fclose(bf);

    mkdir(E2E_DIR, 0755);
    FILE *f = fopen(E2E_DIR "/manual.md", "w");
    if (f) {
        fputs("# X200 blender\nThe warranty period for the X200 blender is 24 months.\n", f);
        fclose(f);
    }
    setenv("GEIST_DOCS", E2E_DIR, 1); /* the child inherits this */

    /* one-shot, small step budget to keep it quick */
    char cmd[4096];
    snprintf(cmd,
             sizeof cmd,
             "'%s' '%s' 'How long is the warranty on the X200 blender?' -n 2 > '%s' 2>/dev/null",
             bin,
             model_path,
             OUTFILE);
    int rc = system(cmd);
    fails += geist_expect(rc == 0, "geist_agent exited 0");

    char buf[8192];
    mind_slurp(OUTFILE, buf, sizeof buf);
    fprintf(stderr, "answer: \"%.200s\"\n", buf);
    fails += geist_expect(buf[0] != '\0', "geist_agent printed a non-empty answer");

    remove(E2E_DIR "/manual.md");
    remove(OUTFILE);
    remove(E2E_DIR);

    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("geist_agent e2e: CLI answers from a doc folder\n");
    return GEIST_TEST_PASS;
}

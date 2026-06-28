/*
 * test_listdir_unit — the list_dir tool, no model.
 *
 * Lists a throwaway directory and checks: non-hidden entries are returned,
 * dotfiles are skipped (like `ls`), and a shell-metacharacter "path" is treated
 * as a plain (unopenable) directory name — proving there is no command
 * execution, just opendir(). No assert(); exit code carries PASS/FAIL.
 */
#define _POSIX_C_SOURCE 200809L

#include "test_helpers.h"

#include "../tools/agent_listdir.h"

#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define DIR_ "./.listdir_unit_test"

static int run(const char *json, char *out, size_t cap) {
    size_t n = 0;
    listdir_invoke(nullptr, strlen(json), json, cap, out, &n);
    return (int) n;
}

int main(void) {
    int fails = 0;

    mkdir(DIR_, 0755);
    FILE *f;
    if ((f = fopen(DIR_ "/alpha.txt", "w"))) {
        fclose(f);
    }
    if ((f = fopen(DIR_ "/beta.md", "w"))) {
        fclose(f);
    }
    if ((f = fopen(DIR_ "/.hidden", "w"))) {
        fclose(f); /* must be skipped */
    }

    char out[4096];

    run("{\"path\":\"" DIR_ "\"}", out, sizeof out);
    fails += geist_expect(strstr(out, "alpha.txt") != nullptr, "listdir: lists a file");
    fails += geist_expect(strstr(out, "beta.md") != nullptr, "listdir: lists another file");
    fails += geist_expect(strstr(out, ".hidden") == nullptr, "listdir: skips dotfiles (like ls)");

    /* A shell-metacharacter path is just a directory name that fails to open —
     * never executed. This is the no-shell guarantee. */
    run("{\"path\":\"" DIR_ "/nope; touch pwned\"}", out, sizeof out);
    fails += geist_expect(strstr(out, "error: cannot open") != nullptr,
                          "listdir: injection-y path is an unopenable name, not a command");
    fails += geist_expect(access("pwned", F_OK) != 0, "listdir: no command ran (no 'pwned' file)");

    /* Missing path defaults to "." (cwd) — non-empty, no crash. */
    fails += geist_expect(run("{}", out, sizeof out) > 0, "listdir: empty args defaults to cwd");

    remove(DIR_ "/alpha.txt");
    remove(DIR_ "/beta.md");
    remove(DIR_ "/.hidden");
    remove(DIR_);

    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("list_dir: lists entries, skips dotfiles, no shell\n");
    return GEIST_TEST_PASS;
}

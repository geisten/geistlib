/*
 * test_agent_memory_unit — the remember / recall tools, no model.
 *
 * Writes a note via remember(text) (title auto-derived from the first line),
 * checks the note + INDEX.md landed, then loads it back via recall(slug) and
 * checks the bad-input paths return errors (not crashes). No assert(); exit code
 * carries PASS/FAIL.
 */
#define _POSIX_C_SOURCE 200809L

#include "test_helpers.h"

#include "../tools/agent_memory.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MIND_ "./.agent_memory_unit"

static int remember(const char *json, char *out, size_t cap) {
    size_t n = 0;
    memory_remember_invoke(nullptr, strlen(json), json, cap, out, &n);
    return (int) n;
}
static int recall(const char *json, char *out, size_t cap) {
    size_t n = 0;
    memory_recall_invoke(nullptr, strlen(json), json, cap, out, &n);
    return (int) n;
}

int main(void) {
    int fails = 0;
    setenv("GEIST_MIND_DIR", MIND_, 1);

    char out[8192];

    /* remember: the (single-line) text becomes the title -> slug. */
    remember("{\"text\":\"Buy milk from the store\"}", out, sizeof out);
    fails += geist_expect(strstr(out, "remembered") != nullptr, "remember: confirms write");
    fails += geist_expect(strstr(out, "buy-milk-from-the-store") != nullptr,
                          "remember: slug derived from the text");

    char idx[4096];
    mind_slurp(MIND_ "/INDEX.md", idx, sizeof idx);
    fails += geist_expect(strstr(idx, "buy-milk-from-the-store.md") != nullptr,
                          "remember: indexed the note");

    /* recall the slug we just wrote -> body present. */
    long got = recall("{\"slug\":\"buy-milk-from-the-store\"}", out, sizeof out);
    fails += geist_expect(got > 0 && strstr(out, "Buy milk from the store") != nullptr,
                          "recall: loads the note body");

    /* bad inputs return errors, never crash. */
    remember("{}", out, sizeof out);
    fails += geist_expect(strstr(out, "error") != nullptr, "remember: empty text -> error");
    recall("{\"slug\":\"does-not-exist\"}", out, sizeof out);
    fails += geist_expect(strstr(out, "error") != nullptr, "recall: missing note -> error");

    remove(MIND_ "/buy-milk-from-the-store.md");
    remove(MIND_ "/INDEX.md");
    remove(MIND_);

    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("memory tools: remember writes + indexes, recall loads, bad input errors\n");
    return GEIST_TEST_PASS;
}

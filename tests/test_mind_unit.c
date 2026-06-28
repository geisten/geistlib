/*
 * test_mind_unit — the file-based memory palace (tools/mind.h), no model.
 *
 * Covers slugify edge cases, the remember -> recall roundtrip, the INDEX.md
 * append, and the missing-note failure path. Runs in a throwaway dir under
 * GEIST_MIND_DIR so it never touches a real ./mind. No assert() (AGENT.md):
 * checks set a fail flag and the exit code carries PASS/FAIL.
 */
#define _POSIX_C_SOURCE 200809L

#include "test_helpers.h"

#include "../tools/mind.h"

#include <stdio.h>
#include <string.h>

#define MIND_TEST_DIR "./.mind_unit_test"

static int fails = 0;

static void expect_slug(const char *title, const char *want) {
    char got[256];
    mind_slugify(title, got, sizeof got);
    if (strcmp(got, want) != 0) {
        fprintf(stderr, "FAIL: slugify(\"%s\") = \"%s\", want \"%s\"\n", title, got, want);
        fails++;
    }
}

static void cleanup(void) {
    remove(MIND_TEST_DIR "/test-note.md");
    remove(MIND_TEST_DIR "/second.md");
    remove(MIND_TEST_DIR "/INDEX.md");
    remove(MIND_TEST_DIR);
}

int main(void) {
    setenv("GEIST_MIND_DIR", MIND_TEST_DIR, 1);
    cleanup(); /* start from a clean slate even if a prior run died */

    /* ---- slugify: case, punctuation, trimming, empty ---- */
    expect_slug("My Web Note!", "my-web-note");
    expect_slug("ALLCAPS", "allcaps");
    expect_slug("  --Spaces & dashes--  ", "spaces-dashes");
    expect_slug("café 2026", "caf-2026"); /* non-ASCII bytes are non-alnum -> separators */
    expect_slug("!!!", "note");           /* all-punctuation collapses to a valid slug */
    expect_slug("", "note");              /* empty title */

    /* ---- remember writes a note with frontmatter + body ---- */
    fails += geist_expect(mind_remember("Test Note", "hello world contents") == 0,
                          "remember returns 0");
    char buf[4096];
    long n = mind_recall("test-note", buf, sizeof buf);
    fails += geist_expect(n > 0, "recall finds the note");
    fails += geist_expect(strstr(buf, "title: Test Note") != nullptr, "note has frontmatter title");
    fails += geist_expect(strstr(buf, "hello world contents") != nullptr, "note has the body");

    /* ---- the index gained a linked line ---- */
    long ni = mind_slurp(MIND_TEST_DIR "/INDEX.md", buf, sizeof buf);
    fails += geist_expect(ni > 0, "INDEX.md exists");
    fails += geist_expect(strstr(buf, "[Test Note](test-note.md)") != nullptr,
                          "index links the note");

    /* ---- a second note appends, not overwrites ---- */
    fails +=
            geist_expect(mind_remember("Second", "another note") == 0, "remember second returns 0");
    mind_slurp(MIND_TEST_DIR "/INDEX.md", buf, sizeof buf);
    fails += geist_expect(strstr(buf, "[Test Note](test-note.md)") != nullptr,
                          "first index line survives");
    fails += geist_expect(strstr(buf, "[Second](second.md)") != nullptr,
                          "second index line appended");

    /* ---- recall of a missing note fails cleanly ---- */
    fails += geist_expect(mind_recall("does-not-exist", buf, sizeof buf) < 0,
                          "recall missing note returns -1");

    cleanup();
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("mind palace: all checks passed\n");
    return GEIST_TEST_PASS;
}

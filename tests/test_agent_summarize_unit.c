/*
 * test_agent_summarize_unit — the model-free parts of summarize_file.
 *
 * Pins the path confinement (reject absolute / "..") and the chunk-boundary
 * logic (paragraph preferred, line fallback, hard split). The refine loop itself
 * is exercised in the _int test against a real model. No assert().
 */
#define _POSIX_C_SOURCE 200809L

#include "test_helpers.h"

#include "../tools/agent_summarize.h"

#include <stdio.h>
#include <string.h>

static int fails = 0;

static void test_confine(void) {
    fails += geist_expect(summ_path_ok("notes.md"), "confine: plain name ok");
    fails += geist_expect(summ_path_ok("sub/dir/file.txt"), "confine: relative subdir ok");
    fails += geist_expect(!summ_path_ok("/etc/passwd"), "confine: absolute path rejected");
    fails += geist_expect(!summ_path_ok("../secret"), "confine: leading .. rejected");
    fails += geist_expect(!summ_path_ok("a/../../etc/x"), "confine: embedded .. rejected");
}

static void test_chunk(void) {
    /* Two paragraphs; a ~10-byte target should cut at the blank line between. */
    const char *para = "first line\n\nsecond paragraph here";
    size_t      end  = summ_next_chunk(para, strlen(para), 0, 10);
    fails += geist_expect(end == strlen("first line\n\n"),
                          "chunk: cuts at the paragraph (blank-line) boundary");

    /* No blank line within range -> fall back to the last newline. */
    const char *lines = "line one\nline two\nline three";
    end               = summ_next_chunk(lines, strlen(lines), 0, 12);
    fails += geist_expect(end == strlen("line one\n"), "chunk: falls back to a line boundary");

    /* A single overlong line with no boundary -> hard split at the target. */
    const char *one = "aaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    end             = summ_next_chunk(one, strlen(one), 0, 8);
    fails += geist_expect(end == 8, "chunk: hard-splits a boundary-less long line");

    /* Whole remainder fits -> returns end of text. */
    end = summ_next_chunk(one, strlen(one), 0, 1000);
    fails += geist_expect(end == strlen(one), "chunk: returns len when the rest fits");
}

static void test_refine_usable(void) {
    fails += geist_expect(summ_refine_usable(strlen("The WM 2026 is in three countries."),
                                             "The WM 2026 is in three countries."),
                          "refine: real content is usable");
    fails += geist_expect(!summ_refine_usable(0, ""), "refine: empty is not usable");
    fails += geist_expect(!summ_refine_usable(3, "   "), "refine: whitespace-only is not usable");
    fails += geist_expect(!summ_refine_usable(strlen("Summary so far:  "), "Summary so far:  "),
                          "refine: scaffold echo is not usable");
    fails += geist_expect(!summ_refine_usable(strlen("\n Summary so far:"), "\n Summary so far:"),
                          "refine: scaffold echo after whitespace is not usable");
}

int main(void) {
    test_confine();
    test_chunk();
    test_refine_usable();
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("summarize: path confinement + chunk boundaries pass\n");
    return GEIST_TEST_PASS;
}

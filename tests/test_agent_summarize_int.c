/*
 * test_agent_summarize_int — the summarize_file core end to end (model-gated).
 *
 * Writes a multi-chunk text file (> SUMM_CHUNK, so the refine/rolling loop runs
 * more than once), summarizes it, and asserts the MODEL-INDEPENDENT mechanics:
 * returns OK, a non-empty summary, SHORTER than the input (it summarized, didn't
 * echo). We do NOT assert the summary's wording — that's model-quality- and
 * build-dependent (same lesson as the forced-arg value in the list_dir test) —
 * but we print it for eyeballing. SKIPs cleanly without GEIST_GGUF_PATH.
 */
#define _POSIX_C_SOURCE 200809L

#include "test_helpers.h"

#include "../tools/agent_summarize.h"

#include <geist.h>

#include <stdio.h>
#include <string.h>

#define TMPFILE "./.summarize_int_test.txt"

int main(void) {
    GEIST_REQUIRE_GGUF(model_path);

    struct geist_backend *be = nullptr;
    if (geist_backend_create("auto", nullptr, nullptr, &be) != GEIST_OK) {
        GEIST_SKIP("backend_create failed");
    }
    struct geist_model *model = nullptr;
    if (geist_model_load(model_path, be, &model) != GEIST_OK) {
        geist_backend_destroy(be);
        GEIST_SKIP("model_load failed (set GEIST_GGUF_PATH)");
    }

    /* A multi-paragraph document > SUMM_CHUNK bytes so chunking + refine kick in. */
    FILE *f = fopen(TMPFILE, "w");
    if (f == nullptr) {
        geist_model_destroy(model);
        geist_backend_destroy(be);
        GEIST_SKIP("could not write the test file");
    }
    size_t input_len = 0;
    for (int p = 0; p < 24; p++) {
        input_len += (size_t) fprintf(
                f,
                "Paragraph %d. The Apollo program landed astronauts on the Moon between 1969 and "
                "1972. It was run by NASA and remains a landmark of human spaceflight.\n\n",
                p);
    }
    fclose(f);

    char              summary[2048];
    size_t            sn = 0;
    enum geist_status st = summarize_file(model, be, ".", TMPFILE, sizeof summary, summary, &sn);

    fprintf(stderr, "input=%zu bytes, summary=%zu bytes\n", input_len, sn);
    fprintf(stderr, "summary: %.300s\n", summary);

    int rc = GEIST_TEST_PASS;
    if (st != GEIST_OK) {
        fprintf(stderr, "FAIL: summarize_file did not return OK\n");
        rc = GEIST_TEST_FAIL;
    } else if (sn == 0 || summary[0] == '\0') {
        fprintf(stderr, "FAIL: empty summary\n");
        rc = GEIST_TEST_FAIL;
    } else if (sn >= input_len) {
        fprintf(stderr, "FAIL: summary not shorter than the input (no summarization)\n");
        rc = GEIST_TEST_FAIL;
    }
    if (rc == GEIST_TEST_PASS) {
        printf("summarize_file: multi-chunk file -> non-empty summary shorter than input\n");
    }

    remove(TMPFILE);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return rc;
}

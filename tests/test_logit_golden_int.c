/*
 * test_logit_golden_int — per-family greedy-continuation golden (#440).
 *
 * For every CI fixture that is present, prefill one fixed prompt and take
 * GOLDEN_STEPS greedy steps; before each step the argmax of the next-token
 * logits and its margin over the runner-up are compared with the committed
 * golden in tests/data/logit_golden/<model>.txt.
 *
 * What this catches that the five-item cloze cannot: a RoPE pair rotated at
 * the wrong rate (#433), a norm applied once too often, a dispatch table
 * handing back stale scratch (#434) — anything that moves a confident
 * decision. What it must NOT catch: the last-bit differences between
 * cpu_neon, cpu_x86, scalar, Accelerate and Metal. So a mismatch counts
 * only where the golden decided with a margin of at least MARGIN_MIN
 * logits; at a narrower margin the two runs simply took different but
 * equally justified branches, the contexts diverge from there, and the
 * comparison stops — reported, not failed.
 *
 *   test_logit_golden_int            compare (the runner's call)
 *   test_logit_golden_int --record   rewrite the golden files from this build
 *
 * A fixture without a golden is skipped with a note (add one with --record);
 * no fixture at all is a fixture skip (strict-aware).
 */
#define _POSIX_C_SOURCE 200809L

#include "test_helpers.h"

#include <geist.h>
#include <geist_util.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Counting: every family continues it with margins of several logits per
 * step, each step depends on the position (a RoPE fault breaks the count),
 * and no instruct model ends its turn after one token — the first prompt
 * tried, a world-knowledge cloze, had gemma emit <end_of_turn> and EOS by
 * step 2 and qwen3 on a 0.05-logit tie at step 1, guarding nothing. */
static const char *PROMPT = "Count from one to thirty: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,";

enum { GOLDEN_STEPS = 12 };
static const float MARGIN_MIN = 0.5f;

static const struct {
    const char *gguf;
    const char *golden;
} FIXTURES[] = {
        {"gguf_artifacts/gemma4-e2b-Q4_K_M.gguf", "tests/data/logit_golden/gemma4-e2b-Q4_K_M.txt"},
        {"gguf_artifacts/smollm2-360m-instruct-q8_0.gguf",
         "tests/data/logit_golden/smollm2-360m-instruct-q8_0.txt"},
        {"gguf_artifacts/qwen3-0.6b-q8_0.gguf", "tests/data/logit_golden/qwen3-0.6b-q8_0.txt"},
        {"gguf_artifacts/qwen3.5-0.8b-q8_0.gguf", "tests/data/logit_golden/qwen3.5-0.8b-q8_0.txt"},
};
enum { N_FIXTURES = sizeof FIXTURES / sizeof FIXTURES[0] };

struct step {
    geist_token_t id;
    float         margin;
};

static bool file_exists(const char *path) {
    FILE *f = fopen(path, "rb");
    if (f == nullptr)
        return false;
    fclose(f);
    return true;
}

/* argmax and its margin over the runner-up. */
static struct step top(size_t n, const float logits[static n]) {
    size_t best = 0, second = 1;
    if (logits[1] > logits[0])
        best = 1, second = 0;
    for (size_t i = 2; i < n; i++) {
        if (logits[i] > logits[best])
            second = best, best = i;
        else if (logits[i] > logits[second])
            second = i;
    }
    return (struct step) {.id = (geist_token_t) best, .margin = logits[best] - logits[second]};
}

/* Run the prompt on one fixture; fills out[] and returns the step count,
 * 0 on API error. Stops after the step whose argmax is EOS: past it the
 * argmax is noise, not a decision worth guarding. */
static int run(const char *gguf, struct step out[static GOLDEN_STEPS]) {
    struct geist_backend *be    = nullptr;
    struct geist_model   *model = nullptr;
    struct geist_session *sess  = nullptr;
    int                   steps = 0;
    if (geist_backend_create("auto", nullptr, nullptr, &be) != GEIST_OK)
        return false;
    if (geist_model_load(gguf, be, &model) != GEIST_OK)
        goto out;
    struct geist_session_opts opts = {.max_seq_len = 256, .temperature = 0.0f};
    if (geist_session_create(model, be, &opts, &sess) != GEIST_OK)
        goto out;
    if (geist_session_set_prompt(sess, PROMPT) != GEIST_OK) {
        fprintf(stderr, "  set_prompt: %s\n", geist_session_errmsg(sess));
        goto out;
    }
    const geist_token_t eos = geist_model_eos_token(model);
    for (int i = 0; i < GOLDEN_STEPS; i++) {
        size_t       n      = 0;
        const float *logits = geist_session_peek_logits(&n, sess);
        if (logits == nullptr || n < 2) {
            steps = 0;
            goto out;
        }
        out[i] = top(n, logits);
        steps  = i + 1;
        if (out[i].id == eos)
            break;
        geist_token_t t;
        if (geist_session_decode_step(sess, &t) != GEIST_OK) {
            steps = 0;
            goto out;
        }
    }
out:
    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return steps;
}

static bool write_golden(const char *path, int n, const struct step got[static n]) {
    FILE *f = fopen(path, "w");
    if (f == nullptr)
        return false;
    fprintf(f,
            "# test_logit_golden_int --record: argmax id and margin over the runner-up,\n"
            "# one greedy step per line. Regenerate only for a reviewed model change.\n");
    for (int i = 0; i < n; i++)
        fprintf(f, "%d %.4f\n", (int) got[i].id, (double) got[i].margin);
    fclose(f);
    return true;
}

static int read_golden(const char *path, struct step want[static GOLDEN_STEPS]) {
    FILE *f = fopen(path, "r");
    if (f == nullptr)
        return -1;
    char line[128];
    int  n = 0;
    while (n < GOLDEN_STEPS && fgets(line, sizeof line, f) != nullptr) {
        if (line[0] == '#')
            continue;
        int   id;
        float margin;
        if (sscanf(line, "%d %f", &id, &margin) != 2)
            break;
        want[n++] = (struct step) {.id = (geist_token_t) id, .margin = margin};
    }
    fclose(f);
    return n;
}

/* 0 match, 1 confident mismatch (fail), 2 diverged on a tie (stop, pass). */
static int compare(const char       *name,
                   int               n,
                   const struct step want[static n],
                   const struct step got[static n]) {
    for (int i = 0; i < n; i++) {
        if (want[i].id == got[i].id)
            continue;
        if (want[i].margin < MARGIN_MIN) {
            fprintf(stderr,
                    "  %s: step %d took %d over golden %d on a %.3f-logit tie — stop\n",
                    name,
                    i,
                    (int) got[i].id,
                    (int) want[i].id,
                    (double) want[i].margin);
            return 2;
        }
        fprintf(stderr,
                "  %s: step %d argmax %d, golden %d decided by %.3f logits (>= %.1f)\n",
                name,
                i,
                (int) got[i].id,
                (int) want[i].id,
                (double) want[i].margin,
                (double) MARGIN_MIN);
        return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    const bool record  = argc > 1 && strcmp(argv[1], "--record") == 0;
    int        present = 0, compared = 0, failed = 0;
    for (int f = 0; f < N_FIXTURES; f++) {
        const char *gguf = FIXTURES[f].gguf, *golden = FIXTURES[f].golden;
        if (!file_exists(gguf))
            continue;
        present++;
        struct step got[GOLDEN_STEPS];
        const int   n_got = run(gguf, got);
        if (n_got == 0) {
            fprintf(stderr, "  %s: forward failed\n", gguf);
            failed++;
            continue;
        }
        if (record) {
            if (!write_golden(golden, n_got, got)) {
                fprintf(stderr, "  cannot write %s\n", golden);
                return GEIST_TEST_ERROR;
            }
            fprintf(stderr, "  wrote %s\n", golden);
            continue;
        }
        struct step want[GOLDEN_STEPS];
        const int   n_want = read_golden(golden, want);
        if (n_want < 1) {
            fprintf(stderr, "  %s: no golden at %s — run with --record\n", gguf, golden);
            continue;
        }
        compared++;
        if (compare(gguf, n_want < n_got ? n_want : n_got, want, got) == 1)
            failed++;
    }
    if (present == 0) {
        GEIST_SKIP_FIXTURE(
                "no CI fixture present (gguf_artifacts/{gemma4-e2b,smollm2,qwen3,qwen3.5})");
    }
    if (record) {
        printf("logit golden: recorded %d fixture(s)\n", present);
        return GEIST_TEST_PASS;
    }
    printf("logit golden: %d fixture(s) present, %d compared, %d failed\n",
           present,
           compared,
           failed);
    return failed == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}

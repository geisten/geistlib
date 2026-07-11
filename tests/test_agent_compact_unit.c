/*
 * test_agent_compact_unit — the sliding-window context bound, no model.
 *
 * Builds a synthetic chat transcript past GEIST_AGENT_CTX_BUDGET, calls
 * agent_compact(), and checks the contract: the protected system prefix is kept
 * verbatim, the oldest turns are evicted, the most recent turn survives, the
 * result is back under TARGET and well-framed (a turn_close after the prefix),
 * and a small transcript / non-conversation agent is left untouched. No model is
 * loaded — agent_compact only touches the transcript string + lengths.
 */
#define _POSIX_C_SOURCE 200809L

#include "test_helpers.h"

#include "../tools/agent.h"

#include <stdio.h>
#include <string.h>

static struct geist_agent A; /* large (32 KB transcript) -> static, not stack */

static void add_turn(const char *user, const char *model) {
    A.tlen += (size_t) snprintf(A.transcript + A.tlen,
                                sizeof A.transcript - A.tlen,
                                "%s%s%s%s%s%s",
                                A.tmpl.user_open,
                                user,
                                A.tmpl.turn_close,
                                A.tmpl.model_open,
                                model,
                                A.tmpl.turn_close);
}

int main(void) {
    int fails = 0;

    A.conversation = true;
    A.tmpl         = GEIST_CHAT_GEMMA;

    /* protected system prefix (opens a user turn, like agent_system_prompt) */
    A.tlen    = (size_t) snprintf(A.transcript,
                                  sizeof A.transcript,
                                  "%sYou are a test agent. Tools: list_dir, recall.\n",
                                  A.tmpl.user_open);
    A.sys_len = A.tlen;
    char sys_copy[1024];
    memcpy(sys_copy, A.transcript, A.sys_len);

    /* turn 0 is FUSED into the system user turn (no leading user_open), matching
     * how the first request is appended — this is the OLDEST content. */
    A.tlen += (size_t) snprintf(A.transcript + A.tlen,
                                sizeof A.transcript - A.tlen,
                                "OLDEST_MARKER first question%s%sfirst reply%s",
                                A.tmpl.turn_close,
                                A.tmpl.model_open,
                                A.tmpl.turn_close);

    /* fill with whole turns (each opens with user_open) until well over budget */
    for (int i = 0; A.tlen < A.sys_len + GEIST_AGENT_CTX_CARRY + 6000; i++) {
        char u[128], m[128];
        snprintf(u, sizeof u, "filler question number %d with some padding text here", i);
        snprintf(m, sizeof m, "filler reply number %d with padding", i);
        add_turn(u, m);
    }
    /* the newest turn — must survive compaction */
    add_turn("NEWEST_MARKER last question", "last reply");

    size_t before = A.tlen;
    fails += geist_expect(before > A.sys_len + GEIST_AGENT_CTX_CARRY,
                          "setup: transcript exceeds budget");

    agent_compact(&A);

    size_t tcl_budget = strlen(A.tmpl.turn_close);
    fails += geist_expect(A.tlen < before, "compact: transcript shrank");
    fails += geist_expect(A.tlen <= A.sys_len + tcl_budget + GEIST_AGENT_CTX_CARRY,
                          "compact: down to target");
    fails += geist_expect(strlen(A.transcript) == A.tlen, "compact: NUL-terminated at tlen");
    fails += geist_expect(memcmp(A.transcript, sys_copy, A.sys_len) == 0,
                          "compact: system prefix kept verbatim");
    /* a turn_close right after the prefix -> the kept history is well-framed */
    size_t tcl = strlen(A.tmpl.turn_close);
    fails += geist_expect(memcmp(A.transcript + A.sys_len, A.tmpl.turn_close, tcl) == 0,
                          "compact: turn_close inserted after the prefix");
    fails += geist_expect(strstr(A.transcript, "NEWEST_MARKER") != nullptr,
                          "compact: most recent turn survives");
    fails += geist_expect(strstr(A.transcript, "OLDEST_MARKER") == nullptr,
                          "compact: oldest turn evicted");

    /* idempotent / no-op once under budget */
    size_t t2 = A.tlen;
    agent_compact(&A);
    fails += geist_expect(A.tlen == t2, "compact: no-op when already under budget");

    /* non-conversation agents are never compacted */
    A.conversation = false;
    A.tlen         = sizeof A.transcript - 1; /* pretend it's huge */
    agent_compact(&A);
    fails += geist_expect(A.tlen == sizeof A.transcript - 1, "compact: no-op outside conversation");

    /* regression: a user_open the user pasted into the first request, sitting in
     * the (sys_len, sys_len+tcl) gap, must NOT be chosen as the cut boundary — a
     * forward-shifting memmove there would write past the buffer. Build a
     * near-cap transcript whose only marker is one byte past sys_len and confirm
     * agent_compact leaves it untouched (no boundary past the prefix) rather than
     * corrupting tlen / overrunning transcript. */
    memset(&A, 0, sizeof A);
    A.conversation = true;
    A.tmpl         = GEIST_CHAT_GEMMA;
    A.sys_len =
            (size_t) snprintf(A.transcript, sizeof A.transcript, "%ssystem.\n", A.tmpl.user_open);
    A.transcript[A.sys_len] = 'x'; /* first request byte: NOT a marker */
    size_t mk               = A.sys_len + 1;
    size_t uol              = strlen(A.tmpl.user_open);
    memcpy(A.transcript + mk, A.tmpl.user_open, uol); /* pasted marker in the gap */
    size_t fill = mk + uol;
    while (fill < sizeof A.transcript - 1) {
        A.transcript[fill++] = 'x'; /* padding, no further markers */
    }
    A.transcript[fill] = '\0';
    A.tlen             = fill;
    size_t t3          = A.tlen;
    agent_compact(&A); /* must be a no-op, and must not overrun the buffer */
    fails += geist_expect(A.tlen == t3, "compact: skips a gap-marker, no-op (no OOB)");
    fails += geist_expect(strlen(A.transcript) == A.tlen,
                          "compact: transcript intact after gap-marker");

    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("agent_compact: keeps system + recent, evicts oldest, bounds to target\n");
    return GEIST_TEST_PASS;
}

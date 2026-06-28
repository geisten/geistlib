/*
 * test_agent_event_unit — the agent progress-event type, no model.
 *
 * Covers the three pure pieces of the new output type: the phase-name table,
 * agent_emit forwarding to a host callback (and no-op when unset), and the
 * ready-made agent_event_print formatter (glyph + phase + tool + truncated
 * detail). No assert() — checks set a flag, the exit code carries PASS/FAIL.
 */
#define _POSIX_C_SOURCE 200809L

#include "test_helpers.h"

#include "../tools/agent.h"

#include <stdio.h>
#include <string.h>

static int fails = 0;

/* capture the last event a callback received */
static struct geist_agent_event captured;
static int                      capture_calls = 0;
static void                     capture(void *ctx, const struct geist_agent_event *ev) {
    (void) ctx;
    captured = *ev; /* shallow copy; the pointers stay valid for the test's scope */
    capture_calls++;
}

static void test_phase_name(void) {
    fails += geist_expect(strcmp(geist_agent_phase_name(GEIST_AGENT_ROUTING), "routing") == 0,
                          "phase: routing");
    fails += geist_expect(strcmp(geist_agent_phase_name(GEIST_AGENT_CALLING), "calling") == 0,
                          "phase: calling");
    fails += geist_expect(strcmp(geist_agent_phase_name(GEIST_AGENT_RUNNING), "running") == 0,
                          "phase: running");
    fails += geist_expect(strcmp(geist_agent_phase_name(GEIST_AGENT_OBSERVED), "observed") == 0,
                          "phase: observed");
    fails += geist_expect(strcmp(geist_agent_phase_name(GEIST_AGENT_ANSWERING), "answering") == 0,
                          "phase: answering");
}

static void test_emit(void) {
    static struct geist_agent a; /* no model needed — agent_emit only reads on_event */
    a.on_event    = nullptr;
    capture_calls = 0;
    agent_emit(&a, GEIST_AGENT_ROUTING, 0, nullptr, nullptr); /* unset -> no-op, no crash */
    fails += geist_expect(capture_calls == 0, "emit: nullptr callback is a no-op");

    a.on_event     = capture;
    a.on_event_ctx = nullptr;
    agent_emit(&a, GEIST_AGENT_RUNNING, 2, "web_search", "the query");
    fails += geist_expect(capture_calls == 1, "emit: callback fired once");
    fails += geist_expect(captured.phase == GEIST_AGENT_RUNNING && captured.step == 2 &&
                                  strcmp(captured.tool, "web_search") == 0 &&
                                  strcmp(captured.detail, "the query") == 0,
                          "emit: event fields forwarded verbatim");
}

/* render one event with agent_event_print into a buffer via a tmpfile */
static void render(const struct geist_agent_event *ev, char *out, size_t cap) {
    FILE *f = tmpfile();
    if (!f) {
        out[0] = '\0';
        return;
    }
    agent_event_print(f, ev);
    rewind(f);
    size_t n = fread(out, 1, cap - 1, f);
    out[n]   = '\0';
    fclose(f);
}

static void test_print(void) {
    char buf[256];

    struct geist_agent_event call = {.phase  = GEIST_AGENT_CALLING,
                                     .step   = 0,
                                     .tool   = "web_search",
                                     .detail = "{\"query\":\"x\"}"};
    render(&call, buf, sizeof buf);
    fails += geist_expect(strstr(buf, "calling") && strstr(buf, "web_search") &&
                                  strstr(buf, "{\"query\":\"x\"}"),
                          "print: phase + tool + detail rendered");

    /* a long detail is truncated with an ellipsis */
    char longdetail[200];
    memset(longdetail, 'a', sizeof longdetail - 1);
    longdetail[sizeof longdetail - 1] = '\0';
    struct geist_agent_event obs      = {
            .phase = GEIST_AGENT_OBSERVED, .step = 1, .tool = "list_dir", .detail = longdetail};
    render(&obs, buf, sizeof buf);
    fails += geist_expect(strstr(buf, "\xe2\x80\xa6") != nullptr,
                          "print: long detail truncated (…)");

    /* tool/detail nullptr -> just the phase line, no crash */
    struct geist_agent_event bare = {
            .phase = GEIST_AGENT_ROUTING, .step = 0, .tool = nullptr, .detail = nullptr};
    render(&bare, buf, sizeof buf);
    fails += geist_expect(strstr(buf, "routing") != nullptr, "print: nullptr tool/detail is fine");
}

int main(void) {
    test_phase_name();
    test_emit();
    test_print();
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("agent event: phase names + emit forwarding + printer pass\n");
    return GEIST_TEST_PASS;
}

#define _GNU_SOURCE /* recv() MSG_DONTWAIT on the musl CI leg (CONTRIBUTING.md) */
/* test_dynamic_stream_unit — dynamic-tools-v1 §Streaming: the strict `stream`
 * request field and the conversation.delta writer (UTF-8 carry, JSON escaping,
 * write-failure latch). No model needed. */
#include "../tools/agent_main.h"

#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>

static int fails = 0;

static int geist_expect(int condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        return 1;
    }
    return 0;
}

/* Minimal valid request skeleton shared by the parse cases. */
#define REQ_HEAD                                                                    \
    "{\"input\":\"Add 5 and 7\",\"tools\":[{\"name\":\"Add\",\"description\":\"add" \
    " two integers\",\"parameters\":{\"type\":\"object\",\"properties\":{\"a\":"    \
    "{\"type\":\"integer\"},\"b\":{\"type\":\"integer\"}},\"required\":[\"a\","     \
    "\"b\"]}}]"

static void test_stream_field(void) {
    struct geist_dynamic_request request;
    enum geist_dynamic_status    ts;

    static const char absent[] = REQ_HEAD "}";
    fails += geist_expect(geist_dynamic_request_parse(absent, sizeof absent - 1u, &request, &ts) ==
                                          GEIST_DYNAMIC_REQUEST_OK &&
                                  !request.stream,
                          "stream: absent -> false (v0.4.0 requests unchanged)");

    static const char on[] = REQ_HEAD ",\"stream\":true}";
    fails += geist_expect(geist_dynamic_request_parse(on, sizeof on - 1u, &request, &ts) ==
                                          GEIST_DYNAMIC_REQUEST_OK &&
                                  request.stream,
                          "stream: true -> opt-in");

    static const char off[] = REQ_HEAD ",\"stream\":false}";
    fails += geist_expect(geist_dynamic_request_parse(off, sizeof off - 1u, &request, &ts) ==
                                          GEIST_DYNAMIC_REQUEST_OK &&
                                  !request.stream,
                          "stream: explicit false -> off");

    static const char bad[] = REQ_HEAD ",\"stream\":\"yes\"}";
    fails += geist_expect(geist_dynamic_request_parse(bad, sizeof bad - 1u, &request, &ts) ==
                                  GEIST_DYNAMIC_REQUEST_E_INVALID_REQUEST,
                          "stream: non-boolean rejected (strict boundary)");
}

/* Read whatever frames are pending on fd into out (bounded, non-blocking-ish:
 * the writer already wrote before we read). */
static size_t drain(int fd, size_t cap, char out[]) {
    size_t  n = 0;
    ssize_t r;
    while (n + 1 < cap && (r = recv(fd, out + n, cap - 1 - n, MSG_DONTWAIT)) > 0) {
        n += (size_t) r;
    }
    out[n] = '\0';
    return n;
}

static void test_delta_writer(void) {
    int sv[2];
    fails += geist_expect(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0, "delta: socketpair");

    struct agent_main_delta_writer dw = {.fd = sv[0]};
    char                           frames[4096];

    /* alnum gate: text below 2 alphanumeric chars is held, not sent (a turn the
     * agent would discard as degenerate never reaches the client). ". " has 0
     * alnum -> zero frames so far. */
    fails += geist_expect(agent_main_delta_write(&dw, 2, ". "), "delta: sub-threshold accepted");
    fails += geist_expect(drain(sv[1], sizeof frames, frames) == 0 && !dw.live,
                          "delta: <2 alnum held back, nothing streamed yet");

    /* Crossing 2 alnum opens the gate and flushes everything held so far. */
    fails += geist_expect(agent_main_delta_write(&dw, 2, "Hi"), "delta: threshold piece accepted");
    drain(sv[1], sizeof frames, frames);
    fails += geist_expect(dw.live && strstr(frames, "\"text\":\". Hi\"") != nullptr,
                          "delta: gate opens, buffered prefix + trigger flushed as one frame");

    /* Now live: UTF-8 carry — "é" arrives as [C3] then [A9]; the split lead is
     * carried, no frame, then completes into one valid é frame. */
    fails += geist_expect(agent_main_delta_write(&dw, 1, "\xc3"), "delta: split lead accepted");
    fails += geist_expect(drain(sv[1], sizeof frames, frames) == 0,
                          "delta: incomplete UTF-8 lead carried, no frame");
    fails += geist_expect(agent_main_delta_write(&dw, 1, "\xa9"), "delta: continuation accepted");
    drain(sv[1], sizeof frames, frames);
    fails += geist_expect(strstr(frames, "\"text\":\"\xc3\xa9\"") != nullptr,
                          "delta: carried bytes complete into valid UTF-8");

    /* Malformed bytes are dropped, never emitted raw (valid-UTF-8-per-frame). A
     * lone continuation 0xA9 before 'X' is dropped; only 'X' is framed. */
    fails += geist_expect(agent_main_delta_write(&dw, 2, "\xa9X"),
                          "delta: lone-cont piece accepted");
    drain(sv[1], sizeof frames, frames);
    fails += geist_expect(strstr(frames, "\"text\":\"X\"") != nullptr &&
                                  strstr(frames, "\xa9") == nullptr,
                          "delta: lone continuation byte dropped, not emitted");

    /* JSON escaping inside a delta frame. */
    fails += geist_expect(agent_main_delta_write(&dw, 4, "a\"b\n"), "delta: escape piece accepted");
    drain(sv[1], sizeof frames, frames);
    fails += geist_expect(strstr(frames, "\"text\":\"a\\\"b\\n\"") != nullptr,
                          "delta: quote and newline escaped");
    fails += geist_expect(strstr(frames, "{\"type\":\"conversation.delta\"") != nullptr &&
                                  frames[strlen(frames) - 1] == '\n',
                          "delta: framed as newline-terminated conversation.delta");

    /* Client gone: write fails, writer latches, decode is told to stop. */
    close(sv[1]);
    fails += geist_expect(!agent_main_delta_write(&dw, 5, "later") && dw.failed,
                          "delta: write failure latches failed and returns false");
    fails += geist_expect(!agent_main_delta_write(&dw, 5, "again"),
                          "delta: latched writer keeps refusing");
    close(sv[0]);

    /* A degenerate answer (never 2 alnum) streams nothing at all. */
    int sv2[2];
    fails += geist_expect(socketpair(AF_UNIX, SOCK_STREAM, 0, sv2) == 0, "delta: socketpair 2");
    struct agent_main_delta_writer dg = {.fd = sv2[0]};
    (void) agent_main_delta_write(&dg, 3, "..!");
    (void) agent_main_delta_write(&dg, 1, "A"); /* 1 alnum total — still degenerate */
    char none[256];
    fails += geist_expect(drain(sv2[1], sizeof none, none) == 0 && !dg.live,
                          "delta: degenerate answer (<2 alnum) streams zero frames");
    close(sv2[0]);
    close(sv2[1]);
}

int main(void) {
    signal(SIGPIPE, SIG_IGN); /* mirror agent_main_serve — the writer's home */
    test_stream_field();
    test_delta_writer();
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return 1;
    }
    printf("dynamic stream: strict stream field + UTF-8/escape-safe delta writer pass\n");
    return 0;
}

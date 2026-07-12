#include "test_helpers.h"

#include "tools/home_daemon_v2.h"

#include <sys/socket.h>

struct runner_state {
    unsigned calls;
};

static enum home_executor_status runner(void                                *context,
                                        int                                  connection,
                                        const struct home_daemon_v2_request *request,
                                        size_t                               response_cap,
                                        char    response[static response_cap],
                                        size_t *response_len) {
    struct runner_state *state = (struct runner_state *) context;
    state->calls++;
    if (strcmp(request->utterance, "Licht \"Flur\" an\nbitte") != 0 ||
        strcmp(request->locale, "de-DE") != 0 || request->registry_version != 7u ||
        request->max_tool_calls != 2u) {
        return HOME_EXECUTOR_INVALID_REQUEST;
    }
    struct home_protocol_executor protocol = {
            .roundtrip         = home_daemon_v2_roundtrip,
            .transport_context = &connection,
            .next_request_id   = 1u,
    };
    struct home_executor               executor = home_protocol_executor_adapter(&protocol);
    const struct home_executor_request tool     = {
            .operation        = HOME_EXECUTOR_CALL_SERVICE,
            .entity_id        = "light.flur",
            .domain           = "light",
            .service          = "turn_on",
            .arguments        = "",
            .registry_version = request->registry_version,
    };
    char   raw[128];
    size_t raw_len = 0u;
    if (home_executor_run(&executor, &tool, sizeof raw, raw, &raw_len) != HOME_EXECUTOR_OK ||
        strcmp(raw, "[]") != 0) {
        return HOME_EXECUTOR_UNAVAILABLE;
    }
    *response_len = (size_t) snprintf(response, response_cap, "Licht ist an.");
    return HOME_EXECUTOR_OK;
}

static size_t encode(const char *json, uint8_t *wire, size_t cap) {
    size_t len = 0u;
    if (geist_home_v2_encode((const uint8_t *) json, strlen(json), wire, cap, &len) !=
        GEIST_HOME_V2_OK) {
        return 0u;
    }
    return len;
}

int main(void) {
    int           fails         = 0;
    const uint8_t v2_prefix[]   = {0u, 0u, 0u, 80u};
    const uint8_t v1_prefix[]   = {'H', 'a', 'l', 'l'};
    const uint8_t zero_prefix[] = {0u, 0u, 0u, 0u};
    fails += geist_expect(home_daemon_v2_is_prefix(v2_prefix), "v2 prefix detected");
    fails += geist_expect(!home_daemon_v2_is_prefix(v1_prefix), "v1 UTF-8 prefix stays v1");
    fails += geist_expect(!home_daemon_v2_is_prefix(zero_prefix), "empty frame rejected");
    int pair[2];
    fails += geist_expect(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0, "socketpair");
    if (fails != 0) {
        return GEIST_TEST_ERROR;
    }

    uint8_t start[1024], tool_result[512];
    size_t  start_len =
            encode("{\"version\":2,\"request_id\":\"conv-1\",\"type\":\"conversation.start\","
                   "\"utterance\":\"Licht \\\"Flur\\\" an\\nbitte\",\"locale\":\"de-DE\","
                   "\"registry_version\":7,\"max_tool_calls\":2}",
                   start,
                   sizeof start);
    size_t tool_result_len =
            encode("{\"version\":2,\"request_id\":\"exec-1\",\"type\":\"tool.result\","
                   "\"status\":\"ok\",\"result\":[]}",
                   tool_result,
                   sizeof tool_result);
    fails += geist_expect(start_len > 0u && tool_result_len > 0u, "fixtures encode");
    fails += geist_expect(
            home_daemon_v2_write_full(pair[0], start, start_len) == 0 &&
                    home_daemon_v2_write_full(pair[0], tool_result, tool_result_len) == 0,
            "client preloads start + tool result");

    struct runner_state state = {0};
    fails += geist_expect(home_daemon_v2_connection(pair[1], runner, &state) == 0,
                          "daemon connection succeeds");
    fails += geist_expect(state.calls == 1u, "runner called once");

    uint8_t first[1024], second[1024];
    size_t  first_len = 0u, second_len = 0u;
    fails += geist_expect(
            home_daemon_v2_read_wire(pair[0], sizeof first, first, &first_len) ==
                            GEIST_HOME_V2_OK &&
                    home_daemon_v2_read_wire(pair[0], sizeof second, second, &second_len) ==
                            GEIST_HOME_V2_OK,
            "client receives tool call + final result");
    struct geist_home_v2_frame frame    = {0};
    size_t                     consumed = 0u;
    fails += geist_expect(geist_home_v2_decode(first, first_len, &frame, &consumed) ==
                                          GEIST_HOME_V2_OK &&
                                  frame.type == GEIST_HOME_V2_TYPE_TOOL_CALL &&
                                  strcmp(frame.request_id, "exec-1") == 0,
                          "first response is correlated tool.call");
    struct geist_home_v2_json_span registry = {0};
    fails += geist_expect(geist_home_v2_top_field(
                                  frame.json, frame.json_len, "registry_version", &registry) == 1 &&
                                  registry.len == 1u && registry.p[0] == '7',
                          "tool.call carries registry version");
    fails += geist_expect(geist_home_v2_decode(second, second_len, &frame, &consumed) ==
                                          GEIST_HOME_V2_OK &&
                                  frame.type == GEIST_HOME_V2_TYPE_CONVERSATION_RESULT &&
                                  strcmp(frame.request_id, "conv-1") == 0,
                          "second response is correlated conversation.result");
    struct geist_home_v2_json_span text = {0};
    char                           decoded[64];
    fails += geist_expect(geist_home_v2_top_field(frame.json, frame.json_len, "text", &text) == 1 &&
                                  home_daemon_v2_string(&text, sizeof decoded, decoded) > 0 &&
                                  strcmp(decoded, "Licht ist an.") == 0,
                          "final text roundtrips");

    close(pair[0]);
    close(pair[1]);
    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}

#include "test_helpers.h"

#include "tools/home_protocol_executor.h"

struct memory_transport {
    unsigned                            calls;
    enum home_protocol_transport_status transport_status;
    const char                         *result_status;
    const char                         *result_json;
    bool                                mismatch_id;
    bool                                duplicate_status;
    bool                                wrong_type;
    char                                last_request[HOME_PROTOCOL_EXECUTOR_JSON_CAP];
};

static enum home_protocol_transport_status memory_roundtrip(void          *context,
                                                            const uint8_t *request,
                                                            size_t         request_len,
                                                            uint8_t       *response,
                                                            size_t         response_cap,
                                                            size_t        *response_len) {
    struct memory_transport *transport = (struct memory_transport *) context;
    transport->calls++;
    *response_len = 0u;
    if (transport->transport_status != HOME_PROTOCOL_TRANSPORT_OK) {
        return transport->transport_status;
    }

    struct geist_home_v2_frame frame    = {0};
    size_t                     consumed = 0u;
    if (geist_home_v2_decode(request, request_len, &frame, &consumed) != GEIST_HOME_V2_OK ||
        consumed != request_len || frame.type != GEIST_HOME_V2_TYPE_TOOL_CALL ||
        frame.json_len >= sizeof transport->last_request) {
        return HOME_PROTOCOL_TRANSPORT_OVERFLOW;
    }
    memcpy(transport->last_request, frame.json, frame.json_len);
    transport->last_request[frame.json_len] = '\0';

    const char *type   = transport->wrong_type ? "health" : "tool.result";
    const char *id     = transport->mismatch_id ? "other-request" : frame.request_id;
    const char *status = transport->result_status != NULL ? transport->result_status : "ok";
    const char *result = transport->result_json != NULL ? transport->result_json : "{}";
    char        json[HOME_PROTOCOL_EXECUTOR_JSON_CAP];
    int json_len = transport->duplicate_status
                           ? snprintf(json,
                                      sizeof json,
                                      "{\"version\":2,\"request_id\":\"%s\",\"type\":\"%s\","
                                      "\"status\":\"%s\",\"status\":\"ok\",\"result\":%s}",
                                      id,
                                      type,
                                      status,
                                      result)
                           : snprintf(json,
                                      sizeof json,
                                      "{\"version\":2,\"request_id\":\"%s\",\"type\":\"%s\","
                                      "\"status\":\"%s\",\"result\":%s}",
                                      id,
                                      type,
                                      status,
                                      result);
    if (json_len <= 0 || (size_t) json_len >= sizeof json ||
        geist_home_v2_encode(
                (const uint8_t *) json, (size_t) json_len, response, response_cap, response_len) !=
                GEIST_HOME_V2_OK) {
        return HOME_PROTOCOL_TRANSPORT_OVERFLOW;
    }
    return HOME_PROTOCOL_TRANSPORT_OK;
}

int main(void) {
    int                           fails     = 0;
    struct memory_transport       transport = {.result_json = "{\"state\":\"off\"}"};
    struct home_protocol_executor protocol  = {
            .roundtrip         = memory_roundtrip,
            .transport_context = &transport,
            .next_request_id   = 7u,
    };
    struct home_executor executor = home_protocol_executor_adapter(&protocol);
    char                 out[256];
    size_t               out_len = 0u;

    const struct home_executor_request get = {
            .operation = HOME_EXECUTOR_GET_STATE,
            .entity_id = "light.flur",
            .domain    = "light",
    };
    enum home_executor_status status =
            home_executor_run(&executor, &get, sizeof out, out, &out_len);
    fails += geist_expect(status == HOME_EXECUTOR_OK, "in-memory state roundtrip succeeds");
    fails += geist_expect(strcmp(out, "{\"state\":\"off\"}") == 0 && out_len == strlen(out),
                          "tool.result JSON reaches planner");
    fails += geist_expect(
            strstr(transport.last_request, "\"request_id\":\"exec-7\"") != NULL &&
                    strstr(transport.last_request, "\"operation\":\"get_state\"") != NULL &&
                    strstr(transport.last_request, "\"entity_id\":\"light.flur\"") != NULL,
            "get_state tool.call schema");

    transport.result_json                  = "[]";
    const struct home_executor_request set = {
            .operation = HOME_EXECUTOR_CALL_SERVICE,
            .entity_id = "climate.bad",
            .domain    = "climate",
            .service   = "set_temperature",
            .arguments = "\"temperature\":22",
    };
    status = home_executor_run(&executor, &set, sizeof out, out, &out_len);
    fails += geist_expect(status == HOME_EXECUTOR_OK && strcmp(out, "[]") == 0,
                          "in-memory service roundtrip succeeds");
    fails += geist_expect(
            strstr(transport.last_request, "\"request_id\":\"exec-8\"") != NULL &&
                    strstr(transport.last_request, "\"operation\":\"call_service\"") != NULL &&
                    strstr(transport.last_request, "\"arguments\":{\"temperature\":22}") != NULL,
            "call_service tool.call schema");

    transport.result_status = "denied";
    status                  = home_executor_run(&executor, &set, sizeof out, out, &out_len);
    fails += geist_expect(status == HOME_EXECUTOR_DENIED, "remote policy denial preserved");
    fails += geist_expect(out_len == 0u && out[0] == '\0', "denied result body discarded");

    transport.result_status = "unknown-status";
    status                  = home_executor_run(&executor, &set, sizeof out, out, &out_len);
    fails += geist_expect(status == HOME_EXECUTOR_INVALID_REQUEST,
                          "unknown remote status fails closed");

    transport.result_status = "ok";
    transport.mismatch_id   = true;
    status                  = home_executor_run(&executor, &get, sizeof out, out, &out_len);
    fails += geist_expect(status == HOME_EXECUTOR_INVALID_REQUEST,
                          "mismatched response request id rejected");
    transport.mismatch_id = false;

    transport.wrong_type = true;
    status               = home_executor_run(&executor, &get, sizeof out, out, &out_len);
    fails += geist_expect(status == HOME_EXECUTOR_INVALID_REQUEST,
                          "non-tool-result response rejected");
    transport.wrong_type = false;

    transport.duplicate_status = true;
    status                     = home_executor_run(&executor, &get, sizeof out, out, &out_len);
    fails += geist_expect(status == HOME_EXECUTOR_INVALID_REQUEST,
                          "duplicate result status rejected");
    transport.duplicate_status = false;

    transport.result_json = "\"not-an-object\"";
    status                = home_executor_run(&executor, &get, sizeof out, out, &out_len);
    fails += geist_expect(status == HOME_EXECUTOR_INVALID_REQUEST, "scalar tool result rejected");
    transport.result_json = "{\"state\":\"off\"}";

    transport.transport_status = HOME_PROTOCOL_TRANSPORT_UNAVAILABLE;
    status                     = home_executor_run(&executor, &get, sizeof out, out, &out_len);
    fails += geist_expect(status == HOME_EXECUTOR_UNAVAILABLE,
                          "transport outage maps to unavailable");
    transport.transport_status = HOME_PROTOCOL_TRANSPORT_OK;

    const unsigned               calls_before_injection = transport.calls;
    struct home_executor_request injected               = set;
    injected.arguments = "\"temperature\":22},\"service\":\"unlock\",\"x\":{";
    status             = home_executor_run(&executor, &injected, sizeof out, out, &out_len);
    fails += geist_expect(status == HOME_EXECUTOR_INVALID_REQUEST,
                          "argument object injection rejected");
    fails += geist_expect(transport.calls == calls_before_injection,
                          "invalid JSON never reaches transport");

    transport.result_json                = "{\"state\":\"on\"}";
    struct home_executor_request escaped = get;
    escaped.entity_id                    = "light.quo\"te";
    status = home_executor_run(&executor, &escaped, sizeof out, out, &out_len);
    fails += geist_expect(
            status == HOME_EXECUTOR_OK &&
                    strstr(transport.last_request, "\"entity_id\":\"light.quo\\\"te\"") != NULL,
            "entity identifiers are JSON escaped");

    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}

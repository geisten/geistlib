/* tools/home_protocol_executor.h — adapt typed home executor requests to
 * protocol-v2 tool.call/tool.result round trips. Transport remains injected:
 * a later slice supplies Unix-socket/private-TCP I/O; unit tests use memory.
 */
#ifndef GEIST_HOME_PROTOCOL_EXECUTOR_H
#define GEIST_HOME_PROTOCOL_EXECUTOR_H

#include "home_executor.h"
#include "home_protocol_v2.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { HOME_PROTOCOL_EXECUTOR_JSON_CAP = 4096, HOME_PROTOCOL_EXECUTOR_WIRE_CAP = 8192 };

enum home_protocol_transport_status {
    HOME_PROTOCOL_TRANSPORT_OK = 0,
    HOME_PROTOCOL_TRANSPORT_UNAVAILABLE,
    HOME_PROTOCOL_TRANSPORT_OVERFLOW,
};

typedef enum home_protocol_transport_status (*home_protocol_roundtrip_fn)(void          *context,
                                                                          const uint8_t *request,
                                                                          size_t   request_len,
                                                                          uint8_t *response,
                                                                          size_t   response_cap,
                                                                          size_t  *response_len);

struct home_protocol_executor {
    home_protocol_roundtrip_fn roundtrip;
    void                      *transport_context;
    uint64_t                   next_request_id;
};

static inline size_t home_protocol_json_escape(const char *src, size_t cap, char out[static cap]) {
    if (src == NULL || out == NULL || cap == 0u) {
        return 0u;
    }
    static const char hex[] = "0123456789abcdef";
    size_t            w     = 0u;
    for (const unsigned char *p = (const unsigned char *) src; *p != '\0'; p++) {
        const unsigned char c = *p;
        if (c == '"' || c == '\\') {
            if (w + 2u >= cap) {
                return 0u;
            }
            out[w++] = '\\';
            out[w++] = (char) c;
        } else if (c < 0x20u) {
            if (w + 6u >= cap) {
                return 0u;
            }
            out[w++] = '\\';
            out[w++] = 'u';
            out[w++] = '0';
            out[w++] = '0';
            out[w++] = hex[c >> 4u];
            out[w++] = hex[c & 0x0fu];
        } else {
            if (w + 1u >= cap) {
                return 0u;
            }
            out[w++] = (char) c;
        }
    }
    out[w] = '\0';
    return w;
}

static inline enum home_executor_status
home_protocol_result_status(const struct geist_home_v2_json_span *span) {
    if (geist_home_v2_span_string_is(span, "ok")) {
        return HOME_EXECUTOR_OK;
    }
    if (geist_home_v2_span_string_is(span, "unconfigured")) {
        return HOME_EXECUTOR_UNCONFIGURED;
    }
    if (geist_home_v2_span_string_is(span, "unavailable")) {
        return HOME_EXECUTOR_UNAVAILABLE;
    }
    if (geist_home_v2_span_string_is(span, "denied")) {
        return HOME_EXECUTOR_DENIED;
    }
    return HOME_EXECUTOR_INVALID_REQUEST;
}

static inline bool home_protocol_arguments_valid(const char *arguments) {
    if (arguments == NULL) {
        return false;
    }
    char object[HOME_PROTOCOL_EXECUTOR_JSON_CAP];
    int  len = snprintf(object, sizeof object, "{%s}", arguments);
    if (len <= 0 || (size_t) len >= sizeof object ||
        !geist_home_v2_utf8_valid((const uint8_t *) object, (size_t) len)) {
        return false;
    }
    struct geist_home_v2_json_cursor cursor = {
            .p   = (const uint8_t *) object,
            .end = (const uint8_t *) object + (size_t) len,
    };
    if (!geist_home_v2_json_value(&cursor, 0u)) {
        return false;
    }
    geist_home_v2_skip_space(&cursor);
    return cursor.p == cursor.end;
}

static inline enum home_executor_status
home_protocol_execute(void                               *context,
                      const struct home_executor_request *request,
                      size_t                              out_cap,
                      char                                out[static out_cap],
                      size_t                             *out_len) {
    struct home_protocol_executor *executor = (struct home_protocol_executor *) context;
    if (executor == NULL || executor->roundtrip == NULL || out == NULL || out_len == NULL ||
        out_cap == 0u || !home_executor_request_valid(request)) {
        return HOME_EXECUTOR_INVALID_REQUEST;
    }
    if (request->operation == HOME_EXECUTOR_CALL_SERVICE &&
        !home_protocol_arguments_valid(request->arguments)) {
        return HOME_EXECUTOR_INVALID_REQUEST;
    }

    char entity[256], domain[128], service[128];
    if (home_protocol_json_escape(request->entity_id, sizeof entity, entity) == 0u ||
        home_protocol_json_escape(request->domain, sizeof domain, domain) == 0u) {
        return HOME_EXECUTOR_INVALID_REQUEST;
    }
    if (request->operation == HOME_EXECUTOR_CALL_SERVICE &&
        home_protocol_json_escape(request->service, sizeof service, service) == 0u) {
        return HOME_EXECUTOR_INVALID_REQUEST;
    }

    char request_id[GEIST_HOME_V2_MAX_REQUEST_ID + 1u];
    int  id_len = snprintf(request_id,
                           sizeof request_id,
                           "exec-%llu",
                           (unsigned long long) executor->next_request_id++);
    if (id_len <= 0 || (size_t) id_len >= sizeof request_id) {
        return HOME_EXECUTOR_INVALID_REQUEST;
    }

    char json[HOME_PROTOCOL_EXECUTOR_JSON_CAP];
    int  json_len;
    if (request->operation == HOME_EXECUTOR_GET_STATE) {
        json_len = snprintf(json,
                            sizeof json,
                            "{\"version\":2,\"request_id\":\"%s\",\"type\":\"tool.call\","
                            "\"operation\":\"get_state\",\"entity_id\":\"%s\",\"domain\":\"%s\","
                            "\"registry_version\":%llu}",
                            request_id,
                            entity,
                            domain,
                            (unsigned long long) request->registry_version);
    } else {
        json_len = snprintf(json,
                            sizeof json,
                            "{\"version\":2,\"request_id\":\"%s\",\"type\":\"tool.call\","
                            "\"operation\":\"call_service\",\"entity_id\":\"%s\","
                            "\"domain\":\"%s\",\"service\":\"%s\",\"arguments\":{%s},"
                            "\"registry_version\":%llu}",
                            request_id,
                            entity,
                            domain,
                            service,
                            request->arguments,
                            (unsigned long long) request->registry_version);
    }
    if (json_len <= 0 || (size_t) json_len >= sizeof json) {
        return HOME_EXECUTOR_INVALID_REQUEST;
    }

    uint8_t request_wire[HOME_PROTOCOL_EXECUTOR_WIRE_CAP];
    size_t  request_wire_len = 0u;
    if (geist_home_v2_encode((const uint8_t *) json,
                             (size_t) json_len,
                             request_wire,
                             sizeof request_wire,
                             &request_wire_len) != GEIST_HOME_V2_OK) {
        return HOME_EXECUTOR_INVALID_REQUEST;
    }

    uint8_t                             response_wire[HOME_PROTOCOL_EXECUTOR_WIRE_CAP];
    size_t                              response_wire_len = 0u;
    enum home_protocol_transport_status transport = executor->roundtrip(executor->transport_context,
                                                                        request_wire,
                                                                        request_wire_len,
                                                                        response_wire,
                                                                        sizeof response_wire,
                                                                        &response_wire_len);
    if (transport != HOME_PROTOCOL_TRANSPORT_OK) {
        return transport == HOME_PROTOCOL_TRANSPORT_UNAVAILABLE ? HOME_EXECUTOR_UNAVAILABLE
                                                                : HOME_EXECUTOR_INVALID_REQUEST;
    }

    struct geist_home_v2_frame response = {0};
    size_t                     consumed = 0u;
    if (geist_home_v2_decode(response_wire, response_wire_len, &response, &consumed) !=
                GEIST_HOME_V2_OK ||
        consumed != response_wire_len || response.type != GEIST_HOME_V2_TYPE_TOOL_RESULT ||
        strcmp(response.request_id, request_id) != 0) {
        return HOME_EXECUTOR_INVALID_REQUEST;
    }

    struct geist_home_v2_json_span status_span = {0};
    if (geist_home_v2_top_field(response.json, response.json_len, "status", &status_span) != 1) {
        return HOME_EXECUTOR_INVALID_REQUEST;
    }
    enum home_executor_status result_status = home_protocol_result_status(&status_span);
    if (result_status != HOME_EXECUTOR_OK) {
        return result_status;
    }

    struct geist_home_v2_json_span result_span = {0};
    if (geist_home_v2_top_field(response.json, response.json_len, "result", &result_span) != 1 ||
        result_span.len == 0u || (result_span.p[0] != '{' && result_span.p[0] != '[') ||
        result_span.len >= out_cap) {
        return HOME_EXECUTOR_INVALID_REQUEST;
    }
    memcpy(out, result_span.p, result_span.len);
    out[result_span.len] = '\0';
    *out_len             = result_span.len;
    return HOME_EXECUTOR_OK;
}

static inline struct home_executor
home_protocol_executor_adapter(struct home_protocol_executor *executor) {
    return (struct home_executor) {.execute = home_protocol_execute, .context = executor};
}

#endif /* GEIST_HOME_PROTOCOL_EXECUTOR_H */

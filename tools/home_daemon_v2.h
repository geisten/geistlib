/* tools/home_daemon_v2.h — POSIX connection-side protocol-v2 session shell.
 * It parses conversation.start, lets an injected runner perform synchronous
 * tool.call/tool.result round trips on the same fd, then emits the correlated
 * conversation.result. Model/agent binding is deliberately a separate layer.
 */
#ifndef GEIST_HOME_DAEMON_V2_H
#define GEIST_HOME_DAEMON_V2_H

#include "home_protocol_executor.h"
#include "home_protocol_v2.h"

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

enum {
    HOME_DAEMON_V2_TEXT_CAP    = 4096,
    HOME_DAEMON_V2_ESCAPED_CAP = HOME_DAEMON_V2_TEXT_CAP * 6 + 1,
    HOME_DAEMON_V2_JSON_CAP    = HOME_DAEMON_V2_ESCAPED_CAP + 256,
    HOME_DAEMON_V2_WIRE_CAP    = HOME_DAEMON_V2_JSON_CAP + GEIST_HOME_V2_PREFIX_BYTES,
};

struct home_daemon_v2_request {
    char     request_id[GEIST_HOME_V2_MAX_REQUEST_ID + 1u];
    char     utterance[HOME_DAEMON_V2_TEXT_CAP + 1u];
    char     locale[33];
    uint64_t registry_version;
    unsigned max_tool_calls;
};

typedef enum home_executor_status (*home_daemon_v2_runner_fn)(
        void                                *context,
        int                                  connection,
        const struct home_daemon_v2_request *request,
        size_t                               response_cap,
        char                                 response[static response_cap],
        size_t                              *response_len);

/* The first four bytes of every permitted v2 frame are a plausible network-
 * order payload length. UTF-8 v1 requests cannot satisfy this under the 1 MiB
 * cap, so the resident daemon can select v1/v2 without consuming ambiguity. */
static inline int home_daemon_v2_is_prefix(const uint8_t prefix[static 4]) {
    const uint32_t payload_len = ((uint32_t) prefix[0] << 24u) | ((uint32_t) prefix[1] << 16u) |
                                 ((uint32_t) prefix[2] << 8u) | (uint32_t) prefix[3];
    return payload_len > 0u && payload_len <= GEIST_HOME_V2_MAX_PAYLOAD;
}

static inline int home_daemon_v2_read_full(int fd, void *buffer, size_t len) {
    uint8_t *bytes = (uint8_t *) buffer;
    size_t   off   = 0u;
    while (off < len) {
        ssize_t got = read(fd, bytes + off, len - off);
        if (got > 0) {
            off += (size_t) got;
        } else if (got == 0) {
            return -1;
        } else if (errno != EINTR) {
            return -1;
        }
    }
    return 0;
}

static inline int home_daemon_v2_write_full(int fd, const void *buffer, size_t len) {
    const uint8_t *bytes = (const uint8_t *) buffer;
    size_t         off   = 0u;
    while (off < len) {
        ssize_t put = write(fd, bytes + off, len - off);
        if (put > 0) {
            off += (size_t) put;
        } else if (put < 0 && errno == EINTR) {
            continue;
        } else {
            return -1;
        }
    }
    return 0;
}

static inline enum geist_home_v2_status
home_daemon_v2_read_wire(int fd, size_t cap, uint8_t wire[static cap], size_t *wire_len) {
    *wire_len = 0u;
    if (cap < GEIST_HOME_V2_PREFIX_BYTES ||
        home_daemon_v2_read_full(fd, wire, GEIST_HOME_V2_PREFIX_BYTES) != 0) {
        return GEIST_HOME_V2_NEED_MORE;
    }
    const uint32_t payload_len = ((uint32_t) wire[0] << 24u) | ((uint32_t) wire[1] << 16u) |
                                 ((uint32_t) wire[2] << 8u) | (uint32_t) wire[3];
    if (payload_len > GEIST_HOME_V2_MAX_PAYLOAD ||
        (size_t) payload_len + GEIST_HOME_V2_PREFIX_BYTES > cap) {
        return GEIST_HOME_V2_E_FRAME_TOO_LARGE;
    }
    if (payload_len == 0u) {
        return GEIST_HOME_V2_E_INVALID_JSON;
    }
    if (home_daemon_v2_read_full(fd, wire + GEIST_HOME_V2_PREFIX_BYTES, payload_len) != 0) {
        return GEIST_HOME_V2_NEED_MORE;
    }
    *wire_len = GEIST_HOME_V2_PREFIX_BYTES + (size_t) payload_len;
    return GEIST_HOME_V2_OK;
}

static inline int home_daemon_v2_hex(uint8_t c) {
    if (c >= '0' && c <= '9') {
        return (int) (c - '0');
    }
    if (c >= 'a' && c <= 'f') {
        return (int) (c - 'a') + 10;
    }
    if (c >= 'A' && c <= 'F') {
        return (int) (c - 'A') + 10;
    }
    return -1;
}

static inline int
home_daemon_v2_utf8(uint32_t codepoint, size_t cap, char out[static cap], size_t *w) {
    if (codepoint <= 0x7fu) {
        if (*w + 1u >= cap) {
            return -1;
        }
        out[(*w)++] = (char) codepoint;
    } else if (codepoint <= 0x7ffu) {
        if (*w + 2u >= cap) {
            return -1;
        }
        out[(*w)++] = (char) (0xc0u | (codepoint >> 6u));
        out[(*w)++] = (char) (0x80u | (codepoint & 0x3fu));
    } else if (codepoint <= 0xffffu && !(codepoint >= 0xd800u && codepoint <= 0xdfffu)) {
        if (*w + 3u >= cap) {
            return -1;
        }
        out[(*w)++] = (char) (0xe0u | (codepoint >> 12u));
        out[(*w)++] = (char) (0x80u | ((codepoint >> 6u) & 0x3fu));
        out[(*w)++] = (char) (0x80u | (codepoint & 0x3fu));
    } else {
        return -1;
    }
    return 0;
}

static inline int home_daemon_v2_string(const struct geist_home_v2_json_span *span,
                                        size_t                                cap,
                                        char                                  out[static cap]) {
    if (span == NULL || span->len < 2u || span->p[0] != '"' || span->p[span->len - 1u] != '"') {
        return -1;
    }
    size_t w = 0u;
    for (size_t i = 1u; i + 1u < span->len; i++) {
        uint8_t c = span->p[i];
        if (c != '\\') {
            if (w + 1u >= cap) {
                return -1;
            }
            out[w++] = (char) c;
            continue;
        }
        if (++i + 1u >= span->len) {
            return -1;
        }
        c = span->p[i];
        if (c == 'u') {
            if (i + 4u >= span->len - 1u) {
                return -1;
            }
            uint32_t codepoint = 0u;
            for (unsigned j = 0u; j < 4u; j++) {
                int nibble = home_daemon_v2_hex(span->p[++i]);
                if (nibble < 0) {
                    return -1;
                }
                codepoint = (codepoint << 4u) | (uint32_t) nibble;
            }
            if (home_daemon_v2_utf8(codepoint, cap, out, &w) != 0) {
                return -1;
            }
        } else {
            char decoded;
            switch (c) {
            case '"':
            case '\\':
            case '/':
                decoded = (char) c;
                break;
            case 'b':
                decoded = '\b';
                break;
            case 'f':
                decoded = '\f';
                break;
            case 'n':
                decoded = '\n';
                break;
            case 'r':
                decoded = '\r';
                break;
            case 't':
                decoded = '\t';
                break;
            default:
                return -1;
            }
            if (w + 1u >= cap) {
                return -1;
            }
            out[w++] = decoded;
        }
    }
    out[w] = '\0';
    return (int) w;
}

static inline int
home_daemon_v2_uint(const struct geist_home_v2_json_span *span, uint64_t maximum, uint64_t *out) {
    if (span == NULL || span->len == 0u || out == NULL) {
        return -1;
    }
    uint64_t value = 0u;
    for (size_t i = 0u; i < span->len; i++) {
        if (span->p[i] < '0' || span->p[i] > '9' ||
            value > (maximum - (uint64_t) (span->p[i] - '0')) / 10u) {
            return -1;
        }
        value = value * 10u + (uint64_t) (span->p[i] - '0');
    }
    *out = value;
    return 0;
}

static inline int home_daemon_v2_parse_start(const struct geist_home_v2_frame *frame,
                                             struct home_daemon_v2_request    *request) {
    memset(request, 0, sizeof *request);
    if (frame->type != GEIST_HOME_V2_TYPE_CONVERSATION_START) {
        return -1;
    }
    struct geist_home_v2_json_span utterance = {0}, locale = {0}, registry = {0}, budget = {0};
    if (geist_home_v2_top_field(frame->json, frame->json_len, "utterance", &utterance) != 1 ||
        geist_home_v2_top_field(frame->json, frame->json_len, "locale", &locale) != 1 ||
        geist_home_v2_top_field(frame->json, frame->json_len, "registry_version", &registry) != 1 ||
        geist_home_v2_top_field(frame->json, frame->json_len, "max_tool_calls", &budget) != 1 ||
        home_daemon_v2_string(&utterance, sizeof request->utterance, request->utterance) <= 0 ||
        home_daemon_v2_string(&locale, sizeof request->locale, request->locale) <= 0) {
        return -1;
    }
    uint64_t max_tools = 0u;
    if (home_daemon_v2_uint(&registry, UINT64_MAX, &request->registry_version) != 0 ||
        home_daemon_v2_uint(&budget, 32u, &max_tools) != 0 || max_tools == 0u) {
        return -1;
    }
    request->max_tool_calls = (unsigned) max_tools;
    snprintf(request->request_id, sizeof request->request_id, "%s", frame->request_id);
    return 0;
}

static inline enum home_protocol_transport_status home_daemon_v2_roundtrip(void          *context,
                                                                           const uint8_t *request,
                                                                           size_t   request_len,
                                                                           uint8_t *response,
                                                                           size_t   response_cap,
                                                                           size_t  *response_len) {
    if (context == NULL || home_daemon_v2_write_full(*(int *) context, request, request_len) != 0) {
        return HOME_PROTOCOL_TRANSPORT_UNAVAILABLE;
    }
    enum geist_home_v2_status status =
            home_daemon_v2_read_wire(*(int *) context, response_cap, response, response_len);
    return status == GEIST_HOME_V2_OK ? HOME_PROTOCOL_TRANSPORT_OK
                                      : HOME_PROTOCOL_TRANSPORT_UNAVAILABLE;
}

static inline int
home_daemon_v2_connection(int connection, home_daemon_v2_runner_fn runner, void *runner_context) {
    if (runner == NULL) {
        return -1;
    }
    static uint8_t request_wire[HOME_DAEMON_V2_WIRE_CAP];
    size_t         request_wire_len = 0u;
    if (home_daemon_v2_read_wire(
                connection, sizeof request_wire, request_wire, &request_wire_len) !=
        GEIST_HOME_V2_OK) {
        return -1;
    }
    struct geist_home_v2_frame frame    = {0};
    size_t                     consumed = 0u;
    if (geist_home_v2_decode(request_wire, request_wire_len, &frame, &consumed) !=
                GEIST_HOME_V2_OK ||
        consumed != request_wire_len) {
        return -1;
    }
    struct home_daemon_v2_request request = {0};
    if (home_daemon_v2_parse_start(&frame, &request) != 0) {
        return -1;
    }

    static char response[HOME_DAEMON_V2_TEXT_CAP + 1u];
    size_t      response_len = 0u;
    if (runner(runner_context, connection, &request, sizeof response, response, &response_len) !=
                HOME_EXECUTOR_OK ||
        response_len == 0u || response_len > HOME_DAEMON_V2_TEXT_CAP) {
        return -1;
    }
    static char escaped[HOME_DAEMON_V2_ESCAPED_CAP];
    if (home_protocol_json_escape(response, sizeof escaped, escaped) == 0u) {
        return -1;
    }
    static char json[HOME_DAEMON_V2_JSON_CAP];
    int         json_len = snprintf(json,
                                    sizeof json,
                                    "{\"version\":2,\"request_id\":\"%s\","
                                    "\"type\":\"conversation.result\",\"text\":\"%s\"}",
                                    request.request_id,
                                    escaped);
    if (json_len <= 0 || (size_t) json_len >= sizeof json) {
        return -1;
    }
    static uint8_t result_wire[HOME_DAEMON_V2_WIRE_CAP];
    size_t         result_wire_len = 0u;
    if (geist_home_v2_encode((const uint8_t *) json,
                             (size_t) json_len,
                             result_wire,
                             sizeof result_wire,
                             &result_wire_len) != GEIST_HOME_V2_OK) {
        return -1;
    }
    return home_daemon_v2_write_full(connection, result_wire, result_wire_len);
}

#endif /* GEIST_HOME_DAEMON_V2_H */

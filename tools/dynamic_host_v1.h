/* dynamic_host_v1.h — newline-framed host-owned tool.call/tool.result bridge. */
#ifndef GEIST_DYNAMIC_HOST_V1_H
#define GEIST_DYNAMIC_HOST_V1_H

#include "agent.h"

#include <errno.h>
#include <unistd.h>

enum { GEIST_DYNAMIC_RESULT_CAP = 4096, GEIST_DYNAMIC_CALL_WIRE_CAP = 6144 };

struct geist_dynamic_host_session {
    int      fd;
    unsigned next_call_id;
    unsigned max_retries;
    bool     cancelled;
};

struct geist_dynamic_host_tool {
    struct geist_dynamic_host_session *session;
    const char                        *name;
};

static inline int geist_dynamic_host_write(int fd, const char *data, size_t len) {
    size_t off = 0u;
    while (off < len) {
        ssize_t written = write(fd, data + off, len - off);
        if (written > 0)
            off += (size_t) written;
        else if (written < 0 && errno == EINTR)
            continue;
        else
            return 0;
    }
    return 1;
}

static inline int
geist_dynamic_host_read_line(int fd, size_t cap, char out[static cap], size_t *len) {
    size_t used = 0u;
    while (used + 1u < cap) {
        ssize_t got = read(fd, out + used, 1u);
        if (got == 1) {
            if (out[used++] == '\n') {
                out[used - 1u] = '\0';
                *len           = used - 1u;
                return 1;
            }
        } else if (got < 0 && errno == EINTR) {
            continue;
        } else {
            return 0;
        }
    }
    return 0;
}

static inline enum geist_status geist_dynamic_host_invoke(void      *context,
                                                          size_t     args_len,
                                                          const char args[static args_len],
                                                          size_t     out_cap,
                                                          char       out[static out_cap],
                                                          size_t    *out_len) {
    struct geist_dynamic_host_tool *tool = (struct geist_dynamic_host_tool *) context;
    if (tool == NULL || tool->session == NULL || tool->session->fd < 0 || tool->name == NULL ||
        tool->session->cancelled)
        return GEIST_E_INVALID_STATE;
    for (unsigned retry = 0u;; retry++) {
        unsigned call_id = tool->session->next_call_id++;
        char     wire[GEIST_DYNAMIC_CALL_WIRE_CAP];
        int      length = snprintf(wire,
                                   sizeof wire,
                                   "{\"type\":\"tool.call\",\"call_id\":\"%u\",\"name\":\"%s\","
                                   "\"arguments\":%.*s}\n",
                                   call_id,
                                   tool->name,
                                   (int) args_len,
                                   args);
        if (length <= 0 || (size_t) length >= sizeof wire ||
            !geist_dynamic_host_write(tool->session->fd, wire, (size_t) length))
            return GEIST_E_IO;

        char   response[GEIST_DYNAMIC_RESULT_CAP];
        size_t response_len = 0u;
        if (!geist_dynamic_host_read_line(
                    tool->session->fd, sizeof response, response, &response_len))
            return GEIST_E_IO;
        struct jsv1_token tokens[JSV1_MAX_TOKENS];
        struct jsv1_doc   doc;
        if (jsv1_parse(response, response_len, JSV1_MAX_TOKENS, tokens, &doc) != JSV1_OK ||
            doc.tokens[0].type != JSV1_OBJECT || !jsv1_object_unique(&doc, 0))
            return GEIST_E_INVALID_ARG;
        int  type   = jsv1_object_get(&doc, 0, "type");
        int  id     = jsv1_object_get(&doc, 0, "call_id");
        int  status = jsv1_object_get(&doc, 0, "status");
        int  result = jsv1_object_get(&doc, 0, "result");
        char expected[16];
        snprintf(expected, sizeof expected, "%u", call_id);
        if (jsv1_token_is(&doc, type, "cancel") && jsv1_token_is(&doc, id, expected)) {
            tool->session->cancelled = true;
            return GEIST_E_INVALID_STATE;
        }
        if (!jsv1_token_is(&doc, type, "tool.result") || !jsv1_token_is(&doc, id, expected) ||
            result < 0)
            return GEIST_E_INVALID_ARG;
        if (status >= 0 && jsv1_token_is(&doc, status, "retryable") &&
            retry < tool->session->max_retries)
            continue;
        size_t result_len = doc.tokens[result].end - doc.tokens[result].start;
        if (result_len + 1u > out_cap)
            return GEIST_E_INVALID_ARG;
        memcpy(out, doc.json + doc.tokens[result].start, result_len);
        out[result_len] = '\0';
        if (out_len != NULL)
            *out_len = result_len;
        return GEIST_OK;
    }
}

#endif /* GEIST_DYNAMIC_HOST_V1_H */

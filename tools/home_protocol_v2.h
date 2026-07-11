/* tools/home_protocol_v2.h — allocation-free framing and envelope validation
 * for the geist Home Assistant protocol v2.
 *
 * Wire format:
 *   4-byte unsigned big-endian payload length
 *   exactly that many bytes of UTF-8 JSON
 *
 * The codec deliberately owns no socket or heap state. Callers can use it over
 * a Unix socket or a private TCP stream and retain control of deadlines and
 * cancellation. Payload-specific schemas are layered on top of the validated
 * common envelope.
 */
#ifndef GEIST_HOME_PROTOCOL_V2_H
#define GEIST_HOME_PROTOCOL_V2_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define GEIST_HOME_V2_VERSION 2u
#define GEIST_HOME_V2_PREFIX_BYTES 4u
#define GEIST_HOME_V2_MAX_PAYLOAD (1024u * 1024u)
#define GEIST_HOME_V2_MAX_REQUEST_ID 64u
#define GEIST_HOME_V2_MAX_JSON_DEPTH 32u

enum geist_home_v2_status {
    GEIST_HOME_V2_OK = 0,
    GEIST_HOME_V2_NEED_MORE,
    GEIST_HOME_V2_E_INVALID_ARG,
    GEIST_HOME_V2_E_FRAME_TOO_LARGE,
    GEIST_HOME_V2_E_INVALID_JSON,
    GEIST_HOME_V2_E_MISSING_FIELD,
    GEIST_HOME_V2_E_INVALID_FIELD,
    GEIST_HOME_V2_E_UNSUPPORTED_VERSION,
    GEIST_HOME_V2_E_UNKNOWN_TYPE,
};

enum geist_home_v2_type {
    GEIST_HOME_V2_TYPE_HELLO = 0,
    GEIST_HOME_V2_TYPE_HEALTH,
    GEIST_HOME_V2_TYPE_REGISTRY_REPLACE,
    GEIST_HOME_V2_TYPE_CONVERSATION_START,
    GEIST_HOME_V2_TYPE_TOOL_CALL,
    GEIST_HOME_V2_TYPE_TOOL_RESULT,
    GEIST_HOME_V2_TYPE_CONVERSATION_RESULT,
    GEIST_HOME_V2_TYPE_CANCEL,
};

struct geist_home_v2_frame {
    const uint8_t          *json;
    size_t                  json_len;
    uint32_t                version;
    enum geist_home_v2_type type;
    char                    request_id[GEIST_HOME_V2_MAX_REQUEST_ID + 1u];
};

static inline const char *geist_home_v2_status_string(enum geist_home_v2_status status) {
    switch (status) {
    case GEIST_HOME_V2_OK:
        return "ok";
    case GEIST_HOME_V2_NEED_MORE:
        return "need_more";
    case GEIST_HOME_V2_E_INVALID_ARG:
        return "invalid_arg";
    case GEIST_HOME_V2_E_FRAME_TOO_LARGE:
        return "frame_too_large";
    case GEIST_HOME_V2_E_INVALID_JSON:
        return "invalid_json";
    case GEIST_HOME_V2_E_MISSING_FIELD:
        return "missing_field";
    case GEIST_HOME_V2_E_INVALID_FIELD:
        return "invalid_field";
    case GEIST_HOME_V2_E_UNSUPPORTED_VERSION:
        return "unsupported_version";
    case GEIST_HOME_V2_E_UNKNOWN_TYPE:
        return "unknown_type";
    }
    return "unknown_error";
}

static inline const char *geist_home_v2_type_string(enum geist_home_v2_type type) {
    static const char *const names[] = {
            "hello",
            "health",
            "registry.replace",
            "conversation.start",
            "tool.call",
            "tool.result",
            "conversation.result",
            "cancel",
    };
    const size_t count = sizeof names / sizeof names[0];
    return (size_t) type < count ? names[type] : NULL;
}

static inline bool
geist_home_v2_type_parse(const uint8_t *text, size_t len, enum geist_home_v2_type *out) {
    if (text == NULL || out == NULL) {
        return false;
    }
    for (unsigned i = 0; i <= (unsigned) GEIST_HOME_V2_TYPE_CANCEL; i++) {
        const char *name = geist_home_v2_type_string((enum geist_home_v2_type) i);
        if (strlen(name) == len && memcmp(text, name, len) == 0) {
            *out = (enum geist_home_v2_type) i;
            return true;
        }
    }
    return false;
}

struct geist_home_v2_json_cursor {
    const uint8_t *p;
    const uint8_t *end;
};

struct geist_home_v2_json_string {
    const uint8_t *p;
    size_t         len;
    bool           escaped;
};

static inline bool geist_home_v2_is_space(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

static inline void geist_home_v2_skip_space(struct geist_home_v2_json_cursor *cursor) {
    while (cursor->p < cursor->end && geist_home_v2_is_space(*cursor->p)) {
        cursor->p++;
    }
}

static inline bool geist_home_v2_is_hex(uint8_t c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

static inline bool geist_home_v2_utf8_valid(const uint8_t *text, size_t len) {
    size_t i = 0u;
    while (i < len) {
        const uint8_t first = text[i++];
        if (first <= 0x7fu) {
            continue;
        }
        unsigned continuation = 0u;
        uint32_t codepoint    = 0u;
        uint32_t minimum      = 0u;
        if (first >= 0xc2u && first <= 0xdfu) {
            continuation = 1u;
            codepoint    = (uint32_t) (first & 0x1fu);
            minimum      = 0x80u;
        } else if (first >= 0xe0u && first <= 0xefu) {
            continuation = 2u;
            codepoint    = (uint32_t) (first & 0x0fu);
            minimum      = 0x800u;
        } else if (first >= 0xf0u && first <= 0xf4u) {
            continuation = 3u;
            codepoint    = (uint32_t) (first & 0x07u);
            minimum      = 0x10000u;
        } else {
            return false;
        }
        if (len - i < continuation) {
            return false;
        }
        for (unsigned j = 0u; j < continuation; j++) {
            const uint8_t next = text[i++];
            if ((next & 0xc0u) != 0x80u) {
                return false;
            }
            codepoint = (codepoint << 6u) | (uint32_t) (next & 0x3fu);
        }
        if (codepoint < minimum || codepoint > 0x10ffffu ||
            (codepoint >= 0xd800u && codepoint <= 0xdfffu)) {
            return false;
        }
    }
    return true;
}

static inline bool geist_home_v2_json_string(struct geist_home_v2_json_cursor *cursor,
                                             struct geist_home_v2_json_string *out) {
    if (cursor->p >= cursor->end || *cursor->p != '"') {
        return false;
    }
    cursor->p++;
    const uint8_t *start   = cursor->p;
    bool           escaped = false;
    while (cursor->p < cursor->end) {
        const uint8_t c = *cursor->p++;
        if (c == '"') {
            if (out != NULL) {
                out->p       = start;
                out->len     = (size_t) ((cursor->p - 1) - start);
                out->escaped = escaped;
            }
            return true;
        }
        if (c < 0x20u) {
            return false;
        }
        if (c != '\\') {
            continue;
        }
        escaped = true;
        if (cursor->p >= cursor->end) {
            return false;
        }
        const uint8_t escape = *cursor->p++;
        if (escape == 'u') {
            if ((size_t) (cursor->end - cursor->p) < 4u) {
                return false;
            }
            for (unsigned i = 0; i < 4u; i++) {
                if (!geist_home_v2_is_hex(cursor->p[i])) {
                    return false;
                }
            }
            cursor->p += 4;
        } else if (escape != '"' && escape != '\\' && escape != '/' && escape != 'b' &&
                   escape != 'f' && escape != 'n' && escape != 'r' && escape != 't') {
            return false;
        }
    }
    return false;
}

static inline bool geist_home_v2_json_value(struct geist_home_v2_json_cursor *cursor,
                                            unsigned                          depth);

static inline bool geist_home_v2_json_literal(struct geist_home_v2_json_cursor *cursor,
                                              const char                       *literal) {
    const size_t len = strlen(literal);
    if ((size_t) (cursor->end - cursor->p) < len || memcmp(cursor->p, literal, len) != 0) {
        return false;
    }
    cursor->p += len;
    return true;
}

static inline bool geist_home_v2_json_number(struct geist_home_v2_json_cursor *cursor) {
    const uint8_t *p = cursor->p;
    if (p < cursor->end && *p == '-') {
        p++;
    }
    if (p >= cursor->end) {
        return false;
    }
    if (*p == '0') {
        p++;
        if (p < cursor->end && *p >= '0' && *p <= '9') {
            return false;
        }
    } else {
        if (*p < '1' || *p > '9') {
            return false;
        }
        do {
            p++;
        } while (p < cursor->end && *p >= '0' && *p <= '9');
    }
    if (p < cursor->end && *p == '.') {
        p++;
        if (p >= cursor->end || *p < '0' || *p > '9') {
            return false;
        }
        do {
            p++;
        } while (p < cursor->end && *p >= '0' && *p <= '9');
    }
    if (p < cursor->end && (*p == 'e' || *p == 'E')) {
        p++;
        if (p < cursor->end && (*p == '+' || *p == '-')) {
            p++;
        }
        if (p >= cursor->end || *p < '0' || *p > '9') {
            return false;
        }
        do {
            p++;
        } while (p < cursor->end && *p >= '0' && *p <= '9');
    }
    cursor->p = p;
    return true;
}

static inline bool geist_home_v2_json_array(struct geist_home_v2_json_cursor *cursor,
                                            unsigned                          depth) {
    cursor->p++;
    geist_home_v2_skip_space(cursor);
    if (cursor->p < cursor->end && *cursor->p == ']') {
        cursor->p++;
        return true;
    }
    while (geist_home_v2_json_value(cursor, depth + 1u)) {
        geist_home_v2_skip_space(cursor);
        if (cursor->p < cursor->end && *cursor->p == ']') {
            cursor->p++;
            return true;
        }
        if (cursor->p >= cursor->end || *cursor->p++ != ',') {
            return false;
        }
        geist_home_v2_skip_space(cursor);
    }
    return false;
}

static inline bool geist_home_v2_json_object(struct geist_home_v2_json_cursor *cursor,
                                             unsigned                          depth) {
    cursor->p++;
    geist_home_v2_skip_space(cursor);
    if (cursor->p < cursor->end && *cursor->p == '}') {
        cursor->p++;
        return true;
    }
    for (;;) {
        if (!geist_home_v2_json_string(cursor, NULL)) {
            return false;
        }
        geist_home_v2_skip_space(cursor);
        if (cursor->p >= cursor->end || *cursor->p++ != ':') {
            return false;
        }
        if (!geist_home_v2_json_value(cursor, depth + 1u)) {
            return false;
        }
        geist_home_v2_skip_space(cursor);
        if (cursor->p < cursor->end && *cursor->p == '}') {
            cursor->p++;
            return true;
        }
        if (cursor->p >= cursor->end || *cursor->p++ != ',') {
            return false;
        }
        geist_home_v2_skip_space(cursor);
    }
}

static inline bool geist_home_v2_json_value(struct geist_home_v2_json_cursor *cursor,
                                            unsigned                          depth) {
    if (depth > GEIST_HOME_V2_MAX_JSON_DEPTH) {
        return false;
    }
    geist_home_v2_skip_space(cursor);
    if (cursor->p >= cursor->end) {
        return false;
    }
    switch (*cursor->p) {
    case '"':
        return geist_home_v2_json_string(cursor, NULL);
    case '{':
        return geist_home_v2_json_object(cursor, depth);
    case '[':
        return geist_home_v2_json_array(cursor, depth);
    case 't':
        return geist_home_v2_json_literal(cursor, "true");
    case 'f':
        return geist_home_v2_json_literal(cursor, "false");
    case 'n':
        return geist_home_v2_json_literal(cursor, "null");
    default:
        return geist_home_v2_json_number(cursor);
    }
}

static inline bool geist_home_v2_key_is(const struct geist_home_v2_json_string *key,
                                        const char                             *name) {
    const size_t len = strlen(name);
    return !key->escaped && key->len == len && memcmp(key->p, name, len) == 0;
}

static inline bool geist_home_v2_request_id_valid(const struct geist_home_v2_json_string *id) {
    if (id->escaped || id->len == 0u || id->len > GEIST_HOME_V2_MAX_REQUEST_ID) {
        return false;
    }
    for (size_t i = 0; i < id->len; i++) {
        const uint8_t c = id->p[i];
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
              c == '-' || c == '_')) {
            return false;
        }
    }
    return true;
}

static inline enum geist_home_v2_status
geist_home_v2_envelope(const uint8_t *json, size_t json_len, struct geist_home_v2_frame *out) {
    if (out != NULL) {
        *out = (struct geist_home_v2_frame) {0};
    }
    if (json == NULL || out == NULL || json_len == 0u) {
        return GEIST_HOME_V2_E_INVALID_ARG;
    }
    if (!geist_home_v2_utf8_valid(json, json_len)) {
        return GEIST_HOME_V2_E_INVALID_JSON;
    }
    struct geist_home_v2_json_cursor cursor = {.p = json, .end = json + json_len};
    geist_home_v2_skip_space(&cursor);
    if (cursor.p >= cursor.end || *cursor.p++ != '{') {
        return GEIST_HOME_V2_E_INVALID_JSON;
    }
    geist_home_v2_skip_space(&cursor);

    bool                             have_version = false;
    bool                             have_id      = false;
    bool                             have_type    = false;
    uint32_t                         version      = 0u;
    enum geist_home_v2_type          type         = GEIST_HOME_V2_TYPE_HELLO;
    struct geist_home_v2_json_string request_id   = {0};

    if (cursor.p < cursor.end && *cursor.p == '}') {
        cursor.p++;
    } else {
        for (;;) {
            struct geist_home_v2_json_string key = {0};
            if (!geist_home_v2_json_string(&cursor, &key)) {
                return GEIST_HOME_V2_E_INVALID_JSON;
            }
            geist_home_v2_skip_space(&cursor);
            if (cursor.p >= cursor.end || *cursor.p++ != ':') {
                return GEIST_HOME_V2_E_INVALID_JSON;
            }
            geist_home_v2_skip_space(&cursor);

            if (geist_home_v2_key_is(&key, "version")) {
                if (have_version || cursor.p >= cursor.end || *cursor.p < '0' || *cursor.p > '9') {
                    return GEIST_HOME_V2_E_INVALID_FIELD;
                }
                if (*cursor.p == '0' && cursor.p + 1 < cursor.end && cursor.p[1] >= '0' &&
                    cursor.p[1] <= '9') {
                    return GEIST_HOME_V2_E_INVALID_FIELD;
                }
                uint64_t parsed = 0u;
                do {
                    parsed = parsed * 10u + (uint64_t) (*cursor.p++ - '0');
                    if (parsed > UINT32_MAX) {
                        return GEIST_HOME_V2_E_INVALID_FIELD;
                    }
                } while (cursor.p < cursor.end && *cursor.p >= '0' && *cursor.p <= '9');
                version      = (uint32_t) parsed;
                have_version = true;
            } else if (geist_home_v2_key_is(&key, "request_id")) {
                if (have_id || !geist_home_v2_json_string(&cursor, &request_id) ||
                    !geist_home_v2_request_id_valid(&request_id)) {
                    return GEIST_HOME_V2_E_INVALID_FIELD;
                }
                have_id = true;
            } else if (geist_home_v2_key_is(&key, "type")) {
                struct geist_home_v2_json_string type_text = {0};
                if (have_type || !geist_home_v2_json_string(&cursor, &type_text) ||
                    type_text.escaped) {
                    return GEIST_HOME_V2_E_INVALID_FIELD;
                }
                if (!geist_home_v2_type_parse(type_text.p, type_text.len, &type)) {
                    return GEIST_HOME_V2_E_UNKNOWN_TYPE;
                }
                have_type = true;
            } else if (!geist_home_v2_json_value(&cursor, 1u)) {
                return GEIST_HOME_V2_E_INVALID_JSON;
            }

            geist_home_v2_skip_space(&cursor);
            if (cursor.p < cursor.end && *cursor.p == '}') {
                cursor.p++;
                break;
            }
            if (cursor.p >= cursor.end || *cursor.p++ != ',') {
                return GEIST_HOME_V2_E_INVALID_JSON;
            }
            geist_home_v2_skip_space(&cursor);
        }
    }
    geist_home_v2_skip_space(&cursor);
    if (cursor.p != cursor.end) {
        return GEIST_HOME_V2_E_INVALID_JSON;
    }
    if (!have_version || !have_id || !have_type) {
        return GEIST_HOME_V2_E_MISSING_FIELD;
    }
    if (version != GEIST_HOME_V2_VERSION) {
        return GEIST_HOME_V2_E_UNSUPPORTED_VERSION;
    }

    out->json     = json;
    out->json_len = json_len;
    out->version  = version;
    out->type     = type;
    memcpy(out->request_id, request_id.p, request_id.len);
    out->request_id[request_id.len] = '\0';
    return GEIST_HOME_V2_OK;
}

static inline enum geist_home_v2_status geist_home_v2_encode(
        const uint8_t *json, size_t json_len, uint8_t *dst, size_t dst_cap, size_t *written) {
    if (written != NULL) {
        *written = 0u;
    }
    if (json == NULL || dst == NULL || written == NULL || json_len == 0u) {
        return GEIST_HOME_V2_E_INVALID_ARG;
    }
    if (json_len > GEIST_HOME_V2_MAX_PAYLOAD) {
        return GEIST_HOME_V2_E_FRAME_TOO_LARGE;
    }
    struct geist_home_v2_frame ignored = {0};
    enum geist_home_v2_status  status  = geist_home_v2_envelope(json, json_len, &ignored);
    if (status != GEIST_HOME_V2_OK) {
        return status;
    }
    if (dst_cap < GEIST_HOME_V2_PREFIX_BYTES + json_len) {
        return GEIST_HOME_V2_NEED_MORE;
    }
    const uint32_t len = (uint32_t) json_len;
    dst[0]             = (uint8_t) (len >> 24u);
    dst[1]             = (uint8_t) (len >> 16u);
    dst[2]             = (uint8_t) (len >> 8u);
    dst[3]             = (uint8_t) len;
    memcpy(dst + GEIST_HOME_V2_PREFIX_BYTES, json, json_len);
    *written = GEIST_HOME_V2_PREFIX_BYTES + json_len;
    return GEIST_HOME_V2_OK;
}

static inline enum geist_home_v2_status geist_home_v2_decode(const uint8_t              *src,
                                                             size_t                      src_len,
                                                             struct geist_home_v2_frame *out,
                                                             size_t                     *consumed) {
    if (out != NULL) {
        *out = (struct geist_home_v2_frame) {0};
    }
    if (consumed != NULL) {
        *consumed = 0u;
    }
    if (src == NULL || out == NULL || consumed == NULL) {
        return GEIST_HOME_V2_E_INVALID_ARG;
    }
    if (src_len < GEIST_HOME_V2_PREFIX_BYTES) {
        return GEIST_HOME_V2_NEED_MORE;
    }
    const uint32_t len = ((uint32_t) src[0] << 24u) | ((uint32_t) src[1] << 16u) |
                         ((uint32_t) src[2] << 8u) | (uint32_t) src[3];
    if (len > GEIST_HOME_V2_MAX_PAYLOAD) {
        return GEIST_HOME_V2_E_FRAME_TOO_LARGE;
    }
    if (len == 0u) {
        return GEIST_HOME_V2_E_INVALID_JSON;
    }
    if (src_len - GEIST_HOME_V2_PREFIX_BYTES < (size_t) len) {
        return GEIST_HOME_V2_NEED_MORE;
    }
    enum geist_home_v2_status status =
            geist_home_v2_envelope(src + GEIST_HOME_V2_PREFIX_BYTES, (size_t) len, out);
    if (status == GEIST_HOME_V2_OK) {
        *consumed = GEIST_HOME_V2_PREFIX_BYTES + (size_t) len;
    }
    return status;
}

#endif /* GEIST_HOME_PROTOCOL_V2_H */

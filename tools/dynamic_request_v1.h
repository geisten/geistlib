/* dynamic_request_v1.h — strict JSON request boundary for dynamic tools v1. */
#ifndef GEIST_DYNAMIC_REQUEST_V1_H
#define GEIST_DYNAMIC_REQUEST_V1_H

#include "dynamic_tools_v1.h"

#include <stdint.h>

enum {
    GEIST_DYNAMIC_INPUT_CAP    = 4096,
    GEIST_DYNAMIC_LANGUAGE_CAP = 16,
    GEIST_DYNAMIC_CONTEXT_CAP  = 4096
};

enum geist_dynamic_request_status {
    GEIST_DYNAMIC_REQUEST_OK = 0,
    GEIST_DYNAMIC_REQUEST_E_INVALID_JSON,
    GEIST_DYNAMIC_REQUEST_E_INVALID_REQUEST,
    GEIST_DYNAMIC_REQUEST_E_INPUT_LIMIT,
    GEIST_DYNAMIC_REQUEST_E_TOOLSET,
};

struct geist_dynamic_request {
    char input[GEIST_DYNAMIC_INPUT_CAP];
    char language[GEIST_DYNAMIC_LANGUAGE_CAP];
    char context[GEIST_DYNAMIC_CONTEXT_CAP];
    /* dynamic-tools-v1 §Streaming: opt-in conversation.delta frames. A client
     * sets this only after seeing "streaming" in health.result features. */
    bool                         stream;
    struct geist_dynamic_toolset toolset;
};

static inline int geist_dynamic_request_language_valid(const char *language) {
    if (language == NULL || language[0] == '\0')
        return 1;
    for (size_t i = 0u; language[i] != '\0'; i++) {
        char c = language[i];
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
              c == '-'))
            return 0;
    }
    return 1;
}

static inline int geist_dynamic_request_hex(char c) {
    if (c >= '0' && c <= '9')
        return c - '0';
    if (c >= 'a' && c <= 'f')
        return c - 'a' + 10;
    return c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
}

static inline int geist_dynamic_request_utf8(uint32_t cp, size_t cap, char *out, size_t *w) {
    size_t need = cp <= 0x7fu ? 1u : cp <= 0x7ffu ? 2u : cp <= 0xffffu ? 3u : 4u;
    if (cp > 0x10ffffu || (cp >= 0xd800u && cp <= 0xdfffu) || *w + need >= cap)
        return 0;
    if (need == 1u) {
        out[(*w)++] = (char) cp;
    } else if (need == 2u) {
        out[(*w)++] = (char) (0xc0u | (cp >> 6u));
        out[(*w)++] = (char) (0x80u | (cp & 0x3fu));
    } else if (need == 3u) {
        out[(*w)++] = (char) (0xe0u | (cp >> 12u));
        out[(*w)++] = (char) (0x80u | ((cp >> 6u) & 0x3fu));
        out[(*w)++] = (char) (0x80u | (cp & 0x3fu));
    } else {
        out[(*w)++] = (char) (0xf0u | (cp >> 18u));
        out[(*w)++] = (char) (0x80u | ((cp >> 12u) & 0x3fu));
        out[(*w)++] = (char) (0x80u | ((cp >> 6u) & 0x3fu));
        out[(*w)++] = (char) (0x80u | (cp & 0x3fu));
    }
    return 1;
}

static inline int geist_dynamic_request_string(const struct jsv1_doc *doc,
                                               int                    token,
                                               size_t                 cap,
                                               char                   out[static cap]) {
    if (doc == NULL || token < 0 || doc->tokens[token].type != JSV1_STRING || cap == 0u)
        return 0;
    size_t w = 0u;
    for (size_t i = doc->tokens[token].start; i < doc->tokens[token].end; i++) {
        unsigned char c = (unsigned char) doc->json[i];
        if (c != '\\') {
            if (w + 1u >= cap)
                return 0;
            out[w++] = (char) c;
            continue;
        }
        if (++i >= doc->tokens[token].end)
            return 0;
        c = (unsigned char) doc->json[i];
        if (c == 'u') {
            if (i + 4u >= doc->tokens[token].end)
                return 0;
            uint32_t cp = 0u;
            for (unsigned d = 0u; d < 4u; d++) {
                int v = geist_dynamic_request_hex(doc->json[++i]);
                if (v < 0)
                    return 0;
                cp = (cp << 4u) | (uint32_t) v;
            }
            if (cp >= 0xd800u && cp <= 0xdbffu) {
                if (i + 6u >= doc->tokens[token].end || doc->json[i + 1u] != '\\' ||
                    doc->json[i + 2u] != 'u')
                    return 0;
                i += 2u;
                uint32_t low = 0u;
                for (unsigned d = 0u; d < 4u; d++) {
                    int v = geist_dynamic_request_hex(doc->json[++i]);
                    if (v < 0)
                        return 0;
                    low = (low << 4u) | (uint32_t) v;
                }
                if (low < 0xdc00u || low > 0xdfffu)
                    return 0;
                cp = 0x10000u + ((cp - 0xd800u) << 10u) + low - 0xdc00u;
            }
            if (!geist_dynamic_request_utf8(cp, cap, out, &w))
                return 0;
        } else {
            char decoded;
            switch (c) {
            case '"':
                decoded = '"';
                break;
            case '\\':
                decoded = '\\';
                break;
            case '/':
                decoded = '/';
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
                return 0;
            }
            if (w + 1u >= cap)
                return 0;
            out[w++] = decoded;
        }
    }
    out[w] = '\0';
    return 1;
}

static inline int geist_dynamic_request_keys(const struct jsv1_doc *doc,
                                             int                    object,
                                             const char *const      allowed[],
                                             size_t                 allowed_count) {
    if (!jsv1_object_unique(doc, object))
        return 0;
    int cursor = object;
    for (unsigned pair = 0u; pair < doc->tokens[object].size; pair++) {
        int key   = jsv1_child(doc, object, cursor);
        int value = jsv1_child(doc, object, key);
        int found = 0;
        for (size_t i = 0u; i < allowed_count; i++)
            found |= jsv1_token_is(doc, key, allowed[i]);
        if (key < 0 || value < 0 || !found)
            return 0;
        cursor = value;
    }
    return 1;
}

static inline enum geist_dynamic_request_status
geist_dynamic_request_parse(const char                   *json,
                            size_t                        json_len,
                            struct geist_dynamic_request *out,
                            enum geist_dynamic_status    *toolset_status) {
    if (out == NULL)
        return GEIST_DYNAMIC_REQUEST_E_INVALID_REQUEST;
    *out = (struct geist_dynamic_request) {0};
    if (toolset_status != NULL)
        *toolset_status = GEIST_DYNAMIC_OK;
    struct jsv1_token tokens[JSV1_MAX_TOKENS];
    struct jsv1_doc   doc;
    if (jsv1_parse(json, json_len, JSV1_MAX_TOKENS, tokens, &doc) != JSV1_OK ||
        doc.tokens[0].type != JSV1_OBJECT)
        return GEIST_DYNAMIC_REQUEST_E_INVALID_JSON;
    const char *const root_keys[] = {
            "input", "language", "context", "max_tool_steps", "tools", "stream"};
    if (!geist_dynamic_request_keys(&doc, 0, root_keys, 6u))
        return GEIST_DYNAMIC_REQUEST_E_INVALID_REQUEST;
    int input    = jsv1_object_get(&doc, 0, "input");
    int tools    = jsv1_object_get(&doc, 0, "tools");
    int budget   = jsv1_object_get(&doc, 0, "max_tool_steps");
    int language = jsv1_object_get(&doc, 0, "language");
    int context  = jsv1_object_get(&doc, 0, "context");
    int stream   = jsv1_object_get(&doc, 0, "stream");
    if (stream >= 0) { /* strict boundary: present -> must be a JSON boolean */
        if (doc.tokens[stream].type == JSV1_TRUE)
            out->stream = true;
        else if (doc.tokens[stream].type != JSV1_FALSE)
            return GEIST_DYNAMIC_REQUEST_E_INVALID_REQUEST;
    }
    if (input < 0 || tools < 0 || doc.tokens[input].type != JSV1_STRING ||
        doc.tokens[tools].type != JSV1_ARRAY || doc.tokens[tools].size > GEIST_DYNAMIC_MAX_TOOLS)
        return GEIST_DYNAMIC_REQUEST_E_INVALID_REQUEST;
    if (!geist_dynamic_request_string(&doc, input, sizeof out->input, out->input))
        return GEIST_DYNAMIC_REQUEST_E_INPUT_LIMIT;
    if (language >= 0 &&
        (!geist_dynamic_request_string(&doc, language, sizeof out->language, out->language) ||
         !geist_dynamic_request_language_valid(out->language)))
        return GEIST_DYNAMIC_REQUEST_E_INVALID_REQUEST;
    if (context >= 0 &&
        !geist_dynamic_request_string(&doc, context, sizeof out->context, out->context))
        return GEIST_DYNAMIC_REQUEST_E_INPUT_LIMIT;
    double steps_number = GEIST_DYNAMIC_DEFAULT_STEPS;
    if (budget >= 0 &&
        (!jsv1_integer_token(&doc, budget) || !jsv1_number(&doc, budget, &steps_number) ||
         steps_number < 1.0 || steps_number > GEIST_DYNAMIC_MAX_STEPS))
        return GEIST_DYNAMIC_REQUEST_E_INVALID_REQUEST;

    char names[GEIST_DYNAMIC_MAX_TOOLS][GEIST_DYNAMIC_NAME_CAP];
    char descriptions[GEIST_DYNAMIC_MAX_TOOLS][GEIST_DYNAMIC_DESCRIPTION_CAP];
    char schemas[GEIST_DYNAMIC_MAX_TOOLS][GEIST_DYNAMIC_SCHEMA_CAP];
    struct geist_dynamic_tool_spec specs[GEIST_DYNAMIC_MAX_TOOLS];
    int                            cursor      = tools;
    const char *const              tool_keys[] = {"name", "description", "parameters"};
    for (unsigned i = 0u; i < doc.tokens[tools].size; i++) {
        int tool = jsv1_child(&doc, tools, cursor);
        if (tool < 0 || doc.tokens[tool].type != JSV1_OBJECT ||
            !geist_dynamic_request_keys(&doc, tool, tool_keys, 3u))
            return GEIST_DYNAMIC_REQUEST_E_INVALID_REQUEST;
        int name        = jsv1_object_get(&doc, tool, "name");
        int description = jsv1_object_get(&doc, tool, "description");
        int parameters  = jsv1_object_get(&doc, tool, "parameters");
        if (name < 0 || description < 0 || parameters < 0 ||
            doc.tokens[parameters].type != JSV1_OBJECT ||
            !geist_dynamic_request_string(&doc, name, sizeof names[i], names[i]) ||
            !geist_dynamic_request_string(
                    &doc, description, sizeof descriptions[i], descriptions[i]))
            return GEIST_DYNAMIC_REQUEST_E_INVALID_REQUEST;
        size_t schema_len = doc.tokens[parameters].end - doc.tokens[parameters].start;
        if (schema_len >= sizeof schemas[i])
            return GEIST_DYNAMIC_REQUEST_E_TOOLSET;
        memcpy(schemas[i], doc.json + doc.tokens[parameters].start, schema_len);
        schemas[i][schema_len] = '\0';
        specs[i]               = (struct geist_dynamic_tool_spec) {
                .name = names[i], .description = descriptions[i], .parameters = schemas[i]};
        cursor = tool;
    }
    enum geist_dynamic_status compiled = geist_dynamic_toolset_compile(
            specs, doc.tokens[tools].size, (size_t) steps_number, &out->toolset);
    if (toolset_status != NULL)
        *toolset_status = compiled;
    return compiled == GEIST_DYNAMIC_OK ? GEIST_DYNAMIC_REQUEST_OK
                                        : GEIST_DYNAMIC_REQUEST_E_TOOLSET;
}

#endif /* GEIST_DYNAMIC_REQUEST_V1_H */

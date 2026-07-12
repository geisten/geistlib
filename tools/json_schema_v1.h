/* tools/json_schema_v1.h — fixed-memory JSON parser and validator for the
 * documented dynamic-tools-v1 JSON Schema subset. Host-neutral, no HA/model
 * dependency, no allocation, no coercion.
 */
#ifndef GEIST_JSON_SCHEMA_V1_H
#define GEIST_JSON_SCHEMA_V1_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum {
    JSV1_MAX_TOKENS      = 512,
    JSV1_MAX_DEPTH       = 8,
    JSV1_MAX_PROPERTIES  = 64,
    JSV1_MAX_ENUM_VALUES = 32,
    JSV1_MAX_ARRAY_ITEMS = 32,
};

enum jsv1_type {
    JSV1_OBJECT = 0,
    JSV1_ARRAY,
    JSV1_STRING,
    JSV1_NUMBER,
    JSV1_TRUE,
    JSV1_FALSE,
    JSV1_NULL,
};

struct jsv1_token {
    enum jsv1_type type;
    size_t         start;
    size_t         end;
    int            parent;
    unsigned       size;
    bool           escaped;
};

struct jsv1_doc {
    const char        *json;
    size_t             len;
    struct jsv1_token *tokens;
    size_t             count;
    size_t             cap;
};

enum jsv1_status {
    JSV1_OK = 0,
    JSV1_E_INVALID_JSON,
    JSV1_E_TOKEN_LIMIT,
    JSV1_E_DEPTH_LIMIT,
    JSV1_E_INVALID_SCHEMA,
    JSV1_E_UNSUPPORTED_KEYWORD,
    JSV1_E_TYPE,
    JSV1_E_REQUIRED,
    JSV1_E_ENUM,
    JSV1_E_ADDITIONAL_PROPERTY,
    JSV1_E_RANGE,
    JSV1_E_ITEM_LIMIT,
};

static inline const char *jsv1_status_string(enum jsv1_status status) {
    switch (status) {
    case JSV1_OK:
        return "ok";
    case JSV1_E_INVALID_JSON:
        return "invalid_json";
    case JSV1_E_TOKEN_LIMIT:
        return "token_limit";
    case JSV1_E_DEPTH_LIMIT:
        return "depth_limit";
    case JSV1_E_INVALID_SCHEMA:
        return "invalid_schema";
    case JSV1_E_UNSUPPORTED_KEYWORD:
        return "unsupported_keyword";
    case JSV1_E_TYPE:
        return "type_mismatch";
    case JSV1_E_REQUIRED:
        return "required_missing";
    case JSV1_E_ENUM:
        return "enum_mismatch";
    case JSV1_E_ADDITIONAL_PROPERTY:
        return "additional_property";
    case JSV1_E_RANGE:
        return "range";
    case JSV1_E_ITEM_LIMIT:
        return "item_limit";
    }
    return "unknown_error";
}

struct jsv1_parser {
    struct jsv1_doc *doc;
    size_t           pos;
    enum jsv1_status status;
};

static inline int jsv1_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

static inline void jsv1_skip(struct jsv1_parser *parser) {
    while (parser->pos < parser->doc->len && jsv1_space(parser->doc->json[parser->pos])) {
        parser->pos++;
    }
}

static inline int
jsv1_new(struct jsv1_parser *parser, enum jsv1_type type, size_t start, int parent) {
    if (parser->doc->count >= parser->doc->cap) {
        parser->status = JSV1_E_TOKEN_LIMIT;
        return -1;
    }
    int index                  = (int) parser->doc->count++;
    parser->doc->tokens[index] = (struct jsv1_token) {
            .type   = type,
            .start  = start,
            .end    = start,
            .parent = parent,
    };
    return index;
}

static inline int jsv1_hex(char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

static inline int jsv1_parse_value(struct jsv1_parser *parser, int parent, unsigned depth);

static inline int jsv1_parse_string(struct jsv1_parser *parser, int parent) {
    if (parser->pos >= parser->doc->len || parser->doc->json[parser->pos] != '"') {
        parser->status = JSV1_E_INVALID_JSON;
        return -1;
    }
    parser->pos++;
    int index = jsv1_new(parser, JSV1_STRING, parser->pos, parent);
    if (index < 0) {
        return -1;
    }
    while (parser->pos < parser->doc->len) {
        unsigned char c = (unsigned char) parser->doc->json[parser->pos++];
        if (c == '"') {
            parser->doc->tokens[index].end = parser->pos - 1u;
            return index;
        }
        if (c < 0x20u) {
            parser->status = JSV1_E_INVALID_JSON;
            return -1;
        }
        if (c != '\\') {
            continue;
        }
        parser->doc->tokens[index].escaped = true;
        if (parser->pos >= parser->doc->len) {
            parser->status = JSV1_E_INVALID_JSON;
            return -1;
        }
        char escape = parser->doc->json[parser->pos++];
        if (escape == 'u') {
            if (parser->doc->len - parser->pos < 4u) {
                parser->status = JSV1_E_INVALID_JSON;
                return -1;
            }
            for (unsigned i = 0u; i < 4u; i++) {
                if (!jsv1_hex(parser->doc->json[parser->pos++])) {
                    parser->status = JSV1_E_INVALID_JSON;
                    return -1;
                }
            }
        } else if (escape != '"' && escape != '\\' && escape != '/' && escape != 'b' &&
                   escape != 'f' && escape != 'n' && escape != 'r' && escape != 't') {
            parser->status = JSV1_E_INVALID_JSON;
            return -1;
        }
    }
    parser->status = JSV1_E_INVALID_JSON;
    return -1;
}

static inline int jsv1_parse_number(struct jsv1_parser *parser, int parent) {
    size_t start = parser->pos;
    if (parser->doc->json[parser->pos] == '-') {
        parser->pos++;
    }
    if (parser->pos >= parser->doc->len) {
        parser->status = JSV1_E_INVALID_JSON;
        return -1;
    }
    if (parser->doc->json[parser->pos] == '0') {
        parser->pos++;
        if (parser->pos < parser->doc->len && parser->doc->json[parser->pos] >= '0' &&
            parser->doc->json[parser->pos] <= '9') {
            parser->status = JSV1_E_INVALID_JSON;
            return -1;
        }
    } else {
        if (parser->doc->json[parser->pos] < '1' || parser->doc->json[parser->pos] > '9') {
            parser->status = JSV1_E_INVALID_JSON;
            return -1;
        }
        while (parser->pos < parser->doc->len && parser->doc->json[parser->pos] >= '0' &&
               parser->doc->json[parser->pos] <= '9') {
            parser->pos++;
        }
    }
    if (parser->pos < parser->doc->len && parser->doc->json[parser->pos] == '.') {
        parser->pos++;
        if (parser->pos >= parser->doc->len || parser->doc->json[parser->pos] < '0' ||
            parser->doc->json[parser->pos] > '9') {
            parser->status = JSV1_E_INVALID_JSON;
            return -1;
        }
        while (parser->pos < parser->doc->len && parser->doc->json[parser->pos] >= '0' &&
               parser->doc->json[parser->pos] <= '9') {
            parser->pos++;
        }
    }
    if (parser->pos < parser->doc->len &&
        (parser->doc->json[parser->pos] == 'e' || parser->doc->json[parser->pos] == 'E')) {
        parser->pos++;
        if (parser->pos < parser->doc->len &&
            (parser->doc->json[parser->pos] == '+' || parser->doc->json[parser->pos] == '-')) {
            parser->pos++;
        }
        if (parser->pos >= parser->doc->len || parser->doc->json[parser->pos] < '0' ||
            parser->doc->json[parser->pos] > '9') {
            parser->status = JSV1_E_INVALID_JSON;
            return -1;
        }
        while (parser->pos < parser->doc->len && parser->doc->json[parser->pos] >= '0' &&
               parser->doc->json[parser->pos] <= '9') {
            parser->pos++;
        }
    }
    int index = jsv1_new(parser, JSV1_NUMBER, start, parent);
    if (index >= 0) {
        parser->doc->tokens[index].end = parser->pos;
    }
    return index;
}

static inline int
jsv1_literal(struct jsv1_parser *parser, int parent, const char *literal, enum jsv1_type type) {
    size_t len = strlen(literal);
    if (parser->doc->len - parser->pos < len ||
        memcmp(parser->doc->json + parser->pos, literal, len) != 0) {
        parser->status = JSV1_E_INVALID_JSON;
        return -1;
    }
    int index = jsv1_new(parser, type, parser->pos, parent);
    parser->pos += len;
    if (index >= 0) {
        parser->doc->tokens[index].end = parser->pos;
    }
    return index;
}

static inline int jsv1_parse_object(struct jsv1_parser *parser, int parent, unsigned depth) {
    size_t start = parser->pos++;
    int    index = jsv1_new(parser, JSV1_OBJECT, start, parent);
    if (index < 0) {
        return -1;
    }
    jsv1_skip(parser);
    if (parser->pos < parser->doc->len && parser->doc->json[parser->pos] == '}') {
        parser->doc->tokens[index].end = ++parser->pos;
        return index;
    }
    for (;;) {
        if (jsv1_parse_string(parser, index) < 0) {
            return -1;
        }
        jsv1_skip(parser);
        if (parser->pos >= parser->doc->len || parser->doc->json[parser->pos++] != ':') {
            parser->status = JSV1_E_INVALID_JSON;
            return -1;
        }
        jsv1_skip(parser);
        if (jsv1_parse_value(parser, index, depth + 1u) < 0) {
            return -1;
        }
        parser->doc->tokens[index].size++;
        jsv1_skip(parser);
        if (parser->pos < parser->doc->len && parser->doc->json[parser->pos] == '}') {
            parser->doc->tokens[index].end = ++parser->pos;
            return index;
        }
        if (parser->pos >= parser->doc->len || parser->doc->json[parser->pos++] != ',') {
            parser->status = JSV1_E_INVALID_JSON;
            return -1;
        }
        jsv1_skip(parser);
    }
}

static inline int jsv1_parse_array(struct jsv1_parser *parser, int parent, unsigned depth) {
    size_t start = parser->pos++;
    int    index = jsv1_new(parser, JSV1_ARRAY, start, parent);
    if (index < 0) {
        return -1;
    }
    jsv1_skip(parser);
    if (parser->pos < parser->doc->len && parser->doc->json[parser->pos] == ']') {
        parser->doc->tokens[index].end = ++parser->pos;
        return index;
    }
    for (;;) {
        if (jsv1_parse_value(parser, index, depth + 1u) < 0) {
            return -1;
        }
        parser->doc->tokens[index].size++;
        jsv1_skip(parser);
        if (parser->pos < parser->doc->len && parser->doc->json[parser->pos] == ']') {
            parser->doc->tokens[index].end = ++parser->pos;
            return index;
        }
        if (parser->pos >= parser->doc->len || parser->doc->json[parser->pos++] != ',') {
            parser->status = JSV1_E_INVALID_JSON;
            return -1;
        }
        jsv1_skip(parser);
    }
}

static inline int jsv1_parse_value(struct jsv1_parser *parser, int parent, unsigned depth) {
    if (depth > JSV1_MAX_DEPTH) {
        parser->status = JSV1_E_DEPTH_LIMIT;
        return -1;
    }
    jsv1_skip(parser);
    if (parser->pos >= parser->doc->len) {
        parser->status = JSV1_E_INVALID_JSON;
        return -1;
    }
    switch (parser->doc->json[parser->pos]) {
    case '{':
        return jsv1_parse_object(parser, parent, depth);
    case '[':
        return jsv1_parse_array(parser, parent, depth);
    case '"':
        return jsv1_parse_string(parser, parent);
    case 't':
        return jsv1_literal(parser, parent, "true", JSV1_TRUE);
    case 'f':
        return jsv1_literal(parser, parent, "false", JSV1_FALSE);
    case 'n':
        return jsv1_literal(parser, parent, "null", JSV1_NULL);
    default:
        return jsv1_parse_number(parser, parent);
    }
}

static inline enum jsv1_status jsv1_parse(const char       *json,
                                          size_t            len,
                                          size_t            token_cap,
                                          struct jsv1_token tokens[static token_cap],
                                          struct jsv1_doc  *out) {
    if (json == NULL || tokens == NULL || out == NULL || len == 0u || token_cap == 0u) {
        return JSV1_E_INVALID_JSON;
    }
    *out = (struct jsv1_doc) {.json = json, .len = len, .tokens = tokens, .cap = token_cap};
    struct jsv1_parser parser = {.doc = out, .status = JSV1_OK};
    if (jsv1_parse_value(&parser, -1, 0u) < 0) {
        return parser.status;
    }
    jsv1_skip(&parser);
    return parser.pos == len ? JSV1_OK : JSV1_E_INVALID_JSON;
}

static inline int jsv1_child(const struct jsv1_doc *doc, int parent, int after) {
    for (int i = after + 1; i < (int) doc->count; i++) {
        if (doc->tokens[i].parent == parent) {
            return i;
        }
    }
    return -1;
}

static inline int jsv1_token_is(const struct jsv1_doc *doc, int index, const char *text) {
    if (index < 0 || doc->tokens[index].type != JSV1_STRING || doc->tokens[index].escaped) {
        return 0;
    }
    size_t len = strlen(text);
    return doc->tokens[index].end - doc->tokens[index].start == len &&
           memcmp(doc->json + doc->tokens[index].start, text, len) == 0;
}

/* -1 absent, -2 duplicate/malformed, else value token index. */
static inline int jsv1_object_get(const struct jsv1_doc *doc, int object, const char *key) {
    if (object < 0 || doc->tokens[object].type != JSV1_OBJECT) {
        return -2;
    }
    int found  = -1;
    int cursor = object;
    for (unsigned pair = 0u; pair < doc->tokens[object].size; pair++) {
        int name  = jsv1_child(doc, object, cursor);
        int value = jsv1_child(doc, object, name);
        if (name < 0 || value < 0 || doc->tokens[name].type != JSV1_STRING) {
            return -2;
        }
        if (jsv1_token_is(doc, name, key)) {
            if (found >= 0) {
                return -2;
            }
            found = value;
        }
        cursor = value;
    }
    return found;
}

static inline int
jsv1_same(const struct jsv1_doc *left, int li, const struct jsv1_doc *right, int ri) {
    if (li < 0 || ri < 0 || left->tokens[li].type != right->tokens[ri].type) {
        return 0;
    }
    size_t ln = left->tokens[li].end - left->tokens[li].start;
    size_t rn = right->tokens[ri].end - right->tokens[ri].start;
    return ln == rn &&
           memcmp(left->json + left->tokens[li].start, right->json + right->tokens[ri].start, ln) ==
                   0;
}

static inline int jsv1_number(const struct jsv1_doc *doc, int index, double *out) {
    if (index < 0 || doc->tokens[index].type != JSV1_NUMBER) {
        return 0;
    }
    size_t len = doc->tokens[index].end - doc->tokens[index].start;
    if (len == 0u || len >= 64u) {
        return 0;
    }
    char text[64];
    memcpy(text, doc->json + doc->tokens[index].start, len);
    text[len] = '\0';
    char *end = NULL;
    *out      = strtod(text, &end);
    return end != text && *end == '\0';
}

static inline int jsv1_integer_token(const struct jsv1_doc *doc, int index) {
    if (index < 0 || doc->tokens[index].type != JSV1_NUMBER) {
        return 0;
    }
    for (size_t i = doc->tokens[index].start; i < doc->tokens[index].end; i++) {
        if (doc->json[i] == '.' || doc->json[i] == 'e' || doc->json[i] == 'E') {
            return 0;
        }
    }
    return 1;
}

static inline int jsv1_object_unique(const struct jsv1_doc *doc, int object) {
    if (object < 0 || doc->tokens[object].type != JSV1_OBJECT) {
        return 0;
    }
    int cursor = object;
    for (unsigned pair = 0u; pair < doc->tokens[object].size; pair++) {
        int key   = jsv1_child(doc, object, cursor);
        int value = jsv1_child(doc, object, key);
        if (key < 0 || value < 0 || doc->tokens[key].type != JSV1_STRING ||
            doc->tokens[key].escaped) {
            return 0;
        }
        int prior_cursor = object;
        for (unsigned prior = 0u; prior < pair; prior++) {
            int prior_key   = jsv1_child(doc, object, prior_cursor);
            int prior_value = jsv1_child(doc, object, prior_key);
            if (prior_key < 0 || prior_value < 0 || jsv1_same(doc, prior_key, doc, key)) {
                return 0;
            }
            prior_cursor = prior_value;
        }
        cursor = value;
    }
    return 1;
}

static inline enum jsv1_status jsv1_validate_at(const struct jsv1_doc *schema,
                                                int                    schema_index,
                                                const struct jsv1_doc *value,
                                                int                    value_index,
                                                unsigned               depth);

static inline enum jsv1_status
jsv1_schema_check(const struct jsv1_doc *schema, int schema_index, unsigned depth);

static inline enum jsv1_status jsv1_validate_enum(const struct jsv1_doc *schema,
                                                  int                    schema_index,
                                                  const struct jsv1_doc *value,
                                                  int                    value_index) {
    int choices = jsv1_object_get(schema, schema_index, "enum");
    if (choices == -1) {
        return JSV1_OK;
    }
    if (choices < 0 || schema->tokens[choices].type != JSV1_ARRAY ||
        schema->tokens[choices].size == 0u || schema->tokens[choices].size > JSV1_MAX_ENUM_VALUES) {
        return JSV1_E_INVALID_SCHEMA;
    }
    int cursor = choices;
    for (unsigned i = 0u; i < schema->tokens[choices].size; i++) {
        int choice = jsv1_child(schema, choices, cursor);
        if (choice < 0 || (schema->tokens[choice].type != JSV1_STRING &&
                           schema->tokens[choice].type != JSV1_NUMBER &&
                           schema->tokens[choice].type != JSV1_TRUE &&
                           schema->tokens[choice].type != JSV1_FALSE)) {
            return JSV1_E_INVALID_SCHEMA;
        }
        if (jsv1_same(schema, choice, value, value_index)) {
            return JSV1_OK;
        }
        cursor = choice;
    }
    return JSV1_E_ENUM;
}

static inline int jsv1_allowed_keyword(const struct jsv1_doc *schema, int key, const char *type) {
    if (jsv1_token_is(schema, key, "type") || jsv1_token_is(schema, key, "enum")) {
        return 1;
    }
    if (strcmp(type, "object") == 0) {
        return jsv1_token_is(schema, key, "properties") || jsv1_token_is(schema, key, "required") ||
               jsv1_token_is(schema, key, "additionalProperties");
    }
    if (strcmp(type, "number") == 0 || strcmp(type, "integer") == 0) {
        return jsv1_token_is(schema, key, "minimum") || jsv1_token_is(schema, key, "maximum");
    }
    if (strcmp(type, "array") == 0) {
        return jsv1_token_is(schema, key, "items") || jsv1_token_is(schema, key, "minItems") ||
               jsv1_token_is(schema, key, "maxItems");
    }
    return 0;
}

static inline enum jsv1_status
jsv1_schema_check(const struct jsv1_doc *schema, int schema_index, unsigned depth) {
    if (depth > JSV1_MAX_DEPTH) {
        return JSV1_E_DEPTH_LIMIT;
    }
    if (schema_index < 0 || schema->tokens[schema_index].type != JSV1_OBJECT ||
        !jsv1_object_unique(schema, schema_index)) {
        return JSV1_E_INVALID_SCHEMA;
    }
    int type_token = jsv1_object_get(schema, schema_index, "type");
    if (type_token < 0 || schema->tokens[type_token].type != JSV1_STRING ||
        schema->tokens[type_token].escaped) {
        return JSV1_E_INVALID_SCHEMA;
    }
    size_t type_len = schema->tokens[type_token].end - schema->tokens[type_token].start;
    if (type_len >= 16u) {
        return JSV1_E_INVALID_SCHEMA;
    }
    char type[16];
    memcpy(type, schema->json + schema->tokens[type_token].start, type_len);
    type[type_len] = '\0';
    if (strcmp(type, "object") != 0 && strcmp(type, "array") != 0 && strcmp(type, "string") != 0 &&
        strcmp(type, "number") != 0 && strcmp(type, "integer") != 0 &&
        strcmp(type, "boolean") != 0) {
        return JSV1_E_INVALID_SCHEMA;
    }
    int cursor = schema_index;
    for (unsigned pair = 0u; pair < schema->tokens[schema_index].size; pair++) {
        int key   = jsv1_child(schema, schema_index, cursor);
        int child = jsv1_child(schema, schema_index, key);
        if (key < 0 || child < 0 || !jsv1_allowed_keyword(schema, key, type)) {
            return JSV1_E_UNSUPPORTED_KEYWORD;
        }
        cursor = child;
    }

    int enum_token = jsv1_object_get(schema, schema_index, "enum");
    if (enum_token >= 0) {
        if (schema->tokens[enum_token].type != JSV1_ARRAY ||
            schema->tokens[enum_token].size == 0u ||
            schema->tokens[enum_token].size > JSV1_MAX_ENUM_VALUES || strcmp(type, "object") == 0 ||
            strcmp(type, "array") == 0) {
            return JSV1_E_INVALID_SCHEMA;
        }
        enum jsv1_type expected    = JSV1_NULL;
        int            enum_cursor = enum_token;
        for (unsigned i = 0u; i < schema->tokens[enum_token].size; i++) {
            int choice = jsv1_child(schema, enum_token, enum_cursor);
            if (choice < 0 || (schema->tokens[choice].type != JSV1_STRING &&
                               schema->tokens[choice].type != JSV1_NUMBER &&
                               schema->tokens[choice].type != JSV1_TRUE &&
                               schema->tokens[choice].type != JSV1_FALSE)) {
                return JSV1_E_INVALID_SCHEMA;
            }
            enum jsv1_type normalized = schema->tokens[choice].type;
            if (normalized == JSV1_FALSE) {
                normalized = JSV1_TRUE;
            }
            if (i > 0u && normalized != expected) {
                return JSV1_E_INVALID_SCHEMA;
            }
            expected    = normalized;
            enum_cursor = choice;
        }
    }

    if (strcmp(type, "object") == 0) {
        int properties = jsv1_object_get(schema, schema_index, "properties");
        if (properties >= 0) {
            if (schema->tokens[properties].type != JSV1_OBJECT ||
                schema->tokens[properties].size > JSV1_MAX_PROPERTIES ||
                !jsv1_object_unique(schema, properties)) {
                return JSV1_E_INVALID_SCHEMA;
            }
            int property_cursor = properties;
            for (unsigned i = 0u; i < schema->tokens[properties].size; i++) {
                int              name   = jsv1_child(schema, properties, property_cursor);
                int              child  = jsv1_child(schema, properties, name);
                enum jsv1_status status = jsv1_schema_check(schema, child, depth + 1u);
                if (status != JSV1_OK) {
                    return status;
                }
                property_cursor = child;
            }
        }
        int additional = jsv1_object_get(schema, schema_index, "additionalProperties");
        if (additional >= 0 && schema->tokens[additional].type != JSV1_TRUE &&
            schema->tokens[additional].type != JSV1_FALSE) {
            return JSV1_E_INVALID_SCHEMA;
        }
        int required = jsv1_object_get(schema, schema_index, "required");
        if (required >= 0) {
            if (properties < 0 || schema->tokens[required].type != JSV1_ARRAY ||
                schema->tokens[required].size > JSV1_MAX_PROPERTIES) {
                return JSV1_E_INVALID_SCHEMA;
            }
            int required_cursor = required;
            for (unsigned i = 0u; i < schema->tokens[required].size; i++) {
                int name = jsv1_child(schema, required, required_cursor);
                if (name < 0 || schema->tokens[name].type != JSV1_STRING ||
                    schema->tokens[name].escaped) {
                    return JSV1_E_INVALID_SCHEMA;
                }
                for (int prior = required, p = 0; p < (int) i; p++) {
                    prior = jsv1_child(schema, required, prior);
                    if (prior < 0 || jsv1_same(schema, prior, schema, name)) {
                        return JSV1_E_INVALID_SCHEMA;
                    }
                }
                size_t length = schema->tokens[name].end - schema->tokens[name].start;
                char   property_name[128];
                if (length == 0u || length >= sizeof property_name) {
                    return JSV1_E_INVALID_SCHEMA;
                }
                memcpy(property_name, schema->json + schema->tokens[name].start, length);
                property_name[length] = '\0';
                if (jsv1_object_get(schema, properties, property_name) < 0) {
                    return JSV1_E_INVALID_SCHEMA;
                }
                required_cursor = name;
            }
        }
    } else if (strcmp(type, "array") == 0) {
        int items = jsv1_object_get(schema, schema_index, "items");
        if (items < 0) {
            return JSV1_E_INVALID_SCHEMA;
        }
        enum jsv1_status status = jsv1_schema_check(schema, items, depth + 1u);
        if (status != JSV1_OK) {
            return status;
        }
        double minimum = 0.0, maximum = JSV1_MAX_ARRAY_ITEMS;
        int    min_token = jsv1_object_get(schema, schema_index, "minItems");
        int    max_token = jsv1_object_get(schema, schema_index, "maxItems");
        if ((min_token >= 0 && (!jsv1_integer_token(schema, min_token) ||
                                !jsv1_number(schema, min_token, &minimum) || minimum < 0.0)) ||
            (max_token >= 0 &&
             (!jsv1_integer_token(schema, max_token) || !jsv1_number(schema, max_token, &maximum) ||
              maximum < 0.0 || maximum > JSV1_MAX_ARRAY_ITEMS)) ||
            minimum > maximum) {
            return JSV1_E_INVALID_SCHEMA;
        }
    } else if (strcmp(type, "number") == 0 || strcmp(type, "integer") == 0) {
        double minimum = 0.0, maximum = 0.0;
        int    min_token = jsv1_object_get(schema, schema_index, "minimum");
        int    max_token = jsv1_object_get(schema, schema_index, "maximum");
        if ((min_token >= 0 && !jsv1_number(schema, min_token, &minimum)) ||
            (max_token >= 0 && !jsv1_number(schema, max_token, &maximum)) ||
            (min_token >= 0 && max_token >= 0 && minimum > maximum)) {
            return JSV1_E_INVALID_SCHEMA;
        }
    }
    return JSV1_OK;
}

static inline enum jsv1_status jsv1_validate_object(const struct jsv1_doc *schema,
                                                    int                    schema_index,
                                                    const struct jsv1_doc *value,
                                                    int                    value_index,
                                                    unsigned               depth) {
    if (value->tokens[value_index].type != JSV1_OBJECT) {
        return JSV1_E_TYPE;
    }
    if (value->tokens[value_index].size > JSV1_MAX_PROPERTIES) {
        return JSV1_E_ITEM_LIMIT;
    }
    int properties = jsv1_object_get(schema, schema_index, "properties");
    if (properties >= 0 && schema->tokens[properties].type != JSV1_OBJECT) {
        return JSV1_E_INVALID_SCHEMA;
    }
    int additional       = jsv1_object_get(schema, schema_index, "additionalProperties");
    int allow_additional = additional >= 0 && schema->tokens[additional].type == JSV1_TRUE;
    if (additional >= 0 && schema->tokens[additional].type != JSV1_TRUE &&
        schema->tokens[additional].type != JSV1_FALSE) {
        return JSV1_E_INVALID_SCHEMA;
    }

    int cursor = value_index;
    for (unsigned pair = 0u; pair < value->tokens[value_index].size; pair++) {
        int key   = jsv1_child(value, value_index, cursor);
        int child = jsv1_child(value, value_index, key);
        if (key < 0 || child < 0 || value->tokens[key].type != JSV1_STRING ||
            value->tokens[key].escaped) {
            return JSV1_E_INVALID_JSON;
        }
        for (int prior = value_index, p = 0; p < (int) pair; p++) {
            prior           = jsv1_child(value, value_index, prior);
            int prior_value = jsv1_child(value, value_index, prior);
            if (jsv1_same(value, prior, value, key)) {
                return JSV1_E_INVALID_JSON;
            }
            prior = prior_value;
        }
        int property_schema = -1;
        if (properties >= 0) {
            size_t key_len = value->tokens[key].end - value->tokens[key].start;
            int    pcursor = properties;
            for (unsigned i = 0u; i < schema->tokens[properties].size; i++) {
                int    pkey   = jsv1_child(schema, properties, pcursor);
                int    pvalue = jsv1_child(schema, properties, pkey);
                size_t plen =
                        pkey >= 0 ? schema->tokens[pkey].end - schema->tokens[pkey].start : 0u;
                if (pkey < 0 || pvalue < 0 || schema->tokens[pkey].type != JSV1_STRING) {
                    return JSV1_E_INVALID_SCHEMA;
                }
                if (!schema->tokens[pkey].escaped && plen == key_len &&
                    memcmp(schema->json + schema->tokens[pkey].start,
                           value->json + value->tokens[key].start,
                           key_len) == 0) {
                    property_schema = pvalue;
                    break;
                }
                pcursor = pvalue;
            }
        }
        if (property_schema < 0) {
            if (!allow_additional) {
                return JSV1_E_ADDITIONAL_PROPERTY;
            }
        } else {
            enum jsv1_status status =
                    jsv1_validate_at(schema, property_schema, value, child, depth + 1u);
            if (status != JSV1_OK) {
                return status;
            }
        }
        cursor = child;
    }

    int required = jsv1_object_get(schema, schema_index, "required");
    if (required >= 0) {
        if (schema->tokens[required].type != JSV1_ARRAY ||
            schema->tokens[required].size > JSV1_MAX_PROPERTIES) {
            return JSV1_E_INVALID_SCHEMA;
        }
        int rcursor = required;
        for (unsigned i = 0u; i < schema->tokens[required].size; i++) {
            int required_key = jsv1_child(schema, required, rcursor);
            if (required_key < 0 || schema->tokens[required_key].type != JSV1_STRING ||
                schema->tokens[required_key].escaped) {
                return JSV1_E_INVALID_SCHEMA;
            }
            size_t rlen    = schema->tokens[required_key].end - schema->tokens[required_key].start;
            int    present = 0, vcursor = value_index;
            for (unsigned p = 0u; p < value->tokens[value_index].size; p++) {
                int    key   = jsv1_child(value, value_index, vcursor);
                int    child = jsv1_child(value, value_index, key);
                size_t klen  = key >= 0 ? value->tokens[key].end - value->tokens[key].start : 0u;
                present |= key >= 0 && !value->tokens[key].escaped && klen == rlen &&
                           memcmp(value->json + value->tokens[key].start,
                                  schema->json + schema->tokens[required_key].start,
                                  rlen) == 0;
                vcursor = child;
            }
            if (!present) {
                return JSV1_E_REQUIRED;
            }
            rcursor = required_key;
        }
    }
    return JSV1_OK;
}

static inline enum jsv1_status jsv1_validate_at(const struct jsv1_doc *schema,
                                                int                    schema_index,
                                                const struct jsv1_doc *value,
                                                int                    value_index,
                                                unsigned               depth) {
    if (depth > JSV1_MAX_DEPTH || schema_index < 0 || value_index < 0 ||
        schema->tokens[schema_index].type != JSV1_OBJECT) {
        return depth > JSV1_MAX_DEPTH ? JSV1_E_DEPTH_LIMIT : JSV1_E_INVALID_SCHEMA;
    }
    int type_token = jsv1_object_get(schema, schema_index, "type");
    if (type_token < 0 || schema->tokens[type_token].type != JSV1_STRING ||
        schema->tokens[type_token].escaped) {
        return JSV1_E_INVALID_SCHEMA;
    }
    size_t type_len = schema->tokens[type_token].end - schema->tokens[type_token].start;
    if (type_len >= 16u) {
        return JSV1_E_INVALID_SCHEMA;
    }
    char type[16];
    memcpy(type, schema->json + schema->tokens[type_token].start, type_len);
    type[type_len] = '\0';
    if (strcmp(type, "object") != 0 && strcmp(type, "array") != 0 && strcmp(type, "string") != 0 &&
        strcmp(type, "number") != 0 && strcmp(type, "integer") != 0 &&
        strcmp(type, "boolean") != 0) {
        return JSV1_E_INVALID_SCHEMA;
    }
    int cursor = schema_index;
    for (unsigned pair = 0u; pair < schema->tokens[schema_index].size; pair++) {
        int key   = jsv1_child(schema, schema_index, cursor);
        int child = jsv1_child(schema, schema_index, key);
        if (key < 0 || child < 0 || !jsv1_allowed_keyword(schema, key, type)) {
            return JSV1_E_UNSUPPORTED_KEYWORD;
        }
        cursor = child;
    }
    enum jsv1_status status;
    if (strcmp(type, "object") == 0) {
        status = jsv1_validate_object(schema, schema_index, value, value_index, depth);
    } else if (strcmp(type, "array") == 0) {
        if (value->tokens[value_index].type != JSV1_ARRAY) {
            return JSV1_E_TYPE;
        }
        int items = jsv1_object_get(schema, schema_index, "items");
        if (items < 0) {
            return JSV1_E_INVALID_SCHEMA;
        }
        uint64_t minimum = 0u, maximum = JSV1_MAX_ARRAY_ITEMS;
        int      min_token = jsv1_object_get(schema, schema_index, "minItems");
        int      max_token = jsv1_object_get(schema, schema_index, "maxItems");
        double   number;
        if (min_token >= 0 && (!jsv1_integer_token(schema, min_token) ||
                               !jsv1_number(schema, min_token, &number) || number < 0.0)) {
            return JSV1_E_INVALID_SCHEMA;
        } else if (min_token >= 0) {
            minimum = (uint64_t) number;
        }
        if (max_token >= 0 &&
            (!jsv1_integer_token(schema, max_token) || !jsv1_number(schema, max_token, &number) ||
             number < 0.0 || number > JSV1_MAX_ARRAY_ITEMS)) {
            return JSV1_E_INVALID_SCHEMA;
        } else if (max_token >= 0) {
            maximum = (uint64_t) number;
        }
        if (minimum > maximum || value->tokens[value_index].size < minimum ||
            value->tokens[value_index].size > maximum ||
            value->tokens[value_index].size > JSV1_MAX_ARRAY_ITEMS) {
            return JSV1_E_ITEM_LIMIT;
        }
        int item = value_index;
        for (unsigned i = 0u; i < value->tokens[value_index].size; i++) {
            item   = jsv1_child(value, value_index, item);
            status = jsv1_validate_at(schema, items, value, item, depth + 1u);
            if (status != JSV1_OK) {
                return status;
            }
        }
        status = JSV1_OK;
    } else if (strcmp(type, "string") == 0) {
        status = value->tokens[value_index].type == JSV1_STRING ? JSV1_OK : JSV1_E_TYPE;
    } else if (strcmp(type, "number") == 0) {
        status = value->tokens[value_index].type == JSV1_NUMBER ? JSV1_OK : JSV1_E_TYPE;
    } else if (strcmp(type, "integer") == 0) {
        status = jsv1_integer_token(value, value_index) ? JSV1_OK : JSV1_E_TYPE;
    } else {
        status = value->tokens[value_index].type == JSV1_TRUE ||
                                 value->tokens[value_index].type == JSV1_FALSE
                         ? JSV1_OK
                         : JSV1_E_TYPE;
    }
    if (status != JSV1_OK) {
        return status;
    }
    status = jsv1_validate_enum(schema, schema_index, value, value_index);
    if (status != JSV1_OK) {
        return status;
    }
    if (strcmp(type, "number") == 0 || strcmp(type, "integer") == 0) {
        double actual, bound;
        if (!jsv1_number(value, value_index, &actual)) {
            return JSV1_E_TYPE;
        }
        int minimum = jsv1_object_get(schema, schema_index, "minimum");
        int maximum = jsv1_object_get(schema, schema_index, "maximum");
        if ((minimum >= 0 && (!jsv1_number(schema, minimum, &bound) || actual < bound)) ||
            (maximum >= 0 && (!jsv1_number(schema, maximum, &bound) || actual > bound))) {
            return JSV1_E_RANGE;
        }
    }
    return JSV1_OK;
}

static inline enum jsv1_status jsv1_validate(const char *schema_json,
                                             size_t      schema_len,
                                             const char *value_json,
                                             size_t      value_len) {
    struct jsv1_token schema_tokens[JSV1_MAX_TOKENS];
    struct jsv1_token value_tokens[JSV1_MAX_TOKENS];
    struct jsv1_doc   schema, value;
    enum jsv1_status  status =
            jsv1_parse(schema_json, schema_len, JSV1_MAX_TOKENS, schema_tokens, &schema);
    if (status != JSV1_OK) {
        return status == JSV1_E_DEPTH_LIMIT || status == JSV1_E_TOKEN_LIMIT ? status
                                                                            : JSV1_E_INVALID_SCHEMA;
    }
    status = jsv1_schema_check(&schema, 0, 0u);
    if (status != JSV1_OK) {
        return status;
    }
    status = jsv1_parse(value_json, value_len, JSV1_MAX_TOKENS, value_tokens, &value);
    if (status != JSV1_OK) {
        return status;
    }
    return jsv1_validate_at(&schema, 0, &value, 0, 0u);
}

#endif /* GEIST_JSON_SCHEMA_V1_H */

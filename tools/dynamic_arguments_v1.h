/* dynamic_arguments_v1.h — bounded schema-driven forced argument extraction. */
#ifndef GEIST_DYNAMIC_ARGUMENTS_V1_H
#define GEIST_DYNAMIC_ARGUMENTS_V1_H

#include "json_schema_v1.h"

#include <ctype.h>
#include <stdio.h>
#include <string.h>

static inline int geist_dynamic_text_contains(size_t      text_len,
                                              const char  text[static text_len],
                                              const char *needle,
                                              size_t      needle_len) {
    if (needle_len == 0u || needle_len > text_len)
        return 0;
    for (size_t i = 0u; i + needle_len <= text_len; i++) {
        size_t j = 0u;
        while (j < needle_len &&
               tolower((unsigned char) text[i + j]) == tolower((unsigned char) needle[j]))
            j++;
        if (j == needle_len)
            return 1;
    }
    return 0;
}

static inline size_t geist_dynamic_json_string(size_t     text_len,
                                               const char text[static text_len],
                                               size_t     cap,
                                               char       out[static cap],
                                               size_t     written) {
    if (written + 2u >= cap)
        return 0u;
    out[written++] = '"';
    for (size_t i = 0u; i < text_len; i++) {
        unsigned char c      = (unsigned char) text[i];
        const char   *escape = NULL;
        switch (c) {
        case '"':
            escape = "\\\"";
            break;
        case '\\':
            escape = "\\\\";
            break;
        case '\b':
            escape = "\\b";
            break;
        case '\f':
            escape = "\\f";
            break;
        case '\n':
            escape = "\\n";
            break;
        case '\r':
            escape = "\\r";
            break;
        case '\t':
            escape = "\\t";
            break;
        default:
            break;
        }
        if (escape != NULL) {
            size_t length = strlen(escape);
            if (written + length + 2u >= cap)
                return 0u;
            memcpy(out + written, escape, length);
            written += length;
        } else if (c < 0x20u) {
            if (written + 7u >= cap)
                return 0u;
            written += (size_t) snprintf(out + written, cap - written, "\\u%04x", c);
        } else {
            if (written + 2u >= cap)
                return 0u;
            out[written++] = (char) c;
        }
    }
    out[written++] = '"';
    out[written]   = '\0';
    return written;
}

static inline int
geist_dynamic_required(const struct jsv1_doc *schema, int object_schema, int name) {
    int required = jsv1_object_get(schema, object_schema, "required");
    if (required < 0)
        return 0;
    int cursor = required;
    for (unsigned i = 0u; i < schema->tokens[required].size; i++) {
        int item = jsv1_child(schema, required, cursor);
        if (jsv1_same(schema, item, schema, name))
            return 1;
        cursor = item;
    }
    return 0;
}

static inline int geist_dynamic_number_n(size_t     text_len,
                                         const char text[static text_len],
                                         unsigned   wanted,
                                         size_t    *start,
                                         size_t    *length) {
    unsigned found = 0u;
    for (size_t i = 0u; i < text_len;) {
        int begins =
                (text[i] >= '0' && text[i] <= '9') ||
                (text[i] == '-' && i + 1u < text_len && text[i + 1u] >= '0' && text[i + 1u] <= '9');
        if (!begins) {
            i++;
            continue;
        }
        size_t begin = i++;
        while (i < text_len && text[i] >= '0' && text[i] <= '9')
            i++;
        if (i < text_len && text[i] == '.') {
            i++;
            while (i < text_len && text[i] >= '0' && text[i] <= '9')
                i++;
        }
        if (found++ == wanted) {
            *start  = begin;
            *length = i - begin;
            return 1;
        }
    }
    return 0;
}

static inline int
geist_dynamic_boolean(size_t text_len, const char text[static text_len], int *value) {
    static const char *const yes[] = {"true", " yes", " on", "enable", "aktiv", " ein", "ja"};
    static const char *const no[]  = {"false", " no", " off", "disable", "inaktiv", " aus", "nein"};
    for (size_t i = 0u; i < sizeof yes / sizeof *yes; i++) {
        if (geist_dynamic_text_contains(text_len, text, yes[i], strlen(yes[i]))) {
            *value = 1;
            return 1;
        }
    }
    for (size_t i = 0u; i < sizeof no / sizeof *no; i++) {
        if (geist_dynamic_text_contains(text_len, text, no[i], strlen(no[i]))) {
            *value = 0;
            return 1;
        }
    }
    return 0;
}

static inline int geist_dynamic_enum_match(const struct jsv1_doc *schema,
                                           int                    value_schema,
                                           size_t                 text_len,
                                           const char             text[static text_len],
                                           int                    after) {
    int choices = jsv1_object_get(schema, value_schema, "enum");
    if (choices < 0)
        return -1;
    int cursor = choices;
    for (unsigned i = 0u; i < schema->tokens[choices].size; i++) {
        int choice = jsv1_child(schema, choices, cursor);
        if (choice > after && schema->tokens[choice].type == JSV1_STRING &&
            !schema->tokens[choice].escaped) {
            size_t length = schema->tokens[choice].end - schema->tokens[choice].start;
            if (geist_dynamic_text_contains(
                        text_len, text, schema->json + schema->tokens[choice].start, length))
                return choice;
        }
        cursor = choice;
    }
    return -1;
}

static inline size_t geist_dynamic_append_raw(
        const struct jsv1_doc *doc, int token, size_t cap, char out[static cap], size_t written) {
    size_t length = doc->tokens[token].end - doc->tokens[token].start;
    if (written + length + 1u >= cap)
        return 0u;
    memcpy(out + written, doc->json + doc->tokens[token].start, length);
    written += length;
    out[written] = '\0';
    return written;
}

/* Returns JSON bytes, or 0 when a required value cannot be safely derived. */
static inline size_t geist_dynamic_arguments_build(const char *schema_json,
                                                   size_t      schema_len,
                                                   size_t      text_len,
                                                   const char  text[static text_len],
                                                   size_t      cap,
                                                   char        out[static cap]) {
    struct jsv1_token tokens[JSV1_MAX_TOKENS];
    struct jsv1_doc   schema;
    if (jsv1_parse(schema_json, schema_len, JSV1_MAX_TOKENS, tokens, &schema) != JSV1_OK ||
        jsv1_schema_check(&schema, 0, 0u) != JSV1_OK || cap < 3u)
        return 0u;
    int properties = jsv1_object_get(&schema, 0, "properties");
    if (properties < 0) {
        memcpy(out, "{}", 3u);
        return 2u;
    }
    size_t written   = 1u;
    out[0]           = '{';
    unsigned emitted = 0u, numeric_index = 0u;
    int      cursor = properties;
    for (unsigned i = 0u; i < schema.tokens[properties].size; i++) {
        int         name         = jsv1_child(&schema, properties, cursor);
        int         value_schema = jsv1_child(&schema, properties, name);
        int         type         = jsv1_object_get(&schema, value_schema, "type");
        int         required     = geist_dynamic_required(&schema, 0, name);
        size_t      name_len     = schema.tokens[name].end - schema.tokens[name].start;
        const char *name_text    = schema.json + schema.tokens[name].start;
        int include = required || geist_dynamic_text_contains(text_len, text, name_text, name_len);
        int enum_choice     = geist_dynamic_enum_match(&schema, value_schema, text_len, text, -1);
        size_t number_start = 0u, number_len = 0u;
        int    boolean_value = 0;
        if (enum_choice >= 0)
            include = 1;
        if ((jsv1_token_is(&schema, type, "number") || jsv1_token_is(&schema, type, "integer")) &&
            geist_dynamic_number_n(text_len, text, numeric_index, &number_start, &number_len))
            include = 1;
        if (jsv1_token_is(&schema, type, "boolean") &&
            geist_dynamic_boolean(text_len, text, &boolean_value))
            include = 1;
        if (!include) {
            cursor = value_schema;
            continue;
        }
        if (emitted++ > 0u)
            out[written++] = ',';
        written = geist_dynamic_json_string(name_len, name_text, cap, out, written);
        if (written == 0u || written + 2u >= cap)
            return 0u;
        out[written++] = ':';
        if (jsv1_token_is(&schema, type, "string")) {
            if (enum_choice >= 0) {
                if (written + 2u >= cap)
                    return 0u;
                out[written++] = '"';
                written        = geist_dynamic_append_raw(&schema, enum_choice, cap, out, written);
                if (written == 0u || written + 2u >= cap)
                    return 0u;
                out[written++] = '"';
            } else if (jsv1_object_get(&schema, value_schema, "enum") >= 0) {
                return 0u;
            } else {
                written = geist_dynamic_json_string(text_len, text, cap, out, written);
                if (written == 0u)
                    return 0u;
            }
        } else if (jsv1_token_is(&schema, type, "number") ||
                   jsv1_token_is(&schema, type, "integer")) {
            if (number_len == 0u || written + number_len + 2u >= cap)
                return 0u;
            memcpy(out + written, text + number_start, number_len);
            written += number_len;
            numeric_index++;
        } else if (jsv1_token_is(&schema, type, "boolean")) {
            if (!geist_dynamic_boolean(text_len, text, &boolean_value))
                return 0u;
            const char *literal = boolean_value ? "true" : "false";
            size_t      length  = strlen(literal);
            if (written + length + 2u >= cap)
                return 0u;
            memcpy(out + written, literal, length);
            written += length;
        } else if (jsv1_token_is(&schema, type, "array")) {
            int items     = jsv1_object_get(&schema, value_schema, "items");
            int item_type = jsv1_object_get(&schema, items, "type");
            if (!jsv1_token_is(&schema, item_type, "string") ||
                jsv1_object_get(&schema, items, "enum") < 0 || written + 2u >= cap)
                return 0u;
            out[written++]   = '[';
            unsigned matches = 0u;
            int      after   = -1;
            while ((enum_choice =
                            geist_dynamic_enum_match(&schema, items, text_len, text, after)) >= 0) {
                if (matches++ > 0u)
                    out[written++] = ',';
                out[written++] = '"';
                written        = geist_dynamic_append_raw(&schema, enum_choice, cap, out, written);
                if (written == 0u || written + 3u >= cap)
                    return 0u;
                out[written++] = '"';
                after          = enum_choice;
            }
            if (matches == 0u)
                return 0u;
            out[written++] = ']';
        } else {
            return 0u; /* nested forced extraction needs an explicit host value */
        }
        cursor = value_schema;
    }
    if (written + 2u >= cap)
        return 0u;
    out[written++] = '}';
    out[written]   = '\0';
    return jsv1_validate(schema_json, schema_len, out, written) == JSV1_OK ? written : 0u;
}

#endif /* GEIST_DYNAMIC_ARGUMENTS_V1_H */

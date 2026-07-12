/* dynamic_tools_v1.h — immutable, host-neutral per-request capability set. */
#ifndef GEIST_DYNAMIC_TOOLS_V1_H
#define GEIST_DYNAMIC_TOOLS_V1_H

#include "json_schema_v1.h"

#include <stddef.h>
#include <string.h>

enum {
    GEIST_DYNAMIC_MAX_TOOLS       = 16,
    GEIST_DYNAMIC_NAME_CAP        = 64,
    GEIST_DYNAMIC_DESCRIPTION_CAP = 512,
    GEIST_DYNAMIC_SCHEMA_CAP      = 4096,
    GEIST_DYNAMIC_DEFAULT_STEPS   = 4,
    GEIST_DYNAMIC_MAX_STEPS       = 16,
};

enum geist_dynamic_status {
    GEIST_DYNAMIC_OK = 0,
    GEIST_DYNAMIC_E_INVALID_ARGUMENT,
    GEIST_DYNAMIC_E_TOOL_LIMIT,
    GEIST_DYNAMIC_E_STEP_LIMIT,
    GEIST_DYNAMIC_E_INVALID_NAME,
    GEIST_DYNAMIC_E_DUPLICATE_NAME,
    GEIST_DYNAMIC_E_DESCRIPTION_LIMIT,
    GEIST_DYNAMIC_E_SCHEMA_LIMIT,
    GEIST_DYNAMIC_E_SCHEMA,
};

struct geist_dynamic_tool_spec {
    const char *name;
    const char *description;
    const char *parameters;
};

struct geist_dynamic_tool {
    char name[GEIST_DYNAMIC_NAME_CAP];
    char description[GEIST_DYNAMIC_DESCRIPTION_CAP];
    char parameters[GEIST_DYNAMIC_SCHEMA_CAP];
};

struct geist_dynamic_toolset {
    struct geist_dynamic_tool tools[GEIST_DYNAMIC_MAX_TOOLS];
    size_t                    count;
    size_t                    max_steps;
};

static inline const char *geist_dynamic_status_string(enum geist_dynamic_status status) {
    switch (status) {
    case GEIST_DYNAMIC_OK:
        return "ok";
    case GEIST_DYNAMIC_E_INVALID_ARGUMENT:
        return "invalid_argument";
    case GEIST_DYNAMIC_E_TOOL_LIMIT:
        return "tool_limit";
    case GEIST_DYNAMIC_E_STEP_LIMIT:
        return "step_limit";
    case GEIST_DYNAMIC_E_INVALID_NAME:
        return "invalid_name";
    case GEIST_DYNAMIC_E_DUPLICATE_NAME:
        return "duplicate_name";
    case GEIST_DYNAMIC_E_DESCRIPTION_LIMIT:
        return "description_limit";
    case GEIST_DYNAMIC_E_SCHEMA_LIMIT:
        return "schema_limit";
    case GEIST_DYNAMIC_E_SCHEMA:
        return "invalid_schema";
    }
    return "unknown_error";
}

static inline int geist_dynamic_name_valid(const char *name) {
    if (name == NULL || name[0] == '\0') {
        return 0;
    }
    for (size_t i = 0u; name[i] != '\0'; i++) {
        unsigned char c = (unsigned char) name[i];
        if (i >= GEIST_DYNAMIC_NAME_CAP - 1u ||
            !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (i > 0u && c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.')) {
            return 0;
        }
    }
    return 1;
}

static inline enum geist_dynamic_status geist_dynamic_schema_check(const char *schema_json,
                                                                   size_t      schema_len) {
    if (schema_json == NULL || schema_len == 0u || schema_len >= GEIST_DYNAMIC_SCHEMA_CAP) {
        return schema_len >= GEIST_DYNAMIC_SCHEMA_CAP ? GEIST_DYNAMIC_E_SCHEMA_LIMIT
                                                      : GEIST_DYNAMIC_E_SCHEMA;
    }
    struct jsv1_token tokens[JSV1_MAX_TOKENS];
    struct jsv1_doc   doc;
    enum jsv1_status  status = jsv1_parse(schema_json, schema_len, JSV1_MAX_TOKENS, tokens, &doc);
    if (status != JSV1_OK || jsv1_schema_check(&doc, 0, 0u) != JSV1_OK) {
        return GEIST_DYNAMIC_E_SCHEMA;
    }
    int type = jsv1_object_get(&doc, 0, "type");
    return jsv1_token_is(&doc, type, "object") ? GEIST_DYNAMIC_OK : GEIST_DYNAMIC_E_SCHEMA;
}

/* Compile once at the request boundary. `out` owns all copied bytes; changing
 * the caller's specs afterwards cannot widen the request's capability set. */
static inline enum geist_dynamic_status
geist_dynamic_toolset_compile(const struct geist_dynamic_tool_spec *specs,
                              size_t                                count,
                              size_t                                max_steps,
                              struct geist_dynamic_toolset         *out) {
    if (out == NULL || (count > 0u && specs == NULL)) {
        return GEIST_DYNAMIC_E_INVALID_ARGUMENT;
    }
    *out = (struct geist_dynamic_toolset) {0};
    if (count > GEIST_DYNAMIC_MAX_TOOLS) {
        return GEIST_DYNAMIC_E_TOOL_LIMIT;
    }
    if (max_steps == 0u) {
        max_steps = GEIST_DYNAMIC_DEFAULT_STEPS;
    }
    if (max_steps > GEIST_DYNAMIC_MAX_STEPS) {
        return GEIST_DYNAMIC_E_STEP_LIMIT;
    }
    for (size_t i = 0u; i < count; i++) {
        const struct geist_dynamic_tool_spec *spec = &specs[i];
        if (!geist_dynamic_name_valid(spec->name) || spec->description == NULL ||
            spec->parameters == NULL) {
            return spec->name == NULL || !geist_dynamic_name_valid(spec->name)
                           ? GEIST_DYNAMIC_E_INVALID_NAME
                           : GEIST_DYNAMIC_E_INVALID_ARGUMENT;
        }
        for (size_t prior = 0u; prior < i; prior++) {
            if (strcmp(specs[prior].name, spec->name) == 0) {
                return GEIST_DYNAMIC_E_DUPLICATE_NAME;
            }
        }
        size_t description_len = strlen(spec->description);
        size_t schema_len      = strlen(spec->parameters);
        if (description_len == 0u || description_len >= GEIST_DYNAMIC_DESCRIPTION_CAP) {
            return GEIST_DYNAMIC_E_DESCRIPTION_LIMIT;
        }
        enum geist_dynamic_status schema_status =
                geist_dynamic_schema_check(spec->parameters, schema_len);
        if (schema_status != GEIST_DYNAMIC_OK) {
            return schema_status;
        }
        memcpy(out->tools[i].name, spec->name, strlen(spec->name) + 1u);
        memcpy(out->tools[i].description, spec->description, description_len + 1u);
        memcpy(out->tools[i].parameters, spec->parameters, schema_len + 1u);
    }
    out->count     = count;
    out->max_steps = max_steps;
    return GEIST_DYNAMIC_OK;
}

static inline const struct geist_dynamic_tool *
geist_dynamic_tool_find(const struct geist_dynamic_toolset *set, const char *name) {
    if (set == NULL || name == NULL) {
        return NULL;
    }
    for (size_t i = 0u; i < set->count; i++) {
        if (strcmp(set->tools[i].name, name) == 0) {
            return &set->tools[i];
        }
    }
    return NULL;
}

/* Temporary prompt/forced-call compatibility projection. Validation always uses
 * `parameters`; this compact view only teaches the existing decoder key names. */
static inline size_t geist_dynamic_tool_prompt_schema(const struct geist_dynamic_tool *tool,
                                                      size_t                           cap,
                                                      char out[static cap]) {
    if (tool == NULL || cap < 3u)
        return 0u;
    struct jsv1_token tokens[JSV1_MAX_TOKENS];
    struct jsv1_doc   doc;
    if (jsv1_parse(tool->parameters, strlen(tool->parameters), JSV1_MAX_TOKENS, tokens, &doc) !=
        JSV1_OK)
        return 0u;
    int    properties = jsv1_object_get(&doc, 0, "properties");
    size_t written    = 0u;
    out[written++]    = '{';
    if (properties >= 0) {
        int cursor = properties;
        for (unsigned i = 0u; i < doc.tokens[properties].size; i++) {
            int    name     = jsv1_child(&doc, properties, cursor);
            int    schema   = jsv1_child(&doc, properties, name);
            int    type     = jsv1_object_get(&doc, schema, "type");
            size_t name_len = doc.tokens[name].end - doc.tokens[name].start;
            size_t type_len = type >= 0 ? doc.tokens[type].end - doc.tokens[type].start : 0u;
            size_t need     = name_len + type_len + (i > 0u ? 3u : 2u) + 1u;
            if (written + need >= cap || doc.tokens[name].escaped ||
                doc.tokens[type].type != JSV1_STRING || doc.tokens[type].escaped)
                return 0u;
            if (i > 0u)
                out[written++] = ',';
            out[written++] = '"';
            memcpy(out + written, doc.json + doc.tokens[name].start, name_len);
            written += name_len;
            out[written++] = '"';
            out[written++] = ':';
            memcpy(out + written, doc.json + doc.tokens[type].start, type_len);
            written += type_len;
            cursor = schema;
        }
    }
    out[written++] = '}';
    out[written]   = '\0';
    return written;
}

static inline enum jsv1_status geist_dynamic_tool_validate(const struct geist_dynamic_toolset *set,
                                                           const char                         *name,
                                                           const char *arguments,
                                                           size_t      arguments_len) {
    const struct geist_dynamic_tool *tool = geist_dynamic_tool_find(set, name);
    if (tool == NULL) {
        return JSV1_E_INVALID_SCHEMA; /* off-list: fail closed, never remap */
    }
    return jsv1_validate(tool->parameters, strlen(tool->parameters), arguments, arguments_len);
}

#endif /* GEIST_DYNAMIC_TOOLS_V1_H */

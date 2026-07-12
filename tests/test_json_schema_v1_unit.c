/* test_json_schema_v1_unit — host-neutral dynamic tool argument validation. */
#include "../tools/json_schema_v1.h"

#include <stdio.h>
#include <string.h>

static int
expect(const char *label, const char *schema, const char *value, enum jsv1_status expected) {
    enum jsv1_status actual = jsv1_validate(schema, strlen(schema), value, strlen(value));
    if (actual == expected) {
        return 0;
    }
    fprintf(stderr,
            "%s: expected %s, got %s\n",
            label,
            jsv1_status_string(expected),
            jsv1_status_string(actual));
    return 1;
}

int main(void) {
    int         failures = 0;
    const char *turn_on  = "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"}},"
                           "\"required\":[\"name\"],\"additionalProperties\":false}";
    const char *multi    = "{\"type\":\"object\",\"properties\":{"
                           "\"room\":{\"type\":\"string\",\"enum\":[\"kitchen\",\"office\"]},"
                           "\"level\":{\"type\":\"integer\",\"minimum\":0,\"maximum\":100},"
                           "\"transition\":{\"type\":\"number\",\"minimum\":0},"
                           "\"announce\":{\"type\":\"boolean\"}},"
                           "\"required\":[\"room\",\"level\"]}";
    const char *array    = "{\"type\":\"object\",\"properties\":{\"modes\":{\"type\":\"array\","
                           "\"items\":{\"type\":\"string\",\"enum\":[\"eco\",\"boost\"]},"
                           "\"minItems\":1,\"maxItems\":2}},\"required\":[\"modes\"]}";
    const char *nested   = "{\"type\":\"object\",\"properties\":{\"target\":{\"type\":\"object\","
                           "\"properties\":{\"id\":{\"type\":\"string\"}},\"required\":[\"id\"]}},"
                           "\"required\":[\"target\"]}";

    failures += expect("basic", turn_on, "{\"name\":\"kitchen } light\"}", JSV1_OK);
    failures += expect("required", turn_on, "{}", JSV1_E_REQUIRED);
    failures += expect("extra",
                       turn_on,
                       "{\"name\":\"light\",\"area\":\"kitchen\"}",
                       JSV1_E_ADDITIONAL_PROPERTY);
    failures += expect("wrong type", turn_on, "{\"name\":7}", JSV1_E_TYPE);
    failures += expect("multi",
                       multi,
                       "{\"room\":\"office\",\"level\":42,\"transition\":0.5,"
                       "\"announce\":false}",
                       JSV1_OK);
    failures += expect("optional", multi, "{\"room\":\"kitchen\",\"level\":0}", JSV1_OK);
    failures += expect("enum", multi, "{\"room\":\"garage\",\"level\":1}", JSV1_E_ENUM);
    failures += expect("integer", multi, "{\"room\":\"office\",\"level\":1.0}", JSV1_E_TYPE);
    failures += expect("range", multi, "{\"room\":\"office\",\"level\":101}", JSV1_E_RANGE);
    failures += expect("array", array, "{\"modes\":[\"eco\",\"boost\"]}", JSV1_OK);
    failures += expect("array enum", array, "{\"modes\":[\"turbo\"]}", JSV1_E_ENUM);
    failures += expect("array size", array, "{\"modes\":[]}", JSV1_E_ITEM_LIMIT);
    failures += expect("nested", nested, "{\"target\":{\"id\":\"lamp-1\"}}", JSV1_OK);
    failures += expect("nested extra",
                       nested,
                       "{\"target\":{\"id\":\"lamp-1\",\"unsafe\":true}}",
                       JSV1_E_ADDITIONAL_PROPERTY);
    failures += expect("duplicate value key",
                       turn_on,
                       "{\"name\":\"one\",\"name\":\"two\"}",
                       JSV1_E_INVALID_JSON);
    failures += expect("duplicate schema key",
                       "{\"type\":\"object\",\"type\":\"object\"}",
                       "{}",
                       JSV1_E_INVALID_SCHEMA);
    failures += expect("unsupported",
                       "{\"type\":\"string\",\"pattern\":\".*\"}",
                       "\"x\"",
                       JSV1_E_UNSUPPORTED_KEYWORD);
    failures += expect("invalid unused property",
                       "{\"type\":\"object\",\"properties\":{\"optional\":{"
                       "\"type\":\"string\",\"pattern\":\".*\"}}}",
                       "{}",
                       JSV1_E_UNSUPPORTED_KEYWORD);
    failures += expect("required not declared",
                       "{\"type\":\"object\",\"properties\":{},\"required\":[\"x\"]}",
                       "{}",
                       JSV1_E_INVALID_SCHEMA);
    failures += expect("mixed enum",
                       "{\"type\":\"string\",\"enum\":[\"x\",1]}",
                       "\"x\"",
                       JSV1_E_INVALID_SCHEMA);
    failures += expect(
            "bad bounds", "{\"type\":\"number\",\"minimum\":\"zero\"}", "1", JSV1_E_INVALID_SCHEMA);
    failures += expect("malformed value", turn_on, "{\"name\":", JSV1_E_INVALID_JSON);
    failures += expect("no coercion", turn_on, "{\"name\":true}", JSV1_E_TYPE);

    if (failures != 0) {
        fprintf(stderr, "test_json_schema_v1_unit: %d failure(s)\n", failures);
        return 1;
    }
    puts("test_json_schema_v1_unit: pass");
    return 0;
}

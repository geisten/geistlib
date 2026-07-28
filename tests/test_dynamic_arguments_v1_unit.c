#include "../tools/dynamic_arguments_v1.h"

#include <stdio.h>
#include <string.h>

static int build(const char *label, const char *schema, const char *text, const char *expected) {
    char   out[1024];
    size_t length = geist_dynamic_arguments_build(
            schema, strlen(schema), strlen(text), text, sizeof out, out);
    if ((expected == NULL && length == 0u) ||
        (expected != NULL && length > 0u && strcmp(out, expected) == 0))
        return 0;
    fprintf(stderr,
            "%s: got '%s' (%zu), expected '%s'\n",
            label,
            length > 0u ? out : "<clarify>",
            length,
            expected != NULL ? expected : "<clarify>");
    return 1;
}

int main(void) {
    int         failures = 0;
    const char *multi    = "{\"type\":\"object\",\"properties\":{"
                           "\"name\":{\"type\":\"string\"},\"level\":{\"type\":\"integer\","
                           "\"minimum\":0,\"maximum\":100},\"transition\":{\"type\":\"number\","
                           "\"minimum\":0},\"announce\":{\"type\":\"boolean\"}},"
                           "\"required\":[\"name\",\"level\"]}";
    failures += build("multi typed",
                      multi,
                      "Set kitchen to 42 transition 1.5 announce yes",
                      "{\"name\":\"Set kitchen to 42 transition 1.5 announce yes\","
                      "\"level\":42,\"transition\":1.5,\"announce\":true}");
    const char *enum_array = "{\"type\":\"object\",\"properties\":{"
                             "\"mode\":{\"type\":\"string\",\"enum\":[\"eco\",\"boost\"]},"
                             "\"rooms\":{\"type\":\"array\",\"items\":{\"type\":\"string\","
                             "\"enum\":[\"kitchen\",\"office\",\"garage\"]},\"minItems\":1}},"
                             "\"required\":[\"mode\",\"rooms\"]}";
    failures += build("enum array",
                      enum_array,
                      "Use boost in office and kitchen",
                      "{\"mode\":\"boost\",\"rooms\":[\"kitchen\",\"office\"]}");
    const char *optional = "{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"},"
                           "\"limit\":{\"type\":\"integer\"}},\"required\":[\"query\"]}";
    failures += build("optional omitted",
                      optional,
                      "find local inference",
                      "{\"query\":\"find local inference\"}");
    failures += build("required enum unclear", enum_array, "Do it somewhere", NULL);

    /* Identifier-shaped enum values. A request says "the kitchen light"; the
     * permitted value is written "kitchen_light" or "light.kitchen", so a
     * literal substring test never fires and the argument could not be filled
     * at all — see geistlib#135. Matching is per alphanumeric run instead. */
    const char *ids = "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\","
                      "\"enum\":[\"kitchen_light\",\"hallway_light\",\"desk_lamp\"]}},"
                      "\"required\":[\"name\"]}";
    failures += build("underscore id", ids, "Turn on the kitchen light",
                      "{\"name\":\"kitchen_light\"}");
    failures += build("underscore id de", ids, "Schalte desk lamp ein",
                      "{\"name\":\"desk_lamp\"}");
    const char *dotted = "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\","
                         "\"enum\":[\"light.kitchen\",\"light.hallway\"]}},"
                         "\"required\":[\"name\"]}";
    failures += build("dotted entity id", dotted, "Is the kitchen light on?",
                      "{\"name\":\"light.kitchen\"}");
    /* Fail closed rather than pick a device: "the light" fits two permitted
     * values equally well, and acting on a guess is the failure mode the
     * clarification path exists for. */
    failures += build("ambiguous scalar enum", ids, "Turn on the light", NULL);
    failures += build("no enum value mentioned", ids, "What is the weather tomorrow?", NULL);
    const char *boolean =
            "{\"type\":\"object\",\"properties\":{\"enabled\":{\"type\":\"boolean\"}},"
            "\"required\":[\"enabled\"]}";
    failures += build("required boolean unclear", boolean, "change the setting", NULL);
    if (failures != 0)
        return 1;
    puts("test_dynamic_arguments_v1_unit: pass");
    return 0;
}

/* test_dynamic_tools_v1_unit — immutable offered capability set. */
#include "../tools/dynamic_tools_v1.h"

#include <stdio.h>
#include <string.h>

static int check(int condition, const char *message) {
    if (condition) {
        return 0;
    }
    fprintf(stderr, "%s\n", message);
    return 1;
}

int main(void) {
    int  failures         = 0;
    char mutable_name[]   = "SetLevel";
    char mutable_schema[] = "{\"type\":\"object\",\"properties\":{\"level\":{\"type\":\"integer\","
                            "\"minimum\":0,\"maximum\":100}},\"required\":[\"level\"]}";
    struct geist_dynamic_tool_spec specs[] = {
            {.name        = mutable_name,
             .description = "Set a generic device level",
             .parameters  = mutable_schema},
            {.name        = "Search.docs-v1",
             .description = "Search a standalone documentation index",
             .parameters  = "{\"type\":\"object\",\"properties\":{\"query\":{\"type\":"
                            "\"string\"}},\"required\":[\"query\"]}"},
    };
    struct geist_dynamic_toolset set;
    failures += check(geist_dynamic_toolset_compile(specs, 2u, 5u, &set) == GEIST_DYNAMIC_OK,
                      "valid toolset rejected");
    failures += check(set.count == 2u && set.max_steps == 5u, "bounds not retained");
    mutable_name[0]   = 'X';
    mutable_schema[0] = '[';
    failures += check(geist_dynamic_tool_find(&set, "SetLevel") != NULL,
                      "toolset borrowed mutable name");
    char prompt_schema[128];
    failures += check(geist_dynamic_tool_prompt_schema(
                              &set.tools[0], sizeof prompt_schema, prompt_schema) > 0u &&
                              strcmp(prompt_schema, "{\"level\":integer}") == 0,
                      "typed schema was not projected into the decoder prompt");
    failures +=
            check(geist_dynamic_tool_validate(&set, "SetLevel", "{\"level\":42}", 12u) == JSV1_OK,
                  "valid arguments rejected");
    failures += check(geist_dynamic_tool_validate(&set, "SetLevel", "{\"level\":101}", 13u) ==
                              JSV1_E_RANGE,
                      "range not enforced");
    failures += check(geist_dynamic_tool_validate(&set, "DeleteEverything", "{}", 2u) != JSV1_OK,
                      "off-list tool accepted");

    struct geist_dynamic_tool_spec duplicate[] = {
            {.name = "same", .description = "one", .parameters = specs[1].parameters},
            {.name = "same", .description = "two", .parameters = specs[1].parameters},
    };
    failures += check(geist_dynamic_toolset_compile(duplicate, 2u, 1u, &set) ==
                              GEIST_DYNAMIC_E_DUPLICATE_NAME,
                      "duplicate names accepted");
    struct geist_dynamic_tool_spec invalid_schema = {
            .name = "Bad", .description = "bad", .parameters = "{\"type\":\"string\"}"};
    failures += check(geist_dynamic_toolset_compile(&invalid_schema, 1u, 1u, &set) ==
                              GEIST_DYNAMIC_E_SCHEMA,
                      "non-object parameter root accepted");
    failures +=
            check(geist_dynamic_toolset_compile(specs, 2u, 17u, &set) == GEIST_DYNAMIC_E_STEP_LIMIT,
                  "unbounded steps accepted");

    if (failures != 0) {
        fprintf(stderr, "test_dynamic_tools_v1_unit: %d failure(s)\n", failures);
        return 1;
    }
    puts("test_dynamic_tools_v1_unit: pass");
    return 0;
}

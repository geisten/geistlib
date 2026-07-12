#include "../tools/dynamic_request_v1.h"

#include <stdio.h>
#include <string.h>

static int expect(const char *label, const char *json, enum geist_dynamic_request_status wanted) {
    static struct geist_dynamic_request request;
    enum geist_dynamic_status           detail;
    enum geist_dynamic_request_status   got =
            geist_dynamic_request_parse(json, strlen(json), &request, &detail);
    if (got == wanted)
        return 0;
    fprintf(stderr,
            "%s: got %d detail=%s, wanted %d\n",
            label,
            got,
            geist_dynamic_status_string(detail),
            wanted);
    return 1;
}

int main(void) {
    int         failures = 0;
    const char *valid =
            "{\"input\":\"Schalte K\\u00fcche an \\ud83d\\udca1\",\"max_tool_steps\":3,"
            "\"tools\":[{\"name\":\"HassTurnOn\",\"description\":\"Turn on an exposed entity\","
            "\"parameters\":{\"type\":\"object\",\"properties\":{\"name\":{\"type\":"
            "\"string\"}},\"required\":[\"name\"]}}]}";
    static struct geist_dynamic_request request;
    enum geist_dynamic_status           detail;
    failures += expect("valid", valid, GEIST_DYNAMIC_REQUEST_OK);
    failures += geist_dynamic_request_parse(valid, strlen(valid), &request, &detail) !=
                GEIST_DYNAMIC_REQUEST_OK;
    failures += strcmp(request.input, "Schalte Küche an 💡") != 0;
    failures += request.toolset.count != 1u || request.toolset.max_steps != 3u;
    failures +=
            geist_dynamic_tool_validate(
                    &request.toolset, "HassTurnOn", "{\"name\":\"light.kitchen\"}", 24u) != JSV1_OK;
    failures += expect("unknown root",
                       "{\"input\":\"x\",\"tools\":[],\"host\":\"ha\"}",
                       GEIST_DYNAMIC_REQUEST_E_INVALID_REQUEST);
    failures += expect("duplicate",
                       "{\"input\":\"x\",\"input\":\"y\",\"tools\":[]}",
                       GEIST_DYNAMIC_REQUEST_E_INVALID_REQUEST);
    failures += expect("bad budget",
                       "{\"input\":\"x\",\"max_tool_steps\":17,\"tools\":[]}",
                       GEIST_DYNAMIC_REQUEST_E_INVALID_REQUEST);
    failures += expect("duplicate names",
                       "{\"input\":\"x\",\"tools\":[{\"name\":\"A\",\"description\":\"a\","
                       "\"parameters\":{\"type\":\"object\"}},{\"name\":\"A\","
                       "\"description\":\"b\",\"parameters\":{\"type\":\"object\"}}]}",
                       GEIST_DYNAMIC_REQUEST_E_TOOLSET);
    failures += expect("malformed", "{\"input\":", GEIST_DYNAMIC_REQUEST_E_INVALID_JSON);
    if (failures != 0) {
        fprintf(stderr, "test_dynamic_request_v1_unit: %d failure(s)\n", failures);
        return 1;
    }
    puts("test_dynamic_request_v1_unit: pass");
    return 0;
}

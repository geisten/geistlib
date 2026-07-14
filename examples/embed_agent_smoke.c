/*
 * embed_agent_smoke.c — proves the SDK ships a usable tool-use interface.
 *
 * Links against the packaged libgeist SDK and includes agent.h (the tool-use
 * interface). Constructs a trivial geist_tool and drives its invoke directly —
 * model-free, so it needs no GGUF — exercising agent_json_str + the geist_tool
 * contract from the shipped headers. The release workflow compiles this against
 * the packaged libgeist-<platform>.tar.gz. For a full agent run see geistwissen.
 */
#include <geist.h>
#include "agent.h" /* tool-use interface: geist_tool, agent_json_str, … */

#include <stdio.h>
#include <string.h>

static enum geist_status echo_invoke(void      *ctx,
                                     size_t     args_len,
                                     const char args[static args_len],
                                     size_t     out_cap,
                                     char       out[static out_cap],
                                     size_t    *out_len) {
    (void) ctx;
    (void) args_len;
    char query[128];
    query[0] = '\0';
    agent_json_str(args, "query", sizeof query, query);
    int w = snprintf(out, out_cap, "echo: %s", query);
    if (out_len) {
        *out_len = (size_t) (w > 0 ? w : 0);
    }
    return GEIST_OK;
}

int main(void) {
    struct geist_tool t = {
            .name        = "echo",
            .description = "echo the query back",
            .args_schema = "{\"query\": string}",
            .invoke      = echo_invoke,
    };
    const char *req = "{\"query\":\"hello\"}";
    char        out[64];
    size_t      n = 0;
    t.invoke(NULL, strlen(req), req, sizeof out, out, &n);
    printf("agent interface OK — %.*s\n", (int) n, out);
    return 0;
}

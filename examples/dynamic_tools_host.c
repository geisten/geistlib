/* Standalone non-HA dynamic-tools-v1 host: calculator + profile selector. */
#include "tools/dynamic_tools_v1.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

static int write_all(int fd, const char *text, size_t len) {
    for (size_t off = 0u; off < len;) {
        ssize_t n = write(fd, text + off, len - off);
        if (n > 0)
            off += (size_t) n;
        else if (n < 0 && errno == EINTR)
            continue;
        else
            return 0;
    }
    return 1;
}

static int read_line(int fd, size_t cap, char out[static cap]) {
    size_t used = 0u;
    while (used + 1u < cap) {
        ssize_t n = read(fd, out + used, 1u);
        if (n == 1 && out[used] == '\n') {
            out[used] = '\0';
            return 1;
        }
        if (n == 1)
            used++;
        else if (n < 0 && errno == EINTR)
            continue;
        else
            return 0;
    }
    return 0;
}

static size_t escape_input(const char *input, size_t cap, char out[static cap]) {
    size_t used = 0u;
    for (size_t i = 0u; input[i] != '\0'; i++) {
        if (input[i] == '"' || input[i] == '\\') {
            if (used + 2u >= cap)
                return 0u;
            out[used++] = '\\';
            out[used++] = input[i];
        } else if (input[i] == '\n') {
            if (used + 2u >= cap)
                return 0u;
            out[used++] = '\\';
            out[used++] = 'n';
        } else {
            if (used + 1u >= cap)
                return 0u;
            out[used++] = input[i];
        }
    }
    out[used] = '\0';
    return used;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <geist.sock> <request>\n", argv[0]);
        return 2;
    }
    static const struct geist_dynamic_tool_spec specs[] = {
            {.name        = "CalculatorAdd",
             .description = "Add two integer values",
             .parameters  = "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":"
                            "\"integer\"},\"b\":{\"type\":\"integer\"}},\"required\":["
                            "\"a\",\"b\"]}"},
            {.name        = "SelectProfile",
             .description = "Select a local runtime profile",
             .parameters  = "{\"type\":\"object\",\"properties\":{\"profile\":{\"type\":"
                            "\"string\",\"enum\":[\"quiet\",\"balanced\",\"fast\"]}},"
                            "\"required\":[\"profile\"]}"},
    };
    static struct geist_dynamic_toolset offered;
    if (geist_dynamic_toolset_compile(specs, 2u, 4u, &offered) != GEIST_DYNAMIC_OK)
        return 2;

    int                fd      = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un address = {.sun_family = AF_UNIX};
    if (fd < 0 || strlen(argv[1]) >= sizeof address.sun_path)
        return 2;
    strcpy(address.sun_path, argv[1]);
    if (connect(fd, (struct sockaddr *) &address, sizeof address) != 0) {
        perror("connect");
        return 1;
    }
    char input[4096];
    if (escape_input(argv[2], sizeof input, input) == 0u)
        return 2;
    char request[12288];
    int  request_len = snprintf(request,
                                sizeof request,
                                "{\"input\":\"%s\",\"max_tool_steps\":4,\"tools\":["
                                "{\"name\":\"%s\",\"description\":\"%s\",\"parameters\":%s},"
                                "{\"name\":\"%s\",\"description\":\"%s\",\"parameters\":%s}]}\n",
                                input,
                                specs[0].name,
                                specs[0].description,
                                specs[0].parameters,
                                specs[1].name,
                                specs[1].description,
                                specs[1].parameters);
    if (request_len <= 0 || (size_t) request_len >= sizeof request ||
        !write_all(fd, request, (size_t) request_len))
        return 1;

    char line[8192];
    while (read_line(fd, sizeof line, line)) {
        struct jsv1_token tokens[JSV1_MAX_TOKENS];
        struct jsv1_doc   doc;
        if (jsv1_parse(line, strlen(line), JSV1_MAX_TOKENS, tokens, &doc) != JSV1_OK ||
            doc.tokens[0].type != JSV1_OBJECT || !jsv1_object_unique(&doc, 0))
            return 1;
        int type = jsv1_object_get(&doc, 0, "type");
        if (jsv1_token_is(&doc, type, "conversation.result")) {
            int text = jsv1_object_get(&doc, 0, "text");
            if (text < 0)
                return 1;
            printf("%.*s\n",
                   (int) (doc.tokens[text].end - doc.tokens[text].start),
                   doc.json + doc.tokens[text].start);
            close(fd);
            return 0;
        }
        if (!jsv1_token_is(&doc, type, "tool.call"))
            return 1;
        int id        = jsv1_object_get(&doc, 0, "call_id");
        int name      = jsv1_object_get(&doc, 0, "name");
        int arguments = jsv1_object_get(&doc, 0, "arguments");
        if (id < 0 || name < 0 || arguments < 0 || doc.tokens[name].escaped)
            return 1;
        char   tool_name[GEIST_DYNAMIC_NAME_CAP];
        size_t name_len = doc.tokens[name].end - doc.tokens[name].start;
        if (name_len >= sizeof tool_name)
            return 1;
        memcpy(tool_name, doc.json + doc.tokens[name].start, name_len);
        tool_name[name_len]  = '\0';
        const char *args     = doc.json + doc.tokens[arguments].start;
        size_t      args_len = doc.tokens[arguments].end - doc.tokens[arguments].start;
        if (geist_dynamic_tool_validate(&offered, tool_name, args, args_len) != JSV1_OK)
            return 1;

        char result[256];
        if (strcmp(tool_name, "CalculatorAdd") == 0) {
            int    a_token = jsv1_object_get(&doc, arguments, "a");
            int    b_token = jsv1_object_get(&doc, arguments, "b");
            double a, b;
            if (!jsv1_number(&doc, a_token, &a) || !jsv1_number(&doc, b_token, &b))
                return 1;
            snprintf(result, sizeof result, "{\"sum\":%.0f}", a + b);
        } else if (strcmp(tool_name, "SelectProfile") == 0) {
            int profile = jsv1_object_get(&doc, arguments, "profile");
            snprintf(result,
                     sizeof result,
                     "{\"selected\":\"%.*s\"}",
                     (int) (doc.tokens[profile].end - doc.tokens[profile].start),
                     doc.json + doc.tokens[profile].start);
        } else {
            return 1; /* defense in depth: offered lookup above must already reject this */
        }
        char response[512];
        int  response_len = snprintf(response,
                                     sizeof response,
                                     "{\"type\":\"tool.result\",\"call_id\":\"%.*s\","
                                     "\"status\":\"ok\",\"result\":%s}\n",
                                     (int) (doc.tokens[id].end - doc.tokens[id].start),
                                     doc.json + doc.tokens[id].start,
                                     result);
        if (response_len <= 0 || (size_t) response_len >= sizeof response ||
            !write_all(fd, response, (size_t) response_len))
            return 1;
    }
    return 1;
}

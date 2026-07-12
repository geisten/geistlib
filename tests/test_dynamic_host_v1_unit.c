#include "../tools/dynamic_host_v1.h"

#include <stdio.h>
#include <string.h>
#include <sys/socket.h>

int main(void) {
    int pair[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, pair) != 0)
        return 2;
    struct geist_dynamic_host_session session = {
            .fd = pair[0], .next_call_id = 7u, .max_retries = 1u};
    struct geist_dynamic_host_tool context = {.session = &session, .name = "SetLevel"};
    const char                    *result  = "{\"type\":\"tool.result\",\"call_id\":\"7\","
                                             "\"result\":{\"ok\":true,\"level\":42}}\n";
    if (!geist_dynamic_host_write(pair[1], result, strlen(result)))
        return 2;
    char        out[256];
    size_t      out_len  = 0u;
    const char *args     = "{\"level\":42}";
    int         failures = geist_dynamic_host_invoke(
                                   &context, strlen(args), args, sizeof out, out, &out_len) != GEIST_OK;
    char        call[512];
    size_t      call_len = 0u;
    failures += !geist_dynamic_host_read_line(pair[1], sizeof call, call, &call_len);
    failures += strstr(call, "\"type\":\"tool.call\"") == NULL;
    failures += strstr(call, "\"name\":\"SetLevel\"") == NULL;
    failures += strstr(call, "\"arguments\":{\"level\":42}") == NULL;
    failures += strcmp(out, "{\"ok\":true,\"level\":42}") != 0;
    const char *retryable = "{\"type\":\"tool.result\",\"call_id\":\"8\","
                            "\"status\":\"retryable\",\"result\":{\"error\":\"busy\"}}\n";
    const char *retried   = "{\"type\":\"tool.result\",\"call_id\":\"9\","
                            "\"status\":\"ok\",\"result\":{\"ok\":true}}\n";
    failures += !geist_dynamic_host_write(pair[1], retryable, strlen(retryable));
    failures += !geist_dynamic_host_write(pair[1], retried, strlen(retried));
    failures += geist_dynamic_host_invoke(
                        &context, strlen(args), args, sizeof out, out, &out_len) != GEIST_OK;
    failures += strcmp(out, "{\"ok\":true}") != 0;
    failures += !geist_dynamic_host_read_line(pair[1], sizeof call, call, &call_len) ||
                strstr(call, "\"call_id\":\"8\"") == NULL;
    failures += !geist_dynamic_host_read_line(pair[1], sizeof call, call, &call_len) ||
                strstr(call, "\"call_id\":\"9\"") == NULL;

    const char *cancel = "{\"type\":\"cancel\",\"call_id\":\"10\"}\n";
    failures += !geist_dynamic_host_write(pair[1], cancel, strlen(cancel));
    failures +=
            geist_dynamic_host_invoke(&context, strlen(args), args, sizeof out, out, &out_len) !=
            GEIST_E_INVALID_STATE;
    failures += !session.cancelled || session.next_call_id != 11u;
    failures += !geist_dynamic_host_read_line(pair[1], sizeof call, call, &call_len) ||
                strstr(call, "\"call_id\":\"10\"") == NULL;
    unsigned before = session.next_call_id;
    failures +=
            geist_dynamic_host_invoke(&context, strlen(args), args, sizeof out, out, &out_len) !=
            GEIST_E_INVALID_STATE;
    failures += session.next_call_id != before;
    close(pair[0]);
    close(pair[1]);
    if (failures != 0) {
        fprintf(stderr, "test_dynamic_host_v1_unit: %d failure(s)\n", failures);
        return 1;
    }
    puts("test_dynamic_host_v1_unit: pass");
    return 0;
}

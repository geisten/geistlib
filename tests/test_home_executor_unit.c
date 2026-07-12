#include "test_helpers.h"

#include "tools/home_executor.h"

struct fake_executor_state {
    unsigned                     calls;
    enum home_executor_status    next_status;
    struct home_executor_request last;
};

static enum home_executor_status fake_execute(void                               *context,
                                              const struct home_executor_request *request,
                                              size_t                              out_cap,
                                              char                                out[],
                                              size_t                             *out_len) {
    struct fake_executor_state *state = (struct fake_executor_state *) context;
    state->calls++;
    state->last = *request;
    if (state->next_status != HOME_EXECUTOR_OK) {
        snprintf(out, out_cap, "must be discarded");
        *out_len = strlen(out);
        return state->next_status;
    }
    *out_len = (size_t) snprintf(out, out_cap, "{\"state\":\"off\"}");
    return HOME_EXECUTOR_OK;
}

int main(void) {
    int                        fails    = 0;
    struct fake_executor_state fake     = {0};
    const struct home_executor executor = {.execute = fake_execute, .context = &fake};
    char                       out[128];
    size_t                     out_len = 0u;

    const struct home_executor_request get = {
            .operation = HOME_EXECUTOR_GET_STATE,
            .entity_id = "light.flur",
            .domain    = "light",
    };
    enum home_executor_status status =
            home_executor_run(&executor, &get, sizeof out, out, &out_len);
    fails += geist_expect(status == HOME_EXECUTOR_OK, "fake state read succeeds");
    fails += geist_expect(fake.calls == 1u && fake.last.operation == HOME_EXECUTOR_GET_STATE,
                          "typed state request reaches fake");
    fails += geist_expect(strcmp(fake.last.entity_id, "light.flur") == 0 &&
                                  strcmp(fake.last.domain, "light") == 0,
                          "state request identity preserved");
    fails += geist_expect(out_len == strlen(out) && strstr(out, "off") != NULL,
                          "successful fake response preserved");

    const struct home_executor_request set = {
            .operation = HOME_EXECUTOR_CALL_SERVICE,
            .entity_id = "climate.bad",
            .domain    = "climate",
            .service   = "set_temperature",
            .arguments = "\"temperature\":22",
    };
    status = home_executor_run(&executor, &set, sizeof out, out, &out_len);
    fails += geist_expect(status == HOME_EXECUTOR_OK && fake.calls == 2u,
                          "typed service request reaches fake");
    fails += geist_expect(strcmp(fake.last.service, "set_temperature") == 0 &&
                                  strcmp(fake.last.arguments, "\"temperature\":22") == 0,
                          "service and arguments preserved");

    fake.next_status = HOME_EXECUTOR_DENIED;
    status           = home_executor_run(&executor, &set, sizeof out, out, &out_len);
    fails += geist_expect(status == HOME_EXECUTOR_DENIED, "executor denial preserved");
    fails += geist_expect(out_len == 0u && out[0] == '\0', "denied response body discarded");

    fake.next_status                                  = HOME_EXECUTOR_OK;
    const unsigned               calls_before_invalid = fake.calls;
    struct home_executor_request invalid              = set;
    invalid.entity_id                                 = "";
    snprintf(out, sizeof out, "stale");
    out_len = 5u;
    status  = home_executor_run(&executor, &invalid, sizeof out, out, &out_len);
    fails += geist_expect(status == HOME_EXECUTOR_INVALID_REQUEST,
                          "empty entity rejected before executor");
    fails += geist_expect(fake.calls == calls_before_invalid, "invalid request never reaches fake");
    fails += geist_expect(out_len == 0u && out[0] == '\0', "invalid request clears output");

    invalid           = get;
    invalid.service   = "turn_on";
    invalid.arguments = "";
    status            = home_executor_run(&executor, &invalid, sizeof out, out, &out_len);
    fails += geist_expect(status == HOME_EXECUTOR_INVALID_REQUEST,
                          "state read with service is rejected");

    invalid           = set;
    invalid.arguments = NULL;
    status            = home_executor_run(&executor, &invalid, sizeof out, out, &out_len);
    fails += geist_expect(status == HOME_EXECUTOR_INVALID_REQUEST,
                          "service call without arguments is rejected");

    fails += geist_expect(
            strcmp(home_executor_status_string(HOME_EXECUTOR_UNCONFIGURED), "unconfigured") == 0 &&
                    strcmp(home_executor_status_string(HOME_EXECUTOR_UNAVAILABLE), "unavailable") ==
                            0 &&
                    strcmp(home_executor_status_string(HOME_EXECUTOR_DENIED), "denied") == 0,
            "executor status strings are stable");

    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}

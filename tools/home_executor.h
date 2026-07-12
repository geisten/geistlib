/* tools/home_executor.h — transport-neutral execution boundary for the home
 * planner. The planner produces typed state reads and bounded service calls;
 * an executor decides how to perform them (legacy HA REST, protocol v2, or a
 * model-free fake).
 */
#ifndef GEIST_HOME_EXECUTOR_H
#define GEIST_HOME_EXECUTOR_H

#include <stddef.h>
#include <stdint.h>

enum home_executor_operation {
    HOME_EXECUTOR_GET_STATE = 0,
    HOME_EXECUTOR_CALL_SERVICE,
};

enum home_executor_status {
    HOME_EXECUTOR_OK = 0,
    HOME_EXECUTOR_UNCONFIGURED,
    HOME_EXECUTOR_UNAVAILABLE,
    HOME_EXECUTOR_DENIED,
    HOME_EXECUTOR_INVALID_REQUEST,
};

struct home_executor_request {
    enum home_executor_operation operation;
    const char                  *entity_id;
    const char                  *domain;
    const char                  *service;
    const char                  *arguments;
    uint64_t                     registry_version;
};

typedef enum home_executor_status (*home_executor_fn)(void                               *context,
                                                      const struct home_executor_request *request,
                                                      size_t                              out_cap,
                                                      char    out[static out_cap],
                                                      size_t *out_len);

struct home_executor {
    home_executor_fn execute;
    void            *context;
};

static inline const char *home_executor_status_string(enum home_executor_status status) {
    switch (status) {
    case HOME_EXECUTOR_OK:
        return "ok";
    case HOME_EXECUTOR_UNCONFIGURED:
        return "unconfigured";
    case HOME_EXECUTOR_UNAVAILABLE:
        return "unavailable";
    case HOME_EXECUTOR_DENIED:
        return "denied";
    case HOME_EXECUTOR_INVALID_REQUEST:
        return "invalid_request";
    }
    return "unknown_error";
}

static inline int home_executor_request_valid(const struct home_executor_request *request) {
    if (request == NULL || request->entity_id == NULL || request->entity_id[0] == '\0' ||
        request->domain == NULL || request->domain[0] == '\0') {
        return 0;
    }
    if (request->operation == HOME_EXECUTOR_GET_STATE) {
        return request->service == NULL && request->arguments == NULL;
    }
    if (request->operation == HOME_EXECUTOR_CALL_SERVICE) {
        return request->service != NULL && request->service[0] != '\0' &&
               request->arguments != NULL;
    }
    return 0;
}

static inline enum home_executor_status
home_executor_run(const struct home_executor         *executor,
                  const struct home_executor_request *request,
                  size_t                              out_cap,
                  char                                out[static out_cap],
                  size_t                             *out_len) {
    if (out_len != NULL) {
        *out_len = 0u;
    }
    if (out != NULL && out_cap > 0u) {
        out[0] = '\0';
    }
    if (executor == NULL || executor->execute == NULL || out == NULL || out_cap == 0u ||
        out_len == NULL || !home_executor_request_valid(request)) {
        return HOME_EXECUTOR_INVALID_REQUEST;
    }
    enum home_executor_status status =
            executor->execute(executor->context, request, out_cap, out, out_len);
    if (status != HOME_EXECUTOR_OK) {
        out[0]   = '\0';
        *out_len = 0u;
    } else if (*out_len >= out_cap) {
        out[0]   = '\0';
        *out_len = 0u;
        return HOME_EXECUTOR_INVALID_REQUEST;
    } else {
        out[*out_len] = '\0';
    }
    return status;
}

#endif /* GEIST_HOME_EXECUTOR_H */

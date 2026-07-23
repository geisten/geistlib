/*
 * src/engine/error.c — status strings + error-slot plumbing.
 *
 * Layer: ENGINE.
 *
 * Per Q27: handle-attached error messages are the primary detail channel;
 * a thread-local fallback exists only for create-time errors (when no
 * handle exists yet to attach to).
 */
#define GEIST_INTERNAL_ENGINE_LAYER

#include "error.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

/* ---------- Status -> string ---------- */

const char *geist_status_to_string(enum geist_status s) {
    switch (s) {
    case GEIST_OK:
        return "GEIST_OK";
    case GEIST_E_OOM:
        return "GEIST_E_OOM";
    case GEIST_E_INVALID_ARG:
        return "GEIST_E_INVALID_ARG";
    case GEIST_E_INTERNAL:
        return "GEIST_E_INTERNAL";
    case GEIST_E_FILE_NOT_FOUND:
        return "GEIST_E_FILE_NOT_FOUND";
    case GEIST_E_IO:
        return "GEIST_E_IO";
    case GEIST_E_FORMAT:
        return "GEIST_E_FORMAT";
    case GEIST_E_UNSUPPORTED:
        return "GEIST_E_UNSUPPORTED";
    case GEIST_E_NOT_FOUND:
        return "GEIST_E_NOT_FOUND";
    case GEIST_E_BACKEND:
        return "GEIST_E_BACKEND";
    case GEIST_E_INVALID_STATE:
        return "GEIST_E_INVALID_STATE";
    case GEIST_E_TOO_MANY_TOKENS:
        return "GEIST_E_TOO_MANY_TOKENS";
    }
    return "GEIST_E_UNKNOWN";
}

/* ---------- Thread-local create-time fallback ---------- */

static _Thread_local struct geist_error_slot t_create_error = {
        .code        = GEIST_OK,
        .message     = {0},
        .origin_func = nullptr,
};

void geist_error_set_create_time(enum geist_status code,
                                 const char       *origin_func,
                                 const char       *fmt,
                                 ...) {
    t_create_error.code        = code;
    t_create_error.origin_func = origin_func;

    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(t_create_error.message, GEIST_ERR_MSG_LEN, fmt, ap);
    va_end(ap);

    if (n < 0) {
        t_create_error.message[0] = '\0';
    }
}

const char *geist_last_create_error(void) {
    if (t_create_error.message[0] == '\0') {
        return "(no error)";
    }
    return t_create_error.message;
}

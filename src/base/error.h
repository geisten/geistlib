/*
 * src/engine/error.h — internal error-context plumbing.
 *
 * Layer: ENGINE.
 *
 * Defined in (Phase B-4):
 *   src/engine/error.c
 *
 * Pattern (per Q27):
 *   - Status code is the return value of every fallible API.
 *   - Detailed error message is attached to the relevant handle (backend,
 *     model, session) via geist_*_errmsg().
 *   - For create-time errors where no handle exists yet, errors land in
 *     a thread-local fallback retrievable via geist_last_create_error().
 */
#ifndef GEIST_INTERNAL_ERROR_H
#define GEIST_INTERNAL_ERROR_H

#ifndef GEIST_INTERNAL_ENGINE_LAYER
#error "error.h is internal to the engine layer."
#endif

#include <geist.h>

/* Fixed-size message buffer to keep allocation deterministic. */
#define GEIST_ERR_MSG_LEN 512

struct geist_error_slot {
    enum geist_status code;
    char              message[GEIST_ERR_MSG_LEN];
    const char       *origin_func;
};

/* Set the thread-local create-time error slot. Used when no handle
 * exists to attach the error to. */
void geist_error_set_create_time(enum geist_status code,
                                 const char       *origin_func,
                                 const char       *fmt,
                                 ...);

#endif /* GEIST_INTERNAL_ERROR_H */

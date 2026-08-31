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

#include <stdbool.h>

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

/* Empty the create-time slot. A multi-stage create (geist_model_load ->
 * decoder state_create -> ...) calls this on entry so that a generic
 * message at the top can tell "nothing below said anything" from "a lower
 * layer already named the cause". */
void geist_error_clear_create_time(void);

/* True iff the create-time slot currently holds a message. Lets an outer
 * frame keep a specific diagnosis instead of overwriting it with a guess:
 * `state_create` returns void*, so the outer frame cannot see WHY it
 * failed and would otherwise report GEIST_E_IO for every cause. */
[[nodiscard]] bool geist_have_create_error(void);

#endif /* GEIST_INTERNAL_ERROR_H */

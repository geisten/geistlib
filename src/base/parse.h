//
// Integer knobs from text (GEIST_* environment variables, mostly).
//
// atoi/atol are undefined behaviour on a value that does not fit the result
// type, and they turn "abc" into 0 without a word, which then reads as a
// deliberate setting. strtol reports both; this wraps the three checks it
// needs so every call site does not have to repeat them.
//
#pragma once

#include <errno.h>
#include <stdlib.h>

/* `s` as a whole decimal long. False, with *out untouched, when `s` is
 * nullptr, empty, has anything after the number, or does not fit a long.
 * Range checks beyond that stay with the caller. */
[[nodiscard]] static inline bool geist_parse_long(const char *s, long *out) {
    if (s == nullptr || s[0] == '\0') {
        return false;
    }
    char *end    = nullptr;
    errno        = 0;
    const long v = strtol(s, &end, 10);
    if (errno != 0 || *end != '\0') {
        return false;
    }
    *out = v;
    return true;
}

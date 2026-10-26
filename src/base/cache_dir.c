/*
 * src/base/cache_dir.c — see cache_dir.h.
 *
 * Layer: ENGINE.
 */
#include "cache_dir.h"

#include <stdio.h>
#include <stdlib.h>

bool geist_cache_dir(size_t n, const char *sub, char dir[static n]) {
    const char *xdg  = getenv("XDG_CACHE_HOME");
    const char *home = getenv("HOME");
    const char *sep  = sub != nullptr && sub[0] != '\0' ? "/" : "";
    sub              = sub != nullptr ? sub : "";
    const int w = xdg != nullptr && xdg[0] != '\0' ? snprintf(dir, n, "%s/geist%s%s", xdg, sep, sub)
                  : home != nullptr && home[0] != '\0'
                          ? snprintf(dir, n, "%s/.cache/geist%s%s", home, sep, sub)
                          : -1;
    return w > 0 && (size_t) w < n;
}

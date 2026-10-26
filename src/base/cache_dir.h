/*
 * src/base/cache_dir.h — geist's per-user cache directory.
 *
 * Layer: ENGINE.
 */
#ifndef GEIST_BASE_CACHE_DIR_H
#define GEIST_BASE_CACHE_DIR_H

#include <stddef.h>

/* $XDG_CACHE_HOME/geist, else $HOME/.cache/geist, followed by "/<sub>"
 * unless sub is nullptr or empty, into dir; false when neither variable
 * is set or the path does not fit in n bytes. Creates nothing. */
[[nodiscard]] bool geist_cache_dir(size_t n, const char *sub, char dir[static n]);

#endif /* GEIST_BASE_CACHE_DIR_H */

/*
 * src/io/file_map.h — a whole file mapped read-only, for the model and
 * tokenizer readers (GGUF, safetensors, tokenizer.bin).
 */
#pragma once

#include <stddef.h>

struct io_map {
    int    fd; /* left open: the readers re-map ranges of it and close it last */
    void  *map;
    size_t size;
};

/* Opens `path` and maps all of it PROT_READ with `flags` (MAP_PRIVATE or
 * MAP_SHARED). On failure closes what it opened, keeps errno from the
 * failing call, and sets *err (when err is non-null) to "open() failed",
 * "fstat() failed" or "mmap() failed". */
[[nodiscard]] bool io_map_ro(const char *path, int flags, struct io_map *out, const char **err);

/* *out = msg when out is non-null; returns msg. The readers' errmsg
 * out-parameter. */
const char *io_set_err(const char **out, const char *msg);

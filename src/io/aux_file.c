/*
 * src/io/aux_file.c - locate a model's auxiliary file (aux_file.h).
 */
#include "aux_file.h"

#include "heap.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool readable(const char *path) {
    FILE *f = fopen(path, "rb");
    if (f == nullptr) {
        return false;
    }
    fclose(f);
    return true;
}

char *geist_find_aux_file(const char        *env_name,
                          const char        *dir,
                          const char        *basename,
                          const char *const *fallbacks) {
    const char *env = env_name != nullptr ? getenv(env_name) : nullptr;
    if (env != nullptr && env[0] != '\0' && readable(env)) {
        return strdup(env);
    }
    if (dir != nullptr) {
        const size_t need = strlen(dir) + 1 + strlen(basename) + 1;
        char        *cand = heap_alloc_aligned(need, alignof(char));
        if (cand != nullptr) {
            snprintf(cand, need, "%s/%s", dir, basename);
            if (readable(cand)) {
                return cand;
            }
            safe_free((void **) &cand);
        }
    }
    for (size_t i = 0; fallbacks != nullptr && fallbacks[i] != nullptr; i++) {
        if (readable(fallbacks[i])) {
            return strdup(fallbacks[i]);
        }
    }
    return nullptr;
}

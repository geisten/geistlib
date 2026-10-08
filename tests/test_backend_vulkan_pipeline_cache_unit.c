/*
 * test_backend_vulkan_pipeline_cache_unit — the persisted pipeline cache
 * (#469).
 *
 * The NVIDIA driver keys its own shader cache by executable, so every new
 * binary paid ~2 s of pipeline compiles at backend creation; geist keeps a
 * VkPipelineCache file instead. Checked, with GEIST_VK_PIPELINE_CACHE naming
 * a file in /tmp:
 *   - the first backend writes the file;
 *   - a second backend reads it and leaves it as it was (nothing new);
 *   - a file of garbage does not stop a backend from being created;
 *   - GEIST_VK_PIPELINE_CACHE=0 writes nothing.
 * SKIPs without a Vulkan device.
 */
#define _POSIX_C_SOURCE 200809L

#include "test_helpers.h"

#include <geist.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static bool create_destroy(void) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create("vulkan", nullptr, nullptr, &be) != GEIST_OK || be == nullptr) {
        return false;
    }
    geist_backend_destroy(be);
    return true;
}

static long file_size(const char *path) {
    struct stat sb;
    return stat(path, &sb) == 0 ? (long) sb.st_size : -1;
}

int main(void) {
    char path[64];
    snprintf(path, sizeof path, "/tmp/geist_pcache_%ld.bin", (long) getpid());
    (void) remove(path);
    setenv("GEIST_VK_PIPELINE_CACHE", path, 1);
    struct geist_backend *probe = nullptr;
    if (geist_backend_create("vulkan", nullptr, nullptr, &probe) != GEIST_OK) {
        (void) remove(path);
        GEIST_SKIP("Vulkan backend unavailable");
    }
    geist_backend_destroy(probe);

    int        fails = 0;
    const long first = file_size(path);
    fails += geist_expect(first > 0, "the first backend writes the cache file");

    fails += geist_expect(create_destroy() && file_size(path) == first,
                          "a second backend reads it and leaves it unchanged");

    FILE *f = fopen(path, "wb");
    if (f != nullptr) {
        fputs("not a pipeline cache", f);
        fclose(f);
    }
    fails += geist_expect(create_destroy(), "a garbage cache file does not stop the backend");
    fails += geist_expect(file_size(path) > 64, "and is replaced by a real cache");

    (void) remove(path);
    setenv("GEIST_VK_PIPELINE_CACHE", "0", 1);
    fails += geist_expect(create_destroy() && file_size(path) == -1,
                          "GEIST_VK_PIPELINE_CACHE=0 writes nothing");
    unsetenv("GEIST_VK_PIPELINE_CACHE");

    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: Vulkan pipeline cache written, reused, garbage-tolerant, can be disabled\n");
    return GEIST_TEST_PASS;
}

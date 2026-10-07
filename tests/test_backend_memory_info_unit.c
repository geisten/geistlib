/*
 * test_backend_memory_info_unit — geist_backend_memory_info reports the
 * device memory a GPU backend allocates from (#665).
 *
 * A runtime sizes a model's context window from it, so the numbers have to
 * move with what is allocated. Checked here:
 *
 *   - null handles and outputs are GEIST_E_INVALID_ARG, a backend without a
 *     provider (cpu_scalar) is GEIST_E_UNSUPPORTED, and failure zeroes out;
 *   - Vulkan: total > 0 and free <= total; a 256 MiB device-local buffer
 *     takes roughly 256 MiB off free and its release gives it back (64 MiB
 *     tolerance: with VK_EXT_memory_budget the numbers are device-wide and
 *     another process may allocate meanwhile; on an integrated GPU, whose
 *     budget follows free system RAM, only the drop's sign is checked);
 *     GEIST_VK_VRAM_BUDGET lowers total.
 *
 * SKIPs when the Vulkan backend is not built or has no device.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <stdio.h>
#include <stdlib.h>

constexpr uint64_t MIB = (uint64_t) 1 << 20;

static int check_generic(void) {
    int                         fails = 0;
    struct geist_backend_memory m     = {.total_bytes = 1, .free_bytes = 1, .device_wide = true};
    fails += geist_expect(geist_backend_memory_info(nullptr, &m) == GEIST_E_INVALID_ARG &&
                                  m.total_bytes == 0 && m.free_bytes == 0 && !m.device_wide,
                          "a null backend is INVALID_ARG and zeroes out");
    fails += geist_expect(geist_backend_memory_info(nullptr, nullptr) == GEIST_E_INVALID_ARG,
                          "a null output is INVALID_ARG");

    struct geist_backend *cpu = nullptr;
    if (geist_backend_create("cpu_scalar", nullptr, nullptr, &cpu) != GEIST_OK) {
        return fails + geist_expect(false, "cpu_scalar backend");
    }
    m                         = (struct geist_backend_memory) {.total_bytes = 1};
    const enum geist_status s = geist_backend_memory_info(cpu, &m);
    fails += geist_expect(s == GEIST_E_UNSUPPORTED && m.total_bytes == 0,
                          "cpu_scalar is UNSUPPORTED and zeroes out");
    geist_backend_destroy(cpu);
    return fails;
}

static uint64_t gap(uint64_t a, uint64_t b) {
    return a > b ? a - b : b - a;
}

static int check_vulkan(struct geist_backend *be) {
    int                         fails = 0;
    char                        msg[256];
    struct geist_backend_memory before = {0}, held = {0}, after = {0};
    if (geist_backend_memory_info(be, &before) != GEIST_OK) {
        return geist_expect(false, "vulkan memory_info is OK");
    }
    printf("vulkan: total %llu MiB, free %llu MiB, device_wide %d, unified %d\n",
           (unsigned long long) (before.total_bytes / MIB),
           (unsigned long long) (before.free_bytes / MIB),
           (int) before.device_wide,
           (int) before.unified_memory);
    fails += geist_expect(before.total_bytes > 0, "total > 0");
    fails += geist_expect(before.free_bytes <= before.total_bytes, "free <= total");
    fails += geist_expect(before.free_bytes > 0, "free > 0 on an idle backend");

    constexpr uint64_t n = 256 * MIB;
    if (before.free_bytes < 2 * n) {
        printf("only %llu MiB free: allocation step skipped\n",
               (unsigned long long) (before.free_bytes / MIB));
        return fails;
    }
    const struct geist_backend_vtbl *vt  = be->desc->vtbl;
    struct geist_buffer             *buf = nullptr;
    if (vt->buffer_create(be, (size_t) n, GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE, &buf) !=
        GEIST_OK) {
        return fails + geist_expect(false, "256 MiB device buffer");
    }
    fails += geist_expect(geist_backend_memory_info(be, &held) == GEIST_OK, "OK while held");
    vt->buffer_destroy(be, buf);
    fails += geist_expect(geist_backend_memory_info(be, &after) == GEIST_OK, "OK after release");

    const uint64_t drop =
            before.free_bytes > held.free_bytes ? before.free_bytes - held.free_bytes : 0;
    if (before.unified_memory && before.device_wide) {
        /* The budget of an integrated GPU follows free system RAM, which
         * every process on the host moves: report, do not judge. */
        printf("unified memory: 256 MiB took %llu MiB off free, release left %llu of %llu MiB\n",
               (unsigned long long) (drop / MIB),
               (unsigned long long) (after.free_bytes / MIB),
               (unsigned long long) (before.free_bytes / MIB));
        return fails + geist_expect(held.free_bytes < before.free_bytes, "free drops while held");
    }
    snprintf(msg,
             sizeof msg,
             "a 256 MiB buffer takes about 256 MiB off free (took %llu MiB)",
             (unsigned long long) (drop / MIB));
    fails += geist_expect(gap(drop, n) <= 64 * MIB, msg);
    snprintf(msg,
             sizeof msg,
             "releasing it gives the memory back (%llu vs %llu MiB)",
             (unsigned long long) (after.free_bytes / MIB),
             (unsigned long long) (before.free_bytes / MIB));
    fails += geist_expect(gap(after.free_bytes, before.free_bytes) <= 64 * MIB, msg);
    fails += geist_expect(held.total_bytes == before.total_bytes, "total does not move");
    return fails;
}

static int check_vulkan_budget(void) {
    setenv("GEIST_VK_VRAM_BUDGET", "64M", 1);
    struct geist_backend *be = nullptr;
    const bool            ok = geist_backend_create("vulkan", nullptr, nullptr, &be) == GEIST_OK;
    unsetenv("GEIST_VK_VRAM_BUDGET");
    if (!ok) {
        return geist_expect(false, "vulkan backend with GEIST_VK_VRAM_BUDGET=64M");
    }
    struct geist_backend_memory m = {0};
    const int fails = geist_expect(geist_backend_memory_info(be, &m) == GEIST_OK &&
                                           m.total_bytes == 64 * MIB && m.free_bytes <= 64 * MIB,
                                   "GEIST_VK_VRAM_BUDGET=64M caps total and free");
    geist_backend_destroy(be);
    return fails;
}

int main(void) {
    int fails = check_generic();

    unsetenv("GEIST_VK_VRAM_BUDGET");
    struct geist_backend *be = nullptr;
    if (geist_backend_create("vulkan", nullptr, nullptr, &be) != GEIST_OK) {
        if (fails != 0) {
            return 1;
        }
        GEIST_SKIP("no Vulkan backend or device");
    }
    fails += check_vulkan(be);
    geist_backend_destroy(be);
    fails += check_vulkan_budget();

    if (fails != 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return 1;
    }
    puts("backend memory info: invalid, unsupported, allocation and budget checks passed");
    return 0;
}

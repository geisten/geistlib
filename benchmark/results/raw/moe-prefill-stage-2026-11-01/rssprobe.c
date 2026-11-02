#include <geist.h>
#include <geist_backend.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static long rss_kb(void) {
    FILE *f = fopen("/proc/self/status", "r"); char l[256]; long v = 0;
    while (fgets(l, sizeof l, f)) if (!strncmp(l, "VmRSS:", 6)) v = atol(l + 6);
    fclose(f); return v;
}
static long avail_kb(void) {
    FILE *f = fopen("/proc/meminfo", "r"); char l[256]; long v = 0;
    while (fgets(l, sizeof l, f)) if (!strncmp(l, "MemAvailable:", 13)) v = atol(l + 13);
    fclose(f); return v;
}
static long pss_kb(void) {
    FILE *f = fopen("/proc/self/smaps_rollup", "r"); char l[256]; long v = 0;
    while (fgets(l, sizeof l, f)) if (!strncmp(l, "Pss:", 4)) v = atol(l + 4);
    fclose(f); return v;
}
int main(int argc, char **argv) {
    const int n = argc > 1 ? atoi(argv[1]) : 1000; const size_t rows = argc > 2 ? atol(argv[2]) : 288;
    struct geist_backend *be = nullptr;
    if (geist_backend_create("vulkan", nullptr, nullptr, &be) != GEIST_OK) return 77;
    const size_t bytes = rows * 512 * 4;
    float *blob = malloc(n * bytes); memset(blob, 0, n * bytes);
    struct geist_weight *w = calloc(n, sizeof *w);
    long r0 = rss_kb(), a0 = avail_kb(), p0 = pss_kb();
    for (int i = 0; i < n; i++) {
        w[i] = (struct geist_weight){.raw = (char *) blob + i * bytes, .raw_nbytes = bytes, .n_in = 512, .n_out = (int) rows, .dtype = GEIST_DTYPE_F32, .flags = GEIST_W_HOST};
        if (be->desc->vtbl->resolve_weight(be, &w[i]) != GEIST_OK) { printf("fail at %d\n", i); break; }
    }
    long r1 = rss_kb(), a1 = avail_kb(), p1 = pss_kb();
    printf("MemAvailable -%ld MiB, PSS +%ld MiB\n", (a0 - a1) >> 10, (p1 - p0) >> 10);
    printf("%d host weights of %zu KiB: RSS +%ld MiB (%.2fx the bytes)\n", n, bytes >> 10, (r1 - r0) >> 10, (double) (r1 - r0) * 1024 / ((double) n * bytes));
    return 0;
}

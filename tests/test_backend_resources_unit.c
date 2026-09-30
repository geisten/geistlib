/* Optional capability / well-defined failure output, no model fixture. */
#include <geist_util.h>
#include "test_helpers.h"
#include <geist.h>
#include <geist_backend.h>
#include <stdio.h>

static enum geist_status broken(const struct geist_backend     *be,
                                struct geist_backend_resources *out) {
    (void) be;
    out->allocated_bytes = 123;
    return GEIST_E_INTERNAL;
}
int main(void) {
    struct geist_backend_resources sample = {.allocated_bytes = 123};
    if (geist_backend_resources_snapshot(nullptr, &sample) != GEIST_E_INVALID_ARG ||
        sample.allocated_bytes || sample.source != GEIST_RESOURCE_NONE || sample.unified_memory)
        return 1;
    if (geist_backend_resources_snapshot(nullptr, nullptr) != GEIST_E_INVALID_ARG)
        return 2;
    const struct geist_backend_descriptor unsupported = {0};
    const struct geist_backend_descriptor failure     = {.resources_snapshot = broken};
    struct geist_backend                  fixture     = {.desc = &unsupported};
    if (geist_backend_resources_snapshot(&fixture, &sample) != GEIST_E_UNSUPPORTED ||
        sample.allocated_bytes)
        return 3;
    fixture.desc = &failure;
    if (geist_backend_resources_snapshot(&fixture, &sample) != GEIST_E_INTERNAL ||
        sample.allocated_bytes)
        return 4;
    struct geist_backend *cpu = nullptr;
    if (geist_backend_create("cpu_scalar", nullptr, nullptr, &cpu) != GEIST_OK)
        return 5;
    enum geist_status status = geist_backend_resources_snapshot(cpu, &sample);
    geist_backend_destroy(cpu);
    cpu = nullptr;
    if (status != GEIST_E_UNSUPPORTED || sample.allocated_bytes || sample.source)
        return 6;
    if (geist_backend_resources_snapshot(cpu, &sample) != GEIST_E_INVALID_ARG)
        return 7;
    puts("backend resources: unsupported, missing, failing and closed handles passed");
    return 0;
}

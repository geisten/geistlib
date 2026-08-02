/* Smoke test for gguf_reader: open file, dump tensor count + dtype histogram +
 * a handful of named lookups. */
#include "gguf_reader.h"
#include "test_helpers.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *model = geist_test_gguf_arg(argc, argv, 1);

    const char      *err = nullptr;
    struct gguf_ctx *ctx = gguf_open(model, &err);
    if (!ctx) {
        fprintf(stderr, "gguf_open: %s\n", err ? err : "?");
        return 1;
    }

    size_t n = gguf_tensor_count(ctx);
    fprintf(stderr, "Loaded %zu tensors from %s\n", n, model);

    /* dtype histogram */
    int    hist[40]    = {0};
    size_t total_bytes = 0;
    for (size_t i = 0; i < n; i++) {
        const struct gguf_tensor_t *t = gguf_tensor_at(ctx, i);
        if ((int) t->dtype < 40)
            hist[t->dtype]++;
        total_bytes += t->nbytes;
    }
    fprintf(stderr, "dtype histogram:\n");
    for (int i = 0; i < 40; i++) {
        if (hist[i] > 0)
            fprintf(stderr,
                    "  [%2d] %s: %d tensors\n",
                    i,
                    gguf_dtype_name((gguf_dtype_t) i),
                    hist[i]);
    }
    fprintf(stderr, "total tensor data: %.1f MB\n", total_bytes / 1e6);

    /* For any unknown dtype, list the first few tensor names so we can see
     * which weights use the format. */
    bool unknown_seen[40] = {false};
    for (size_t i = 0; i < n; i++) {
        const struct gguf_tensor_t *t  = gguf_tensor_at(ctx, i);
        int                         dt = (int) t->dtype;
        if (dt >= 40 || dt < 0)
            continue;
        const char *nm = gguf_dtype_name((gguf_dtype_t) dt);
        if (strcmp(nm, "?") == 0 && !unknown_seen[dt]) {
            unknown_seen[dt] = true;
            fprintf(stderr, "  unknown dtype id=%d first tensor: %s (dims=[", dt, t->name);
            for (int d = 0; d < t->n_dims; d++) {
                if (d > 0)
                    fprintf(stderr, ",");
                fprintf(stderr, "%llu", (unsigned long long) t->dims[d]);
            }
            fprintf(stderr, "] nbytes=%zu)\n", t->nbytes);
        }
    }

    /* Print first 5 tensors as sanity */
    fprintf(stderr, "\nFirst 5 tensors:\n");
    for (size_t i = 0; i < (n < 5 ? n : 5); i++) {
        const struct gguf_tensor_t *t = gguf_tensor_at(ctx, i);
        fprintf(stderr, "  [%zu] %s  %s  dims=[", i, t->name, gguf_dtype_name(t->dtype));
        for (int d = 0; d < t->n_dims; d++) {
            if (d > 0)
                fprintf(stderr, ",");
            fprintf(stderr, "%llu", (unsigned long long) t->dims[d]);
        }
        fprintf(stderr, "]  nbytes=%zu  off=%llu\n", t->nbytes, (unsigned long long) t->offset);
    }

    /* Lookup user-supplied names */
    for (int a = 2; a < argc; a++) {
        const struct gguf_tensor_t *t = gguf_get_tensor(ctx, argv[a]);
        if (!t) {
            printf("MISSING: %s\n", argv[a]);
            continue;
        }
        printf("%s  %s  dims=[", t->name, gguf_dtype_name(t->dtype));
        for (int d = 0; d < t->n_dims; d++) {
            if (d > 0)
                printf(",");
            printf("%llu", (unsigned long long) t->dims[d]);
        }
        printf("]  nbytes=%zu\n", t->nbytes);
    }

    gguf_close(ctx);
    return 0;
}

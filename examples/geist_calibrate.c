/*
 * examples/geist_calibrate.c — reference consumer for the calibration
 * API (PR 3 of the calibration program).
 *
 * Demonstrates the POLICY side the library deliberately does not own:
 * when to measure (quiet-machine gate), where to persist (a file named
 * by the opaque key), and how to react to staleness. The library only
 * measures, serializes and validates.
 *
 *   geist-calibrate <dir> [--force]  measure and write <dir>/<key>
 *                                    (--force skips the load gate)
 *   geist-calibrate --apply <file>   validate a stored blob against
 *                                    THIS machine (exit 0 = would apply)
 *
 * A real consumer (daemon, app) does the same around startup: build the
 * key, look up its own cache, geist_backend_apply_calibration before
 * the first model load, recalibrate on GEIST_E_STALE_CALIBRATION.
 */
#include <geist.h>
#include <geist_backend.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#if !defined(_WIN32)
#include <sys/stat.h>
#endif

static void key_to_filename(const char *key, char *out, size_t out_size) {
    size_t j = 0;
    for (size_t i = 0; key[i] != '\0' && j + 1 < out_size; i++) {
        const char c = key[i];
        out[j++]     = (c == '|' || c == '/' || c == '\\' || c == ':' || c == '*' || c == '+')
                               ? '_'
                               : c;
    }
    out[j] = '\0';
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <dir> | --apply <blob-file>\n", argv[0]);
        return 2;
    }

    struct geist_backend *be = NULL;
    if (geist_backend_create("cpu_neon", NULL, NULL, &be) != GEIST_OK) {
        fprintf(stderr, "geist-calibrate: no cpu_neon backend on this build\n");
        return 1;
    }

    if (strcmp(argv[1], "--apply") == 0) {
        if (argc < 3) {
            fprintf(stderr, "--apply needs a blob file\n");
            return 2;
        }
        FILE *f = fopen(argv[2], "rb");
        if (f == NULL) {
            fprintf(stderr, "cannot open %s\n", argv[2]);
            return 1;
        }
        char         blob[8192];
        const size_t n = fread(blob, 1, sizeof blob, f);
        fclose(f);
        const enum geist_status s = geist_backend_apply_calibration(be, blob, n);
        printf("apply: %s\n", geist_status_to_string(s));
        geist_backend_destroy(be);
        return s == GEIST_OK ? 0 : 1;
    }

    /* Policy: refuse to measure on a busy machine — the library will
     * happily measure garbage under load; deciding not to is OUR job. */
#if !defined(_WIN32)
    {
        double load[1] = {0};
        const bool force = argc > 2 && strcmp(argv[2], "--force") == 0;
        if (!force && getloadavg(load, 1) == 1 && load[0] > 1.5) {
            fprintf(stderr,
                    "geist-calibrate: load %.2f > 1.5 — measure on a quiet machine\n",
                    load[0]);
            return 1;
        }
    }
#endif

    char   key[256];
    size_t need = 0;
    if (geist_backend_calibration_key(be, key, sizeof key, &need) != GEIST_OK) {
        fprintf(stderr, "geist-calibrate: no machine identity — cannot key a cache\n");
        return 1;
    }

    size_t blob_need = 0;
    (void) geist_backend_calibrate(be, 0, NULL, 0, &blob_need);
    char *blob = malloc(blob_need);
    if (blob == NULL) {
        return 1;
    }
    printf("calibrating (key %s)...\n", key);
    size_t                  blob_len = 0;
    const enum geist_status s =
            geist_backend_calibrate(be, 4000000000ull /* 4 s */, blob, blob_need, &blob_len);
    if (s != GEIST_OK) {
        fprintf(stderr, "calibrate failed: %s\n", geist_status_to_string(s));
        return 1;
    }
    const char *dis = strstr(blob, "disagreement=");
    if (dis != NULL && atoi(dis + 13) > 2) {
        fprintf(stderr,
                "warning: high measurement disagreement (%d) — machine may not be quiet;"
                " consider re-running\n",
                atoi(dis + 13));
    }

    char fname[300];
    key_to_filename(key, fname, sizeof fname - 1);
    char path[512];
    snprintf(path, sizeof path, "%s/%s", argv[1], fname);
    FILE *out = fopen(path, "wb");
    if (out == NULL) {
        fprintf(stderr, "cannot write %s\n", path);
        return 1;
    }
    fwrite(blob, 1, blob_len - 1, out);
    fclose(out);
    printf("wrote %s\n%s", path, blob);
    free(blob);
    geist_backend_destroy(be);
    return 0;
}

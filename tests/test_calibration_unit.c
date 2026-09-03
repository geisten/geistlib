/* Calibration driver contract (EXPERIMENTAL API): key stability,
 * calibrate blob shape, apply roundtrip, stale/format rejection,
 * atomicity, and the resolve-time lock. Runs the real cpu_neon sondes
 * with a small budget — values are machine-dependent by design, only
 * their kind-validity is asserted. */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>

#include <stdlib.h>
#include <string.h>

static int  g_fail = 0;
static void check(bool ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        g_fail = 1;
    }
}

int main(void) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create("cpu_neon", nullptr, nullptr, &be) != GEIST_OK) {
        printf("test_calibration_unit: SKIP (no cpu_neon backend)\n");
        return 0;
    }

    /* -- key: size query, fetch, stability ---------------------------- */
    size_t key_need = 0;
    check(geist_backend_calibration_key(be, nullptr, 0, &key_need) == GEIST_E_INVALID_ARG,
          "key size query returns INVALID_ARG with required_size set");
    check(key_need > 8 && key_need < 256, "key size plausible");
    char key1[256], key2[256];
    check(geist_backend_calibration_key(be, key1, sizeof key1, &key_need) == GEIST_OK, "key fetch");
    check(geist_backend_calibration_key(be, key2, sizeof key2, &key_need) == GEIST_OK, "key again");
    check(strcmp(key1, key2) == 0, "key is stable across calls");

    /* -- calibrate: blob shape ---------------------------------------- */
    size_t need = 0;
    check(geist_backend_calibrate(be, 0, nullptr, 0, &need) == GEIST_E_INVALID_ARG,
          "calibrate size query");
    check(need > 0 && need < 65536, "calibrate size plausible");
    char *blob = malloc(need);
    check(blob != nullptr, "blob alloc");
    size_t blob_len = 0;
    check(geist_backend_calibrate(be, 50000000ull /* 50 ms */, blob, need, &blob_len) == GEIST_OK,
          "calibrate runs");
    check(blob_len > 0 && blob_len <= need, "blob length sane");
    check(strstr(blob, "key=") != nullptr, "blob carries key");
    check(strstr(blob, key1) != nullptr, "blob key matches calibration_key()");
    check(strstr(blob, "q5k_native_mn=") != nullptr, "blob has bool tunable");
    check(strstr(blob, "qk_sgemm_threshold=") != nullptr, "blob has size tunable");

    /* -- apply roundtrip on a fresh backend --------------------------- */
    struct geist_backend *be2 = nullptr;
    check(geist_backend_create("cpu_neon", nullptr, nullptr, &be2) == GEIST_OK, "second backend");
    check(geist_backend_apply_calibration(be2, blob, blob_len - 1) == GEIST_OK, "apply ok");
    int64_t v = -1;
    check(geist_calibration_lookup(be2, "q5k_native_mn", &v) && (v == 0 || v == 1),
          "applied bool visible and kind-valid");
    check(geist_calibration_lookup(be2, "qk_sgemm_threshold", &v) && v > 0,
          "applied size visible and kind-valid");

    /* -- stale: tampered key ------------------------------------------ */
    struct geist_backend *be3 = nullptr;
    check(geist_backend_create("cpu_neon", nullptr, nullptr, &be3) == GEIST_OK, "third backend");
    {
        char *tampered = malloc(blob_len);
        memcpy(tampered, blob, blob_len);
        char *kp = strstr(tampered, "key=v");
        check(kp != nullptr, "key line found");
        kp[4] = 'X'; /* corrupt the schema field */
        check(geist_backend_apply_calibration(be3, tampered, blob_len - 1) ==
                      GEIST_E_STALE_CALIBRATION,
              "tampered key -> STALE");
        check(be3->calibration.n_values == 0, "stale apply left backend unchanged");
        free(tampered);
    }

    /* -- format errors are atomic ------------------------------------- */
    {
        const char *bad = "key=whatever\nq5k_native_mn=7\n";
        check(geist_backend_apply_calibration(be3, bad, strlen(bad)) == GEIST_E_FORMAT,
              "bool out of range -> FORMAT");
        const char *unknown = "key=whatever\nno_such_tunable=1\n";
        check(geist_backend_apply_calibration(be3, unknown, strlen(unknown)) == GEIST_E_FORMAT,
              "unknown tunable -> FORMAT");
        check(be3->calibration.n_values == 0, "failed applies left backend unchanged");
    }

    /* -- lock: apply after first resolve is refused ------------------- */
    be3->calibration.locked = true; /* what weight-load does */
    check(geist_backend_apply_calibration(be3, blob, blob_len - 1) == GEIST_E_INVALID_STATE,
          "apply after lock -> INVALID_STATE");

    free(blob);
    geist_backend_destroy(be3);
    geist_backend_destroy(be2);
    geist_backend_destroy(be);
    if (g_fail == 0) {
        printf("test_calibration_unit: all checks passed\n");
    }
    return g_fail == 0 ? 0 : 1;
}

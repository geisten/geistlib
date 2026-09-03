/*
 * src/base/calibration.c — generic per-machine calibration driver.
 *
 * Layer: ENGINE.
 *
 * Owns everything AROUND a backend's measurement sondes: the opaque
 * key, the text-blob format, driver-level noise defense (median of
 * three sonde runs + a disagreement note), and the atomic apply into
 * geist_backend.calibration. Backends contribute only name + kind +
 * measure() through the descriptor's `tunables` slot; consumers own
 * persistence entirely (the library never touches the filesystem).
 *
 * Blob format (UTF-8, line-oriented):
 *   # geist-calibration v1
 *   key=<opaque key>
 *   disagreement=<n sonde runs that disagreed with the median>
 *   <tunable>=<int64 value>
 *   ...
 *
 * Key composition (opaque to consumers, may grow factors):
 *   v<schema>|<backend>|g<generation>|t<fnv1a of tunable names>|<uarch>
 *   |c<logical cores>|l<l3 domains>
 * The tunable-name hash auto-invalidates on renames/additions; the
 * per-backend generation is bumped manually when kernel performance
 * character changes without a name change (the #318 case).
 */
#define GEIST_INTERNAL_ENGINE_LAYER

#include "hw_probe.h"

#include <geist.h>
#include <geist_backend.h>

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static constexpr uint32_t CAL_SCHEMA_VERSION = 1u;
static constexpr size_t   CAL_MAX_TUNABLES =
        sizeof(((struct geist_backend *) 0)->calibration.values) /
        sizeof(((struct geist_backend *) 0)->calibration.values[0]);

static uint32_t cal_fnv1a(const struct geist_tunable *t, size_t n) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++) {
        for (const char *p = t[i].name; *p != '\0'; p++) {
            h = (h ^ (uint32_t) (unsigned char) *p) * 16777619u;
        }
        h = (h ^ 0x7cu) * 16777619u; /* name separator */
    }
    return h;
}

/* Builds the key into buf; returns needed size incl. NUL (snprintf
 * convention + 1). buf may be nullptr/short — always safe. */
static size_t cal_key_build(const struct geist_backend *be, char *buf, size_t buf_size) {
    struct geist_hw_probe hw;
    geist_hw_probe_fill(&hw);
    size_t                      n_tun = 0;
    const struct geist_tunable *tun =
            be->desc->tunables != nullptr ? be->desc->tunables(&n_tun) : nullptr;
    const uint32_t names_hash = tun != nullptr ? cal_fnv1a(tun, n_tun) : 0u;
    const int      need       = snprintf(buf,
                                         buf_size,
                                         "v%" PRIu32 "|%s|g%" PRIu32 "|t%08" PRIx32 "|%s|c%zu|l%zu",
                                         CAL_SCHEMA_VERSION,
                                         be->desc->name,
                                         be->desc->caps.calibration_generation,
                                         names_hash,
                                         hw.uarch,
                                         hw.logical_cores,
                                         hw.n_l3_domains);
    return need > 0 ? (size_t) need + 1u : 0u;
}

enum geist_status geist_backend_calibration_key(const struct geist_backend *be,
                                                char                       *buf,
                                                size_t                      buf_size,
                                                size_t                     *required_size) {
    if (be == nullptr || be->desc == nullptr || required_size == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    {
        struct geist_hw_probe hw;
        geist_hw_probe_fill(&hw);
        if (hw.uarch[0] == '\0') {
            /* No identity source — a key would collide across machines. */
            return GEIST_E_UNSUPPORTED;
        }
    }
    const size_t need = cal_key_build(be, buf, buf != nullptr ? buf_size : 0u);
    *required_size    = need;
    if (buf == nullptr || buf_size < need) {
        return GEIST_E_INVALID_ARG;
    }
    return GEIST_OK;
}

enum geist_status geist_backend_calibrate(struct geist_backend *be,
                                          uint64_t              budget_ns,
                                          char                 *buf,
                                          size_t                buf_size,
                                          size_t               *required_size) {
    if (be == nullptr || be->desc == nullptr || required_size == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    if (be->desc->tunables == nullptr) {
        return GEIST_E_UNSUPPORTED;
    }
    size_t                      n_tun = 0;
    const struct geist_tunable *tun   = be->desc->tunables(&n_tun);
    if (tun == nullptr || n_tun == 0 || n_tun > CAL_MAX_TUNABLES) {
        return GEIST_E_UNSUPPORTED;
    }

    char         key[192];
    const size_t key_need = cal_key_build(be, key, sizeof key);
    if (key_need == 0 || key_need > sizeof key) {
        return GEIST_E_UNSUPPORTED;
    }

    /* Size estimate: header + one worst-case line per tunable. */
    const size_t need_estimate = 32u + 4u + key_need + 20u + n_tun * (48u + 21u) + 1u;
    *required_size             = need_estimate;
    if (buf == nullptr || buf_size < need_estimate) {
        return GEIST_E_INVALID_ARG;
    }

    /* Driver-level noise defense: each sonde runs three times, the
     * median value wins, and runs disagreeing with it are counted into
     * the blob header so consumers can judge measurement quality. */
    int64_t        values[CAL_MAX_TUNABLES];
    uint32_t       disagreement = 0;
    const uint64_t per_sonde    = budget_ns / (n_tun * 3u > 0 ? (uint64_t) (n_tun * 3u) : 1u);
    for (size_t i = 0; i < n_tun; i++) {
        int64_t v[3];
        for (int r = 0; r < 3; r++) {
            v[r]                      = 0;
            const enum geist_status s = tun[i].measure(be, per_sonde, &v[r]);
            if (s != GEIST_OK) {
                return s;
            }
        }
        /* median of three */
        int64_t lo  = v[0] < v[1] ? v[0] : v[1];
        int64_t hi  = v[0] < v[1] ? v[1] : v[0];
        int64_t med = v[2] < lo ? lo : (v[2] > hi ? hi : v[2]);
        for (int r = 0; r < 3; r++) {
            if (v[r] != med) {
                disagreement++;
            }
        }
        if (tun[i].kind == GEIST_TUNABLE_BOOL && (med < 0 || med > 1)) {
            return GEIST_E_INTERNAL;
        }
        if (tun[i].kind == GEIST_TUNABLE_SIZE && med <= 0) {
            return GEIST_E_INTERNAL;
        }
        values[i] = med;
    }

    size_t off = 0;
    off += (size_t) snprintf(buf + off,
                             buf_size - off,
                             "# geist-calibration v%" PRIu32 "\nkey=%s\ndisagreement=%" PRIu32 "\n",
                             CAL_SCHEMA_VERSION,
                             key,
                             disagreement);
    for (size_t i = 0; i < n_tun && off < buf_size; i++) {
        off += (size_t) snprintf(
                buf + off, buf_size - off, "%s=%" PRId64 "\n", tun[i].name, values[i]);
    }
    if (off >= buf_size) {
        return GEIST_E_INTERNAL; /* estimate was wrong — bug */
    }
    *required_size = off + 1u;
    return GEIST_OK;
}

enum geist_status
geist_backend_apply_calibration(struct geist_backend *be, const char *blob, size_t blob_size) {
    if (be == nullptr || be->desc == nullptr || blob == nullptr || blob_size == 0) {
        return GEIST_E_INVALID_ARG;
    }
    if (be->calibration.locked) {
        return GEIST_E_INVALID_STATE;
    }
    if (be->desc->tunables == nullptr) {
        return GEIST_E_UNSUPPORTED;
    }
    size_t                      n_tun = 0;
    const struct geist_tunable *tun   = be->desc->tunables(&n_tun);
    if (tun == nullptr || n_tun == 0) {
        return GEIST_E_UNSUPPORTED;
    }

    char         key[192];
    const size_t key_need = cal_key_build(be, key, sizeof key);
    if (key_need == 0 || key_need > sizeof key) {
        return GEIST_E_UNSUPPORTED;
    }

    /* Parse into a staging copy first — apply is atomic. */
    struct geist_calibration_value staged[CAL_MAX_TUNABLES];
    size_t                         n_staged  = 0;
    bool                           key_seen  = false;
    bool                           key_match = false;

    const char *p   = blob;
    const char *end = blob + blob_size;
    while (p < end) {
        const char  *nl   = memchr(p, '\n', (size_t) (end - p));
        const char  *line = p;
        const size_t len  = nl != nullptr ? (size_t) (nl - p) : (size_t) (end - p);
        p                 = nl != nullptr ? nl + 1 : end;
        if (len == 0 || line[0] == '#') {
            continue;
        }
        const char *eq = memchr(line, '=', len);
        if (eq == nullptr) {
            return GEIST_E_FORMAT;
        }
        const size_t klen = (size_t) (eq - line);
        const char  *val  = eq + 1;
        const size_t vlen = len - klen - 1u;
        if (klen == 0 || vlen == 0 || vlen > 63) {
            return GEIST_E_FORMAT;
        }
        char vbuf[64];
        memcpy(vbuf, val, vlen);
        vbuf[vlen] = '\0';

        if (klen == 3 && memcmp(line, "key", 3) == 0) {
            key_seen  = true;
            key_match = strlen(key) == vlen && memcmp(key, vbuf, vlen) == 0;
            continue;
        }
        if (klen == 12 && memcmp(line, "disagreement", 12) == 0) {
            continue; /* informational */
        }
        /* Must be a known tunable with a kind-valid value. */
        const struct geist_tunable *match = nullptr;
        for (size_t i = 0; i < n_tun; i++) {
            if (strlen(tun[i].name) == klen && memcmp(tun[i].name, line, klen) == 0) {
                match = &tun[i];
                break;
            }
        }
        if (match == nullptr) {
            return GEIST_E_FORMAT;
        }
        char         *endp = nullptr;
        const int64_t v    = (int64_t) strtoll(vbuf, &endp, 10);
        if (endp == vbuf || *endp != '\0') {
            return GEIST_E_FORMAT;
        }
        if (match->kind == GEIST_TUNABLE_BOOL && (v < 0 || v > 1)) {
            return GEIST_E_FORMAT;
        }
        if (match->kind == GEIST_TUNABLE_SIZE && v <= 0) {
            return GEIST_E_FORMAT;
        }
        if (n_staged >= CAL_MAX_TUNABLES) {
            return GEIST_E_FORMAT;
        }
        snprintf(staged[n_staged].name, sizeof(staged[n_staged].name), "%.*s", (int) klen, line);
        staged[n_staged].value = v;
        n_staged++;
    }
    if (!key_seen || n_staged == 0) {
        return GEIST_E_FORMAT;
    }
    if (!key_match) {
        return GEIST_E_STALE_CALIBRATION;
    }

    /* Commit — all or nothing. */
    memcpy(be->calibration.values, staged, n_staged * sizeof(staged[0]));
    be->calibration.n_values = n_staged;
    return GEIST_OK;
}

bool geist_calibration_lookup(const struct geist_backend *be, const char *name, int64_t *out) {
    if (be == nullptr || name == nullptr || out == nullptr) {
        return false;
    }
    for (size_t i = 0; i < be->calibration.n_values; i++) {
        if (strcmp(be->calibration.values[i].name, name) == 0) {
            *out = be->calibration.values[i].value;
            return true;
        }
    }
    return false;
}

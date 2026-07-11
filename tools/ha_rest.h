/*
 * ha_rest.h — standalone Home Assistant REST client. Header-only, agent-free:
 * this layer knows HTTP + the HA API shape and NOTHING about tools or models,
 * so a host app (or the `geist ha` debug CLI) can use it directly and the
 * pure parts test without a network.
 *
 *   ha_call_service(url, token, "light", "turn_on", "light.flur", NULL, ...)
 *       POST /api/services/<domain>/<service>   body {"entity_id":"..."}
 *   ha_get_state(url, token, "light.flur", ...)
 *       GET  /api/states/<entity>               -> "state" value + raw JSON
 *
 * Transport: curl via fork+execvp (no shell), Bearer-token header — the
 * webfetch security posture (scheme pinning, size/time caps). Unix/desktop
 * only; an iOS/Android host supplies its own transport over the platform
 * HTTP client and reuses the pure helpers.
 *
 * ponytail: flat JSON field extraction (ha_json_str), no walker — HA's
 * /api/states response nests "attributes", but the fields we read (state,
 * friendly_name, temperature, unit_of_measurement) are reachable by first
 * occurrence. Revisit with a real walker if an integration breaks that.
 */
#ifndef GEIST_HA_REST_H
#define GEIST_HA_REST_H

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#define HA_URL_DEFAULT "http://localhost:8123"
enum { HA_URL_CAP = 512, HA_BODY_CAP = 512 };

/* GEIST_HA_URL / GEIST_HA_TOKEN with defaults ("" token = unconfigured). */
static inline const char *ha_env_url(void) {
    const char *u = getenv("GEIST_HA_URL");
    return (u && u[0]) ? u : HA_URL_DEFAULT;
}
static inline const char *ha_env_token(void) {
    const char *t = getenv("GEIST_HA_TOKEN");
    return t ? t : "";
}

/* ---- pure helpers (no network) ------------------------------------------ */

/* First "key":"value" or "key":123 in json -> out. Strings lose their quotes;
 * numbers/bools are copied verbatim up to the next , } or ]. Returns length,
 * 0 if the key is absent. Flat fields only (see header note). */
static inline size_t
ha_json_str(const char *json, const char *key, size_t cap, char out[static cap]) {
    out[0] = '\0';
    char pat[96];
    snprintf(pat, sizeof pat, "\"%s\"", key);
    const char *p = strstr(json, pat);
    if (p == nullptr) {
        return 0;
    }
    p += strlen(pat);
    while (*p == ' ' || *p == ':') {
        p++;
    }
    size_t w = 0;
    if (*p == '"') { /* string value */
        for (p++; *p && *p != '"' && w + 1 < cap; p++) {
            if (*p == '\\' && p[1]) {
                p++;
            }
            out[w++] = *p;
        }
    } else { /* number / bool / null */
        while (*p && *p != ',' && *p != '}' && *p != ']' && *p != ' ' && w + 1 < cap) {
            out[w++] = *p++;
        }
    }
    out[w] = '\0';
    return w;
}

/* POST /api/services/<domain>/<service> URL into out. Returns length. */
static inline size_t ha_service_url(const char *base,
                                    const char *domain,
                                    const char *service,
                                    size_t      cap,
                                    char        out[static cap]) {
    int n = snprintf(out, cap, "%s/api/services/%s/%s", base, domain, service);
    return n > 0 && (size_t) n < cap ? (size_t) n : 0;
}

/* GET /api/states/<entity_id> URL into out. Returns length. */
static inline size_t
ha_state_url(const char *base, const char *entity, size_t cap, char out[static cap]) {
    int n = snprintf(out, cap, "%s/api/states/%s", base, entity);
    return n > 0 && (size_t) n < cap ? (size_t) n : 0;
}

/* Service-call body {"entity_id":"..."} with optional extra fields merged in
 * ("\"temperature\":21.5"). Returns length, 0 on overflow. */
static inline size_t
ha_service_body(const char *entity, const char *extra, size_t cap, char out[static cap]) {
    int n = (extra && extra[0]) ? snprintf(out, cap, "{\"entity_id\":\"%s\",%s}", entity, extra)
                                : snprintf(out, cap, "{\"entity_id\":\"%s\"}", entity);
    return n > 0 && (size_t) n < cap ? (size_t) n : 0;
}

/* ---- transport (curl, no shell) ------------------------------------------ */

/* Run curl against `url` with the Bearer token; POST with body when body is
 * non-null, else GET. Response bytes into raw. Returns bytes, -1 on failure.
 * Mirrors webfetch_curl: argv exec (no shell), scheme pinned, caps. */
static inline long
ha_curl(const char *url, const char *token, const char *body, size_t cap, char raw[static cap]) {
    int fds[2];
    if (pipe(fds) != 0) {
        return -1;
    }
    pid_t pid = fork();
    if (pid < 0) {
        close(fds[0]);
        close(fds[1]);
        return -1;
    }
    if (pid == 0) {
        dup2(fds[1], STDOUT_FILENO);
        int nul = open("/dev/null", O_WRONLY);
        if (nul >= 0) {
            dup2(nul, STDERR_FILENO);
        }
        close(fds[0]);
        close(fds[1]);
        char auth[600];
        snprintf(auth, sizeof auth, "Authorization: Bearer %s", token);
        const char *argv[24];
        size_t      a = 0;
        argv[a++]     = "curl";
        argv[a++]     = "-sS";
        argv[a++]     = "--max-time";
        argv[a++]     = "10";
        argv[a++]     = "--max-filesize";
        argv[a++]     = "1000000";
        argv[a++]     = "--proto";
        argv[a++]     = "=http,https";
        argv[a++]     = "-H";
        argv[a++]     = auth;
        if (body != nullptr) {
            argv[a++] = "-H";
            argv[a++] = "Content-Type: application/json";
            argv[a++] = "-X";
            argv[a++] = "POST";
            argv[a++] = "-d";
            argv[a++] = body;
        }
        argv[a++] = url;
        argv[a]   = nullptr;
        execvp("curl", (char *const *) argv);
        _exit(127);
    }
    close(fds[1]);
    size_t  n = 0;
    ssize_t r;
    while (n + 1 < cap && (r = read(fds[0], raw + n, cap - 1 - n)) > 0) {
        n += (size_t) r;
    }
    raw[n] = '\0';
    close(fds[0]);
    int status = 0;
    if (waitpid(pid, &status, 0) < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        return -1;
    }
    return (long) n;
}

/* ---- API calls ------------------------------------------------------------ */

/* Call a service; response (HA echoes the changed states) into out.
 * Returns bytes, -1 on transport failure / URL overflow. */
static inline long ha_call_service(const char *base,
                                   const char *token,
                                   const char *domain,
                                   const char *service,
                                   const char *entity,
                                   const char *extra,
                                   size_t      cap,
                                   char        out[static cap]) {
    char url[HA_URL_CAP];
    char body[HA_BODY_CAP];
    if (ha_service_url(base, domain, service, sizeof url, url) == 0 ||
        ha_service_body(entity, extra, sizeof body, body) == 0) {
        return -1;
    }
    return ha_curl(url, token, body, cap, out);
}

/* Read an entity; raw JSON into out. Returns bytes, -1 on failure. Use
 * ha_json_str(out, "state", ...) for the state value. */
static inline long ha_get_state(
        const char *base, const char *token, const char *entity, size_t cap, char out[static cap]) {
    char url[HA_URL_CAP];
    if (ha_state_url(base, entity, sizeof url, url) == 0) {
        return -1;
    }
    return ha_curl(url, token, nullptr, cap, out);
}

#endif /* GEIST_HA_REST_H */

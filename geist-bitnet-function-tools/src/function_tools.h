/*
 * function_tools.h — expose plain C functions as whitelist-gated geist tools.
 *
 * This is the "function tooling" extension on top of the imported engine
 * (geistenlib.a + the header-only agent layer): each entry wraps a native C
 * function into a `struct geist_tool` with a dynamic-tools-v1 JSON Schema
 * (`parameters_schema`), so the agent validates every argument object BEFORE
 * the function runs and the forced-call path can build typed multi-field
 * calls even on the untrained BitNet 2B-4T model.
 *
 * Shipped function tools:
 *   calculator  add/subtract/multiply/divide two numbers
 *   clock_now   current date + time (utc or local)
 *   sysinfo     host cores, memory, load average
 *
 * Conventions follow geisten (CONTRIBUTING.md): caller-provided buffers,
 * enum geist_status, no assert(), no hidden heap.
 */
#ifndef FUNCTION_TOOLS_H
#define FUNCTION_TOOLS_H

#include <geist.h>

#include "agent.h" /* struct geist_tool, agent_json_str, agent_obs */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h> /* sysconf */

/* Extract a JSON number field:  "<key>": <number>  (a quoted "5" also parses,
 * so a model that stringifies numbers still works). Returns 1 on success. */
static inline int ft_json_num(const char *json, const char *key, double *out) {
    char pat[GEIST_AGENT_NAME_CAP + 4];
    snprintf(pat, sizeof pat, "\"%s\"", key);
    const char *p = strstr(json, pat);
    if (!p) {
        return 0;
    }
    p += strlen(pat);
    while (*p == ':' || *p == ' ' || *p == '\t' || *p == '"') {
        p++;
    }
    char *end = nullptr;
    double v = strtod(p, &end);
    if (end == p) {
        return 0;
    }
    *out = v;
    return 1;
}

/* ---- calculator ------------------------------------------------------------ */

static enum geist_status ft_calculator(void *ctx, size_t args_len,
                                       const char args[static args_len],
                                       size_t out_cap, char out[static out_cap],
                                       size_t *out_len) {
    (void) ctx;
    (void) args_len;
    char   op[24];
    double a = 0.0, b = 0.0;
    if (!agent_json_str(args, "op", sizeof op, op) || !ft_json_num(args, "a", &a) ||
        !ft_json_num(args, "b", &b)) {
        return agent_obs(out_cap, out, out_len,
                         "error: calculator needs \"op\", \"a\" and \"b\"");
    }
    double r;
    if (strcmp(op, "add") == 0) {
        r = a + b;
    } else if (strcmp(op, "subtract") == 0) {
        r = a - b;
    } else if (strcmp(op, "multiply") == 0) {
        r = a * b;
    } else if (strcmp(op, "divide") == 0) {
        if (b == 0.0) {
            return agent_obs(out_cap, out, out_len, "error: division by zero");
        }
        r = a / b;
    } else {
        return agent_obs(out_cap, out, out_len, "error: unknown op \"%s\"", op);
    }
    return agent_obs(out_cap, out, out_len, "%.10g", r);
}

/* ---- clock_now -------------------------------------------------------------- */

static enum geist_status ft_clock_now(void *ctx, size_t args_len,
                                      const char args[static args_len],
                                      size_t out_cap, char out[static out_cap],
                                      size_t *out_len) {
    (void) ctx;
    (void) args_len;
    char zone[16] = "utc";
    agent_json_str(args, "zone", sizeof zone, zone); /* optional; default utc */
    time_t    now = time(nullptr);
    struct tm tm;
    bool      local = strcmp(zone, "local") == 0;
    if (local) {
        localtime_r(&now, &tm);
    } else {
        gmtime_r(&now, &tm);
    }
    char stamp[64];
    strftime(stamp, sizeof stamp, "%Y-%m-%d %H:%M:%S (%A)", &tm);
    return agent_obs(out_cap, out, out_len, "%s %s", stamp, local ? "local" : "UTC");
}

/* ---- sysinfo ----------------------------------------------------------------- */

static enum geist_status ft_sysinfo(void *ctx, size_t args_len,
                                    const char args[static args_len],
                                    size_t out_cap, char out[static out_cap],
                                    size_t *out_len) {
    (void) ctx;
    (void) args_len;
    (void) args;
    long cores = sysconf(_SC_NPROCESSORS_ONLN);
    long pages = sysconf(_SC_PHYS_PAGES);
    long psize = sysconf(_SC_PAGESIZE);
    double mem_gib = (pages > 0 && psize > 0)
                             ? (double) pages * (double) psize / (1024.0 * 1024.0 * 1024.0)
                             : 0.0;
    double load[3] = {0.0, 0.0, 0.0};
    int    nload   = getloadavg(load, 3);
    return agent_obs(out_cap, out, out_len,
                     "cores: %ld, memory: %.1f GiB, load: %.2f %.2f %.2f%s", cores,
                     mem_gib, load[0], load[1], load[2],
                     nload == 3 ? "" : " (load unavailable)");
}

/* ---- registry ----------------------------------------------------------------
 * Field order per struct geist_tool: name, args_schema (legacy display),
 * description (routing), parameters_schema (dynamic-tools-v1 validation),
 * invoke, ctx. The whole array is compile-time immutable — the model cannot
 * grant itself capabilities beyond this whitelist. */

static const struct geist_tool FUNCTION_TOOLS[] = {
        {
                .name        = "calculator",
                .args_schema = "{\"op\": string, \"a\": number, \"b\": number}",
                .description = "do arithmetic: add, subtract, multiply or divide two numbers",
                .parameters_schema =
                        "{\"type\":\"object\",\"properties\":{"
                        "\"op\":{\"type\":\"string\",\"enum\":[\"add\",\"subtract\","
                        "\"multiply\",\"divide\"]},"
                        "\"a\":{\"type\":\"number\"},\"b\":{\"type\":\"number\"}},"
                        "\"required\":[\"op\",\"a\",\"b\"],\"additionalProperties\":false}",
                .invoke = ft_calculator,
                .ctx    = nullptr,
        },
        {
                .name        = "clock_now",
                .args_schema = "{\"zone\": string}",
                .description = "tell the current date and time (zone: utc or local)",
                .parameters_schema =
                        "{\"type\":\"object\",\"properties\":{"
                        "\"zone\":{\"type\":\"string\",\"enum\":[\"utc\",\"local\"]}},"
                        "\"additionalProperties\":false}",
                .invoke = ft_clock_now,
                .ctx    = nullptr,
        },
        {
                .name        = "sysinfo",
                .args_schema = "{}",
                .description = "report this machine's CPU cores, memory and load average",
                .parameters_schema =
                        "{\"type\":\"object\",\"properties\":{},"
                        "\"additionalProperties\":false}",
                .invoke = ft_sysinfo,
                .ctx    = nullptr,
        },
};

enum { FUNCTION_TOOLS_N = sizeof FUNCTION_TOOLS / sizeof FUNCTION_TOOLS[0] };

#endif /* FUNCTION_TOOLS_H */

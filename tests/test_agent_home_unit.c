/* test_agent_home_unit — the pure (no-model, no-network) core of the home
 * tool family: registry parsing, alias-phrase matching incl. the deliberate
 * ambiguity answer, action/value mapping, pronoun fallback via last_entity,
 * and the v1 write-whitelist (locks are refused). Transport is stubbed. */
#define _POSIX_C_SOURCE 200809L
#include "test_helpers.h"
#include "../tools/agent_home.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;

/* transport stub: records the call, answers like HA would */
static char g_last_service[64], g_last_extra[64], g_last_entity[64];
static long stub_set(struct home_ctx          *c,
                     const struct home_device *d,
                     const char               *service,
                     const char               *extra,
                     size_t                    cap,
                     char                      out[]) {
    (void) c;
    snprintf(g_last_service, sizeof g_last_service, "%s", service);
    snprintf(g_last_extra, sizeof g_last_extra, "%s", extra);
    snprintf(g_last_entity, sizeof g_last_entity, "%s", d->entity);
    return (long) (size_t) snprintf(out, cap, "[]");
}
static long stub_get(struct home_ctx *c, const struct home_device *d, size_t cap, char out[]) {
    (void) c;
    if (strcmp(d->domain, "sensor") == 0) {
        return (long) (size_t) snprintf(
                out, cap, "{\"state\":\"19.5\",\"attributes\":{\"unit_of_measurement\":\"°C\"}}");
    }
    return (long) (size_t) snprintf(out, cap, "{\"state\":\"off\"}");
}

static struct home_ctx ctx;

static void setup(void) {
    memset(&ctx, 0, sizeof ctx);
    ctx.set             = stub_set;
    ctx.get             = stub_get;
    const char *lines[] = {
            "light.flur   | light   | flurlicht, licht flur, hallway light, licht",
            "light.bad    | light   | badlicht, licht bad, licht",
            "climate.bad  | climate | heizung bad, thermostat bad, heizung",
            "cover.wz     | cover   | rollladen wohnzimmer, rollladen",
            "sensor.bad_t | sensor  | temperatur bad, warm bad", /* cover 'wie warm' asks */
            "lock.haustuer| lock    | haustür, haustuer, front door",
    };
    for (size_t i = 0; i < sizeof lines / sizeof *lines; i++) {
        if (home_registry_line(lines[i], &ctx.dev[ctx.n_dev])) {
            ctx.n_dev++;
        }
    }
}

static void invoke_cmd(const char *req, size_t cap, char *out) {
    char args[600];
    snprintf(args, sizeof args, "{\"request\":\"%s\"}", req);
    size_t n = 0;
    (void) home_command_invoke(&ctx, strlen(args), args, cap, out, &n);
}

int main(void) {
    setup();
    fails += geist_expect(ctx.n_dev == 6, "registry: six usable lines");
    fails += geist_expect(strcmp(ctx.dev[0].entity, "light.flur") == 0 &&
                                  strcmp(ctx.dev[0].domain, "light") == 0,
                          "registry: fields trimmed");

    /* matching: specific phrase beats the generic one — no false ambiguity */
    size_t idx[8];
    fails += geist_expect(home_match(&ctx, "Schalte das Licht im Flur ein", 8, idx) == 1 &&
                                  idx[0] == 0,
                          "match: 'licht flur' wins over generic 'licht'");
    /* generic word with two candidates -> both, tied = deliberate ambiguity */
    fails += geist_expect(home_match(&ctx, "Schalte das Licht ein", 8, idx) == 2,
                          "match: generic 'licht' is ambiguous across two lights");
    fails += geist_expect(home_match(&ctx, "Mach die Kaffeemaschine an", 8, idx) == 0,
                          "match: unknown device yields none");

    /* action mapping */
    char extra[64];
    fails += geist_expect(strcmp(home_action("Schalte das Licht ein", "light", sizeof extra, extra),
                                 "turn_on") == 0,
                          "action: ein -> turn_on");
    fails += geist_expect(strcmp(home_action("Mach das Licht aus", "light", sizeof extra, extra),
                                 "turn_off") == 0,
                          "action: aus -> turn_off");
    fails += geist_expect(
            strcmp(home_action("Dimme das Licht auf 40 Prozent", "light", sizeof extra, extra),
                   "turn_on") == 0 &&
                    strcmp(extra, "\"brightness_pct\":40") == 0,
            "action: dimme 40 -> brightness_pct");
    fails += geist_expect(
            strcmp(home_action("Stelle die Heizung auf 21.5 Grad", "climate", sizeof extra, extra),
                   "set_temperature") == 0 &&
                    strcmp(extra, "\"temperature\":21.5") == 0,
            "action: climate value lifted");
    fails += geist_expect(home_action("Mach die Heizung wärmer", "climate", sizeof extra, extra) ==
                                  nullptr,
                          "action: climate without a value maps to nothing");
    fails += geist_expect(strcmp(home_action("Rollladen runter", "cover", sizeof extra, extra),
                                 "close_cover") == 0,
                          "action: runter -> close_cover");

    /* command invoke end-to-end against the stub */
    char out[HOME_OBS_CAP];
    invoke_cmd("Schalte das Licht im Flur ein", sizeof out, out);
    fails += geist_expect(strcmp(g_last_service, "turn_on") == 0 &&
                                  strcmp(g_last_entity, "light.flur") == 0 &&
                                  strstr(out, "OK: light.flur") != nullptr,
                          "invoke: command dispatched + OK observation");

    /* pronoun: 'mach es wieder aus' hits the remembered entity */
    invoke_cmd("Mach es wieder aus", sizeof out, out);
    fails += geist_expect(strcmp(g_last_service, "turn_off") == 0 &&
                                  strcmp(g_last_entity, "light.flur") == 0,
                          "invoke: pronoun resolves via last_entity");

    /* ambiguity: clarifying observation, nothing dispatched */
    g_last_service[0] = '\0';
    invoke_cmd("Schalte das Licht ein", sizeof out, out);
    fails += geist_expect(g_last_service[0] == '\0' && strstr(out, "Mehrere Geräte") != nullptr &&
                                  strstr(out, "light.flur") != nullptr &&
                                  strstr(out, "light.bad") != nullptr,
                          "invoke: ambiguity asks back, does not act");

    /* safety: lock in the registry is still refused */
    invoke_cmd("Schließ die Haustür auf", sizeof out, out);
    fails += geist_expect(g_last_service[0] == '\0' && strstr(out, "Abgelehnt") != nullptr,
                          "invoke: lock refused by the write-whitelist");

    /* status invoke: sensor value with unit */
    {
        char   args[] = "{\"request\":\"Wie warm ist es im Bad?\"}";
        size_t n      = 0;
        (void) home_status_invoke(&ctx, strlen(args), args, sizeof out, out, &n);
        fails += geist_expect(strstr(out, "19.5") != nullptr && strstr(out, "°C") != nullptr,
                              "status: sensor value + unit");
    }

    if (fails) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("agent_home: registry + matching + actions + pronoun + safety pass\n");
    return GEIST_TEST_PASS;
}

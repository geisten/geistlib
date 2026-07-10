/* test_ha_rest_unit — the pure (no-network) core of the HA REST lib:
 * URL building, service-call bodies, and flat JSON field extraction against
 * real /api/states response shapes. The transport (ha_curl) is covered by the
 * live smoke against a real Home Assistant, not here. */
#define _POSIX_C_SOURCE 200809L
#include "test_helpers.h"
#include "../tools/ha_rest.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;

int main(void) {
    char out[512];

    /* URL building */
    fails += geist_expect(ha_service_url("http://pi:8123", "light", "turn_on", sizeof out, out) >
                                          0 &&
                                  strcmp(out, "http://pi:8123/api/services/light/turn_on") == 0,
                          "url: service");
    fails += geist_expect(ha_state_url("http://pi:8123", "light.flur", sizeof out, out) > 0 &&
                                  strcmp(out, "http://pi:8123/api/states/light.flur") == 0,
                          "url: state");
    char tiny[8];
    fails += geist_expect(ha_service_url("http://pi:8123", "light", "turn_on", sizeof tiny, tiny) ==
                                  0,
                          "url: overflow yields 0");

    /* body building */
    fails += geist_expect(ha_service_body("light.flur", nullptr, sizeof out, out) > 0 &&
                                  strcmp(out, "{\"entity_id\":\"light.flur\"}") == 0,
                          "body: plain entity");
    fails += geist_expect(
            ha_service_body("climate.bad", "\"temperature\":21.5", sizeof out, out) > 0 &&
                    strcmp(out, "{\"entity_id\":\"climate.bad\",\"temperature\":21.5}") == 0,
            "body: extra field merged");

    /* JSON extraction against a realistic /api/states response */
    const char *resp = "{\"entity_id\":\"climate.bad\",\"state\":\"heat\",\"attributes\":"
                       "{\"current_temperature\":19.5,\"temperature\":21.0,"
                       "\"friendly_name\":\"Thermostat Bad\",\"unit_of_measurement\":\"\\u00b0C\"},"
                       "\"last_changed\":\"2026-07-10T14:00:00+00:00\"}";
    fails += geist_expect(ha_json_str(resp, "state", sizeof out, out) > 0 &&
                                  strcmp(out, "heat") == 0,
                          "json: string field");
    fails += geist_expect(ha_json_str(resp, "current_temperature", sizeof out, out) > 0 &&
                                  strcmp(out, "19.5") == 0,
                          "json: bare number field");
    fails += geist_expect(ha_json_str(resp, "friendly_name", sizeof out, out) > 0 &&
                                  strcmp(out, "Thermostat Bad") == 0,
                          "json: nested attribute reachable flat");
    fails += geist_expect(ha_json_str(resp, "nonexistent", sizeof out, out) == 0,
                          "json: absent key yields 0");

    /* off/on states from a light response */
    const char *light = "{\"entity_id\":\"light.flur\",\"state\":\"off\",\"attributes\":"
                        "{\"friendly_name\":\"Flurlicht\"}}";
    fails += geist_expect(ha_json_str(light, "state", sizeof out, out) > 0 &&
                                  strcmp(out, "off") == 0,
                          "json: light state off");

    if (fails) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("ha_rest: urls + bodies + flat json extraction pass\n");
    return GEIST_TEST_PASS;
}

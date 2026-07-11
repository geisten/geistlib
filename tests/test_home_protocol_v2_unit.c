#include "test_helpers.h"

#include "tools/home_protocol_v2.h"

static int
expect_status(enum geist_home_v2_status got, enum geist_home_v2_status want, const char *what) {
    if (got == want) {
        return 0;
    }
    fprintf(stderr,
            "FAIL: %s: got %s, want %s\n",
            what,
            geist_home_v2_status_string(got),
            geist_home_v2_status_string(want));
    return 1;
}

static int
decode_json(const char *json, enum geist_home_v2_status want, struct geist_home_v2_frame *frame) {
    const size_t len  = strlen(json);
    uint8_t     *wire = malloc(GEIST_HOME_V2_PREFIX_BYTES + len);
    if (wire == NULL) {
        return GEIST_TEST_ERROR;
    }
    wire[0] = (uint8_t) (len >> 24u);
    wire[1] = (uint8_t) (len >> 16u);
    wire[2] = (uint8_t) (len >> 8u);
    wire[3] = (uint8_t) len;
    memcpy(wire + GEIST_HOME_V2_PREFIX_BYTES, json, len);
    size_t                    consumed = 99u;
    enum geist_home_v2_status got =
            geist_home_v2_decode(wire, GEIST_HOME_V2_PREFIX_BYTES + len, frame, &consumed);
    int fails = expect_status(got, want, json);
    if (got != GEIST_HOME_V2_OK) {
        fails += geist_expect(consumed == 0u, "failed decode consumes no bytes");
    }
    free(wire);
    return fails;
}

int main(void) {
    int fails = 0;

    /* Golden hello frame: prefix is the exact network-order JSON byte count. */
    static const char hello[] = "{\"version\":2,\"request_id\":\"req-1\",\"type\":\"hello\","
                                "\"capabilities\":[\"health\"]}";
    uint8_t           wire[sizeof hello + GEIST_HOME_V2_PREFIX_BYTES] = {0};
    size_t            written                                         = 0u;
    enum geist_home_v2_status status                                  = geist_home_v2_encode(
            (const uint8_t *) hello, strlen(hello), wire, sizeof wire, &written);
    fails += expect_status(status, GEIST_HOME_V2_OK, "encode golden hello");
    fails += geist_expect(written == GEIST_HOME_V2_PREFIX_BYTES + strlen(hello),
                          "golden encoded length");
    fails += geist_expect(wire[0] == 0u && wire[1] == 0u && wire[2] == 0u &&
                                  wire[3] == strlen(hello),
                          "golden big-endian prefix");
    fails += geist_expect(memcmp(wire + GEIST_HOME_V2_PREFIX_BYTES, hello, strlen(hello)) == 0,
                          "golden payload unchanged");

    struct geist_home_v2_frame frame    = {0};
    size_t                     consumed = 0u;
    status                              = geist_home_v2_decode(wire, written, &frame, &consumed);
    fails += expect_status(status, GEIST_HOME_V2_OK, "decode golden hello");
    fails += geist_expect(consumed == written, "golden frame consumed exactly");
    fails += geist_expect(frame.version == 2u, "golden version");
    fails += geist_expect(frame.type == GEIST_HOME_V2_TYPE_HELLO, "golden type");
    fails += geist_expect(strcmp(frame.request_id, "req-1") == 0, "golden request id");

    /* Multiple frames may share one receive buffer; decode consumes one. */
    uint8_t pair[2u * sizeof wire];
    memcpy(pair, wire, written);
    memcpy(pair + written, wire, written);
    consumed = 0u;
    status   = geist_home_v2_decode(pair, 2u * written, &frame, &consumed);
    fails += expect_status(status, GEIST_HOME_V2_OK, "decode first of two frames");
    fails += geist_expect(consumed == written, "only first frame consumed");

    /* Every specified frame type has a stable string and round-trips. */
    for (unsigned i = 0u; i <= (unsigned) GEIST_HOME_V2_TYPE_CANCEL; i++) {
        const char *name = geist_home_v2_type_string((enum geist_home_v2_type) i);
        char        json[192];
        int         n = snprintf(json,
                                 sizeof json,
                                 "{\"type\":\"%s\",\"request_id\":\"roundtrip_%u\",\"version\":2}",
                                 name,
                                 i);
        fails += geist_expect(n > 0 && (size_t) n < sizeof json, "roundtrip JSON fits");
        memset(&frame, 0, sizeof frame);
        fails += decode_json(json, GEIST_HOME_V2_OK, &frame);
        fails += geist_expect(frame.type == (enum geist_home_v2_type) i, "roundtrip type");
    }

    /* Streaming boundaries. */
    consumed   = 12u;
    frame.json = (const uint8_t *) "stale";
    status     = geist_home_v2_decode(wire, 3u, &frame, &consumed);
    fails += expect_status(status, GEIST_HOME_V2_NEED_MORE, "partial prefix");
    fails += geist_expect(consumed == 0u, "partial prefix consumes nothing");
    fails += geist_expect(frame.json == NULL, "partial prefix clears frame");
    status = geist_home_v2_decode(wire, written - 1u, &frame, &consumed);
    fails += expect_status(status, GEIST_HOME_V2_NEED_MORE, "partial payload");
    uint8_t tiny[8] = {0};
    status          = geist_home_v2_encode(
            (const uint8_t *) hello, strlen(hello), tiny, sizeof tiny, &written);
    fails += expect_status(status, GEIST_HOME_V2_NEED_MORE, "small destination");
    fails += geist_expect(written == 0u, "small destination writes no reported frame");

    /* Envelope failures are stable and fail closed. */
    fails += decode_json("{}", GEIST_HOME_V2_E_MISSING_FIELD, &frame);
    fails += decode_json("{\"version\":3,\"request_id\":\"x\",\"type\":\"hello\"}",
                         GEIST_HOME_V2_E_UNSUPPORTED_VERSION,
                         &frame);
    fails += decode_json("{\"version\":2,\"request_id\":\"x\",\"type\":\"execute.shell\"}",
                         GEIST_HOME_V2_E_UNKNOWN_TYPE,
                         &frame);
    fails += decode_json("{\"version\":2,\"request_id\":\"\",\"type\":\"hello\"}",
                         GEIST_HOME_V2_E_INVALID_FIELD,
                         &frame);
    fails += decode_json("{\"version\":2,\"request_id\":\"bad id\",\"type\":\"hello\"}",
                         GEIST_HOME_V2_E_INVALID_FIELD,
                         &frame);
    fails += decode_json("{\"version\":2,\"version\":2,\"request_id\":\"x\",\"type\":\"hello\"}",
                         GEIST_HOME_V2_E_INVALID_FIELD,
                         &frame);
    fails += decode_json("{\"version\":02,\"request_id\":\"x\",\"type\":\"hello\"}",
                         GEIST_HOME_V2_E_INVALID_FIELD,
                         &frame);
    fails += decode_json("{\"version\":2,\"request_id\":\"x\",\"type\":\"hello\",}",
                         GEIST_HOME_V2_E_INVALID_JSON,
                         &frame);
    fails += decode_json("{\"version\":2,\"request_id\":\"x\",\"type\":\"hello\"} trailing",
                         GEIST_HOME_V2_E_INVALID_JSON,
                         &frame);
    fails += decode_json("{\"version\":2,\"request_id\":\"x\",\"type\":\"hello\",\"payload\":01}",
                         GEIST_HOME_V2_E_INVALID_JSON,
                         &frame);
    fails += decode_json(
            "{\"version\":2,\"request_id\":\"x\",\"type\":\"hello\",\"payload\":\"\\q\"}",
            GEIST_HOME_V2_E_INVALID_JSON,
            &frame);

    /* Length is checked before payload access or allocation. */
    uint8_t oversized_prefix[] = {0x00u, 0x10u, 0x00u, 0x01u};
    status = geist_home_v2_decode(oversized_prefix, sizeof oversized_prefix, &frame, &consumed);
    fails += expect_status(status, GEIST_HOME_V2_E_FRAME_TOO_LARGE, "oversized prefix");
    uint8_t empty_prefix[] = {0u, 0u, 0u, 0u};
    status = geist_home_v2_decode(empty_prefix, sizeof empty_prefix, &frame, &consumed);
    fails += expect_status(status, GEIST_HOME_V2_E_INVALID_JSON, "empty payload");

    static const uint8_t invalid_utf8_json[] = {
            '{', '"', 'v', 'e', 'r', 's', 'i', 'o', 'n', '"', ':', '2',   ',',   '"', 'r', 'e',
            'q', 'u', 'e', 's', 't', '_', 'i', 'd', '"', ':', '"', 0xc0u, 0xafu, '"', ',', '"',
            't', 'y', 'p', 'e', '"', ':', '"', 'h', 'e', 'l', 'l', 'o',   '"',   '}',
    };
    uint8_t      invalid_utf8_wire[GEIST_HOME_V2_PREFIX_BYTES + sizeof invalid_utf8_json];
    const size_t invalid_utf8_len = sizeof invalid_utf8_json;
    invalid_utf8_wire[0]          = (uint8_t) (invalid_utf8_len >> 24u);
    invalid_utf8_wire[1]          = (uint8_t) (invalid_utf8_len >> 16u);
    invalid_utf8_wire[2]          = (uint8_t) (invalid_utf8_len >> 8u);
    invalid_utf8_wire[3]          = (uint8_t) invalid_utf8_len;
    memcpy(invalid_utf8_wire + GEIST_HOME_V2_PREFIX_BYTES, invalid_utf8_json, invalid_utf8_len);
    frame.json = (const uint8_t *) "stale";
    status = geist_home_v2_decode(invalid_utf8_wire, sizeof invalid_utf8_wire, &frame, &consumed);
    fails += expect_status(status, GEIST_HOME_V2_E_INVALID_JSON, "invalid UTF-8");
    fails += geist_expect(frame.json == NULL && frame.request_id[0] == '\0',
                          "failed envelope clears frame");

    fails += geist_expect(
            strcmp(geist_home_v2_status_string(GEIST_HOME_V2_E_UNKNOWN_TYPE), "unknown_type") == 0,
            "stable status string");
    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}

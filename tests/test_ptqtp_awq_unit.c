/*
 * test_ptqtp_awq_unit — ptqtp_awq_open on an AWQS file and on every way it
 * refuses one (#349): missing file, too short, bad magic, bad version, and
 * every proper prefix of a valid file. Each refusal returns nullptr with a
 * reason and, under ASan, leaks nothing; the valid file's scales come back.
 */
#include "test_helpers.h"

#include "ptqtp_awq.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static char path[] = "/tmp/geist_awq_XXXXXX";

static void put_u32(uint8_t *p, uint32_t v) {
    memcpy(p, &v, 4);
}

static bool write_file(size_t n, const uint8_t *data) { /* n may be 0 */
    FILE *f = fopen(path, "wb");
    if (f == nullptr)
        return false;
    const bool ok = fwrite(data, 1, n, f) == n;
    return fclose(f) == 0 && ok;
}

/* Opens what is at `path`; a refusal must come with a reason. */
static int refused(const char *what) {
    const char           *err = nullptr;
    struct ptqtp_awq_ctx *a   = ptqtp_awq_open(path, &err);
    const int             bad = a != nullptr || err == nullptr;
    ptqtp_awq_close(a);
    return geist_expect(!bad, what);
}

int main(void) {
    const int fd = mkstemp(path);
    if (fd < 0)
        GEIST_SKIP("mkstemp failed");
    close(fd);

    /* "AWQS", version 1, one norm "n" with two scales. */
    uint8_t     file[12 + 4 + 1 + 4 + 8];
    const float sc[2] = {0.5f, 2.0f};
    memcpy(file, "AWQS", 4);
    put_u32(file + 4, 1);
    put_u32(file + 8, 1);
    put_u32(file + 12, 1);
    file[16] = 'n';
    put_u32(file + 17, 2);
    memcpy(file + 21, sc, sizeof sc);

    int fails = 0;
    if (!write_file(sizeof file, file))
        GEIST_SKIP("cannot write the temporary file");
    const char           *err = nullptr;
    struct ptqtp_awq_ctx *a   = ptqtp_awq_open(path, &err);
    size_t                n   = 0;
    const float          *s   = a != nullptr ? ptqtp_awq_get(a, "n", &n) : nullptr;
    fails += geist_expect(s != nullptr && n == 2 && s[0] == 0.5f && s[1] == 2.0f,
                          "a valid file's scales come back");
    fails += geist_expect(a != nullptr && ptqtp_awq_get(a, "m", &n) == nullptr,
                          "an unknown name is not found");
    ptqtp_awq_close(a);

    for (size_t len = 0; len < sizeof file; len++) {
        char what[64];
        snprintf(what, sizeof what, "a %zu-byte prefix is refused", len);
        fails += write_file(len, file) ? refused(what) : 1;
    }
    file[0] = 'X';
    fails += write_file(sizeof file, file) ? refused("bad magic is refused") : 1;
    file[0] = 'A';
    put_u32(file + 4, 2);
    fails += write_file(sizeof file, file) ? refused("version 2 is refused") : 1;

    unlink(path);
    fails += refused("a missing file is refused");

    if (fails == 0)
        printf("PASS: ptqtp_awq_open reads a valid file and refuses every broken one\n");
    return fails ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
}

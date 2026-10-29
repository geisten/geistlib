#define _POSIX_C_SOURCE 200809L
/*
 * test_env_unit — geist_env_flag / geist_env_count (env.h): one rule for
 * every GEIST_* switch. Unset gives the default; empty or "0" is off; any
 * other value is on.
 */
#include "src/base/env.h"
#include "test_helpers.h"

#include <stdio.h>
#include <stdlib.h>

static constexpr char NAME[] = "GEIST_TEST_ENV_UNIT";

static bool flag_with(const char *value, bool dflt) {
    if (value == nullptr) {
        unsetenv(NAME);
    } else {
        setenv(NAME, value, 1);
    }
    return geist_env_flag(NAME, dflt);
}

static int count_with(const char *value) {
    if (value == nullptr) {
        unsetenv(NAME);
    } else {
        setenv(NAME, value, 1);
    }
    return geist_env_count(NAME);
}

int main(void) {
    int fails = 0;
    fails += geist_expect(!flag_with(nullptr, false) && flag_with(nullptr, true), "unset: default");
    fails += geist_expect(!flag_with("", true), "empty: off, even with default on");
    fails += geist_expect(!flag_with("0", true), "\"0\": off");
    fails += geist_expect(flag_with("1", false), "\"1\": on");
    fails += geist_expect(flag_with("yes", false), "\"yes\": on");
    fails += geist_expect(flag_with("2", false), "\"2\": on");
    fails += geist_expect(flag_with("00", false), "\"00\": on (only \"0\" is off)");
    fails += geist_expect(flag_with("no", false), "\"no\": on (only \"0\" is off)");

    fails += geist_expect(count_with(nullptr) == -1, "count unset: -1");
    fails += geist_expect(count_with("") == -1, "count empty: -1");
    fails += geist_expect(count_with("6") == 6, "count 6");
    fails += geist_expect(count_with("0") == 0, "count 0: not a count");
    fails += geist_expect(count_with("-2") == 0, "count -2: not a count");
    fails += geist_expect(count_with("4x") == 0, "count 4x: not a count");
    fails += geist_expect(count_with("99999999999") == 0, "count past INT_MAX: not a count");
    unsetenv(NAME);
    if (fails == 0)
        printf("PASS: GEIST_* flags and counts follow env.h's rule\n");
    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}

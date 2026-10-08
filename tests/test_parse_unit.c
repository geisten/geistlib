/*
 * test_parse_unit — geist_parse_long (parse.h) takes a whole decimal long
 * and rejects what atoi would have quietly turned into a number.
 */
#include "src/base/parse.h"
#include "test_helpers.h"

#include <stdio.h>

static bool rejects(const char *s) {
    long v = 42;
    return !geist_parse_long(s, &v) && v == 42;
}

int main(void) {
    int  fails = 0;
    long v     = 0;
    fails += geist_expect(geist_parse_long("17", &v) && v == 17, "17");
    fails += geist_expect(geist_parse_long("-3", &v) && v == -3, "-3");
    fails += geist_expect(geist_parse_long("0", &v) && v == 0, "0");
    fails += geist_expect(rejects(nullptr), "unset");
    fails += geist_expect(rejects(""), "empty");
    fails += geist_expect(rejects("abc"), "a word, which atoi reads as 0");
    fails += geist_expect(rejects("4x"), "trailing junk, which atoi reads as 4");
    fails += geist_expect(rejects("99999999999999999999999"), "out of range, undefined for atoi");
    if (fails == 0)
        printf("PASS: geist_parse_long takes whole decimal longs, rejects the rest\n");
    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}

/*
 * tests/static/seeded_defects.c - defects `make check-static` must find (#342).
 *
 * Never compiled into anything; scripts/check_static.py --selftest runs
 * clang-tidy on it and fails unless each `// expect:` check is reported.
 */
#include <stddef.h>
#include <stdlib.h>

int seeded_null_dereference(void) {
    int *p = nullptr;
    return *p; // expect: clang-analyzer-core.NullDereference
}

int seeded_use_after_free(void) {
    int *p = malloc(sizeof *p);
    if (p == nullptr)
        return 0;
    *p = 1;
    free(p);
    return *p; // expect: clang-analyzer-unix.Malloc
}

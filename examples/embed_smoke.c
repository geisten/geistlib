/*
 * embed_smoke.c — the smallest possible "the libgeist SDK links" check.
 *
 * Calls only @stability STABLE, model-free entry points (version + status), so
 * it links against libgeist.a with nothing but -lm — no backend, no model, no
 * OpenMP/BLAS. The release workflow compiles this against the *packaged*
 * `libgeist-<platform>.tar.gz` (headers + archive) to prove the SDK artifact is
 * self-consistent. For real embedding (backend -> model -> session -> generate)
 * see simple_generate.c.
 *
 *   cc -Iinclude examples/embed_smoke.c lib/.../libgeist.a -lm -o smoke && ./smoke
 */
#include <geist.h>

#include <stdio.h>

int main(void) {
    int major = 0, minor = 0, patch = 0;
    geist_version_components(&major, &minor, &patch);
    printf("libgeist %s (%d.%d.%d) — %s\n",
           geist_version_string(),
           major,
           minor,
           patch,
           geist_status_to_string(GEIST_OK));
    return 0;
}

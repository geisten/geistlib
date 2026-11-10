/* rowinv_mb: decode GEMV timings (m = 1, the whole team) of cpu_x86 at Llama
 * 3.2 3B's shapes, best of 7 blocks of 50 calls. Built against a revision's
 * libgeist.a by the probe workflow. */
#include "test_helpers.h"
#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>
#include "heap.h"
#include "par.h"
#include "quant_fixtures.h"
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (double) t.tv_sec + 1e-9 * (double) t.tv_nsec; }
static size_t di(enum geist_dtype dt) { for (size_t d = 0; d < N_DTYPES; d++) if (DTYPES[d].dt == dt) return d; return 0; }
int main(int argc, char **argv) {
    const char *bname = argc > 1 ? argv[1] : "cpu_x86";
    struct geist_backend *be; if (geist_backend_create(bname, nullptr, nullptr, &be) != GEIST_OK) return 1;
    struct { const char *n; enum geist_dtype dt; size_t ni, no; size_t m; } S[] = {
        {"Q4_K 3072->8192 (gate/up)", GEIST_DTYPE_Q4_K, 3072, 8192, 1},
        {"Q4_K 8192->3072 (down)", GEIST_DTYPE_Q4_K, 8192, 3072, 1},
        {"Q6_K 8192->3072 (down)", GEIST_DTYPE_Q6_K, 8192, 3072, 1},
        {"Q6_K 3072->128256 (lm_head)", GEIST_DTYPE_Q6_K, 3072, 128256, 1},
        {"Q4_K 2048->768 m=4 (expert)", GEIST_DTYPE_Q4_K, 2048, 768, 4},
        {"Q6_K 768->2048 m=4 (expert)", GEIST_DTYPE_Q6_K, 768, 2048, 4},
    };
    float *x = xmalloc(8 * 8192 * sizeof(float)), *y = xmalloc(8 * 128256 * sizeof(float));
    for (size_t i = 0; i < 8 * 8192; i++) x[i] = next_f();
    for (size_t s = 0; s < sizeof S / sizeof S[0]; s++) {
        size_t d = di(S[s].dt), bytes; uint8_t *raw = make_weight(d, S[s].ni, S[s].no, &bytes);
        struct geist_weight w = {.raw = raw, .raw_nbytes = bytes, .n_in = (int32_t) S[s].ni, .n_out = (int32_t) S[s].no, .dtype = (uint16_t) S[s].dt};
        be->desc->vtbl->resolve_weight(be, &w);
        double best = 1e9;
        for (int r = 0; r < 7; r++) {
            const double t0 = now();
            for (int k = 0; k < 50; k++) {
                if (S[s].m == 1) w.linear_m1(x, &w, be, y); else w.linear_mN(S[s].m, x, &w, be, y);
            }
            const double dt = (now() - t0) / 50; if (dt < best) best = dt;
        }
        printf("%-30s %9.1f us\n", S[s].n, best * 1e6);
        if ((w.flags & GEIST_W_AUX_HEAP_OWNED) != 0) { void *aux = (void *) w.aux_fp32; safe_free(&aux); }
        free(raw);
    }
    return 0;
}

/*
 * bench_attention_int8 — attention_int8_via_buffers, the INT8-KV attention
 * core every CPU backend runs (cpu_x86 defaults to the INT8 cache), timed on
 * its own over context lengths and head layouts.
 *
 * Model-level runs mix it with the GEMVs and, on synthetic weights, with
 * expf slow paths from saturated scores; this isolates the kernel. Random
 * int8 K/V with moderate scales and random Q keep the softmax spread out.
 *
 * Layouts: 32/8 heads hd 64 (Llama-3.2-1B), 16/8 hd 128 (Qwen3-0.6B),
 * 15/5 hd 64 (SmolLM2-360M), 8/1 hd 256 (MQA), 32/32 hd 64 (MHA,
 * SmolLM2-1.7B). Decode is n_q = 1 at the end of the context; prefill is
 * one 64-token chunk ending there.
 * Reports the median of the timed calls.
 *
 * Usage: bench_attention_int8 [iters]   (default 20)
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "src/archs/transformer/forward/internal.h"
#include "src/archs/transformer/forward.h"

#include "heap.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec * 1e3 + (double) ts.tv_nsec * 1e-6;
}

static uint32_t g_rng = 0x9E3779B9u;
static uint32_t next_u32(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng;
}

static int cmp_double(const void *a, const void *b) {
    const double x = *(const double *) a, y = *(const double *) b;
    return (x > y) - (x < y);
}

struct layout {
    const char *name;
    size_t      n_q_heads, n_kv_heads, head_dim;
};

static double time_one(const struct layout *l, size_t n_kv, size_t n_q, int iters) {
    const size_t kv_elems = n_kv * l->n_kv_heads * l->head_dim;
    const size_t q_elems  = n_q * l->n_q_heads * l->head_dim;
    int8_t      *k        = heap_alloc_array_aligned(int8_t, kv_elems);
    int8_t      *v        = heap_alloc_array_aligned(int8_t, kv_elems);
    float       *ks       = heap_alloc_array_aligned(float, n_kv * l->n_kv_heads);
    float       *vs       = heap_alloc_array_aligned(float, n_kv * l->n_kv_heads);
    float       *q        = heap_alloc_array_aligned(float, q_elems);
    float       *out      = heap_alloc_array_aligned(float, q_elems);
    double      *t        = heap_alloc_array_aligned(double, (size_t) iters);
    const size_t n_scr    = attention_int8_scratch_floats(l->n_q_heads, l->head_dim);
    float       *scratch  = heap_alloc_array_aligned(float, n_scr);
    double       med      = -1.0;
    if (k != nullptr && v != nullptr && ks != nullptr && vs != nullptr && q != nullptr &&
        out != nullptr && t != nullptr && scratch != nullptr) {
        for (size_t i = 0; i < kv_elems; i++) {
            k[i] = (int8_t) (next_u32() % 255u - 127u);
            v[i] = (int8_t) (next_u32() % 255u - 127u);
        }
        for (size_t i = 0; i < n_kv * l->n_kv_heads; i++) {
            ks[i] = 0.0005f + (float) (next_u32() % 1000u) * 1e-6f;
            vs[i] = 0.0005f + (float) (next_u32() % 1000u) * 1e-6f;
        }
        for (size_t i = 0; i < q_elems; i++) {
            q[i] = ((float) (next_u32() % 2001u) - 1000.0f) * 1e-3f;
        }
        const size_t q_offset = n_kv - n_q;
        attention_int8_via_buffers(n_q,
                                   l->n_q_heads,
                                   l->head_dim,
                                   n_kv,
                                   l->n_kv_heads,
                                   n_scr,
                                   q_offset,
                                   0,
                                   q,
                                   k,
                                   ks,
                                   v,
                                   vs,
                                   out,
                                   scratch); /* warm */
        for (int it = 0; it < iters; it++) {
            const double t0 = now_ms();
            attention_int8_via_buffers(n_q,
                                       l->n_q_heads,
                                       l->head_dim,
                                       n_kv,
                                       l->n_kv_heads,
                                       n_scr,
                                       q_offset,
                                       0,
                                       q,
                                       k,
                                       ks,
                                       v,
                                       vs,
                                       out,
                                       scratch);
            t[it] = now_ms() - t0;
        }
        qsort(t, (size_t) iters, sizeof *t, cmp_double);
        med = t[iters / 2];
    }
    safe_free((void **) &k);
    safe_free((void **) &v);
    safe_free((void **) &ks);
    safe_free((void **) &vs);
    safe_free((void **) &q);
    safe_free((void **) &out);
    safe_free((void **) &t);
    safe_free((void **) &scratch);
    return med;
}

int main(int argc, char **argv) {
    const int                  iters = argc > 1 && atoi(argv[1]) > 0 ? atoi(argv[1]) : 20;
    static const struct layout L[]   = {
            {"llama32-1b 32/8 hd64", 32, 8, 64},
            {"qwen3-0.6b 16/8 hd128", 16, 8, 128},
            {"smollm2 15/5 hd64", 15, 5, 64},
            {"mqa 8/1 hd256", 8, 1, 256},
            {"mha 32/32 hd64", 32, 32, 64},
    };
    static const size_t CTX[] = {512, 1024, 2048, 8192};
    printf("{\"bench\":\"attention_int8\",\"iters\":%d,\"rows\":[\n", iters);
    bool first = true;
    for (size_t li = 0; li < sizeof L / sizeof *L; li++) {
        for (size_t ci = 0; ci < sizeof CTX / sizeof *CTX; ci++) {
            const double dec = time_one(&L[li], CTX[ci], 1, iters);
            const double pre = CTX[ci] >= 64 ? time_one(&L[li], CTX[ci], 64, iters / 4 + 1) : -1.0;
            printf("%s {\"layout\":\"%s\",\"ctx\":%zu,\"decode_ms\":%.4f,\"prefill64_ms\":%.3f}",
                   first ? " " : ",\n ",
                   L[li].name,
                   CTX[ci],
                   dec,
                   pre);
            first = false;
        }
    }
    printf("\n]}\n");
    return 0;
}

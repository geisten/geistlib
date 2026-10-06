/* Exact selected-tile versus ordinary resolved head kernels, including tails. */
#include "test_helpers.h"
#include "quant.h"
#include "heap.h"
#include <geist_backend.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static enum geist_status (*borrowed_resolver)(struct geist_backend *, struct geist_weight *);
[[nodiscard]] static enum geist_status forwarding_resolver(struct geist_backend *be,
                                                           struct geist_weight  *w) {
    return borrowed_resolver(be, w);
}

[[nodiscard]] static int borrowed_capability(struct geist_backend *be) {
    /* Reproduce cpu_x86's resolver delegation on every host, even when
     * the host cannot run x86 code. Borrowed kernels are later rebound. */
    struct geist_backend_vtbl       v       = *be->desc->vtbl;
    struct geist_backend_descriptor desc    = *be->desc;
    struct geist_backend            wrapper = *be;
    borrowed_resolver                       = v.resolve_weight;
    v.resolve_weight                        = forwarding_resolver;
    desc.vtbl                               = &v;
    desc.name                               = "test_borrowed_scalar";
    wrapper.desc                            = &desc;
    const float         raw[8]              = {1, 2, 3, 4, 5, 6, 7, 8};
    struct geist_weight w                   = {
            .raw = raw, .raw_nbytes = sizeof raw, .n_in = 4, .n_out = 2, .dtype = GEIST_DTYPE_F32};
    int fails = geist_expect(borrowed_resolver(be, &w) == GEIST_OK && w.linear_rows != nullptr,
                             "native Scalar rows available");
    fails += geist_expect(v.resolve_weight(&wrapper, &w) == GEIST_OK && w.linear_rows == nullptr &&
                                  w.linear_rows_tile == 0 && w.linear_rows_prepare == nullptr,
                          "delegating backend cannot inherit stale Scalar row capability");
    return fails;
}

static uint32_t rng = 42;
static uint8_t  byte(void) {
    rng = rng * 1664525u + 1013904223u;
    return (uint8_t) (rng >> 24);
}
static uint64_t now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t) t.tv_sec * 1000000000ULL + (uint64_t) t.tv_nsec;
}

static int run(struct geist_backend *be, enum geist_dtype dtype, size_t h, size_t vocab) {
    const struct geist_backend_vtbl *v  = be->desc->vtbl;
    struct geist_buffer             *bw = nullptr, *bx = nullptr, *bd = nullptr, *bs = nullptr;
    struct geist_weight              w      = {0};
    size_t                           stride = 0;
    if (quant_raw_bytes(dtype, h, &stride))
        return geist_expect(false, "fixture stride");
    const size_t bytes = stride * vocab;
    int          fails = 0;
    if (v->buffer_create(be, bytes, GEIST_BUFFER_WEIGHT, GEIST_MEMORY_HOST_VISIBLE, &bw) !=
                GEIST_OK ||
        v->buffer_create(
                be, h * sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_HOST_VISIBLE, &bx) !=
                GEIST_OK ||
        v->buffer_create(
                be, vocab * sizeof(float), GEIST_BUFFER_SCRATCH, GEIST_MEMORY_HOST_VISIBLE, &bd) !=
                GEIST_OK ||
        v->buffer_create(
                be, 4 * 32 * sizeof(float), GEIST_BUFFER_SCRATCH, GEIST_MEMORY_HOST_VISIBLE, &bs) !=
                GEIST_OK) {
        fails = geist_expect(false, "selected fixture buffers");
        goto done;
    }
    uint8_t *blob = v->buffer_map(bw);
    float   *x    = v->buffer_map(bx);
    if (blob == nullptr || x == nullptr) {
        fails++;
        goto done;
    }
    for (size_t i = 0; i < bytes; i++)
        blob[i] = byte();
    if (dtype == GEIST_DTYPE_F16) {
        for (size_t i = 0; i < h * vocab; i++) {
            const uint16_t value =
                    (uint16_t) (0x3000u | (byte() << 2) | ((uint16_t) (byte() & 1) << 15));
            memcpy(blob + i * 2, &value, 2);
        }
    } else if (dtype == GEIST_DTYPE_F32) {
        for (size_t i = 0; i < h * vocab; i++) {
            const float value = ((float) byte() - 127.5f) / 64;
            memcpy(blob + i * 4, &value, 4);
        }
    } else {
        size_t block, block_bytes, tail;
        if (!quant_block_layout(dtype, &block, &block_bytes, &tail)) {
            fails++;
            goto done;
        }
        for (size_t i = 0; i < bytes / block_bytes; i++) {
            const size_t   scale = dtype == GEIST_DTYPE_Q6_K ? 208 : 0;
            const uint16_t half  = (uint16_t) (0x3800u + (byte() & 3u) * 0x400u);
            memcpy(blob + i * block_bytes + scale, &half, 2);
            if (dtype == GEIST_DTYPE_Q4_K || dtype == GEIST_DTYPE_Q5_K) {
                const uint16_t min = 0x3400;
                memcpy(blob + i * block_bytes + 2, &min, 2);
            }
        }
    }
    for (size_t i = 0; i < h; i++)
        x[i] = ((float) byte() - 127.5f) / 32;
    w = (struct geist_weight) {.raw        = blob,
                               .raw_nbytes = bytes,
                               .n_in       = (int32_t) h,
                               .n_out      = (int32_t) vocab,
                               .dtype      = (uint16_t) dtype};
    v->buffer_unmap(bw);
    v->buffer_unmap(bx);
    if (v->resolve_weight(be, &w) != GEIST_OK) {
        fails++;
        goto done;
    }
    const bool required = strcmp(geist_backend_name(be), "cpu_scalar") == 0 ||
                          strcmp(geist_backend_name(be), "metal") == 0 ||
                          dtype == GEIST_DTYPE_F16 || dtype == GEIST_DTYPE_Q8_0;
    fails += geist_expect(!required || w.linear_rows != nullptr,
                          "required source/Metal head supported");
    if (w.linear_rows == nullptr) {
        printf("  %s dtype=%u layout=%u: explicit unsupported\n",
               geist_backend_name(be),
               dtype,
               w.backend_layout);
        goto done;
    }
    if (w.linear_rows_prepare != nullptr) {
        fails += geist_expect(w.linear_rows_prepare(&w, be) == GEIST_OK,
                              "head implementation prepared at creation");
    }
    const size_t tile = w.linear_rows_tile;
    fails += geist_expect(tile > 0 && tile <= 32, "bounded tile width");
    if (tile == 0 || tile > 32)
        goto done;
    const geist_token_t ids[4] = {0,
                                  (geist_token_t) tile,
                                  (geist_token_t) ((vocab / 2) / tile * tile),
                                  (geist_token_t) ((vocab - 1) / tile * tile)};
    struct geist_tensor tx     = {.buffer = bx,
                                  .dtype  = GEIST_DTYPE_F32,
                                  .layout = GEIST_LAYOUT_DENSE,
                                  .ndim   = 1,
                                  .shape  = {(int64_t) h},
                                  .stride = {1}};
    struct geist_tensor tw     = {.buffer = bw,
                                  .dtype  = dtype,
                                  .layout = dtype == GEIST_DTYPE_F32 || dtype == GEIST_DTYPE_F16
                                                    ? GEIST_LAYOUT_DENSE
                                                    : GEIST_LAYOUT_BLOCK_QUANTIZED,
                                  .ndim   = 2,
                                  .shape  = {(int64_t) vocab, (int64_t) h},
                                  .stride = {(int64_t) h, 1}};
    struct geist_tensor td     = {.buffer = bd,
                                  .dtype  = GEIST_DTYPE_F32,
                                  .layout = GEIST_LAYOUT_DENSE,
                                  .ndim   = 1,
                                  .shape  = {(int64_t) vocab},
                                  .stride = {1}};
    struct geist_tensor ts     = {.buffer = bs,
                                  .dtype  = GEIST_DTYPE_F32,
                                  .layout = GEIST_LAYOUT_DENSE,
                                  .ndim   = 2,
                                  .shape  = {4, (int64_t) tile},
                                  .stride = {(int64_t) tile, 1}};
    const struct geist_backend_fused *f = geist_backend_fused_tbl(be);
    if (strcmp(geist_backend_name(be), "metal") == 0) {
        fails += geist_expect(f->linear_t(be, &tx, &w, &tw, 1, &td) == GEIST_OK,
                              "dense device head");
    } else {
        x            = v->buffer_map(bx);
        float *dense = v->buffer_map(bd);
        w.linear_m1(x, &w, be, dense);
        v->buffer_unmap(bx);
        v->buffer_unmap(bd);
    }
    struct geist_tensor bad = tx;
    bad.offset              = SIZE_MAX;
    fails += geist_expect(w.linear_rows(4, ids, &bad, &w, &tw, &ts, be) != GEIST_OK,
                          "selected input offset cannot escape its buffer");
    bad        = ts;
    bad.offset = SIZE_MAX;
    fails += geist_expect(w.linear_rows(4, ids, &tx, &w, &tw, &bad, be) != GEIST_OK,
                          "selected output offset cannot escape its buffer");
    bad          = tx;
    bad.shape[0] = (int64_t) h + 1;
    fails += geist_expect(w.linear_rows(4, ids, &bad, &w, &tw, &ts, be) != GEIST_OK,
                          "selected input extent cannot escape its buffer");
#ifdef _OPENMP
    const int thread_count = omp_get_max_threads();
#endif
    fails += geist_expect(w.linear_rows(4, ids, &tx, &w, &tw, &ts, be) == GEIST_OK,
                          "selected tile dispatch");
    const geist_token_t invalid_ids[4] = {-1, ids[1], ids[2], ids[3]};
    fails += geist_expect(w.linear_rows(4, invalid_ids, &tx, &w, &tw, &ts, be) != GEIST_OK,
                          "invalid tile rejected without escaping its row buffer");
#ifdef _OPENMP
    fails += geist_expect(omp_get_max_threads() == thread_count,
                          "selected kernel restores caller threads even after failure");
#endif
    fails += geist_expect(w.linear_rows(4, ids, &tx, &w, &tw, &ts, be) == GEIST_OK,
                          "selected dispatch recovers after validation failure");
    const float *dense = v->buffer_map(bd), *selected = v->buffer_map(bs);
    for (size_t i = 0; i < 4; i++)
        for (size_t j = 0; j < tile && (size_t) ids[i] + j < vocab; j++) {
            fails += geist_expect(memcmp(&dense[(size_t) ids[i] + j],
                                         &selected[i * tile + j],
                                         sizeof(float)) == 0,
                                  "selected tile bit-identical to dense kernel");
        }
    v->buffer_unmap(bd);
    v->buffer_unmap(bs);
    const size_t allocations = heap_alloc_count();
    fails += geist_expect(w.linear_rows(4, ids, &tx, &w, &tw, &ts, be) == GEIST_OK &&
                                  heap_alloc_count() == allocations,
                          "warm selected tile allocates nothing");
    const uint64_t warm_allocations = heap_alloc_count() - allocations;
    if (getenv("GEIST_BENCH_SELECTED_ROWS") != nullptr) {
        uint64_t selected_ns = 0, dense_ns = 0;
        for (size_t i = 0; i < 10; i++) {
            uint64_t start = now();
            fails += geist_expect(w.linear_rows(4, ids, &tx, &w, &tw, &ts, be) == GEIST_OK,
                                  "timed selected");
            v->buffer_map(bs);
            v->buffer_unmap(bs);
            selected_ns += now() - start;
            start = now();
            if (strcmp(geist_backend_name(be), "metal") == 0) {
                fails += geist_expect(f->linear_t(be, &tx, &w, &tw, 1, &td) == GEIST_OK,
                                      "timed dense");
            } else {
                x        = v->buffer_map(bx);
                float *y = v->buffer_map(bd);
                w.linear_m1(x, &w, be, y);
                v->buffer_unmap(bx);
                v->buffer_unmap(bd);
            }
            v->buffer_map(bd);
            v->buffer_unmap(bd);
            dense_ns += now() - start;
        }
        printf("{\"backend\":\"%s\",\"dtype\":%u,\"hidden\":%zu,\"vocab\":%zu,\"tile\":%zu,"
               "\"selected_rows\":%zu,\"warm_heap_allocations\":%llu,\"dense_mean_ns\":%llu,"
               "\"selected_mean_ns\":%llu}\n",
               geist_backend_name(be),
               dtype,
               h,
               vocab,
               tile,
               4 * tile,
               (unsigned long long) warm_allocations,
               (unsigned long long) (dense_ns / 10),
               (unsigned long long) (selected_ns / 10));
    }
    printf("  %s dtype=%u layout=%u tile=%zu vocab=%zu: exact %s\n",
           geist_backend_name(be),
           dtype,
           w.backend_layout,
           tile,
           vocab,
           fails ? "FAIL" : "PASS");
done:
    if ((w.flags & GEIST_W_AUX_HEAP_OWNED) != 0) {
        void *aux = (void *) w.aux_fp32;
        safe_free(&aux);
    }
    if (bs)
        v->buffer_destroy(be, bs);
    if (bd)
        v->buffer_destroy(be, bd);
    if (bx)
        v->buffer_destroy(be, bx);
    if (bw)
        v->buffer_destroy(be, bw);
    return fails;
}

int main(void) {
    const char            *backends[] = {"cpu_scalar", "cpu_neon", "metal"};
    const enum geist_dtype dtypes[]   = {GEIST_DTYPE_Q8_0,
                                         GEIST_DTYPE_F32,
                                         GEIST_DTYPE_F16,
                                         GEIST_DTYPE_Q4_K,
                                         GEIST_DTYPE_Q6_K,
                                         GEIST_DTYPE_PQ2_0,
                                         GEIST_DTYPE_Q5_K};
    int                    fails      = 0;
    for (size_t b = 0; b < 3; b++) {
        struct geist_backend *be = nullptr;
        if (geist_backend_create(backends[b], nullptr, nullptr, &be) != GEIST_OK)
            continue;
        if (strcmp(backends[b], "cpu_scalar") == 0)
            fails += borrowed_capability(be);
        for (size_t d = 0; d < sizeof dtypes / sizeof dtypes[0]; d++) {
            fails += run(be, dtypes[d], 512, 384);
            fails += run(be, dtypes[d], 1024, 387);
        }
        fails += run(be, GEIST_DTYPE_PQ2_0, 512, 6144);
        fails += run(be, GEIST_DTYPE_PQ2_0, 512, 6150);
        if (getenv("GEIST_BENCH_SELECTED_ROWS") != nullptr &&
            strcmp(backends[b], "cpu_scalar") != 0) {
            fails += run(be, GEIST_DTYPE_Q8_0, 1024, 151936);
            fails += run(be, GEIST_DTYPE_PQ2_0, 5120, 248320);
        }
        geist_backend_destroy(be);
    }
    return fails ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
}

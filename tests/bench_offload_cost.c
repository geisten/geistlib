/*
 * tests/bench_offload_cost — phase 0 of the offload proposal (#78,
 * docs/OFFLOAD.md): the cost model behind the row threshold m_min.
 *
 * For one weight shape (default 5120 x 17408, a Qwen3-14B FFN gate/up) and
 * the dtypes Q4_0, Q4_K, Q8_0 it measures, per batch size m:
 *
 *   cpu      the CPU backend's resolved kernel (linear_m1 / linear_mN in
 *            slices of caps.max_m rows), inside its parallel region
 *   gpu      upload x, fused->linear_t on a VRAM-resident weight, download y:
 *            what one routed linear costs once the weight is on the device
 *   gpu_host the same with the weight in host memory, read by the shaders
 *            over the bus (#466; GEIST_VK_WEIGHT_RESERVE forces the spill):
 *            the v0 route of the proposal
 *   up/down  the activation upload and output download alone
 *
 * and once per dtype the bulk upload of the weight into VRAM (bw), the cost
 * the v1 staging ring pays per chunk. From these:
 *
 *   m_min v0 = smallest m from which gpu_host < cpu
 *   m_min v1 = smallest m from which B/bw + gpu < cpu
 *
 * ("from which": for that m and every larger one measured). Medians over
 * repeats; one JSON line per point on stdout, the summary table on stderr.
 *
 * Env: BENCH_CPU (default "auto"), BENCH_N_IN, BENCH_N_OUT, BENCH_MS
 * (target time per point, default 300). Exits 77 without a Vulkan device.
 * Bench only, not a correctness test (test_backend_vulkan_linear_parity is).
 */
#define _POSIX_C_SOURCE 200809L

#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { N_M = 8, MAX_REPS = 50 };
static const size_t M_GRID[N_M] = {1, 4, 16, 32, 64, 128, 256, 512};

struct qfmt {
    int         dtype;
    const char *name;
    size_t      elems, bytes;
    int         scale_bytes; /* leading f16 fields pinned finite */
};
static const struct qfmt QF[] = {
        {GEIST_DTYPE_Q4_0, "q4_0", 32, 18, 2},
        {GEIST_DTYPE_Q4_K, "q4_K", 256, 144, 4},
        {GEIST_DTYPE_Q8_0, "q8_0", 32, 34, 2},
};

static uint32_t rng_state = 0x9e3779b9u;
static uint8_t  rng_u8(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (uint8_t) (rng_state >> 24);
}

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t) t.tv_sec * 1000000000ull + (uint64_t) t.tv_nsec;
}

static int cmp_u64(const void *a, const void *b) {
    const uint64_t x = *(const uint64_t *) a, y = *(const uint64_t *) b;
    return (x > y) - (x < y);
}

/* Random-but-finite quant blob (as test_backend_vulkan_linear_parity). */
static void fill_blob(size_t n_in, size_t n_out, const struct qfmt *q, uint8_t *dst) {
    const size_t n_blk = n_out * (n_in / q->elems);
    for (size_t b = 0; b < n_blk; b++) {
        uint8_t *blk = dst + b * q->bytes;
        for (size_t i = 0; i < q->bytes; i++)
            blk[i] = rng_u8();
        blk[0] = 0x00, blk[1] = 0x2C; /* d = fp16(1/16): keeps sums in range */
        if (q->scale_bytes >= 4)
            blk[2] = 0x00, blk[3] = 0x28; /* dmin = fp16(1/32) */
    }
}

static struct geist_tensor mat_view(size_t rows, size_t cols, struct geist_buffer *b) {
    return (struct geist_tensor) {.buffer = b,
                                  .dtype  = GEIST_DTYPE_F32,
                                  .layout = GEIST_LAYOUT_DENSE,
                                  .ndim   = 2,
                                  .shape  = {(int64_t) rows, (int64_t) cols},
                                  .stride = {(int64_t) cols, 1}};
}

/* One measured operation. */
struct op {
    enum { OP_CPU, OP_GPU, OP_UP, OP_DOWN, OP_BULK } kind;
    size_t                     m, n_in, n_out;
    struct geist_backend      *be;
    const struct geist_weight *w;
    const float               *x;
    float                     *y;
    struct geist_buffer       *bx, *by, *bw; /* device buffers (GPU kinds) */
    const uint8_t             *blob;
    size_t                     blob_bytes;
};

static bool op_run(const struct op *o) {
    const struct geist_backend_vtbl *v = o->be->desc->vtbl;
    switch (o->kind) {
    case OP_CPU: {
        const size_t cap = o->be->desc->caps.max_m ? o->be->desc->caps.max_m : o->m;
        for (size_t r = 0; r < o->m; r += cap) {
            const size_t n = o->m - r < cap ? o->m - r : cap;
            if (n == 1)
                o->w->linear_m1(o->x + r * o->n_in, o->w, o->be, o->y + r * o->n_out);
            else
                o->w->linear_mN(n, o->x + r * o->n_in, o->w, o->be, o->y + r * o->n_out);
        }
        return true;
    }
    case OP_GPU: {
        const struct geist_backend_fused *f  = geist_backend_fused_tbl(o->be);
        struct geist_tensor               tx = mat_view(o->m, o->n_in, o->bx);
        struct geist_tensor               ty = mat_view(o->m, o->n_out, o->by);
        struct geist_tensor               tw = {.dtype = o->w->dtype};
        return v->buffer_upload(o->bx, o->m * o->n_in * 4, (const uint8_t *) o->x) == GEIST_OK &&
               f->linear_t(o->be, &tx, o->w, &tw, o->m, &ty) == GEIST_OK &&
               v->buffer_download(o->m * o->n_out * 4, (uint8_t *) o->y, o->by) == GEIST_OK;
    }
    case OP_UP:
        return v->buffer_upload(o->bx, o->m * o->n_in * 4, (const uint8_t *) o->x) == GEIST_OK;
    case OP_DOWN:
        return v->buffer_download(o->m * o->n_out * 4, (uint8_t *) o->y, o->by) == GEIST_OK;
    case OP_BULK:
        return v->buffer_upload(o->bw, o->blob_bytes, o->blob) == GEIST_OK;
    }
    return false;
}

/* Median ns of `o`: one warmup, then repeats until target_ns or MAX_REPS
 * (at least 3). 0 on failure. */
static uint64_t measure(uint64_t target_ns, const struct op *o) {
    const struct geist_backend_vtbl *v   = o->be->desc->vtbl;
    const bool                       cpu = o->kind == OP_CPU && v->parallel_region_begin;
    const int                        token =
            cpu ? v->parallel_region_begin(
                          o->be, o->m == 1 ? GEIST_REGION_DECODE_STEP : GEIST_REGION_PREFILL_BATCH)
                : 0;
    uint64_t samples[MAX_REPS];
    size_t   n  = 0;
    bool     ok = op_run(o);
    for (uint64_t t_end = now_ns() + target_ns;
         ok && n < MAX_REPS && (n < 3 || now_ns() < t_end);) {
        const uint64_t t0 = now_ns();
        ok                = op_run(o);
        samples[n++]      = now_ns() - t0;
    }
    if (cpu)
        v->parallel_region_end(o->be, token);
    if (!ok)
        return 0;
    qsort(samples, n, sizeof samples[0], cmp_u64);
    return samples[n / 2];
}

static size_t env_size(const char *name, size_t dflt) {
    const char *s = getenv(name);
    return s != nullptr && *s != '\0' ? (size_t) strtoull(s, nullptr, 10) : dflt;
}

/* Smallest grid m from which t_gpu < t_cpu holds for every larger m;
 * 0 = never. */
static size_t m_min_of(const uint64_t extra_ns,
                       const uint64_t t_cpu[static N_M],
                       const uint64_t t_gpu[static N_M]) {
    size_t m_min = 0;
    for (size_t i = N_M; i-- > 0;) {
        if (t_cpu[i] == 0 || t_gpu[i] == 0 || extra_ns + t_gpu[i] >= t_cpu[i])
            break;
        m_min = M_GRID[i];
    }
    return m_min;
}

int main(void) {
    const char    *cpu_name = getenv("BENCH_CPU") ? getenv("BENCH_CPU") : "auto";
    const size_t   n_in     = env_size("BENCH_N_IN", 5120);
    const size_t   n_out    = env_size("BENCH_N_OUT", 17408);
    const uint64_t target   = env_size("BENCH_MS", 300) * 1000000ull;

    struct geist_backend *cpu = nullptr, *vk = nullptr, *vk_host = nullptr;
    if (geist_backend_create("vulkan", nullptr, nullptr, &vk) != GEIST_OK) {
        printf("SKIP: no Vulkan device\n");
        return 77;
    }
    /* Every weight of this instance spills to host memory (#466). */
    setenv("GEIST_VK_WEIGHT_RESERVE", "1024G", 1);
    const enum geist_status hs = geist_backend_create("vulkan", nullptr, nullptr, &vk_host);
    unsetenv("GEIST_VK_WEIGHT_RESERVE");
    if (hs != GEIST_OK || geist_backend_create(cpu_name, nullptr, nullptr, &cpu) != GEIST_OK) {
        fprintf(stderr, "backend create failed\n");
        return 99;
    }
    if (geist_backend_fused_tbl(vk)->linear_t == nullptr) {
        fprintf(stderr, "vulkan has no linear_t\n");
        return 99;
    }
    fprintf(stderr,
            "offload cost: cpu=%s, weight %zu x %zu, m up to %zu, OMP_NUM_THREADS=%s\n",
            geist_backend_name(cpu),
            n_in,
            n_out,
            M_GRID[N_M - 1],
            getenv("OMP_NUM_THREADS") ? getenv("OMP_NUM_THREADS") : "(unset)");

    const size_t m_top = M_GRID[N_M - 1];
    float       *x     = aligned_alloc(64, m_top * n_in * sizeof(float));
    float       *y     = aligned_alloc(64, m_top * n_out * sizeof(float));
    if (x == nullptr || y == nullptr)
        return 99;
    for (size_t i = 0; i < m_top * n_in; i++)
        x[i] = ((float) rng_u8() - 127.5f) / 128.0f;

    const struct geist_backend_vtbl *vv    = vk->desc->vtbl;
    struct geist_buffer             *bx[2] = {nullptr}, *by[2] = {nullptr};
    struct geist_backend            *gpus[2] = {vk, vk_host};
    for (int g = 0; g < 2; g++) {
        if (vv->buffer_create(
                    gpus[g], m_top * n_in * 4, GEIST_BUFFER_SCRATCH, GEIST_MEMORY_AUTO, &bx[g]) !=
                    GEIST_OK ||
            vv->buffer_create(
                    gpus[g], m_top * n_out * 4, GEIST_BUFFER_SCRATCH, GEIST_MEMORY_AUTO, &by[g]) !=
                    GEIST_OK) {
            fprintf(stderr, "scratch buffers: %s\n", geist_backend_errmsg(gpus[g]));
            return 99;
        }
    }

    fprintf(stderr,
            "%-5s %5s %12s %12s %12s %10s %10s\n",
            "dtype",
            "m",
            "cpu us",
            "gpu us",
            "gpu_host us",
            "up us",
            "down us");
    /* Blobs live to the end: the Vulkan weight registry is keyed by the host
     * pointer, so a freed and reused address would find a stale entry. */
    uint8_t *blobs[sizeof QF / sizeof QF[0]] = {nullptr};
    int      rc                              = 0;
    for (size_t q = 0; q < sizeof QF / sizeof QF[0]; q++) {
        const struct qfmt *f     = &QF[q];
        const size_t       bytes = n_out * (n_in / f->elems) * f->bytes;
        uint8_t           *blob  = aligned_alloc(64, (bytes + 63) / 64 * 64);
        if (blob == nullptr)
            return 99;
        blobs[q] = blob;
        fill_blob(n_in, n_out, f, blob);
        const struct geist_weight base  = {.raw        = blob,
                                           .raw_nbytes = bytes,
                                           .n_in       = (int32_t) n_in,
                                           .n_out      = (int32_t) n_out,
                                           .dtype      = (uint16_t) f->dtype};
        struct geist_weight       w_cpu = base, w_vk = base, w_vh = base;
        if (cpu->desc->vtbl->resolve_weight(cpu, &w_cpu) != GEIST_OK ||
            vv->resolve_weight(vk, &w_vk) != GEIST_OK ||
            vv->resolve_weight(vk_host, &w_vh) != GEIST_OK || w_cpu.linear_mN == nullptr) {
            fprintf(stderr, "%s: resolve failed (%s)\n", f->name, geist_backend_errmsg(vk));
            rc = 1;
            continue;
        }

        /* Bulk upload of the weight into a device-local buffer: bw. */
        struct geist_buffer *bw     = nullptr;
        uint64_t             t_bulk = 0;
        if (vv->buffer_create(vk, bytes, GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE, &bw) ==
            GEIST_OK) {
            t_bulk = measure(target,
                             &(struct op) {.kind       = OP_BULK,
                                           .be         = vk,
                                           .bw         = bw,
                                           .blob       = blob,
                                           .blob_bytes = bytes});
            vv->buffer_destroy(vk, bw);
        }

        uint64_t t_cpu[N_M] = {0}, t_gpu[N_M] = {0}, t_host[N_M] = {0};
        for (size_t i = 0; i < N_M; i++) {
            const size_t m = M_GRID[i];
            struct op    o = {.m = m, .n_in = n_in, .n_out = n_out, .x = x, .y = y};
            o.kind = OP_CPU, o.be = cpu, o.w = &w_cpu;
            t_cpu[i] = measure(target, &o);
            o.kind = OP_GPU, o.be = vk, o.w = &w_vk, o.bx = bx[0], o.by = by[0];
            t_gpu[i]              = measure(target, &o);
            o.kind                = OP_UP;
            const uint64_t t_up   = measure(target, &o);
            o.kind                = OP_DOWN;
            const uint64_t t_down = measure(target, &o);
            o.kind = OP_GPU, o.be = vk_host, o.w = &w_vh, o.bx = bx[1], o.by = by[1];
            t_host[i] = measure(target, &o);
            if (t_cpu[i] == 0 || t_gpu[i] == 0 || t_host[i] == 0)
                rc = 1;
            printf("{\"dtype\":\"%s\",\"n_in\":%zu,\"n_out\":%zu,\"weight_bytes\":%zu,\"m\":%zu,"
                   "\"cpu_ns\":%llu,\"gpu_ns\":%llu,\"gpu_host_ns\":%llu,\"up_ns\":%llu,"
                   "\"down_ns\":%llu,\"bulk_upload_ns\":%llu}\n",
                   f->name,
                   n_in,
                   n_out,
                   bytes,
                   m,
                   (unsigned long long) t_cpu[i],
                   (unsigned long long) t_gpu[i],
                   (unsigned long long) t_host[i],
                   (unsigned long long) t_up,
                   (unsigned long long) t_down,
                   (unsigned long long) t_bulk);
            fprintf(stderr,
                    "%-5s %5zu %12.1f %12.1f %12.1f %10.1f %10.1f\n",
                    f->name,
                    m,
                    t_cpu[i] / 1e3,
                    t_gpu[i] / 1e3,
                    t_host[i] / 1e3,
                    t_up / 1e3,
                    t_down / 1e3);
        }
        const double bw_gbs   = t_bulk ? (double) bytes / (double) t_bulk : 0.0;
        const double host_gbs = t_host[0] ? (double) bytes / (double) t_host[0] : 0.0;
        fprintf(stderr,
                "%s: %.1f MB, bulk upload %.2f GB/s (%.1f ms), shader host read at m=1 %.2f GB/s; "
                "m_min v0 (host-read) = %zu, m_min v1 (stream + resident) = %zu  (0 = never)\n\n",
                f->name,
                bytes / 1e6,
                bw_gbs,
                t_bulk / 1e6,
                host_gbs,
                m_min_of(0, t_cpu, t_host),
                m_min_of(t_bulk, t_cpu, t_gpu));
    }

    for (int g = 0; g < 2; g++) {
        vv->buffer_destroy(gpus[g], bx[g]);
        vv->buffer_destroy(gpus[g], by[g]);
    }
    geist_backend_destroy(vk_host);
    geist_backend_destroy(vk);
    geist_backend_destroy(cpu);
    for (size_t q = 0; q < sizeof QF / sizeof QF[0]; q++)
        free(blobs[q]);
    free(x);
    free(y);
    return rc;
}

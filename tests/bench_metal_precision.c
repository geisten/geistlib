/* Isolated FP16/FP8/INT8 GEMM. Run under the host lock. JSONL contains
 * conversion-only, prepared-GEMM-only and conversion+GEMM GPU and wall
 * times. The same geometry, tile, source values and command batching are
 * used for all precisions. No model or external runtime dependency.
 *
 * bench_metal_precision [M N K [repeats [cycles]]]
 * Shapes are bounded to keep all size products and INT32 sums safe.
 * Default: M=512, N=4096, K=4096, 32 repeats, six interleaved cycles.
 * K=32 probes the block length of Q8_0; strides are padded to 128 elements
 * for all modes to meet FP8 alignment without changing logical K. */
#include "test_helpers.h"
#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef GEIST_BACKEND_METAL
#include "../src/backends/metal/metal_internal.h"
#include "metal_precision_shaders.h"

struct precision_params {
    uint32_t m, n, k, stride;
};

static double cmd_time(struct metal_state *st, void *cmd, const char *selector) {
    union {
        void *raw;
        double (*fn)(void *, void *);
    } msg = {.raw = st->objc_msgSend};
    return msg.fn(cmd, metal_sel_register_name(st, selector));
}

static void *compile(struct metal_state *st) {
    const char *parts[] = {precision_common,
                           precision_convert,
                           precision_gemm_f16,
                           precision_gemm_fp8,
                           precision_gemm_i8,
                           precision_calibrate};
    size_t      bytes   = 1;
    for (size_t i = 0; i < 6; i++)
        bytes += strlen(parts[i]);
    char *source = malloc(bytes);
    if (source == nullptr)
        return nullptr;
    source[0] = '\0';
    for (size_t i = 0; i < 6; i++)
        strcat(source, parts[i]);
    void *ns  = metal_objc_get_class(st, "NSString");
    void *str = metal_msg_send_id_cstr(st, ns, "stringWithUTF8String:", source);
    free(source);
    void *options = metal_msg_send_id0(st, metal_objc_get_class(st, "MTLCompileOptions"), "new");
    metal_msg_send_void_ulong(st, options, "setLanguageVersion:", (4ul << 16) + 1ul);
    void *error   = nullptr;
    void *library = metal_msg_send_id_id_id_err(
            st, st->device, "newLibraryWithSource:options:error:", str, options, &error);
    metal_msg_send_void0(st, options, "release");
    if (library == nullptr)
        fprintf(stderr, "precision shader: %s\n", metal_nserror_message(st, error));
    return library;
}

static void *pipeline(struct metal_state *st, void *library, const char *name) {
    void *str = metal_msg_send_id_cstr(
            st, metal_objc_get_class(st, "NSString"), "stringWithUTF8String:", name);
    void *fn = metal_msg_send_id_id(st, library, "newFunctionWithName:", str), *error = nullptr;
    void *pipe = fn != nullptr
                         ? metal_msg_send_id_id_err(st,
                                                    st->device,
                                                    "newComputePipelineStateWithFunction:error:",
                                                    fn,
                                                    &error)
                         : nullptr;
    metal_msg_send_void0(st, fn, "release");
    if (pipe == nullptr)
        fprintf(stderr, "precision pipeline %s: %s\n", name, metal_nserror_message(st, error));
    return pipe;
}

static bool timed(struct metal_state            *st,
                  int                            phase,
                  unsigned                       repeats,
                  const struct precision_params *p,
                  size_t                         elements,
                  void                          *cvt,
                  void                          *mm,
                  const struct geist_buffer     *src,
                  const struct geist_buffer     *packed,
                  const struct geist_buffer     *out,
                  double                        *gpu_us,
                  double                        *wall_us) {
    [[gnu::cleanup(metal_pool_end)]] struct metal_pool pool  = metal_standalone_pool(st);
    const uint64_t                                     start = geist_now_ns();
    void *cmd = metal_msg_send_id0(st, st->command_queue, "commandBuffer");
    void *enc = cmd != nullptr ? metal_msg_send_id0(st, cmd, "computeCommandEncoder") : nullptr;
    if (enc == nullptr)
        return false;
    const uint32_t n4 = (uint32_t) (elements / 4);
    for (unsigned i = 0; i < repeats; i++) {
        if (phase != 1) {
            metal_msg_send_set_pipeline(st, enc, cvt);
            metal_msg_send_set_buffer(st, enc, src->buffer, src->base_off, 0);
            metal_msg_send_set_buffer(st, enc, packed->buffer, packed->base_off, 1);
            metal_msg_send_set_bytes(st, enc, &n4, sizeof n4, 2);
            if (phase == 3)
                metal_msg_send_set_buffer(st, enc, out->buffer, out->base_off, 3);
            metal_msg_send_dispatch(
                    st,
                    enc,
                    (struct metal_size) {
                            (phase == 3 ? (n4 / 8u + 7u) / 8u : (n4 + 255u) / 256u), 1, 1},
                    (struct metal_size) {256, 1, 1});
        }
        if (phase == 1 || phase == 2) {
            metal_msg_send_set_pipeline(st, enc, mm);
            metal_msg_send_set_buffer(st, enc, packed->buffer, packed->base_off, 0);
            metal_msg_send_set_buffer(st, enc, out->buffer, out->base_off, 1);
            metal_msg_send_set_bytes(st, enc, p, sizeof *p, 2);
            metal_msg_send_dispatch(
                    st,
                    enc,
                    (struct metal_size) {(p->m + 127u) / 128u, (p->n + 63u) / 64u, 1},
                    (struct metal_size) {256, 1, 1});
        }
    }
    metal_msg_send_void0(st, enc, "endEncoding");
    metal_msg_send_void0(st, cmd, "commit");
    metal_msg_send_void0(st, cmd, "waitUntilCompleted");
    *wall_us           = (double) (geist_now_ns() - start) / (1000.0 * repeats);
    const double begin = cmd_time(st, cmd, "GPUStartTime"), end = cmd_time(st, cmd, "GPUEndTime");
    if (metal_msg_send_id0(st, cmd, "error") != nullptr || !(end > begin) || begin <= 0)
        return false;
    *gpu_us = (end - begin) * 1e6 / repeats;
    return true;
}

static bool
verify(unsigned mode, const struct precision_params *p, const float *src, const void *out) {
    /* A fixed sample includes first/last rows and columns. Integer-valued
     * inputs give an exact oracle in FP16, FP8 and INT8, without relying
     * on a second GPU kernel or relaxed precision-dependent tolerance. */
    for (size_t i = 0; i < 32; i++) {
        const size_t r   = i == 0 ? 0 : i == 1 ? p->m - 1 : (i * 73u) % p->m;
        const size_t n   = i == 0 ? 0 : i == 1 ? p->n - 1 : (i * 137u) % p->n;
        int64_t      sum = 0;
        for (size_t k = 0; k < p->k; k++) {
            sum += (int64_t) src[r * p->stride + k] *
                   (int64_t) src[((size_t) p->m + n) * p->stride + k];
        }
        const double got = mode == 2 ? ((const int32_t *) out)[r * p->n + n]
                                     : ((const float *) out)[r * p->n + n];
        if (got != (double) sum) {
            fprintf(stderr,
                    "precision oracle mode=%u row=%zu col=%zu: %.9g != %lld\n",
                    mode,
                    r,
                    n,
                    got,
                    (long long) sum);
            return false;
        }
    }
    return true;
}

static bool verify_calibration(unsigned     mode,
                               size_t       elements,
                               const float *src,
                               const void  *packed,
                               const void  *metadata) {
    const size_t blocks = elements / 32u;
    for (size_t sample = 0; sample < 32; sample++) {
        const size_t b    = sample == 31 ? blocks - 1 : sample * blocks / 32u;
        float        amax = 0;
        for (size_t k = 0; k < 32; k++)
            amax = fmaxf(amax, fabsf(src[b * 32u + k]));
        if (mode == 2) {
            const float d = amax > 0 ? amax / 127.0f : 1.0f;
            if (fabsf(((const float *) metadata)[b] - d) > d * 1e-6f)
                return false;
            int sum = 0;
            for (size_t k = 0; k < 32; k++) {
                const int q = (int) nearbyintf(src[b * 32u + k] / d);
                if (((const int8_t *) packed)[b * 32u + k] != q)
                    return false;
                sum += q;
            }
            if (((const int32_t *) metadata)[blocks + b] != sum)
                return false;
        } else {
            const int e = ((const uint8_t *) metadata)[b] - 127;
            if (e != (amax > 0 ? (int) ceilf(log2f(amax / 448.0f)) : -126))
                return false;
            for (size_t k = 0; k < 32; k++) {
                const uint8_t  q   = ((const uint8_t *) packed)[b * 32u + k];
                const unsigned exp = (q >> 3) & 15u, mant = q & 7u;
                float          value = exp ? ldexpf(1.0f + (float) mant / 8.0f, (int) exp - 7)
                                           : ldexpf((float) mant / 8.0f, -6);
                value                = ldexpf((q & 128u) ? -value : value, e);
                if (value != src[b * 32u + k])
                    return false;
            }
        }
    }
    return true;
}
#endif

int main(int argc, char **argv) {
#ifndef GEIST_BACKEND_METAL
    (void) argc;
    (void) argv;
    return GEIST_TEST_SKIP;
#else
    unsigned values[5] = {512, 4096, 4096, 32, 6};
    if (argc != 1 && (argc < 4 || argc > 6))
        return GEIST_TEST_ERROR;
    for (int i = 1; i < argc; i++) {
        char *end                 = nullptr;
        errno                     = 0;
        unsigned long       value = strtoul(argv[i], &end, 10);
        const unsigned long limit = i <= 3 ? 16384ul : i == 4 ? 256ul : 64ul;
        if (errno || *end || value == 0 || value > limit)
            return GEIST_TEST_ERROR;
        values[i - 1] = (unsigned) value;
    }
    if (values[2] % 32u != 0)
        return GEIST_TEST_ERROR;
    struct precision_params p = {values[0], values[1], values[2], (values[2] + 127u) & ~127u};
    const size_t elements = ((size_t) p.m + p.n) * p.stride, outputs = (size_t) p.m * p.n;
    /* Explicit bound also limits memory and makes worst-case dot sums
     * <2^24, exactly representable in the FP32 destinations. */
    if (elements > 128u * 1024u * 1024u || outputs > 64u * 1024u * 1024u)
        return GEIST_TEST_ERROR;
    struct geist_backend   *be     = nullptr;
    const enum geist_status status = geist_backend_create("metal", nullptr, nullptr, &be);
    if (status == GEIST_E_UNSUPPORTED || status == GEIST_E_NOT_FOUND)
        return GEIST_TEST_SKIP;
    if (status != GEIST_OK)
        return GEIST_TEST_FAIL;
    struct metal_state *st = be->state;
    if (!metal_msg_send_bool_long(st, st->device, "supportsFamily:", 1011)) {
        geist_backend_destroy(be);
        return GEIST_TEST_SKIP;
    }
    void                *pool = metal_pool_push(st), *library = compile(st);
    void                *cvt[3] = {nullptr}, *mm[3] = {nullptr}, *calibrate[3] = {nullptr};
    struct geist_buffer *src = nullptr, *packed[3] = {nullptr}, *out = nullptr, *metadata = nullptr;
    const struct geist_backend_vtbl *v  = be->desc->vtbl;
    int                              rc = GEIST_TEST_FAIL;
    if (library == nullptr ||
        v->buffer_create(be, elements * sizeof(float), GEIST_BUFFER_ACTIVATION, 0, &src) !=
                GEIST_OK ||
        v->buffer_create(be, outputs * sizeof(float), GEIST_BUFFER_ACTIVATION, 0, &out) != GEIST_OK)
        goto cleanup;
    if (v->buffer_create(be, elements / 32u * 8u, GEIST_BUFFER_SCRATCH, 0, &metadata) != GEIST_OK)
        goto cleanup;
    float *data = v->buffer_map(src);
    if (data == nullptr)
        goto cleanup;
    uint32_t rng = 0x54321u;
    for (size_t i = 0; i < elements; i++) {
        rng     = rng * 1664525u + 1013904223u;
        data[i] = i % p.stride < p.k ? (float) ((int) (rng >> 28) - 8) : 0.0f;
    }
    const char *names[3] = {"f16", "fp8", "i8"};
    for (unsigned mode = 0; mode < 3; mode++) {
        char name[32];
        snprintf(name, sizeof name, "cvt_%s", names[mode]);
        cvt[mode] = pipeline(st, library, name);
        if (mode != 0) {
            snprintf(name, sizeof name, "calibrate_%s", names[mode]);
            calibrate[mode] = pipeline(st, library, name);
            if (calibrate[mode] == nullptr)
                goto cleanup;
        }
        snprintf(name, sizeof name, "gemm_%s", names[mode]);
        mm[mode] = pipeline(st, library, name);
        if (cvt[mode] == nullptr || mm[mode] == nullptr ||
            v->buffer_create(be,
                             elements * (mode == 0 ? 2u : 1u),
                             GEIST_BUFFER_ACTIVATION,
                             0,
                             &packed[mode]) != GEIST_OK)
            goto cleanup;
        double gpu = 0, wall = 0;
        if (!timed(st,
                   2,
                   4,
                   &p,
                   elements,
                   cvt[mode],
                   mm[mode],
                   src,
                   packed[mode],
                   out,
                   &gpu,
                   &wall) ||
            !verify(mode, &p, data, v->buffer_map(out)))
            goto cleanup;
    }
    for (unsigned cycle = 0; cycle < values[4]; cycle++) {
        for (unsigned arm = 0; arm < 3; arm++) {
            const unsigned mode = (arm + cycle) % 3;
            for (int phase = 0; phase < (mode == 0 ? 3 : 4); phase++) {
                double gpu = 0, wall = 0;
                if (!timed(st,
                           phase,
                           values[3],
                           &p,
                           elements,
                           phase == 3 ? calibrate[mode] : cvt[mode],
                           mm[mode],
                           src,
                           packed[mode],
                           phase == 3 ? metadata : out,
                           &gpu,
                           &wall))
                    goto cleanup;
                if ((phase == 1 || phase == 2) && !verify(mode, &p, data, v->buffer_map(out)))
                    goto cleanup;
                if (phase == 3 && !verify_calibration(mode,
                                                      elements,
                                                      data,
                                                      v->buffer_map(packed[mode]),
                                                      v->buffer_map(metadata))) {
                    fprintf(stderr, "calibration oracle mode=%u failed\n", mode);
                    goto cleanup;
                }
                printf("{\"mode\":\"%s\",\"phase\":\"%s\",\"cycle\":%u,\"m\":%u,\"n\":%u,\"k\":%u,"
                       "\"stride\":%u,"
                       "\"repeats\":%u,\"gpu_us\":%.6f,\"wall_us\":%.6f,\"operand_bytes\":%zu,"
                       "\"output_bytes\":%zu,"
                       "\"tflops_or_tops\":%.6f,\"oracle\":\"%s\",\"metadata_bytes\":%zu}\n",
                       names[mode],
                       phase == 0   ? "convert"
                       : phase == 1 ? "gemm"
                       : phase == 2 ? "combined"
                                    : "calibrate32",
                       cycle,
                       p.m,
                       p.n,
                       p.k,
                       p.stride,
                       values[3],
                       gpu,
                       wall,
                       elements * (mode == 0 ? 2u : 1u),
                       outputs * sizeof(float),
                       (phase == 0 || phase == 3) ? 0.0
                                                  : 2.0 * (double) p.m * p.n * p.k / (gpu * 1e6),
                       phase == 3   ? "calibrated_32_blocks"
                       : phase == 0 ? "verified_by_following_gemm"
                                    : "exact_32_outputs",
                       phase == 3 ? elements / 32u * (mode == 2 ? 8u : 1u) : 0u);
                fflush(stdout);
            }
        }
    }
    rc = GEIST_TEST_PASS;
cleanup:
    for (unsigned mode = 0; mode < 3; mode++) {
        metal_msg_send_void0(st, cvt[mode], "release");
        metal_msg_send_void0(st, mm[mode], "release");
        metal_msg_send_void0(st, calibrate[mode], "release");
        v->buffer_destroy(be, packed[mode]);
    }
    v->buffer_destroy(be, metadata);
    v->buffer_destroy(be, src);
    v->buffer_destroy(be, out);
    metal_msg_send_void0(st, library, "release");
    metal_pool_pop(st, pool);
    geist_backend_destroy(be);
    return rc;
#endif
}

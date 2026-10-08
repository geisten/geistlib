/*
 * test_quant_dequant_row_unit — quant_dequant_row (quant.h), the one
 * per-dtype row-dequant dispatch, against what the copies it replaced did
 * (#465).
 *
 * The copies — cpu_scalar / Vulkan host path (common/linear_ref.c), the
 * transformer's embedding / PLE row lookup (forward/layer.c), the Metal host
 * fallback, the spec head and formats/gguf's two dispatchers — each decoded
 * F32 by memcpy, F16 / BF16 element by element, I2_S by its reversed 2-bit
 * fields and one per-tensor scale, and every other format with its quant.h
 * row codec at raw + e0 / block_elems * block_bytes. old_decode below is that
 * switch. For every dtype in quant_fixtures.h, random rows must come out
 * bit-identical:
 *   - each whole row, and single-block / multi-block runs inside one;
 *   - the whole tensor in one call;
 *   - through gguf_dequant_row_to_fp32 / gguf_dequant_to_fp32.
 * Runs that are not whole blocks or overrun the tensor, and dtypes with no
 * decoder, return false with out zeroed; quant_dequant_row decodes exactly
 * the dtypes geist_linear_ref_decodes lists.
 */
#include "test_helpers.h"

#include <geist_types.h>

#include "gguf_dequant.h"
#include "gguf_reader.h"
#include "heap.h"
#include "linear_ref.h"
#include "quant.h"
#include "quant_fixtures.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Wide enough for two tiles of every block size (32, 128, 256). */
constexpr size_t N_IN  = 512;
constexpr size_t N_OUT = 6;

/* Elements [e0, e0 + n) of a tensor of n_total elements, decoded the way
 * the replaced copies did. */
static void
old_decode(size_t d, size_t n_total, size_t e0, size_t n, const uint8_t *raw, float *out) {
    switch (DTYPES[d].dt) {
    case GEIST_DTYPE_F32:
        memcpy(out, raw + e0 * sizeof(float), n * sizeof(float));
        return;
    case GEIST_DTYPE_F16:
        for (size_t i = 0; i < n; i++) {
            uint16_t h;
            memcpy(&h, raw + (e0 + i) * 2, sizeof h);
            out[i] = fp16_to_fp32(h);
        }
        return;
    case GEIST_DTYPE_BF16:
        for (size_t i = 0; i < n; i++) {
            uint16_t h;
            memcpy(&h, raw + (e0 + i) * 2, sizeof h);
            const uint32_t u = (uint32_t) h << 16;
            memcpy(&out[i], &u, sizeof u);
        }
        return;
    case GEIST_DTYPE_I2_S: {
        float scale;
        memcpy(&scale, raw + i2_s_scale_offset(n_total), sizeof scale);
        for (size_t i = 0; i < n; i++) {
            /* Element e of a 256-block: byte h*32 + e%32 of its 64, h =
             * (e%256)/128, at shift 6 - 2*((e%128)/32). */
            const size_t  e = e0 + i, r = e % 256;
            const uint8_t byte = raw[(e / 256) * 64 + (r / 128) * 32 + r % 32];
            out[i] = (float) ((int) ((byte >> (6 - 2 * ((r % 128) / 32))) & 3) - 1) * scale;
        }
        return;
    }
    default:
        DTYPES[d].row(n, raw + e0 / DTYPES[d].blk * DTYPES[d].bytes, out);
        return;
    }
}

static gguf_dtype_t gguf_type_of(enum geist_dtype dt) {
    switch (dt) {
    case GEIST_DTYPE_F32:
        return GGUF_TYPE_F32;
    case GEIST_DTYPE_F16:
        return GGUF_TYPE_F16;
    case GEIST_DTYPE_BF16:
        return GGUF_TYPE_BF16;
    case GEIST_DTYPE_Q4_0:
        return GGUF_TYPE_Q4_0;
    case GEIST_DTYPE_Q4_1:
        return GGUF_TYPE_Q4_1;
    case GEIST_DTYPE_Q5_0:
        return GGUF_TYPE_Q5_0;
    case GEIST_DTYPE_Q8_0:
        return GGUF_TYPE_Q8_0;
    case GEIST_DTYPE_Q3_K:
        return GGUF_TYPE_Q3_K;
    case GEIST_DTYPE_Q4_K:
        return GGUF_TYPE_Q4_K;
    case GEIST_DTYPE_Q5_K:
        return GGUF_TYPE_Q5_K;
    case GEIST_DTYPE_Q6_K:
        return GGUF_TYPE_Q6_K;
    case GEIST_DTYPE_IQ2_S:
        return GGUF_TYPE_IQ2_S;
    case GEIST_DTYPE_IQ3_S:
        return GGUF_TYPE_IQ3_S;
    case GEIST_DTYPE_IQ4_NL:
        return GGUF_TYPE_IQ4_NL;
    case GEIST_DTYPE_IQ4_XS:
        return GGUF_TYPE_IQ4_XS;
    case GEIST_DTYPE_TQ2_0:
        return GGUF_TYPE_TQ2_0;
    case GEIST_DTYPE_PQ2_0:
        return GGUF_TYPE_PQ2_0;
    case GEIST_DTYPE_I2_S:
        return GGUF_TYPE_I2_S;
    default:
        return GGUF_TYPE_TQ1_0;
    }
}

static bool all_zero(size_t n, const float *v) {
    for (size_t i = 0; i < n; i++) {
        if (v[i] != 0.0f || signbit(v[i])) {
            return false;
        }
    }
    return true;
}

/* One run through both, compared bitwise. */
static int check_run(size_t d, const uint8_t *raw, size_t e0, size_t n, float *want, float *got) {
    const size_t total = N_IN * N_OUT;
    char         msg[128];
    old_decode(d, total, e0, n, raw, want);
    memset(got, 0xff, n * sizeof *got);
    const bool ok = quant_dequant_row(DTYPES[d].dt, total, e0, n, raw, got);
    snprintf(msg, sizeof msg, "%s e0=%zu n=%zu: decodes", DTYPES[d].name, e0, n);
    int fails = geist_expect(ok, msg);
    snprintf(msg, sizeof msg, "%s e0=%zu n=%zu: bit-identical", DTYPES[d].name, e0, n);
    fails += geist_expect(memcmp(want, got, n * sizeof *got) == 0, msg);
    return fails;
}

static int run_dtype(size_t d) {
    const size_t total = N_IN * N_OUT;
    const size_t blk   = DTYPES[d].blk;
    size_t       bytes = 0;
    uint8_t     *raw   = make_weight(d, N_IN, N_OUT, &bytes);
    float       *want  = malloc(total * sizeof *want);
    float       *got   = malloc(total * sizeof *got);
    if (raw == nullptr || want == nullptr || got == nullptr) {
        fprintf(stderr, "%s: alloc failed\n", DTYPES[d].name);
        free(raw);
        free(want);
        free(got);
        return 1;
    }
    int  fails = 0;
    char msg[128];

    for (size_t j = 0; j < N_OUT; j++) {
        fails += check_run(d, raw, j * N_IN, N_IN, want, got);
        /* One block, then a two-block run, inside the row. */
        const size_t b = (next_u32() % (N_IN / blk - 1));
        fails += check_run(d, raw, j * N_IN + b * blk, blk, want, got);
        fails += check_run(d, raw, j * N_IN + b * blk, 2 * blk, want, got);
    }
    fails += check_run(d, raw, 0, total, want, got);

    /* The GGUF dispatchers: a [N_OUT][N_IN] tensor (dims fastest first). */
    const struct gguf_tensor_t t = {
            .name   = "w",
            .dtype  = gguf_type_of(DTYPES[d].dt),
            .n_dims = 2,
            .dims   = {N_IN, N_OUT},
            .nbytes = bytes,
            .data   = raw,
    };
    old_decode(d, total, 0, total, raw, want);
    for (size_t j = 0; j < N_OUT; j++) {
        snprintf(msg, sizeof msg, "%s: gguf_dequant_row_to_fp32 row %zu", DTYPES[d].name, j);
        fails += geist_expect(gguf_dequant_row_to_fp32(&t, j, N_IN, got) &&
                                      memcmp(want + j * N_IN, got, N_IN * sizeof *got) == 0,
                              msg);
    }
    snprintf(msg, sizeof msg, "%s: gguf_dequant_row_to_fp32 past the last row", DTYPES[d].name);
    fails += geist_expect(!gguf_dequant_row_to_fp32(&t, N_OUT, N_IN, got), msg);
    float *whole = gguf_dequant_to_fp32(&t);
    snprintf(msg, sizeof msg, "%s: gguf_dequant_to_fp32", DTYPES[d].name);
    fails += geist_expect(whole != nullptr && memcmp(want, whole, total * sizeof *want) == 0, msg);
    safe_free((void **) &whole);

    /* Rejections zero the output. */
    if (blk > 1) {
        memset(got, 0xff, blk * sizeof *got);
        snprintf(msg, sizeof msg, "%s: e0 inside a block is refused", DTYPES[d].name);
        fails += geist_expect(!quant_dequant_row(DTYPES[d].dt, total, 1, blk, raw, got) &&
                                      all_zero(blk, got),
                              msg);
        memset(got, 0xff, (blk + 1) * sizeof *got);
        snprintf(msg, sizeof msg, "%s: a partial block is refused", DTYPES[d].name);
        fails += geist_expect(!quant_dequant_row(DTYPES[d].dt, total, 0, blk + 1, raw, got) &&
                                      all_zero(blk + 1, got),
                              msg);
    }
    memset(got, 0xff, N_IN * sizeof *got);
    snprintf(msg, sizeof msg, "%s: a run past the tensor is refused", DTYPES[d].name);
    fails += geist_expect(
            !quant_dequant_row(DTYPES[d].dt, total, total - N_IN + blk, N_IN, raw, got) &&
                    all_zero(N_IN, got),
            msg);

    free(raw);
    free(want);
    free(got);
    return fails;
}

int main(void) {
    int fails = 0;
    for (size_t d = 0; d < N_DTYPES; d++) {
        fails += run_dtype(d);
    }

    /* Exactly the dtypes the reference linear kernel claims, and nothing
     * else: no decoder for I8 / U8 (a raw layout but no f32 meaning here),
     * TQ1_0, BINARY, TERNARY, CUSTOM or any value past the enum (64 covers it). */
    uint8_t blob[4096] = {0};
    float   out[256];
    for (int dt = 0; dt < 64; dt++) {
        memset(out, 0xff, sizeof out);
        const bool ok = quant_dequant_row((enum geist_dtype) dt, 256, 0, 256, blob, out);
        char       msg[96];
        snprintf(msg, sizeof msg, "dtype %d: decodes iff geist_linear_ref_decodes", dt);
        fails += geist_expect(ok == geist_linear_ref_decodes((uint16_t) dt), msg);
        if (!ok) {
            snprintf(msg, sizeof msg, "dtype %d: refused with out zeroed", dt);
            fails += geist_expect(all_zero(256, out), msg);
        }
    }

    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("quant_dequant_row: %d dtypes bit-identical to the replaced copies\n", (int) N_DTYPES);
    return GEIST_TEST_PASS;
}

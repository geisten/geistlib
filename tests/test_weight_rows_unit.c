/*
 * test_weight_rows_unit — resolve_weight refuses a weight whose rows are
 * not whole blocks.
 *
 * Every CPU kernel indexes a quantized weight by row: row j starts at
 * j * (n_in / block_elems) * block_bytes and is decoded block by block. A
 * row that ends inside a block breaks both: the next row starts at the
 * wrong byte and the tail of this one is never decoded. The GGUF reader
 * checks only that the whole tensor is whole blocks, which Q4_0 with
 * n_in = 48 and n_out = 2 (three blocks in all, one and a half per row)
 * passes.
 *
 * For every block-quantized dtype and every CPU backend in the build: rows
 * of one and a half blocks give GEIST_E_FORMAT; a total that is not whole
 * blocks does too; rows of two blocks are not refused as malformed (they
 * resolve, or the backend has no kernel for the dtype).
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>

#include "quant.h"

#include <stdio.h>
#include <stdlib.h>

static const char *const BACKENDS[] = {"cpu_x86", "cpu_neon", "cpu_scalar"};

static const struct {
    enum geist_dtype dt;
    const char      *name;
    size_t           blk; /* elements per block */
} DTYPES[] = {
        {GEIST_DTYPE_Q4_0, "Q4_0", Q4_0_BLOCK_ELEMS},
        {GEIST_DTYPE_Q4_1, "Q4_1", Q4_1_BLOCK_ELEMS},
        {GEIST_DTYPE_Q8_0, "Q8_0", Q8_0_BLOCK_ELEMS},
        {GEIST_DTYPE_Q3_K, "Q3_K", Q3_K_BLOCK_ELEMS},
        {GEIST_DTYPE_Q4_K, "Q4_K", Q4_K_BLOCK_ELEMS},
        {GEIST_DTYPE_Q5_K, "Q5_K", Q5_K_BLOCK_ELEMS},
        {GEIST_DTYPE_Q6_K, "Q6_K", Q6_K_BLOCK_ELEMS},
        {GEIST_DTYPE_IQ2_S, "IQ2_S", IQ2_S_BLOCK_ELEMS},
        {GEIST_DTYPE_IQ3_S, "IQ3_S", IQ3_S_BLOCK_ELEMS},
        {GEIST_DTYPE_IQ4_NL, "IQ4_NL", IQ4_NL_BLOCK_ELEMS},
        {GEIST_DTYPE_IQ4_XS, "IQ4_XS", IQ4_XS_BLOCK_ELEMS},
        {GEIST_DTYPE_TQ2_0, "TQ2_0", TQ2_0_BLOCK_ELEMS},
        {GEIST_DTYPE_PQ2_0, "PQ2_0", PQ2_0_BLOCK_ELEMS},
        {GEIST_DTYPE_I2_S, "I2_S", I2_S_BLOCK_ELEMS},
};

/* resolve_weight on zeroed storage of `bytes` for an [n_out, n_in] weight. */
static enum geist_status
resolve(struct geist_backend *be, enum geist_dtype dt, size_t n_in, size_t n_out, size_t bytes) {
    void *raw = calloc(1, bytes);
    if (raw == nullptr) {
        return GEIST_E_OOM;
    }
    struct geist_weight     w = {.raw        = raw,
                                 .raw_nbytes = bytes,
                                 .n_in       = (int32_t) n_in,
                                 .n_out      = (int32_t) n_out,
                                 .dtype      = (uint16_t) dt};
    const enum geist_status s = be->desc->vtbl->resolve_weight(be, &w);
    if ((w.flags & GEIST_W_AUX_HEAP_OWNED) != 0) {
        void *aux = (void *) w.aux_fp32;
        free(aux);
    }
    free(raw);
    return s;
}

static int run_backend(const char *backend, struct geist_backend *be) {
    int fails = 0;
    for (size_t i = 0; i < sizeof DTYPES / sizeof DTYPES[0]; i++) {
        const enum geist_dtype dt  = DTYPES[i].dt;
        const size_t           blk = DTYPES[i].blk;
        char                   msg[160];

        /* One and a half blocks per row, three in all. Storage for the
         * whole tensor, so the extent is not what gets it refused. */
        size_t bytes = 0;
        if (quant_raw_bytes(dt, 3 * blk, &bytes)) {
            fprintf(stderr, "FAIL: %s: quant_raw_bytes cannot size 3 blocks\n", DTYPES[i].name);
            fails++;
            continue;
        }
        enum geist_status s = resolve(be, dt, blk + blk / 2, 2, bytes);
        snprintf(msg,
                 sizeof msg,
                 "%s %s: rows of 1.5 blocks are refused as malformed (got %s)",
                 backend,
                 DTYPES[i].name,
                 geist_status_to_string(s));
        fails += geist_expect(s == GEIST_E_FORMAT, msg);

        /* Half a block in all: not even the tensor is whole blocks. */
        s = resolve(be, dt, blk / 2, 1, bytes);
        snprintf(msg,
                 sizeof msg,
                 "%s %s: half a block is refused as malformed (got %s)",
                 backend,
                 DTYPES[i].name,
                 geist_status_to_string(s));
        fails += geist_expect(s == GEIST_E_FORMAT, msg);

        /* Two blocks per row, two rows: well-formed. */
        if (quant_raw_bytes(dt, 4 * blk, &bytes)) {
            fprintf(stderr, "FAIL: %s: quant_raw_bytes cannot size 4 blocks\n", DTYPES[i].name);
            fails++;
            continue;
        }
        s = resolve(be, dt, 2 * blk, 2, bytes);
        snprintf(msg,
                 sizeof msg,
                 "%s %s: rows of 2 blocks are not refused as malformed (got %s)",
                 backend,
                 DTYPES[i].name,
                 geist_status_to_string(s));
        fails += geist_expect(s == GEIST_OK || s == GEIST_E_UNSUPPORTED, msg);
    }
    return fails;
}

int main(void) {
    int fails = 0, ran = 0;
    for (size_t i = 0; i < sizeof BACKENDS / sizeof BACKENDS[0]; i++) {
        struct geist_backend *be = nullptr;
        if (geist_backend_create(BACKENDS[i], nullptr, nullptr, &be) != GEIST_OK || be == nullptr) {
            continue; /* not in this build */
        }
        ran++;
        fails += run_backend(BACKENDS[i], be);
        geist_backend_destroy(be);
    }
    if (ran == 0) {
        printf("SKIP: no CPU backend in this build\n");
        return GEIST_TEST_SKIP;
    }
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: resolve_weight refuses rows that are not whole blocks\n");
    return GEIST_TEST_PASS;
}

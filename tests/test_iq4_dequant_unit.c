/*
 * tests/test_iq4_dequant_unit.c — IQ4_NL / IQ4_XS block decode, model-free.
 *
 * Exists for two reasons: pin the LUT/scale/nibble layout against
 * hand-computed values (the gguf-py cross-check needs a model file and
 * only runs where fixtures exist), and keep the coverage ratchet honest —
 * without it the iq4 decoders and the cpu_scalar resolve path for the two
 * dtypes only execute under the UD model fixture.
 */
#include "test_helpers.h"

#include <geist_backend.h>
#include <geist_types.h>

#include "quant.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int g_fail = 0;
#define check(ok, what) (g_fail |= geist_expect((ok), (what)))

/* The fixed non-linear table both formats index into (ggml kvalues_iq4nl). */
static const float KV[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

static void put_f16_one(uint8_t *p) { /* fp16(1.0) */
    p[0] = 0x00;
    p[1] = 0x3C;
}

int main(void) {
    /* ---- IQ4_NL: one 18-byte block, d = 1.0, qs[j] = (j+1)<<4 | j:
     * element j (low nibble) = KV[j], element j+16 (high) = KV[j+1&15]. */
    {
        uint8_t blk[18];
        put_f16_one(blk);
        for (int j = 0; j < 16; j++) {
            blk[2 + j] = (uint8_t) ((((j + 1) & 15) << 4) | j);
        }
        float out[32];
        dequant_iq4_nl_row(32, blk, out);
        bool ok = true;
        for (int j = 0; j < 16; j++) {
            ok &= out[j] == KV[j];
            ok &= out[j + 16] == KV[(j + 1) & 15];
        }
        check(ok, "IQ4_NL nibble/LUT layout");
    }

    /* ---- IQ4_XS: one 136-byte block, d = 1.0. Sub-block scales:
     * scales_l nibbles = ib32, scales_h 2-bit fields = 1
     * -> ls = 16 + ib32, dl = ls - 32 = ib32 - 16. All qs = 0x98:
     * low nibble 8 -> KV[8] = 1, high nibble 9 -> KV[9] = 13. */
    {
        uint8_t blk[136];
        memset(blk, 0x98, sizeof blk);
        put_f16_one(blk);
        blk[2] = 0x55; /* scales_h: eight 2-bit fields, all = 1 */
        blk[3] = 0x55;
        for (int i = 0; i < 4; i++) {
            const int lo = 2 * i, hi = 2 * i + 1;
            blk[4 + i] = (uint8_t) ((hi << 4) | lo);
        }
        float out[256];
        dequant_iq4_xs_row(256, blk, out);
        bool ok = true;
        for (int ib32 = 0; ib32 < 8; ib32++) {
            const float dl = (float) (16 + ib32 - 32);
            for (int j = 0; j < 16; j++) {
                ok &= out[32 * ib32 + j] == dl * 1.0f;       /* KV[8]  */
                ok &= out[32 * ib32 + 16 + j] == dl * 13.0f; /* KV[9]  */
            }
        }
        check(ok, "IQ4_XS sub-scale/nibble layout");
    }

    /* ---- cpu_scalar linear over an IQ4_XS weight: resolve installs the
     * row-dequant kernel; y must equal the dequant-then-dot reference.
     * Covers the weight_resolve dtype switch and dequant_one_row_for. */
    {
        enum { N_IN = 256, N_OUT = 3 };
        /* Row bytes: IQ4_XS 136 (1 block), IQ4_NL 144 (8x18) — size for
         * the larger of the two, gcc's stringop-overflow checks the
         * IQ4_NL writes against the full object. */
        static uint8_t wblob[N_OUT * 144];
        for (size_t i = 0; i < sizeof wblob; i++) {
            wblob[i] = (uint8_t) (i * 37u + 11u);
        }
        for (int r = 0; r < N_OUT; r++) {
            put_f16_one(wblob + (size_t) r * 136); /* keep d finite */
        }
        struct geist_backend *be = nullptr;
        if (geist_backend_create("cpu_scalar", nullptr, nullptr, &be) != GEIST_OK) {
            check(false, "cpu_scalar create");
            return 1;
        }
        struct geist_weight w = {.raw        = wblob,
                                 .raw_nbytes = sizeof wblob,
                                 .n_in       = N_IN,
                                 .n_out      = N_OUT,
                                 .dtype      = GEIST_DTYPE_IQ4_XS};
        check(be->desc->vtbl->resolve_weight(be, &w) == GEIST_OK, "IQ4_XS resolve");
        check(w.linear_m1 != nullptr, "IQ4_XS kernel installed");
        if (w.linear_m1 != nullptr) {
            float x[N_IN], y[N_OUT], row[N_IN];
            for (int i = 0; i < N_IN; i++) {
                x[i] = (float) ((i % 17) - 8) * 0.25f;
            }
            w.linear_m1(x, &w, be, y);
            for (int r = 0; r < N_OUT; r++) {
                dequant_iq4_xs_row(N_IN, wblob + (size_t) r * 136, row);
                float ref = 0.0f;
                for (int i = 0; i < N_IN; i++) {
                    ref += row[i] * x[i];
                }
                check(fabsf(y[r] - ref) <= 1e-3f * fmaxf(fabsf(ref), 1.0f),
                      "IQ4_XS scalar linear matches dequant+dot");
            }
        }
        /* Same round-trip for IQ4_NL (32-elem blocks). */
        struct geist_weight wn = {.raw        = wblob,
                                  .raw_nbytes = sizeof wblob,
                                  .n_in       = N_IN,
                                  .n_out      = N_OUT,
                                  .dtype      = GEIST_DTYPE_IQ4_NL};
        /* 256/32 = 8 blocks/row x 18 B = 144 B/row < 136*... reuse blob,
         * pin each block's d. */
        for (int r = 0; r < N_OUT; r++) {
            for (int b = 0; b < N_IN / 32; b++) {
                put_f16_one(wblob + (size_t) r * (N_IN / 32) * 18 + (size_t) b * 18);
            }
        }
        check(be->desc->vtbl->resolve_weight(be, &wn) == GEIST_OK, "IQ4_NL resolve");
        if (wn.linear_m1 != nullptr) {
            float x[N_IN], y[N_OUT], row[N_IN];
            for (int i = 0; i < N_IN; i++) {
                x[i] = (float) ((i % 13) - 6) * 0.5f;
            }
            wn.linear_m1(x, &wn, be, y);
            for (int r = 0; r < N_OUT; r++) {
                dequant_iq4_nl_row(N_IN, wblob + (size_t) r * (N_IN / 32) * 18, row);
                float ref = 0.0f;
                for (int i = 0; i < N_IN; i++) {
                    ref += row[i] * x[i];
                }
                check(fabsf(y[r] - ref) <= 1e-3f * fmaxf(fabsf(ref), 1.0f),
                      "IQ4_NL scalar linear matches dequant+dot");
            }
        }
        geist_backend_destroy(be);

        /* Same round-trips through cpu_neon: exercises the W4A8 decode
         * GEMVs (vqtbl1q LUT + SDOT). Activations are int8-quantized on
         * the way in, so parity vs the f32 dequant+dot reference is
         * tolerance-based, not exact. */
        struct geist_backend *bn = nullptr;
        if (geist_backend_create("cpu_neon", nullptr, nullptr, &bn) == GEIST_OK) {
            struct geist_weight wx = {.raw        = wblob,
                                      .raw_nbytes = sizeof wblob,
                                      .n_in       = N_IN,
                                      .n_out      = N_OUT,
                                      .dtype      = GEIST_DTYPE_IQ4_XS};
            check(bn->desc->vtbl->resolve_weight(bn, &wx) == GEIST_OK, "neon IQ4_XS resolve");
            if (wx.linear_m1 != nullptr) {
                float x[N_IN], y[N_OUT], row[N_IN];
                for (int i = 0; i < N_IN; i++) {
                    x[i] = (float) ((i % 17) - 8) * 0.25f;
                }
                wx.linear_m1(x, &wx, bn, y);
                for (int r = 0; r < N_OUT; r++) {
                    dequant_iq4_xs_row(N_IN, wblob + (size_t) r * 136, row);
                    float ref = 0.0f, mag = 0.0f;
                    for (int i = 0; i < N_IN; i++) {
                        ref += row[i] * x[i];
                        mag += fabsf(row[i] * x[i]);
                    }
                    check(fabsf(y[r] - ref) <= 2e-2f * fmaxf(mag, 1.0f),
                          "neon IQ4_XS w4a8 matches dequant+dot (a8 tolerance)");
                }
            }
#if defined(__ARM_NEON)
            /* #321 mN prefill: per (row, token) the op order matches the
             * m1 GEMV exactly, so the batched result must be BIT-equal
             * to m separate m1 calls — no tolerance. m=5 covers the
             * 4-token register tile plus the tail. */
            {
                enum { M_TOK = 5 };
                float xs[M_TOK * N_IN], ym[M_TOK * N_OUT], y1[N_OUT];
                for (int t = 0; t < M_TOK; t++) {
                    for (int i = 0; i < N_IN; i++) {
                        xs[t * N_IN + i] = (float) (((i + 3 * t) % 19) - 9) * 0.3f;
                    }
                }
                linear_iq4xs_w4a8_prefill(M_TOK, N_IN, N_OUT, xs, wblob, ym);
                for (int t = 0; t < M_TOK; t++) {
                    linear_iq4xs_decode_w4a8(N_IN, N_OUT, xs + t * N_IN, wblob, y1);
                    check(memcmp(ym + t * N_OUT, y1, sizeof y1) == 0,
                          "neon IQ4_XS mN prefill bit-equal to m1 per token");
                }
            }
#endif
            struct geist_weight wn2 = {.raw        = wblob,
                                       .raw_nbytes = sizeof wblob,
                                       .n_in       = N_IN,
                                       .n_out      = N_OUT,
                                       .dtype      = GEIST_DTYPE_IQ4_NL};
            check(bn->desc->vtbl->resolve_weight(bn, &wn2) == GEIST_OK, "neon IQ4_NL resolve");
            if (wn2.linear_m1 != nullptr) {
                float x[N_IN], y[N_OUT], row[N_IN];
                for (int i = 0; i < N_IN; i++) {
                    x[i] = (float) ((i % 13) - 6) * 0.5f;
                }
                wn2.linear_m1(x, &wn2, bn, y);
                for (int r = 0; r < N_OUT; r++) {
                    dequant_iq4_nl_row(N_IN, wblob + (size_t) r * (N_IN / 32) * 18, row);
                    float ref = 0.0f, mag = 0.0f;
                    for (int i = 0; i < N_IN; i++) {
                        ref += row[i] * x[i];
                        mag += fabsf(row[i] * x[i]);
                    }
                    check(fabsf(y[r] - ref) <= 2e-2f * fmaxf(mag, 1.0f),
                          "neon IQ4_NL w4a8 matches dequant+dot (a8 tolerance)");
                }
            }
            geist_backend_destroy(bn);
        }
    }

    if (g_fail == 0) {
        printf("test_iq4_dequant_unit: all checks passed\n");
    }
    return g_fail == 0 ? 0 : 1;
}

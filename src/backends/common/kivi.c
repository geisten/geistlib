/*
 * kivi.c — 2-bit asymmetric KV-cache quantization.
 *
 * Layer: BACKEND (common). Pure C23; portable scalar implementation.
 * NEON specialization will land alongside the runtime integration in
 * arch_state once the format is locked in.
 */
#include "kivi.h"

#include <math.h>
#include <stdint.h>

static inline uint8_t kivi_quant_one(float x, float zero, float inv_scale) {
    /* Asymmetric 2-bit: q = round((x - zero) / scale), clipped to [0,3]. */
    float t = (x - zero) * inv_scale;
    int   q = (int) lrintf(t);
    if (q < 0)
        q = 0;
    if (q > 3)
        q = 3;
    return (uint8_t) q;
}

/* Pack four 2-bit quants in [0..3] into one byte. Order: q0 in bits 0-1. */
static inline uint8_t kivi_pack4(uint8_t q0, uint8_t q1, uint8_t q2, uint8_t q3) {
    return (uint8_t) ((q3 << 6) | (q2 << 4) | (q1 << 2) | q0);
}

static inline void kivi_unpack4(uint8_t byte, uint8_t out[4]) {
    out[0] = (uint8_t) ((byte >> 0) & 0x3);
    out[1] = (uint8_t) ((byte >> 2) & 0x3);
    out[2] = (uint8_t) ((byte >> 4) & 0x3);
    out[3] = (uint8_t) ((byte >> 6) & 0x3);
}

void kivi_pack_v_row(size_t      n,
                     const float in[static n],
                     uint8_t     out_q[static n / 4],
                     float      *out_scale,
                     float      *out_zero) {
    /* Per-token: min/max across n elements; scale = range/3, zero = min. */
    float vmin = in[0], vmax = in[0];
    for (size_t i = 1; i < n; i++) {
        if (in[i] < vmin)
            vmin = in[i];
        if (in[i] > vmax)
            vmax = in[i];
    }
    const float range = vmax - vmin;
    /* Degenerate (constant row): scale=1, all q=0, reconstructed = zero. */
    const float scale     = range > 0.0f ? range / 3.0f : 1.0f;
    const float inv_scale = range > 0.0f ? 3.0f / range : 0.0f;
    *out_scale            = scale;
    *out_zero             = vmin;

    for (size_t i = 0; i < n; i += 4) {
        const uint8_t q0 = kivi_quant_one(in[i + 0], vmin, inv_scale);
        const uint8_t q1 = kivi_quant_one(in[i + 1], vmin, inv_scale);
        const uint8_t q2 = kivi_quant_one(in[i + 2], vmin, inv_scale);
        const uint8_t q3 = kivi_quant_one(in[i + 3], vmin, inv_scale);
        out_q[i / 4]     = kivi_pack4(q0, q1, q2, q3);
    }
}

void kivi_unpack_v_row(
        size_t n, const uint8_t in_q[static n / 4], float scale, float zero, float out[static n]) {
    for (size_t i = 0; i < n; i += 4) {
        uint8_t q[4];
        kivi_unpack4(in_q[i / 4], q);
        out[i + 0] = (float) q[0] * scale + zero;
        out[i + 1] = (float) q[1] * scale + zero;
        out[i + 2] = (float) q[2] * scale + zero;
        out[i + 3] = (float) q[3] * scale + zero;
    }
}

void kivi_pack_k_group(size_t      g_tokens,
                       size_t      n_channels,
                       const float in[static g_tokens * n_channels],
                       uint8_t     out_q[static(g_tokens * n_channels) / 4],
                       float       out_scales[static n_channels],
                       float       out_zeros[static n_channels]) {
    /* Pass 1: per-channel min/max across all g_tokens rows. */
    for (size_t c = 0; c < n_channels; c++) {
        float cmin = in[c], cmax = in[c];
        for (size_t t = 1; t < g_tokens; t++) {
            const float v = in[t * n_channels + c];
            if (v < cmin)
                cmin = v;
            if (v > cmax)
                cmax = v;
        }
        const float range = cmax - cmin;
        out_scales[c]     = range > 0.0f ? range / 3.0f : 1.0f;
        out_zeros[c]      = cmin;
    }

    /* Pass 2: quantize each (token, channel). Pack 4 channel-consecutive
     * values per byte. Channel block outer so the per-channel reciprocals
     * are computed once, not once per token row. */
    for (size_t c = 0; c < n_channels; c += 4) {
        const float inv0 = out_scales[c + 0] > 0.0f ? 1.0f / out_scales[c + 0] : 0.0f;
        const float inv1 = out_scales[c + 1] > 0.0f ? 1.0f / out_scales[c + 1] : 0.0f;
        const float inv2 = out_scales[c + 2] > 0.0f ? 1.0f / out_scales[c + 2] : 0.0f;
        const float inv3 = out_scales[c + 3] > 0.0f ? 1.0f / out_scales[c + 3] : 0.0f;
        for (size_t t = 0; t < g_tokens; t++) {
            const float  *row     = in + t * n_channels;
            uint8_t      *row_out = out_q + t * (n_channels / 4);
            const uint8_t q0      = kivi_quant_one(row[c + 0], out_zeros[c + 0], inv0);
            const uint8_t q1      = kivi_quant_one(row[c + 1], out_zeros[c + 1], inv1);
            const uint8_t q2      = kivi_quant_one(row[c + 2], out_zeros[c + 2], inv2);
            const uint8_t q3      = kivi_quant_one(row[c + 3], out_zeros[c + 3], inv3);
            row_out[c / 4]        = kivi_pack4(q0, q1, q2, q3);
        }
    }
}

void kivi_unpack_k_group(size_t        g_tokens,
                         size_t        n_channels,
                         const uint8_t in_q[static(g_tokens * n_channels) / 4],
                         const float   scales[static n_channels],
                         const float   zeros[static n_channels],
                         float         out[static g_tokens * n_channels]) {
    for (size_t t = 0; t < g_tokens; t++) {
        const uint8_t *row_in = in_q + t * (n_channels / 4);
        float         *row    = out + t * n_channels;
        for (size_t c = 0; c < n_channels; c += 4) {
            uint8_t q[4];
            kivi_unpack4(row_in[c / 4], q);
            row[c + 0] = (float) q[0] * scales[c + 0] + zeros[c + 0];
            row[c + 1] = (float) q[1] * scales[c + 1] + zeros[c + 1];
            row[c + 2] = (float) q[2] * scales[c + 2] + zeros[c + 2];
            row[c + 3] = (float) q[3] * scales[c + 3] + zeros[c + 3];
        }
    }
}

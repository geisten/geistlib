#ifndef QWEN_M1_DELTANET_OUTPUT_H
#define QWEN_M1_DELTANET_OUTPUT_H

static inline float
q1_deltanet_output(float out, float inv, float weight, float gate, float gate_exp, float gate_rcp) {
#if defined(__clang__)
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
#endif
    /* Match the M1's FP32 multiplication tree around the SiLU gate. */
    const float normalized = (out * inv) * weight;
    const float numerator  = gate >= 0.0f ? gate : gate * gate_exp;
    return (normalized * numerator) * gate_rcp;
}

#endif

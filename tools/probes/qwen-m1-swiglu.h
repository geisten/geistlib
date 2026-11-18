#ifndef QWEN_M1_SWIGLU_H
#define QWEN_M1_SWIGLU_H

static inline float q1_swiglu(float gate, float up, float sigmoid) {
#if defined(__clang__)
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
#endif
    /* The M1 rounds the product of the projections before applying sigmoid. */
    const float product = gate * up;
    return product * sigmoid;
}

#endif

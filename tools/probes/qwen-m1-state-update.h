#ifndef QWEN_M1_STATE_UPDATE_H
#define QWEN_M1_STATE_UPDATE_H

#ifdef __METAL_VERSION__
#define Q1_FMA fma
#else
#include <math.h>
#define Q1_FMA fmaf
#endif

static inline float
q1_state_update(float key, float residual, float beta, float previous, float decay) {
#if defined(__clang__)
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
#endif
    /* Round K times the residual before multiplying by beta. Fuse the second
     * multiply with accumulation of the previously rounded state term. */
    const float kr    = key * residual;
    const float carry = previous * decay;
    return Q1_FMA(kr, beta, carry);
}

#undef Q1_FMA
#endif

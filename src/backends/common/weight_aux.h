/*
 * src/backends/common/weight_aux.h — frees a geist_weight's owned aux_fp32.
 *
 * Layer: BACKEND (common).
 */
#ifndef GEIST_WEIGHT_AUX_H
#define GEIST_WEIGHT_AUX_H

#include <geist_weight.h>

#include "heap.h"

#include <stdint.h>

/* Frees aux_fp32 when the weight owns it (GEIST_W_AUX_HEAP_OWNED): mapped
 * pages of aux_n bytes under GEIST_W_AUX_PAGES, a heap block otherwise.
 * Clears aux_fp32 and both flags; a weight without owned aux is left as is. */
static inline void weight_aux_free(struct geist_weight *w) {
    if ((w->flags & GEIST_W_AUX_HEAP_OWNED) == 0 || w->aux_fp32 == nullptr) {
        return;
    }
    void *p = (void *) (uintptr_t) w->aux_fp32;
    if ((w->flags & GEIST_W_AUX_PAGES) != 0) {
        heap_free_pages(&p, (size_t) w->aux_n);
    } else {
        safe_free(&p);
    }
    w->aux_fp32 = nullptr;
    w->flags &= (uint16_t) ~(GEIST_W_AUX_HEAP_OWNED | GEIST_W_AUX_PAGES);
}

#endif /* GEIST_WEIGHT_AUX_H */

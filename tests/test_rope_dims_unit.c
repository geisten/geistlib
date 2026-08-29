/*
 * test_rope_dims_unit — the rotary head_dim contract (issue #329).
 *
 * Rotary position embedding rotates channel i against channel
 * i + head_dim/2. With an odd head_dim the last channel has no partner,
 * and rope_compute_at writes only 2*(head_dim/2) of the head_dim entries
 * it is handed — leaving the last one exactly as the allocator left it.
 * The arch layer then copies that tail into an activation. head_dim is
 * model metadata (d_model / n_q_heads for Llama and BitNet), so a
 * malformed file can choose it.
 *
 * The contract is that odd head_dim is rejected, not silently half-
 * handled. These cases pin both halves: the predicate answers correctly
 * on the boundary, and for every head_dim it accepts, rope_compute_at
 * leaves no table entry unwritten. Hermetic — no model, no backend.
 */
#include "test_helpers.h"

#include "gemma4_kernels.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            g_fail = 1;                                                     \
        }                                                                   \
    } while (0)

/* Sentinel that no cos/sin value can legitimately take. */
static const float POISON = -7.5e30f;

/* Fill both tables with the sentinel, compute, and report how many entries
 * the kernel left untouched. Zero is the only acceptable answer: a table
 * entry nobody wrote is read on every forward pass. */
static size_t unwritten_after_compute(size_t n_positions, size_t head_dim, size_t n_rotated) {
    const size_t n   = n_positions * head_dim;
    float       *cos = malloc(n * sizeof(float));
    float       *sin = malloc(n * sizeof(float));
    if (cos == nullptr || sin == nullptr) {
        free(cos);
        free(sin);
        fprintf(stderr, "alloc failed\n");
        exit(GEIST_TEST_ERROR);
    }
    for (size_t i = 0; i < n; i++) {
        cos[i] = POISON;
        sin[i] = POISON;
    }
    rope_compute_at(0, n_positions, head_dim, n_rotated, 10000.0f, cos, sin);
    size_t untouched = 0;
    for (size_t i = 0; i < n; i++) {
        untouched += (cos[i] == POISON) + (sin[i] == POISON);
    }
    free(cos);
    free(sin);
    return untouched;
}

int main(void) {
    /* ---- the predicate on the boundary ---------------------------------- */
    CHECK(!rope_head_dim_supported(0)); /* no channels at all */
    CHECK(!rope_head_dim_supported(1));
    CHECK(rope_head_dim_supported(2)); /* smallest usable: one pair */
    CHECK(!rope_head_dim_supported(3));
    CHECK(rope_head_dim_supported(4));
    /* The head_dims the shipped families actually use. */
    CHECK(rope_head_dim_supported(64));
    CHECK(rope_head_dim_supported(128)); /* Llama, BitNet */
    CHECK(rope_head_dim_supported(256)); /* Gemma 4 sliding */
    CHECK(rope_head_dim_supported(512)); /* Gemma 4 full */
    /* A d_model that is not a multiple of 2*n_q_heads lands here. */
    CHECK(!rope_head_dim_supported(255));
    CHECK(!rope_head_dim_supported(1537));

    /* ---- accepted head_dims leave nothing unwritten ---------------------- */
    static const size_t EVEN[] = {2, 4, 64, 128, 256, 512};
    for (size_t i = 0; i < sizeof(EVEN) / sizeof(EVEN[0]); i++) {
        const size_t hd = EVEN[i];
        /* Full rotation, and Gemma 4's partial rotation (n_rotated < hd),
         * which still has to fill the whole table — the un-rotated tail
         * gets cos=1/sin=0 from a zero inverse frequency, not garbage. */
        const size_t left_full = unwritten_after_compute(4, hd, hd);
        const size_t left_part = unwritten_after_compute(4, hd, hd / 4);
        if (left_full != 0 || left_part != 0) {
            fprintf(stderr,
                    "FAIL: head_dim=%zu left %zu (full) / %zu (partial) table entries "
                    "unwritten\n",
                    hd,
                    left_full,
                    left_part);
            g_fail = 1;
        }
    }

    /* ---- and the reason odd is rejected --------------------------------
     * Not a hypothetical. Calling the kernel with an odd head_dim is safe
     * (it writes fewer entries, it does not run off the end), so we can
     * show exactly what it leaves behind: one cos and one sin per position.
     * That tail is what the arch layer used to copy into an activation.
     * If a future change ever makes odd head_dim fully defined, this
     * assertion is the one to revisit — together with the predicate. */
    const size_t odd_left = unwritten_after_compute(4, 5, 5);
    if (odd_left != 8) {
        fprintf(stderr,
                "FAIL: expected head_dim=5 to leave 8 entries unwritten (1 cos + 1 sin "
                "per position over 4 positions), got %zu\n",
                odd_left);
        g_fail = 1;
    }

    if (g_fail) {
        return GEIST_TEST_FAIL;
    }
    printf("PASS: rotary head_dim contract — odd and zero rejected, every accepted "
           "head_dim fills its whole cos/sin table\n");
    return GEIST_TEST_PASS;
}

/*
 * test_rope_dims_unit — the rotary head_dim and rotated-width contracts
 * (issues #329 and #432).
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

#include <math.h>
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
    /* A table row covers the ROTATED dims, not the whole head. */
    const size_t n_rot = n_rotated < head_dim ? n_rotated : head_dim;
    const size_t n     = n_positions * n_rot;
    float       *cos   = malloc(n * sizeof(float));
    float       *sin   = malloc(n * sizeof(float));
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
    rope_compute_at(0, n_positions, head_dim, n_rot, true, 10000.0f, cos, sin);
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
        /* Full rotation, and a partial one (n_rotated < hd, as qwen35's
         * 64 of 256): the table is as wide as the rotated block and every
         * entry of it has to be written. An even rotated width is part of
         * the contract, same as an even head_dim. */
        const size_t part      = hd / 4 >= 2 ? (hd / 4) & ~(size_t) 1 : 2;
        const size_t left_full = unwritten_after_compute(4, hd, hd);
        const size_t left_part = unwritten_after_compute(4, hd, part);
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

    /* ---- partial rotary rotates the right pairs at the right rate ------
     * The bug this pins (#432): pairing on head_dim/2 instead of n_rot/2
     * rotated dims 0..31 against 128..159 for qwen35's 64-of-256 head, and
     * the frequency exponent divided by head_dim instead of n_rot. Both
     * collapse to the correct formula when n_rot == head_dim, which is why
     * every other family stayed correct. The reference below is the
     * definition, written out: pair (i, i + n_rot/2), angle
     * pos * theta^(-2i/n_rot), dims at or above n_rot untouched. */
    {
        const size_t HD = 256, N_ROT = 64, SEQ = 3, HEADS = 2, POS0 = 5;
        const float  THETA = 1.0e7f;
        float       *x     = malloc(SEQ * HEADS * HD * sizeof(float));
        float       *want  = malloc(SEQ * HEADS * HD * sizeof(float));
        float       *cos_t = malloc(SEQ * N_ROT * sizeof(float));
        float       *sin_t = malloc(SEQ * N_ROT * sizeof(float));
        if (x == nullptr || want == nullptr || cos_t == nullptr || sin_t == nullptr) {
            fprintf(stderr, "alloc failed\n");
            exit(GEIST_TEST_ERROR);
        }
        for (size_t i = 0; i < SEQ * HEADS * HD; i++) {
            x[i] = (float) ((int) (i % 37u) - 18) * 0.05f;
        }
        memcpy(want, x, SEQ * HEADS * HD * sizeof(float));
        const size_t h = N_ROT / 2;
        for (size_t s = 0; s < SEQ; s++) {
            for (size_t head = 0; head < HEADS; head++) {
                float *row = want + (s * HEADS + head) * HD;
                for (size_t i = 0; i < h; i++) {
                    const double freq  = pow((double) THETA, -(double) (2 * i) / (double) N_ROT);
                    const double angle = (double) (POS0 + s) * freq;
                    const float  a = row[i], b = row[i + h];
                    row[i]     = (float) ((double) a * cos(angle) - (double) b * sin(angle));
                    row[i + h] = (float) ((double) b * cos(angle) + (double) a * sin(angle));
                }
            }
        }
        rope_compute_at(POS0, SEQ, HD, N_ROT, true, THETA, cos_t, sin_t);
        rope_apply(SEQ, HEADS, HD, N_ROT, x, cos_t, sin_t);
        size_t bad = 0, touched_tail = 0;
        for (size_t i = 0; i < SEQ * HEADS * HD; i++) {
            const size_t d = i % HD;
            if (fabsf(x[i] - want[i]) > 1e-5f) {
                bad++;
                if (d >= N_ROT) {
                    touched_tail++;
                }
            }
        }
        if (bad != 0) {
            fprintf(stderr,
                    "FAIL: partial rotary (head_dim=%zu, n_rot=%zu): %zu of %zu values differ "
                    "from the definition (%zu of them outside the rotated block)\n",
                    HD,
                    N_ROT,
                    bad,
                    SEQ * HEADS * HD,
                    touched_tail);
            g_fail = 1;
        }
        free(x);
        free(want);
        free(cos_t);
        free(sin_t);
    }

    if (g_fail) {
        return GEIST_TEST_FAIL;
    }
    printf("PASS: rotary head_dim contract — odd and zero rejected, every accepted "
           "head_dim fills its whole cos/sin table\n");
    return GEIST_TEST_PASS;
}

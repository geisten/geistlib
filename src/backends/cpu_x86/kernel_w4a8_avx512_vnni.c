/*
 * src/backends/cpu_x86/kernel_w4a8_avx512_vnni.c — AVX-512 + AVX-512_VNNI
 * variant of the W4A8 dot kernel.
 *
 * Compiled with -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni; the
 * dispatcher (kernel_w4a8.c, baseline -march) only ever calls this on a host
 * whose runtime cpuid reports the corresponding feature bits, so there is no
 * SIGILL risk on AVX2-only or older CPUs.
 *
 * --- Kernel sketch ---------------------------------------------------------
 *
 * One iteration processes TWO 32-element W4A8 blocks per 512-bit VPDPBUSD:
 *   1. Load 32 packed weight bytes (= 16 packed + 16 packed = 2 blocks).
 *   2. Extract low+high nibbles per byte (unsigned [0, 15]).
 *   3. Interleave the 32-nibble unpacked form per 128-bit lane, then
 *      reassemble so bytes 0..31 are block 0 in original order and
 *      bytes 32..63 are block 1 in original order.
 *   4. Load 64 int8 activations (= block 0 acts | block 1 acts).
 *   5. _mm512_dpbusd_epi32 → 16 int32 lanes; the first 8 contain block 0's
 *      4-wide subsums and the second 8 contain block 1's.
 *   6. Horizontal-reduce each half → d_b0, d_b1.
 *   7. acc += w_scales[b]*d_b0 - w_offsets[b]*sum_a_per_block[b];
 *      acc += w_scales[b+1]*d_b1 - w_offsets[b+1]*sum_a_per_block[b+1];
 *
 * Two-block-per-iteration is allowed unconditionally because Q4_K guarantees
 * n_in is a multiple of 256 (one super-block) → n_blocks = n_in / 32 is a
 * multiple of 8, always even. For correctness across non-Q4_K callers the
 * scalar reference handles odd n_blocks; this kernel doesn't need to.
 *
 * Caller pre-computes sum_a_per_block once per activation row, so the
 * inner loop is pure VPDPBUSD + load + small per-block fp arithmetic.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "kernel_w4a8.h"

#include <immintrin.h>
#include <stdint.h>

/* Horizontal sum of 8 int32 lanes in a __m256i → int32. */
static inline int32_t hsum_i32_avx2(__m256i v) {
    const __m128i lo = _mm256_castsi256_si128(v);
    const __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i       s4 = _mm_add_epi32(lo, hi);
    s4               = _mm_hadd_epi32(s4, s4);
    s4               = _mm_hadd_epi32(s4, s4);
    return _mm_cvtsi128_si32(s4);
}

[[nodiscard]] float
w4a8_dot_avx512_vnni(size_t        n_blocks,
                     const uint8_t weights[static n_blocks * W4A8_BLOCK_BYTES_WEIGHTS],
                     const float   w_scales[static n_blocks],
                     const float   w_offsets[static n_blocks],
                     const int8_t  acts[static n_blocks * W4A8_BLOCK_ELEMS],
                     const int32_t sum_a_per_block[static n_blocks],
                     float         scale_x) {
    const __m256i lo_mask256 = _mm256_set1_epi8(0x0F);

    float  acc = 0.0f;
    size_t b   = 0;

    /* Main loop: two W4A8 blocks per 512-bit VPDPBUSD. */
    for (; b + 2 <= n_blocks; b += 2) {
        /* 32 packed bytes = 16 (block 0) + 16 (block 1) in 256-bit reg. */
        const __m256i packed =
                _mm256_loadu_si256((const __m256i *) (weights + b * W4A8_BLOCK_BYTES_WEIGHTS));

        /* Per-byte unsigned nibble extraction. */
        const __m256i lo_nibs = _mm256_and_si256(packed, lo_mask256);
        const __m256i hi_nibs = _mm256_and_si256(_mm256_srli_epi16(packed, 4), lo_mask256);

        /* Interleave WITHIN each 128-bit lane: low lane = block 0 nibble
         * pairs, high lane = block 1 nibble pairs. unpacklo gives the
         * first 8 byte-pairs in each lane → 16 interleaved bytes per lane.
         * unpackhi gives the second 8 → 16 more per lane. */
        const __m256i nibs_lo = _mm256_unpacklo_epi8(lo_nibs, hi_nibs);
        const __m256i nibs_hi = _mm256_unpackhi_epi8(lo_nibs, hi_nibs);

        /* Extract per-block halves:
         *   block 0 first 16 elements   = low 128 of nibs_lo
         *   block 0 second 16 elements  = low 128 of nibs_hi
         *   block 1 first 16 elements   = high 128 of nibs_lo
         *   block 1 second 16 elements  = high 128 of nibs_hi
         */
        const __m128i b0_first  = _mm256_castsi256_si128(nibs_lo);
        const __m128i b0_second = _mm256_castsi256_si128(nibs_hi);
        const __m128i b1_first  = _mm256_extracti128_si256(nibs_lo, 1);
        const __m128i b1_second = _mm256_extracti128_si256(nibs_hi, 1);

        /* Reassemble per block (32 unsigned bytes each, in element order). */
        const __m256i b0_u_w = _mm256_set_m128i(b0_second, b0_first);
        const __m256i b1_u_w = _mm256_set_m128i(b1_second, b1_first);

        /* Concatenate into a 64-byte __m512i: bytes 0..31 = block 0,
         * bytes 32..63 = block 1. */
        const __m512i u_w_512 = _mm512_inserti64x4(_mm512_castsi256_si512(b0_u_w), b1_u_w, 1);

        /* 64 int8 activations: block 0 acts | block 1 acts. */
        const __m512i s_a_512 = _mm512_loadu_si512((const __m512i *) (acts + b * W4A8_BLOCK_ELEMS));

        /* 16 int32 lanes. Lanes 0..7 sum block 0's 4-wide groups; lanes
         * 8..15 sum block 1's. */
        const __m512i dot512 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), u_w_512, s_a_512);

        const __m256i d_low  = _mm512_castsi512_si256(dot512);
        const __m256i d_high = _mm512_extracti64x4_epi64(dot512, 1);
        const int32_t d_b0   = hsum_i32_avx2(d_low);
        const int32_t d_b1   = hsum_i32_avx2(d_high);

        acc += w_scales[b] * (float) d_b0 - w_offsets[b] * (float) sum_a_per_block[b];
        acc += w_scales[b + 1] * (float) d_b1 - w_offsets[b + 1] * (float) sum_a_per_block[b + 1];
    }

    /* Tail: one block (kernel called with odd n_blocks). 256-bit path. */
    if (b < n_blocks) {
        const __m128i packed_128 =
                _mm_loadu_si128((const __m128i *) (weights + b * W4A8_BLOCK_BYTES_WEIGHTS));
        const __m256i packed_256 = _mm256_set_m128i(packed_128, packed_128);
        const __m256i lo_nibs    = _mm256_and_si256(packed_256, lo_mask256);
        const __m256i hi_nibs    = _mm256_and_si256(_mm256_srli_epi16(packed_256, 4), lo_mask256);
        const __m256i nibs_lo    = _mm256_unpacklo_epi8(lo_nibs, hi_nibs);
        const __m256i nibs_hi    = _mm256_unpackhi_epi8(lo_nibs, hi_nibs);
        const __m256i u_w        = _mm256_permute2x128_si256(nibs_lo, nibs_hi, 0x20);
        const __m256i s_a   = _mm256_loadu_si256((const __m256i *) (acts + b * W4A8_BLOCK_ELEMS));
        const __m256i dot32 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), u_w, s_a);
        const int32_t d_b   = hsum_i32_avx2(dot32);
        acc += w_scales[b] * (float) d_b - w_offsets[b] * (float) sum_a_per_block[b];
    }

    return scale_x * acc;
}
